"""Issue 28's cloud runs: plans of ``run.py`` side by side on one 24-vCPU machine.

A smoke phase first (each plan end to end on a few hundred seeds, its pair leakage checks
included); nothing else starts unless every smoke passes. Then each plan as its own process
with ``--threads`` threads (LightGBM, DuckDB). Each finished job leaves
``<out>/<job>/done.json`` (written last), so a restarted machine skips it, and a half-done
plan resumes from its cached fits in ``<out>/<job>/work``; pair chunks go to ``--pairs-root``,
outside the checkpoint. The driver prints each running job's last log line every
``--heartbeat`` minutes.

    uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-v2/cloud28.py \
        --out runs/cloud28 --plans curve:14 ablation:10
"""

import argparse
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "rec-board"))
from cloud_jobs import heartbeat, run_job

RUN = "experiments/rel-trial/site-sponsor-run/cand-lgbm-v2/run.py"


def job(plan: str, out: Path, pairs_root: Path, threads: int, smoke: int | None, extra=()):
    name = f"{'smoke' if smoke else 'plan'}_{plan}"
    work = out / name / "work"
    cmd = [sys.executable, RUN, "--plan", plan, "--work-dir", str(work)]
    cmd += ["--pairs-dir", str(pairs_root / name), *extra]
    if smoke:
        cmd += ["--smoke", str(smoke)]

    def collect(j: Path) -> None:
        for f in work.rglob("summary*.json"):
            shutil.copy(f, j / f.name)
        for f in work.rglob("tune.jsonl"):
            shutil.copy(f, j / f.name)

    return run_job(name, cmd, out, threads, collect)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cloud28")
    p.add_argument("--pairs-root", default=str(Path.home() / "rc28pairs"))
    p.add_argument("--plans", nargs="+", required=True, help="plan:threads, e.g. curve:14")
    p.add_argument("--smoke-rows", type=int, default=150)
    p.add_argument("--heartbeat", type=float, default=5)
    p.add_argument("--no-smoke", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed to run.py")
    a = p.parse_args()
    out, pairs_root = Path(a.out), Path(a.pairs_root)
    out.mkdir(parents=True, exist_ok=True)
    plans = [(s.split(":")[0], int(s.split(":")[1])) for s in a.plans]
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.heartbeat, stop), daemon=True).start()
    if not a.no_smoke:
        with ThreadPoolExecutor(len(plans)) as pool:
            futs = [
                pool.submit(job, pl, out / "smoke", pairs_root, 4, a.smoke_rows, a.extra)
                for pl, _ in plans
            ]
            msgs = [f.result() for f in futs]
        if any("FAILED" in m for m in msgs):
            sys.exit("smoke phase failed; not starting the real jobs:\n" + "\n".join(msgs))
    with ThreadPoolExecutor(len(plans)) as pool:
        futs = [pool.submit(job, pl, out, pairs_root, th, None, a.extra) for pl, th in plans]
        msgs = [f.result() for f in futs]
    stop.set()
    print("\n".join(msgs), flush=True)
    (out / "done.json").write_text(json.dumps({"jobs": msgs}, indent=2))
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
