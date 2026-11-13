"""Issue 21's cloud run: the recommendation board's wave 2 (rel-amazon's three link tasks and
rel-hm user-item-purchase), side by side on one 24-vCPU machine.

* A smoke phase first: each ranker end to end on a few hundred seeds (its pair leakage check
  included). Nothing else starts unless all pass.
* The four rankers (``relconduit.link_big``: val-only choices, every val and test row scored,
  test scored once, 3 seeds), ``--threads`` each, and upstream's trivial baselines (past visit,
  global popularity; task tables only) beside them. Each ranker waits until ``--free-gb`` of
  memory is available before it starts, so they don't all load their databases at once.

Each finished job leaves ``<out>/<job>/done.json`` (written last, by rename) with its minutes
and peak RSS; a restarted machine skips it, and a half-done ranker resumes from its cached fits
in ``<out>/<job>/work`` (pair chunks go to ``--pairs-root``, outside the checkpoint). Each
ranker's seed-0 test table stays at ``<out>/<job>/work/.../submission/`` on the checkpoint.
The driver prints each running job's last log line every ``--heartbeat`` minutes.

    uv run python experiments/rec-board/cloud21.py --out runs/cloud21
"""

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cloud_jobs import heartbeat, run_job, trivial, wait_for_memory

# Biggest first.
TASKS = [
    ("rel-amazon", "user-item-purchase"),
    ("rel-amazon", "user-item-rate"),
    ("rel-amazon", "user-item-review"),
    ("rel-hm", "user-item-purchase"),
]


def ranker(ds, task, out: Path, pairs_root: Path, threads: int, smoke=None, free_gb=0.0, delay=0.0):
    name = f"{'smoke' if smoke else 'ranker'}_{ds}_{task}"
    if free_gb and not (out / name / "done.json").exists():
        time.sleep(delay * 60)  # let the job before load its database first
        wait_for_memory(name, free_gb)
    work = out / name / "work"
    cmd = [sys.executable, f"experiments/{ds}/{task}/cand-lgbm-ranker/run.py"]
    cmd += ["--work-dir", str(work), "--pairs-dir", str(pairs_root / name)]
    cmd += ["--smoke", str(smoke)] if smoke else ["--test"]

    def collect(job: Path) -> None:
        rows = Path(f"results/{ds}/{task}.jsonl")
        if not smoke and rows.exists():
            shutil.copy(rows, job / "rows.jsonl")
        for f in work.rglob("summary*.json"):
            shutil.copy(f, job / f.name)
        for f in work.rglob("tune.jsonl"):
            shutil.copy(f, job / f.name)

    return run_job(name, cmd, out, threads, collect)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cloud21")
    p.add_argument("--pairs-root", default=str(Path.home() / "rc21pairs"))
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--parallel", type=int, default=4, help="rankers at once")
    p.add_argument("--free-gb", type=float, default=20)
    p.add_argument("--stagger", type=float, default=4, help="minutes between ranker starts")
    p.add_argument("--smoke-rows", type=int, default=200)
    p.add_argument("--heartbeat", type=float, default=5)
    p.add_argument("--smoke", action="store_true", help="only the smoke phase")
    p.add_argument("--no-smoke", action="store_true")
    p.add_argument("--no-trivial", action="store_true")
    a = p.parse_args()
    out, pairs_root = Path(a.out), Path(a.pairs_root)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.heartbeat, stop), daemon=True).start()
    if not a.no_smoke:
        with ThreadPoolExecutor(len(TASKS)) as pool:
            futs = [
                pool.submit(ranker, d, t, out / "smoke", pairs_root, a.threads, a.smoke_rows)
                for d, t in TASKS
            ]
            msgs = [f.result() for f in futs]
        if any("FAILED" in m for m in msgs):
            sys.exit("smoke phase failed; not starting the real jobs:\n" + "\n".join(msgs))
        if a.smoke:
            return
    msgs = []
    with ThreadPoolExecutor(a.parallel) as pool, ThreadPoolExecutor(1) as tpool:
        tfuts = [] if a.no_trivial else [tpool.submit(trivial, d, t, out) for d, t in TASKS]
        futs = []
        for i, (d, t) in enumerate(TASKS):
            futs.append(
                pool.submit(
                    ranker, d, t, out, pairs_root, a.threads, None, a.free_gb, i * a.stagger
                )
            )
        msgs = [f.result() for f in futs] + [f.result() for f in tfuts]
    stop.set()
    print("\n".join(msgs), flush=True)
    (out / "done.json").write_text(json.dumps({"jobs": msgs}, indent=2))
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
