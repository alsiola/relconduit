"""Issue 18's cloud run: cand-lgbm-ranker-plus on rel-stack/user-post-comment and
rel-trial/condition-sponsor-run, side by side on one 24-vCPU machine.

A smoke phase first (both searches end to end on a few hundred seeds, their pair leakage
checks included); nothing else starts unless both pass. Then both real searches
(``relconduit.link_search``: val-only choices, test scored once, 5 seeds), ``--threads``
each. Each finished job leaves ``<out>/<job>/done.json`` (written last, by rename) with its
minutes and peak RSS; a restarted machine skips it, and a half-done search resumes from its
cached fits in ``<out>/<job>/work``. Pair tables go to ``--pairs-root``, outside the
checkpoint. The driver prints each running job's last log line every ``--heartbeat`` minutes.

    uv run python experiments/rec-board/cloud18.py --out runs/cloud18
"""

import argparse
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cloud_jobs import heartbeat, run_job

TASKS = [("rel-stack", "user-post-comment"), ("rel-trial", "condition-sponsor-run")]


def search(ds, task, out: Path, pairs_root: Path, threads: int, smoke: int | None = None):
    name = f"{'smoke' if smoke else 'search'}_{ds}_{task}"
    work = out / name / "work"
    cmd = [sys.executable, f"experiments/{ds}/{task}/cand-lgbm-ranker-plus/run.py"]
    cmd += ["--work-dir", str(work), "--pairs-dir", str(pairs_root / name)]
    cmd += ["--smoke", str(smoke)] if smoke else ["--test"]

    def collect(job: Path) -> None:
        rows = Path(f"results/{ds}/{task}.jsonl")
        if not smoke and rows.exists():
            shutil.copy(rows, job / "rows.jsonl")
        for f in work.rglob("summary*.json"):
            shutil.copy(f, job / f.name)

    return run_job(name, cmd, out, threads, collect)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cloud18")
    p.add_argument("--pairs-root", default=str(Path.home() / "rc18pairs"))
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--smoke-rows", type=int, default=150)
    p.add_argument("--heartbeat", type=float, default=5)
    p.add_argument("--smoke", action="store_true", help="only the smoke phase")
    p.add_argument("--no-smoke", action="store_true")
    a = p.parse_args()
    out, pairs_root = Path(a.out), Path(a.pairs_root)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.heartbeat, stop), daemon=True).start()
    with ThreadPoolExecutor(len(TASKS)) as pool:
        if not a.no_smoke:
            futs = [
                pool.submit(search, d, t, out / "smoke", pairs_root, a.threads, a.smoke_rows)
                for d, t in TASKS
            ]
            msgs = [f.result() for f in futs]
            if any("FAILED" in m for m in msgs):
                sys.exit("smoke phase failed; not starting the real jobs:\n" + "\n".join(msgs))
            if a.smoke:
                return
        futs = [pool.submit(search, d, t, out, pairs_root, a.threads) for d, t in TASKS]
        msgs = [f.result() for f in futs]
    stop.set()
    print("\n".join(msgs), flush=True)
    (out / "done.json").write_text(json.dumps({"jobs": msgs}, indent=2))
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
