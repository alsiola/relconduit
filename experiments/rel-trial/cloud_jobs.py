"""Issue 6's cloud run: the rel-trial baselines and dfs-lightgbm runs, plus avito test.

Runs each job as its own process, side by side, on one 24-vCPU machine. Each finished job
leaves ``<out>/<job>/done.json`` (written last, by rename), so a restarted machine skips it.
Per job: ``log`` (stdout+stderr), and for the baselines ``pred/`` and ``summary.txt``, for
dfs-lightgbm the summary JSON and the results rows it appended (``rows.jsonl``).

    uv run python experiments/rel-trial/cloud_jobs.py --out runs/cloud6
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UPSTREAM = Path("experiments/rel-f1/baselines/upstream/lightgbm_entity.py")

BASELINES = [
    ("rel-trial", "study-adverse", 42),
    ("rel-trial", "study-adverse", 43),
    ("rel-trial", "study-adverse", 44),
    ("rel-trial", "site-success", 43),
    ("rel-trial", "site-success", 44),
]
# Longest first, so the short one fills a slot at the end.
DFS = [
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-avito", "ad-ctr"),
]


def run_job(name: str, cmd: list[str], out: Path, threads: int, collect) -> str:
    job = out / name
    if (job / "done.json").exists():
        return f"{name}: already done"
    job.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTHONUNBUFFERED": "1",
    }
    t0 = time.time()
    with open(job / "log", "w") as log:
        rc = subprocess.run(
            cmd, check=False, stdout=log, stderr=subprocess.STDOUT, env=env
        ).returncode
    minutes = (time.time() - t0) / 60
    if rc != 0:
        return f"{name}: FAILED rc={rc} after {minutes:.1f} min (see {job / 'log'})"
    collect(job)
    tmp = job / "done.json.tmp"
    tmp.write_text(json.dumps({"minutes": round(minutes, 2), "cmd": cmd}))
    tmp.rename(job / "done.json")
    return f"{name}: done in {minutes:.1f} min"


def baseline(ds: str, task: str, seed: int, out: Path, threads: int) -> str:
    name = f"baseline_{ds}_{task}_s{seed}"
    pred = out / name / "pred"
    cmd = [
        sys.executable,
        str(UPSTREAM),
        "--dataset",
        ds,
        "--task",
        task,
        "--seed",
        str(seed),
        "--cache_dir",
        str(Path.home() / "bl_cache" / name),
        "--pred_dir",
        str(pred),
    ]

    def collect(job: Path) -> None:
        lines = [
            ln
            for ln in (job / "log").read_text().splitlines()
            if ln.startswith(("Train:", "Val:", "Test:"))
        ]
        (job / "summary.txt").write_text("\n".join(lines) + "\n")

    return run_job(name, cmd, out, threads, collect)


def dfs(ds: str, task: str, out: Path, threads: int) -> str:
    name = f"dfs_{ds}_{task}"
    cmd = [sys.executable, f"experiments/{ds}/{task}/dfs-lightgbm/run.py", "--test"]

    def collect(job: Path) -> None:
        rows = Path(f"results/{ds}/{task}.jsonl")
        shutil.copy(rows, job / "rows.jsonl")
        for f in Path(f"runs/{ds}/{task}/dfs-lightgbm").glob("*.json"):
            shutil.copy(f, job / f.name)

    return run_job(name, cmd, out, threads, collect)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cloud6")
    p.add_argument("--baseline-threads", type=int, default=2)
    p.add_argument("--dfs-threads", type=int, default=7)
    p.add_argument("--dfs-parallel", type=int, default=2)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (
        ThreadPoolExecutor(len(BASELINES)) as bl_pool,
        ThreadPoolExecutor(a.dfs_parallel) as dfs_pool,
    ):
        futs = [dfs_pool.submit(dfs, d, t, out, a.dfs_threads) for d, t in DFS]
        futs += [
            bl_pool.submit(baseline, d, t, s, out, a.baseline_threads) for d, t, s in BASELINES
        ]
        msgs = [f.result() for f in futs]
    for m in msgs:
        print(m, flush=True)
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
