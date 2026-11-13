"""Issue 8's cloud run: the big regression tasks' baselines and dfs-lightgbm runs.

Runs each job as its own process, side by side, on one 24-vCPU machine, in three pools:

* the four dfs-lightgbm runs (``--dfs-threads`` each, DuckDB capped at ``--duckdb-memory``);
* the rel-amazon baseline seeds, ``--amazon-parallel`` at a time: upstream's script reads the
  whole database into pandas twice (``task.get_db()`` and ``dataset.get_db()``), review text
  included, and rel-amazon's review table alone is about 25 GB in pandas;
* the rel-hm and rel-stack baseline seeds.

Each finished job leaves ``<out>/<job>/done.json`` (written last, by rename), so a restarted
machine skips it; dfs jobs also keep their finished val fits in ``<out>/<job>/work`` and
resume from them. Per job: ``log`` (stdout+stderr), ``peak_rss_gb`` in ``done.json``, and for
the baselines ``pred/`` and ``summary.txt``, for dfs-lightgbm the summary JSON and the
results rows it appended (``rows.jsonl``). Every ``--heartbeat`` minutes the driver prints
each running job's last log line, so the cloud-run log shows the jobs are alive.

    uv run python experiments/regression-board/cloud_jobs.py --out runs/cloud8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UPSTREAM = Path("experiments/rel-f1/baselines/upstream/lightgbm_entity.py")
SEEDS = (42, 43, 44)
AMAZON = [("rel-amazon", "user-ltv"), ("rel-amazon", "item-ltv")]
SMALL = [("rel-hm", "item-sales"), ("rel-stack", "post-votes")]
# Longest first.
DFS = [
    ("rel-amazon", "user-ltv"),
    ("rel-hm", "item-sales"),
    ("rel-amazon", "item-ltv"),
    ("rel-stack", "post-votes"),
]
# Reports the job's peak resident memory (the largest process it waited on), in GB.
PEAK = (
    "import resource, subprocess, sys; rc = subprocess.call(sys.argv[1:]); "
    "m = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss; "
    "gb = m / 1e9 if sys.platform == 'darwin' else m / 1e6; "  # bytes on macOS, KB on Linux
    "print(f'peak_rss_gb {gb:.2f}', flush=True); sys.exit(rc)"
)
RUNNING: dict[str, Path] = {}
LOCK = threading.Lock()


def log(msg: str) -> None:
    print(f"[jobs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_job(name: str, cmd: list[str], out: Path, threads: int, collect, env_extra=None) -> str:
    job = out / name
    if (job / "done.json").exists():
        log(f"{name}: already done")
        return f"{name}: already done"
    job.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTHONUNBUFFERED": "1",
        **(env_extra or {}),
    }
    log(f"{name}: started ({threads} threads)")
    t0 = time.time()
    with LOCK:
        RUNNING[name] = job / "log"
    with open(job / "log", "w") as f:
        rc = subprocess.run(
            [sys.executable, "-c", PEAK, *cmd],
            check=False,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        ).returncode
    with LOCK:
        RUNNING.pop(name, None)
    minutes = (time.time() - t0) / 60
    lines = (job / "log").read_text(errors="replace").splitlines()
    peak = next((ln.split()[1] for ln in reversed(lines) if ln.startswith("peak_rss_gb")), None)
    if rc != 0:
        tail = "\n    ".join(lines[-15:])
        msg = f"{name}: FAILED rc={rc} after {minutes:.1f} min, peak {peak} GB\n    {tail}"
        log(msg)
        return msg
    collect(job)
    tmp = job / "done.json.tmp"
    tmp.write_text(json.dumps({"minutes": round(minutes, 2), "peak_rss_gb": peak, "cmd": cmd}))
    tmp.rename(job / "done.json")
    log(f"{name}: done in {minutes:.1f} min, peak {peak} GB")
    return f"{name}: done in {minutes:.1f} min, peak {peak} GB"


def baseline(ds: str, task: str, seed: int, out: Path, threads: int) -> str:
    name = f"baseline_{ds}_{task}_s{seed}"
    cmd = [
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
        str(out / name / "pred"),
    ]

    def collect(job: Path) -> None:
        lines = [
            ln
            for ln in (job / "log").read_text().splitlines()
            if ln.startswith(("Train:", "Val:", "Test:"))
        ]
        (job / "summary.txt").write_text("\n".join(lines) + "\n")

    return run_job(name, [sys.executable, *cmd], out, threads, collect)


def dfs(ds: str, task: str, out: Path, threads: int, duckdb_memory: str, test: bool) -> str:
    name = f"dfs_{ds}_{task}"
    work = out / name / "work"
    cmd = [sys.executable, f"experiments/{ds}/{task}/dfs-lightgbm/run.py", "--work-dir", str(work)]
    if test:
        cmd.append("--test")

    def collect(job: Path) -> None:
        rows = Path(f"results/{ds}/{task}.jsonl")
        if test and rows.exists():
            shutil.copy(rows, job / "rows.jsonl")
        for f in (work / ds / task / "dfs-lightgbm").glob("*.json"):
            shutil.copy(f, job / f.name)

    env = {"RELCONDUIT_DUCKDB_THREADS": str(threads), "RELCONDUIT_DUCKDB_MEMORY": duckdb_memory}
    return run_job(name, cmd, out, threads, collect, env)


def smoke(out: Path, threads: int) -> bool:
    """Each dfs run end to end on a few thousand rows (``--smoke``), one after another."""
    ok = True
    for d, t in DFS:
        cmd = [sys.executable, f"experiments/{d}/{t}/dfs-lightgbm/run.py", "--smoke"]
        cmd += ["--work-dir", str(out / "smoke" / d / t)]
        msg = run_job(f"smoke_{d}_{t}", cmd, out / "smoke", threads, lambda job: None)
        ok = ok and "FAILED" not in msg
    return ok


def heartbeat(minutes: float, stop: threading.Event) -> None:
    while not stop.wait(minutes * 60):
        with LOCK:
            running = dict(RUNNING)
        for name, path in running.items():
            try:
                last = path.read_text(errors="replace").strip().splitlines()[-1][:200]
            except (OSError, IndexError):
                last = "(no output yet)"
            log(f"  {name}: {last}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cloud8")
    p.add_argument("--dfs-threads", type=int, default=5)
    p.add_argument("--duckdb-memory", default="10GB")
    p.add_argument("--baseline-threads", type=int, default=2)
    p.add_argument("--amazon-parallel", type=int, default=1)
    p.add_argument("--small-parallel", type=int, default=2)
    p.add_argument("--no-test", action="store_true", help="dfs: tune on val only")
    p.add_argument("--only", nargs="*", help="run only jobs whose name contains one of these")
    p.add_argument("--heartbeat", type=float, default=10)
    p.add_argument("--smoke", action="store_true", help="only the smoke phase")
    p.add_argument("--no-smoke", action="store_true", help="skip the smoke phase")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Smoke phase: a minute or two that catches a broken dfs job before hours are paid for.
    if not a.no_smoke and not smoke(out, a.dfs_threads):
        sys.exit("smoke phase failed; not starting the real jobs")
    if a.smoke:
        return

    def wanted(name: str) -> bool:
        return not a.only or any(s in name for s in a.only)

    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.heartbeat, stop), daemon=True).start()
    with (
        ThreadPoolExecutor(len(DFS)) as dfs_pool,
        ThreadPoolExecutor(a.amazon_parallel) as amazon_pool,
        ThreadPoolExecutor(a.small_parallel) as small_pool,
    ):
        futs = [
            dfs_pool.submit(dfs, d, t, out, a.dfs_threads, a.duckdb_memory, not a.no_test)
            for d, t in DFS
            if wanted(f"dfs_{d}_{t}")
        ]
        for pool, tasks in ((amazon_pool, AMAZON), (small_pool, SMALL)):
            futs += [
                pool.submit(baseline, d, t, s, out, a.baseline_threads)
                for d, t in tasks
                for s in SEEDS
                if wanted(f"baseline_{d}_{t}_s{s}")
            ]
        msgs = [f.result() for f in futs]
    stop.set()
    print("\n".join(msgs), flush=True)
    (out / "done.json").write_text(json.dumps({"jobs": msgs}, indent=2))
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
