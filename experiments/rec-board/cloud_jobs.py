"""Issue 12's cloud run: the recommendation board's wave 1 (every link task but amazon and hm).

Runs each job as its own process, side by side, on one 24-vCPU machine:

* a smoke phase first: every cand-lgbm-ranker run end to end on a few hundred seeds (its own
  pair leakage check included), all at once. Nothing else starts unless all pass;
* the six cand-lgbm-ranker runs (``--ranker-threads`` each), tuned on val, test scored once;
* upstream's trivial baselines (past visit, global popularity; deterministic, one run each);
* upstream's LightGBM recommendation baseline, seeds 42, 43, 44 (``--upstream-parallel`` at
  a time, ``--upstream-threads`` each).

Each finished job leaves ``<out>/<job>/done.json`` (written last, by rename), so a restarted
machine skips it; ranker jobs also keep finished fits in ``<out>/<job>/work`` and resume from
them (their pair tables go to ``--pairs-root``, outside the checkpoint). Per job: ``log``
(stdout+stderr), ``peak_rss_gb`` in ``done.json``; baselines leave ``pred/`` and
``summary.txt``, rankers their summary JSON and the results rows they appended
(``rows.jsonl``). Every ``--heartbeat`` minutes the driver prints each running job's last log
line, so the cloud-run log shows the jobs are alive.

    uv run python experiments/rec-board/cloud_jobs.py --out runs/cloud12
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

UPSTREAM = Path("experiments/rel-f1/baselines/upstream")
SEEDS = (42, 43, 44)
# Longest first.
TASKS = [
    ("rel-avito", "user-ad-visit"),
    ("rel-stack", "user-post-comment"),
    ("rel-trial", "site-sponsor-run"),
    ("rel-stack", "post-post-related"),
    ("rel-trial", "condition-sponsor-run"),
    ("rel-f1", "driver-circuit-compete"),
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


def run_job(name, cmd, out: Path, threads: int, collect, cwd=None) -> str:
    job = out / name
    if (job / "done.json").exists():
        log(f"{name}: already done")
        return f"{name}: already done"
    job.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "RELCONDUIT_DUCKDB_THREADS": str(threads),
        "PYTHONUNBUFFERED": "1",
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
            cwd=cwd,
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


def _summary_lines(job: Path) -> None:
    keep = ("Train:", "Val:", "Test:", "past_visit:", "global_popularity:")
    lines = [ln for ln in (job / "log").read_text().splitlines() if ln.startswith(keep)]
    (job / "summary.txt").write_text("\n".join(lines) + "\n")


def trivial(ds: str, task: str, out: Path) -> str:
    name = f"trivial_{ds}_{task}"
    pred = (out / name / "pred").resolve()
    cmd = [sys.executable, "trivial_recommendation.py", "--dataset", ds, "--task", task]
    cmd += ["--pred_dir", str(pred)]
    return run_job(name, cmd, out, 1, _summary_lines, cwd=UPSTREAM)


def upstream_lgbm(ds: str, task: str, seed: int, out: Path, threads: int) -> str:
    name = f"lgbm_{ds}_{task}_s{seed}"
    cmd = [sys.executable, "lightgbm_recommendation.py", "--dataset", ds, "--task", task]
    cmd += ["--seed", str(seed), "--cache_dir", str(Path.home() / "bl_cache" / name)]
    cmd += ["--pred_dir", str((out / name / "pred").resolve())]
    return run_job(name, cmd, out, threads, _summary_lines, cwd=UPSTREAM)


def ranker(ds, task, out: Path, pairs_root: Path, threads: int, smoke: int | None = None):
    name = f"{'smoke' if smoke else 'ranker'}_{ds}_{task}"
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

    return run_job(name, cmd, out, threads, collect)


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
    p.add_argument("--out", default="runs/cloud12")
    p.add_argument("--pairs-root", default=str(Path.home() / "rc12pairs"))
    p.add_argument("--ranker-threads", type=int, default=4)
    p.add_argument("--upstream-threads", type=int, default=2)
    p.add_argument("--upstream-parallel", type=int, default=3)
    p.add_argument("--smoke-rows", type=int, default=150)
    p.add_argument("--only", nargs="*", help="run only jobs whose name contains one of these")
    p.add_argument("--heartbeat", type=float, default=5)
    p.add_argument("--smoke", action="store_true", help="only the smoke phase")
    p.add_argument("--no-smoke", action="store_true", help="skip the smoke phase")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_root = Path(a.pairs_root)

    def wanted(name: str) -> bool:
        return not a.only or any(s in name for s in a.only)

    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.heartbeat, stop), daemon=True).start()
    if not a.no_smoke:
        with ThreadPoolExecutor(len(TASKS)) as pool:
            futs = [
                pool.submit(ranker, d, t, out / "smoke", pairs_root, 4, a.smoke_rows)
                for d, t in TASKS
            ]
            msgs = [f.result() for f in futs]
        if any("FAILED" in m for m in msgs):
            sys.exit("smoke phase failed; not starting the real jobs:\n" + "\n".join(msgs))
        if a.smoke:
            return
    with (
        ThreadPoolExecutor(len(TASKS)) as ranker_pool,
        ThreadPoolExecutor(2) as trivial_pool,
        ThreadPoolExecutor(a.upstream_parallel) as upstream_pool,
    ):
        futs = [
            ranker_pool.submit(ranker, d, t, out, pairs_root, a.ranker_threads)
            for d, t in TASKS
            if wanted(f"ranker_{d}_{t}")
        ]
        futs += [
            trivial_pool.submit(trivial, d, t, out) for d, t in TASKS if wanted(f"trivial_{d}_{t}")
        ]
        futs += [
            upstream_pool.submit(upstream_lgbm, d, t, s, out, a.upstream_threads)
            for d, t in TASKS
            for s in SEEDS
            if wanted(f"lgbm_{d}_{t}_s{s}")
        ]
        msgs = [f.result() for f in futs]
    stop.set()
    print("\n".join(msgs), flush=True)
    (out / "done.json").write_text(json.dumps({"jobs": msgs}, indent=2))
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
