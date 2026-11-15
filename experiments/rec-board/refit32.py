"""Issue 32: refit the eight recommendation tasks whose runs kept no per-candidate scores, at
their recorded final configs and seeds, keeping each seed's and the seed ensemble's top-k
tables for val and test (:mod:`relconduit.ensemble`). Nothing is tuned: every choice is the one
the original run made on val. The ensemble's test table is written, not scored; ``ensemble32.py``
decides on val, then scores test once per final table.

* ``relconduit.link_big`` (issue 21): rel-amazon user-item-purchase, -rate, -review and rel-hm
  user-item-purchase, each experiment's ``run.py`` plan, seeds 0-2.
* ``refit22.py``'s jobs (issues 12 and 20): rel-avito user-ad-visit and rel-f1
  driver-circuit-compete (seeds 0-2), rel-stack user-post-comment and rel-trial
  condition-sponsor-run (seeds 0-4).

rel-stack post-post-related (issue 29) and rel-trial site-sponsor-run (issue 28) kept their
per-candidate scores, so they need no refit.

Each job leaves ``<out>/<job>/ens/{val,test}_top_{seed<s>,ens}.npy``, its ``summary*.json`` and
``done.json`` (written last; a restarted machine skips the job). A smoke phase (every job on a
few hundred val rows, val tables only) runs first; nothing else starts unless all pass. Jobs
start longest first, at most ``--parallel`` at once, each waiting for ``--free-gb`` free.

    uv run python experiments/rec-board/refit32.py --out runs/refit32
    uv run python experiments/rec-board/refit32.py --job rel-hm/user-item-purchase --work W ...
"""

import argparse
import importlib.util
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]
BIG = [
    "rel-amazon/user-item-purchase",
    "rel-amazon/user-item-rate",
    "rel-amazon/user-item-review",
    "rel-hm/user-item-purchase",
]
REFIT22 = [
    "rel-stack/user-post-comment",
    "rel-trial/condition-sponsor-run",
    "rel-avito/user-ad-visit",
    "rel-f1/driver-circuit-compete",
]
# Longest first (minutes and peak GB of the issue 21 and 22 runs).
ORDER = [
    ("rel-stack/user-post-comment", 129, 75),
    ("rel-amazon/user-item-purchase", 112, 26),
    ("rel-amazon/user-item-rate", 104, 32),
    ("rel-amazon/user-item-review", 85, 26),
    ("rel-trial/condition-sponsor-run", 74, 27),
    ("rel-hm/user-item-purchase", 73, 35),
    ("rel-avito/user-ad-visit", 12, 32),
    ("rel-f1/driver-circuit-compete", 1, 2),
]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"run_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def refit_one(name: str, work: Path, pairs: Path, ens: Path, smoke: int | None) -> dict:
    ds, task = name.split("/")
    if name in REFIT22:
        from refit22 import refit_one as refit22_one

        return refit22_one(name, work, pairs, smoke, score_test=not smoke, ensemble_dir=ens)
    from relconduit.link_big import run_big

    mod = _load(ROOT / "experiments" / ds / task / "cand-lgbm-ranker" / "run.py")
    return run_big(
        ds,
        task,
        f"experiments/{ds}/{task}/cand-lgbm-ranker (refit, issue 32)",
        mod.PLAN,
        mod.FEATURES,
        mod.FEATURES,
        work_dir=work,
        pairs_dir=pairs,
        score_test=not smoke,
        smoke_rows=smoke,
        leakage_seeds=8 if smoke else 40,
        ensemble_dir=ens,
    )


def _driver_job(name, out, pairs_root, threads, smoke, free_gb, delay) -> str:
    from cloud_jobs import run_job, wait_for_memory

    slug = name.replace("/", "_")
    job_name = f"{'smoke' if smoke else 'refit'}_{slug}"
    job = out / job_name
    if free_gb and not (job / "done.json").exists():
        time.sleep(delay * 60)  # let the job before load its database first
        wait_for_memory(job_name, free_gb)
    work = job / "work"
    cmd = [sys.executable, __file__, "--job", name, "--work", str(work)]
    cmd += ["--pairs", str(pairs_root / job_name), "--ens", str(job / "ens")]
    cmd += ["--smoke", "--smoke-rows", str(smoke)] if smoke else []

    def collect(j: Path) -> None:
        for f in work.rglob("summary*.json"):
            shutil.copy(f, j / f.name)

    return run_job(job_name, cmd, out, threads, collect)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job", help="refit this one task in-process (dataset/task)")
    p.add_argument("--work")
    p.add_argument("--pairs")
    p.add_argument("--ens")
    p.add_argument("--out", default="runs/refit32")
    p.add_argument("--pairs-root", default=str(Path.home() / "rc32pairs"))
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--free-gb", type=float, default=40)
    p.add_argument("--stagger", type=float, default=3, help="minutes between job starts")
    p.add_argument("--only", nargs="*")
    p.add_argument("--smoke-rows", type=int, default=200)
    p.add_argument("--no-smoke", action="store_true")
    p.add_argument("--smoke", action="store_true", help="only the smoke phase")
    p.add_argument("--heartbeat", type=float, default=5)
    a = p.parse_args()
    if a.job:
        smoke = a.smoke_rows if a.smoke else None
        s = refit_one(a.job, Path(a.work), Path(a.pairs), Path(a.ens), smoke)
        keys = ("val_scores", "test_scores", "runtime_minutes")
        print(json.dumps({k: s.get(k) for k in keys}), flush=True)
        return

    from cloud_jobs import heartbeat

    out, pairs_root = Path(a.out), Path(a.pairs_root)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n, _, _ in ORDER if not a.only or n in a.only]
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.heartbeat, stop), daemon=True).start()
    if not a.no_smoke:
        with ThreadPoolExecutor(4) as pool:
            futs = [
                pool.submit(_driver_job, n, out / "smoke", pairs_root, 3, a.smoke_rows, 0, 0)
                for n in names
            ]
            msgs = [f.result() for f in futs]
        print("\n".join(msgs), flush=True)
        if any("FAILED" in m for m in msgs):
            sys.exit("smoke phase failed; not starting the real jobs")
        if a.smoke:
            return
    with ThreadPoolExecutor(a.parallel) as pool:
        futs = [
            pool.submit(
                _driver_job,
                n,
                out,
                pairs_root,
                a.threads,
                None,
                a.free_gb,
                (i if i < a.parallel else 0) * a.stagger,
            )
            for i, n in enumerate(names)
        ]
        msgs = [f.result() for f in futs]
    stop.set()
    print("\n".join(msgs), flush=True)
    (out / "done.json").write_text(json.dumps({"jobs": msgs}, indent=2))
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
