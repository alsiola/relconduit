"""Issue 22: refit the six recommendation tasks whose test tables weren't kept, at their recorded
final configs and seeds, for the leaderboard package. A reproduction, not a new experiment: no
choice is made here, nothing is tuned and no results row is appended.

* Wave 1 (issue 12, ``relconduit.link_lgbm``): rel-f1 driver-circuit-compete, rel-trial
  site-sponsor-run, rel-avito user-ad-visit, rel-stack post-post-related. Each one's run.py
  config (candidates, features, train sample) with the grid cut to the config that val chose,
  seeds 0, 1, 2.
* Issue 20 (``relconduit.link_search``): rel-stack user-post-comment and rel-trial
  condition-sponsor-run, ``cand-lgbm-ranker-plus``. The plan cut to the chosen candidate set,
  train size and config, so every stage of the search is that one fit; seeds 0-4.

Each job scores test once per seed (``write_prediction_table`` + ``evaluate_task``), and leaves
``<out>/<job>/summary.json``, its seed-0 table at ``<out>/<job>/<dataset>__<task>.csv`` and
``done.json`` (written last; a restarted machine skips the job). Pair leakage checks run as in
the original runs.

    uv run python experiments/rec-board/refit22.py --out runs/refit22           # all six
    uv run python experiments/rec-board/refit22.py --out runs/refit22 --smoke   # 150-seed check
    uv run python experiments/rec-board/refit22.py --job rel-f1/driver-circuit-compete --work W
"""

import argparse
import importlib.util
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]

# The config val chose, from each task's recorded test row in results/ (notes field).
JOBS = {
    "rel-f1/driver-circuit-compete": {
        "kind": "lgbm",
        "approach": "cand-lgbm-ranker",
        "features": ("FEATURES", "FEATURES"),
        "train_sample": None,
        "config": {
            "objective": "binary",
            "num_leaves": 63,
            "min_child_samples": 20,
            "feature_fraction": 0.7,
        },
        "seeds": (0, 1, 2),
    },
    "rel-trial/site-sponsor-run": {
        "kind": "lgbm",
        "approach": "cand-lgbm-ranker",
        "features": ("SRC_FEATURES", "DST_FEATURES"),
        "train_sample": 25_000,
        "config": {
            "objective": "lambdarank",
            "num_leaves": 63,
            "min_child_samples": 20,
            "feature_fraction": 0.7,
        },
        "seeds": (0, 1, 2),
    },
    "rel-avito/user-ad-visit": {
        "kind": "lgbm",
        "approach": "cand-lgbm-ranker",
        "features": ("FEATURES", "FEATURES"),
        "train_sample": 20_000,
        "config": {
            "objective": "lambdarank",
            "num_leaves": 15,
            "min_child_samples": 20,
            "feature_fraction": 0.7,
        },
        "seeds": (0, 1, 2),
    },
    "rel-stack/post-post-related": {
        "kind": "lgbm",
        "approach": "cand-lgbm-ranker",
        "features": ("FEATURES", "FEATURES"),
        "train_sample": None,
        "config": {
            "objective": "lambdarank",
            "num_leaves": 15,
            "min_child_samples": 100,
            "feature_fraction": 0.7,
        },
        "seeds": (0, 1, 2),
    },
    "rel-stack/user-post-comment": {
        "kind": "search",
        "approach": "cand-lgbm-ranker-plus",
        "features": ("FEATURES", "FEATURES"),
        "variant": "plus",
        "size": None,  # all 21,240 train seed rows
        "config": {
            "objective": "lambdarank",
            "num_leaves": 15,
            "min_child_samples": 20,
            "feature_fraction": 0.7,
            "learning_rate": 0.02,
        },
        "seeds": (0, 1, 2, 3, 4),
    },
    "rel-trial/condition-sponsor-run": {
        "kind": "search",
        "approach": "cand-lgbm-ranker-plus",
        "features": ("SRC_FEATURES", "DST_FEATURES"),
        "variant": "plus",
        "size": None,  # all 36,934 train seed rows
        "config": {
            "objective": "lambdarank",
            "num_leaves": 63,
            "min_child_samples": 100,
            "feature_fraction": 0.7,
            "learning_rate": 0.02,
        },
        "seeds": (0, 1, 2, 3, 4),
    },
}
SUBMISSION_SEED = 0


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"run_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def refit_one(
    name: str, work: Path, pairs: Path, smoke: int | None = None, score_test: bool = True
) -> dict:
    """Refit one job in this process; returns its summary."""
    ds, task = name.split("/")
    job = JOBS[name]
    mod = _load(ROOT / "experiments" / ds / task / job["approach"] / "run.py")
    approach = f"experiments/{ds}/{task}/{job['approach']} (refit, issue 22)"
    if job["kind"] == "lgbm":
        from relconduit.link_lgbm import run

        src, dst = (getattr(mod, a) for a in job["features"])
        return run(
            ds,
            task,
            approach,
            mod.LINK,
            src,
            dst,
            seeds=job["seeds"],
            train_sample=job["train_sample"],
            grid=[job["config"]],
            score_test=score_test,
            work_dir=work,
            smoke_rows=smoke,
            leakage_seeds=10 if smoke else 40,
            pairs_dir=pairs,
        )
    from relconduit.link_search import run_search

    src, dst = (getattr(mod.base, a) for a in job["features"])
    variant = job["variant"]
    plan = replace(
        mod.PLAN,
        variants={variant: mod.PLAN.variants[variant]},
        ref_config=job["config"],
        grid=[job["config"]],
        stage1_rows=job["size"],
        train_sizes=(job["size"],),
        final_seeds=job["seeds"],
        leader=None,  # no verdict or submission check here; the package is checked whole
    )
    return run_search(
        ds,
        task,
        approach,
        plan,
        src,
        dst,
        work_dir=work,
        pairs_dir=pairs,
        score_test=score_test,
        smoke_rows=smoke,
        leakage_seeds=10 if smoke else 40,
        n_boot=200,
    )


def _driver_job(name, out: Path, pairs_root: Path, threads: int, smoke: int | None) -> str:
    from cloud_jobs import run_job

    slug = name.replace("/", "_")
    job_name = f"{'smoke' if smoke else 'refit'}_{slug}"
    work = out / job_name / "work"
    cmd = [sys.executable, __file__, "--job", name, "--work", str(work)]
    cmd += ["--pairs", str(pairs_root / job_name)]
    cmd += ["--smoke", str(smoke)] if smoke else []

    def collect(job: Path) -> None:
        for f in work.rglob("summary*.json"):
            shutil.copy(f, job / f.name)
        ds, task = name.split("/")
        for f in work.rglob(f"seed{SUBMISSION_SEED}/{ds}__{task}.csv"):
            shutil.copy(f, job / f.name)

    return run_job(job_name, cmd, out, threads, collect)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--job", help="refit this one task in-process (dataset/task)")
    p.add_argument("--work", help="with --job: work dir")
    p.add_argument("--pairs", help="with --job: pair cache dir")
    p.add_argument("--out", default="runs/refit22")
    p.add_argument("--pairs-root", default=str(Path.home() / "rc22pairs"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--only", nargs="*", help="driver: only these tasks")
    p.add_argument("--smoke", type=int, nargs="?", const=150, default=None)
    p.add_argument("--heartbeat", type=float, default=5)
    a = p.parse_args()
    if a.job:
        summary = refit_one(a.job, Path(a.work), Path(a.pairs), a.smoke, score_test=not a.smoke)
        keys = ("val_scores", "test_scores", "runtime_minutes")
        print(json.dumps({k: summary.get(k) for k in keys}), flush=True)
        return

    from cloud_jobs import heartbeat

    out, pairs_root = Path(a.out), Path(a.pairs_root)
    if a.smoke:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    names = a.only or list(JOBS)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(a.heartbeat, stop), daemon=True).start()
    with ThreadPoolExecutor(len(names)) as pool:
        futs = [pool.submit(_driver_job, n, out, pairs_root, a.threads, a.smoke) for n in names]
        msgs = [f.result() for f in futs]
    stop.set()
    print("\n".join(msgs), flush=True)
    (out / "done.json").write_text(json.dumps({"jobs": msgs}, indent=2))
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
