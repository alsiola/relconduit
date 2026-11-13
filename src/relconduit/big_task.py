"""Command line shared by the dfs-lightgbm runs on the big regression tasks (issue 8).

Each experiment's ``run.py`` sets its feature config and calls :func:`main`. The protocol is
:func:`relconduit.entity_lgbm.run` with the slim database (:func:`relconduit.db.load_task_db`)
and a fixed uniform sample of the train table. ``--smoke`` runs the code end to end on a few
thousand rows with one config, for checking a machine; it never writes results.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence

from relconduit.entity_lgbm import records, run
from relconduit.features import FeatureConfig
from relconduit.results import append_record, git_commit

TRAIN_SAMPLE = 500_000
PRUNE_TOP_K = (100, 300)
SMOKE_GRID = [
    {
        "num_leaves": 15,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "features": "all",
        "objective": "regression_l1",
    }
]


def main(
    dataset: str,
    task: str,
    features: FeatureConfig,
    notes: str = "",
    db_transform: Callable | None = None,
    prune_top_k: Sequence[int] = PRUNE_TOP_K,
    argv: Sequence[str] | None = None,
) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="score test once, append to results/")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end check; no results")
    p.add_argument("--work-dir", default="runs")
    p.add_argument("--train-sample", type=int, default=TRAIN_SAMPLE)
    a = p.parse_args(argv)
    approach = f"experiments/{dataset}/{task}/dfs-lightgbm"
    kw = {
        "feature_config": features,
        "work_dir": a.work_dir,
        "slim_db": True,
        "db_transform": db_transform,
        # Every test seed sits at the test timestamp, so upto_test_timestamp=False would give
        # the same features: one test row.
        "full_db_variant": False,
    }
    if a.smoke:
        summary = run(
            dataset,
            task,
            approach + " (smoke)",
            seeds=(0,),
            train_sample=3000,
            val_sample=2000,
            grid=SMOKE_GRID,
            score_test=False,
            **kw,
        )
        print(json.dumps({k: summary[k] for k in ("n_features", "val_scores")}), flush=True)
        return summary
    summary = run(
        dataset,
        task,
        approach,
        train_sample=a.train_sample,
        prune_top_k=prune_top_k,
        score_test=a.test,
        **kw,
    )
    out = {k: summary[k] for k in ("best_config", "val_scores", "test_scores")}
    print(json.dumps(out, indent=2), flush=True)
    if a.test:
        n, full = summary["n_train_rows"], summary["n_train_rows_full"]
        sample = f"Trained on a fixed uniform sample of {n:,} of the {full:,} train rows."
        for row in records(summary, commit=git_commit(), notes=f"{sample} {notes}".strip()):
            append_record(f"results/{dataset}/{task}.jsonl", row)
    return summary
