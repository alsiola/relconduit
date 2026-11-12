"""LightGBM on time-safe relational features for a RelBench entity task.

The protocol, in :func:`run`:

1. Build features (:mod:`relconduit.features`) for the train and val tables from
   ``task.get_db(upto_test_timestamp=True)``.
2. For every config in a small fixed grid and each seed: fit on train, with early stopping on
   val. Pick the config with the best mean val score over seeds. The grid includes the
   feature set (with or without hop-2 paths), so feature selection is decided on val too.
3. Only then, score test once with the chosen config, for each seed, through the official
   path (:func:`relbench.submit.write_prediction_table` + ``evaluate_task``): with test
   features from ``get_db(upto_test_timestamp=True)`` (main result) and from
   ``get_db(upto_test_timestamp=False)`` (separate row). Train and val features are the
   same under both: every row before a train or val seed time is before the test timestamp.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from relconduit.features import FeatureConfig, build_features

FEATURE_SETS = ("all", "no_hop2")

BASE_PARAMS = {
    "learning_rate": 0.03,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}
GRID = {
    "num_leaves": [15, 63],
    "min_child_samples": [20, 100],
    "feature_fraction": [0.3, 0.8],
    "features": list(FEATURE_SETS),
}
REGRESSION_OBJECTIVES = ["regression", "regression_l1"]
MAX_ROUNDS = 3000
EARLY_STOPPING = 100


def grid_configs(task_type: str) -> list[dict]:
    """The fixed grid, as a list of configs. Regression also searches L2 vs L1 loss."""
    grid = dict(GRID)
    if task_type == "regression":
        grid["objective"] = REGRESSION_OBJECTIVES
    else:
        grid["objective"] = ["binary"]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def select_columns(columns: Sequence[str], feature_set: str) -> list[str]:
    if feature_set == "all":
        return list(columns)
    if feature_set == "no_hop2":
        return [c for c in columns if ">" not in c]
    raise ValueError(f"unknown feature set {feature_set!r}")


def fit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    config: dict,
    seed: int,
) -> lgb.Booster:
    """Fit one model; ``X_val`` is used for early stopping only."""
    cfg = dict(config)
    cols = select_columns(X_train.columns, cfg.pop("features"))
    metric = "auc" if cfg["objective"] == "binary" else "l1"
    params = {**BASE_PARAMS, **cfg, "metric": metric, "seed": seed, "deterministic": True}
    dtrain = lgb.Dataset(X_train[cols], y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val[cols], y_val, reference=dtrain, free_raw_data=False)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )


def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X[model.feature_name()], num_iteration=model.best_iteration)


@dataclass
class TuneResult:
    config: dict
    val_scores: list[float]
    best_iterations: list[int]

    @property
    def mean(self) -> float:
        return float(np.mean(self.val_scores))


def _score(task, pred: np.ndarray, table) -> float:
    (value,) = task.evaluate(pred, table).values()
    return float(value)


def run(
    dataset_name: str,
    task_name: str,
    approach: str,
    seeds: Sequence[int] = (0, 1, 2),
    work_dir: str | Path = "runs",
    feature_config: FeatureConfig | None = None,
    score_test: bool = False,
) -> dict:
    """Tune on val; with ``score_test``, then score test once with the chosen config.

    Returns a summary dict (also written to ``work_dir``). Without ``score_test`` the test
    table's labels are never read, so the grid and the features can be iterated on val.
    """
    import relbench
    from relbench.submit import evaluate_task, write_prediction_table

    t0 = time.time()
    ds = relbench.load_dataset(f"stanford-star/relbench-v1/{dataset_name}")
    task = ds.load_task(task_name)
    task_type = task.task_type.value
    higher_is_better = task_type != "regression"
    work = Path(work_dir) / dataset_name / task_name / "dfs-lightgbm"
    work.mkdir(parents=True, exist_ok=True)

    tables = {s: task.get_table(s) for s in ("train", "val", "test")}
    key = [task.entity_col, task.time_col]

    def features(db, split):
        df = tables[split].df
        return build_features(
            db, df[key], task.entity_table, task.entity_col, task.time_col, feature_config
        )

    db_main = task.get_db(upto_test_timestamp=True)
    X_train, X_val = features(db_main, "train"), features(db_main, "val")
    y_train = tables["train"].df[task.target_col].to_numpy(dtype=float)
    y_val = tables["val"].df[task.target_col].to_numpy(dtype=float)
    t_feat = time.time() - t0
    print(f"features: train {X_train.shape}, val {X_val.shape} in {t_feat:.0f}s", flush=True)

    # --- tune on val ------------------------------------------------------------------
    results: list[TuneResult] = []
    for config in grid_configs(task_type):
        scores, iters = [], []
        for seed in seeds:
            model = fit(X_train, y_train, X_val, y_val, config, seed)
            scores.append(_score(task, predict(model, X_val), tables["val"]))
            iters.append(model.best_iteration)
        results.append(TuneResult(config, scores, iters))
        print(f"  {config} val {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}", flush=True)
    best = (max if higher_is_better else min)(results, key=lambda r: r.mean)
    print(f"best: {best.config} val {best.mean:.4f}", flush=True)

    summary = {
        "dataset": dataset_name,
        "task": task_name,
        "approach": approach,
        "metric": task.metrics[0].__name__,
        "seeds": list(seeds),
        "n_features": X_train.shape[1],
        "n_features_selected": len(select_columns(X_train.columns, best.config["features"])),
        "best_config": best.config,
        "best_iterations": best.best_iterations,
        "val_scores": best.val_scores,
        "test_scores": None,
        "grid": [
            {"config": r.config, "val_scores": r.val_scores, "best_iterations": r.best_iterations}
            for r in results
        ],
        "feature_minutes": t_feat / 60,
    }
    if not score_test:
        summary["runtime_minutes"] = (time.time() - t0) / 60
        (work / "summary_val.json").write_text(json.dumps(summary, indent=2))
        return summary

    # --- score test once ----------------------------------------------------------------
    X_test = {
        "upto_test_timestamp=True": features(db_main, "test"),
        "upto_test_timestamp=False": features(task.get_db(upto_test_timestamp=False), "test"),
    }
    test_scores: dict[str, list[float]] = {v: [] for v in X_test}
    for seed in seeds:
        model = fit(X_train, y_train, X_val, y_val, best.config, seed)
        for variant, X in X_test.items():
            tag = "upto_test" if variant.endswith("True") else "full_db"
            path = work / "pred" / tag / f"seed{seed}" / f"{dataset_name}__{task_name}.csv"
            write_prediction_table(task, predict(model, X), path)
            (value,) = evaluate_task(f"{dataset_name}/{task_name}", path).values()
            test_scores[variant].append(float(value))
    summary["test_scores"] = test_scores
    summary["runtime_minutes"] = (time.time() - t0) / 60
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def records(summary: dict, commit: str, notes: str = "") -> list[dict]:
    """Results rows for a :func:`run` summary: val, test (main) and test (full db)."""
    from relconduit.results import make_record

    cfg = summary["best_config"]
    base = (
        f"LightGBM on relconduit.features, config chosen on val by mean over seeds: {cfg}; "
        f"{summary['n_features_selected']} features. Train on train only; early stopping on val."
    )
    if notes:
        base += " " + notes
    common = {
        "approach": summary["approach"],
        "metric": summary["metric"],
        "runtime_minutes": summary["runtime_minutes"],
        "commit": commit,
    }
    rows = [
        make_record(
            split="val",
            values=summary["val_scores"],
            notes=base + " Val is optimistic: it chose the config and the stopping round.",
            db="upto_test_timestamp=True",
            **common,
        )
    ]
    for variant, values in summary["test_scores"].items():
        main = variant.endswith("True")
        label = "Main result." if main else "Leaderboard-allowed variant, test features only."
        rows.append(
            make_record(split="test", values=values, notes=f"{base} {label}", db=variant, **common)
        )
    return rows
