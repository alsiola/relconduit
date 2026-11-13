"""LightGBM on time-safe relational features for a RelBench entity task.

The protocol, in :func:`run`:

1. Build features (:mod:`relconduit.features`) for the train and val tables from
   ``task.get_db(upto_test_timestamp=True)``.
2. For every config in a small fixed grid and each seed: fit on train, with early stopping on
   val. Pick the config with the best mean val score over seeds. The grid includes the
   feature set (with or without hop-2 paths), so feature selection is decided on val too.
3. Only then, score test once with the chosen config, for each seed, through the official
   path (:func:`relbench.submit.write_prediction_table` + ``evaluate_task``): with test
   features from ``get_db(upto_test_timestamp=True)`` (main result) and, with
   ``full_db_variant``, from ``get_db(upto_test_timestamp=False)`` (separate row). Train and
   val features are the same under both: every row before a train or val seed time is before
   the test timestamp.

With ``train_sample``, step 1 uses a fixed uniform sample of that many train rows (drawn with
``numpy.random.default_rng(0)``, the same for every seed and config), as upstream's LightGBM
baseline does with 50,000. With ``slim_db``, the databases come from
:func:`relconduit.db.load_task_db` (the parquet files without free text) instead of
``task.get_db``: same rows and the same features, a fraction of the memory.

With ``prune_top_k``, step 2 runs in two stages instead, for wide feature tables. Stage 1 ranks
the features by total gain of one reference fit (seed 0, :data:`REF_CONFIG`, early stopping on
val), then scores each candidate feature set (``all``, ``no_hop2``, ``no_hop2_all`` when there
are ``>>`` features, and the top ``k`` by gain) with the reference config over every seed, and
keeps the best on val. Stage 2 searches the hyperparameter grid on that feature set.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from relbench.base import Database

from relconduit.entity_encoding import EncodingConfig, EntityEncoder
from relconduit.features import FeatureConfig, build_features, feature_vocab

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
REF_CONFIG = {"num_leaves": 31, "min_child_samples": 50, "feature_fraction": 0.5}
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


def select_columns(
    columns: Sequence[str], feature_set: str, ranking: Sequence[str] | None = None
) -> list[str]:
    """The columns of a named feature set; ``top<k>`` needs ``ranking`` (best first)."""
    if feature_set == "all":
        return list(columns)
    if feature_set == "no_hop2":
        return [c for c in columns if ">" not in c]
    if feature_set == "no_hop2_all":
        return [c for c in columns if ">>" not in c]
    if feature_set.startswith("drop:"):
        # drop:<a>,<b>: every column whose name contains none of the substrings.
        subs = feature_set[5:].split(",")
        return [c for c in columns if not any(s in c for s in subs)]
    if feature_set.startswith("top") and feature_set[3:].isdigit():
        if ranking is None:
            raise ValueError(f"feature set {feature_set!r} needs a ranking")
        keep = set(ranking[: int(feature_set[3:])])
        return [c for c in columns if c in keep]
    raise ValueError(f"unknown feature set {feature_set!r}")


def fit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    config: dict,
    seed: int,
    ranking: Sequence[str] | None = None,
) -> lgb.Booster:
    """Fit one model; ``X_val`` is used for early stopping only."""
    cfg = dict(config)
    cols = select_columns(X_train.columns, cfg.pop("features"), ranking)
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


def _tune(task, configs, data, seeds, ranking=None, label="", cache: Path | None = None):
    """Fit every config for every seed and score it on val.

    With ``cache`` (a JSON-lines file), each finished config is appended to it and configs
    already there are not fitted again, so a restarted run resumes where it stopped.
    """
    X_train, y_train, X_val, y_val, val_table = data
    done = {}
    if cache is not None and cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    out: list[TuneResult] = []
    for config in configs:
        key = label + json.dumps({"config": config, "seeds": list(seeds)}, sort_keys=True)
        if key in done:
            scores, iters = done[key]["val_scores"], done[key]["best_iterations"]
        else:
            scores, iters = [], []
            for seed in seeds:
                model = fit(X_train, y_train, X_val, y_val, config, seed, ranking)
                scores.append(_score(task, predict(model, X_val), val_table))
                iters.append(model.best_iteration)
            if cache is not None:
                row = {"key": key, "val_scores": scores, "best_iterations": iters}
                with open(cache, "a") as f:
                    f.write(json.dumps(row) + "\n")
        out.append(TuneResult(config, scores, iters))
        sd = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        print(f"  {label}{config} val {np.mean(scores):.4f} ± {sd:.4f}", flush=True)
    return out


def gain_ranking(model: lgb.Booster) -> list[str]:
    """Feature names by total split gain, best first (ties by name, for determinism)."""
    gain = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(model.feature_name(), gain), key=lambda p: (-p[1], p[0]))
    return [name for name, _ in pairs]


def run(
    dataset_name: str,
    task_name: str,
    approach: str,
    seeds: Sequence[int] = (0, 1, 2),
    work_dir: str | Path = "runs",
    feature_config: FeatureConfig | None = None,
    score_test: bool = False,
    prune_top_k: Sequence[int] = (),
    full_db_variant: bool = True,
    train_sample: int | None = None,
    slim_db: bool = False,
    grid: Sequence[dict] | None = None,
    val_sample: int | None = None,
    db_transform: Callable[[Database], Database] | None = None,
    encoding: EncodingConfig | None = None,
    extra_feature_sets: Sequence[str] = (),
    smoke_rows: int | None = None,
) -> dict:
    """Tune on val; with ``score_test``, then score test once with the chosen config.

    ``db_transform`` is applied to every database loaded (it must work row by row, e.g. a
    dtype change, or it could leak). ``val_sample`` subsamples val like ``train_sample``: for
    smoke tests only, never for a reported number. ``grid`` replaces :func:`grid_configs` (each config needs ``features``, used only without
    ``prune_top_k``). Returns a summary dict (also written to ``work_dir``). Without
    ``score_test`` the test table's labels are never read, so the grid and the features can be
    iterated on val. Val fits are cached in ``work_dir`` (see :func:`_tune`): delete
    ``tune.jsonl`` and ``ranking.json`` there after changing the features.

    With ``encoding``, :class:`~relconduit.entity_encoding.EntityEncoder` columns (target
    encoding over the train labels only) are added to every split, and ``extra_feature_sets``
    (e.g. ``drop:enc__``) join the feature sets compared on val in stage 1.

    ``smoke_rows`` is for smoke tests: train and val are cut to a random sample of that many
    rows each, the grid to its first config, and test is never touched.
    """
    import relbench
    from relbench.base import Table
    from relbench.submit import evaluate_task, write_prediction_table

    t0 = time.time()
    ds = relbench.load_dataset(f"stanford-star/relbench-v1/{dataset_name}")
    task = ds.load_task(task_name)
    task_type = task.task_type.value
    higher_is_better = task_type != "regression"
    pick = max if higher_is_better else min
    work = Path(work_dir) / dataset_name / task_name / "dfs-lightgbm"
    work.mkdir(parents=True, exist_ok=True)

    tables = {s: task.get_table(s) for s in ("train", "val", "test")}
    n_train_full = len(tables["train"].df)
    if train_sample is not None and train_sample < n_train_full:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n_train_full, size=train_sample, replace=False))
        tables["train"].df = tables["train"].df.iloc[idx].reset_index(drop=True)
    if val_sample is not None and val_sample < len(tables["val"].df):
        rng = np.random.default_rng(1)
        idx = np.sort(rng.choice(len(tables["val"].df), size=val_sample, replace=False))
        tables["val"].df = tables["val"].df.iloc[idx].reset_index(drop=True)
    if smoke_rows:
        if score_test:
            raise ValueError("a smoke run never scores test")
        for s in ("train", "val"):
            t = tables[s]
            n = min(smoke_rows, len(t.df))
            df = t.df.sample(n=n, random_state=0).reset_index(drop=True)
            tables[s] = Table(df, t.fkey_col_to_pkey_table, t.pkey_col, t.time_col)
    key = [task.entity_col, task.time_col]

    def get_db(upto_test_timestamp: bool):
        if slim_db:
            from relconduit.db import load_task_db

            db = load_task_db(task, upto_test_timestamp=upto_test_timestamp)
        else:
            db = task.get_db(upto_test_timestamp=upto_test_timestamp)
        return db if db_transform is None else db_transform(db)

    db_main = get_db(True)
    print(f"db loaded in {time.time() - t0:.0f}s", flush=True)
    # One text vocabulary for every split, so all share one feature set and category codes,
    # read only from rows before the val timestamp (no value first seen in val or test).
    cfg = replace(feature_config or FeatureConfig())
    if cfg.vocab is None:
        cfg.vocab = feature_vocab(db_main, cfg, before=ds.val_timestamp)

    encoders: dict[int, EntityEncoder] = {}

    def features(db, split):
        df = tables[split].df
        X = build_features(db, df[key], task.entity_table, task.entity_col, task.time_col, cfg)
        if encoding is None:
            return X
        if id(db) not in encoders:
            encoders[id(db)] = EntityEncoder(
                db,
                task.entity_table,
                task.entity_col,
                task.time_col,
                encoding,
                labels=tables["train"].df,
                target=task.target_col,
                horizon=task.timedelta,
            )
        return pd.concat([X, encoders[id(db)].transform(df[key])], axis=1)

    X_train, X_val = features(db_main, "train"), features(db_main, "val")
    y_train = tables["train"].df[task.target_col].to_numpy(dtype=float)
    y_val = tables["val"].df[task.target_col].to_numpy(dtype=float)
    t_feat = time.time() - t0
    print(f"features: train {X_train.shape}, val {X_val.shape} in {t_feat:.0f}s", flush=True)
    data = (X_train, y_train, X_val, y_val, tables["val"])
    # Finished fits survive a restart (a Spot machine taken back): see _tune.
    tune_cache = work / "tune.jsonl"

    # --- tune on val ------------------------------------------------------------------
    ranking, stage1 = None, []
    configs = list(grid) if grid is not None else grid_configs(task_type)
    if prune_top_k:
        objective = "regression_l1" if task_type == "regression" else "binary"
        ref = {**REF_CONFIG, "objective": objective}
        ranking_path = work / "ranking.json"
        if ranking_path.exists():
            ranking = json.loads(ranking_path.read_text())
        else:
            ref_all = ref | {"features": "all"}
            ranking = gain_ranking(fit(X_train, y_train, X_val, y_val, ref_all, 0))
            ranking_path.write_text(json.dumps(ranking))
        sets = ["all", "no_hop2"]
        if any(">>" in c for c in X_train.columns):
            sets.append("no_hop2_all")
        sets += list(extra_feature_sets)
        sets += [f"top{k}" for k in prune_top_k if k < X_train.shape[1]]
        stage1 = _tune(
            task, [ref | {"features": f} for f in sets], data, seeds, ranking, "[1] ", tune_cache
        )
        chosen = pick(stage1, key=lambda r: r.mean).config["features"]
        print(f"feature set: {chosen}", flush=True)
        configs = [c | {"features": chosen} for c in configs if c["features"] == "all"]
    if smoke_rows:
        configs = configs[:1]
    results = _tune(task, configs, data, seeds, ranking, cache=tune_cache)
    best = pick(results, key=lambda r: r.mean)
    print(f"best: {best.config} val {best.mean:.4f}", flush=True)

    def _grid(rs):
        return [
            {"config": r.config, "val_scores": r.val_scores, "best_iterations": r.best_iterations}
            for r in rs
        ]

    selected = select_columns(X_train.columns, best.config["features"], ranking)
    summary = {
        "dataset": dataset_name,
        "task": task_name,
        "approach": approach,
        "metric": task.metrics[0].__name__,
        "seeds": list(seeds),
        "n_train_rows": len(X_train),
        "n_train_rows_full": n_train_full,
        "n_features": X_train.shape[1],
        "n_features_selected": len(selected),
        "best_config": best.config,
        "best_iterations": best.best_iterations,
        "val_scores": best.val_scores,
        "test_scores": None,
        "feature_sets": _grid(stage1),
        "grid": _grid(results),
        "top_features": (ranking or [])[:50],
        "feature_minutes": t_feat / 60,
    }
    if not score_test:
        summary["runtime_minutes"] = (time.time() - t0) / 60
        name = "summary_smoke.json" if smoke_rows else "summary_val.json"
        (work / name).write_text(json.dumps(summary, indent=2))
        return summary

    # --- score test once ----------------------------------------------------------------
    X_test = {"upto_test_timestamp=True": features(db_main, "test")}
    if full_db_variant:
        db_full = get_db(False)
        X_test["upto_test_timestamp=False"] = features(db_full, "test")
    test_scores: dict[str, list[float]] = {v: [] for v in X_test}
    for seed in seeds:
        model = fit(X_train, y_train, X_val, y_val, best.config, seed, ranking)
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
