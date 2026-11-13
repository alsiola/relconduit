"""Candidate generation + a LightGBM ranker for a RelBench recommendation task.

The protocol, in :func:`run`:

1. Candidates and pair features for the train, val and test seed rows
   (:class:`relconduit.link.PairBuilder`), from ``task.get_db()`` (rows up to the test
   timestamp; every feature reads rows strictly before its seed time). Entity features of each
   side from :func:`relconduit.features.build_features`: the src entity per seed row, the dst
   entity per distinct ``(dst, seed time)``. Text vocabularies come from rows before the val
   timestamp only.
2. Train rows are a fixed sample of the train table (``numpy.random.default_rng(0)``). A pair's
   label is 1 when its dst is in the seed row's true list. Train seeds with no true dst among
   their candidates are dropped (they carry no ranking signal).
3. For each config in a small grid and each seed: fit on train with early stopping on val
   (LightGBM ``map@eval_k`` over val seeds that have a true candidate). Score the val table
   with ``task.evaluate`` (official MAP@k, every val row). Pick the best mean over seeds.
4. Only then, score test once per seed with the chosen config through
   :func:`relbench.submit.write_prediction_table` + ``evaluate_task``.

Rows with fewer than ``eval_k`` candidates are padded with ``-1`` (never a true dst).

Before any fitting, :func:`relconduit.link.pair_leakage_mismatches` is run on a sample of val
and train seeds of the real database (full vs truncated at each seed time), and the run stops
if it finds any difference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from relconduit.features import FeatureConfig, build_features, feature_vocab
from relconduit.link import (
    DST,
    SEED,
    LinkConfig,
    PairBuilder,
    candidate_recall,
    label_pairs,
    pair_leakage_mismatches,
)

BASE_PARAMS = {
    "learning_rate": 0.05,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}
GRID = {
    "objective": ["lambdarank", "binary"],
    "num_leaves": [15, 63],
    "min_child_samples": [20, 100],
    "feature_fraction": [0.7],
}
MAX_ROUNDS = 2000
EARLY_STOPPING = 50


def grid_configs() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def _group_sizes(seed_col: np.ndarray) -> np.ndarray:
    """Sizes of consecutive runs of equal seeds (pairs are sorted by seed)."""
    change = np.flatnonzero(np.diff(seed_col)) + 1
    bounds = np.concatenate([[0], change, [len(seed_col)]])
    return np.diff(bounds)


def _with_positive(pairs: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Mask of pairs whose seed has at least one true candidate."""
    has = pd.Series(y, index=pairs[SEED].to_numpy()).groupby(level=0).transform("max")
    return has.to_numpy() > 0


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in (SEED, DST)]


def fit(train, val, config: dict, seed: int, eval_k: int) -> lgb.Booster:
    """``train``/``val`` are ``(X, y)`` with ``X`` sorted by seed; val stops early only."""
    (X_tr, y_tr), (X_va, y_va) = train, val
    cols = feature_columns(X_tr)
    params = {
        **BASE_PARAMS,
        **config,
        "metric": "map",
        "eval_at": [eval_k],
        "seed": seed,
        "deterministic": True,
    }
    if config["objective"] == "lambdarank":
        params["lambdarank_truncation_level"] = max(2 * eval_k, 20)
    dtr = lgb.Dataset(X_tr[cols], y_tr, group=_group_sizes(X_tr[SEED].to_numpy()))
    dva = lgb.Dataset(X_va[cols], y_va, group=_group_sizes(X_va[SEED].to_numpy()), reference=dtr)
    return lgb.train(
        params,
        dtr,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )


def top_k(X: pd.DataFrame, score: np.ndarray, n_rows: int, k: int) -> np.ndarray:
    """``(n_rows, k)`` dst ids, best first per seed row; ``-1`` pads short rows."""
    df = pd.DataFrame({SEED: X[SEED].to_numpy(), DST: X[DST].to_numpy(), "s": score})
    df = df.sort_values([SEED, "s", DST], ascending=[True, False, True], kind="mergesort")
    df["r"] = df.groupby(SEED).cumcount()
    df = df[df["r"] < k]
    out = np.full((n_rows, k), -1, dtype=np.int64)
    out[df[SEED].to_numpy(), df["r"].to_numpy()] = df[DST].to_numpy()
    return out


def _evaluate(task, pred: np.ndarray, table) -> float:
    (value,) = task.evaluate(pred, table).values()
    return float(value)


def run(
    dataset_name: str,
    task_name: str,
    approach: str,
    link_config: LinkConfig,
    src_features: FeatureConfig | None = None,
    dst_features: FeatureConfig | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    train_sample: int | None = None,
    grid: Sequence[dict] | None = None,
    score_test: bool = False,
    work_dir: str | Path = "runs",
    smoke_rows: int | None = None,
    leakage_seeds: int = 40,
    pairs_dir: str | Path | None = None,
) -> dict:
    """Tune on val; with ``score_test``, then score test once. Returns a summary dict.

    ``src_features``/``dst_features`` ``None`` leaves that side's entity features out.
    ``smoke_rows`` cuts train and val to that many seed rows, the grid to one config and the
    seeds to one, and never touches test. Finished fits are cached in ``work_dir`` (models and
    ``tune.jsonl``), as are the built pairs (in ``pairs_dir`` if given: they can be large), so
    a restarted run resumes.
    """
    import relbench
    from relbench.submit import evaluate_task, write_prediction_table

    t0 = time.time()
    ds = relbench.load_dataset(f"stanford-star/relbench-v1/{dataset_name}")
    task = ds.load_task(task_name)
    k = task.eval_k
    src_col, dst_col, time_col = task.src_entity_col, task.dst_entity_col, task.time_col
    work = Path(work_dir) / dataset_name / task_name / "cand-lgbm-ranker"
    if smoke_rows:
        work = work / "smoke"
        if score_test:
            raise ValueError("a smoke run never scores test")
    work.mkdir(parents=True, exist_ok=True)
    pairs_dir = Path(pairs_dir) if pairs_dir is not None else work
    pairs_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "train": task.get_table("train", mask_input_cols=False).df,
        "val": task.get_table("val", mask_input_cols=False).df,
    }
    n_train_full = len(tables["train"])
    if train_sample is not None and train_sample < n_train_full:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n_train_full, size=train_sample, replace=False))
        tables["train"] = tables["train"].iloc[idx].reset_index(drop=True)
    if smoke_rows:
        for s in ("train", "val"):
            n = min(smoke_rows, len(tables[s]))
            tables[s] = tables[s].sample(n=n, random_state=0).reset_index(drop=True)
        grid = list(grid or grid_configs())[:1]
        seeds = tuple(seeds)[:1]
    configs = list(grid) if grid is not None else grid_configs()
    val_table = task.get_table("val", mask_input_cols=False)
    val_table.df = tables["val"]

    db = task.get_db()
    print(f"[{_m(t0)}] db loaded", flush=True)
    builder = PairBuilder(db, link_config)
    print(f"[{_m(t0)}] edge sets: { {n: len(e) for n, e in builder.edges.items()} }", flush=True)

    # --- leakage check on the real database ------------------------------------------------
    leak_rows = []
    for s, n in (("val", leakage_seeds), ("train", leakage_seeds)):
        df = tables[s]
        sample = df.sample(n=min(n, len(df)), random_state=0) if n else df.iloc[:0]
        leak_rows.append(sample[[src_col, time_col]])
    leak_seeds = pd.concat(leak_rows, ignore_index=True)
    n_leak = 0
    if len(leak_seeds):
        bad = pair_leakage_mismatches(db, leak_seeds, src_col, time_col, link_config)
        n_leak = len(bad)
        print(
            f"[{_m(t0)}] pair leakage check on {len(leak_seeds)} seeds "
            f"({leak_seeds[time_col].nunique()} seed times): {n_leak} mismatches",
            flush=True,
        )
        if bad:
            raise RuntimeError(f"pair leakage: {bad[:10]}")

    # --- entity features --------------------------------------------------------------------
    sides = {}
    for side, cfg in (("src", src_features), ("dst", dst_features)):
        if cfg is not None:
            cfg = replace(cfg)
            if cfg.vocab is None:
                cfg.vocab = feature_vocab(db, cfg, before=ds.val_timestamp)
            sides[side] = cfg

    def entity_features(pairs: pd.DataFrame, seed_rows: pd.DataFrame) -> pd.DataFrame:
        X = pairs
        if "src" in sides:
            s = seed_rows[[src_col, time_col]].reset_index(drop=True)
            F = build_features(db, s, task.src_entity_table, src_col, time_col, sides["src"])
            F = F.astype({c: np.float32 for c in F.columns if F[c].dtype == np.float64})
            F.columns = [f"src.{c}" for c in F.columns]
            X = X.join(F, on=SEED)
        if "dst" in sides:
            u = pd.DataFrame(
                {"dst_id": X[DST].to_numpy(), "t": seed_rows[time_col].to_numpy()[X[SEED]]}
            ).drop_duplicates(ignore_index=True)
            F = build_features(db, u, task.dst_entity_table, "dst_id", "t", sides["dst"])
            F = F.astype({c: np.float32 for c in F.columns if F[c].dtype == np.float64})
            F.columns = [f"dst.{c}" for c in F.columns]
            F = pd.concat([u, F], axis=1)
            X = X.assign(t=seed_rows[time_col].to_numpy()[X[SEED]])
            X = X.merge(F, left_on=[DST, "t"], right_on=["dst_id", "t"], how="left")
            X = X.drop(columns=["dst_id", "t"])
        return X

    def split_data(split: str, seed_rows: pd.DataFrame) -> pd.DataFrame:
        path = pairs_dir / f"pairs_{split}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        t1 = time.time()
        pairs = builder.build(seed_rows, src_col, time_col)
        print(
            f"[{_m(t0)}] {split}: {len(pairs):,} pairs for {len(seed_rows):,} seeds "
            f"({time.time() - t1:.0f}s)",
            flush=True,
        )
        X = entity_features(pairs, seed_rows)
        X = X.astype({c: np.float32 for c in X.columns if X[c].dtype == np.float64})
        print(f"[{_m(t0)}] {split}: features {X.shape}", flush=True)
        tmp = path.with_suffix(".tmp")
        X.to_parquet(tmp)
        tmp.rename(path)
        return X

    X_train = split_data("train", tables["train"])
    y_train = label_pairs(X_train, tables["train"], dst_col)
    X_val = split_data("val", tables["val"])
    y_val = label_pairs(X_val, tables["val"], dst_col)

    sources = [c for c in X_val.columns if c.startswith("cand__")]
    recall = {"all": candidate_recall(X_val, tables["val"], dst_col)}
    for c in sources:
        recall[c[len("cand__") :]] = candidate_recall(X_val[X_val[c] > 0], tables["val"], dst_col)
    cand_per_seed = float(X_val.groupby(SEED).size().mean())
    print(
        f"[{_m(t0)}] val candidate recall {json.dumps(recall)}; "
        f"{cand_per_seed:.0f} candidates per seed",
        flush=True,
    )

    keep = _with_positive(X_train, y_train)
    X_tr, y_tr = X_train[keep].reset_index(drop=True), y_train[keep]
    vkeep = _with_positive(X_val, y_val)
    X_va_es, y_va_es = X_val[vkeep].reset_index(drop=True), y_val[vkeep]
    print(
        f"[{_m(t0)}] train {len(X_tr):,} pairs ({X_tr[SEED].nunique():,} seeds with a "
        f"true candidate of {len(tables['train']):,}); positives {y_tr.mean():.4f}",
        flush=True,
    )
    t_feat = time.time() - t0

    # --- tune on val ------------------------------------------------------------------------
    cache = work / "tune.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    models_dir = work / "models"
    models_dir.mkdir(exist_ok=True)
    results = []
    for config in configs:
        scores, iters = [], []
        for seed in seeds:
            key = json.dumps({"config": config, "seed": seed}, sort_keys=True)
            if key in done:
                scores.append(done[key]["val"])
                iters.append(done[key]["best_iteration"])
                continue
            t1 = time.time()
            model = fit((X_tr, y_tr), (X_va_es, y_va_es), config, seed, k)
            score = model.predict(X_val[model.feature_name()], num_iteration=model.best_iteration)
            val = _evaluate(task, top_k(X_val, score, len(tables["val"]), k), val_table)
            model.save_model(models_dir / f"{_slug(key)}.txt", num_iteration=model.best_iteration)
            row = {"key": key, "val": val, "best_iteration": model.best_iteration}
            with open(cache, "a") as f:
                f.write(json.dumps(row) + "\n")
            scores.append(val)
            iters.append(model.best_iteration)
            print(
                f"[{_m(t0)}]   {config} seed {seed}: val {val:.4f} "
                f"({model.best_iteration} rounds, {time.time() - t1:.0f}s)",
                flush=True,
            )
        results.append({"config": config, "val_scores": scores, "best_iterations": iters})
        sd = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        print(f"[{_m(t0)}] {config} val {np.mean(scores):.4f} ± {sd:.4f}", flush=True)
    best = max(results, key=lambda r: np.mean(r["val_scores"]))
    print(f"[{_m(t0)}] best: {best['config']} val {np.mean(best['val_scores']):.4f}", flush=True)

    summary = {
        "dataset": dataset_name,
        "task": task_name,
        "approach": approach,
        "metric": task.metrics[0].__name__,
        "eval_k": k,
        "seeds": list(seeds),
        "n_train_seeds": len(tables["train"]),
        "n_train_seeds_full": n_train_full,
        "n_train_pairs": len(X_tr),
        "n_features": len(feature_columns(X_tr)),
        "val_candidate_recall": recall,
        "val_candidates_per_seed": cand_per_seed,
        "leakage_check_seeds": len(leak_seeds),
        "leakage_mismatches": n_leak,
        "best_config": best["config"],
        "best_iterations": best["best_iterations"],
        "val_scores": best["val_scores"],
        "test_scores": None,
        "grid": results,
        "feature_minutes": t_feat / 60,
    }
    top = lgb.Booster(model_file=str(models_dir / f"{_slug(_key(best['config'], seeds[0]))}.txt"))
    gain = top.feature_importance(importance_type="gain")
    order = np.argsort(-gain)[:30]
    summary["top_features"] = [[top.feature_name()[i], round(float(gain[i]), 1)] for i in order]
    if not score_test:
        summary["runtime_minutes"] = (time.time() - t0) / 60
        name = "summary_smoke.json" if smoke_rows else "summary_val.json"
        (work / name).write_text(json.dumps(summary, indent=2))
        return summary

    # --- score test once ----------------------------------------------------------------------
    test_rows = task.get_table("test").df  # masked: src and time only
    X_test = split_data("test", test_rows)
    test_scores = []
    for seed in seeds:
        model = lgb.Booster(model_file=str(models_dir / f"{_slug(_key(best['config'], seed))}.txt"))
        score = model.predict(X_test[model.feature_name()])
        pred = top_k(X_test, score, len(test_rows), k)
        path = work / "pred" / f"seed{seed}" / f"{dataset_name}__{task_name}.csv"
        write_prediction_table(task, pred, path)
        (value,) = evaluate_task(f"{dataset_name}/{task_name}", path).values()
        test_scores.append(float(value))
        print(f"[{_m(t0)}] test seed {seed}: {value:.4f}", flush=True)
    summary["test_scores"] = test_scores
    summary["runtime_minutes"] = (time.time() - t0) / 60
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _key(config: dict, seed: int) -> str:
    return json.dumps({"config": config, "seed": seed}, sort_keys=True)


def _slug(key: str) -> str:
    import hashlib

    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _m(t0: float) -> str:
    return f"{(time.time() - t0) / 60:6.1f} min"


def records(summary: dict, commit: str, notes: str = "") -> list[dict]:
    """Results rows (val and test) for a :func:`run` summary."""
    from relconduit.results import make_record

    rec = {k: round(v, 4) for k, v in summary["val_candidate_recall"].items()}
    base = (
        f"Candidates (val recall {rec['all']:.3f}, {summary['val_candidates_per_seed']:.0f} per "
        f"seed) + LightGBM ranker on {summary['n_features']} pair and entity features, config "
        f"chosen on val by mean over seeds: {summary['best_config']}. Trained on "
        f"{summary['n_train_seeds']:,} of {summary['n_train_seeds_full']:,} train seed rows; "
        f"early stopping on val. Pair leakage check: {summary['leakage_mismatches']} mismatches "
        f"on {summary['leakage_check_seeds']} seeds."
    )
    if notes:
        base += " " + notes
    common = {
        "approach": summary["approach"],
        "metric": summary["metric"],
        "runtime_minutes": summary["runtime_minutes"],
        "commit": commit,
        "candidate_recall_val": rec,
    }
    return [
        make_record(
            split="val",
            values=summary["val_scores"],
            notes=base + " Val is optimistic: it chose the config and the stopping round.",
            **common,
        ),
        make_record(split="test", values=summary["test_scores"], notes=base, **common),
    ]


def main(
    dataset: str,
    task: str,
    link_config: LinkConfig,
    src_features: FeatureConfig | None,
    dst_features: FeatureConfig | None,
    train_sample: int | None = None,
    notes: str = "",
    argv: Sequence[str] | None = None,
) -> dict:
    """Command line for an experiment's ``run.py``: ``--smoke``, ``--test``, ``--work-dir``."""
    from relconduit.results import append_record, git_commit

    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="score test once, append to results/")
    p.add_argument(
        "--smoke",
        type=int,
        nargs="?",
        const=300,
        default=None,
        help="tiny end-to-end check on this many seeds per split; no results",
    )
    p.add_argument("--work-dir", default="runs")
    p.add_argument("--pairs-dir", help="where the built pairs are cached (default: work dir)")
    p.add_argument("--train-sample", type=int, default=train_sample)
    a = p.parse_args(argv)
    approach = f"experiments/{dataset}/{task}/cand-lgbm-ranker"
    summary = run(
        dataset,
        task,
        approach + (" (smoke)" if a.smoke else ""),
        link_config,
        src_features,
        dst_features,
        train_sample=a.train_sample,
        score_test=a.test and not a.smoke,
        work_dir=a.work_dir,
        smoke_rows=a.smoke,
        leakage_seeds=10 if a.smoke else 40,
        pairs_dir=a.pairs_dir,
    )
    keys = ("val_candidate_recall", "best_config", "val_scores", "test_scores", "top_features")
    print(json.dumps({k: summary[k] for k in keys}, indent=2), flush=True)
    if a.test and not a.smoke:
        for row in records(summary, commit=git_commit(), notes=notes):
            append_record(f"results/{dataset}/{task}.jsonl", row)
    return summary
