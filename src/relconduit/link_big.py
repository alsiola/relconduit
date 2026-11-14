"""Candidates + a LightGBM ranker for the big recommendation tasks (rel-amazon, rel-hm).

:mod:`relconduit.link_lgbm` and :mod:`relconduit.link_search` hold a split's pairs in memory,
which rel-amazon's val and test (350k-390k seed rows at one seed time, 100+ candidates each)
don't allow. This module runs the same kind of pipeline at that scale (issue 21):

1. **Database**: :func:`relconduit.db.load_task_db` (parquet, no free text), with any extra
   options the plan gives (e.g. ``text_lengths``).
2. **Leakage**: :func:`relconduit.link.pair_leakage_mismatches` (full vs truncated at each seed
   time) on a sample of val and train seeds of the real database, before anything is built or
   fitted. The run stops on any mismatch.
3. **Pairs**, a chunk of seeds at a time (:func:`seed_chunks`: seed times in order, a big seed
   time split into pieces; a seed's pairs depend only on its own src and time). Entity
   features: the src's per seed row, the dst's per distinct ``(dst, seed time)``, built once per
   seed time for every dst seen so far. Each chunk is written to parquet (``float32``;
   categorical features as their codes over the fixed vocabulary) and skipped on a restart.
   Val and test are built on **every** row.
4. **Train rows**: :attr:`BigPlan.train_rows` seed rows sampled (``default_rng(0)``) from the
   latest :attr:`BigPlan.train_times` train seed times. Seeds with no true dst among their
   candidates are dropped (no ranking signal).
5. **Tuning on val**: every config of :attr:`BigPlan.grid` is fitted with seed 0, early
   stopping on the pairs of :attr:`BigPlan.es_seeds` val seeds that have a true candidate. Each
   fit is scored with ``task.evaluate`` on **all** val rows (scored chunk by chunk, top
   ``eval_k`` kept). The best config is fitted with the other seeds.
6. **Test, once**, with the chosen config, per seed: ``write_prediction_table`` +
   ``evaluate_task``. The seed-0 table is copied to ``<work>/submission/`` and checked with
   :func:`relbench.submit.evaluate_submission` (the per-task check of ``python -m
   relbench.submit``).

Fits are cached (``<work>/tune.jsonl`` and models), so a restarted machine resumes.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
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
    label_pairs,
    pair_leakage_mismatches,
)
from relconduit.link_lgbm import _slug, _with_positive, add_entity_features, top_k
from relconduit.link_search import _dataset, _params, _rss_gb, pick_best

LABEL = "__y"

_REF = {
    "objective": "lambdarank",
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
}
# The small grid of issue 21: the reference, the binary objective, a bigger tree.
GRID = [
    _REF,
    {**_REF, "objective": "binary"},
    {**_REF, "num_leaves": 127, "min_child_samples": 300},
]


@dataclass
class BigPlan:
    """What :func:`run_big` does for one task (see the module doc). Configs are LightGBM params
    on top of :data:`relconduit.link_lgbm.BASE_PARAMS`; ``objective`` is required."""

    link: LinkConfig
    grid: Sequence[dict]
    train_rows: int
    train_times: int | None = None  # sample from the latest this-many seed times (None: all)
    seeds: Sequence[int] = (0, 1, 2)
    es_seeds: int = 20_000
    chunk_rows: int = 20_000
    leader: tuple[str, float] | None = None
    max_rounds: int = 3000
    early_stopping: int = 100
    db_kwargs: dict = field(default_factory=dict)
    # ``(task, **db_kwargs) -> Database``; default :func:`relconduit.db.load_task_db`.
    db_loader: Callable | None = None
    extra: dict = field(default_factory=dict)


# --- pure helpers (tested) ---------------------------------------------------------------


def seed_chunks(times: pd.Series, max_rows: int) -> list[np.ndarray]:
    """Positions of ``times`` in chunks of at most ``max_rows``: seed times in order, each one
    split into consecutive pieces; a chunk never mixes seed times."""
    t = pd.to_datetime(times).to_numpy()
    out = []
    for u in np.unique(t):
        idx = np.flatnonzero(t == u)
        out.extend(idx[i : i + max_rows] for i in range(0, len(idx), max_rows))
    return out


def sample_train(times: pd.Series, n: int, last_times: int | None, seed: int = 0) -> np.ndarray:
    """Sorted positions of ``n`` rows drawn without replacement from the rows at the latest
    ``last_times`` seed times (all of them if ``None`` or fewer rows than ``n``)."""
    t = pd.to_datetime(times).to_numpy()
    pool = np.arange(len(t))
    if last_times is not None:
        keep = np.unique(t)[-last_times:]
        pool = np.flatnonzero(np.isin(t, keep))
    if n >= len(pool):
        return pool
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


def fill_top_k(out: np.ndarray, X: pd.DataFrame, score: np.ndarray) -> None:
    """Write each seed's top ``out.shape[1]`` dsts of ``X`` (by ``score``, ties by lower dst)
    into its row of ``out``; rows of seeds absent from ``X`` are left as they are."""
    if not len(X):
        return
    seeds = np.unique(X[SEED].to_numpy())
    local = pd.Series(np.arange(len(seeds)), index=seeds)
    Xl = pd.DataFrame({SEED: local.loc[X[SEED].to_numpy()].to_numpy(), DST: X[DST].to_numpy()})
    out[seeds] = top_k(Xl, score, len(seeds), out.shape[1])


def recall_from_hits(hits: np.ndarray, n_true: np.ndarray) -> float:
    """Mean over seed rows of the share of their true dsts found (rows with none count 0)."""
    n = np.asarray(n_true, dtype=float)
    return float(np.mean(np.where(n > 0, hits / np.maximum(n, 1), 0.0)))


def to_codes(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns as float32 codes (NaN for missing), so chunks written separately
    agree: their categories come from the fixed feature vocabulary."""
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            codes = X[c].cat.codes.to_numpy().astype(np.float32)
            X[c] = np.where(codes < 0, np.nan, codes).astype(np.float32)
        elif X[c].dtype == np.float64 or pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            raise TypeError(f"feature {c} is free text; give it a vocabulary or leave it out")
    return X


# --- the run -----------------------------------------------------------------------------


def _m(t0: float) -> str:
    return f"{(time.time() - t0) / 60:6.1f} min"


class _DstFeatures:
    """The dst entity's features at one seed time, built for every dst seen so far."""

    def __init__(self, db, task, cfg: FeatureConfig):
        self.db, self.task, self.cfg = db, task, cfg
        self.t, self.F = None, None

    def get(self, dsts: np.ndarray, t: pd.Timestamp) -> pd.DataFrame:
        if self.t != t:
            self.t, self.F = t, None
        have = self.F.index.to_numpy() if self.F is not None else np.zeros(0, dtype=np.int64)
        missing = np.setdiff1d(np.unique(dsts), have)
        if len(missing):
            u = pd.DataFrame(
                {"dst_id": missing, "t": np.repeat(np.datetime64(t, "ns"), len(missing))}
            )
            F = build_features(self.db, u, self.task.dst_entity_table, "dst_id", "t", self.cfg)
            F = to_codes(F)
            F.columns = [f"dst.{c}" for c in F.columns]
            F.index = pd.Index(missing, name=DST)
            self.F = F if self.F is None else pd.concat([self.F, F])
        return self.F


class _Pairs:
    """Pair chunks of one split, built and cached as parquet."""

    def __init__(self, ctx, builder: PairBuilder, dst_features: _DstFeatures | None):
        self.c, self.builder, self.dst = ctx, builder, dst_features

    def build(self, split: str, rows: pd.DataFrame, labels: bool, positive_only: bool):
        """Paths of the chunk files of ``rows`` (``seed`` = position in ``rows``)."""
        c = self.c
        paths = []
        chunks = seed_chunks(rows[c.time_col], c.plan.chunk_rows)
        for i, idx in enumerate(chunks):
            path = c.pairs_dir / f"{split}_{i:04d}.parquet"
            paths.append(path)
            if path.exists():
                continue
            t1 = time.time()
            sub = rows.iloc[idx].reset_index(drop=True)
            pairs = self.builder.build(sub, c.src_col, c.time_col)
            n_pairs = len(pairs)
            if labels:
                y = label_pairs(pairs, sub, c.dst_col)
                if positive_only:
                    keep = _with_positive(pairs, y)
                    pairs, y = pairs[keep].reset_index(drop=True), y[keep]
            X = pairs
            if "src" in c.sides:
                X = add_entity_features(c.db, c.task, X, sub, {"src": c.sides["src"]})
            if self.dst is not None:
                t = pd.Timestamp(sub[c.time_col].iloc[0])
                F = self.dst.get(X[DST].to_numpy(), t)
                X = X.join(F, on=DST)
            X = to_codes(X)
            if labels:
                X[LABEL] = y
            X[SEED] = idx[X[SEED].to_numpy()]
            tmp = path.with_suffix(".tmp")
            X.to_parquet(tmp)
            tmp.rename(path)
            c.log(
                f"{split} chunk {i + 1}/{len(chunks)}: {len(sub):,} seeds, {n_pairs:,} pairs, "
                f"kept {len(X):,} x {X.shape[1]} ({time.time() - t1:.0f}s, "
                f"peak rss {_rss_gb():.1f} GB)"
            )
        return paths


@dataclass
class _Ctx:
    task: object
    db: object
    plan: BigPlan
    sides: dict
    pairs_dir: Path
    src_col: str
    dst_col: str
    time_col: str
    t0: float

    def log(self, msg: str) -> None:
        print(f"[{_m(self.t0)}] {msg}", flush=True)


def _features(paths: Sequence[Path]) -> list[str]:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(paths[0]).schema_arrow.names
    return [c for c in names if c not in (SEED, DST, LABEL) and not c.startswith("__index")]


def _concat(paths: Sequence[Path], seeds: np.ndarray | None = None) -> pd.DataFrame:
    parts = []
    for p in paths:
        X = pd.read_parquet(p)
        if seeds is not None:
            X = X[np.isin(X[SEED].to_numpy(), seeds)]
        parts.append(X)
    # Chunks hold disjoint seeds, each sorted by (seed, dst): their concatenation keeps every
    # seed's pairs together, which is all LightGBM's groups need.
    return pd.concat(parts, ignore_index=True)


def _predict_top(models, paths, cols, n_rows: int, k: int) -> list[np.ndarray]:
    """Each model's ``(n_rows, k)`` predictions over the chunk files, read once."""
    out = [np.full((n_rows, k), -1, dtype=np.int64) for _ in models]
    for p in paths:
        X = pd.read_parquet(p, columns=[SEED, DST, *cols])
        for m, pred in zip(models, out, strict=True):
            fill_top_k(pred, X, m.predict(X[cols]))
    return out


def run_big(
    dataset_name: str,
    task_name: str,
    approach: str,
    plan: BigPlan,
    src_features: FeatureConfig | None,
    dst_features: FeatureConfig | None,
    work_dir: str | Path = "runs",
    pairs_dir: str | Path | None = None,
    score_test: bool = False,
    smoke_rows: int | None = None,
    leakage_seeds: int = 40,
) -> dict:
    """Run the pipeline (see the module doc); returns the summary, also written to
    ``<work>/summary.json`` (``summary_val.json`` without test, ``summary_smoke.json`` for a
    smoke run, which cuts train and val to ``smoke_rows`` seed rows, the grid to two configs
    and the seeds to one, and never touches test)."""
    import relbench
    from relbench.submit import evaluate_submission, evaluate_task, write_prediction_table

    from relconduit.db import load_task_db

    t0 = time.time()
    ds = relbench.load_dataset(f"stanford-star/relbench-v1/{dataset_name}")
    task = ds.load_task(task_name)
    k = task.eval_k
    work = Path(work_dir) / dataset_name / task_name / "cand-lgbm-ranker"
    if smoke_rows:
        if score_test:
            raise ValueError("a smoke run never scores test")
        work = work / "smoke"
        plan = replace(
            plan,
            grid=list(plan.grid)[:2],
            seeds=tuple(plan.seeds)[:1],
            train_rows=min(plan.train_rows, 3 * smoke_rows),
            train_times=1,
            es_seeds=smoke_rows,
            chunk_rows=max(smoke_rows // 2, 1),
        )
    work.mkdir(parents=True, exist_ok=True)
    pairs_dir = Path(pairs_dir) if pairs_dir is not None else work / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    train_all = task.get_table("train", mask_input_cols=False).df.reset_index(drop=True)
    tpos = sample_train(train_all[task.time_col], plan.train_rows, plan.train_times)
    train_rows = train_all.iloc[tpos].reset_index(drop=True)
    val_rows = task.get_table("val", mask_input_cols=False).df.reset_index(drop=True)
    if smoke_rows:
        val_rows = val_rows.sample(n=min(smoke_rows, len(val_rows)), random_state=0)
        val_rows = val_rows.reset_index(drop=True)
    val_table = task.get_table("val", mask_input_cols=False)
    val_table.df = val_rows

    db = (plan.db_loader or load_task_db)(task, **plan.db_kwargs)
    sides = {}
    for side, cfg in (("src", src_features), ("dst", dst_features)):
        if cfg is not None:
            cfg = replace(cfg)
            if cfg.vocab is None:
                cfg.vocab = feature_vocab(db, cfg, before=ds.val_timestamp)
            sides[side] = cfg
    c = _Ctx(
        task,
        db,
        plan,
        sides,
        pairs_dir,
        task.src_entity_col,
        task.dst_entity_col,
        task.time_col,
        t0,
    )
    c.log(
        f"{dataset_name}/{task_name}: db loaded (peak rss {_rss_gb():.1f} GB); "
        f"{len(train_rows):,} of {len(train_all):,} train seed rows "
        f"({train_rows[c.time_col].nunique()} seed times), {len(val_rows):,} val rows"
    )

    # --- leakage check on the real database ------------------------------------------------
    sample = pd.concat(
        [
            r.sample(n=min(leakage_seeds, len(r)), random_state=0)[[c.src_col, c.time_col]]
            for r in (val_rows, train_rows)
        ],
        ignore_index=True,
    )
    bad = pair_leakage_mismatches(db, sample, c.src_col, c.time_col, plan.link)
    c.log(
        f"pair leakage check on {len(sample)} seeds ({sample[c.time_col].nunique()} seed "
        f"times): {len(bad)} mismatches"
    )
    if bad:
        raise RuntimeError(f"pair leakage: {bad[:10]}")

    builder = PairBuilder(db, plan.link)
    c.log(f"edge sets { {n: len(e) for n, e in builder.edges.items()} }")
    dstf = _DstFeatures(db, task, sides["dst"]) if "dst" in sides else None
    pairs = _Pairs(c, builder, dstf)

    # --- val pairs (every row), recall ------------------------------------------------------
    val_paths = pairs.build("val", val_rows, labels=True, positive_only=False)
    cols = _features(val_paths)
    sources = [x for x in cols if x.startswith("cand__")]
    n_val = len(val_rows)
    hits = {s: np.zeros(n_val) for s in ["all", *sources]}
    n_cand = np.zeros(n_val)
    for p in val_paths:
        X = pd.read_parquet(p, columns=[SEED, LABEL, *sources])
        s, y = X[SEED].to_numpy(), X[LABEL].to_numpy().astype(float)
        np.add.at(hits["all"], s, y)
        np.add.at(n_cand, s, 1)
        for src in sources:
            np.add.at(hits[src], s, y * (X[src].to_numpy() > 0))
    n_true = val_rows[c.dst_col].map(len).to_numpy()
    recall = {
        (s if s == "all" else s[len("cand__") :]): recall_from_hits(h, n_true)
        for s, h in hits.items()
    }
    recall["per_seed"] = float(n_cand.mean())
    recall["seeds_with_true"] = float(np.mean(hits["all"] > 0))
    c.log(f"val recall {json.dumps({a: round(b, 4) for a, b in recall.items()})}")

    # --- fits ---------------------------------------------------------------------------------
    cache = work / "tune.jsonl"
    done: dict[str, dict] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    models_dir = work / "models"
    models_dir.mkdir(exist_ok=True)
    state: dict = {}

    def datasets():
        if "dtr" in state:
            return state["dtr"], state["dva"]
        paths = pairs.build("train", train_rows, labels=True, positive_only=True)
        X = _concat(paths)
        y = X.pop(LABEL).to_numpy()
        info = {
            "train_seed_rows": len(train_rows),
            "train_seeds_with_true": int(X[SEED].nunique()),
            "train_pairs": len(X),
            "positive_rate": float(y.mean()),
            "n_features": len(cols),
        }
        state["info"] = info
        dtr = _dataset(X, y, cols)
        del X
        pos = np.flatnonzero(hits["all"] > 0)
        rng = np.random.default_rng(0)
        es = np.sort(rng.choice(pos, size=min(plan.es_seeds, len(pos)), replace=False))
        Xv = _concat(val_paths, es)
        yv = Xv.pop(LABEL).to_numpy()
        dva = _dataset(Xv, yv, cols, reference=dtr)
        info["es_val_seeds"] = len(es)
        c.log(f"datasets {json.dumps(info)}, peak rss {_rss_gb():.1f} GB")
        state["dtr"], state["dva"] = dtr, dva
        return dtr, dva

    def key(config: dict, seed: int) -> str:
        return json.dumps({"config": config, "seed": seed}, sort_keys=True)

    def fit(config: dict, seed: int) -> float:
        kk = key(config, seed)
        if kk in done:
            return done[kk]["val"]
        dtr, dva = datasets()
        t1 = time.time()
        model = lgb.train(
            _params(config, seed, k),
            dtr,
            num_boost_round=plan.max_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(plan.early_stopping, verbose=False)],
        )
        model.save_model(models_dir / f"{_slug(kk)}.txt", num_iteration=model.best_iteration)
        m = lgb.Booster(model_file=str(models_dir / f"{_slug(kk)}.txt"))
        (pred,) = _predict_top([m], val_paths, cols, n_val, k)
        (val,) = task.evaluate(pred, val_table).values()
        row = {
            "key": kk,
            "val": float(val),
            "best_iteration": model.best_iteration,
            "fit_minutes": (time.time() - t1) / 60,
        }
        with open(cache, "a") as f:
            f.write(json.dumps(row) + "\n")
        done[kk] = row
        c.log(
            f"  {config} seed {seed}: val {val:.5f} ({model.best_iteration} rounds, "
            f"{time.time() - t1:.0f}s, peak rss {_rss_gb():.1f} GB)"
        )
        return float(val)

    seeds = tuple(plan.seeds)
    grid = []
    for config in plan.grid:
        grid.append({"config": config, "scores": [fit(config, seeds[0])]})
    best = pick_best(grid)
    config = best["config"]
    c.log(f"grid (seed {seeds[0]}): best {config} val {best['scores'][0]:.5f}")
    final = [fit(config, s) for s in seeds]
    sd = float(np.std(final, ddof=1)) if len(final) > 1 else 0.0
    c.log(f"chosen config, seeds {list(seeds)}: val {np.mean(final):.5f} ± {sd:.5f}")
    if "info" not in state:  # every fit came from the cache
        paths = pairs.build("train", train_rows, labels=True, positive_only=True)
        tr = _concat(paths)
        state["info"] = {
            "train_seed_rows": len(train_rows),
            "train_seeds_with_true": int(tr[SEED].nunique()),
            "train_pairs": len(tr),
            "positive_rate": float(tr[LABEL].mean()),
            "n_features": len(cols),
        }
        del tr
    state.pop("dtr", None), state.pop("dva", None)

    models = [
        lgb.Booster(model_file=str(models_dir / f"{_slug(key(config, s))}.txt")) for s in seeds
    ]
    gain = models[0].feature_importance(importance_type="gain")
    summary = {
        "dataset": dataset_name,
        "task": task_name,
        "approach": approach,
        "metric": task.metrics[0].__name__,
        "eval_k": k,
        "n_train_seeds_full": len(train_all),
        "train_times": plan.train_times,
        "train": state["info"],
        "val_rows": n_val,
        "val_candidate_recall": recall,
        "leakage_check_seeds": len(sample),
        "leakage_mismatches": len(bad),
        "grid": grid,
        "chosen_config": config,
        "seeds": list(seeds),
        "val_scores": final,
        "best_iterations": [done[key(config, s)]["best_iteration"] for s in seeds],
        "test_scores": None,
        "plan_extra": plan.extra,
        "top_features": [
            [models[0].feature_name()[i], round(float(gain[i]), 1)] for i in np.argsort(-gain)[:30]
        ],
    }
    if not score_test:
        summary["runtime_minutes"] = (time.time() - t0) / 60
        summary["peak_rss_gb"] = _rss_gb()
        name = "summary_smoke.json" if smoke_rows else "summary_val.json"
        (work / name).write_text(json.dumps(summary, indent=2))
        return summary

    # --- test, once ---------------------------------------------------------------------------
    test_rows = task.get_table("test").df.reset_index(drop=True)  # masked: src and time only
    test_paths = pairs.build("test", test_rows, labels=False, positive_only=False)
    preds = _predict_top(models, test_paths, cols, len(test_rows), k)
    test_scores = []
    for seed, pred in zip(seeds, preds, strict=True):
        path = work / "pred" / f"seed{seed}" / f"{dataset_name}__{task_name}.csv"
        write_prediction_table(task, pred, path)
        (value,) = evaluate_task(f"{dataset_name}/{task_name}", path).values()
        test_scores.append(float(value))
        c.log(f"test seed {seed}: {value:.5f}")
    summary["test_scores"] = test_scores
    sub = work / "submission"
    sub.mkdir(exist_ok=True)
    src = work / "pred" / f"seed{seeds[0]}" / f"{dataset_name}__{task_name}.csv"
    shutil.copy(src, sub / src.name)
    res = evaluate_submission(str(sub), num_workers=1, verbose=False)
    entry = res["tasks"][f"{dataset_name}/{task_name}"]
    summary["submission_check"] = {
        "file": str(sub / src.name),
        "seed": seeds[0],
        "status": entry["status"],
        "metric": entry["metric"],
        "error": entry["error"],
        "family": entry["family"],
    }
    c.log(f"submission check: {json.dumps(summary['submission_check'])}")
    if plan.leader is not None:
        lname, lvalue = plan.leader
        summary["leader"] = {
            "name": lname,
            "test": lvalue,
            "margin": float(np.mean(test_scores)) - lvalue,
        }
    summary["runtime_minutes"] = (time.time() - t0) / 60
    summary["peak_rss_gb"] = _rss_gb()
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def records(summary: dict, commit: str, notes: str = "") -> list[dict]:
    """Results rows (val and test) for a :func:`run_big` summary with test scored."""
    from relconduit.results import make_record

    rec = summary["val_candidate_recall"]
    tr = summary["train"]
    base = (
        f"Candidates (val recall {rec['all']:.3f}, {rec['per_seed']:.0f} per seed) + LightGBM "
        f"ranker on {tr['n_features']} pair and entity features; config chosen on val (all "
        f"{summary['val_rows']:,} rows) from {len(summary['grid'])}: {summary['chosen_config']}. "
        f"Trained on {tr['train_seed_rows']:,} of {summary['n_train_seeds_full']:,} train seed "
        f"rows (latest {summary['train_times']} seed times; {tr['train_seeds_with_true']:,} with "
        f"a true candidate); early stopping on {tr.get('es_val_seeds', 0):,} val seeds. Pair "
        f"leakage check: {summary['leakage_mismatches']} mismatches on "
        f"{summary['leakage_check_seeds']} seeds."
    )
    if notes:
        base += " " + notes
    common = {
        "approach": summary["approach"],
        "metric": summary["metric"],
        "runtime_minutes": summary["runtime_minutes"],
        "commit": commit,
        "candidate_recall_val": {a: round(b, 4) for a, b in rec.items()},
    }
    test_extra = {}
    if "submission_check" in summary:
        test_extra["submission_check_seed0"] = summary["submission_check"]["status"]
    return [
        make_record(
            split="val",
            values=summary["val_scores"],
            notes=base + " Val is optimistic: it chose the config and the stopping round.",
            **common,
        ),
        make_record(
            split="test", values=summary["test_scores"], notes=base, **common, **test_extra
        ),
    ]


def main(
    dataset: str,
    task: str,
    plan: BigPlan,
    src_features: FeatureConfig | None,
    dst_features: FeatureConfig | None,
    notes: str = "",
    argv: Sequence[str] | None = None,
) -> dict:
    """Command line for an experiment's ``run.py``: ``--smoke``, ``--test``, ``--work-dir``."""
    import argparse

    from relconduit.results import append_record, git_commit

    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="score test once, append to results/")
    p.add_argument("--smoke", type=int, nargs="?", const=300, default=None)
    p.add_argument("--work-dir", default="runs")
    p.add_argument("--pairs-dir")
    p.add_argument("--train-rows", type=int, help="override the plan's train sample")
    a = p.parse_args(argv)
    if a.train_rows:
        plan = replace(plan, train_rows=a.train_rows)
    approach = f"experiments/{dataset}/{task}/cand-lgbm-ranker"
    summary = run_big(
        dataset,
        task,
        approach + (" (smoke)" if a.smoke else ""),
        plan,
        src_features,
        dst_features,
        work_dir=a.work_dir,
        pairs_dir=a.pairs_dir,
        score_test=a.test and not a.smoke,
        smoke_rows=a.smoke,
        leakage_seeds=8 if a.smoke else 40,
    )
    keys = ("val_candidate_recall", "chosen_config", "val_scores", "test_scores", "top_features")
    print(json.dumps({x: summary.get(x) for x in keys}, indent=2), flush=True)
    if a.test and not a.smoke:
        for row in records(summary, commit=git_commit(), notes=notes):
            append_record(f"results/{dataset}/{task}.jsonl", row)
    return summary
