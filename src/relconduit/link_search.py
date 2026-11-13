"""A staged, val-only search over candidates, train size and ranker config for a link task.

:mod:`relconduit.link_lgbm` runs one candidate set, one train sample and a small grid. This
module runs the follow-up protocol of issue 18, where every choice is made on val and test is
scored once, for the final config only:

1. **Candidates.** Each candidate set (:attr:`SearchPlan.variants`, the first being the
   reference) is built on val, with its recall per source, and checked for leakage on the real
   database (:func:`relconduit.link.pair_leakage_mismatches`, full vs truncated, on a sample of
   val and train seeds) before anything is fitted. Each is fitted on
   :attr:`SearchPlan.stage1_rows` train seed rows with the reference ranker config, over the
   tuning seeds. The best mean val MAP wins.
2. **Train size.** The winning candidate set is fitted on each of
   :attr:`SearchPlan.train_sizes` train seed rows (``None``: all of them). Subsets are nested:
   the first ``n`` rows of one fixed permutation (``numpy.random.default_rng(0)``).
3. **Config.** The winning size is fitted with each config in :attr:`SearchPlan.grid`.
4. **Final.** The best config is fitted on the remaining :attr:`SearchPlan.final_seeds`. With
   ``score_test``, test is scored once per final seed (``write_prediction_table`` +
   ``evaluate_task``), and :func:`bootstrap_map` gives a CI over test rows for the seed mean.

Train seeds with no true dst among their candidates are dropped before entity features are
built (as in :mod:`relconduit.link_lgbm`, they carry no ranking signal), and pairs are built
and stored a chunk of seed times at a time (``float32``), so the full train table fits in
memory. One LightGBM ``Dataset`` per (candidate set, train size) is shared by every config and
seed (``feature_pre_filter=False``), so a config change never rebins the data.

Every fit is cached in ``<work>/tune.jsonl`` with its model, and every pair chunk in
``pairs_dir``, so a restarted machine resumes.
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from relconduit.features import FeatureConfig, feature_vocab
from relconduit.link import (
    DST,
    SEED,
    LinkConfig,
    PairBuilder,
    candidate_recall,
    label_pairs,
    pair_leakage_mismatches,
)
from relconduit.link_lgbm import (
    BASE_PARAMS,
    _group_sizes,
    _slug,
    _with_positive,
    add_entity_features,
    top_k,
)

MAX_ROUNDS = 5000
EARLY_STOPPING = 100


@dataclass
class SearchPlan:
    """What the search tries (see the module doc). Configs are LightGBM params on top of
    :data:`relconduit.link_lgbm.BASE_PARAMS`; ``objective`` is required."""

    variants: dict[str, LinkConfig]
    ref_config: dict
    grid: Sequence[dict]
    stage1_rows: int = 6_000
    train_sizes: Sequence[int | None] = (12_000, None)
    tune_seeds: Sequence[int] = (0, 1, 2)
    final_seeds: Sequence[int] = (0, 1, 2, 3, 4)
    leader: tuple[str, float] | None = None  # (name, test value) of the board's leader
    submission_seed: int = 0  # the seed whose test table is the candidate submission
    chunk_rows: int = 3_000  # seed rows per pair-building chunk (whole seed times)
    extra: dict = field(default_factory=dict)


# --- pure helpers (tested) ---------------------------------------------------------------


def nested_subset(n_total: int, n: int | None, seed: int = 0) -> np.ndarray:
    """Sorted positions of the first ``n`` rows of a fixed permutation (``None``: all)."""
    if n is None or n >= n_total:
        return np.arange(n_total)
    perm = np.random.default_rng(seed).permutation(n_total)
    return np.sort(perm[:n])


def time_chunks(times: pd.Series, max_rows: int) -> list[np.ndarray]:
    """Positions of ``times`` grouped into chunks of whole seed times, each up to
    ``max_rows`` rows (a single seed time with more rows is its own chunk)."""
    t = pd.to_datetime(times).to_numpy()
    chunks, cur, n = [], [], 0
    for u in np.unique(t):
        idx = np.flatnonzero(t == u)
        if cur and n + len(idx) > max_rows:
            chunks.append(np.concatenate(cur))
            cur, n = [], 0
        cur.append(idx)
        n += len(idx)
    if cur:
        chunks.append(np.concatenate(cur))
    return chunks


def pick_best(results: Sequence[dict], key: str = "scores") -> dict:
    """The result with the highest mean of ``key``; ties go to the earliest."""
    best = None
    for r in results:
        if best is None or np.mean(r[key]) > np.mean(best[key]):
            best = r
    return best


def bootstrap_map(
    evaluate: Callable[[np.ndarray, np.ndarray], float],
    preds: Sequence[np.ndarray],
    n_rows: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap over rows of the seed mean: for each resample of row positions,
    ``mean(evaluate(rows, pred[rows]) for pred in preds)``. ``evaluate`` must score a
    prediction array against the target rows at those positions (the official metric)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, n_rows, size=n_rows)
        out[b] = np.mean([evaluate(rows, p[rows]) for p in preds])
    return out


def lead_verdict(values: Sequence[float], leader: float, boot: np.ndarray | None = None) -> dict:
    """The clear-lead bar of issue 18: mean above ``leader`` by more than 2x the seed std
    (ddof=1). With ``boot``, the 95% percentile CI and the share of resamples above it."""
    v = np.asarray(values, dtype=float)
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out = {
        "mean": mean,
        "std": sd,
        "leader": leader,
        "margin": mean - leader,
        "bar": 2 * sd,
        "clear_lead": bool(mean - leader > 2 * sd),
    }
    if boot is not None:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out |= {"ci95": [float(lo), float(hi)], "p_above_leader": float(np.mean(boot > leader))}
    return out


# --- the search --------------------------------------------------------------------------


def _m(t0: float) -> str:
    return f"{(time.time() - t0) / 60:6.1f} min"


def _rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


class _Data:
    """Pair tables for one candidate set, built in chunks and cached as parquet."""

    def __init__(self, ctx, name: str, link: LinkConfig):
        self.ctx, self.name = ctx, name
        self.builder = PairBuilder(ctx.db, link)
        self.link = link
        self.dir = ctx.pairs_dir / name
        self.dir.mkdir(parents=True, exist_ok=True)

    def split(
        self,
        split: str,
        rows: pd.DataFrame,
        positive_only: bool,
        positions: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """All pairs of ``rows.iloc[positions]`` (all rows if ``None``), features joined;
        ``seed`` is the position in ``rows``."""
        c = self.ctx
        out = []
        pos = np.arange(len(rows)) if positions is None else np.asarray(positions)
        for i, local in enumerate(time_chunks(rows[c.time_col].iloc[pos], c.plan.chunk_rows)):
            idx = pos[local]
            path = self.dir / f"{split}_{i:03d}.parquet"
            if path.exists():
                out.append(pd.read_parquet(path))
                continue
            t1 = time.time()
            sub = rows.iloc[idx].reset_index(drop=True)
            pairs = self.builder.build(sub, c.src_col, c.time_col)
            n_pairs = len(pairs)
            if positive_only:
                y = label_pairs(pairs, sub, c.dst_col)
                pairs = pairs[_with_positive(pairs, y)].reset_index(drop=True)
            X = add_entity_features(c.db, c.task, pairs, sub, c.sides)
            X = X.astype({k: np.float32 for k in X.columns if X[k].dtype == np.float64})
            X[SEED] = idx[X[SEED].to_numpy()]
            tmp = path.with_suffix(".tmp")
            X.to_parquet(tmp)
            tmp.rename(path)
            out.append(X)
            c.log(
                f"{self.name} {split} chunk {i}: {len(sub):,} seeds, {n_pairs:,} pairs, "
                f"kept {len(X):,} x {X.shape[1]} ({time.time() - t1:.0f}s, "
                f"peak rss {_rss_gb():.1f} GB)"
            )
        X = _categorical(pd.concat(out, ignore_index=True))
        return X.sort_values([SEED, DST], kind="mergesort").reset_index(drop=True)


def _categorical(X: pd.DataFrame) -> pd.DataFrame:
    """Non-numeric columns as plain ``category`` (chunks read back from parquet can disagree
    on categories; LightGBM maps val/test categories onto train's by value)."""
    for c in X.columns:
        cat = isinstance(X[c].dtype, pd.CategoricalDtype)
        if c not in (SEED, DST) and (cat or not pd.api.types.is_numeric_dtype(X[c])):
            X[c] = X[c].astype(object).astype("category")
    return X


@dataclass
class _Ctx:
    task: object
    db: object
    plan: SearchPlan
    sides: dict
    pairs_dir: Path
    src_col: str
    dst_col: str
    time_col: str
    t0: float

    def log(self, msg: str) -> None:
        print(f"[{_m(self.t0)}] {msg}", flush=True)


def _dataset(X: pd.DataFrame, y: np.ndarray, cols: list[str], reference=None) -> lgb.Dataset:
    ds = lgb.Dataset(
        X[cols],
        y,
        group=_group_sizes(X[SEED].to_numpy()),
        reference=reference,
        params={"feature_pre_filter": False, "verbosity": -1},
        free_raw_data=True,
    )
    return ds.construct()


def _params(config: dict, seed: int, k: int) -> dict:
    params = {
        **BASE_PARAMS,
        **config,
        "metric": "map",
        "eval_at": [k],
        "seed": seed,
        "deterministic": True,
        "feature_pre_filter": False,
    }
    if config["objective"] == "lambdarank" and "lambdarank_truncation_level" not in config:
        params["lambdarank_truncation_level"] = max(2 * k, 20)
    return params


def run_search(
    dataset_name: str,
    task_name: str,
    approach: str,
    plan: SearchPlan,
    src_features: FeatureConfig | None,
    dst_features: FeatureConfig | None,
    work_dir: str | Path = "runs",
    pairs_dir: str | Path | None = None,
    score_test: bool = False,
    smoke_rows: int | None = None,
    leakage_seeds: int = 40,
    n_boot: int = 2000,
) -> dict:
    """Run the search (see the module doc). Returns the summary, also written to
    ``<work>/summary.json`` (``summary_val.json`` without test, ``summary_smoke.json`` for a
    smoke run). ``smoke_rows`` cuts train and val to that many seed rows, every stage to its
    first two options and one seed, and never touches test."""
    import relbench
    from relbench.submit import evaluate_task, write_prediction_table

    t0 = time.time()
    ds = relbench.load_dataset(f"stanford-star/relbench-v1/{dataset_name}")
    task = ds.load_task(task_name)
    k = task.eval_k
    work = Path(work_dir) / dataset_name / task_name / "cand-lgbm-ranker-plus"
    if smoke_rows:
        if score_test:
            raise ValueError("a smoke run never scores test")
        work = work / "smoke"
        plan = replace(
            plan,
            grid=list(plan.grid)[:2],
            train_sizes=list(plan.train_sizes)[:1],
            stage1_rows=smoke_rows,
            tune_seeds=tuple(plan.tune_seeds)[:1],
            final_seeds=tuple(plan.final_seeds)[:2],
            chunk_rows=max(smoke_rows // 3, 1),
        )
    work.mkdir(parents=True, exist_ok=True)
    pairs_dir = Path(pairs_dir) if pairs_dir is not None else work / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    train_rows = task.get_table("train", mask_input_cols=False).df.reset_index(drop=True)
    val_rows = task.get_table("val", mask_input_cols=False).df.reset_index(drop=True)
    if smoke_rows:
        train_rows = train_rows.sample(n=min(3 * smoke_rows, len(train_rows)), random_state=0)
        train_rows = train_rows.reset_index(drop=True)
        val_rows = val_rows.sample(n=min(smoke_rows, len(val_rows)), random_state=0)
        val_rows = val_rows.reset_index(drop=True)
    val_table = task.get_table("val", mask_input_cols=False)
    val_table.df = val_rows

    db = task.get_db()
    sides = {}
    for side, cfg in (("src", src_features), ("dst", dst_features)):
        if cfg is not None:
            cfg = replace(cfg)
            if cfg.vocab is None:
                cfg.vocab = feature_vocab(db, cfg, before=ds.val_timestamp)
            sides[side] = cfg
    ctx = _Ctx(
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
    ctx.log(f"{dataset_name}/{task_name}: db loaded; {len(train_rows):,} train seed rows")

    # --- cache of finished fits ------------------------------------------------------------
    cache = work / "tune.jsonl"
    done: dict[str, dict] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row
    models = work / "models"
    models.mkdir(exist_ok=True)

    def key(variant: str, size: int | None, config: dict, seed: int) -> str:
        return json.dumps(
            {"variant": variant, "size": size, "config": config, "seed": seed}, sort_keys=True
        )

    # --- candidates: val pairs, recall, leakage -------------------------------------------
    data: dict[str, _Data] = {}
    val_X: dict[str, pd.DataFrame] = {}
    recall: dict[str, dict] = {}
    leak: dict[str, int] = {}
    for name, link in plan.variants.items():
        d = data[name] = _Data(ctx, name, link)
        ctx.log(f"{name}: edge sets { {n: len(e) for n, e in d.builder.edges.items()} }")
        sample = pd.concat(
            [
                r.sample(n=min(leakage_seeds, len(r)), random_state=0)[[ctx.src_col, ctx.time_col]]
                for r in (val_rows, train_rows)
            ],
            ignore_index=True,
        )
        bad = pair_leakage_mismatches(db, sample, ctx.src_col, ctx.time_col, link)
        leak[name] = len(bad)
        ctx.log(
            f"{name}: pair leakage check on {len(sample)} seeds "
            f"({sample[ctx.time_col].nunique()} seed times): {len(bad)} mismatches"
        )
        if bad:
            raise RuntimeError(f"pair leakage in {name}: {bad[:10]}")
        X = val_X[name] = d.split("val", val_rows, positive_only=False)
        rec = {"all": candidate_recall(X, val_rows, ctx.dst_col)}
        for c in [c for c in X.columns if c.startswith("cand__")]:
            rec[c[len("cand__") :]] = candidate_recall(X[X[c] > 0], val_rows, ctx.dst_col)
        rec["per_seed"] = float(X.groupby(SEED).size().mean())
        recall[name] = rec
        ctx.log(f"{name}: val recall {json.dumps({a: round(b, 4) for a, b in rec.items()})}")

    # --- datasets --------------------------------------------------------------------------
    datasets: dict[tuple, tuple] = {}
    train_X: dict[str, pd.DataFrame] = {}
    stage1_winner: dict[str, str] = {}

    def get_train(name: str, size: int | None) -> pd.DataFrame:
        """Train pairs: stage 1 builds only its subset; later stages the whole table."""
        full = name in train_X or size is None or name == stage1_winner.get("v")
        if full:
            if name not in train_X:
                train_X[name] = data[name].split("train", train_rows, positive_only=True)
                ctx.log(f"{name}: train pairs {train_X[name].shape}, peak rss {_rss_gb():.1f} GB")
            return train_X[name]
        sub = nested_subset(len(train_rows), size)
        return data[name].split(f"train{size}", train_rows, positive_only=True, positions=sub)

    def get_datasets(name: str, size: int | None):
        if (name, size) in datasets:
            return datasets[(name, size)]
        X = get_train(name, size)
        keep = np.isin(X[SEED].to_numpy(), nested_subset(len(train_rows), size))
        Xs = X[keep]
        ys = label_pairs(Xs, train_rows, ctx.dst_col)
        cols = [c for c in X.columns if c not in (SEED, DST)]
        dtr = _dataset(Xs, ys, cols)
        Xv = val_X[name]
        yv = label_pairs(Xv, val_rows, ctx.dst_col)
        vk = _with_positive(Xv, yv)
        dva = _dataset(Xv[vk], yv[vk], cols, reference=dtr)
        info = {
            "train_seed_rows": len(nested_subset(len(train_rows), size)),
            "train_seeds_with_true": int(Xs[SEED].nunique()),
            "train_pairs": len(Xs),
            "positive_rate": float(ys.mean()),
            "n_features": len(cols),
        }
        ctx.log(f"{name} size {size}: dataset {json.dumps(info)}, peak rss {_rss_gb():.1f} GB")
        datasets[(name, size)] = (dtr, dva, cols, info)
        return datasets[(name, size)]

    def fit_scores(name: str, size: int | None, config: dict, seeds: Sequence[int]) -> dict:
        scores, iters = [], []
        for seed in seeds:
            kk = key(name, size, config, seed)
            if kk in done:
                scores.append(done[kk]["val"])
                iters.append(done[kk]["best_iteration"])
                continue
            dtr, dva, cols, _ = get_datasets(name, size)
            t1 = time.time()
            model = lgb.train(
                _params(config, seed, k),
                dtr,
                num_boost_round=MAX_ROUNDS,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            Xv = val_X[name]
            s = model.predict(Xv[cols], num_iteration=model.best_iteration)
            (val,) = task.evaluate(top_k(Xv, s, len(val_rows), k), val_table).values()
            model.save_model(models / f"{_slug(kk)}.txt", num_iteration=model.best_iteration)
            row = {"key": kk, "val": float(val), "best_iteration": model.best_iteration}
            with open(cache, "a") as f:
                f.write(json.dumps(row) + "\n")
            done[kk] = row
            scores.append(float(val))
            iters.append(model.best_iteration)
            ctx.log(
                f"  {name} size {size} {config} seed {seed}: val {val:.4f} "
                f"({model.best_iteration} rounds, {time.time() - t1:.0f}s)"
            )
        sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        ctx.log(f"{name} size {size} {config}: val {np.mean(scores):.4f} ± {sd:.4f}")
        return {
            "variant": name,
            "size": size,
            "config": config,
            "scores": scores,
            "best_iterations": iters,
        }

    seeds = tuple(plan.tune_seeds)
    stages: dict[str, list[dict]] = {}

    # 1. candidates
    stages["candidates"] = [
        fit_scores(name, plan.stage1_rows, plan.ref_config, seeds) for name in plan.variants
    ]
    variant = pick_best(stages["candidates"])["variant"]
    stage1_winner["v"] = variant
    ctx.log(f"stage 1 (candidates): {variant}")
    for name in list(train_X):
        if name != variant:  # free the losers
            del train_X[name]
    for kk in [kk for kk in datasets if kk[0] != variant]:
        del datasets[kk]

    # 2. train size
    stage2 = [r for r in stages["candidates"] if r["variant"] == variant]
    for size in plan.train_sizes:
        datasets.pop((variant, plan.stage1_rows), None)
        stage2.append(fit_scores(variant, size, plan.ref_config, seeds))
    stages["train_size"] = stage2
    size = pick_best(stage2)["size"]
    ctx.log(f"stage 2 (train size): {size}")
    for kk in [kk for kk in datasets if kk != (variant, size)]:
        del datasets[kk]

    # 3. config
    stage3 = [r for r in stage2 if r["size"] == size and r["config"] == plan.ref_config]
    for config in plan.grid:
        if config != plan.ref_config:
            stage3.append(fit_scores(variant, size, config, seeds))
    stages["config"] = stage3
    best = pick_best(stage3)
    config = best["config"]
    ctx.log(f"stage 3 (config): {config}")

    # 4. final seeds
    final = fit_scores(variant, size, config, tuple(plan.final_seeds))
    _, _, cols, info = get_datasets(variant, size)
    train_X.clear()
    summary = {
        "dataset": dataset_name,
        "task": task_name,
        "approach": approach,
        "metric": task.metrics[0].__name__,
        "eval_k": k,
        "n_train_seeds_full": len(train_rows),
        "val_candidate_recall": recall,
        "leakage_mismatches": leak,
        "leakage_check_seeds": 2 * leakage_seeds,
        "stages": stages,
        "chosen": {"variant": variant, "size": size, "config": config},
        "train": info,
        "final_seeds": list(plan.final_seeds),
        "val_scores": final["scores"],
        "best_iterations": final["best_iterations"],
        "test_scores": None,
        "plan_extra": plan.extra,
    }
    m0 = lgb.Booster(
        model_file=str(models / f"{_slug(key(variant, size, config, plan.final_seeds[0]))}.txt")
    )
    gain = m0.feature_importance(importance_type="gain")
    summary["top_features"] = [
        [m0.feature_name()[i], round(float(gain[i]), 1)] for i in np.argsort(-gain)[:30]
    ]
    if not score_test:
        summary["runtime_minutes"] = (time.time() - t0) / 60
        summary["peak_rss_gb"] = _rss_gb()
        name = "summary_smoke.json" if smoke_rows else "summary_val.json"
        (work / name).write_text(json.dumps(summary, indent=2))
        return summary

    # --- test, once ---------------------------------------------------------------------------
    test_rows = task.get_table("test").df.reset_index(drop=True)  # masked: src and time only
    X_test = data[variant].split("test", test_rows, positive_only=False)
    preds, test_scores = [], []
    for seed in plan.final_seeds:
        model = lgb.Booster(
            model_file=str(models / f"{_slug(key(variant, size, config, seed))}.txt")
        )
        s = model.predict(X_test[cols])
        pred = top_k(X_test, s, len(test_rows), k)
        path = work / "pred" / f"seed{seed}" / f"{dataset_name}__{task_name}.csv"
        write_prediction_table(task, pred, path)
        (value,) = evaluate_task(f"{dataset_name}/{task_name}", path).values()
        preds.append(pred)
        test_scores.append(float(value))
        ctx.log(f"test seed {seed}: {value:.4f}")
    summary["test_scores"] = test_scores

    target = task.get_table("test", mask_input_cols=False)
    tdf = target.df.reset_index(drop=True)

    def evaluate(rows: np.ndarray, pred: np.ndarray) -> float:
        target.df = tdf.iloc[rows].reset_index(drop=True)
        (v,) = task.evaluate(pred, target).values()
        return float(v)

    check = [evaluate(np.arange(len(tdf)), p) for p in preds]
    summary["evaluate_check"] = {
        "task_evaluate": check,
        "same": bool(np.allclose(check, test_scores, atol=1e-9)),
    }
    if not summary["evaluate_check"]["same"]:
        ctx.log(f"WARNING: task.evaluate {check} != evaluate_task {test_scores}")
    boot = bootstrap_map(evaluate, preds, len(tdf), n_boot=n_boot)
    target.df = tdf
    summary["test_bootstrap"] = {
        "n_boot": n_boot,
        "n_rows": len(tdf),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "std": float(boot.std(ddof=1)),
    }
    if plan.leader is not None:
        lname, lvalue = plan.leader
        summary["leader"] = lname
        summary["verdict"] = lead_verdict(test_scores, lvalue, boot)
        ctx.log(f"verdict vs {lname}: {json.dumps(summary['verdict'])}")
        if summary["verdict"]["clear_lead"]:
            src = work / "pred" / f"seed{plan.submission_seed}" / f"{dataset_name}__{task_name}.csv"
            sub = work / "submission"
            sub.mkdir(exist_ok=True)
            (sub / src.name).write_bytes(src.read_bytes())
            res = subprocess.run(
                [sys.executable, "-m", "relbench.submit", str(sub)],
                capture_output=True,
                text=True,
                check=False,
            )
            summary["submission_check"] = {
                "dir": str(sub),
                "seed": plan.submission_seed,
                "returncode": res.returncode,
                "output": (res.stdout + res.stderr)[-4000:],
            }
            ctx.log(f"submission check rc={res.returncode}:\n{res.stdout[-2000:]}")
    summary["runtime_minutes"] = (time.time() - t0) / 60
    summary["peak_rss_gb"] = _rss_gb()
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def records(summary: dict, commit: str, notes: str = "") -> list[dict]:
    """Results rows (val and test) for a :func:`run_search` summary with test scored."""
    from relconduit.results import make_record

    ch = summary["chosen"]
    rec = summary["val_candidate_recall"][ch["variant"]]
    tr = summary["train"]
    base = (
        f"Staged val-only search (candidates, train size, config); chosen: candidate set "
        f"'{ch['variant']}' (val recall {rec['all']:.3f}, {rec['per_seed']:.0f} per seed), "
        f"{tr['train_seed_rows']:,} of {summary['n_train_seeds_full']:,} train seed rows, "
        f"{tr['n_features']} features, LightGBM {ch['config']}; early stopping on val. Pair "
        f"leakage check: {summary['leakage_mismatches']} mismatches on "
        f"{summary['leakage_check_seeds']} seeds per candidate set."
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
    if "test_bootstrap" in summary:
        test_extra["bootstrap_ci95_test_rows"] = [
            round(x, 4) for x in summary["test_bootstrap"]["ci95"]
        ]
    if "verdict" in summary:
        v = summary["verdict"]
        test_extra["clear_lead_vs_" + summary["leader"]] = v["clear_lead"]
    return [
        make_record(
            split="val",
            values=summary["val_scores"],
            notes=base + " Val is optimistic: it made every choice and the stopping round.",
            **common,
        ),
        make_record(
            split="test", values=summary["test_scores"], notes=base, **common, **test_extra
        ),
    ]


def main(
    dataset: str,
    task: str,
    plan: SearchPlan,
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
    p.add_argument("--smoke", type=int, nargs="?", const=200, default=None)
    p.add_argument("--work-dir", default="runs")
    p.add_argument("--pairs-dir")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args(argv)
    approach = f"experiments/{dataset}/{task}/cand-lgbm-ranker-plus"
    summary = run_search(
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
        n_boot=a.n_boot,
    )
    keys = ("chosen", "val_scores", "test_scores", "test_bootstrap", "verdict", "top_features")
    print(json.dumps({x: summary.get(x) for x in keys}, indent=2), flush=True)
    if a.test and not a.smoke:
        for row in records(summary, commit=git_commit(), notes=notes):
            append_record(f"results/{dataset}/{task}.jsonl", row)
    return summary
