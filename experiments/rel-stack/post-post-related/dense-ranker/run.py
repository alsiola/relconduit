"""Issue 29, phase 2: ablations against the misses phase 1 measured (val only), then test once.

One **superset** of candidates and pair features is built per split, with every source any
variant uses: issue 12's sources (``../cand-lgbm-ranker/run.py``), tag and title-word overlap
counts, and dense kNN over each encoder's index (``relconduit.post_embed``, n = 300). A
variant is a view of it: the rows some of its sources proposed (a dense source with ``n``
below 300 proposes the rows its rank keeps), its feature columns, and the walk ranks and
``n_candidates`` recomputed on those rows exactly as :mod:`relconduit.link` computes them, so
a variant equals what a separate build with its own config would give.

Protocol (every choice on val, official ``task.evaluate``):

1. Pair leakage check on the real database for the superset config (40 val + 40 train seeds,
   full vs ``truncate_before``): the run stops on any mismatch.
2. Superset pairs for train (and the extra offset train seed times, labelled by
   ``task.make_table``), and val. Train seeds with no true candidate are dropped before
   entity features (no ranking signal).
3. Stage A: each variant fitted with issue 12's chosen config, seeds 0-4, early stopping on
   val. Per variant: val MAP per seed, candidate recall, per-row AP, a paired bootstrap of the
   per-row AP (seed mean) against ``ref``. Stage B: the best stage-A variant with the extra
   train rows. Stage C: the best so far with a few ranker configs.
4. ``--final VARIANT [--config JSON]`` with ``--test``: test pairs, the variant's models for
   seeds 0-2, test scored once per seed (``write_prediction_table`` + ``evaluate_task``),
   results rows appended.

Every fit is cached (``<out>/tune.jsonl`` + model), every pair table in ``--pairs-dir``; per
variant and seed the val (and test) per-candidate scores and top-100 tables are kept in
``<out>/scores/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import relbench

from relconduit.features import feature_vocab
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
    _slug,
    _with_positive,
    add_entity_features,
    fit,
    top_k,
)
from relconduit.post_embed import DenseIndex, DenseSource

HERE = Path(__file__).resolve().parent
APPROACH = "experiments/rel-stack/post-post-related/dense-ranker"
DST_COL = "postLinksIdList"
CONFIG = {
    "objective": "lambdarank",
    "num_leaves": 15,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
}
ENCODERS = {"minilm": "all-MiniLM-L6-v2", "bge": "bge-small-en-v1.5"}
DENSE_N = 300
BASE = ["link_any", "two_hop", "tags", "words", "link_any__top3650d"]
PATHS = ["two_hop", "tags", "words"]
SEEDS = (0, 1, 2, 3, 4)

# name -> (dense sources {encoder: n proposed, 0 = feature only}, overlap features?)
VARIANTS = {
    "ref": ({}, False),
    "overlap": ({}, True),
    "cos_minilm": ({"minilm": 0}, True),
    "dense_minilm_100": ({"minilm": 100}, True),
    "dense_minilm_300": ({"minilm": 300}, True),
    "dense_bge_300": ({"bge": 300}, True),
    "dense_both_300": ({"minilm": 300, "bge": 300}, True),
}
STAGE_C = [
    {"learning_rate": 0.02},
    {"num_leaves": 31},
    {"min_child_samples": 300},
]


def _load_run_module():
    path = HERE.parent / "cand-lgbm-ranker" / "run.py"
    spec = importlib.util.spec_from_file_location("ppr_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN12 = _load_run_module()


def load_index(emb_dir: Path, enc: str) -> DenseIndex:
    z = np.load(emb_dir / f"{ENCODERS[enc]}.npz")
    return DenseIndex(z["ids"], z["times"], z["emb"].astype(np.float32))


def superset_config(emb_dir: Path) -> LinkConfig:
    return replace(
        RUN12.LINK,
        dense={
            f"dense_{enc}": DenseSource(load_index(emb_dir, enc), n=DENSE_N) for enc in ENCODERS
        },
        overlaps={"tag_overlap": "tag", "word_overlap": "word"},
    )


def extra_seed_rows(task, db) -> pd.DataFrame:
    """Train rows at seed times halfway between the train table's, labelled by RelBench's
    own ``task.make_table``. Each label window ends before the val timestamp, so every label
    is a row dated before val."""
    val_ts = pd.Timestamp(task.dataset.val_timestamp)
    half = task.timedelta / 2
    start = val_ts - task.timedelta - half
    times = pd.date_range(start=start, end=db.min_timestamp, freq=-task.timedelta)
    times = [t for t in times if t + task.timedelta < val_ts]
    table = task.make_table(db, pd.Series(times))
    table = task.filter_dangling_entities(table, db)
    df = table.df.reset_index(drop=True)
    assert (pd.to_datetime(df["timestamp"]) + task.timedelta < val_ts).all()
    return df


# --- variants as views of the superset ----------------------------------------------------


def variant_view(X: pd.DataFrame, name: str) -> pd.DataFrame:
    dense, overlap = VARIANTS[name] if name in VARIANTS else _parse(name)
    keep = np.zeros(len(X), dtype=bool)
    for s in BASE:
        keep |= X[f"cand__{s}"].to_numpy() > 0
    for enc, n in dense.items():
        if n > 0:
            keep |= X[f"dense_{enc}__rank"].fillna(np.inf).to_numpy() <= n
    V = X[keep].reset_index(drop=True)
    drop = []
    for enc in ENCODERS:
        cols = [f"cand__dense_{enc}", f"dense_{enc}__score", f"dense_{enc}__rank"]
        if enc not in dense:
            drop += cols
        elif dense[enc] == 0:
            drop.append(f"cand__dense_{enc}")
    if not overlap:
        drop += [c for c in V.columns if "_overlap__" in c]
    V = V.drop(columns=[c for c in drop if c in V.columns])
    for enc, n in dense.items():
        r = V.groupby(SEED)[f"dense_{enc}__score"].rank(ascending=False, method="min")
        V[f"dense_{enc}__rank"] = r.where(r <= max(n, 100)).astype(np.float32)
        if n > 0:
            V[f"cand__dense_{enc}"] = (V[f"dense_{enc}__rank"] <= n).astype(np.float32)
    for p in PATHS:
        r = V.groupby(SEED)[f"{p}__score"].rank(ascending=False, method="min")
        V[f"{p}__rank"] = r.where(V[f"{p}__score"] > 0).astype(np.float32)
    V["n_candidates"] = V.groupby(SEED)[DST].transform("size").astype(np.float32)
    return V


def _parse(name: str):
    raise KeyError(f"unknown variant {name}")


def features_of(V: pd.DataFrame) -> list[str]:
    return [c for c in V.columns if c not in (SEED, DST, "y")]


# --- evaluation helpers --------------------------------------------------------------------


def ap_rows(pred: np.ndarray, true_lists: list[set]) -> np.ndarray:
    """Per-row AP@k as ``relbench.metrics.map`` computes it (every val row has a true dst)."""
    k = pred.shape[1]
    out = np.zeros(len(true_lists))
    for i, (p, tr) in enumerate(zip(pred, true_lists)):
        isin = np.fromiter((d in tr for d in p), dtype=float, count=k)
        prec = np.cumsum(isin) / (np.arange(k) + 1)
        out[i] = (prec * isin).sum() / min(len(tr), k)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000) -> dict:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (a - b)[idx].mean(axis=1)
    return {
        "delta": float((a - b).mean()),
        "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
        "p_le_0": float((d <= 0).mean()),
    }


def _m(t0):
    return f"{(time.time() - t0) / 60:6.1f} min"


# --- main ---------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="runs/p29/emb")
    p.add_argument("--out", default="runs/p29/ablate")
    p.add_argument("--pairs-dir", default="runs/p29-pairs")
    p.add_argument("--smoke", type=int, help="this many train/val seeds, 1 seed, no test")
    p.add_argument("--final", help="the variant to score test with")
    p.add_argument("--config", default="{}", help="JSON overrides of CONFIG for --final")
    p.add_argument("--extra", action="store_true", help="--final: with the extra train rows")
    p.add_argument("--test", action="store_true")
    p.add_argument("--leakage-seeds", type=int, default=40)
    p.add_argument(
        "--train-minutes",
        type=float,
        default=0.0,
        help="--final: minutes the run that trained the saved models took, added to this "
        "run's own time in results/ (else a scoring-only run records only its own minutes)",
    )
    a = p.parse_args()
    t0 = time.time()
    out = Path(a.out) / ("smoke" if a.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    pairs_dir = Path(a.pairs_dir) / ("smoke" if a.smoke else "")
    pairs_dir.mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    ds = relbench.load_dataset("stanford-star/relbench-v1/rel-stack")
    task = ds.load_task("post-post-related")
    k = task.eval_k
    db = task.get_db()
    tables = {
        "train": task.get_table("train", mask_input_cols=False).df.reset_index(drop=True),
        "val": task.get_table("val", mask_input_cols=False).df.reset_index(drop=True),
    }
    tables["extra"] = extra_seed_rows(task, db)
    seeds = SEEDS
    if a.smoke:
        for s, df in list(tables.items()):
            n = min(a.smoke, len(df))
            tables[s] = df.sample(n=n, random_state=0).reset_index(drop=True)
        seeds = (0,)
    val_table = task.get_table("val", mask_input_cols=False)
    val_table.df = tables["val"]
    print(
        f"[{_m(t0)}] rows: train {len(tables['train']):,}, extra {len(tables['extra']):,} "
        f"({tables['extra']['timestamp'].nunique()} seed times), val {len(tables['val'])}",
        flush=True,
    )

    cfg = superset_config(Path(a.emb))
    builder = PairBuilder(db, cfg)
    print(f"[{_m(t0)}] builder ready; dense {[len(i.ids) for i in builder.dense.values()]}")

    # 1. leakage check, before anything is fitted
    leak_path = out / "leakage.json"
    if not leak_path.exists():
        n = 10 if a.smoke else a.leakage_seeds
        rows = [
            tables[s].sample(n=min(n, len(tables[s])), random_state=0)[["PostId", "timestamp"]]
            for s in ("val", "train", "extra")
        ]
        ls = pd.concat(rows, ignore_index=True)
        bad = pair_leakage_mismatches(db, ls, "PostId", "timestamp", cfg)
        msg = (
            f"pair leakage check on {len(ls)} seeds ({ls['timestamp'].nunique()} seed times): "
            f"{len(bad)} mismatches"
        )
        print(f"[{_m(t0)}] {msg}", flush=True)
        if bad:
            raise RuntimeError(f"pair leakage: {bad[:10]}")
        leak_path.write_text(json.dumps({"seeds": len(ls), "mismatches": 0}))

    feat_cfg = replace(RUN12.FEATURES)
    feat_cfg.vocab = feature_vocab(db, feat_cfg, before=ds.val_timestamp)
    sides = {"src": feat_cfg, "dst": feat_cfg}

    def pairs_of(split: str, rows: pd.DataFrame, only_positive: bool) -> pd.DataFrame:
        path = pairs_dir / f"{split}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        t1 = time.time()
        P = builder.build(rows, "PostId", "timestamp")
        if only_positive:
            y = label_pairs(P, rows, DST_COL)
            P = P[_with_positive(P, y)].reset_index(drop=True)
        print(f"[{_m(t0)}] {split}: {len(P):,} pairs ({time.time() - t1:.0f}s)", flush=True)
        X = add_entity_features(db, task, P, rows, sides)
        X = X.astype({c: np.float32 for c in X.columns if X[c].dtype == np.float64})
        if split != "test":
            X["y"] = label_pairs(X, rows, DST_COL)
        tmp = path.with_suffix(".tmp")
        X.to_parquet(tmp)
        tmp.rename(path)
        print(f"[{_m(t0)}] {split}: features {X.shape}", flush=True)
        return X

    X_train = pairs_of("train", tables["train"], True)
    X_val = pairs_of("val", tables["val"], False)
    true_val = [{int(d) for d in v} for v in tables["val"][DST_COL]]

    # Fitting ---------------------------------------------------------------------------
    cache = out / "tune.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r

    def run_variant(name: str, extra: bool, config: dict) -> dict:
        tag = (
            name
            + ("+extra" if extra else "")
            + "".join(f"+{k_}={v_}" for k_, v_ in sorted(config.items()) if CONFIG.get(k_) != v_)
        )
        Vtr = variant_view(X_train, name)
        if extra:
            Xe = pairs_of("extra", tables["extra"], True)
            Ve = variant_view(Xe, name)
            Ve[SEED] = Ve[SEED] + int(Vtr[SEED].max()) + 1
            Vtr = pd.concat([Vtr, Ve[Vtr.columns]], ignore_index=True)
        Vva = variant_view(X_val, name)
        cols = features_of(Vva)
        vkeep = _with_positive(Vva, Vva["y"].to_numpy())
        es = Vva[vkeep].reset_index(drop=True)
        recall = {"all": candidate_recall(Vva, tables["val"], DST_COL)}
        for c in [c for c in Vva.columns if c.startswith("cand__")]:
            recall[c[6:]] = candidate_recall(Vva[Vva[c] > 0], tables["val"], DST_COL)
        res = {
            "variant": tag,
            "config": config,
            "recall": recall,
            "cands_per_row": float(Vva.groupby(SEED).size().mean()),
            "n_features": len(cols),
            "train_pairs": len(Vtr),
            "train_seeds": int(Vtr[SEED].nunique()),
            "val": [],
            "iters": [],
        }
        per_seed_scores, per_seed_ap = [], []
        for seed in seeds:
            key = json.dumps({"variant": tag, "seed": seed}, sort_keys=True)
            mpath = out / "models" / f"{_slug(key)}.txt"
            if key in done and mpath.exists():
                model = lgb.Booster(model_file=str(mpath))
                it = done[key]["best_iteration"]
            else:
                t1 = time.time()
                model = fit(
                    (Vtr[[SEED, DST, *cols]], Vtr["y"].to_numpy()),
                    (es[[SEED, DST, *cols]], es["y"].to_numpy()),
                    config,
                    seed,
                    k,
                )
                it = model.best_iteration
                model.save_model(mpath, num_iteration=it)
                model = lgb.Booster(model_file=str(mpath))
                print(f"[{_m(t0)}]   {tag} seed {seed}: {it} rounds ({time.time() - t1:.0f}s)")
            s = model.predict(Vva[cols])
            pred = top_k(Vva, s, len(tables["val"]), k)
            (v,) = task.evaluate(pred, val_table).values()
            if key not in done:
                done[key] = {"key": key, "val": float(v), "best_iteration": it}
                with open(cache, "a") as f:
                    f.write(json.dumps(done[key]) + "\n")
            res["val"].append(float(v))
            res["iters"].append(it)
            per_seed_scores.append(s)
            per_seed_ap.append(ap_rows(pred, true_val))
            np.save(out / "scores" / f"{_slug(key)}_val_top{k}.npy", pred)
        S = Vva[[SEED, DST]].copy()
        for seed, s in zip(seeds, per_seed_scores):
            S[f"score{seed}"] = s.astype(np.float32)
        S.to_parquet(out / "scores" / f"{tag}_val_scores.parquet")
        # Seed bag: the mean of the per-seed scores, each z-scored per row.
        z = [
            (
                pd.Series(s)
                .groupby(Vva[SEED].to_numpy())
                .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            ).to_numpy()
            for s in per_seed_scores
        ]
        bag = top_k(Vva, np.mean(z, axis=0), len(tables["val"]), k)
        (vb,) = task.evaluate(bag, val_table).values()
        res["val_bag"] = float(vb)
        res["ap_seed_mean"] = np.mean(per_seed_ap, axis=0)
        res["ap_bag"] = ap_rows(bag, true_val)
        v3 = res["val"][:3]
        res["val3_mean"] = float(np.mean(v3))
        res["val3_std"] = float(np.std(v3, ddof=1)) if len(v3) > 1 else 0.0
        res["val_mean"] = float(np.mean(res["val"]))
        print(
            f"[{_m(t0)}] {tag}: recall {recall['all']:.3f}, {res['cands_per_row']:.0f}/row, "
            f"val {res['val3_mean']:.4f} ± {res['val3_std']:.4f} (seeds 0-2), "
            f"{res['val_mean']:.4f} (all {len(seeds)}), bag {vb:.4f}",
            flush=True,
        )
        return res

    if a.final:
        config = {**CONFIG, **json.loads(a.config)}
        res = run_variant(a.final, a.extra, config)
        if not a.test and not a.smoke:
            return
        final_test(a, task, db, res, pairs_of, out, config, t0)
        return

    results = {}
    summary_path = out / "summary.json"

    def save():
        rows = {}
        ref = results.get("ref")
        for name, r in results.items():
            row = {k_: v for k_, v in r.items() if not k_.startswith("ap_")}
            if ref is not None and name != "ref":
                row["paired_vs_ref_seed_mean"] = paired_bootstrap(
                    r["ap_seed_mean"], ref["ap_seed_mean"]
                )
                row["paired_vs_ref_bag"] = paired_bootstrap(r["ap_bag"], ref["ap_bag"])
            rows[name] = row
        summary_path.write_text(json.dumps(rows, indent=2))

    # Stage A
    for name in VARIANTS:
        results[name] = run_variant(name, False, CONFIG)
        save()
    best = max(VARIANTS, key=lambda n: results[n]["val_mean"])
    print(f"[{_m(t0)}] stage A best: {best}", flush=True)
    # Stage B: more train rows
    rb = run_variant(best, True, CONFIG)
    results[rb["variant"]] = rb
    save()
    use_extra = rb["val_mean"] > results[best]["val_mean"]
    lead = rb if use_extra else results[best]
    print(f"[{_m(t0)}] stage B: extra rows {'help' if use_extra else 'do not help'}", flush=True)
    # Stage C: configs
    for over in STAGE_C:
        rc = run_variant(best, use_extra, {**CONFIG, **over})
        rc["override"] = over
        results[rc["variant"]] = rc
        save()
        if rc["val_mean"] > lead["val_mean"]:
            lead = rc
    print(f"[{_m(t0)}] lead: {lead['variant']} {lead['config']} val {lead['val_mean']:.4f}")
    (out / "lead.json").write_text(
        json.dumps({"variant": best, "extra": use_extra, "config": lead["config"]}, indent=2)
    )
    print(f"[{_m(t0)}] done", flush=True)


def final_test(a, task, db, res, pairs_of, out, config, t0):
    """Score test once per seed 0-2 with the chosen variant; append the results rows."""
    from relbench.submit import evaluate_task, write_prediction_table

    from relconduit.results import append_record, git_commit, make_record

    test_rows = task.get_table("test").df.reset_index(drop=True)
    if a.smoke:  # code path check only: no prediction table written, test never scored
        test_rows = test_rows.sample(n=min(a.smoke, len(test_rows)), random_state=0)
        test_rows = test_rows.reset_index(drop=True)
    X_test = pairs_of("test", test_rows, False)
    tag = res["variant"]
    Vte = variant_view(X_test, a.final)
    cols = features_of(Vte)
    k = task.eval_k
    scores = []
    S = Vte[[SEED, DST]].copy()
    for seed in (0,) if a.smoke else (0, 1, 2):
        key = json.dumps({"variant": tag, "seed": seed}, sort_keys=True)
        model = lgb.Booster(model_file=str(out / "models" / f"{_slug(key)}.txt"))
        s = model.predict(Vte[cols])
        S[f"score{seed}"] = s.astype(np.float32)
        pred = top_k(Vte, s, len(test_rows), k)
        if a.smoke:
            print(f"[{_m(t0)}] smoke: test seed {seed} predicted {pred.shape}, not scored")
            continue
        path = out / "pred" / f"seed{seed}" / "rel-stack__post-post-related.csv"
        write_prediction_table(task, pred, path)
        (v,) = evaluate_task("rel-stack/post-post-related", path).values()
        scores.append(float(v))
        print(f"[{_m(t0)}] test seed {seed}: {v:.4f}", flush=True)
    S.to_parquet(out / "scores" / f"{tag}_test_scores.parquet")
    if a.smoke:
        return
    runtime = a.train_minutes + (time.time() - t0) / 60
    notes = (
        f"Variant {tag} (issue 29): issue 12's candidates and features plus "
        f"{res['n_features']} features in all; val candidate recall {res['recall']['all']:.3f} "
        f"({res['cands_per_row']:.0f} per row); LightGBM {config}, early stopping on val. "
        "Pair leakage check 0 mismatches. Google Cloud Spot, CPU."
    )
    if a.train_minutes:
        notes += (
            f" runtime_minutes is the training run's {a.train_minutes:.1f} min plus this "
            f"scoring run's {runtime - a.train_minutes:.1f} min."
        )
    common = {
        "approach": APPROACH,
        "metric": "map",
        "runtime_minutes": runtime,
        "commit": git_commit(),
    }
    rows = [
        make_record(
            split="val",
            values=res["val"][:3],
            notes=notes + " Val is optimistic: it chose the variant and the stopping round.",
            **common,
        ),
        make_record(split="test", values=scores, notes=notes, **common),
    ]
    for r in rows:
        append_record("results/rel-stack/post-post-related.jsonl", r)
    (out / "final.json").write_text(
        json.dumps(
            {"test": scores, "val": res["val"][:3], "variant": tag, "config": config}, indent=2
        )
    )
    print(json.dumps({"test": scores, "mean": np.mean(scores), "std": np.std(scores, ddof=1)}))


if __name__ == "__main__":
    main()
