"""rel-stack/post-votes: LightGBM on all train rows, tuned on val, with time-safe text and tag
features. See README.md.

    run.py tune --out runs/pv-full [--jobs 6] [--smoke]   # features + every val stage
    run.py test --out runs/pv-full                        # score test once, append results

``tune`` builds the features (train, val and test seeds; test labels are never read), then runs
the stages below, every choice by val nMAE. Train and val features go to ``<out>-features/``,
test features to ``<out>-test/`` (small, brought back from the cloud). Each fit is written to
``<out>/fits/<key>.json`` (and its model to ``<key>.txt``) as soon as it ends, and skipped if
already there, so a restarted machine resumes from the ``<out>`` checkpoint. ``test`` loads the chosen config's three models and scores test
through ``relbench.submit``: the only place test labels are read.

Stages (seed 0 unless said):

* A, sample size: the #11 config on 500k, 1M and all train rows, plus the #11 feature set
  (``base11``, which still has the posts' dump-time ``ContentLicense``) at 500k and all rows.
* B, features, at the chosen size: ``base`` (``base11`` without ``ContentLicense``),
  ``+text``, ``+tags``, ``+text+tags``.
* C, one change at a time from the #11 config: ``num_leaves`` 127 / 255,
  ``min_child_samples`` 100 / 500, ``feature_fraction`` 0.5, ``lambda_l2`` 10, Huber loss.
  Then every change that helped, together, if more than one did.
* D, side by side: seeds 1 and 2 of the best so far, and seeds 0, 1 and 2 of it at learning
  rate 0.01 (up to 10,000 rounds). The one with the better mean over three seeds is the final
  config, and its three models are the test models.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

DATASET, TASK = "rel-stack", "post-votes"
APPROACH = "experiments/rel-stack/post-votes/dfs-lightgbm-full"
SAMPLES = {"500k": 500_000, "1m": 1_000_000, "all": None}
REF = {
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "objective": "regression_l1",
    "learning_rate": 0.03,
}
BASE = {"bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "metric": "l1"}
CHANGES = [
    {"num_leaves": 127},
    {"num_leaves": 255},
    {"min_child_samples": 100},
    {"min_child_samples": 500},
    {"feature_fraction": 0.5},
    {"lambda_l2": 10.0},
    {"objective": "huber", "alpha": 1.0},
]
ROUNDS = {0.03: 3000, 0.01: 10000}
EARLY_STOPPING = 100
FEATURE_SETS = (
    "base11",
    "base",
    "base+text",
    "base+tags",
    "base+text+tags",
    "base+text+tags+graph",
    "base+text+tags+short",
    "base+text+tags+graph+short",
)
ROUND2_SETS = FEATURE_SETS[-3:]
SHORT_WINDOWS = (7, 14)
SHORT_SUFFIXES = ("__7d", "__14d")


def _dfs11():
    """The #11 experiment's feature config and type-code transform, unchanged."""
    import importlib.util

    path = HERE.parent / "dfs-lightgbm" / "run.py"
    spec_ = importlib.util.spec_from_file_location("post_votes_dfs_lightgbm", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod.FEATURES, mod.codes_as_text


# --- features ------------------------------------------------------------------------------


def is_dump_licence(col: str) -> bool:
    """A ``ContentLicense`` read off a ``posts`` row: the licence of the post's *last* edit,
    from the 2023 dump, so it can reveal edits after the seed time. ``postHistory`` and
    ``comments`` carry the licence of their own row and are kept."""
    if "ContentLicense" not in col:
        return False
    path = col.split("__")[0]
    return path.split(">")[-1].split(".")[0] == "posts"


def is_short(col: str) -> bool:
    return col.endswith(SHORT_SUFFIXES)


def select(columns, feature_set: str) -> list[str]:
    plain = [c for c in columns if not c.startswith("text__") and not is_short(c)]
    base11 = [c for c in plain if ">" not in c]
    if feature_set == "base11":
        return base11
    base = [c for c in base11 if not is_dump_licence(c)]
    text = [c for c in columns if c.startswith("text__") and not c.startswith("text__tag_")]
    tags = [c for c in columns if c.startswith("text__tag_")]
    graph = [c for c in plain if ">" in c and not is_dump_licence(c)]
    short = [c for c in columns if is_short(c) and ">" not in c and not is_dump_licence(c)]
    return {
        "base": base,
        "base+text": base + text,
        "base+tags": base + tags,
        "base+text+tags": base + text + tags,
        "base+text+tags+graph": base + text + tags + graph,
        "base+text+tags+short": base + text + tags + short,
        "base+text+tags+graph+short": base + text + tags + graph + short,
    }[feature_set]


def build(out: Path, smoke: bool) -> None:
    """Feature parquet files for train, val and test, and the labels of train and val."""
    fdir, tdir = feature_dirs(out)
    done = fdir / "done.json"
    built = done.exists() and (tdir / "X_test.parquet").exists()
    if built and any(is_short(c) for c in json.loads(done.read_text())["columns"]):
        return
    import relbench

    FEATURES, codes_as_text = _dfs11()

    from relconduit.db import load_task_db
    from relconduit.features import build_features, feature_vocab
    from relconduit.stack_text import load_revision_stats, post_text_features, revisions_table

    t0 = time.time()
    ds = relbench.load_dataset(f"stanford-star/relbench-v1/{DATASET}")
    task = ds.load_task(TASK)
    db = codes_as_text(load_task_db(task, upto_test_timestamp=True))
    stats = load_revision_stats(Path(ds.db_dir) / "postHistory.parquet", ds.test_timestamp)
    db.table_dict["revisions"] = revisions_table(stats)
    print(f"[features] db loaded in {time.time() - t0:.0f}s, {len(stats):,} revisions", flush=True)
    base_db = type(db)({k: v for k, v in db.table_dict.items() if k != "revisions"})
    cfg = FEATURES
    cfg.vocab = feature_vocab(base_db, cfg, before=ds.val_timestamp)
    # Round 2: the same paths over the last 7 and 14 days (hop 1 only).
    short_cfg = replace(
        cfg, windows=SHORT_WINDOWS, aggs=("mean", "sum"), hop2=False, siblings_via=()
    )
    key = [task.entity_col, task.time_col]
    fdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = task.get_table(split).df  # test: labels masked
        if smoke:
            df = df.sample(n=min(len(df), 20_000 if split == "train" else 5_000), random_state=0)
            df = df.reset_index(drop=True)
        parts = []
        chunk = 400_000
        for lo in range(0, len(df), chunk):
            seeds = df.iloc[lo : lo + chunk][key]
            X = build_features(base_db, seeds, task.entity_table, *key, cfg)
            T = post_text_features(db, seeds, *key)
            S = build_features(base_db, seeds, task.entity_table, *key, short_cfg)
            S = S[[c for c in S.columns if is_short(c)]]
            X = pd.concat([X, T, S], axis=1)
            num = X.select_dtypes(include="number").columns
            X[num] = X[num].astype("float32")
            parts.append(X)
            print(
                f"[features] {split} rows {lo:,}+{len(seeds):,} at {time.time() - t0:.0f}s",
                flush=True,
            )
        X = pd.concat(parts)
        where = tdir if split == "test" else fdir
        X.reset_index(drop=True).to_parquet(where / f"X_{split}.parquet")
        df[key].reset_index(drop=True).to_parquet(where / f"seeds_{split}.parquet")
        if split != "test":
            np.save(fdir / f"y_{split}.npy", df[task.target_col].to_numpy(dtype=float))
        print(f"[features] {split} {X.shape} done at {time.time() - t0:.0f}s", flush=True)
        del X, parts
    meta = {
        "columns": list(pd.read_parquet(fdir / "X_val.parquet").columns),
        "minutes": (time.time() - t0) / 60,
    }
    (fdir / "done.json").write_text(json.dumps(meta))


def feature_dirs(out: Path) -> tuple[Path, Path]:
    return out.with_name(out.name + "-features"), out.with_name(out.name + "-test")


# --- fits -------------------------------------------------------------------------------------


def fit_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def fit_name(spec: dict) -> str:
    import hashlib

    return hashlib.sha1(fit_key(spec).encode()).hexdigest()[:12]


_DATA: dict = {}


def _load(fdir: Path, feature_set: str):
    """Train and val features of one feature set (one set held per process, to bound memory)."""
    if _DATA.get("set") != feature_set:
        _DATA.clear()
        columns = json.loads((fdir / "done.json").read_text())["columns"]
        cols = select(columns, feature_set)
        _DATA["X_train"] = pd.read_parquet(fdir / "X_train.parquet", columns=cols)
        _DATA["y_train"] = np.load(fdir / "y_train.npy")
        _DATA["X_val"] = pd.read_parquet(fdir / "X_val.parquet", columns=cols)
        _DATA["y_val"] = np.load(fdir / "y_val.npy")
        _DATA["set"] = feature_set
    return _DATA


def run_fit(out: str, spec: dict, threads: int, smoke: bool) -> dict:
    """Fit one spec (``sample``, ``features``, ``config``, ``seed``), score val, save it."""
    import lightgbm as lgb
    import relbench

    out = Path(out)
    fits = out / "fits"
    name = fit_name(spec)
    if (fits / f"{name}.json").exists():
        return json.loads((fits / f"{name}.json").read_text())
    t0 = time.time()
    print(f"[fit {name}] start {fit_key(spec)}", flush=True)
    try:
        fdir, _ = feature_dirs(out)
        d = _load(fdir, spec["features"])
        cols = list(d["X_train"].columns)
        n = len(d["X_train"])
        size = SAMPLES[spec["sample"]]
        if smoke and size is not None:
            size = min(size // 100, n)
        idx = np.arange(n)
        if size is not None and size < n:
            idx = np.sort(np.random.default_rng(0).choice(n, size=size, replace=False))
        cfg = dict(spec["config"])
        lr = cfg["learning_rate"]
        params = {
            **BASE,
            **cfg,
            "seed": spec["seed"],
            "num_threads": threads,
            "deterministic": True,
        }
        X_tr = d["X_train"] if len(idx) == n else d["X_train"].iloc[idx]
        dtrain = lgb.Dataset(X_tr, d["y_train"][idx])
        dval = lgb.Dataset(d["X_val"][cols], d["y_val"], reference=dtrain)
        rounds = 60 if smoke else ROUNDS[lr]
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(250)],
        )
        pred = model.predict(d["X_val"][cols], num_iteration=model.best_iteration)
        task = relbench.load_dataset(f"stanford-star/relbench-v1/{DATASET}").load_task(TASK)
        val_table = task.get_table("val")
        if smoke:
            seeds = pd.read_parquet(fdir / "seeds_val.parquet")
            val_table.df = val_table.df.merge(seeds, on=list(seeds.columns), how="right")
        (value,) = task.evaluate(pred, val_table).values()
        row = {
            "key": fit_key(spec),
            **spec,
            "val": float(value),
            "best_iteration": model.best_iteration,
            "n_train": len(idx),
            "n_features": len(cols),
            "minutes": (time.time() - t0) / 60,
        }
        fits.mkdir(parents=True, exist_ok=True)
        model.save_model(str(fits / f"{name}.txt.tmp"), num_iteration=model.best_iteration)
        os.replace(fits / f"{name}.txt.tmp", fits / f"{name}.txt")
        (fits / f"{name}.json.tmp").write_text(json.dumps(row))
        os.replace(fits / f"{name}.json.tmp", fits / f"{name}.json")
        print(
            f"[fit {name}] val {value:.5f} iter {model.best_iteration} in {row['minutes']:.1f} min",
            flush=True,
        )
        return row
    except Exception:
        import traceback

        print(f"[fit {name}] FAILED\n{traceback.format_exc()}", flush=True)
        raise


def run_all(out: Path, specs: list[dict], jobs: int, smoke: bool) -> list[dict]:
    cpus = os.cpu_count() or 1
    jobs = max(1, min(jobs, len(specs)))
    threads = max(1, cpus // jobs)
    print(f"[stage] {len(specs)} fits, {jobs} at a time, {threads} threads each", flush=True)
    if jobs == 1:
        return [run_fit(str(out), s, threads, smoke) for s in specs]
    import multiprocessing as mp

    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
        return list(
            ex.map(
                run_fit,
                [str(out)] * len(specs),
                specs,
                [threads] * len(specs),
                [smoke] * len(specs),
            )
        )


def spec(sample="all", features="base", config=None, seed=0) -> dict:
    return {"sample": sample, "features": features, "config": config or dict(REF), "seed": seed}


def tune(out: Path, jobs: int, smoke: bool) -> dict:
    t0 = time.time()
    build(out, smoke)
    log = {}

    a = run_all(
        out,
        [spec("500k", "base11"), spec("500k"), spec("1m"), spec("all"), spec("all", "base11")],
        jobs,
        smoke,
    )
    log["A"] = a
    sample = min((r for r in a if r["features"] == "base"), key=lambda r: r["val"])["sample"]
    print(f"[stage A] sample: {sample}", flush=True)

    b_specs = [spec(sample, f) for f in ("base", "base+text", "base+tags", "base+text+tags")]
    b = run_all(out, b_specs, jobs, smoke)
    log["B"] = b
    best_b = min(b, key=lambda r: r["val"])
    features = best_b["features"]
    print(f"[stage B] features: {features}", flush=True)

    ref_val = best_b["val"]
    c_specs = [spec(sample, features, {**REF, **ch}) for ch in CHANGES]
    c = run_all(out, c_specs, jobs, smoke)
    helped = [ch for ch, r in zip(CHANGES, c) if r["val"] < ref_val]
    # One value per parameter: the best of those that helped.
    by_param: dict[str, tuple[float, dict]] = {}
    for ch, r in zip(CHANGES, c):
        if ch in helped:
            p = next(iter(ch))
            if p not in by_param or r["val"] < by_param[p][0]:
                by_param[p] = (r["val"], ch)
    candidates = [best_b] + c
    if len(by_param) > 1:
        combo = dict(REF)
        for _, ch in by_param.values():
            combo.update(ch)
        candidates += run_all(out, [spec(sample, features, combo)], 1, smoke)
    log["C"] = candidates
    best_c = min(candidates, key=lambda r: r["val"])
    print(f"[stage C] config: {best_c['config']} val {best_c['val']:.5f}", flush=True)

    # D: seeds 1 and 2 of that config, and seeds 0-2 at learning rate 0.01, side by side; keep
    # whichever has the better mean over the three seeds.
    fast = best_c["config"]
    slow = {**fast, "learning_rate": 0.01}
    d_specs = [spec(sample, features, fast, s) for s in (1, 2)]
    d_specs += [spec(sample, features, slow, s) for s in (0, 1, 2)]
    d = run_all(out, d_specs, jobs, smoke)
    log["D"] = d
    runs = {"fast": [best_c] + d[:2], "slow": d[2:]}
    pick = min(runs, key=lambda k: np.mean([r["val"] for r in runs[k]]))
    final = runs[pick]
    best = final[0]
    vals = [r["val"] for r in final]
    print(f"[stage D] config: {best['config']} val {np.mean(vals):.5f}", flush=True)
    summary = {
        "sample": sample,
        "features": features,
        "config": best["config"],
        "final_fits": [fit_name(spec(sample, features, best["config"], s)) for s in (0, 1, 2)],
        "val_scores": vals,
        "val_mean": float(np.mean(vals)),
        "val_std": float(np.std(vals, ddof=1)),
        "best_iterations": [r["best_iteration"] for r in final],
        "stages": log,
        "runtime_minutes": (time.time() - t0) / 60,
        "smoke": smoke,
    }
    (out / "summary_val.json").write_text(json.dumps(summary, indent=2))
    (out / "done.json").write_text(json.dumps({"ok": True}))
    print(json.dumps({k: v for k, v in summary.items() if k != "stages"}, indent=2), flush=True)
    return summary


def tune2(out: Path, jobs: int, smoke: bool) -> dict:
    """Round 2, after round 1 (``tune``) and before test: the graph paths (hop 2 and the owner's
    other posts, licences still dropped) and 7- and 14-day windows, on top of round 1's
    features. Each set at round 1's lr-0.03 config over three seeds; the best, if its mean beats
    round 1's at that config, again at lr 0.01 over three seeds, kept if it beats round 1's final.
    """
    t0 = time.time()
    r1_path = out / "summary_val_round1.json"
    if not r1_path.exists():
        (out / "summary_val.json").rename(r1_path)
    r1 = json.loads(r1_path.read_text())
    build(out, smoke)
    sample, base_set, slow = r1["sample"], r1["features"], r1["config"]
    fast = {**slow, "learning_rate": 0.03}
    seeds = (0, 1, 2)
    fast_r1 = run_all(out, [spec(sample, base_set, fast, s) for s in seeds], jobs, smoke)
    f = run_all(out, [spec(sample, fs, fast, s) for fs in ROUND2_SETS for s in seeds], jobs, smoke)
    means = {base_set: np.mean([r["val"] for r in fast_r1])}
    for i, fs in enumerate(ROUND2_SETS):
        means[fs] = float(np.mean([r["val"] for r in f[3 * i : 3 * i + 3]]))
    print(f"[stage F] 3-seed val means at lr 0.03: {means}", flush=True)
    best_set = min(means, key=means.get)
    final_set, final = base_set, None
    g = []
    if best_set != base_set:
        g = run_all(out, [spec(sample, best_set, slow, s) for s in seeds], jobs, smoke)
        if np.mean([r["val"] for r in g]) < r1["val_mean"]:
            final_set, final = best_set, g
    if final is None:
        final = [json.loads((out / "fits" / f"{n}.json").read_text()) for n in r1["final_fits"]]
    vals = [r["val"] for r in final]
    print(f"[stage G] final features {final_set}: val {np.mean(vals):.5f}", flush=True)
    summary = {
        **r1,
        "features": final_set,
        "final_fits": [fit_name(spec(sample, final_set, slow, s)) for s in seeds],
        "val_scores": vals,
        "val_mean": float(np.mean(vals)),
        "val_std": float(np.std(vals, ddof=1)),
        "best_iterations": [r["best_iteration"] for r in final],
        "round2": {"F_means": means, "F": fast_r1 + f, "G": g},
        "runtime_minutes": r1["runtime_minutes"] + (time.time() - t0) / 60,
    }
    (out / "summary_val.json").write_text(json.dumps(summary, indent=2))
    (out / "done2.json").write_text(json.dumps({"ok": True}))
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("stages", "round2")}, indent=2),
        flush=True,
    )
    return summary


# --- test, once ---------------------------------------------------------------------------------


def test(out: Path) -> None:
    import lightgbm as lgb
    import relbench
    from relbench.submit import evaluate_task, write_prediction_table

    from relconduit.results import append_record, git_commit, make_record

    summary = json.loads((out / "summary_val.json").read_text())
    if summary["smoke"]:
        raise SystemExit("a smoke run never scores test")
    task = relbench.load_dataset(f"stanford-star/relbench-v1/{DATASET}").load_task(TASK)
    _, tdir = feature_dirs(out)
    X = pd.read_parquet(tdir / "X_test.parquet")
    seeds = pd.read_parquet(tdir / "seeds_test.parquet")
    ref = task.get_table("test").df[seeds.columns].reset_index(drop=True)
    assert ref.equals(seeds), "test features are not in the test table's row order"
    scores = []
    for seed, name in enumerate(summary["final_fits"]):
        model = lgb.Booster(model_file=str(out / "fits" / f"{name}.txt"))
        path = out / "pred" / f"seed{seed}" / f"{DATASET}__{TASK}.csv"
        write_prediction_table(task, model.predict(X[model.feature_name()]), path)
        (value,) = evaluate_task(f"{DATASET}/{TASK}", path).values()
        scores.append(float(value))
        print(f"test seed {seed}: {value:.5f}", flush=True)
    summary["test_scores"] = scores
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    notes = (
        f"LightGBM on relconduit.features + relconduit.stack_text, every choice on val "
        f"(sample {summary['sample']}, features {summary['features']}, config "
        f"{summary['config']}, rounds by early stopping on val). Train on train only."
    )
    common = {
        "approach": APPROACH,
        "metric": task.metrics[0].__name__,
        "runtime_minutes": summary["runtime_minutes"],
        "commit": git_commit(),
    }
    for split, values, extra in (
        ("val", summary["val_scores"], " Val is optimistic: it made every choice."),
        ("test", scores, " Test scored once, for the final config."),
    ):
        append_record(
            f"results/{DATASET}/{TASK}.jsonl",
            make_record(
                split=split,
                values=values,
                notes=notes + extra,
                db="upto_test_timestamp=True",
                **common,
            ),
        )
    print(json.dumps({"val": summary["val_scores"], "test": scores}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["tune", "tune2", "test"])
    p.add_argument("--out", default="runs/pv-full")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--smoke", action="store_true", help="tiny slice, 60 rounds; no results")
    a = p.parse_args()
    if a.command == "tune":
        tune(Path(a.out), a.jobs, a.smoke)
    elif a.command == "tune2":
        tune2(Path(a.out), a.jobs, a.smoke)
    else:
        test(Path(a.out))
