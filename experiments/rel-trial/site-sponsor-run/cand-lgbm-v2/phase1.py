"""Issue 28, phase 1: where does cand-lgbm-ranker lose MAP on rel-trial/site-sponsor-run val?

Val only; test is never loaded. Rebuilds the baseline's val pairs (same LinkConfig and entity
features as ``../cand-lgbm-ranker/run.py``), scores them with the baseline's three saved
models (best config, seeds 0-2; from the issue-12 cloud checkpoint) and breaks the true
(facility, sponsor) pairs down by what the database knew before the seed time.

    uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-v2/phase1.py \
        --models runs/i28/base/ranker_rel-trial_site-sponsor-run/work/rel-trial/site-sponsor-run/cand-lgbm-ranker/models \
        --out runs/i28/phase1

The val labels are read here only to *measure*; no feature or candidate reads them.
"""

import argparse
import importlib.util
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import relbench

from relconduit.features import feature_vocab
from relconduit.link import DST, SEED, PairBuilder, label_pairs
from relconduit.link_lgbm import _key, _slug, add_entity_features, top_k

HERE = Path(__file__).resolve().parent
BEST = {
    "objective": "lambdarank",
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
}
K = 10


def load_baseline():
    spec = importlib.util.spec_from_file_location(
        "base_run", HERE.parent / "cand-lgbm-ranker" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ap_terms(pred: np.ndarray, true_lists: list[set]) -> tuple[np.ndarray, np.ndarray]:
    """Per seed and slot, the hit flag and the AP term it adds (official MAP@k formula)."""
    hit = np.array([[d in t for d in row] for row, t in zip(pred, true_lists)], dtype=float)
    n = np.array([min(len(t), K) for t in true_lists], dtype=float)
    prec = np.cumsum(hit, axis=1) / (np.arange(K) + 1)
    return hit, prec * hit / n[:, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--out", default="runs/i28/phase1")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = load_baseline()

    ds = relbench.load_dataset("stanford-star/relbench-v1/rel-trial")
    task = ds.load_task("site-sponsor-run")
    val_table = task.get_table("val", mask_input_cols=False)
    val = val_table.df.reset_index(drop=True)
    T = pd.Timestamp(val["timestamp"].iloc[0])
    db = task.get_db()
    print(f"[{time.time() - t0:.0f}s] db loaded; val seeds {len(val):,} at {T.date()}", flush=True)

    # --- baseline val pairs, features and scores --------------------------------------------
    path = out / "val_scored.parquet"
    if path.exists():
        X = pd.read_parquet(path)
    else:
        builder = PairBuilder(db, base.LINK)
        pairs = builder.build(val, "facility_id", "timestamp")
        print(f"[{time.time() - t0:.0f}s] pairs {len(pairs):,}", flush=True)
        sides = {}
        for side, cfg in (("src", base.SRC_FEATURES), ("dst", base.DST_FEATURES)):
            cfg = replace(cfg)
            cfg.vocab = feature_vocab(db, cfg, before=ds.val_timestamp)
            sides[side] = cfg
        X = add_entity_features(db, task, pairs, val, sides)
        X = X.astype({c: np.float32 for c in X.columns if X[c].dtype == np.float64})
        for s in (0, 1, 2):
            m = lgb.Booster(model_file=str(Path(a.models) / f"{_slug(_key(BEST, s))}.txt"))
            X[f"score{s}"] = m.predict(X[m.feature_name()])
        keep = [SEED, DST, "n_candidates", "score0", "score1", "score2"]
        keep += [
            c for c in X.columns if c.startswith("cand__") or c.endswith(("__score", "__rank"))
        ]
        keep += ["ran__count"]
        X = X[keep]
        X.to_parquet(path)
        print(f"[{time.time() - t0:.0f}s] features and scores done", flush=True)
    X["y"] = label_pairs(X, val, "sponsor_id")

    stats = {}
    true_lists = [set(map(int, v)) for v in val["sponsor_id"]]
    n_true = np.array([len(t) for t in true_lists])
    official = []
    for s in (0, 1, 2):
        pred = top_k(X, X[f"score{s}"].to_numpy(), len(val), K)
        (v,) = task.evaluate(pred, val_table).values()
        official.append(float(v))
    stats["baseline_val_map"] = official
    pred = top_k(X, X["score0"].to_numpy(), len(val), K)
    hit, terms = ap_terms(pred, true_lists)
    ap = terms.sum(axis=1)
    assert abs(ap.mean() - official[0]) < 1e-9, (ap.mean(), official[0])

    # --- what the database knew before T --------------------------------------------------
    fs = db.table_dict["facilities_studies"].df
    ss = db.table_dict["sponsors_studies"].df
    fs_b, ss_b = fs[fs["date"] < T], ss[ss["date"] < T]
    ran = fs_b[["facility_id", "nct_id"]].merge(ss_b[["nct_id", "sponsor_id"]], on="nct_id")
    ran_pairs = set(zip(ran["facility_id"], ran["sponsor_id"]))
    warm_fac = set(fs_b["facility_id"])
    spon_n_all = ss_b.groupby("sponsor_id").size()
    spon_n_365 = ss_b[ss_b["date"] >= T - pd.Timedelta(days=365)].groupby("sponsor_id").size()
    fac = db.table_dict["facilities"].df.set_index("facility_id")
    spn = db.table_dict["sponsors"].df.set_index("sponsor_id")

    pos = val[["facility_id", "sponsor_id"]].explode("sponsor_id").reset_index()
    pos = pos.rename(columns={"index": SEED, "sponsor_id": DST})
    pos[DST] = pos[DST].astype(np.int64)
    cand = X[[SEED, DST, "score0"] + [c for c in X.columns if c.startswith("cand__")]]
    pos = pos.merge(cand, on=[SEED, DST], how="left")
    pos["in_cand"] = pos["score0"].notna()
    ranks = X.sort_values([SEED, "score0", DST], ascending=[True, False, True])
    ranks["rank"] = ranks.groupby(SEED).cumcount() + 1
    pos = pos.merge(ranks[[SEED, DST, "rank"]], on=[SEED, DST], how="left")
    pos["in_top10"] = pos["rank"] <= K
    pos["warm"] = pos["facility_id"].isin(warm_fac)
    pos["seen_pair"] = [(f, s) in ran_pairs for f, s in zip(pos["facility_id"], pos[DST])]
    pos["spon_n_all"] = pos[DST].map(spon_n_all).fillna(0).astype(int)
    pos["spon_n_365"] = pos[DST].map(spon_n_365).fillna(0).astype(int)

    # Graph reach: is the sponsor reachable from the facility by each relation before T?
    def reach_place(col):
        e = ran.merge(fac[[col]], left_on="facility_id", right_index=True)[[col, "sponsor_id"]]
        e = e.dropna().drop_duplicates()
        key = set(zip(e[col], e["sponsor_id"]))
        vals = pos["facility_id"].map(fac[col])
        return np.array([(v, s) in key for v, s in zip(vals, pos[DST])])

    for col in ("name", "city", "zip", "state", "country"):
        pos[f"reach_{col}"] = reach_place(col)

    def reach_via(link_table, col):
        """Facility's past studies share a ``col`` value with the sponsor's past studies."""
        lt = db.table_dict[link_table].df
        lt = lt[lt["date"] < T][["nct_id", col]].drop_duplicates()
        fc = fs_b[["facility_id", "nct_id"]].merge(lt, on="nct_id")[["facility_id", col]]
        fc = fc.drop_duplicates()
        sc = ss_b[["sponsor_id", "nct_id"]].merge(lt, on="nct_id")[["sponsor_id", col]]
        sc = set(map(tuple, sc.drop_duplicates().to_numpy()))
        m = pos[[SEED, "facility_id", DST]].merge(fc, on="facility_id")
        hit_ = np.array([(s, c) in sc for s, c in zip(m[DST], m[col])])
        ok = set(map(tuple, m.loc[hit_, [SEED, DST]].drop_duplicates().to_numpy()))
        return np.array([(a_, b_) in ok for a_, b_ in zip(pos[SEED], pos[DST])])

    pos["reach_condition"] = reach_via("conditions_studies", "condition_id")
    pos["reach_intervention"] = reach_via("interventions_studies", "intervention_id")

    # co-sponsor: the sponsor ran a study with one of the facility's past sponsors
    co = ss_b[["nct_id", "sponsor_id"]].merge(ss_b[["nct_id", "sponsor_id"]], on="nct_id")
    co = co[co["sponsor_id_x"] != co["sponsor_id_y"]][["sponsor_id_x", "sponsor_id_y"]]
    co = set(map(tuple, co.drop_duplicates().to_numpy()))
    fac_sp = ran[["facility_id", "sponsor_id"]].drop_duplicates()
    m = pos[[SEED, "facility_id", DST]].merge(fac_sp, on="facility_id")
    hit_ = np.array([(a_, b_) in co for a_, b_ in zip(m["sponsor_id"], m[DST])])
    ok = set(map(tuple, m.loc[hit_, [SEED, DST]].drop_duplicates().to_numpy()))
    pos["reach_cosponsor"] = [(a_, b_) in ok for a_, b_ in zip(pos[SEED], pos[DST])]

    # Text: a sponsor-name token (>= 4 letters, not generic) inside the facility name.
    generic = {
        "university",
        "hospital",
        "medical",
        "center",
        "centre",
        "clinical",
        "research",
        "institute",
        "health",
        "clinic",
        "national",
        "foundation",
        "college",
        "school",
        "group",
        "study",
        "site",
        "sites",
        "pharmaceuticals",
        "pharma",
        "pharmaceutical",
        "inc",
        "ltd",
        "llc",
        "corporation",
        "company",
        "limited",
        "department",
        "general",
    }

    def toks(s):
        return {w for w in re.findall(r"[a-z]{4,}", str(s).lower()) if w not in generic}

    fn = pos["facility_id"].map(fac["name"]).map(toks)
    sn = pos[DST].map(spn["name"]).map(toks)
    pos["name_token"] = [bool(x & y) for x, y in zip(fn, sn)]

    reach_cols = [
        "seen_pair",
        "reach_name",
        "reach_cosponsor",
        "reach_city",
        "reach_zip",
        "reach_state",
        "reach_country",
        "reach_condition",
        "reach_intervention",
        "name_token",
    ]

    def first_reach(r):
        for c in reach_cols:
            if r[c]:
                return c
        return "none" if r["spon_n_all"] > 0 else "new_sponsor"

    pos["first_reach"] = pos.apply(first_reach, axis=1)
    pos.to_parquet(out / "val_positives.parquet")

    def table(df, by):
        g = df.groupby(by, dropna=False)
        return pd.DataFrame(
            {
                "positives": g.size(),
                "share": g.size() / len(df),
                "in_cand": g["in_cand"].mean(),
                "in_top10": g["in_top10"].mean(),
            }
        ).round(4)

    pos["spon_365_bucket"] = pd.cut(
        pos["spon_n_365"], [-1, 0, 2, 10, 50, 10**6], labels=["0", "1-2", "3-10", "11-50", ">50"]
    )
    pos["spon_status"] = np.where(pos["spon_n_all"] == 0, "new", "established")
    tabs = {
        "warm": table(pos, "warm"),
        "sponsor": table(pos, "spon_status"),
        "seen_pair_by_warm": table(pos, ["warm", "seen_pair"]),
        "first_reach_by_warm": table(pos, ["warm", "first_reach"]),
        "sponsor_recent_activity": table(pos, "spon_365_bucket"),
    }
    for c in reach_cols:
        tabs[f"reach::{c}"] = table(pos, ["warm", c])

    # --- ceilings --------------------------------------------------------------------------
    hits_in_cand = pos.groupby(SEED)["in_cand"].sum().reindex(range(len(val)), fill_value=0)
    oracle_cand = np.minimum(hits_in_cand.to_numpy(), K) / np.minimum(n_true, K)
    warm_seed = val["facility_id"].isin(warm_fac).to_numpy()
    stats["ceiling"] = {
        "current_map": float(ap.mean()),
        "oracle_rank_of_candidates": float(oracle_cand.mean()),
        "perfect_recall_and_rank": 1.0,
        "candidate_recall": float((hits_in_cand / n_true).mean()),
        "share_of_seeds_with_any_true_candidate": float((hits_in_cand > 0).mean()),
        "current_map_warm": float(ap[warm_seed].mean()),
        "current_map_cold": float(ap[~warm_seed].mean()),
        "oracle_cand_warm": float(oracle_cand[warm_seed].mean()),
        "oracle_cand_cold": float(oracle_cand[~warm_seed].mean()),
        "share_warm_seeds": float(warm_seed.mean()),
        # Of seeds with every true dst in the candidates, how much MAP does the ranker get?
        "current_map_on_fully_covered": float(ap[(hits_in_cand.to_numpy() >= n_true)].mean()),
        "share_fully_covered": float((hits_in_cand.to_numpy() >= n_true).mean()),
        "true_candidates_ranked_below_10": int(((pos["in_cand"]) & (~pos["in_top10"])).sum()),
        "true_candidates_in_top10": int(pos["in_top10"].sum()),
        "positives": len(pos),
    }
    # Oracle by candidate-pool size: candidates ordered by the ranker; best MAP if the true
    # ones in the top N were placed first. Shows how much ranking deeper candidates matters.
    stats["oracle_within_ranker_top_n"] = {}
    for n in (10, 20, 30, 50, 100, 200):
        h = (pos["rank"] <= n).groupby(pos[SEED]).sum().reindex(range(len(val)), fill_value=0)
        h = h.to_numpy()
        stats["oracle_within_ranker_top_n"][n] = float(
            (np.minimum(h, K) / np.minimum(n_true, K)).mean()
        )

    # --- where the 10 slots go ----------------------------------------------------------------
    seeds_idx = np.repeat(np.arange(len(val)), K)
    slot = pd.DataFrame(
        {SEED: seeds_idx, DST: pred.ravel(), "hit": hit.ravel(), "term": terms.ravel() / len(val)}
    )
    slot = slot[slot[DST] >= 0]
    slot["facility_id"] = val["facility_id"].to_numpy()[slot[SEED]]
    slot["seen_pair"] = [(f, s) in ran_pairs for f, s in zip(slot["facility_id"], slot[DST])]
    slot["warm"] = slot["facility_id"].isin(warm_fac)
    g = slot.groupby(["warm", "seen_pair"])
    tabs["slots"] = pd.DataFrame(
        {
            "slots": g.size(),
            "hits": g["hit"].sum(),
            "precision": g["hit"].mean(),
            "map_contrib": g["term"].sum(),
        }
    ).round(4)

    # Is the number of true sponsors predictable? Correlate with pre-T facts.
    fac_n365 = fs_b[fs_b["date"] >= T - pd.Timedelta(days=365)].groupby("facility_id").size()
    fac_sp_n = ran.groupby("facility_id")["sponsor_id"].nunique()
    fac_sp_365 = ran[ran["facility_id"].isin(fac_n365.index)]
    v = pd.DataFrame(
        {
            "n_true": n_true,
            "fac_studies_365": val["facility_id"].map(fac_n365).fillna(0).to_numpy(),
            "fac_distinct_sponsors_all": val["facility_id"].map(fac_sp_n).fillna(0).to_numpy(),
            "warm": warm_seed.astype(int),
            "ap": ap,
        }
    )
    stats["n_true"] = {
        "distribution": v["n_true"].clip(upper=11).value_counts().sort_index().to_dict(),
        "spearman_vs_fac_studies_365": float(
            v["n_true"].corr(v["fac_studies_365"], method="spearman")
        ),
        "spearman_vs_fac_distinct_sponsors": float(
            v["n_true"].corr(v["fac_distinct_sponsors_all"], method="spearman")
        ),
        "map_by_n_true": v.groupby(v["n_true"].clip(upper=11))["ap"].mean().round(4).to_dict(),
    }
    # How concentrated are the positives? A sponsor true for many val facilities = a big
    # multi-site trial starting in the window.
    breadth = pos.groupby(DST).size().sort_values(ascending=False)
    stats["sponsor_breadth"] = {
        "distinct_true_sponsors": len(breadth),
        "share_positives_top10_sponsors": float(breadth.head(10).sum() / len(pos)),
        "share_positives_top100_sponsors": float(breadth.head(100).sum() / len(pos)),
        "share_positives_sponsor_true_for_1_facility": float(
            breadth[breadth == 1].sum() / len(pos)
        ),
    }
    pos["breadth"] = pos[DST].map(breadth)
    pos["breadth_bucket"] = pd.cut(
        pos["breadth"], [0, 1, 5, 20, 100, 10**6], labels=["1", "2-5", "6-20", "21-100", ">100"]
    )
    tabs["breadth_by_warm"] = table(pos, ["warm", "breadth_bucket"])
    _ = fac_sp_365
    stats["minutes"] = (time.time() - t0) / 60

    with open(out / "phase1.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    with open(out / "phase1_tables.md", "w") as f:
        f.writelines(f"### {name}\n\n```\n{t.to_string()}\n```\n\n" for name, t in tabs.items())
    print(json.dumps(stats, indent=2, default=str))
    for name, t in tabs.items():
        print(f"\n### {name}\n{t.to_string()}")


if __name__ == "__main__":
    main()
