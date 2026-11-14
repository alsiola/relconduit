"""Issue 29, phase 1: where does cand-lgbm-ranker lose MAP on rel-stack post-post-related?

Val only; test is never read. Rebuilds the val pairs of ``../cand-lgbm-ranker`` (same config,
same vocab), scores them with that run's three saved models for the chosen config (downloaded
from the issue-12 checkpoint into ``--models``), checks they reproduce the logged val MAPs,
then breaks the misses down. Writes ``<out>/phase1.json`` and ``<out>/positives.parquet``.

Some diagnostics below read the ``posts`` table's ``Title``/``Tags``/``Body`` as of the 2023
dump (that is what the ranker reads too); the as-of-seed-time tags and title from
``postHistory`` are measured beside them, to show how much the dump's later edits matter.
"""

from __future__ import annotations

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
CONFIG = {
    "objective": "lambdarank",
    "num_leaves": 15,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
}


def _load_run_module():
    path = HERE.parent / "cand-lgbm-ranker" / "run.py"
    spec = importlib.util.spec_from_file_location("ppr_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ap_rows(pred: np.ndarray, true_lists: list[set]) -> np.ndarray:
    """Per-row AP@k exactly as ``relbench.metrics.map`` (rows with no true dst dropped)."""
    k = pred.shape[1]
    out = []
    for p, tr in zip(pred, true_lists):
        if not tr:
            continue
        isin = np.array([d in tr for d in p], dtype=float)
        prec = np.cumsum(isin) / (np.arange(k) + 1)
        out.append(float((prec * isin).sum() / min(len(tr), k)))
    return np.array(out)


def _same(a, b) -> bool:
    return bool(pd.notna(a) and pd.notna(b) and a == b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="the issue-12 cand-lgbm-ranker models dir")
    p.add_argument("--out", default="runs/p29/phase1")
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run = _load_run_module()

    ds = relbench.load_dataset("stanford-star/relbench-v1/rel-stack")
    task = ds.load_task("post-post-related")
    k = task.eval_k
    val_table = task.get_table("val", mask_input_cols=False)
    val = val_table.df.reset_index(drop=True)
    T = pd.Timestamp(val["timestamp"].iloc[0])
    assert (val["timestamp"] == T).all()
    db = task.get_db()
    print(f"[{time.time() - t0:.0f}s] db loaded; val {len(val)} rows at {T}", flush=True)

    # --- rebuild val pairs and score -----------------------------------------------------
    pairs_path = out / "pairs_val.parquet"
    if pairs_path.exists():
        X = pd.read_parquet(pairs_path)
    else:
        builder = PairBuilder(db, run.LINK)
        pairs = builder.build(val, "PostId", "timestamp")
        cfg = replace(run.FEATURES)
        cfg.vocab = feature_vocab(db, cfg, before=ds.val_timestamp)
        X = add_entity_features(db, task, pairs, val, {"src": cfg, "dst": cfg})
        X = X.astype({c: np.float32 for c in X.columns if X[c].dtype == np.float64})
        X.to_parquet(pairs_path)
    y = label_pairs(X, val, "postLinksIdList")
    print(f"[{time.time() - t0:.0f}s] val pairs {X.shape}, positives {int(y.sum())}", flush=True)

    true_lists = [{int(d) for d in v} for v in val["postLinksIdList"]]
    scores, maps, preds = [], [], []
    for seed in (0, 1, 2):
        m = lgb.Booster(model_file=str(Path(a.models) / f"{_slug(_key(CONFIG, seed))}.txt"))
        s = m.predict(X[m.feature_name()])
        pred = top_k(X, s, len(val), k)
        (v,) = task.evaluate(pred, val_table).values()
        scores.append(s)
        maps.append(float(v))
        preds.append(pred)
        print(f"seed {seed}: val MAP {v:.4f}", flush=True)
    X = X.assign(y=y).copy()
    for i, s in enumerate(scores):
        X[f"score{i}"] = s
        X[f"rank{i}"] = X.groupby(SEED)[f"score{i}"].rank(ascending=False, method="first")
    # Seed-mean ensemble (rank-average of the three) for the per-row analysis.
    X["score_mean"] = np.mean([X[f"rank{i}"] for i in range(3)], axis=0) * -1
    X["rank_mean"] = X.groupby(SEED)["score_mean"].rank(ascending=False, method="first")
    pred_mean = top_k(X, X["score_mean"].to_numpy(), len(val), k)
    (map_mean,) = task.evaluate(pred_mean, val_table).values()

    # --- per-positive table ---------------------------------------------------------------
    posts = db.table_dict["posts"].df[
        ["Id", "PostTypeId", "ParentId", "CreationDate", "Title", "Tags", "OwnerUserId"]
    ]
    posts = posts.set_index("Id")
    links = db.table_dict["postLinks"].df
    before = links[links["CreationDate"] < T]
    window = links[(links["CreationDate"] > T) & (links["CreationDate"] <= T + task.timedelta)]
    link_type = window.groupby(["PostId", "RelatedPostId"])["LinkTypeId"].max()
    deg = pd.concat([before["PostId"], before["RelatedPostId"]]).value_counts()

    # As-of-T tags and titles from postHistory (latest revision strictly before T).
    ph = db.table_dict["postHistory"].df
    ph = ph[(ph["CreationDate"] < T) & ph["PostHistoryTypeId"].isin([1, 3, 4, 6])]
    ph = ph.sort_values(["PostId", "CreationDate", "Id"])
    tag_rev = ph[ph["PostHistoryTypeId"].isin([3, 6])].groupby("PostId")["Text"].last()
    title_rev = ph[ph["PostHistoryTypeId"].isin([1, 4])].groupby("PostId")["Text"].last()

    def tags_of(pid, asof):
        if asof:
            s = tag_rev.get(pid)
        else:
            s = posts["Tags"].get(pid) if pid in posts.index else None
        if not isinstance(s, str):
            return set()
        return set(re.findall(r"<([^>]+)>", s)) or set(s.strip("<>").split("><")) - {""}

    def words_of(pid, asof):
        s = title_rev.get(pid) if asof else posts["Title"].get(pid)
        return set(run.title_words(s)) if isinstance(s, str) else set()

    def question_of(pid):
        """The post itself if a question, else its parent question (answers carry no tags)."""
        if pid in posts.index and posts.at[pid, "PostTypeId"] == 2:
            par = posts.at[pid, "ParentId"]
            return int(par) if pd.notna(par) else pid
        return pid

    rows = []
    cand = X[X["y"] == 1].set_index([SEED, DST])
    src_cols = [c for c in X.columns if c.startswith("cand__")]
    n_cand = X.groupby(SEED).size()
    for i, r in val.iterrows():
        src = int(r["PostId"])
        for d in r["postLinksIdList"]:
            d = int(d)
            hit = (i, d) in cand.index
            row = {
                "seed": i,
                "src": src,
                "dst": d,
                "n_true": len(true_lists[i]),
                "candidate": hit,
                "link_type": int(link_type.get((src, d), -1)),
                "dst_type": int(posts.at[d, "PostTypeId"]) if d in posts.index else -1,
                "src_type": int(posts.at[src, "PostTypeId"]) if src in posts.index else -1,
                "dst_age_days": (T - posts.at[d, "CreationDate"]) / pd.Timedelta(days=1),
                "src_age_days": (T - posts.at[src, "CreationDate"]) / pd.Timedelta(days=1),
                "dst_links_before": int(deg.get(d, 0)),
                "src_links_before": int(deg.get(src, 0)),
                "same_owner": _same(posts.at[src, "OwnerUserId"], posts.at[d, "OwnerUserId"]),
            }
            qs, qd = question_of(src), question_of(d)
            for asof in (False, True):
                suf = "_asof" if asof else "_dump"
                row["tag" + suf] = len(tags_of(qs, asof) & tags_of(qd, asof))
                row["word" + suf] = len(words_of(qs, asof) & words_of(qd, asof))
            for c in src_cols:
                row[c] = bool(hit and cand.at[(i, d), c] > 0)
            for j in range(3):
                row[f"rank{j}"] = float(cand.at[(i, d), f"rank{j}"]) if hit else np.nan
            row["rank_mean"] = float(cand.at[(i, d), "rank_mean"]) if hit else np.nan
            rows.append(row)
    P = pd.DataFrame(rows)
    P.to_parquet(out / "positives.parquet")
    print(f"[{time.time() - t0:.0f}s] {len(P)} positives", flush=True)

    # --- breakdowns -------------------------------------------------------------------------
    def bucket(name, col, bins, labels):
        b = pd.cut(P[col], bins=bins, labels=labels, right=False)
        g = P.groupby(b, observed=False).agg(
            n=("candidate", "size"),
            recall=("candidate", "mean"),
            in_top100=("rank_mean", lambda s: float((s <= 100).sum())),
        )
        g["share_of_positives"] = g["n"] / len(P)
        g["top100_rate"] = g["in_top100"] / g["n"].clip(lower=1)
        g["share_of_misses"] = P.groupby(b, observed=False)["candidate"].apply(
            lambda s: (~s).sum()
        ).to_numpy() / max(1, (~P["candidate"]).sum())
        return {name: g.reset_index().astype({col: str}).to_dict(orient="records")}

    stats = {
        "val_rows": len(val),
        "positives": len(P),
        "mean_true_per_row": float(P.groupby("seed").size().mean()),
        "true_per_row_hist": P.groupby("seed").size().value_counts().sort_index().to_dict(),
        "candidates_per_row_mean": float(n_cand.mean()),
        "val_map_per_seed": maps,
        "val_map_seed_mean": float(np.mean(maps)),
        "val_map_seed_std": float(np.std(maps, ddof=1)),
        "val_map_rank_avg_ensemble": float(map_mean),
        "recall_all_positives": float(P["candidate"].mean()),
        "recall_per_row_mean (official-style)": float(P.groupby("seed")["candidate"].mean().mean()),
    }
    breakdowns = {}
    breakdowns.update(
        bucket(
            "dst_age",
            "dst_age_days",
            [0, 30, 91, 365, 3 * 365, 100 * 365],
            ["<30d", "30-91d", "91d-1y", "1-3y", ">3y"],
        )
    )
    breakdowns.update(
        bucket(
            "src_age",
            "src_age_days",
            [0, 7, 30, 91, 365, 100 * 365],
            ["<7d", "7-30d", "30-91d", "91d-1y", ">1y"],
        )
    )
    breakdowns.update(
        bucket(
            "dst_popularity",
            "dst_links_before",
            [0, 1, 2, 5, 20, 10**9],
            ["0", "1", "2-4", "5-19", "20+"],
        )
    )
    breakdowns.update(bucket("src_links_before", "src_links_before", [0, 1, 10**9], ["0", "1+"]))
    for c in ("tag_dump", "tag_asof", "word_dump", "word_asof"):
        breakdowns.update(bucket(c, c, [0, 1, 2, 3, 100], ["0", "1", "2", "3+"]))
    for c in ("link_type", "dst_type", "src_type", "same_owner"):
        P[c + "_s"] = P[c].astype(str)
        g = P.groupby(c + "_s").agg(
            n=("candidate", "size"),
            recall=("candidate", "mean"),
            top100=("rank_mean", lambda s: float((s <= 100).mean())),
        )
        breakdowns[c] = g.reset_index().to_dict(orient="records")
    # Tag edits after T: positives whose shared-tag status differs between dump and as-of-T.
    stats["tag_share_dump_vs_asof"] = (
        pd.crosstab(P["tag_dump"] > 0, P["tag_asof"] > 0).astype(int).to_dict()
    )
    stats["word_share_dump_vs_asof"] = (
        pd.crosstab(P["word_dump"] > 0, P["word_asof"] > 0).astype(int).to_dict()
    )
    # For contrast: the same on a random sample of non-positive candidates.
    neg = X[X["y"] == 0].sample(n=min(5000, int((X["y"] == 0).sum())), random_state=0)
    diffs = 0
    for s_, d_ in zip(neg[SEED], neg[DST]):
        qs, qd = question_of(int(val.at[int(s_), "PostId"])), question_of(int(d_))
        diffs += (len(tags_of(qs, False) & tags_of(qd, False)) > 0) != (
            len(tags_of(qs, True) & tags_of(qd, True)) > 0
        )
    stats["neg_candidates_tag_share_flip_rate"] = diffs / len(neg)
    stats["pos_tag_share_flip_rate"] = float(((P["tag_dump"] > 0) != (P["tag_asof"] > 0)).mean())

    # --- ranks and the MAP decomposition ---------------------------------------------------
    rk = P["rank_mean"]
    stats["positive_rank_buckets (rank-avg ensemble)"] = {
        "not_candidate": int(rk.isna().sum()),
        "1": int((rk == 1).sum()),
        "2-5": int(((rk >= 2) & (rk <= 5)).sum()),
        "6-20": int(((rk >= 6) & (rk <= 20)).sum()),
        "21-100": int(((rk >= 21) & (rk <= 100)).sum()),
        ">100 (candidate, cut)": int((rk > 100).sum()),
    }
    per_seed_buckets = []
    for j in range(3):
        r = P[f"rank{j}"]
        per_seed_buckets.append(
            {
                "1": int((r == 1).sum()),
                "2-5": int(((r >= 2) & (r <= 5)).sum()),
                "6-20": int(((r >= 6) & (r <= 20)).sum()),
                "21-100": int(((r >= 21) & (r <= 100)).sum()),
                ">100": int((r > 100).sum()),
            }
        )
    stats["positive_rank_buckets_per_seed"] = per_seed_buckets

    # Where each row's AP comes from: AP = sum over hits of P@r / n_true.
    ap_mean = ap_rows(pred_mean, true_lists)
    contrib = {"rank1": 0.0, "2-5": 0.0, "6-20": 0.0, "21-100": 0.0}
    for row, (pr, tr) in enumerate(zip(pred_mean, true_lists)):
        isin = np.array([d in tr for d in pr], dtype=float)
        prec = np.cumsum(isin) / (np.arange(k) + 1)
        for pos in np.flatnonzero(isin):
            r = pos + 1
            key = "rank1" if r == 1 else "2-5" if r <= 5 else "6-20" if r <= 20 else "21-100"
            contrib[key] += prec[pos] / min(len(tr), k) / len(val)
    stats["map_contribution_by_rank (rank-avg ensemble)"] = contrib
    stats["rows_with_positive_at_rank1"] = float(
        np.mean([pr[0] in tr for pr, tr in zip(pred_mean, true_lists)])
    )

    # Ceilings, with the official evaluate.
    def oracle(include_missing: bool):
        pr = np.full((len(val), k), -1, dtype=np.int64)
        for i, tr in enumerate(true_lists):
            have = X.loc[(X[SEED] == i) & (X["y"] == 1), DST].to_numpy()
            put = sorted(tr) if include_missing else list(have)
            rest = X.loc[(X[SEED] == i) & (X["y"] == 0)].sort_values("rank_mean")[DST]
            row = (list(put) + list(rest.to_numpy()))[:k]
            pr[i, : len(row)] = row
        (v,) = task.evaluate(pr, val_table).values()
        return float(v)

    stats["ceiling_perfect_ranking_of_candidates"] = oracle(False)
    stats["ceiling_perfect_recall_and_ranking"] = oracle(True)

    # "Perfect recall, today's ranking quality": put each missed positive at a rank drawn
    # from the found positives' rank distribution (rank-avg ensemble, ranks <= 624 kept).
    rng = np.random.default_rng(0)
    found_ranks = P.loc[P["candidate"], "rank_mean"].to_numpy()
    sims = []
    for _ in range(200):
        aps = []
        for i, tr in enumerate(true_lists):
            sub = P[P["seed"] == i]
            ranks = sub["rank_mean"].to_numpy().copy()
            miss = np.isnan(ranks)
            ranks[miss] = rng.choice(found_ranks, size=miss.sum())
            ranks = np.sort(ranks)
            ap, last = 0.0, 0.0
            for j, r in enumerate(ranks):
                pos = max(r, last + 1)  # two hits can't share a rank
                last = pos
                if pos <= k:
                    ap += (j + 1) / pos
            aps.append(ap / min(len(tr), k))
        sims.append(np.mean(aps))
    stats["map_if_misses_found_at_todays_rank_distribution"] = float(np.mean(sims))

    # Recall-limited vs rank-limited rows.
    per_row = pd.DataFrame({"ap": ap_mean, "n_true": [len(t) for t in true_lists]})
    per_row["n_found"] = P.groupby("seed")["candidate"].sum().reindex(range(len(val))).values
    per_row["all_missed"] = per_row["n_found"] == 0
    stats["rows_all_positives_missed"] = float(per_row["all_missed"].mean())
    stats["map_on_rows_with_a_candidate_positive"] = float(
        per_row.loc[~per_row.all_missed, "ap"].mean()
    )

    # --- noise: ranker seed vs 226 rows (bootstrap) ---------------------------------------
    per_seed_ap = np.stack([ap_rows(pr, true_lists) for pr in preds])  # (3, rows)
    n = per_seed_ap.shape[1]
    idx = rng.integers(0, n, size=(a.boot, n))
    boot_seedmean = per_seed_ap.mean(axis=0)[idx].mean(axis=1)
    boot_single = per_seed_ap[0][idx].mean(axis=1)
    diff_01 = (per_seed_ap[0] - per_seed_ap[1])[idx].mean(axis=1)
    stats["noise"] = {
        "seed_std_of_map (3 seeds)": float(np.std(maps, ddof=1)),
        "bootstrap_std_rows_seed0": float(boot_single.std()),
        "bootstrap_std_rows_seed_mean_ap": float(boot_seedmean.std()),
        "bootstrap_ci95_seed_mean_ap": [
            float(np.quantile(boot_seedmean, 0.025)),
            float(np.quantile(boot_seedmean, 0.975)),
        ],
        "bootstrap_std_paired_diff_seed0_minus_seed1": float(diff_01.std()),
        "per_row_ap_std_across_seeds_mean": float(per_seed_ap.std(axis=0, ddof=1).mean()),
        "rows_whose_ap_changes_across_seeds": float(
            (per_seed_ap.max(axis=0) - per_seed_ap.min(axis=0) > 1e-9).mean()
        ),
        "share_of_map_from_top10_rows": float(
            np.sort(per_seed_ap.mean(axis=0))[::-1][:10].sum() / per_seed_ap.mean(axis=0).sum()
        ),
        "rows_with_ap_zero_all_seeds": float((per_seed_ap.max(axis=0) == 0).mean()),
    }

    # --- lead indicator for text retrieval: TF-IDF kNN over title+body ---------------------
    from sklearn.feature_extraction.text import TfidfVectorizer

    pp = db.table_dict["posts"].df
    pp = pp[pp["CreationDate"] < T][["Id", "PostTypeId", "ParentId", "Title", "Body"]]
    titles = pp.set_index("Id")["Title"]
    parent_title = pp["ParentId"].map(titles)
    text = (pp["Title"].fillna(parent_title).fillna("") + " " + pp["Body"].fillna("")).str.replace(
        r"<[^>]+>", " ", regex=True
    )
    vec = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True, max_features=200_000)
    M = vec.fit_transform(text.to_numpy())
    id_pos = pd.Series(np.arange(len(pp)), index=pp["Id"].to_numpy())
    srcs = val["PostId"].to_numpy()
    S = (M[id_pos.reindex(srcs).to_numpy()] @ M.T).toarray()
    S[np.arange(len(srcs)), id_pos.reindex(srcs).to_numpy()] = -1  # not itself
    order = np.argsort(-S, axis=1)
    rank_of = {}
    for i in range(len(srcs)):
        inv = np.empty(len(pp), dtype=np.int64)
        inv[order[i]] = np.arange(1, len(pp) + 1)
        for d in true_lists[i]:
            if d in id_pos.index:
                rank_of[(i, d)] = int(inv[id_pos[d]])
    P["tfidf_rank"] = [rank_of.get((s, d), np.nan) for s, d in zip(P["seed"], P["dst"])]
    tf = {}
    for kk in (10, 50, 100, 300, 1000):
        hit = P["tfidf_rank"] <= kk
        tf[f"recall@{kk}"] = float(hit.mean())
        tf[f"recall@{kk}_of_missed"] = float(hit[~P["candidate"]].mean())
        tf[f"union_recall_with_candidates@{kk}"] = float((hit | P["candidate"]).mean())
    stats["tfidf_knn_title_body (dump text, diagnostic)"] = tf
    P.to_parquet(out / "positives.parquet")

    stats["breakdowns"] = breakdowns
    stats["runtime_minutes"] = (time.time() - t0) / 60
    (out / "phase1.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
