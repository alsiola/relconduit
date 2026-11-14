"""Tests for relconduit.post_embed: initial texts, the time-safe index and its leak test."""

from dataclasses import replace

import numpy as np
import pandas as pd
from relbench.base import Database, Table

from relconduit.link import DST, SEED, LinkConfig, build_pairs, pair_leakage_mismatches
from relconduit.post_embed import DenseIndex, DenseSource, initial_post_texts, text_times

D = pd.Timestamp
SEED_T = D("2020-06-01")


def tiny_db() -> Database:
    """Five posts; post 4 is created exactly at a seed time, post 5 after it."""
    posts = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5],
            "PostTypeId": [1, 2, 1, 1, 1],
            "ParentId": [np.nan, 1, np.nan, np.nan, np.nan],
            "CreationDate": [
                D("2020-01-01"),
                D("2020-02-01"),
                D("2020-03-01"),
                SEED_T,
                D("2020-07-01"),
            ],
        }
    )
    ph = pd.DataFrame(
        {
            "Id": range(9),
            "PostId": [1, 1, 2, 3, 3, 4, 4, 5, 1],
            "PostHistoryTypeId": [1, 2, 2, 1, 2, 1, 2, 2, 4],
            "Text": ["t1", "<p>b1</p>", "a2", "t3", "b3", "t4", "b4", "b5", "edited"],
            "CreationDate": [
                D("2020-01-01"),
                D("2020-01-01"),
                D("2020-02-01"),
                D("2020-03-01"),
                D("2020-03-02"),
                SEED_T,
                SEED_T,
                D("2020-07-01"),
                D("2020-08-01"),
            ],
        }
    )
    links = pd.DataFrame(
        {
            "Id": [0],
            "PostId": [3],
            "RelatedPostId": [1],
            "CreationDate": [D("2020-04-01")],
        }
    )
    return Database(
        {
            "posts": Table(posts, {}, pkey_col="Id", time_col="CreationDate"),
            "postHistory": Table(ph, {"PostId": "posts"}, pkey_col="Id", time_col="CreationDate"),
            "postLinks": Table(
                links,
                {"PostId": "posts", "RelatedPostId": "posts"},
                pkey_col="Id",
                time_col="CreationDate",
            ),
        }
    )


def index(db) -> DenseIndex:
    tx = initial_post_texts(db.table_dict["postHistory"].df, db.table_dict["posts"].df)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(tx), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return DenseIndex(tx["Id"].to_numpy(), tx["t"].to_numpy(), emb)


def config(db) -> LinkConfig:
    return LinkConfig(edges={}, dense={"dense": DenseSource(index(db), n=10)})


def seeds() -> pd.DataFrame:
    return pd.DataFrame(
        {"PostId": [1, 3, 2, 4], "timestamp": [SEED_T, SEED_T, D("2020-02-15"), SEED_T]}
    )


def test_initial_texts_use_first_revisions_and_the_questions_title():
    db = tiny_db()
    tx = initial_post_texts(db.table_dict["postHistory"].df, db.table_dict["posts"].df)
    tx = tx.set_index("Id")
    assert tx.at[1, "text"] == "t1\nb1"  # the later title edit is ignored, HTML stripped
    assert tx.at[1, "t"] == D("2020-01-01")
    assert tx.at[2, "text"] == "t1\na2"  # an answer gets its question's initial title
    assert tx.at[2, "t"] == D("2020-02-01")
    assert tx.at[3, "t"] == D("2020-03-02")  # the latest of the rows used
    assert text_times(db).set_index("Id")["t"].equals(tx["t"])


def test_search_only_sees_posts_known_strictly_before_the_seed_time():
    idx = index(tiny_db())
    r, d, s = idx.search(np.array([1, 4]), SEED_T, n=10)
    assert set(d[r == 0]) == {2, 3}  # not itself, not 4 (at T) or 5 (after)
    assert (r == 1).sum() == 0  # post 4's own text is dated at T: no query
    assert np.all(np.diff(s[r == 0]) <= 0)
    r, d, _ = idx.search(np.array([1]), SEED_T, n=10, inclusive=True)
    assert 4 in set(d)


def test_cosine_is_nan_unless_both_posts_known_before():
    idx = index(tiny_db())
    t = np.full(3, np.datetime64(SEED_T, "ns"))
    c = idx.cosine(np.array([1, 1, 4]), np.array([3, 4, 1]), t)
    assert np.isfinite(c[0]) and np.isnan(c[1]) and np.isnan(c[2])


def test_dense_candidates_and_features_in_pairs():
    db = tiny_db()
    pairs = build_pairs(db, seeds(), "PostId", "timestamp", config(db))
    one = pairs[pairs[SEED] == 0].set_index(DST)
    assert set(one.index) == {2, 3}
    assert one["cand__dense"].eq(1).all()
    assert one["dense__rank"].min() == 1
    early = pairs[pairs[SEED] == 2]
    assert set(early[DST]) == {1}  # at 2020-02-15 only post 1 is known


def test_no_dense_leakage():
    db = tiny_db()
    assert pair_leakage_mismatches(db, seeds(), "PostId", "timestamp", config(db)) == []


def test_dense_leakage_test_catches_a_planted_leak():
    db = tiny_db()
    leaky = replace(config(db), _inclusive=True)
    bad = pair_leakage_mismatches(db, seeds(), "PostId", "timestamp", leaky)
    assert bad
    assert any(b[1] == 4 for b in bad)  # post 4, dated exactly at T, leaked in


def tag_db() -> Database:
    """``tiny_db`` plus a tags table: post 3 gains tag ``b`` exactly at the seed time."""
    db = tiny_db()
    tags = pd.DataFrame(
        {
            "Id": range(5),
            "PostId": [1, 1, 3, 3, 2],
            "tag": ["a", "b", "a", "b", "c"],
            "CreationDate": [
                D("2020-01-01"),
                D("2020-01-01"),
                D("2020-03-01"),
                SEED_T,
                D("2020-02-01"),
            ],
        }
    )
    db.table_dict["tags"] = Table(tags, {"PostId": "posts"}, pkey_col="Id", time_col="CreationDate")
    return db


def overlap_config(db) -> LinkConfig:
    from relconduit.link import table_edges

    def tag(d):
        return table_edges(d, "tags", "PostId", "tag")

    cfg = config(db)
    return replace(cfg, edges={"tag": tag}, overlaps={"tag_overlap": "tag"})


def test_overlap_counts_shared_tokens_known_before_the_seed_time():
    db = tag_db()
    pairs = build_pairs(db, seeds(), "PostId", "timestamp", overlap_config(db))
    one = pairs[pairs[SEED] == 0].set_index(DST)
    assert one.at[3, "tag_overlap__shared"] == 1  # "b" on post 3 is dated at T: not yet
    assert one.at[3, "tag_overlap__jaccard"] == 0.5
    assert one.at[2, "tag_overlap__shared"] == 0
    assert pair_leakage_mismatches(db, seeds(), "PostId", "timestamp", overlap_config(db)) == []


def test_overlap_leakage_test_catches_a_planted_leak():
    db = tag_db()
    leaky = replace(overlap_config(db), _inclusive=True)
    bad = pair_leakage_mismatches(db, seeds(), "PostId", "timestamp", leaky)
    assert any(b[2] == "tag_overlap__shared" for b in bad)
