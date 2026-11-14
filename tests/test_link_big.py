"""Tests for relconduit.link_big's helpers and relconduit.link_amazon's edges."""

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

from relconduit.link import DST, NEVER, SEED, pair_leakage_mismatches
from relconduit.link_amazon import five, link_config, long, product_codes
from relconduit.link_big import fill_top_k, recall_from_hits, sample_train, seed_chunks, to_codes

D = pd.Timestamp


def test_seed_chunks_split_big_times_and_never_mix_times():
    times = pd.Series([D("2020-02-01")] * 5 + [D("2020-01-01")] * 2 + [D("2020-02-01")])
    chunks = seed_chunks(times, 3)
    assert [list(c) for c in chunks] == [[5, 6], [0, 1, 2], [3, 4, 7]]
    for c in chunks:
        assert times.iloc[c].nunique() == 1
    assert sorted(np.concatenate(chunks)) == list(range(len(times)))


def test_sample_train_takes_latest_times():
    times = pd.Series([D("2020-01-01")] * 4 + [D("2020-02-01")] * 4 + [D("2020-03-01")] * 4)
    pos = sample_train(times, 5, last_times=2)
    assert len(pos) == 5 and (pos >= 4).all() and (np.diff(pos) > 0).all()
    np.testing.assert_array_equal(pos, sample_train(times, 5, last_times=2))  # fixed rng
    assert list(sample_train(times, 100, last_times=1)) == [8, 9, 10, 11]
    assert len(sample_train(times, 100, last_times=None)) == 12


def test_fill_top_k_writes_only_seeds_in_the_chunk():
    out = np.full((4, 2), -1, dtype=np.int64)
    X = pd.DataFrame({SEED: [1, 1, 1, 3], DST: [10, 11, 12, 20]})
    fill_top_k(out, X, np.array([0.1, 0.9, 0.9, 0.5]))
    assert out.tolist() == [[-1, -1], [11, 12], [-1, -1], [20, -1]]  # ties: lower dst first


def test_recall_from_hits():
    assert recall_from_hits(np.array([1, 0, 2]), np.array([2, 1, 2])) == pytest.approx(0.5)


def test_to_codes_is_stable_across_chunks():
    cat = pd.CategoricalDtype(["a", "b", "c"])
    a = to_codes(pd.DataFrame({"x": pd.Series(["c", None], dtype=cat), "y": [1.0, 2.0]}))
    b = to_codes(pd.DataFrame({"x": pd.Series(["a"], dtype=cat), "y": [3.0]}))
    assert a["x"].tolist()[0] == 2 and np.isnan(a["x"].tolist()[1]) and b["x"].tolist() == [0]
    assert a["y"].dtype == np.float32
    with pytest.raises(TypeError):
        to_codes(pd.DataFrame({"z": ["free text"]}))


def amazon_db() -> Database:
    meta = pd.DataFrame(
        {
            "product_id": [0, 1, 2, 3],
            "category": [
                ["Books", "Mystery", "Noir"],
                ["Books", "Mystery"],
                None,
                ["Books", "Mystery", "Noir"],
            ],
            "brand": ["A", "B", "A", None],
        }
    )
    prod = product_codes(meta)
    reviews = pd.DataFrame(
        {
            "review_time": [
                D("2015-01-01"),
                D("2015-02-01"),
                D("2015-03-01"),
                D("2015-04-01"),
                D("2015-05-01"),
                D("2015-06-01"),
            ],
            "customer_id": [0, 0, 1, 1, 2, 2],
            "product_id": [0, 1, 0, 2, 3, 1],
            "rating": [5.0, 3.0, 5.0, 1.0, 5.0, 4.0],
            "verified": [True] * 6,
            "review_text_len": [400.0, 10.0, np.nan, 301.0, 300.0, 1000.0],
        }
    )
    return Database(
        {
            "customer": Table(pd.DataFrame({"customer_id": [0, 1, 2]}), {}, pkey_col="customer_id"),
            "product": Table(prod, {}, pkey_col="product_id"),
            "review": Table(
                reviews,
                {"customer_id": "customer", "product_id": "product"},
                time_col="review_time",
            ),
        }
    )


def test_product_codes():
    db = amazon_db()
    p = db.table_dict["product"].df.set_index("product_id")
    assert p.loc[0, "brand_id"] == p.loc[2, "brand_id"] != p.loc[1, "brand_id"]
    assert p.loc[3, "brand_id"] == -1 and p.loc[2, "leaf_id"] == -1
    assert p.loc[0, "leaf_id"] == p.loc[3, "leaf_id"] != p.loc[1, "leaf_id"]
    assert p.loc[0, "cat2_id"] == p.loc[1, "cat2_id"]


def test_target_edges_are_the_label_rows():
    db = amazon_db()
    assert sorted(zip(*five(db)[["src", "dst"]].to_numpy().T)) == [(0, 0), (1, 0), (2, 3)]
    assert sorted(zip(*long(db)[["src", "dst"]].to_numpy().T)) == [(0, 0), (1, 2), (2, 1)]


@pytest.mark.parametrize("task", ["user-item-purchase", "user-item-rate", "user-item-review"])
def test_amazon_pairs_do_not_leak(task):
    from dataclasses import replace

    db = amazon_db()
    seeds = pd.DataFrame(
        {
            "customer_id": [0, 1, 2, 2],
            "timestamp": [D("2015-03-01"), D("2015-05-01"), D("2015-06-01"), D("2015-05-01")],
        }
    )
    cfg = link_config(task)
    assert pair_leakage_mismatches(db, seeds, "customer_id", "timestamp", cfg) == []
    # A planted `<=` is caught: reviews sit exactly on the seed times.
    leaky = replace(cfg, _inclusive=True)
    assert pair_leakage_mismatches(db, seeds, "customer_id", "timestamp", leaky)
    # Static product codes are known from the start.
    e = cfg.edges["prod_brand"](db)
    assert (e["t"] == NEVER).all() and DST in e
