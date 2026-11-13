"""Tests for relconduit.link: edge sets, candidates, pair features, and the pair leakage test."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

from relconduit.link import (
    DST,
    NEVER,
    SEED,
    LinkConfig,
    Path,
    build_pairs,
    candidate_recall,
    join_edges,
    label_pairs,
    pair_leakage_mismatches,
    reverse_edges,
    table_edges,
    token_edges,
)

D = pd.Timestamp


def tiny_db() -> Database:
    """Three users, four items with tags, visits. Some visits sit exactly on a seed time."""
    users = pd.DataFrame({"user_id": [0, 1, 2]})
    items = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "tags": ["<a><b>", "<b>", "<c>", "<a>"],
            "created": [D("2019-01-01"), D("2019-01-01"), D("2019-01-01"), D("2020-06-01")],
        }
    )
    visits = pd.DataFrame(
        {
            "visit_id": range(8),
            "user_id": [0, 0, 1, 1, 2, 0, 1, 2],
            "item_id": [0, 1, 1, 2, 0, 0, 3, 2],
            "time": [
                D("2020-01-01"),
                D("2020-02-01"),
                D("2020-02-15"),
                D("2020-03-01"),
                D("2020-04-01"),
                D("2020-05-01"),
                D("2020-06-01"),  # exactly on a seed time below
                D("2020-07-01"),  # after every seed
            ],
        }
    )
    return Database(
        {
            "users": Table(users, {}, pkey_col="user_id"),
            "items": Table(items, {}, pkey_col="item_id", time_col="created"),
            "visits": Table(
                visits,
                {"user_id": "users", "item_id": "items"},
                pkey_col="visit_id",
                time_col="time",
            ),
        }
    )


def split_tags(s: str) -> list[str]:
    return s.strip("<>").split("><")


def config() -> LinkConfig:
    def visits(db):
        return table_edges(db, "visits", "user_id", "item_id")

    def item_tags(db):
        return token_edges(db, "items", "item_id", "tags", split_tags)

    def user_tags(db):
        return join_edges(visits(db), item_tags(db))

    return LinkConfig(
        edges={
            "visits": visits,
            "user_tags": user_tags,
            "tag_targets": _tag_to_visited,
        },
        direct=["visits"],
        paths={
            "covisit": Path(("visits", "~visits", "visits"), n=10),
            "tags": Path(("user_tags", "tag_targets"), n=10),
        },
        popularity=["visits"],
        pop_windows=(30, 365),
        pop_candidates={"visits": (365, 2)},
    )


def _tag_to_visited(db):
    """tag -> item for every visit of an item carrying the tag (at the later time)."""
    tags = token_edges(db, "items", "item_id", "tags", split_tags)  # item -> tag
    visits = table_edges(db, "visits", "user_id", "item_id")  # user -> item
    item_visit = reverse_edges(visits)  # item -> user
    # tag -> item via item -> user: keep the item end by joining back on the item.
    m = reverse_edges(tags).merge(item_visit, left_on=DST, right_on="src", suffixes=("", "_v"))
    t = np.maximum(m["t"].to_numpy(), m["t_v"].to_numpy())
    return pd.DataFrame({"src": m["src"], "dst": m["dst"], "t": t})


SEEDS = pd.DataFrame(
    {
        "user_id": [0, 1, 2, 0],
        "time": [D("2020-06-01"), D("2020-06-01"), D("2020-06-01"), D("2020-03-01")],
        "item_id": [[0, 3], [2], [1], [0]],
    }
)


def test_table_edges_dates_static_rows_never_and_joins_take_the_later_time():
    db = tiny_db()
    static = table_edges(
        Database({"users": Table(pd.DataFrame({"a": [1], "b": [2]}), {}, None)}), "users", "a", "b"
    )
    assert static["t"].iloc[0] == NEVER
    tags = token_edges(db, "items", "item_id", "tags", split_tags)
    assert sorted(tags[tags["src"] == 0]["dst"]) == ["a", "b"]
    j = join_edges(table_edges(db, "visits", "user_id", "item_id"), tags)
    # user 1 visited item 3 (tag a) on 2020-06-01; item 3 was created 2020-06-01.
    row = j[(j["src"] == 1) & (j["dst"] == "a")]
    assert row["t"].max() == D("2020-06-01")


def test_direct_features_and_candidates():
    pairs = build_pairs(tiny_db(), SEEDS, "user_id", "time", config())
    p = pairs.set_index([SEED, DST])
    # Seed 0: user 0 at 2020-06-01 visited item 0 twice (Jan 1, May 1) and item 1 once.
    assert p.loc[(0, 0), "visits__count"] == 2
    assert p.loc[(0, 0), "visits__days_since_last"] == 31
    assert p.loc[(0, 0), "visits__days_since_first"] == (D("2020-06-01") - D("2020-01-01")).days
    assert p.loc[(0, 1), "visits__count"] == 1
    # Seed 3: user 0 at 2020-03-01 has only seen items 0 and 1, once each.
    assert p.loc[(3, 0), "visits__count"] == 1
    # Item 3's only visit is at 2020-06-01, not before any seed time, so no source proposes it.
    assert 3 not in pairs[DST].to_numpy()


def test_popularity_and_global_candidates():
    pairs = build_pairs(tiny_db(), SEEDS, "user_id", "time", config())
    p = pairs.set_index([SEED, DST])
    # Before 2020-06-01: item 0 visited 3 times, items 1 and 2 twice each.
    assert p.loc[(1, 0), "visits__pop_all"] == 3
    assert p.loc[(1, 0), "visits__pop_365d"] == 3
    assert p.loc[(1, 0), "visits__pop_30d"] == 0  # May 1 is 31 days back
    # Global top 2 over 365 days (count, then lower id): items 0 and 1, proposed to every seed.
    for s in range(3):
        assert {(s, 0), (s, 1)} <= set(p.index)


def test_covisit_path_scores_are_walk_probabilities():
    pairs = build_pairs(tiny_db(), SEEDS, "user_id", "time", config())
    p = pairs.set_index([SEED, DST])
    # User 2 at 2020-06-01 visited only item 0. Item 0's visitors: user 0 (twice), user 2 (once).
    # User 0 -> {0: 2/3, 1: 1/3}; user 2 -> {0: 1}. Walk: 2/3 * (2/3, 1/3) + 1/3 * (1, 0).
    assert p.loc[(2, 0), "covisit__score"] == pytest.approx(2 / 3 * 2 / 3 + 1 / 3)
    assert p.loc[(2, 1), "covisit__score"] == pytest.approx(2 / 3 * 1 / 3)
    assert p.loc[(2, 0), "covisit__rank"] == 1
    assert p.loc[(2, 1), "covisit__rank"] == 2


def test_label_pairs_and_recall():
    pairs = build_pairs(tiny_db(), SEEDS, "user_id", "time", config())
    y = label_pairs(pairs, SEEDS, "item_id")
    hit = set(map(tuple, pairs.loc[y == 1, [SEED, DST]].to_numpy()))
    assert (0, 0) in hit and (2, 1) in hit
    assert all(d in SEEDS["item_id"].iloc[s] for s, d in hit)
    r = candidate_recall(pairs, SEEDS, "item_id")
    assert 0 < r <= 1


def test_no_pair_leakage():
    assert pair_leakage_mismatches(tiny_db(), SEEDS, "user_id", "time", config()) == []


def test_pair_leakage_test_catches_a_planted_leak():
    """With ``<=`` the visit at exactly 2020-06-01 leaks into seeds at that time."""
    leaky = replace(config(), _inclusive=True)
    bad = pair_leakage_mismatches(tiny_db(), SEEDS, "user_id", "time", leaky)
    assert bad
    # Item 3 (created and first visited at 2020-06-01) is among the leaked values.
    assert any(dst == 3 for _, dst, *_ in bad)


def test_source_columns_mark_who_proposed_each_pair():
    pairs = build_pairs(tiny_db(), SEEDS, "user_id", "time", config())
    cols = [c for c in pairs.columns if c.startswith("cand__")]
    assert cols == ["cand__visits", "cand__covisit", "cand__tags", "cand__visits__top365d"]
    assert pairs[cols].sum(axis=1).ge(1).all()
    p = pairs.set_index([SEED, DST])
    assert p.loc[(0, 1), "cand__visits"] == 1  # user 0 visited item 1
    assert p.loc[(1, 0), "cand__visits"] == 0  # user 1 never visited item 0 ...
    assert p.loc[(1, 0), "cand__visits__top365d"] == 1  # ... but it is the most popular


def test_windows_and_half_lives_keep_static_edges():
    """A static edge (dated NEVER) is in every window and never decays."""
    db = tiny_db()
    users = pd.DataFrame({"user_id": [0, 1, 2], "city": ["x", "x", "y"]})
    db.table_dict["users"] = Table(users, {}, pkey_col="user_id")
    cfg = LinkConfig(
        edges={
            "city": lambda db: table_edges(db, "users", "user_id", "city"),
            "visits": lambda db: table_edges(db, "visits", "user_id", "item_id"),
        },
        paths={
            "windowed": Path(("city", "~city", "visits"), n=10, window_days=365),
            "decayed": Path(("city", "~city", "visits"), n=10, half_life_days=30),
        },
    )
    seeds = pd.DataFrame({"user_id": [1], "time": [D("2020-06-01")]})
    p = build_pairs(db, seeds, "user_id", "time", cfg).set_index(DST)
    # User 1 shares city x with user 0, who visited items 0 and 1 before 2020-06-01.
    assert p.loc[0, "windowed__score"] > 0 and p.loc[1, "windowed__score"] > 0
    assert p.loc[0, "decayed__score"] > 0
