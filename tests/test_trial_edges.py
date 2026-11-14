"""Tests for relconduit.trial_edges: rel-trial's edge sets and issue 28's new candidate sources.

Each new source gets the pair leakage check on a tiny rel-trial-shaped database with a study
dated exactly at the seed time: no mismatch with the real rule (``<``), and at least one with a
planted ``<=`` (``LinkConfig._inclusive``).
"""

from dataclasses import replace

import pandas as pd
import pytest
from relbench.base import Database, Table

from relconduit.link import NEVER, LinkConfig, build_pairs, pair_leakage_mismatches
from relconduit.trial_edges import EDGES, NEW_PATHS, cosponsor, fac_condition, ran, spon_condition

D = pd.Timestamp
T = D("2020-01-01")


def trial_db() -> Database:
    """Five sites, four sponsors, four studies; study 2 starts exactly at the seed time ``T``
    and study 3 after it."""
    facilities = pd.DataFrame(
        {
            "facility_id": [0, 1, 2, 3, 4],
            "name": ["a", "b", "c", "d", "e"],
            "city": ["A", "A", "B", "B", "A"],
            "zip": ["Z1", "Z1", "Z2", "Z2", "Z1"],
            "state": ["S1", "S1", "S1", "S2", "S1"],
            "country": ["C"] * 5,
        }
    )
    sponsors = pd.DataFrame({"sponsor_id": [0, 1, 2, 3], "name": ["p", "q", "r", "s"]})
    conditions = pd.DataFrame({"condition_id": [0, 1], "mesh_term": ["x", "y"]})
    dates = {0: D("2019-01-01"), 1: D("2019-06-01"), 2: T, 3: D("2020-03-01")}
    studies = pd.DataFrame({"nct_id": list(dates), "start_date": list(dates.values())})
    fs = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 3), (3, 4)]  # (study, facility)
    ss = [(0, 0), (0, 1), (1, 2), (2, 1), (2, 3), (3, 3)]  # (study, sponsor)
    cs = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)]  # (study, condition)

    def link(rows, col):
        return pd.DataFrame(
            {
                "id": range(len(rows)),
                "nct_id": [s for s, _ in rows],
                col: [x for _, x in rows],
                "date": [dates[s] for s, _ in rows],
            }
        )

    def timed(df, fks):
        return Table(df, fks, pkey_col="id", time_col="date")

    return Database(
        {
            "facilities": Table(facilities, {}, pkey_col="facility_id"),
            "sponsors": Table(sponsors, {}, pkey_col="sponsor_id"),
            "conditions": Table(conditions, {}, pkey_col="condition_id"),
            "studies": Table(studies, {}, pkey_col="nct_id", time_col="start_date"),
            "facilities_studies": timed(
                link(fs, "facility_id"), {"nct_id": "studies", "facility_id": "facilities"}
            ),
            "sponsors_studies": timed(
                link(ss, "sponsor_id"), {"nct_id": "studies", "sponsor_id": "sponsors"}
            ),
            "conditions_studies": timed(
                link(cs, "condition_id"), {"nct_id": "studies", "condition_id": "conditions"}
            ),
        }
    )


SEEDS = pd.DataFrame(
    {
        "facility_id": [0, 1, 2, 3, 4, 0],
        "timestamp": [T, T, T, T, T, D("2019-07-01")],
    }
)


def one_source(name: str) -> LinkConfig:
    return LinkConfig(edges=EDGES, paths={name: NEW_PATHS[name]})


def test_edges_are_dated_at_their_latest_row_and_cosponsors_exclude_self():
    db = trial_db()
    r = ran(db)
    assert set(zip(r["src"], r["dst"])) == {(0, 0), (0, 1), (2, 0), (2, 1), (1, 2), (0, 3)} | {
        (3, 1),
        (3, 3),
        (4, 3),
    }
    co = cosponsor(db)
    assert set(zip(co["src"], co["dst"])) == {(0, 1), (1, 0), (1, 3), (3, 1)}
    assert (co.loc[(co["src"] == 1) & (co["dst"] == 3), "t"] == T).all()
    fc = fac_condition(db)
    assert set(fc.loc[fc["src"] == 3, "dst"]) == {0, 1}
    sc = spon_condition(db)
    assert set(sc.loc[sc["src"] == 3, "dst"]) == {0, 1}
    assert EDGES["city"](db)["t"].eq(NEVER).all()  # static column, known from the start


@pytest.mark.parametrize("name", sorted(NEW_PATHS))
def test_new_source_has_no_pair_leakage(name):
    assert (
        pair_leakage_mismatches(trial_db(), SEEDS, "facility_id", "timestamp", one_source(name))
        == []
    )


@pytest.mark.parametrize("name", sorted(NEW_PATHS))
def test_leakage_check_catches_a_planted_leak_in_each_new_source(name):
    leaky = replace(one_source(name), _inclusive=True)
    bad = pair_leakage_mismatches(trial_db(), SEEDS, "facility_id", "timestamp", leaky)
    assert bad, f"a planted <= in {name} went unnoticed"


def test_new_sources_reach_what_they_should():
    db = trial_db()
    seeds = SEEDS.iloc[:5]
    cfg = LinkConfig(edges=EDGES, paths=NEW_PATHS)
    p = build_pairs(db, seeds, "facility_id", "timestamp", cfg)
    got = lambda src, seed: set(p.loc[(p["seed"] == seed) & (p[f"cand__{src}"] > 0), "dst"])
    # Site 0 ran studies with sponsors 0 and 1 (before T): their co-sponsors are each other.
    assert got("cosponsor", 0) == {0, 1}
    # Site 1 ran a study on condition 0, which sponsors 0, 1, 2 have studies on before T.
    assert got("condition", 1) == {0, 1, 2}
    # Site 4 (cold) shares zip Z1 with sites 0 and 1.
    assert got("zip", 4) == {0, 1, 2}
    # Site 3 is alone in state S2 and has no study before T: nothing.
    assert got("state", 3) == set()
    # City A, last 365 days before T: studies 0 (2019-01-01) and 1 (2019-06-01).
    assert got("city_recent", 4) == {0, 1, 2}
