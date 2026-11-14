"""Edge sets of the rel-trial database for its facility -> sponsor task (issue 28).

Each function maps the database to time-stamped ``(src, dst, t)`` edges for
:class:`relconduit.link.LinkConfig` (see :mod:`relconduit.link`): an edge made by joining rows
is dated at the latest of them, so it exists before a seed time exactly when every row it came
from does. Static tables (``facilities``, ``sponsors``) are dated :data:`relconduit.link.NEVER`.

* :func:`ran`: facility -> sponsor of a study at the facility (the task's own relation).
* :func:`sponsor_trials`: study -> sponsor.
* :func:`place`: facility -> one of its static columns (city, zip, state, country).
* :func:`place_key`: facility -> a key made from its static columns (city within country, a
  zip prefix within country).
* :func:`cosponsor`: sponsor -> another sponsor of one of its studies.
* :func:`fac_condition`, :func:`spon_condition`: facility / sponsor -> a condition one of its
  studies is on.
"""

from __future__ import annotations

import pandas as pd

from relconduit.link import NEVER, Path, _frame, join_edges, reverse_edges, table_edges


def ran(db):
    """facility -> sponsor, through a study at the facility run by the sponsor."""
    fac = table_edges(db, "facilities_studies", "facility_id", "nct_id")
    spon = table_edges(db, "sponsors_studies", "nct_id", "sponsor_id")
    return join_edges(fac, spon)


def sponsor_trials(db):
    """study -> its sponsor."""
    return table_edges(db, "sponsors_studies", "nct_id", "sponsor_id")


def place(col: str):
    """facility -> its ``col`` (a static column of the facilities table)."""

    def edges(db):
        return table_edges(db, "facilities", "facility_id", col)

    edges.__name__ = f"place_{col}"
    return edges


def place_key(name: str, key):
    """facility -> ``key(facilities)``, a string per facility (``None``/NaN: no edge)."""

    def edges(db):
        f = db.table_dict["facilities"].df
        k = pd.Series(key(f), index=f.index)
        ok = k.notna()
        return _frame(f.loc[ok, "facility_id"].to_numpy(), k[ok].to_numpy(), NEVER)

    edges.__name__ = f"place_{name}"
    return edges


def city_country(f: pd.DataFrame) -> pd.Series:
    """The city within its country, so Paris, France and Paris, Texas differ."""
    return (f["city"].astype(str) + "|" + f["country"].astype(str)).where(f["city"].notna())


def zip3(f: pd.DataFrame) -> pd.Series:
    """The first three characters of the zip within the country (a US zip3 is a region
    between city and state)."""
    z = f["zip"].astype(str).str.strip().str[:3]
    return (z + "|" + f["country"].astype(str)).where(f["zip"].notna() & (z.str.len() == 3))


def cosponsor(db):
    """sponsor -> each other sponsor of one of its studies, at the study's date."""
    ss = sponsor_trials(db)
    e = join_edges(reverse_edges(ss), ss)
    return e[e["src"] != e["dst"]].reset_index(drop=True)


def _study_condition(db):
    return table_edges(db, "conditions_studies", "nct_id", "condition_id")


def fac_condition(db):
    """facility -> a condition of one of its studies."""
    fs = table_edges(db, "facilities_studies", "facility_id", "nct_id")
    return join_edges(fs, _study_condition(db))


def spon_condition(db):
    """sponsor -> a condition of one of its studies."""
    return join_edges(reverse_edges(sponsor_trials(db)), _study_condition(db))


# The edge sets every config below may walk, by name.
EDGES = {
    "ran": ran,
    "sponsor_trials": sponsor_trials,
    "city": place("city"),
    "zip": place("zip"),
    "state": place("state"),
    "country": place("country"),
    "name": place("name"),
    "city_country": place_key("city_country", city_country),
    "zip3": place_key("zip3", zip3),
    "cosponsor": cosponsor,
    "fac_condition": fac_condition,
    "spon_condition": spon_condition,
}

# Issue 28's new candidate sources (each one is a path over EDGES; see relconduit.link.Path).
NEW_PATHS = {
    # Warm sites: co-sponsors of the sponsors that ran trials here.
    "cosponsor": Path(("ran", "cosponsor"), n=50, half_life_days=730),
    # Warm sites: sponsors with studies on the conditions this site has run trials on.
    "condition": Path(
        ("fac_condition", "~spon_condition"), n=50, half_life_days=730, max_fanout=3000
    ),
    # Cold sites: sponsors at other sites with the same zip / in the same state.
    "zip": Path(("zip", "~zip", "ran"), n=30, half_life_days=730),
    "state": Path(("state", "~state", "ran"), n=30, window_days=730, max_fanout=5000),
    # Sponsors that opened a site in the same city in the last year.
    "city_recent": Path(("city", "~city", "ran"), n=50, window_days=(None, None, 365)),
    # Run 2 (geo did best in run 1): the city walk without cross-country homonyms, and a
    # zip-prefix region between city and state.
    "city_country": Path(("city_country", "~city_country", "ran"), n=50, half_life_days=730),
    "zip3": Path(("zip3", "~zip3", "ran"), n=30, half_life_days=730, max_fanout=5000),
}
