"""Tests for relconduit.entity_encoding: as-of sums, counts, target encoding, leakage."""

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

import relconduit.entity_encoding as ee
from relconduit.entity_encoding import EncodingConfig, EntityEncoder, asof_group_sums, template
from relconduit.features import leakage_mismatches

D = pd.Timestamp
H = pd.Timedelta(days=10)


def site_db() -> Database:
    """Four static sites; a time-stamped link table says when each first appears."""
    sites = pd.DataFrame(
        {
            "site_id": [0, 1, 2, 3],
            "city": ["paris", "paris", "rome", None],
            "name": ["Site 12", "Site 7", "Hospital", "Site 3"],
        }
    )
    links = pd.DataFrame(
        {
            "link_id": range(5),
            "site_id": [0, 1, 1, 2, 3],
            "date": [D("2020-01-01"), D("2020-02-01"), D("2020-01-15"), D("2020-03-01"), None],
        }
    )
    return Database(
        {
            "sites": Table(sites, {}, pkey_col="site_id"),
            "links": Table(links, {"site_id": "sites"}, pkey_col="link_id", time_col="date"),
        }
    )


LABELS = pd.DataFrame(
    {
        "site_id": [0, 1, 0, 2],
        "t": [D("2020-01-01"), D("2020-01-01"), D("2020-01-11"), D("2020-01-01")],
        "y": [1.0, 0.0, 0.5, 1.0],
    }
)
CFG = EncodingConfig(
    columns=("city",), templates=("name",), first_seen_via=("links", "site_id"), smoothing=1.0
)


def encoder(db=None, cfg=CFG, labels=LABELS):
    return EntityEncoder(db or site_db(), "sites", "site_id", "t", cfg, labels, "y", H)


def seeds(ids, times):
    return pd.DataFrame({"site_id": ids, "t": [D(x) for x in times]})


def test_asof_group_sums_matches_brute_force():
    rng = np.random.default_rng(0)
    n_ev, n_q = 300, 200
    ev_key, q_key = rng.integers(-1, 5, n_ev), rng.integers(-1, 5, n_q)
    base = np.datetime64("2020-01-01", "D")
    ev_t, q_t = base + rng.integers(0, 30, n_ev), base + rng.integers(0, 30, n_q)
    ev_y = rng.normal(size=n_ev)
    for strict in (True, False):
        cnt, tot = asof_group_sums(ev_key, ev_t, ev_y, q_key, q_t, strict)
        for i in range(n_q):
            before = ev_t < q_t[i] if strict else ev_t <= q_t[i]
            m = (ev_key == q_key[i]) & before & (q_key[i] >= 0)
            assert cnt[i] == m.sum()
            assert tot[i] == pytest.approx(ev_y[m].sum())


def test_template():
    s = template(pd.Series(["Site 12 / ID# 4", None, "  Hospital  "]))
    assert s.tolist()[0] == "Site # / ID# #"
    assert pd.isna(s.tolist()[1]) and s.tolist()[2] == "Hospital"


def test_counts_only_see_sites_first_linked_strictly_before_seed():
    X = encoder().transform(
        seeds([0, 0, 2, 3], ["2020-02-01", "2020-03-02", "2020-03-01", "2020-03-02"])
    )
    # 2020-02-01: site 0 (2020-01-01) and site 1 (first link 2020-01-15) are visible.
    assert X["enc__city__count"].tolist()[:2] == [2, 2]
    assert X["enc__city__share"].iloc[0] == 1.0
    # Site 2 first appears exactly at its seed time: not counted, so rome has 0.
    assert X["enc__city__count"].iloc[2] == 0
    # Missing city: NaN, not a count of other missing values. Site 3 has no dated link, so
    # it is never visible and never counted.
    assert np.isnan(X["enc__city__count"].iloc[3])
    assert X["enc__city__share"].iloc[1] == pytest.approx(2 / 3)
    # Templates: "Site 12" and "Site 7" are both "Site #".
    assert X["enc__name_tpl__count"].iloc[1] == 2


def test_target_encoding_uses_only_labels_known_by_seed_time():
    enc = encoder()
    # Labels at t=2020-01-01 are known from 2020-01-11; the one at 2020-01-11 from 01-21.
    X = enc.transform(seeds([1, 1, 1], ["2020-01-10", "2020-01-11", "2020-01-21"]))
    assert np.isnan(X["enc__city__te"].iloc[0])  # nothing known yet: no prior either
    assert X["enc__city__te_n"].tolist() == [0, 2, 3]
    # 2020-01-11: paris labels 1.0 (site 0) and 0.0 (site 1); prior over all known = 2/3.
    assert X["enc__city__te"].iloc[1] == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)


def test_target_encoding_never_sees_own_label():
    base = encoder().transform(seeds([0], ["2020-01-11"]))
    lab = LABELS.copy()
    lab.loc[2, "y"] = 99.0  # site 0's own label for its seed at 2020-01-11
    other = encoder(labels=lab).transform(seeds([0], ["2020-01-11"]))
    pd.testing.assert_frame_equal(base, other)


def test_target_key_is_own_past_labels():
    cfg = EncodingConfig(columns=(), first_seen_via=("links", "site_id"), target_key=True)
    X = encoder(cfg=cfg).transform(seeds([0], ["2020-01-21"]))
    assert X["enc__site_id__te_n"].iloc[0] == 2  # both of site 0's labels are known
    assert "enc__site_id__count" not in X


def test_no_leakage_full_vs_truncated_db():
    s = seeds(
        [0, 1, 2, 2, 3], ["2020-01-15", "2020-01-15", "2020-03-01", "2020-03-02", "2020-02-01"]
    )

    def build(db, sd, entity_table, entity_col, time_col, cfg):
        return encoder(db).transform(sd)

    assert leakage_mismatches(site_db(), s, "sites", "site_id", "t", builder=build) == []


def test_categorical_codebook_and_embeddings(monkeypatch):
    monkeypatch.setattr(ee, "embed_texts", lambda texts, model: np.eye(len(texts), 4))
    cfg = EncodingConfig(
        categorical=("city",), max_categories=1, embed=("name",), embed_dims=2, target=False
    )
    X = EntityEncoder(site_db(), "sites", "site_id", "t", cfg).transform(
        seeds([0, 2, 3], ["2020-01-01"] * 3)
    )
    assert list(X["enc__city__cat"].cat.categories) == ["paris"]
    assert X["enc__city__cat"].isna().tolist() == [False, True, True]
    assert {"enc__name__emb0", "enc__name__emb1"} <= set(X.columns)


# --- rel-trial/site-success ------------------------------------------------------------


@pytest.fixture(scope="module")
def trial():
    relbench = pytest.importorskip("relbench")
    try:
        ds = relbench.load_dataset("stanford-star/relbench-v1/rel-trial")
        task = ds.load_task("site-success")
        db = task.get_db(upto_test_timestamp=False)
        tables = {s: task.get_table(s).df for s in ("train", "val", "test")}
    except Exception as e:  # noqa: BLE001  (offline, no HF cache: skip, don't fail)
        pytest.skip(f"rel-trial not available: {e}")
    return task, db, tables


def _trial_leaks(trial, inclusive):
    task, db, tables = trial
    rng = np.random.default_rng(0)
    key = [task.entity_col, task.time_col]
    parts = [df.iloc[rng.choice(len(df), 2, replace=False)][key] for df in tables.values()]
    # Adversarial seeds: a site exactly at its first link to a study, when it becomes visible.
    first = db.table_dict["facilities_studies"].df.groupby("facility_id")["date"].min()
    first = first.iloc[rng.choice(len(first), 3, replace=False)]
    parts.append(pd.DataFrame({key[0]: first.index, key[1]: first.to_numpy()}))
    seeds = pd.concat(parts, ignore_index=True)
    cfg = EncodingConfig(
        columns=("city", "zip", "country"),
        templates=("name",),
        first_seen_via=("facilities_studies", "facility_id"),
        target_key=True,
        categorical=("country",),
        _inclusive=inclusive,
    )
    labels = tables["train"]

    def build(db, sd, entity_table, entity_col, time_col, _cfg):
        enc = EntityEncoder(
            db, entity_table, entity_col, time_col, cfg, labels, task.target_col, task.timedelta
        )
        return enc.transform(sd)

    return leakage_mismatches(
        db, seeds, task.entity_table, task.entity_col, task.time_col, builder=build
    )


def test_no_leakage_on_rel_trial(trial):
    assert _trial_leaks(trial, inclusive=False) == []


def test_planted_leak_is_caught_on_rel_trial(trial):
    bad = _trial_leaks(trial, inclusive=True)
    assert {i for i, *_ in bad} >= {6, 7, 8}  # every seed at a site's first link
    assert all(c.endswith(("__count", "__share")) for _, c, *_ in bad)
