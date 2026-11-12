"""Tests for relconduit.features: values on a tiny synthetic database, and leakage."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

from relconduit.features import (
    FeatureConfig,
    build_features,
    feature_vocab,
    leakage_mismatches,
    truncate_before,
)

D = pd.Timestamp


def tiny_db() -> Database:
    """Two users, two teams, events and team scores. Some rows sit exactly on seed times."""
    users = pd.DataFrame(
        {
            "user_id": [0, 1],
            "height": [1.8, np.nan],
            "country": ["uk", "fr"],
            "born": [D("2000-01-01"), D("1990-01-01")],
            "name": ["alice", "bob"],
        }
    )
    teams = pd.DataFrame({"team_id": [0, 1], "label": ["red", "blue"]})
    events = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4, 5],
            "user_id": [0, 0, 0, 0, 1, 0],
            "team_id": [0, 0, 1, 1, 0, 0],
            "time": [
                D("2020-01-01"),
                D("2020-03-01"),
                D("2020-05-01"),
                D("2020-06-01"),  # exactly on a seed time below
                D("2020-02-01"),
                D("2020-07-01"),  # after every seed
            ],
            "value": [1.0, 3.0, np.nan, 100.0, 7.0, 1000.0],
            "points": [10, 20, 30, 40, 50, 60],
        }
    )
    team_scores = pd.DataFrame(
        {
            "score_id": [0, 1, 2, 3],
            "team_id": [0, 1, 1, 1],
            "time": [D("2020-01-15"), D("2020-04-01"), D("2020-05-20"), D("2020-06-01")],
            "score": [5.0, 1.0, 2.0, 50.0],
        }
    )
    return Database(
        {
            "users": Table(users, {}, pkey_col="user_id"),
            "teams": Table(teams, {}, pkey_col="team_id"),
            "events": Table(
                events,
                {"user_id": "users", "team_id": "teams"},
                pkey_col="event_id",
                time_col="time",
            ),
            "team_scores": Table(
                team_scores, {"team_id": "teams"}, pkey_col="score_id", time_col="time"
            ),
        }
    )


SEEDS = pd.DataFrame(
    {
        "user_id": [0, 0, 1, 0, 1],
        "time": [
            D("2020-06-01"),
            D("2020-02-15"),
            D("2020-06-01"),
            D("2019-06-01"),
            D("2020-02-01"),
        ],
    },
    index=[10, 11, 12, 13, 14],
)
CFG = FeatureConfig(windows=(30, 90, None))


def feats(db=None, seeds=SEEDS, cfg=CFG):
    return build_features(db or tiny_db(), seeds, "users", "user_id", "time", cfg)


def test_shape_order_and_index():
    X = feats()
    assert list(X.index) == [10, 11, 12, 13, 14]
    assert not any(
        c.startswith(("events__event_id", "events__user_id", "events__team_id")) for c in X
    )
    assert "events__time__mean__all" not in X


def test_hop1_values_are_strictly_before_seed():
    X = feats().loc[10]  # user 0 at 2020-06-01: events 0, 1, 2 (event 3 is *at* the seed time)
    assert X["events__count__all"] == 3
    assert X["events__value__mean__all"] == pytest.approx(2.0)  # NaN ignored
    assert X["events__value__max__all"] == 3.0
    assert X["events__value__nullfrac__all"] == pytest.approx(1 / 3)
    assert np.isnan(X["events__value__last"])  # the most recent value is missing; kept as NaN
    assert X["events__points__last"] == 30
    assert X["events__days_since_last"] == pytest.approx(31.0)
    # Windows: 30d = [2020-05-02, 2020-06-01) holds nothing; 90d = [2020-03-03, ...) holds
    # only event 2 (event 1, on 03-01, is two days outside).
    assert X["events__count__30d"] == 0
    assert X["events__count__90d"] == 1
    assert X["events__points__mean__90d"] == pytest.approx(30.0)
    assert np.isnan(X["events__points__mean__30d"])


def test_int_columns_have_no_nullfrac():
    X = feats()
    assert "events__points__nullfrac__all" not in X
    assert "events__value__nullfrac__all" in X


def test_no_history_gives_zero_counts_and_nan():
    X = feats()
    for idx in (13, 14):  # user 0 before any event; user 1 exactly at their only event
        assert X.loc[idx, "events__count__all"] == 0
        assert np.isnan(X.loc[idx, "events__value__mean__all"])
        assert np.isnan(X.loc[idx, "events__days_since_last"])


def test_upward_join_of_static_parent_without_numeric_columns_adds_nothing():
    assert not any("teams." in c.split("__")[1] for c in feats().columns if "__" in c)


def test_hop2_through_current_team():
    X = feats().loc[10]
    # user 0's latest event before 2020-06-01 is event 2 (team 1). Team 1 scores before then:
    # 1.0 (04-01) and 2.0 (05-20); the 06-01 score is at the seed time and excluded.
    p = "events.team_id>team_scores"
    assert X[f"{p}__count__all"] == 2
    assert X[f"{p}__score__mean__all"] == pytest.approx(1.5)
    assert X[f"{p}__score__last"] == 2.0
    assert X[f"{p}__count__30d"] == 1
    # And events of that team (here only user 0's own event 2): T2 may be T itself.
    assert X["events.team_id>events__count__all"] == 1
    # Seed 11 (2020-02-15): latest event is event 0 (team 0): its 01-15 score.
    assert feats().loc[11, f"{p}__score__last"] == 5.0


def test_hop2_off():
    X = feats(cfg=FeatureConfig(windows=(None,), hop2=False))
    assert not any(">" in c for c in X.columns)


def test_entity_static_features():
    X = feats()
    assert X.loc[10, "users__height"] == 1.8
    assert np.isnan(X.loc[12, "users__height"])
    assert X.loc[10, "users__born__days_before"] == (D("2020-06-01") - D("2000-01-01")).days
    assert X.loc[12, "users__country"] == "fr"
    assert isinstance(X["users__country"].dtype, pd.CategoricalDtype)
    assert list(X["users__country"].cat.categories) == ["fr", "uk"]


def test_exclusions():
    cfg = FeatureConfig(
        windows=(None,), exclude_tables=["team_scores"], exclude_columns=[("events", "value")]
    )
    X = feats(cfg=cfg)
    assert not any("team_scores" in c for c in X.columns)
    assert not any("__value__" in c for c in X.columns)
    assert "events__points__mean__all" in X


def test_truncate_before_is_strict():
    t = truncate_before(tiny_db(), D("2020-06-01"))
    assert t.table_dict["events"].df["time"].max() < D("2020-06-01")
    assert len(t.table_dict["events"]) == 4
    assert len(t.table_dict["users"]) == 2  # static tables untouched


def test_no_leakage_on_tiny_db():
    assert leakage_mismatches(tiny_db(), SEEDS, "users", "user_id", "time", CFG) == []


def test_planted_leak_is_caught_on_tiny_db():
    leaky = FeatureConfig(windows=CFG.windows, _inclusive=True)
    bad = leakage_mismatches(tiny_db(), SEEDS, "users", "user_id", "time", leaky)
    assert bad, "a feature using <= must make the leakage check fail"
    assert {SEEDS.index[i] for i, *_ in bad} == {10, 14}  # exactly the seeds with a same-time row


# --- time-stamped entity, text columns, entity parents, hop 2 over all memberships -------


def timed_db() -> Database:
    """Users with a sign-up time, a static club each, a static friendship table, events."""
    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "joined": [D("2020-01-01"), D("2020-03-01"), D("2020-01-15")],
            "height": [1.8, 1.6, 1.7],
            "country": ["uk", "fr", "uk"],
            "club_id": [0, 1, 1],
        }
    )
    clubs = pd.DataFrame({"club_id": [0, 1], "size": [10.0, 20.0], "kind": ["a", "b"]})
    friends = pd.DataFrame({"user": [0, 0, 1, 2], "friend": [1, 2, 0, 0]})
    events = pd.DataFrame(
        {
            "event_id": range(8),
            "user_id": [0, 0, 1, 1, 2, 2, 2, 0],
            "time": [
                D("2020-02-01"),
                D("2020-04-01"),  # exactly on a seed time below
                D("2020-03-05"),
                D("2020-04-01"),  # exactly on a seed time below
                D("2020-01-20"),
                D("2020-03-20"),
                D("2020-05-01"),
                D("2020-06-01"),
            ],
            "status": ["yes", "no", "yes", "maybe", "yes", None, "no", "yes"],
            "value": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0],
        }
    )
    return Database(
        {
            "users": Table(users, {"club_id": "clubs"}, pkey_col="user_id", time_col="joined"),
            "clubs": Table(clubs, {}, pkey_col="club_id"),
            "friends": Table(friends, {"user": "users", "friend": "users"}),
            "events": Table(events, {"user_id": "users"}, pkey_col="event_id", time_col="time"),
        }
    )


TIMED_SEEDS = pd.DataFrame(
    {
        "user_id": [0, 1, 1, 0, 2],
        "time": [
            D("2020-04-01"),
            D("2020-03-01"),
            D("2020-04-01"),
            D("2020-01-01"),
            D("2020-04-01"),
        ],
    },
    index=[20, 21, 22, 23, 24],
)
TIMED_CFG = FeatureConfig(windows=(30, None), hop2_all_via=("users",))


def timed_feats(cfg=TIMED_CFG, db=None, seeds=TIMED_SEEDS):
    return build_features(db or timed_db(), seeds, "users", "user_id", "time", cfg)


def test_timed_entity_columns_only_after_its_own_timestamp():
    X = timed_feats()
    # Seed 22: user 1 (joined 03-01) at 04-01 is visible, with its age and its club.
    assert X.loc[22, "users__height"] == 1.6
    assert X.loc[22, "users__joined__days_before"] == 31
    assert X.loc[22, "users__country"] == "fr"
    assert X.loc[22, "users.club_id__size"] == 20.0
    assert X.loc[22, "users.club_id__kind"] == "b"
    # Seeds 21 and 23: the seed time equals the user's own timestamp, so nothing is known.
    for idx in (21, 23):
        assert np.isnan(X.loc[idx, "users__height"])
        assert np.isnan(X.loc[idx, "users__joined__days_before"])
        assert pd.isna(X.loc[idx, "users__country"])
        assert np.isnan(X.loc[idx, "users.club_id__size"])
        assert pd.isna(X.loc[idx, "users.club_id__kind"])
    assert not any(c.startswith("users__club_id") for c in X.columns)  # keys are not features


def test_text_columns_give_value_counts_and_last_category():
    X = timed_feats()
    # User 0 at 04-01: only event 0 (yes) is strictly before; event 1 is at the seed time.
    yes = [c for c in X.columns if c.startswith("events__status=") and "_yes__count__all" in c]
    assert len(yes) == 1
    assert X.loc[20, yes[0]] == 1
    assert X.loc[20, "events__status__last"] == "yes"
    assert isinstance(X["events__status__last"].dtype, pd.CategoricalDtype)
    assert list(X["events__status__last"].cat.categories) == ["maybe", "no", "yes"]
    # User 2 at 04-01: events 4 (yes) and 5 (status missing); last keeps the missing value.
    assert pd.isna(X.loc[24, "events__status__last"])
    assert X.loc[24, yes[0]] == 1


def test_hop2_all_pools_every_linked_parent():
    X = timed_feats()
    p = "friends.friend>>events"
    # User 0 at 04-01: friends 1 and 2. Their events strictly before 04-01: 2 (3.0), 4 (NaN),
    # 5 (6.0); event 3 is at the seed time.
    assert X.loc[20, f"{p}__n_parents"] == 2
    assert X.loc[20, f"{p}__count__all"] == 3
    assert X.loc[20, f"{p}__value__mean__all"] == pytest.approx(4.5)
    assert X.loc[20, f"{p}__value__min__all"] == 3.0
    assert X.loc[20, f"{p}__value__max__all"] == 6.0
    assert X.loc[20, f"{p}__value__nullfrac__all"] == pytest.approx(1 / 3)
    assert X.loc[20, f"{p}__count__30d"] == 2  # 03-05 and 03-20, not 01-20
    assert X.loc[20, f"{p}__days_since_last"] == pytest.approx(12.0)
    yes = next(
        c for c in X.columns if c.startswith(f"{p}__status=") and c.endswith("_yes__count__all")
    )
    assert X.loc[20, yes] == 2
    # User 1's only friend is user 0: event 0 before 04-01.
    assert X.loc[22, f"{p}__count__all"] == 1
    # The other direction of the same table is a separate path.
    assert X.loc[20, "friends.user>>events__count__all"] == 3


def test_hop2_all_rollup_matches_brute_force():
    db = timed_db()
    X = timed_feats(FeatureConfig(windows=(None,), hop2_all_via=("users",)))
    ev = db.table_dict["events"].df
    fr = db.table_dict["friends"].df
    for idx, row in TIMED_SEEDS.iterrows():
        friends = fr.loc[fr["user"] == row["user_id"], "friend"].unique()
        rows = ev[ev["user_id"].isin(friends) & (ev["time"] < row["time"])]
        assert X.loc[idx, "friends.friend>>events__count__all"] == len(rows)
        got = X.loc[idx, "friends.friend>>events__value__mean__all"]
        want = rows["value"].mean()
        assert (np.isnan(got) and np.isnan(want)) or got == pytest.approx(want)


def test_hop2_all_chain_goes_one_link_further():
    # "users>users": friends' friends (through the friendship table again), then their events.
    cfg = FeatureConfig(windows=(None,), hop2_all_via=("users>users",))
    X = timed_feats(cfg)
    db = timed_db()
    ev, fr = db.table_dict["events"].df, db.table_dict["friends"].df
    col = "friends.friend>>friends.friend>>events__count__all"
    assert col in X
    for idx, row in TIMED_SEEDS.iterrows():
        f1 = fr.loc[fr["user"] == row["user_id"], "friend"].unique()
        f2 = fr.loc[fr["user"].isin(f1), "friend"].unique()
        want = (ev["user_id"].isin(f2) & (ev["time"] < row["time"])).sum()
        assert X.loc[idx, col] == want
    assert not any(c.startswith("friends.friend>>events") for c in X.columns)  # chain only
    assert leakage_mismatches(db, TIMED_SEEDS, "users", "user_id", "time", cfg) == []
    leaky = replace(cfg, _inclusive=True)
    bad = leakage_mismatches(db, TIMED_SEEDS, "users", "user_id", "time", leaky)
    assert any(c == col for _, c, *_ in bad)


def test_hop2_all_is_opt_in():
    X = timed_feats(FeatureConfig(windows=(None,)))
    assert not any(">>" in c for c in X.columns)


def test_no_leakage_with_timed_entity():
    assert leakage_mismatches(timed_db(), TIMED_SEEDS, "users", "user_id", "time", TIMED_CFG) == []


def test_planted_leak_is_caught_with_timed_entity():
    leaky = replace(TIMED_CFG, _inclusive=True)
    bad = leakage_mismatches(timed_db(), TIMED_SEEDS, "users", "user_id", "time", leaky)
    rows = {TIMED_SEEDS.index[i] for i, *_ in bad}
    cols = {c for _, c, *_ in bad}
    # 21 and 23 sit on the user's own timestamp; 20, 22 and 24 on an event's.
    assert {21, 23} <= rows
    assert "users__height" in cols and "users.club_id__size" in cols
    assert any(">>" in c for c in cols)


def test_vocab_is_shared_by_leakage_builds():
    db = timed_db()
    early = truncate_before(db, D("2020-02-15"))
    assert feature_vocab(early)[("events", "status")] == ["yes"]
    assert feature_vocab(db)[("events", "status")] == ["maybe", "no", "yes"]
    cfg = replace(TIMED_CFG, vocab=feature_vocab(db))
    seeds = TIMED_SEEDS.iloc[[3]]
    assert list(timed_feats(cfg, db=early, seeds=seeds).columns) == list(
        timed_feats(cfg, seeds=seeds).columns
    )


# --- rel-f1 -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def f1():
    relbench = pytest.importorskip("relbench")
    try:
        ds = relbench.load_dataset("stanford-star/relbench-v1/rel-f1")
        task = ds.load_task("driver-dnf")
        db = task.get_db(upto_test_timestamp=False)
        tables = {s: task.get_table(s).df for s in ("train", "val", "test")}
    except Exception as e:  # noqa: BLE001  (offline, no HF cache: skip, don't fail)
        pytest.skip(f"rel-f1 not available: {e}")
    return task, db, tables


def _f1_seeds(task, db, tables, n_per_split=5, n_adversarial=6, seed=0):
    rng = np.random.default_rng(seed)
    parts = [df.iloc[rng.choice(len(df), n_per_split, replace=False)] for df in tables.values()]
    # Adversarial seeds: a driver at the exact timestamp of one of their own result rows, so a
    # `<=` anywhere would pull that row in.
    res = db.table_dict["results"].df
    res = res.iloc[rng.choice(len(res), n_adversarial, replace=False)]
    parts.append(pd.DataFrame({task.entity_col: res["driverId"], task.time_col: res["date"]}))
    return pd.concat([p[[task.entity_col, task.time_col]] for p in parts], ignore_index=True)


def test_no_leakage_on_rel_f1(f1):
    task, db, tables = f1
    seeds = _f1_seeds(task, db, tables)
    bad = leakage_mismatches(db, seeds, task.entity_table, task.entity_col, task.time_col)
    assert bad == []


def test_planted_leak_is_caught_on_rel_f1(f1):
    task, db, tables = f1
    seeds = _f1_seeds(task, db, tables)
    leaky = FeatureConfig(_inclusive=True)
    bad = leakage_mismatches(db, seeds, task.entity_table, task.entity_col, task.time_col, leaky)
    assert bad


# --- rel-event: a time-stamped entity table (users.joinedAt) -------------------------------

EVENT_CFG = FeatureConfig(windows=(7, None), hop2_all_via=("users",))


@pytest.fixture(scope="module")
def event():
    relbench = pytest.importorskip("relbench")
    try:
        ds = relbench.load_dataset("stanford-star/relbench-v1/rel-event")
        task = ds.load_task("user-attendance")
        db = task.get_db(upto_test_timestamp=False)
        tables = {s: task.get_table(s).df for s in ("train", "val", "test")}
    except Exception as e:  # noqa: BLE001  (offline, no HF cache: skip, don't fail)
        pytest.skip(f"rel-event not available: {e}")
    return task, db, tables


def _event_seeds(task, db, tables, seed=0):
    rng = np.random.default_rng(seed)
    parts = [df.iloc[rng.choice(len(df), 2, replace=False)] for df in tables.values()]
    key = [task.entity_col, task.time_col]
    parts = [p[key] for p in parts]
    # Adversarial seeds: a user exactly at their own sign-up time (the entity's own columns
    # must be hidden), and a user exactly at the start of an event they responded to.
    users = db.table_dict["users"].df
    u = users.iloc[rng.choice(len(users), 3, replace=False)]
    parts.append(pd.DataFrame({key[0]: u["user_id"].astype("int64"), key[1]: u["joinedAt"]}))
    att = db.table_dict["event_attendees"].df.dropna(subset=["user_id"])
    a = att.iloc[rng.choice(len(att), 2, replace=False)]
    parts.append(pd.DataFrame({key[0]: a["user_id"].astype("int64"), key[1]: a["start_time"]}))
    return pd.concat(parts, ignore_index=True)


def test_no_leakage_on_rel_event(event):
    task, db, tables = event
    seeds = _event_seeds(task, db, tables)
    bad = leakage_mismatches(
        db, seeds, task.entity_table, task.entity_col, task.time_col, EVENT_CFG
    )
    assert bad == []


def test_planted_leak_is_caught_on_rel_event(event):
    task, db, tables = event
    seeds = _event_seeds(task, db, tables)
    leaky = replace(EVENT_CFG, _inclusive=True)
    bad = leakage_mismatches(db, seeds, task.entity_table, task.entity_col, task.time_col, leaky)
    rows = {i for i, *_ in bad}
    cols = {c for _, c, *_ in bad}
    assert set(range(6, 9)) <= rows  # every seed at a user's own sign-up time
    assert any(c.startswith("users__") for c in cols)  # through the entity's own columns
    assert any(c.startswith("event_attendees__") for c in cols)
