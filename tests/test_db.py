"""Tests for relconduit.db (the slim parquet loader), and the leakage check on real rel-hm data."""

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table

from relconduit.db import load_tables

D = pd.Timestamp

TABLES = {
    "users": {"pkey": "user_id", "time_col": None, "fkeys": {}},
    "events": {"pkey": None, "time_col": "time", "fkeys": {"user_id": "users"}},
}


@pytest.fixture
def db_dir(tmp_path):
    n = 300
    rng = np.random.default_rng(0)
    users = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "country": rng.choice(["uk", "fr", "de"], n),  # low cardinality: kept
            "name": [f"user {i}" for i in range(n)],  # free text: left out
            "tags": [["a", "b"]] * n,  # nested: left out
            "secret": rng.normal(size=n),  # a task's remove_columns: left out
        }
    )
    events = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=n, freq="D"),
            "user_id": rng.integers(0, n, n),
            "value": rng.normal(size=n),
            "kind": rng.choice(["x", "y"], n),
        }
    )
    users.to_parquet(tmp_path / "users.parquet")
    events.to_parquet(tmp_path / "events.parquet")
    return tmp_path


def test_columns_kept_and_left_out(db_dir):
    db, left_out = load_tables(
        db_dir, TABLES, None, remove_columns=[("users", "secret")], max_categories=100
    )
    assert list(db.table_dict["users"].df.columns) == ["user_id", "country"]
    assert list(db.table_dict["events"].df.columns) == ["time", "user_id", "value", "kind"]
    assert left_out == {"users": ["name", "tags"], "events": []}


def test_cutoff_matches_table_upto(db_dir):
    cutoff = D("2020-03-01")
    db, _ = load_tables(db_dir, TABLES, cutoff)
    full = pd.read_parquet(db_dir / "events.parquet")
    full["kind"] = full["kind"].astype(object)  # relbench's loader does the same
    want = Table(full, {"user_id": "users"}, time_col="time").upto(cutoff).df
    got = db.table_dict["events"].df
    assert got["time"].max() == cutoff  # inclusive, as relbench's upto
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True))


def test_keys_are_kept_even_if_text(tmp_path):
    pd.DataFrame({"k": [f"id{i}" for i in range(500)], "v": range(500)}).to_parquet(
        tmp_path / "t.parquet"
    )
    db, left_out = load_tables(tmp_path, {"t": {"pkey": "k", "time_col": None, "fkeys": {}}}, None)
    assert list(db.table_dict["t"].df.columns) == ["k", "v"]
    assert left_out == {"t": []}


# --- real data -----------------------------------------------------------------------------


def _cached(dataset: str) -> bool:
    from huggingface_hub import CacheNotFound, scan_cache_dir

    try:
        repos = scan_cache_dir().repos
    except CacheNotFound:
        return False
    for repo in repos:
        if repo.repo_id == "stanford-star/relbench-v1":
            for rev in repo.revisions:
                if any(f"{dataset}/db/" in str(f.file_path) for f in rev.files):
                    return True
    return False


@pytest.mark.skipif(not _cached("rel-f1"), reason="rel-f1 not in the Hugging Face cache")
def test_slim_db_equals_get_db_on_kept_columns():
    import relbench

    from relconduit.db import load_task_db

    task = relbench.load_dataset("stanford-star/relbench-v1/rel-f1").load_task("driver-position")
    slim, full = load_task_db(task), task.get_db()
    assert set(slim.table_dict) == set(full.table_dict)
    for name, t in full.table_dict.items():
        s = slim.table_dict[name]
        assert (s.pkey_col, s.time_col, s.fkey_col_to_pkey_table) == (
            t.pkey_col,
            t.time_col,
            t.fkey_col_to_pkey_table,
        )
        pd.testing.assert_frame_equal(
            s.df.reset_index(drop=True), t.df[s.df.columns].reset_index(drop=True)
        )
        # Only text columns are ever left out.
        for col in set(t.df.columns) - set(s.df.columns):
            assert t.df[col].dtype == object, (name, col)


@pytest.mark.skipif(not _cached("rel-hm"), reason="rel-hm not in the Hugging Face cache")
def test_no_leakage_on_a_sample_of_rel_hm_item_sales():
    """Full vs truncated database on real rows: 40 train seeds of rel-hm/item-sales.

    The database is cut down to the sampled articles' transactions and every other purchase
    of their buyers (so hop 2, the last buyer's purchases, has real rows), loaded through
    the slim loader. The features are the experiment's, with windows trimmed for speed.
    """
    import relbench
    from relbench.base import Database

    from relconduit.db import load_task_db
    from relconduit.features import FeatureConfig, leakage_mismatches

    task = relbench.load_dataset("stanford-star/relbench-v1/rel-hm").load_task("item-sales")
    db = load_task_db(task)
    train = task.get_table("train").df
    rng = np.random.default_rng(0)
    tx = db.table_dict["transactions"].df
    # Articles with some history, so most seeds have rows before them and on their day.
    busy = tx["article_id"].value_counts()
    articles = rng.choice(busy[(busy > 50) & (busy < 400)].index.to_numpy(), 40, replace=False)
    seeds = train[train["article_id"].isin(articles)]
    seeds = seeds.iloc[rng.choice(len(seeds), 40, replace=False)].reset_index(drop=True)
    buyers = tx.loc[tx["article_id"].isin(articles), "customer_id"].unique()
    keep = tx["article_id"].isin(articles) | tx["customer_id"].isin(buyers)
    tables = dict(db.table_dict)
    t = tables["transactions"]
    tables["transactions"] = Table(
        tx[keep].reset_index(drop=True), t.fkey_col_to_pkey_table, t.pkey_col, t.time_col
    )
    small = Database(tables)
    cfg = FeatureConfig(windows=(7, None), aggs=("mean", "max", "sum"))
    args = (small, seeds[["article_id", "timestamp"]], "article", "article_id", "timestamp")
    assert leakage_mismatches(*args, cfg) == []
    bad = leakage_mismatches(
        *args, FeatureConfig(windows=cfg.windows, aggs=cfg.aggs, _inclusive=True)
    )
    assert bad, "a feature using <= must make the leakage check fail on real data"
    # The planted leak shows in the paths that read transactions, including hop 2.
    cols = {c for _, c, *_ in bad}
    assert any(c.startswith("transactions__") for c in cols)
    assert any(c.startswith("transactions.customer_id>transactions") for c in cols)
