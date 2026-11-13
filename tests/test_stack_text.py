"""Tests for relconduit.stack_text: values on a tiny post history, and leakage."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

from relconduit.features import FeatureConfig, leakage_mismatches
from relconduit.stack_text import (
    builder,
    load_revision_stats,
    post_text_features,
    revision_stats,
    revisions_table,
)

D = pd.Timestamp


def history() -> pd.DataFrame:
    rows = [
        # post 0: asked 2020-01-01, retagged 2020-03-01 (exactly a seed time), body edited later
        (0, 0, 1, "How do I fit a GLM?", "2020-01-01"),
        (1, 0, 2, "I have `y` and $x$. See http://a.b ![img](c)", "2020-01-01"),
        (2, 0, 3, "<glm><r>", "2020-01-01"),
        (3, 0, 6, "<glm><regression><r>", "2020-03-01"),
        (4, 0, 5, "much longer body now", "2020-05-01"),
        # post 1: asked 2020-02-01 with tag glm
        (5, 1, 1, "GLM link functions", "2020-02-01"),
        (6, 1, 2, "body", "2020-02-01"),
        (7, 1, 3, "<glm>", "2020-02-01"),
        # post 2: asked after every seed
        (8, 2, 1, "Late", "2020-09-01"),
        (9, 2, 3, "<glm>", "2020-09-01"),
        # not a text revision: ignored
        (10, 0, 10, "closed", "2020-01-15"),
    ]
    df = pd.DataFrame(rows, columns=["Id", "PostId", "PostHistoryTypeId", "Text", "CreationDate"])
    df["CreationDate"] = pd.to_datetime(df["CreationDate"])
    return df


def tiny_db() -> Database:
    votes = pd.DataFrame(
        {
            "Id": range(6),
            "PostId": [0, 0, 1, 1, 1, 0],
            "VoteTypeId": [2, 3, 2, 2, 2, 2],
            "CreationDate": pd.to_datetime(
                ["2020-01-10", "2020-01-11", "2020-02-10", "2020-03-01", "2020-04-20", "2020-03-01"]
            ),
        }
    )
    posts = pd.DataFrame({"Id": [0, 1, 2]})
    return Database(
        {
            "posts": Table(posts, {}, pkey_col="Id"),
            "revisions": revisions_table(revision_stats(history())),
            "votes": Table(votes, {"PostId": "posts"}, pkey_col="Id", time_col="CreationDate"),
        }
    )


SEEDS = pd.DataFrame(
    {
        "PostId": [0, 0, 1, 2, 0],
        "timestamp": pd.to_datetime(
            ["2020-03-01", "2020-06-01", "2020-03-01", "2020-06-01", "2019-12-01"]
        ),
    },
    index=[5, 6, 7, 8, 9],
)


def test_revision_stats_row_by_row():
    st = revision_stats(history()).set_index("Id")
    assert 10 not in st.index
    assert st.loc[0, "kind"] == "title" and st.loc[0, "qmark"] == 1 and st.loc[0, "words"] == 6
    body = st.loc[1]
    assert body["kind"] == "body" and body["code"] == 2 and body["latex"] == 2
    assert body["links"] == 1 and body["images"] == 1
    assert sorted(st.loc[3, "tag_list"]) == ["glm", "r", "regression"] and st.loc[3, "n"] == 3


def test_load_revision_stats_matches_and_cuts(tmp_path):
    path = tmp_path / "postHistory.parquet"
    history().to_parquet(path)
    a = load_revision_stats(path).sort_values("Id").reset_index(drop=True)
    b = revision_stats(history()).sort_values("Id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns="tag_list"), b.drop(columns="tag_list"))
    cut = load_revision_stats(path, cutoff=D("2020-03-01"))
    assert cut["CreationDate"].max() == D("2020-03-01")


def test_values_are_as_of_the_seed_time():
    f = post_text_features(tiny_db(), SEEDS)
    assert list(f.index) == [5, 6, 7, 8, 9]
    # Seed at 2020-03-01: the retag on that day is not yet visible.
    assert f.loc[5, "text__tags_n"] == 2 and f.loc[5, "text__tags_revisions"] == 1
    assert f.loc[5, "text__body_len"] == len("I have `y` and $x$. See http://a.b ![img](c)")
    # Later: the retag and the body edit are.
    assert f.loc[6, "text__tags_n"] == 3 and f.loc[6, "text__body_len"] == 20
    assert f.loc[6, "text__body_revisions"] == 2
    # Tag glm on 2020-03-01: questions 0 and 1; upvotes before then: post 0 one, post 1 one.
    assert f.loc[5, "text__tag_questions_max"] == 2
    assert f.loc[7, "text__tag_up_all_sum"] == 2
    assert f.loc[7, "text__tag_up_per_q_mean"] == 1
    # Seed 6 (2020-06-01), tags glm, regression, r: glm has 2 questions and 5 upvotes.
    assert f.loc[6, "text__tag_up_all_max"] == 5 and f.loc[6, "text__tag_questions_min"] == 1
    # Before the post exists, and for a post asked after the seed: no text.
    assert f.loc[9, ["text__body_len", "text__tag_questions_max"]].isna().all()
    assert f.loc[9, "text__tags_revisions"] == 0
    assert np.isnan(f.loc[8, "text__title_len"])


CFG = FeatureConfig(vocab={})


def test_no_leakage_on_tiny_db():
    assert leakage_mismatches(tiny_db(), SEEDS, "posts", "PostId", "timestamp", CFG, builder) == []


def test_planted_leak_is_caught_on_tiny_db():
    leaky = replace(CFG, _inclusive=True)
    bad = leakage_mismatches(tiny_db(), SEEDS, "posts", "PostId", "timestamp", leaky, builder)
    cols = {c for _, c, *_ in bad}
    assert "text__tags_n" in cols and "text__tag_up_all_sum" in cols


# --- rel-stack ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stack():
    relbench = pytest.importorskip("relbench")
    try:
        ds = relbench.load_dataset("stanford-star/relbench-v1/rel-stack")
        task = ds.load_task("post-votes")
        db_dir = ds.db_dir
        tables = {s: task.get_table(s).df for s in ("train", "val")}
    except Exception as e:  # noqa: BLE001  (offline, no HF cache: skip, don't fail)
        pytest.skip(f"rel-stack not available: {e}")
    import duckdb

    # A slice: 300 questions, every revision and vote of theirs (the tag aggregates then
    # cover only these questions, which is enough to test the cut).
    rng = np.random.default_rng(0)
    train = tables["train"]
    pids = rng.choice(train["PostId"].unique(), 300, replace=False)
    ph = duckdb.sql(
        f"SELECT Id, PostId, PostHistoryTypeId, Text, CreationDate "
        f"FROM read_parquet('{db_dir}/postHistory.parquet') "
        f"WHERE PostId IN ({','.join(map(str, pids))})"
    ).df()
    votes = duckdb.sql(
        f"SELECT Id, PostId, VoteTypeId, CreationDate FROM read_parquet('{db_dir}/votes.parquet') "
        f"WHERE PostId IN ({','.join(map(str, pids))})"
    ).df()
    db = Database(
        {
            "posts": Table(pd.DataFrame({"Id": pids}), {}, pkey_col="Id"),
            "revisions": revisions_table(revision_stats(ph)),
            "votes": Table(votes, {"PostId": "posts"}, pkey_col="Id", time_col="CreationDate"),
        }
    )
    seeds = train[train["PostId"].isin(pids)].sample(12, random_state=0)
    # Adversarial seeds: a post exactly at the time of one of its retags and of one of its votes.
    retag = ph[ph["PostHistoryTypeId"] == 6].head(3)
    up = votes[votes["VoteTypeId"] == 2].head(3)
    adv = pd.DataFrame(
        {
            "PostId": pd.concat([retag["PostId"], up["PostId"]]).to_numpy(),
            "timestamp": pd.concat([retag["CreationDate"], up["CreationDate"]]).to_numpy(),
        }
    )
    seeds = pd.concat([seeds[["PostId", "timestamp"]], adv], ignore_index=True)
    return db, seeds


def test_no_leakage_on_rel_stack(stack):
    db, seeds = stack
    assert leakage_mismatches(db, seeds, "posts", "PostId", "timestamp", CFG, builder) == []


def test_planted_leak_is_caught_on_rel_stack(stack):
    db, seeds = stack
    leaky = replace(CFG, _inclusive=True)
    assert leakage_mismatches(db, seeds, "posts", "PostId", "timestamp", leaky, builder)
