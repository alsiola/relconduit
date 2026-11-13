"""Tests for relconduit.link_search's helpers: subsets, chunks, selection, bootstrap, verdict."""

import numpy as np
import pandas as pd
import pytest

from relconduit.link_search import (
    _categorical,
    bootstrap_map,
    lead_verdict,
    nested_subset,
    pick_best,
    time_chunks,
)


def test_nested_subsets_are_nested_sorted_and_fixed():
    a, b, c = nested_subset(100, 10), nested_subset(100, 30), nested_subset(100, None)
    assert len(a) == 10 and len(b) == 30 and len(c) == 100
    assert set(a) <= set(b) <= set(c)
    assert np.all(np.diff(b) > 0)
    np.testing.assert_array_equal(a, nested_subset(100, 10))
    np.testing.assert_array_equal(nested_subset(5, 10), np.arange(5))


def test_time_chunks_keep_seed_times_whole():
    t = pd.Series(pd.to_datetime(["2020-01-01"] * 3 + ["2020-02-01"] * 2 + ["2020-03-01"] * 4))
    t = t.sample(frac=1, random_state=0).reset_index(drop=True)  # order must not matter
    chunks = time_chunks(t, max_rows=5)
    assert sorted(np.concatenate(chunks)) == list(range(9))
    for c in chunks:
        times = set(t.iloc[c])
        for u in times:  # every row of a seed time is in this chunk
            assert set(np.flatnonzero(t == u)) <= set(c)
    assert [len(c) for c in chunks] == [5, 4]
    assert [len(c) for c in time_chunks(t, max_rows=2)] == [3, 2, 4]


def test_pick_best_takes_the_highest_mean_and_the_first_on_a_tie():
    rs = [
        {"n": 0, "scores": [0.1, 0.4]},
        {"n": 1, "scores": [0.25, 0.25]},
        {"n": 2, "scores": [0.2]},
    ]
    assert pick_best(rs)["n"] == 0
    assert pick_best(rs[1:])["n"] == 1
    assert pick_best([{"n": 3, "scores": [0.1]}, *rs])["n"] == 0


def test_bootstrap_averages_seeds_over_the_same_resampled_rows():
    truth = np.array([1.0, 0.0, 1.0, 1.0])
    preds = [np.array([1.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0, 0.0])]

    def evaluate(rows, pred):  # accuracy against the resampled truth
        return float(np.mean(truth[rows] == pred))

    boot = bootstrap_map(evaluate, preds, n_rows=4, n_boot=200, seed=1)
    # Seed 0 is always right (1.0); seed 1's accuracy is the share of resampled 0-rows.
    assert boot.min() >= 0.5 and boot.max() <= 1.0
    assert abs(boot.mean() - (1.0 + 0.25) / 2) < 0.05
    np.testing.assert_array_equal(boot, bootstrap_map(evaluate, preds, 4, n_boot=200, seed=1))


def test_lead_verdict_applies_the_two_std_bar():
    v = lead_verdict([0.140, 0.141, 0.139], leader=0.1313)
    assert v["clear_lead"] and v["bar"] == pytest.approx(0.002)
    tie = lead_verdict([0.133, 0.131, 0.135], leader=0.1313)
    assert not tie["clear_lead"] and tie["margin"] == pytest.approx(0.0017)
    boot = np.array([0.12, 0.13, 0.14, 0.15])
    with_ci = lead_verdict([0.14, 0.141], leader=0.1313, boot=boot)
    assert with_ci["p_above_leader"] == 0.5 and len(with_ci["ci95"]) == 2


def test_categorical_unifies_string_columns_across_chunks():
    a = pd.DataFrame({"seed": [0], "dst": [1], "c": pd.Categorical(["x"]), "v": [1.0]})
    b = pd.DataFrame({"seed": [1], "dst": [2], "c": pd.Categorical(["y"]), "v": [2.0]})
    X = _categorical(pd.concat([a, b], ignore_index=True))
    assert isinstance(X["c"].dtype, pd.CategoricalDtype)
    assert set(X["c"].cat.categories) == {"x", "y"}
    assert X["v"].dtype == np.float64 and X["seed"].dtype == np.int64
