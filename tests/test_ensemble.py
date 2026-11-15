import numpy as np
import pandas as pd
import pytest
from relbench.metrics import map as official_map

from relconduit.ensemble import (
    TopAccumulator,
    paired_bootstrap,
    per_row_ap,
    rank_average,
    save_top,
    within_row_rank,
)
from relconduit.link import DST, SEED


def _pairs():
    # Row 0 has candidates 10, 11, 12; row 1 has 20, 21.
    return pd.DataFrame({SEED: [0, 0, 0, 1, 1], DST: [10, 11, 12, 20, 21]})


def test_within_row_rank_is_per_row_and_ties_share():
    X = _pairs()
    r = within_row_rank(X[SEED].to_numpy(), np.array([0.1, 0.9, 0.5, 2.0, 2.0]))
    assert r.tolist() == [2.0, 0.0, 1.0, 0.5, 0.5]


def test_rank_average_ignores_scale():
    X = _pairs()
    ids = X[SEED].to_numpy()
    a = np.array([0.1, 0.9, 0.5, 1.0, 0.0])
    # Same order as a, wildly different scale: the average equals a's own ranks.
    b = 1000 * a - 50
    assert np.allclose(rank_average(ids, [a, b]), -within_row_rank(ids, a))
    # Two seeds that disagree on row 0: 11 is first for one and second for the other.
    c = np.array([0.5, 0.4, 0.1, 1.0, 0.0])
    ens = rank_average(ids, [a, c])
    assert ens[:3].tolist() == [-1.0, -0.5, -1.5]
    with pytest.raises(ValueError):
        rank_average(ids, [a[:3]])


def test_save_top_writes_seeds_and_ensemble(tmp_path):
    X = _pairs()
    a = np.array([0.1, 0.9, 0.5, 1.0, 0.0])
    c = np.array([0.5, 0.4, 0.1, 1.0, 0.0])
    tables = save_top(tmp_path, "val", X, {0: a, 1: c}, n_rows=3, k=2)
    assert tables["seed0"].tolist() == [[11, 12], [20, 21], [-1, -1]]
    assert tables["seed1"].tolist() == [[10, 11], [20, 21], [-1, -1]]
    assert tables["ens"].tolist() == [[11, 10], [20, 21], [-1, -1]]
    assert (np.load(tmp_path / "val_top_ens.npy") == tables["ens"]).all()


def test_accumulator_matches_save_top_over_chunks(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({SEED: np.repeat(np.arange(6), 5), DST: rng.permutation(30)})
    scores = {0: rng.normal(size=30), 1: rng.normal(size=30), 2: rng.normal(size=30)}
    whole = save_top(tmp_path / "a", "test", X, scores, n_rows=7, k=3)
    acc = TopAccumulator([0, 1, 2], n_rows=7, k=3)
    for lo, hi in ((0, 10), (10, 30)):  # chunks hold whole seed rows
        acc.add(X.iloc[lo:hi], [s[lo:hi] for s in scores.values()])
    chunked = acc.save(tmp_path / "b", "test")
    for name in whole:
        assert (whole[name] == chunked[name]).all(), name


def test_per_row_ap_mean_is_relbench_map():
    rng = np.random.default_rng(1)
    isin = rng.random((50, 10)) < 0.2
    count = rng.integers(0, 6, size=50)
    ap = per_row_ap(isin, count)
    assert len(ap) == int((count > 0).sum())
    assert ap.mean() == pytest.approx(official_map(isin, count))


def test_paired_bootstrap():
    a = np.array([0.5, 0.6, 0.7, 0.8] * 50)
    b = a - 0.1
    r = paired_bootstrap(a, b, n_boot=200)
    assert r["diff"] == pytest.approx(0.1)
    assert r["ci_lo"] == pytest.approx(0.1) and r["p_better"] == 1.0
    same = paired_bootstrap(a, a, n_boot=50)
    assert same["diff"] == 0 and same["p_better"] == 0
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:3])
