"""Seed ensembling for the recommendation rankers (issue 32).

A ranker's seeds share one candidate set per seed row and differ only in the LightGBM seed.
:func:`rank_average` combines them: within each seed row, every seed's scores are turned into
ranks (0 is that seed's best candidate, ties share their mean rank), and the ensemble score is
minus the mean rank over seeds, so the ensemble's top-k is the candidates with the best average
rank. Ranks, not raw scores, because a lambdarank model's scores have no fixed scale from one
seed to the next.

:func:`save_top` writes each seed's and the ensemble's ``(n_rows, k)`` top-k table for one
split, so the val decision and its bootstrap (:func:`per_row_ap`, :func:`paired_bootstrap`)
run later without the pairs or the models.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from relconduit.link import SEED

ENSEMBLE_RULE = "rank average: mean within-row rank over seeds (ties share their mean rank)"


def within_row_rank(seed_ids: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Each pair's rank among its seed row's pairs by ``score``, 0 for the best; tied scores
    get their mean rank."""
    s = pd.Series(np.asarray(score, dtype=np.float64))
    return s.groupby(np.asarray(seed_ids)).rank(method="average", ascending=False).to_numpy() - 1


def rank_average(seed_ids: np.ndarray, scores: Sequence[np.ndarray]) -> np.ndarray:
    """The ensemble score of each pair: minus its mean within-row rank over ``scores`` (one
    array per seed, aligned with ``seed_ids``). Higher is better, as for the seeds' scores."""
    if not len(scores):
        raise ValueError("no scores to average")
    total = np.zeros(len(seed_ids), dtype=np.float64)
    for s in scores:
        if len(s) != len(seed_ids):
            raise ValueError("every score array must align with seed_ids")
        total += within_row_rank(seed_ids, s)
    return -total / len(scores)


def save_top(
    out_dir: str | Path,
    split: str,
    X: pd.DataFrame,
    scores: dict[int, np.ndarray],
    n_rows: int,
    k: int,
) -> dict[str, np.ndarray]:
    """Write ``<split>_top_seed<s>.npy`` per seed and ``<split>_top_ens.npy`` (the
    :func:`rank_average` of all of them) into ``out_dir``: ``(n_rows, k)`` dst ids, best first,
    ``-1`` padding. ``X`` holds the pairs (``SEED`` is the row index, ``DST`` the candidate)."""
    from relconduit.link_lgbm import top_k

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {f"seed{s}": top_k(X, sc, n_rows, k) for s, sc in scores.items()}
    tables["ens"] = top_k(X, rank_average(X[SEED].to_numpy(), list(scores.values())), n_rows, k)
    for name, t in tables.items():
        np.save(out / f"{split}_top_{name}.npy", t)
    return tables


class TopAccumulator:
    """:func:`save_top` for pairs that arrive a chunk at a time (every seed row's pairs within
    one chunk, as :mod:`relconduit.link_big` writes them)."""

    def __init__(self, seeds: Sequence[int], n_rows: int, k: int):
        self.seeds = list(seeds)
        names = [f"seed{s}" for s in self.seeds] + ["ens"]
        self.tables = {n: np.full((n_rows, k), -1, dtype=np.int64) for n in names}

    def add(self, X: pd.DataFrame, scores: Sequence[np.ndarray]) -> None:
        from relconduit.link_big import fill_top_k

        if not len(X):
            return
        for s, sc in zip(self.seeds, scores, strict=True):
            fill_top_k(self.tables[f"seed{s}"], X, sc)
        fill_top_k(self.tables["ens"], X, rank_average(X[SEED].to_numpy(), list(scores)))

    def save(self, out_dir: str | Path, split: str) -> dict[str, np.ndarray]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, t in self.tables.items():
            np.save(out / f"{split}_top_{name}.npy", t)
        return self.tables


def per_row_ap(pred_isin: np.ndarray, dst_count: np.ndarray) -> np.ndarray:
    """Per-row AP@k, as :func:`relbench.metrics.map` computes it before taking the mean (rows
    with no true dst dropped). Pass it to ``task.evaluate(pred, table, metrics=[per_row_ap])``
    so the hits come from RelBench's own matching; its mean is the official MAP."""
    keep = dst_count > 0
    pred_isin, dst_count = pred_isin[keep], dst_count[keep]
    eval_k = pred_isin.shape[1]
    precision = np.cumsum(pred_isin, axis=1) / (np.arange(eval_k) + 1)
    return (precision * pred_isin).sum(axis=1) / dst_count.clip(max=eval_k)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Mean of ``a - b`` (per-row scores of two tables on the same rows), its 95% bootstrap
    interval over rows, and the share of resamples where ``a`` beats ``b``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must score the same rows")
    d = a - b
    rng = np.random.default_rng(seed)
    batch = max(1, 20_000_000 // max(len(d), 1))  # resamples at a time, ~160 MB of indices
    means = np.concatenate(
        [
            d[rng.integers(0, len(d), size=(min(batch, n_boot - i), len(d)))].mean(axis=1)
            for i in range(0, n_boot, batch)
        ]
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_better": float((means > 0).mean()),
    }
