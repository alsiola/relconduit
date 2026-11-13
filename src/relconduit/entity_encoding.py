"""Time-safe encodings of an entity table's high-cardinality columns.

:mod:`relconduit.features` drops text columns with more than ``MAX_CATEGORIES`` values, so an
entity like a trial site (``facilities``: ``name``, ``city``, ``state``, ``zip``,
``country``) loses all of its own columns. :class:`EntityEncoder` adds them back, per seed row
``(entity id, seed time)``, as:

* **counts** (``enc__<col>__count``, ``enc__<col>__share``): how many entities with the seed
  entity's value were *visible* strictly before the seed time, and that as a share of all
  visible entities. An entity is visible from its own timestamp if the entity table has a
  time column, else from its first row in :attr:`EncodingConfig.first_seen_via` (a
  time-stamped link table such as ``facilities_studies``). So a static table's counts still
  move with the seed time: an entity that first shows up after the seed is not counted. With
  neither, every row of the static table counts at every seed time (static tables are always
  whole under the leaderboard rule).
* **target encoding** (``enc__<col>__te``, ``enc__<col>__te_n``): the smoothed mean label of
  entities sharing the value, over **train labels only**, and only those whose label window
  had closed by the seed time: a label for seed time ``s`` is computed from rows in
  ``[s, s + horizon)``, so it is known at ``t`` iff ``s + horizon <= t``. This is
  out-of-fold in time: a row never sees its own label (its window ends after its own seed
  time), nor any label from its future. Encoding the primary key gives the entity's own past
  labels (a lagged target).
* **categoricals** (``enc__<col>__cat``): the ``max_categories`` most common values over the
  whole static table, as a LightGBM categorical.
* **text embeddings** (``enc__<col>__emb<i>``): an open-source sentence-embedding model
  (GloVe word averages by default, as upstream RelBench's LightGBM uses), reduced by PCA fitted
  on the whole static table.

Columns in :attr:`EncodingConfig.templates` are also encoded as a template, digits replaced
by ``#`` (``"Teva Investigational Site 123"`` becomes ``"Teva Investigational Site #"``):
anonymised sponsor sites share one.

Every value an encoding reads is the entity's own static row, a whole static table, a
time-stamped row strictly before the seed time, or a train label known by the seed time.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from relbench.base import Database

PREFIX = "enc__"
GLOVE = "sentence-transformers/average_word_embeddings_glove.6B.300d"


@dataclass
class EncodingConfig:
    """Which columns of the entity table get which encodings."""

    columns: Sequence[str] = ()  # count and target encoding
    templates: Sequence[str] = ()  # also encode digits-as-# templates of these columns
    first_seen_via: tuple[str, str] | None = None  # (link table, its fk to the entity)
    count: bool = True
    target: bool = True
    target_key: bool = False  # also target-encode the primary key (own past labels)
    smoothing: float = 20.0  # pseudo-count pulling a value's mean label to the prior
    categorical: Sequence[str] = ()
    max_categories: int = 250
    embed: Sequence[str] = ()
    embed_dims: int = 16
    embed_model: str = GLOVE
    # Testing only: count entities visible at ``<=`` the seed time instead of ``<``, so the
    # leakage test can prove it catches a leak.
    _inclusive: bool = False


def template(values: pd.Series) -> pd.Series:
    """Digits replaced by ``#`` and whitespace collapsed, missing kept missing."""
    s = values.astype("string").str.replace(r"\d+", "#", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def asof_group_sums(
    ev_key: np.ndarray,
    ev_t: np.ndarray,
    ev_y: np.ndarray,
    q_key: np.ndarray,
    q_t: np.ndarray,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per query, the count and sum of ``ev_y`` over events with its key, timed before it.

    Events count when ``ev_t < q_t`` (``strict``) or ``ev_t <= q_t``. Keys are integer
    codes; a negative key (missing) matches nothing and gives count 0, sum 0.
    """
    n_ev, n_q = len(ev_key), len(q_key)
    key = np.concatenate([ev_key, q_key])
    t = np.concatenate([ev_t, q_t]).astype("datetime64[ns]").astype(np.int64)
    is_q = np.concatenate([np.zeros(n_ev, bool), np.ones(n_q, bool)])
    # On a tie, strict puts the query before the event (not counted), else after it.
    tie = is_q if not strict else ~is_q
    order = np.lexsort((tie, t, key))
    k, q = key[order], is_q[order]
    w = np.where(q, 0.0, 1.0)
    y = np.where(q, 0.0, np.concatenate([ev_y, np.zeros(n_q)])[order])
    cw, cy = np.cumsum(w), np.cumsum(y)
    # Subtract the running totals at the start of each key's block.
    start = np.r_[True, k[1:] != k[:-1]]
    idx = np.maximum.accumulate(np.where(start, np.arange(len(k)), 0))
    base_w = np.where(idx > 0, cw[idx - 1], 0.0)
    base_y = np.where(idx > 0, cy[idx - 1], 0.0)
    cnt = np.empty(n_q)
    tot = np.empty(n_q)
    qpos = order[q] - n_ev
    cnt[qpos] = (cw - base_w)[q]
    tot[qpos] = (cy - base_y)[q]
    missing = q_key < 0
    cnt[missing], tot[missing] = 0.0, 0.0
    return cnt, tot


class EntityEncoder:
    """Encodings of ``entity_table``'s columns for seed rows. See the module doc.

    ``labels`` (train only) has ``entity_col``, ``time_col`` (the seed time) and ``target``;
    it is needed for target encoding, with ``horizon`` (the task's ``timedelta``).
    """

    def __init__(
        self,
        db: Database,
        entity_table: str,
        entity_col: str,
        time_col: str,
        cfg: EncodingConfig,
        labels: pd.DataFrame | None = None,
        target: str | None = None,
        horizon: pd.Timedelta | None = None,
    ):
        self.cfg = cfg
        self.entity_col, self.time_col = entity_col, time_col
        t = db.table_dict[entity_table]
        ent = t.df.set_index(t.pkey_col)
        self.key_name = t.pkey_col
        cols = {c: ent[c] for c in cfg.columns}
        for c in cfg.templates:
            cols[f"{c}_tpl"] = template(ent[c])
        if cfg.target_key:
            cols[t.pkey_col] = pd.Series(ent.index, index=ent.index)
        # One integer code per value, per column; -1 for missing.
        self.codes = {
            c: pd.Series(pd.factorize(s.astype("string"))[0], index=ent.index)
            for c, s in cols.items()
        }
        self.ids = ent.index
        self.visible_from = self._visible_from(db, t, ent)
        self.ent = ent
        if cfg.target:
            if labels is None or target is None or horizon is None:
                raise ValueError("target encoding needs labels, target and horizon")
            lab = labels[[entity_col, time_col, target]].dropna()
            self.lab_id = lab[entity_col].to_numpy()
            self.lab_known = (pd.to_datetime(lab[time_col]) + horizon).to_numpy()
            self.lab_y = lab[target].to_numpy(dtype=float)
        self._cat = {c: self._codebook(ent[c]) for c in cfg.categorical}
        self._emb = {c: self._embed(ent[c]) for c in cfg.embed}

    def _visible_from(self, db: Database, t, ent: pd.DataFrame) -> pd.Series | None:
        if t.time_col is not None:
            return pd.to_datetime(ent[t.time_col])
        if self.cfg.first_seen_via is not None:
            table, fk = self.cfg.first_seen_via
            link = db.table_dict[table]
            first = link.df.groupby(fk)[link.time_col].min()
            return first.reindex(ent.index)
        return None  # static: every entity always visible

    def _codebook(self, values: pd.Series) -> pd.CategoricalDtype:
        counts = values.dropna().astype(str).value_counts().rename("n").reset_index()
        counts = counts.sort_values(["n", counts.columns[0]], ascending=[False, True])
        top = counts.iloc[: self.cfg.max_categories, 0]
        return pd.CategoricalDtype(categories=sorted(top))

    def _embed(self, values: pd.Series) -> pd.DataFrame:
        from sklearn.decomposition import PCA

        texts = values.fillna("").astype(str)
        uniq, inv = np.unique(texts.to_numpy(), return_inverse=True)
        vecs = embed_texts(list(uniq), self.cfg.embed_model)
        pca = PCA(n_components=self.cfg.embed_dims, random_state=0)
        red = pca.fit_transform(vecs).astype(np.float32)  # one row per distinct text
        return pd.DataFrame(red[inv], index=values.index)

    def transform(self, seeds: pd.DataFrame) -> pd.DataFrame:
        """One row per row of ``seeds`` (same index), ``enc__`` columns only."""
        ids = seeds[self.entity_col].to_numpy()
        seed_t = pd.to_datetime(seeds[self.time_col]).to_numpy().astype("datetime64[ns]")
        out: dict[str, np.ndarray] = {}
        cfg = self.cfg
        if cfg.count:
            if self.visible_from is None:
                vis = np.ones(len(self.ids), bool)
                ev_t = np.full(len(self.ids), np.datetime64("1677-09-22", "ns"))
            else:
                vis = self.visible_from.notna().to_numpy()
                ev_t = self.visible_from.to_numpy().astype("datetime64[ns]")
            n_vis, _ = asof_group_sums(
                np.zeros(vis.sum(), np.int64),
                ev_t[vis],
                np.zeros(vis.sum()),
                np.zeros(len(ids), np.int64),
                seed_t,
                strict=not cfg._inclusive,
            )
            for c in [c for c in self.codes if c != self.key_name]:
                code = self.codes[c].to_numpy()
                q = self.codes[c].reindex(ids).fillna(-1).to_numpy(np.int64)
                cnt, _ = asof_group_sums(
                    code[vis], ev_t[vis], np.zeros(vis.sum()), q, seed_t, not cfg._inclusive
                )
                cnt = np.where(q < 0, np.nan, cnt)
                out[f"{PREFIX}{c}__count"] = cnt
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{PREFIX}{c}__share"] = cnt / n_vis
        if cfg.target:
            n_all, s_all = asof_group_sums(
                np.zeros(len(self.lab_y), np.int64),
                self.lab_known,
                self.lab_y,
                np.zeros(len(ids), np.int64),
                seed_t,
                strict=False,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                prior = s_all / n_all
            for c, codes in self.codes.items():
                ev_key = codes.reindex(self.lab_id).fillna(-1).to_numpy(np.int64)
                q = codes.reindex(ids).fillna(-1).to_numpy(np.int64)
                n, s = asof_group_sums(ev_key, self.lab_known, self.lab_y, q, seed_t, False)
                m = cfg.smoothing
                with np.errstate(invalid="ignore", divide="ignore"):
                    te = (s + m * prior) / (n + m)
                out[f"{PREFIX}{c}__te"] = np.where(q < 0, np.nan, te)
                out[f"{PREFIX}{c}__te_n"] = n
        frame = pd.DataFrame(out, index=seeds.index)
        for c, dtype in self._cat.items():
            vals = self.ent[c].reindex(ids)
            vals = vals.where(vals.isna(), vals.astype(str))
            vals = vals.where(vals.isin(dtype.categories))  # outside the codebook: missing
            frame[f"{PREFIX}{c}__cat"] = pd.Categorical(vals.to_numpy(), dtype=dtype)
        for c, emb in self._emb.items():
            e = emb.reindex(ids).to_numpy()
            for i in range(e.shape[1]):
                frame[f"{PREFIX}{c}__emb{i}"] = e[:, i]
        return frame


def embed_texts(texts: list[str], model: str = GLOVE) -> np.ndarray:
    """Sentence embeddings with ``sentence-transformers`` on cuda, mps or cpu.

    Runs in a child process (``python -m relconduit.entity_encoding``): on macOS, torch
    segfaults if LightGBM's OpenMP runtime was loaded first, as it is when this is called from
    :mod:`relconduit.entity_lgbm`.
    """
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "texts.json", Path(d) / "vecs.npy"
        src.write_text(json.dumps(texts))
        cmd = [sys.executable, "-m", "relconduit.entity_encoding", model, str(src), str(dst)]
        subprocess.run(cmd, check=True)
        return np.load(dst)


def _embed_worker(texts: list[str], model: str, batch_size: int = 4096) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    st = SentenceTransformer(model, device=device)
    return np.asarray(st.encode(texts, batch_size=batch_size, convert_to_numpy=True))


if __name__ == "__main__":
    import sys

    _model, _src, _dst = sys.argv[1:4]
    with open(_src) as f:
        np.save(_dst, _embed_worker(json.load(f), _model))
