"""Time-safe dense text retrieval over rel-stack posts: a candidate source and a pair feature.

**What is embedded.** The ``posts`` table holds each post's text as of the 2023 dump, after
every later edit (see :mod:`relconduit.stack_text`), so it is not used. Instead each post is
embedded from its *initial* revision in ``postHistory``: the first title (``PostHistoryTypeId``
1) and the first body (2). An answer has no title; it gets its question's initial title. A
post's embedding is dated by the latest of the revision rows it was made from
(:func:`initial_post_texts`), so it exists before a seed time ``T`` exactly when every row
behind it is dated strictly before ``T``. Edits after the first revision are ignored, which
costs a little signal and keeps one vector per post.

**What is searched.** :class:`DenseIndex` holds the vectors (L2-normalised, so a dot product is
the cosine), their post ids and their times. For a seed ``(post, T)`` it searches only vectors
dated strictly before ``T``, and the query is the seed post's own vector, which must be dated
before ``T`` too (else the seed gets no dense candidates and NaN scores).

**Wiring.** A :class:`DenseSource` in :attr:`relconduit.link.LinkConfig.dense` adds, per seed:
the ``n`` nearest posts as candidates (``cand__<name>``), and for every pair of the seed's
candidate set the cosine (``<name>__score``) and its rank among the seed's neighbours
(``<name>__rank``, NaN beyond ``n``). Its ``available`` function reads, from the database the
pairs are built on, which posts have their text rows yet and when; so
:func:`relconduit.link.pair_leakage_mismatches` (full vs ``truncate_before(db, T)``) checks the
index's time filter like any edge set, and a planted ``<=`` (``LinkConfig._inclusive``) is
caught by ``tests/test_post_embed.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from relbench.base import Database

TITLE, BODY = 1, 2
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean(text: str, max_chars: int = 2000) -> str:
    """HTML tags and runs of whitespace out, cut to ``max_chars``."""
    return _WS.sub(" ", _TAG.sub(" ", text)).strip()[:max_chars]


def initial_post_texts(
    post_history: pd.DataFrame, posts: pd.DataFrame, max_chars: int = 2000
) -> pd.DataFrame:
    """One row per post with an initial title or body: ``Id``, ``t``, ``text``.

    ``text`` is ``title + "\\n" + body`` (an answer's title is its question's). ``t`` is the
    latest time of the revision rows used, so the text is known strictly before ``T`` exactly
    when ``t < T``. ``post_history`` needs ``Id``, ``PostId``, ``PostHistoryTypeId``, ``Text``,
    ``CreationDate``; ``posts`` needs ``Id``, ``PostTypeId``, ``ParentId``.
    """
    ph = post_history[post_history["PostHistoryTypeId"].isin([TITLE, BODY])]
    ph = ph.dropna(subset=["Text", "PostId"]).sort_values(["PostId", "CreationDate", "Id"])
    first = ph.groupby(["PostId", "PostHistoryTypeId"], sort=False).head(1)
    title = first[first["PostHistoryTypeId"] == TITLE].set_index("PostId")
    body = first[first["PostHistoryTypeId"] == BODY].set_index("PostId")

    p = posts[["Id", "PostTypeId", "ParentId"]].copy()
    p["Id"] = p["Id"].astype(np.int64)
    # Whose title a post gets: its own, or an answer's question's.
    is_answer = (p["PostTypeId"] == 2) & p["ParentId"].notna()
    p["title_of"] = np.where(is_answer, p["ParentId"].fillna(-1), p["Id"]).astype(np.int64)
    p = p[p["Id"].isin(body.index) | p["title_of"].isin(title.index)]
    t_title = p["title_of"].map(title["CreationDate"])
    t_body = p["Id"].map(body["CreationDate"])
    text_title = p["title_of"].map(title["Text"]).fillna("")
    text_body = p["Id"].map(body["Text"]).fillna("")
    t = pd.concat([t_title, t_body], axis=1).max(axis=1)
    text = (text_title.astype(str) + "\n" + text_body.astype(str).map(clean)).str.strip()
    out = pd.DataFrame({"Id": p["Id"].to_numpy(), "t": t.to_numpy(), "text": text.to_numpy()})
    out["text"] = out["text"].str.slice(0, max_chars)
    out = out[out["text"].str.len() > 0].dropna(subset=["t"])
    out["t"] = pd.to_datetime(out["t"]).astype("datetime64[ns]")
    return out.sort_values("Id", kind="mergesort").reset_index(drop=True)


def text_times(db: Database) -> pd.DataFrame:
    """``Id``, ``t`` of :func:`initial_post_texts` on ``db``, without the text: when each
    post's embedded text became known, by the rows this database holds."""
    ph = (
        db.table_dict["postHistory"]
        .df[["Id", "PostId", "PostHistoryTypeId", "CreationDate"]]
        .assign(Text="x")
    )
    return initial_post_texts(ph, db.table_dict["posts"].df)[["Id", "t"]]


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: str | None = None,
    max_seq_length: int = 256,
    progress: bool = False,
) -> np.ndarray:
    """L2-normalised ``float32`` sentence embeddings (``cuda``, then ``mps``, then ``cpu``)."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    # Longest first keeps padding low; the order is restored after.
    order = np.argsort([-len(s) for s in texts], kind="stable")
    emb = model.encode(
        [texts[i] for i in order],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=progress,
    ).astype(np.float32)
    out = np.empty_like(emb)
    out[order] = emb
    return out


@dataclass
class DenseIndex:
    """Post vectors (unit length), their ids and the times their text became known."""

    ids: np.ndarray
    times: np.ndarray
    emb: np.ndarray

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=np.int64)
        self.times = np.asarray(self.times, dtype="datetime64[ns]")
        self.emb = np.ascontiguousarray(self.emb, dtype=np.float32)
        self._pos = pd.Series(np.arange(len(self.ids)), index=self.ids)

    def positions(self, ids) -> np.ndarray:
        """Row of each id in the index; -1 where absent."""
        return self._pos.reindex(np.asarray(ids, dtype=np.int64)).fillna(-1).to_numpy(np.int64)

    def restrict(self, available: pd.DataFrame) -> DenseIndex:
        """Only posts in ``available`` (``Id``, ``t``), each dated the later of the two times."""
        a = available.set_index("Id")["t"]
        keep = np.isin(self.ids, a.index.to_numpy())
        t = np.maximum(self.times[keep], a.reindex(self.ids[keep]).to_numpy("datetime64[ns]"))
        return DenseIndex(self.ids[keep], t, self.emb[keep])

    def usable(self, t: pd.Timestamp, inclusive: bool = False) -> np.ndarray:
        t = np.datetime64(pd.Timestamp(t), "ns")
        return self.times <= t if inclusive else self.times < t

    def search(
        self,
        srcs: np.ndarray,
        t: pd.Timestamp,
        n: int,
        inclusive: bool = False,
        block: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ``n`` nearest posts known before ``t`` to each src (not itself).

        Returns ``(row, dst, score)``: ``row`` indexes ``srcs``. A src whose own vector is not
        known before ``t`` gets nothing. Ties break by lower post id."""
        ok = self.usable(t, inclusive)
        cand = np.flatnonzero(ok)  # sorted by id, since ids are
        qpos = self.positions(srcs)
        has = (qpos >= 0) & np.where(qpos >= 0, ok[np.maximum(qpos, 0)], False)
        rows, dsts, scores = [], [], []
        if len(cand) == 0:
            return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
        E = self.emb[cand]
        q_rows = np.flatnonzero(has)
        m = min(n + 1, len(cand))
        for a in range(0, len(q_rows), block):
            r = q_rows[a : a + block]
            S = self.emb[qpos[r]] @ E.T
            # Exclude the src itself.
            self_col = np.searchsorted(self.ids[cand], self.ids[qpos[r]])
            S[np.arange(len(r)), self_col] = -np.inf
            S = np.round(S, 6)
            top = np.argpartition(-S, m - 1, axis=1)[:, :m]
            for i, row in enumerate(r):
                cols = top[i]
                s = S[i, cols]
                # Order: score desc, then lower id (cand is id-sorted, so lower column).
                o = np.lexsort((cols, -s))
                cols, s = cols[o], s[o]
                # argpartition may cut a tie at the boundary arbitrarily; recompute ties.
                kth = s[min(n, len(s)) - 1] if len(s) else -np.inf
                full = np.flatnonzero(S[i] >= kth)
                if len(full) > len(cols):
                    fs = S[i, full]
                    o = np.lexsort((full, -fs))
                    cols, s = full[o], fs[o]
                keep = np.isfinite(s)
                cols, s = cols[keep][:n], s[keep][:n]
                rows.append(np.full(len(cols), row, np.int64))
                dsts.append(self.ids[cand[cols]])
                scores.append(s.astype(np.float32))
        if not rows:
            return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
        return np.concatenate(rows), np.concatenate(dsts), np.concatenate(scores)

    def cosine(
        self, src: np.ndarray, dst: np.ndarray, t: np.ndarray, inclusive: bool = False
    ) -> np.ndarray:
        """Cosine of each ``(src, dst)`` pair at its time; NaN unless both known before it."""
        ps, pd_ = self.positions(src), self.positions(dst)
        t = np.asarray(t, dtype="datetime64[ns]")
        out = np.full(len(ps), np.nan, dtype=np.float64)
        ok = (ps >= 0) & (pd_ >= 0)
        if inclusive:
            ok[ok] &= (self.times[ps[ok]] <= t[ok]) & (self.times[pd_[ok]] <= t[ok])
        else:
            ok[ok] &= (self.times[ps[ok]] < t[ok]) & (self.times[pd_[ok]] < t[ok])
        out[ok] = np.einsum("ij,ij->i", self.emb[ps[ok]], self.emb[pd_[ok]])
        return np.round(out, 6)


@dataclass
class DenseSource:
    """A :class:`DenseIndex` as a candidate source and pair feature (see the module doc).

    ``available(db)`` gives ``Id``, ``t`` of the posts whose text rows ``db`` holds; the index
    is cut to them, so a truncated database sees a truncated index."""

    index: DenseIndex
    n: int = 100
    available: Callable[[Database], pd.DataFrame] | None = field(default=text_times)
