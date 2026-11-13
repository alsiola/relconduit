"""Time-safe candidates and pair features for RelBench recommendation (link) tasks.

A recommendation task asks, for each seed row ``(src entity, seed time T)``, for the ``eval_k``
destination entities most likely to be linked to it in ``[T, T + timedelta)``. This module
does the first half of a two-stage recommender: it proposes candidate destinations per seed and
describes each ``(seed, candidate)`` pair with features. A ranker (:mod:`relconduit.link_lgbm`)
does the second half.

**Everything is read from database rows dated strictly before the seed time.** Nothing reads
the task's label tables, whose rows are stamped with the start of a label window that runs
past it. :func:`pair_leakage_mismatches` checks the claim, the same way
:func:`relconduit.features.leakage_mismatches` does for entity features: it compares the output
on the full database with the output on ``truncate_before(db, T)``, per seed time.

The parts:

* **Edge sets.** A named function turns the database into a frame of time-stamped edges
  ``(src, dst, t)`` between any two kinds of node (a user and a post, a post and a tag, a
  facility and a country). :func:`table_edges`, :func:`join_edges`, :func:`reverse_edges`,
  :func:`concat_edges` and :func:`token_edges` build them. An edge's time is the latest time
  of the rows it was made from (static rows count as :data:`NEVER`, known from the start), so a
  joined edge exists before ``T`` exactly when every row it came from does.
* **Direct relations** (:attr:`LinkConfig.direct`): edge sets from the task's src entity to
  its dst entity, such as the src's own past targets. Per pair: count, days since the last and
  the first edge. Each one also proposes the src's top ``n_direct`` dsts by count, then
  recency.
* **Paths** (:attr:`LinkConfig.paths`): random walks over a sequence of edge sets, e.g.
  ``user -> ad -> user -> ad`` (co-visitation) or ``post -> tag -> link target``. Each step's
  edge counts (optionally decayed by age) are row-normalised, so a pair's score is the walk's
  probability of ending at the dst. Per pair: score and rank. Each path proposes its top ``n``.
  Intermediate nodes keep their ``max_fanout`` heaviest out-edges and each partial walk its
  ``beam`` heaviest nodes, which bounds the cost on hub-heavy graphs.
* **Popularity** (:attr:`LinkConfig.popularity`): edge sets ending at the dst entity. Per
  candidate: its edge count in each of :attr:`LinkConfig.pop_windows` days before ``T``, all
  time, and days since its last edge. :attr:`LinkConfig.pop_candidates` also proposes the
  global top ``n`` of an edge set over a recent window, the same for every seed at ``T``.

The candidate set of a seed is the union of every source's proposals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from relbench.base import Database

from relconduit.features import truncate_before

SRC, DST, T = "src", "dst", "t"
SEED = "seed"  # positional index of the seed row a pair belongs to
NEVER = pd.Timestamp("1900-01-01")  # the time of an edge made only from static rows
DAY = pd.Timedelta(days=1)
_SOURCE = "__source"

EdgeFn = Callable[[Database], pd.DataFrame]


# --- edge sets ---------------------------------------------------------------------------


def _frame(src, dst, t) -> pd.DataFrame:
    df = pd.DataFrame({SRC: src, DST: dst, T: t})
    df = df.dropna(subset=[SRC, DST, T])
    for c in (SRC, DST):
        if pd.api.types.is_numeric_dtype(df[c]) or isinstance(df[c].dtype, pd.Int64Dtype):
            df[c] = df[c].astype(np.int64)
    df[T] = pd.to_datetime(df[T]).astype("datetime64[ns]")
    return df.reset_index(drop=True)


def table_edges(db: Database, table: str, src_col: str, dst_col: str) -> pd.DataFrame:
    """One edge per row of ``table`` with both columns set, at the row's time.

    Either column may be the primary key (e.g. ``posts``: ``OwnerUserId -> Id``). Rows of a
    static table are dated :data:`NEVER`; rows of a time-stamped table with no time are left
    out.
    """
    t = db.table_dict[table]
    df = t.df
    time = df[t.time_col] if t.time_col is not None else NEVER
    return _frame(df[src_col].to_numpy(), df[dst_col].to_numpy(), time)


def token_edges(
    db: Database, table: str, id_col: str, col: str, split: Callable[[str], list[str]]
) -> pd.DataFrame:
    """Edges from each row's ``id_col`` to each token ``split`` finds in its ``col`` text.

    For example a post's tags: ``split=lambda s: s.strip("<>").split("><")``.
    """
    t = db.table_dict[table]
    df = t.df[[id_col, col] + ([t.time_col] if t.time_col else [])].dropna(subset=[col])
    tokens = df[col].astype(str).map(split)
    ex = df.assign(**{col: tokens}).explode(col)
    ex = ex[ex[col].notna() & (ex[col] != "")]
    time = ex[t.time_col] if t.time_col else NEVER
    return _frame(ex[id_col].to_numpy(), ex[col].astype(str).to_numpy(), time)


def join_edges(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Edges ``a.src -> b.dst`` through ``a.dst == b.src``, at the later of the two times."""
    m = a.merge(b, left_on=DST, right_on=SRC, suffixes=("_a", "_b"))
    t = np.maximum(m[T + "_a"].to_numpy(), m[T + "_b"].to_numpy())
    return _frame(m[SRC + "_a"].to_numpy(), m[DST + "_b"].to_numpy(), t)


def reverse_edges(e: pd.DataFrame) -> pd.DataFrame:
    return _frame(e[DST].to_numpy(), e[SRC].to_numpy(), e[T].to_numpy())


def concat_edges(*es: pd.DataFrame) -> pd.DataFrame:
    return _frame(
        np.concatenate([e[SRC].to_numpy() for e in es]),
        np.concatenate([e[DST].to_numpy() for e in es]),
        np.concatenate([e[T].to_numpy() for e in es]),
    )


# --- config ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Path:
    """A random walk from the seed's src over named edge sets, ending at a dst.

    A step named ``"~name"`` walks the edge set backwards. With ``window_days``, only edges in
    the last that many days before the seed time are walked (a tuple gives one window per
    step, ``None`` for none); with ``half_life_days``, each edge counts
    ``0.5 ** (age / half_life)`` instead of 1 (edges from static rows always count 1).
    """

    steps: tuple[str, ...]
    n: int = 100
    window_days: float | tuple[float | None, ...] | None = None
    half_life_days: float | None = None
    max_fanout: int = 1000
    beam: int = 2000


@dataclass
class LinkConfig:
    """How candidates and pair features are made for one task (see the module doc)."""

    edges: dict[str, EdgeFn]
    direct: Sequence[str] = ()
    n_direct: int = 100
    paths: dict[str, Path] = field(default_factory=dict)
    popularity: Sequence[str] = ()
    pop_windows: Sequence[int] = (7, 30, 91, 365)
    # edge set -> (window days, n): the global top n dsts by edge count over the window.
    pop_candidates: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Testing only: keep edges dated exactly at the seed time (``<=``), a planted leak.
    _inclusive: bool = field(default=False, repr=False)


# --- building ----------------------------------------------------------------------------


def _top_per_group(df: pd.DataFrame, group: str, by: list[str], asc: list[bool], n: int):
    df = df.sort_values([group, *by], ascending=[True, *asc], kind="mergesort")
    return df[df.groupby(group, sort=False).cumcount() < n]


def _row_normalise(m: sp.csr_matrix) -> sp.csr_matrix:
    s = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, s, out=np.zeros_like(s, dtype=float), where=s > 0)
    return sp.diags(inv) @ m


def _keep_top_per_row(m: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Each row's ``k`` largest entries (ties: lower column, i.e. lower node value, first)."""
    m = m.tocsr()
    m.sum_duplicates()
    lengths = np.diff(m.indptr)
    if lengths.max(initial=0) <= k:
        return m
    rows = np.repeat(np.arange(m.shape[0]), lengths)
    order = np.lexsort((m.indices, -np.round(m.data, 12), rows))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order)) - np.repeat(m.indptr[:-1], lengths)[order]
    keep = rank < k
    return sp.csr_matrix((m.data[keep], (rows[keep], m.indices[keep])), shape=m.shape).tocsr()


def _node_index(arrays) -> pd.Index:
    """The sorted distinct nodes of ``arrays`` (sorted, so ties break by node value whichever
    rows the database holds). Integer nodes stay integers, which is much faster on big edge
    sets; anything else (strings, mixed) is compared as objects."""
    arrays = [np.asarray(x) for x in arrays]
    if all(np.issubdtype(x.dtype, np.integer) for x in arrays):
        return pd.Index(np.unique(np.concatenate(arrays).astype(np.int64)))
    vals = pd.unique(np.concatenate([x.astype(object) for x in arrays]))
    return pd.Index(np.sort(vals))


def _positions(index: pd.Index, values) -> np.ndarray:
    """Position of each value in ``index``; -1 where absent."""
    values = np.asarray(values)
    if index.dtype == np.int64 and np.issubdtype(values.dtype, np.integer):
        return index.get_indexer(values.astype(np.int64))
    return index.get_indexer(values.astype(object))


class PairBuilder:
    """Candidates and pair features for seeds on one database (see the module doc)."""

    def __init__(self, db: Database, config: LinkConfig):
        self.cfg = config
        self.edges = {name: fn(db) for name, fn in config.edges.items()}
        # Work that depends only on the seed time (filtered edge sets, walk matrices,
        # popularity), kept for the last seed time seen, so one seed time built in several
        # chunks of seeds does it once.
        self._cache_t: pd.Timestamp | None = None
        self._cache: dict = {}

    def _at(self, t: pd.Timestamp, key, make):
        if self._cache_t != t:
            self._cache_t, self._cache = t, {}
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def source_names(self) -> list[str]:
        """The candidate sources, in the order of their ``cand__<source>`` columns."""
        cfg = self.cfg
        pop = [f"{n}__top{w}d" for n, (w, _) in cfg.pop_candidates.items()]
        return [*cfg.direct, *cfg.paths, *pop]

    def _before(self, name: str, t: pd.Timestamp, window_days: float | None = None):
        return self._at(
            t, ("before", name, window_days), lambda: self._filter(name, t, window_days)
        )

    def _filter(self, name: str, t: pd.Timestamp, window_days: float | None):
        e = self.edges[name]
        mask = (e[T] <= t) if self.cfg._inclusive else (e[T] < t)
        if window_days is not None:  # static rows (NEVER) are always in the window
            mask &= (e[T] >= t - pd.Timedelta(days=window_days)) | (e[T] == NEVER)
        return e[mask]

    def build(self, seeds: pd.DataFrame, src_col: str, time_col: str) -> pd.DataFrame:
        """One row per ``(seed, candidate dst)``: ``seed`` (position in ``seeds``), ``dst`` and
        the pair features, ordered by seed then dst. Every source's proposals are included.

        Each seed's pairs depend only on its own src and seed time (and the database), so
        building ``seeds`` in chunks and concatenating gives the same rows."""
        parts = []
        times = pd.to_datetime(seeds[time_col]).to_numpy()
        srcs = seeds[src_col].to_numpy().astype(np.int64)
        for t in np.unique(times):
            idx = np.flatnonzero(times == t)
            parts.append(self._build_at(pd.Timestamp(t), idx, srcs[idx]))
        if not parts:
            return pd.DataFrame({SEED: [], DST: []})
        out = pd.concat(parts, ignore_index=True)
        return out.sort_values([SEED, DST], kind="mergesort").reset_index(drop=True)

    def _build_at(self, t: pd.Timestamp, idx: np.ndarray, srcs: np.ndarray) -> pd.DataFrame:
        cfg = self.cfg
        seed_src = pd.DataFrame({SEED: idx, SRC: srcs})
        cands = []  # frames of (seed, dst)
        direct_feats = {}
        for name in cfg.direct:
            e = self._before(name, t)
            e = e[e[SRC].isin(np.unique(srcs))]
            g = e.groupby([SRC, DST], sort=False)[T].agg(["size", "max", "min"]).reset_index()
            g = g.rename(columns={"size": "n", "max": "last", "min": "first"})
            f = pd.DataFrame(
                {
                    SRC: g[SRC],
                    DST: g[DST],
                    f"{name}__count": g["n"].astype(float),
                    f"{name}__days_since_last": (t - g["last"]) / DAY,
                    f"{name}__days_since_first": (t - g["first"]) / DAY,
                }
            )
            direct_feats[name] = f
            top = _top_per_group(
                f,
                SRC,
                [f"{name}__count", f"{name}__days_since_last", DST],
                [False, True, True],
                cfg.n_direct,
            )
            cands.append((name, seed_src.merge(top[[SRC, DST]], on=SRC)[[SEED, DST]]))

        path_scores = {}
        for name, path in cfg.paths.items():
            scores, dst_values = self._walk(path, t, srcs)
            path_scores[name] = (scores, dst_values)
            top = _keep_top_per_row(scores, path.n).tocoo()
            cands.append((name, pd.DataFrame({SEED: idx[top.row], DST: dst_values[top.col]})))

        pop_counts = {
            name: self._at(t, ("pop", name), lambda n=name: self._pop(n, t))
            for name in cfg.popularity
        }
        for name, (w, n) in cfg.pop_candidates.items():
            vc = self._at(
                t, ("popcand", name, w, n), lambda a=name, b=w, c=n: self._top(a, t, b, c)
            )
            top = pd.DataFrame(
                {SEED: np.repeat(idx, len(vc)), DST: np.tile(vc[DST].to_numpy(), len(idx))}
            )
            cands.append((f"{name}__top{w}d", top))

        # Which sources proposed each pair: a 0/1 column per source.
        tagged = pd.concat(
            [c.assign(**{_SOURCE: name}) for name, c in cands], ignore_index=True
        ).drop_duplicates()
        tagged[DST] = tagged[DST].astype(np.int64)
        pairs = tagged.assign(_one=1.0).pivot_table(
            index=[SEED, DST], columns=_SOURCE, values="_one", aggfunc="max", fill_value=0.0
        )
        pairs.columns = [f"cand__{c}" for c in pairs.columns]
        pairs = pairs.reindex(columns=[f"cand__{n}" for n in self.source_names()], fill_value=0.0)
        pairs = pairs.reset_index().merge(seed_src, on=SEED)
        for name, f in direct_feats.items():
            pairs = pairs.merge(f, on=[SRC, DST], how="left")
            pairs[f"{name}__count"] = pairs[f"{name}__count"].fillna(0.0)
        pos = pd.Series(np.arange(len(idx)), index=idx)
        for name, (scores, dst_values) in path_scores.items():
            col_of = pd.Series(np.arange(len(dst_values)), index=dst_values)
            rows = pos.loc[pairs[SEED]].to_numpy()
            cols = col_of.reindex(pairs[DST]).to_numpy()
            ok = ~np.isnan(cols)
            vals = np.zeros(len(pairs))
            if ok.any():
                vals[ok] = np.asarray(scores[rows[ok], cols[ok].astype(np.int64)]).ravel()
            pairs[f"{name}__score"] = np.round(vals, 12)
            rank = pairs.groupby(SEED)[f"{name}__score"].rank(ascending=False, method="min")
            pairs[f"{name}__rank"] = rank.where(pairs[f"{name}__score"] > 0)
        for name, pc in pop_counts.items():
            pairs = pairs.merge(pc, left_on=DST, right_index=True, how="left")
            count_cols = [c for c in pc.columns if "__pop_days" not in c]
            pairs[count_cols] = pairs[count_cols].fillna(0.0)
        pairs["n_candidates"] = pairs.groupby(SEED)[DST].transform("size").astype(float)
        return pairs.drop(columns=[SRC])

    def _pop(self, name: str, t: pd.Timestamp) -> pd.DataFrame:
        e = self._before(name, t)
        cols = {}
        for w in self.cfg.pop_windows:
            recent = e[DST][e[T] >= t - pd.Timedelta(days=w)]
            cols[f"{name}__pop_{w}d"] = recent.value_counts()
        cols[f"{name}__pop_all"] = e[DST].value_counts()
        cols[f"{name}__pop_days_since_last"] = (t - e.groupby(DST)[T].max()) / DAY
        return pd.DataFrame(cols)

    def _top(self, name: str, t: pd.Timestamp, w: int, n: int) -> pd.DataFrame:
        e = self._before(name, t, window_days=w)
        vc = e.groupby(DST).size().rename("n").reset_index()
        return vc.sort_values(["n", DST], ascending=[False, True], kind="mergesort").head(n)

    def _walk(self, path: Path, t: pd.Timestamp, srcs: np.ndarray):
        """Walk scores, a sparse ``(len(srcs), n dst nodes)`` matrix, and the dst node values."""
        index, mats, dst_values = self._at(t, ("walk", path), lambda: self._walk_mats(path, t))
        rows = _positions(index[0], srcs)
        ok = rows >= 0  # a src with no out-edge in step 0 walks nowhere
        m = sp.csr_matrix(
            (np.ones(ok.sum()), (np.flatnonzero(ok), rows[ok])),
            shape=(len(srcs), len(index[0])),
        )
        for mat in mats:
            m = _keep_top_per_row(m @ mat, path.beam)
        return m.tocsr(), dst_values

    def _walk_mats(self, path: Path, t: pd.Timestamp):
        """Per seed time: each node space's index, and each step's row-normalised matrix."""
        steps = []
        windows = path.window_days
        if not isinstance(windows, tuple):
            windows = (windows,) * len(path.steps)
        for step, window in zip(path.steps, windows, strict=True):
            name = step.lstrip("~")
            e = self._before(name, t, window_days=window)
            a, b = (e[DST], e[SRC]) if step.startswith("~") else (e[SRC], e[DST])
            if path.half_life_days:
                # Static rows (dated NEVER) have no age: they count 1.
                age = ((t - e[T]) / DAY).to_numpy()
                w = np.where(e[T].to_numpy() == NEVER, 1.0, 0.5 ** (age / path.half_life_days))
            else:
                w = np.ones(len(e))
            steps.append((a.to_numpy(), b.to_numpy(), w))
        # One node index per position along the walk: the left ends of step 0, then each
        # step's right ends together with the next step's left ends.
        spaces = []
        left = [steps[0][0]]
        for i in range(len(steps)):
            right = [steps[i][1]] + ([steps[i + 1][0]] if i + 1 < len(steps) else [])
            spaces.append(left)
            left = right
        spaces.append(left)
        index = [_node_index(arrays) for arrays in spaces]
        mats = []
        for i, (a, b, w) in enumerate(steps):
            mat = sp.csr_matrix(
                (
                    w,
                    (
                        _positions(index[i], a),
                        _positions(index[i + 1], b),
                    ),
                ),
                shape=(len(index[i]), len(index[i + 1])),
            )
            mat.sum_duplicates()
            if i > 0:
                mat = _keep_top_per_row(mat, path.max_fanout)
            mats.append(_row_normalise(mat).tocsr())
        dst_values = np.asarray(index[-1], dtype=object)
        try:
            dst_values = dst_values.astype(np.int64)
        except (TypeError, ValueError):
            pass
        return index, mats, dst_values


def build_pairs(
    db: Database, seeds: pd.DataFrame, src_col: str, time_col: str, config: LinkConfig
) -> pd.DataFrame:
    """:meth:`PairBuilder.build` in one call."""
    return PairBuilder(db, config).build(seeds, src_col, time_col)


def label_pairs(pairs: pd.DataFrame, seeds: pd.DataFrame, dst_col: str) -> np.ndarray:
    """1 where a pair's dst is in its seed row's ``dst_col`` list, else 0."""
    true = seeds[dst_col].reset_index(drop=True)
    ex = true.explode().dropna()
    pos = set(zip(ex.index.to_numpy(), ex.to_numpy().astype(np.int64)))
    return np.fromiter(
        ((s, d) in pos for s, d in zip(pairs[SEED].to_numpy(), pairs[DST].to_numpy())),
        dtype=np.int8,
        count=len(pairs),
    )


def candidate_recall(pairs: pd.DataFrame, seeds: pd.DataFrame, dst_col: str) -> float:
    """Mean over seed rows of the share of its true dsts that are among its candidates."""
    y = label_pairs(pairs, seeds, dst_col)
    hits = pd.Series(y, index=pairs[SEED].to_numpy()).groupby(level=0).sum()
    n_true = seeds[dst_col].reset_index(drop=True).map(len)
    hits = hits.reindex(range(len(seeds)), fill_value=0).to_numpy()
    return float(np.mean(np.where(n_true > 0, hits / n_true.clip(lower=1), np.nan)))


def pair_leakage_mismatches(
    db: Database,
    seeds: pd.DataFrame,
    src_col: str,
    time_col: str,
    config: LinkConfig,
    atol: float = 1e-9,
) -> list[tuple]:
    """Compare pairs built on ``db`` with pairs built on ``truncate_before(db, T)``, per seed
    time ``T``. Returns every difference: ``(seed, dst, column, full, truncated)``, with column
    ``"<candidate>"`` for a pair present on one side only. Empty means no leak was found."""
    full = build_pairs(db, seeds, src_col, time_col, config)
    bad = []
    times = pd.to_datetime(seeds[time_col])
    for t in times.unique():
        idx = np.flatnonzero((times == t).to_numpy())
        sub = seeds.iloc[idx]
        trunc = build_pairs(truncate_before(db, pd.Timestamp(t)), sub, src_col, time_col, config)
        trunc[SEED] = idx[trunc[SEED].to_numpy()]
        a = full[full[SEED].isin(idx)].set_index([SEED, DST])
        b = trunc.set_index([SEED, DST])
        for key in a.index.symmetric_difference(b.index):
            bad.append((*key, "<candidate>", key in a.index, key in b.index))
        common = a.index.intersection(b.index)
        cols = sorted(set(a.columns) | set(b.columns))
        for col in cols:
            x = a[col].reindex(common).to_numpy(dtype=float) if col in a else None
            y = b[col].reindex(common).to_numpy(dtype=float) if col in b else None
            if x is None or y is None:
                bad.append((None, None, col, x is not None, y is not None))
                continue
            same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, atol=atol, rtol=0)
            for k in np.flatnonzero(~same):
                bad.append((*common[k], col, x[k], y[k]))
    return bad
