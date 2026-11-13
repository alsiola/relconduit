"""Time-safe relational features for RelBench entity tasks, computed with DuckDB.

Given a RelBench :class:`~relbench.base.Database` and a seed table (one row per
``(entity id, seed time)``), :func:`build_features` returns one feature row per seed row,
aggregated over the database's foreign-key paths. **Every row a feature reads has a
timestamp strictly before that seed row's seed time.** :func:`truncate_before` and
:func:`leakage_mismatches` exist to check that claim.

Paths, with ``E`` the entity table:

* ``E`` itself: its numeric columns, low-cardinality text columns as categoricals and its
  datetime columns as days before the seed time. If ``E`` has a time column, all of these are
  NaN unless the entity's own timestamp is strictly before the seed time, and its age
  (``<E>__<time col>__days_before``) is added under the same rule.
* ``E -> P``: the same columns of each parent ``P`` of ``E`` (e.g. an ad's category), under the
  same rule if ``P`` has a time column.
* hop 1, ``E <- T``: every time-stamped table ``T`` with a foreign key to ``E``. Aggregates of
  ``T``'s columns over ``T``'s rows for the entity, plus the numeric columns of ``T``'s other
  parents ``T -> P`` joined onto each row (hop 2 upward, e.g. a result row's race ``year``).
* hop 2, ``E <- T -> P <- T2`` for each *static* parent ``P`` (no time column, e.g. a
  constructor): take the ``P`` of the entity's most recent ``T`` row before the seed time
  (its current team) and aggregate ``T2``'s rows for that ``P``. Parents with a time column
  (e.g. races) are events, not memberships, so they are not walked this way.
* hop 2 over all memberships, ``E <- T -> P <- T2``, only for the parent tables named in
  :attr:`FeatureConfig.hop2_all_via`: aggregate ``T2``'s rows over *every* distinct ``P`` the
  entity is linked to by a ``T`` row before the seed time (``T`` may be static, e.g. a
  friendship table). Computed per distinct ``(P, seed time)`` and rolled up, which keeps
  fan-out manageable and gives exactly the counts, means, minima, maxima and null fractions of
  the pooled ``T2`` rows. Named ``<T>.<fk to P>>><T2>__...``.
* siblings, ``E -> P <- E' <- T``, for the foreign keys of ``E`` named in
  :attr:`FeatureConfig.siblings_via`: the same pooled aggregates over the ``T`` rows of every
  entity sharing the seed entity's parent (e.g. the click-through of all ads in its
  category). Named ``<E>.<fk>>><T>__...``.

Aggregates per numeric column, for each window (days before the seed time, ``None`` for all
time): ``count`` (one per path, not per column), ``mean``, ``min``, ``max`` (plus ``sum`` if
:attr:`FeatureConfig.aggs` asks for it) and, for float
columns (which can hold missing values, e.g. a finishing position that is NaN for a
retirement), ``nullfrac``. Over all time, also ``last`` (the value on the most recent row,
nulls kept) and per path ``days_since_last``. Text columns with at most
``MAX_COUNT_CATEGORIES`` values get a count per value and window; with at most
``MAX_CATEGORIES`` values, a ``last`` categorical.

Which features exist depends only on the schema, the dtypes and the vocabulary of text
columns (:func:`feature_vocab`, the distinct values in the database given). Pass a fixed
vocabulary in :attr:`FeatureConfig.vocab` to make the feature set independent of the rows;
:func:`leakage_mismatches` does, so truncating the database cannot change the feature set.

Primary- and foreign-key columns are never used as features.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import duckdb
import numpy as np
import pandas as pd
from relbench.base import Database, Table

DEFAULT_WINDOWS: tuple[int | None, ...] = (30, 90, 365, None)
WINDOW_AGGS = ("mean", "min", "max")
MAX_CATEGORIES = 100  # text columns with more distinct values than this are dropped
MAX_COUNT_CATEGORIES = 10  # text columns with at most this many values also get value counts

_SEED_IDX = "__seed_idx"
_SEED_ID = "__eid"
_SEED_T = "__t"
_EV_T = "__et"
_EV_ORDER = "__ord"
_ROW = "__row"  # tie-break column added to time-stamped tables that have no primary key

Vocab = dict[tuple[str, str], list[str]]


@dataclass
class FeatureConfig:
    """Which paths and aggregates :func:`build_features` computes."""

    windows: Sequence[int | None] = DEFAULT_WINDOWS
    # Per-window aggregates of each numeric column: any of "mean", "min", "max", "sum".
    aggs: Sequence[str] = WINDOW_AGGS
    # Text columns with at most this many values get a count per value and window.
    max_count_categories: int = MAX_COUNT_CATEGORIES
    # Hop 2 through static parents (E <- T -> P <- T2). Off gives hop 1 (+ upward joins) only.
    hop2: bool = True
    # Parent tables P to walk as E <- T -> P <- T2 over every linked P (see the module doc).
    hop2_all_via: Sequence[str] = ()
    # Foreign keys of E to walk as E -> P <- E' <- T (see :meth:`_Builder.siblings`).
    siblings_via: Sequence[str] = ()
    # Tables never read (whole tables, as path steps or targets).
    exclude_tables: Sequence[str] = ()
    # (table, column) pairs never used as feature values.
    exclude_columns: Sequence[tuple[str, str]] = ()
    # Text-column vocabulary; None computes it from the database passed to build_features.
    vocab: Vocab | None = field(default=None, repr=False)
    # Testing only: compare event times with ``<=`` instead of ``<``, which leaks rows stamped
    # exactly at the seed time. Exists so the leakage test can prove it catches a leak.
    _inclusive: bool = field(default=False, repr=False)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _window_name(w: int | None) -> str:
    return "all" if w is None else f"{w}d"


def _value_name(i: int, value: str) -> str:
    """A feature-name-safe label for a category value (LightGBM rejects JSON characters)."""
    return f"v{i}_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:24]


def _skip_cols(table: Table, name: str, cfg: FeatureConfig) -> set[str]:
    skip = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
    return skip | {c for t, c in cfg.exclude_columns if t == name}


def _numeric_cols(table: Table, name: str, cfg: FeatureConfig) -> list[str]:
    """Columns aggregated as values: numeric or bool, and not a key or the time column."""
    skip = _skip_cols(table, name, cfg)
    out = []
    for col in table.df.columns:
        if col in skip:
            continue
        dtype = table.df[col].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            out.append(col)
    return out


def _is_text(dtype) -> bool:
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _nullable(table: Table, col: str) -> bool:
    return pd.api.types.is_float_dtype(table.df[col].dtype)


def feature_vocab(db: Database, config: FeatureConfig | None = None) -> Vocab:
    """Sorted distinct values of every text column with at most ``MAX_CATEGORIES`` of them."""
    cfg = config or FeatureConfig()
    vocab: Vocab = {}
    for name, t in db.table_dict.items():
        if name in set(cfg.exclude_tables):
            continue
        skip = _skip_cols(t, name, cfg)
        for col in t.df.columns:
            if col in skip or not _is_text(t.df[col].dtype):
                continue
            vals = t.df[col].dropna()
            # Cheap rejection of free text: a sample already holding too many values.
            if vals.head(20_000).nunique() > MAX_CATEGORIES:
                continue
            uniq = pd.unique(vals.astype(str))
            if len(uniq) <= MAX_CATEGORIES:
                vocab[(name, col)] = sorted(uniq.tolist())
    return vocab


def truncate_before(db: Database, timestamp: pd.Timestamp) -> Database:
    """The database with every time-stamped row at or after ``timestamp`` removed.

    Unlike :meth:`Database.upto` (``<=``), this is strict: it keeps exactly the rows a feature
    for a seed at ``timestamp`` may read. Static tables are kept whole.
    """
    tables = {}
    for name, t in db.table_dict.items():
        df = t.df if t.time_col is None else t.df[t.df[t.time_col] < timestamp]
        tables[name] = Table(
            df=df,
            fkey_col_to_pkey_table=t.fkey_col_to_pkey_table,
            pkey_col=t.pkey_col,
            time_col=t.time_col,
        )
    return Database(tables)


class _Builder:
    def __init__(self, db: Database, entity_table: str, cfg: FeatureConfig, vocab: Vocab):
        self.db = db
        self.entity_table = entity_table
        self.cfg = cfg
        self.vocab = vocab
        self.lt = "<=" if cfg._inclusive else "<"
        self.con = duckdb.connect()
        # Several builds can share a machine: cap each one's DuckDB (defaults: all threads, 80%
        # of RAM) with RELCONDUIT_DUCKDB_THREADS / RELCONDUIT_DUCKDB_MEMORY (e.g. "12GB").
        if os.environ.get("RELCONDUIT_DUCKDB_THREADS"):
            self.con.execute(f"SET threads = {int(os.environ['RELCONDUIT_DUCKDB_THREADS'])}")
        if os.environ.get("RELCONDUIT_DUCKDB_MEMORY"):
            self.con.execute(f"SET memory_limit = {_lit(os.environ['RELCONDUIT_DUCKDB_MEMORY'])}")
        self.con.execute("SET preserve_insertion_order = false")
        self.tables = {n: t for n, t in db.table_dict.items() if n not in set(cfg.exclude_tables)}
        for name, t in self.tables.items():
            df = t.df
            if t.time_col is not None and t.pkey_col is None:
                # Row order survives truncation, so (time, row) is a stable "most recent" order.
                df = df.assign(**{_ROW: np.arange(len(df))})
            if isinstance(df.index, pd.RangeIndex) and df.index.start == 0:
                df = df.reset_index(drop=True)
            self.con.register(name, df)

    def close(self) -> None:
        self.con.close()

    # --- helpers ---------------------------------------------------------------------

    def _order_expr(self, alias: str, table: Table) -> str:
        """Tie-break key for "most recent row": time, then primary key (or row order)."""
        tie = table.pkey_col if table.pkey_col is not None else _ROW
        return f"({alias}.{_q(table.time_col)}, {alias}.{_q(tie)})"

    def _window_filter(self, w: int | None) -> str | None:
        if w is None:
            return None
        return f"{_EV_T} >= {_SEED_T} - INTERVAL {int(w)} DAY"

    @staticmethod
    def _when(cond: str | None, expr: str) -> str:
        """``expr`` where ``cond`` holds, else NULL. (DuckDB's ``FILTER (WHERE ...)`` is
        pathologically slow with hundreds of filtered aggregates; this is not.)"""
        return expr if cond is None else f"CASE WHEN {cond} THEN {expr} END"

    def _cat_cols(self, table: Table, name: str) -> list[str]:
        skip = _skip_cols(table, name, self.cfg)
        return [c for c in table.df.columns if c not in skip and (name, c) in self.vocab]

    def _agg_select(
        self,
        prefix: str,
        value_cols: list[tuple[str, str, bool]],
        cat_cols: Sequence[tuple[str, str, list[str]]] = (),
    ) -> list[str]:
        """Aggregate expressions over the ``ev`` relation.

        ``value_cols`` are ``(column in ev, feature-name stem, nullable)`` triples; ``cat_cols``
        are ``(column in ev, stem, vocabulary)``.
        """
        exprs = []
        for w in self.cfg.windows:
            wf = self._window_filter(w)
            wn = _window_name(w)
            exprs.append(f"count({self._when(wf, '1')}) AS {_q(f'{prefix}__count__{wn}')}")
            for col, stem, nullable in value_cols:
                c = _q(col)
                v = self._when(wf, f"CAST({c} AS DOUBLE)")
                for agg in self.cfg.aggs:
                    fn = {"mean": "avg", "min": "min", "max": "max", "sum": "sum"}[agg]
                    exprs.append(f"{fn}({v}) AS {_q(f'{prefix}__{stem}__{agg}__{wn}')}")
                if nullable:
                    isnull = self._when(wf, f"CAST({c} IS NULL AS DOUBLE)")
                    exprs.append(f"avg({isnull}) AS {_q(f'{prefix}__{stem}__nullfrac__{wn}')}")
            for col, stem, values in cat_cols:
                if len(values) > self.cfg.max_count_categories:
                    continue
                for i, val in enumerate(values):
                    cond = f"CAST({_q(col)} AS VARCHAR) = {_lit(val)}"
                    cond = f"{wf} AND {cond}" if wf else cond
                    name = f"{prefix}__{stem}={_value_name(i, val)}__count__{wn}"
                    exprs.append(f"count({self._when(cond, '1')}) AS {_q(name)}")
        for col, stem, _ in value_cols:
            exprs.append(
                f"arg_max_null(CAST({_q(col)} AS DOUBLE), {_EV_ORDER}) "
                f"AS {_q(f'{prefix}__{stem}__last')}"
            )
        for col, stem, _ in cat_cols:
            exprs.append(
                f"arg_max_null(CAST({_q(col)} AS VARCHAR), {_EV_ORDER}) "
                f"AS {_q(f'{prefix}__{stem}__last')}"
            )
        exprs.append(
            f"(epoch({_SEED_T}) - epoch(max({_EV_T}))) / 86400.0 "
            f"AS {_q(f'{prefix}__days_since_last')}"
        )
        return exprs

    def _run(self, sql: str) -> pd.DataFrame:
        return self.con.sql(sql).df().set_index(_SEED_IDX)

    def _cat_spec(self, table: Table, name: str, stem_prefix: str = ""):
        return [(c, stem_prefix + c, self.vocab[(name, c)]) for c in self._cat_cols(table, name)]

    # --- paths -----------------------------------------------------------------------

    def child_paths(self, timed: bool = True) -> list[tuple[str, str]]:
        """``(T, fkey col)`` for every table with a foreign key to E (time-stamped only by
        default)."""
        out = []
        for name, t in self.tables.items():
            if timed and t.time_col is None:
                continue
            for fk, pk_table in t.fkey_col_to_pkey_table.items():
                if pk_table == self.entity_table:
                    out.append((name, fk))
        return out

    def hop1(self, child: str, fk: str) -> pd.DataFrame:
        t = self.tables[child]
        prefix = child if self._n_fks(child) == 1 else f"{child}.{fk}"
        cols = [(c, c, _nullable(t, c)) for c in _numeric_cols(t, child, self.cfg)]
        cats = self._cat_spec(t, child)
        selects = [f"e.{_q(c)} AS {_q(c)}" for c, _, _ in cols]
        selects += [f"e.{_q(c)} AS {_q(c)}" for c, _, _ in cats]
        joins = []
        # Upward joins T -> P: P's numeric columns onto each T row.
        for i, (pfk, ptable) in enumerate(t.fkey_col_to_pkey_table.items()):
            if ptable == self.entity_table or ptable not in self.tables:
                continue
            p = self.tables[ptable]
            if p.pkey_col is None:
                continue
            pcols = _numeric_cols(p, ptable, self.cfg)
            if not pcols:
                continue
            alias = f"p{i}"
            cond = f"e.{_q(pfk)} = {alias}.{_q(p.pkey_col)}"
            if p.time_col is not None:
                cond += f" AND {alias}.{_q(p.time_col)} {self.lt} s.{_SEED_T}"
            joins.append(f"LEFT JOIN {_q(ptable)} {alias} ON {cond}")
            for pc in pcols:
                name = f"{ptable}.{pc}"
                selects.append(f"{alias}.{_q(pc)} AS {_q(name)}")
                cols.append((name, name, _nullable(p, pc)))
        sql = f"""
        WITH ev AS (
            SELECT s.{_SEED_IDX}, s.{_SEED_T}, e.{_q(t.time_col)} AS {_EV_T},
                   {self._order_expr("e", t)} AS {_EV_ORDER}
                   {"".join(", " + x for x in selects)}
            FROM seeds s
            JOIN {_q(child)} e
              ON e.{_q(fk)} = s.{_SEED_ID} AND e.{_q(t.time_col)} {self.lt} s.{_SEED_T}
            {" ".join(joins)}
        )
        SELECT {_SEED_IDX}, {", ".join(self._agg_select(prefix, cols, cats))}
        FROM ev GROUP BY {_SEED_IDX}, {_SEED_T}
        """
        return self._run(sql)

    def _t2_children(self, ptable: str):
        """``(T2 name, T2, fk to P, name suffix)`` for every time-stamped child of P."""
        for t2_name, t2 in self.tables.items():
            if t2.time_col is None:
                continue
            fks = [fk2 for fk2, pt2 in t2.fkey_col_to_pkey_table.items() if pt2 == ptable]
            for fk2 in fks:
                yield t2_name, t2, fk2, (f".{fk2}" if len(fks) > 1 else "")

    def hop2(self, child: str, fk: str) -> list[pd.DataFrame]:
        t = self.tables[child]
        out = []
        for pfk, ptable in t.fkey_col_to_pkey_table.items():
            if ptable == self.entity_table or ptable not in self.tables:
                continue
            p = self.tables[ptable]
            if p.time_col is not None or p.pkey_col is None:
                continue
            # The parent of each seed's most recent T row, e.g. its current constructor.
            self.con.execute(f"""
                CREATE OR REPLACE TEMP TABLE lp AS
                SELECT s.{_SEED_IDX}, s.{_SEED_T},
                       arg_max(e.{_q(pfk)}, {self._order_expr("e", t)}) AS __pid
                FROM seeds s
                JOIN {_q(child)} e
                  ON e.{_q(fk)} = s.{_SEED_ID} AND e.{_q(t.time_col)} {self.lt} s.{_SEED_T}
                GROUP BY s.{_SEED_IDX}, s.{_SEED_T}
            """)
            for t2_name, t2, fk2, suffix in self._t2_children(ptable):
                prefix = f"{child}.{pfk}>{t2_name}{suffix}"
                cols = [(c, c, _nullable(t2, c)) for c in _numeric_cols(t2, t2_name, self.cfg)]
                cats = self._cat_spec(t2, t2_name)
                selects = "".join(f", e2.{_q(c)} AS {_q(c)}" for c, _, _ in cols + cats)
                # Aggregate once per distinct (parent, seed time): teammates share one.
                aggs = self._agg_select(prefix, cols, cats)
                names = ", ".join(f"a.{a.rsplit(' AS ', 1)[1]}" for a in aggs)
                sql = f"""
                WITH k AS (SELECT DISTINCT __pid, {_SEED_T} FROM lp),
                ev AS (
                    SELECT k.__pid, k.{_SEED_T}, e2.{_q(t2.time_col)} AS {_EV_T},
                           {self._order_expr("e2", t2)} AS {_EV_ORDER} {selects}
                    FROM k
                    JOIN {_q(t2_name)} e2
                      ON e2.{_q(fk2)} = k.__pid
                     AND e2.{_q(t2.time_col)} {self.lt} k.{_SEED_T}
                ),
                a AS (
                    SELECT __pid, {_SEED_T}, {", ".join(aggs)}
                    FROM ev GROUP BY __pid, {_SEED_T}
                )
                SELECT lp.{_SEED_IDX}, {names}
                FROM lp JOIN a ON a.__pid = lp.__pid AND a.{_SEED_T} = lp.{_SEED_T}
                """
                out.append(self._run(sql))
        return out

    def hop2_all(self, child: str, fk: str) -> list[pd.DataFrame]:
        """E <- T -> P <- T2 over every distinct P linked before the seed time, rolled up.

        A ``"P>Q"`` entry in ``hop2_all_via`` goes one link further,
        E <- T -> P <- L -> Q <- T2: every distinct Q linked to one of those P by a row of a
        link table L (e.g. the other studies of a study's sponsors).
        """
        t = self.tables[child]
        out = []
        chains = [v.split(">") for v in self.cfg.hop2_all_via]
        for pfk, ptable in t.fkey_col_to_pkey_table.items():
            wanted = [c for c in chains if c[0] == ptable]
            if pfk == fk or not wanted or ptable not in self.tables:
                continue
            time_cond = ""
            if t.time_col is not None:
                time_cond = f"AND e.{_q(t.time_col)} {self.lt} s.{_SEED_T}"
            self.con.execute(f"""
                CREATE OR REPLACE TEMP TABLE lk AS
                SELECT DISTINCT s.{_SEED_IDX}, s.{_SEED_T}, e.{_q(pfk)} AS __pid
                FROM seeds s
                JOIN {_q(child)} e ON e.{_q(fk)} = s.{_SEED_ID} {time_cond}
                WHERE e.{_q(pfk)} IS NOT NULL
            """)
            base = f"{child}.{pfk}"
            for chain in wanted:
                if len(chain) == 1:
                    out.extend(self._rollup_children("lk", ptable, base))
                    continue
                (qtable,) = chain[1:]
                for lname, link in self.tables.items():
                    fks = link.fkey_col_to_pkey_table
                    for lfk, lp in fks.items():
                        if lp != ptable:
                            continue
                        for qfk, lq in fks.items():
                            if lq != qtable or qfk == lfk:
                                continue
                            cond = ""
                            if link.time_col is not None:
                                cond = f"AND l.{_q(link.time_col)} {self.lt} lk.{_SEED_T}"
                            self.con.execute(f"""
                                CREATE OR REPLACE TEMP TABLE lk2 AS
                                SELECT DISTINCT lk.{_SEED_IDX}, lk.{_SEED_T}, l.{_q(qfk)} AS __pid
                                FROM lk JOIN {_q(lname)} l ON l.{_q(lfk)} = lk.__pid {cond}
                                WHERE l.{_q(qfk)} IS NOT NULL
                            """)
                            prefix = f"{base}>>{lname}.{qfk}"
                            out.extend(self._rollup_children("lk2", qtable, prefix))
        return out

    def _rollup_children(self, links: str, ptable: str, base: str) -> list[pd.DataFrame]:
        """For a ``(seed, __pid)`` link relation over ``ptable``: per seed, the pooled
        aggregates of each time-stamped child T2 of ``ptable`` over all the linked parents."""
        out = []
        for t2_name, t2, fk2, suffix in self._t2_children(ptable):
            prefix = f"{base}>>{t2_name}{suffix}"
            out.append(self._rollup(links, _q(t2_name), t2_name, t2, fk2, prefix))
        return out

    def _rollup(
        self, links: str, source: str, t2_name: str, t2: Table, fk2: str, prefix: str
    ) -> pd.DataFrame:
        """Pooled aggregates of the rows of ``source`` (T2's columns, or a relation with them)
        whose ``fk2`` is one of the seed's linked ``__pid``, strictly before the seed time."""
        cols = [(c, _nullable(t2, c)) for c in _numeric_cols(t2, t2_name, self.cfg)]
        cats = [
            (c, v)
            for c, _, v in self._cat_spec(t2, t2_name)
            if len(v) <= self.cfg.max_count_categories
        ]
        inner, outer = self._rollup_selects(prefix, cols, cats)
        sql = f"""
        WITH k AS (SELECT DISTINCT __pid, {_SEED_T} FROM {links}),
        ev AS (
            SELECT k.__pid, k.{_SEED_T}, e2.{_q(t2.time_col)} AS {_EV_T}
                   {"".join(f", e2.{_q(c)} AS {_q(c)}" for c, _ in cols + cats)}
            FROM k
            JOIN {source} e2
              ON e2.{_q(fk2)} = k.__pid
             AND e2.{_q(t2.time_col)} {self.lt} k.{_SEED_T}
        ),
        a AS (
            SELECT __pid, {_SEED_T}, {", ".join(inner)}
            FROM ev GROUP BY __pid, {_SEED_T}
        )
        SELECT lk.{_SEED_IDX}, {", ".join(outer)}
        FROM {links} lk JOIN a ON a.__pid = lk.__pid AND a.{_SEED_T} = lk.{_SEED_T}
        GROUP BY lk.{_SEED_IDX}, lk.{_SEED_T}
        """
        return self._run(sql)

    def siblings(self) -> list[pd.DataFrame]:
        """E -> P <- E' <- T for E's foreign keys named in ``siblings_via``: the pooled rows of
        each time-stamped child T over every entity sharing the seed entity's parent P
        (e.g. all ads in the ad's category), strictly before the seed time."""
        e = self.tables[self.entity_table]
        out = []
        for fk in self.cfg.siblings_via:
            seen = ""
            if e.time_col is not None:  # an unborn entity's parent is not known
                seen = f"AND ent.{_q(e.time_col)} {self.lt} s.{_SEED_T}"
            self.con.execute(f"""
                CREATE OR REPLACE TEMP TABLE lks AS
                SELECT s.{_SEED_IDX}, s.{_SEED_T}, ent.{_q(fk)} AS __pid
                FROM seeds s
                JOIN {_q(self.entity_table)} ent ON ent.{_q(e.pkey_col)} = s.{_SEED_ID} {seen}
                WHERE ent.{_q(fk)} IS NOT NULL
            """)
            for child, tfk in self.child_paths():
                t = self.tables[child]
                source = (
                    f"(SELECT c.*, ent.{_q(fk)} AS __sibp FROM {_q(child)} c "
                    f"JOIN {_q(self.entity_table)} ent ON c.{_q(tfk)} = ent.{_q(e.pkey_col)})"
                )
                prefix = f"{self.entity_table}.{fk}>>{child}"
                if self._n_fks(child) > 1:
                    prefix += f".{tfk}"
                out.append(self._rollup("lks", source, child, t, "__sibp", prefix))
        return out

    def _rollup_selects(
        self,
        prefix: str,
        cols: list[tuple[str, bool]],
        cats: list[tuple[str, list[str]]],
    ) -> tuple[list[str], list[str]]:
        """Per-(P, seed time) partial aggregates, and the per-seed expressions combining them.

        The combination is exact: pooled count, mean (sum / non-null count), min, max and null
        fraction of the T2 rows of all the seed's parents.
        """
        inner, outer = [], []
        k = 0

        def part(expr: str) -> str:
            nonlocal k
            k += 1
            inner.append(f"{expr} AS __a{k}")
            return f"__a{k}"

        for w in self.cfg.windows:
            wf = self._window_filter(w)
            wn = _window_name(w)
            n = part(f"count({self._when(wf, '1')})")
            outer.append(f"sum({n}) AS {_q(f'{prefix}__count__{wn}')}")
            for col, nullable in cols:
                v = self._when(wf, f"CAST({_q(col)} AS DOUBLE)")
                nn, sm = part(f"count({v})"), part(f"sum({v})")
                mn, mx = part(f"min({v})"), part(f"max({v})")
                pooled = {
                    "mean": f"sum({sm}) / nullif(sum({nn}), 0)",
                    "min": f"min({mn})",
                    "max": f"max({mx})",
                    "sum": f"sum({sm})",
                }
                for agg in self.cfg.aggs:
                    outer.append(f"{pooled[agg]} AS {_q(f'{prefix}__{col}__{agg}__{wn}')}")
                if nullable:
                    outer.append(
                        f"1 - sum({nn}) / nullif(sum({n}), 0) "
                        f"AS {_q(f'{prefix}__{col}__nullfrac__{wn}')}"
                    )
            for col, values in cats:
                for i, val in enumerate(values):
                    cond = f"CAST({_q(col)} AS VARCHAR) = {_lit(val)}"
                    cond = f"{wf} AND {cond}" if wf else cond
                    c = part(f"count({self._when(cond, '1')})")
                    name = f"{prefix}__{col}={_value_name(i, val)}__count__{wn}"
                    outer.append(f"sum({c}) AS {_q(name)}")
        last = part(f"max({_EV_T})")
        outer.append(
            f"(epoch(lk.{_SEED_T}) - epoch(max({last}))) / 86400.0 "
            f"AS {_q(f'{prefix}__days_since_last')}"
        )
        outer.append(f"count(*) AS {_q(f'{prefix}__n_parents')}")
        return inner, outer

    def _n_fks(self, child: str) -> int:
        fks = self.tables[child].fkey_col_to_pkey_table.values()
        return sum(1 for v in fks if v == self.entity_table)

    def own_columns(
        self, table_name: str, ids: np.ndarray, seed_t: np.ndarray, prefix: str
    ) -> pd.DataFrame:
        """A table's own columns for the rows with primary key ``ids``, as of ``seed_t``.

        For a time-stamped table, a row is visible only if its timestamp is strictly before the
        seed time; otherwise every column is NaN. Its age is added as
        ``<prefix>__<time col>__days_before``.
        """
        t = self.db.table_dict[table_name]
        df = t.df.set_index(t.pkey_col)
        skip = _skip_cols(t, table_name, self.cfg) - {t.time_col}
        joined = df.reindex(ids)
        out = pd.DataFrame(index=np.arange(len(ids)))
        visible = np.ones(len(ids), dtype=bool)
        if t.time_col is not None:
            own_t = joined[t.time_col].to_numpy().astype("datetime64[ns]")
            visible = own_t < seed_t if not self.cfg._inclusive else own_t <= seed_t
        for col in df.columns:
            if col in skip:
                continue
            name = f"{prefix}__{col}"
            dtype = df[col].dtype
            vals = joined[col].to_numpy()
            if pd.api.types.is_datetime64_any_dtype(dtype):
                delta = (seed_t - vals.astype("datetime64[ns]")) / np.timedelta64(1, "D")
                out[f"{name}__days_before"] = np.where(visible, delta.astype(float), np.nan)
            elif pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
                num = pd.array(vals).astype("Float64").to_numpy(dtype=float, na_value=np.nan)
                out[name] = np.where(visible, num, np.nan)
            elif (table_name, col) in self.vocab:
                dtype_c = pd.CategoricalDtype(categories=self.vocab[(table_name, col)])
                ok = pd.notna(vals) & visible
                s = pd.Series(vals, index=out.index).astype(object).where(ok, None)
                out[name] = s.where(~ok, s.astype(str)).astype(dtype_c)
        return out

    def entity_static(self, seeds: pd.DataFrame) -> pd.DataFrame:
        ids = seeds[_SEED_ID].to_numpy()
        seed_t = seeds[_SEED_T].to_numpy().astype("datetime64[ns]")
        parts = [self.own_columns(self.entity_table, ids, seed_t, self.entity_table)]
        # E -> P: the entity's parents' own columns.
        e = self.db.table_dict[self.entity_table]
        if e.fkey_col_to_pkey_table:
            ekeys = e.df.set_index(e.pkey_col).reindex(ids)
            for fk, ptable in e.fkey_col_to_pkey_table.items():
                if ptable not in self.tables or self.tables[ptable].pkey_col is None:
                    continue
                pids = ekeys[fk].to_numpy()
                if e.time_col is not None:  # an unborn entity has no known parent
                    own_t = ekeys[e.time_col].to_numpy().astype("datetime64[ns]")
                    vis = own_t <= seed_t if self.cfg._inclusive else own_t < seed_t
                    pids = np.where(vis, pids, None)
                parts.append(self.own_columns(ptable, pids, seed_t, f"{self.entity_table}.{fk}"))
        out = pd.concat(parts, axis=1)
        out.index = pd.Index(seeds[_SEED_IDX].to_numpy(), name=_SEED_IDX)
        return out


def build_features(
    db: Database,
    seeds: pd.DataFrame,
    entity_table: str,
    entity_col: str,
    time_col: str,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """One feature row per row of ``seeds``, in the same order and with the same index.

    ``seeds`` needs ``entity_col`` (ids of ``entity_table``'s primary key) and ``time_col``
    (the seed time). Counts are 0 where an entity has no history; other aggregates are NaN.
    """
    cfg = config or FeatureConfig()
    vocab = cfg.vocab if cfg.vocab is not None else feature_vocab(db, cfg)
    s = pd.DataFrame(
        {
            _SEED_IDX: np.arange(len(seeds)),
            _SEED_ID: seeds[entity_col].to_numpy(),
            _SEED_T: pd.to_datetime(seeds[time_col]).to_numpy(),
        }
    )
    b = _Builder(db, entity_table, cfg, vocab)
    try:
        b.con.register("seeds", s)
        parts = [b.entity_static(s)]
        for child, fk in b.child_paths():
            parts.append(b.hop1(child, fk))
            if cfg.hop2:
                parts.extend(b.hop2(child, fk))
        if cfg.hop2_all_via:
            for child, fk in b.child_paths(timed=False):
                parts.extend(b.hop2_all(child, fk))
        if cfg.siblings_via:
            parts.extend(b.siblings())
    finally:
        b.close()
    base = pd.DataFrame(index=pd.Index(s[_SEED_IDX], name=_SEED_IDX))
    out = base.join(parts, how="left") if len(parts) > 1 else base.join(parts[0])
    count_cols = [c for c in out.columns if "__count__" in c or c.endswith("__n_parents")]
    out[count_cols] = out[count_cols].fillna(0)
    # `last` of a text column comes back from DuckDB as strings: make it categorical.
    for c in out.columns:
        if c.endswith("__last") and _is_text(out[c].dtype):
            out[c] = out[c].astype(pd.CategoricalDtype(categories=_last_vocab(c, vocab)))
    out.index = seeds.index
    return out


def _last_vocab(col: str, vocab: Vocab) -> list[str]:
    """The vocabulary behind a ``...__<text col>__last`` feature (its path's final table)."""
    path, stem, _ = col.rsplit("__", 2)
    table = re.split(r">>?", path)[-1].split(".")[0]
    return vocab[(table, stem)]


def leakage_mismatches(
    db: Database,
    seeds: pd.DataFrame,
    entity_table: str,
    entity_col: str,
    time_col: str,
    config: FeatureConfig | None = None,
    builder: Callable[..., pd.DataFrame] = build_features,
    atol: float = 1e-9,
) -> list[tuple[int, str, object, object]]:
    """Compare features on the full database with features on the truncated one, per seed.

    For each seed row, the features from ``db`` must equal those from
    ``truncate_before(db, seed_time)``. Returns ``(row, column, full, truncated)`` for every
    value that differs; an empty list means no leak was found for these seeds. Both builds
    share the full database's text vocabulary, so they have the same feature set.
    """
    cfg = config or FeatureConfig()
    if cfg.vocab is None:
        cfg = replace(cfg, vocab=feature_vocab(db, cfg))
    full = builder(db, seeds, entity_table, entity_col, time_col, cfg)
    bad = []
    for i in range(len(seeds)):
        one = seeds.iloc[[i]]
        t = pd.Timestamp(one[time_col].iloc[0])
        trunc = builder(truncate_before(db, t), one, entity_table, entity_col, time_col, cfg)
        a, b = full.iloc[i], trunc.iloc[0]
        for col in full.columns:
            x, y = a[col], b.get(col, np.nan)
            if pd.isna(x) and pd.isna(y):
                continue
            if isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)):
                if abs(float(x) - float(y)) <= atol * max(1.0, abs(float(x))):
                    continue
            elif x == y:
                continue
            bad.append((i, col, x, y))
    return bad
