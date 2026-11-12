"""Time-safe relational features for RelBench entity tasks, computed with DuckDB.

Given a RelBench :class:`~relbench.base.Database` and a seed table (one row per
``(entity id, seed time)``), :func:`build_features` returns one feature row per seed row,
aggregated over the database's foreign-key paths. **Every row a feature reads has a
timestamp strictly before that seed row's seed time.** :func:`truncate_before` and
:func:`leakage_mismatches` exist to check that claim.

Paths, with ``E`` the entity table:

* ``E`` itself: its numeric columns, low-cardinality text columns as categoricals and its
  datetime columns as days before the seed time (``E`` must be a static table).
* hop 1, ``E <- T``: every time-stamped table ``T`` with a foreign key to ``E``. Aggregates of
  ``T``'s numeric columns over ``T``'s rows for the entity, plus the numeric columns of
  ``T``'s other parents ``T -> P`` joined onto each row (hop 2 upward, e.g. a result row's
  race ``year`` and ``round``).
* hop 2, ``E <- T -> P <- T2`` for each *static* parent ``P`` (no time column, e.g. a
  constructor): take the ``P`` of the entity's most recent ``T`` row before the seed time
  (its current team) and aggregate ``T2``'s rows for that ``P``. Parents with a time column
  (e.g. races) are events, not memberships, so they are not walked this way.

Aggregates per numeric column, for each window (days before the seed time, ``None`` for all
time): ``count`` (one per path, not per column), ``mean``, ``min``, ``max`` and, for float
columns (which can hold missing values, e.g. a finishing position that is NaN for a
retirement), ``nullfrac``. Which features exist depends only on the schema and dtypes, never
on the rows, so truncating the database cannot change the feature set. Over all time, also ``last`` (the value on
the most recent row, nulls kept) and per path ``days_since_last``.

Primary- and foreign-key columns are never used as features.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd
from relbench.base import Database, Table

DEFAULT_WINDOWS: tuple[int | None, ...] = (30, 90, 365, None)
WINDOW_AGGS = ("mean", "min", "max")
MAX_CATEGORIES = 100  # text columns of E with more distinct values than this are dropped

_SEED_IDX = "__seed_idx"
_SEED_ID = "__eid"
_SEED_T = "__t"
_EV_T = "__et"
_EV_ORDER = "__ord"


@dataclass
class FeatureConfig:
    """Which paths and aggregates :func:`build_features` computes."""

    windows: Sequence[int | None] = DEFAULT_WINDOWS
    # Hop 2 through static parents (E <- T -> P <- T2). Off gives hop 1 (+ upward joins) only.
    hop2: bool = True
    # Tables never read (whole tables, as path steps or targets).
    exclude_tables: Sequence[str] = ()
    # (table, column) pairs never used as feature values.
    exclude_columns: Sequence[tuple[str, str]] = ()
    # Testing only: compare event times with ``<=`` instead of ``<``, which leaks rows stamped
    # exactly at the seed time. Exists so the leakage test can prove it catches a leak.
    _inclusive: bool = field(default=False, repr=False)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _window_name(w: int | None) -> str:
    return "all" if w is None else f"{w}d"


def _numeric_cols(table: Table, name: str, cfg: FeatureConfig) -> list[str]:
    """Columns aggregated as values: numeric or bool, and not a key or the time column."""
    skip = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
    skip |= {c for t, c in cfg.exclude_columns if t == name}
    out = []
    for col in table.df.columns:
        if col in skip:
            continue
        dtype = table.df[col].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            out.append(col)
    return out


def _nullable(table: Table, col: str) -> bool:
    return pd.api.types.is_float_dtype(table.df[col].dtype)


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
    def __init__(self, db: Database, entity_table: str, cfg: FeatureConfig):
        self.db = db
        self.entity_table = entity_table
        self.cfg = cfg
        self.lt = "<=" if cfg._inclusive else "<"
        self.con = duckdb.connect()
        self.tables = {n: t for n, t in db.table_dict.items() if n not in set(cfg.exclude_tables)}
        for name, t in self.tables.items():
            df = t.df
            if isinstance(df.index, pd.RangeIndex) and df.index.start == 0:
                df = df.reset_index(drop=True)
            self.con.register(name, df)

    def close(self) -> None:
        self.con.close()

    # --- helpers ---------------------------------------------------------------------

    def _order_expr(self, alias: str, table: Table) -> str:
        """Tie-break key for "most recent row": time, then primary key when there is one."""
        if table.pkey_col is not None:
            return f"({alias}.{_q(table.time_col)}, {alias}.{_q(table.pkey_col)})"
        return f"{alias}.{_q(table.time_col)}"

    def _window_filter(self, w: int | None) -> str | None:
        if w is None:
            return None
        return f"{_EV_T} >= {_SEED_T} - INTERVAL {int(w)} DAY"

    def _agg_select(self, prefix: str, value_cols: list[tuple[str, str, bool]]) -> list[str]:
        """Aggregate expressions over the ``ev`` relation.

        ``value_cols`` are ``(column in ev, feature-name stem, nullable)`` triples.
        """
        exprs = []
        for w in self.cfg.windows:
            wf = self._window_filter(w)
            filt = f" FILTER (WHERE {wf})" if wf else ""
            wn = _window_name(w)
            exprs.append(f"count(*){filt} AS {_q(f'{prefix}__count__{wn}')}")
            for col, stem, nullable in value_cols:
                c = _q(col)
                v = f"CAST({c} AS DOUBLE)"
                for agg in WINDOW_AGGS:
                    fn = {"mean": "avg", "min": "min", "max": "max"}[agg]
                    exprs.append(f"{fn}({v}){filt} AS {_q(f'{prefix}__{stem}__{agg}__{wn}')}")
                if nullable:
                    exprs.append(
                        f"avg(CAST({c} IS NULL AS DOUBLE)){filt} "
                        f"AS {_q(f'{prefix}__{stem}__nullfrac__{wn}')}"
                    )
        for col, stem, _ in value_cols:
            exprs.append(
                f"arg_max_null(CAST({_q(col)} AS DOUBLE), {_EV_ORDER}) "
                f"AS {_q(f'{prefix}__{stem}__last')}"
            )
        exprs.append(
            f"(epoch({_SEED_T}) - epoch(max({_EV_T}))) / 86400.0 "
            f"AS {_q(f'{prefix}__days_since_last')}"
        )
        return exprs

    def _run(self, sql: str) -> pd.DataFrame:
        return self.con.sql(sql).df().set_index(_SEED_IDX)

    # --- paths -----------------------------------------------------------------------

    def child_paths(self) -> list[tuple[str, str]]:
        """``(T, fkey col)`` for every time-stamped table with a foreign key to E."""
        out = []
        for name, t in self.tables.items():
            if t.time_col is None:
                continue
            for fk, pk_table in t.fkey_col_to_pkey_table.items():
                if pk_table == self.entity_table:
                    out.append((name, fk))
        return out

    def hop1(self, child: str, fk: str) -> pd.DataFrame:
        t = self.tables[child]
        prefix = child if self._n_fks(child) == 1 else f"{child}.{fk}"
        cols = [(c, c, _nullable(t, c)) for c in _numeric_cols(t, child, self.cfg)]
        selects = [f"e.{_q(c)} AS {_q(c)}" for c, _, _ in cols]
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
        SELECT {_SEED_IDX}, {", ".join(self._agg_select(prefix, cols))}
        FROM ev GROUP BY {_SEED_IDX}, {_SEED_T}
        """
        return self._run(sql)

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
            for t2_name, t2 in self.tables.items():
                if t2.time_col is None:
                    continue
                for fk2, pt2 in t2.fkey_col_to_pkey_table.items():
                    if pt2 != ptable:
                        continue
                    prefix = f"{child}.{pfk}>{t2_name}"
                    if sum(1 for v in t2.fkey_col_to_pkey_table.values() if v == ptable) > 1:
                        prefix += f".{fk2}"
                    cols = [(c, c, _nullable(t2, c)) for c in _numeric_cols(t2, t2_name, self.cfg)]
                    selects = "".join(f", e2.{_q(c)} AS {_q(c)}" for c, _, _ in cols)
                    # Aggregate once per distinct (parent, seed time): teammates share one.
                    aggs = self._agg_select(prefix, cols)
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

    def _n_fks(self, child: str) -> int:
        fks = self.tables[child].fkey_col_to_pkey_table.values()
        return sum(1 for v in fks if v == self.entity_table)

    def entity_static(self, seeds: pd.DataFrame) -> pd.DataFrame:
        e = self.db.table_dict[self.entity_table]
        if e.time_col is not None:
            raise ValueError(f"entity table {self.entity_table!r} has a time column; unsupported")
        df = e.df.set_index(e.pkey_col)
        skip = {e.pkey_col, *e.fkey_col_to_pkey_table}
        skip |= {c for t, c in self.cfg.exclude_columns if t == self.entity_table}
        joined = df.reindex(seeds[_SEED_ID].to_numpy())
        out = pd.DataFrame(index=seeds[_SEED_IDX].to_numpy())
        out.index.name = _SEED_IDX
        seed_t = seeds[_SEED_T].to_numpy()
        for col in df.columns:
            if col in skip:
                continue
            name = f"{self.entity_table}__{col}"
            dtype = df[col].dtype
            vals = joined[col].to_numpy()
            if pd.api.types.is_datetime64_any_dtype(dtype):
                delta = (seed_t - vals.astype("datetime64[ns]")) / np.timedelta64(1, "D")
                out[f"{name}__days_before"] = delta.astype(float)
            elif pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
                out[name] = pd.array(vals).astype("Float64").to_numpy(dtype=float, na_value=np.nan)
            else:
                cats = pd.Series(df[col].dropna().unique()).astype(str).sort_values()
                if len(cats) > MAX_CATEGORIES:
                    continue
                dtype_c = pd.CategoricalDtype(categories=cats.tolist())
                out[name] = (
                    pd.Series(vals, index=out.index)
                    .astype(object)
                    .where(pd.notna(vals), None)
                    .astype(str)
                    .where(pd.notna(vals), None)
                    .astype(dtype_c)
                )
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
    s = pd.DataFrame(
        {
            _SEED_IDX: np.arange(len(seeds)),
            _SEED_ID: seeds[entity_col].to_numpy(),
            _SEED_T: pd.to_datetime(seeds[time_col]).to_numpy(),
        }
    )
    b = _Builder(db, entity_table, cfg)
    try:
        b.con.register("seeds", s)
        parts = [b.entity_static(s)]
        for child, fk in b.child_paths():
            parts.append(b.hop1(child, fk))
            if cfg.hop2:
                parts.extend(b.hop2(child, fk))
    finally:
        b.close()
    base = pd.DataFrame(index=pd.Index(s[_SEED_IDX], name=_SEED_IDX))
    out = base.join(parts, how="left") if len(parts) > 1 else base.join(parts[0])
    count_cols = [c for c in out.columns if "__count__" in c]
    out[count_cols] = out[count_cols].fillna(0)
    out.index = seeds.index
    return out


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
    value that differs; an empty list means no leak was found for these seeds.
    """
    full = builder(db, seeds, entity_table, entity_col, time_col, config)
    bad = []
    for i in range(len(seeds)):
        one = seeds.iloc[[i]]
        t = pd.Timestamp(one[time_col].iloc[0])
        trunc = builder(truncate_before(db, t), one, entity_table, entity_col, time_col, config)
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
