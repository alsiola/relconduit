"""Load a RelBench task's database straight from its parquet files, without the free text.

``task.get_db()`` reads every column of every table into pandas. On the big datasets almost
all of that is free text that :mod:`relconduit.features` never uses: rel-amazon's ``review``
table is about 25 GB in pandas, 24 GB of it ``review_text``. :func:`load_task_db` reads the
same parquet files with column projection and the time filter pushed down, and leaves out
the columns the feature builder would drop anyway:

* text columns with more than ``max_categories`` distinct values in their first
  ``sample_rows`` rows, the rule :func:`relconduit.features.feature_vocab` uses to reject free
  text (a column that passes here can still be rejected there, on all its rows);
* list and other nested columns, which the builder has no aggregate for.

Keys and time columns are always kept. Otherwise the result is what ``task.get_db()`` gives,
column for column: rows dated after the test timestamp dropped with the same ``<=``
(``upto_test_timestamp=True``), the task's ``remove_columns`` dropped, and the dataset's own
``validate_and_correct_db`` applied (dangling foreign keys set to null).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from relbench.base import Database, Table

from relconduit.features import MAX_CATEGORIES

SAMPLE_ROWS = 20_000


def _kept_columns(
    pf: pq.ParquetFile,
    keys: set[str],
    removed: set[str],
    max_categories: int,
    sample_rows: int,
) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` column names of one parquet file."""
    kept, dropped = [], []
    schema = pf.schema_arrow
    for field in schema:
        name, typ = field.name, field.type
        if name in removed:
            continue
        if name in keys:
            kept.append(name)
        elif pa.types.is_nested(typ):
            dropped.append(name)
        elif pa.types.is_string(typ) or pa.types.is_large_string(typ):
            head = pf.read_row_group(0, columns=[name]).column(0)[:sample_rows]
            n = len(pc.unique(head.drop_null()))
            (kept if n <= max_categories else dropped).append(name)
        else:
            kept.append(name)
    return kept, dropped


def load_tables(
    db_dir: str | Path,
    tables: dict[str, dict],
    cutoff: pd.Timestamp | None,
    remove_columns: list[tuple[str, str]] = (),
    max_categories: int = MAX_CATEGORIES,
    sample_rows: int = SAMPLE_ROWS,
) -> tuple[Database, dict[str, list[str]]]:
    """A :class:`Database` from ``<db_dir>/<table>.parquet``, and the columns left out.

    ``tables`` maps each table name to ``{"pkey": ..., "time_col": ..., "fkeys": {...}}``
    (a RelBench dataset manifest's ``tables``). With ``cutoff``, rows of time-stamped tables
    dated after it are dropped (``<=`` kept, as :meth:`relbench.base.Table.upto`).
    """
    db_dir = Path(db_dir)
    removed: dict[str, set[str]] = {}
    for t, c in remove_columns:
        removed.setdefault(t, set()).add(c)
    table_dict, left_out = {}, {}
    for name, spec in tables.items():
        pf = pq.ParquetFile(db_dir / f"{name}.parquet")
        keys = {spec["pkey"], spec["time_col"], *spec["fkeys"]} - {None}
        kept, dropped = _kept_columns(
            pf, keys, removed.get(name, set()), max_categories, sample_rows
        )
        time_col = spec["time_col"]
        filters = None
        if cutoff is not None and time_col is not None:
            filters = [(time_col, "<=", pd.Timestamp(cutoff))]
        arrow = pq.read_table(db_dir / f"{name}.parquet", columns=kept, filters=filters)
        df = arrow.to_pandas()
        del arrow
        for col in df.columns:  # as relbench's loader: DuckDB can't ingest StringDtype
            if isinstance(df[col].dtype, pd.StringDtype):
                df[col] = df[col].astype(object)
        table_dict[name] = Table(
            df=df,
            fkey_col_to_pkey_table=dict(spec["fkeys"]),
            pkey_col=spec["pkey"],
            time_col=time_col,
        )
        left_out[name] = dropped
    return Database(table_dict), left_out


def load_task_db(task, upto_test_timestamp: bool = True, **kwargs) -> Database:
    """``task.get_db(upto_test_timestamp)`` without its free-text and nested columns.

    Keyword arguments go to :func:`load_tables`. The columns left out are printed.
    """
    ds = task.dataset
    tables = {
        name: {"pkey": s.pkey, "time_col": s.time_col, "fkeys": dict(s.fkeys)}
        for name, s in ds.manifest.tables.items()
    }
    cutoff = ds.test_timestamp if upto_test_timestamp else None
    db, left_out = load_tables(ds.db_dir, tables, cutoff, task.remove_columns, **kwargs)
    ds.validate_and_correct_db(db)
    dropped = {t: c for t, c in left_out.items() if c}
    print(f"load_task_db: left out {dropped}", flush=True)
    return db
