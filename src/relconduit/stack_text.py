"""Time-safe text-length and tag features for rel-stack posts.

The ``posts`` table holds each post's ``Title``, ``Body``, ``Tags`` and ``ContentLicense`` as
of the dump (2023), *after* every later edit. A feature read from those columns at an earlier
seed time can see the future: about 6% of questions had their tags retagged later, and
``ContentLicense`` is the licence of the post's last edit (CC BY-SA 4.0 means an edit after
May 2018). So the text features here come from ``postHistory`` instead, where each revision is
its own time-stamped row: initial title, body and tags (``PostHistoryTypeId`` 1, 2, 3) and
their edits (4, 5, 6). For a seed ``(post, t)`` a feature reads only revisions dated strictly
before ``t``, and the latest of each kind is the post as it stood at ``t``.

Two steps:

1. :func:`revision_stats` turns each revision row into numbers, row by row (so it can't
   leak): text length, words, lines, and counts of code spans, links, images and LaTeX ``$``
   for bodies; length, words and a trailing ``?`` for titles; the tag list and its length for
   tags. :func:`load_revision_stats` does the same straight from the parquet file, so the
   free text never reaches pandas.
2. :func:`post_text_features` builds one row per seed: the latest title, body and tags
   revision before the seed time (its stats, its age in days, the number of revisions of that
   kind so far), and tag aggregates. For each tag of the question as it stood at ``t``: the
   number of questions carrying it at ``t`` (each by its own latest tags revision before
   ``t``), how many of those were asked in the 91 days before ``t``, and the upvotes
   (``VoteTypeId`` 2) those questions got before ``t``, over all time and in the last 91 days,
   with the per-question means. Aggregated over the question's tags with mean, max, min and sum.

:func:`leakage_mismatches` (in :mod:`relconduit.features`) checks it, with :func:`builder` as
the builder: features on the full database must equal those on the database cut strictly
before each seed time.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from relbench.base import Database, Table

TITLE_TYPES = (1, 4)
BODY_TYPES = (2, 5)
TAGS_TYPES = (3, 6)
KINDS = {"title": TITLE_TYPES, "body": BODY_TYPES, "tags": TAGS_TYPES}
UPVOTE = "2"
TAG_WINDOW_DAYS = 91

# Per-revision numbers, by kind. Each is a SQL expression over the revision's ``Text``.
_STATS = {
    "len": "length(Text)",
    "words": "length(string_split_regex(trim(Text), '\\s+'))",
    "lines": "length(Text) - length(replace(Text, chr(10), '')) + 1",
    "code": "(length(Text) - length(replace(Text, '`', '')))",
    "links": "(length(Text) - length(replace(Text, 'http', ''))) // 4",
    "images": "(length(Text) - length(replace(Text, '![', ''))) // 2",
    "latex": "(length(Text) - length(replace(Text, '$', '')))",
    "qmark": "CASE WHEN rtrim(Text) LIKE '%?' THEN 1 ELSE 0 END",
    "n": "length(regexp_extract_all(Text, '<([^>]+)>', 1))",
}
KIND_STATS = {
    "title": ("len", "words", "qmark"),
    "body": ("len", "words", "lines", "code", "links", "images", "latex"),
    "tags": ("n",),
}
STAT_COLS = sorted({s for stats in KIND_STATS.values() for s in stats})


def _kind_case() -> str:
    parts = [
        f"WHEN PostHistoryTypeId IN ({', '.join(map(str, types))}) THEN '{kind}'"
        for kind, types in KINDS.items()
    ]
    return "CASE " + " ".join(parts) + " END"


def _stats_sql(source: str, where: str = "") -> str:
    types = sorted(t for ts in KINDS.values() for t in ts)
    cols = []
    for stat in STAT_COLS:
        kinds = [k for k, s in KIND_STATS.items() if stat in s]
        cond = " OR ".join(f"PostHistoryTypeId IN ({', '.join(map(str, KINDS[k]))})" for k in kinds)
        cols.append(f"CASE WHEN {cond} THEN CAST({_STATS[stat]} AS DOUBLE) END AS {stat}")
    tags = (
        f"CASE WHEN PostHistoryTypeId IN ({', '.join(map(str, TAGS_TYPES))}) "
        "THEN list_distinct(regexp_extract_all(Text, '<([^>]+)>', 1)) END AS tag_list"
    )
    cond = f"PostHistoryTypeId IN ({', '.join(map(str, types))}) AND Text IS NOT NULL"
    if where:
        cond += f" AND {where}"
    return (
        f"SELECT Id, PostId, CreationDate, {_kind_case()} AS kind, {', '.join(cols)}, {tags} "
        f"FROM {source} WHERE {cond}"
    )


def revision_stats(post_history: pd.DataFrame) -> pd.DataFrame:
    """One row per title, body or tags revision: its post, time, kind and numbers.

    Row by row: each output row depends only on its own input row, so cutting the input at a
    time cuts the output at the same time.
    """
    con = duckdb.connect()
    con.register("ph", post_history)
    return con.execute(_stats_sql("ph")).df()


def load_revision_stats(path: str | Path, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """:func:`revision_stats` of ``postHistory.parquet``, with rows after ``cutoff`` dropped
    (``<=`` kept, as ``task.get_db(upto_test_timestamp=True)``)."""
    where = ""
    if cutoff is not None:
        where = f"CreationDate <= TIMESTAMP '{pd.Timestamp(cutoff)}'"
    con = duckdb.connect()
    return con.execute(_stats_sql(f"read_parquet('{path}')", where)).df()


def revisions_table(stats: pd.DataFrame) -> Table:
    """``stats`` as a time-stamped table, so :func:`relconduit.features.truncate_before` cuts it."""
    return Table(stats, {"PostId": "posts"}, pkey_col="Id", time_col="CreationDate")


def post_text_features(
    db: Database,
    seeds: pd.DataFrame,
    entity_col: str = "PostId",
    time_col: str = "timestamp",
    inclusive: bool = False,
) -> pd.DataFrame:
    """Text and tag features for each seed row, from ``db``'s ``revisions`` and ``votes``.

    ``db["revisions"]`` is :func:`revisions_table`; ``db["votes"]`` needs ``PostId``,
    ``VoteTypeId`` (int or text) and ``CreationDate``. Returns one row per seed row, in
    order, on ``seeds``'s index. ``inclusive`` compares with ``<=`` instead of ``<``: for the
    leakage test only, to show it catches a leak.
    """
    lt = "<=" if inclusive else "<"
    rev = db.table_dict["revisions"].df
    votes = db.table_dict["votes"].df[["PostId", "VoteTypeId", "CreationDate"]]
    s = pd.DataFrame(
        {
            "__i": np.arange(len(seeds)),
            "pid": seeds[entity_col].to_numpy(),
            "t": pd.to_datetime(seeds[time_col]).to_numpy(),
        }
    )
    con = duckdb.connect()
    con.register("rev_df", rev)
    con.register("votes_df", votes)
    con.register("s", s)
    con.execute(
        "CREATE TEMP TABLE rev AS SELECT *, row_number() OVER "
        "(PARTITION BY PostId, kind ORDER BY CreationDate, Id) AS ord FROM rev_df"
    )
    con.execute("CREATE TEMP TABLE ts AS SELECT DISTINCT t FROM s")

    # 1. The seed post's latest revision of each kind before t.
    out = s[["__i"]].copy()
    for kind, stats in KIND_STATS.items():
        sel = ", ".join(f"arg_max(r.{st}, r.ord) AS {kind}_{st}" for st in stats)
        if kind == "tags":
            sel = "arg_max(r.n, r.ord) AS tags_n"
        q = f"""
            SELECT s.__i, {sel}, count(*) AS {kind}_revisions,
                   date_diff('second', max(r.CreationDate), any_value(s.t)) / 86400.0
                       AS {kind}_days_since_edit
            FROM s JOIN rev r ON r.PostId = s.pid AND r.kind = '{kind}'
                AND r.CreationDate {lt} s.t
            GROUP BY s.__i
        """
        part = con.execute(q).df()
        out = out.merge(part, on="__i", how="left")
        out[f"{kind}_revisions"] = out[f"{kind}_revisions"].fillna(0)

    # 2. Tags of every question as it stood at each seed time, and its upvotes before then.
    con.execute(f"""
        CREATE TEMP TABLE qt AS
        SELECT r.PostId AS pid, ts.t, arg_max(r.tag_list, r.ord) AS tags,
               min(r.CreationDate) AS asked
        FROM rev r JOIN ts ON r.kind = 'tags' AND r.CreationDate {lt} ts.t
        GROUP BY r.PostId, ts.t
    """)
    con.execute(f"""
        CREATE TEMP TABLE qv AS
        SELECT v.PostId AS pid, ts.t, count(*) AS up_all,
               count(*) FILTER (WHERE v.CreationDate >= ts.t - INTERVAL {TAG_WINDOW_DAYS} DAY)
                   AS up_win
        FROM votes_df v JOIN ts ON CAST(v.VoteTypeId AS VARCHAR) = '{UPVOTE}'
            AND v.CreationDate {lt} ts.t
        GROUP BY v.PostId, ts.t
    """)
    con.execute(f"""
        CREATE TEMP TABLE tagstat AS
        SELECT tag, qt.t, count(*) AS questions,
               count(*) FILTER (WHERE qt.asked >= qt.t - INTERVAL {TAG_WINDOW_DAYS} DAY)
                   AS questions_win,
               coalesce(sum(qv.up_all), 0) AS up_all,
               coalesce(sum(qv.up_win), 0) AS up_win
        FROM (SELECT pid, t, asked, unnest(tags) AS tag FROM qt) qt
        LEFT JOIN qv ON qv.pid = qt.pid AND qv.t = qt.t
        GROUP BY tag, qt.t
    """)
    stat_cols = ["questions", "questions_win", "up_all", "up_win", "up_per_q", "up_win_per_q"]
    aggs = []
    for c in stat_cols:
        for a in ("mean", "max", "min", "sum"):
            if a == "sum" and c.endswith("per_q"):
                continue
            aggs.append(f"{a}({c}) AS tag_{c}_{a}")
    part = con.execute(f"""
        SELECT __i, {", ".join(aggs)} FROM (
            SELECT s.__i, st.questions, st.questions_win, st.up_all, st.up_win,
                   st.up_all / st.questions AS up_per_q,
                   st.up_win / st.questions AS up_win_per_q
            FROM s
            JOIN (SELECT pid, t, unnest(tags) AS tag FROM qt) mine
                ON mine.pid = s.pid AND mine.t = s.t
            JOIN tagstat st ON st.tag = mine.tag AND st.t = s.t
        ) GROUP BY __i
    """).df()
    out = out.merge(part, on="__i", how="left").sort_values("__i")
    out = out.drop(columns="__i").astype("float64")
    out.columns = [f"text__{c}" for c in out.columns]
    out.index = seeds.index
    return out


def builder(db, seeds, entity_table, entity_col, time_col, cfg=None) -> pd.DataFrame:
    """:func:`post_text_features` with :func:`relconduit.features.leakage_mismatches`'s
    builder signature; ``cfg._inclusive`` plants the ``<=`` leak."""
    return post_text_features(
        db, seeds, entity_col, time_col, inclusive=bool(getattr(cfg, "_inclusive", False))
    )
