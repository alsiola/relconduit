"""rel-stack/post-votes: LightGBM on time-safe relational features. See README.md."""

import pandas as pd
from relbench.base import Database, Table

from relconduit.big_task import main
from relconduit.features import FeatureConfig

# Type codes are labels, not quantities: as text they get a count per value and window (the
# label counts upvotes, VoteTypeId 2, in the next 91 days).
TYPE_CODES = {
    "votes": ["VoteTypeId"],
    "posts": ["PostTypeId"],
    "postLinks": ["LinkTypeId"],
    "postHistory": ["PostHistoryTypeId"],
    "badges": ["Class"],
}


def codes_as_text(db: Database) -> Database:
    """Row by row (a dtype change), so it can't leak."""
    tables = {}
    for name, t in db.table_dict.items():
        df = t.df
        cols = [c for c in TYPE_CODES.get(name, []) if c in df.columns]
        if cols:
            df = df.assign(
                **{c: df[c].map(lambda v: None if pd.isna(v) else str(int(v))) for c in cols}
            )
        tables[name] = Table(df, t.fkey_col_to_pkey_table, t.pkey_col, t.time_col)
    return Database(tables)


# A question's votes, comments, answers (posts by ParentId), links and edit history, and the
# owner's record pooled over all their posts' votes and answers (siblings via OwnerUserId).
# AccountId is an id, not a quantity.
FEATURES = FeatureConfig(
    windows=(30, 91, 365, None),
    aggs=("mean", "min", "max", "sum"),
    siblings_via=("OwnerUserId",),
    exclude_columns=(("users", "AccountId"),),
    max_count_categories=20,
)

if __name__ == "__main__":
    main("rel-stack", "post-votes", FEATURES, db_transform=codes_as_text)
