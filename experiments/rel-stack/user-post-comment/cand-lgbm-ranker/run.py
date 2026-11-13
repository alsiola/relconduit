"""rel-stack/user-post-comment: candidates + LightGBM ranker. See README.md."""

import numpy as np
import pandas as pd

from relconduit.features import FeatureConfig
from relconduit.link import (
    LinkConfig,
    Path,
    concat_edges,
    join_edges,
    table_edges,
    token_edges,
)
from relconduit.link_lgbm import main


def comment(db):
    return table_edges(db, "comments", "UserId", "PostId")


def history(db):
    """user -> post the user edited (or created: postHistory has a row for each)."""
    return table_edges(db, "postHistory", "UserId", "PostId")


def own(db):
    return table_edges(db, "posts", "OwnerUserId", "Id")


def touch(db):
    return concat_edges(comment(db), history(db), own(db))


def _root(posts: pd.DataFrame) -> np.ndarray:
    """A post's thread: its question (its parent for an answer, else itself)."""
    return posts["ParentId"].fillna(posts["Id"]).astype(np.int64).to_numpy()


def thread(db):
    """post -> its thread's question, at the post's creation."""
    posts = db.table_dict["posts"].df
    return concat_edges(
        pd.DataFrame({"src": posts["Id"], "dst": _root(posts), "t": posts["CreationDate"]})
    )


def split_tags(s: str) -> list[str]:
    return s.strip("<>").split("><")


def post_tag(db):
    """post -> each tag of its thread's question (answers carry no tags of their own)."""
    tags = token_edges(db, "posts", "Id", "Tags", split_tags)  # question -> tag
    return join_edges(thread(db), tags)


def tag_comments(db):
    """tag -> post, one edge per comment on a post in a thread with the tag."""
    c = db.table_dict["comments"].df[["PostId", "CreationDate"]].dropna()
    pt = post_tag(db)
    m = c.merge(pt, left_on="PostId", right_on="src")
    t = np.maximum(m["CreationDate"].to_numpy(), m["t"].to_numpy())
    return concat_edges(pd.DataFrame({"src": m["dst"], "dst": m["PostId"], "t": t}))


def post_activity(db):
    """anything -> post: comments, edits and votes on the post (recent activity)."""
    votes = table_edges(db, "votes", "Id", "PostId")
    hist = table_edges(db, "postHistory", "Id", "PostId")
    com = table_edges(db, "comments", "Id", "PostId")
    return concat_edges(votes, hist, com)


LINK = LinkConfig(
    edges={
        "comment": comment,
        "history": history,
        "own": own,
        "touch": touch,
        "thread": thread,
        "post_tag": post_tag,
        "tag_comments": tag_comments,
        "post_activity": post_activity,
    },
    direct=["comment", "history", "own"],
    n_direct=200,
    paths={
        # Posts in the threads the user has commented on, edited or written.
        "thread": Path(("touch", "thread", "~thread"), n=300, half_life_days=180),
        # Posts commented on by users who commented where this user did.
        "covisit": Path(("comment", "~comment", "comment"), n=200, half_life_days=180),
        # Posts with comments in the last 30 days in the tags the user is active in.
        "tags": Path(("touch", "post_tag", "tag_comments"), n=200, window_days=(None, None, 30)),
    },
    popularity=["comment", "post_activity"],
    pop_windows=(7, 30, 91, 365),
    pop_candidates={"comment": (30, 100)},
)
FEATURES = FeatureConfig(windows=(91, 365, None), hop2=False)

if __name__ == "__main__":
    main("rel-stack", "user-post-comment", LINK, FEATURES, FEATURES, train_sample=6_000)
