"""rel-stack/user-post-comment: wider candidates, all train rows, tuning, 5 seeds. See README.md."""

import importlib.util
from dataclasses import replace
from pathlib import Path as _P

from relconduit.link import Path, concat_edges, join_edges, reverse_edges, table_edges
from relconduit.link_search import SearchPlan, main

_spec = importlib.util.spec_from_file_location(
    "base_run", _P(__file__).parents[1] / "cand-lgbm-ranker" / "run.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def links(db):
    """question -> question through a post link (duplicate or related), both directions, at the
    latest of the link and the two posts."""
    th = base.thread(db)  # post -> its question
    pl = table_edges(db, "postLinks", "PostId", "RelatedPostId")
    one = join_edges(join_edges(reverse_edges(th), pl), th)  # question -> post -> post -> q
    return concat_edges(one, reverse_edges(one))


def interact(db):
    """user -> user: a commenter and the owner of the post, each way, at the later time."""
    c = join_edges(base.comment(db), reverse_edges(base.own(db)))  # commenter -> owner
    c = c[c["src"] != c["dst"]]
    return concat_edges(c, reverse_edges(c))


PLUS = replace(
    base.LINK,
    edges={**base.LINK.edges, "links": links, "interact": interact},
    n_direct=300,
    paths={
        # Wider versions of #12's sources.
        "thread": Path(("touch", "thread", "~thread"), n=1000, half_life_days=180),
        "covisit": Path(("comment", "~comment", "comment"), n=300, half_life_days=180),
        "tags": Path(("touch", "post_tag", "tag_comments"), n=300, window_days=(None, None, 30)),
        # New: posts in threads linked (duplicate/related) to threads the user touched.
        "linked": Path(("touch", "thread", "links", "~thread"), n=300, half_life_days=365),
        # New: recent posts by users this user comments with (either way round).
        "people": Path(("interact", "own"), n=300, half_life_days=180),
        # New: posts with comments in the last 91 days in the user's tags.
        "tags91": Path(("touch", "post_tag", "tag_comments"), n=300, window_days=(None, None, 91)),
    },
    pop_candidates={"comment": (30, 100), "post_activity": (30, 100)},
)

REF = {
    "objective": "lambdarank",
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
}
LR = {**REF, "learning_rate": 0.02}
GRID = [
    REF,  # #12's choice, learning rate 0.05
    LR,
    {**LR, "num_leaves": 31},
    {**LR, "min_child_samples": 100},
    {**LR, "lambda_l2": 10.0},
    {**LR, "feature_fraction": 0.5},
    {**LR, "lambdarank_truncation_level": 20},
    {**LR, "objective": "binary"},
]

PLAN = SearchPlan(
    variants={"base": base.LINK, "plus": PLUS},
    ref_config=REF,
    grid=GRID,
    stage1_rows=6_000,
    train_sizes=(12_000, None),
    leader=("Kapso", 0.1313),
)

if __name__ == "__main__":
    main("rel-stack", "user-post-comment", PLAN, base.FEATURES, base.FEATURES)
