"""rel-stack/post-post-related: candidates + LightGBM ranker. See README.md."""

import re

import numpy as np
import pandas as pd

from relconduit.features import FeatureConfig
from relconduit.link import (
    LinkConfig,
    Path,
    concat_edges,
    reverse_edges,
    table_edges,
    token_edges,
)
from relconduit.link_lgbm import main

STOP = {
    "what",
    "which",
    "when",
    "where",
    "does",
    "with",
    "from",
    "that",
    "this",
    "have",
    "there",
    "their",
    "about",
    "between",
    "using",
    "used",
    "would",
    "could",
    "should",
    "into",
    "than",
    "then",
    "them",
    "they",
    "your",
    "other",
    "only",
    "more",
    "most",
    "over",
    "same",
    "some",
    "such",
    "very",
    "just",
    "make",
    "made",
    "also",
    "each",
    "many",
    "much",
}


def link(db):
    return table_edges(db, "postLinks", "PostId", "RelatedPostId")


def link_any(db):
    e = link(db)
    return concat_edges(e, reverse_edges(e))


def split_tags(s: str) -> list[str]:
    return s.strip("<>").split("><")


def title_words(s: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9\-]{3,}", s.lower())
    return sorted({w for w in words if w not in STOP})


def tokens(col, split):
    def edges(db):
        return token_edges(db, "posts", "Id", col, split)

    return edges


def token_targets(col, split):
    """token -> post, one edge per link to a post carrying the token, at the later time."""

    def edges(db):
        tok = token_edges(db, "posts", "Id", col, split)  # post -> token
        e = link_any(db)
        m = e.merge(tok, left_on="dst", right_on="src", suffixes=("", "_tok"))
        t = np.maximum(m["t"].to_numpy(), m["t_tok"].to_numpy())
        return concat_edges(pd.DataFrame({"src": m["dst_tok"], "dst": m["dst"], "t": t}))

    return edges


LINK = LinkConfig(
    edges={
        "link": link,
        "link_any": link_any,
        "tag": tokens("Tags", split_tags),
        "tag_target": token_targets("Tags", split_tags),
        "word": tokens("Title", title_words),
        "word_target": token_targets("Title", title_words),
    },
    direct=["link_any"],
    n_direct=100,
    paths={
        # Posts linked to the posts this one is linked to.
        "two_hop": Path(("link_any", "link_any"), n=100),
        # Often-linked posts in this post's tags / sharing words of its title.
        "tags": Path(("tag", "tag_target"), n=300, max_fanout=2000),
        "words": Path(("word", "word_target"), n=300, max_fanout=2000),
    },
    popularity=["link", "link_any"],
    pop_windows=(91, 365, 1095),
    pop_candidates={"link_any": (3650, 100)},
)
FEATURES = FeatureConfig(windows=(91, 365, None), hop2=False)

if __name__ == "__main__":
    main("rel-stack", "post-post-related", LINK, FEATURES, FEATURES)
