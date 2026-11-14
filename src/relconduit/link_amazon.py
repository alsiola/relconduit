"""rel-amazon's database, edge sets and candidate config for its three recommendation tasks.

All three tasks ask which products a customer reviews in the next 91 days: any review
(``user-item-purchase``), a 5-star one (``user-item-rate``), or one longer than 300 characters
(``user-item-review``). So they share one edge set of reviews, and each adds its own target
edges (``five``, ``long``) as direct relations, popularity and popular-candidate sources.

:func:`load_db` is :func:`relconduit.db.load_task_db` with the review length kept
(``review_text_len``, the column ``user-item-review``'s label is about) and three static codes
on ``product`` from columns the slim loader leaves out: ``brand_id`` (in this Books catalogue,
mostly the author: "Visit Amazon's Agatha Christie Page"), ``leaf_id`` (the last element of the
category list) and ``cat2_id`` (its first two). Codes are factorised over the whole static
product table, so they don't depend on time. They feed walks only: :data:`FEATURES` leaves
them out of the entity features, where the numbers would mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relconduit.features import FeatureConfig
from relconduit.link import NEVER, LinkConfig, Path, concat_edges, table_edges

STATIC_CODES = ("brand_id", "leaf_id", "cat2_id")
TARGETS = {"user-item-purchase": None, "user-item-rate": "five", "user-item-review": "long"}


def product_codes(meta: pd.DataFrame) -> pd.DataFrame:
    """``product_id`` and the static codes of :data:`STATIC_CODES` (-1 where unknown)."""

    def last(x):
        return x[-1] if x is not None and len(x) else None

    def first2(x):
        return ">".join(x[:2]) if x is not None and len(x) else None

    out = pd.DataFrame({"product_id": meta["product_id"].to_numpy()})
    out["brand_id"] = pd.factorize(meta["brand"], sort=True)[0]
    out["leaf_id"] = pd.factorize(meta["category"].map(last), sort=True)[0]
    out["cat2_id"] = pd.factorize(meta["category"].map(first2), sort=True)[0]
    return out


def load_db(task, **kwargs):
    from relconduit.db import load_task_db

    kwargs.setdefault("text_lengths", {"review": ["review_text"]})
    db = load_task_db(task, **kwargs)
    meta = pd.read_parquet(
        f"{task.dataset.db_dir}/product.parquet", columns=["product_id", "category", "brand"]
    )
    prod = db.table_dict["product"]
    prod.df = prod.df.merge(product_codes(meta), on="product_id", how="left")
    return db


def review(db):
    return table_edges(db, "review", "customer_id", "product_id")


def _reviews_where(db, mask_fn):
    r = db.table_dict["review"].df
    r = r[mask_fn(r)]
    return concat_edges(
        pd.DataFrame({"src": r["customer_id"], "dst": r["product_id"], "t": r["review_time"]})
    )


def five(db):
    """customer -> product, 5-star reviews (``user-item-rate``'s label rows)."""
    return _reviews_where(db, lambda r: r["rating"] == 5.0)


def long(db):
    """customer -> product, reviews over 300 characters (``user-item-review``'s label rows)."""
    return _reviews_where(db, lambda r: r["review_text_len"] > 300)


def _static(db, code: str):
    p = db.table_dict["product"].df
    p = p[p[code] >= 0]
    return concat_edges(
        pd.DataFrame({"src": p["product_id"], "dst": p[code], "t": np.repeat(NEVER, len(p))})
    )


def _code_reviews(db, code: str):
    """code -> product, one edge per review of a product with that code, at its time."""
    r = db.table_dict["review"].df[["product_id", "review_time"]]
    p = db.table_dict["product"].df[["product_id", code]]
    m = r.merge(p[p[code] >= 0], on="product_id")
    return concat_edges(
        pd.DataFrame({"src": m[code], "dst": m["product_id"], "t": m["review_time"]})
    )


def edges() -> dict:
    return {
        "review": review,
        "five": five,
        "long": long,
        "prod_brand": lambda db: _static(db, "brand_id"),
        "brand_reviews": lambda db: _code_reviews(db, "brand_id"),
        "prod_leaf": lambda db: _static(db, "leaf_id"),
        "leaf_reviews": lambda db: _code_reviews(db, "leaf_id"),
    }


def link_config(task_name: str) -> LinkConfig:
    """Candidates for one of the three tasks (see the module doc)."""
    target = TARGETS[task_name]
    tgt = [target] if target else []
    names = ["review", *tgt]
    e = edges()
    keep = {"review", "prod_brand", "brand_reviews", "prod_leaf", "leaf_reviews", *tgt}
    pop_cands = {"review": (30, 30)}
    if target:
        pop_cands[target] = (30, 30)
    return LinkConfig(
        edges={n: f for n, f in e.items() if n in keep},
        direct=names,
        n_direct=20,
        paths={
            # Products reviewed by customers who reviewed the same products, recent ones up.
            "covisit": Path(
                ("review", "~review", "review"), n=100, half_life_days=365, max_fanout=300, beam=500
            ),
            # Other books by the authors (brands) the customer reviewed, popular in the last year.
            "brand": Path(
                ("review", "prod_brand", "brand_reviews"), n=40, window_days=(None, None, 365)
            ),
            # Books in the customer's leaf categories, popular in the last 91 days.
            "leaf": Path(
                ("review", "prod_leaf", "leaf_reviews"), n=30, window_days=(None, None, 91)
            ),
        },
        popularity=names,
        pop_windows=(30, 91, 365),
        pop_candidates=pop_cands,
    )


FEATURES = FeatureConfig(
    windows=(91, 365, None),
    hop2=False,
    exclude_columns=[("product", c) for c in STATIC_CODES],
)
