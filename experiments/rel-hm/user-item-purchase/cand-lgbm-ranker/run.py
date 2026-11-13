"""rel-hm/user-item-purchase: candidates + LightGBM ranker at scale (issue 21). See README.md."""

import pandas as pd

from relconduit.features import FeatureConfig
from relconduit.link import LinkConfig, Path, concat_edges, table_edges
from relconduit.link_big import GRID, BigPlan, main


def buy(db):
    return table_edges(db, "transactions", "customer_id", "article_id")


def _code(col):
    def static(db):
        return table_edges(db, "article", "article_id", col)

    def buys(db):
        """code -> article, one edge per purchase of an article with that code, at its time."""
        tr = db.table_dict["transactions"].df[["article_id", "t_dat"]]
        art = db.table_dict["article"].df[["article_id", col]]
        m = tr.merge(art, on="article_id").dropna()
        return concat_edges(pd.DataFrame({"src": m[col], "dst": m["article_id"], "t": m["t_dat"]}))

    return static, buys


art_prod, prod_buys = _code("product_code")
art_type, type_buys = _code("product_type_no")

LINK = LinkConfig(
    edges={
        "buy": buy,
        "art_prod": art_prod,
        "prod_buys": prod_buys,
        "art_type": art_type,
        "type_buys": type_buys,
    },
    direct=["buy"],
    n_direct=30,  # repurchases
    paths={
        # Articles bought by customers who bought the same articles, the last weeks up.
        "covisit": Path(("buy", "~buy", "buy"), n=50, half_life_days=14, max_fanout=300),
        # Other colours / sizes of the products the customer bought, bought in the last 7 days.
        "product": Path(
            ("buy", "art_prod", "prod_buys"), n=30, half_life_days=60, window_days=(None, None, 7)
        ),
        # Best sellers of the last 7 days in the customer's product types.
        "type": Path(
            ("buy", "art_type", "type_buys"), n=30, half_life_days=60, window_days=(None, None, 7)
        ),
    },
    popularity=["buy"],
    pop_windows=(7, 14, 28, 91),
    pop_candidates={"buy": (7, 60)},  # the 7-day window
)
FEATURES = FeatureConfig(windows=(7, 28, 91, None), hop2=False)
PLAN = BigPlan(
    link=LINK,
    grid=GRID,
    train_rows=120_000,  # ~300 features: about 40k kept seeds, 6 GB of pairs
    train_times=8,  # the last eight weekly seed times
    leader=("Kapso", 0.0325602),
)

if __name__ == "__main__":
    main("rel-hm", "user-item-purchase", PLAN, FEATURES, FEATURES)
