"""rel-avito/user-ad-visit: candidates + LightGBM ranker. See README.md."""

import pandas as pd

from relconduit.features import FeatureConfig
from relconduit.link import LinkConfig, Path, concat_edges, join_edges, table_edges
from relconduit.link_lgbm import main


def visit(db):
    return table_edges(db, "VisitStream", "UserID", "AdID")


def phone(db):
    return table_edges(db, "PhoneRequestsStream", "UserID", "AdID")


def shown(db):
    """user -> ad shown in one of the user's searches, at the search's time."""
    search = table_edges(db, "SearchInfo", "UserID", "SearchID")
    stream = table_edges(db, "SearchStream", "SearchID", "AdID")
    return join_edges(search, stream)


def ad_category(db):
    return table_edges(db, "AdsInfo", "AdID", "CategoryID")


def category_visits(db):
    """category -> ad, one edge per visit to an ad in the category, at the visit's time."""
    v = db.table_dict["VisitStream"].df[["AdID", "ViewDate"]]
    ads = db.table_dict["AdsInfo"].df[["AdID", "CategoryID"]]
    m = v.merge(ads, on="AdID").dropna()
    return concat_edges(
        pd.DataFrame({"src": m["CategoryID"], "dst": m["AdID"], "t": m["ViewDate"]})
    )


def user_category(db):
    """user -> category of each ad the user visited."""
    return join_edges(visit(db), ad_category(db))


LINK = LinkConfig(
    edges={
        "visit": visit,
        "phone": phone,
        "shown": shown,
        "user_category": user_category,
        "category_visits": category_visits,
    },
    direct=["visit", "phone", "shown"],
    n_direct=40,
    paths={
        # Ads visited by users who visited the same ads, recent visits weighted up.
        "covisit": Path(("visit", "~visit", "visit"), n=60, half_life_days=3, max_fanout=500),
        # Ads popular in the last four days in the categories this user visits.
        "category": Path(("user_category", "category_visits"), n=40, window_days=(None, 4)),
    },
    popularity=["visit", "shown", "phone"],
    pop_windows=(1, 4, 8),
    pop_candidates={"visit": (4, 20)},
)
# Ads have a few static columns (category, location, price, title); users none, so their
# features are their visit, search and phone-request history.
FEATURES = FeatureConfig(windows=(4, 8, None), hop2=False)

if __name__ == "__main__":
    main("rel-avito", "user-ad-visit", LINK, FEATURES, FEATURES, train_sample=20_000)
