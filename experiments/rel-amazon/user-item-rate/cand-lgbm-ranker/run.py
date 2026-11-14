"""rel-amazon/user-item-rate: candidates + LightGBM ranker at scale (issue 21). See README.md."""

from relconduit.link_amazon import FEATURES, link_config, load_db
from relconduit.link_big import GRID, BigPlan, main

TASK = "user-item-rate"
PLAN = BigPlan(
    link=link_config(TASK),
    grid=GRID,
    train_rows=400_000,
    train_times=4,  # the 2014-10 to 2015-07 seed times
    db_loader=load_db,
    leader=("Kapso", 0.0231075),
)

if __name__ == "__main__":
    main("rel-amazon", TASK, PLAN, FEATURES, FEATURES)
