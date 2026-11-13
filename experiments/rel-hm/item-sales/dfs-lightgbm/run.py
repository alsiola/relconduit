"""rel-hm/item-sales: LightGBM on time-safe relational features. See README.md."""

from relconduit.big_task import main
from relconduit.features import FeatureConfig

# An article's own attributes and its transactions (the label is the sum of their prices in
# the next 7 days) with each buyer's age and club flags, in weekly-aligned windows; hop 2: the
# purchases of the article's most recent buyer.
FEATURES = FeatureConfig(windows=(7, 14, 28, 91, None), aggs=("mean", "min", "max", "sum"))

if __name__ == "__main__":
    main("rel-hm", "item-sales", FEATURES)
