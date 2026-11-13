"""rel-amazon/item-ltv: LightGBM on time-safe relational features. See README.md."""

from relconduit.big_task import main
from relconduit.features import FeatureConfig

# A product's price and its reviews (the label is price times the reviews in the next 91
# days), and hop 2: the reviews of the product's most recent reviewer.
FEATURES = FeatureConfig(windows=(30, 91, 182, 365, None), aggs=("mean", "min", "max", "sum"))

if __name__ == "__main__":
    main("rel-amazon", "item-ltv", FEATURES)
