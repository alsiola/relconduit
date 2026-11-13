"""rel-amazon/user-ltv: LightGBM on time-safe relational features. See README.md."""

from relconduit.big_task import main
from relconduit.features import FeatureConfig

# A customer's reviews, each with its product's price (the label is the sum of prices of the
# products reviewed in the next 91 days, so the windowed sums are the obvious inputs), and
# hop 2: the reviews of the customer's most recently reviewed product.
FEATURES = FeatureConfig(windows=(30, 91, 182, 365, None), aggs=("mean", "min", "max", "sum"))

if __name__ == "__main__":
    main("rel-amazon", "user-ltv", FEATURES)
