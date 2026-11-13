"""rel-avito/ad-ctr: LightGBM on time-safe relational features. See README.md."""

import argparse
import json

from relconduit.entity_lgbm import records, run
from relconduit.features import FeatureConfig
from relconduit.results import append_record, git_commit

DATASET = "rel-avito"
TASK = "ad-ctr"
APPROACH = "experiments/rel-avito/ad-ctr/dfs-lightgbm"
# Windows: 1 day, the 4-day horizon, and all time (the database spans 19 days). Hop 2 over
# all memberships walks to the users who saw the ad and the searches it appeared in; siblings
# pool the history of every ad in the same category and in the same location.
FEATURES = FeatureConfig(
    windows=(1, 4, None),
    hop2_all_via=("UserInfo", "SearchInfo"),
    siblings_via=("CategoryID", "LocationID"),
)
# Feature sets tried on val besides all / no hop 2: the top k by gain of one reference fit.
PRUNE_TOP_K = (100, 300, 1000)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    action="store_true",
    help="after tuning on val, score test once and append the rows to results/",
)
args = parser.parse_args()

# Every test seed sits at the test timestamp, so get_db(upto_test_timestamp=False) gives the
# same features as the main database: one test row.
summary = run(
    DATASET,
    TASK,
    APPROACH,
    feature_config=FEATURES,
    score_test=args.test,
    prune_top_k=PRUNE_TOP_K,
    full_db_variant=False,
)
print(json.dumps({k: summary[k] for k in ("best_config", "val_scores", "test_scores")}, indent=2))
if args.test:
    for row in records(summary, commit=git_commit()):
        append_record(f"results/{DATASET}/{TASK}.jsonl", row)
