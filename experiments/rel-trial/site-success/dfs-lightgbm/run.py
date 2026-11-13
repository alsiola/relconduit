"""rel-trial/site-success: LightGBM on time-safe relational features. See README.md."""

import argparse
import json

from relconduit.entity_encoding import EncodingConfig
from relconduit.entity_lgbm import records, run
from relconduit.features import FeatureConfig
from relconduit.results import append_record, git_commit

DATASET = "rel-trial"
TASK = "site-success"
APPROACH = "experiments/rel-trial/site-success/dfs-lightgbm"
# Windows: the 365-day horizon, 4x it, and all time. Hop 2 over all memberships walks
# from the site to every study it has run and their outcomes, analyses and adverse events.
FEATURES = FeatureConfig(windows=(365, 1460, None), hop2_all_via=("studies",))
# The site's own columns, all text with too many values for features.py (issue 9): counts
# over sites first linked to a study before the seed time, target encoding over train labels
# known by the seed time (by location, by name template, and the site's own past labels),
# country as a categorical, and GloVe embeddings of all five, 32 PCA dims each.
TEXT = ("name", "city", "state", "zip", "country")
ENCODING = EncodingConfig(
    columns=("city", "state", "zip", "country"),
    templates=("name",),
    first_seen_via=("facilities_studies", "facility_id"),
    target_key=True,
    categorical=("country",),
    embed=TEXT,
    embed_dims=32,
)
# Feature sets compared on val in stage 1, besides all / no hop 2 / top k: without the
# encodings (issue 6's features), and without each family of them.
EXTRA_SETS = (
    "drop:enc__",
    "drop:__te",
    "drop:__count,__share",
    "drop:__te,__count,__share",
    "drop:__emb",
)
# Feature sets tried on val besides all / no hop 2: the top k by gain of one reference fit.
PRUNE_TOP_K = (100, 300, 1000)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    action="store_true",
    help="after tuning on val, score test once and append the rows to results/",
)
parser.add_argument("--smoke", type=int, help="smoke test on this many train and val rows")
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
    encoding=ENCODING,
    extra_feature_sets=EXTRA_SETS,
    smoke_rows=args.smoke,
)
print(json.dumps({k: summary[k] for k in ("best_config", "val_scores", "test_scores")}, indent=2))
if args.test:
    notes = "Entity columns encoded by relconduit.entity_encoding (issue 9)."
    for row in records(summary, commit=git_commit(), notes=notes):
        append_record(f"results/{DATASET}/{TASK}.jsonl", row)
