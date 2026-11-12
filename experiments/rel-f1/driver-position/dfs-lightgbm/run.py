"""rel-f1/driver-position: LightGBM on time-safe relational features. See README.md."""

import argparse
import json

from relconduit.entity_lgbm import records, run
from relconduit.results import append_record, git_commit

TASK = "driver-position"
APPROACH = "experiments/rel-f1/driver-position/dfs-lightgbm"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    action="store_true",
    help="after tuning on val, score test once and append the rows to results/",
)
args = parser.parse_args()

summary = run("rel-f1", TASK, APPROACH, score_test=args.test)
print(json.dumps({k: summary[k] for k in ("best_config", "val_scores", "test_scores")}, indent=2))
if args.test:
    for row in records(summary, commit=git_commit()):
        append_record(f"results/rel-f1/{TASK}.jsonl", row)
