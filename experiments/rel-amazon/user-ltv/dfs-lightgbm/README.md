# rel-amazon/user-ltv: time-safe relational features + LightGBM

Predict how much a customer will spend (the summed price of the products they review) over the next 91 days. Entity
table `customer` (1.85M rows, static); the time-stamped table is `review` (20.9M rows, `review_time`). Train / val /
test: 4,708,383 / 409,792 / 351,885 rows. Metric: nMAE ↓. Part of issue 8.

## What it does

1. Loads the database with `relconduit.db.load_task_db`. It reads the parquet files directly, keeps keys, times and
   low-cardinality columns, and leaves out free text (`review_text`, `summary`, product `title`, `description`,
   `category`, `brand`, `customer_name`). That takes the database from about 25 GB in pandas to 0.4 GB. Rows are cut
   at the test timestamp, as `get_db(upto_test_timestamp=True)` does.
2. Features from `relconduit.features.build_features`:
   - aggregates of the customer's reviews, with each product's `price` joined on, over 30, 91, 182 and 365 days and
     all time (mean, min, max, sum, count, days since last);
   - hop 2: the reviews of the customer's most recently reviewed product.

   That gives 132 features. Every aggregate reads only rows strictly before the seed time.
3. Train on a fixed uniform sample of 500,000 of the 4.7M train rows, to keep each fit to minutes. Val and test are
   scored in full.
4. Feature set chosen on val at a reference config: all features (.2096), essentially tied with top 100 (.2097) and
   without hop 2 (.2099).
5. Grid (`relconduit.entity_lgbm.GRID`): leaves, min child samples, feature fraction, L2 vs L1. Early stopping on
   val, seeds 0, 1 and 2. Test scored once.

Chosen config: num_leaves 63, min_child_samples 20, feature_fraction 0.8, L1 loss, all features. It stops early,
at 115–161 rounds.

The top features are the customer's all-time summed spend, their review counts (all time, 365 and 182 days), summed
spend over 365 days, `days_since_last` and the highest price they have paid.

## Run

```sh
uv run python experiments/rel-amazon/user-ltv/dfs-lightgbm/run.py --test
```

Ran on Google Cloud (`n2-custom-24-196608` Spot, 5 threads, beside the other jobs of the issue 8 run) in 38.8 min,
with feature building under 30 s. Peak RSS 5.2 GB. The driver is `experiments/regression-board/cloud_jobs.py`.

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (optimistic) | .2082 ± .0000 | .2082 / .2082 / .2081 |
| **test** | **.2430 ± .0004** | .2432 / .2434 / .2426 |

Published (test):
- official LightGBM .292 (our reproduction .292 ± .000). That equals the Global Median and Zero baselines:
  upstream's entity-only model predicts close to the median;
- GNN .249;
- RT-J .243;
- RT .242;
- leader Kapso .238.

**Where it sits:** .049 better than the official LightGBM and .006 better than the GNN. It ties RT-J (.0001 apart)
and is .005 behind the leader.
