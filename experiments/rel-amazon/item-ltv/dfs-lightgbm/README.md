# rel-amazon/item-ltv: time-safe relational features + LightGBM

Predict a product's revenue (its price times the reviews it gets) over the next 91 days. Entity table `product`
(506k rows, static); the time-stamped table is `review` (20.9M rows, `review_time`). Train / val / test: 2,707,679 /
166,978 / 178,334 rows. Metric: nMAE ↓. Part of issue 8.

## What it does

1. Loads the database with `relconduit.db.load_task_db`, which leaves out the free text (25 GB → 0.4 GB). See
   `experiments/rel-amazon/user-ltv/dfs-lightgbm/README.md`.
2. Features from `relconduit.features.build_features`:
   - the product's `price`;
   - aggregates of its reviews over 30, 91, 182 and 365 days and all time;
   - hop 2: the reviews of its most recent reviewer.

   That gives 107 features. Every aggregate reads only rows strictly before the seed time.
3. Train on a fixed uniform sample of 500,000 of the 2.7M train rows. Val and test are scored in full.
4. Feature set chosen on val at a reference config: without hop 2 (.0687), tied with top 100 (.0687) and just ahead
   of all features (.0689). That leaves 54 features.
5. Grid (`relconduit.entity_lgbm.GRID`), early stopping on val, seeds 0, 1 and 2. Test scored once.

Chosen config: num_leaves 63, min_child_samples 20, feature_fraction 0.3, L1 loss, no hop 2. It runs 1,969–2,284
rounds, under the 3,000 cap. L1 matters a lot here: the best L2 config scores about .086 on val.

The top features are `price`, `days_since_last`, the summed ratings and review counts over every window, and the
count of verified reviews.

## Run

```sh
uv run python experiments/rel-amazon/item-ltv/dfs-lightgbm/run.py --test
```

Ran on Google Cloud (`n2-custom-24-196608` Spot, 5 threads, beside the other jobs of the issue 8 run) in 57.0 min,
with feature building under 30 s. Peak RSS 4.7 GB.

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (optimistic) | .0682 ± .0002 | .0684 / .0680 / .0681 |
| **test** | **.0714 ± .0001** | .0714 / .0716 / .0713 |

Published (test):
- official LightGBM .103 (our reproduction in `experiments/rel-amazon/baselines`);
- GNN .085;
- RT-J .072;
- RT-PluRel .073;
- leader Kapso .066.

**Where it sits:** .031 better than the official LightGBM and .014 better than the GNN. It is .0005 ahead of RT-J,
about five of our seed stds, but RT-J's own seed spread isn't published, so call it level with RT-J. It is .006
behind the leader.
