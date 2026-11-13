# rel-hm/item-sales: time-safe relational features + LightGBM

Predict an article's sales (the summed price of its transactions) over the next 7 days. Entity table `article`
(105,542 rows, static); the time-stamped table is `transactions` (15.5M rows, `t_dat`). Train / val / test:
5,488,184 / 105,542 / 105,542 rows. Metric: nMAE ↓. Part of issue 8.

## What it does

1. Loads the database with `relconduit.db.load_task_db`. It leaves out the free-text and high-cardinality columns
   (`prod_name`, `department_name`, `detail_desc`, `postal_code`) and cuts rows at the test timestamp.
2. Features from `relconduit.features.build_features`:
   - the article's own attributes (section, colour, garment group, and so on);
   - aggregates of its transactions, with each buyer's age and club flags joined on, over 7, 14, 28 and 91 days and
     all time (weekly-aligned windows, matching the 7-day label);
   - hop 2: the purchases of the article's most recent buyer.

   That gives 204 features. Every aggregate reads only rows strictly before the seed time.
3. Train on a fixed uniform sample of 500,000 of the 5.5M train rows. Val and test are scored in full.
4. Feature set chosen on val at a reference config: top 100 by gain (.0915), essentially tied with without hop 2
   (.0916) and all features (.0918).
5. Grid (`relconduit.entity_lgbm.GRID`), early stopping on val, seeds 0, 1 and 2. Test scored once.

Chosen config: num_leaves 63, min_child_samples 20, feature_fraction 0.3, L1 loss, top 100.

The top features are the article's last 14 days of transactions (count, summed price, summed `sales_channel_id`,
summed buyer age), its 7- and 28-day sums, `days_since_last`, and its section, colour group and garment group.

## Run

```sh
uv run python experiments/rel-hm/item-sales/dfs-lightgbm/run.py --test
```

Ran on Google Cloud (`n2-custom-24-196608` Spot, 5 threads, beside the other jobs of the issue 8 run) in 51.0 min,
with feature building under 40 s. Peak RSS 5.8 GB.

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (optimistic) | .0911 ± .0002 | .0910 / .0913 / .0909 |
| **test** | **.0752 ± .0002** | .0751 / .0755 / .0751 |

Test is lower than val on every model here, the official baseline included (.154 test, .173 val): that's the split,
not a sign of leakage.

Published (test):
- official LightGBM .154 (our reproduction .154 ± .000). That equals the Global Median and Zero baselines;
- GNN .112;
- RT-J .076;
- RT-PluRel .075;
- leader Kapso .063.

**Where it sits:** about half the official LightGBM's error, and .037 better than the GNN. It ties RT-PluRel (.0001
apart) and is .0005 ahead of RT-J. It is .012 behind the leader.
