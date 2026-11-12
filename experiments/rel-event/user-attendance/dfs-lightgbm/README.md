# rel-event/user-attendance: time-safe relational features + LightGBM

Predict how many events a user will respond "yes" or "maybe" to in the next 7 days. Metric: nMAE ↓. Part of
issue 6.

## What it does

1. Features from `relconduit.features.build_features` on `task.get_db()`. The entity table `users` has its own
   time column (`joinedAt`), so the user's own columns are used only when `joinedAt` is strictly before the seed
   time, and `users__joinedAt__days_before` (the user's age at the seed time) is added. Aggregates of
   `event_attendees`, `event_interest` and the user's own `events` (with each event's columns joined on) over 7 and
   28 days and all time, plus hop 2 through `user_friends` to the friends' activity. Every aggregate reads only rows
   strictly before the seed time. That gives 5,142 features.
2. Feature set chosen on val (stage 1, one reference config): all, without hop 2, or the top 100 / 300 / 1000 by
   gain of a single reference fit on train. Top 100 won (.3244 vs .3276 for all, .3271 without hop 2).
3. Fixed grid (`relconduit.entity_lgbm.GRID`) on that set: num_leaves {15, 63} × min_child_samples {20, 100} ×
   feature_fraction {0.3, 0.8} × L2/L1 loss. Fit on train only, with early stopping on val. Seeds 0, 1, 2.
4. Test scored once with the chosen config, through `write_prediction_table` + `evaluate_task`. Every test seed is at
   the test timestamp, so `upto_test_timestamp=False` gives the same database: one test row.

Chosen config: num_leaves 63, min_child_samples 100, feature_fraction 0.8, L1 loss, top 100 features. L1 loss beat L2
by a wide margin on val, as on rel-f1.

## Run

```sh
uv run python experiments/rel-event/user-attendance/dfs-lightgbm/run.py         # tune on val only
uv run python experiments/rel-event/user-attendance/dfs-lightgbm/run.py --test  # then score test once
```

The `--test` run took 37.7 min on the Mac (features 1.7 min, then 63 fits).

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (chose the features, the config and the stopping round, so optimistic) | .324 ± .001 | .323 / .325 / .323 |
| **test** | **.332 ± .001** | .331 / .332 / .333 |

Published (test): official LightGBM .344 (our reproduction .345 ± .000, `experiments/rel-event/baselines`), GNN .326,
RT-J .316, leader GelGT .306.

**Where it sits:** it beats the official LightGBM by .012, more than ten seed stds. It is .006 behind the GNN,
.016 behind RT-J and .026 behind the leader. So the rel-f1 result (RT-J level) does not carry over to this task.
The top features are almost all the user's own past responses (counts by status, and the columns of the events they
responded to). Friends' activity added little: dropping hop 2 cost only .0005 on val.
