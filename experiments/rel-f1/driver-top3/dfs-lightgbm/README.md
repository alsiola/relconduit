# rel-f1/driver-top3: time-safe relational features + LightGBM

Predict whether the driver qualifies in the top 3 in the next 30 days. Metric: AUROC ↑. Part of issue 3.

## What it does

1. Features from `relconduit.features.build_features` on `task.get_db()`: the driver's own columns
   (age from `dob`, nationality), aggregates of `results`, `qualifying` and `standings` (with
   each row's race `year` and `round` joined on), and hop-2 aggregates through the driver's
   current constructor (its `results`, `qualifying`, `constructor_results` and
   `constructor_standings`). Windows are 30, 90 and 365 days and all time. Every aggregate reads
   only rows strictly before the seed time; `tests/test_features.py` checks that on rel-f1.
   That gives 839 features.
2. Fixed grid (`relconduit.entity_lgbm.GRID`): num_leaves {15, 63} × min_child_samples {20, 100} ×
   feature_fraction {0.3, 0.8} × feature set {all, without hop 2} (× L2/L1 loss for regression).
   Fit on train only, with early stopping on val. Seeds 0, 1, 2. The config with the best mean
   val score wins, so feature selection is decided on val too.
3. Test scored once with that config, through `write_prediction_table` + `evaluate_task`.

Chosen config: num_leaves 15, min_child_samples 100, feature_fraction 0.3, 319 features (hop 2 dropped). Hop-2 features hurt on val (.897 with, .902 without), so val dropped them.

## Run

```sh
uv run python experiments/rel-f1/driver-top3/dfs-lightgbm/run.py         # tune on val only
uv run python experiments/rel-f1/driver-top3/dfs-lightgbm/run.py --test  # then score test once, append to results/
```

The `--test` run took 2.2 min on the Mac (the three tasks ran in parallel, feature building included).

## Results (3 seeds, mean ± std)

| | AUROC ↑ | per seed |
|---|---|---|
| val (chose the config and the stopping round, so optimistic) | .902 ± .002 | |
| **test, `get_db(upto_test_timestamp=True)`: main result** | **.832 ± .033** | .843 / .795 / .858 |
| test, `get_db(upto_test_timestamp=False)`: leaderboard-allowed variant | .928 ± .001 | .929 / .927 / .929 |

Published: official LightGBM .649 (our reproduction .768 ± .041, see
`experiments/rel-f1/baselines`), GNN .680, RT-J .917, leader Kapso .936. Issue target: AUROC above .85.

- **Main result:** Well above the official LightGBM (.649) and the GNN (.680), but below RT-J and the .85 target, with high seed variance.
- **`upto_test_timestamp=False`:** Beats RT-J by .011 (just above noise) and sits .008 behind the leader (within noise, so not a claimed win). Meets the target (> .85).

## Why the two test rows differ so much

The test seeds run from 2010 to 2016. With `upto_test_timestamp=True` the database stops at
2010-01-01, so for a 2013 seed every recency feature is computed against a history that ended
three years earlier: `days_since_last` is over 1,000, and the 30- and 90-day windows are empty.
In training that pattern means a driver who has stopped racing, so the model predicts that for
everyone. These features are built around recency, which makes them fragile under a frozen
database. The upstream LightGBM uses no history at all, so it isn't affected. With the full
database the features see each test seed's true recent past (still strictly before the seed
time), and the val-level quality carries over.

A follow-up could anchor recency features at `min(seed_time, db cutoff)` in the frozen setting.
That has to be chosen without looking at test again.
