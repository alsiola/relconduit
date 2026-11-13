# rel-avito/ad-ctr: time-safe relational features + LightGBM

Predict an ad's click-through rate over the next 4 days. Metric: nMAE ↓. Part of issue 6.

## What it does

1. Features from `relconduit.features.build_features` on `task.get_db()`:
   - aggregates of the ad's `SearchStream`, `VisitStream` and `PhoneRequestsStream` rows, with each search's columns joined on, over 1 and 4
     days and all time (the database spans 19 days);
   - hop 2 through `UserInfo` and `SearchInfo` (the users who saw the ad, the searches it appeared in);
   - siblings through `CategoryID` and `LocationID`: the pooled history of the other ads in the same category and
     location.

   DuckDB does the joins over `AdsInfo`'s 6M rows. Every aggregate reads only rows strictly before the seed time.
   That gives 709 features.
2. Feature set chosen on val: top 100 by gain (.3157), essentially tied with top 300 (.3157), all features (.3160)
   and without hop 2 (.3159).
3. Grid (`relconduit.entity_lgbm.GRID`), with train on train only (5,100 rows), early stopping on val, and seeds 0,
   1 and 2. Test scored once.

Chosen config: num_leaves 63, min_child_samples 20, feature_fraction 0.8, L1 loss, top 100.

The top features are the ad's own recent impressions and `HistCTR` (1- and 4-day windows), the same counts over the
searches it appeared in, `days_since_last` and `Price`.

## Run

```sh
uv run python experiments/rel-avito/ad-ctr/dfs-lightgbm/run.py --test
```

Ran in the cloud (`n2-custom-24-65536`, 7 threads) in 1.2 min. The val-only run took about 24 min on the Mac,
sharing it with other jobs. The val scores are identical between the two runs.

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (optimistic) | .3147 ± .0008 | .3155 / .3138 / .3147 |
| **test** | **.3518 ± .0016** | .3499 / .3531 / .3523 |

Published (test):
- official LightGBM .426 (our reproduction .425 ± .000);
- GNN .429;
- RT-J .360;
- GelGT .353;
- Kapso .334;
- leader KumoRelational .332.

**Where it sits:** .074 better than the official LightGBM and the GNN, and .008 better than RT-J (about five seed
stds). It ties GelGT and is .020 behind the leader.
