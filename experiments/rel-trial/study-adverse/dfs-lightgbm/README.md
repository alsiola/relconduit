# rel-trial/study-adverse: time-safe relational features + LightGBM

Predict the number of patients with severe adverse events in a study over the next year. Metric: nMAE ↓. Part of
issue 6.

## What it does

1. Features from `relconduit.features.build_features` on `task.get_db()`:
   - **The study's own columns.** `studies` has its own time column (`start_date`), so these are used only when
     `start_date` is strictly before the seed time, and the study's age at the seed time (`studies__start_date__days_before`) is added.
   - **Aggregates** of every table that points at the study (eligibilities, designs, outcomes, drop-outs, the link
     tables, and so on) over 365 and 1,460 days and all time.
   - **Hop 2** through the link tables to the study's conditions, sponsors, interventions and sites, and one link
     further, to those parents' other studies and what those studies reported before the seed time.

   Every aggregate reads only rows strictly before the seed time. That gives 2,338 features.
2. Feature set chosen on val: top 100 by gain (.1169) beat all features (.1191), without hop 2 (.1210) and top 300 or
   1,000.
3. Grid (`relconduit.entity_lgbm.GRID`) on that set. Train on train only, early stopping on val, seeds 0, 1 and 2.
   Test scored once, through `write_prediction_table` + `evaluate_task`. Every test seed is at the test timestamp,
   so there is one test row.

Chosen config: num_leaves 63, min_child_samples 20, feature_fraction 0.8, L1 loss, top 100.

Among the top features: the study's site count, enrollment, phase and sponsor class, and the hop-2 sponsor history
(`sponsors_studies.sponsor_id>>sponsors_studies.nct_id>>reported_event_totals__subjects_affected__mean__all`, the
mean number of subjects affected in the sponsor's earlier studies).

## Run

```sh
uv run python experiments/rel-trial/study-adverse/dfs-lightgbm/run.py --test
```

Ran in the cloud (`n2-custom-24-65536`, 7 threads, alongside other jobs) in 20.2 min, of which feature building took
1.8 min.

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (chose the features, the config and the stopping round, so optimistic) | .1152 ± .0005 | .1155 / .1147 / .1156 |
| **test** | **.1078 ± .0005** | .1083 / .1075 / .1077 |

Published (test):
- official LightGBM .129 (our reproduction .126 ± .004, `experiments/rel-trial/baselines`);
- GNN .129;
- RT-J .103;
- KumoRelational .094;
- leader Kapso .087.

**Where it sits:** .021 better than the official LightGBM and the GNN, but .005 behind RT-J (about ten seed stds,
so not a tie), and .021 behind the leader.
