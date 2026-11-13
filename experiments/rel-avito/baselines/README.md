# rel-avito baselines: reproducing the official LightGBM

These check our evaluation path on rel-avito before we trust a new result. They use the upstream
`examples/lightgbm_entity.py`, kept verbatim in `experiments/rel-f1/baselines/upstream/`. It uses only the entity
table's own columns (GloVe for text), with 10 Optuna trials tuned on val, trains on train only, and scores test through
`write_prediction_table` + `evaluate_task`. Seeds 42, 43 and 44. Rows are in `results/rel-avito/<task>.jsonl` with
`approach: experiments/rel-avito/baselines`.

## Run

```sh
cd experiments/rel-f1/baselines/upstream
OMP_NUM_THREADS=1 uv run python lightgbm_entity.py --dataset rel-avito --task ad-ctr --seed 42 --cache_dir ../../../../runs/baselines/rel-avito/ad-ctr/cache_s42 --pred_dir ../../../../runs/baselines/rel-avito/ad-ctr/pred_s42
```

## Results (nMAE ↓, mean ± std over 3 seeds)

| Task | val | test | per seed (test) | official LightGBM (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| ad-ctr | .390 ± .001 | .425 ± .000 | .425 / .426 / .425 | .426 | 7.7 min per seed, Mac, single thread | reproduces |

## Recommendation tasks (issue 12)

Upstream `examples/trivial_recommendation.py` (past visit and global popularity: deterministic, one run each) and
`examples/lightgbm_recommendation.py` (seeds 42, 43 and 44, 10 Optuna trials on val), kept verbatim in
`experiments/rel-f1/baselines/upstream/`. Test is scored through `write_prediction_table` + `evaluate_task`. Rows are in
`results/rel-avito/<task>.jsonl` with `approach: experiments/rel-avito/baselines` and a `baseline` field.

```sh
cd experiments/rel-f1/baselines/upstream
uv run python trivial_recommendation.py --dataset rel-avito --task user-ad-visit --pred_dir ../../../../runs/baselines/rel-avito/user-ad-visit/trivial
OMP_NUM_THREADS=1 uv run python lightgbm_recommendation.py --dataset rel-avito --task user-ad-visit --seed 42 --cache_dir ../../../../runs/bl_cache --pred_dir ../../../../runs/baselines/rel-avito/user-ad-visit/lgbm_s42
```

Issue 12 ran them through `experiments/rec-board/cloud_jobs.py` on Google Cloud Spot machines. The trivial baselines
ran on an `n2-custom-24-131072` (1 thread). LightGBM ran on rel-f1 and rel-stack user-post-comment on the same machine
type and on the other tasks on an `n2-custom-24-196608` (4–5 threads, beside other jobs). "Reproduces" means within 5%
of the published value or two seed stds.

The current upstream LightGBM script adds two label features, `num_past_visit` and `global_popularity_fraction`
(counted from labels before each seed time). The published LightGBM row predates them, so ours is expected to be
higher where past visits carry signal.

### user-ad-visit (MAP@k)

| baseline | val | test | per seed (test) | leaderboard (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| Past visit | 0.0341 | 0.0188 | deterministic | 0.0193 | 0.4 min | reproduces |
| Global popularity | 0.0001 | 0.0000 | deterministic | 0.0000 | 0.4 min | reproduces |
| LightGBM (3 seeds) | 0.0273 ± 0.0004 | 0.0141 ± 0.0004 | 0.0139 / 0.0139 / 0.0146 | 0.0006 | 177.6 min per seed | **higher** by 0.0135 |
