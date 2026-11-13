# rel-trial baselines: reproducing the official LightGBM

These check our evaluation path on rel-trial before we trust a new result. They use the upstream
`examples/lightgbm_entity.py`, kept verbatim in `experiments/rel-f1/baselines/upstream/`. It uses only the entity
table's own columns (GloVe for text), with 10 Optuna trials tuned on val, trains on train only, and scores test through
`write_prediction_table` + `evaluate_task`. Seeds 42, 43 and 44. Rows are in `results/rel-trial/<task>.jsonl` with
`approach: experiments/rel-trial/baselines`.

## Run

Locally, one seed:

```sh
cd experiments/rel-f1/baselines/upstream
OMP_NUM_THREADS=1 uv run python lightgbm_entity.py --dataset rel-trial --task study-adverse --seed 42 --pred_dir ../../../../runs/baselines/rel-trial/study-adverse/pred_s42
```

Issue 6 ran them through `experiments/rel-trial/cloud_jobs.py` on one cloud machine, five seeds side by side at 2
threads each.

## Results (nMAE ↓, mean ± std over 3 seeds)

| Task | val | test | per seed (test) | official LightGBM (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| study-adverse | .132 ± .003 | .126 ± .004 | .130 / .123 / .125 | .129 | 52–124 min per seed (cloud, 2 threads) | reproduces (within seed noise) |
| site-success | .876 ± .008 | .890 ± .002 | .889 / .890 / .893 | .880 | seed 42: 54 min (Mac, 1 thread); seeds 43 and 44: 11–15 min (cloud, 2 threads) | close: .010 worse, about five seed stds |

On site-success, all three seeds land .009 to .013 above the published .880. That is the same direction and size
as rel-f1 driver-position (.591 against .602), so it is probably Optuna tuning differences or a RelBench release
difference. It is small next to the gaps that matter here (the dfs-lightgbm result is .931, and the GNN is .781).

Study-adverse seeds took 52 to 124 min each in the cloud, much longer than site-success. The Optuna trials with very
small learning rates hit the 2,000-round cap.

## Recommendation tasks (issue 12)

Upstream `examples/trivial_recommendation.py` (past visit and global popularity: deterministic, one run each) and
`examples/lightgbm_recommendation.py` (seeds 42, 43 and 44, 10 Optuna trials on val), kept verbatim in
`experiments/rel-f1/baselines/upstream/`. Test is scored through `write_prediction_table` + `evaluate_task`. Rows are in
`results/rel-trial/<task>.jsonl` with `approach: experiments/rel-trial/baselines` and a `baseline` field.

```sh
cd experiments/rel-f1/baselines/upstream
uv run python trivial_recommendation.py --dataset rel-trial --task condition-sponsor-run --pred_dir ../../../../runs/baselines/rel-trial/condition-sponsor-run/trivial
OMP_NUM_THREADS=1 uv run python lightgbm_recommendation.py --dataset rel-trial --task condition-sponsor-run --seed 42 --cache_dir ../../../../runs/bl_cache --pred_dir ../../../../runs/baselines/rel-trial/condition-sponsor-run/lgbm_s42
```

Issue 12 ran them through `experiments/rec-board/cloud_jobs.py` on Google Cloud Spot machines. The trivial baselines
ran on an `n2-custom-24-131072` (1 thread). LightGBM ran on rel-f1 and rel-stack user-post-comment on the same machine
type and on the other tasks on an `n2-custom-24-196608` (4–5 threads, beside other jobs). "Reproduces" means within 5%
of the published value or two seed stds.

The current upstream LightGBM script adds two label features, `num_past_visit` and `global_popularity_fraction`
(counted from labels before each seed time). The published LightGBM row predates them, so ours is expected to be
higher where past visits carry signal.

### condition-sponsor-run (MAP@k)

| baseline | val | test | per seed (test) | leaderboard (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| Past visit | 0.0774 | 0.0806 | deterministic | 0.0830 | 0.1 min | reproduces |
| Global popularity | 0.0263 | 0.0252 | deterministic | 0.0252 | 0.1 min | reproduces |
| LightGBM (3 seeds) | 0.0688 ± 0.0026 | 0.0709 ± 0.0031 | 0.0684 / 0.0744 / 0.0699 | 0.0480 | 79.8 min per seed | **higher** by 0.0229 |

### site-sponsor-run (MAP@k)

| baseline | val | test | per seed (test) | leaderboard (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| Past visit | 0.1560 | 0.1723 | deterministic | 0.1734 | 0.8 min | reproduces |
| Global popularity | 0.0491 | 0.0375 | deterministic | 0.0375 | 0.8 min | reproduces |
| LightGBM (3 seeds) | 0.1829 ± 0.0090 | 0.1664 ± 0.0149 | 0.1825 / 0.1634 / 0.1532 | 0.0904 | 120.3 min per seed | **higher** by 0.0760 |
