# rel-stack baselines: reproducing the official LightGBM

These check our evaluation path on rel-stack before we trust a new result. They use the upstream
`examples/lightgbm_entity.py`, kept verbatim in `experiments/rel-f1/baselines/upstream/`. It uses only the entity
table's own columns (GloVe for text), with 10 Optuna trials tuned on val, trains on train only, and scores test through
`write_prediction_table` + `evaluate_task`. Seeds 42, 43 and 44. Rows are in `results/rel-stack/<task>.jsonl` with
`approach: experiments/rel-stack/baselines`.

Too big for the Mac, so these ran on Google Cloud in the issue 8 run (`n2-custom-24-196608` Spot, 2 threads per seed,
beside the dfs-lightgbm jobs), driven by `experiments/regression-board/cloud_jobs.py`.

## Run

```sh
cd experiments/rel-f1/baselines/upstream
OMP_NUM_THREADS=2 uv run python lightgbm_entity.py --dataset rel-stack --task post-votes --seed 42 --cache_dir ../../../../runs/baselines/rel-stack/post-votes/cache_s42 --pred_dir ../../../../runs/baselines/rel-stack/post-votes/pred_s42
```

## Results (nMAE ↓, mean ± std over 3 seeds)

| Task | val | test | per seed (test) | official LightGBM (test) | runtime per seed | peak RSS | verdict |
|---|---|---|---|---|---|---|---|
| post-votes | .1208 ± .0000 | .1330 ± .0000 | .1330 / .1330 / .1330 | .133 | 7.0 min | 12.9 GB | reproduces |

All three seeds give identical val and test scores, equal to the leaderboard's Global Median and Zero score: the
tuned model predicts a constant (most questions get no upvotes in 91 days).

## Recommendation tasks (issue 12)

Upstream `examples/trivial_recommendation.py` (past visit and global popularity: deterministic, one run each) and
`examples/lightgbm_recommendation.py` (seeds 42, 43 and 44, 10 Optuna trials on val), kept verbatim in
`experiments/rel-f1/baselines/upstream/`. Test is scored through `write_prediction_table` + `evaluate_task`. Rows are in
`results/rel-stack/<task>.jsonl` with `approach: experiments/rel-stack/baselines` and a `baseline` field.

```sh
cd experiments/rel-f1/baselines/upstream
uv run python trivial_recommendation.py --dataset rel-stack --task user-post-comment --pred_dir ../../../../runs/baselines/rel-stack/user-post-comment/trivial
OMP_NUM_THREADS=1 uv run python lightgbm_recommendation.py --dataset rel-stack --task user-post-comment --seed 42 --cache_dir ../../../../runs/bl_cache --pred_dir ../../../../runs/baselines/rel-stack/user-post-comment/lgbm_s42
```

Issue 12 ran them through `experiments/rec-board/cloud_jobs.py` on Google Cloud Spot machines. The trivial baselines
ran on an `n2-custom-24-131072` (1 thread). LightGBM ran on rel-f1 and rel-stack user-post-comment on the same machine
type and on the other tasks on an `n2-custom-24-196608` (4–5 threads, beside other jobs). "Reproduces" means within 5%
of the published value or two seed stds.

The current upstream LightGBM script adds two label features, `num_past_visit` and `global_popularity_fraction`
(counted from labels before each seed time). The published LightGBM row predates them, so ours is expected to be
higher where past visits carry signal.

### user-post-comment (MAP@k)

| baseline | val | test | per seed (test) | leaderboard (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| Past visit | 0.0175 | 0.0132 | deterministic | 0.0138 | 0.1 min | reproduces |
| Global popularity | 0.0003 | 0.0002 | deterministic | 0.0002 | 0.1 min | reproduces |
| LightGBM (3 seeds) | 0.0149 ± 0.0053 | 0.0086 ± 0.0042 | 0.0050 / 0.0076 / 0.0132 | 0.0002 | 25.5 min per seed | **higher** by 0.0084 |

### post-post-related (MAP@k)

| baseline | val | test | per seed (test) | leaderboard (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| Past visit | 0.0000 | 0.0174 | deterministic | 0.0174 | 0.1 min | reproduces |
| Global popularity | 0.0046 | 0.0146 | deterministic | 0.0146 | 0.1 min | reproduces |
| LightGBM (3 seeds) | 0.0069 ± 0.0022 | 0.0130 ± 0.0094 | 0.0237 / 0.0097 / 0.0058 | 0.0168 | 4.5 min per seed | reproduces |

Every baseline is near zero here. LightGBM's three seeds spread from .006 to .024 and cover the published .017.
