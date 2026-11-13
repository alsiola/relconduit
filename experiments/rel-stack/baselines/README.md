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
