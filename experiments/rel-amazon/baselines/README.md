# rel-amazon baselines: reproducing the official LightGBM

These check our evaluation path on rel-amazon before we trust a new result. They use the upstream
`examples/lightgbm_entity.py`, kept verbatim in `experiments/rel-f1/baselines/upstream/`. It uses only the entity
table's own columns (GloVe for text), with 10 Optuna trials tuned on val, trains on train only, and scores test through
`write_prediction_table` + `evaluate_task`. Seeds 42, 43 and 44. Rows are in `results/rel-amazon/<task>.jsonl` with
`approach: experiments/rel-amazon/baselines`.

Too big for the Mac, so these ran on Google Cloud in the issue 8 run (`n2-custom-24-196608` Spot, 2 threads per seed,
beside the dfs-lightgbm jobs), driven by `experiments/regression-board/cloud_jobs.py`.

Upstream's script holds two copies of the database at once (`task.get_db()`, and `dataset.get_db()` for the stype
proposal), including the 24 GB of `review_text`, so each seed peaks at **46 GB**. The seeds ran one at a time.

## Run

```sh
cd experiments/rel-f1/baselines/upstream
OMP_NUM_THREADS=2 uv run python lightgbm_entity.py --dataset rel-amazon --task user-ltv --seed 42 --cache_dir ../../../../runs/baselines/rel-amazon/user-ltv/cache_s42 --pred_dir ../../../../runs/baselines/rel-amazon/user-ltv/pred_s42
```

## Results (nMAE ↓, mean ± std over 3 seeds)

| Task | val | test | per seed (test) | official LightGBM (test) | runtime per seed | peak RSS | verdict |
|---|---|---|---|---|---|---|---|
| user-ltv | .2459 ± .0000 | .2919 ± .0000 | .2919 / .2919 / .2919 | .292 | 4.2 min | 46.3 GB | reproduces |
| item-ltv | .0942 ± .0000 | .1024 ± .0001 | .1024 / .1025 / .1024 | .103 | 53.3 min | 46.3 GB | reproduces |

On user-ltv the tuned model predicts close to the global median: .2919 is also the leaderboard's Global Median and
Zero score. So the seeds agree to four decimals.
