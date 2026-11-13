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
