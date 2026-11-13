# rel-event baselines: reproducing the official LightGBM

These check our evaluation path on rel-event before we trust a new result. They use the upstream
`examples/lightgbm_entity.py`, kept verbatim in `experiments/rel-f1/baselines/upstream/`. It uses only the entity
table's own columns (GloVe for text), with 10 Optuna trials tuned on val, trains on train only, and scores test through
`write_prediction_table` + `evaluate_task`. Seeds 42, 43 and 44. Rows are in `results/rel-event/<task>.jsonl` with
`approach: experiments/rel-event/baselines`.

## Run

```sh
cd experiments/rel-f1/baselines/upstream
OMP_NUM_THREADS=1 uv run python lightgbm_entity.py --dataset rel-event --task user-attendance --seed 42 --cache_dir ../../../../runs/baselines/rel-event/user-attendance/cache_s42 --pred_dir ../../../../runs/baselines/rel-event/user-attendance/pred_s42
```

## Results (nMAE ↓, mean ± std over 3 seeds)

| Task | val | test | per seed (test) | official LightGBM (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| user-attendance | .342 ± .000 | .345 ± .000 | .344 / .345 / .344 | .344 | 1.2 min per seed, Mac, single thread | reproduces |
