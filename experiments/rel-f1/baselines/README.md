# rel-f1 baselines: reproducing the official trivial and LightGBM numbers

Purpose: confirm our evaluation path (RelBench 3.1, `task.evaluate` on val,
`write_prediction_table` + `relbench.submit.evaluate_task` on test) before trusting any new
result. `upstream/` holds the upstream example scripts
(`snap-stanford/relbench@main:examples/`) verbatim. They are excluded from ruff so the copy stays
exact.

## How to run

```sh
cd experiments/rel-f1/baselines/upstream
uv run python trivial_entity.py --dataset rel-f1 --task driver-position --pred_dir ../../../../runs/baselines/trivial
# LightGBM: OMP_NUM_THREADS=1 is required on Apple Silicon. Without it torch's and LightGBM's
# OpenMP runtimes collide and the process segfaults during Optuna tuning.
OMP_NUM_THREADS=1 uv run python lightgbm_entity.py --dataset rel-f1 --task driver-position --seed 42 --pred_dir ../../../../runs/baselines/lgbm_s42
```

The LightGBM script uses only the `drivers` table's own columns (GloVe embeddings for the text
columns), with 10 Optuna trials tuned on val. It trains on train only.

## Results (test unless marked; Mac, single thread)

LightGBM, seeds 42, 43, 44, mean ± std. About 8.5 minutes per task per 3 seeds (25.5 min total).

| Task | Metric | Ours: val | Ours: test | Official LightGBM (test) | Verdict |
|---|---|---|---|---|---|
| driver-position | nMAE ↓ | .496 ± .002 | .591 ± .004 | .602 | reproduces (within ~.01) |
| driver-dnf | AUROC ↑ | .683 ± .018 | .712 ± .019 | .725 | reproduces (within seed noise) |
| driver-top3 | AUROC ↑ | .703 ± .022 | .768 ± .041 | .649 | **higher** than official, high variance |

Per seed (test): position .591 / .586 / .595; dnf .702 / .734 / .700; top3 .814 / .736 / .755.

driver-top3 has only 1,353 train rows and 59 train timestamps, and the Optuna search lands in
quite different places per seed (test range .736–.814). Our three seeds all beat the published
.649, so the published number looks like an unlucky draw, or it came from an older RelBench
release. Either way the evaluation path agrees with the official one on the other two tasks, and
the comparison for our approach is made against both numbers.

Trivial baselines (deterministic except `random`; test):

| Task | Metric | global_mean | global_median | entity_mean | entity_median | random | majority |
|---|---|---|---|---|---|---|---|
| driver-position | nMAE ↓ | .642 | .626 | 1.210 | 1.212 | | |
| driver-dnf | AUROC ↑ | | | .447 | .428 | .509 | .500 |
| driver-top3 | AUROC ↑ | | | .704 | .620 | .404 | .500 |

(The trivial script's test predictions use train+val statistics, as upstream does.)

LightGBM rows are in `results/rel-f1/<task>.jsonl` with `approach: experiments/rel-f1/baselines`.
