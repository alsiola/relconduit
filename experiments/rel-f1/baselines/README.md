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

## Recommendation tasks (issue 12)

Upstream `examples/trivial_recommendation.py` (past visit and global popularity: deterministic, one run each) and
`examples/lightgbm_recommendation.py` (seeds 42, 43 and 44, 10 Optuna trials on val), kept verbatim in
`experiments/rel-f1/baselines/upstream/`. Test is scored through `write_prediction_table` + `evaluate_task`. Rows are in
`results/rel-f1/<task>.jsonl` with `approach: experiments/rel-f1/baselines` and a `baseline` field.

```sh
cd experiments/rel-f1/baselines/upstream
uv run python trivial_recommendation.py --dataset rel-f1 --task driver-circuit-compete --pred_dir ../../../../runs/baselines/rel-f1/driver-circuit-compete/trivial
OMP_NUM_THREADS=1 uv run python lightgbm_recommendation.py --dataset rel-f1 --task driver-circuit-compete --seed 42 --cache_dir ../../../../runs/bl_cache --pred_dir ../../../../runs/baselines/rel-f1/driver-circuit-compete/lgbm_s42
```

Issue 12 ran them through `experiments/rec-board/cloud_jobs.py` on Google Cloud Spot machines. The trivial baselines
ran on an `n2-custom-24-131072` (1 thread). LightGBM ran on rel-f1 and rel-stack user-post-comment on the same machine
type and on the other tasks on an `n2-custom-24-196608` (4–5 threads, beside other jobs). "Reproduces" means within 5%
of the published value or two seed stds.

The current upstream LightGBM script adds two label features, `num_past_visit` and `global_popularity_fraction`
(counted from labels before each seed time). The published LightGBM row predates them, so ours is expected to be
higher where past visits carry signal.

### driver-circuit-compete (MAP@k)

| baseline | val | test | per seed (test) | leaderboard (test) | runtime | verdict |
|---|---|---|---|---|---|---|
| Past visit | 0.5457 | 0.2554 | deterministic | 0.1839 | 0.1 min | **higher** by 0.0715 |
| Global popularity | 0.5519 | 0.5012 | deterministic | 0.5012 | 0.1 min | reproduces |
| LightGBM (3 seeds) | 0.6525 ± 0.0121 | 0.6438 ± 0.0522 | 0.5923 / 0.6967 / 0.6426 | 0.5956 | 2.3 min per seed | reproduces |

Past visit is .07 above the leaderboard. Upstream's script builds the test prediction from train+val labels, and rel-f1's val year is the year just before test, so this probably differs from how the published number was made. We haven't traced it. LightGBM's seed std (.052) covers the gap to .596.
