# rel-f1/driver-circuit-compete: candidate generation + LightGBM ranker

Issue 12, recommendation board wave 1. Task: for each `rel-f1` seed row, the 10 destinations most likely to be linked in the next window; metric MAP@10.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and
`src/relconduit/link_lgbm.py` (the protocol); this folder's `run.py` holds the task's config.

Candidates: every circuit on the calendar in the last three seasons (77 circuits in all), the driver's own past circuits, and a co-racing walk (driver → circuit → driver → circuit, two-year half-life). Pair features: the driver's races at the circuit (count, days since the last and first), walk score and rank, the circuit's race counts over 1/2/5 years, plus `relconduit.features` for the driver and the circuit.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before fitting, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 67 val and train seeds: **0 mismatches**. `tests/test_link.py` shows the same check catches a planted `<=`.

Ranker: LightGBM, lambdarank or binary, early stopping on val (`map@10`), grid of 8 configs × 3 seeds chosen by mean val MAP (official `task.evaluate`), then test scored once per seed through `write_prediction_table` + `evaluate_task`. Train: 2,649 of 2,649 train seed rows (fixed sample), 73,219 pairs from seeds with a true candidate; 701 features.

## Run

```sh
uv run python experiments/rel-f1/driver-circuit-compete/cand-lgbm-ranker/run.py --smoke 150   # end-to-end check, no results
uv run python experiments/rel-f1/driver-circuit-compete/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run went through `experiments/rec-board/cloud_jobs.py` on a Google Cloud `n2-custom-24-131072` Spot machine, 4 threads, beside the other jobs: **2 min** (pairs, features, 24 fits and test), peak RSS 2.12 GB.

## Results (MAP@10, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.7319 ± 0.0079 | 0.7346 / 0.7230 / 0.7381 |
| test | **0.8727 ± 0.0055** | 0.8769 / 0.8747 / 0.8665 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'binary', 'num_leaves': 63, 'min_child_samples': 20, 'feature_fraction': 0.7}`.

Candidate recall on val (52 candidates per seed): **0.956** overall; by source raced 0.652, covisit 0.744, calendar__top1095d 0.956.

Grid (val MAP, mean ± std over 3 seeds):

| objective | num_leaves | min_child_samples | val |
|---|---|---|---|
| lambdarank | 15 | 20 | 0.7024 ± 0.0136 |
| lambdarank | 15 | 100 | 0.7179 ± 0.0055 |
| lambdarank | 63 | 20 | 0.7176 ± 0.0090 |
| lambdarank | 63 | 100 | 0.7073 ± 0.0126 |
| binary | 15 | 20 | 0.7219 ± 0.0013 |
| binary | 15 | 100 | 0.7219 ± 0.0013 |
| binary | 63 | 20 | 0.7319 ± 0.0079 |
| binary | 63 | 100 | 0.7260 ± 0.0057 |

Top features by gain (seed 0): `raced__days_since_last`, `dst.races__year__mean__365d`, `dst.circuits__circuitRef`, `covisit__score`, `dst.circuits__location`, `raced__pop_730d`, `dst.races__year__min__1095d`, `src.standings__races.year__max__all`.

## Against the leaderboard (test MAP@10)

| | test |
|---|---|
| Kapso (leader) | 0.8795 |
| **cand-lgbm-ranker (this)** | 0.8727 |
| ID-GNN | 0.6232 |
| LightGBM (upstream) | 0.5956 |
| Global popularity | 0.5012 |
| GNN | 0.2069 |
| Past visit | 0.1839 |

This sits below Kapso (leader). Second on the board: .0068 below Kapso, about 1.2 seed stds, so close to a tie but not one. Val has only 27 rows (one seed time), so early stopping lands at 3–5 rounds and the config choice is noisy; test has 27 rows too.
