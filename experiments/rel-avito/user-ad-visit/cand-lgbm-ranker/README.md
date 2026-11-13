# rel-avito/user-ad-visit: candidate generation + LightGBM ranker

Issue 12, recommendation board wave 1. Task: for each `rel-avito` seed row, the 12 destinations most likely to be linked in the next window; metric MAP@12.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and
`src/relconduit/link_lgbm.py` (the protocol); this folder's `run.py` holds the task's config.

Edges: visits, phone requests and ads shown in the user's searches (`SearchInfo` ⋈ `SearchStream`). Candidates: the user's own visited / phoned / shown ads, a co-visitation walk (user → ad → user → ad, three-day half-life), the ads popular in the last four days in the categories the user visits, and the four-day global top 20. Pair features: per relation count and recency, walk scores and ranks, ad popularity over 1/4/8 days per relation, plus `relconduit.features` (windows 4/8 days and all time, no hop 2) for the user and the ad. 73% of val's true ads were never visited by anyone before the seed time, which caps candidate recall at .085.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before fitting, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` shows the same check catches a planted `<=`.

Ranker: LightGBM, lambdarank or binary, early stopping on val (`map@12`), grid of 8 configs × 3 seeds chosen by mean val MAP (official `task.evaluate`), then test scored once per seed through `write_prediction_table` + `evaluate_task`. Train: 20,000 of 86,616 train seed rows (fixed sample), 761,073 pairs from seeds with a true candidate; 452 features.

## Run

```sh
uv run python experiments/rel-avito/user-ad-visit/cand-lgbm-ranker/run.py --smoke 150   # end-to-end check, no results
uv run python experiments/rel-avito/user-ad-visit/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run went through `experiments/rec-board/cloud_jobs.py` on a Google Cloud `n2-custom-24-131072` Spot machine, 4 threads, beside the other jobs: **39 min** (pairs, features, 24 fits and test), peak RSS 31.25 GB.

## Results (MAP@12, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.0577 ± 0.0001 | 0.0577 / 0.0577 / 0.0576 |
| test | **0.0403 ± 0.0001** | 0.0402 / 0.0404 / 0.0404 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'lambdarank', 'num_leaves': 15, 'min_child_samples': 20, 'feature_fraction': 0.7}`.

Candidate recall on val (116 candidates per seed): **0.085** overall; by source visit 0.074, phone 0.016, shown 0.004, covisit 0.082, category 0.003, visit__top4d 0.000.

Grid (val MAP, mean ± std over 3 seeds):

| objective | num_leaves | min_child_samples | val |
|---|---|---|---|
| lambdarank | 15 | 20 | 0.0577 ± 0.0001 |
| lambdarank | 15 | 100 | 0.0575 ± 0.0001 |
| lambdarank | 63 | 20 | 0.0574 ± 0.0001 |
| lambdarank | 63 | 100 | 0.0574 ± 0.0001 |
| binary | 15 | 20 | 0.0575 ± 0.0001 |
| binary | 15 | 100 | 0.0575 ± 0.0001 |
| binary | 63 | 20 | 0.0575 ± 0.0000 |
| binary | 63 | 100 | 0.0576 ± 0.0001 |

Top features by gain (seed 0): `covisit__score`, `visit__count`, `covisit__rank`, `visit__days_since_last`, `visit__days_since_first`, `dst.VisitStream__count__4d`, `phone__days_since_last`, `src.VisitStream__count__8d`.

## Against the leaderboard (test MAP@12)

| | test |
|---|---|
| Kapso (leader) | 0.0420 |
| **cand-lgbm-ranker (this)** | 0.0403 |
| ID-GNN | 0.0363 |
| GNN | 0.0363 |
| Past visit | 0.0193 |
| LightGBM (upstream) | 0.0006 |
| Global popularity | 0.0000 |

This sits below Kapso (leader). Second on the board, ahead of the GNN and ID-GNN (.036), .0017 below Kapso with a seed std of .0001: a clear gap, though a small one in absolute terms. The cap here is candidate recall, not the ranker.
