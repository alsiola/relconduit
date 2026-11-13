# rel-stack/user-post-comment: candidate generation + LightGBM ranker

Issue 12, recommendation board wave 1. Task: for each `rel-stack` seed row, the 100 destinations most likely to be linked in the next window; metric MAP@100.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and
`src/relconduit/link_lgbm.py` (the protocol); this folder's `run.py` holds the task's config.

Edges: the user's comments, edits (`postHistory`) and own posts; post → thread (its question). Candidates: posts the user commented on, edited or wrote; posts in the threads the user touched (user → post → thread → post, 180-day half-life); a comment co-visitation walk; posts with comments in the last 30 days in the tags the user is active in; and the 30-day most-commented posts. Pair features: per relation count and recency, walk scores and ranks, post activity (comments, edits, votes) over 7/30/91/365 days and days since the last, plus `relconduit.features` (no hop 2) for the user and the post.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before fitting, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` shows the same check catches a planted `<=`.

Ranker: LightGBM, lambdarank or binary, early stopping on val (`map@100`), grid of 8 configs × 3 seeds chosen by mean val MAP (official `task.evaluate`), then test scored once per seed through `write_prediction_table` + `evaluate_task`. Train: 6,000 of 21,240 train seed rows (fixed sample), 1,327,475 pairs from seeds with a true candidate; 341 features.

## Run

```sh
uv run python experiments/rel-stack/user-post-comment/cand-lgbm-ranker/run.py --smoke 150   # end-to-end check, no results
uv run python experiments/rel-stack/user-post-comment/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run went through `experiments/rec-board/cloud_jobs.py` on a Google Cloud `n2-custom-24-131072` Spot machine, 4 threads, beside the other jobs: **23 min** (pairs, features, 24 fits and test), peak RSS 14.15 GB.

## Results (MAP@100, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.1596 ± 0.0026 | 0.1568 / 0.1599 / 0.1620 |
| test | **0.1330 ± 0.0023** | 0.1305 / 0.1350 / 0.1334 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'lambdarank', 'num_leaves': 15, 'min_child_samples': 20, 'feature_fraction': 0.7}`.

Candidate recall on val (452 candidates per seed): **0.293** overall; by source comment 0.112, history 0.199, own 0.204, thread 0.263, covisit 0.110, tags 0.056, comment__top30d 0.005.

Grid (val MAP, mean ± std over 3 seeds):

| objective | num_leaves | min_child_samples | val |
|---|---|---|---|
| lambdarank | 15 | 20 | 0.1596 ± 0.0026 |
| lambdarank | 15 | 100 | 0.1594 ± 0.0029 |
| lambdarank | 63 | 20 | 0.1517 ± 0.0025 |
| lambdarank | 63 | 100 | 0.1561 ± 0.0029 |
| binary | 15 | 20 | 0.1489 ± 0.0029 |
| binary | 15 | 100 | 0.1510 ± 0.0037 |
| binary | 63 | 20 | 0.1450 ± 0.0028 |
| binary | 63 | 100 | 0.1459 ± 0.0040 |

Top features by gain (seed 0): `thread__rank`, `thread__score`, `post_activity__pop_days_since_last`, `cand__thread`, `own__days_since_last`, `own__count`, `dst.votes__count__all`, `post_activity__pop_30d`.

## Against the leaderboard (test MAP@100)

| | test |
|---|---|
| **cand-lgbm-ranker (this)** | 0.1330 |
| Kapso (leader) | 0.1313 |
| ID-GNN | 0.1303 |
| Past visit | 0.0138 |
| GNN | 0.0004 |
| LightGBM (upstream) | 0.0002 |
| Global popularity | 0.0002 |

This sits at the top of the board on the mean. Test is .133 ± .002 against Kapso's .1313: .0017 above, inside one seed std, so a tie with the leader, not a win.
