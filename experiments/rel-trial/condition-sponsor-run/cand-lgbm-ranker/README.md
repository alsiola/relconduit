# rel-trial/condition-sponsor-run: candidate generation + LightGBM ranker

Issue 12, recommendation board wave 1. Task: for each `rel-trial` seed row, the 10 destinations most likely to be linked in the next window; metric MAP@10.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and
`src/relconduit/link_lgbm.py` (the protocol); this folder's `run.py` holds the task's config.

Edges: condition → sponsor through a study on the condition run by the sponsor (`conditions_studies` ⋈ `sponsors_studies`, at the later date). Candidates: the condition's own past sponsors, co-occurrence walks (condition → sponsor → condition → sponsor; a two-year half-life and a two-year window), and the most active sponsors of the last year. Pair features: past count and recency, walk scores and ranks, sponsor popularity over 1/2/5 years, plus `relconduit.features` (no hop 2) for the condition and the sponsor.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before fitting, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` shows the same check catches a planted `<=`.

Ranker: LightGBM, lambdarank or binary, early stopping on val (`map@10`), grid of 8 configs × 3 seeds chosen by mean val MAP (official `task.evaluate`), then test scored once per seed through `write_prediction_table` + `evaluate_task`. Train: 12,000 of 36,934 train seed rows (fixed sample), 1,102,934 pairs from seeds with a true candidate; 117 features.

## Run

```sh
uv run python experiments/rel-trial/condition-sponsor-run/cand-lgbm-ranker/run.py --smoke 150   # end-to-end check, no results
uv run python experiments/rel-trial/condition-sponsor-run/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run went through `experiments/rec-board/cloud_jobs.py` on a Google Cloud `n2-custom-24-131072` Spot machine, 4 threads, beside the other jobs: **13 min** (pairs, features, 24 fits and test), peak RSS 8.45 GB.

## Results (MAP@10, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.1161 ± 0.0008 | 0.1162 / 0.1152 / 0.1168 |
| test | **0.1221 ± 0.0008** | 0.1224 / 0.1227 / 0.1212 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'binary', 'num_leaves': 63, 'min_child_samples': 20, 'feature_fraction': 0.7}`.

Candidate recall on val (142 candidates per seed): **0.360** overall; by source ran 0.232, covisit 0.321, covisit_recent 0.198, sponsor_trials__top365d 0.125, ran__top365d 0.129.

Grid (val MAP, mean ± std over 3 seeds):

| objective | num_leaves | min_child_samples | val |
|---|---|---|---|
| lambdarank | 15 | 20 | 0.1147 ± 0.0014 |
| lambdarank | 15 | 100 | 0.1158 ± 0.0008 |
| lambdarank | 63 | 20 | 0.1151 ± 0.0014 |
| lambdarank | 63 | 100 | 0.1148 ± 0.0009 |
| binary | 15 | 20 | 0.1149 ± 0.0002 |
| binary | 15 | 100 | 0.1152 ± 0.0003 |
| binary | 63 | 20 | 0.1161 ± 0.0008 |
| binary | 63 | 100 | 0.1155 ± 0.0008 |

Top features by gain (seed 0): `ran__count`, `ran__days_since_last`, `src.conditions_studies__count__365d`, `src.conditions_studies__count__1460d`, `covisit__score`, `covisit__rank`, `covisit_recent__score`, `covisit_recent__rank`.

## Against the leaderboard (test MAP@10)

| | test |
|---|---|
| Kapso (leader) | 0.1228 |
| **cand-lgbm-ranker (this)** | 0.1221 |
| Past visit | 0.0830 |
| LightGBM (upstream) | 0.0480 |
| GNN | 0.0350 |
| Global popularity | 0.0252 |
| ID-GNN | 0.0033 |

This sits below Kapso (leader). Test is .1221 ± .0008 against Kapso's .1228: .0007 below, within one seed std, so a tie with the leader.
