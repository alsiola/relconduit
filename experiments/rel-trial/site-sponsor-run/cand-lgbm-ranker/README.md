# rel-trial/site-sponsor-run: candidate generation + LightGBM ranker

Issue 12, recommendation board wave 1. Task: for each `rel-trial` seed row, the 10 destinations most likely to be linked in the next window; metric MAP@10.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and
`src/relconduit/link_lgbm.py` (the protocol); this folder's `run.py` holds the task's config.

Edges: facility → sponsor through a study at the facility (`facilities_studies` ⋈ `sponsors_studies`). Half the val facilities have no trial before the seed time, so besides the facility's own past sponsors and a co-occurrence walk, candidates come from walks through the facility's static columns: sponsors running trials at other sites in the same city, the same country (two-year window) and with the same name (e.g. "Novartis Investigative Site"), plus last year's most active sponsors. Pair features: past count and recency, every walk's score and rank, sponsor popularity, plus `relconduit.features` (no hop 2) for both sides.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before fitting, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` shows the same check catches a planted `<=`.

Ranker: LightGBM, lambdarank or binary, early stopping on val (`map@10`), grid of 8 configs × 3 seeds chosen by mean val MAP (official `task.evaluate`), then test scored once per seed through `write_prediction_table` + `evaluate_task`. Train: 25,000 of 669,310 train seed rows (fixed sample), 1,470,477 pairs from seeds with a true candidate; 122 features.

## Run

```sh
uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-ranker/run.py --smoke 150   # end-to-end check, no results
uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run went through `experiments/rec-board/cloud_jobs.py` on a Google Cloud `n2-custom-24-131072` Spot machine, 4 threads, beside the other jobs: **40 min** (pairs, features, 24 fits and test), peak RSS 15.30 GB.

## Results (MAP@10, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.2650 ± 0.0017 | 0.2669 / 0.2639 / 0.2642 |
| test | **0.2834 ± 0.0008** | 0.2841 / 0.2836 / 0.2825 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'lambdarank', 'num_leaves': 63, 'min_child_samples': 20, 'feature_fraction': 0.7}`.

Candidate recall on val (91 candidates per seed): **0.553** overall; by source ran 0.206, covisit 0.271, city 0.413, country 0.288, name 0.204, ran__top365d 0.242.

Grid (val MAP, mean ± std over 3 seeds):

| objective | num_leaves | min_child_samples | val |
|---|---|---|---|
| lambdarank | 15 | 20 | 0.2629 ± 0.0020 |
| lambdarank | 15 | 100 | 0.2633 ± 0.0027 |
| lambdarank | 63 | 20 | 0.2650 ± 0.0017 |
| lambdarank | 63 | 100 | 0.2641 ± 0.0010 |
| binary | 15 | 20 | 0.2582 ± 0.0013 |
| binary | 15 | 100 | 0.2586 ± 0.0005 |
| binary | 63 | 20 | 0.2619 ± 0.0005 |
| binary | 63 | 100 | 0.2617 ± 0.0007 |

Top features by gain (seed 0): `covisit__score`, `name__score`, `city__rank`, `covisit__rank`, `city__score`, `sponsor_trials__pop_365d`, `name__rank`, `country__rank`.

## Against the leaderboard (test MAP@10)

| | test |
|---|---|
| Kapso (leader) | 0.3333 |
| **cand-lgbm-ranker (this)** | 0.2834 |
| Past visit | 0.1734 |
| GNN | 0.1059 |
| LightGBM (upstream) | 0.0904 |
| Global popularity | 0.0375 |
| ID-GNN | 0.0014 |

This sits below Kapso (leader). Second on the board, .050 below Kapso, far beyond the seed std: a clear gap. Past visit alone gets .17, so the walks through city and site name add about .11.
