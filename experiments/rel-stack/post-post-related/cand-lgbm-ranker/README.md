# rel-stack/post-post-related: candidate generation + LightGBM ranker

Issue 12, recommendation board wave 1. Task: for each `rel-stack` seed row, the 100 destinations most likely to be linked in the next window; metric MAP@100.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and
`src/relconduit/link_lgbm.py` (the protocol); this folder's `run.py` holds the task's config.

Edges: `postLinks` in both directions; post → tag; post → title word (lowercased, 4+ letters, a short stop list); tag / word → linked post (one edge per link to a post carrying it). Candidates: the post's own linked posts, two-hop links, often-linked posts in the post's tags and sharing words of its title, and the most-linked posts of the last ten years. Pair features: link count and recency, walk scores and ranks, link popularity over 91/365/1095 days, plus `relconduit.features` (no hop 2) for both posts. Tags and title words carry most of the recall (.45 and .36 of .57).

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before fitting, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` shows the same check catches a planted `<=`.

Ranker: LightGBM, lambdarank or binary, early stopping on val (`map@100`), grid of 8 configs × 3 seeds chosen by mean val MAP (official `task.evaluate`), then test scored once per seed through `write_prediction_table` + `evaluate_task`. Train: 5,855 of 5,855 train seed rows (fixed sample), 2,205,078 pairs from seeds with a true candidate; 353 features.

## Run

```sh
uv run python experiments/rel-stack/post-post-related/cand-lgbm-ranker/run.py --smoke 150   # end-to-end check, no results
uv run python experiments/rel-stack/post-post-related/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run went through `experiments/rec-board/cloud_jobs.py` on a Google Cloud `n2-custom-24-131072` Spot machine, 4 threads, beside the other jobs: **25 min** (pairs, features, 24 fits and test), peak RSS 19.47 GB.

## Results (MAP@100, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.1721 ± 0.0050 | 0.1778 / 0.1683 / 0.1703 |
| test | **0.1993 ± 0.0047** | 0.2033 / 0.1941 / 0.2005 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'lambdarank', 'num_leaves': 15, 'min_child_samples': 100, 'feature_fraction': 0.7}`.

Candidate recall on val (624 candidates per seed): **0.574** overall; by source link_any 0.096, two_hop 0.049, tags 0.451, words 0.356, link_any__top3650d 0.069.

Grid (val MAP, mean ± std over 3 seeds):

| objective | num_leaves | min_child_samples | val |
|---|---|---|---|
| lambdarank | 15 | 20 | 0.1714 ± 0.0042 |
| lambdarank | 15 | 100 | 0.1721 ± 0.0050 |
| lambdarank | 63 | 20 | 0.1663 ± 0.0061 |
| lambdarank | 63 | 100 | 0.1707 ± 0.0028 |
| binary | 15 | 20 | 0.1539 ± 0.0028 |
| binary | 15 | 100 | 0.1520 ± 0.0050 |
| binary | 63 | 20 | 0.1442 ± 0.0063 |
| binary | 63 | 100 | 0.1513 ± 0.0103 |

Top features by gain (seed 0): `tags__rank`, `tags__score`, `link_any__days_since_last`, `link_any__days_since_first`, `words__rank`, `words__score`, `two_hop__score`, `two_hop__rank`.

## Against the leaderboard (test MAP@100)

| | test |
|---|---|
| Kapso (leader) | 0.2178 |
| **cand-lgbm-ranker (this)** | 0.1993 |
| ID-GNN | 0.1075 |
| Past visit | 0.0174 |
| LightGBM (upstream) | 0.0168 |
| Global popularity | 0.0146 |
| GNN | 0.0012 |

This sits below Kapso (leader). Second on the board, ahead of ID-GNN (.108), .018 below Kapso, about four seed stds: a clear gap. Text similarity beyond tags and title words is the obvious next lever.
