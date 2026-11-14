# rel-amazon/user-item-review: candidate generation + LightGBM ranker at scale

Issue 21, recommendation board wave 2. Task: for each `rel-amazon` seed row, which products a customer writes a review of over 300 characters for in the next 91 days; metric MAP@10.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and `src/relconduit/link_big.py`
(chunked pairs and scoring at this scale), with the amazon edge sets in `src/relconduit/link_amazon.py`; this folder's
`run.py` holds the task's plan.

Edges: reviews and long reviews (`long`, over 300 characters, the label's own relation) as customer → product, plus static product → author (`brand`) and product → leaf-category codes, joined back to the reviews of each code. Candidates: the customer's own past products (top 20), a co-review walk (customer → product → customer → product, one-year half-life, beam 500, top 100), other books by the authors the customer reviewed that were popular in the last year (top 40), books in the customer's leaf categories popular in the last 91 days (top 30), and the 30-day global top 30 of each target relation. Pair features: per relation count and recency, walk scores and ranks, product popularity over 30/91/365 days per relation, plus `relconduit.features` (windows 91/365 days and all time, no hop 2) for the customer and the product; the static codes feed walks only.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before any fit, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` and `tests/test_link_big.py` show the same check catches a planted `<=`, including on the amazon edge sets and with pairs built in chunks.

Ranker: LightGBM (`relconduit.link_big`), a 3-config grid on val (lambdarank 31 leaves, binary 31, lambdarank 127; seed 0), early stopping on 19,990 val seeds, then the winner on seeds 0–2. Val and test are scored on **every** row, in chunks of 20,000 seeds, through `write_prediction_table` + `evaluate_task`; test was scored once. Train: 400,000 of 2,324,177 train seed rows from the latest 4 seed times (71,386 with a true candidate, 12,456,744 pairs); 124 features.

## Run

```sh
uv run python experiments/rel-amazon/user-item-review/cand-lgbm-ranker/run.py --smoke 200   # end-to-end check on 200 seeds, no results
uv run python experiments/rel-amazon/user-item-review/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run (run 4 of issue 21, commit e3e3f0e) went through `experiments/rec-board/cloud21.py` on a Google Cloud
`n2-custom-24-131072` Spot machine in europe-west2, 6 threads, beside the other three rankers: **85 min** (pairs, features,
5 fits, test), peak RSS 25.5 GB. The whole machine ran 2.03 h for $0.85.

## Results (MAP@10, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.0309 ± 0.0002 | 0.0309 / 0.0311 / 0.0308 |
| test | **0.0256 ± 0.0001** | 0.0255 / 0.0256 / 0.0256 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'lambdarank', 'num_leaves': 127, 'min_child_samples': 300, 'feature_fraction': 0.7}`; best iterations 171, 208, 244.

Candidate recall on val, every row (167 candidates per seed): **0.114** overall; 17.1% of seeds have at least one true candidate. By source: review 0.002, long 0.001, covisit 0.076, brand 0.042, leaf 0.018, review__top30d 0.027, long__top30d 0.024.

Grid (val MAP@10, seed 0, every val row; the winner then ran seeds 1 and 2):

| objective | num_leaves | min_child_samples | val | best iteration | fit minutes |
|---|---|---|---|---|---|
| lambdarank | 31 | 100 | 0.0306 | 391 | 10.1 |
| binary | 31 | 100 | 0.0304 | 827 | 14.2 |
| lambdarank | 127 | 300 | 0.0309 (chosen) | 171 | 7.8 |

Top features by gain (seed 0): `brand__score`, `long__pop_30d`, `covisit__score`, `covisit__rank`, `review__pop_all`, `dst.review__review_text_len__mean__91d`, `brand__rank`, `review__days_since_last`.

## Against the leaderboard (test MAP@10)

| | test |
|---|---|
| Kapso (leader) | 0.0295 |
| **cand-lgbm-ranker (this)** | 0.0256 |
| LEDGER | 0.0079 |
| GNN | 0.0044 |
| GelGT | 0.0013 |
| LightGBM (upstream) | 0.0012 |
| Global popularity | 0.0011 |
| ID-GNN | 0.0009 |
| Past visit | 0.0004 |

This sits **second**, .0039 below Kapso (.0295), and well above everything else (LEDGER .0079, GNN .0044). Recall is lower here than on the other two amazon tasks (.114 against .143 with more candidates per seed), and long reviews are where Kapso does best of its three amazon numbers, so the gap is probably in the candidates: something that picks out the customers who write long reviews and what they write them about.

Leaderboard values are from the raw `leaderboard.json` (`stanford-star/relbench`, `leaderboard/`, read 2026-11-14).

The seed-0 test table passes `relbench.submit`'s per-task check (status `ok`, MAP 0.0255). It is on the cloud checkpoint at `gs://oddsworth-lab-runs/relconduit-21-wave2/runs/cloud21/ranker_rel-amazon_user-item-review/work/rel-amazon/user-item-review/cand-lgbm-ranker/submission/rel-amazon__user-item-review.csv`; it is not committed and not submitted.
