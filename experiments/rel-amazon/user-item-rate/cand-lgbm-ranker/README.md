# rel-amazon/user-item-rate: candidate generation + LightGBM ranker at scale

Issue 21, recommendation board wave 2. Task: for each `rel-amazon` seed row, which products a customer gives a 5-star review in the next 91 days; metric MAP@10.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and `src/relconduit/link_big.py`
(chunked pairs and scoring at this scale), with the amazon edge sets in `src/relconduit/link_amazon.py`; this folder's
`run.py` holds the task's plan.

Edges: reviews and 5-star reviews (`five`, the label's own relation) as customer → product, plus static product → author (`brand`) and product → leaf-category codes, joined back to the reviews of each code. Candidates: the customer's own past products (top 20), a co-review walk (customer → product → customer → product, one-year half-life, beam 500, top 100), other books by the authors the customer reviewed that were popular in the last year (top 40), books in the customer's leaf categories popular in the last 91 days (top 30), and the 30-day global top 30 of each target relation. Pair features: per relation count and recency, walk scores and ranks, product popularity over 30/91/365 days per relation, plus `relconduit.features` (windows 91/365 days and all time, no hop 2) for the customer and the product; the static codes feed walks only.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before any fit, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` and `tests/test_link_big.py` show the same check catches a planted `<=`, including on the amazon edge sets and with pairs built in chunks.

Ranker: LightGBM (`relconduit.link_big`), a 3-config grid on val (lambdarank 31 leaves, binary 31, lambdarank 127; seed 0), early stopping on 20,000 val seeds, then the winner on seeds 0–2. Val and test are scored on **every** row, in chunks of 20,000 seeds, through `write_prediction_table` + `evaluate_task`; test was scored once. Train: 400,000 of 3,667,157 train seed rows from the latest 4 seed times (80,879 with a true candidate, 13,028,272 pairs); 124 features.

## Run

```sh
uv run python experiments/rel-amazon/user-item-rate/cand-lgbm-ranker/run.py --smoke 200   # end-to-end check on 200 seeds, no results
uv run python experiments/rel-amazon/user-item-rate/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run (run 4 of issue 21, commit e3e3f0e) went through `experiments/rec-board/cloud21.py` on a Google Cloud
`n2-custom-24-131072` Spot machine in europe-west2, 6 threads, beside the other three rankers: **104 min** (pairs, features,
5 fits, test), peak RSS 31.5 GB. The whole machine ran 2.03 h for $0.85.

## Results (MAP@10, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.0381 ± 0.0001 | 0.0382 / 0.0381 / 0.0381 |
| test | **0.0320 ± 0.0000** | 0.0320 / 0.0320 / 0.0321 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'lambdarank', 'num_leaves': 127, 'min_child_samples': 300, 'feature_fraction': 0.7}`; best iterations 290, 208, 244.

Candidate recall on val, every row (154 candidates per seed): **0.143** overall; 19.7% of seeds have at least one true candidate. By source: review 0.002, five 0.002, covisit 0.098, brand 0.058, leaf 0.023, review__top30d 0.035, five__top30d 0.036.

Grid (val MAP@10, seed 0, every val row; the winner then ran seeds 1 and 2):

| objective | num_leaves | min_child_samples | val | best iteration | fit minutes |
|---|---|---|---|---|---|
| lambdarank | 31 | 100 | 0.0381 | 460 | 12.9 |
| binary | 31 | 100 | 0.0369 | 558 | 12.7 |
| lambdarank | 127 | 300 | 0.0382 (chosen) | 290 | 12.5 |

Top features by gain (seed 0): `brand__score`, `five__pop_30d`, `covisit__rank`, `covisit__score`, `review__pop_all`, `brand__rank`, `review__pop_30d`, `dst.review__review_text_len__mean__91d`.

## Against the leaderboard (test MAP@10)

| | test |
|---|---|
| **cand-lgbm-ranker (this)** | 0.0320 |
| Kapso (leader) | 0.0231 |
| LEDGER | 0.0117 |
| GNN | 0.0079 |
| LightGBM (upstream) | 0.0017 |
| Global popularity | 0.0015 |
| GelGT | 0.0012 |
| ID-GNN | 0.0012 |
| Past visit | 0.0007 |

This sits **above every leaderboard entry** on test with the official evaluation: .0320 ± .0000 against Kapso's .0231, a gap of .0089. It beats the GNN (.0079) 4× and every published baseline. It is not submitted yet. As on purchase, recall is the cap (20% of val seeds with a true candidate).

Leaderboard values are from the raw `leaderboard.json` (`stanford-star/relbench`, `leaderboard/`, read 2026-11-14).

The seed-0 test table passes `relbench.submit`'s per-task check (status `ok`, MAP 0.0320). It is on the cloud checkpoint at `gs://oddsworth-lab-runs/relconduit-21-wave2/runs/cloud21/ranker_rel-amazon_user-item-rate/work/rel-amazon/user-item-rate/cand-lgbm-ranker/submission/rel-amazon__user-item-rate.csv`; it is not committed and not submitted.
