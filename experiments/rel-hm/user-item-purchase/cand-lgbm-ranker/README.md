# rel-hm/user-item-purchase: candidate generation + LightGBM ranker at scale

Issue 21, recommendation board wave 2. Task: for each `rel-hm` seed row, which articles a customer buys in the next 7 days; metric MAP@12.

## What it does

Two stages, with the shared code in `src/relconduit/link.py` (candidates and pair features) and `src/relconduit/link_big.py`
(chunked pairs and scoring at this scale); this folder's
`run.py` holds the task's plan.

Edges: purchases (customer → article), plus static article → product code (other colours and sizes of the same garment) and article → product type, joined back to the purchases of each code. Candidates: the customer's own past articles (top 30, repurchases), a co-purchase walk (14-day half-life, beam 500, top 50), other colours and sizes of the products the customer bought that sold in the last 7 days (top 30), the 7-day best sellers in the customer's product types (top 30), and the 7-day global top 60. Pair features: purchase count and recency, walk scores and ranks, article popularity over 7/14/28/91 days, plus `relconduit.features` (windows 7/28/91 days and all time, no hop 2) for the customer and the article.

Every feature and candidate reads only database rows dated strictly before the seed time; the task's label tables are never used as features. Before any fit, the run compares pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds: **0 mismatches**. `tests/test_link.py` and `tests/test_link_big.py` show the same check catches a planted `<=`, including on the amazon edge sets and with pairs built in chunks.

Ranker: LightGBM (`relconduit.link_big`), a 3-config grid on val (lambdarank 31 leaves, binary 31, lambdarank 127; seed 0), early stopping on 20,000 val seeds, then the winner on seeds 0–2. Val and test are scored on **every** row, in chunks of 20,000 seeds, through `write_prediction_table` + `evaluate_task`; test was scored once. Train: 150,000 of 3,878,451 train seed rows from the latest 8 seed times (44,838 with a true candidate, 6,471,542 pairs); 297 features.

## Run

```sh
uv run python experiments/rel-hm/user-item-purchase/cand-lgbm-ranker/run.py --smoke 200   # end-to-end check on 200 seeds, no results
uv run python experiments/rel-hm/user-item-purchase/cand-lgbm-ranker/run.py --test        # tune on val, score test once, append to results/
```

The reported run (run 4 of issue 21, commit e3e3f0e) went through `experiments/rec-board/cloud21.py` on a Google Cloud
`n2-custom-24-131072` Spot machine in europe-west2, 6 threads, beside the other three rankers: **73 min** (pairs, features,
5 fits, test), peak RSS 35.2 GB. The whole machine ran 2.03 h for $0.85.

## Results (MAP@12, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | 0.0297 ± 0.0002 | 0.0297 / 0.0298 / 0.0295 |
| test | **0.0329 ± 0.0002** | 0.0327 / 0.0329 / 0.0330 |

Val is optimistic: it chose the config and the stopping round. Chosen config: `{'objective': 'lambdarank', 'num_leaves': 31, 'min_child_samples': 100, 'feature_fraction': 0.7}`; best iterations 238, 271, 314.

Candidate recall on val, every row (142 candidates per seed): **0.157** overall; 31.9% of seeds have at least one true candidate. By source: buy 0.036, covisit 0.074, product 0.057, type 0.033, buy__top7d 0.071.

Grid (val MAP@12, seed 0, every val row; the winner then ran seeds 1 and 2):

| objective | num_leaves | min_child_samples | val | best iteration | fit minutes |
|---|---|---|---|---|---|
| lambdarank | 31 | 100 | 0.0297 (chosen) | 238 | 8.3 |
| binary | 31 | 100 | 0.0295 | 447 | 10.9 |
| lambdarank | 127 | 300 | 0.0295 | 135 | 7.8 |

Top features by gain (seed 0): `dst.article__product_code`, `buy__days_since_last`, `product__score`, `covisit__score`, `covisit__rank`, `buy__pop_7d`, `src.transactions__sales_channel_id__mean__7d`, `src.transactions__days_since_last`.

## Against the leaderboard (test MAP@12)

| | test |
|---|---|
| **cand-lgbm-ranker (this)** | 0.0329 |
| Kapso (leader) | 0.0326 |
| ID-GNN | 0.0281 |
| LEDGER | 0.0241 |
| Past visit | 0.0089 |
| GNN | 0.0082 |
| GelGT | 0.0057 |
| LightGBM (upstream) | 0.0037 |
| Global popularity | 0.0030 |

This sits **just above Kapso** on test with the official evaluation: .0329 ± .0002 against .0326, a gap of .0003, about 2 seed stds. That is a narrow lead, not a clear one. It beats ID-GNN (.0281) and every published baseline. It is not submitted yet.

Leaderboard values are from the raw `leaderboard.json` (`stanford-star/relbench`, `leaderboard/`, read 2026-11-14).

The seed-0 test table passes `relbench.submit`'s per-task check (status `ok`, MAP 0.0327). It is on the cloud checkpoint at `gs://oddsworth-lab-runs/relconduit-21-wave2/runs/cloud21/ranker_rel-hm_user-item-purchase/work/rel-hm/user-item-purchase/cand-lgbm-ranker/submission/rel-hm__user-item-purchase.csv`; it is not committed and not submitted.
