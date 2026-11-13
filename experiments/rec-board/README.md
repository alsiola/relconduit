# Recommendation board, wave 1 (issue 12)

Every link-prediction (recommendation) task in RelBench 3.x except rel-amazon's and rel-hm's, which come in a later
wave. rel-event has no recommendation task in 3.x. The official metric is MAP@k throughout (`task.evaluate`).

## Tasks

| task | src entity → dst entity | eval_k | window | train rows | val rows | test rows | avg true dst per row (val) | wave |
|---|---|---|---|---|---|---|---|---|
| rel-f1/driver-circuit-compete | `drivers.driverId` → `circuits.circuitId` | 10 | 365 d | 2,649 | 27 | 27 | 13.93 | 1 |
| rel-trial/condition-sponsor-run | `conditions.condition_id` → `sponsors.sponsor_id` | 10 | 365 d | 36,934 | 2,081 | 2,057 | 14.63 | 1 |
| rel-trial/site-sponsor-run | `facilities.facility_id` → `sponsors.sponsor_id` | 10 | 365 d | 669,310 | 37,003 | 27,428 | 2.16 | 1 |
| rel-avito/user-ad-visit | `UserInfo.UserID` → `AdsInfo.AdID` | 12 | 4 d | 86,616 | 29,979 | 36,129 | 29.27 | 1 |
| rel-stack/user-post-comment | `users.UserId` → `posts.PostId` | 100 | 91 d | 21,240 | 825 | 758 | 1.94 | 1 |
| rel-stack/post-post-related | `posts.PostId` → `posts.postLinksIdList` | 100 | 91 d | 5,855 | 226 | 258 | 1.3 | 1 |
| rel-amazon/user-item-purchase | `customer.customer_id` → `product.product_id` | 10 | 91 d | 5,112,803 | 351,876 | 393,985 | 2.28 | later |
| rel-amazon/user-item-rate | `customer.customer_id` → `product.product_id` | 10 | 91 d | 3,667,157 | 257,939 | 292,609 | 2.01 | later |
| rel-amazon/user-item-review | `customer.customer_id` → `product.product_id` | 10 | 91 d | 2,324,177 | 116,970 | 127,021 | 2.3 | later |
| rel-hm/user-item-purchase | `customer.customer_id` → `article.article_id` | 12 | 7 d | 3,878,451 | 74,575 | 67,144 | 3.18 | later |

Rows are the task tables' rows: one per (src entity, seed time), with a list of true destinations.

## What is here

- `src/relconduit/link.py`: time-safe candidates (the entity's own past targets, recent popularity, co-occurrence and
  2-hop walks) and pair features, with `tests/test_link.py`. The tests include the pair leakage test: pairs built on the
  full database equal pairs built on `truncate_before(db, T)`, and a planted `<=` is caught.
- `src/relconduit/link_lgbm.py`: the protocol (8 configs × 3 seeds tuned on val, test scored once).
- `experiments/<dataset>/<task>/cand-lgbm-ranker/`: one folder per task, with its config, candidate recall and grid.
- `experiments/<dataset>/baselines/README.md`: upstream trivial and LightGBM baselines reproduced.
- `cloud_jobs.py`: the cloud driver. `baseline_rows.py` appends the baseline rows from its checkpoint.

## Results (test MAP@k, 3 seeds, mean ± std)

| task | cand-lgbm-ranker | leader (Kapso) | ID-GNN | GNN | past visit | LightGBM, published | LightGBM, ours | verdict |
|---|---|---|---|---|---|---|---|---|
| rel-f1 driver-circuit-compete | **.8727 ± .0055** | .8795 | .6232 | .2069 | .1839 | .5956 | .6438 ± .0522 | 2nd, .007 below the leader (about 1.2 std) |
| rel-trial condition-sponsor-run | **.1221 ± .0008** | .1228 | .0033 | .0350 | .0830 | .0480 | .0709 ± .0031 | tie with the leader (.0007 gap, inside the seed std) |
| rel-trial site-sponsor-run | **.2834 ± .0008** | .3333 | .0014 | .1059 | .1734 | .0904 | .1664 ± .0149 | 2nd, .050 below the leader |
| rel-avito user-ad-visit | **.0403 ± .0001** | .0420 | .0363 | .0363 | .0193 | .0006 | .0141 ± .0004 | 2nd, .0017 below the leader |
| rel-stack user-post-comment | **.1330 ± .0023** | .1313 | .1303 | .0004 | .0138 | .0002 | .0086 ± .0042 | tie with the leader (+.0017, inside the seed std, not a win) |
| rel-stack post-post-related | **.1993 ± .0047** | .2178 | .1075 | .0012 | .0174 | .0168 | .0130 ± .0094 | 2nd, .019 below the leader |

Leaderboard values are from the raw `leaderboard.json` (2026-11-13). RT-J has no recommendation entries there.
The ranker beats every published baseline and both GNNs on all six tasks. It ties the leader on two and is second on
the other four. That is no state-of-the-art claim. Val numbers, candidate recall and the grids are in each experiment
README. Rows are in `results/<dataset>/<task>.jsonl`.

The trivial baselines reproduce the published values, except rel-f1 past visit (.255 against .184; see
`experiments/rel-f1/baselines/README.md`). Upstream's current LightGBM script beats its own published row on four tasks,
because it now has past-visit and popularity label features. Details are in each dataset's baselines README.

## Runs and cost

All on Google Cloud Spot in europe-west2, through `uv run cloud-run`, under the issue's $8 approval.

| run | machine | hours | cost | what it finished |
|---|---|---|---|---|
| 1 (3222743) | n2-custom-24-131072 | 0.85 | $0.31 | smoke phase with leakage checks on all six; all trivial baselines; rel-f1 ranker. Stalled out of memory (three rel-avito upstream seeds at about 25 GB each) |
| 2 (d6e168f) | n2-custom-24-131072 ×3 (two preemptions) | 2.33 + 0.14 + 0.55 | $1.15 | the other five rankers; upstream LightGBM on rel-f1 and rel-stack user-post-comment |
| 3 (efd3ad0) | n2-custom-24-196608 | 5.25 | $2.42 | upstream LightGBM on rel-avito, rel-trial (both), rel-stack post-post-related |
| **total** | | 9.1 | **$3.88** | |

Per-job runtimes (from each job's `done.json`): rankers 2 min (rel-f1) to 41 min (rel-trial site-sponsor-run).
Upstream LightGBM took 2–204 min per seed, and rel-avito, at 156–204 min per seed with about 30 GB peak, was the
slowest. Mac use was about 17 min of CPU for development and smoke tests on 150–300 seed samples.

## Reproduce

```sh
uv run cloud-run --issue 12 --approved <comment> --machine n2-custom-24-196608 --hours 5.5 \
    --checkpoint runs/cloud12 --run rec-wave1 -- \
    uv run python experiments/rec-board/cloud_jobs.py --out runs/cloud12
uv run python experiments/rec-board/baseline_rows.py --out runs/cloud12 --commit <sha> --machine "<machine>"
```

Each ranker also runs on its own: `uv run python experiments/<dataset>/<task>/cand-lgbm-ranker/run.py --test`.
