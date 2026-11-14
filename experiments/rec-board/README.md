# Recommendation board (issues 12, 20 and 21)

All ten link-prediction (recommendation) tasks in RelBench 3.x, the whole recommendation family that `relbench.submit`
packages. Wave 1 (issue 12) covered six; issue 20 pushed two of them further; wave 2 (issue 21) added rel-amazon's three
and rel-hm's one. rel-event has no recommendation task in 3.x. The official metric is MAP@k throughout (`task.evaluate`).

## Tasks

| task | src entity → dst entity | eval_k | window | train rows | val rows | test rows | avg true dst per row (val) | wave |
|---|---|---|---|---|---|---|---|---|
| rel-f1/driver-circuit-compete | `drivers.driverId` → `circuits.circuitId` | 10 | 365 d | 2,649 | 27 | 27 | 13.93 | 1 |
| rel-trial/condition-sponsor-run | `conditions.condition_id` → `sponsors.sponsor_id` | 10 | 365 d | 36,934 | 2,081 | 2,057 | 14.63 | 1 |
| rel-trial/site-sponsor-run | `facilities.facility_id` → `sponsors.sponsor_id` | 10 | 365 d | 669,310 | 37,003 | 27,428 | 2.16 | 1 |
| rel-avito/user-ad-visit | `UserInfo.UserID` → `AdsInfo.AdID` | 12 | 4 d | 86,616 | 29,979 | 36,129 | 29.27 | 1 |
| rel-stack/user-post-comment | `users.UserId` → `posts.PostId` | 100 | 91 d | 21,240 | 825 | 758 | 1.94 | 1 |
| rel-stack/post-post-related | `posts.PostId` → `posts.postLinksIdList` | 100 | 91 d | 5,855 | 226 | 258 | 1.3 | 1 |
| rel-amazon/user-item-purchase | `customer.customer_id` → `product.product_id` | 10 | 91 d | 5,112,803 | 351,876 | 393,985 | 2.28 | 2 |
| rel-amazon/user-item-rate | `customer.customer_id` → `product.product_id` | 10 | 91 d | 3,667,157 | 257,939 | 292,609 | 2.01 | 2 |
| rel-amazon/user-item-review | `customer.customer_id` → `product.product_id` | 10 | 91 d | 2,324,177 | 116,970 | 127,021 | 2.3 | 2 |
| rel-hm/user-item-purchase | `customer.customer_id` → `article.article_id` | 12 | 7 d | 3,878,451 | 74,575 | 67,144 | 3.18 | 2 |

Rows are the task tables' rows: one per (src entity, seed time), with a list of true destinations.

## What is here

- `src/relconduit/link.py`: time-safe candidates (the entity's own past targets, recent popularity, co-occurrence and
  2-hop walks) and pair features, with `tests/test_link.py`. The tests include the pair leakage test: pairs built on the
  full database equal pairs built on `truncate_before(db, T)`, and a planted `<=` is caught.
- `src/relconduit/link_lgbm.py`: the protocol (8 configs × 3 seeds tuned on val, test scored once).
- `experiments/<dataset>/<task>/cand-lgbm-ranker/`: one folder per task, with its config, candidate recall and grid.
- `experiments/<dataset>/baselines/README.md`: upstream trivial and LightGBM baselines reproduced.
- `cloud_jobs.py`: the cloud driver. `baseline_rows.py` appends the baseline rows from its checkpoint.

## Results: all ten tasks (test MAP@k, 3 seeds, mean ± std)

Our best approach per task, chosen on val, against the leaderboard. Leaderboard values are from the raw
`leaderboard.json` (`stanford-star/relbench`, `leaderboard/`; wave 1 read 2026-11-13, wave 2 2026-11-14). RT-J has no
recommendation entries there.

| task | ours | approach | leader (Kapso) | next best published | ID-GNN | GNN | past visit | verdict |
|---|---|---|---|---|---|---|---|---|
| rel-f1 driver-circuit-compete | .8727 ± .0055 | cand-lgbm-ranker | .8795 | GelGT .7456 | .6232 | .2069 | .1839 | 2nd, .007 below (about 1.2 std) |
| rel-trial condition-sponsor-run | .1223 ± .0030 | cand-lgbm-ranker-plus | .1228 | LEDGER .0598 | .0033 | .0350 | .0830 | tie (.0005 gap, inside the seed std) |
| rel-trial site-sponsor-run | .2834 ± .0008 | cand-lgbm-ranker | .3333 | LEDGER .1133 | .0014 | .1059 | .1734 | 2nd, .050 below |
| rel-avito user-ad-visit | .0403 ± .0001 | cand-lgbm-ranker | .0420 | ID-GNN .0363 | .0363 | .0363 | .0193 | 2nd, .0017 below |
| rel-stack user-post-comment | **.1363 ± .0010** | cand-lgbm-ranker-plus | .1313 | ID-GNN .1303 | .1303 | .0004 | .0138 | **1st**, +.0050 (5 std) |
| rel-stack post-post-related | .1993 ± .0047 | cand-lgbm-ranker | .2178 | ID-GNN .1075 | .1075 | .0012 | .0174 | 2nd, .019 below |
| rel-amazon user-item-purchase | **.0288 ± .0001** | cand-lgbm-ranker | .0254 | LEDGER .0113 | .0010 | .0082 | .0006 | **1st**, +.0034 (about 50 std) |
| rel-amazon user-item-rate | **.0320 ± .0000** | cand-lgbm-ranker | .0231 | LEDGER .0117 | .0012 | .0079 | .0007 | **1st**, +.0089 |
| rel-amazon user-item-review | .0256 ± .0001 | cand-lgbm-ranker | .0295 | LEDGER .0079 | .0009 | .0044 | .0004 | 2nd, .0039 below |
| rel-hm user-item-purchase | **.0329 ± .0002** | cand-lgbm-ranker | .0326 | ID-GNN .0281 | .0281 | .0082 | .0089 | 1st by a hair, +.0003 (about 2 std) |

So on test, with the official evaluation: above every leaderboard entry on four tasks (rel-stack user-post-comment and
rel-amazon purchase and rate clearly, rel-hm purchase narrowly), a tie on one, second on five. Nothing is submitted yet;
the family's seed-0 tables wait for the submission issue. Ours beats every published baseline and both GNNs on all ten.
Val numbers, candidate recall and the grids are in each experiment README. Rows are in `results/<dataset>/<task>.jsonl`.

### Wave 1 detail (issue 12)

| task | cand-lgbm-ranker | LightGBM, published | LightGBM, ours |
|---|---|---|---|
| rel-f1 driver-circuit-compete | .8727 ± .0055 | .5956 | .6438 ± .0522 |
| rel-trial condition-sponsor-run | .1221 ± .0008 | .0480 | .0709 ± .0031 |
| rel-trial site-sponsor-run | .2834 ± .0008 | .0904 | .1664 ± .0149 |
| rel-avito user-ad-visit | .0403 ± .0001 | .0006 | .0141 ± .0004 |
| rel-stack user-post-comment | .1330 ± .0023 | .0002 | .0086 ± .0042 |
| rel-stack post-post-related | .1993 ± .0047 | .0168 | .0130 ± .0094 |

The trivial baselines reproduce the published values, except rel-f1 past visit (.255 against .184; see
`experiments/rel-f1/baselines/README.md`). Upstream's current LightGBM script beats its own published row on four tasks,
because it now has past-visit and popularity label features. Details are in each dataset's baselines README.

## Runs and cost

### Wave 1 (issue 12)

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

### Wave 2 (issue 21)

All on Google Cloud Spot in europe-west2, through `uv run cloud-run` with `experiments/rec-board/cloud21.py`, under the
issue's $10 approval. The four rankers ran side by side, 6 threads each; the trivial baselines on 1 thread beside them.

| run | machine | hours | cost | what it finished |
|---|---|---|---|---|
| 1 (709a569) | n2-custom-24-65536 | 0.14 | $0.05 | smoke: all four rankers on 200 seeds, leakage checks (0 mismatches), peak RSS |
| 2 (1ae6b3e) | n2-custom-24-131072 | 0.60 | $0.25 | the four trivial baselines; stopped: pair building too slow (a full lexsort in the walks' top-k) |
| 3 (24f32f5) | n2-custom-24-131072 | 0.08 | $0.04 | nothing; stopped: an `IndexError` in the new top-k on empty rows |
| 4 (e3e3f0e) | n2-custom-24-131072 | 2.03 | $0.85 | smoke again, then all four rankers: 3-config grid, 3 seeds, test once |
| **total** | | 2.85 | **$1.19** | |

Per-ranker runtimes in run 4: 73 min (rel-hm) to 112 min (amazon purchase), peak RSS 25–35 GB each. The trivial
baselines took 2–6 min each. Mac use was about 18 min of CPU for development and a rel-hm smoke. Each run's ledger push
failed inside the task (no `os` credential helper), so each was recorded with `uv run costs add`.

The trivial baselines: global popularity reproduces the published values exactly on all four. Past visit comes out a
little lower than published (amazon purchase .00058 against .00063, rate .00070 against .00074, review .00035 against
.00042, hm .0079 against .0089); both are far below everything else on the board.

## Reproduce

```sh
uv run cloud-run --issue 12 --approved <comment> --machine n2-custom-24-196608 --hours 5.5 \
    --checkpoint runs/cloud12 --run rec-wave1 -- \
    uv run python experiments/rec-board/cloud_jobs.py --out runs/cloud12
uv run python experiments/rec-board/baseline_rows.py --out runs/cloud12 --commit <sha> --machine "<machine>"

# wave 2
uv run cloud-run --issue 21 --approved <comment> --machine n2-custom-24-131072 --disk 300 --hours 6 \
    --checkpoint runs/cloud21 --run wave2 --download runs/cloud21 -- \
    uv run python experiments/rec-board/cloud21.py --out runs/cloud21 --threads 6 --free-gb 30 --heartbeat 5
cat runs/cloud21/ranker_<ds>_<task>/rows.jsonl >> results/<ds>/<task>.jsonl
uv run python experiments/rec-board/baseline_rows.py --out runs/cloud21 --only trivial --commit <sha> \
    --machine "<machine>" --tasks rel-amazon/user-item-purchase rel-amazon/user-item-rate \
    rel-amazon/user-item-review rel-hm/user-item-purchase
```

Each ranker also runs on its own: `uv run python experiments/<dataset>/<task>/cand-lgbm-ranker/run.py --test`.
