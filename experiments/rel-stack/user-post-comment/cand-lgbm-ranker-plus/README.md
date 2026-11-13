# rel-stack/user-post-comment: wider candidates, all train rows, tuning, 5 seeds

Issue 18 follows up `../cand-lgbm-ranker/` (issue 12), which tied Kapso at .1330 ± .0023 against .1313. The task: for each `rel-stack` user seed row, rank the 100 posts it will most likely comment on in the next window. The metric is MAP@100.

**Result: test MAP@100 0.1363 ± 0.0010 over 5 seeds, against Kapso's 0.1313.** That beats the clear-lead bar the issue set (more than 2× the seed std above the leader). The bootstrap over the 758 test rows is much wider, though: see below.

## What it does

`run.py` holds the task's config. The protocol is in `src/relconduit/link_search.py`: a staged search where every choice is made on val and test is scored once, for the final config only. Candidates and pair features come from `src/relconduit/link.py`, as in issue 12.

The candidate set `plus` is #12's set (`base`) with:

- wider sources (thread walk n 300 → 1000, covisit and tags n 300, direct sources n 300);
- `linked`: posts in threads linked (duplicate or related, `postLinks`) to threads the user touched;
- `people`: recent posts by users this user comments with, either way round (commenter ↔ post owner);
- `tags91`: posts with comments in the last 91 days in the user's tags (beside the 30-day one);
- `post_activity__top30d`: the 30-day most active posts.

**Leakage.** Every edge reads only rows dated strictly before the seed time. Before any fit, the run compared pairs built on the full database with pairs built on `truncate_before(db, T)` for 80 val and train seeds, for each candidate set. Both had **0 mismatches**. `tests/test_link.py` shows the check catches a planted `<=`.

**Shared Dataset and binning.** The search builds one LightGBM `Dataset` per (candidate set, train size), with `feature_pre_filter=False`. Every config and seed reuses it, so the feature bins (LightGBM defaults, `max_bin` 255) are fixed per train size and only tree parameters change between configs. Comparisons within a stage are therefore like for like. A config whose best setting would need different binning (`max_bin`, `min_data_in_bin`) is out of reach of this grid. Stage 3's reference row is stage 2's all-rows fit, reused from cache. The final stage reuses tuning seeds 0–2 and fits only seeds 3 and 4.

## Val search (MAP@100, 3 tuning seeds, mean ± std)

| stage | candidates | train seed rows | config | val |
|---|---|---|---|---|
| 1 candidates | base | 6,000 | #12's: lambdarank, 15 leaves, mcs 20, ff 0.7, lr 0.05 | 0.1586 ± 0.0011 |
| 1 candidates | **plus** | 6,000 | same | 0.1606 ± 0.0005 |
| 2 train size | plus | 12,000 | same | 0.1620 ± 0.0006 |
| 2 train size | plus | **all 21,240** | same | 0.1633 ± 0.0012 |
| 3 config | plus | all | **lr 0.02** | **0.1649 ± 0.0011** |
| 3 config | plus | all | lr 0.02, 31 leaves | 0.1639 ± 0.0005 |
| 3 config | plus | all | lr 0.02, mcs 100 | 0.1645 ± 0.0008 |
| 3 config | plus | all | lr 0.02, `lambda_l2` 10 | 0.1642 ± 0.0016 |
| 3 config | plus | all | lr 0.02, ff 0.5 | 0.1637 ± 0.0006 |
| 3 config | plus | all | lr 0.02, `lambdarank_truncation_level` 20 | 0.1634 ± 0.0012 |
| 3 config | plus | all | lr 0.02, binary | 0.1568 ± 0.0028 |

Each lever adds a little: wider candidates +.0020, all train rows +.0027, lr 0.02 +.0016. Only lr 0.02 among the configs is clearly above the reference; the rest sit within noise of each other. Train: 21,240 seed rows, 9,011 with a true post among their candidates (the rest are dropped), 11.9M pairs, 351 features, positive rate 0.11%.

## Candidate recall on val

| source | base | plus |
|---|---|---|
| **all** | **0.293** | **0.314** |
| candidates per seed | 452 | 1,097 |
| comment | 0.112 | 0.116 |
| history | 0.199 | 0.207 |
| own | 0.204 | 0.212 |
| thread | 0.263 | 0.286 |
| covisit | 0.110 | 0.113 |
| tags (30d) | 0.056 | 0.060 |
| linked | | 0.039 |
| people | | 0.038 |
| tags91 | | 0.067 |
| comment top 30d | 0.005 | 0.005 |
| post activity top 30d | | 0.003 |

Recall is still the ceiling. Two-thirds of the posts a user goes on to comment on are not candidates. The new sources mostly overlap the thread walk.

## Final (5 seeds, chosen config, test scored once)

| split | MAP@100 | per seed |
|---|---|---|
| val | 0.1643 ± 0.0011 | 0.1654 / 0.1636 / 0.1656 / 0.1634 / 0.1633 |
| test | **0.1363 ± 0.0010** | 0.1353 / 0.1365 / 0.1371 / 0.1353 / 0.1374 |

Best rounds 317–440. `task.evaluate` on the prediction tables gives the same numbers. Top features by gain: `thread__rank`, `thread__score`, `own__count`, `post_activity__pop_days_since_last`, `own__days_since_last`, `cand__thread`.

**Test-sample noise.** The seed std covers only training noise. A 2,000-resample bootstrap over the 758 test rows (seed-mean predictions) gives a **95% CI of [0.1156, 0.1582]** and a bootstrap std of 0.011. Kapso's .1313 is well inside it. In 68% of resamples this run's MAP is above .1313. Kapso's own number has the same test-sample noise. Both are scored on the same 758 rows, so the difference between them is less noisy than either number alone. Still, a .005 margin on 758 rows is not a statistically settled lead.

## Against the leaderboard (test MAP@100)

| | test |
|---|---|
| **cand-lgbm-ranker-plus (this), 5 seeds** | **0.1363 ± 0.0010** |
| cand-lgbm-ranker (issue 12), 3 seeds | 0.1330 ± 0.0023 |
| Kapso (leader) | 0.1313 |
| ID-GNN | 0.1303 |
| Past visit | 0.0138 |
| GNN | 0.0004 |
| LightGBM (upstream) | 0.0002 |
| Global popularity | 0.0002 |

The clear-lead bar: the margin over Kapso is 0.0050, and the bar is 2 × 0.00095 = 0.0019. **It clears the bar.** This is the top of the board on test with the official evaluation, under the bar fixed in issue 18, with the test-sample caveat above.

**Submission check.** The issue asks for this because the task cleared the bar. The seed-0 prediction table, written by `write_prediction_table`, is at `runs/cloud18/search_rel-stack_user-post-comment/work/rel-stack/user-post-comment/cand-lgbm-ranker-plus/submission/`. It is not committed. `uv run python -m relbench.submit <dir>` validates it: `✓ rel-stack/user-post-comment 13.53`. The command still exits 1, because a full leaderboard package needs every task and this folder holds one. Whether and how to submit is the owner's call on issue 4.

## Run

```sh
uv run python experiments/rel-stack/user-post-comment/cand-lgbm-ranker-plus/run.py --smoke 150   # end to end on a sample, no results
uv run python experiments/rel-stack/user-post-comment/cand-lgbm-ranker-plus/run.py --test        # the full search, test once, rows to runs/
uv run python experiments/rec-board/cloud18.py --out runs/cloud18                                 # both tasks side by side, as reported
```

The reported run used `experiments/rec-board/cloud18.py` (commit a853ca1) on a Google Cloud `n2-custom-24-131072` Spot machine (24 vCPU, 128 GB, europe-west2-b). It ran beside rel-trial/condition-sponsor-run with 12 threads each. Runtime was **224 min** for the search (pairs, leakage checks, 11 distinct tuning fits × 3 seeds, 2 final fits, test), and peak RSS was 76.3 GB. The machine cost **$1.52** for both tasks (ledger entry `relconduit-18-1a1d5b1ea97`, recorded by hand because the in-task ledger push failed).
