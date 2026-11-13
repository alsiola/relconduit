# rel-trial/condition-sponsor-run: wider candidates, all train rows, tuning, 5 seeds

Issue 18 follows up `../cand-lgbm-ranker/` (issue 12), which tied Kapso at .1221 ± .0008 against .1228. The task: for each condition seed row, rank the 10 sponsors most likely to run a study on it in the next window. The metric is MAP@10.

**Result: test MAP@10 0.1223 ± 0.0030 over 5 seeds, against Kapso's 0.1228. Still a tie, not a lead; a negative result.** Candidate recall went from .360 to .477. The extra recall and all the train rows bought about .001 on val and nothing on test.

## What it does

`run.py` holds the task's config. The protocol is `src/relconduit/link_search.py`, the same staged, val-only search as the rel-stack task (see `experiments/rel-stack/user-post-comment/cand-lgbm-ranker-plus/README.md`). Candidates and pair features come from `src/relconduit/link.py`.

The candidate set `plus` is #12's set (`base`) with:

- wider sources (direct n 200, covisit n 200 with a 2-year half-life, recent covisit n 100);
- `covisit_long`: co-occurrence with a 5-year half-life;
- `cocond`, `cocond_recent`: condition → study → co-listed condition → sponsor (half-life and 2-year window);
- `words`: sponsors of conditions whose MeSH term shares a word with this one (stop words dropped);
- `facility`: sponsors running studies at the facilities this condition's sponsors use, within 2 years, fan-out capped at 300;
- popularity sources over 365 days at n 100.

**Leakage.** Every edge reads only rows dated strictly before the seed time; MeSH words are static. The real-database pair leakage check (full vs `truncate_before`, 80 val and train seeds) gave **0 mismatches** for both candidate sets before any fit.

**Shared Dataset and binning.** The search builds one LightGBM `Dataset` per (candidate set, train size) with `feature_pre_filter=False`. Every config and seed reuses it, so bins are fixed per train size and configs differ only in tree parameters. Stage 3's reference is stage 2's cached all-rows fit. The final stage reuses tuning seeds 0–2 and fits seeds 3 and 4.

## Val search (MAP@10, 3 tuning seeds, mean ± std)

| stage | candidates | train seed rows | config | val |
|---|---|---|---|---|
| 1 candidates | base | 12,000 | #12's: binary, 63 leaves, mcs 20, ff 0.7, lr 0.05 | 0.1152 ± 0.0006 |
| 1 candidates | **plus** | 12,000 | same | 0.1161 ± 0.0013 |
| 2 train size | plus | 24,000 | same | 0.1162 ± 0.0009 |
| 2 train size | plus | **all 36,934** | same | 0.1174 ± 0.0005 |
| 3 config | plus | all | lr 0.02 | 0.1173 ± 0.0009 |
| 3 config | plus | all | lr 0.02, 127 leaves | 0.1178 ± 0.0002 |
| 3 config | plus | all | lr 0.02, mcs 100 | 0.1173 ± 0.0008 |
| 3 config | plus | all | lr 0.02, `lambda_l2` 10 | 0.1171 ± 0.0006 |
| 3 config | plus | all | lr 0.02, ff 0.5 | 0.1169 ± 0.0007 |
| 3 config | plus | all | **lr 0.02, lambdarank, mcs 100** | **0.1181 ± 0.0006** |
| 3 config | plus | all | same, `lambdarank_truncation_level` 40 | 0.1167 ± 0.0037 |

Every step is within about one std of the last. The search picked lambdarank by .0003 over 127-leaf binary. Train: 36,934 seed rows, 30,287 with a true sponsor among their candidates, 10.1M pairs, 132 features, positive rate 2.4%.

## Candidate recall on val

| source | base | plus |
|---|---|---|
| **all** | **0.360** | **0.477** |
| candidates per seed | 142 | 377 |
| ran (direct) | 0.232 | 0.249 |
| covisit | 0.321 | 0.401 |
| covisit_recent | 0.198 | 0.260 |
| covisit_long | | 0.306 |
| cocond | | 0.377 |
| cocond_recent | | 0.260 |
| words | | 0.256 |
| facility | | 0.249 |
| sponsor trials top 365d | 0.125 | 0.190 |
| ran top 365d | 0.129 | 0.188 |

Recall rose by a third, but the new true sponsors are mostly ones the ranker can't lift into the top 10. MAP@10 is set by the head of the list, and the head comes from `cocond` and `covisit` ranks (the top features by gain: `cocond__rank`, `cocond__score`, `covisit__rank`, `covisit__score`, `ran__count`).

## Final (5 seeds, chosen config, test scored once)

| split | MAP@10 | per seed |
|---|---|---|
| val | 0.1171 ± 0.0020 | 0.1176 / 0.1187 / 0.1180 / 0.1136 / 0.1176 |
| test | **0.1223 ± 0.0030** | 0.1233 / 0.1236 / 0.1232 / 0.1169 / 0.1245 |

**Seed 3 stopped early.** Its best val round was 6, and patience is 100 rounds. The other seeds stopped at rounds 505–609. That seed's model is under-trained, and it accounts for most of the std. It is reported as it fell. Dropping it after seeing test would be choosing on test. Without it, the mean would be about .1237, still inside 2×std of Kapso. lambdarank at lr 0.02 with early stopping looks fragile here. A follow-up should set a minimum number of rounds, or stop on a smoothed val curve.

`task.evaluate` on the prediction tables gives the same numbers. A 2,000-resample bootstrap over the 2,057 test rows gives a **95% CI of [0.1131, 0.1317]** (bootstrap std 0.0047). In 45% of resamples this run's MAP is above .1228.

## Against the leaderboard (test MAP@10)

| | test |
|---|---|
| Kapso (leader) | 0.1228 |
| **cand-lgbm-ranker-plus (this), 5 seeds** | **0.1223 ± 0.0030** |
| cand-lgbm-ranker (issue 12), 3 seeds | 0.1221 ± 0.0008 |
| Past visit | 0.0830 |
| LightGBM (upstream) | 0.0480 |
| GNN | 0.0350 |
| Global popularity | 0.0252 |
| ID-GNN | 0.0033 |

The clear-lead bar: the margin is −0.0005, and the bar is 2 × 0.0030 = 0.0061. **It does not clear the bar.** This is a tie with Kapso, as in issue 12. No submission check was run, because the issue asks for one only on a task that clears the bar.

## Run

```sh
uv run python experiments/rel-trial/condition-sponsor-run/cand-lgbm-ranker-plus/run.py --smoke 150   # end to end on a sample, no results
uv run python experiments/rel-trial/condition-sponsor-run/cand-lgbm-ranker-plus/run.py --test        # the full search, test once
uv run python experiments/rec-board/cloud18.py --out runs/cloud18                                     # both tasks side by side, as reported
```

The reported run used `experiments/rec-board/cloud18.py` (commit a853ca1) on a Google Cloud `n2-custom-24-131072` Spot machine (24 vCPU, 128 GB, europe-west2-b), beside rel-stack/user-post-comment, with 12 threads each. Runtime was **211 min**, and peak RSS was 28.0 GB. The machine cost **$1.52** for both tasks (ledger entry `relconduit-18-1a1d5b1ea97`, recorded by hand because the in-task ledger push failed).
