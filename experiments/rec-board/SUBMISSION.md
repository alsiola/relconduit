# Leaderboard submission draft: recommendation family (issues 22 and 32)

**Nothing has been submitted.** This is a draft of what the RelBench submission form asks for. The package is built
and checked. Submitting is the owner's call (issue 4).

## The package

- Ten prediction tables, one per task (`<dataset>__<task>.csv`), written by `relbench.submit.write_prediction_table`.
  Each table is either seed 0 of the task's recorded final config, or the rank average of that config's recorded seeds.
  The choice was made on val by a rule fixed before any test scoring (below).
- `python -m relbench.submit runs/ensemble32/preds --out runs/ensemble32/relconduit.zip` reports the recommendation
  family **validated** (10/10 tasks) and writes `relconduit-recommendation.zip` (27.1 MB, sha256 `6ecc5f7f…89ade9`).
  The zip is in the private bucket at `gs://oddsworth-lab-runs/relconduit-32-refit32/package/` (kept 30 days) and on
  the Mac under `runs/ensemble32/`. It and the tables stay out of git. The classification and regression families
  report FAIL in the same check because this package holds no tables for them.
- The leaderboard takes one table per task, with no seeds or mean. Picking the best seed on test would select on test
  labels, which rule 1 forbids. The seed mean ± std is in the Note and below.

Rebuild it:

```sh
uv run python experiments/rec-board/refit32.py --out runs/refit32   # eight refits (cloud: 2.9 h on 24 vCPUs)
uv run python experiments/rec-board/ensemble32.py --phase val       # score seeds and ensembles on val, apply the rule
uv run python experiments/rec-board/ensemble32.py --phase test      # write the final tables, score once, validate, zip
```

`ensemble32.py` reads two more inputs from `runs/inputs32/`. These are per-candidate score parquets from the
checkpoints of issues 28 and 29 in `gs://oddsworth-lab-runs`: rel-stack post-post-related `dense_minilm_100_*` and
rel-trial site-sponsor-run `seed{0,1,2}/*_scores.parquet`. Download them with
`CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS=1` (sliced downloads crash gcloud). The test phase does
`package22.py`'s gather and validate step and uses its task `ORDER`. `package22.py` itself is not rerun, because two
of its sources (post-post-related and site-sponsor-run) now come from newer experiments.

## Form fields

| field | draft |
|---|---|
| Name | `relconduit: candidates + LightGBM ranker` |
| In-context? | **No**. LightGBM is trained on each target database's train table. |
| URL | https://github.com/alsiola/relconduit |
| Repro link | https://github.com/alsiola/relconduit/blob/main/experiments/rec-board/SUBMISSION.md |
| Note | see below |
| File(s) | `relconduit-recommendation.zip` |

Note (shown on hover):

> Per task: time-safe candidate generation (the entity's own past targets, recent popularity, co-occurrence and
> multi-hop walks over the relational graph, plus frozen MiniLM text kNN on rel-stack post-post-related, all from rows
> strictly before the seed time), pair and entity aggregate features, and a LightGBM lambdarank/binary ranker.
> Configs were chosen on val only. Each table is seed 0 or the rank average of the seeds, chosen on val by a fixed
> rule. Test MAP over 3 seeds (5 on two tasks), mean ± std, is in the repro link. Built by Claude Code agents.

## Per-task numbers (test MAP@k, official `evaluate_task`)

| task | k | approach | table | submitted | seed mean ± std (seeds) | Kapso (leader) | next best published | submitted vs Kapso |
|---|---|---|---|---|---|---|---|---|
| rel-amazon/user-item-purchase | 10 | cand-lgbm-ranker | ensemble | 0.0290 | 0.0288 ± 0.0001 (3) | 0.0254 | LEDGER .0113 | +0.0036 |
| rel-amazon/user-item-rate | 10 | cand-lgbm-ranker (refit32 seeds) | ensemble | 0.0321 | 0.0321 ± 0.0001 (3) | 0.0231 | LEDGER .0117 | +0.0090 |
| rel-amazon/user-item-review | 10 | cand-lgbm-ranker | ensemble | 0.0257 | 0.0256 ± 0.0001 (3) | 0.0295 | LEDGER .0079 | -0.0038 |
| rel-avito/user-ad-visit | 12 | cand-lgbm-ranker | ensemble | 0.0404 | 0.0403 ± 0.0001 (3) | 0.0420 | ID-GNN .0363 | -0.0016 |
| rel-f1/driver-circuit-compete | 10 | cand-lgbm-ranker | seed 0 | 0.8769 | 0.8727 ± 0.0055 (3) | 0.8795 | GelGT .7456 | -0.0026 |
| rel-hm/user-item-purchase | 12 | cand-lgbm-ranker | ensemble | 0.0331 | 0.0329 ± 0.0002 (3) | 0.0326 | ID-GNN .0281 | +0.0005 |
| rel-stack/user-post-comment | 100 | cand-lgbm-ranker-plus | seed 0 | 0.1353 | 0.1363 ± 0.0010 (5) | 0.1313 | ID-GNN .1303 | +0.0040 |
| rel-stack/post-post-related | 100 | dense-ranker (issue 29) | seed 0 | 0.2504 | 0.2484 ± 0.0019 (3) | 0.2178 | ID-GNN .1075 | +0.0326 |
| rel-trial/condition-sponsor-run | 10 | cand-lgbm-ranker-plus | ensemble | 0.1233 | 0.1223 ± 0.0030 (5) | 0.1228 | LEDGER .0598 | +0.0005 |
| rel-trial/site-sponsor-run | 10 | cand-lgbm-v2/final (issue 28) | ensemble | 0.3048 | 0.3037 ± 0.0009 (3) | 0.3333 | LEDGER .1133 | -0.0285 |
| **board mean** | | | | **0.1851** | 0.1843 | **.1837** | ID-GNN .0933 | **+0.0014** |

On the recommendation board (`leaderboard.json`, read 2026-11-14), the submitted tables' mean, **.1851**, is above
Kapso's .1837 by .0014. It would place **1st of 9**, ahead of Kapso, ID-GNN (.0933), LEDGER (.0875), GelGT (.0793) and
LightGBM (.0759). The margin is small. From the seed stds, the board mean has a seed-to-seed sd of about .0007, so .0014
is about two sds. The mean of the seed means, .1843, is only .0006 above Kapso. Kapso publishes one number per task,
with no spread. The leaderboard must be read again before submitting.

The tables beat Kapso on **six** tasks: rel-amazon purchase and rate, rel-hm purchase, rel-stack user-post-comment and
post-post-related, and rel-trial condition-sponsor-run. By the seed mean the count is **five**: condition-sponsor-run
(.1223 against .1228) is a tie within its seed std, and the ensemble table puts it just above. rel-hm purchase is +.0005
by table and +.0003 by mean, also close. The tables beat every other entry on all ten tasks. The four tasks below Kapso
are rel-amazon review, rel-avito, rel-f1 and rel-trial site-sponsor-run. Site-sponsor-run is still the largest gap,
-.0285, down from -.0492 before issue 28.

Seed 0 is not the best seed on every task. On rel-stack user-post-comment it is the second lowest of five (.1353; the
seeds run .1353 to .1374). There the ensemble was not better on val, so seed 0 stands. It is not swapped after scoring.

## The ensemble rule (issue 32)

Fixed in commit 3cc7350, before any ensemble's test table was scored. For each task, the ensemble is the rank average
of the final seeds' scores: the mean within-row rank over seeds, with ties sharing their mean rank. It replaces seed 0
only if, on val (official `task.evaluate`, every val row):

- its MAP is above both seed 0's and the seeds' mean MAP, and
- a paired bootstrap over val rows (2,000 resamples) puts it above the seed mean in at least 90% of resamples.

The decisions, scored on val only, were committed in 0a15397 before the test phase ran
(`ensemble32_val.md`, `ensemble32_val.json`):

| task | seeds | val seed 0 | val seed mean | val ensemble | ens − seed mean [95% CI] | P(ens > mean) | use |
|---|---|---|---|---|---|---|---|
| rel-amazon/user-item-purchase | 3 | 0.0352 | 0.0353 | 0.0354 | +0.0002 [+0.0001, +0.0002] | 1.00 | ensemble |
| rel-amazon/user-item-rate | 3 | 0.0382 | 0.0382 | 0.0383 | +0.0001 [+0.0000, +0.0001] | 0.99 | ensemble |
| rel-amazon/user-item-review | 3 | 0.0309 | 0.0309 | 0.0311 | +0.0001 [+0.0001, +0.0002] | 1.00 | ensemble |
| rel-avito/user-ad-visit | 3 | 0.0577 | 0.0577 | 0.0578 | +0.0001 [-0.0000, +0.0002] | 0.95 | ensemble |
| rel-f1/driver-circuit-compete | 3 | 0.7346 | 0.7319 | 0.7406 | +0.0086 [-0.0104, +0.0276] | 0.80 | seed 0 |
| rel-hm/user-item-purchase | 3 | 0.0297 | 0.0297 | 0.0299 | +0.0002 [+0.0001, +0.0003] | 1.00 | ensemble |
| rel-stack/user-post-comment | 5 | 0.1654 | 0.1643 | 0.1646 | +0.0003 [-0.0009, +0.0016] | 0.68 | seed 0 |
| rel-stack/post-post-related | 3 | 0.2319 | 0.2309 | 0.2298 | -0.0011 [-0.0049, +0.0029] | 0.30 | seed 0 |
| rel-trial/condition-sponsor-run | 5 | 0.1176 | 0.1171 | 0.1178 | +0.0007 [-0.0002, +0.0015] | 0.94 | ensemble |
| rel-trial/site-sponsor-run | 3 | 0.2781 | 0.2780 | 0.2785 | +0.0005 [+0.0002, +0.0009] | 1.00 | ensemble |

On test, each of the seven ensemble tables beat its own seeds' test mean (by .00006 to .0011; for amazon rate those are the refit32 seeds), so val's call held on every task.
Each final table was scored on test exactly once. The seven ensemble rows are in `results/` under
`experiments/rec-board/ensemble32 (<approach>)`.

## How the two rules are met

**Rule 1: no information at or after the prediction time.**

- Every candidate source and pair feature (`src/relconduit/link.py`) reads only database rows dated strictly before
  the row's seed time (`e[T] < t`). Entity features (`src/relconduit/features.py`) do the same per `(entity, seed
  time)`. Text vocabularies come from rows before the val timestamp. post-post-related's MiniLM kNN index holds only
  posts dated before each seed time (issue 29).
- Before any fit, each run checks this on the real database. It compares pairs built on the full database with pairs
  built on `truncate_before(db, T)`, per seed time, for up to 40 val and 40 train seeds. Every run, including these
  refits, found 0 mismatches. `tests/test_link.py` shows the check catches a planted `<=`.
- The database is `task.get_db()` (rows up to the test timestamp). `upto_test_timestamp=False` is not used.
- Test labels were never used for training, early stopping, choosing a candidate set, train size or config, or
  choosing between seed 0 and the ensemble. All of those were chosen on val with the official `task.evaluate`.
- What test results did influence: which tasks got more work. Issue 20 widened the candidates on the two tasks closest
  to the leader after wave 1's test scores. Issues 28 and 29 took on site-sponsor-run and post-post-related because
  they were furthest below Kapso on test. In each case the choice between the old and new approach was still made on
  val. The leaderboard's other entries face the same trade-off, and the README's rule is about selection on test
  labels, which this is not. We disclose it all the same.

**Rule 2 (in-context):** not claimed. The form says No.

## Reproduction (recorded vs refit32, per task)

For eight tasks, `refit32.py` refit the recorded final config and seeds to keep each seed's top-k table. The other
two (post-post-related, site-sponsor-run) kept their per-candidate scores in issues 29 and 28, so their tables come
straight from those scores.

| task | approach | seeds | recorded test (mean ± std) | refit test (mean ± std) | largest per-seed gap | source |
|---|---|---|---|---|---|---|
| rel-amazon/user-item-purchase | cand-lgbm-ranker | 3 | 0.0288 ± 0.0001 | 0.0288 ± 0.0001 | 0.000038 | refit32 |
| rel-amazon/user-item-rate | cand-lgbm-ranker | 3 | 0.0320 ± 0.0000 | 0.0321 ± 0.0001 | 0.000148 | refit32 (other config, see below) |
| rel-amazon/user-item-review | cand-lgbm-ranker | 3 | 0.0256 ± 0.0001 | 0.0256 ± 0.0001 | 0.000001 | refit32 |
| rel-avito/user-ad-visit | cand-lgbm-ranker | 3 | 0.0403 ± 0.0001 | 0.0403 ± 0.0001 | 0.000000 | refit32 |
| rel-f1/driver-circuit-compete | cand-lgbm-ranker | 3 | 0.8727 ± 0.0055 | 0.8727 ± 0.0055 | 0.000000 | refit32 |
| rel-hm/user-item-purchase | cand-lgbm-ranker | 3 | 0.0329 ± 0.0002 | 0.0329 ± 0.0002 | 0.000000 | refit32 |
| rel-stack/user-post-comment | cand-lgbm-ranker-plus | 5 | 0.1363 ± 0.0010 | 0.1363 ± 0.0010 | 0.000000 | refit32 |
| rel-stack/post-post-related | dense-ranker | 3 | 0.2484 ± 0.0019 | (kept scores) | — | issue 29 checkpoint |
| rel-trial/condition-sponsor-run | cand-lgbm-ranker-plus | 5 | 0.1223 ± 0.0030 | 0.1223 ± 0.0030 | 0.000000 | refit32 |
| rel-trial/site-sponsor-run | cand-lgbm-v2/final | 3 | 0.3037 ± 0.0009 | (kept scores) | — | issue 28 checkpoint |

Six refits reproduce every recorded per-seed test value to 6 decimals; avito, f1, hm, user-post-comment and
condition-sponsor-run match exactly, and amazon review is off by .000001. Val matches the record on all of them, and on
the two kept-score tasks. The refits ran with 3 LightGBM threads, but issue 21's amazon records used 6, so two amazon tasks moved:

- **rel-amazon/user-item-purchase**: seed 2 stopped at round 239 instead of 242 (test .028857 against .028895).
  Seeds 0 and 1 match. This is LightGBM's multithreaded histogram summation under a different thread count.
- **rel-amazon/user-item-rate**: the same effect moved the 31-leaf lambdarank fit in the 3-config val grid from .0381
  to .03824, enough to beat the recorded choice (127 leaves, .03818). So the refit's three seeds are the 31-leaf
  config: val .0382 ± .0000, test .0321 ± .0001, against the record's .0320 ± .0000. Its rows are in `results/` as
  `experiments/rec-board/refit32 (…/user-item-rate/cand-lgbm-ranker)`. The rate ensemble averages these seeds, not the
  recorded ones. The config was still chosen on val. The difference is within noise, and the submitted table's test
  value (.0321) is what the board would see.

The test phase also checks each seed-0 table it submits against its recorded row (tolerance 5e-4). All three seed-0
tables (f1, user-post-comment, post-post-related) match.

Refit run: one Google Cloud Spot `n2-custom-24-196608` (europe-west2), eight jobs, at most four side by side at 3
threads each, after a smoke phase on all eight. The command ran 2.91 h and the machine about 2.95 h, **$1.46** (ledger
entry `relconduit-32-1a1dca80a5b`, entered by hand because the runner's push to main failed). Per job: rel-f1 0.6 min,
avito 11 min, hm 66 min, condition-sponsor-run 87 min, amazon review 108 min, user-post-comment 111 min (peak 61 GB),
amazon purchase 152 min, amazon rate 158 min. Mac: the val and test phases take about a minute each.
