# Leaderboard submission draft: recommendation family (issue 22)

**Nothing has been submitted.** This is the draft of what the RelBench submission form asks for, with the package
built and checked. Submitting is the owner's call (issue 4).

## The package

- Ten prediction tables, one per task (`<dataset>__<task>.csv`), each written by
  `relbench.submit.write_prediction_table` from **seed 0** of that task's recorded final config.
- `uv run python -m relbench.submit runs/submission22/preds --out runs/submission22/relconduit.zip` reports the
  recommendation family **validated** and writes `relconduit-recommendation.zip` (27.1 MB, sha256 `e3e28d7c…70d3`). It is in the
  private bucket at `gs://oddsworth-lab-runs/relconduit-22-refit22/package/` (kept 30 days) and on the Mac under
  `runs/submission22/`; it and the tables stay out of git.
- The leaderboard takes one table per task, with no seeds or mean. Seed 0 was fixed before anything was scored.
  Picking the best seed on test would select on test labels, which rule 1 forbids. The 3-seed (or 5-seed) mean ± std is
  in the Note and below.

Rebuild it:

```sh
uv run python experiments/rec-board/refit22.py --out runs/refit22                # the six refits (cloud: 2.3 h on 24 vCPUs)
uv run python experiments/rec-board/package22.py --fetch --out runs/submission22  # gather, compare, validate, zip
```

`--fetch` copies wave 2's four seed-0 tables from issue 21's cloud checkpoint (`gs://oddsworth-lab-runs`, kept 30
days). Without the bucket, rerun those four with `experiments/rec-board/cloud21.py` (about 2 h on 24 vCPUs).

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
> multi-hop walks over the relational graph, all from rows strictly before the seed time), pair and entity
> aggregate features, and a LightGBM lambdarank/binary ranker. Configs chosen on val only. Tables are seed 0. Test
> MAP over 3 seeds (5 on two tasks), mean ± std, is in the repro link. Built by Claude Code agents.

## Per-task numbers (test MAP@k, official `evaluate_task`)

| task | k | table (seed 0) | mean ± std (seeds) | Kapso (leader) | next best published | seed 0 vs Kapso |
|---|---|---|---|---|---|---|
| rel-amazon/user-item-purchase | 10 | 0.0289 | 0.0288 ± 0.0001 (3) | 0.0254 | LEDGER .0113 | +0.0035 |
| rel-amazon/user-item-rate | 10 | 0.0320 | 0.0320 ± 0.0000 (3) | 0.0231 | LEDGER .0117 | +0.0089 |
| rel-amazon/user-item-review | 10 | 0.0255 | 0.0256 ± 0.0001 (3) | 0.0295 | LEDGER .0079 | -0.0040 |
| rel-avito/user-ad-visit | 12 | 0.0402 | 0.0403 ± 0.0001 (3) | 0.0420 | ID-GNN .0363 | -0.0018 |
| rel-f1/driver-circuit-compete | 10 | 0.8769 | 0.8727 ± 0.0055 (3) | 0.8795 | GelGT .7456 | -0.0026 |
| rel-hm/user-item-purchase | 12 | 0.0327 | 0.0329 ± 0.0002 (3) | 0.0326 | ID-GNN .0281 | +0.0001 |
| rel-stack/user-post-comment | 100 | 0.1353 | 0.1363 ± 0.0010 (5) | 0.1313 | ID-GNN .1303 | +0.0040 |
| rel-stack/post-post-related | 100 | 0.2033 | 0.1993 ± 0.0047 (3) | 0.2178 | ID-GNN .1075 | -0.0145 |
| rel-trial/condition-sponsor-run | 10 | 0.1233 | 0.1223 ± 0.0031 (5) | 0.1228 | LEDGER .0598 | +0.0005 |
| rel-trial/site-sponsor-run | 10 | 0.2841 | 0.2834 ± 0.0008 (3) | 0.3333 | LEDGER .1133 | -0.0492 |
| **board mean** | | **0.1782** | | **.1837** | ID-GNN .0933 | -0.0055 |

On the recommendation board (`leaderboard.json`, read 2026-11-14) this would place **2nd of 9**: below Kapso (.1837),
well above ID-GNN (.0933), LEDGER (.0875), GelGT (.0793) and LightGBM (.0759). On the submitted seed-0 tables it is above
Kapso on five tasks: rel-amazon purchase and rate, rel-stack user-post-comment, rel-hm purchase and rel-trial
condition-sponsor-run. By the seed mean it is four, since condition-sponsor-run (.1223 against .1228) is a tie within
the seed std. It is above every other entry on all ten. Not state of the art: the board ranks the mean, and Kapso
leads it. Seed 0 is not the best seed on every task (on rel-stack user-post-comment it is the second lowest of five);
it was fixed before scoring and is not swapped after.

## How the two rules are met

**Rule 1: no information at or after the prediction time.**

- Every candidate source and pair feature (`src/relconduit/link.py`) reads only database rows dated strictly before
  the row's seed time (`e[T] < t`). Entity features (`src/relconduit/features.py`) do the same per `(entity, seed
  time)`. Text vocabularies come from rows before the val timestamp.
- Before any fit, each run checks this on the real database. It compares pairs built on the full database with pairs
  built on `truncate_before(db, T)`, per seed time, for up to 40 val and 40 train seeds. Every run, including these refits,
  found 0 mismatches. `tests/test_link.py` shows the check catches a planted `<=`.
- The database is `task.get_db()` (rows up to the test timestamp). `upto_test_timestamp=False` is not used.
- Test labels were never used for training, early stopping, or choosing a candidate set, train size or config. All
  of those were chosen on val with the official `task.evaluate`, and test was scored once per run.
- What test results did influence: which tasks got a second round of work. Issue 20 widened the candidates on the two
  tasks closest to the leader after wave 1's test scores. The choice between the two rounds was still made on val:
  `cand-lgbm-ranker-plus` has the higher val MAP on both tasks. The leaderboard's other entries face the same
  trade-off, and the README's rule is about selection on test labels, which this is not. We disclose it all the same.

**Rule 2 (in-context):** not claimed. The form says No.

## Reproduction (recorded vs refit, per task)

| task | approach | seeds | recorded test (mean ± std) | refit test (mean ± std) | largest per-seed gap | seed-0 table | source |
|---|---|---|---|---|---|---|---|
| rel-amazon/user-item-purchase | cand-lgbm-ranker | 3 | 0.0288 ± 0.0001 | 0.0288 ± 0.0001 | 0.000000 | 0.0289 | issue 21 checkpoint (seed-0 table as written) |
| rel-amazon/user-item-rate | cand-lgbm-ranker | 3 | 0.0320 ± 0.0000 | 0.0320 ± 0.0000 | 0.000000 | 0.0320 | issue 21 checkpoint (seed-0 table as written) |
| rel-amazon/user-item-review | cand-lgbm-ranker | 3 | 0.0256 ± 0.0001 | 0.0256 ± 0.0001 | 0.000000 | 0.0255 | issue 21 checkpoint (seed-0 table as written) |
| rel-avito/user-ad-visit | cand-lgbm-ranker | 3 | 0.0403 ± 0.0001 | 0.0403 ± 0.0001 | 0.000000 | 0.0402 | refit (refit22.py) |
| rel-f1/driver-circuit-compete | cand-lgbm-ranker | 3 | 0.8727 ± 0.0055 | 0.8727 ± 0.0055 | 0.000000 | 0.8769 | refit (refit22.py) |
| rel-hm/user-item-purchase | cand-lgbm-ranker | 3 | 0.0329 ± 0.0002 | 0.0329 ± 0.0002 | 0.000000 | 0.0327 | issue 21 checkpoint (seed-0 table as written) |
| rel-stack/user-post-comment | cand-lgbm-ranker-plus | 5 | 0.1363 ± 0.0010 | 0.1363 ± 0.0010 | 0.000000 | 0.1353 | refit (refit22.py) |
| rel-stack/post-post-related | cand-lgbm-ranker | 3 | 0.1993 ± 0.0047 | 0.1993 ± 0.0047 | 0.000000 | 0.2033 | refit (refit22.py) |
| rel-trial/condition-sponsor-run | cand-lgbm-ranker-plus | 5 | 0.1223 ± 0.0030 | 0.1223 ± 0.0031 | 0.000284 | 0.1233 | refit (refit22.py) |
| rel-trial/site-sponsor-run | cand-lgbm-ranker | 3 | 0.2834 ± 0.0008 | 0.2834 ± 0.0008 | 0.000000 | 0.2841 | refit (refit22.py) |

Board mean of the seed-0 tables: 0.1782.

Wave 2's four tables are the ones issue 21 wrote, re-scored with `evaluate_task`: identical. The six refits reproduce
every recorded per-seed test value to 6 decimals except one: rel-trial condition-sponsor-run seed 4, .124763 against
.124479 (gap .00028, a tenth of the seed std). Its val (.1178) matches the record to 4 decimals; the likely cause is
LightGBM's multithreaded histogram summation under a different thread count (4 here, 12 in issue 18). The submitted
table is seed 0, which matches exactly (.123260). Val per seed matches the record on every task.

Refit run: one Google Cloud Spot `n2-custom-24-131072` (europe-west2), six jobs side by side at 4 threads each, after
a 150-seed smoke phase on all six; 2.31 h, **$0.89** (ledger entry `relconduit-22-1a1d98aa261`). Per job: rel-f1 0.6
min, post-post-related 10 min, user-ad-visit 12 min, site-sponsor-run 14 min, condition-sponsor-run 74 min,
user-post-comment 129 min (peak 75 GB). Mac use: about 2 min of CPU for a smoke test, plus packaging and scoring.
