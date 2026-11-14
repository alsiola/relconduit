# Leaderboard submission draft: recommendation family (issue 22)

**Nothing has been submitted.** This is the draft of what the RelBench submission form asks for, with the package
built and checked. Submitting is the owner's call (issue 4).

## The package

- Ten prediction tables, one per task (`<dataset>__<task>.csv`), each written by
  `relbench.submit.write_prediction_table` from **seed 0** of that task's recorded final config.
- `uv run python -m relbench.submit runs/submission22/preds --out runs/submission22/relconduit.zip` reports the
  recommendation family **validated** and writes `relconduit-recommendation.zip`. The zip is attached to issue 22; it
  and the tables stay out of git.
- The leaderboard takes one table per task, with no seeds or mean. Seed 0 was fixed before anything was scored.
  Picking the best seed on test would select on test labels, which rule 1 forbids. The 3-seed (or 5-seed) mean ± std is
  in the Note and below.

Rebuild it:

```sh
uv run python experiments/rec-board/refit22.py --out runs/refit22                # the six refits (cloud: ~TBD on 24 vCPUs)
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

TBD

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

TBD
