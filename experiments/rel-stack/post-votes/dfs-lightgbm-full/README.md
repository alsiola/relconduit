# rel-stack/post-votes: all train rows, val-tuned LightGBM, time-safe text and tag features

The aim is first place on this one task (issue 16, for the programme in issue 4). It builds on
`../dfs-lightgbm` (issue 11), which got test nMAE .1238 ± .0001 on a 500k sample of train.

## Leaderboard (test nMAE ↓, raw `leaderboard.json`, 13 Nov 2026)

| | nMAE |
|---|---|
| Kapso (Leeroo Team), leader | .12214 |
| LEDGER | .12432 |
| KumoRelational | .12481 |
| InRTL | .12722 |
| GNN + LightGBM | .12737 |
| GNN | .12761 |
| RT-J | .12791 |
| LightGBM (RelBench baseline), Zero, Global Median | .13303 |
| issue 11 `dfs-lightgbm` (not submitted) | .12380 |

## What changed from issue 11

1. **All 2,454,038 train rows**, not a 500k sample. Stage A measures what the sample size buys.
2. **Text from `postHistory`, not `posts`.** `posts.Title`, `Body` and `Tags` are the 2023 dump
   versions, after every later edit: 6% of questions were retagged after they were asked. Every
   revision in `postHistory` is its own dated row (types 1/2/3 initial title, body, tags; 4/5/6
   their edits). `relconduit.stack_text` takes the latest revision of each kind strictly before
   the seed time and gives:
   - title length, words and a trailing `?`; body length, words, lines, and counts of backticks,
     links, images and LaTeX `$`; number of tags; revisions so far and days since the last one
     (`text__*`);
   - tag aggregates (`text__tag_*`): for each of the question's tags at the seed time, the number
     of questions carrying it then (each question's own latest tags before the seed), how many
     were asked in the last 91 days, and the upvotes those questions got before the seed (all
     time and in the last 91 days, with per-question means). Mean, max, min and sum over the
     question's tags.
3. **`posts.ContentLicense` dropped** (`base` feature set). It is the licence of the post's
   *last* edit, so CC BY-SA 4.0 on a 2012 seed means an edit after May 2018. Issue 11 used it,
   for the question and its answers. `base11` is the issue 11 set with it, run in stage A to show
   what it was worth on val. `postHistory` and `comments` licences belong to their own dated row
   and are kept.
4. **Tuning on val**, in stages, every choice by val nMAE (see `run.py`):
   - A: sample size;
   - B: feature set;
   - C: one change at a time from issue 11's config, then the helpful ones combined;
   - D: learning rate 0.03 vs 0.01, three seeds each.

   Test is scored once, for the final config, from its three saved models.

The issue 11 relational features (`relconduit.features`, the issue 11 `FeatureConfig` and
`codes_as_text`, without hop 2 and owner siblings, as chosen on val there) are unchanged.

Leakage: `tests/test_stack_text.py` checks the text and tag features on the full database
against the database cut strictly before each seed time, on a synthetic history (rows exactly
on seed times) and on a real rel-stack slice (seeds exactly at a retag and at an upvote). Both
pass, and a planted `<=` is caught on both.

## Run

```sh
uv run python experiments/rel-stack/post-votes/dfs-lightgbm-full/run.py tune --smoke   # ~1 min
uv run cloud-run --issue 16 --approved <url> --hours 4 --machine n2-custom-24-65536 \
  --checkpoint runs/pv-full --run pv-full --download runs/pv-full --download runs/pv-full-test \
  -- uv run python experiments/rel-stack/post-votes/dfs-lightgbm-full/run.py tune --out runs/pv-full --jobs 8
uv run python experiments/rel-stack/post-votes/dfs-lightgbm-full/run.py test --out runs/pv-full
```

## Results

RESULTS
