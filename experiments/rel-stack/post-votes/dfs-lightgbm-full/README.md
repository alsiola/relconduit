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
   - Round 2 (`tune2`, still val only), since round 1 projected short of the leader. F adds the
     graph paths (hop 2 and the owner's other posts, licences dropped) and 7- and 14-day windows
     (`short`), each over three seeds at the lr 0.03 config. G runs the best of them at lr 0.01
     over three seeds, and it becomes final only if it beats round 1's final on val.

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
uv run cloud-run --issue 16 --approved <url> --hours 2 --machine n2-custom-24-98304 \
  --checkpoint runs/pv-full --run pv-full --download runs/pv-full --download runs/pv-full-test \
  -- uv run python experiments/rel-stack/post-votes/dfs-lightgbm-full/run.py tune2 --out runs/pv-full --jobs 5
uv run python experiments/rel-stack/post-votes/dfs-lightgbm-full/run.py test --out runs/pv-full   # seconds
```

## Results

**Final config:**
- all 2,454,038 train rows;
- `base+text+tags+short`, 273 features;
- L1 loss, `num_leaves` 63, `min_child_samples` 20, `feature_fraction` 0.5, `lambda_l2` 1;
- learning rate 0.01, early stopping on val at 604 to 1,023 rounds.

| | nMAE ↓ (3 seeds) | per seed |
|---|---|---|
| val (optimistic: it made every choice) | .11195 ± .00013 | .11196 / .11208 / .11182 |
| **test** (scored once) | **.12302 ± .00003** | .12301 / .12305 / .12299 |
| issue 11 test, for comparison | .12380 ± .0001 | |

**Where it sits:** .0008 better than issue 11 on test. It would place **second**, .0013 ahead of
LEDGER (.12432), but it is **.0009 behind the leader, Kapso (.12214)**. Not a state-of-the-art
result, so no prediction table was written. The gain on val (-.0009) carried over to test
(-.0008) almost one for one.

### Val, by stage (seed 0 unless said; default config is issue 11's, lr 0.03)

| stage | sample | features | change | val |
|---|---|---|---|---|
| A | 500k | `base11` (issue 11's set) | | .11275 |
| A | 500k | `base` | | .11287 |
| A | 1M | `base` | | .11259 |
| A | all | `base` | | .11204 |
| A | all | `base11` | | .11204 |
| B | all | `base+text` | | .11206 |
| B | all | `base+tags` | | .11201 |
| B | all | `base+text+tags` | | .11194 |
| C | all | `base+text+tags` | `num_leaves` 127 / 255 | .11197 / .11212 |
| C | all | `base+text+tags` | `min_child_samples` 100 / 500 | .11204 / .11252 |
| C | all | `base+text+tags` | `feature_fraction` 0.5 | .11189 (seeds 1, 2: .11207, .11210) |
| C | all | `base+text+tags` | `lambda_l2` 10 | .11211 |
| C | all | `base+text+tags` | Huber (alpha 1) | .17936 |
| D | all | `base+text+tags` | `feature_fraction` 0.5, lr 0.01, 3 seeds | .11197 ± .00003 |
| F | all | `base+text+tags` | `feature_fraction` 0.5, 3 seeds | .11202 (mean) |
| F | all | `+graph` | same | .11213 (mean) |
| F | all | `+short` | same | .11200 (mean) |
| F | all | `+graph+short` | same | .11212 (mean) |
| G | all | `base+text+tags+short` | `feature_fraction` 0.5, lr 0.01, 3 seeds | **.11195 ± .00013** |

What it shows:
- **Train size is the one real lever:** -.0008 from 500k to all rows.
- Text and tag features add about .0001. The tuning changes, and the short windows (which won
  stage G by .00002), are within seed noise (std about .0001 at lr 0.03).
- The graph paths hurt slightly, as in issue 11.
- Huber at alpha 1 is squared loss on residuals under 1, so it pulls towards the mean, not the
  median, and is far worse on this mostly-zero target.
- The posts' dump-time `ContentLicense` is worth nothing (`base11` = `base` at all rows), so
  issue 11's result does not rest on it.

## Runtime, machine, cost

Both runs were on Google Cloud Spot in `europe-west2`, with features built on the machine (all
rows in about 2 minutes) and fits five to eight at a time:
- `tune`: `n2-custom-24-65536` (24 vCPU, 64 GB), 0.23 h of command time, $0.07;
- `tune2`: `n2-custom-24-98304` (96 GB, for five 429-column workers), 0.21 h, $0.08.

The test step ran on the Mac in 13 s from the three saved models. Total: **$0.15**, against
the $4 approved. Development and smoke tests on the Mac took under 10 minutes of CPU.
