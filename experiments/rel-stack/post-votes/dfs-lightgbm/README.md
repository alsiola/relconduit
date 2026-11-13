# rel-stack/post-votes: time-safe relational features + LightGBM

Predict how many upvotes (`VoteTypeId` 2) a question gets over the next 91 days. Entity table `posts` (415,913 rows,
`CreationDate`). Its activity is in `votes` (1.67M), `postHistory` (1.49M), `comments` (795k), `badges` (591k),
`users` (334k) and `postLinks` (104k), all time-stamped. Train / val / test: 2,454,038 / 156,230 / 160,903 rows.
Metric: nMAE ↓. Part of issue 8.

## What it does

1. Loads the database with `relconduit.db.load_task_db`. It leaves out the free text (post `Body`, `Title`, `Tags`,
   comment and history `Text`, user `AboutMe`, names, URLs) and cuts rows at the test timestamp.
2. `codes_as_text` (in `run.py`) turns the type codes (`VoteTypeId`, `PostTypeId`, `LinkTypeId`,
   `PostHistoryTypeId`, badge `Class`) into text, row by row, so each code gets its own count per window rather than
   a meaningless mean.
3. Features from `relconduit.features.build_features`:
   - aggregates of the question's votes, comments, answers (posts by `ParentId`), links and edit history over 30, 91
     and 365 days and all time;
   - siblings through `OwnerUserId`: the owner's record pooled over all their posts;
   - hop 2.

   That gives 345 features. Every aggregate reads only rows strictly before the seed time.
4. Train on a fixed uniform sample of 500,000 of the 2.5M train rows. Val and test are scored in full.
5. Feature set chosen on val at a reference config: `no_hop2` (.1128), which drops both the hop-2 and the
   owner-sibling (`>>`) columns. All five candidates are within .0002 of it: all features, top 100, top 300, and
   `no_hop2_all` (siblings dropped, hop 2 kept). That leaves 177 features.
6. Grid (`relconduit.entity_lgbm.GRID`), early stopping on val, seeds 0, 1 and 2. Test scored once.

Chosen config: num_leaves 63, min_child_samples 20, feature_fraction 0.8, L1 loss, `no_hop2`.

By gain over all features, the top ones are the question's past upvotes (all time, 365 and 91 days), its total
votes, `days_since_last` for votes, edits, answers and linked posts, the last vote on any of the owner's posts, and
the post's age. The owner-sibling feature isn't needed: dropping it costs nothing on val.

## Run

```sh
uv run python experiments/rel-stack/post-votes/dfs-lightgbm/run.py --test
```

Ran on Google Cloud (`n2-custom-24-196608` Spot, 5 threads, beside the other jobs of the issue 8 run) in 26.8 min,
with feature building under 30 s. Peak RSS 5.4 GB.

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (optimistic) | .1128 ± .0000 | .1128 / .1128 / .1129 |
| **test** | **.1238 ± .0001** | .1238 / .1237 / .1239 |

Published (test):
- official LightGBM .133 (our reproduction .133 ± .000). That equals the Global Median and Zero baselines;
- GNN .128;
- RT-J .128;
- LEDGER .124;
- KumoRelational .125;
- leader Kapso .122.

**Where it sits:** .009 better than the official LightGBM, and .004 better than the GNN and RT-J. It would place
second on the leaderboard, .0005 ahead of LEDGER and .0017 behind the leader. It is not a state-of-the-art result.
