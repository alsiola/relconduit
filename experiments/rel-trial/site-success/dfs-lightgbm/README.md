# rel-trial/site-success: time-safe relational features + entity encodings + LightGBM (negative result)

Predict a site's success rate over its trials in the next year. Metric: nMAE ↓. Issue 6 built it; issue 9 added the
site's own columns.

## What it does

1. Relational features from `relconduit.features.build_features` on `task.get_db()`:
   - aggregates of the site's `facilities_studies` links, with each study's columns joined on, over 365 and 1,460
     days and all time;
   - hop 2 through those links to every study the site has run, with their outcomes, analyses and adverse events.

   Every aggregate reads only rows strictly before the seed time. Since issue 9, the text-column vocab is built only
   from rows dated strictly before the val timestamp.
2. The site's own columns (`name`, `city`, `state`, `zip`, `country`), from `relconduit.entity_encoding` (issue 9).
   `features.py` drops all of them as too high-cardinality. The encodings:
   - **counts** (`__count`, `__share`): sites with the same city, state, zip or country, counted only over sites first
     linked to a study strictly before the seed time;
   - **target encoding** (`__te`, `__te_n`): by city, state, zip, country, name template (digits as `#`) and the site
     itself, over **train labels only**, and only labels whose 365-day window had closed by the seed time
     (`s + 365d <= t`). That is out-of-fold in time: a row never sees its own label or a later one. K-fold target
     encoding was rejected, because a train row would see labels from its own future;
   - **country** as a categorical;
   - **GloVe** word-average embeddings of all five columns (the model upstream's LightGBM uses), 32 PCA dims each.

   That gives 672 features in total.
3. Feature set chosen on val in stage 1 (one config, 3 seeds):

   | features | val nMAE |
   |---|---|
   | all | .8887 |
   | without hop 2 | .8968 |
   | without any encoding (issue 6's features) | .8916 |
   | without target encoding | .8855 |
   | without counts | .8913 |
   | without target encoding or counts | .8890 |
   | **without embeddings** | **.8793** |
   | top 100 / top 300 by gain | .8828 / .8844 |

4. Grid (`relconduit.entity_lgbm.GRID`) on the chosen set, with train on train only (151,407 rows), early stopping
   on val, and seeds 0, 1 and 2. Test scored once.

Chosen config: num_leaves 63, min_child_samples 20, feature_fraction 0.8, L1 loss, 512 features (all but the 160
embedding dims).

## Run

```sh
uv run python experiments/rel-trial/site-success/dfs-lightgbm/run.py --test
```

Issue 9 ran it in the cloud on `n2-custom-24-65536` (Spot, 18 threads, alongside three rel-f1 val checks), at commit
eef75b1. It took 17.7 min, and the whole machine cost $0.11. The cloud copy had no `.git`, so writing the results rows
failed after test was scored. The rows were written on the Mac from the run's printed summary, and `cloud-run` now
passes the commit to the machine as `RELCONDUIT_COMMIT`.

## Results (3 seeds, mean ± std)

| | issue 6 (no encodings) | **issue 9 (with encodings)** | per seed (issue 9) |
|---|---|---|---|
| val (optimistic) | .901 ± .003 | .883 ± .003 | .885 / .879 / .884 |
| **test** | .931 ± .004 | **.962 ± .002** | .960 / .964 / .963 |

Published (test):
- official LightGBM .880 (our reproduction .890 ± .002);
- GNN .781;
- RT-J .765;
- leader Kapso .659.

**Negative result, and worse than issue 6.** Val improved by .018 and now nearly matches the reproduced LightGBM's val
(.876). Test got worse by .031 and is .08 behind the official LightGBM. It doesn't even beat the trivial
entity-mean baseline (.941).

## Why (hypothesis, not tested on test)

Most of the val gain came from the time-dependent encodings, chosen over the static embeddings: dropping the embeddings
helped on val, and dropping target encoding barely mattered. Those features aren't distributed the same way at test
time:
- counts of visible sites grow with the seed time;
- the target encodings at test can read a whole further year of train labels (every train label's window has closed
  by then), which the model never saw during training or on val.

A tree split on an absolute count or on a TE built from a shorter label history doesn't carry over. Val can't show
this, because val sits at the same distance from train as the windows the model was fitted on. The static encodings
(embeddings, country) don't move with time: an encoder-only probe on the Mac with the embeddings got .890 val.

Diagnosing this needs a check that doesn't touch test, for example a pseudo-test that holds out the last train
timestamp, or embeddings and country only with counts and TE turned off. It isn't done here: issue 9 scored test once.
