# rel-trial/site-success: time-safe relational features + LightGBM (negative result)

Predict a site's success rate over its trials in the next year. Metric: nMAE ↓. Part of issue 6.

## What it does

1. Features from `relconduit.features.build_features` on `task.get_db()`:
   - aggregates of the site's `facilities_studies` links, with each study's columns joined on, over 365 and 1,460
     days and all time;
   - hop 2 through those links to every study the site has run, with their outcomes, analyses and adverse events.

   Every aggregate reads only rows strictly before the seed time. That gives 492 features.
2. Feature set chosen on val: all features (.8916) beat top 300 (.8924), top 100 (.8978) and without hop 2 (.9049).
3. Grid (`relconduit.entity_lgbm.GRID`), with train on train only (151,407 rows), early stopping on val, and seeds
   0, 1 and 2. Test scored once.

Chosen config: num_leaves 63, min_child_samples 100, feature_fraction 0.3, L1 loss, all 492 features.

## Run

```sh
uv run python experiments/rel-trial/site-success/dfs-lightgbm/run.py --test
```

Ran in the cloud (`n2-custom-24-65536`, 7 threads, alongside other jobs) in 23.8 min.

## Results (3 seeds, mean ± std)

| | nMAE ↓ | per seed |
|---|---|---|
| val (optimistic) | .901 ± .003 | .902 / .897 / .903 |
| **test** | **.931 ± .004** | .932 / .926 / .933 |

Published (test):
- official LightGBM .880 (our reproduction .890 ± .002);
- GNN .781;
- RT-J .765;
- leader Kapso .659.

**Negative result.** This is worse than the official LightGBM by .05, and it was already worse on val (.901 against
the baseline's .876), so this is not a val/test gap. It beats only the trivial baselines (entity mean .941, global
median .971).

## Why

The site's own columns never reach the model. `facilities` has `name`, `city`, `state`, `zip` and `country`, all
text with more than `MAX_CATEGORIES = 100` distinct values (country has 210, city 28,165, zip 48,347). So
`features.py` drops all five, and no `facilities__` feature exists. The upstream LightGBM uses only those columns
(GloVe-embedded) and does better. Where a site is evidently matters more than the aggregates of its past studies.

A follow-up could add a time-safe encoding of high-cardinality entity columns, chosen on val. Options include
frequency counts, out-of-fold target encoding on train, or the same text embeddings upstream uses, with country
kept as a categorical of about 200 levels.
