# rel-stack/post-post-related: time-safe dense retrieval on top of the issue-12 ranker

Issue 29. Task: for each `rel-stack` seed post, rank the 100 posts most likely to be linked to it in the next
window. Metric: MAP@100. The starting point is issue 12's `cand-lgbm-ranker`: test .1993 ± .0047 against the
leader Kapso's .2178. No Kapso code or write-up was read; everything below comes from our own measurements.

**Result:** the final config adds frozen `all-MiniLM-L6-v2` kNN candidates (built only from posts dated before the
seed time) and the cosine feature. It scores **test .2484 ± .0019** (3 seeds, official evaluation), above Kapso's
.2178 and issue 12's .1993.

## Phase 1: where the MAP goes (val only)

`../diagnose/phase1.py` rebuilds issue 12's val pairs and rescores them with its three saved models, reproducing
val .1778 / .1683 / .1703. The full tables are in the
[issue comment](https://github.com/alsiola/relconduit/issues/29#issuecomment-6877806062). The numbers that
shaped Phase 2:

- **Recall is the ceiling.** Candidate recall is .575. With perfect ranking of today's candidates MAP would be
  .574, and 38.5% of val rows have no true target among their candidates at all.
- **15.6% of true targets had never been linked before the seed time**, and every issue-12 content source (tags,
  title words) only ends at already-linked posts. These are 37% of the misses, and no source could reach them.
- **27% of targets share no tag with the seed** (recall .28 there). TF-IDF kNN over title and body found 12% of
  today's misses at k = 100, a weak lower bound for a sentence encoder.
- **The MAP is at rank 1**: with about 1.3 targets per row, rank-1 hits carry .111 of the .170 total.
- **Noise.** The ranker-seed std is .005, the bootstrap std over the 226 val rows is .021, and a paired difference
  between configs under about .007 is noise. Test has 258 rows, so Kapso's .019 lead is within one test-sample std.
- Every seed and every target is a question. The dump's titles and tags drift by later edits, but I found no
  label-correlated drift. The dense source embeds only each question's *initial* title and body from
  `postHistory`, so it does not read the dump text.

## What it does

`src/relconduit/post_embed.py` (tested in `tests/test_post_embed.py`) adds a `DenseSource` to `relconduit.link`'s
pair builder:

- `embed.py` embeds the initial title and body of all 163,748 questions with a **frozen** open encoder,
  `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`. Each vector is dated by the latest
  `postHistory` row it was made from.
- For a seed `(post, T)`, the index searches only vectors dated **strictly before T**, with the seed's own vector
  (also dated before T) as the query. The `n` nearest neighbours become candidates (`cand__dense_*`); every
  candidate pair gets the cosine and its neighbour rank as features.
- Tag and title-word overlap features (`*_overlap__shared`, `__jaccard`) count the tokens both posts carry before T.

`run.py` builds one superset of pairs and features and runs every variant as a view of it.

- **Leakage check.** Before any fit, it compares pairs built on the full database with pairs built on
  `truncate_before(db, T)`, for 120 val, train and extra seeds over 44 seed times, with the dense index and overlaps
  in: **0 mismatches**. The tests show that the same check catches a planted `<=` in the dense search and in the
  overlap counts.
- **Ranker.** Issue 12's ranker is unchanged: LightGBM lambdarank (`num_leaves` 15, `min_child_samples` 100,
  `feature_fraction` 0.7), early stopping on val, 5 seeds per variant. Variants are chosen on the 5-seed val mean
  with the official `task.evaluate`.
- **Extra rows.** The "extra" train rows come from 40 seed times halfway between the train table's, labelled by
  RelBench's own `task.make_table`, with every label window ending before the val timestamp.

## Phase 2: ablations (val MAP@100; test not touched)

`summary.json` from run 1; each row includes the changes above it except where noted. Δ is the paired bootstrap of
the 5-seed mean AP against `ref` over the 226 val rows (2,000 resamples).

| variant | cand. recall | cands/row | seeds 0–2 | 5 seeds | 5-seed bag | Δ vs ref [95% CI] |
|---|---|---|---|---|---|---|
| ref (issue 12) | .574 | 624 | .1694 ± .0025 | .1708 | .1649 | |
| + tag and title overlap | .574 | 624 | .1867 ± .0012 | .1866 | .1811 | +.016 [−.001, .035] |
| + MiniLM cosine (feature only) | .574 | 624 | .2151 ± .0039 | .2135 | .2114 | +.043 [.014, .073] |
| **+ MiniLM kNN, n = 100 (final)** | **.682** | **707** | **.2309 ± .0011** | **.2328** | **.2350** | **+.062 [.027, .099]** |
| + MiniLM kNN, n = 300 | .730 | 887 | .2314 ± .0018 | .2319 | .2323 | +.061 [.024, .100] |
| bge-small kNN, n = 300 (instead of MiniLM) | .720 | 885 | .2146 ± .0010 | .2135 | .2136 | +.043 [.012, .073] |
| both encoders, n = 300 each | .753 | 1049 | .2359 ± .0063 | .2335 | .2309 | +.063 [.026, .100] |
| both + extra train rows (9,820 seeds) | .753 | 1049 | .2341 ± .0042 | .2337 | .2311 | +.063 [.025, .102] |
| both + extra, `learning_rate` .02 | .753 | 1049 | .2285 ± .0048 | .2238 | .2214 | +.053 [.016, .091] |
| both + extra, `min_child_samples` 300 | .753 | 1049 | .2348 ± .0015 | .2377 | .2347 | +.067 [.030, .107] |
| both + extra, `num_leaves` 31 | .753 | 1049 | .2392 ± .0007 | .2389 | .2379 | +.068 [.030, .109] |

Paired bootstraps between the top configs (`compare.py`, 5-seed mean AP):

| comparison | Δ | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| best on val (both + extra + `num_leaves` 31) vs MiniLM n = 100 | +.0061 | [−.0072, .0209] | .19 |
| best on val vs both, n = 300 | +.0055 | [−.0049, .0161] | .16 |
| both + extra vs both (the extra rows) | +.0002 | [−.0063, .0063] | .49 |
| both, n = 300 vs MiniLM n = 100 | +.0007 | [−.0079, .0094] | .46 |
| MiniLM n = 300 vs n = 100 | −.0008 | [−.0087, .0072] | .60 |

What this says, hypothesis by hypothesis:

- **H1, H2 (reach unlinked and tag-disjoint targets): confirmed.** Dense kNN lifts recall from .574 to .682 for 83
  more candidates per row, and MAP by .062. MiniLM beats bge-small by about .017 at the same n.
- **H3 (a sharper rank 1): confirmed.** The cosine as a feature alone is worth +.043; the overlap counts +.016.
- **More recall past .68 doesn't turn into MAP (negative).** 300 neighbours or two encoders reach recall .730–.753,
  but neither adds MAP beyond noise. The extra candidates are harder ones that the ranker doesn't place near the top.
- **H4 (variance): negative.** Extra train rows from 40 more seed times add +.0002. A 5-seed score bag tracks the
  5-seed mean and doesn't tighten the result. The three ranker configs move val by at most .006, within noise.
- **H5 (fine-tuning the encoder): not run.** The frozen encoder already supplies the recall, and the val noise floor
  (about .007) couldn't have resolved a smaller ranking gain on top.

**Final config: `dense_minilm_100`**, the simplest setup that nothing else beats clearly, as decided before the
numbers were in. The best-on-val config is +.006 inside the noise, and it is the max of 11 configs picked on the
same 226 rows, so it carries a winner's-curse bias.

## Results (MAP@100, 3 seeds, mean ± std)

| split | MAP | per seed |
|---|---|---|
| val | .2309 ± .0011 | .2319, .2297, .2310 |
| test | **.2484 ± .0019** | .2504, .2467, .2480 |

Val is optimistic: it chose the variant and the stopping round.

## Against the leaderboard (test MAP@100)

| | test |
|---|---|
| Kapso (leader) | .2178 |
| **dense-ranker (this)** | **.2484 ± .0019** |
| cand-lgbm-ranker (issue 12) | .1993 ± .0047 |
| ID-GNN | .1075 |
| LightGBM (upstream) | .0168 |

On the official `relbench.submit` evaluation, the 3-seed test mean is **.031 above Kapso's .2178** and .049 above
issue 12's .1993. Every seed beats Kapso on its own. The seed std (.0019) is small, but the 258 test rows carry about
.02 of row-sampling noise (the phase 1 bootstrap on val's 226 rows gave .021). So the lead over Kapso is about 1.5
sampling stds: a clear lead on the published number, not a statistically settled one. Without Kapso's predictions,
there can be no paired test. Test was scored once, for the config chosen on val beforehand. Nothing has been
submitted to the leaderboard.

## Run

Everything ran on Google Cloud Spot `n2-custom-24-131072` (24 vCPUs, 128 GB, europe-west2).

- **Run 1**, embeddings and all val ablations: 2.24 h (embedding both encoders in parallel 46 min; pairs and
  features 6 min; 55 fits 87 min), $0.88.
- **Run 2**, test once with the saved seed 0–2 models: 6 min of work (rebuilding the pairs 2.5 min, scoring
  0.2 min, plus setup), about 0.12 h billed, $0.05. Val reproduced run 1's .2309 exactly before test was scored.

The first val/test pair in `results/rel-stack/post-post-related.jsonl` (2026-11-14, commit 2194f20) records
`runtime_minutes` 2.7, which is run 2's process alone. `results/` is append-only, so a superseding pair dated
2026-11-15 has the same values with 137.1 min (run 1's 134.4 plus run 2's 2.7; issue 31). Read the later pair.
`--train-minutes` now adds the training run's time, so a scoring-only `--final` run records the whole cost.

```sh
# Mac smoke test (150 seeds, 1 ranker seed, no results; 2.6 min): embed a small slice of each encoder with
# embed.py --limit, put the two .npz files under the encoder names in a folder, and point --emb at it
uv run python experiments/rel-stack/post-post-related/dense-ranker/run.py --smoke 150 --emb <folder>
# run 1: embeddings + val ablations
uv run cloud-run --issue 29 --approved <link> --hours 5 --machine n2-custom-24-131072 --checkpoint runs/p29 --run p29-ablate \
  --download runs/p29/ablate/summary.json -- bash experiments/rel-stack/post-post-related/dense-ranker/cloud29.sh
uv run python experiments/rel-stack/post-post-related/dense-ranker/compare.py
# run 2: test, once, with run 1's models
uv run cloud-run --issue 29 --approved <link> --hours 1.5 --machine n2-custom-24-131072 --checkpoint runs/p29 --run p29-ablate \
  --download runs/p29/ablate/final.json --download results/rel-stack/post-post-related.jsonl \
  -- uv run python -W ignore experiments/rel-stack/post-post-related/dense-ranker/run.py --final dense_minilm_100 --test --train-minutes 134.4
```

The checkpoint `gs://oddsworth-lab-runs/relconduit-29-p29-ablate/` (kept 30 days) holds the embeddings, every
model, the per-candidate val scores of every variant (`scores/<variant>_val_scores.parquet`), each seed's top-100
val tables, the final config's per-candidate test scores, and each seed's test prediction table (`pred/seed<k>/`).
