# rel-trial/site-sponsor-run: where the MAP goes, and sharper locality for the ranker

Issue 28. It starts from `../cand-lgbm-ranker` (issue 12; test MAP@10 .2834 ± .0008, against .3333
for Kapso, the leader). First it measures where that ranker loses MAP, then it builds against the
largest measured losses. Every choice is made on val with the official `task.evaluate`, and test
is scored once, for the final config. The work is our own: no code or write-up from the leader
was read.

## Phase 1: the statistics (val only)

Script: `phase1.py`. Mac, 3 min. It rebuilds the baseline's val pairs and scores them with the
issue-12 models.

**Ceilings.** The current ranker gets .267. Perfect ordering of its own top 50 would give .524,
and perfect ordering of all of today's candidates (91 per seed, recall .553) would give .558.
Most of the reachable headroom is in **ranking**, not recall. Of the positives, 13.6% belong to a
sponsor with no study at all before the seed time. No time-safe source can reach them, which caps
recall at about .86.

**Where the positives are** (80,103 true val pairs):

| segment | share | in candidates | in top 10 |
|---|---|---|---|
| warm facility, pair seen before | 20.9% | .95 | .79 |
| warm facility, sponsor new to it | 33.4% | .27 | .08 |
| cold facility (no trial before T) | 45.8% | .39 | .22 |

- Cold positives are reached mostly through the city: 16.6k, at recall .72. Next come the
  country only (7.0k, recall .17) and the state (3.0k, .24).
- Warm, sponsor-new positives are reached mostly through a co-sponsor of a past sponsor: 9.9k, at
  recall .47 but only .14 in the top 10.
- Repeat sponsors give 63% of the MAP from 19% of the slots.
- The per-facility number of true sponsors is only weakly predictable (Spearman .23–.28).
- Facility names are unique (453,232 names for 453,233 facilities), so the `name` walk only
  replays a site's own history.

## Phase 2: experiments (val MAP@10, 3 seeds, mean ± std)

Every candidate set passed the real-database pair leakage check (80 val and train seeds, full
database against `truncate_before`) with **0 mismatches** before any fit. The new sources are in
`src/relconduit/trial_edges.py`. `tests/test_trial_edges.py` checks that each one has 0
mismatches and that a planted `<=` in each is caught.

### Run 1: the train-size curve (baseline candidates and config)

| train seed rows | 25k | 50k | 100k | 200k | all 669,310 |
|---|---|---|---|---|---|
| val MAP@10 | .2652 ± .0008 | .2645 ± .0013 | .2656 ± .0015 | .2656 ± .0004 | .2655 ± .0008 |

**Flat.** The ranker is not data-limited, so everything after this stays at 25k rows.

### Run 1: candidate ablation (25k rows, reference config)

| set | recall | per seed | val MAP@10 | Δ |
|---|---|---|---|---|
| base | .553 | 91 | .2652 ± .0008 | |
| + co-sponsor walk | .557 | 102 | .2644 ± .0016 | −.0008 |
| + condition walk | .559 | 107 | .2642 ± .0004 | −.0010 |
| **+ zip and state walks (`geo`)** | .569 | 101 | **.2766 ± .0005** | **+.0114** |
| + recent city walk (365 d) | .565 | 105 | .2648 ± .0012 | −.0004 |
| all + 90-day popularity | .586 | 140 | .2771 ± .0002 | +.0119 |

The co-sponsor, condition and recent-city walks reach the positives they aimed at but add nothing
to MAP. These are negative results. Locality helps, mostly through ranking: `geo` gains +.011 MAP
for +.016 recall.

### Run 2: sharper locality on `geo`

| set | recall | per seed | val MAP@10 |
|---|---|---|---|
| `geo` | .569 | 101 | .2765 ± .0005 |
| `geo_cc`: city walk keyed on city within country | .569 | 101 | .2762 ± .0010 |
| **`geo_zip3`: + US zip3 (metro region) within country** | .572 | 104 | **.2780 ± .0008** |
| `geo_loc`: both | .572 | 104 | .2780 ± .0010 |
| `geo_loc_wide`: both, wider place walks | .602 | 152 | .2773 ± .0009 |

City-name homonyms are real (3,743 names span several states or countries), but keying on the
country changes nothing. Wider walks add .03 recall but no MAP.

### Run 2: ranker grid on `geo_zip3`

| config | val MAP@10 |
|---|---|
| **lambdarank, 63 leaves, min_child 20, feature fraction .7 (reference)** | **.2780 ± .0008** |
| 127 leaves, min_child 50 | .2775 ± .0005 |
| 255 leaves, min_child 100 | .2753 ± .0006 |
| learning rate .02 | .2780 ± .0009 |
| feature fraction .5 | .2778 ± .0009 |
| `rank_xendcg` | .2744 ± .0012 |
| `binary` | .2740 ± .0006 |

Nothing beats the reference, so the other objectives (listwise `rank_xendcg` among them) are a
negative result. **Chosen on val:** `geo_zip3` with the reference config.

## Final (`--plan final`, test scored once)

`geo_zip3` candidates (val recall .572, 104 per seed), 131 features, the reference config, and
25,000 train seed rows. Refitted with seeds 0–2. The pair leakage check gave 0 mismatches on 80
seeds. Test was scored through `write_prediction_table` + `task.evaluate`, and `evaluate_task`
on the saved tables gave the same numbers.

| split | MAP@10 (mean ± std, 3 seeds) | per seed |
|---|---|---|
| val | .2780 ± .0008 | .2781 / .2772 / .2787 |
| test | **.3037 ± .0009** | .3047 / .3031 / .3032 |

The bootstrap 95% CI over the 27,428 test rows (seed 0) is [.2990, .3085].

### Against the leaderboard (test MAP@10)

| | test |
|---|---|
| Kapso (leader) | .3333 |
| **cand-lgbm-v2 (this)** | **.3037** |
| cand-lgbm-ranker (issue 12, ours) | .2834 |
| Past visit | .1734 |
| GNN | .1059 |
| LightGBM (upstream) | .0904 |

- **+.0203 over our issue-12 ranker on test**, about 20× the seed std. On val the gain was .0128
  under the same protocol, so it held up on test and grew.
- **Still .030 below Kapso**, well outside the bootstrap CI (its top is .3085). This does not lead
  the board. It closes about 40% of the .050 gap and stays second.
- Nearly all of the gain comes from locality: the zip, state and zip3 walks used as candidate
  sources and as rank features. More train rows, the co-sponsor, condition and recent-city walks,
  wider walks, bigger trees and other objectives all gave nothing on val.
- **What is left, by phase 1's measurements:** ranking is still the binding constraint, since
  perfect ordering of today's candidates gives about .56 on val. The warm-facility, sponsor-new
  positives (33% of positives, top-10 rate .08) remain the largest group the ranker cannot
  separate.

**The test number was computed twice.** The first final machine was pre-empted after it had
scored all three test seeds but before it finished the bootstrap step. `cloud-run` restarted
the same command from the checkpoint on a new machine, which refitted and rescored with the same
config and seeds and got the same per-seed test MAP (.3031 and .3032 for seeds 1 and 2 on both).
Nothing was chosen between the two runs.

## Run

```sh
uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-v2/phase1.py               # phase 1 statistics, Mac
uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-v2/run.py --plan run2 --smoke 150
uv run python experiments/rel-trial/site-sponsor-run/cand-lgbm-v2/run.py --plan final --test
```

Only `final` accepts `--test`. The cloud runs used `cloud28.py`, which runs a smoke of each plan
before the real jobs, on a Google Cloud `n2-custom-24-131072` Spot machine (europe-west2-b):

- Run 1: `--plans curve:14 ablation:10`, 1.2 h, $0.47.
- Run 2: `--plans run2:22`, 63 min job, $0.57.
- Final: `--plans final:22 --extra --test`, 62 min job (peak 62.5 GB). The first machine was pre-empted after 0.78 h ($0.30); the restart took 1.07 h ($0.41). Issue total: $1.75.

Each final seed's val and test prediction tables and per-candidate scores are in the checkpoint
`gs://oddsworth-lab-runs/relconduit-28-i28-final/`, not in git.
