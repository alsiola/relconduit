---
name: cloud-compute
description: "Rent a Google Cloud Spot machine for one long run (a multi-seed training run, a hyperparameter sweep, a dataset too big for the Mac) with `uv run cloud-run`: when it's worth it, getting the owner's go-ahead first, the $50 budget, running it as a task that outlives your turn, checkpoints, and reporting the cost. Use before any run you expect to take the Mac more than about an hour, and before running anything on Google Cloud."
---

# Cloud compute

`uv run cloud-run` (`src/relconduit/cloud.py`) rents one Spot machine on the shared `oddsworth` Google Cloud project, copies this checkout up, runs `uv sync --all-extras --frozen`, runs one command, copies the results back and deletes the machine. Spot machines are cheap because Google can take one back at short notice. The machine deletes itself after `--hours` even if the script dies, so nothing is left running.

## The budget: $50, total

This project has **$50** for all cloud spend (`budgetUsd` in `config/cloud.json`). The ledger, `costs/ledger.json` on main, records every machine's hours and price. `uv run costs` shows the entries and what's left.

Before renting, `cloud-run` prices the machine from Google's catalogue and refuses if `--hours` at that price, plus what's spent, plus the most that machines still running could cost, would pass $50. That's a backstop, not a plan: spend deliberately, and say in every request how much of the budget is left.

## Ask first, every time

Never start a run without the owner's go-ahead on the issue, for that run. Post one comment saying:

- what the run is for and what it will produce, in one line ("LightGBM on rel-hm user-churn, 5 seeds × 12 settings");
- why the Mac won't do: time a small slice here first and extrapolate, or say it won't fit in memory;
- the machine, the `--hours` cap, the most it can cost (`--dry-run` prints it), and the budget left after it;
- what it uploads and what it brings back.

Then end the turn asking for help (status `help`) so the owner can answer. `--approved` takes the link to the owner's comment saying yes, and the script refuses to run without one. A yes covers that run only. A rerun with different settings needs another yes, unless the owner said it covers reruns.

## When it's worth it

- Runs that would hold the Mac for more than about two hours and split across cores: many seeds or settings run side by side (`joblib`, `multiprocessing`, LightGBM's `n_jobs`), feature builds over large tables.
- The big RelBench datasets (`rel-amazon`, `rel-hm`, `rel-stack`, `rel-avito`, `rel-event`) when they don't fit comfortably in the Mac's memory. The 24-vCPU machine has 24 GB; give it more with `--machine n2-custom-24-65536` (about $0.31/h) if needed.
- GPU training, once a GPU is allowed (below).

Not worth it:

- Runs under an hour. Setting up takes 10–15 minutes (machine, `uv sync`, dataset download).
- Single-threaded work: a bigger machine won't help unless the script splits its work. Check first, and add the split if it's easy.
- Anything needing secrets. The machine gets no keys and no `.env`.

## Machines

Spot prices in London, `europe-west2`, on 12 Nov 2026 (`--dry-run` prints the current figure):

| Machine | What | Per hour |
|---|---|---|
| `n2-custom-24-24576` (default) | 24 vCPUs, 24 GB | about $0.26 |
| `n2-custom-24-65536` | 24 vCPUs, 64 GB | about $0.31 |
| `--gpu nvidia-tesla-t4` (`n1-standard-8`) | 8 vCPUs, 30 GB, T4 16 GB | about $0.33 |
| `--gpu nvidia-l4` (`g2-standard-8`) | 8 vCPUs, 32 GB, L4 24 GB | about $0.60 |

The trial project allows **24 vCPUs across all regions**, shared with another project, and **no GPUs** (`GPUS_ALL_REGIONS = 0`). `cloud-run` checks quota before renting. A GPU needs the billing account upgraded (the credit is kept) and a quota increase, and both are the owner's to ask for. If a plan needs a GPU, say which and for how many hours, and ask. GPU machines use Google's Deep Learning image (CUDA drivers included), and the locked torch wheels are CUDA builds on Linux, so `torch.cuda.is_available()` works there. Pick the device as `cuda`, then `mps`, then `cpu`.

## Running it

A run outlives your turn, so start it with `claude-code.task.start` and park on it with `claude-code.task.wait`. A shell `&` or `nohup` dies with the turn. Give `timeout_minutes` comfortably over `--hours` (the default is 240):

```json
{"name": "cloud-hm-sweep", "timeout_minutes": 400,
 "command": "uv run cloud-run --issue 12 --approved https://github.com/alsiola/relconduit/issues/12#issuecomment-123 --hours 6 --checkpoint runs/hm-sweep --run hm-sweep --download results/rel-hm -- uv run python experiments/rel-hm/user-churn/lightgbm/sweep.py --out runs/hm-sweep"}
```

- Run `--dry-run` first. It does the real quota, price and budget checks and prints every step without renting anything.
- The command runs from the machine's copy of this checkout, uncommitted files included, after `uv sync --all-extras --frozen`. Commit or stage `uv.lock` changes first.
- RelBench datasets download from the Hugging Face Hub on the machine. To skip the download, `--upload ~/.cache/huggingface/hub/<dataset folder>`: a path under home lands at the same place under the machine's home.
- `--upload <path>` (repeatable): inside the checkout or under home. `--download <path>` (repeatable): relative to the checkout, copied back whether the command passed or failed.
- Write bulky outputs under `runs/` (ignored by git) and bring back only what you'll commit or read.
- The machine's full log lands in `.local/cloud/<machine>/run.log`, and the task log has the live copy.

### Confirm it runs before you park

"started" only means the command was launched. It doesn't mean any job works. Before ending the turn on a run:

- Smoke-test new code on the target first: the first part of a run, or a `--dry-run`-sized slice, on the machine
  image. The Mac is not a Debian VM: system libraries differ (LightGBM needs `libgomp1`, for one).
- Make each job say when it starts, and when it fails, in the log that `cloud-run` echoes. A driver that only
  prints at the end hides failures.
- Once it starts, ssh in (`gcloud compute ssh <machine> --zone=... --project=oddsworth`, signed in with
  `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=~/.config/oddsworth/gcp-mac-runner.json`) and read each job's own log.
  Check every job has done real work (features built, first fit scored, no `Traceback`), and look again partway
  through a long run. In issue 6, three of eight jobs died at import in seconds and went unnoticed for 90 paid minutes.

## Checkpoints

Google can take a Spot machine back at any time, so build every run of more than an hour to resume, and checkpoint it. Being taken back five hours into a six-hour run should cost minutes, not hours.

- Write the run as a list of parts (a seed, a fold, a block of settings). Write each finished part to its own file in one output folder as soon as it's done, and skip parts already there. Write to a temporary name and rename when complete.
- For a single long training run, save model and optimiser state every few minutes, and load it on start if it's there.
- Pass the folder as `--checkpoint <folder>` (relative to the checkout, e.g. `runs/hm-sweep`) with `--run <name>`. The machine restores it from `gs://oddsworth-lab-runs/relconduit-<issue>-<name>/`, saves it every 10 minutes and at the end, and once more in the 30 seconds Google gives before taking the machine back. A failed save prints a `[checkpoint]` line. If you see one, say so in the report.
- If the machine is taken back, `cloud-run` starts a new one and resumes, up to `--retries` (2) times, within the original `--hours`. The checkpoint folder is copied back at the end whatever happens. The bucket deletes everything after 30 days.
- Running again with the same `--run` name resumes the same run. That's a new run, so ask first.

## Reporting

When the run ends, say on the issue how long it ran, what it cost (the ledger entry `cloud-run` printed), what's left of the $50 (`uv run costs`), and where the results are, along with the findings. If it failed, quote the end of the log and say whether a retry is worth it. A retry is another run, so ask again.

`cloud-run` signs in as the Mac's service account (key at `~/.config/oddsworth/gcp-mac-runner.json`), so a person's lapsed `gcloud` login doesn't stop a run. If the ledger push to main fails, the script prints the `uv run costs add ...` line that records it by hand. Run it: an unrecorded cost breaks the budget check.
