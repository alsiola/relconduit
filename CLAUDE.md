# relconduit

This repo runs experiments on RelBench, the benchmark for deep learning on relational databases
(https://relbench.stanford.edu/, paper: https://arxiv.org/abs/2407.20060). Work arrives as GitHub issues, is picked up
by an Open Conduit agent (Claude Code), and lands as a pull request that a reviewer bot reviews. There is no CI: run the
checks yourself before you open or update a PR.

## Environment

- Python is managed by `uv`. Use `uv run ...` and `uv add ...`; never `pip install`. Don't add a `.python-version`
  file: this machine's pyenv shims intercept it and break `uv`.
- Run `uv sync --all-extras` after pulling dependency changes.
- Compute is an Apple Silicon Mac, plus Google Cloud Spot machines for long runs within a **$50 total budget**: 24-vCPU
  CPU machines now, and a single T4 or L4 GPU only if the owner enables one. Read the `cloud-compute` skill before any
  run you expect to take the Mac more than about an hour, and never rent a machine without the owner's approval on the
  issue. Pick the torch device as `cuda`, then `mps`, then `cpu`. Size experiments to finish in hours, not days, and say
  in the PR how long a run took and where it ran.
- Anything that should keep running after your turn (a long training run, a cloud run) starts with
  `claude-code.task.start` and ends the turn with `claude-code.task.wait`. Background shell jobs die with the turn.
- RelBench 3.x loads datasets from the Hugging Face Hub (`stanford-star/relbench-v1/<dataset>`) into the HF cache
  (`~/.cache/huggingface`). Never commit datasets, caches, checkpoints or large artefacts
  (`.gitignore` covers the usual ones). Anything over about 1 MB doesn't belong in git.

## Layout

- `src/relconduit/`: shared, tested code used by more than one experiment.
- `experiments/<dataset>/<task>/<approach>/`: one experiment, for example `experiments/rel-f1/driver-position/lightgbm/`.
  Each has a `README.md` saying what it tries, how to run it, and what it got. Experiment-specific code stays inside it.
- `results/<dataset>/<task>.jsonl`: one JSON object per run, appended, never rewritten. Fields: `date`, `approach`
  (its experiment path), `commit`, `metric`, `split` (`val` or `test`), `value`, `seeds`, `runtime_minutes`, `notes`.
- `tests/`: pytest. Shared code in `src/` needs tests.
- `config/cloud.json`, `src/relconduit/cloud.py`, `src/relconduit/costs.py`: the cloud runner and the cost ledger
  (`costs/ledger.json`, written to main by the runner, never edited by hand). `uv run costs` shows the budget.

## Benchmark integrity

- RelBench is 3.x: `ds = relbench.load_dataset("stanford-star/relbench-v1/rel-f1")`, `ds.get_task_names()`,
  `task = ds.load_task("driver-position")`, `ds.get_db()`, `task.get_table("train")`, `task.evaluate(pred)`.
  Older tutorials use the 1.x `relbench.datasets.get_dataset` / `relbench.tasks.get_task` API, which no longer exists.
  GNN helpers are in `relbench.modeling`. Read the installed source under `.venv/lib/python*/site-packages/relbench/`
  when unsure.
- Use RelBench's own splits and `task.evaluate` (or `relbench.submit.evaluate_task` on a prediction table). Never
  re-implement a metric or re-split the data. Test predictions for the leaderboard are written with
  `relbench.submit.write_prediction_table` and checked with `uv run python -m relbench.submit <pred_dir>`.
- Respect the temporal split. Train only on the train table, tune on val, touch test only for a final reported number.
  Any feature must be computable from rows whose timestamp is at or before the seed time; leakage invalidates a result.
- Report the task's official metric, the number of seeds (at least 3 for a reported result), and mean and std.
- Compare against the published baselines (the RelBench leaderboard and paper) and say where a result sits. Don't claim
  a state-of-the-art result unless it beats the leaderboard on test with the official evaluation.
- A negative result is a result. Record it in `results/` and the experiment README just as you would a win.

## Working an issue

- One issue is one experiment or one change. Keep PRs focused.
- Before you open or update a PR: `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` all pass.
- The PR description says what was tried, the numbers (with seeds), the runtime, and how to reproduce it.
- If an issue is ambiguous about scope, the dataset or task, or how much compute to spend, ask in an issue comment rather
  than guessing.
- Never commit secrets, API keys or personal data.
