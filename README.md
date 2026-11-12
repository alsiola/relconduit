# relconduit

Agent-driven experiments on [RelBench](https://relbench.stanford.edu/) ([project page](https://star-project.stanford.edu/relbench/)),
the Stanford benchmark for deep learning over relational databases.

Work arrives as GitHub issues and is picked up by an [Open Conduit](https://github.com/open-conduit/open-conduit)
agent (Claude Code, Opus), which lands it as a pull request. A separate reviewer app reviews every PR. Nothing runs in
CI. Run checks locally.

## Getting started

```sh
uv sync --all-extras   # relbench + LightGBM + GNN (torch, PyG, PyTorch Frame)
uv run pytest
```

RelBench 3.x downloads datasets from the Hugging Face Hub on first use (into `~/.cache/huggingface`).

## Cloud runs

Long runs go to Google Cloud Spot machines with `uv run cloud-run` (24 vCPUs; GPUs once enabled), within a $50 budget
tracked in `costs/ledger.json`. Run `uv run costs` for the spend so far. See `.claude/skills/cloud-compute/SKILL.md`.

## Layout

- `src/relconduit/`: shared code (data loading, feature engineering, models, evaluation helpers).
- `experiments/<dataset>/<task>/<approach>/`: one directory per experiment, with a `README.md` and its scripts.
- `results/`: committed result records (`results/<dataset>/<task>.jsonl`), one line per run.
- `tests/`: pytest.

See [CLAUDE.md](CLAUDE.md) for the rules agents follow.
