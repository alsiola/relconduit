"""Append the baseline rows of issue 12's cloud run to ``results/<dataset>/<task>.jsonl``.

Reads ``<out>/trivial_<ds>_<task>/summary.txt`` (past visit and global popularity:
deterministic, one run) and ``<out>/lgbm_<ds>_<task>_s<seed>/summary.txt`` (upstream LightGBM,
seeds 42-44), with each job's ``done.json`` for its runtime and peak memory.

    uv run python experiments/rec-board/baseline_rows.py --out runs/cloud12 --commit <sha> \
        --machine "Google Cloud n2-custom-24-196608 Spot" \
        --trivial-commit <sha> --trivial-machine "Google Cloud n2-custom-24-131072 Spot"

The trivial baselines and the upstream LightGBM jobs can come from different runs of the driver,
so each gets its own commit and machine (the trivial ones default to ``--commit``/``--machine``).
"""

import argparse
import ast
import json
import re
from pathlib import Path

from relconduit.results import append_record, make_record

TASKS = [
    ("rel-avito", "user-ad-visit"),
    ("rel-stack", "user-post-comment"),
    ("rel-trial", "site-sponsor-run"),
    ("rel-stack", "post-post-related"),
    ("rel-trial", "condition-sponsor-run"),
    ("rel-f1", "driver-circuit-compete"),
    # Wave 2 (issue 21): trivial baselines only (``--only trivial``).
    ("rel-amazon", "user-item-purchase"),
    ("rel-amazon", "user-item-rate"),
    ("rel-amazon", "user-item-review"),
    ("rel-hm", "user-item-purchase"),
]
SEEDS = (42, 43, 44)
UPSTREAM = "experiments/rel-f1/baselines/upstream"


def _maps(text: str) -> dict[str, dict[str, float]]:
    """``{section: {"Val": map, "Test": map}}``; section "" when there is no heading."""
    out, section = {}, ""
    for line in text.splitlines():
        if line.endswith(":") and not line.startswith(("Train", "Val", "Test")):
            section = line[:-1]
            continue
        m = re.match(r"(Train|Val|Test): (\{.*\})", line)
        if m:
            d = ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", m.group(2)))
            out.setdefault(section, {})[m.group(1)] = float(d["map"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cloud12")
    p.add_argument("--commit", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--trivial-commit", help="commit the trivial jobs ran at (default --commit)")
    p.add_argument("--trivial-machine", help="machine the trivial jobs ran on (default --machine)")
    p.add_argument("--only", choices=("trivial", "lightgbm"), help="append only these rows")
    p.add_argument("--tasks", nargs="+", metavar="DS/TASK", help="only these tasks (default all)")
    a = p.parse_args()
    trivial_commit = a.trivial_commit or a.commit
    trivial_machine = a.trivial_machine or a.machine
    out = Path(a.out)
    for ds, task in TASKS:
        if a.tasks and f"{ds}/{task}" not in a.tasks:
            continue
        path = f"results/{ds}/{task}.jsonl"
        approach = f"experiments/{ds}/baselines"
        job = out / f"trivial_{ds}_{task}"
        if a.only != "lightgbm" and (job / "done.json").exists():
            done = json.loads((job / "done.json").read_text())
            for name, m in _maps((job / "summary.txt").read_text()).items():
                for split in ("val", "test"):
                    append_record(
                        path,
                        make_record(
                            approach=approach,
                            metric="map",
                            split=split,
                            values=[m[split.capitalize()]],
                            runtime_minutes=done["minutes"],
                            commit=trivial_commit,
                            notes=(
                                f"Upstream examples/trivial_recommendation.py ({UPSTREAM}), "
                                f"{name}: deterministic, one run. Test predictions use train+val "
                                f"labels, as upstream does. {trivial_machine}, 1 thread."
                            ),
                            baseline=name,
                        ),
                    )
        if a.only == "trivial":
            continue
        vals, tests, minutes, peaks, seeds = [], [], [], [], []
        for s in SEEDS:
            job = out / f"lgbm_{ds}_{task}_s{s}"
            if not (job / "done.json").exists():
                continue
            done = json.loads((job / "done.json").read_text())
            m = _maps((job / "summary.txt").read_text())[""]
            vals.append(m["Val"])
            tests.append(m["Test"])
            minutes.append(done["minutes"])
            peaks.append(float(done["peak_rss_gb"] or "nan"))
            seeds.append(s)
        if not vals:
            print(f"{ds}/{task}: no upstream LightGBM seed finished")
            continue
        notes = (
            f"Upstream examples/lightgbm_recommendation.py ({UPSTREAM}): src and dst entity "
            f"columns plus past-visit and popularity counts, 10 Optuna trials on val, seeds "
            f"{', '.join(str(s) for s in seeds)}. Runtime is mean minutes per seed; "
            f"{a.machine}; peak RSS {max(peaks):.1f} GB."
        )
        for split, values in (("val", vals), ("test", tests)):
            append_record(
                path,
                make_record(
                    approach=approach,
                    metric="map",
                    split=split,
                    values=values,
                    runtime_minutes=sum(minutes) / len(minutes),
                    commit=a.commit,
                    notes=notes,
                    baseline="lightgbm",
                ),
            )
        print(f"{ds}/{task}: lightgbm val {vals} test {tests}")


if __name__ == "__main__":
    main()
