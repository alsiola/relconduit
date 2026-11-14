"""Issue 22: assemble and check the recommendation-family leaderboard package. Never submits.

1. Gathers the ten seed-0 test tables into ``<out>/preds/``:
   * wave 2 (rel-amazon x3, rel-hm) from issue 21's cloud checkpoint
     (``gs://oddsworth-lab-runs/relconduit-21-wave2``; ``--fetch`` copies them down);
   * the other six from ``refit22.py``'s output (``--refits``).
2. Compares every seed's test score with its recorded row in ``results/`` (seed by seed, and
   the mean), and writes ``<out>/reproduction.json`` and ``<out>/reproduction.md``.
3. Runs ``python -m relbench.submit <out>/preds --out <out>/relconduit.zip`` (no ``--submit``),
   which scores every table officially and writes ``relconduit-recommendation.zip``.

    uv run python experiments/rec-board/package22.py --fetch --out runs/submission22
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from refit22 import JOBS, SUBMISSION_SEED

BUCKET = "gs://oddsworth-lab-runs/relconduit-21-wave2/runs/cloud21"
WAVE2 = {
    "rel-amazon/user-item-purchase": "cand-lgbm-ranker",
    "rel-amazon/user-item-rate": "cand-lgbm-ranker",
    "rel-amazon/user-item-review": "cand-lgbm-ranker",
    "rel-hm/user-item-purchase": "cand-lgbm-ranker",
}
# Leaderboard order (relbench.submit.LEADERBOARD_TASKS["recommendation"]).
ORDER = [
    "rel-amazon/user-item-purchase",
    "rel-amazon/user-item-rate",
    "rel-amazon/user-item-review",
    "rel-avito/user-ad-visit",
    "rel-f1/driver-circuit-compete",
    "rel-hm/user-item-purchase",
    "rel-stack/user-post-comment",
    "rel-stack/post-post-related",
    "rel-trial/condition-sponsor-run",
    "rel-trial/site-sponsor-run",
]


def recorded(name: str, approach: str) -> dict:
    """The recorded test row for ``approach`` on ``name`` (the last one, if several)."""
    rows = [json.loads(line) for line in Path(f"results/{name}.jsonl").read_text().splitlines()]
    want = f"experiments/{name}/{approach}"
    rows = [r for r in rows if r["split"] == "test" and r["approach"] == want]
    if not rows:
        raise SystemExit(f"no recorded test row for {want}")
    return rows[-1]


def fetch_wave2(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in WAVE2:
        ds, task = name.split("/")
        job = f"{BUCKET}/ranker_{ds}_{task}"
        csv = f"{job}/work/{ds}/{task}/cand-lgbm-ranker/submission/{ds}__{task}.csv"
        for src, out in ((csv, dest / f"{ds}__{task}.csv"), (f"{job}/summary.json", None)):
            out = out or dest / f"summary_{ds}_{task}.json"
            if not out.exists():
                subprocess.run(["gcloud", "storage", "cp", "-q", src, str(out)], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/submission22")
    p.add_argument("--wave2", default="runs/wave2ckpt")
    p.add_argument("--refits", default="runs/refit22")
    p.add_argument("--fetch", action="store_true", help="copy wave 2's tables from the bucket")
    a = p.parse_args()
    out, wave2, refits = Path(a.out), Path(a.wave2), Path(a.refits)
    if a.fetch:
        fetch_wave2(wave2)
    preds = out / "preds"
    if preds.exists():
        shutil.rmtree(preds)
    preds.mkdir(parents=True)

    rows = []
    for name in ORDER:
        ds, task = name.split("/")
        fname = f"{ds}__{task}.csv"
        if name in WAVE2:
            approach, source = WAVE2[name], "issue 21 checkpoint (seed-0 table as written)"
            summary = json.loads((wave2 / f"summary_{ds}_{task}.json").read_text())
            table = wave2 / fname
        else:
            approach, source = JOBS[name]["approach"], "refit (refit22.py)"
            job = refits / f"refit_{ds}_{task}"
            summary = json.loads((job / "summary.json").read_text())
            table = job / fname
        shutil.copy(table, preds / fname)
        rec = recorded(name, approach)
        refit = [float(x) for x in summary["test_scores"]]
        per_seed = rec.get("per_seed") or []
        diffs = [r - s for r, s in zip(refit, per_seed)] if per_seed else []
        rows.append(
            {
                "task": name,
                "approach": f"experiments/{name}/{approach}",
                "source": source,
                "seeds": len(refit),
                "recorded_mean": rec["value"],
                "recorded_std": rec["std"],
                "recorded_per_seed": per_seed,
                "refit_per_seed": refit,
                "refit_mean": float(np.mean(refit)),
                "refit_std": float(np.std(refit, ddof=1)),
                "max_abs_seed_diff": max((abs(d) for d in diffs), default=None),
                "within_std": abs(float(np.mean(refit)) - rec["value"]) <= max(rec["std"], 1e-4),
                "submitted_seed": SUBMISSION_SEED,
                "submitted_value": refit[SUBMISSION_SEED],
            }
        )

    res = subprocess.run(
        [sys.executable, "-m", "relbench.submit", str(preds), "--out", str(out / "relconduit.zip")],
        capture_output=True,
        text=True,
        check=False,
    )
    (out / "submit_check.txt").write_text(res.stdout + res.stderr)
    print(res.stdout[-6000:], res.stderr[-2000:], flush=True)

    mean = float(np.mean([r["submitted_value"] for r in rows]))
    report = {"rows": rows, "board_mean_seed0": mean, "submit_returncode": res.returncode}
    (out / "reproduction.json").write_text(json.dumps(report, indent=2))
    lines = [
        (
            "| task | approach | seeds | recorded test (mean ± std) | refit test (mean ± std) "
            "| largest per-seed gap | seed-0 table | source |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        gap = "–" if r["max_abs_seed_diff"] is None else f"{r['max_abs_seed_diff']:.6f}"
        lines.append(
            f"| {r['task']} | {r['approach'].split('/')[-1]} | {r['seeds']} "
            f"| {r['recorded_mean']:.4f} ± {r['recorded_std']:.4f} "
            f"| {r['refit_mean']:.4f} ± {r['refit_std']:.4f} | {gap} "
            f"| {r['submitted_value']:.4f} | {r['source']} |"
        )
    lines.append(f"\nBoard mean of the seed-0 tables: {mean:.4f}.")
    (out / "reproduction.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if res.returncode != 0 or not all(r["within_std"] for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
