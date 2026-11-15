"""Issue 32: seed ensembling on the recommendation board, decided on val, then the package.

Rule, fixed before any ensemble's test table is scored: for each task the ensemble is the
rank average of its final seeds' scores (:data:`relconduit.ensemble.ENSEMBLE_RULE`). It
replaces the seed-0 table only if, on val (official ``task.evaluate``, every val row):

* its MAP is above both seed 0's and the mean of the seeds' MAPs, and
* a paired bootstrap over val rows (2,000 resamples) puts it above the seed mean in at least
  90% of resamples.

``--phase val`` (the default) scores each seed's and the ensemble's val table, runs the
bootstraps, applies the rule and writes ``<out>/val.json`` and ``<out>/val.md``. It reads no
test label. ``--phase test`` reads those decisions, writes each task's final table (the
ensemble or seed 0) into ``<out>/preds/``, scores each final table once with
``evaluate_task``, checks the seed-0 tables against their recorded rows, runs ``python -m
relbench.submit <out>/preds --out <out>/relconduit.zip`` (no ``--submit``) and writes the
board comparison (``<out>/board.json``, ``<out>/board.md``).

Inputs per task:

* the eight refits of ``refit32.py`` (``--refits``): ``<job>/ens/{val,test}_top_*.npy``;
* rel-stack post-post-related (issue 29, variant ``dense_minilm_100``): its kept score
  parquets, ``score0``-``score2`` (the three recorded seeds);
* rel-trial site-sponsor-run (issue 28, ``cand-lgbm-v2/final``): each seed's kept score
  parquets.

    uv run python experiments/rec-board/ensemble32.py --phase val
    uv run python experiments/rec-board/ensemble32.py --phase test
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from package22 import ORDER

from relconduit.ensemble import ENSEMBLE_RULE, paired_bootstrap, per_row_ap, save_top
from relconduit.link import DST, SEED

P_BETTER = 0.90
N_BOOT = 2000
# The final approach of each task (results/ rows) and its seeds.
FINAL = {
    "rel-amazon/user-item-purchase": ("cand-lgbm-ranker", 3),
    "rel-amazon/user-item-rate": ("cand-lgbm-ranker", 3),
    "rel-amazon/user-item-review": ("cand-lgbm-ranker", 3),
    "rel-avito/user-ad-visit": ("cand-lgbm-ranker", 3),
    "rel-f1/driver-circuit-compete": ("cand-lgbm-ranker", 3),
    "rel-hm/user-item-purchase": ("cand-lgbm-ranker", 3),
    "rel-stack/user-post-comment": ("cand-lgbm-ranker-plus", 5),
    "rel-stack/post-post-related": ("dense-ranker", 3),
    "rel-trial/condition-sponsor-run": ("cand-lgbm-ranker-plus", 5),
    "rel-trial/site-sponsor-run": ("cand-lgbm-v2/final", 3),
}
# Kapso, the board leader (leaderboard.json read 2026-11-14), test MAP@k.
KAPSO = {
    "rel-amazon/user-item-purchase": 0.0254,
    "rel-amazon/user-item-rate": 0.0231,
    "rel-amazon/user-item-review": 0.0295,
    "rel-avito/user-ad-visit": 0.0420,
    "rel-f1/driver-circuit-compete": 0.8795,
    "rel-hm/user-item-purchase": 0.0326,
    "rel-stack/user-post-comment": 0.1313,
    "rel-stack/post-post-related": 0.2178,
    "rel-trial/condition-sponsor-run": 0.1228,
    "rel-trial/site-sponsor-run": 0.3333,
}
KAPSO_MEAN = 0.1837


# Tasks whose ensemble averages seeds recorded under another row: rel-amazon user-item-rate's
# refit (3 threads) picked a different config from issue 21's record, so its seeds are refit32's.
SEED_ROWS = {
    "rel-amazon/user-item-rate": (
        "experiments/rec-board/refit32 (experiments/rel-amazon/user-item-rate/cand-lgbm-ranker)"
    ),
}


def recorded(name: str, split: str) -> dict:
    """The results row holding the seeds that ``name``'s tables come from."""
    approach = SEED_ROWS.get(name, f"experiments/{name}/{FINAL[name][0]}")
    rows = [json.loads(x) for x in Path(f"results/{name}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["approach"] == approach and r["split"] == split]
    if not rows:
        raise SystemExit(f"no recorded {split} row for {approach}")
    return rows[-1]


def _tables_from_scores(out: Path, split: str, X: pd.DataFrame, scores: dict, n: int, k: int):
    if not (out / f"{split}_top_ens.npy").exists():
        save_top(out, split, X, scores, n, k)


def load_tables(name: str, split: str, refits: Path, inputs: Path, cache: Path, task) -> dict:
    """``{"seed<s>": table, ..., "ens": table}`` of ``(n_rows, k)`` dst ids for ``split``."""
    ds, t = name.split("/")
    k = task.eval_k
    n = len(task.get_table(split).df)
    if name == "rel-stack/post-post-related":
        d = cache / "ppr"
        X = pd.read_parquet(inputs / "ppr" / f"dense_minilm_100_{split}_scores.parquet")
        scores = {s: X[f"score{s}"].to_numpy() for s in range(3)}
        _tables_from_scores(d, split, X[[SEED, DST]], scores, n, k)
    elif name == "rel-trial/site-sponsor-run":
        d = cache / "ssr"
        parts = [
            pd.read_parquet(inputs / "ssr" / f"seed{s}" / f"{split}_scores.parquet")
            for s in range(3)
        ]
        for p in parts[1:]:
            if not (
                np.array_equal(p[SEED], parts[0][SEED]) and np.array_equal(p[DST], parts[0][DST])
            ):
                raise SystemExit("site-sponsor-run: seeds score different pairs")
        scores = {s: p["score"].to_numpy() for s, p in enumerate(parts)}
        _tables_from_scores(d, split, parts[0][[SEED, DST]], scores, n, k)
    else:
        d = refits / f"refit_{ds}_{t}" / "ens"
    seeds = FINAL[name][1]
    out = {f"seed{s}": np.load(d / f"{split}_top_seed{s}.npy") for s in range(seeds)}
    out["ens"] = np.load(d / f"{split}_top_ens.npy")
    for key, tab in out.items():
        if tab.shape != (n, k):
            raise SystemExit(f"{name} {split} {key}: shape {tab.shape}, want {(n, k)}")
    return out


def val_phase(a) -> None:
    import relbench
    from relbench.metrics import map as official_map

    out = Path(a.out)
    rows = []
    for name in a.only or ORDER:
        ds, t = name.split("/")
        task = relbench.load_dataset(f"stanford-star/relbench-v1/{ds}").load_task(t)
        table = task.get_table("val", mask_input_cols=False)
        tabs = load_tables(name, "val", Path(a.refits), Path(a.inputs), out / "tables", task)
        score, per_row = {}, {}
        for key, pred in tabs.items():
            res = task.evaluate(pred, table, metrics=[official_map, per_row_ap])
            score[key], per_row[key] = float(res["map"]), res["per_row_ap"]
            if not np.isclose(per_row[key].mean(), score[key], rtol=0, atol=1e-12):
                raise SystemExit(f"{name} {key}: per-row AP mean != official MAP")
        seed_keys = [x for x in tabs if x != "ens"]
        seed_vals = [score[x] for x in seed_keys]
        rec = recorded(name, "val")["per_seed"]
        seed_mean_rows = np.mean([per_row[x] for x in seed_keys], axis=0)
        vs0 = paired_bootstrap(per_row["ens"], per_row["seed0"], N_BOOT)
        vsm = paired_bootstrap(per_row["ens"], seed_mean_rows, N_BOOT)
        use = (
            score["ens"] > score["seed0"]
            and score["ens"] > float(np.mean(seed_vals))
            and vsm["p_better"] >= P_BETTER
        )
        row = {
            "task": name,
            "approach": f"experiments/{name}/{FINAL[name][0]}",
            "seeds": len(seed_keys),
            "val_rows_scored": len(per_row["ens"]),
            "val_seed": seed_vals,
            "val_seed_recorded": rec,
            "val_seed_max_gap_vs_recorded": float(np.max(np.abs(np.array(seed_vals) - rec))),
            "val_seed0": score["seed0"],
            "val_seed_mean": float(np.mean(seed_vals)),
            "val_ens": score["ens"],
            "vs_seed0": vs0,
            "vs_seed_mean": vsm,
            "use_ensemble": bool(use),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("task", "val_seed0", "val_seed_mean", "val_ens", "use_ensemble")
                }
            ),
            flush=True,
        )
    if a.only:
        out = out / "partial"
    out.mkdir(parents=True, exist_ok=True)
    report = {"rule": ENSEMBLE_RULE, "p_better": P_BETTER, "n_boot": N_BOOT, "rows": rows}
    (out / "val.json").write_text(json.dumps(report, indent=2))
    lines = [
        (
            "| task | seeds | val seed 0 | val seed mean | val ensemble | ens − seed 0 [95% CI] "
            "| ens − seed mean [95% CI] | P(ens > mean) | use |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a0, am = r["vs_seed0"], r["vs_seed_mean"]
        lines.append(
            f"| {r['task']} | {r['seeds']} | {r['val_seed0']:.4f} | {r['val_seed_mean']:.4f} "
            f"| {r['val_ens']:.4f} | {a0['diff']:+.4f} [{a0['ci_lo']:+.4f}, {a0['ci_hi']:+.4f}] "
            f"| {am['diff']:+.4f} [{am['ci_lo']:+.4f}, {am['ci_hi']:+.4f}] | {am['p_better']:.2f} "
            f"| {'ensemble' if r['use_ensemble'] else 'seed 0'} |"
        )
    (out / "val.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def test_phase(a) -> None:
    import relbench
    from relbench.submit import evaluate_task, write_prediction_table

    from relconduit.results import append_record, git_commit, make_record

    out = Path(a.out)
    val = json.loads((out / "val.json").read_text())
    preds = out / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in val["rows"]:
        name = r["task"]
        ds, t = name.split("/")
        task = relbench.load_dataset(f"stanford-star/relbench-v1/{ds}").load_task(t)
        tabs = load_tables(name, "test", Path(a.refits), Path(a.inputs), out / "tables", task)
        key = "ens" if r["use_ensemble"] else "seed0"
        path = preds / f"{ds}__{t}.csv"
        write_prediction_table(task, tabs[key], path)
        (value,) = evaluate_task(name, path).values()  # once per final table
        rec = recorded(name, "test")
        if key == "seed0" and abs(value - rec["per_seed"][0]) > 5e-4:
            raise SystemExit(f"{name}: seed-0 test {value:.6f} != recorded {rec['per_seed'][0]}")
        row = {
            "task": name,
            "table": "ensemble" if key == "ens" else "seed 0",
            "test": float(value),
            "test_seed0_recorded": rec["per_seed"][0],
            "test_seed_mean": rec["value"],
            "test_seed_std": rec["std"],
            "seeds": rec["seeds"],
            "kapso": KAPSO[name],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if key == "ens" and a.append:
            append_record(
                f"results/{name}.jsonl",
                make_record(
                    approach=f"experiments/rec-board/ensemble32 ({r['approach']})",
                    metric="map",
                    split="test",
                    values=[float(value)],
                    runtime_minutes=0.0,
                    notes=(
                        f"Issue 32: one table, the {ENSEMBLE_RULE} of the {r['seeds']} recorded "
                        f"seeds of {r['approach']}; chosen on val (ensemble {r['val_ens']:.4f} vs "
                        f"seed mean {r['val_seed_mean']:.4f}, P(better) "
                        f"{r['vs_seed_mean']['p_better']:.2f}). Test scored once. The seeds' own "
                        f"test mean is {rec['value']:.4f} ± {rec['std']:.4f} ({rec['seeds']})."
                    ),
                    commit=git_commit(),
                    ensemble_of_seeds=r["seeds"],
                    val_ensemble=round(r["val_ens"], 6),
                ),
            )
    res = subprocess.run(
        [sys.executable, "-m", "relbench.submit", str(preds), "--out", str(out / "relconduit.zip")],
        capture_output=True,
        text=True,
        check=False,
    )
    (out / "submit_check.txt").write_text(res.stdout + res.stderr)
    print(res.stdout[-4000:], res.stderr[-2000:], flush=True)
    mean = float(np.mean([r["test"] for r in rows]))
    mean_of_means = float(np.mean([r["test_seed_mean"] for r in rows]))
    lines = [
        "| task | table | ours (submitted) | seed mean ± std (seeds) | Kapso | gap | above Kapso: table / mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        gap = r["test"] - r["kapso"]
        lines.append(
            f"| {r['task']} | {r['table']} | {r['test']:.4f} | {r['test_seed_mean']:.4f} ± "
            f"{r['test_seed_std']:.4f} ({r['seeds']}) | {r['kapso']:.4f} | {gap:+.4f} "
            f"| {'yes' if r['test'] > r['kapso'] else 'no'} / "
            f"{'yes' if r['test_seed_mean'] > r['kapso'] else 'no'} |"
        )
    n_tab = sum(r["test"] > r["kapso"] for r in rows)
    n_mean = sum(r["test_seed_mean"] > r["kapso"] for r in rows)
    lines.append(
        f"| **board mean** | | **{mean:.4f}** | {mean_of_means:.4f} | **{KAPSO_MEAN:.4f}** "
        f"| {mean - KAPSO_MEAN:+.4f} | {n_tab} / {n_mean} tasks |"
    )
    board = {
        "rows": rows,
        "board_mean_submitted": mean,
        "board_mean_seed_means": mean_of_means,
        "kapso_mean": KAPSO_MEAN,
        "above_kapso_by_table": int(n_tab),
        "above_kapso_by_mean": int(n_mean),
        "submit_returncode": res.returncode,
    }
    (out / "board.json").write_text(json.dumps(board, indent=2))
    (out / "board.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if res.returncode != 0:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["val", "test"], default="val")
    p.add_argument("--out", default="runs/ensemble32")
    p.add_argument("--refits", default="runs/refit32")
    p.add_argument("--inputs", default="runs/inputs32")
    p.add_argument("--append", action="store_true", help="test: append results rows")
    p.add_argument("--only", nargs="*", help="val: only these tasks (a dry check)")
    a = p.parse_args()
    val_phase(a) if a.phase == "val" else test_phase(a)


if __name__ == "__main__":
    main()
