"""Issue 9's cloud run: site-success with entity encodings, and the rel-f1 vocab check.

Jobs, side by side on one 24-vCPU machine, each its own process with ``<out>/<job>/log``
and ``done.json`` written last (by rename), so a restarted machine skips finished jobs:

* ``smoke``: site-success on 2,000 train and val rows, first, so a broken build fails in
  minutes;
* ``site-success``: ``run.py --test`` (tune on val, score test once, append results rows);
* ``f1_<task>``: each rel-f1 dfs-lightgbm run on val only, twice on this code: with the old
  vocab (every row of the main database) and the new one (rows before the val timestamp).
  If the val mean moved by more than the old run's seed std, the job then runs ``--test``
  and appends the rows (issue 9, item 4).

    uv run python experiments/rel-trial/site-success/dfs-lightgbm/cloud_jobs.py --out runs/cloud9
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SITE = "experiments/rel-trial/site-success/dfs-lightgbm/run.py"
F1_TASKS = ("driver-position", "driver-dnf", "driver-top3")


def sh(cmd: list[str], log: Path, threads: int) -> int:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "PYTHONUNBUFFERED": "1",
    }
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(
            cmd, check=False, stdout=f, stderr=subprocess.STDOUT, env=env
        ).returncode


def job(name: str, out: Path, body) -> str:
    d = out / name
    if (d / "done.json").exists():
        return f"{name}: already done"
    d.mkdir(parents=True, exist_ok=True)
    print(f"{name}: started", flush=True)
    t0 = time.time()
    info = body(d)
    minutes = (time.time() - t0) / 60
    if info is None:
        msg = f"{name}: FAILED after {minutes:.1f} min (see {d / 'log'})"
        print(msg, flush=True)
        return msg
    tmp = d / "done.json.tmp"
    tmp.write_text(json.dumps({"minutes": round(minutes, 2), **info}))
    tmp.rename(d / "done.json")
    msg = f"{name}: done in {minutes:.1f} min {info}"
    print(msg, flush=True)
    return msg


def site(out: Path, threads: int, smoke: bool) -> str:
    def body(d: Path):
        cmd = [sys.executable, SITE] + (["--smoke", "2000"] if smoke else ["--test"])
        if sh(cmd, d / "log", threads) != 0:
            return None
        for f in Path("runs/rel-trial/site-success/dfs-lightgbm").glob("*.json"):
            shutil.copy(f, d / f.name)
        if not smoke:
            shutil.copy("results/rel-trial/site-success.jsonl", d / "rows.jsonl")
        return {"smoke": smoke}

    return job("smoke" if smoke else "site-success", out, body)


# The pre-issue-9 vocab: every row of the main database, passed in so run() keeps it.
OLD_VOCAB = """
import relbench, sys
from relconduit.entity_lgbm import run
from relconduit.features import FeatureConfig, feature_vocab
task = sys.argv[1]
db = relbench.load_dataset("stanford-star/relbench-v1/rel-f1").load_task(task).get_db()
cfg = FeatureConfig(vocab=feature_vocab(db))
run("rel-f1", task, "old-vocab", feature_config=cfg, work_dir="runs/oldvocab")
"""


def _mean_std(vals: list[float]) -> tuple[float, float]:
    m = sum(vals) / len(vals)
    return m, (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def f1(task: str, out: Path, threads: int) -> str:
    def body(d: Path):
        script = f"experiments/rel-f1/{task}/dfs-lightgbm/run.py"
        if sh([sys.executable, "-c", OLD_VOCAB, task], d / "log", threads) != 0:
            return None
        if sh([sys.executable, script], d / "log", threads) != 0:
            return None
        scores = {}
        for tag, root in (("old", "runs/oldvocab"), ("new", "runs")):
            f = Path(f"{root}/rel-f1/{task}/dfs-lightgbm/summary_val.json")
            shutil.copy(f, d / f"summary_val_{tag}.json")
            scores[tag] = json.loads(f.read_text())["val_scores"]
        (old, old_sd), (new, _) = _mean_std(scores["old"]), _mean_std(scores["new"])
        material = abs(new - old) > old_sd
        info = {"old": scores["old"], "new": scores["new"], "material": material}
        if material:
            if sh([sys.executable, script, "--test"], d / "log", threads) != 0:
                return None
            shutil.copy(f"results/rel-f1/{task}.jsonl", d / "rows.jsonl")
        return info

    return job(f"f1_{task}", out, body)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cloud9")
    p.add_argument("--site-threads", type=int, default=18)
    p.add_argument("--f1-threads", type=int, default=2)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    msgs = [site(out, a.site_threads, smoke=True)]
    if "FAILED" not in msgs[0]:
        with ThreadPoolExecutor(1 + len(F1_TASKS)) as pool:
            futs = [pool.submit(site, out, a.site_threads, False)]
            futs += [pool.submit(f1, t, out, a.f1_threads) for t in F1_TASKS]
            msgs += [f.result() for f in futs]
    for m in msgs:
        print(m, flush=True)
    if any("FAILED" in m for m in msgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
