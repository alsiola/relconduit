"""Append run records to ``results/<dataset>/<task>.jsonl`` (one JSON object per line)."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import numpy as np

REQUIRED = (
    "date",
    "approach",
    "commit",
    "metric",
    "split",
    "value",
    "seeds",
    "runtime_minutes",
    "notes",
)


def git_commit(cwd: str | Path | None = None) -> str:
    """The current commit's short SHA, with ``-dirty`` if the tree has uncommitted changes.

    ``RELCONDUIT_COMMIT`` overrides it: a cloud machine gets the checkout's files but no
    ``.git``, so the launcher passes the commit it was started from.
    """
    if os.environ.get("RELCONDUIT_COMMIT"):
        return os.environ["RELCONDUIT_COMMIT"]
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def make_record(
    *,
    approach: str,
    metric: str,
    split: str,
    values: list[float],
    runtime_minutes: float,
    notes: str,
    commit: str,
    date: str | None = None,
    **extra,
) -> dict:
    """A results row: ``value`` is the mean over seeds, ``std`` the sample std (ddof=1)."""
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if not values:
        raise ValueError("values is empty")
    arr = np.asarray(values, dtype=float)
    rec = {
        "date": date or dt.date.today().isoformat(),
        "approach": approach,
        "commit": commit,
        "metric": metric,
        "split": split,
        "value": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else 0.0,
        "seeds": len(arr),
        "per_seed": [round(float(v), 6) for v in arr],
        "runtime_minutes": round(float(runtime_minutes), 2),
        "notes": notes,
    }
    rec.update(extra)
    return rec


def append_record(path: str | Path, record: dict) -> None:
    """Append one record as a JSON line. Never rewrites existing lines."""
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"record is missing {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
