"""The cost ledger, costs/ledger.json: every dollar this project spends, against config/cloud.json's budgetUsd.

Each rented machine (relconduit.cloud) records its hours and Google's Spot price for them; anything paid by hand goes
in with `uv run costs add`. Entries are pushed straight to main with git plumbing (no checkout or branch switch), so a
run from any worktree lands at once and every later budget check sees it.

    uv run costs                       # entries, total and what's left of the budget
    uv run costs add --what "..." --usd 3.20 [--issue 7] [--at 2026-11-20]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from relconduit.retry import retry

ROOT = Path(__file__).resolve().parents[2]
LEDGER = "costs/ledger.json"
CONFIG = ROOT / "config" / "cloud.json"
COMMENT = json.loads((ROOT / LEDGER).read_text())["$comment"] if (ROOT / LEDGER).exists() else ""
COMPUTE_SERVICE = "6F81-5844-456A"  # Compute Engine in the Cloud Billing catalogue
HOURS_PER_MONTH = 730


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def git(*args: str, env: dict | None = None, check: bool = True, strip: bool = True) -> str:
    res = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, env=env, check=False
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {res.stderr.strip()}")
    return res.stdout.strip() if strip else res.stdout


def ledger_on_main() -> dict:
    """main's ledger, fetched first, so the budget counts every run from every worktree."""
    git("fetch", "-q", "origin", "main")
    raw = git("show", f"origin/main:{LEDGER}", check=False)
    return json.loads(raw) if raw else {"$comment": COMMENT, "entries": []}


def spent(ledger: dict) -> float:
    return round(sum(e.get("usd") or 0 for e in ledger["entries"]), 2)


def push_to_main(
    path: str,
    content: str | Callable[[str], str],
    message: str,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Commits one file onto origin/main and pushes it, without touching the working tree. Returns the commit.

    `content` is the file's new text, or a function from main's current text ("" when it has none) to the new text,
    called again on every try so a push that lost a race to another worktree rebuilds on top of what won. When main
    already holds that text (a try whose push landed though git reported it failed), nothing is pushed and main's
    commit is returned. A failed
    fetch or push (a race, or a blip in the network or the credential helper) is retried with backoff and jitter,
    about two minutes in all; the error names git's last complaint.
    """

    def attempt() -> str:
        git("fetch", "-q", "origin", "main")
        current = git("show", f"origin/main:{path}", check=False, strip=False)
        text = content(current) if callable(content) else content
        if text == current:
            return git("rev-parse", "origin/main")
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
            git("read-tree", "origin/main", env=env)
            blob = subprocess.run(
                ["git", "-C", str(ROOT), "hash-object", "-w", "--stdin"],
                input=text,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
            tree = git("write-tree", env=env)
        commit = git("commit-tree", tree, "-p", "origin/main", "-m", message)
        git("push", "-q", "origin", f"{commit}:refs/heads/main")
        return commit

    try:
        return retry(
            attempt,
            attempts=attempts,
            base=4.0,
            cap=60.0,
            what=f"pushing {path} to main",
            sleep=sleep,
        )
    except Exception as e:  # every failed try ends here; say which file and why
        raise RuntimeError(f"couldn't push {path} to main after {attempts} tries: {e}") from e


def add_entry(raw: str, entry: dict) -> str:
    """The ledger file's text with `entry` added, given its current text ("" for none). Unchanged if `entry` is
    already there, so retrying a push that in fact landed can't record the same cost twice."""
    ledger = json.loads(raw) if raw.strip() else {"$comment": COMMENT, "entries": []}
    if entry in ledger["entries"]:
        return raw
    entries = sorted([*ledger["entries"], entry], key=lambda e: e["at"])
    return (
        json.dumps({"$comment": ledger.get("$comment", COMMENT), "entries": entries}, indent=2)
        + "\n"
    )


def record_cost(entry: dict, sleep: Callable[[float], None] = time.sleep) -> str:
    """Adds an entry to main's ledger and pushes it. Returns the commit."""
    usd = "unpriced" if entry.get("usd") is None else f"${entry['usd']:.2f}"
    return push_to_main(
        LEDGER, lambda raw: add_entry(raw, entry), f"Cost: {entry['what'][:80]}, {usd}", sleep=sleep
    )


# --- Prices -------------------------------------------------------------------------------------------------------


def unit_price(sku: dict) -> float | None:
    rates = sku.get("pricingInfo", [{}])[0].get("pricingExpression", {}).get("tieredRates", [])
    if not rates:
        return None
    u = rates[-1]["unitPrice"]
    return float(u.get("units", 0)) + u.get("nanos", 0) / 1e9


GPU_SKU = {"nvidia-tesla-t4": "Tesla T4", "nvidia-l4": "L4"}


def spot_price_per_hour(
    skus: list[dict],
    machine: str,
    region_name: str,
    vcpus: int,
    memory_gb: float,
    disk_gb: float,
    gpu: str | None = None,
) -> float | None:
    """A Spot machine's dollars an hour from the catalogue: its family's Spot core and RAM rates in the region, the
    GPU's Spot rate, and the balanced disk's monthly rate over 730 hours. None when a rate is missing.

    region_name is the catalogue's name for the region ("London" for europe-west2).
    """
    family = machine.split("-")[0].upper()
    custom = "-custom-" in machine
    suffix = f"running in {region_name}"

    def rate(pred) -> float | None:
        for s in skus:
            if pred(s["description"]):
                return unit_price(s)
        return None

    def kind(what: str):
        def pred(d: str) -> bool:
            if not d.startswith("Spot Preemptible ") or not d.endswith(suffix):
                return False
            rest = d.removeprefix("Spot Preemptible ").removesuffix(" " + suffix)
            for vendor in ("", " AMD", " Intel"):
                for c in [" Custom"] if custom else ["", " Predefined"]:
                    if rest == f"{family}{vendor}{c} Instance {what}":
                        return True
            return False

        return pred

    core = rate(kind("Core"))
    ram = rate(kind("Ram"))
    disk = rate(lambda d: d == f"Balanced PD Capacity in {region_name}")
    total = (
        None
        if None in (core, ram, disk)
        else core * vcpus + ram * memory_gb + disk * disk_gb / HOURS_PER_MONTH
    )
    if gpu and total is not None:
        g = rate(
            lambda d: (
                d == f"Nvidia {GPU_SKU.get(gpu, gpu)} GPU attached to Spot Preemptible VMs {suffix}"
            )
        )
        total = None if g is None else total + g
    return total


def compute_skus(token: str) -> list[dict]:
    """Every Compute Engine SKU priced in dollars."""
    out, page = [], ""
    while True:
        url = (
            f"https://cloudbilling.googleapis.com/v1/services/{COMPUTE_SERVICE}/skus"
            f"?currencyCode=USD&pageSize=5000" + (f"&pageToken={page}" if page else "")
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as res:
            body = json.load(res)
        out += body.get("skus", [])
        page = body.get("nextPageToken", "")
        if not page:
            return out


# --- CLI ----------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="costs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd")
    add = sub.add_parser("add", help="record a spend paid by hand")
    add.add_argument("--what", required=True)
    add.add_argument("--usd", type=float, required=True)
    add.add_argument("--kind", default="other", choices=["cloud", "service", "data", "other"])
    add.add_argument("--issue")
    add.add_argument("--at")
    args = p.parse_args(argv)
    budget = load_config()["budgetUsd"]
    if args.cmd == "add":
        entry = {
            "at": args.at or datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": args.kind,
            "what": args.what,
            "issue": args.issue,
            "usd": round(args.usd, 2),
            "priced": "entered",
        }
        print(
            f"[costs] added {entry['what']}, ${entry['usd']:.2f}; pushed {record_cost(entry)[:8]} to main"
        )
        return 0
    ledger = ledger_on_main()
    for e in ledger["entries"]:
        usd = "  unpriced" if e.get("usd") is None else f"${e['usd']:>9.2f}"
        issue = f" (issue {e['issue']})" if e.get("issue") else ""
        print(f"{e['at'][:16]}  {usd}  {e['what']}{issue}")
    total = spent(ledger)
    unpriced = any(e.get("usd") is None for e in ledger["entries"])
    print(
        f"{len(ledger['entries'])} entries, ${total:.2f} of ${budget:.2f} spent, ${budget - total:.2f} left"
        + (" (some unpriced)" if unpriced else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
