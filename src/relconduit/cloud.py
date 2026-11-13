"""Runs one long command on a rented Google Cloud Spot machine, then deletes it (the cloud-compute skill):

    uv run cloud-run --issue 7 --approved <owner's comment url> --hours 4 [--gpu nvidia-l4] [--machine TYPE]
        [--upload PATH]... [--download PATH]... [--checkpoint DIR --run NAME] [--dry-run] -- <command>

Only after the owner has approved the run on its issue. Steps:
 1. check the run fits what's left of the budget (config/cloud.json budgetUsd, less costs/ledger.json on main, less
    the most that machines still running could cost) at Google's Spot price for --hours, and that quota covers it;
 2. create the machine: Spot, labelled with the issue and its most possible cost, and set to delete itself after
    --hours whatever happens to this script, so a crash here can never leave one running;
 3. copy up this checkout (tracked and untracked files, not ignored ones) and each --upload, install uv and
    `uv sync --all-extras --frozen`;
 4. restore any --checkpoint folders from Cloud Storage, run the command detached from SSH, save the checkpoints every
    few minutes and at the end, and echo the log here;
 5. copy each --download back, delete the machine, and record its hours and price in the cost ledger on main.

A machine Google takes back is replaced and the run resumes from its checkpoints, up to --retries times, within the
original --hours. RelBench datasets download from the Hugging Face Hub on the machine; nothing secret is copied up.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

from relconduit import costs

ROOT = costs.ROOT
POLL_SECONDS = 60
REGION_NAMES = {
    "europe-west2": "London"
}  # the Cloud Billing catalogue's names for the regions we use


class Gone(Exception):
    """The machine went away mid-run: taken back by Google, or past its hours."""


# --- Pure helpers (tested) ---------------------------------------------------------------------------------------


def vcpus(machine: str) -> int:
    """The vCPUs in a machine type's name: `n2-custom-24-24576` -> 24, `g2-standard-8` -> 8."""
    m = re.search(r"-custom-(\d+)-", machine)
    n = m.group(1) if m else machine.rsplit("-", 1)[-1]
    if not n.isdigit() or int(n) <= 0:
        raise ValueError(f"can't read the vCPU count from {machine}")
    return int(n)


def remote_path(local: str | Path, root: Path = ROOT, home: Path | None = None) -> str:
    """Where a local path lands on the machine: `repo/<rel>` inside the checkout, `<rel>` under home."""
    p = Path(local).expanduser().resolve()
    home = home or Path.home()
    for base, prefix in ((root.resolve(), "repo/"), (home.resolve(), "")):
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if str(rel) != ".":
            return prefix + rel.as_posix()
    raise ValueError(
        f"--upload {local}: only paths inside the checkout or the home folder can be copied up"
    )


def checkpoint_path(p: str) -> str:
    clean = posixpath.normpath(p).rstrip("/")
    if not clean or clean == "." or clean.startswith("..") or posixpath.isabs(clean):
        raise ValueError(
            f"--checkpoint {p}: give a folder inside the checkout, relative to it (e.g. runs/sweep)"
        )
    return clean


def gpu_quota_metric(gpu: str) -> str:
    """`nvidia-tesla-t4` -> `NVIDIA_T4_GPUS`, `nvidia-l4` -> `NVIDIA_L4_GPUS`."""
    return gpu.upper().replace("-TESLA", "").replace("-", "_") + "_GPUS"


def quota_shortfalls(
    machine: str, gpu: str | None, region_quotas: list[dict], project_quotas: list[dict]
) -> list[str]:
    """What quota is missing for one Spot machine (and its GPU) in a region."""
    need = vcpus(machine)
    family = machine.split("-")[0].upper()

    def find(qs: list[dict], metric: str) -> dict | None:
        return next((q for q in qs if q["metric"] == metric), None)

    checks = [(find(project_quotas, "CPUS_ALL_REGIONS"), need)]
    pre = find(region_quotas, "PREEMPTIBLE_CPUS")
    # A region with its own Spot quota counts Spot machines there instead of in the family's quota.
    checks.append(
        (
            pre
            if pre and pre["limit"] > 0
            else find(region_quotas, f"{family}_CPUS") or find(region_quotas, "CPUS"),
            need,
        )
    )
    if gpu:
        metric = gpu_quota_metric(gpu)
        checks.append(
            (
                find(project_quotas, "GPUS_ALL_REGIONS")
                or {"metric": "GPUS_ALL_REGIONS", "limit": 0, "usage": 0},
                1,
            )
        )
        pre_gpu = find(region_quotas, f"PREEMPTIBLE_{metric}")
        checks.append(
            (
                pre_gpu
                if pre_gpu and pre_gpu["limit"] > 0
                else find(region_quotas, metric) or {"metric": metric, "limit": 0, "usage": 0},
                1,
            )
        )
    return [
        f"{q['metric']}: {q['limit'] - q['usage']:g} free of {q['limit']:g}, the machine needs {n}"
        for q, n in checks
        if q and q["limit"] - q["usage"] < n
    ]


def budget_check(budget: float, spent: float, reserved: float, cost: float) -> str | None:
    """Why a run of at most `cost` dollars can't start, or None if it fits."""
    left = budget - spent - reserved
    if cost > left:
        return (
            f"the run could cost up to ${cost:.2f}, but only ${left:.2f} of the ${budget:.2f} budget is left "
            f"(${spent:.2f} spent, ${reserved:.2f} held by machines still running)"
        )
    return None


def sync_lines(
    checkpoints: list[str], prefix: str, direction: str, repo: str = "$HOME/repo"
) -> str:
    """Shell copying each checkpoint folder between the machine's checkout and the bucket. A failed save prints a
    `[checkpoint]` line and marks ~/checkpoint.failed; a restore with nothing saved yet starts fresh; any other restore
    failure stops the setup."""
    out = []
    for p in checkpoints:
        if direction == "save":
            out.append(
                f'if [ -d "{repo}/{p}" ]; then gcloud storage rsync --recursive --quiet "{repo}/{p}" "{prefix}/{p}" '
                f'|| {{ echo "[checkpoint] saving {p} to {prefix} failed (exit $?)" >&2; touch "$HOME/checkpoint.failed"; }}; fi'
            )
        else:
            out.append(
                f'mkdir -p "{repo}/{p}"\n'
                f'if listing=$(gcloud storage ls "{prefix}/{p}/" 2>&1); then\n'
                f'  gcloud storage rsync --recursive --quiet "{prefix}/{p}" "{repo}/{p}" && echo "[checkpoint] restored {p}"\n'
                f'elif echo "$listing" | grep -q "matched no objects"; then\n'
                f'  echo "[checkpoint] nothing saved for {p} yet; starting fresh"\n'
                f'else\n  echo "[checkpoint] can\'t read {prefix}/{p}: $listing" >&2; exit 1\nfi'
            )
    return "\n".join(out)


def setup_script(gpu: bool) -> str:
    check_gpu = "nvidia-smi --query-gpu=name,memory.total --format=csv\n" if gpu else ""
    return f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq curl ca-certificates git libgomp1 >/dev/null
{check_gpu}curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/repo && tar xzf ~/repo.tgz -C ~/repo && rm ~/repo.tgz
cd ~/repo
uv sync --all-extras --frozen
echo setup done"""


def start_script(command: str, save: str, checkpoint_minutes: int) -> str:
    saver = (
        f"save() {{\n{save}\n}}\n( while sleep {checkpoint_minutes * 60}; do save; done ) &\nsaver=$!\n"
        if save
        else ""
    )
    finish = "kill $saver 2>/dev/null\nsave\n" if save else ""
    return f"""cat > ~/run.sh <<'RUN_EOF'
export PATH="$HOME/.local/bin:$PATH"
cd ~/repo
{saver}{command}
code=$?
{finish}echo $code > ~/run.exit
RUN_EOF
setsid nohup bash ~/run.sh > ~/run.log 2>&1 < /dev/null &
echo started"""


# --- gcloud ------------------------------------------------------------------------------------------------------


class Gcloud:
    def __init__(self, project: str, zone: str, dry_run: bool, key: str):
        self.project, self.zone, self.dry_run = project, zone, dry_run
        key_path = Path(key).expanduser()
        # Sign in as the Mac's service account when its key is here, so a person's lapsed login never stops a run.
        self.env = (
            {**os.environ, "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": str(key_path)}
            if key_path.exists()
            else None
        )

    def run(
        self,
        args: list[str],
        echo: bool = False,
        allow_fail: bool = False,
        input: str = "",
        mutating: bool = False,
    ):
        full = ["gcloud", *args, f"--project={self.project}", "--quiet"]
        if self.dry_run and mutating:
            print(f"[dry run] {' '.join(full)}")
            return 0, ""
        proc = subprocess.Popen(
            full,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if echo else subprocess.PIPE,
            text=True,
            env=self.env,
        )
        out_lines = []
        proc.stdin.write(input)
        proc.stdin.close()
        for line in proc.stdout:
            out_lines.append(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        err = "" if echo else proc.stderr.read()
        code = proc.wait()
        out = "".join(out_lines)
        if code != 0 and not allow_fail:
            tail = "\n".join((err or out).strip().splitlines()[-5:])
            raise RuntimeError(f"gcloud {' '.join(args[:3])} exited {code}: {tail}")
        return code, out

    def ssh(self, name: str, script: str, **kw):
        return self.run(
            ["compute", "ssh", name, f"--zone={self.zone}", "--command=bash -s"],
            input=script,
            mutating=True,
            **kw,
        )

    def scp(self, src: str, dst: str):
        return self.run(
            ["compute", "scp", "--recurse", "--compress", f"--zone={self.zone}", src, dst],
            echo=True,
            mutating=True,
        )


def pack_checkout(dest: Path) -> None:
    files = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(dest, "w:gz") as tar:
        for f in filter(None, files.decode().split("\0")):
            if (ROOT / f).exists():
                tar.add(ROOT / f, arcname=f, recursive=False)


def memory_gb(gc: Gcloud, machine: str) -> float:
    m = re.search(r"-custom-\d+-(\d+)", machine)
    if m:
        return int(m.group(1)) / 1024
    _, out = gc.run(
        [
            "compute",
            "machine-types",
            "describe",
            machine,
            f"--zone={gc.zone}",
            "--format=value(memoryMb)",
        ]
    )
    return int(out.strip()) / 1024


def price_per_hour(gc: Gcloud, machine: str, disk: int, gpu: str | None) -> float | None:
    region = gc.zone.rsplit("-", 1)[0]
    try:
        _, token = gc.run(["auth", "print-access-token"])
        return costs.spot_price_per_hour(
            costs.compute_skus(token.strip()),
            machine,
            REGION_NAMES.get(region, region),
            vcpus(machine),
            memory_gb(gc, machine),
            disk,
            gpu,
        )
    except Exception as e:  # noqa: BLE001 - reported, and the caller refuses to run unpriced
        print(f"[cloud] couldn't read Google's prices: {e}", file=sys.stderr)
        return None


def reserved_by_running(gc: Gcloud, prefix: str) -> float:
    """The most that this project's machines still running could cost: each carries its cap as a label, in cents."""
    _, out = gc.run(
        ["compute", "instances", "list", f"--filter=labels.project={prefix}", "--format=json"]
    )
    return sum(int(i.get("labels", {}).get("max-cents", 0)) for i in json.loads(out or "[]")) / 100


# --- One machine ---------------------------------------------------------------------------------------------------


def run_once(plan: dict, name: str, hours: float) -> int | None:
    gc: Gcloud = plan["gc"]
    cfg = plan["config"]
    work = ROOT / ".local" / "cloud" / name
    work.mkdir(parents=True, exist_ok=True)
    created, started = False, time.time()
    gpu = plan["gpu"]
    image = cfg["gpuImage"] if gpu else cfg["image"]
    try:
        shutdown = work / "shutdown.sh"
        shutdown.write_text(
            "#!/bin/bash\nfor home in /home/*; do\n"
            + sync_lines(plan["checkpoints"], plan["prefix"], "save", "$home/repo")
            + "\ndone\n"
        )
        max_cents = math.ceil(plan["per_hour"] * hours * 100)
        gc.run(
            [
                "compute",
                "instances",
                "create",
                name,
                f"--zone={gc.zone}",
                f"--machine-type={plan['machine']}",
                "--provisioning-model=SPOT",
                "--instance-termination-action=DELETE",
                f"--max-run-duration={math.ceil(hours * 3600)}s",
                f"--image-family={image['family']}",
                f"--image-project={image['project']}",
                f"--boot-disk-size={plan['disk']}GB",
                "--boot-disk-type=pd-balanced",
                f"--service-account={cfg['serviceAccount']}",
                "--scopes=cloud-platform",
                f"--labels=project={cfg['bucketPrefix']},issue={plan['issue']},max-cents={max_cents}",
                *(
                    [f"--accelerator=type={gpu},count=1"]
                    if gpu and not plan["machine"].startswith("g2-")
                    else []
                ),
                *(
                    ["--maintenance-policy=TERMINATE", "--metadata=install-nvidia-driver=True"]
                    if gpu
                    else []
                ),
                *(
                    [f"--metadata-from-file=shutdown-script={shutdown}"]
                    if plan["checkpoints"]
                    else []
                ),
            ],
            echo=True,
            mutating=True,
        )
        created = True
        print(
            f"[cloud] {name} created ({plan['machine']}{' + ' + gpu if gpu else ''}, {gc.zone}); deletes itself after {hours:.2f}h, at most ${max_cents / 100:.2f}"
        )

        for i in range(
            40
        ):  # SSH (and, on GPU images, the driver install) comes up a little after the machine
            if gc.dry_run or gc.ssh(name, "true", allow_fail=True)[0] == 0:
                break
            if i == 39:
                raise RuntimeError(f"can't reach {name} over SSH")
            time.sleep(15)

        tarball = work / "repo.tgz"
        print("[cloud] packing the checkout")
        if not gc.dry_run:
            pack_checkout(tarball)
        gc.scp(str(tarball), f"{name}:repo.tgz")
        print("[cloud] installing uv and the project's dependencies")
        gc.ssh(name, setup_script(bool(gpu)), echo=True)
        for u in plan["uploads"]:
            parent = posixpath.dirname(u["remote"])
            gc.ssh(name, f'mkdir -p "$HOME/{parent}"')
            gc.scp(u["local"], f"{name}:{parent}/")
        if plan["checkpoints"]:
            probe = f'echo probe | gcloud storage cp - "{plan["prefix"]}/.probe" --quiet || {{ echo "[checkpoint] can\'t write to {plan["prefix"]}" >&2; exit 1; }}'
            gc.ssh(
                name,
                f"set -e\n{probe}\n{sync_lines(plan['checkpoints'], plan['prefix'], 'restore')}",
                echo=True,
            )
        gc.ssh(
            name,
            start_script(
                plan["command"],
                sync_lines(plan["checkpoints"], plan["prefix"], "save"),
                cfg["checkpointMinutes"],
            ),
            echo=True,
        )
        print(f"[cloud] running: {plan['command']}")

        shown, misses, exit_code, warned = 0, 0, None, False
        while not gc.dry_run and exit_code is None:
            time.sleep(POLL_SECONDS)
            code, out = gc.ssh(
                name,
                f"tail -c +{shown + 1} ~/run.log; echo; echo \"@@exit=$(cat ~/run.exit 2>/dev/null)$( [ -f ~/checkpoint.failed ] && echo ' ckpt-failed')\"",
                allow_fail=True,
            )
            if code != 0:
                misses += 1
                if misses >= 3:
                    _, listed = gc.run(
                        [
                            "compute",
                            "instances",
                            "list",
                            f"--filter=name={name}",
                            "--format=value(name)",
                        ]
                    )
                    if not listed.strip():
                        created = False
                        raise Gone(f"{name} is gone (taken back by Google, or past its hours)")
                continue
            misses = 0
            cut = out.rfind("\n@@exit=")
            log, marker = out[:cut], out[cut + 8 :].strip()
            body = log.removesuffix("\n")
            if body:
                print(body)
            shown += len(body.encode())
            if "ckpt-failed" in marker and not warned:
                warned = True
                print(
                    f"[cloud] a checkpoint save to {plan['prefix']} failed; work since the last good save is at risk",
                    file=sys.stderr,
                )
            if marker.replace("ckpt-failed", "").strip():
                exit_code = int(marker.replace("ckpt-failed", "").strip())
        print(
            f"[cloud] command exited {exit_code if exit_code is not None else '(dry run)'} after {(time.time() - started) / 3600:.2f}h"
        )

        try:  # copy back whether the command passed or not: a failed run's partial output is still worth reading
            gc.scp(f"{name}:run.log", str(work / "run.log"))
        except RuntimeError:
            pass
        for d in plan["downloads"]:
            local = ROOT / d
            local.parent.mkdir(parents=True, exist_ok=True)
            gc.scp(f"{name}:repo/{d}", str(local.parent))
        return exit_code
    finally:
        if created:
            gc.run(
                ["compute", "instances", "delete", name, f"--zone={gc.zone}"],
                echo=True,
                allow_fail=True,
                mutating=True,
            )
            print(f"[cloud] {name} deleted; log in {work.relative_to(ROOT)}")
            if not gc.dry_run:
                record_machine(plan, name, started, time.time())


def record_machine(plan: dict, name: str, started: float, ended: float) -> None:
    hours = (ended - started) / 3600
    usd = round(plan["per_hour"] * hours, 2)
    entry = {
        "at": datetime.fromtimestamp(ended, UTC).isoformat(timespec="seconds"),
        "kind": "cloud",
        "what": f"{name}: {plan['command'][:120]}",
        "issue": plan["issue"],
        "usd": usd,
        "priced": "catalogue",
        "cloud": {
            "machine": plan["machine"],
            "gpu": plan["gpu"],
            "zone": plan["gc"].zone,
            "diskGb": plan["disk"],
            "hours": round(hours, 3),
            "usdPerHour": round(plan["per_hour"], 4),
        },
    }
    try:
        commit = costs.record_cost(entry)
        print(
            f"[cloud] {hours:.2f}h at ${plan['per_hour']:.3f}/h = ${usd:.2f}; in the cost ledger at {commit[:8]}"
        )
    except Exception as e:  # noqa: BLE001 - the run's work is already home; say how to fix the ledger
        print(
            f"[cloud] couldn't write the cost ledger ({e}): add it by hand with "
            f'`uv run costs add --kind cloud --what "{name}" --usd {usd} --issue {plan["issue"]}`',
            file=sys.stderr,
        )


# --- CLI -----------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sep = argv.index("--") if "--" in argv else len(argv)
    cfg = costs.load_config()
    p = argparse.ArgumentParser(
        prog="cloud-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--issue", required=True)
    p.add_argument(
        "--approved", required=True, help="link to the owner's comment approving this run"
    )
    p.add_argument("--hours", type=float, required=True)
    p.add_argument(
        "--gpu",
        choices=sorted(cfg["gpus"]),
        help="attach one GPU; picks its machine unless --machine",
    )
    p.add_argument("--machine")
    p.add_argument("--zone", default=cfg["zone"])
    p.add_argument("--disk", type=int, default=cfg["diskGb"])
    p.add_argument("--upload", action="append", default=[])
    p.add_argument("--download", action="append", default=[])
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--run", help="checkpoint run name; the same name resumes the same run")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv[:sep])
    command = " ".join(argv[sep + 1 :])
    if not command:
        p.error("give the command after --")
    if not re.fullmatch(r"https://github\.com/.+#issuecomment-\d+", a.approved):
        p.error(
            "--approved must link the owner's comment approving this run (https://github.com/...#issuecomment-...)"
        )
    if not 0 < a.hours <= cfg["maxHours"]:
        p.error(
            f"--hours must be over 0 and at most the {cfg['maxHours']}-hour cap in config/cloud.json"
        )
    checkpoints = [checkpoint_path(c) for c in a.checkpoint]
    if checkpoints and not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.run or ""):
        p.error(
            "--checkpoint needs --run <name> (lower case, digits, hyphens): the same name resumes the same run"
        )
    if shutil.which("gcloud") is None:
        raise SystemExit("[cloud] gcloud isn't installed")
    machine = a.machine or (cfg["gpus"][a.gpu]["machine"] if a.gpu else cfg["machine"])
    gc = Gcloud(cfg["project"], a.zone, a.dry_run, cfg["runnerKey"])
    region = a.zone.rsplit("-", 1)[0]

    # Quota: the trial project allows 24 vCPUs and no GPUs; a machine over quota fails only after a wait.
    _, region_out = gc.run(["compute", "regions", "describe", region, "--format=json"])
    _, project_out = gc.run(["compute", "project-info", "describe", "--format=json"])
    short = quota_shortfalls(
        machine,
        a.gpu,
        json.loads(region_out).get("quotas", []),
        json.loads(project_out).get("quotas", []),
    )
    if short:
        raise SystemExit(
            f"[cloud] not enough quota for {machine}{' + ' + a.gpu if a.gpu else ''} in {region}:\n  "
            + "\n  ".join(short)
            + "\nRaising it is the owner's call (GPUs need the billing account upgraded and a quota increase)."
        )

    # Budget: refuse anything that could take total spend past budgetUsd.
    per_hour = price_per_hour(gc, machine, a.disk, a.gpu)
    if per_hour is None:
        raise SystemExit(
            "[cloud] no price for this machine, so the budget can't be checked; not running"
        )
    spent = costs.spent(costs.ledger_on_main())
    reserved = reserved_by_running(gc, cfg["bucketPrefix"])
    cost = per_hour * a.hours
    why = budget_check(cfg["budgetUsd"], spent, reserved, cost)
    if why:
        raise SystemExit(f"[cloud] not running: {why}")
    print(
        f"[cloud] {machine}{' + ' + a.gpu if a.gpu else ''} at ${per_hour:.3f}/h; up to ${cost:.2f} for {a.hours}h; "
        f"${cfg['budgetUsd'] - spent - reserved - cost:.2f} of the budget left after it"
    )

    uploads = []
    for u in a.upload:
        local = Path(u).expanduser().resolve()
        if not local.exists():
            raise SystemExit(f"[cloud] --upload {u} doesn't exist")
        uploads.append({"local": str(local), "remote": remote_path(local)})
    prefix = f"gs://{cfg['bucket']}/{cfg['bucketPrefix']}-{a.issue}-{a.run}"
    plan = {
        "gc": gc,
        "config": cfg,
        "machine": machine,
        "gpu": a.gpu,
        "disk": a.disk,
        "issue": a.issue,
        "command": command,
        "uploads": uploads,
        "downloads": a.download,
        "checkpoints": checkpoints,
        "prefix": prefix,
        "per_hour": per_hour,
    }
    retries = a.retries if checkpoints else 0
    deadline = time.time() + a.hours * 3600
    exit_code = 0
    try:
        for attempt in range(retries + 1):
            name = f"{cfg['bucketPrefix']}-{a.issue}-{int(time.time() * 1000):x}"[:63]
            try:
                exit_code = run_once(plan, name, (deadline - time.time()) / 3600) or 0
                break
            except Gone as e:
                left = (deadline - time.time()) / 3600
                if attempt >= retries or left < 0.25:
                    raise
                print(
                    f"[cloud] {e}; starting again from the checkpoints ({retries - attempt} retries, {left:.2f}h left)"
                )
    finally:
        for c in checkpoints:  # the checkpoints come home even when every machine was lost
            (ROOT / c).mkdir(parents=True, exist_ok=True)
            code, _ = gc.run(
                ["storage", "rsync", "--recursive", f"{prefix}/{c}", str(ROOT / c)],
                echo=True,
                allow_fail=True,
                mutating=True,
            )
            if code != 0:
                print(
                    f"[cloud] copying checkpoints back from {prefix}/{c} failed; they stay in the bucket for 30 days",
                    file=sys.stderr,
                )
                exit_code = exit_code or 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
