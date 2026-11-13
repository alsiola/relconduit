import json
import subprocess
from pathlib import Path

import pytest

from relconduit import costs


def sh(cwd: Path, *args: str) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path: Path, monkeypatch):
    """A bare origin with a ledger on main, and a clone of it standing in for this checkout."""
    origin, seed, clone = tmp_path / "origin.git", tmp_path / "seed", tmp_path / "clone"
    sh(tmp_path, "git", "init", "-q", "--bare", "-b", "main", str(origin))
    sh(tmp_path, "git", "clone", "-q", str(origin), str(seed))
    for k, v in {"user.name": "t", "user.email": "t@t"}.items():
        sh(seed, "git", "config", k, v)
    (seed / "costs").mkdir()
    (seed / "costs" / "ledger.json").write_text(
        costs.add_entry("", {"at": "2026-01-01", "usd": 1.0})
    )
    for args in [
        ("git", "add", "."),
        ("git", "commit", "-q", "-m", "seed"),
        ("git", "push", "-q", "origin", "HEAD:main"),
    ]:
        sh(seed, *args)
    sh(tmp_path, "git", "clone", "-q", str(origin), str(clone))
    for k, v in {"user.name": "t", "user.email": "t@t"}.items():
        sh(clone, "git", "config", k, v)
    monkeypatch.setattr(costs, "ROOT", clone)
    return origin, clone


def entries(origin: Path) -> list[dict]:
    return json.loads(sh(origin, "git", "show", f"main:{costs.LEDGER}"))["entries"]


def test_record_cost_appends_on_main(repos):
    origin, _ = repos
    costs.record_cost({"at": "2026-02-01", "what": "m", "usd": 2.0}, sleep=lambda s: None)
    assert [e["usd"] for e in entries(origin)] == [1.0, 2.0]


def test_push_retries_and_rebuilds_on_what_won_the_race(repos, monkeypatch):
    origin, _ = repos
    real_git, pushes = costs.git, []

    def git(*args, **kw):
        if args[0] == "push" and not pushes:
            pushes.append(1)  # another worktree's entry lands first, and this push fails
            costs.ROOT, here = origin.parent / "seed", costs.ROOT
            try:
                costs.record_cost(
                    {"at": "2026-03-01", "what": "other", "usd": 3.0}, sleep=lambda s: None
                )
            finally:
                costs.ROOT = here
            raise RuntimeError("git push failed: rejected")
        return real_git(*args, **kw)

    monkeypatch.setattr(costs, "git", git)
    costs.record_cost({"at": "2026-02-01", "what": "mine", "usd": 2.0}, sleep=lambda s: None)
    assert sorted(e["usd"] for e in entries(origin)) == [1.0, 2.0, 3.0]


def test_push_gives_up_with_the_reason(repos, monkeypatch):
    real_git = costs.git

    def git(*args, **kw):
        if args[0] == "push":
            raise RuntimeError("git push failed: could not read Username")
        return real_git(*args, **kw)

    monkeypatch.setattr(costs, "git", git)
    with pytest.raises(RuntimeError, match="after 3 tries.*could not read Username"):
        costs.push_to_main(costs.LEDGER, "{}", "m", attempts=3, sleep=lambda s: None)


def test_a_push_that_landed_but_reported_failure_is_not_recorded_twice(repos, monkeypatch):
    origin, _ = repos
    real_git, pushes = costs.git, []

    def git(*args, **kw):
        out = real_git(*args, **kw)
        if args[0] == "push" and not pushes:
            pushes.append(
                1
            )  # the server took it, then the connection dropped before git heard back
            raise RuntimeError("git push failed: Connection reset by peer")
        return out

    monkeypatch.setattr(costs, "git", git)
    commit = costs.record_cost(
        {"at": "2026-02-01", "what": "mine", "usd": 2.0}, sleep=lambda s: None
    )
    assert [e["usd"] for e in entries(origin)] == [1.0, 2.0]
    assert commit == sh(origin, "git", "rev-parse", "main")


def test_add_entry_is_idempotent():
    once = costs.add_entry("", {"at": "2026-02-01", "what": "m", "usd": 2.0})
    assert costs.add_entry(once, {"at": "2026-02-01", "what": "m", "usd": 2.0}) == once
