from relconduit.results import git_commit


def test_git_commit_env_override(monkeypatch):
    monkeypatch.setenv("RELCONDUIT_COMMIT", "abc1234")
    assert git_commit() == "abc1234"


def test_git_commit_from_repo(monkeypatch):
    monkeypatch.delenv("RELCONDUIT_COMMIT", raising=False)
    sha = git_commit()
    assert len(sha.split("-")[0]) >= 7
