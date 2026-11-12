import relbench
from relbench import submit  # noqa: F401  (the leaderboard evaluation path must import)


def test_relbench_imports():
    assert relbench.__version__ >= "3"
    assert callable(relbench.load_dataset)
