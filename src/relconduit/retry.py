"""Retrying a flaky step (a git push, a copy from Cloud Storage) with exponential backoff and jitter."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def backoff_delays(
    attempts: int, base: float, cap: float, rng: random.Random | None = None
) -> list[float]:
    """The waits between `attempts` tries: "equal jitter", each uniform in [base * 2**i / 2, base * 2**i], at most cap.

    Half the nominal wait is always kept, so a retry never fires straight back into whatever just failed.
    """
    rng = rng or random.Random()
    out = []
    for i in range(attempts - 1):
        top = min(cap, base * 2**i)
        out.append(rng.uniform(top / 2, top))
    return out


def retry(
    fn: Callable[[], T],
    attempts: int = 5,
    base: float = 2.0,
    cap: float = 60.0,
    what: str = "",
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Calls fn until it returns without raising, at most `attempts` times; re-raises the last error."""
    delays = backoff_delays(attempts, base, cap, rng)
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # anything fn raises counts as one failed try
            if i == attempts - 1:
                raise
            wait = delays[i]
            print(
                f"[retry] {what or 'step'} failed ({e}); try {i + 2} of {attempts} in {wait:.0f}s"
            )
            sleep(wait)
    raise AssertionError("unreachable")
