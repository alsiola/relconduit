import random

import pytest

from relconduit.retry import backoff_delays, retry


def test_backoff_delays_grow_with_jitter_and_cap():
    d = backoff_delays(6, base=4.0, cap=20.0, rng=random.Random(0))
    assert len(d) == 5
    for i, w in enumerate(d):
        top = min(20.0, 4.0 * 2**i)
        assert top / 2 <= w <= top
    assert backoff_delays(1, 4.0, 20.0) == []


def test_retry_succeeds_after_failures():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("blip")
        return "ok"

    assert retry(flaky, attempts=5, sleep=waits.append) == "ok"
    assert len(calls) == 3 and len(waits) == 2


def test_retry_reraises_the_last_error():
    waits = []

    def broken():
        raise RuntimeError(f"fail {len(waits)}")

    with pytest.raises(RuntimeError, match="fail 2"):
        retry(broken, attempts=3, sleep=waits.append)
    assert len(waits) == 2
