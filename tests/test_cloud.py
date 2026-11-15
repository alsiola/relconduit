import time
from pathlib import Path

import pytest

from relconduit import cloud, costs


def test_vcpus():
    assert cloud.vcpus("n2-custom-24-24576") == 24
    assert cloud.vcpus("g2-standard-8") == 8
    with pytest.raises(ValueError):
        cloud.vcpus("weird")


def test_remote_path(tmp_path: Path):
    root, home = tmp_path / "home" / "repo", tmp_path / "home"
    (root / "data").mkdir(parents=True)
    assert cloud.remote_path(root / "data", root, home) == "repo/data"
    assert cloud.remote_path(home / ".cache" / "x", root, home) == ".cache/x"
    with pytest.raises(ValueError):
        cloud.remote_path(tmp_path / "elsewhere", root, home)


def test_checkpoint_path():
    assert cloud.checkpoint_path("runs/sweep/") == "runs/sweep"
    for bad in ["", ".", "../x", "/abs"]:
        with pytest.raises(ValueError):
            cloud.checkpoint_path(bad)


def q(metric, limit, usage=0):
    return {"metric": metric, "limit": limit, "usage": usage}


def test_quota_trial_project_has_no_gpus():
    project = [q("CPUS_ALL_REGIONS", 24), q("GPUS_ALL_REGIONS", 0)]
    region = [q("N2_CPUS", 200), q("G2_CPUS", 0), q("PREEMPTIBLE_NVIDIA_L4_GPUS", 1)]
    assert cloud.quota_shortfalls("n2-custom-24-24576", None, region, project) == []
    assert cloud.quota_shortfalls("n2-custom-32-32768", None, region, project)[0].startswith(
        "CPUS_ALL_REGIONS"
    )
    short = cloud.quota_shortfalls("g2-standard-8", "nvidia-l4", region, project)
    assert any(s.startswith("GPUS_ALL_REGIONS") for s in short)
    assert cloud.gpu_quota_metric("nvidia-tesla-t4") == "NVIDIA_T4_GPUS"


def test_budget_check():
    assert cloud.budget_check(50, 10, 5, 30) is None
    assert "only $5.00" in cloud.budget_check(50, 40, 5, 6)


def sku(desc, price):
    units, nanos = int(price), round((price - int(price)) * 1e9)
    return {
        "description": desc,
        "pricingInfo": [
            {
                "pricingExpression": {
                    "tieredRates": [{"unitPrice": {"units": str(units), "nanos": nanos}}]
                }
            }
        ],
    }


SKUS = [
    sku("Spot Preemptible N2 Custom Instance Core running in London", 0.00893),
    sku("Spot Preemptible N2 Custom Instance Ram running in London", 0.0012),
    sku(
        "Spot Preemptible N2 Instance Core running in London", 0.5
    ),  # predefined, not custom: ignored
    sku("Spot Preemptible G2 Instance Core running in London", 0.01707),
    sku("Spot Preemptible G2 Instance Ram running in London", 0.002),
    sku("Nvidia L4 GPU attached to Spot Preemptible VMs running in London", 0.3826),
    sku("Balanced PD Capacity in London", 0.12),
]


def test_spot_price():
    p = costs.spot_price_per_hour(SKUS, "n2-custom-24-24576", "London", 24, 24, 100)
    assert p == pytest.approx(24 * 0.00893 + 24 * 0.0012 + 100 * 0.12 / 730)
    g = costs.spot_price_per_hour(SKUS, "g2-standard-8", "London", 8, 32, 100, "nvidia-l4")
    assert g == pytest.approx(8 * 0.01707 + 32 * 0.002 + 100 * 0.12 / 730 + 0.3826)
    assert costs.spot_price_per_hour(SKUS, "c2d-highcpu-16", "London", 16, 32, 100) is None
    assert (
        costs.spot_price_per_hour(SKUS, "g2-standard-8", "London", 8, 32, 100, "nvidia-tesla-t4")
        is None
    )


def test_spent_ignores_unpriced():
    assert costs.spent({"entries": [{"usd": 1.234}, {"usd": None}, {"usd": 2}]}) == 3.23


def test_start_script_exports_commit():
    # The machine has no .git, so the run's results rows get the commit from the environment.
    script = cloud.start_script(
        "uv run python x.py", save="", checkpoint_minutes=10, commit="abc1234"
    )
    assert "export RELCONDUIT_COMMIT=abc1234\n" in script
    assert script.index("RELCONDUIT_COMMIT") < script.index("uv run python x.py")
    assert "RELCONDUIT_COMMIT" not in cloud.start_script("x", save="", checkpoint_minutes=10)


def test_finish_keeps_the_command_result_apart_from_bookkeeping(capsys):
    assert cloud.finish(0, []) == 0
    assert cloud.finish(2, []) == 2
    assert cloud.finish(0, ["ledger push failed; `uv run costs add ...`"]) == cloud.POST_RUN_FAILED
    err = capsys.readouterr().err
    assert "the command succeeded" in err and "uv run costs add" in err
    assert cloud.finish(2, ["checkpoints didn't copy back"]) == 2


def test_sync_lines_skip_gcloud_temp_files():
    for direction in ("save", "restore"):
        assert "gstmp" in cloud.sync_lines(["runs/x"], "gs://b/p", direction)


def test_remove_partials(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.npy").write_bytes(b"1")
    (tmp_path / "a" / "x.npy_.gstmp").write_bytes(b"1")
    (tmp_path / "y.gstmp").write_bytes(b"1")
    assert cloud.remove_partials(tmp_path) == 2
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["x.npy"]
    assert cloud.remove_partials(tmp_path / "missing") == 0


def test_stream_stops_a_hung_command_and_its_children():
    start = time.monotonic()
    # the shell starts a child that would hold the output pipe open; killing the group must reach it too
    code, out, _, timed_out = cloud.stream(
        ["bash", "-c", "echo started; sleep 30 & wait"], echo=False, timeout=1
    )
    assert timed_out and code != 0 and "started" in out
    assert time.monotonic() - start < 10


def test_stream_without_a_hang():
    code, out, err, timed_out = cloud.stream(["bash", "-c", "cat; echo oops >&2"], input="hi")
    assert (code, out, err.strip(), timed_out) == (0, "hi", "oops", False)


def test_redo_copy_back_runs_as_the_runner_one_file_at_a_time():
    cmd = cloud.redo_copy_back("~/key.json", "gs://b/p/runs/x", "runs/x")
    assert "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=~/key.json" in cmd
    assert "CLOUDSDK_STORAGE_PROCESS_COUNT=1" in cmd and "CLOUDSDK_STORAGE_THREAD_COUNT=1" in cmd
    assert cmd.endswith("gs://b/p/runs/x runs/x") and "gstmp" in cmd


def test_stream_drains_a_chatty_stderr():
    # over a pipe's worth of stderr before any stdout: reading stderr only at the end would deadlock here
    code, out, err, timed_out = cloud.stream(
        ["bash", "-c", "head -c 300000 /dev/zero | tr '\\0' x >&2; echo done"], timeout=20
    )
    assert (code, out, timed_out) == (0, "done\n", False) and len(err) == 300000
