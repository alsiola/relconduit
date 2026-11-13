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
