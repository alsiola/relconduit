import json

import numpy as np
import pandas as pd
import pytest

from relconduit.entity_lgbm import (
    fit,
    gain_ranking,
    grid_configs,
    predict,
    records,
    select_columns,
)
from relconduit.results import append_record, make_record


def test_grid_sizes_and_objectives():
    reg, clf = grid_configs("regression"), grid_configs("binary_classification")
    assert len(reg) == 2 * len(clf)
    assert {c["objective"] for c in reg} == {"regression", "regression_l1"}
    assert {c["objective"] for c in clf} == {"binary"}
    assert {c["features"] for c in clf} == {"all", "no_hop2"}


def test_select_columns():
    cols = ["drivers__dob__days_before", "results__count__all", "results.constructorId>results__x"]
    assert select_columns(cols, "all") == cols
    assert select_columns(cols, "no_hop2") == cols[:2]
    with pytest.raises(ValueError):
        select_columns(cols, "nope")


def test_select_columns_pruned_sets():
    cols = ["a", "t.fk>u__x", "t.fk>>u__y", "b"]
    assert select_columns(cols, "no_hop2_all") == ["a", "t.fk>u__x", "b"]
    assert select_columns(cols, "top2", ranking=["b", "t.fk>>u__y", "a"]) == ["t.fk>>u__y", "b"]
    with pytest.raises(ValueError):
        select_columns(cols, "top2")


def test_select_columns_drop_sets():
    cols = ["a__x", "enc__city__te", "enc__city__count", "enc__name__emb0"]
    assert select_columns(cols, "drop:enc__") == ["a__x"]
    assert select_columns(cols, "drop:__te,__emb") == ["a__x", "enc__city__count"]


def test_gain_ranking_puts_the_signal_first():
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({"noise": rng.normal(size=n), "signal": rng.normal(size=n)})
    y = X["signal"].to_numpy() * 3 + 0.1 * rng.normal(size=n)
    cfg = grid_configs("regression")[0] | {"features": "all"}
    model = fit(X[:600], y[:600], X[600:], y[600:], cfg, seed=0)
    assert gain_ranking(model)[0] == "signal"
    pruned = fit(X[:600], y[:600], X[600:], y[600:], cfg | {"features": "top1"}, 0, ["signal"])
    assert pruned.feature_name() == ["signal"]


def test_fit_predict_binary_learns_signal():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"a": rng.normal(size=n), "b>c": rng.normal(size=n)})
    X["cat"] = pd.Categorical(rng.choice(["x", "y"], n), categories=["x", "y"])
    y = (X["a"] + 0.3 * rng.normal(size=n) > 0).astype(float).to_numpy()
    cfg = grid_configs("binary_classification")[0] | {"features": "no_hop2"}
    model = fit(X[:400], y[:400], X[400:], y[400:], cfg, seed=0)
    assert "b>c" not in model.feature_name()
    p = predict(model, X[400:])
    assert ((p > 0.5) == (y[400:] > 0.5)).mean() > 0.8


def test_records_and_append(tmp_path):
    summary = {
        "approach": "experiments/x/y/z",
        "metric": "roc_auc",
        "runtime_minutes": 1.234,
        "best_config": {"num_leaves": 15},
        "n_features_selected": 10,
        "val_scores": [0.7, 0.72, 0.71],
        "test_scores": {
            "upto_test_timestamp=True": [0.6, 0.62, 0.61],
            "upto_test_timestamp=False": [0.65, 0.66, 0.64],
        },
    }
    rows = records(summary, commit="abc123")
    assert [(r["split"], r["db"]) for r in rows] == [
        ("val", "upto_test_timestamp=True"),
        ("test", "upto_test_timestamp=True"),
        ("test", "upto_test_timestamp=False"),
    ]
    assert rows[1]["value"] == pytest.approx(0.61)
    assert rows[1]["std"] == pytest.approx(0.01)
    assert rows[1]["seeds"] == 3
    path = tmp_path / "r" / "t.jsonl"
    for r in rows:
        append_record(path, r)
    append_record(path, rows[0])
    lines = path.read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0]) == rows[0]


def test_make_record_validates():
    with pytest.raises(ValueError):
        make_record(
            approach="a",
            metric="m",
            split="train",
            values=[1.0],
            runtime_minutes=0,
            notes="",
            commit="c",
        )
    with pytest.raises(ValueError):
        append_record("/tmp/never.jsonl", {"value": 1})
