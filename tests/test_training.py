"""Model selection logic and the training entry point."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from heartsense import settings as cfg
from heartsense import train
from heartsense.evaluate import classification_metrics, plot_leaderboard


def test_search_space_has_five_families_with_pipelines():
    space = train.search_space(quick=True)
    assert set(space) == {
        "logistic_regression",
        "random_forest",
        "svm_rbf",
        "hist_gradient_boosting",
        "knn",
    }
    for est, params, n_iter in space.values():
        assert n_iter >= 2
        assert all(k.startswith("classifier__") for k in params)
        assert hasattr(est, "fit")


def test_classification_metrics_perfect_and_prefix():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), prefix="t_")
    assert m["t_accuracy"] == 1.0 and m["t_roc_auc"] == 1.0 and m["t_false_negatives"] == 0
    assert set(m) == {
        f"t_{k}"
        for k in (
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
            "average_precision",
            "brier",
            "false_negatives",
            "false_positives",
        )
    }


def test_classification_metrics_threshold_changes_recall():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.4, 0.45, 0.9, 0.1])
    assert classification_metrics(y, s, threshold=0.5)["recall"] == 0.5
    assert classification_metrics(y, s, threshold=0.42)["recall"] == 1.0


def test_jsonable_strips_prefix_and_stringifies():
    out = train._jsonable(
        {"classifier__C": 1.5, "classifier__class_weight": None, "classifier__x": np.float64(2)}
    )
    assert out == {"C": 1.5, "class_weight": None, "x": 2.0}
    assert type(out["x"]) is float


def test_plot_leaderboard_runs():
    board = pd.DataFrame(
        {
            "model": ["a", "b"],
            "cv_roc_auc_mean": [0.9, 0.85],
            "cv_roc_auc_std": [0.01, 0.02],
            "test_roc_auc": [0.92, 0.88],
        }
    )
    fig = plot_leaderboard(board)
    assert fig is not None


@pytest.mark.slow
def test_quick_training_end_to_end(tmp_path, monkeypatch, clean_frame):
    """Runs the real training loop with tiny search spaces against a temporary MLflow store."""
    import mlflow

    data = tmp_path / "clean.csv"
    clean_frame.to_csv(data, index=False)
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(cfg, "MLFLOW_EXPERIMENT", "heartsense-test")
    monkeypatch.chdir(tmp_path)
    out = train.run(
        quick=True,
        data_path=data,
        model_store=tmp_path / "store",
        reports_dir=tmp_path / "reports",
        register=True,
    )

    assert (tmp_path / "store" / cfg.MODEL_FILE.name).exists()
    meta = json.loads((tmp_path / "store" / cfg.METADATA_FILE.name).read_text())
    assert meta["family"] == out["champion"]
    assert meta["metrics"]["cv_roc_auc_mean"] > 0.8
    assert (tmp_path / "reports" / "leaderboard.csv").exists()
    board = out["leaderboard"]
    assert len(board) == 5
    assert board["cv_roc_auc_mean"].is_monotonic_decreasing

    client = mlflow.MlflowClient(tracking_uri=cfg.MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name("heartsense-test")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) >= 5 + 1 + 2 * 5  # families + selection + at least 2 trials each
    mv = client.get_model_version_by_alias(cfg.MLFLOW_REGISTERED_MODEL, cfg.MLFLOW_PRODUCTION_ALIAS)
    assert str(mv.version) == "1"
