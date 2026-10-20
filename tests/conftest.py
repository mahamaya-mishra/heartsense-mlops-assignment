"""Shared fixtures: a cleaned dataset, a small fitted pipeline and an API client."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartsense import settings as cfg
from heartsense.ingest import clean, read_raw
from heartsense.pipeline import build_pipeline, split_xy

RAW = cfg.RAW_DATA_PATH


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    if not RAW.exists():
        pytest.skip("raw dataset not present; run `python -m heartsense.ingest`")
    return read_raw(RAW)


@pytest.fixture(scope="session")
def clean_frame(raw_frame) -> pd.DataFrame:
    return clean(raw_frame)


@pytest.fixture(scope="session")
def xy(clean_frame):
    return split_xy(clean_frame)


@pytest.fixture(scope="session")
def fitted_pipeline(xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=2000, random_state=cfg.SEED))
    return pipe.fit(X, y)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, fitted_pipeline) -> Path:
    """A throwaway model_store with a fitted pipeline and minimal metadata."""
    d = tmp_path_factory.mktemp("model_store")
    joblib.dump(fitted_pipeline, d / cfg.MODEL_FILE.name)
    (d / cfg.METADATA_FILE.name).write_text(
        json.dumps(
            {
                "model_version": "test-v0",
                "family": "logistic_regression",
                "decision_threshold": 0.5,
                "git_sha": "test",
                "input_features": cfg.INPUT_FEATURES,
            }
        )
    )
    return d


@pytest.fixture()
def api_client(model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_FILE", str(model_dir / cfg.MODEL_FILE.name))
    monkeypatch.setenv("METADATA_FILE", str(model_dir / cfg.METADATA_FILE.name))
    from service.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def high_risk_patient() -> dict:
    return json.loads((cfg.PROJECT_ROOT / "examples" / "patient_high_risk.json").read_text())


@pytest.fixture()
def low_risk_patient() -> dict:
    return json.loads((cfg.PROJECT_ROOT / "examples" / "patient_low_risk.json").read_text())


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    """A tiny hand-made frame with edge cases (missing values, extreme readings)."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 78, n),
            "resting_bp": rng.integers(95, 180, n).astype(float),
            "cholesterol": rng.integers(150, 400, n).astype(float),
            "max_heart_rate": rng.integers(90, 200, n).astype(float),
            "st_depression": rng.uniform(0, 4, n).round(1),
            "major_vessels": rng.integers(0, 4, n),
            "sex": rng.integers(0, 2, n),
            "fasting_blood_sugar": rng.integers(0, 2, n),
            "exercise_angina": rng.integers(0, 2, n),
            "chest_pain_type": rng.integers(1, 5, n),
            "rest_ecg": rng.integers(0, 3, n),
            "st_slope": rng.integers(1, 4, n),
            "thalassemia": rng.choice([3, 6, 7], n),
            "target": rng.integers(0, 2, n),
        }
    )
    df.loc[0, "major_vessels"] = np.nan
    df.loc[1, "thalassemia"] = np.nan
    return df


@pytest.fixture(autouse=True)
def _quiet_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
