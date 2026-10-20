"""Feature engineering and preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartsense import settings as cfg
from heartsense.pipeline import (
    ENGINEERED_FEATURES,
    ClinicalFeatureEngineer,
    build_pipeline,
    build_preprocessor,
    output_feature_names,
    split_xy,
)


def test_engineer_adds_expected_columns(synthetic_frame):
    X, _ = split_xy(synthetic_frame)
    out = ClinicalFeatureEngineer().fit_transform(X)
    assert list(out.columns) == [*cfg.INPUT_FEATURES, *ENGINEERED_FEATURES]
    assert len(out) == len(X)


def test_engineer_formulas():
    X = pd.DataFrame(
        [
            {
                **{c: 0 for c in cfg.INPUT_FEATURES},
                "age": 60,
                "max_heart_rate": 120,
                "st_depression": 2.0,
                "st_slope": 3,
                "cholesterol": 250,
                "resting_bp": 145,
                "fasting_blood_sugar": 1,
                "exercise_angina": 1,
            }
        ]
    )
    out = ClinicalFeatureEngineer().fit_transform(X)
    assert out.loc[0, "hr_reserve_pct"] == pytest.approx(120 / 160)
    assert out.loc[0, "st_burden"] == pytest.approx(4.0)  # downsloping weight 2.0
    assert out.loc[0, "risk_factor_count"] == 5


def test_engineer_propagates_missing_without_crashing(synthetic_frame):
    X, _ = split_xy(synthetic_frame)
    X.loc[0, "max_heart_rate"] = np.nan
    X.loc[1, "st_slope"] = np.nan
    out = ClinicalFeatureEngineer().fit_transform(X)
    assert np.isnan(out.loc[0, "hr_reserve_pct"])
    assert np.isnan(out.loc[1, "st_burden"])


def test_engineer_does_not_mutate_input(synthetic_frame):
    X, _ = split_xy(synthetic_frame)
    snapshot = X.copy()
    ClinicalFeatureEngineer().fit_transform(X)
    pd.testing.assert_frame_equal(X, snapshot)


def test_preprocessor_output_is_dense_and_finite(synthetic_frame):
    X, _ = split_xy(synthetic_frame)
    engineered = ClinicalFeatureEngineer().fit_transform(X)
    Z = build_preprocessor().fit_transform(engineered)
    assert isinstance(Z, np.ndarray)
    assert np.isfinite(Z).all()
    # 6 scaled raw + 3 engineered + 3 binary + one-hot(4+3+3+3)
    assert Z.shape == (len(X), 6 + 3 + 3 + 13)


def test_pipeline_fits_and_predicts_probabilities(synthetic_frame):
    X, y = split_xy(synthetic_frame)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_handles_unseen_category_and_missing_at_inference(fitted_pipeline, xy):
    X, _ = xy
    row = X.head(1).astype("float64").copy()
    row["major_vessels"] = np.nan
    row["thalassemia"] = np.nan
    assert fitted_pipeline.predict_proba(row).shape == (1, 2)


def test_output_feature_names_match_classifier_input(fitted_pipeline):
    names = output_feature_names(fitted_pipeline)
    assert fitted_pipeline.named_steps["classifier"].coef_.shape[1] == len(names)
    assert "hr_reserve_pct" in names
    assert any(n.startswith("thalassemia_7") for n in names)


def test_split_xy_requires_all_features(synthetic_frame):
    with pytest.raises(KeyError):
        split_xy(synthetic_frame.drop(columns=["thalassemia"]))


def test_fitted_pipeline_is_reasonably_accurate(fitted_pipeline, xy):
    X, y = xy
    assert (fitted_pipeline.predict(X) == y).mean() > 0.8
