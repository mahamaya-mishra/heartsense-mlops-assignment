"""Feature engineering and the preprocessing pipeline shared by training and serving.

The whole pipeline (engineered features -> imputation -> encoding/scaling -> classifier) is one
scikit-learn ``Pipeline`` object, so the exact transformations fitted at training time are
replayed at prediction time. No preprocessing lives in the API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartsense import settings as cfg

ENGINEERED_FEATURES = ["hr_reserve_pct", "st_burden", "risk_factor_count"]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive three interpretable cardiology features from the raw measurements.

    * ``hr_reserve_pct``: achieved share of the age-predicted maximum heart rate
      (``max_heart_rate / (220 - age)``). Low values suggest chronotropic incompetence.
    * ``st_burden``: exercise ST depression weighted by slope (flat x1.5, downsloping x2),
      because a depressed *and* non-upsloping ST segment is the classic ischaemia pattern.
    * ``risk_factor_count``: number of conventional risk flags present
      (age >= 55, cholesterol > 240, resting BP >= 140, high fasting sugar, exercise angina).
    """

    SLOPE_WEIGHTS = {1: 1.0, 2: 1.5, 3: 2.0}

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        num = {c: pd.to_numeric(df[c], errors="coerce").astype("float64") for c in df.columns}
        predicted_max = 220.0 - num["age"]
        df["hr_reserve_pct"] = num["max_heart_rate"] / predicted_max.where(predicted_max > 0)
        weights = num["st_slope"].map(self.SLOPE_WEIGHTS)
        df["st_burden"] = num["st_depression"] * weights
        flags = pd.concat(
            [
                num["age"] >= 55,
                num["cholesterol"] > 240,
                num["resting_bp"] >= 140,
                num["fasting_blood_sugar"] == 1,
                num["exercise_angina"] == 1,
            ],
            axis=1,
        )
        df["risk_factor_count"] = flags.sum(axis=1).astype("float64")
        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray([*base, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Column-wise preprocessing. Imputers are fitted on training data only."""
    scaled = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    nominal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("scaled", scaled, cfg.CONTINUOUS_FEATURES + cfg.COUNT_FEATURES + ENGINEERED_FEATURES),
            ("binary", binary, cfg.BINARY_FEATURES),
            ("nominal", nominal, cfg.NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Full inference pipeline: engineer -> preprocess -> classifier."""
    return Pipeline(
        [
            ("engineer", ClinicalFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in cfg.INPUT_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"dataset is missing feature columns: {missing}")
    return df[cfg.INPUT_FEATURES].copy(), df[cfg.TARGET].astype(int)


def output_feature_names(fitted: Pipeline) -> list[str]:
    """Names of the columns that reach the classifier of a fitted pipeline."""
    return list(fitted.named_steps["preprocess"].get_feature_names_out())
