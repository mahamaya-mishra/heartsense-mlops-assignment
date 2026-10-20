"""Project-wide settings: paths, dataset schema, feature groups and tracking configuration.

Everything that another module might need to agree on lives here so that training, the
prediction service and the tests never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "heart_disease_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_STORE = PROJECT_ROOT / "model_store"
MODEL_FILE = MODEL_STORE / "heartsense_pipeline.joblib"
METADATA_FILE = MODEL_STORE / "metadata.json"

# ----------------------------------------------------------------------- data source
# UCI Machine Learning Repository, dataset 45 "Heart Disease" (Cleveland clinic subset).
# Licence: CC BY 4.0. Janosi, Steinbrunn, Pfisterer & Detrano (1988).
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
DATASET_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
DATASET_ARCHIVE_MEMBER = "processed.cleveland.data"
DATASET_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

# --------------------------------------------------------------------------- schema
# The raw file has no header. We rename the terse UCI column names to descriptive ones
# so the API, notebooks and report all speak the same language.
RAW_TO_CLEAN_NAMES: dict[str, str] = {
    "age": "age",
    "sex": "sex",
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "rest_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "ca": "major_vessels",
    "thal": "thalassemia",
    "num": "diagnosis",
}
RAW_COLUMNS = list(RAW_TO_CLEAN_NAMES)
TARGET = "target"  # 1 = heart disease present (diagnosis > 0), 0 = absent

CONTINUOUS_FEATURES = ["age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression"]
COUNT_FEATURES = ["major_vessels"]
BINARY_FEATURES = ["sex", "fasting_blood_sugar", "exercise_angina"]
NOMINAL_FEATURES = ["chest_pain_type", "rest_ecg", "st_slope", "thalassemia"]
INPUT_FEATURES = CONTINUOUS_FEATURES + COUNT_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES

# Allowed codes for discrete fields (from the UCI data dictionary).
CODEBOOK: dict[str, dict[int, str]] = {
    "sex": {0: "female", 1: "male"},
    "chest_pain_type": {
        1: "typical angina",
        2: "atypical angina",
        3: "non-anginal",
        4: "asymptomatic",
    },
    "fasting_blood_sugar": {0: "<= 120 mg/dl", 1: "> 120 mg/dl"},
    "rest_ecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exercise_angina": {0: "no", 1: "yes"},
    "st_slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "major_vessels": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thalassemia": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}

# Clinically plausible bounds; anything outside is treated as a recording error -> NaN.
PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "resting_bp": (70, 250),
    "cholesterol": (90, 650),
    "max_heart_rate": (50, 230),
    "st_depression": (0, 8),
}

# ------------------------------------------------------------------- reproducibility
SEED = 2025
TEST_FRACTION = 0.2
CV_SPLITS = 5
CV_REPEATS = 2
DECISION_THRESHOLD = 0.5
RISK_TIERS: list[tuple[float, str]] = [  # upper bound (exclusive) -> tier name
    (0.25, "low"),
    (0.50, "moderate"),
    (0.75, "elevated"),
    (1.01, "high"),
]

# ------------------------------------------------------------------------- tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "heartsense"
MLFLOW_REGISTERED_MODEL = "heartsense-heart-disease"
MLFLOW_PRODUCTION_ALIAS = "production"


def risk_tier(score: float) -> str:
    """Map a disease probability to a human-readable risk tier."""
    for upper, name in RISK_TIERS:
        if score < upper:
            return name
    return RISK_TIERS[-1][1]
