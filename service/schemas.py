"""Request/response contracts of the prediction API (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    """One patient's clinical measurements. Codes follow the UCI Heart Disease data dictionary."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "age": 61,
                    "sex": 1,
                    "chest_pain_type": 4,
                    "resting_bp": 148,
                    "cholesterol": 203,
                    "fasting_blood_sugar": 0,
                    "rest_ecg": 0,
                    "max_heart_rate": 161,
                    "exercise_angina": 0,
                    "st_depression": 0.0,
                    "st_slope": 1,
                    "major_vessels": 1,
                    "thalassemia": 7,
                }
            ]
        },
    )

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    chest_pain_type: Literal[1, 2, 3, 4] = Field(
        ..., description="1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic"
    )
    resting_bp: float = Field(
        ..., ge=70, le=250, description="Resting systolic blood pressure (mm Hg)"
    )
    cholesterol: float = Field(..., ge=90, le=650, description="Serum cholesterol (mg/dl)")
    fasting_blood_sugar: Literal[0, 1] = Field(
        ..., description="1 if fasting blood sugar > 120 mg/dl"
    )
    rest_ecg: Literal[0, 1, 2] = Field(
        ..., description="0 normal, 1 ST-T abnormality, 2 LV hypertrophy"
    )
    max_heart_rate: float = Field(
        ..., ge=50, le=230, description="Maximum heart rate achieved (bpm)"
    )
    exercise_angina: Literal[0, 1] = Field(..., description="1 if exercise induced angina")
    st_depression: float = Field(
        ..., ge=0, le=8, description="ST depression induced by exercise (mm)"
    )
    st_slope: Literal[1, 2, 3] = Field(..., description="1 upsloping, 2 flat, 3 downsloping")
    major_vessels: Literal[0, 1, 2, 3] | None = Field(
        None, description="Major vessels coloured by fluoroscopy; omit if unknown (imputed)"
    )
    thalassemia: Literal[3, 6, 7] | None = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect; omit if unknown (imputed)"
    )


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients: list[PatientRecord] = Field(..., min_length=1, max_length=200)


class Prediction(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: Literal["disease", "no_disease"]
    risk_score: float = Field(..., ge=0, le=1, description="Predicted probability of heart disease")
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class")
    risk_tier: Literal["low", "moderate", "elevated", "high"]


class PredictionResponse(Prediction):
    model_version: str
    request_id: str
    latency_ms: float


class BatchResponse(BaseModel):
    model_version: str
    request_id: str
    count: int
    results: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str | None = None
    detail: str | None = None
