"""HTTP contract of the prediction service."""

from __future__ import annotations

import json

import pytest


def test_root_points_to_docs(api_client):
    body = api_client.get("/").json()
    assert body["docs"] == "/docs"


def test_healthz(api_client):
    r = api_client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "X-Request-ID" in r.headers


def test_readyz_reports_model_version(api_client):
    body = api_client.get("/readyz").json()
    assert body == {"ready": True, "model_version": "test-v0", "detail": None}


def test_metadata_endpoint(api_client):
    body = api_client.get("/metadata").json()
    assert body["family"] == "logistic_regression"


def test_predict_high_risk(api_client, high_risk_patient):
    r = api_client.post("/predict", json=high_risk_patient, headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "disease" and body["prediction"] == 1
    assert body["request_id"] == "abc123"
    assert 0 <= body["risk_score"] <= 1 and body["confidence"] >= 0.5
    assert body["risk_tier"] in {"elevated", "high"}


def test_predict_low_risk(api_client, low_risk_patient):
    body = api_client.post("/predict", json=low_risk_patient).json()
    assert body["label"] == "no_disease" and body["prediction"] == 0


def test_predict_optional_fields_can_be_omitted(api_client, high_risk_patient):
    rec = {k: v for k, v in high_risk_patient.items() if k not in ("major_vessels", "thalassemia")}
    assert api_client.post("/predict", json=rec).status_code == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 5),
        ("age", 150),
        ("sex", 2),
        ("chest_pain_type", 0),
        ("resting_bp", 20),
        ("cholesterol", 9999),
        ("thalassemia", 4),
        ("st_slope", "flat"),
        ("max_heart_rate", -1),
    ],
)
def test_predict_rejects_out_of_range(api_client, high_risk_patient, field, value):
    rec = {**high_risk_patient, field: value}
    r = api_client.post("/predict", json=rec)
    assert r.status_code == 422
    assert "request_id" in r.json()


def test_predict_rejects_unknown_field_and_missing_required(api_client, high_risk_patient):
    assert api_client.post("/predict", json={**high_risk_patient, "smoker": 1}).status_code == 422
    rec = dict(high_risk_patient)
    rec.pop("age")
    assert api_client.post("/predict", json=rec).status_code == 422


def test_batch_predict(api_client, high_risk_patient, low_risk_patient):
    r = api_client.post("/predict/batch", json={"patients": [high_risk_patient, low_risk_patient]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["results"]) == 2
    assert body["results"][0]["risk_score"] > body["results"][1]["risk_score"]


def test_batch_rejects_empty(api_client):
    assert api_client.post("/predict/batch", json={"patients": []}).status_code == 422


def test_metrics_exposes_prediction_counters(api_client, high_risk_patient):
    api_client.post("/predict", json=high_risk_patient)
    api_client.post("/predict", json={"age": 1})
    text = api_client.get("/metrics").text
    assert "heartsense_predictions_total" in text
    assert "heartsense_http_requests_total" in text
    assert "heartsense_validation_errors_total" in text
    assert "heartsense_model_ready 1.0" in text


def test_access_log_is_json_without_payload(api_client, high_risk_patient, caplog):
    from service.telemetry import JsonFormatter

    with caplog.at_level("INFO", logger="heartsense-api"):
        api_client.post("/predict", json=high_risk_patient)
    lines = [JsonFormatter().format(r) for r in caplog.records if r.name == "heartsense-api"]
    assert lines, "expected log records from the service logger"
    events = []
    for ln in lines:
        rec = json.loads(ln)  # every line is valid JSON
        events.append(rec.get("event"))
        assert "cholesterol" not in ln and "290" not in ln  # no patient data in logs
    assert "prediction" in events and "http_request" in events
    http = next(json.loads(ln) for ln in lines if json.loads(ln).get("event") == "http_request")
    assert http["path"] == "/predict" and http["status"] == 200 and "latency_ms" in http


def test_readyz_503_when_model_missing(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_FILE", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("METADATA_FILE", str(tmp_path / "missing.json"))
    from service.app import app

    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 503
        assert client.get("/healthz").json()["status"] == "degraded"
        assert client.post("/predict", json={"age": 50}).status_code == 422  # validation runs first
