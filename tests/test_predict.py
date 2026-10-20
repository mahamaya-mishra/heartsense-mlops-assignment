"""Offline prediction helpers."""

from __future__ import annotations

import json

import numpy as np

from heartsense import predict
from heartsense import settings as cfg


def test_records_to_frame_orders_and_fills(high_risk_patient):
    rec = dict(high_risk_patient)
    rec.pop("thalassemia")
    rec["unexpected"] = 42
    frame = predict.records_to_frame([rec])
    assert list(frame.columns) == cfg.INPUT_FEATURES
    assert np.isnan(frame.loc[0, "thalassemia"])
    assert "unexpected" not in frame.columns


def test_score_records_shape_and_fields(fitted_pipeline, high_risk_patient, low_risk_patient):
    out = predict.score_records(fitted_pipeline, [high_risk_patient, low_risk_patient])
    assert len(out) == 2
    for r in out:
        assert set(r) == {"prediction", "label", "risk_score", "confidence", "risk_tier"}
        assert r["confidence"] >= 0.5
        assert r["risk_tier"] == cfg.risk_tier(r["risk_score"])


def test_high_risk_scores_above_low_risk(fitted_pipeline, high_risk_patient, low_risk_patient):
    hi, lo = predict.score_records(fitted_pipeline, [high_risk_patient, low_risk_patient])
    assert hi["risk_score"] > lo["risk_score"]
    assert hi["label"] == "disease" and lo["label"] == "no_disease"


def test_threshold_is_respected(fitted_pipeline, low_risk_patient):
    strict = predict.score_records(fitted_pipeline, [low_risk_patient], threshold=0.01)[0]
    assert strict["prediction"] == 1


def test_cli_prints_json(model_dir, high_risk_patient, tmp_path, capsys):
    inp = tmp_path / "p.json"
    inp.write_text(json.dumps(high_risk_patient))
    predict.main(["--input", str(inp), "--model", str(model_dir / cfg.MODEL_FILE.name)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["label"] in {"disease", "no_disease"}
