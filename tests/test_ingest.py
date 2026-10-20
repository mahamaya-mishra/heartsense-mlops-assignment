"""Data acquisition and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartsense import settings as cfg
from heartsense.ingest import DatasetIntegrityError, clean, fetch_raw, file_sha256, summary


def test_raw_checksum_matches_pin(raw_frame):
    assert file_sha256(cfg.RAW_DATA_PATH) == cfg.DATASET_SHA256


def test_fetch_raw_rejects_tampered_file(tmp_path):
    bad = tmp_path / "processed.cleveland.data"
    bad.write_text("1,2,3\n")
    with pytest.raises(DatasetIntegrityError):
        fetch_raw(bad)


def test_raw_shape_and_columns(raw_frame):
    assert raw_frame.shape == (303, 14)
    assert list(raw_frame.columns) == list(cfg.RAW_TO_CLEAN_NAMES.values())


def test_clean_output_schema(clean_frame):
    assert list(clean_frame.columns) == [*cfg.INPUT_FEATURES, cfg.TARGET]
    assert len(clean_frame) == 303


def test_target_is_binary_and_derived_from_diagnosis(raw_frame, clean_frame):
    assert set(clean_frame[cfg.TARGET].unique()) == {0, 1}
    expected = (raw_frame["diagnosis"] > 0).astype(int)
    assert (clean_frame[cfg.TARGET].to_numpy() == expected.to_numpy()).all()
    assert clean_frame[cfg.TARGET].sum() == 139


def test_question_marks_become_missing(clean_frame):
    assert int(clean_frame.isna().sum().sum()) == 6
    assert clean_frame["major_vessels"].isna().sum() == 4
    assert clean_frame["thalassemia"].isna().sum() == 2


def test_invalid_codes_are_nulled():
    row = {c: 1 for c in cfg.INPUT_FEATURES}
    row.update(
        {
            "age": 50,
            "resting_bp": 120,
            "cholesterol": 200,
            "max_heart_rate": 150,
            "st_depression": 1.0,
            "thalassemia": 5,
            "chest_pain_type": 9,
            "diagnosis": 2,
        }
    )
    out = clean(pd.DataFrame([row]))
    assert pd.isna(out.loc[0, "thalassemia"])
    assert pd.isna(out.loc[0, "chest_pain_type"])
    assert out.loc[0, cfg.TARGET] == 1


def test_implausible_measurements_are_nulled():
    row = {c: 1 for c in cfg.INPUT_FEATURES}
    row.update(
        {
            "age": 50,
            "resting_bp": 999,
            "cholesterol": 10,
            "max_heart_rate": 150,
            "st_depression": 1.0,
            "thalassemia": 3,
            "chest_pain_type": 2,
            "st_slope": 1,
            "rest_ecg": 0,
            "diagnosis": 0,
        }
    )
    out = clean(pd.DataFrame([row]))
    assert pd.isna(out.loc[0, "resting_bp"]) and pd.isna(out.loc[0, "cholesterol"])


def test_clean_is_pure(raw_frame):
    before = raw_frame.copy()
    clean(raw_frame)
    pd.testing.assert_frame_equal(before, raw_frame)


def test_duplicates_are_dropped(raw_frame):
    doubled = pd.concat([raw_frame, raw_frame.head(5)], ignore_index=True)
    assert len(clean(doubled)) == 303


def test_summary_values(clean_frame):
    s = summary(clean_frame)
    assert s["rows"] == 303 and s["missing_cells"] == 6
    assert np.isclose(s["positive_rate"], 139 / 303, atol=1e-4)


def test_codebook_covers_every_discrete_feature():
    for col in cfg.BINARY_FEATURES + cfg.NOMINAL_FEATURES + cfg.COUNT_FEATURES:
        assert col in cfg.CODEBOOK


def test_risk_tier_boundaries():
    assert cfg.risk_tier(0.0) == "low"
    assert cfg.risk_tier(0.249) == "low"
    assert cfg.risk_tier(0.25) == "moderate"
    assert cfg.risk_tier(0.5) == "elevated"
    assert cfg.risk_tier(0.75) == "high"
    assert cfg.risk_tier(1.0) == "high"
