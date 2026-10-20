"""Offline inference with the exported pipeline.

python -m heartsense.predict --input examples/patient_high_risk.json
python -m heartsense.predict --input examples/batch_request.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from heartsense import settings as cfg


def load_artifacts(model_file: Path = cfg.MODEL_FILE, metadata_file: Path = cfg.METADATA_FILE):
    model = joblib.load(model_file)
    metadata = json.loads(Path(metadata_file).read_text()) if Path(metadata_file).exists() else {}
    return model, metadata


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Build the model input frame; unknown keys are ignored, missing optional ones become NaN."""
    frame = pd.DataFrame.from_records(records)
    for col in cfg.INPUT_FEATURES:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[cfg.INPUT_FEATURES].astype("float64")


def score_records(
    model, records: list[dict], threshold: float = cfg.DECISION_THRESHOLD
) -> list[dict]:
    probs = model.predict_proba(records_to_frame(records))[:, 1]
    out = []
    for p in probs:
        p = float(p)
        positive = p >= threshold
        out.append(
            {
                "prediction": int(positive),
                "label": "disease" if positive else "no_disease",
                "risk_score": round(p, 4),
                "confidence": round(p if positive else 1 - p, 4),
                "risk_tier": cfg.risk_tier(p),
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path, help="JSON object or list of objects")
    parser.add_argument("--model", type=Path, default=cfg.MODEL_FILE)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text())
    records = (
        payload["patients"] if isinstance(payload, dict) and "patients" in payload else payload
    )
    records = records if isinstance(records, list) else [records]
    model, metadata = load_artifacts(args.model)
    results = score_records(
        model, records, metadata.get("decision_threshold", cfg.DECISION_THRESHOLD)
    )
    print(
        json.dumps(
            {"model_version": metadata.get("model_version", "unknown"), "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
