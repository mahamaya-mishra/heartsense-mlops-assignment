"""Acquire, verify and clean the UCI Heart Disease (Cleveland) dataset.

    python -m heartsense.ingest              # download if needed, verify SHA-256, clean, save
    python -m heartsense.ingest --refresh    # force re-download

The cleaned CSV uses descriptive column names and a binary ``target`` column.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from heartsense import settings as cfg

log = logging.getLogger(__name__)


class DatasetIntegrityError(RuntimeError):
    """Raised when the downloaded file does not match the pinned checksum."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _http_get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "heartsense-mlops/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_raw(destination: Path = cfg.RAW_DATA_PATH, refresh: bool = False) -> Path:
    """Download the raw data file (direct URL first, UCI zip archive as fallback)."""
    destination = Path(destination)
    if destination.exists() and not refresh:
        log.info("raw file present, skipping download: %s", destination)
    else:
        try:
            blob = _http_get(cfg.DATASET_URL)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("direct download failed (%s), falling back to zip archive", exc)
            with zipfile.ZipFile(io.BytesIO(_http_get(cfg.DATASET_ARCHIVE_URL))) as zf:
                blob = zf.read(cfg.DATASET_ARCHIVE_MEMBER)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        log.info("downloaded %d bytes -> %s", len(blob), destination)

    digest = file_sha256(destination)
    if digest != cfg.DATASET_SHA256:
        raise DatasetIntegrityError(
            f"SHA-256 mismatch for {destination.name}: expected {cfg.DATASET_SHA256}, got {digest}"
        )
    log.info("checksum verified (sha256=%s...)", digest[:12])
    return destination


def read_raw(path: Path = cfg.RAW_DATA_PATH) -> pd.DataFrame:
    """Parse the header-less raw file; '?' marks a missing value."""
    frame = pd.read_csv(path, header=None, names=cfg.RAW_COLUMNS, na_values=["?"])
    return frame.rename(columns=cfg.RAW_TO_CLEAN_NAMES)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules. Pure function: never mutates its input.

    1. Binarise the diagnosis (0 = no disease, 1-4 = disease) into ``target``.
    2. Coerce every column to numeric; invalid category codes become NaN.
    3. Null out continuous readings outside clinically plausible bounds.
    4. Drop exact duplicate rows.
    Missing values are *kept* as NaN: imputation belongs to the model pipeline so that it is
    fitted on training folds only.
    """
    df = frame.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cfg.TARGET] = (df["diagnosis"] > 0).astype("int64")
    df = df.drop(columns=["diagnosis"])

    for col, codes in cfg.CODEBOOK.items():
        df.loc[~df[col].isin(list(codes)) & df[col].notna(), col] = np.nan

    for col, (lo, hi) in cfg.PLAUSIBLE_BOUNDS.items():
        mask = (df[col] < lo) | (df[col] > hi)
        df.loc[mask, col] = np.nan

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if before != len(df):
        log.info("dropped %d duplicate rows", before - len(df))

    # Integer-valued columns that may hold NaN keep a nullable integer dtype.
    for col in cfg.BINARY_FEATURES + cfg.NOMINAL_FEATURES + cfg.COUNT_FEATURES + ["age"]:
        df[col] = df[col].astype("Int64")
    return df[[*cfg.INPUT_FEATURES, cfg.TARGET]]


def save_clean(df: pd.DataFrame, path: Path = cfg.CLEAN_DATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path = cfg.CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset (building it from the raw file if it does not exist)."""
    path = Path(path)
    if not path.exists():
        log.info("clean dataset missing, building it")
        save_clean(clean(read_raw(fetch_raw())), path)
    df = pd.read_csv(path)
    for col in cfg.BINARY_FEATURES + cfg.NOMINAL_FEATURES + cfg.COUNT_FEATURES + ["age"]:
        df[col] = df[col].astype("Int64")
    return df


def summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": int(df.shape[1]),
        "positive_rate": round(float(df[cfg.TARGET].mean()), 4),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--refresh", action="store_true", help="re-download the raw file")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    raw = fetch_raw(refresh=args.refresh)
    df = clean(read_raw(raw))
    out = save_clean(df)
    log.info("clean dataset saved -> %s", out)
    print(json.dumps(summary(df), indent=2))


if __name__ == "__main__":
    main()
