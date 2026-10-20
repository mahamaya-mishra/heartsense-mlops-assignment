"""Send realistic traffic to the API so the dashboards have something to show.

    python scripts/load_generator.py --base-url http://127.0.0.1:8000 --duration 300 --rps 3

Patients are sampled from the cleaned dataset (with small jitter); ~3 % of requests are
deliberately invalid to exercise the 422 path, and ~5 % use the batch endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "rest_ecg",
    "max_heart_rate",
    "exercise_angina",
    "st_depression",
    "st_slope",
    "major_vessels",
    "thalassemia",
]
INTS = {
    "age",
    "sex",
    "chest_pain_type",
    "fasting_blood_sugar",
    "rest_ecg",
    "exercise_angina",
    "st_slope",
    "major_vessels",
    "thalassemia",
}


def post(url: str, payload: dict, timeout: float = 5.0) -> int:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError):
        return 0


def sample_patient(df: pd.DataFrame, rng: random.Random) -> dict:
    row = df.iloc[rng.randrange(len(df))]
    rec = {}
    for f in FEATURES:
        v = row[f]
        if pd.isna(v):
            continue
        if f in INTS:
            rec[f] = int(v)
        else:
            rec[f] = round(float(v) * rng.uniform(0.97, 1.03), 1)
    rec["st_depression"] = max(0.0, rec.get("st_depression", 0.0))
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--duration", type=int, default=300, help="seconds")
    p.add_argument("--rps", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    df = pd.read_csv(ROOT / "data" / "processed" / "heart_disease_clean.csv")
    rng = random.Random(a.seed)
    deadline = time.time() + a.duration
    counts: dict[int, int] = {}
    n = 0
    while time.time() < deadline:
        t0 = time.time()
        r = rng.random()
        if r < 0.03:
            status = post(f"{a.base_url}/predict", {"age": 999, "sex": 1})
        elif r < 0.08:
            pts = [sample_patient(df, rng) for _ in range(rng.randint(2, 6))]
            status = post(f"{a.base_url}/predict/batch", {"patients": pts})
        else:
            status = post(f"{a.base_url}/predict", sample_patient(df, rng))
        counts[status] = counts.get(status, 0) + 1
        n += 1
        if n % 50 == 0:
            print(f"{n} requests, status counts {counts}", flush=True)
        time.sleep(max(0.0, 1.0 / a.rps - (time.time() - t0)))
    print(f"done: {n} requests, status counts {counts}")
    sys.exit(0 if counts.get(0, 0) == 0 else 1)


if __name__ == "__main__":
    main()
