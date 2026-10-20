"""Structured JSON logging and Prometheus metrics for the service.

Logging policy: every request produces exactly one JSON line with method, path, status,
latency and request id. Patient payloads are **never** logged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge, Histogram, Info

SERVICE_NAME = "heartsense-api"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status",
            "latency_ms",
            "label",
            "risk_tier",
            "event",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str | None = None) -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    logging.getLogger("uvicorn.access").disabled = True  # we emit our own access log
    return logging.getLogger(SERVICE_NAME)


# ------------------------------------------------------------------- Prometheus metrics
REQUESTS = Counter("heartsense_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "heartsense_http_request_duration_seconds",
    "Request latency (s)",
    ["path"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("heartsense_predictions_total", "Predictions served", ["label", "risk_tier"])
RISK_SCORE = Histogram(
    "heartsense_risk_score",
    "Distribution of predicted disease probability",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
VALIDATION_ERRORS = Counter(
    "heartsense_validation_errors_total", "Rejected (422) prediction requests"
)
MODEL_READY = Gauge("heartsense_model_ready", "1 when the model is loaded")
MODEL_INFO = Info("heartsense_model", "Metadata of the loaded model")
INFLIGHT = Gauge("heartsense_inflight_requests", "Requests currently being processed")


class Stopwatch:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.start) * 1000.0
