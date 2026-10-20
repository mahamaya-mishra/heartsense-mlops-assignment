"""HeartSense prediction API.

    uvicorn service.app:app --host 0.0.0.0 --port 8000

Environment: MODEL_FILE, METADATA_FILE (paths), LOG_LEVEL.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heartsense import __version__
from heartsense import settings as cfg
from heartsense.predict import score_records
from service import telemetry
from service.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PatientRecord,
    PredictionResponse,
    ReadyResponse,
)

log = telemetry.configure_logging()
STARTED_AT = time.time()


class ModelState:
    model = None
    metadata: dict = {}
    error: str | None = None

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", cfg.DECISION_THRESHOLD))


state = ModelState()


def load_model(model_file: Path | None = None, metadata_file: Path | None = None) -> None:
    model_file = Path(model_file or os.getenv("MODEL_FILE", cfg.MODEL_FILE))
    metadata_file = Path(metadata_file or os.getenv("METADATA_FILE", cfg.METADATA_FILE))
    try:
        state.model = joblib.load(model_file)
        state.metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        state.error = None
        telemetry.MODEL_READY.set(1)
        telemetry.MODEL_INFO.info(
            {
                "version": state.version,
                "family": str(state.metadata.get("family", "?")),
                "git_sha": str(state.metadata.get("git_sha", "?")),
            }
        )
        log.info("model loaded", extra={"event": "model_loaded", "label": state.version})
    except Exception as exc:
        state.model, state.error = None, f"{type(exc).__name__}: {exc}"
        telemetry.MODEL_READY.set(0)
        log.error("model load failed: %s", state.error, extra={"event": "model_load_failed"})


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="HeartSense API",
    version=__version__,
    description="Predicts the probability of heart disease from 13 clinical measurements "
    "(UCI Cleveland). Educational project, not a medical device.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    telemetry.INFLIGHT.inc()
    with telemetry.Stopwatch() as sw:
        try:
            response: Response = await call_next(request)
        except Exception:
            telemetry.INFLIGHT.dec()
            telemetry.REQUESTS.labels(request.method, request.url.path, "500").inc()
            log.exception(
                "unhandled error", extra={"request_id": request_id, "path": request.url.path}
            )
            raise
    telemetry.INFLIGHT.dec()
    path = request.url.path if request.url.path in KNOWN_PATHS else "other"
    telemetry.REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    telemetry.LATENCY.labels(path).observe(sw.ms / 1000.0)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "request",
        extra={
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(sw.ms, 2),
        },
    )
    return response


KNOWN_PATHS = {
    "/",
    "/healthz",
    "/readyz",
    "/metadata",
    "/predict",
    "/predict/batch",
    "/metrics",
    "/docs",
}


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/predict"):
        telemetry.VALIDATION_ERRORS.inc()
    return JSONResponse(
        status_code=422, content={"detail": exc.errors(), "request_id": request.state.request_id}
    )


def _require_model():
    if state.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {state.error or 'unknown'}")


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": telemetry.SERVICE_NAME,
        "docs": "/docs",
        "health": "/healthz",
        "predict": "/predict",
    }


@app.get("/healthz", response_model=HealthResponse, tags=["operations"])
def healthz():
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        service=telemetry.SERVICE_NAME,
        version=__version__,
        uptime_seconds=round(time.time() - STARTED_AT, 1),
    )


@app.get(
    "/readyz",
    response_model=ReadyResponse,
    tags=["operations"],
    responses={503: {"description": "model not loaded"}},
)
def readyz():
    if state.model is None:
        return JSONResponse(
            status_code=503, content=ReadyResponse(ready=False, detail=state.error).model_dump()
        )
    return ReadyResponse(ready=True, model_version=state.version)


@app.get("/metadata", tags=["model"])
def metadata():
    _require_model()
    return state.metadata


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["model"],
    responses={422: {"description": "invalid input"}, 503: {"description": "model not loaded"}},
)
def predict(patient: PatientRecord, request: Request):
    _require_model()
    with telemetry.Stopwatch() as sw:
        result = score_records(state.model, [patient.model_dump()], state.threshold)[0]
    telemetry.PREDICTIONS.labels(result["label"], result["risk_tier"]).inc()
    telemetry.RISK_SCORE.observe(result["risk_score"])
    log.info(
        "prediction",
        extra={
            "event": "prediction",
            "request_id": request.state.request_id,
            "label": result["label"],
            "risk_tier": result["risk_tier"],
        },
    )
    return PredictionResponse(
        **result,
        model_version=state.version,
        request_id=request.state.request_id,
        latency_ms=round(sw.ms, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["model"])
def predict_batch(batch: BatchRequest, request: Request):
    _require_model()
    results = score_records(state.model, [p.model_dump() for p in batch.patients], state.threshold)
    for r in results:
        telemetry.PREDICTIONS.labels(r["label"], r["risk_tier"]).inc()
        telemetry.RISK_SCORE.observe(r["risk_score"])
    return BatchResponse(
        model_version=state.version,
        request_id=request.state.request_id,
        count=len(results),
        results=results,
    )


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
