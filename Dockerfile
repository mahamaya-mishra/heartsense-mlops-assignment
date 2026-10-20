# syntax=docker/dockerfile:1.7
# HeartSense prediction API. Two stages: dependencies are built into a virtualenv, then only the
# virtualenv, the package and the trained model are copied into a slim runtime image.
#
#   docker build -t heartsense-api:1.0.0 .
#   docker run --rm -p 8000:8000 heartsense-api:1.0.0

ARG PYTHON_VERSION=3.12

# ------------------------------------------------------------------ build stage
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .

# ---------------------------------------------------------------- runtime stage
FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="heartsense-api" \
      org.opencontainers.image.description="Heart-disease risk prediction API (UCI Cleveland)" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_FILE=/srv/model_store/heartsense_pipeline.joblib \
    METADATA_FILE=/srv/model_store/metadata.json \
    LOG_LEVEL=INFO PORT=8000

RUN groupadd --system --gid 10001 heartsense \
 && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin heartsense
WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
COPY service/ service/
COPY model_store/heartsense_pipeline.joblib model_store/metadata.json model_store/
RUN chown -R 10001:10001 /srv
USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status==200 else 1)"]

# One uvicorn worker per container: Prometheus counters are per-process, scale with replicas.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--proxy-headers"]
