# HeartSense: one target per pipeline stage. `make help` lists them.
PY        ?= .venv/bin/python
IMAGE     ?= heartsense-api
TAG       ?= 1.0.0
PROFILE   ?= heartsense
NAMESPACE ?= heartsense
export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONPATH=src

.PHONY: help venv data eda notebook train predict lint format test test-fast docker-build docker-run docker-smoke \
        docker-stop mlflow-ui k8s-up k8s-deploy k8s-smoke k8s-forward k8s-down traffic architecture clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:            ## create .venv and install every dependency
	python3.12 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt

data:            ## download (SHA-256 verified) and clean the dataset
	$(PY) -m heartsense.ingest

eda:             ## EDA figures -> reports/figures/eda, profile -> reports/eda_profile.json
	$(PY) -m heartsense.explore

notebook:        ## build and execute the EDA notebook
	$(PY) scripts/build_notebook.py && $(PY) -m nbconvert --to notebook --execute --inplace notebooks/01_exploratory_analysis.ipynb

train:           ## tune 5 model families, track in MLflow, register + export champion
	$(PY) -m heartsense.train

train-quick:     ## tiny search (CI smoke)
	$(PY) -m heartsense.train --quick

predict:         ## offline prediction on the example patient
	$(PY) -m heartsense.predict --input examples/patient_high_risk.json

lint:            ## ruff lint + format check
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

format:          ## auto-format
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .

test:            ## full test suite with coverage
	$(PY) -m pytest --cov --cov-report=term-missing

test-fast:       ## tests without the slow end-to-end training test
	$(PY) -m pytest -m "not slow"

docker-build:    ## build the API image (needs model_store/heartsense_pipeline.joblib)
	docker build -t $(IMAGE):$(TAG) .

docker-run:      ## run the image on :8000
	docker run -d --rm --name heartsense-api -p 8000:8000 $(IMAGE):$(TAG)

docker-smoke:    ## smoke test the running container
	bash scripts/smoke_test.sh http://127.0.0.1:8000

docker-stop:
	-docker rm -f heartsense-api

mlflow-ui:       ## MLflow UI on http://127.0.0.1:5001
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001

k8s-up:          ## start minikube (docker driver)
	bash scripts/minikube_up.sh

k8s-deploy:      ## helm install API + apply monitoring stack, load local image into minikube
	IMAGE=$(IMAGE):$(TAG) bash scripts/deploy.sh

k8s-smoke:       ## smoke test through the Kubernetes LoadBalancer service (port-forward)
	bash scripts/smoke_test.sh http://127.0.0.1:8000

k8s-forward:     ## port-forward API :8000, Prometheus :9090, Grafana :3000
	bash scripts/port_forward.sh

k8s-down:        ## remove the release and the monitoring namespace
	bash scripts/teardown.sh

traffic:         ## 5 minutes of synthetic traffic for the dashboards
	$(PY) scripts/load_generator.py --duration 300 --rps 3

architecture:    ## render docs/architecture.png
	$(PY) docs/render_architecture.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml junit.xml htmlcov
