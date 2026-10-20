# Video script: HeartSense MLOps pipeline (target 4 to 5 minutes)

Record the screen with QuickTime (File > New Screen Recording) or OBS. Use a full-screen terminal
with a large font (Cmd +) and a browser with the tabs below opened in advance. Speak slowly.
Pausing between segments and joining clips afterwards is fine.

## Before pressing record (prep, about 5 minutes)

```bash
cd ~/Desktop/mlops_assgn
source .venv/bin/activate
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=src

# Cluster + services (if not already running)
minikube -p heartsense status || bash scripts/minikube_up.sh
kubectl -n heartsense get pods                      # expect 2 pods Running
bash scripts/port_forward.sh                        # API :8000, Prometheus :9090, Grafana :3000

# MLflow UI on :5001
nohup .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001 >/dev/null 2>&1 &

# Background traffic so the dashboard has live data (15 minutes)
nohup python scripts/load_generator.py --duration 900 --rps 4 >/dev/null 2>&1 &
clear
```

Browser tabs, in this order:

| # | Tab | URL |
|---|-----|-----|
| 1 | Architecture | `docs/architecture.png` (open the file) |
| 2 | GitHub repo | https://github.com/mahamaya-mishra/heartsense-mlops-assignment |
| 3 | GitHub Actions | https://github.com/mahamaya-mishra/heartsense-mlops-assignment/actions (open the latest run) |
| 4 | MLflow | http://127.0.0.1:5001 (experiment **heartsense**) |
| 5 | Swagger UI | http://127.0.0.1:8000/docs |
| 6 | Grafana | http://127.0.0.1:3000/d/heartsense-api |
| 7 | Prometheus | http://127.0.0.1:9090/targets |

Do not show: the Grafana admin password, any personal browser tabs, `docker ps` without a filter.

---

## Segment 1: introduction (0:00 to 0:25) - Tab 1, architecture

> "Hi, I am Mahamaya Mishra. This is my MLOps Assignment 1: HeartSense, a heart-disease risk
> classifier trained on the UCI Cleveland dataset and served as a monitored API on Kubernetes.
> The top lane is build time: data ingestion with a pinned checksum, EDA, training with MLflow
> tracking, and a GitHub Actions pipeline that lints, tests, trains and publishes a container
> image. The bottom lane is run time: the image runs on Kubernetes through a Helm chart, behind a
> LoadBalancer service, scraped by Prometheus and visualised in Grafana."

## Segment 2: repository, data and EDA (0:25 to 0:55) - Tab 2

Scroll the README: results table, repository layout. Open `notebooks/01_exploratory_analysis.ipynb`
on GitHub and scroll through two or three figures.

> "Everything lives in this repository. The ingest module downloads the dataset, verifies its
> SHA-256 hash and cleans it: invalid codes and implausible readings become missing values, and
> imputation happens inside the model pipeline so nothing leaks from the test set. The EDA shows
> a near-balanced outcome, 46 percent positive, only six missing cells, and the strongest signals
> from the thallium test, number of vessels, chest-pain type and exercise response."

## Segment 3: training and MLflow (0:55 to 1:40) - Tab 4

1. Runs table: expand a family run (click the arrow next to `svm_rbf` or `logistic_regression`) to show its nested `trial-xx` runs.
2. Open the champion family run: Parameters, Metrics, then Artifacts > `plots/`.
3. Open the `model-selection` run: Artifacts > `plots/leaderboard.png` and `plots/roc_curves.png`.
4. Left menu > Models > `heartsense-heart-disease`: point at the alias `@production`.

> "I compared five model families: logistic regression, random forest, an RBF support-vector
> machine, histogram gradient boosting and k-nearest neighbours. Each family is tuned with a
> randomised search under repeated stratified five-fold cross-validation, and every sampled
> configuration is a nested MLflow run with its parameters and metrics. The champion is chosen on
> mean cross-validated ROC-AUC only, never on the test set, and it is registered in the MLflow
> Model Registry under the alias production. The hold-out numbers are in the leaderboard."

## Segment 4: tests and CI/CD (1:40 to 2:20) - terminal, then Tab 3

```bash
make lint
pytest -m "not slow" -q
```

> "Ruff enforces lint and formatting. Fifty-five unit tests cover data cleaning, the feature
> engineering, the pipeline, model selection helpers, offline prediction and the HTTP contract of
> the API, including rejected inputs and the no-patient-data-in-logs rule. A slow end-to-end
> training test also runs in CI."

Tab 3: show the job graph; click **Unit tests**, then **Train, track & export champion** (scroll to the
leaderboard in the job summary), then **Build image, smoke test & publish**; show **Artifacts** at the bottom.

> "On every push, GitHub Actions runs lint and Helm validation, the tests on Python 3.11 and 3.12,
> a full training run whose model and MLflow store are uploaded as artifacts, and finally a Docker
> build with a live smoke test. Only an image that passes all of that is pushed to the GitHub
> Container Registry."

## Segment 5: Docker (2:20 to 2:45) - terminal

```bash
docker build -t heartsense-api:1.0.0 . | tail -n 3
docker run -d --rm --name demo -p 18000:8000 heartsense-api:1.0.0
bash scripts/smoke_test.sh http://127.0.0.1:18000
docker rm -f demo
```

> "The image is multi-stage, runs as a non-root user with a health check, and passes the smoke
> test: readiness, metadata, a high-risk and a low-risk prediction, a batch request, a rejected
> invalid payload and the metrics endpoint."

## Segment 6: Kubernetes (2:45 to 3:20) - terminal, then Tab 5

```bash
helm list -n heartsense
kubectl -n heartsense get deploy,pods,svc,hpa,pdb
curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d @examples/patient_high_risk.json | python -m json.tool
```

> "The Helm chart deploys two replicas with startup, readiness and liveness probes, resource
> limits, a horizontal pod autoscaler, a disruption budget and a LoadBalancer service. The
> response carries the prediction, the probability, a confidence value and a risk tier."

Tab 5: expand **POST /predict** > Try it out > Execute; scroll to the response.

## Segment 7: monitoring and logging (3:20 to 4:00) - Tabs 6, 7, terminal

Tab 6: point at replicas scraped, request rate, p95 latency, error ratio, predictions by risk tier,
and the **positive share vs training prevalence** panel.

> "Prometheus discovers the pods through annotations and scrapes both replicas. Grafana shows
> traffic, latency, errors and model behaviour. This panel compares the share of positive
> predictions with the 46 percent prevalence seen in training: a cheap drift signal, and there is
> an alert rule for it."

Tab 7: both targets UP; click **Alerts** to show the five rules. Then:

```bash
kubectl -n heartsense logs deployment/heartsense --tail=5
```

> "Every request is logged as one JSON line with request id, path, status and latency, plus the
> predicted label and risk tier. Patient measurements are never written to logs."

## Segment 8: close (4:00 to 4:15) - Tab 2

> "Code, report, screenshots and this video link are all in the repository. Thank you."

---

## After recording

```bash
pkill -f load_generator.py; pkill -f "mlflow ui"; pkill -f "port-forward svc/"
```

The cluster keeps running until `minikube -p heartsense stop`. Keep the video under about 100 MB;
if the portal limits size, upload to Google Drive (anyone with the link can view) and submit the link.
