"""Capture screenshots of the web UIs (MLflow, Swagger, Prometheus, Grafana) with Playwright.

python tools/capture_web.py --only mlflow,api          # subset
python tools/capture_web.py                             # everything that is reachable
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
MLFLOW = "http://127.0.0.1:5001"
API = "http://127.0.0.1:8000"
PROM = "http://127.0.0.1:9090"
GRAFANA = "http://127.0.0.1:3000"


def reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def shot(page, url: str, name: str, wait: float = 2.0, full: bool = True, actions=None):
    page.goto(url, wait_until="load", timeout=60000)
    time.sleep(wait)
    if actions:
        actions(page)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print("wrote", name)


def mlflow_ids() -> dict:
    """Find experiment and run ids through the REST API."""

    def post(path, body):
        req = urllib.request.Request(
            f"{MLFLOW}/api/2.0/mlflow/{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.load(urllib.request.urlopen(req, timeout=10))

    def get(path):
        return json.load(urllib.request.urlopen(f"{MLFLOW}/api/2.0/mlflow/{path}", timeout=10))

    exp = get("experiments/get-by-name?experiment_name=heartsense")["experiment"]["experiment_id"]
    runs = post("runs/search", {"experiment_ids": [exp], "max_results": 500})["runs"]
    by_name = {}
    for r in runs:
        tags = {t["key"]: t["value"] for t in r["data"].get("tags", [])}
        by_name[tags.get("mlflow.runName", "")] = r["info"]["run_id"]
    meta = json.loads((ROOT / "model_store" / "metadata.json").read_text())
    return {
        "exp": exp,
        "champion": by_name.get(meta["family"]),
        "selection": by_name.get("model-selection"),
        "family": meta["family"],
    }


def capture_mlflow(page):
    ids = mlflow_ids()
    e, c, s = ids["exp"], ids["champion"], ids["selection"]
    shot(page, f"{MLFLOW}/#/experiments/{e}", "mlflow_01_experiment_runs", wait=4, full=False)
    shot(
        page,
        f"{MLFLOW}/#/experiments/{e}/runs/{c}",
        "mlflow_02_champion_run_overview",
        wait=4,
        full=False,
    )
    shot(
        page,
        f"{MLFLOW}/#/experiments/{e}/runs/{c}/model-metrics",
        "mlflow_03_champion_metrics",
        wait=4,
        full=False,
    )
    shot(
        page,
        f"{MLFLOW}/#/experiments/{e}/runs/{c}/artifacts/plots/permutation_importance.png",
        "mlflow_04_champion_artifact_importance",
        wait=4,
        full=False,
    )
    shot(
        page,
        f"{MLFLOW}/#/experiments/{e}/runs/{s}/artifacts/plots/roc_curves.png",
        "mlflow_05_selection_roc_curves",
        wait=4,
        full=False,
    )
    shot(
        page,
        f"{MLFLOW}/#/experiments/{e}/runs/{s}/artifacts/plots/leaderboard.png",
        "mlflow_06_selection_leaderboard",
        wait=4,
        full=False,
    )
    shot(
        page,
        f"{MLFLOW}/#/models/heartsense-heart-disease",
        "mlflow_07_registered_model",
        wait=4,
        full=False,
    )
    shot(
        page,
        f"{MLFLOW}/#/experiments/{e}?searchFilter=tags.stage%20%3D%20%22trial%22&orderByKey=metrics.cv_roc_auc_mean&orderByAsc=false",
        "mlflow_08_trial_runs_sorted",
        wait=4,
        full=False,
    )


def capture_api(page):
    shot(page, f"{API}/docs", "api_01_swagger_overview", wait=2, full=True)

    def try_it(p):
        p.locator("#operations-model-predict_predict_post").click()
        time.sleep(0.8)
        p.get_by_role("button", name="Try it out").first.click()
        time.sleep(0.5)
        body = (ROOT / "examples" / "patient_high_risk.json").read_text()
        ta = p.locator("#operations-model-predict_predict_post textarea").first
        ta.fill(body)
        p.get_by_role("button", name="Execute").first.click()
        time.sleep(2.0)
        p.locator("#operations-model-predict_predict_post").scroll_into_view_if_needed()

    shot(page, f"{API}/docs", "api_02_predict_try_it_out", wait=2, full=True, actions=try_it)
    shot(page, f"{API}/metadata", "api_03_metadata", wait=1, full=True)
    shot(page, f"{API}/metrics", "api_04_prometheus_metrics", wait=1, full=False)


def capture_prom(page):
    shot(page, f"{PROM}/targets", "prometheus_01_targets", wait=3, full=True)
    shot(page, f"{PROM}/alerts", "prometheus_02_alert_rules", wait=3, full=True)
    q = "sum by (path, status) (rate(heartsense_http_requests_total[5m]))"
    shot(
        page,
        f"{PROM}/query?g0.expr={urllib.request.quote(q)}&g0.tab=graph&g0.range_input=15m",
        "prometheus_03_request_rate_graph",
        wait=4,
        full=True,
    )


def capture_grafana(page):
    shot(
        page,
        f"{GRAFANA}/d/heartsense-api?orgId=1&from=now-15m&to=now&kiosk",
        "grafana_01_dashboard_kiosk",
        wait=8,
        full=True,
    )
    shot(
        page,
        f"{GRAFANA}/d/heartsense-api?orgId=1&from=now-15m&to=now",
        "grafana_02_dashboard",
        wait=8,
        full=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="mlflow,api,prom,grafana")
    a = ap.parse_args()
    want = set(a.only.split(","))
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        for key, url, fn in [
            ("mlflow", MLFLOW, capture_mlflow),
            ("api", f"{API}/healthz", capture_api),
            ("prom", f"{PROM}/-/ready", capture_prom),
            ("grafana", f"{GRAFANA}/api/health", capture_grafana),
        ]:
            if key not in want:
                continue
            if not reachable(url):
                print(f"skip {key}: {url} not reachable")
                continue
            try:
                fn(page)
            except Exception as exc:
                print(f"{key} failed: {exc}")
        browser.close()


if __name__ == "__main__":
    main()
