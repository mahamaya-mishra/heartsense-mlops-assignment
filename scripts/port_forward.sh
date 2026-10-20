#!/usr/bin/env bash
# Forward the three LoadBalancer services to localhost (minikube has no external IPs without
# `minikube tunnel`). Runs in the background; logs in /tmp/heartsense-pf-*.log.
set -euo pipefail
pkill -f "port-forward svc/heartsense " 2>/dev/null || true
pkill -f "port-forward svc/prometheus" 2>/dev/null || true
pkill -f "port-forward svc/grafana" 2>/dev/null || true
sleep 1
nohup kubectl -n heartsense     port-forward svc/heartsense 8000:8000 >/tmp/heartsense-pf-api.log 2>&1 &
nohup kubectl -n observability  port-forward svc/prometheus 9090:9090 >/tmp/heartsense-pf-prom.log 2>&1 &
nohup kubectl -n observability  port-forward svc/grafana    3000:3000 >/tmp/heartsense-pf-grafana.log 2>&1 &
sleep 3
for url in http://127.0.0.1:8000/readyz http://127.0.0.1:9090/-/ready http://127.0.0.1:3000/api/health; do
  printf '%-40s ' "$url"; curl -s -o /dev/null -w '%{http_code}\n' "$url" || true
done
echo "API docs  http://127.0.0.1:8000/docs"
echo "Prometheus http://127.0.0.1:9090/targets"
echo "Grafana    http://127.0.0.1:3000/d/heartsense-api"
