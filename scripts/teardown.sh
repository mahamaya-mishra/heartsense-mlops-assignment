#!/usr/bin/env bash
# Remove the HeartSense release and the observability stack (keeps the cluster running).
set -uo pipefail
pkill -f "port-forward svc/(heartsense|prometheus|grafana)" 2>/dev/null || true
helm uninstall heartsense -n heartsense 2>/dev/null || true
kubectl delete namespace heartsense observability --ignore-not-found --wait=false
echo "removed. Stop the cluster with: minikube -p heartsense stop   (delete: minikube -p heartsense delete)"
