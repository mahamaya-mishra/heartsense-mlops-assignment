#!/usr/bin/env bash
# Start (or reuse) the local Kubernetes cluster used for the HeartSense deployment.
#   bash scripts/minikube_up.sh            # profile "heartsense", docker driver
set -euo pipefail
PROFILE="${PROFILE:-heartsense}"
CPUS="${CPUS:-3}"
MEMORY="${MEMORY:-5120}"

command -v docker >/dev/null || { echo "docker CLI not found"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not running (start Docker Desktop or 'colima start')"; exit 1; }
command -v minikube >/dev/null || { echo "minikube not found (brew install minikube)"; exit 1; }

if minikube -p "${PROFILE}" status >/dev/null 2>&1; then
  echo "==> minikube profile '${PROFILE}' already running"
else
  echo "==> starting minikube profile '${PROFILE}' (${CPUS} CPU, ${MEMORY} MB)"
  minikube start -p "${PROFILE}" --driver=docker --cpus="${CPUS}" --memory="${MEMORY}"
fi
minikube -p "${PROFILE}" addons enable metrics-server >/dev/null   # needed by the HPA
kubectl config use-context "${PROFILE}" >/dev/null
kubectl wait --for=condition=Ready node --all --timeout=180s
kubectl get nodes -o wide
