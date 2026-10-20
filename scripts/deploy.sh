#!/usr/bin/env bash
# Deploy the HeartSense API (Helm) and the observability stack (Prometheus + Grafana) to the
# current kubectl context.
#
#   bash scripts/deploy.sh                                   # image from ghcr.io (CI-published)
#   IMAGE=heartsense-api:1.0.0 bash scripts/deploy.sh        # locally built image (loaded into minikube)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RELEASE="${RELEASE:-heartsense}"
NAMESPACE="${NAMESPACE:-heartsense}"
MON_NS="observability"
PROFILE="${PROFILE:-heartsense}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/mahamaya-mishra/heartsense-api:latest}"
IMAGE="${IMAGE:-${GHCR_IMAGE}}"
REPO="${IMAGE%:*}"; TAG="${IMAGE##*:}"

echo "==> context: $(kubectl config current-context)   image: ${IMAGE}"

# A locally built image has to be copied into the minikube node (no registry involved).
if [[ "${IMAGE}" != ghcr.io/* ]] && command -v minikube >/dev/null && minikube -p "${PROFILE}" status >/dev/null 2>&1; then
  echo "==> loading ${IMAGE} into minikube profile ${PROFILE}"
  minikube -p "${PROFILE}" image load "${IMAGE}"
fi

echo "==> helm upgrade --install ${RELEASE} (namespace ${NAMESPACE})"
helm upgrade --install "${RELEASE}" deploy/helm/heartsense-api \
  --namespace "${NAMESPACE}" --create-namespace \
  --set image.repository="${REPO}" --set image.tag="${TAG}" \
  --set image.pullPolicy="$([[ "${IMAGE}" == ghcr.io/* ]] && echo Always || echo IfNotPresent)" \
  --wait --timeout 5m
kubectl -n "${NAMESPACE}" rollout status deployment/"${RELEASE}" --timeout=300s

echo "==> observability stack (namespace ${MON_NS})"
kubectl apply -f deploy/monitoring/k8s/namespace.yaml
cm() { kubectl -n "${MON_NS}" create configmap "$1" "${@:2}" --dry-run=client -o yaml | kubectl apply -f -; }
cm prometheus-config      --from-file=deploy/monitoring/prometheus/
cm grafana-datasources    --from-file=deploy/monitoring/grafana/provisioning/datasources/
cm grafana-dashboard-provider --from-file=deploy/monitoring/grafana/provisioning/dashboards/
cm grafana-dashboards     --from-file=deploy/monitoring/grafana/dashboards/
if ! kubectl -n "${MON_NS}" get secret grafana-admin >/dev/null 2>&1; then
  kubectl -n "${MON_NS}" create secret generic grafana-admin \
    --from-literal=admin-user=admin --from-literal=admin-password="$(openssl rand -base64 18)" >/dev/null
  echo "    created secret ${MON_NS}/grafana-admin (read it with: make grafana-password)"
fi
# Annotate with a config hash so pods restart when their configuration changes.
stamp() { local h; h="$(cat "${@:2}" | shasum -a 256 | cut -c1-12)"; sed "s/CONFIG_HASH/${h}/" "$1" | kubectl apply -f -; }
stamp deploy/monitoring/k8s/prometheus.yaml deploy/monitoring/prometheus/*
stamp deploy/monitoring/k8s/grafana.yaml deploy/monitoring/grafana/provisioning/*/* deploy/monitoring/grafana/dashboards/*
kubectl -n "${MON_NS}" rollout status deployment/prometheus --timeout=300s
kubectl -n "${MON_NS}" rollout status deployment/grafana --timeout=300s

echo
kubectl -n "${NAMESPACE}" get deploy,pods,svc,hpa,pdb -o wide
echo
kubectl -n "${MON_NS}" get deploy,pods,svc
cat <<'EOT'

Next: expose the services on this machine and run the smoke test
  bash scripts/port_forward.sh            # API :8000, Prometheus :9090, Grafana :3000
  bash scripts/smoke_test.sh http://127.0.0.1:8000
(or `minikube -p heartsense tunnel` to get real LoadBalancer IPs: kubectl get svc -A)
EOT
