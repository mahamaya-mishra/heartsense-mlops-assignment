#!/usr/bin/env bash
# End-to-end smoke test against a running HeartSense API.
#   scripts/smoke_test.sh [BASE_URL]       default: http://127.0.0.1:8000
# Checks: readiness, liveness, metadata, a high-risk and a low-risk prediction, a batch request,
# validation (422) and the Prometheus endpoint. Exits non-zero on the first failure.
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX="${HERE}/../examples"
fail() { echo "FAIL: $*" >&2; exit 1; }
need() { grep -q "\"$2\"" <<<"$1" || fail "field '$2' missing in: $1"; }

echo "==> waiting for ${BASE}/readyz"
for i in $(seq 1 40); do
  curl -fsS "${BASE}/readyz" >/dev/null 2>&1 && break
  [[ $i -eq 40 ]] && fail "API not ready after 80 s"
  sleep 2
done

echo "==> GET /healthz";  curl -fsS "${BASE}/healthz"; echo
echo "==> GET /readyz";   curl -fsS "${BASE}/readyz"; echo
echo "==> GET /metadata (model_version)"
curl -fsS "${BASE}/metadata" | python3 -c 'import json,sys; m=json.load(sys.stdin); print(m["model_version"], m["family"])'

echo "==> POST /predict (high-risk patient)"
hi="$(curl -fsS -X POST "${BASE}/predict" -H 'Content-Type: application/json' -H 'X-Request-ID: smoke-high' \
      -d @"${EX}/patient_high_risk.json")"
echo "${hi}"
for f in prediction label risk_score confidence risk_tier model_version request_id; do need "${hi}" "${f}"; done
grep -q '"label":"disease"' <<<"${hi}" || fail "expected label=disease for the high-risk example"

echo "==> POST /predict (low-risk patient)"
lo="$(curl -fsS -X POST "${BASE}/predict" -H 'Content-Type: application/json' -d @"${EX}/patient_low_risk.json")"
echo "${lo}"
grep -q '"label":"no_disease"' <<<"${lo}" || fail "expected label=no_disease for the low-risk example"

echo "==> POST /predict/batch (3 patients)"
batch="$(curl -fsS -X POST "${BASE}/predict/batch" -H 'Content-Type: application/json' -d @"${EX}/batch_request.json")"
grep -q '"count":3' <<<"${batch}" || fail "batch count != 3: ${batch}"
echo "count=3 ok"

echo "==> POST /predict with invalid payload (expect 422)"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/predict" -H 'Content-Type: application/json' \
        -d '{"age": 200, "sex": 1}')"
[[ "${code}" == "422" ]] || fail "expected 422, got ${code}"
echo "422 ok"

echo "==> GET /metrics (heartsense_* excerpt)"
curl -fsS "${BASE}/metrics" | grep -E '^heartsense_(predictions_total|model_ready|http_requests_total)' | head -8

echo "SMOKE TEST PASSED (${BASE})"
