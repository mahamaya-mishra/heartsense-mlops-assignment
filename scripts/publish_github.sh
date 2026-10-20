#!/usr/bin/env bash
# Create the GitHub repository (if needed) and push the current branch.
#   bash scripts/publish_github.sh                 # repo: <gh user>/heartsense-mlops-assignment
#   REPO=myname/other-name bash scripts/publish_github.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gh auth status >/dev/null 2>&1 || { echo "not logged in: run  gh auth login -h github.com -p https -w"; exit 1; }
OWNER="$(gh api user --jq .login)"
OWNER_LC="$(printf "%s" "${OWNER}" | tr "[:upper:]" "[:lower:]")"
REPO="${REPO:-${OWNER}/heartsense-mlops-assignment}"
if ! gh repo view "${REPO}" >/dev/null 2>&1; then
  echo "==> creating ${REPO} (public)"
  gh repo create "${REPO}" --public --description "HeartSense: heart-disease risk API with an end-to-end MLOps pipeline (BITS MLOps Assignment 1)"
fi
git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/${REPO}.git"
git remote set-url origin "https://github.com/${REPO}.git"
# Point the Helm chart and deploy script at this owner's GHCR namespace.
sed -i.bak "s#ghcr.io/OWNER/#ghcr.io/${OWNER_LC}/#g" deploy/helm/heartsense-api/values.yaml scripts/deploy.sh README.md 2>/dev/null || true
find . -name "*.bak" -not -path "./.venv/*" -delete
git add -A
git diff --cached --quiet || git commit -q -m "chore: point image references at ghcr.io/${OWNER_LC}" || true
git push -u origin "$(git branch --show-current)"
echo "==> pushed: https://github.com/${REPO}"
echo "    Actions: https://github.com/${REPO}/actions"
