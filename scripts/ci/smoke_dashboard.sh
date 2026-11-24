#!/usr/bin/env bash
set -euo pipefail

dashboard_url="${DASHBOARD_URL:-http://localhost:5173}"

curl --fail --silent --show-error "${dashboard_url}/" >/dev/null
curl --fail --silent --show-error "${dashboard_url}/api/health" >/dev/null

echo "Dashboard and API smoke checks passed at ${dashboard_url}."
