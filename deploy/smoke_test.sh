#!/usr/bin/env bash
# Smoke test for a running Nu Score On the Go API.
# Usage: deploy/smoke_test.sh [base_url] [csv_path]
set -euo pipefail

BASE="${1:-http://127.0.0.1:8096}"
CSV="${2:-input/customer_churn_1yr.csv}"
PY=python3

json() { $PY -c "import sys,json;d=json.load(sys.stdin);print($1)"; }

wait_job() {
  local job="$1" status
  for _ in $(seq 1 300); do
    status=$(curl -sf "$BASE/api/jobs/$job")
    if echo "$status" | grep -q '"succeeded"\|"failed"'; then echo "$status"; return; fi
    sleep 2
  done
  echo "$status"
}

echo "health: $(curl -sf "$BASE/api/health")"
echo "index:  $(curl -s -o /dev/null -w '%{http_code}' "$BASE/")"

echo "-- synthetic job"
job=$(curl -sf -X POST "$BASE/api/jobs/train" -H 'content-type: application/json' \
      -d '{"n_rows":2000,"n_features":30,"n_categorical":5,"tier":"quick"}' | json "d['job_id']")
wait_job "$job" | json "d['status'], d['message']"
curl -sf "$BASE/api/jobs/$job/result" | json "'leader:', d['leaderboard'][0]"

echo "-- csv upload + job ($CSV)"
up=$(curl -sf -X POST "$BASE/api/uploads?filename=$(basename "$CSV")" \
     -H 'content-type: application/octet-stream' --data-binary @"$CSV")
echo "$up" | json "'rows', d['row_count'], 'cols', d['column_count'], 'target', d['ai_analysis'].get('suggested_target_col')"
sid=$(echo "$up" | json "d['session_id']")
tgt=$(echo "$up" | json "d['ai_analysis'].get('suggested_target_col') or ''")
job=$(curl -sf -X POST "$BASE/api/jobs/train-csv" -H 'content-type: application/json' \
      -d "{\"session_id\":\"$sid\",\"target_col\":\"$tgt\",\"tier\":\"quick\"}" | json "d['job_id']")
wait_job "$job" | json "d['status'], d['message']"
curl -sf "$BASE/api/jobs/$job/result" | json "'leader:', d['leaderboard'][0], 'rows', d['dataset']['n_rows']"
