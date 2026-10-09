#!/usr/bin/env bash

# Simple LAN worker that pulls jobs from the cloud API, runs the local model
# server (on the same machine/LAN), and posts results back via a webhook.
#
# Usage:
#   chmod +x scripts/worker.sh
#   WORKER_TOKEN=change-me API_BASE="http://34.105.69.230/chippygpt-staging" \
#     LOCAL_MODEL_BASE="http://localhost:8000" ./scripts/worker.sh
#
# Environment variables:
#   API_BASE          Base URL of the public API (cloud) exposing /jobs endpoints
#   WORKER_TOKEN      Shared secret for /jobs/next and /workers/heartbeat
#   LOCAL_MODEL_BASE  Base URL for the local model server (default http://localhost:8000)
#   API_KEY           Optional: Bearer key for LOCAL_MODEL_BASE /v1 calls (if the
#                     server has API-key auth enabled; header omitted when empty)
#   POLL_DELAY_MS     Delay between polls when idle (default 800)
#   HEARTBEAT_SEC     Interval between heartbeats (default 10)
#   TIMEOUT_SEC       Timeout for HTTP requests (default 60)

set -euo pipefail

API_BASE=${API_BASE:-"http://127.0.0.1:8000"}
WORKER_TOKEN=${WORKER_TOKEN:-"change-me"}
LOCAL_MODEL_BASE=${LOCAL_MODEL_BASE:-"http://localhost:8000"}
API_KEY=${API_KEY:-""}
POLL_DELAY_MS=${POLL_DELAY_MS:-800}
HEARTBEAT_SEC=${HEARTBEAT_SEC:-10}
TIMEOUT_SEC=${TIMEOUT_SEC:-60}

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required. Install with: sudo apt-get install -y jq" >&2
  exit 1
fi

last_hb=$(date +%s)

while true; do
  now=$(date +%s)
  if (( now - last_hb >= HEARTBEAT_SEC )); then
    curl -s -m "$TIMEOUT_SEC" -X POST "${API_BASE}/workers/heartbeat?worker_token=${WORKER_TOKEN}" \
      -H "Content-Type: application/json" -d '{}' >/dev/null || true
    last_hb=$now
  fi

  JOB=$(curl -s -m "$TIMEOUT_SEC" -X POST "${API_BASE}/jobs/next?worker_token=${WORKER_TOKEN}" \
    -H "Content-Type: application/json" -d '{}') || JOB='{"job":null}'

  JOB_ID=$(echo "$JOB" | jq -r '.job_id // empty')
  if [[ -z "$JOB_ID" ]]; then
    # no work; brief sleep
    usleep $((POLL_DELAY_MS*1000))
    continue
  fi

  PROBE=$(echo "$JOB" | jq -r '.probe // false')
  PAYLOAD=$(echo "$JOB" | jq -c '.payload // {}')

  # Execute against local model server (Authorization only when API_KEY is set)
  AUTH_ARGS=()
  if [[ -n "$API_KEY" ]]; then
    AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
  fi
  RESULT=$(curl -s -m "$TIMEOUT_SEC" -X POST "${LOCAL_MODEL_BASE}/v1/completions" \
    "${AUTH_ARGS[@]}" -H "Content-Type: application/json" -d "$PAYLOAD" || echo '{}')
  CURL_RC=$?

  SUCCESS=true
  if [[ "$CURL_RC" -ne 0 ]]; then
    SUCCESS=false
  fi

  # Post back to webhook with result
  CB_PAYLOAD=$(jq -n --arg jid "$JOB_ID" --argjson res "$RESULT" --argjson probe "$PROBE" --argjson ok $SUCCESS \
    '{job_id: $jid, result: $res, probe: $probe, success: $ok}')

  curl -s -m "$TIMEOUT_SEC" -X POST "${API_BASE}/webhooks/completions" \
    -H "Content-Type: application/json" -d "$CB_PAYLOAD" >/dev/null || true
done


