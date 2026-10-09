#!/bin/bash
#
# test_auth.sh — API-key authentication test suite for llmlocal (self-contained).
#
# Usage:
#   ./scripts/test_auth.sh
#       Starts its OWN server on LLM_PORT=8010 / LLM_BACKEND_PORT=8110 so it
#       cannot collide with a dev server on 8000/8100 or another agent's run,
#       waits for /health, runs the checks, then stops the server by port-PID
#       (never by process name). API_KEY and WORKER_TOKEN are read from .env
#       and never echoed.
#
#   MANAGE_SERVER=0 BASE_URL=http://localhost:8010 SERVER_LOG=/tmp/llmlocal_auth.log ./scripts/test_auth.sh
#       Use a server you started yourself (then this script neither starts nor
#       stops it). SERVER_LOG is required for check 14 (server stdout scan).
#
#   PowerShell launcher for external mode (run BEFORE the script):
#       $env:LLM_PORT='8010'; $env:LLM_BACKEND_PORT='8110'
#       python server.py *> "$env:TEMP\llmlocal_auth.log"
#   Git-Bash launcher for external mode:
#       LLM_PORT=8010 LLM_BACKEND_PORT=8110 python server.py > /tmp/llmlocal_auth.log 2>&1 &
#
# Overrides: PORT, BASE_URL, BACKEND_PORT, MANAGE_SERVER, SERVER_LOG, HEALTH_WAIT, PYTHON
# Exit codes: 0 all passed | 1 failures | 2 API_KEY empty | 3 auth not wired
#             into the server yet (blocked on the FastAPI migration) | 4 server
#             failed to start.

set -u
cd "$(dirname "$0")/.." || exit 1

PY="${PYTHON:-python}"
PORT="${PORT:-8010}"
BACKEND_PORT="${BACKEND_PORT:-8110}"
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
MANAGE_SERVER="${MANAGE_SERVER:-1}"
HEALTH_WAIT="${HEALTH_WAIT:-240}"
TIMEOUT=60

PASS=0
FAIL=0

WORK=$(mktemp -d)
BODY="$WORK/body.json"
SERVER_LOG="${SERVER_LOG:-$WORK/server.log}"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

env_get() {
    # Print value of KEY from .env (CR/quotes stripped). Never echoed anywhere.
    local line v
    line=$(grep -E "^$1=" .env 2>/dev/null | head -1)
    [ -z "$line" ] && return 0
    v=${line#*=}
    v=${v//$'\r'/}
    v=${v#\"}; v=${v%\"}
    v=${v#\'}; v=${v%\'}
    printf '%s' "$v"
}

check() {
    local name="$1" cond="$2"
    if eval "$cond"; then
        echo "PASS  $name"
        PASS=$((PASS+1))
    else
        echo "FAIL  $name"
        FAIL=$((FAIL+1))
    fi
}

# http METHOD PATH [curl options...] -> prints HTTP status, body in $BODY
http() {
    local method="$1" path="$2"
    shift 2
    curl -sS -o "$BODY" -w '%{http_code}' --max-time "$TIMEOUT" \
        -X "$method" "$@" "${BASE_URL}${path}"
}

# _json_ok <file> <python expr over d>   -> exit 0 when expr is truthy
_json_ok() {
    "$PY" -c 'import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    ok = bool(eval(sys.argv[2], {"d": d}))
except Exception:
    ok = False
sys.exit(0 if ok else 1)' "$1" "$2"
}

pid_on_port() {
    # PID of the process LISTENING on a TCP port (Windows). Prints "" if none.
    local port="$1" pid=""
    if command -v netstat >/dev/null 2>&1; then
        pid=$(netstat -ano 2>/dev/null | awk -v p=":$port" \
            '$4 == "LISTENING" && $2 ~ p"$" {print $NF; exit}')
    fi
    if [ -z "$pid" ] && command -v powershell.exe >/dev/null 2>&1; then
        pid=$(powershell.exe -NoProfile -Command \
            "(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)" \
            2>/dev/null | tr -d '\r\n')
    fi
    case "$pid" in '' | *[!0-9]*) pid="";; esac
    printf '%s' "$pid"
}

kill_pid() {
    # Kill by PID only (never by process name — other agents may run their
    # own llama-server/python concurrently on other ports).
    taskkill //F //PID "$1" >/dev/null 2>&1 ||
        powershell.exe -NoProfile -Command "Stop-Process -Id $1 -Force" >/dev/null 2>&1 ||
        true
}

stop_server() {
    # Stop ONLY our own ports: llama-server on $BACKEND_PORT, API on $PORT.
    local pid
    for p in "$BACKEND_PORT" "$PORT"; do
        pid=$(pid_on_port "$p")
        if [ -n "$pid" ]; then
            kill_pid "$pid"
            sleep 1
        fi
    done
    if [ -n "${SERVER_SHELL_PID:-}" ]; then
        kill "$SERVER_SHELL_PID" 2>/dev/null || true
    fi
}

cleanup() {
    if [ "$MANAGE_SERVER" = "1" ] && [ -n "${SERVER_SHELL_PID:-}" ]; then
        stop_server
    fi
    return 0
}
trap cleanup EXIT

complete_job() {
    # complete_job <job_id> -> prints webhook HTTP status
    curl -sS -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
        -H 'Content-Type: application/json' \
        -d "{\"job_id\":\"$1\",\"success\":true,\"result\":{\"text\":\"test_auth drain\"}}" \
        "${BASE_URL}/webhooks/completions" 2>/dev/null || true
}

drain_queue() {
    # Pop one queued job (valid worker token) and complete it via webhook.
    # Leaves the queue empty when it was holding exactly our job(s).
    local resp jid code
    resp=$(curl -sS --max-time 40 -X POST \
        "${BASE_URL}/jobs/next?worker_token=${WORKER_TOKEN}" 2>/dev/null || true)
    jid=$(printf '%s' "$resp" | "$PY" -c 'import json, sys
try:
    print(json.load(sys.stdin).get("job_id") or "")
except Exception:
    print("")')
    [ -z "$jid" ] && return 1
    code=$(complete_job "$jid")
    [ "$code" = "200" ]
}

# ---------------------------------------------------------------------------
# secrets (never echoed) + configuration gate
# ---------------------------------------------------------------------------

API_KEY=$(env_get API_KEY)
WORKER_TOKEN=$(env_get WORKER_TOKEN)
WORKER_TOKEN="${WORKER_TOKEN:-change-me}"

if [ -z "$API_KEY" ]; then
    echo "ERROR: API_KEY is empty in .env — API-key auth is disabled (development mode),"
    echo "       so auth checks cannot run. Set API_KEY in .env and re-run."
    exit 2
fi

echo "API auth test suite — $BASE_URL (MANAGE_SERVER=$MANAGE_SERVER, port=$PORT/backend=$BACKEND_PORT)"

# ---------------------------------------------------------------------------
# start server (own ports only)
# ---------------------------------------------------------------------------

if [ "$MANAGE_SERVER" = "1" ]; then
    for p in "$BACKEND_PORT" "$PORT"; do
        stale=$(pid_on_port "$p")
        if [ -n "$stale" ]; then
            echo "note: port $p busy (pid $stale) — stopping stale listener on our own port"
            kill_pid "$stale"
            sleep 2
        fi
    done
    LLM_PORT="$PORT" LLM_BACKEND_PORT="$BACKEND_PORT" LLM_LOG="$WORK/llama-server.log" \
        "$PY" server.py > "$SERVER_LOG" 2>&1 &
    SERVER_SHELL_PID=$!
    echo "started: python server.py (pid $SERVER_SHELL_PID), log $SERVER_LOG"
fi

echo "waiting for GET /health (up to ${HEALTH_WAIT}s, includes model load)..."
ready=0
for _ in $(seq 1 "$HEALTH_WAIT"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$BASE_URL/health" 2>/dev/null || true)
    if [ "$code" = "200" ]; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" != "1" ]; then
    echo "ERROR: server did not become healthy within ${HEALTH_WAIT}s (last /health code: ${code:-none})"
    if [ "$MANAGE_SERVER" = "1" ] && [ -f "$SERVER_LOG" ]; then
        echo "--- last 15 lines of server log ---"
        tail -15 "$SERVER_LOG"
    fi
    exit 4
fi

# ---------------------------------------------------------------------------
# gate: is API-key auth actually enforced? (pending Agent 3's FastAPI wiring)
# ---------------------------------------------------------------------------

TIMEOUT=10
GATE_CODE=$(http GET /v1/models)
for _ in $(seq 1 30); do
    [ "$GATE_CODE" = "401" ] && break
    [ "$GATE_CODE" = "503" ] || break   # 503 = model still warming; anything else is conclusive
    sleep 1
    GATE_CODE=$(http GET /v1/models)
done
if [ "$GATE_CODE" != "401" ]; then
    echo "BLOCKED: GET /v1/models without a key returned '$GATE_CODE' (expected 401)."
    echo "         The server does not enforce API-key auth yet — pending the FastAPI"
    echo "         migration (app/main.py + routes wiring require_api_key). Auth checks"
    echo "         not run; nothing outside app/auth was modified to force a pass."
    exit 3
fi
TIMEOUT=60

# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

echo "----------------------------------------------"

# 1
CODE=$(http GET /health)
check "1. GET /health without key -> 200" '[ "$CODE" = "200" ]'

# 2
CODE=$(http GET /v1/models)
cp "$BODY" "$WORK/missing_401.json"
check "2. GET /v1/models without key -> 401 error.code=invalid_api_key" \
    '[ "$CODE" = "401" ] && _json_ok "$BODY" '"'"'d.get("error", {}).get("code") == "invalid_api_key"'"'"''

# 3
CODE=$(http GET /v1/models -H "Authorization: Bearer invalid-key-for-testing-000")
check "3. GET /v1/models with invalid key -> 401, body identical to check 2" \
    '[ "$CODE" = "401" ] && cmp -s "$WORK/missing_401.json" "$BODY"'

# 4
CODE=$(http GET /v1/models -H "Authorization: Bearer $API_KEY")
check "4. GET /v1/models with valid key -> 200" '[ "$CODE" = "200" ]'

# 5
CHAT_PAYLOAD='{"model":"local","messages":[{"role":"user","content":"Say OK"}],"max_tokens":16,"thinking":false}'
CODE_NO=$(http POST /v1/chat/completions -H 'Content-Type: application/json' -d "$CHAT_PAYLOAD")
TIMEOUT=180
CODE_OK=$(http POST /v1/chat/completions -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' -d "$CHAT_PAYLOAD")
TIMEOUT=60
check "5. POST /v1/chat/completions: 401 without key, 200 with valid key" \
    '[ "$CODE_NO" = "401" ] && [ "$CODE_OK" = "200" ]'

# 6
COMPLETION_PAYLOAD='{"model":"local","prompt":"Say OK","max_tokens":16}'
TIMEOUT=180
CODE=$(http POST /v1/completions -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' -d "$COMPLETION_PAYLOAD")
TIMEOUT=60
check "6. POST /v1/completions with valid key -> 200" '[ "$CODE" = "200" ]'

# 7
JOB_PAYLOAD='{"model":"local","prompt":"test_auth job","max_tokens":16}'
CODE_NO=$(http POST /jobs -H 'Content-Type: application/json' -d "$JOB_PAYLOAD")
CODE_OK=$(http POST /jobs -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' -d "$JOB_PAYLOAD")
JOB_ID=$("$PY" -c 'import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("job_id") or "")
except Exception:
    print("")' "$BODY")
DRAIN_OK=0
if [ -n "$JOB_ID" ] && drain_queue; then
    DRAIN_OK=1
fi
check "7. POST /jobs: 401 without key, 200 with key, job drained via /jobs/next + webhook" \
    '[ "$CODE_NO" = "401" ] && [ "$CODE_OK" = "200" ] && [ "$DRAIN_OK" = "1" ]'

# 8
CODE_NO=$(http GET /queue/metrics)
CODE_OK=$(http GET /queue/metrics -H "Authorization: Bearer $API_KEY")
check "8. GET /queue/metrics: 401 without key, 200 with valid key" \
    '[ "$CODE_NO" = "401" ] && [ "$CODE_OK" = "200" ]'

# 9
TIMEOUT=20
CODE=$(http POST "/jobs/next?worker_token=wrong-token-test" \
    -H "Authorization: Bearer $API_KEY")
TIMEOUT=60
check "9. POST /jobs/next?worker_token=WRONG with valid API key header -> 403 (auth worlds independent)" \
    '[ "$CODE" = "403" ]'

# 10
CODE_ENQ=$(http POST /jobs -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' -d "$JOB_PAYLOAD")
TIMEOUT=40
CODE_NEXT=$(http POST "/jobs/next?worker_token=${WORKER_TOKEN}")   # no API key header
TIMEOUT=60
NEXT_JOB=$("$PY" -c 'import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("job_id") or "")
except Exception:
    print("")' "$BODY")
WH_OK=0
if [ -n "$NEXT_JOB" ]; then
    [ "$(complete_job "$NEXT_JOB")" = "200" ] && WH_OK=1
fi
check "10. POST /jobs/next?worker_token=<valid> with NO API key header -> 200 (then drained)" \
    '[ "$CODE_ENQ" = "200" ] && [ "$CODE_NEXT" = "200" ] && [ "$WH_OK" = "1" ]'

# 11
CODE_BAD=$(http POST "/workers/heartbeat?worker_token=wrong-token-test")
CODE_GOOD=$(http POST "/workers/heartbeat?worker_token=${WORKER_TOKEN}")
check "11. POST /workers/heartbeat: 403 with wrong worker_token, 200 with valid" \
    '[ "$CODE_BAD" = "403" ] && [ "$CODE_GOOD" = "200" ]'

# 12
CODE_ENQ=$(http POST /jobs -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' -d "$JOB_PAYLOAD")
WEBHOOK_JOB=$("$PY" -c 'import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("job_id") or "")
except Exception:
    print("")' "$BODY")
CODE_WH=000
if [ -n "$WEBHOOK_JOB" ]; then
    CODE_WH=$(complete_job "$WEBHOOK_JOB")   # no API key header on purpose
fi
drain_queue >/dev/null 2>&1 || true          # pop the completed job so queue ends empty
check "12. POST /webhooks/completions without key -> 200 (documented current behavior)" \
    '[ "$CODE_ENQ" = "200" ] && [ "$CODE_WH" = "200" ]'

# 13
CODE=$(http GET "/v1/models?api_key=${API_KEY}")   # correct key, but in QUERY string
check "13. GET /v1/models?api_key=<valid> (key in query string) -> 401 (never accepted)" \
    '[ "$CODE" = "401" ]'

# 14
LOG_OK=1
LOG_REASON=""
if [ ! -f "$SERVER_LOG" ]; then
    LOG_OK=0
    LOG_REASON="server log not found (set SERVER_LOG=... when using MANAGE_SERVER=0)"
elif [ ! -s "$SERVER_LOG" ]; then
    LOG_OK=0
    LOG_REASON="server log is empty — nothing was scanned"
else
    if grep -Fq -- "$API_KEY" "$SERVER_LOG"; then
        LOG_OK=0
        LOG_REASON="API_KEY value appears in server stdout"
    elif grep -Fq -- "$WORKER_TOKEN" "$SERVER_LOG"; then
        LOG_OK=0
        LOG_REASON="WORKER_TOKEN value appears in server stdout"
    elif grep -Fq "Authorization" "$SERVER_LOG"; then
        LOG_OK=0
        LOG_REASON="Authorization header value appears in server stdout"
    fi
fi
if [ "$LOG_OK" = "1" ]; then
    echo "PASS  14. server stdout log contains no API_KEY, no WORKER_TOKEN, no Authorization value"
    PASS=$((PASS+1))
else
    echo "FAIL  14. server stdout log contains no API_KEY, no WORKER_TOKEN, no Authorization value"
    echo "      -> $LOG_REASON"
    FAIL=$((FAIL+1))
fi

# 15 (queue left empty by checks 7/10/12)
CODE=$(http GET /queue/metrics -H "Authorization: Bearer $API_KEY")
check "15. queue empty after drains (queue_depth == 0)" \
    '[ "$CODE" = "200" ] && _json_ok "$BODY" '"'"'d.get("queue_depth") == 0'"'"''

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

echo "=============================================="
echo "Passed: $PASS  Failed: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "Some auth checks failed"
    exit 1
fi
echo "All auth checks passed"
