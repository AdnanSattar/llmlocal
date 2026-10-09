#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

# OpenAI API Compatible Local LLM Server Test Script
# Override with env vars: BASE_URL=http://host:port MODEL=my-model ./scripts/test_api.sh

BASE_URL="${BASE_URL:-http://localhost:8000}"

# Model id echoed back by the server (defaults to whatever .env configures)
MODEL="${MODEL:-$(grep -E '^LLM_MODEL=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r"')}"
MODEL="${MODEL:-local}"

# API key for protected endpoints (read from .env, never echoed).
# First NON-EMPTY API_KEY= wins (mirrors the server: an empty API_KEY means
# auth disabled, and then no header is sent — protected routes stay open).
# /health is always called WITHOUT the header (it is public).
API_KEY="${API_KEY:-$(grep -E '^API_KEY=.' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\r\"'" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')}"
AUTH_ARGS=()
if [ -n "$API_KEY" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer $API_KEY")
fi

PASS=0
FAIL=0

check() {
    local name="$1" cond="$2"
    if eval "$cond"; then
        echo "✅ $name"
        PASS=$((PASS+1))
    else
        echo "❌ $name"
        FAIL=$((FAIL+1))
    fi
}

post() {
    local path="$1" body="$2"
    local code
    code=$(curl -s -o /tmp/llmlocal_body.json -w "%{http_code}" \
        -H "Content-Type: application/json" "${AUTH_ARGS[@]}" -d "$body" "${BASE_URL}${path}")
    echo "$code"
}

PY="${PYTHON:-python}"

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

# _json_get <file> <python expr over d>  -> prints the value as text
_json_get() {
    "$PY" -c 'import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    v = eval(sys.argv[2], {"d": d})
    print("" if v is None or v is False else v)
except Exception:
    pass' "$1" "$2"
}

echo "🚀 Testing OpenAI Compatible Local LLM Server ($BASE_URL, model=$MODEL)"
echo "=============================================="

# Test 1: Basic Health Check
echo "1. Health Check"
CODE=$(curl -s -o /tmp/llmlocal_health.txt -w "%{http_code}" "$BASE_URL/health")
check "health returns 200" '[ "$CODE" = "200" ]'
check "health body is OK" 'grep -q "OK" /tmp/llmlocal_health.txt'

# Test 2: List Models
echo "2. List Available Models"
CODE=$(curl -s -o /tmp/llmlocal_models.json -w "%{http_code}" "${AUTH_ARGS[@]}" "$BASE_URL/v1/models")
check "models endpoint returns 200" '[ "$CODE" = "200" ]'
check "models entry has an id" '_json_ok /tmp/llmlocal_models.json '"'"'d["data"][0]["id"]'"'"''
echo "   → $(_json_get /tmp/llmlocal_models.json 'd["data"][0]')"

# Test 3: Basic Text Completion
echo "3. Basic Text Completion"
CODE=$(post /v1/completions "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The future of artificial intelligence is\",
    \"max_tokens\": 50,
    \"temperature\": 0.7
}")
check "completion returns 200" '[ "$CODE" = "200" ]'
check "completion has choices" '_json_ok /tmp/llmlocal_body.json '"'"'d["choices"][0]'"'"''
check "completion echoes configured model" '[ "$(_json_get /tmp/llmlocal_body.json '"'"'d.get("model")'"'"')" = "$MODEL" ]'

# Test 4: Text Completion with Advanced Parameters
echo "4. Text Completion with Advanced Parameters"
CODE=$(post /v1/completions "{
    \"model\": \"$MODEL\",
    \"prompt\": \"Once upon a time in a distant galaxy\",
    \"max_tokens\": 100,
    \"temperature\": 0.8,
    \"top_p\": 0.9,
    \"stop\": [\"\\n\", \"The end\"]
}")
check "advanced completion returns 200" '[ "$CODE" = "200" ]'
check "advanced completion has usage" '_json_ok /tmp/llmlocal_body.json '"'"'d.get("usage", {}).get("total_tokens", 0) > 0'"'"''

# Test 5: Chat Completion with System Prompt
echo "5. Chat Completion with System Prompt"
CODE=$(post /v1/chat/completions "{
    \"model\": \"$MODEL\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a helpful coding assistant. Provide clear and concise answers.\"},
        {\"role\": \"user\", \"content\": \"How do I create a Python function?\"}
    ],
    \"max_tokens\": 150,
    \"temperature\": 0.5
}")
check "chat returns 200" '[ "$CODE" = "200" ]'
check "chat has assistant message" '_json_ok /tmp/llmlocal_body.json '"'"'d["choices"][0].get("message")'"'"''

# Test 6: Multi-turn Conversation
echo "6. Multi-turn Conversation"
CODE=$(post /v1/chat/completions "{
    \"model\": \"$MODEL\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a friendly assistant.\"},
        {\"role\": \"user\", \"content\": \"Hello! How are you today?\"},
        {\"role\": \"assistant\", \"content\": \"Hello! I'm doing well, thank you for asking.\"},
        {\"role\": \"user\", \"content\": \"Can you tell me a short joke?\"}
    ],
    \"max_tokens\": 100,
    \"temperature\": 0.7
}")
check "multi-turn returns 200" '[ "$CODE" = "200" ]'
check "multi-turn has content field" '_json_ok /tmp/llmlocal_body.json '"'"'d["choices"][0].get("message", {}).get("content") is not None'"'"''

# Test 7: Creative Writing with High Temperature
echo "7. Creative Writing (High Temperature)"
CODE=$(post /v1/chat/completions "{
    \"model\": \"$MODEL\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a creative writer.\"},
        {\"role\": \"user\", \"content\": \"Write a short story about a robot learning to paint.\"}
    ],
    \"max_tokens\": 200,
    \"temperature\": 1.2,
    \"top_p\": 0.95
}")
check "creative writing returns 200" '[ "$CODE" = "200" ]'

# Test 8: Code Generation
echo "8. Code Generation"
CODE=$(post /v1/chat/completions "{
    \"model\": \"$MODEL\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a Python programming expert.\"},
        {\"role\": \"user\", \"content\": \"Write a Python function to calculate fibonacci numbers.\"}
    ],
    \"max_tokens\": 150,
    \"temperature\": 0.3
}")
check "code generation returns 200" '[ "$CODE" = "200" ]'

# Test 9: Structured Output Request
echo "9. Structured Output Request"
CODE=$(post /v1/chat/completions "{
    \"model\": \"$MODEL\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a helpful assistant that provides structured information.\"},
        {\"role\": \"user\", \"content\": \"List 3 benefits of renewable energy.\"}
    ],
    \"max_tokens\": 100,
    \"temperature\": 0.5
}")
check "structured output returns 200" '[ "$CODE" = "200" ]'

# Test 10: Streaming (SSE)
echo "10. Streaming Chat"
STREAM_OUT=$(curl -s -N -H "Content-Type: application/json" "${AUTH_ARGS[@]}" -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one short sentence.\"}],
    \"max_tokens\": 64,
    \"stream\": true
}" "${BASE_URL}/v1/chat/completions")
check "stream emits data: chunks" 'echo "$STREAM_OUT" | grep -q "^data: "'
check "stream ends with [DONE]" 'echo "$STREAM_OUT" | grep -q "data: \[DONE\]"'

# Test 11: Per-request thinking toggle (ignored by the legacy backend)
echo "11. Thinking toggle"
CODE=$(post /v1/chat/completions "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Answer with just the number.\"}],
    \"max_tokens\": 512,
    \"thinking\": false
}")
check "thinking:false returns 200" '[ "$CODE" = "200" ]'
check "thinking:false has content" '_json_ok /tmp/llmlocal_body.json '"'"'d["choices"][0].get("message", {}).get("content")'"'"''

# Test 12: Error handling — invalid JSON
echo "12. Error Handling (malformed JSON)"
CODE=$(curl -s -o /tmp/llmlocal_body.json -w "%{http_code}" \
    -H "Content-Type: application/json" "${AUTH_ARGS[@]}" -d '{not json' "${BASE_URL}/v1/chat/completions")
check "malformed JSON returns 400" '[ "$CODE" = "400" ]'
check "error uses OpenAI shape" '_json_ok /tmp/llmlocal_body.json '"'"'d.get("error", {}).get("message")'"'"''

# Test 13: Error handling — missing messages
echo "13. Error Handling (missing messages)"
CODE=$(post /v1/chat/completions "{\"model\": \"$MODEL\"}")
check "missing messages returns 400" '[ "$CODE" = "400" ]'
check "error message mentions messages" '_json_get /tmp/llmlocal_body.json '"'"'d.get("error", {}).get("message", "")'"'"' | grep -qi "messages"'

# Test 14: Unknown model is not rejected (model field is informational)
echo "14. Unknown model field"
CODE=$(post /v1/completions "{
    \"model\": \"invalid-model\",
    \"prompt\": \"Say OK.\",
    \"max_tokens\": 16
}")
check "unknown model still returns 200" '[ "$CODE" = "200" ]'

echo "=============================================="
echo "✅ Passed: $PASS  ❌ Failed: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "Some tests failed"
    exit 1
fi
echo "🎉 All tests passed — the server is OpenAI API compatible"
