#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
# OpenAI-compatible /v1/completions examples
# Usage: BASE_URL=http://localhost:8000 ./scripts/examples.sh

BASE_URL=${BASE_URL:-http://localhost:8000}

# API key (read from .env, never echoed); header omitted when auth is disabled.
API_KEY="${API_KEY:-$(grep -E '^API_KEY=.' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\r\"'" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')}"
AUTH_ARGS=()
if [ -n "$API_KEY" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer $API_KEY")
fi

# 1) Basic completion
curl "$BASE_URL/v1/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "Hello, how are you?",
    "max_tokens": 80,
    "temperature": 0.7
  }'

echo -e "\n\n---\n"

# 2) With system prompt (recommended)
curl "$BASE_URL/v1/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a helpful assistant that responds briefly and clearly.",
    "prompt": "Explain what an API is.",
    "max_tokens": 120,
    "temperature": 0.5
  }'

echo -e "\n\n---\n"

# 3) Creative writing (high temperature)
curl "$BASE_URL/v1/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a creative fiction writer.",
    "prompt": "In a city of glass towers, a courier discovers a secret.",
    "max_tokens": 200,
    "temperature": 1.1,
    "top_p": 0.95
  }'

echo -e "\n\n---\n"

# 4) Constrained output with stop sequences
curl "$BASE_URL/v1/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a helpful assistant.",
    "prompt": "List three benefits of unit testing:",
    "max_tokens": 80,
    "temperature": 0.4,
    "stop": ["\\n", "User:"]
  }'

echo -e "\n\n---\n"

# 5) Low-variance (more deterministic)
curl "$BASE_URL/v1/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "Answer as precisely and concisely as possible.",
    "prompt": "What is the time complexity of binary search?",
    "max_tokens": 60,
    "temperature": 0.2,
    "top_p": 1.0
  }'

echo -e "\n\n---\n"

# 6) Summarizer (concise 2-3 sentences)
curl "$BASE_URL/v1/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a professional summarizer. Produce a concise, factual summary in 2-3 sentences. Avoid repetition and opinions.",
    "prompt": "Summarize: Artificial intelligence (AI) refers to computer systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Recent advances in machine learning and large datasets have accelerated AI capabilities across industries, from healthcare to finance, enabling automation and improved decision-making.",
    "max_tokens": 120,
    "temperature": 0.3,
    "top_p": 0.9,
    "stop": ["\\n\\n"]
  }'

echo -e "\n\n---\n"

# 7) Summarizer (bulleted key points)
curl "$BASE_URL/v1/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You extract key points as short bullet items. No preamble, no conclusion.",
    "prompt": "Key points: Artificial intelligence (AI) refers to computer systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Recent advances in machine learning and large datasets have accelerated AI capabilities across industries, from healthcare to finance, enabling automation and improved decision-making.",
    "max_tokens": 120,
    "temperature": 0.35,
    "stop": ["\\n\\n"]
  }'
