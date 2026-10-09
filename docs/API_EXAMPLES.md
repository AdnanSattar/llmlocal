# OpenAI Compatible Local LLM Server — `/v1` API

OpenAI-style examples for the local server. The `model` field is
informational: responses echo the model configured in `.env` (`LLM_MODEL`,
default `Qwen3-1.7B-Q4_K_M` with the llama.cpp backend). Endpoints:
`/v1/completions`, `/v1/chat/completions` (both support `stream: true`).

## Base URL

Replace `localhost:8000` with your server address if different.

---

## Interactive docs (Swagger / ReDoc / Postman)

Prefer clicking to curling? The server ships its own references:

- **`GET /docs`** — Swagger UI. Has an **Authorize** button for the
  `Authorization: Bearer <API_KEY>` header (paste the `.env` `API_KEY`) and a
  pre-filled request-body editor on every POST route, so each endpoint can be
  tried straight from the browser.
- **`GET /redoc`** — ReDoc reference (read-only).
- **`GET /openapi.json`** — the raw OpenAPI schema.
- **`docs/llmlocal.postman_collection.json`** — a Postman collection covering
  every endpoint; set its `base_url`, `api_key` and `worker_token` variables.

These three HTTP routes are public (no API key) — they expose endpoint shapes,
not data.

---

## Authentication

When `API_KEY` is set in `.env`, every endpoint except `GET /health` requires
it (full policy table below). Send the key as a Bearer token in the
`Authorization` header:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"What is 17 × 23?"}],"thinking":true,"max_tokens":1024}'
```

### 401 response (missing == invalid)

A missing key and an invalid key return the *same* status and body — there is
no way to tell from the response whether a given key exists:

```json
{
  "error": {
    "message": "Invalid or missing API key",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

### Rules

- The key is accepted **only** from the `Authorization` header. Keys passed in
  query strings (`/v1/models?api_key=...`) are never accepted — a *correct*
  key passed that way still gets a 401.
- Scheme is case-insensitive (`Bearer`, `bearer`, `BEARER`); extra whitespace
  around the header value is tolerated. Non-Bearer schemes (`Basic …`) are
  treated as a missing key.
- **Development mode:** when `API_KEY` is empty/unset in `.env`, API-key auth
  is disabled and the protected routes are open. Do NOT run this way on an
  exposed port.
- Keys are compared in constant time, never logged, and never echoed in error
  messages or tracebacks.

### API key vs worker token

Two independent authentication worlds — neither substitutes for the other:

| Endpoint | Auth |
|---|---|
| GET /health | public |
| GET /v1/models, POST /v1/completions, POST /v1/chat/completions | API key |
| POST /jobs, GET /jobs/{id}, GET /queue/metrics | API key |
| POST /jobs/next, POST /workers/heartbeat | WORKER_TOKEN (existing `worker_token` query param) — independent of API key |
| POST /webhooks/completions | unchanged today (worker.sh posts none) — capability = unguessable job UUID; known design point |

- `API_KEY` empty/unset → `/v1` and job/queue routes are open (development
  mode). `WORKER_TOKEN` stays enforced on worker endpoints regardless of
  `API_KEY`: a valid API key does **not** unlock `/jobs/next` or
  `/workers/heartbeat` (still 403 without the right `worker_token`), and the
  worker token is not accepted as an API key.

---

## 🚀 Quick Start

### Basic Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 50
  }'
```

---

## 📚 Complete API Examples

### 1. Basic Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### 2. With Temperature Control

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "Once upon a time in a distant galaxy",
    "max_tokens": 150,
    "temperature": 0.8
  }'
```

### 3. Advanced Parameters

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "The secret to happiness is",
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["\n", "The end"]
  }'
```

### 4. Creative Writing (High Temperature)

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "In a world where robots have emotions",
    "max_tokens": 250,
    "temperature": 1.2,
    "top_p": 0.95
  }'
```

### 5. Code Generation (Low Temperature)

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "def fibonacci(n):",
    "max_tokens": 200,
    "temperature": 0.3
  }'
```

### 6. Question Answering

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "Question: What is machine learning?\nAnswer:",
    "max_tokens": 150,
    "temperature": 0.6
  }'
```

### 7. Translation Task

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "English: Hello, how are you?\nFrench:",
    "max_tokens": 50,
    "temperature": 0.2
  }'
```

### 8. Story Continuation

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "The detective entered the dark room and found",
    "max_tokens": 200,
    "temperature": 0.9
  }'
```

---

## 🔧 Supported Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | — | Informational; the response echoes the configured model |
| `prompt` | string | required | Text to complete |
| `system` | string | — | Optional instruction: prepended to completions / injected as a system message in chat (see note at the end) |
| `max_tokens` | integer | 1024 | Maximum tokens to generate (hard cap `LLM_MAX_TOKENS_CAP`, default 4096); reasoning models spend part of this on thinking — give them room |
| `temperature` | float | 0.7 | Controls randomness (0.0+) |
| `top_p` | float | 1.0 | Nucleus sampling (0.0-1.0] |
| `stop` | string/array | null | Stop sequences |
| `min_tokens` | integer | 12 | Minimum generated tokens (legacy/transformers backend only) |
| `stream` | bool | false | SSE streaming (`data:` chunks + `data: [DONE]`) |

---

## 📋 Response Format

Your server returns responses in the exact OpenAI API format:

```json
{
  "id": "cmpl-123456789012345678901",
  "object": "text_completion",
  "created": 1677858242,
  "model": "Qwen3-1.7B-Q4_K_M",
  "choices": [
    {
      "text": "Generated text here...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 10,
    "total_tokens": 15
  }
}
```

---

## 🧪 Testing

### Health Check

```bash
curl http://localhost:8000/health
```

### List Models

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

### Run Test Suite

```bash
chmod +x scripts/test_api.sh
./scripts/test_api.sh
```

---

## 💬 Chat, streaming and reasoning

### Chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "system", "content": "Answer briefly."},
      {"role": "user", "content": "What is an API?"}
    ],
    "max_tokens": 512
  }'
```

### Streaming (SSE, both endpoints)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
    -d '{"model": "local", "messages": [{"role": "user", "content": "Hello!"}], "stream": true, "max_tokens": 256}'
```

### Reasoning models (chain-of-thought)

With the llama.cpp backend and a hybrid reasoning model (Qwen3/Qwen3.5,
SmolLM3), the chain-of-thought comes back in `message.reasoning_content`
while `message.content` holds the final answer:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "How many primes below 50?"}],
    "max_tokens": 2048,
    "thinking": true
  }'
```

Use `"thinking": false` to skip reasoning for a single request. Give
reasoning prompts enough `max_tokens` (thinking consumes part of the budget).

---

## 💡 Usage Tips

### Temperature Guidelines:

- **0.0-0.3**: Factual, deterministic responses
- **0.4-0.7**: Balanced creativity and accuracy
- **0.8-1.2**: Creative, diverse responses
- **1.3+**: Very creative, potentially incoherent

### Prompt Engineering:

- Be specific and clear in your prompts
- Use examples when possible
- Include context for better results
- Use stop sequences to control output length

### Performance:

- Lower `max_tokens` = faster responses
- Higher `temperature` = more creative but slower
- Use `stop` sequences to end generation early

---

## 🎯 Integration Examples

### Python

```python
import os
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    headers={"Authorization": f"Bearer {os.environ['API_KEY']}"},
    json={
        "model": "local",
        "prompt": "Hello, how are you?",
        "max_tokens": 100,
        "temperature": 0.7
    }
)
print(response.json())
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/v1/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`
  },
  body: JSON.stringify({
    model: 'local',
    prompt: 'Hello, how are you?',
    max_tokens: 100,
    temperature: 0.7
  })
});
const data = await response.json();
console.log(data);
```

---

## 🎉 Your server is ready!

You now have a **fully OpenAI-compatible** local LLM server that works with any application expecting the standard `/v1/completions` API.

**Happy coding!** 🚀

---

## ➕ Additional `/v1/completions` Examples (with system and summarizer)

### Warmup (fast, short)

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "prompt": "warmup",
    "max_tokens": 8
  }'
```

### Completion with System Prompt (recommended)

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a helpful assistant that responds briefly and clearly.",
    "prompt": "Explain what an API is.",
    "max_tokens": 120,
    "temperature": 0.5
  }'
```

### Creative Writing (High Temperature)

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a creative fiction writer.",
    "prompt": "In a city of glass towers, a courier discovers a secret.",
    "max_tokens": 250,
    "temperature": 1.2,
    "top_p": 0.95
  }'
```

### Constrained Output with Stop Sequences

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a helpful assistant.",
    "prompt": "List three benefits of unit testing:",
    "max_tokens": 80,
    "temperature": 0.4,
    "stop": ["\n", "User:"]
  }'
```

### Summarizer (Concise 2–3 sentences)

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You are a professional summarizer. Produce a concise, factual summary in 2-3 sentences. Avoid repetition and opinions.",
    "prompt": "Summarize: Artificial intelligence (AI) refers to computer systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Recent advances in machine learning and large datasets have accelerated AI capabilities across industries, from healthcare to finance, enabling automation and improved decision-making.",
    "max_tokens": 120,
    "temperature": 0.3,
    "top_p": 0.9,
    "stop": ["\n\n"]
  }'
```

### Summarizer (Bulleted key points)

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "system": "You extract key points as short bullet items. No preamble, no conclusion.",
    "prompt": "Key points: Artificial intelligence (AI) refers to computer systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Recent advances in machine learning and large datasets have accelerated AI capabilities across industries, from healthcare to finance, enabling automation and improved decision-making.",
    "max_tokens": 120,
    "temperature": 0.35,
    "stop": ["\n\n"]
  }'
```

### Note on `system` field

- Optional top-level `system` field, applied server-side:
  - **llama.cpp backend** (default): on `/v1/completions` the text is prepended
    to the prompt; on `/v1/chat/completions` it is injected as a
    `{"role": "system"}` message (only if the messages don't already contain
    one — an explicit system message always wins).
  - **transformers (legacy) backend**: wrapped into the original
    `Instruction: …\nQuestion: …\nAnswer:` template (original behavior).
- On chat, prefer putting the instruction in the messages array — it's the
  standard OpenAI shape.
- For more factual outputs, use lower temperatures (0.2–0.5) and provide clear instructions.
