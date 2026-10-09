# llmlocal — OpenAI-compatible local LLM API (CPU-first)

`server.py` exposes **OpenAI-compatible `/v1` endpoints** over a locally hosted
model, plus a reverse-webhook **job queue** for cloud/LAN workers. Inference runs
through a pluggable backend:

| Backend | `LLM_BACKEND=` | What it is |
| --- | --- | --- |
| **llama.cpp (default, recommended)** | `llama_cpp` | official `llama-server` subprocess serving a GGUF model — fast, quantized, native Windows, streaming, chain-of-thought support |
| Transformers (legacy/rollback) | `transformers` | the original HuggingFace `flan-t5-small` pipeline (fp32 CPU) |

Model investigation, benchmark results and migration details (all under
`docs/`): `INVESTIGATION.md`, `MODEL_RESEARCH.md`, `MODEL_REEVALUATION.md`,
`MODEL_COMPARISON.md`, `MIGRATION_PLAN.md`, `API_EXAMPLES.md`; local numbers
in `benchmarks/RESULTS.md`. A ready-made Postman collection lives at
`docs/llmlocal.postman_collection.json`.

---

## Quick start (native Windows/Linux — recommended)

Prerequisites: Python 3.10+, ~4 GB free disk per model.

```bash
# 1. configure
cp .env.example .env          # set LLM_MODEL / LLM_MODEL_PATH / tokens

# 2. runtime: llama.cpp release (Windows CPU zip) extracted into tools/llama.cpp/
#    https://github.com/ggml-org/llama.cpp/releases  →  llama-b*-bin-win-cpu-x64.zip

# 3. model: a Q4_K_M GGUF into models/gguf/ (see .env.example for the default)
#    e.g. from Qwen, ggml-org, unsloth or bartowski on Hugging Face

# 4. run
./scripts/run.sh              # or: python server.py

# 5. verify
./scripts/test_api.sh
```

Health & models:

```bash
curl http://localhost:8000/health                        # always public
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"                    # when API_KEY is set
```

Chat completion (OpenAI-style):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

---

## Reasoning models (chain-of-thought)

Hybrid reasoning models (Qwen3/Qwen3.5, SmolLM3) think before answering. With
the default `LLM_REASONING_FORMAT=deepseek` the reasoning is returned in a
separate field and never leaks into the answer:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "How many primes below 50?"}],
    "max_tokens": 2048,
    "thinking": true
  }'
```

```jsonc
// response.choices[0].message
{
  "reasoning_content": "Let me list them... 2, 3, 5, 7, ...  (the chain of thought)",
  "content": "There are 15 primes below 50."
}
```

- `"thinking": false` turns thinking off for that request (mapped to the
  model's `chat_template_kwargs.enable_thinking`).
- Streaming works on both endpoints: add `"stream": true` (SSE,
  `data: {...}` chunks ending with `data: [DONE]`).
- Server-wide defaults: `LLM_THINKING=auto|on|off`,
  `LLM_REASONING_FORMAT=deepseek|none`, optional `LLM_REASONING_BUDGET`.

---

## Configuration (`.env`)

Key variables (full list with comments in `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_BACKEND` | `llama_cpp` | `llama_cpp` \| `transformers` |
| `LLM_MODEL` | `Qwen3-1.7B-Q4_K_M` | id reported by `/v1/models` and in responses (benchmark pick — see `docs/MODEL_COMPARISON.md`) |
| `LLM_MODEL_PATH` | `models/gguf/Qwen3-1.7B-Q4_K_M.gguf` | GGUF file |
| `LLM_PORT` | `8000` | public API port (`VLLM_PORT` accepted as alias) |
| `LLM_BACKEND_PORT` | `8100` | internal llama-server port (127.0.0.1 only) |
| `LLM_CONTEXT_SIZE` | `8192` | context window |
| `LLM_THREADS` | `4` | CPU threads (`0` = auto) |
| `LLM_MAX_TOKENS` / `LLM_MAX_TOKENS_CAP` | `1024` / `4096` | default + hard cap for `max_tokens` |
| `LLM_THINKING` | `auto` | default thinking mode for reasoning models |
| `API_KEY` | _(empty)_ | required `Authorization: Bearer …` key for `/v1/*` and job endpoints; empty = dev mode (open) |
| `TRANSFORMERS_MODEL` | `google/flan-t5-small` | legacy backend model |

Rollback to the original runtime without touching anything else:

```bash
# .env
LLM_BACKEND=transformers
```

The legacy backend is optional and imported lazily: it needs `transformers`
and a PyTorch build installed (`pip install transformers torch`) on top of the
web dependencies in `requirements.txt`, which is why it is not in the default
install.

---

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | 200 `OK` when the model is ready, 503 otherwise |
| `GET /v1/models` | OpenAI-style model list |
| `POST /v1/completions` | text completion (supports `stream`) |
| `POST /v1/chat/completions` | chat completion (supports `stream`, `thinking`) |
| `POST /jobs` | enqueue a job (reverse-webhook queue) |
| `POST /jobs/next?worker_token=…` | worker long-poll (25 s) |
| `GET /jobs/{job_id}` | job status/result |
| `POST /webhooks/completions` | worker posts `{job_id, result, success}` |
| `POST /workers/heartbeat?worker_token=…` | worker heartbeat + breaker state |
| `GET /queue/metrics` | queue depth/age + circuit breaker state |

**Authentication:** when `API_KEY` is set in `.env`, every `/v1/*` endpoint,
`POST /jobs`, `GET /jobs/{id}` and `GET /queue/metrics` require
`Authorization: Bearer <API_KEY>` (missing and invalid keys get the same 401).
`GET /health` stays public. Worker endpoints (`/jobs/next`,
`/workers/heartbeat`) use the separate `WORKER_TOKEN` query parameter. With
`API_KEY` empty/unset the server runs in dev mode (routes open; only the mode
is logged at startup, never the key).

**Interactive docs:** `/docs` (Swagger UI), `/redoc` and `/openapi.json` are
served unauthenticated. Swagger exposes an **Authorize** button for the
`BearerAuth` scheme and a pre-filled request-body editor on every POST route,
so the whole surface can be exercised from the browser (see the note in
[Security notes](#security-notes) on turning docs off).

Errors use the OpenAI shape: `{"error": {"message", "type", "code"}}`.
The `model` field in requests is informational (echoed responses always carry
the configured `LLM_MODEL`).

### Reverse-webhook queue (unchanged)

```
cloud/LAN client ──POST /jobs──► server.py queue ──long-poll──► worker.sh
     ▲                               │                             │
     └────── GET /jobs/{id} ◄────────┘◄──POST /webhooks/completions─┘
```

The circuit breaker (`CLOSED/OPEN/HALF_OPEN`) opens when workers go stale, the
queue ages out, or jobs fail repeatedly — see tunables in `.env.example`.

---

## Docker path (optional)

CPU-only container running the **llama.cpp** backend (pinned llama-b11438 —
the same build as native). Base image `ubuntu:24.04`; Python deps
(FastAPI/uvicorn from `requirements.txt`) are installed in the image:

```bash
docker compose build             # ubuntu:24.04 + pinned llama.cpp CPU release
docker compose up -d             # default service: llama_cpp (port 8000)
./scripts/run.sh --docker        # build + start + healthcheck + warmup

RUN_BACKEND=transformers ./scripts/run.sh --docker   # legacy rollback path (profile)
RUN_BACKEND=vllm ./scripts/run.sh --docker           # GPU vLLM profile (hardware required)
```

- **host port**: `LLM_PORT` (default 8000). Only port 8000 is published — the
  internal llama-server port 8100 stays bound to 127.0.0.1 inside the
  container (in-container healthcheck probes the fixed internal port 8000).
- **model is mounted, not baked**: `./models/gguf → /app/models/gguf:ro`.
  Required file: `models/gguf/Qwen3-1.7B-Q4_K_M.gguf` (gitignored; the image
  stays model-independent and small).
- **no secrets in the image**: `HF_TOKEN` is not passed (the mounted GGUF
  needs no download); `API_KEY` and `WORKER_TOKEN` come from your `.env` at
  deploy time and are never baked into image layers.
- container: `llama_cpp_server`, logs: `docker compose logs -f llama_cpp`.

---

## Benchmarks

`benchmarks/` contains a reproducible local suite (16 prompts, llama-bench
pp512/tg64, TTFT/decode t/s, peak RSS, reasoning separation checks):

```bash
python benchmarks/benchmark_models.py --bench            # all models in models.json
python benchmarks/benchmark_models.py --only qwen3.5-4b  # one model
python benchmarks/benchmark_models.py --base http://localhost:8000 --label my-endpoint
```

Measured results and the model recommendation: `benchmarks/RESULTS.md`,
`docs/MODEL_COMPARISON.md`, `docs/MODEL_RESEARCH.md`,
`docs/MODEL_REEVALUATION.md`.

---

## Security notes

- `.env` holds secrets (`HF_TOKEN`, `WORKER_TOKEN`, `API_KEY`) — never commit
  it; rotate all three if the file was ever shared or committed.
- Set a long random `API_KEY` to guard `/v1/*` and the job endpoints; with it
  unset the API is open (dev mode) — bind to localhost/LAN only then, and keep
  `WORKER_TOKEN` non-default if workers are used across machines.
- The worker token travels in the query string (legacy behavior) — treat worker
  endpoints as LAN-trusted. `/docs`, `/redoc` and `/openapi.json` are enabled
  and unauthenticated (they expose endpoint schemas, not data); to turn them
  off again, set `docs_url=None, redoc_url=None, openapi_url=None` in the
  `FastAPI(...)` constructor in `app/main.py`.

---

## Repo layout

```
server.py              thin entrypoint (loads the FastAPI app, runs uvicorn)
app/                   FastAPI app: routes/, auth/ (API key), state, http
backend/               backend abstraction (llama_cpp | transformers)
requirements.txt       web dependencies (FastAPI/Starlette/uvicorn)
tools/llama.cpp/       llama-server runtime (downloaded, gitignored)
models/gguf/           GGUF weights (downloaded, gitignored)
logs/                  runtime logs, e.g. llama-server.log (gitignored)
benchmarks/            reproducible benchmark suite + measured results
scripts/               run.sh, worker.sh, test_api.sh, test_auth.sh, examples.sh
docs/                  investigation, migration plan, model research, API examples, Postman collection
.env.example           documented configuration template (copy to .env)
Dockerfile             optional CPU container image (ubuntu:24.04 + pinned llama.cpp)
docker-compose.yml     optional container path (llama_cpp default; transformers/vllm profiles)
```
