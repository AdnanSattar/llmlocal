# MIGRATION PLAN — llama.cpp backend for llmlocal

What changed, why, in which order, and how to roll back. Companion documents:
`INVESTIGATION.md` (audit + recommendation), `benchmarks/RESULTS.md` (evidence),
`MODEL_COMPARISON.md` (model data).

---

## 1. Goals / non-goals

**Goals**

1. Serve a CPU-feasible reasoning LLM (chain-of-thought with a per-request
   on/off switch) through the existing OpenAI-compatible API.
2. Keep the cloud/LAN worker architecture byte-compatible: `/jobs`,
   `/jobs/next`, `/jobs/{id}`, `/webhooks/completions`, `/workers/heartbeat`,
   `/queue/metrics` request and response shapes are unchanged; `worker.sh` is
   unchanged.
3. Keep `/v1/completions` and `/v1/chat/completions` working for existing
   clients.
4. Retain the original transformers runtime as a rollback path
   (`LLM_BACKEND=transformers`).
5. Fix the correctness bugs found in the audit (INVESTIGATION §2) that directly
   affect API consumers.

**Non-goals**

- No auth system, no persistence of the job queue, no multi-model hot-swap,
  no GPU scheduling. No git commits (per instructions).
- Docker remains optional/legacy; the primary runtime is native
  (llama-server as a subprocess), because Windows containers here would add a
  WSL2 VM for zero CPU benefit.

---

## 2. Architecture: before → after

**Before**

```
server.py ──► transformers pipeline("text2text-generation", flan-t5-small)   (single-threaded HTTP)
```

**After**

```
                     ┌─ backend/llama_cpp.py ──► llama-server.exe (child process, 127.0.0.1:8100, GGUF)
server.py (threads) ─┤
                     └─ backend/transformers_legacy.py ──► transformers pipeline (original behavior + bug fixes)
```

`server.py` keeps the HTTP surface, validation, queue and breaker; the
`backend/` package owns model lifecycle and generation:

| File | Role |
| --- | --- |
| `backend/base.py` | `Backend` interface (`start/health/complete/stream/shutdown`) + `BackendError` |
| `backend/llama_cpp.py` | spawns/health-checks `llama-server`, proxies non-stream + SSE requests, maps `thinking` → `chat_template_kwargs.enable_thinking`, rewrites the `model` field |
| `backend/transformers_legacy.py` | the original pipeline behavior (prompt wrapper, sampling knobs) with documented fixes; uses `_Text2TextShim` because transformers v5 removed the `text2text-generation` pipeline task |
| `backend/__init__.py` | env-driven factory + `.env` loader + port/limit helpers |

---

## 3. Configuration migration

| Old (still accepted) | New | Default | Purpose |
| --- | --- | --- | --- |
| — | `LLM_BACKEND` | `llama_cpp` | backend selection (`transformers` = rollback) |
| `TRANSFORMERS_MODEL` (ignored before, now honored) | same | `google/flan-t5-small` | legacy model |
| — | `LLM_MODEL` | `Qwen3-1.7B-Q4_K_M` | id echoed in responses/`/v1/models` (benchmark pick) |
| — | `LLM_MODEL_PATH` | `models/gguf/…gguf` | GGUF file |
| `VLLM_PORT` | `LLM_PORT` (alias: `VLLM_PORT`) | `8000` | API port |
| — | `LLM_BACKEND_PORT` | `8100` | internal llama-server port (loopback) |
| — | `LLM_CONTEXT_SIZE` / `LLM_THREADS` | `8192` / `4` | llama-server sizing |
| — | `LLM_THINKING` / `LLM_REASONING_FORMAT` / `LLM_REASONING_BUDGET` | `auto` / `deepseek` / – | reasoning controls |
| — | `LLM_MAX_TOKENS` / `LLM_MAX_TOKENS_CAP` | `1024` / `4096` | default + cap (default raised from 256: thinking spends the same budget — see B10) |
| `RUN_BACKEND` | unchanged | `transformers` | Docker profile selector |

`.env.example` documents everything; `.env` keeps working as-is (VLLM_* aliases).

---

## 4. Deliberate behavior changes

Bug fixes (audit IDs from INVESTIGATION §2):

| ID | Change |
| --- | --- |
| B1 | chat answers are no longer character-sliced (`result[len(prompt):]` removed) |
| B2/A2 | `model` field = configured `LLM_MODEL` everywhere (was `gpt2` / hard-coded FLAN); `/v1/models` built from config |
| B3 | HTTP server is threaded — a `/jobs/next` long-poll no longer freezes `/v1/*` and `/health` (legacy pipeline serialized behind a lock) |
| B4/A4 | `stream: true` supported on both `/v1` endpoints (SSE + `data: [DONE]`); previously silently ignored |
| B5 | chat budget is `max_new_tokens` (was word-count `max_length`) |
| B6 | malformed/missing bodies → HTTP 400/411 with OpenAI-shaped error (was connection reset) |
| B7 | stop sequences applied once |
| B8 | legacy `min_tokens` default 12 preserved for `/v1/completions` only (documented) |
| B9 | `usage` = tokenizer counts (was whitespace) |
| A1 | errors are `{"error":{"message","type","code"}}` with correct status codes |
| A5 | `max_tokens` cap raised 256 → `LLM_MAX_TOKENS_CAP` (4096); required for reasoning output |
| B10 | `max_tokens` default raised 256 → `LLM_MAX_TOKENS` (1024): hybrid reasoning models spend the request budget on thinking, so at 256 a client omitting `max_tokens` got an empty/truncated answer (observed in benchmark pass 1 — 11/16 prompts ended `finish=length` with no content) |
| A3 | `/v1/models` returns full OpenAI model objects |
| O1 | full payload logging replaced by one-line request summaries |
| O5 | `run.sh` no longer prints an `HF_TOKEN` prefix; failure path logs the container that actually failed |
| O3 | compose healthcheck probes the fixed in-container port 8000 |
| O8 | `tools/` and `logs/` added to `.gitignore`; `.dockerignore` added |
| NEW | transformers v5 dropped the `text2text-generation` pipeline task — the legacy backend now loads the seq2seq model directly (`_Text2TextShim`, tokenizer-truncated at 512 source tokens) with the same call contract |

New capabilities:

- `reasoning_content` on chat responses (chain-of-thought kept out of `content`)
  when `LLM_BACKEND=llama_cpp` and the model has hybrid thinking.
- Per-request `"thinking": true|false` (chat endpoint; mapped to
  `chat_template_kwargs.enable_thinking`; ignored by the legacy backend).
- Native startup: `python server.py` / `./scripts/run.sh` needs no Docker.
- `.env` is loaded automatically by `server.py` (no dependency).

Unchanged on purpose:

- All queue/worker/breaker endpoints, tunables, and `worker.sh`.
- `/v1/completions` prompt wrapper + sampling knobs **in the legacy backend**
  (faithful rollback).
- Port 8000 default, `/health` semantics (200 `OK` / 503 + detail).
- The `model` request field stays informational (never rejected) — matches the
  previous behavior that existing clients rely on.

---

## 5. Rollout steps (executed in this order)

1. **Evidence first**: download runtime + candidate GGUFs, smoke-test each
   model in llama-server (arch support, reasoning separation), run the
   benchmark suite (`benchmarks/`) — see RESULTS.md.
2. **Backend package**: `backend/` with the interface, llama_cpp and legacy
   implementations (no behavior change until wired).
3. **`server.py` refactor**: swap generation call sites to the backend, add
   threading, validation, SSE, OpenAI errors; leave queue code untouched.
4. **Config & scripts**: `.env.example`, `scripts/run.sh`,
   `scripts/test_api.sh`, `docker-compose.yml`, `Dockerfile`, `.gitignore`,
   `.dockerignore`.
5. **Docs**: README, INVESTIGATION, MODEL_COMPARISON, MIGRATION_PLAN, RESULTS.
6. **Verification**: syntax checks, `./scripts/test_api.sh` against the new server
   (all 25 checks), endpoint spot-checks (streaming, thinking, errors), queue
   smoke (`/jobs` → `/jobs/next` → `/webhooks/completions` → `/jobs/{id}`),
   legacy rollback start (`LLM_BACKEND=transformers`).

---

## 6. Rollback

Level 1 — configuration only (no code change):

```bash
# .env
LLM_BACKEND=transformers
```

Restores the original pipeline runtime through the same (fixed) API surface.
The legacy backend reproduces the original prompt wrapper, sampling knobs and
`min_tokens` behavior; only the documented bug fixes differ.

Level 2 — full revert: the migration was not committed until requested;
`git checkout -- server.py` (etc.) or discarding the working tree restores the
original code at any time.

Operational:

- llama-server logs: `logs/llama-server.log` (also mirrored by
  `LLM_LOG`); if the child dies, `/health` → 503 with the exit reason, and the
  backend can be restarted by restarting `server.py` (no auto-respawn loop —
  fail loudly instead of flapping).
- Port conflicts: API `LLM_PORT` (8000), llama-server `LLM_BACKEND_PORT`
  (8100, loopback only), benchmarks use 8099.
- Disk: `models/gguf/` holds the model zoo (~10 GB) and `tools/llama.cpp/`
  the runtime (~30 MB); both are gitignored and safe to delete — only the
  model referenced by `LLM_MODEL_PATH` is needed at runtime.

---

## 7. Verification checklist (run before calling the migration done)

- [x] `python -m py_compile server.py backend/*.py` passes
- [x] `./scripts/test_api.sh` — all 25 checks green against the llama_cpp backend
- [x] `./scripts/test_api.sh` — all 25 checks green with `LLM_BACKEND=transformers` (rollback; verified 2026-10-08 — the legacy path serves through `_Text2TextShim` on transformers v5)
- [x] streaming: `stream: true` returns SSE chunks + `[DONE]`
- [x] reasoning: `reasoning_content` present, `content` clean of think tags
- [x] `thinking: false` shortens/omits reasoning
- [x] queue round-trip: enqueue → long-poll → webhook → status
- [x] `/jobs/next` long-poll does not block `/health` (200 in 41 ms during an
      active 25 s long-poll — the B3 fix)
- [x] benchmark suite results written under `benchmarks/results/` (a git
      commit is left to the user — nothing was committed)
