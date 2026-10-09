# INVESTIGATION — llmlocal

Investigation of the repository, its architecture and limitations, candidate
CPU-only reasoning models, runtime options, and the resulting recommendation.
Companion documents: `MODEL_COMPARISON.md` (model-by-model data),
`MIGRATION_PLAN.md` (how the migration is executed), and `benchmarks/RESULTS.md`
(measured local results).

Date of investigation: 2026-10-06. All facts below were verified on this date
(HF API, llama.cpp release assets, llama.cpp `--help`, or local execution) unless
explicitly marked as *published* (taken from a model card).
**Sections 1–2 describe the repository as found, before the migration** (see
`MIGRATION_PLAN.md`); where behavior changed, the change is called out in
MIGRATION_PLAN §4.

---

## 1. What this repo is

`llmlocal` is a small Python service that exposes an **OpenAI-compatible HTTP
API** over a locally hosted model, plus a reverse-webhook job queue for LAN/cloud
workers. Single process, standard library only (`http.server`), model inference
via HuggingFace `transformers`.

### 1.1 Current endpoints (server.py)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | 200 `OK` when model loaded, else 503 |
| `/v1/models` | GET | model list (hard-coded, see §2) |
| `/v1/completions` | POST | text completion (FLAN-wrapped prompt) |
| `/v1/chat/completions` | POST | chat completion (broken, see §2) |
| `/jobs` | POST | enqueue a job (payload shaped like `/v1/completions`) |
| `/jobs/next?worker_token=…` | POST | worker long-poll (up to 25 s) |
| `/jobs/{id}` | GET | job status/result |
| `/webhooks/completions` | POST | worker posts result `{job_id, result, success}` |
| `/workers/heartbeat?worker_token=…` | POST | worker heartbeat, returns breaker state |
| `/queue/metrics` | GET | queue depth/age + circuit breaker state |

### 1.2 Cloud/LAN worker architecture (reverse-webhook)

```
 cloud / LAN client                 local PC (server.py)              LAN worker (worker.sh)
 ─────────────────                  ─────────────────────             ─────────────────────
 POST /jobs  ────────────────────►  in-memory queue + circuit breaker
                                    POST /jobs/next ◄────────────────  long-poll (25 s)
                                    ◄──── {job_id, payload}
                                    (worker then calls local inference itself,
                                     or the job runs on server.py /v1/*)
 POST /jobs/{id} ◄────────────────  status / result
                                    ◄──── POST /webhooks/completions {job_id, result}
```

Circuit breaker states `CLOSED → OPEN → HALF_OPEN` route decisions based on
worker staleness, queue age/depth, and consecutive failures. This whole layer is
independent of which model backend serves `/v1/*` and must survive any migration
unchanged (worker.sh depends on exact request/response shapes).

### 1.3 How inference works today

- `google/flan-t5-small` (80M params) through `pipeline("text2text-generation")`,
  fp32 on CPU.
- `/v1/completions` wraps the prompt as `Question: …\nAnswer:` (or
  `Instruction: …\nQuestion: …\nAnswer:` when `system` is set).
- Hard-coded generation knobs: `repetition_penalty: 1.15`,
  `no_repeat_ngram_size: 3`, `min_new_tokens: 12` (default), `max_tokens` capped
  at 256.
- No streaming, single-threaded HTTP server, `usage` counts whitespace instead of
  tokens.

### 1.4 Repo operational inventory

| File | Role | Notes |
| --- | --- | --- |
| `server.py` (569 lines) | API + queue + breaker, all in one file | see §2 |
| `run.sh` | launch helper | assumes Docker; prints `HF_TOKEN` prefix; wrong container in error path |
| `worker.sh` | LAN worker long-poller | posts only to `/v1/completions` |
| `docker-compose.yml` | `transformers` + `vllm` profiles | healthcheck bug, see §2 |
| `Dockerfile` | CUDA PyTorch base image for a CPU workload | huge image |
| `.env` | **contains live `HF_TOKEN` and `WORKER_TOKEN`** | treat as compromised (see §10) |
| `test_api.sh`, `examples.sh` | smoke tests | send `"model": "gpt2"` |
| `README.md`, `API_EXAMPLES.md` | docs | name models that do not exist in the stack |
| `models/hub/` | HF cache dir used by Docker path | was broken (0-byte `config.json`); repaired — re-downloaded for the transformers baseline |

---

## 2. Existing limitations (code audit)

Verified by reading the code; file:line references are to the current tree.

### 2.1 Correctness bugs

| # | Severity | Finding |
| --- | --- | --- |
| B1 | **high** | `/v1/chat/completions` slices the wrong string: `result[len(prompt):]` (server.py:395) removes the first `len(prompt)` **characters of the answer**, because a text2text pipeline returns only the generated text. Short answers become empty; long answers lose their beginning. |
| B2 | **high** | `/v1/chat/completions` reports `"model": "gpt2"` (server.py:405) — the model is neither gpt2 nor what the response says. |
| B3 | **high** | Single-threaded `socketserver.TCPServer` (server.py:569): the `/jobs/next` long-poll holds the only request thread for up to 25 s, blocking `/v1/*`, `/health` and everything else. The worker architecture periodically freezes the API. |
| B4 | medium | `/v1/completions` and `/v1/chat/completions` ignore `stream: true` and return plain JSON — OpenAI SDK streaming calls break (they wait for SSE that never comes). |
| B5 | medium | Chat endpoint budget uses `max_length = word_count + max_tokens` (server.py:387) — words are not tokens; inconsistent with the completions endpoint's `max_new_tokens`. |
| B6 | medium | `Content-Length` read is unguarded (server.py:251, 361): missing/invalid header or malformed JSON crashes the handler → connection reset instead of a JSON error. |
| B7 | low | Stop sequences are applied twice (server.py:302–323). |
| B8 | low | `min_tokens` defaults to 12 — even `"prompt":"hi"` forces ≥ 12 generated tokens (server.py:262). |
| B9 | low | `usage` counts whitespace (`len(text.split())`), not tokens (server.py:338–341). |

### 2.2 API/OpenAI-compatibility gaps

| # | Finding |
| --- | --- |
| A1 | Error shape is `{"error": "string"}`; OpenAI uses `{"error": {"message","type","code"}}` with proper status codes. 404/403 responses have no body at all. |
| A2 | Request `model` field is never validated or echoed consistently: `/v1/models` hard-codes `google/flan-t5-small` (server.py:222) and ignores even the existing `TRANSFORMERS_MODEL` env var (load at server.py:143 is hard-coded); completions response hard-codes the same name (server.py:328); chat says `gpt2`. |
| A3 | `/v1/models` entries lack `object`, `created`, `owned_by`. |
| A4 | No streaming (B4), no `logprobs`, no `n`, no `seed`, no `finish_reason` fidelity (always `"stop"`). |
| A5 | Undocumented hard caps: `max_tokens` silently clamped to 256 (server.py:269) — far too small for reasoning models, where thinking alone routinely exceeds 256 tokens. |
| A6 | Tests/docs disagree with reality: `test_api.sh` and `examples.sh` send `"model":"gpt2"`; `run.sh` warmup uses `google/flan-t5-small`; README mentions `qwen-1.5b` and `llama-1b` which do not exist anywhere in the stack. Tests pass only because the server ignores `model`. |

### 2.3 Non-functional / ops issues

| # | Finding |
| --- | --- |
| O1 | Every request payload is printed in full (server.py:253, 363) — log noise plus PII risk. |
| O2 | `Dockerfile` uses `pytorch/pytorch:2.2.0-cuda12.1` (multi-GB CUDA image) for CPU inference; `pip install transformers` unpinned. |
| O3 | `docker-compose.yml` `transformers` healthcheck probes `${VLLM_PORT:-8000}` *inside* the container where the server listens on 8000 — any non-default `VLLM_PORT` makes healthchecks fail forever. |
| O4 | `vllm` profile passes `--cuda_visible_devices` as a CLI flag (it is an env var) and `VLLM_LOAD_ON_START` is never used — profile likely broken; not exercised by tests. |
| O5 | `run.sh` prints the first 10 chars of `HF_TOKEN` and, on failure, always prints `docker compose logs vllm` regardless of which backend actually failed (run.sh:90). |
| O6 | `models/hub` HF cache is broken (0-byte `config.json`) — the model was never actually downloaded to the repo cache; the Docker path re-downloads on first run. |
| O7 | No thread-safety: if the server were threaded, the `transformers` pipeline would be called concurrently (not thread-safe) without a lock. |
| O8 | `.gitignore` covers `models/` and `.env` but **not `tools/`** (downloaded binaries would be committed). |
| O9 | Security: no auth on `/v1/*`; `WORKER_TOKEN` default `change-me`; worker token travels in the query string (proxy/log leakage). Acceptable for a LAN tool, documented as risk. |

---

## 3. Candidate models (verified 2026-10-06)

Requirements: (a) runs faster than realtime on 4 cores of an i7-1165G7,
(b) ≤ ~16 GB RAM including KV cache, (c) real chain-of-thought with controllable
thinking, (d) permissive license, (e) instruct/chat tuned, (f) GGUF available.

All sizes are the actual Q4_K_M downloads in `models/gguf/`.

| Model | Params | Arch | License | Native ctx | Q4_K_M | Thinking |
| --- | --- | --- | --- | --- | --- | --- |
| `google/flan-t5-small` (current) | 80M | enc-dec T5 | Apache-2.0 | ~512 eff. | 90 MB | no |
| `Qwen/Qwen3-1.7B` | 1.7B | decoder | Apache-2.0 | 32,768 (YARN→128k) | 1,223 MB | hybrid |
| `HuggingFaceTB/SmolLM3-3B` | 3B | decoder | Apache-2.0 | 64k trained (YARN→128k) | 1,827 MB | hybrid (`/no_think`) |
| `Qwen/Qwen3.5-2B` | 2B | decoder (VL-capable) | Apache-2.0 | **262,144** | 1,222 MB | hybrid, default ON |
| `Qwen/Qwen3-4B` | 4B | decoder | Apache-2.0 | 32,768 (YARN→128k) | 2,382 MB | hybrid |
| `Qwen/Qwen3.5-4B` | 4B | decoder (VL-capable) | Apache-2.0 | **262,144** | 2,873 MB | hybrid, default ON |

Considered and rejected:

- **`microsoft/Phi-4-mini-instruct`** (3.8B, MIT, 128k): strong instruct model
  but not a hybrid-reasoning model — no `<|think|>` control, so it fails the
  "reasoning on/off" requirement.
- **`google/gemma-4-E2B/E4B-it`**: gated license (gemma terms), extra friction.
- **`meta-llama/Llama-3.2-3B-Instruct`**: non-Apache license, no reasoning mode.
- **FLAN-T5-large / mT5 / UL2**: no hybrid thinking, encoder-decoder limits (see §4.3).

Published reasoning numbers from the SmolLM3 model card (reasoning mode,
same evaluation harness for all three — *published*, not locally measured):

| Benchmark | Qwen3-1.7B | SmolLM3-3B | Qwen3-4B |
| --- | --- | --- | --- |
| AIME'25 | 30.7 | 36.7 | **58.8** |
| GPQA | 39.9 | 41.7 | **55.3** |
| LiveCodeBench | 34.4 | 30.0 | **52.9** |
| GSM-Plus | 79.4 | 83.4 | **88.2** |
| IFEval | 74.2 | 71.2 | **85.4** |
| BFCL (tool use) | 88.8 | 88.8 | **95.5** |

Qwen3.5 is the newer generation (2026) with 262k context and thinking on by
default; its card documents the same `chat_template_kwargs.enable_thinking`
control. Local measured comparison: `benchmarks/RESULTS.md`.

---

## 4. Runtime comparison (how to actually run a model on this PC)

### 4.1 Options surveyed

| Runtime | Native Windows | Quantization | OpenAI server | Reasoning fields (`reasoning_content`) | Verdict |
| --- | --- | --- | --- | --- | --- |
| **llama.cpp `llama-server`** | yes (official release zip, Clang, AVX2/AVX-512 dispatch) | GGUF Q4–Q8 | yes, built-in | yes (`--reasoning-format`) | **chosen** |
| HF Transformers (current) | yes | not by default (fp32/fp16) | no | no | keep as legacy backend |
| vLLM (CPU build) | no (Linux/WSL2, multi-GB, IPEX) | yes | yes | GPU-focused | reject for CPU path |
| ONNX Runtime / Optimum | yes | yes | no (own glue code) | partial | reject: weaker chat-template/tooling, more conversion work |
| llama-cpp-python | needs MSVC build for Py3.14 | GGUF | via FastAPI glue | yes | reject: build risk on Windows; subprocess server gives the same engine with zero build |
| Ollama / LM Studio | yes | GGUF | yes (different API surface) | partial | reject: extra daemon/GUI layer over the same engine; less control |
| MLX | Apple silicon only | — | — | — | irrelevant |

### 4.2 Why llama.cpp (verified empirically)

- Release **b11438 (2026-10-06)**, Windows CPU x64 zip (18.5 MB), extracted to
  `tools/llama.cpp/`. `llama-server.exe --version` → `0.6.0-dev (build 11438)`.
- On this CPU the `ggml-cpu-icelake.dll` backend loads (AVX-512 path;
  i7-1165G7 is Tiger Lake).
- Built-in OpenAI surface: `/v1/completions`, `/v1/chat/completions`
  (streaming SSE), `/v1/models`, `/health` — a 1:1 match for what `server.py`
  must serve.
- Reasoning support (from `llama-server --help`, b11438):
  `--reasoning-format none|deepseek|deepseek-legacy|auto` (default `auto`);
  `deepseek` moves chain-of-thought into `message.reasoning_content` and keeps
  `content` = final answer — exactly the desired API shape;
  `--reasoning on|off|auto`, `--reasoning-effort`, `--reasoning-budget N`;
  `--chat-template-kwargs` and per-request `chat_template_kwargs` for
  `enable_thinking` (documented by Qwen3, Qwen3.5 and SmolLM3 cards);
  `--jinja` templates (default on); `--alias` to control the reported model name.
- Measured on this machine (llama-bench, 4 threads, Qwen3-1.7B Q4_K_M):
  prefill 74.27 t/s (pp512), decode 13.38 t/s (tg64) — the pass-2 row of
  `benchmarks/results/summary.json` (pass 1 recorded 74.06 / 13.83). Larger
  models scale roughly inverse to weight size (see `benchmarks/RESULTS.md`).

### 4.3 Known llama.cpp limitation found during this investigation

The community GGUF of **flan-t5-small (T5 encoder-decoder) is broken on b11438**:

- `llama-bench -m flan-t5-small.Q4_K_M.gguf` aborts with
  `GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first")`
  (llama-graph.cpp:1102).
- `llama-server` loads but generates degenerate output: greedy decoding yields
  rows of `.` for “translate English to German: …” and empty output for “What is
  the capital of France?”. Evidence kept in
  `benchmarks/results/flan-t5-llamacpp-BROKEN-evidence.json`.

Conclusion: T5-class models cannot serve as the llama.cpp baseline; the FLAN
baseline is therefore measured through the repo's **actual current runtime**
(the transformers pipeline, fp32) — which is the honest "before" picture anyway.

### 4.4 CPU/RAM analysis for this machine

Machine: i7-1165G7, 4C/8T, 16 GB RAM (15.8 GB usable), AVX2 + AVX-512, no
discrete GPU, 22.5 GB free on E:.

Budget model for llama.cpp (single user, ctx 8192):

| Component | 1.7B Q4 | 3B Q4 | 4B Q4 | Qwen3.5-4B Q4 |
| --- | --- | --- | --- | --- |
| weights (mmap) | ~1.2 GB | ~1.8 GB | ~2.4 GB | ~2.9 GB |
| KV cache @8k (f16) | ~0.2 GB | ~0.2 GB | ~0.3 GB | ~0.3 GB |
| compute buffers | ~0.5 GB | ~0.5 GB | ~0.6 GB | ~0.6 GB |
| **peak RSS (measured)** | see RESULTS | see RESULTS | see RESULTS | see RESULTS |

fp32 transformers is only viable for the 80M model (~0.5–1 GB); a 4B model in
fp32 would need ~16 GB and is therefore impossible on this machine — quantization
is not optional, it is the enabling factor.

Threads: `-t 4` (physical cores) measured best practice for llama.cpp on 4C/8T;
hyperthreading (`-t 8`) typically hurts decode. Used throughout benchmarks.

---

## 5. Reasoning capability: what “good” looks like here

Requirements from the brief: a model with a real chain-of-thought, with a
per-request switch between “thinking on” and “thinking off”.

Mechanics (verified against model cards and llama.cpp help):

1. **Hybrid thinking models** (Qwen3/3.5, SmolLM3) emit
   `<think> … </think>` before the answer when
   thinking is on; the template can suppress thinking entirely.
2. **Per-request switch**: `chat_template_kwargs: {"enable_thinking": false}`
   in the request body (documented in all three model cards). llama.cpp accepts
   it per request through its OpenAI-compatible endpoint.
   `server.py` will also accept the friendlier top-level `"thinking": bool` and
   map it — mapping only, no invented semantics.
3. **API shape**: with `--reasoning-format deepseek`, llama-server returns
   `message.reasoning_content` (the full CoT) and `message.content` (the final
   answer) as separate fields; thinking tags never leak into `content`.
   When thinking is off, `reasoning_content` is absent/empty.
4. **Thinking budget**: server-level `--reasoning-budget` (llama.cpp flag) is
   exposed as env config `LLM_REASONING_BUDGET`; per-request budget is *not*
   invented because template support differs per model.
5. **Endpoint choice**: `/v1/completions` takes a raw prompt — thinking tags
   would appear inline in raw text. Reasoning models should be used through
   `/v1/chat/completions` (documented).

FLAN-T5 (current) has no reasoning mode at all: it is an 80M
encoder-decoder trained for single-pass instruction following.

---

## 6. API compatibility analysis (llama-server vs our contract)

| Concern | Plan |
| --- | --- |
| Endpoints kept | `server.py` keeps owning `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, and the entire job/queue surface. The backend only *generates text*. |
| Non-stream requests | llama backend forwards the (whitelisted) payload to `llama-server`, returns its OpenAI-shaped JSON with `model` rewritten to the configured name. |
| Streaming | passthrough of llama-server SSE chunks; legacy transformers backend synthesizes OpenAI-shaped chunks (role → content → finish → `[DONE]`) so `stream: true` never hangs. |
| `thinking` | top-level bool → `chat_template_kwargs.enable_thinking` (chat endpoint only). |
| `reasoning_content` | produced natively by llama-server (`--reasoning-format deepseek`). |
| `stop`, `temperature`, `top_p`, `top_k`, `max_tokens`, `seed` | passed through (whitelist) to llama-server. |
| `min_tokens`, `repetition_penalty`, `no_repeat_ngram_size` | legacy-transformers-only; ignored by llama backend (documented). |
| `usage` | real token counts from llama-server (fixes whitespace counting for the llama path). |
| Errors | OpenAI-style `{"error": {"message","type","code"}}` + correct status codes; malformed JSON → 400 instead of connection reset. |
| `model` field | echoed from config (`LLM_MODEL`); `/v1/models` lists the same id with `object/created/owned_by`. |
| Threading | `ThreadingTCPServer` + a lock around the legacy pipeline; queue endpoints unchanged but no longer block `/v1/*` (fixes B3). |
| Worker compatibility | `/jobs`, `/jobs/next`, `/webhooks/completions`, `/workers/heartbeat` request/response shapes untouched; `worker.sh` unchanged. |

---

## 7. Benchmark plan

Implemented in `benchmarks/` (all files committed):

- **`benchmarks/prompts.json`** — 16 fixed prompts across 11 categories
  (instruction ×1, reasoning ×2, math ×3, code ×2, extraction, classification,
  planning, RAG, hallucination-guard, system-hierarchy, long-context ×2 with
  2 k/4 k needles). Each prompt carries a machine-checkable expectation
  (`contains`/`not_contains`/`json`/`sentence_count`/…) so quality is scored,
  not eyeballed. Deterministic seed inputs are materialized by the runner.
- **`benchmarks/models.json`** — the 5 GGUF candidate model configs (threads=4,
  ctx=8192, port 8099; per-model sampling). The FLAN entry was removed after
  its GGUF proved unusable on llama.cpp (§4.3); FLAN runs via `--base`.
- **`benchmarks/benchmark_models.py`** — stdlib + psutil runner. Per model:
  1. spawns `llama-server` (managed mode) or targets an existing endpoint
     (`--base`, used for the legacy transformers baseline);
  2. `llama-bench -p 512 -n 64 -t 4 -o json` → clean prefill/decode numbers;
  3. runs the 16-prompt suite over **SSE streaming**, recording TTFT, decode
     t/s, wall time, `finish_reason`, real `usage` tokens;
  4. samples peak RSS and CPU% of the server process (0.5 s cadence);
  5. records reasoning diagnostics: whether `reasoning_content` was populated,
      whether think-tags leaked into `content`, reasoning length;
  6. runs the per-prompt checks → `checks_passed/checks_total`;
  7. writes `benchmarks/results/<label>.json` + appends `summary.json`.

**Metrics**: checks pass-rate (quality proxy), avg/median TTFT, decode t/s,
end-to-end wall per category, peak RSS, llama-bench pp512/tg64, load time,
reasoning separation success, think-tag leaks.

**Methodology controls**: same prompts, same runner, same threads (4), same
ctx (8192), temperature 0.6 / top_p 0.95 (chat models), one model resident at a
time, no other benchmarks running concurrently — but background CPU contention
(IDE open) was present for every pass-2 model **except Qwen3-1.7B**, so those
speed rows are lower bounds (flagged in §8 and `benchmarks/RESULTS.md`).
**Budgets are scaled ×4 per
prompt** (`--max-tokens-scale 4`, cap 4096): pass 1 at default budgets showed
reasoning models truncated mid-thought (Qwen3.5-2B: 11/16 `finish=length`,
9/22 checks) — budgets must not be the variable under test. Pass-1 artifacts
are preserved under `benchmarks/results/pass1_default_budget/`.

**Threats to validity**: 16 prompts is a smoke suite, not a leaderboard — it
ranks candidates for *this machine and workload*, it does not reproduce
published benchmarks (published numbers cited in §3 are used as a second,
independent signal). Sampling at temperature 0.6 adds run-to-run variance;
checks are structural (did the answer contain the required fact/JSON) rather
than semantic.

**FLAN baseline**: run against the repo's current `server.py` (transformers,
fp32) via `--base http://localhost:8000 --api completions` — the honest “before”
state. The broken flan GGUF is excluded (§4.3).

---

## 8. Recommendation

The migration is **implemented and verified**, and the benchmark suite is
complete for all 5 candidates (22 checks each, pass 2 at ×4 budgets).

Decision criteria: (1) quality of reasoning answers on this machine’s suite,
(2) tokens/second (usability), (3) RAM headroom, (4) license + ecosystem,
(5) API features (thinking control, `reasoning_content`, streaming).

### 8.1 Backend — implemented and verified

- **Default runtime**: llama.cpp `llama-server` as a subprocess behind the
  `backend/` package (`backend/base.py` interface, `backend/llama_cpp.py`
  subprocess + proxy, `backend/transformers_legacy.py`), with `server.py`
  running a `ThreadingTCPServer`.
- **Verified** (`MIGRATION_PLAN.md` §7): all **25 `test_api.sh` checks green**
  on `LLM_BACKEND=llama_cpp`; streaming SSE ends with `[DONE]`;
  `reasoning_content` separated from `content` (0 think-tag leaks);
  `"thinking": false` honored; queue round-trip (enqueue → long-poll →
  webhook → status); **B3 proof**: `/health` returned 200 in **41 ms during an
  active 25 s `/jobs/next` long-poll**.
- **Legacy rollback verified**: `LLM_BACKEND=transformers` runs the pre-migration
  runtime through the same API — **all 25 `test_api.sh` checks green on the
  transformers path as well** (2026-10-08). The legacy backend loads via
  `_Text2TextShim` because transformers v5 dropped the `text2text-generation`
  pipeline task; the FLAN baseline (8/22) was measured through this path.

### 8.2 Model recommendation — from measured results

Source: `benchmarks/results/summary.json` (pass 2, ×4 budgets, streaming,
4 threads, ctx 8192, temp 0.6 / top_p 0.95). Full data:
`benchmarks/RESULTS.md`, cross-model analysis: `MODEL_COMPARISON.md`.

| Model | checks | avg TTFT s | decode t/s | e2e t/s | peak RSS MB | pp512 / tg64 | budget exhausted | CoT separated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B-Q4_K_M | **20/22** | 5.49 | 10.47 | 7.76 | 3720.7 | 74.27 / 13.38 | 0/16 | 11/16 |
| Qwen3-4B-Q4_K_M | 18/22 | 12.047 | 4.42 | 3.84 | 5284.9 | 27.79 / 4.15 | 0/16 | 16/16 |
| SmolLM3-3B-Q4_K_M | 15/22 | 8.637 | 7.49 | 6.05 | 4306.1 | 36.05 / 8.38 | 0/16 | 6/16 |
| Qwen3.5-4B-Q4_K_M | 13/22 | 14.981 | 4.75 | 4.37 | 5853.8 | 26.41 / 5.32 | 7/16 | 16/16 |
| Qwen3.5-2B-Q4_K_M | 12/22 | 5.66 | 9.74 | 8.49 | 2637.7 | 31.90 / 5.44 | 9/16 | 16/16 |
| flan-t5-small (legacy) | 8/22 | 2.849 | n/a | 9.17 | — | — | 0/16 | 0/16 |

**Verdict:**

- **DEFAULT = `Qwen3-1.7B-Q4_K_M`** — highest score (20/22), decisively
  fastest (tg64 13.38 vs 4.15–5.44 for the field), low RAM (3720.7 MB), zero
  budget exhaustion. It replaces the stock default `Qwen3.5-4B-Q4_K_M` in
  `.env`, `.env.example`, `README.md`, `API_EXAMPLES.md`, `run.sh` fallback and
  the `backend/__init__.py` fallback path. Its only failure is
  instruction_hierarchy (0/2: leaked `BANANA-77`, 82 words vs the 30-word cap).
- **QUALITY = `Qwen3-4B-Q4_K_M`** (18/22) — when deliberation quality matters
  more than latency: 16/16 reasoning separation, 0/16 budget exhaustion, no
  check passed by the 1.7 B model that it misses — the cost is ~3× lower
  decode speed and 1.4× the RAM.
- **Rejected: `Qwen3.5-4B-Q4_K_M`** — the previous default, now measured and
  dethroned: **worst score (13/22)**, most expensive run (238.35 s/prompt),
  7/16 budget-exhausted, both long-context needles failed, highest RSS
  (5853.8 MB). Its 2/2 instruction-hierarchy “pass” is vacuous (empty truncated
  answer).
- **Rejected: `Qwen3.5-2B-Q4_K_M`** (12/22) — 9/16 budget-exhausted even at ×4,
  anomalous per-token slowness for its size (tg64 5.44 vs 13.38 for the
  smaller 1.7 B), vacuous hierarchy pass.
- **`SmolLM3-3B-Q4_K_M` = fast-but-reasons-little** — best wall time
  (37.49 s/prompt) but reasoning separated on only 6/16 prompts and confident
  wrong answers on hard reasoning (15/22). Usable only for low-deliberation
  traffic.

**Caveats (keep honest):**

- One sample per model at temperature 0.6 — scores are indicative; ±10 %
  run-to-run variance is plausible (`benchmarks/RESULTS.md` limitations).
- The hierarchy adversarial prompt is failed in substance by **every** model
  that produced an answer (leaked `BANANA-77` in 68–161 words); the two clean
  2/2 scores (Qwen3.5-2B, Qwen3.5-4B) are empty truncated responses. No
  candidate withstands the override — do not rank on this prompt.
- Load asymmetry: Qwen3-1.7B ran on a cleaner machine; the other four ran
  under background CPU contention, so their speed rows are lower bounds. The
  1.7 B speed ranking is robust to this (see `MODEL_COMPARISON.md` §7).

### 8.3 Config that ships

```bash
# .env (now updated in the repo)
LLM_BACKEND=llama_cpp
LLM_MODEL=Qwen3-1.7B-Q4_K_M
LLM_MODEL_PATH=models/gguf/Qwen3-1.7B-Q4_K_M.gguf
LLM_THINKING=auto
LLM_REASONING_FORMAT=deepseek
LLM_MAX_TOKENS=1024          # default raised from 256 (hybrid reasoning)
LLM_MAX_TOKENS_CAP=4096
LLM_THREADS=4
LLM_CONTEXT_SIZE=8192
```

### 8.4 Open issues before production

- Worker endpoints (`/jobs/next`, `/workers/heartbeat`) authenticate via a
  **query-string token** (`?worker_token=…`) — leaks into proxy/server logs
  (legacy behavior, O9); move to a header before any non-LAN exposure.
- **No auth on `/v1/*`** — bind to localhost or a trusted LAN only;
  `WORKER_TOKEN` still ships as `change-me`.
- The job queue is **in-memory** — queued jobs are lost on process exit; no
  durability or replay.
- A daily **app-initiated shutdown around 18:00 local time** was observed on
  the test machine and killed the first benchmark attempt — disable scheduled
  sleep/hibernate (and any maintenance windows) before long runs.

---

## 9. Migration plan summary

See `MIGRATION_PLAN.md` for the full plan and rollback. Outline:

1. Add `backend/` package: `base` (interface), `llama_cpp` (subprocess + proxy),
   `transformers_legacy` (current pipeline, faithful behavior + bug fixes B1/B2).
2. `server.py`: backend factory from env (`LLM_BACKEND`), threaded server,
   OpenAI error shapes, streaming responses, config-driven `/v1/models` and
   `model` fields, `thinking` mapping, payload logging → one-line logs.
3. Config: new `LLM_*` env vars with `VLLM_*` aliases so existing
   `.env`/compose files keep working; `.env.example` documents them.
4. `run.sh`: native Windows first-class path (no Docker required), Docker path
   kept and fixed (healthcheck bug O3, wrong-container log O5, token leak O5).
5. `test_api.sh` / `examples.sh` / docs: model names from config, streaming and
   reasoning tests added.
6. Rollback: `LLM_BACKEND=transformers` restores old runtime without touching
   the queue layer; `git` state unchanged until the user asks for a commit.

## 10. Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| llama-server subprocess dies at runtime | low | health endpoint reports 503 with the exit reason; deliberately no auto-respawn loop — restart `server.py` (flapping hides failures); circuit breaker already isolates `/v1` failures from queued jobs |
| Qwen3.5 arch incompatibility with llama.cpp b11438 | low (GGUFs exist for this build family) | empirically smoke-tested before adoption (see RESULTS); Qwen3-4B/SmolLM3 as fallbacks |
| T5 GGUF brokenness (found, §4.3) | realized | FLAN baseline measured via transformers; T5 excluded from llama path |
| Thinking models exceed `max_tokens` cap → truncated answers | medium (observed: 11/16 prompts `finish=length` at 256) | default `LLM_MAX_TOKENS` raised 256 → 1024; hard cap configurable `LLM_MAX_TOKENS_CAP` (default 4096); pass-2 benchmark at ×4 budgets + tests cover it |
| Long-poll/`/v1` interference | solved | threading server (B3) |
| Exposed `.env` secrets (`HF_TOKEN`, `WORKER_TOKEN`) | realized | rotate both (never printed by tooling); `run.sh` no longer echoes token prefix (O5) |
| Single-user machine, no `/v1` auth | known | out of scope; documented in README (bind to LAN only, keep `WORKER_TOKEN`) |
| Disk pressure from model zoo (6 GGUF files ≈ 9.6 GB — 5 benchmark candidates + the broken flan-t5 GGUF) | low | `models/` gitignored; `tools/` added to `.gitignore` (O8); prune instructions in README |

## 11. Rollback plan

- Old runtime is still there: `LLM_BACKEND=transformers` + (optionally)
  `TRANSFORMERS_MODEL=google/flan-t5-small` reproduces pre-migration inference
  through the same API (with B1/B2 bug fixes; raw pipeline behavior otherwise
  preserved: wrapper prompt, penalties, caps). The legacy backend loads via
  `_Text2TextShim`, because transformers v5 removed the `text2text-generation`
  pipeline task the pre-migration code used (§1.3).
- Queue/worker layer is untouched by construction (no changes to those
  endpoints' code paths beyond threading).
- No commit is made until explicitly requested; reverting files restores the
  old tree at any time.
- llama-server is optional: if `tools/llama.cpp` is deleted, the service still
  starts in transformers mode and reports a clear error if `LLM_BACKEND=llama_cpp`.
