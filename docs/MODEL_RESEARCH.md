# MODEL_RESEARCH — external Jetson evidence & new model candidates

Date: 2026-10-08. **Research only** — no code changes, no benchmark runs, no
commits. Only this file was created.
Companions: `INVESTIGATION.md` (model verdicts), `MODEL_COMPARISON.md`
(recommendations), `benchmarks/RESULTS.md` (authoritative local results).

---

## 1. External benchmark summary

**Source (fetched 2026-10-08):**
<https://turingpi.com/jetson-orin-nano-8gb-llm-benchmark/>
— "What LLMs Actually Fit on a Jetson Orin Nano 8GB? Models, Context, and
Runtimes Tested", Turing Pi, published 2026-09-26.

**Hardware / method (as published):**

- NVIDIA **Jetson Orin Nano 8GB** (Turing Pi 2.5 carrier), Ubuntu 24.04,
  JetPack 7.2.1, CUDA 13.2, 25 W power mode, shared LPDDR5.
- **llama.cpp build 10706 (CUDA backend)** + Ollama 0.34.0; 6 CPU threads,
  `-ngl 999` full GPU offload, FP16 KV, ctx 4096, batch 512, flash attention,
  temperature 0 / seed 42.
- Measured: 5-rep `llama-bench` `pp512` / `tg128` for **7 Q4_K_M models**,
  tokenizer-counted filled-context sweeps (3/3 reproducibility criterion),
  Ollama vs llama.cpp on identical GGUF checksums, and Qwen3-8B full vs
  partial offload.
- The article explicitly performs **no answer-quality scoring** ("we did not
  evaluate answer quality").

**Results table (all numbers EXTERNAL — GPU, different hardware):**

| Model | Q4_K_M file | `tg128` GPU t/s | `pp512` GPU t/s | Max filled ctx (3/3) |
| --- | --- | --- | --- | --- |
| Qwen3 1.7B | 1.11 GB | **39.42** | 1,408.3 | 32k |
| Gemma 4 E2B it | 3.46 GB | 26.67 | 926.6 | 32k |
| Llama 3.2 3B Instruct | 2.02 GB | 25.24 | 813.7 | 32k |
| Ministral 3 3B Instruct | 2.15 GB | 23.27 | 766.5 | 32k |
| Qwen3 4B | 2.50 GB | 19.53 | 609.7 | 16k (32k KV OOM) |
| Phi-4-mini-instruct | 2.49 GB | 19.43 | 705.1 | 16k (32k KV OOM) |
| Qwen3 8B | 5.03 GB | 12.64 | 355.5 | 8k (safety cutoff) |

**⚠ Incomparability statement:** these are **GPU tokens/second on different
hardware** (Orin Nano SM87 CUDA, 6 threads, llama.cpp b10706). They must
**never** be compared directly with this repo's local CPU numbers (i7-1165G7,
4 threads, llama.cpp b11438 — e.g. 10.47 t/s suite decode for Qwen3-1.7B in
`benchmarks/RESULTS.md`). The external data is used here **only** to
(a) sanity-check that our default is not a CPU-only artifact and (b) discover
candidate models that demonstrably fit 8 GB and run on the Orin Nano.

**What is materially new for this project:**

1. **Qwen3-1.7B is also the fastest model on the Jetson** (39.42 t/s) and
   completed filled 32k prompts 3/3 — there is no GPU-speed argument for
   dropping our default.
2. **Qwen3 8B fits fully offloaded (37/37 layers) on the 8 GB Jetson** at 4k
   context (12.64 t/s, 8k filled 3/3; partial offload 24/37 → 5.85 t/s) — a
   previously undocumented option for a Jetson quality profile.
3. Two **Apache-2.0 small models** we had never evaluated surfaced with strong
   Jetson numbers: **Gemma 4 E2B it** and **Ministral 3 3B Instruct** (both
   32k filled 3/3). Gemma 4's license status also **contradicts an earlier
   rejection in this repo** (see §2.4).

---

## 2. Candidate analysis

### 2.1 Already benchmarked locally (authoritative source: `benchmarks/RESULTS.md`)

| Model | Params | Arch | License | Q4_K_M | Hybrid thinking on/off | Native ctx | Local checks | Local decode t/s | Peak RSS MB | EXTERNAL Jetson `tg128` | Local verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Qwen3-1.7B** (default) | 1.7B | decoder | Apache-2.0 | 1.22 GB | yes (`enable_thinking`) | 32,768 (YaRN→128k) | **20/22** | **10.47** | 3720.7 | **39.42** (fastest) | keep — best on every decisive axis |
| Qwen3-4B | 4B | decoder | Apache-2.0 | 2.38 GB | yes | 32,768 (YaRN→128k) | 18/22 | 4.42 | 5284.9 | 19.53 | quality profile; 1.7B's 20/22 is a strict superset |
| SmolLM3-3B | 3B | decoder | Apache-2.0 | 1.83 GB | yes (`/no_think`) | 64k trained (YaRN→128k) | 15/22 | 7.49 | 4306.1 | not tested | fast but reasons little (CoT 6/16) |
| Qwen3.5-4B | 4B | decoder (VL) | Apache-2.0 | 2.87 GB | yes, default ON | 262,144 | 13/22 | 4.75 | 5853.8 | not tested | rejected — worst score, 7/16 exhausted |
| Qwen3.5-2B | 2B | decoder (VL) | Apache-2.0 | 1.22 GB | yes, default ON | 262,144 | 12/22 | 9.74 | 2637.7 | not tested | rejected — over-thinks, 9/16 exhausted |

No new information from the external benchmark materially changes these rows:
bigger Qwen3/Qwen3.5 models **cost speed without winning checks** on this CPU,
and on the Jetson the 1.7B and 4B rows keep the same speed ordering.

### 2.2 New candidates discovered — full field analysis

Fields per project rules. "EXTERNAL" = published by third parties (never
comparable to our local numbers). CPU speeds are **estimates** from the
inverse-to-params heuristic vs the 1.7B baseline (~10 t/s suite / 13.38 tg64),
not measurements.

| Field | **Gemma 4 E2B it** | **Qwen3-8B** | **Ministral 3 3B Instruct 2512** |
| --- | --- | --- | --- |
| Parameters | 5.12B total BF16 (HF safetensors); Google markets "effective 2B" (2.3B eff. per third-party reviews) | 8.19B dense (HF `safetensors.parameters`) | ~3B |
| Architecture | `Gemma4ForConditionalGeneration`, multimodal (text/image/audio in), hybrid-attention state layout (external logs: 204 MiB state/KV @32k) | `Qwen3ForCausalLM`, dense decoder | `Mistral3...`, dense, vision-capable, hybrid per-layer state |
| License | **Apache-2.0** (HF API `cardData.license`, Google launch blog); `gated: false` | **Apache-2.0** (HF API, LICENSE file); `gated: false` | **Apache-2.0** (HF metadata + Mistral 2025-12-02 announcement); `gated: false` |
| GGUF / Q4_K_M publisher | `bartowski/google_gemma-4-E2B-it-GGUF` — Q4_K_M **3.46 GB**, SHA-verified by the external article | **Official `Qwen/Qwen3-8B-GGUF` Q4_K_M 5.03 GB** (verified via HF API: file list + `totalFileSize` 5,027,783,488); also `bartowski/Qwen_Qwen3-8B-GGUF` (SHA-verified externally) | `bartowski/...` Q4_K_M **2.15 GB** (SHA-verified by the external article) |
| Hybrid reasoning | **Yes** — chat template kwarg `enable_thinking`, `default(false)`; ON/OFF states both documented by Google for E2B/E4B (`<\|think\|>` + `<\|channel>thought` blocks) | **Yes** — Qwen3 `enable_thinking` kwarg in the official GGUF chat template (verified in `Qwen/Qwen3-8B-GGUF` metadata) | **No** for Instruct: no thinking tokens anywhere in its template (verified). A separate `Ministral-3-3B-Reasoning-2512` variant exists |
| Thinking on/off | ✓ both states supported | ✓ both states supported | ✗ Instruct: none. ✗ Reasoning variant: always-on via default system prompt, **no documented off switch** (overriding the system prompt = inventing semantics) |
| Native context | 262k (HF card "up to 256K"); externally filled **32k, 3/3** | **40,960** (config `max_position_embeddings` + GGUF metadata, both verified); YaRN extension to ~128k per family docs (not re-verified) | 256k (card); externally filled 32k, 3/3 |
| Q4_K_M size | 3.46 GB | 5.03 GB | 2.15 GB |
| Expected RAM (ctx 8192, our harness) | ~4.5–5.5 GB peak RSS (weights 3.46 GB + buffers; KV tiny thanks to hybrid layout) | ~7–8 GB peak RSS (weights 5.03 GB + KV 8k ≈ 1.15 GB f16 + ~0.6 GB buffers) | ~4.3–4.6 GB peak RSS |
| llama.cpp support | Yes — external article ran it on b10706; our b11438 is newer | Yes — same Qwen3 arch already benchmarked locally; official GGUF ships the template | Yes — ran externally; **known llama-server friction** on the Reasoning variant's `content[].type: "thinking"` (issue #17700 / PR #17713, status at b11438 unverified) |
| Jetson compat (Orin Nano = **SM87**) | ✓ proven externally (CUDA b10706, 26.67 t/s, 32k 3/3) | ✓ proven externally (37/37 full offload @4k, 12.64 t/s, 8k 3/3) | ✓ proven externally (23.27 t/s, 32k 3/3) |
| CPU suitability (estimate) | **~5–9 t/s decode** — by total params (5.12B) ≈ 3.5 t/s; by effective/active ~2B ≈ 8–9 t/s; external GPU ratio vs 1.7B (0.68×) suggests ~9 t/s. Must be measured; even the pessimistic end is "Qwen3-4B class" | **~1.5–2.5 t/s decode** — dense 8B vs the observed 4B row (4.15 tg64 at 4.42 suite); inverse scaling under-delivered at 4B, so expect ≤2.5 tg64. Interactive-unusable on this CPU | ~6–7 t/s (2.15 GB vs SmolLM3's 1.83 GB → 8.38 tg64) |
| GPU suitability | 2nd fastest of the seven (EXTERNAL 26.67 t/s, 32k) | Slowest of the seven but usable (EXTERNAL 12.64 t/s full offload) | 23.27 t/s (EXTERNAL) |
| Published benchmark evidence | **EXTERNAL**: Gemma 4 launch/tech-report figures (aggregated) — IFEval **94.6**, GPQA-Diamond 43.4, LiveCodeBench v6 44.0, AIME'26 37.5, MMLU-Pro 60.0, IFBench 38.0. For comparison, Qwen3-1.7B *published* card numbers (different versions/harness — directional only): IFEval 74.2, GPQA 39.9, LCB 34.4, AIME'25 30.7. **EXTERNAL** controlled study (arXiv 2604.07035): E2B weighted acc. 0.493 few-shot CoT vs Qwen3-8B 0.322 (their harness; prompt-sensitive) | **EXTERNAL** mixed: arXiv 2604.07035 ranked Qwen3-8B in the *lower* cluster (0.322 few-shot CoT) — a counter-signal; Qwen family published cards show 8B > 4B > 1.7B on reasoning suites (versions differ) | **EXTERNAL**: external article ran no quality scores; Mistral announcement claims best-in-class small agentic/JSON (marketing) |
| Expected tradeoff | Older-gen 1.7B gives up ~1.5–2× decode speed and ~1 GB RAM vs our default; E2B must prove its newer generation converts into ≥20/22 here; reasoning-separation path (`<\|channel>thought` → `reasoning_content`) is unverified on our llama.cpp | ~4–5× slower than Qwen3-1.7B on our CPU (~2 t/s); suite run would take ~1.5–2.5 h; local 4B precedent (18/22, strict subset of 1.7B's 20) says size alone may not win checks | No thinking toggle at all → the project's core reasoning/on-off feature is absent (Instruct) or non-standard (Reasoning) |
| Why relevant | The **only** new candidate plausibly able to beat Qwen3-1.7B on checks at tolerable CPU speed, with Jetson proof | The only candidate with a **justified dual-profile CPU/Jetson split** (CPU stays 1.7B; Jetson runs 8B at ~12.6 t/s GPU ≈ current CPU UX, with better quality) | Strong edge numbers but fails the thinking requirement — documented, not shortlisted |

### 2.3 Considered and rejected (with the requirement that fails)

| Model | Why rejected |
| --- | --- |
| Ministral 3 3B **Reasoning** 2512 | Req 4: thinking driven by an always-on default system prompt, no supported on/off switch; plus known llama-server `content[].type: thinking` friction (issue #17700). GGUF existence never checked (rejected before it mattered). |
| Ministral 3 3B **Instruct** 2512 | Req 3 + 4: no reasoning/thinking mode at all (template verified) — fails the project's core reasoning requirement despite Apache-2.0, 256k ctx, 2.15 GB, and good Jetson numbers. |
| `meta-llama/Llama-3.2-3B-Instruct` (surfaced externally) | Req 4 (no thinking mode) and req 14 (Llama 3.2 Community License, not a permissive OSS license) — re-affirms the earlier rejection in INVESTIGATION §3. |
| `microsoft/Phi-4-mini-instruct` (surfaced externally) | Req 4: no hybrid thinking (re-affirms INVESTIGATION §3); Phi-4-mini-reasoning is always-on with no off switch. Mixed EXTERNAL quality signal (arXiv 2604.07035 put its reasoning sibling mid/low cluster). |
| Gemma 4 **E4B**-it | Not in the external benchmark (excluded from their "standardized pass"); 4B-class on CPU has a measured precedent of losing to 1.7B (Qwen3-4B 18/22 @ 4.42 t/s). No credible reason it beats the 1.7B at tolerable speed; would just inflate the model zoo. |
| Qwen3.5-4B / Qwen3.5-2B | Already locally measured and rejected (13/22 and 12/22 with budget exhaustion). |
| Qwen3-0.6B (family floor) | No credible path to ≥20/22 — a priori weaker than the 1.7B across reasoning/coding/math; testing it can only confirm a loss. |
| `openai/gpt-oss-20B` (considered, not surfaced externally) | Req 4: always-reasoning model (effort levels, no thinking-off); not investigated further. |
| **Qwen3-8B** (dropped 2026-10-08) | Explicit decision: skip — "no need to test it". Its CPU estimate (~1.5–2.5 t/s, §2.2) was interactive-unusable regardless; a benchmark run had been started and was stopped mid-flight — no local result. Removed from `benchmarks/models.json`. |

### 2.4 Correction to prior documentation

`INVESTIGATION.md` §3 rejected "**google/gemma-4-E2B/E4B-it**: gated license
(gemma terms), extra friction". Verified 2026-10-08 via the HF API:
`google/gemma-4-E2B-it` has `license: apache-2.0` and **`gated: false`**, and
Google's Gemma 4 launch post states the family is "released under a
commercially permissive Apache 2.0 license". **The original rejection reason
no longer holds** for Gemma 4. (A companion "Gemma 4 license" page exists at
`ai.google.dev/gemma/docs/gemma_4_license` but timed out on fetch — see §5.)

---

## 3. Shortlist — ≤3 new candidates worth benchmark time

### 1) `google/gemma-4-E2B-it` — Q4_K_M, 3.46 GB (bartowski) — **PRIMARY**

Requirement coverage: 1 ✓ instruct · 2 ✓ genuine generation · 3 ✓ useful
reasoning (EXTERNAL IFEval 94.6 / GPQA-D 43.4 / LCB-v6 44.0) · 4 ✓
`enable_thinking` on/off (template verified) · 5 ✓ coding · 6 ✓ math ·
7 ? structured extraction (native tool-calling/JSON; suite will decide) ·
8 ? planning · 9 ? RAG · 10/11 ? hallucination & hierarchy (IFEval signal is
promising but the BANANA-77 override is its own test) · 12 ✓ CPU-friendly
(est. ~5–9 t/s, ~4.5–5.5 GB RSS) · 13 ✓ Q4_K_M (bartowski, SHA-verified) ·
14 ✓ Apache-2.0, ungated · 15 ✓ reasonable RAM · 16 ✓ llama.cpp (proven on
b10706; our b11438 newer) · 17 ✓ Jetson SM87 proven (26.67 t/s, 32k 3/3).

- **Why it might beat Qwen3-1.7B:** it is a April-2026-generation
  effective-~2B model whose published instruction-following and reasoning
  numbers sit above Qwen3-1.7B's published card numbers on every comparable
  axis (IFEval 94.6 vs 74.2; GPQA-D 43.4 vs GPQA 39.9; LCB v6 44.0 vs 34.4 —
  EXTERNAL, different versions, directional only). Our default's only two
  failures are instruction_hierarchy (0/2) and it separates CoT on just 11/16
  — precisely the areas a newer instruction-tuned generation could improve.
  Its expected CPU speed (~5–9 t/s) keeps it in the tolerable band (≥ SmolLM3's
  7.49 t/s class), so a quality win would not cost usability the way the 4B
  class does.
- **Expected tradeoff:** 1.5–2× slower decode than Qwen3-1.7B and ~1 GB more
  RSS; ~3.46 GB weight file; reasoning-separation behavior of Gemma's
  `<|channel>thought` channel through `--reasoning-format deepseek` is
  unverified on our llama.cpp build; thinking defaults OFF (must be enabled
  per request — the harness already maps this).
- **Why it is worth spending benchmark time:** it is the only new model that
  is simultaneously (a) in our CPU speed class, (b) fully satisfying the
  license/thinking/GGUF requirements, and (c) externally evidenced as stronger
  than the incumbent — a genuine, cheap-to-run challenger. One suite run
  (~1 h) answers it definitively on this machine.

---

## 4. Final recommendation

RECOMMENDATION: TEST THESE CANDIDATES

1. `google/gemma-4-E2B-it` Q4_K_M (3.46 GB, bartowski) — primary; same-CPU-class challenger with newer-generation quality evidence and Jetson proof.

**Status (2026-10-08, same day):** tested — see `MODEL_REEVALUATION.md`.
Gemma scored **19/22** vs Qwen3-1.7B's **20/22** at 9.35 vs 10.47 decode t/s;
the default is kept. Qwen3-8B (former secondary candidate) was dropped by
explicit decision before testing (§2.3).

---

## 5. Facts not verified this session

- The Gemma 4 companion license page (`ai.google.dev/gemma/docs/gemma_4_license`)
  timed out on fetch; Apache-2.0 rests on HF API metadata + Google's launch
  blog. A license field of `apache-2.0` with a separate `license_link` was not
  read in full.
- Whether llama.cpp b11438 maps Gemma 4's `<|channel>thought` blocks into
  `message.reasoning_content` under `--reasoning-format deepseek` (expected,
  untested on this box).
- Gemma 4 E2B's exact parameterization: HF reports 5.12B total and Google says
  "effective 2B"; one third-party paper classifies it MoE (5.0B/2.0B-active),
  another source says embedding-heavy — CPU speed estimate spans this
  uncertainty.
- All CPU t/s and RSS figures for the shortlisted model (Gemma 4 E2B it) in
  this document are **estimates**, not measurements — measured results are in
  `MODEL_REEVALUATION.md` (same day, same machine).
- Ministral-3-3B-Reasoning GGUF availability, and the current status of the
  llama.cpp fix for its `content[].type: thinking` (issue #17700 / PR #17713)
  at b11438 — both moot after the req-4 rejection.
- No secrets (HF_TOKEN / WORKER_TOKEN / API_KEY) appear in this document.
