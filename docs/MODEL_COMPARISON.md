# MODEL_COMPARISON — CPU-only reasoning LLMs for llmlocal

Companion documents: `INVESTIGATION.md` (audit, candidate selection),
`benchmarks/RESULTS.md` (raw measured results), `MIGRATION_PLAN.md`
(implementation), `README.md` (usage).

## 1. Machine under test

| | |
| --- | --- |
| CPU | Intel Core i7-1165G7 (4C/8T, Tiger Lake) — AVX2 + AVX-512, loads `ggml-cpu-icelake.dll` |
| RAM | 16 GB (15.8 GB usable) |
| GPU | none — CPU inference only |
| OS / Python | Windows 11 (26200), Python 3.14.7 |
| Runtime | llama.cpp b11438 (llama-server subprocess, 4 threads, ctx 8192) |
| Quant | Q4_K_M GGUF for all candidates |

## 2. Method

Suite: `benchmarks/prompts.json` — 16 prompts across 11 categories
(instruction, reasoning ×2, math ×3, code ×2, extraction, classification,
planning, RAG, hallucination-guard, instruction-hierarchy safety, long-context
×2 with 2 k/4 k needles), each with machine-checkable assertions (22 checks
total).

- **Budgets**: every prompt's `max_tokens` scaled ×4 (capped at 4096,
  matching `LLM_MAX_TOKENS_CAP`). See §5 — reasoning models spend part of the
  budget on chain-of-thought, and the default budgets caused premature
  truncation. Thinking mode `auto` (server default), reasoning returned in
  `reasoning_content` (`LLM_REASONING_FORMAT=deepseek`), so checks run on the
  final answer only.
- **Latency**: streaming SSE — TTFT, decode t/s, end-to-end tokens/s; peak RSS
  via `psutil`.
- **Model-intrinsic speed**: `llama-bench -p 512 -n 64 -t 4`.
- **Ordering**: one model at a time, full suite, same prompts, temperature
  0.6 / top_p 0.95 for all llama-server models. Load caveat in §7/§8.
- **Legacy baseline**: `flan-t5-small` through the repo's transformers backend,
  completions API, same ×4 budgets (it never approaches them).

Reproduce: `python benchmarks/benchmark_models.py --bench --max-tokens-scale 4 --timeout 600`

## 3. Candidates considered

All downloaded, all Apache-2.0, all verified to load in llama-server on this
machine:

| Model | Params | GGUF Q4_K_M | Hybrid thinking | Score (22 checks) |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B | 1.7 B | 1.22 GB | yes | 20 — top scorer, fastest |
| Qwen3.5-2B | 2 B | 1.22 GB | yes | 12 — over-thinks to empty answers |
| SmolLM3-3B | 3 B | 1.83 GB | yes | 15 — reasons on 6/16 prompts |
| Qwen3-4B | 4 B | 2.38 GB | yes | 18 — quality profile |
| Qwen3.5-4B | 4 B | 2.87 GB | yes | 13 — slowest, worst score |
| flan-t5-small | 77 M | 0.09 GB | **no** | 8 — legacy baseline only |

Rejected earlier (see INVESTIGATION §5): Phi-4-mini (no hybrid thinking),
gemma-4 (gated), Llama-3.2 family (license). SmolLM3-3B was kept despite being
non-Qwen. `Qwen3.5-4B-Q4_K_M` was the stock `.env` default when this work
started; it is now measured and deprecated (see §7).

### flan-t5 GGUF is broken in llama.cpp b11438

`llama-bench` aborts with
`GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first")`
(llama-graph.cpp:1102) and llama-server generates garbage (rows of `.`).
Evidence preserved in `benchmarks/results/flan-t5-llamacpp-BROKEN-evidence.json`.
This is why FLAN stays on the transformers baseline and llama.cpp serves only
decoder models.

## 4. Pass-2 results (scale=4 budgets) — primary evidence

| Model | checks | avg TTFT s | decode t/s | e2e t/s | peak RSS MB | llama-bench pp512/tg64 | exhausted¹ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | **20/22** | 5.49 | 10.47 | 7.76 | 3720.7 | 74.27 / 13.38 | 0/16 |
| Qwen3-4B | 18/22 | 12.047 | 4.42 | 3.84 | 5284.9 | 27.79 / 4.15 | 0/16 |
| SmolLM3-3B | 15/22 | 8.637 | 7.49 | 6.05 | 4306.1 | 36.05 / 8.38 | 0/16 |
| Qwen3.5-4B | 13/22 | 14.981 | 4.75 | 4.37 | 5853.8 | 26.41 / 5.32 | 7/16 |
| Qwen3.5-2B | 12/22 | 5.66 | 9.74 | 8.49 | 2637.7 | 31.90 / 5.44 | 9/16 |
| flan-t5-small (transformers) | 8/22 | 2.849 | n/a² | 9.17 | — | — | 0/16 |

¹ prompts that hit `finish_reason=length`. ² legacy backend emits the SSE
stream in a single chunk — decode rate not measurable; `e2e t/s` is used.

Rows match `benchmarks/results/summary.json` field for field. Single run per
model at temperature 0.6 — treat as indicative, not a stability test. Full
per-prompt data: `benchmarks/RESULTS.md`.

## 5. Finding: token budgets decide apparent quality (pass 1)

With the original fixed budgets (128–768 `max_tokens`), reasoning models were
truncated mid-thought and scored badly for a configuration reason, not a
capability reason:

| model | hit `finish=length` | empty answers | checks |
| --- | --- | --- | --- |
| Qwen3-1.7B (default budgets) | 3/16 | 2 | 17/22 |
| Qwen3.5-2B (default budgets) | 11/16 | 11 | 9/22 |
| Qwen3.5-2B (×4 budgets) | 9/16 | 9 | 12/22 |

Mechanism: `thinking` consumes the same `max_tokens` budget as the answer.
Qwen3.5-2B thinks ~1–3 k characters before answering; at 256 tokens it rarely
starts the answer. Consequences:

- **Deployment rule**: for reasoning prompts, `max_tokens` ≥ 1024–2048 (the
  server default cap is `LLM_MAX_TOKENS_CAP=4096`). The repo's default was
  raised 256 → 1024.
- **Benchmark rule**: fair model comparison needs generous budgets — hence
  pass 2 (`--max-tokens-scale 4`).
- Even ×4 did not fix Qwen3.5-2B (9/16 still hit the cap empty) and bit
  Qwen3.5-4B too (7/16). Qwen3.5-2B's pass-2 12/22 therefore undercounts
  capability, and its 2/2 instruction-hierarchy score is a *vacuous* pass on an
  empty answer (see §7). Qwen3.5-4B's 2/2 is likewise vacuous.
- Scaling ×4 raised Qwen3-1.7B from 17/22 → 20/22 and Qwen3.5-2B from 9/22 →
  12/22 — the ranking direction survives, the absolute numbers do not.
- Pass-1 artifacts are preserved in `benchmarks/results/pass1_default_budget/`.

## 6. Finding: Qwen3.5-2B decode is anomalously slow (and Qwen3.5-4B is a worse over-thinker)

Pass 1: Qwen3-1.7B decoded at 11.39 t/s vs Qwen3.5-2B's 8.84 t/s despite a
similar 1.22 GB weight file; llama-bench pp512 74.06 vs 37.54, tg64 13.83 vs
7.44. Pass 2 repeated the gap without the budget confound: pp512 74.27 vs
31.90, tg64 13.38 vs 5.44 — and under identical evening load the tg64 gap
widened further (7.44 → 5.44) while the 1.7 B model held ~13.4 t/s.

Order across the field (pass-2 llama-bench tg64): **13.38 Qwen3-1.7B > 8.38
SmolLM3-3B > 5.44 Qwen3.5-2B > 5.32 Qwen3.5-4B > 4.15 Qwen3-4B**. So
Qwen3.5-2B is the slowest per-token of the small class it nominally belongs to
— and its suite average (140.76 s/prompt) is worse than the 4 B model's
(112.90 s) because 9 of its prompts ran to the cap emitting empty completions.

`Qwen3.5-4B` is a different failure: half the speed-per-token of the 1.7 B
model for a 0.2-part advantage in checks is not available — its 13/22 is the
worst of the run and its 238.35 s/prompt average is the most expensive wall
time of the whole suite. Where the 2 B model thinks *too much*, the 3.5-4 B
model thinks *as much at 5×5.4 tokens/s* — strictly worse on every axis
against Qwen3-1.7B except reasoning separation (16/16 vs 11/16), which buys
nothing when the answer is empty.

## 7. Recommendation

**Primary / default backend: `Qwen3-1.7B-Q4_K_M`** — swap the current `.env`
default (`Qwen3.5-4B-Q4_K_M`) to `Qwen3-1.7B-Q4_K_M`:

- (a) Highest check score in pass 2 — **20/22**, failing only
  instruction_hierarchy (0/2: leaked `BANANA-77`, 82 words vs the 30-word cap).
  Every other prompt passed every check.
- (b) Decisively fastest on this CPU — llama-bench tg64 **13.38 t/s** vs
  4.15 (Qwen3-4B) and 5.32 (Qwen3.5-4B); suite decode 10.47 t/s vs 4.42/4.75;
  ~5.5 s TTFT vs ~12–15 s for the 4 B class.
- (c) Low RAM (peak RSS 3720.7 MB; only the 2 B model measured lower, 2637.7 MB)
  on a 16 GB machine.

Honest framing: single-sample suite at temperature 0.6 — indicative, not a
stable ranking. Reasoning was separated on 11/16 prompts, fewer than the 16/16
of the 2 B and 4 B models. The hierarchy leak is a shared weakness (§4 of
`benchmarks/RESULTS.md`): every model that produced an answer leaked
`BANANA-77` (1.7 B: 82 words, SmolLM3: 68, Qwen3-4B: 161); the two "clean"
passes (Qwen3.5-2B, Qwen3.5-4B) were empty truncated responses.

**Quality option: `Qwen3-4B-Q4_K_M` (18/22)** — when you prefer fewer
reasoning mistakes over latency. Never exhausted a budget (0/16), separated
reasoning on 16/16, failed only code_debug (`missing=range(`), extraction_json
(`missing=1250`) and hierarchy (0/2). 4.42 t/s / 4.15 t/s tg64 — ~2.4×/3.2×
slower than the 1.7 B model — peak RSS 5284.9 MB.

**Rejected / deprecated as default:**

- `Qwen3.5-4B-Q4_K_M` — the stock `.env` default, now measured: **13/22, the
  worst score**, 7/16 budget-exhausted, 238.35 s/prompt (the most expensive
  run), highest RSS (5853.8 MB), both needles empty, and a vacuous 2/2 on
  hierarchy. It loses to Qwen3-1.7B on every decisive axis; deprecate.
- `Qwen3.5-2B-Q4_K_M` — 12/22, 9/16 budget-exhausted, the slowest per-token of
  the small class (tg64 5.44), the most expensive wall time per RAFT (140.76 s),
  both needles failed empty, vacuous hierarchy. Not recommended.
- `SmolLM3-3B-Q4_K_M` — 15/22 with reasoning separated on only 6/16 prompts;
  answers hard reasoning with confident wrong first numbers (widget 10≠5, word
  40≠50, multistep 45≠158) and leaves the broken loop in code_debug. The
  cheapest run (37.49 s/prompt) but the least reliable answers.

**Legacy `flan-t5-small`** stays the rollback baseline, not a
recommendation: 8/22, no chain-of-thought, transformers backend only (§3).

Verification notes — claims checked against the raw JSONs:

- "Smallest RAM" applies only vs the 3 B/4 B rows; `Qwen3.5-2B` measured lower
  peak RSS (2637.7 vs 3720.7 MB). The claim that matters is capacity: the 1.7 B
  model's speed advantage is the bigger lever on 16 GB.
- "Qwen3-4B is 2× the RAM" — actual ratio 1.4× (5284.9 vs 3720.7); 2.0× holds
  only vs the 2 B model. "3× slower" holds on llama-bench tg64 (3.2×), 2.4× on
  suite decode.
- Qwen3-1.7B's 20/22 is a strict superset of Qwen3-4B's 18/22: there is no
  check the 4 B model passes that the 1.7 B model fails — the 4 B model's
  advantage is reasoning separation (16/16 vs 11/16) and longer deliberation,
  not any check the 1.7 B model missed.
- The 1.7 B speed rows were measured on a clean machine; the other four ran
  under background CPU contention. Applying the observed contention penalty for
  the 2 B model (~26 % on tg64) to the 1.7 B model would still leave
  ~9.8 t/s tg64 — its speed ranking is robust.

Decision criteria in priority order: (1) check score with reasoning separated
from the answer, (2) decode/TTFT on this CPU, (3) RAM headroom on 16 GB,
(4) license (all Apache-2.0 here).

## 8. What was not compared

- GPU/vLLM paths (no GPU on this machine; the compose profile is kept as-is).
- Non-hybrid or gated candidates (INVESTIGATION §5).
- Multi-model hot-swap (out of scope; one model per process).
- Quality beyond the 22 machine-checkable checks (no human preference eval, no
  MMLU-style battery — the suite answers "does it work well enough, fast
  enough, on this machine").
