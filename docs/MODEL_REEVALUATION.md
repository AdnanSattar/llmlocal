# MODEL_REEVALUATION — Gemma 4 E2B it vs Qwen3-1.7B on this machine

Date: 2026-10-08 (run finished 16:06, same machine, same day as
`MODEL_RESEARCH.md`). Raw data: `benchmarks/results/gemma-4-e2b-it.json` and
`summary.json`. Companions: `MODEL_RESEARCH.md` (candidate discovery),
`benchmarks/RESULTS.md` (baselines + methodology), `MODEL_COMPARISON.md`
(recommendation).

---

## 1. Run conditions

- Harness: `benchmarks/benchmark_models.py --bench --max-tokens-scale 4
  --timeout 600` (the documented standard in `RESULTS.md`), managed
  llama-server mode — spawns `llama-server` directly on 127.0.0.1:8099, so
  neither `server.py` nor the API-key layer is in the path. 16 prompts /
  22 checks, ctx 8192, threads 4, temp 0.6 / top_p 0.95, streaming,
  `--reasoning-format deepseek`.
- Model: `models/gguf/gemma-4-E2B-it-Q4_K_M.gguf` (bartowski, 3.22 GB on
  disk), label `gemma-4-e2b-it`.
- Server: load 4.67 s, peak RSS 2690.8 MB, avg CPU 325.5%.
- Qwen3-8B: **skipped by explicit decision** — its in-flight run was stopped
  mid-flight, no result; the entry was removed from `benchmarks/models.json`.

## 2. Results vs the incumbent (same suite, same machine, same day)

| Metric | Qwen3-1.7B (default) | Gemma 4 E2B it | Winner |
|---|---|---|---|
| Checks | **20/22** | 19/22 | Qwen3-1.7B |
| Avg decode t/s (suite) | **10.47** | 9.35 | Qwen3-1.7B |
| llama-bench pp512 | **74.27** | 56.05 | Qwen3-1.7B |
| llama-bench tg64 | **13.38** | 11.09 | Qwen3-1.7B |
| Avg TTFT s | **5.49** | 6.36 | Qwen3-1.7B |
| Budget exhausted | **0** | 2 | Qwen3-1.7B |
| Peak RSS MB | 3720.7 | **2690.8** | Gemma (−1.0 GB) |
| Reasoning separated | 11/16 | **13/16** | Gemma |
| Hierarchy check (BANANA-77) | 0/2 | **2/2** | Gemma |
| End-to-end t/s | 7.76 | **8.03** | Gemma † |
| Avg total s/prompt | 58.38 | **43.9** | Gemma † |
| Errors / think-tag leaks | 0 / 0 | 0 / 0 | tie |

† end-to-end and average-total depend on how many tokens each model chose to
emit — Gemma emitted fewer tokens per prompt overall; its per-token decode is
the slower of the two (9.35 vs 10.47 t/s).

For reference, the container baseline (`docker-qwen3-1.7b`) scored the same
20/22 at 10.05 decode t/s.

## 3. The 3 missed checks

1. **`code_debug` (1/2)** — fixed the bug and leaked nothing, but wrote
   `for num in nums` instead of the expected `range(len(nums))` idiom
   (`contains_all: range(` failed). Qwen3-1.7B passes this.
2. **`longctx_2k_needle` (0/1)** and 3. **`longctx_4k_needle` (0/1)** —
   `finish_reason: length` with **empty content**: Gemma spent the entire
   512-token budget on reasoning (`reasoning_chars` 2678 / 2816) and never
   emitted the answer. `budget_exhausted: 2` vs the baseline's 0. The
   baseline answers the same prompts in ~124 tokens (`finish: stop`). This is
   a reasoning-overrun pattern under a hard token cap, not a context or
   retrieval failure.

## 4. Where Gemma beat the incumbent

- **`instruction_hierarchy` (BANANA-77 override): 2/2** — precisely the check
  Qwen3-1.7B fails (0/2; its only misses on the whole suite).
- **Reasoning separation 13/16 vs 11/16**, zero think-tag leaks both.
- **~1 GB lower peak RSS** (2690.8 vs 3720.7 MB).
- Higher end-to-end throughput at lower emitted verbosity (see † above).
- Speed landed at/above the research estimate (§2.2 of `MODEL_RESEARCH.md`
  estimated ~5–9 t/s): usable, clearly in class — the loss is on checks, not
  usability.

## 5. Verdict

**Keep Qwen3-1.7B as the default.** The project rule is replacement only on
clear local evidence; Gemma scored 19/22 vs 20/22, is ~11% slower per decoded
token and ~25%/17% slower in llama-bench pp512/tg64, and exhausted the token
budget on both long-context prompts. Gemma's wins (hierarchy 2/2, CoT
separation 13/16, −1 GB RSS) are real and make it the **strongest known
challenger** — worth keeping on the bench, not installing.

Revisit if any of these change:

- instruction-hierarchy compliance or reasoning separation becomes a hard
  product requirement (Gemma then has a credible case despite the 1-check and
  ~1 t/s deficit), or
- a follow-up run with a larger generation budget (or prompt tweak) fixes the
  two long-context overruns — that alone would make the score 21/22.

## 6. Not verified this session

- No Jetson Orin Nano run (external GPU numbers only — `MODEL_RESEARCH.md` §1).
- Gemma's `<|channel>thought>` → `reasoning_content` mapping worked here
  (13/16 separated, 0 leaks on b11438), but was spot-checked only through the
  suite's fields, not token-level inspection.
- Qwen3-8B remains unmeasured locally by decision (external numbers only).
- No secrets (HF_TOKEN / WORKER_TOKEN / API_KEY) appear in this document.
