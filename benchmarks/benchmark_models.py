#!/usr/bin/env python3
"""CPU LLM benchmark suite for local OpenAI-compatible backends.

Two modes:

1) Managed mode (default): spawns llama-server per model listed in
   benchmarks/models.json, runs the shared prompt suite against each,
   records latency/throughput/RAM, then shuts the server down.

2) External mode: benchmarks any already-running OpenAI-compatible
   endpoint, e.g. the legacy transformers server:

       python benchmarks/benchmark_models.py --base http://localhost:8000 \
           --label flan-t5-small-transformers --api chat

Stdlib only (psutil is optional and used for RAM/CPU sampling).

Outputs: benchmarks/results/<label>.json  (raw outputs + metrics)
         benchmarks/results/summary.json   (one row per label, appended)
"""

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import psutil
except Exception:
    psutil = None

REFUSAL_PATTERNS = [
    r"(?i)\bnot (mentioned|provided|available|stated|specified|included|present|given)\b",
    r"(?i)\bno (information|mention|such|details|record)\b",
    r"(?i)\b(cannot|can't|could not|couldn't|unable to)\b",
    r"(?i)\b(doesn't|does not|don't|do not) (say|mention|specify|provide|contain|include|state|list)\b",
    r"(?i)\bcontext does not\b",
    r"(?i)\b(the (passage|text) (does not|doesn't))\b",
    r"(?i)\bunknown\b",
    r"(?i)\bno way to (know|determine)\b",
]


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def build_passage(spec, filler):
    target = int(spec["target_chars"])
    parts, total, i = [], 0, 0
    while total < target:
        p = filler[i % len(filler)]
        parts.append(p)
        total += len(p) + 2
        i += 1
    text = "\n\n".join(parts)
    pos = int(len(text) * float(spec.get("needle_at", 0.5)))
    text = text[:pos] + spec["needle"] + " " + text[pos:]
    return text


def materialize_prompts(suite):
    filler = suite.get("long_context_filler", [])
    out = []
    for p in suite["prompts"]:
        p = json.loads(json.dumps(p))  # deep copy
        if "passage" in p:
            passage = build_passage(p["passage"], filler)
            suffix = p.pop("user_suffix", "")
            for m in p["messages"]:
                if m.get("content") == "PLACEHOLDER":
                    m["content"] = "Operating logs:\n\n" + passage + suffix
        out.append(p)
    return out


def flan_style_prompt(messages):
    """Mirror the legacy server.py prompt wrapper for completions-only models."""
    system = " ".join(m["content"] for m in messages if m.get("role") == "system").strip()
    user = " ".join(m["content"] for m in messages if m.get("role") == "user").strip()
    if system:
        return f"Instruction: {system}\nQuestion: {user}\nAnswer:".strip()
    return f"Question: {user}\nAnswer:".strip()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _first_token(text):
    m = re.search(r"[A-Za-z0-9']+", text or "")
    return m.group(0).lower() if m else ""


def _first_number(text):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def strip_fences(text):
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", t, re.S)
    if m:
        return m.group(1).strip()
    t = re.sub(r"^```[a-zA-Z]*\s*\n", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def run_check(content, chk):
    t = chk["type"]
    if t == "contains_all":
        low = (content or "").lower()
        ok = all(v.lower() in low for v in chk["values"])
        return ok, "missing=" + ",".join(v for v in chk["values"] if v.lower() not in low)
    if t == "contains_any":
        low = (content or "").lower()
        return any(v.lower() in low for v in chk["values"]), ""
    if t == "not_contains":
        low = (content or "").lower()
        bad = [v for v in chk["values"] if v.lower() in low]
        return not bad, "leaked=" + ",".join(bad)
    if t == "regex":
        return bool(re.search(chk["pattern"], content or "")), ""
    if t == "first_word_in":
        fw = _first_token(content)
        ok = fw in [v.lower() for v in chk["values"]]
        return ok, f"first_word={fw!r}"
    if t == "first_number_equals":
        n = _first_number(content)
        if n is None:
            return False, "no number found"
        ok = abs(n - float(chk["value"])) <= float(chk.get("tol", 0.001))
        return ok, f"found={n}"
    if t == "sentence_count":
        sents = [s for s in re.split(r"(?<=[.!?])\s+", (content or "").strip()) if len(s.strip()) > 2]
        n = len(sents)
        return chk["min"] <= n <= chk["max"], f"sentences={n}"
    if t == "word_count_max":
        n = len((content or "").split())
        return n <= chk["max"], f"words={n}"
    if t == "min_numbered_steps":
        n = len(re.findall(r"^\s*(?:\d+[.)]|[-*•])\s+\S", content or "", re.M))
        return n >= chk["min"], f"steps={n}"
    if t == "refusal":
        for pat in REFUSAL_PATTERNS:
            if re.search(pat, content or ""):
                return True, "refusal matched"
        return False, "no refusal pattern"
    if t == "json_object":
        try:
            obj = json.loads(strip_fences(content))
            if isinstance(obj, dict):
                return True, ""
            return False, f"not an object: {type(obj).__name__}"
        except Exception as e:
            return False, f"json error: {e}"
    if t == "json_keys":
        try:
            obj = json.loads(strip_fences(content))
            have = {k.lower() for k in obj}
            missing = [k for k in chk["keys"] if k.lower() not in have]
            return not missing, "missing=" + ",".join(missing)
        except Exception as e:
            return False, f"json error: {e}"
    if t == "json_contains":
        try:
            obj = json.loads(strip_fences(content))
            blob = json.dumps(obj).lower()
            missing = [v for v in chk["values"] if v.lower() not in blob]
            return not missing, "missing=" + ",".join(missing)
        except Exception as e:
            return False, f"json error: {e}"
    return False, f"unknown check {t}"


def stream_request(url, payload, timeout):
    """POST payload with stream=true; fall back to plain JSON if not SSE.

    Returns dict with content/reasoning/timings. TTFT = first generated token
    (reasoning or content).
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    out = {
        "content": "", "reasoning": "", "ttft_s": None, "total_s": None,
        "completion_tokens": None, "usage": None, "timings": None,
        "finish_reason": None, "streamed": False, "error": None,
    }
    t0 = time.perf_counter()
    t_first = None
    deltas = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "text/event-stream" not in ctype:
                body = resp.read().decode("utf-8", "replace")
                obj = json.loads(body)
                if isinstance(obj, dict) and obj.get("error"):
                    err = obj["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    out["error"] = f"api error: {msg}"
                    return out
                try:
                    choice = obj["choices"][0]
                    out["content"] = choice.get("message", {}).get("content") or choice.get("text") or ""
                    out["finish_reason"] = choice.get("finish_reason")
                except Exception:
                    out["error"] = "unrecognized response shape"
                    return out
                t_first = time.perf_counter()
                out["usage"] = obj.get("usage")
                out["timings"] = obj.get("timings")
                out["streamed"] = False
            else:
                out["streamed"] = True
                buf = b""
                while True:
                    chunk = resp.readline()
                    if not chunk:
                        break
                    buf += chunk
                    if not buf.endswith(b"\n"):
                        continue
                    line = buf.strip()
                    buf = b""
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        break
                    try:
                        obj = json.loads(raw.decode("utf-8", "replace"))
                    except Exception:
                        continue
                    if obj.get("usage"):
                        out["usage"] = obj["usage"]
                    if obj.get("timings"):
                        out["timings"] = obj["timings"]
                    for ch in obj.get("choices", []):
                        if ch.get("finish_reason"):
                            out["finish_reason"] = ch["finish_reason"]
                        if "delta" in ch:
                            delta = ch.get("delta") or {}
                            rc = delta.get("reasoning_content") or ""
                            ct = delta.get("content") or ""
                        else:
                            rc = ""
                            ct = ch.get("text") or ""
                        if rc or ct:
                            if t_first is None:
                                t_first = time.perf_counter()
                            deltas += 1
                            out["reasoning"] += rc
                            out["content"] += ct
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    t_end = time.perf_counter()
    out["total_s"] = t_end - t0
    if t_first is not None:
        out["ttft_s"] = t_first - t0
        if out["usage"] and out["usage"].get("completion_tokens"):
            out["completion_tokens"] = out["usage"]["completion_tokens"]
        else:
            out["completion_tokens"] = deltas
    return out


# ---------------------------------------------------------------------------
# resource sampling
# ---------------------------------------------------------------------------

class ResourceSampler(threading.Thread):
    def __init__(self, proc=None, interval=0.5):
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval
        self.stop_evt = threading.Event()
        self.peak_rss_mb = 0.0
        self.avg_cpu_pct = None
        self._cpu_samples = []
        self._sys_samples = []

    def run(self):
        prev = None
        while not self.stop_evt.wait(self.interval):
            self._sample(prev)
            prev = time.time()
        self._sample(prev)

    def _sample(self, prev):
        if psutil is None:
            return
        try:
            if self.proc is not None:
                p = psutil.Process(self.proc.pid)
                rss = p.memory_info().rss
                self.peak_rss_mb = max(self.peak_rss_mb, rss / (1024 * 1024))
                ct = p.cpu_times()
                prev_ct = getattr(self, "_prev_ct", None)
                if prev_ct is not None:
                    wall = time.time() - getattr(self, "_prev_wall", time.time())
                    cpu = (ct.user - prev_ct.user) + (ct.system - prev_ct.system)
                    if wall > 0:
                        self._cpu_samples.append(100.0 * cpu / wall)
                self._prev_ct = ct
                self._prev_wall = time.time()
            self._sys_samples.append(psutil.cpu_percent(interval=None))
        except Exception:
            pass

    def finish(self):
        self.stop_evt.set()
        self.join(timeout=3)
        if self._cpu_samples:
            self.avg_cpu_pct = sum(self._cpu_samples) / len(self._cpu_samples)


def host_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    if psutil:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    return info


# ---------------------------------------------------------------------------
# llama-server management
# ---------------------------------------------------------------------------

def find_tool(name):
    base = os.path.join(ROOT, "tools", "llama.cpp")
    for r, _d, files in os.walk(base):
        if name in files:
            return os.path.join(r, name)
    return name


def wait_health(base_url, timeout_s=240, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code == 503:
                pass
            else:
                raise
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server at {base_url} not healthy after {timeout_s}s")


def start_llama_server(cfg, port, threads):
    exe = find_tool("llama-server.exe")
    log_path = os.path.join(ROOT, "benchmarks", "results", f"server_{cfg['label']}.log")
    args = [
        exe,
        "-m", os.path.join(ROOT, cfg["gguf"]),
        "-c", str(cfg.get("ctx", 8192)),
        "-t", str(threads),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--alias", cfg["label"],
        "--reasoning-format", "deepseek",
    ]
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    base_url = f"http://127.0.0.1:{port}"
    load_s = wait_health(base_url, proc=proc)
    return proc, base_url, load_s, log


# ---------------------------------------------------------------------------
# suite execution
# ---------------------------------------------------------------------------

def run_suite(base_url, api, prompts, cfg, timeout, skip_categories=()):
    results = []
    endpoint = base_url + ("/v1/chat/completions" if api == "chat" else "/v1/completions")
    for p in prompts:
        if p.get("category") in skip_categories:
            results.append({"id": p["id"], "category": p["category"], "skipped": True})
            print(f"  skip {p['id']} (category skipped)")
            continue
        if api == "chat":
            payload = {
                "model": cfg.get("label", "benchmark"),
                "messages": p["messages"],
                "max_tokens": p["max_tokens"],
                "temperature": cfg.get("temperature", 0.6),
                "top_p": cfg.get("top_p", 0.95),
                "stream": True,
            }
        else:
            payload = {
                "model": cfg.get("label", "benchmark"),
                "prompt": flan_style_prompt(p["messages"]),
                "max_tokens": p["max_tokens"],
                "temperature": cfg.get("temperature", 0.2),
                "top_p": cfg.get("top_p", 1.0),
                "stream": True,
            }
        t_wall = time.perf_counter()
        r = stream_request(endpoint, payload, timeout=timeout)
        wall = time.perf_counter() - t_wall
        content = r["content"]
        checks = []
        if not r["error"]:
            for chk in p.get("checks", []):
                ok, detail = run_check(content, chk)
                checks.append({"type": chk["type"], "passed": ok, "detail": detail})
        passed = sum(1 for c in checks if c["passed"])
        reasoning = r["reasoning"] or ""
        leaked = ("<think>" in (content or "")) or ("</think>" in (content or "")) \
                 or ("<think>" in (content or "").lower())
        tok = r["completion_tokens"] or 0
        decode_tps = None
        decode_window = (r["total_s"] - r["ttft_s"]) if (r["ttft_s"] is not None and r["total_s"] is not None) else 0.0
        if tok and decode_window >= 0.05:
            decode_tps = round((tok - 1) / decode_window, 2)
        row = {
            "id": p["id"],
            "category": p["category"],
            "error": r["error"],
            "streamed": r["streamed"],
            "ttft_s": round(r["ttft_s"], 3) if r["ttft_s"] is not None else None,
            "total_s": round(wall, 3),
            "decode_tps": decode_tps,
            "completion_tokens": tok,
            "usage": r["usage"],
            "timings": r["timings"],
            "finish_reason": r["finish_reason"],
            "reasoning_chars": len(reasoning),
            "reasoning_head": reasoning[:400],
            "reasoning_separated": (len(reasoning) > 0),
            "think_tags_in_content": leaked,
            "content": content,
            "checks": checks,
            "checks_passed": passed,
            "checks_total": len(checks),
        }
        results.append(row)
        status = "ERR" if r["error"] else f"{passed}/{len(checks)}"
        tt = f"{r['ttft_s']:.2f}s ttft" if r["ttft_s"] is not None else "n/a"
        print(f"  {p['id']:<28} {wall:6.1f}s total  {tt:>10}  {decode_tps or 0:6.2f} t/s  checks={status}")
    return results


def summarize(results):
    ok = [r for r in results if not r.get("skipped")]
    passed = sum(r.get("checks_passed", 0) for r in ok)
    total = sum(r.get("checks_total", 0) for r in ok)
    errors = sum(1 for r in ok if r.get("error"))
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    decs = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    totals = [r["total_s"] for r in ok if r.get("total_s") is not None]
    comp_toks = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    total_wall = sum(totals)
    return {
        "prompts": len(ok),
        "errors": errors,
        "checks_passed": passed,
        "checks_total": total,
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "avg_decode_tps": round(sum(decs) / len(decs), 2) if decs else None,
        "avg_total_s": round(sum(totals) / len(totals), 2) if totals else None,
        "end_to_end_tps": round(sum(comp_toks) / total_wall, 2) if (comp_toks and total_wall > 0) else None,
        "budget_exhausted": sum(1 for r in ok if r.get("finish_reason") == "length"),
        "reasoning_separated_prompts": sum(1 for r in ok if r.get("reasoning_separated")),
        "think_tag_leaks": sum(1 for r in ok if r.get("think_tags_in_content")),
    }


def llama_bench(cfg, threads, out_dir):
    exe = find_tool("llama-bench.exe")
    if not os.path.exists(exe):
        print("  llama-bench.exe not found, skipping")
        return None
    args = [exe, "-m", os.path.join(ROOT, cfg["gguf"]), "-p", "512", "-n", "64",
            "-t", str(threads), "-o", "json"]
    try:
        raw = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                             timeout=1800, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:
        print(f"  llama-bench failed: {e}")
        return None
    text = (raw.stdout or "").strip()
    try:
        rows = json.loads(text)
    except Exception:
        tail = ((raw.stderr or "").strip().splitlines() or ["no output"])[-3:]
        print(f"  llama-bench rc={raw.returncode}, skipped: {' | '.join(tail)}")
        return {"error": f"rc={raw.returncode}: {tail[-1][:160]}"}
    out = {}
    for row in rows:
        if row.get("n_prompt") and not row.get("n_gen"):
            out["pp512_tps"] = round(row.get("avg_ts", 0), 2)
        if row.get("n_gen") and not row.get("n_prompt"):
            out["tg64_tps"] = round(row.get("avg_ts", 0), 2)
    return out


def append_summary(out_dir, entry):
    path = os.path.join(out_dir, "summary.json")
    rows = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            rows = []
    rows = [r for r in rows if r.get("label") != entry.get("label")]
    rows.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.path.join(ROOT, "benchmarks", "models.json"))
    ap.add_argument("--prompts", default=os.path.join(ROOT, "benchmarks", "prompts.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results"))
    ap.add_argument("--only", default=None, help="comma-separated label substrings to filter models")
    ap.add_argument("--base", default=None, help="benchmark an external endpoint instead of spawning servers")
    ap.add_argument("--label", default=None, help="label for external endpoint mode")
    ap.add_argument("--api", default="chat", choices=["chat", "completions"])
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=300, help="per-request timeout seconds")
    ap.add_argument("--bench", action="store_true", help="also run llama-bench (pp512/tg64)")
    ap.add_argument("--only-ids", default=None, help="comma-separated prompt ids")
    ap.add_argument("--max-tokens-scale", type=float, default=1.0,
                    help="multiply every prompt's max_tokens; reasoning models spend budget on "
                         "thinking, so scale > 1 keeps them from finishing before they answer")
    ap.add_argument("--max-tokens-cap", type=int, default=4096,
                    help="ceiling for scaled max_tokens; keep <= server LLM_MAX_TOKENS_CAP")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.prompts, encoding="utf-8") as f:
        suite = json.load(f)
    prompts = materialize_prompts(suite)
    if args.only_ids:
        wanted = {s.strip() for s in args.only_ids.split(",")}
        prompts = [p for p in prompts if p["id"] in wanted]
    for p in prompts:
        p["max_tokens"] = max(1, min(int(round(p["max_tokens"] * args.max_tokens_scale)), args.max_tokens_cap))

    print(f"host: {host_info()}")

    if args.base:
        label = args.label or args.base
        cfg = {"label": label, "temperature": args.temperature, "top_p": args.top_p}
        print(f"== benchmarking external endpoint {args.base} as '{label}' ==")
        sampler = ResourceSampler(proc=None)
        sampler.start()
        results = run_suite(args.base, args.api, prompts, cfg, args.timeout)
        sampler.finish()
        entry = {"label": label, "mode": "external", "api": args.api, "summary": summarize(results)}
    else:
        with open(args.models, encoding="utf-8") as f:
            mconf = json.load(f)
        threads = args.threads or mconf.get("threads", 4)
        port = args.port or mconf.get("port", 8099)
        models = mconf["models"]
        if args.only:
            keys = [k.strip().lower() for k in args.only.split(",")]
            models = [m for m in models if any(k in m["label"].lower() for k in keys)]
        entry = None
        for cfg in models:
            print(f"== {cfg['label']} ==")
            proc = None
            log = None
            try:
                proc, base_url, load_s, log = start_llama_server(cfg, port, threads)
                print(f"  loaded in {load_s:.1f}s")
                sampler = ResourceSampler(proc=proc)
                sampler.start()
                bench = llama_bench(cfg, threads, args.out) if args.bench else None
                warm = {
                    "model": cfg["label"],
                    "messages": [{"role": "user", "content": "Say OK."}],
                    "max_tokens": 8, "temperature": 0.6, "stream": True,
                } if cfg.get("api", "chat") == "chat" else {
                    "model": cfg["label"], "prompt": "Say OK.", "max_tokens": 8, "stream": True,
                }
                ep = base_url + ("/v1/chat/completions" if cfg.get("api", "chat") == "chat" else "/v1/completions")
                stream_request(ep, warm, timeout=120)
                results = run_suite(base_url, cfg.get("api", "chat"), prompts, cfg,
                                    args.timeout, skip_categories=cfg.get("skip_categories", ()))
                sampler.finish()
                summary = summarize(results)
                if bench:
                    summary["llama_bench"] = bench
                doc = {
                    "label": cfg["label"],
                    "mode": "llama_server",
                    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "api": cfg.get("api", "chat"),
                    "gguf": cfg.get("gguf"),
                    "threads": threads,
                    "ctx": cfg.get("ctx"),
                    "sampling": {"temperature": cfg.get("temperature"), "top_p": cfg.get("top_p")},
                    "host": host_info(),
                    "server": {
                        "load_s": round(load_s, 2),
                        "peak_rss_mb": round(sampler.peak_rss_mb, 1),
                        "avg_cpu_pct": round(sampler.avg_cpu_pct, 1) if sampler.avg_cpu_pct else None,
                        "sys_cpu_pct": round(sum(sampler._sys_samples) / len(sampler._sys_samples), 1) if sampler._sys_samples else None,
                    },
                    "summary": summary,
                    "results": results,
                }
                with open(os.path.join(args.out, f"{cfg['label']}.json"), "w", encoding="utf-8") as f:
                    json.dump(doc, f, indent=2)
                append_summary(args.out, {
                    "label": cfg["label"], "gguf": cfg.get("gguf"),
                    "mode": "llama_server", "api": cfg.get("api", "chat"),
                    "load_s": doc["server"]["load_s"],
                    "peak_rss_mb": doc["server"]["peak_rss_mb"],
                    **summary,
                })
                entry = doc
                print(f"  summary: {json.dumps(summary)}")
            except Exception as e:
                print(f"  FAILED: {e}")
                append_summary(args.out, {"label": cfg["label"], "failed": str(e)})
            finally:
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        pass
                if log is not None:
                    log.close()
                time.sleep(1.0)

    if args.base and entry:
        with open(os.path.join(args.out, f"{entry['label']}.json"), "w", encoding="utf-8") as f:
            json.dump({"label": entry["label"], "mode": "external", "api": entry["api"],
                       "host": host_info(), "summary": entry["summary"], "results": results},
                      f, indent=2)
        append_summary(args.out, {"label": entry["label"], **entry["summary"]})
        print(f"summary: {json.dumps(entry['summary'])}")


if __name__ == "__main__":
    main()
