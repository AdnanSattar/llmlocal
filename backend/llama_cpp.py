"""llama.cpp backend: spawns ``llama-server`` and proxies OpenAI requests.

This is the recommended CPU backend. It runs the official llama.cpp
release binary (native Windows, no build step) as a child process on
127.0.0.1 and forwards /v1 payloads to it, including SSE streaming
passthrough.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import re
from typing import Iterator, Optional
import time
import urllib.error
import urllib.request
from typing import Iterator

from .base import Backend, BackendError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_model_exists(model_path: str) -> str:
    p = model_path
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    if os.path.isfile(p):
        return p
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(f"model file not found and huggingface_hub unavailable: {p}")
    repo = os.environ.get("LLM_MODEL_REPO") or os.environ.get("LLM_HF_REPO")
    if not repo:
        raise FileNotFoundError(f"model file not found: {p} (set LLM_MODEL_REPO to auto-download)")
    filename = os.path.basename(p)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    local_dir = os.path.dirname(p) or os.path.join(ROOT, "models", "gguf")
    os.makedirs(local_dir, exist_ok=True)
    downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir, token=token, local_dir_use_symlinks=False)
    return downloaded


# Request fields forwarded to llama-server. Unknown fields (our custom
# "thinking", legacy "min_tokens", ...) are consumed here instead.
_ALLOWED = {
    "model", "messages", "prompt", "stream", "stream_options",
    "temperature", "top_p", "top_k", "max_tokens", "stop", "seed", "n",
    "presence_penalty", "frequency_penalty", "logit_bias", "user",
    "chat_template_kwargs", "response_format", "tools", "tool_choice",
    "parallel_tool_calls",
}


def _find_server_exe() -> str:
    candidates = [
        os.path.join(ROOT, "tools", "llama.cpp", "llama-server.exe"),
        os.path.join(ROOT, "tools", "llama.cpp", "llama-server"),
        "llama-server.exe",
        "llama-server",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[-1]


class LlamaCppBackend(Backend):
    def __init__(self, model_id: str, model_path: str, port: int = 8100,
                 context_size: int = 8192, threads: int = 0,
                 thinking: str = "auto", reasoning_format: str = "deepseek",
                 reasoning_budget: int = None, request_timeout: float = 600.0,
                 startup_timeout: float = 180.0, log_path: str = None,
                 extra_args: str = ""):
        self.model_id = model_id
        self.model_path = _ensure_model_exists(model_path)
        self.port = port
        self.context_size = context_size
        self.threads = threads
        self.thinking = thinking
        self.reasoning_format = reasoning_format
        self.reasoning_budget = reasoning_budget
        self.request_timeout = request_timeout
        self.startup_timeout = startup_timeout
        self.extra_args = shlex.split(extra_args) if extra_args else []
        self.log_path = log_path or os.path.join(ROOT, "logs", "llama-server.log")

        self._proc = None
        self._state = "stopped"   # stopped | loading | ready | error
        self._detail = "not started"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._state in ("loading", "ready"):
                return
            if not os.path.isfile(self.model_path):
                self._state = "error"
                self._detail = (
                    f"model file not found: {self.model_path} "
                    "(download a GGUF into models/gguf/)"
                )
                print(f"[backend:llama_cpp] {self._detail}")
                return
            exe = _find_server_exe()
            if not os.path.isfile(exe) and not _which(exe):
                self._state = "error"
                self._detail = (
                    f"llama-server not found (looked for {exe}); download the "
                    "llama.cpp release into tools/llama.cpp/"
                )
                print(f"[backend:llama_cpp] {self._detail}")
                return
            self._state = "loading"
            self._detail = "starting llama-server"

        threading.Thread(target=self._run_server, daemon=True).start()

    def _run_server(self) -> None:
        args = [
            _find_server_exe(),
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.context_size),
            "--alias", self.model_id,
            "--reasoning-format", self.reasoning_format,
            "--reasoning", self.thinking,
            "--parallel", "1",
        ]
        if self.threads:
            args += ["-t", str(self.threads)]
        if self.reasoning_budget:
            args += ["--reasoning-budget", str(self.reasoning_budget)]
        args += self.extra_args

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        try:
            log = open(self.log_path, "ab", buffering=0)
        except OSError:
            log = subprocess.DEVNULL
        try:
            with self._lock:
                if self._state != "loading":
                    return   # shutdown() raced us — do not orphan a child
            print(f"[backend:llama_cpp] spawning: {' '.join(shlex.quote(a) for a in args)}")
            self._proc = subprocess.Popen(
                args,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as e:
            with self._lock:
                self._state = "error"
                self._detail = f"failed to spawn llama-server: {e}"
            print(f"[backend:llama_cpp] {self._detail}")
            return
        finally:
            if hasattr(log, "close"):
                log.close()

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                with self._lock:
                    self._state = "error"
                    self._detail = (
                        f"llama-server exited with code {self._proc.returncode} "
                        f"(see {self.log_path})"
                    )
                print(f"[backend:llama_cpp] {self._detail}")
                return
            ready, detail = self._probe()
            if ready:
                with self._lock:
                    self._state = "ready"
                    self._detail = f"llama-server ready (model {self.model_id})"
                print(f"[backend:llama_cpp] {self._detail}")
                return
            with self._lock:
                self._detail = detail
            time.sleep(1.0)
        with self._lock:
            self._state = "error"
            self._detail = f"llama-server not ready after {self.startup_timeout:.0f}s"
        print(f"[backend:llama_cpp] {self._detail}")

    def health(self):
        with self._lock:
            state, detail = self._state, self._detail
        if state == "ready":
            ready, probe_detail = self._probe()
            return ready, probe_detail if not ready else detail
        return False, detail

    def _probe(self):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True, "llama-server ready"
            return False, "llama-server loading model"
        except urllib.error.HTTPError:
            return False, "llama-server loading model"
        except Exception:
            return False, "llama-server not reachable"

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            self._state = "stopped"
            self._detail = "stopped"
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    # ------------------------------------------------------------------
    # request proxying
    # ------------------------------------------------------------------
    def _prepare(self, payload: dict, chat: bool) -> dict:
        out = {k: v for k, v in payload.items() if k in _ALLOWED}
        system = payload.get("system")
        if isinstance(system, str) and system.strip():
            system = system.strip()
            if chat:
                msgs = list(out.get("messages") or [])
                if not any(isinstance(m, dict) and m.get("role") == "system"
                           for m in msgs):
                    msgs.insert(0, {"role": "system", "content": system})
                    out["messages"] = msgs
            else:
                prompt = out.get("prompt") or ""
                out["prompt"] = (system + "\n\n" + prompt) if prompt else system
        thinking = payload.get("thinking")
        if thinking is not None and chat:
            kwargs = dict(out.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = bool(thinking)
            out["chat_template_kwargs"] = kwargs
        out["stream"] = False
        return out

    def _url(self, chat: bool) -> str:
        path = "/v1/chat/completions" if chat else "/v1/completions"
        return f"http://127.0.0.1:{self.port}{path}"

    def _post(self, data: dict, chat: bool):
        req = urllib.request.Request(
            self._url(chat),
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=self.request_timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            msg = body
            try:
                err = json.loads(body).get("error")
                msg = err.get("message") if isinstance(err, dict) else str(err)
            except Exception:
                pass
            raise BackendError(f"llama-server: {msg}", status=400 if e.code < 500 else 502)
        except (urllib.error.URLError, OSError) as e:
            raise BackendError(f"llama-server unreachable: {e}")

    def complete(self, payload: dict, chat: bool) -> dict:
        ready, detail = self.health()
        if not ready:
            raise BackendError(f"model not ready: {detail}", status=503)
        data = self._prepare(payload, chat)
        with self._post(data, chat) as resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
        if isinstance(obj, dict):
            obj["model"] = self.model_id
        return obj

    def stream(self, payload: dict, chat: bool) -> Iterator[dict]:
        """Eagerly check readiness and open the upstream connection, then
        yield SSE chunks. Errors raise *before* any chunk is yielded so the
        API can still respond with a normal HTTP error."""
        ready, detail = self.health()
        if not ready:
            raise BackendError(f"model not ready: {detail}", status=503)
        data = self._prepare(payload, chat)
        data["stream"] = True
        resp = self._post(data, chat)
        return self._iter_stream(resp)

    def _iter_stream(self, resp) -> Iterator[dict]:
        try:
            ctype = resp.headers.get("Content-Type", "")
            if "text/event-stream" not in ctype:
                obj = json.loads(resp.read().decode("utf-8", "replace"))
                obj["model"] = self.model_id
                yield obj
                return
            for obj in _iter_sse(resp):
                obj["model"] = self.model_id
                yield obj
        finally:
            resp.close()


def _iter_sse(resp) -> Iterator[dict]:
    """Parse an SSE body, yielding the JSON payloads of ``data:`` lines."""
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
            yield json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue


def _which(exe: str) -> bool:
    import shutil
    return shutil.which(exe) is not None
