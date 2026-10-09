"""Backend factory.

``LLM_BACKEND`` selects the runtime:

* ``llama_cpp`` (default) — llama.cpp ``llama-server`` subprocess serving a
  GGUF model. Recommended for CPU-only machines.
* ``transformers`` — the original HuggingFace pipeline (rollback path).

All configuration is environment-based; see ``.env.example``.
``VLLM_*`` names are accepted as aliases for the original compose files.
"""

from __future__ import annotations

import os

from .base import Backend, BackendError  # noqa: F401  (re-exported)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(env, name, default=None):
    value = env.get(name, None)
    if value is None or value == "":
        return default
    return value


def _int(env, name, default):
    try:
        return int(_env(env, name, default))
    except (TypeError, ValueError):
        return default


def load_env_file(path: str = None) -> None:
    """Minimal .env loader (no dependency). Real environment wins."""
    path = path or os.path.join(ROOT, ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def listen_port(env) -> int:
    """API port: LLM_PORT, alias VLLM_PORT, default 8000."""
    return _int(env, "LLM_PORT", _int(env, "VLLM_PORT", 8000))


def token_limits(env):
    """(default max_tokens when the request omits it, hard cap)."""
    default = _int(env, "LLM_MAX_TOKENS", 1024)
    cap = _int(env, "LLM_MAX_TOKENS_CAP", 4096)
    if cap < 1:
        cap = 4096
    default = max(1, min(default, cap))
    return default, cap


def create_backend(env=None) -> Backend:
    env = env if env is not None else os.environ
    kind = (_env(env, "LLM_BACKEND", "llama_cpp") or "llama_cpp").strip().lower()
    max_default, max_cap = token_limits(env)

    if kind in ("llama_cpp", "llamacpp", "llama", "llama-cpp"):
        from .llama_cpp import LlamaCppBackend
        model_path = _env(env, "LLM_MODEL_PATH",
                          "models/gguf/Qwen3-1.7B-Q4_K_M.gguf")
        stem = os.path.splitext(os.path.basename(model_path))[0]
        if stem.startswith("Qwen_Qwen"):
            stem = stem.split("_", 1)[1]
        model_id = _env(env, "LLM_MODEL", stem)
        return LlamaCppBackend(
            model_id=model_id,
            model_path=model_path,
            port=_int(env, "LLM_BACKEND_PORT", 8100),
            context_size=_int(env, "LLM_CONTEXT_SIZE", 8192),
            threads=_int(env, "LLM_THREADS", 0),
            thinking=(_env(env, "LLM_THINKING", "auto") or "auto").lower(),
            reasoning_format=(_env(env, "LLM_REASONING_FORMAT", "deepseek")
                              or "deepseek").lower(),
            reasoning_budget=_int(env, "LLM_REASONING_BUDGET", 0) or None,
            request_timeout=float(_int(env, "LLM_REQUEST_TIMEOUT", 600)),
            startup_timeout=float(_int(env, "LLM_STARTUP_TIMEOUT", 180)),
            log_path=_env(env, "LLM_LOG", os.path.join(ROOT, "logs", "llama-server.log")),
            extra_args=_env(env, "LLM_EXTRA_ARGS", "") or "",
        )

    if kind in ("transformers", "hf", "legacy"):
        from .transformers_legacy import TransformersBackend
        model_name = _env(env, "TRANSFORMERS_MODEL", "google/flan-t5-small")
        model_id = _env(env, "LLM_MODEL", model_name)
        return TransformersBackend(
            model_id=model_id,
            model_name=model_name,
            max_tokens_default=max_default,
            max_tokens_cap=max_cap,
        )

    raise ValueError(
        f"unknown LLM_BACKEND '{kind}' (expected 'llama_cpp' or 'transformers')"
    )
