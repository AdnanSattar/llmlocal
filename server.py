"""llmlocal — OpenAI-compatible local LLM API + reverse-webhook job queue.

FastAPI/uvicorn entrypoint (migrated from stdlib ``http.server``; the HTTP
surface itself lives in ``app/``):

* ``python server.py`` (native) and ``python3 /app/server.py`` (Docker) both
  load ``.env``, create the backend, and serve the app from ``app/main.py``.
* ``LLM_BACKEND=llama_cpp`` (default) — llama.cpp ``llama-server`` subprocess
  serving a GGUF model (recommended on CPU-only machines).
* ``LLM_BACKEND=transformers`` — the original HuggingFace pipeline (rollback).

The job queue, circuit breaker, and worker endpoints are unchanged:

# Flow (reverse-webhook):
# 1) Clients call POST /jobs with a payload shaped like the /v1/completions body.
# 2) A LAN worker process long-polls POST /jobs/next?worker_token=... to fetch work.
# 3) The LAN worker calls local model (e.g., http://localhost:8000/v1/completions).
# 4) The LAN worker POSTs results to /webhooks/completions with {job_id, result}.
# 5) Clients can query GET /jobs/{job_id} for status/result.
#
# Circuit breaker marks local processing as unavailable when:
# - No worker polls for a configured window; or
# - Queue age/depth exceeds thresholds; or
# - Consecutive failures exceed a threshold.
#
# For production, persist jobs to a durable store (e.g., Redis/DB) and protect
# worker endpoints with mTLS or strong auth. This in-memory version is simple.
"""

import os
import sys

from backend import create_backend, load_env_file, listen_port

# Load .env (real environment variables win) before reading config.
load_env_file()


def main() -> None:
    # Create the backend up front: config errors print and exit 1 before
    # anything binds a port (same fail-fast behaviour as the old server).
    try:
        backend = create_backend(os.environ)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    import uvicorn

    from app import state
    from app.main import app

    state.BACKEND = backend

    host = (os.environ.get("HOST") or "").strip() or "0.0.0.0"
    port = listen_port(os.environ)

    # access_log=False: uvicorn's access log prints full URLs including
    # ?worker_token=... — the app logs method+path only (app/main.py).
    uvicorn.run(app, host=host, port=port, access_log=False)


if __name__ == "__main__":
    main()
