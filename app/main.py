"""FastAPI application: lifespan (backend + reaper), auth, routers, access log.

Wiring order: create_app() -> FastAPI(...) -> install_auth(app) -> middleware
-> routers -> catch-all 404 (so unknown paths keep the original
``unknown path: ...`` envelope).
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend import create_backend, listen_port
from app import state
from app.auth import install_auth
from app.http import error_body
from app.routes import completions, health, jobs, metrics, models, workers


class AccessLogMiddleware:
    """One line per request on stderr: client, method, path, status.

    Uvicorn's own access log is disabled (``access_log=False``) because it
    prints full URLs including ``?worker_token=...``; this logger never
    emits query strings and never touches Authorization headers.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "")
        path = scope.get("path", "")
        if "worker_token" in path:
            path = path.split("worker_token", 1)[0] + "worker_token=REDACTED"
        client = scope.get("client")
        ip = client[0] if client else "-"

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                sys.stderr.write(f'{ip} - "{method} {path} HTTP/1.1" {message["status"]}\n')
                sys.stderr.flush()
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.BACKEND is None:
            try:
                state.BACKEND = create_backend(os.environ)
            except ValueError as e:
                print(f"Configuration error: {e}", flush=True)
                raise SystemExit(1)
        backend = state.BACKEND
        atexit.register(backend.shutdown)
        state.start_reaper()
        threading.Thread(target=backend.start, daemon=True).start()
        port = listen_port(os.environ)
        backend_name = os.environ.get("LLM_BACKEND", "llama_cpp")
        print(
            f"Starting HTTP server on port {port} "
            f"(backend={backend_name}, model={backend.model_id})",
            flush=True,
        )
        yield
        backend.shutdown()

    app = FastAPI(
        lifespan=lifespan,
        redirect_slashes=False,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    install_auth(app)
    app.add_middleware(AccessLogMiddleware)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(completions.router)
    app.include_router(jobs.router)
    app.include_router(workers.router)
    app.include_router(metrics.router)

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def unknown_path(request: Request):
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        return JSONResponse(
            status_code=404,
            content=error_body(f"unknown path: {path}", "invalid_request_error"),
        )

    return app


app = create_app()
