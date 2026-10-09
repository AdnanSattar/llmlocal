"""HTTP helpers mirroring the original stdlib server's send/parse/validate
helpers, so status codes, messages, and error envelopes stay identical."""

from __future__ import annotations

import json

from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from backend import BackendError
from app import state


def error_body(message, err_type="server_error"):
    """The exact OpenAI-style error envelope the original server emitted."""
    return {"error": {"message": message, "type": err_type, "code": err_type}}


def error_response(status, message, err_type="server_error"):
    return JSONResponse(status_code=status, content=error_body(message, err_type))


async def read_json_body(request):
    """Read and parse the request body. Returns (obj, error_status)."""
    length = request.headers.get("content-length")
    if length is not None:
        try:
            int(length)
        except (TypeError, ValueError):
            return None, 411
    raw = await request.body()
    if not raw:
        return {}, None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 400
    if not isinstance(obj, dict):
        return None, 400
    return obj, None


def worker_token_from(request):
    """Raw (undecoded) ``worker_token`` query value, as the original parsed it."""
    query = request.scope.get("query_string", b"")
    if isinstance(query, bytes):
        query = query.decode("latin-1")
    for part in query.split("&"):
        if part.startswith("worker_token="):
            return part.split("=", 1)[1]
    return None


def normalize_payload(body):
    """Apply API-level defaults/caps. Returns (payload, error_message)."""
    payload = dict(body)

    raw_max = payload.get("max_tokens", state.MAX_TOKENS_DEFAULT)
    try:
        max_tokens = int(raw_max)
    except (TypeError, ValueError):
        max_tokens = state.MAX_TOKENS_DEFAULT
    payload["max_tokens"] = max(1, min(max_tokens, state.MAX_TOKENS_CAP))

    for field, default in (("temperature", 0.7), ("top_p", 1.0)):
        value = payload.get(field, default)
        if value is None:
            value = default
        try:
            payload[field] = float(value)
        except (TypeError, ValueError):
            return None, f"'{field}' must be a number"
        if field == "top_p" and not 0 < payload[field] <= 1:
            return None, "'top_p' must be in (0, 1]"
        if field == "temperature" and payload[field] < 0:
            return None, "'temperature' must be >= 0"

    if "thinking" in payload and not isinstance(payload["thinking"], bool):
        return None, "'thinking' must be a boolean"
    if "stream" in payload and not isinstance(payload["stream"], bool):
        return None, "'stream' must be a boolean"
    if "messages" in payload and not isinstance(payload["messages"], list):
        return None, "'messages' must be an array"
    return payload, None


def sse_response(chunks):
    """OpenAI-style SSE: ``data: {...}`` per chunk, then ``data: [DONE]``."""
    def gen():
        try:
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except BackendError as e:
            # mid-stream failure: surface it in-band (headers are already sent)
            try:
                err = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                pass
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client went away
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return StreamingResponse(
        gen(),
        status_code=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
