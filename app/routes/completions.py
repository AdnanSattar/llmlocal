"""/v1/completions and /v1/chat/completions — generation (API key guarded).

Request bodies are read raw and validated with the original rules (no
pydantic request models), so error status codes/messages stay identical.
"""

import traceback

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend import BackendError
from app import state
from app.auth import require_api_key
from app.http import error_response, normalize_payload, read_json_body, sse_response

router = APIRouter()

_OPENAI_SHARED_PROPS = {
    "model": {"type": "string", "description": "Echoed back in the response; any value is accepted."},
    "max_tokens": {"type": "integer", "minimum": 1, "description": "Capped by LLM_MAX_TOKENS_CAP."},
    "temperature": {"type": "number", "minimum": 0, "default": 0.7},
    "top_p": {"type": "number", "default": 1.0},
    "stop": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
    "stream": {
        "type": "boolean",
        "default": False,
        "description": "Server-Sent Events: `data: {...}` chunks, then `data: [DONE]`.",
    },
    "thinking": {
        "type": "boolean",
        "description": "Qwen3 reasoning toggle (adds reasoning_content when true).",
    },
}


def _body_docs(schema, example):
    """Document a JSON request body for Swagger UI only.

    The handlers read the raw body themselves (see ``read_json_body``), so this
    adds the OpenAPI ``requestBody`` without FastAPI parsing/validating it —
    status codes and error messages stay exactly as the original server.
    """
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": schema, "example": example}},
        }
    }


_COMPLETIONS_DOCS = _body_docs(
    {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "system": {"type": "string"},
            **_OPENAI_SHARED_PROPS,
        },
        "required": ["prompt"],
        "additionalProperties": True,
    },
    {
        "model": "local",
        "prompt": "Explain what an API is.",
        "system": "You are a helpful assistant.",
        "max_tokens": 120,
        "temperature": 0.5,
    },
)

_CHAT_DOCS = _body_docs(
    {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                },
            },
            "system": {"type": "string"},
            **_OPENAI_SHARED_PROPS,
        },
        "required": ["messages"],
        "additionalProperties": True,
    },
    {
        "model": "local",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is an API?"},
        ],
        "max_tokens": 120,
        "temperature": 0.5,
    },
)


async def _handle_v1(request: Request, chat: bool):
    ready, detail = state.BACKEND.health()
    if not ready:
        return error_response(503, f"model not ready: {detail}")

    body, err = await read_json_body(request)
    if err == 411:
        return error_response(411, "Content-Length required", "invalid_request_error")
    if err:
        return error_response(err, "invalid JSON body", "invalid_request_error")

    payload, err = normalize_payload(body or {})
    if err:
        return error_response(400, err, "invalid_request_error")

    if chat and not payload.get("messages"):
        return error_response(400, "'messages' is required for chat completions",
                              "invalid_request_error")
    if not chat and "prompt" not in payload:
        return error_response(400, "'prompt' is required for completions",
                              "invalid_request_error")

    stream = payload.get("stream", False)
    endpoint = "chat" if chat else "completions"
    print(
        f"[v1] {endpoint} stream={stream} max_tokens={payload['max_tokens']} "
        f"temp={payload['temperature']}"
        + (f" thinking={payload['thinking']}" if "thinking" in payload else "")
    )

    try:
        if stream:
            chunks = await run_in_threadpool(state.BACKEND.stream, payload, chat)
            return sse_response(chunks)
        response = await run_in_threadpool(state.BACKEND.complete, payload, chat)
        response["model"] = state.BACKEND.model_id
        return JSONResponse(response)
    except BackendError as e:
        return error_response(e.status, str(e))
    except Exception as e:
        traceback.print_exc()
        return error_response(500, f"generation failed: {e}")


@router.post(
    "/v1/completions",
    dependencies=[Depends(require_api_key)],
    openapi_extra=_COMPLETIONS_DOCS,
)
async def completions(request: Request):
    return await _handle_v1(request, False)


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_api_key)],
    openapi_extra=_CHAT_DOCS,
)
async def chat_completions(request: Request):
    return await _handle_v1(request, True)
