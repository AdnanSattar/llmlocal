"""/v1/models — OpenAI model list (guarded by the API key)."""

import time

from fastapi import APIRouter, Depends

from app import state
from app.auth import require_api_key
from app.http import error_response

router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
async def models():
    ready, detail = state.BACKEND.health()
    if not ready:
        return error_response(503, f"model not ready: {detail}")
    return {
        "object": "list",
        "data": [
            {
                "id": state.BACKEND.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llmlocal",
            }
        ],
    }
