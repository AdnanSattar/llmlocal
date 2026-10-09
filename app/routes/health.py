"""/health — public readiness probe (200 "OK" / 503 + detail)."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app import state

router = APIRouter()

_TEXT_HEADERS = {"Content-Type": "text/plain"}


@router.get("/health")
async def health():
    ready, detail = state.BACKEND.health()
    if ready:
        return PlainTextResponse("OK", headers=_TEXT_HEADERS)
    return PlainTextResponse(detail, status_code=503, headers=_TEXT_HEADERS)
