"""/webhooks/completions (worker result sink) and /workers/heartbeat.

Worker authentication is the WORKER_TOKEN query param — never the API key.
"""

import time

from fastapi import APIRouter, Request

from app import state
from app.http import error_response, read_json_body, worker_token_from

router = APIRouter()

_WEBHOOK_DOCS = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "result": {"description": "Anything the worker produced; stored on the job."},
                        "success": {"type": "boolean", "default": True},
                        "probe": {
                            "type": "boolean",
                            "default": False,
                            "description": "Marks a breaker HALF_OPEN probe result.",
                        },
                    },
                    "required": ["job_id"],
                    "additionalProperties": True,
                },
                "example": {
                    "job_id": "00000000-0000-0000-0000-000000000000",
                    "result": "ok",
                    "success": True,
                },
            }
        },
    }
}


@router.post("/webhooks/completions", openapi_extra=_WEBHOOK_DOCS)
async def webhook_completions(request: Request):
    body, err = await read_json_body(request)
    if err:
        return error_response(err, "invalid JSON body", "invalid_request_error")
    body = body or {}
    job_id = body.get("job_id")
    success = bool(body.get("success", True))
    job = state.JOBS.get(job_id)
    if not job:
        return error_response(404, f"job {job_id} not found", "invalid_request_error")
    job["result"] = body.get("result")
    job["status"] = "completed" if success else "failed"
    job["lease_until"] = 0
    if success:
        state.CONSECUTIVE_JOB_FAILS = 0
    else:
        state.CONSECUTIVE_JOB_FAILS += 1
    # If this was a probe, inform breaker result
    if body.get("probe") is True:
        state.breaker_probe_result(success)
    return {"ok": True}


@router.post("/workers/heartbeat")
async def workers_heartbeat(request: Request):
    if worker_token_from(request) != state.WORKER_TOKEN:
        return error_response(403, "invalid worker token", "invalid_request_error")
    state.LAST_WORKER_POLL_AT = time.time()
    state.breaker_maybe_transition(state.LAST_WORKER_POLL_AT)
    return {"ok": True, "breaker_state": state.BREAKER_STATE}
