"""/jobs (enqueue, API key), /jobs/{id} (status, API key),
/jobs/next (worker long-poll, WORKER_TOKEN query param).

``jobs_next`` is a plain ``def`` on purpose: FastAPI runs sync endpoints in
the threadpool, so the blocking 25 s condition wait never stalls the event
loop and /health, /v1/* keep answering while a poll is held.
"""

import time
import uuid

from fastapi import APIRouter, Depends, Request

from app import state
from app.auth import require_api_key
from app.http import error_response, read_json_body, worker_token_from

router = APIRouter()

_JOBS_DOCS = {
    "requestBody": {
        "required": False,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Stored verbatim as the job payload and delivered to workers via /jobs/next.",
                },
                "example": {"prompt": "Say ok.", "max_tokens": 16},
            }
        },
    }
}

LONG_POLL_SEC = 25


@router.post("/jobs", dependencies=[Depends(require_api_key)], openapi_extra=_JOBS_DOCS)
async def enqueue_job(request: Request):
    body, err = await read_json_body(request)
    if err:
        return error_response(err, "invalid JSON body", "invalid_request_error")
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "payload": body or {},
        "status": "queued",
        "enqueued_at": time.time(),
        "lease_until": 0,
        "attempts": 0,
        "result": None,
    }
    state.JOBS[job_id] = job
    with state.queue_cv:
        state.JOB_QUEUE.append(job_id)
        state.queue_cv.notify_all()
    # Re-evaluate breaker on enqueue
    state.breaker_maybe_transition(time.time())
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = state.JOBS.get(job_id)
    if not job:
        return error_response(404, f"job {job_id} not found", "invalid_request_error")
    # do not include full payload for brevity
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "enqueued_at": job.get("enqueued_at"),
        "lease_until": job.get("lease_until", 0),
        "attempts": job.get("attempts", 0),
        "result": job.get("result"),
    }


@router.post("/jobs/next")
def jobs_next(request: Request):
    if worker_token_from(request) != state.WORKER_TOKEN:
        return error_response(403, "invalid worker token", "invalid_request_error")
    state.LAST_WORKER_POLL_AT = time.time()
    state.breaker_maybe_transition(state.LAST_WORKER_POLL_AT)
    # Long-poll up to 25s
    deadline = time.time() + LONG_POLL_SEC
    job_id = None
    with state.queue_cv:
        while not state.JOB_QUEUE and time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            state.queue_cv.wait(timeout=min(remaining, 1.0))
        if state.JOB_QUEUE:
            job_id = state.JOB_QUEUE.popleft()
            job = state.JOBS.get(job_id)
            if job:
                job["status"] = "leased"
                job["lease_until"] = time.time() + state.LEASE_SEC
    if not job_id:
        # empty response to indicate no work
        return {"job": None}
    job = state.JOBS.get(job_id, {})
    # Return minimal job view to worker
    return {
        "job_id": job_id,
        "payload": job.get("payload", {}),
        "probe": (state.BREAKER_STATE == "HALF_OPEN"),
    }
