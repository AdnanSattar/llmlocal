"""/queue/metrics — queue and breaker observability (API key guarded)."""

import time

from fastapi import APIRouter, Depends

from app import state
from app.auth import require_api_key

router = APIRouter()


@router.get("/queue/metrics", dependencies=[Depends(require_api_key)])
async def queue_metrics():
    now = time.time()
    state.breaker_maybe_transition(now)
    oldest_age = 0.0
    if state.JOB_QUEUE:
        first_job = state.JOBS[state.JOB_QUEUE[0]]
        oldest_age = max(0.0, now - first_job["enqueued_at"]) if first_job else 0.0
    return {
        "queue_depth": len(state.JOB_QUEUE),
        "queue_oldest_age_sec": oldest_age,
        "last_worker_poll_at": state.LAST_WORKER_POLL_AT,
        "now": now,
        "breaker_state": state.BREAKER_STATE,
        "consecutive_job_fails": state.CONSECUTIVE_JOB_FAILS,
    }
