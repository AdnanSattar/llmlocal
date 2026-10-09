"""Shared process state: backend handle, job queue, and circuit breaker.

Ported from the original stdlib ``server.py``, which held these as module
globals, so queue/breaker/lease semantics are preserved exactly. Mutation
happens from request handlers (threadpool) and the reaper thread, exactly
as before.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from backend import load_env_file, token_limits

# Load .env (real environment variables win) before reading config.
load_env_file()

# Backend instance. server.py assigns it before the app starts; when the app
# is served standalone (uvicorn app.main:app) the lifespan calls
# create_backend(os.environ) instead.
BACKEND = None

MAX_TOKENS_DEFAULT, MAX_TOKENS_CAP = token_limits(os.environ)

# Queue and job tracking
JOB_QUEUE = deque()  # holds job_ids
JOBS = {}  # job_id -> {payload, status, enqueued_at, lease_until, attempts, result}

# Long-poll coordination
queue_cv = threading.Condition()

# Worker/circuit breaker state
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "change-me")
LAST_WORKER_POLL_AT = 0.0
CONSECUTIVE_JOB_FAILS = 0

# Circuit breaker states
BREAKER_STATE = (
    "CLOSED"  # CLOSED -> prefer local, OPEN -> route to fallback, HALF_OPEN -> probe
)

# Tunables (seconds and counts)
POLL_STALE_SEC = int(os.environ.get("POLL_STALE_SEC", "15"))
LEASE_SEC = int(os.environ.get("LEASE_SEC", "30"))
QUEUE_AGE_SLA_SEC = int(os.environ.get("QUEUE_AGE_SLA_SEC", "10"))
QUEUE_DEPTH_LIMIT = int(os.environ.get("QUEUE_DEPTH_LIMIT", "100"))
FAIL_OPEN_THRESHOLD = int(os.environ.get("FAIL_OPEN_THRESHOLD", "3"))
HALF_OPEN_COOLDOWN_SEC = int(os.environ.get("HALF_OPEN_COOLDOWN_SEC", "30"))

BREAKER_LAST_OPEN_AT = 0.0

_reaper_started = False


def breaker_should_open(now):
    """Decide if breaker should open based on worker staleness and queue pressure."""
    oldest_age = 0.0
    if JOB_QUEUE:
        first_job = JOBS[JOB_QUEUE[0]]
        oldest_age = max(0.0, now - first_job["enqueued_at"]) if first_job else 0.0
    if (now - LAST_WORKER_POLL_AT) > POLL_STALE_SEC:
        return True
    if oldest_age > QUEUE_AGE_SLA_SEC:
        return True
    if len(JOB_QUEUE) > QUEUE_DEPTH_LIMIT:
        return True
    if CONSECUTIVE_JOB_FAILS >= FAIL_OPEN_THRESHOLD:
        return True
    return False


def breaker_maybe_transition(now):
    """Update breaker state based on signals and cooldown."""
    global BREAKER_STATE, BREAKER_LAST_OPEN_AT
    if BREAKER_STATE == "CLOSED":
        if breaker_should_open(now):
            BREAKER_STATE = "OPEN"
            BREAKER_LAST_OPEN_AT = now
    elif BREAKER_STATE == "OPEN":
        # After cooldown, try probing (HALF_OPEN)
        if (now - BREAKER_LAST_OPEN_AT) > HALF_OPEN_COOLDOWN_SEC:
            BREAKER_STATE = "HALF_OPEN"
    elif BREAKER_STATE == "HALF_OPEN":
        # Stay HALF_OPEN until a probe result is reported (success -> CLOSED, failure -> OPEN)
        pass


def breaker_probe_result(success):
    """Called when a probe job completes to move from HALF_OPEN to CLOSED/OPEN."""
    global BREAKER_STATE, BREAKER_LAST_OPEN_AT
    if BREAKER_STATE != "HALF_OPEN":
        return
    if success:
        BREAKER_STATE = "CLOSED"
    else:
        BREAKER_STATE = "OPEN"
        BREAKER_LAST_OPEN_AT = time.time()


def reaper_loop():
    """Requeue jobs whose lease has expired to avoid being stuck forever."""
    while True:
        try:
            now = time.time()
            expired = []
            for job_id, job in list(JOBS.items()):
                if job.get("status") == "leased" and job.get("lease_until", 0) < now:
                    expired.append(job_id)
            if expired:
                with queue_cv:
                    for job_id in expired:
                        job = JOBS.get(job_id)
                        if not job:
                            continue
                        job["status"] = "queued"
                        job["lease_until"] = 0
                        job["attempts"] = job.get("attempts", 0) + 1
                        # Only requeue if not already present
                        if job_id not in JOB_QUEUE:
                            JOB_QUEUE.appendleft(job_id)
                    queue_cv.notify_all()
        except RuntimeError:
            # dict mutated while iterating (threaded handlers) — retry next tick
            pass
        time.sleep(1.0)


def start_reaper():
    """Start the lease-requeue reaper thread once (from the app lifespan)."""
    global _reaper_started
    if _reaper_started:
        return
    _reaper_started = True
    reaper_thread = threading.Thread(target=reaper_loop)
    reaper_thread.daemon = True
    reaper_thread.start()
