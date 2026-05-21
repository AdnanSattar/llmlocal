import http.server
import json
import random
import socketserver
import threading
import time
import uuid

from transformers import pipeline

# Global variables
generator = None
model_loading = True
model_loaded = False

# -----------------------------
# Lightweight Job Queue + Circuit Breaker
# -----------------------------
# This section adds a minimal in-memory job queue, worker long-poll endpoint,
# webhook callback endpoint, and a simple circuit breaker to allow a reverse-webhook
# processing model. It is designed to coexist with the existing `/v1/*` endpoints
# and not change their behavior.
#
# Flow (reverse-webhook):
# 1) Clients call POST /jobs with a payload shaped like the /v1/completions body.
# 2) A LAN worker process long-polls POST /jobs/next?worker_token=... to fetch work.
# 3) The LAN worker calls local model (e.g., http://localhost:8000/v1/completions).
# 4) The LAN worker POSTs results to /webhooks/completions with {job_id, result}.
# 5) Clients can query GET /jobs/{job_id} for status/result.
#
# Circuit breaker logic marks local processing as unavailable when:
# - No worker polls for a configured window; or
# - Queue age/depth exceeds thresholds; or
# - Consecutive failures exceed a threshold.
#
# For production, persist jobs to a durable store (e.g., Redis/DB) and protect worker
# endpoints with mTLS or strong auth. This in-memory version is for simplicity.

import os
from collections import deque

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
    """Update breaker state based on signals and cooldowns."""
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
        time.sleep(1.0)

# Start the reaper in background
reaper_thread = threading.Thread(target=reaper_loop)
reaper_thread.daemon = True
reaper_thread.start()

def load_model():
    global generator, model_loading, model_loaded
    try:
        print("Loading FLAN-T5-SMALL (instruction-tuned) model...")
        generator = pipeline("text2text-generation", model="google/flan-t5-small")
        print("Model loaded successfully!")
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loading = False
    finally:
        model_loading = False

# Start model loading in background
model_thread = threading.Thread(target=load_model)
model_thread.daemon = True
model_thread.start()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            if model_loaded:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(bytes("OK", "utf-8"))
            else:
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                status = "Loading model..." if model_loading else "Model failed to load"
                self.wfile.write(bytes(status, "utf-8"))
        elif self.path.startswith("/queue/metrics"):
            # Lightweight metrics for queue and breaker state
            now = time.time()
            breaker_maybe_transition(now)
            oldest_age = 0.0
            if JOB_QUEUE:
                first_job = JOBS[JOB_QUEUE[0]]
                oldest_age = (
                    max(0.0, now - first_job["enqueued_at"]) if first_job else 0.0
                )
            metrics = {
                "queue_depth": len(JOB_QUEUE),
                "queue_oldest_age_sec": oldest_age,
                "last_worker_poll_at": LAST_WORKER_POLL_AT,
                "now": now,
                "breaker_state": BREAKER_STATE,
                "consecutive_job_fails": CONSECUTIVE_JOB_FAILS,
            }
            body = json.dumps(metrics)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(body, "utf-8"))
        elif self.path.startswith("/jobs/") and len(self.path.split("/")) == 3:
            # GET /jobs/{job_id}
            _, _, job_id = self.path.split("/")
            job = JOBS.get(job_id)
            if not job:
                self.send_response(404)
                self.end_headers()
                return
            # do not include full payload for brevity
            job_view = {
                "job_id": job_id,
                "status": job.get("status"),
                "enqueued_at": job.get("enqueued_at"),
                "lease_until": job.get("lease_until", 0),
                "attempts": job.get("attempts", 0),
                "result": job.get("result"),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(job_view), "utf-8"))
        elif self.path == "/v1/models":
            if model_loaded:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    bytes(
                        json.dumps({"data": [{"id": "google/flan-t5-small"}]}), "utf-8"
                    )
                )
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status = "loading" if model_loading else "error"
                self.wfile.write(
                    bytes(json.dumps({"error": f"Model is {status}"}), "utf-8")
                )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global LAST_WORKER_POLL_AT
        global CONSECUTIVE_JOB_FAILS
        if self.path == "/v1/completions":
            if not model_loaded:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status = "loading" if model_loading else "error"
                self.wfile.write(
                    bytes(json.dumps({"error": f"Model is {status}"}), "utf-8")
                )
                return

            content_length = int(self.headers["Content-Length"])
            post_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            print(f"Received request: {post_data}")

            # Extract parameters with defaults
            prompt = post_data.get("prompt", "").strip()
            system = post_data.get("system", "").strip()
            max_tokens = post_data.get("max_tokens", 50)
            temperature = post_data.get("temperature", 0.7)
            top_p = post_data.get("top_p", 1.0)
            stop = post_data.get("stop", None)
            min_tokens = post_data.get("min_tokens", 12)

            # Safety caps for CPU-friendly inference
            try:
                max_tokens = int(max_tokens)
            except Exception:
                max_tokens = 50
            max_tokens = max(1, min(max_tokens, 256))  # cap large requests

            try:
                min_tokens = int(min_tokens)
            except Exception:
                min_tokens = 12
            min_tokens = max(0, min(min_tokens, max_tokens))

            # Compose instruction-following prompt for FLAN-T5 (reduces echo)
            if system:
                effective_prompt = (
                    f"Instruction: {system}\n" f"Question: {prompt}\n" f"Answer:"
                ).strip()
            else:
                effective_prompt = (f"Question: {prompt}\n" f"Answer:").strip()

            # Generate text with parameters (text2text pipeline returns only completion)
            # Only honor user-provided stop sequences to avoid over-truncation
            stop_sequences = (
                (stop if isinstance(stop, list) else [stop]) if stop else []
            )
            stop_sequences = [s for s in stop_sequences if s]

            generation_kwargs = {
                "max_new_tokens": int(max_tokens),
                "min_new_tokens": int(min_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "do_sample": bool(temperature and temperature > 0),
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 3,
            }

            if stop:
                generation_kwargs["stop_strings"] = (
                    stop if isinstance(stop, list) else [stop]
                )

            result = generator(effective_prompt, **generation_kwargs)[0][
                "generated_text"
            ]

            generated_text = result.strip()

            # Apply stop sequences post-generation for T5
            for stop_seq in stop_sequences:
                if stop_seq and stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0].strip()

            # Handle stop sequences
            if stop:
                for stop_seq in stop if isinstance(stop, list) else [stop]:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]

            response = {
                "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "google/flan-t5-small",
                "choices": [
                    {
                        "text": generated_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(effective_prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(effective_prompt.split())
                    + len(generated_text.split()),
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(response), "utf-8"))

        elif self.path == "/v1/chat/completions":
            if not model_loaded:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status = "loading" if model_loading else "error"
                self.wfile.write(
                    bytes(json.dumps({"error": f"Model is {status}"}), "utf-8")
                )
                return

            content_length = int(self.headers["Content-Length"])
            post_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            print(f"Received chat request: {post_data}")

            # Extract chat parameters
            messages = post_data.get("messages", [])
            max_tokens = post_data.get("max_tokens", 50)
            temperature = post_data.get("temperature", 0.7)
            top_p = post_data.get("top_p", 1.0)

            # Convert messages to prompt
            prompt_parts = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            prompt = "\n".join(prompt_parts) + "\nAssistant:"

            # Generate response
            generation_kwargs = {
                "max_length": len(prompt.split()) + max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "pad_token_id": generator.tokenizer.eos_token_id,
            }

            result = generator(prompt, **generation_kwargs)[0]["generated_text"]
            generated_text = result[len(prompt) :].strip()

            # Clean up the response
            if "\n" in generated_text:
                generated_text = generated_text.split("\n")[0]

            response = {
                "id": f"chatcmpl-{random.randint(100000000000000000000, 999999999999999999999)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gpt2",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": generated_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(response), "utf-8"))

        elif self.path == "/jobs":
            # Enqueue a job with provided payload (opaque to the queue)
            content_length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = (
                    json.loads(self.rfile.read(content_length).decode("utf-8"))
                    if content_length > 0
                    else {}
                )
            except Exception:
                body = {}
            job_id = str(uuid.uuid4())
            job = {
                "job_id": job_id,
                "payload": body,
                "status": "queued",
                "enqueued_at": time.time(),
                "lease_until": 0,
                "attempts": 0,
                "result": None,
            }
            JOBS[job_id] = job
            with queue_cv:
                JOB_QUEUE.append(job_id)
                queue_cv.notify_all()
            # Re-evaluate breaker on enqueue
            breaker_maybe_transition(time.time())
            resp = {"job_id": job_id, "status": "queued"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(resp), "utf-8"))

        elif self.path.startswith("/jobs/next"):
            # Worker long-poll to fetch next job
            # Very simple query parsing for worker_token
            token = None
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for part in q.split("&"):
                    if part.startswith("worker_token="):
                        token = part.split("=", 1)[1]
                        break
            if token != WORKER_TOKEN:
                self.send_response(403)
                self.end_headers()
                return
            LAST_WORKER_POLL_AT = time.time()
            breaker_maybe_transition(LAST_WORKER_POLL_AT)
            # Long-poll up to 25s
            deadline = time.time() + 25
            job_id = None
            with queue_cv:
                while not JOB_QUEUE and time.time() < deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    queue_cv.wait(timeout=min(remaining, 1.0))
                if JOB_QUEUE:
                    job_id = JOB_QUEUE.popleft()
                    job = JOBS.get(job_id)
                    if job:
                        job["status"] = "leased"
                        job["lease_until"] = time.time() + LEASE_SEC
            if not job_id:
                # empty response to indicate no work
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(bytes(json.dumps({"job": None}), "utf-8"))
                return
            job = JOBS.get(job_id, {})
            # Return minimal job view to worker
            out = {
                "job_id": job_id,
                "payload": job.get("payload", {}),
                "probe": (BREAKER_STATE == "HALF_OPEN"),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(out), "utf-8"))

        elif self.path == "/webhooks/completions":
            # Worker posts results here: {job_id, result, success: bool}
            content_length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = (
                    json.loads(self.rfile.read(content_length).decode("utf-8"))
                    if content_length > 0
                    else {}
                )
            except Exception:
                body = {}
            job_id = body.get("job_id")
            success = bool(body.get("success", True))
            job = JOBS.get(job_id)
            if not job:
                self.send_response(404)
                self.end_headers()
                return
            job["result"] = body.get("result")
            job["status"] = "completed" if success else "failed"
            job["lease_until"] = 0
            if success:
                CONSECUTIVE_JOB_FAILS = 0
            else:
                CONSECUTIVE_JOB_FAILS += 1
            # If this was a probe, inform breaker result
            if body.get("probe") is True:
                breaker_probe_result(success)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"ok": True}), "utf-8"))

        elif self.path.startswith("/workers/heartbeat"):
            # Optional: worker sends periodic heartbeat
            token = None
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for part in q.split("&"):
                    if part.startswith("worker_token="):
                        token = part.split("=", 1)[1]
                        break
            if token != WORKER_TOKEN:
                self.send_response(403)
                self.end_headers()
                return
            LAST_WORKER_POLL_AT = time.time()
            breaker_maybe_transition(LAST_WORKER_POLL_AT)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                bytes(json.dumps({"ok": True, "breaker_state": BREAKER_STATE}), "utf-8")
            )

        else:
            self.send_response(404)
            self.end_headers()


print("Starting HTTP server on port 8000...")
socketserver.TCPServer(("", 8000), Handler).serve_forever()
