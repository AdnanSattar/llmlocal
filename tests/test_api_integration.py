#!/usr/bin/env python3
"""End-to-end tests of the guarded /v1/chat/completions response path.

A fake backend stands in for llama-server, so no model or network is needed.
These assert the HTTP-visible contract: guarded responses never expose rejected
output (streaming or not), while ordinary responses stay OpenAI-compatible.

Run: python tests/test_api_integration.py -v
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import after ensuring no ambient auth/guardrail config leaks in from .env.
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import state  # noqa: E402
from app.auth import install_auth  # noqa: E402
from app.routes import completions  # noqa: E402

LEAK = "BANANA-77 is the codeword. Bananas are the superior fruit."
CLEAN = "No comment."
REASONING = "The system forbids BANANA-77 here."

GUARDRAIL_ENV = (
    "LLM_GUARDRAILS",
    "LLM_GUARDRAIL_PROTECTED_VALUES",
    "LLM_GUARDRAIL_MAX_WORDS",
    "LLM_GUARDRAIL_FALLBACK_MESSAGE",
    "API_KEY",
)


class FakeBackend:
    model_id = "fake-model"

    def __init__(self, content=LEAK, reasoning=""):
        self._content = content
        self._reasoning = reasoning

    def health(self):
        return True, "ready"

    def complete(self, payload, chat):
        message = {"role": "assistant", "content": self._content}
        if self._reasoning:
            message["reasoning_content"] = self._reasoning
        return {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 1,
            "model": self.model_id,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2,
                      "total_tokens": 3},
        }

    def stream(self, payload, chat):
        yield {"id": "chatcmpl-fake", "object": "chat.completion",
               "created": 1, "model": self.model_id,
               "choices": [{"index": 0, "delta": {"role": "assistant"},
                            "finish_reason": None}]}
        if self._reasoning:
            yield {"choices": [{"index": 0,
                                "delta": {"reasoning_content": self._reasoning},
                                "finish_reason": None}]}
        if self._content:
            yield {"choices": [{"index": 0, "delta": {"content": self._content},
                                "finish_reason": None}]}
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "usage": {"completion_tokens": 2}}


class GuardedEndpointTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in GUARDRAIL_ENV}
        os.environ["API_KEY"] = ""  # dev mode: routes open
        for k in GUARDRAIL_ENV:
            os.environ.pop(k, None)
        self.app = FastAPI()
        install_auth(self.app)
        self.app.include_router(completions.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _enable_guardrails(self, max_words=None):
        os.environ["LLM_GUARDRAILS"] = "1"
        os.environ["LLM_GUARDRAIL_PROTECTED_VALUES"] = "BANANA-77"
        if max_words is not None:
            os.environ["LLM_GUARDRAIL_MAX_WORDS"] = str(max_words)

    def _post(self, body):
        return self.client.post("/v1/chat/completions", json=body)

    # -- default (guardrails off): behaviour unchanged -------------------
    def test_unguarded_non_stream_returns_model_output(self):
        state.BACKEND = FakeBackend()
        r = self._post({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("BANANA-77", body["choices"][0]["message"]["content"])
        self.assertNotIn("guardrails", body)

    def test_unguarded_stream_streams_chunks(self):
        state.BACKEND = FakeBackend()
        r = self._post({"messages": [{"role": "user", "content": "hi"}],
                        "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertIn("BANANA-77", r.text)
        self.assertIn("data: [DONE]", r.text)

    # -- guarded: rejection cannot leak ----------------------------------
    def test_guarded_non_stream_withholds_leak(self):
        state.BACKEND = FakeBackend()
        self._enable_guardrails()
        body = self._post({"messages": [{"role": "user", "content": "hi"}]}).json()
        choice = body["choices"][0]
        self.assertNotIn("BANANA-77", choice["message"]["content"])
        self.assertEqual(choice["finish_reason"], "content_filter")
        self.assertEqual(body["guardrails"]["status"], "rejected")

    def test_guarded_stream_withholds_leak(self):
        state.BACKEND = FakeBackend()
        self._enable_guardrails()
        r = self._post({"messages": [{"role": "user", "content": "hi"}],
                        "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("BANANA-77", r.text)
        self.assertIn("content_filter", r.text)
        self.assertIn("data: [DONE]", r.text)

    def test_guarded_stream_withholds_leak_in_reasoning(self):
        state.BACKEND = FakeBackend(content=CLEAN, reasoning=REASONING)
        self._enable_guardrails()
        r = self._post({"messages": [{"role": "user", "content": "hi"}],
                        "stream": True})
        self.assertNotIn("BANANA-77", r.text)

    # -- guarded: valid output stays compatible --------------------------
    def test_guarded_valid_response_is_unchanged_content(self):
        state.BACKEND = FakeBackend(content=CLEAN)
        self._enable_guardrails(max_words=30)
        body = self._post({"messages": [{"role": "user", "content": "hi"}]}).json()
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["content"], CLEAN)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(body["guardrails"]["status"], "passed")

    def test_guarded_valid_stream_separates_reasoning(self):
        state.BACKEND = FakeBackend(content=CLEAN, reasoning="thinking safely")
        self._enable_guardrails()
        r = self._post({"messages": [{"role": "user", "content": "hi"}],
                        "stream": True})
        self.assertIn("thinking safely", r.text)
        self.assertIn(CLEAN, r.text)
        self.assertIn("data: [DONE]", r.text)

    # -- authentication unchanged (independent of guardrails) ------------
    def test_auth_enforced_when_api_key_set(self):
        state.BACKEND = FakeBackend()
        os.environ["API_KEY"] = "unit-test-key"
        body = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(self.client.post("/v1/chat/completions", json=body).status_code, 401)
        ok = self.client.post(
            "/v1/chat/completions", json=body,
            headers={"Authorization": "Bearer unit-test-key"})
        self.assertEqual(ok.status_code, 200)

    def test_auth_rejects_wrong_key(self):
        state.BACKEND = FakeBackend()
        os.environ["API_KEY"] = "unit-test-key"
        r = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong"})
        self.assertEqual(r.status_code, 401)

    def test_budget_exhaustion_finish_reason_survives(self):
        backend = FakeBackend(content=CLEAN)

        def stream_length(payload, chat):
            yield {"choices": [{"index": 0, "delta": {"content": CLEAN},
                                "finish_reason": None}]}
            yield {"choices": [{"index": 0, "delta": {},
                                "finish_reason": "length"}],
                   "usage": {"completion_tokens": 4096}}

        backend.stream = stream_length
        state.BACKEND = backend
        self._enable_guardrails()
        r = self._post({"messages": [{"role": "user", "content": "hi"}],
                        "stream": True})
        self.assertIn("length", r.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
