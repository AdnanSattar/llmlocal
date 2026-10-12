#!/usr/bin/env python3
"""Focused tests for the benchmark's bearer-auth handling.

No network, no real secrets: the key values here are obvious dummy strings.
Run: python benchmarks/test_benchmark_auth.py -v
"""

import importlib.util
import pathlib
import unittest

_MOD_PATH = pathlib.Path(__file__).with_name("benchmark_models.py")
_spec = importlib.util.spec_from_file_location("benchmark_models", _MOD_PATH)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)


class HeaderConstructionTests(unittest.TestCase):
    def test_header_when_key_present(self):
        self.assertEqual(bm.auth_header("dummy-key"),
                         {"Authorization": "Bearer dummy-key"})

    def test_no_header_when_key_empty(self):
        self.assertEqual(bm.auth_header(""), {})

    def test_key_from_env_stripped(self):
        self.assertEqual(bm.api_key_from_env({"API_KEY": "  k  "}), "k")

    def test_key_from_env_missing(self):
        self.assertIsNone(bm.api_key_from_env({}))
        self.assertIsNone(bm.api_key_from_env({"API_KEY": ""}))


class MissingKeyPolicyTests(unittest.TestCase):
    def test_loopback_allowed_without_key(self):
        self.assertIsNone(bm.external_auth_error("http://localhost:8000", None))
        self.assertIsNone(bm.external_auth_error("http://127.0.0.1:2108", None))

    def test_remote_without_key_is_an_error(self):
        self.assertIsNotNone(
            bm.external_auth_error("http://34.105.69.230/llmlocal", None))

    def test_remote_with_key_is_ok(self):
        self.assertIsNone(
            bm.external_auth_error("http://34.105.69.230/llmlocal", "dummy-key"))

    def test_remote_without_key_allowed_when_opted_in(self):
        self.assertIsNone(
            bm.external_auth_error("http://34.105.69.230/llmlocal", None,
                                   allow_no_auth=True))


class StreamRequestAuthTests(unittest.TestCase):
    """Assert the Authorization header actually reaches the request."""

    class _FakeResp:
        headers = {"Content-Type": "application/json"}

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _captured_headers(self, api_key):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            return self._FakeResp()

        orig = bm.urllib.request.urlopen
        bm.urllib.request.urlopen = fake_urlopen
        try:
            bm.stream_request("http://example.invalid/v1/chat/completions",
                              {"stream": True}, 5, api_key=api_key)
        finally:
            bm.urllib.request.urlopen = orig
        return captured["headers"]

    def test_bearer_sent_when_key_given(self):
        self.assertEqual(self._captured_headers("dummy-key").get("Authorization"),
                         "Bearer dummy-key")

    def test_no_bearer_without_key(self):
        self.assertNotIn("Authorization", self._captured_headers(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
