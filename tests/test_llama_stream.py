#!/usr/bin/env python3
"""Regression tests for the llama.cpp SSE parsing and passthrough behaviour.

These guard the metrics that already pass in the clean benchmark runs:
reasoning separation, zero think-tag leaks, error passthrough, and
token-budget (finish_reason="length") reporting. No server is started; a fake
HTTP response object feeds the parser.

Run: python tests/test_llama_stream.py -v
"""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.llama_cpp import LlamaCppBackend, _iter_sse  # noqa: E402


class FakeSSE:
    """Minimal stand-in for an http.client.HTTPResponse in SSE mode."""

    def __init__(self, lines, ctype="text/event-stream"):
        self._lines = list(lines)
        self.headers = {"Content-Type": ctype}
        self.closed = False

    def readline(self):
        return self._lines.pop(0) if self._lines else b""

    def read(self):
        return b"".join(self._lines)

    def close(self):
        self.closed = True


def _data(obj):
    return b"data: " + json.dumps(obj).encode("utf-8") + b"\n"


class IterSseTests(unittest.TestCase):
    def test_parses_data_lines_until_done(self):
        resp = FakeSSE([
            _data({"a": 1}),
            b": keep-alive comment\n",
            b"event: ping\n",
            _data({"b": 2}),
            b"data: [DONE]\n",
            _data({"never": True}),
        ])
        self.assertEqual([o for o in _iter_sse(resp)], [{"a": 1}, {"b": 2}])

    def test_delimiter_split_across_segments_is_reassembled(self):
        # Simulate the transport delivering the stream one byte at a time
        # (so the "data:" prefix and the JSON body are each split across
        # reads) and prove a readline()-style reader still yields whole lines.
        stream = _data({"n": 1}) + _data({"n": 2})

        class ByteStream:
            def __init__(self, chunks):
                self._chunks = list(chunks)
                self._buf = b""

            def readline(self):
                while b"\n" not in self._buf:
                    if not self._chunks:
                        out, self._buf = self._buf, b""
                        return out
                    self._buf += self._chunks.pop(0)
                idx = self._buf.index(b"\n") + 1
                line, self._buf = self._buf[:idx], self._buf[idx:]
                return line

        one_byte_chunks = [stream[i:i + 1] for i in range(len(stream))]
        self.assertEqual([o for o in _iter_sse(ByteStream(one_byte_chunks))],
                         [{"n": 1}, {"n": 2}])

    def test_final_line_without_newline_is_not_dropped(self):
        # Regression: the old reader discarded an EOF-terminated final line.
        resp = FakeSSE([b'data: {"last": 1}\n', b'data: {"end": 2}'])
        self.assertEqual([o for o in _iter_sse(resp)], [{"last": 1}, {"end": 2}])

    def test_crlf_lines_supported(self):
        resp = FakeSSE([b'data: {"x": 1}\r\n', b"data: [DONE]\r\n"])
        self.assertEqual([o for o in _iter_sse(resp)], [{"x": 1}])

    def test_malformed_json_is_skipped_not_fatal(self):
        resp = FakeSSE([
            b"data: {not json}\n",
            _data({"ok": True}),
            b"data: [DONE]\n",
        ])
        self.assertEqual([o for o in _iter_sse(resp)], [{"ok": True}])

    def test_truncated_stream_without_done_ends(self):
        resp = FakeSSE([_data({"partial": True})])
        self.assertEqual([o for o in _iter_sse(resp)], [{"partial": True}])


class IterStreamTests(unittest.TestCase):
    def test_reasoning_content_passes_through_unchanged(self):
        resp = FakeSSE([
            _data({"id": "chatcmpl-1", "object": "chat.completion",
                   "created": 5, "model": "upstream",
                   "choices": [{"index": 0,
                                "delta": {"reasoning_content": "think "},
                                "finish_reason": None}]}),
            _data({"choices": [{"index": 0, "delta": {"content": "hi"},
                                "finish_reason": None}]}),
            _data({"choices": [{"index": 0, "delta": {},
                                "finish_reason": "stop"}]}),
            b"data: [DONE]\n",
        ])
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend.model_id = "local-model"
        chunks = list(backend._iter_stream(resp))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["reasoning_content"],
                         "think ")
        self.assertEqual(chunks[1]["choices"][0]["delta"]["content"], "hi")
        self.assertTrue(resp.closed)
        for chunk in chunks:
            self.assertEqual(chunk["model"], "local-model")

    def test_no_think_tags_leak_into_content(self):
        resp = FakeSSE([
            _data({"choices": [{"index": 0,
                                "delta": {"reasoning_content": "BANANA-77 is secret"},
                                "finish_reason": None}]}),
            _data({"choices": [{"index": 0, "delta": {"content": "safe"},
                                "finish_reason": "stop"}]}),
            b"data: [DONE]\n",
        ])
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend.model_id = "m"
        content = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in backend._iter_stream(resp))
        self.assertEqual(content, "safe")
        self.assertNotIn("<think>", content)

    def test_budget_exhaustion_finish_reason_preserved(self):
        resp = FakeSSE([
            _data({"choices": [{"index": 0, "delta": {"content": "cut"},
                                "finish_reason": None}]}),
            _data({"choices": [{"index": 0, "delta": {},
                                "finish_reason": "length"}],
                   "usage": {"completion_tokens": 768}}),
            b"data: [DONE]\n",
        ])
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend.model_id = "m"
        chunks = list(backend._iter_stream(resp))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 768)

    def test_non_sse_fallback_yields_single_json_object(self):
        body = json.dumps({"choices": [{"index": 0,
                                        "message": {"role": "assistant",
                                                    "content": "hi"},
                                        "finish_reason": "stop"}]}).encode()
        resp = FakeSSE([body], ctype="application/json")
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend.model_id = "m"
        chunks = list(backend._iter_stream(resp))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["model"], "m")


if __name__ == "__main__":
    unittest.main(verbosity=2)
