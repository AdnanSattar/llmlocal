#!/usr/bin/env python3
"""Unit tests for the deterministic output-guardrail layer.

No network, no server, no model. The protected value used here (BANANA-77) is a
**test fixture**, the same literal the benchmark's hierarchy prompt uses; it is
never hard-coded in the guardrail module.

Run: python tests/test_guardrails.py -v
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import guardrails as g  # noqa: E402


ACTIVE = g.GuardrailPolicy(enabled=True, protected_values=("BANANA-77",),
                           max_words=30)
INACTIVE = g.GuardrailPolicy(enabled=True)  # enabled but no rules -> no-op


class WordCountTests(unittest.TestCase):
    def test_plain_prose(self):
        self.assertEqual(g.count_words("one two three"), 3)

    def test_whitespace_and_newlines(self):
        self.assertEqual(g.count_words("  one\ntwo\tthree  "), 3)

    def test_empty(self):
        self.assertEqual(g.count_words(""), 0)
        self.assertEqual(g.count_words(None), 0)

    def test_markdown_punctuation_only_tokens_not_counted(self):
        self.assertEqual(g.count_words("**bold** --- # -"), 1)

    def test_hyphenated_token_is_one_word(self):
        self.assertEqual(g.count_words("BANANA-77 leaked"), 2)

    def test_unicode_alnum_counts(self):
        self.assertEqual(g.count_words("café naïve"), 2)


class ProtectedValueTests(unittest.TestCase):
    def test_case_insensitive_match(self):
        self.assertEqual(g.find_protected("the code is banana-77", ["BANANA-77"]),
                         ["BANANA-77"])

    def test_no_match(self):
        self.assertEqual(g.find_protected("nothing here", ["BANANA-77"]), [])

    def test_regex_metacharacters_are_literal(self):
        self.assertEqual(g.find_protected("token a.b*c end", ["a.b*c"]), ["a.b*c"])
        self.assertEqual(g.find_protected("token axbxc end", ["a.b*c"]), [])

    def test_empty_inputs(self):
        self.assertEqual(g.find_protected("", ["x"]), [])
        self.assertEqual(g.find_protected("x", []), [])


class PolicyLoadingTests(unittest.TestCase):
    def test_default_is_disabled(self):
        policy = g.load_policy({})
        self.assertFalse(policy.enabled)
        self.assertFalse(policy.active)

    def test_enabled_without_rules_is_inactive(self):
        self.assertFalse(g.load_policy({"LLM_GUARDRAILS": "1"}).active)

    def test_rules_without_switch_are_inactive(self):
        policy = g.load_policy({"LLM_GUARDRAIL_PROTECTED_VALUES": "x"})
        self.assertFalse(policy.active)

    def test_full_configuration(self):
        policy = g.load_policy({
            "LLM_GUARDRAILS": "true",
            "LLM_GUARDRAIL_PROTECTED_VALUES": " alpha , beta ,",
            "LLM_GUARDRAIL_MAX_WORDS": "12",
        })
        self.assertTrue(policy.active)
        self.assertEqual(policy.protected_values, ("alpha", "beta"))
        self.assertEqual(policy.max_words, 12)

    def test_invalid_max_words_ignored(self):
        policy = g.load_policy({"LLM_GUARDRAILS": "1",
                                "LLM_GUARDRAIL_MAX_WORDS": "0"})
        self.assertIsNone(policy.max_words)

    def test_reasoning_screening_default_on_and_can_be_disabled(self):
        on = g.load_policy({"LLM_GUARDRAILS": "1"})
        self.assertTrue(on.guard_reasoning)
        off = g.load_policy({"LLM_GUARDRAILS": "1",
                             "LLM_GUARDRAIL_GUARD_REASONING": "0"})
        self.assertFalse(off.guard_reasoning)


class ValidateTests(unittest.TestCase):
    def test_inactive_policy_reports_nothing(self):
        self.assertEqual(g.validate("BANANA-77 " * 100, INACTIVE), [])

    def test_word_limit_violation(self):
        violations = g.validate(" ".join(["w"] * 31), ACTIVE)
        self.assertEqual([v.rule for v in violations], ["max_words"])

    def test_word_limit_boundary_is_inclusive(self):
        self.assertEqual(g.validate(" ".join(["w"] * 30), ACTIVE), [])

    def test_protected_leak_violation(self):
        violations = g.validate("the codeword is BANANA-77", ACTIVE)
        self.assertEqual([v.rule for v in violations], ["protected_value"])

    def test_reasoning_skips_word_limit(self):
        long_reasoning = " ".join(["w"] * 500)
        self.assertEqual(g.validate(long_reasoning, ACTIVE, is_reasoning=True), [])

    def test_reasoning_still_screens_protected_values(self):
        violations = g.validate("I know BANANA-77", ACTIVE, is_reasoning=True)
        self.assertEqual([v.rule for v in violations], ["protected_value"])


def _chat_response(content, reasoning=""):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
        "model": "test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


class ApplyGuardrailsTests(unittest.TestCase):
    def test_inactive_policy_returns_response_unchanged(self):
        response = _chat_response("BANANA-77 " * 100)
        out, violations = g.apply_guardrails(response, INACTIVE, chat=True)
        self.assertEqual(violations, [])
        self.assertNotIn("guardrails", out)
        self.assertIn("BANANA-77", out["choices"][0]["message"]["content"])

    def test_clean_response_passes_and_is_unchanged(self):
        response = _chat_response("A short clean answer.")
        out, violations = g.apply_guardrails(response, ACTIVE, chat=True)
        self.assertEqual(violations, [])
        self.assertEqual(out["choices"][0]["message"]["content"],
                         "A short clean answer.")
        self.assertEqual(out["guardrails"]["status"], "passed")

    def test_leak_is_replaced_and_not_returned(self):
        response = _chat_response("BANANA-77 is the codeword.")
        out, violations = g.apply_guardrails(response, ACTIVE, chat=True)
        choice = out["choices"][0]
        self.assertEqual(choice["message"]["content"], ACTIVE.fallback_message)
        self.assertEqual(choice["finish_reason"], "content_filter")
        self.assertNotIn("BANANA-77", out["choices"][0]["message"]["content"])
        self.assertEqual(out["guardrails"]["status"], "rejected")

    def test_reasoning_leak_is_dropped(self):
        response = _chat_response("A clean answer.",
                                  reasoning="the codeword is BANANA-77")
        out, _ = g.apply_guardrails(response, ACTIVE, chat=True)
        message = out["choices"][0]["message"]
        self.assertNotIn("BANANA-77", message.get("reasoning_content", ""))
        self.assertNotIn("reasoning_content", message)
        self.assertEqual(out["choices"][0]["finish_reason"], "content_filter")

    def test_metadata_never_contains_protected_value(self):
        response = _chat_response("BANANA-77")
        out, _ = g.apply_guardrails(response, ACTIVE, chat=True)
        blob = repr(out["guardrails"])
        self.assertNotIn("BANANA-77", blob)

    def test_word_limit_violation_is_rejected(self):
        response = _chat_response("word " * 100)
        out, violations = g.apply_guardrails(response, ACTIVE, chat=True)
        self.assertEqual([v.rule for v in violations], ["max_words"])
        self.assertEqual(out["choices"][0]["finish_reason"], "content_filter")

    def test_completions_shape(self):
        response = {
            "id": "cmpl-test", "object": "text_completion", "created": 1,
            "model": "test",
            "choices": [{"index": 0, "text": "leak BANANA-77", "finish_reason": "stop"}],
        }
        out, violations = g.apply_guardrails(response, ACTIVE, chat=False)
        self.assertEqual(len(violations), 1)
        self.assertEqual(out["choices"][0]["text"], ACTIVE.fallback_message)
        self.assertNotIn("BANANA-77", out["choices"][0]["text"])


class AggregateStreamTests(unittest.TestCase):
    def _chunks(self):
        return iter([
            {"id": "chatcmpl-x", "object": "chat.completion", "created": 7,
             "model": "m", "choices": [{"index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None}]},
            {"choices": [{"index": 0,
                          "delta": {"reasoning_content": "think "},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "ans "},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": "wer"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {},
                          "finish_reason": "stop"}],
             "usage": {"completion_tokens": 4}},
        ])

    def test_separates_reasoning_and_content(self):
        agg = g.aggregate_stream(self._chunks(), chat=True)
        self.assertEqual(agg["reasoning"], "think ")
        self.assertEqual(agg["content"], "ans wer")
        self.assertEqual(agg["finish_reason"], "stop")
        self.assertEqual(agg["usage"], {"completion_tokens": 4})
        self.assertEqual(agg["id"], "chatcmpl-x")


class ReplayChunksTests(unittest.TestCase):
    def _agg(self, content, reasoning="", finish="stop"):
        return {"id": "x", "object": "chat.completion", "created": 1,
                "model": "m", "content": content, "reasoning": reasoning,
                "finish_reason": finish, "usage": {"completion_tokens": 3}}

    def test_clean_replay_keeps_reasoning_separated(self):
        chunks = list(g.replay_chunks(self._agg("answer", "reasoning"),
                                      ACTIVE, chat=True))
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertIn({"role": "assistant"}, deltas)
        self.assertIn({"reasoning_content": "reasoning"}, deltas)
        self.assertIn({"content": "answer"}, deltas)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["guardrails"]["status"], "passed")

    def test_rejected_replay_never_emits_secret(self):
        agg = self._agg("BANANA-77 leaked", "reason BANANA-77")
        chunks = list(g.replay_chunks(agg, ACTIVE, chat=True))
        blob = "".join(str(c) for c in chunks)
        self.assertNotIn("BANANA-77", blob)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"],
                         "content_filter")
        contents = [c["choices"][0]["delta"].get("content") for c in chunks
                    if "delta" in c["choices"][0]]
        self.assertIn(ACTIVE.fallback_message, contents)
        self.assertNotIn("reasoning_content", str(chunks[1]))

    def test_budget_exhaustion_finish_reason_preserved(self):
        chunks = list(g.replay_chunks(self._agg("short", finish="length"),
                                      ACTIVE, chat=True))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")


class CollectStreamTests(unittest.TestCase):
    class _Backend:
        def __init__(self, chunks):
            self._chunks = chunks

        def stream(self, payload, chat):
            return iter(self._chunks)

    def test_buffers_backend_stream(self):
        backend = self._Backend([
            {"choices": [{"index": 0, "delta": {"content": "hi"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ])
        agg = g.collect_stream(backend, {"stream": True}, True)
        self.assertEqual(agg["content"], "hi")
        self.assertEqual(agg["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
