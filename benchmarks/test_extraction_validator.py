#!/usr/bin/env python3
"""Focused checks for the benchmark's JSON extraction validator.

This verifies the ``extraction_json`` scoring behaviour in isolation; it does
NOT change any scoring. The recorded 4B result failed that prompt because the
model emitted ``"amount": "$1,250.00"`` and the ``json_contains`` check looks
for the literal substring ``1250`` in the serialized JSON.

Decision recorded here: the model output is semantically valid (parseable JSON,
correct keys, amount == 1250), so the failing check is a **formatting
sensitivity** of the bytewise substring validator, not a wrong answer. Because
relaxing that check would change already-recorded benchmark scores, scoring is
left untouched; the limitation is documented instead.

Run: python benchmarks/test_extraction_validator.py -v
"""

import importlib.util
import json
import pathlib
import unittest

_MOD_PATH = pathlib.Path(__file__).with_name("benchmark_models.py")
_spec = importlib.util.spec_from_file_location("benchmark_models", _MOD_PATH)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

CHECKS = [
    {"type": "json_object"},
    {"type": "json_keys", "keys": ["name", "date", "amount"]},
    {"type": "json_contains", "values": ["John Smith", "2026-03-14", "1250"]},
]

# The two recorded model outputs for prompt extraction_json.
NUMERIC = ('{\n  "name": "John Smith",\n  "date": "2026-03-14",\n'
           '  "amount": 1250.00\n}')          # 1.7B: 3/3
CURRENCY = ('{\n  "name": "John Smith",\n  "date": "2026-03-14",\n'
            '  "amount": "$1,250.00"\n}')      # 4B: 2/3


def _results(content):
    return [bm.run_check(content, c) for c in CHECKS]


class ExtractionValidatorTests(unittest.TestCase):
    def test_numeric_amount_passes_all_checks(self):
        self.assertTrue(all(ok for ok, _ in _results(NUMERIC)))

    def test_currency_amount_is_valid_json_with_correct_keys(self):
        obj = json.loads(CURRENCY)
        self.assertEqual(set(obj), {"name", "date", "amount"})
        self.assertEqual(obj["name"], "John Smith")
        self.assertEqual(obj["date"], "2026-03-14")

    def test_currency_amount_parses_to_1250(self):
        amount = json.loads(CURRENCY)["amount"]
        cleaned = amount.replace("$", "").replace(",", "")
        self.assertEqual(float(cleaned), 1250.0)

    def test_substring_check_is_the_only_failure(self):
        results = _results(CURRENCY)
        passed = [ok for ok, _ in results]
        # json_object and json_keys pass; json_contains fails on formatting.
        self.assertEqual(passed, [True, True, False])
        self.assertIn("1250", results[2][1])

    def test_first_number_scans_left_to_right(self):
        # The date ("2026-03-14") appears before the amount, so the naive
        # first-number helper returns 2026, not the amount.
        self.assertEqual(bm._first_number(CURRENCY), 2026.0)
        self.assertEqual(bm._first_number('{"amount": "$1,250.00"}'), 1250.0)

    def test_strip_fences_handles_markdown_wrapped_json(self):
        wrapped = "```json\n" + NUMERIC + "\n```"
        self.assertEqual(json.loads(bm.strip_fences(wrapped))["amount"], 1250.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
