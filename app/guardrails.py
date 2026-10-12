"""Deterministic output guardrails for generated responses.

The model cannot be trusted to honour system-level instructions (see the
instruction-hierarchy failure: a small model leaks a protected codeword when the
user says "ignore the system prompt"). Prompt wording cannot guarantee an
instruction hierarchy, so the reliable enforcement boundary is *after*
generation: validate the model output against an explicit, operator-configured
policy and withhold anything that violates it.

Design notes
------------
* The policy is **explicit configurability**, never a hard-coded literal. The
  benchmark codeword (``BANANA-77``) is a test fixture passed in through the
  policy, not something baked into this module.
* Default is **disabled**: with no ``LLM_GUARDRAILS=1`` the response path is
  byte-for-byte what it was before (no behaviour change, no latency change).
* Validation is a *detector*, not a prompt-injection solution. A literal,
  case-insensitive substring detector catches the protected values an operator
  declares; it does not (and does not claim to) defeat arbitrary rephrasing or
  encoding-based evasion. Documented limitation.
* On a violation the rejected text is **discarded**, never returned, in both
  the non-streaming and streaming paths. ``finish_reason`` becomes the
  OpenAI-standard ``"content_filter"`` and a small ``guardrails`` metadata block
  reports the *rule kind* only — never the protected value.

Streaming trade-off
-------------------
Guarded streaming requests are **buffered**: the full upstream output is
collected, validated, and only then replayed as SSE chunks. This is the only
way to guarantee that no protected content is emitted before validation
completes. The cost is that guarded requests lose token-by-token streaming
(TTFT equals total generation time). Requests with no active policy keep the
original pass-through streaming and are unaffected.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

__all__ = [
    "GuardrailPolicy",
    "Violation",
    "DEFAULT_FALLBACK_MESSAGE",
    "load_policy",
    "count_words",
    "find_protected",
    "validate",
    "apply_guardrails",
    "collect_stream",
    "aggregate_stream",
    "replay_chunks",
]

DEFAULT_FALLBACK_MESSAGE = (
    "Response withheld: the generated answer violated the configured output policy."
)

_TRUTHY = {"1", "true", "yes", "on"}

# A "word" is a run of non-whitespace characters that contains at least one
# alphanumeric character. This matches ``str.split()`` for ordinary prose while
# excluding punctuation-only Markdown tokens (``**``, ``---``, ``#``), so
# emphasis/bullet syntax does not inflate the count.
_TOKEN_RE = re.compile(r"\S+")


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class GuardrailPolicy:
    """An operator-configured output policy.

    ``active`` is true only when guardrails are enabled *and* at least one rule
    is configured, so enabling the switch with no rules is a harmless no-op.
    """

    enabled: bool = False
    protected_values: tuple = ()
    max_words: Optional[int] = None
    guard_reasoning: bool = True
    fallback_message: str = DEFAULT_FALLBACK_MESSAGE

    @property
    def active(self) -> bool:
        return bool(self.enabled) and (
            len(self.protected_values) > 0 or self.max_words is not None
        )


@dataclass
class Violation:
    """A single failed rule. ``detail`` never contains a protected value."""

    rule: str
    detail: str


def load_policy(env=None) -> GuardrailPolicy:
    """Build the policy from the environment (safe default: disabled).

    Keys (all optional):
      * ``LLM_GUARDRAILS`` — master switch; ``1``/``true``/``yes``/``on`` enables.
      * ``LLM_GUARDRAIL_PROTECTED_VALUES`` — comma-separated literals to withhold.
      * ``LLM_GUARDRAIL_MAX_WORDS`` — maximum words in the assistant answer.
      * ``LLM_GUARDRAIL_GUARD_REASONING`` — also screen ``reasoning_content``
        (default on; reasoning is streamed to clients under the deepseek format).
      * ``LLM_GUARDRAIL_FALLBACK_MESSAGE`` — safe replacement text on rejection.
    """
    env = os.environ if env is None else env
    enabled = _truthy(env.get("LLM_GUARDRAILS"))

    raw = env.get("LLM_GUARDRAIL_PROTECTED_VALUES") or ""
    values = tuple(v.strip() for v in raw.split(",") if v.strip())

    max_words: Optional[int] = None
    raw_words = env.get("LLM_GUARDRAIL_MAX_WORDS")
    if raw_words not in (None, ""):
        try:
            n = int(raw_words)
            if n > 0:
                max_words = n
        except (TypeError, ValueError):
            max_words = None

    guard_reasoning = True
    if env.get("LLM_GUARDRAIL_GUARD_REASONING") is not None:
        guard_reasoning = _truthy(env.get("LLM_GUARDRAIL_GUARD_REASONING"))

    fallback = (env.get("LLM_GUARDRAIL_FALLBACK_MESSAGE") or "").strip() \
        or DEFAULT_FALLBACK_MESSAGE

    return GuardrailPolicy(
        enabled=enabled,
        protected_values=values,
        max_words=max_words,
        guard_reasoning=guard_reasoning,
        fallback_message=fallback,
    )


def count_words(text: Optional[str]) -> int:
    """Count words, tolerating whitespace/unicode and Markdown decorations."""
    tokens = _TOKEN_RE.findall(text or "")
    return sum(1 for t in tokens if any(ch.isalnum() for ch in t))


def find_protected(text: Optional[str], values: Sequence[str]) -> list:
    """Case-insensitive literal matches of ``values`` inside ``text``.

    Returns the matching values for internal decisions; callers must never emit
    them. Plain substring matching (not regex), so values may contain any
    characters.
    """
    if not text or not values:
        return []
    low = text.lower()
    return [v for v in values if v and v.lower() in low]


def validate(content: Optional[str], policy: GuardrailPolicy,
             *, is_reasoning: bool = False) -> list:
    """Return the list of :class:`Violation` for ``content`` (empty == clean).

    The word limit applies to the final answer only; protected-value screening
    applies to the answer and (when enabled) to reasoning text too.
    """
    violations: list = []
    if not policy.active:
        return violations

    if not is_reasoning and policy.max_words is not None:
        n = count_words(content)
        if n > policy.max_words:
            violations.append(
                Violation("max_words",
                          f"answer has {n} words (limit {policy.max_words})"))

    if policy.protected_values:
        hits = find_protected(content, policy.protected_values)
        if hits:
            violations.append(
                Violation("protected_value",
                          f"answer contained {len(hits)} protected value(s)"))

    return violations


def _choices(response):
    try:
        return response["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None


def apply_guardrails(response: dict, policy: GuardrailPolicy, chat: bool):
    """Validate a non-streaming OpenAI response in place.

    Returns ``(response, violations)``. On a violation the answer is replaced by
    the policy's fallback message, any leaked reasoning is dropped,
    ``finish_reason`` becomes ``"content_filter"``, and only rule kinds are
    recorded. When the policy is inactive the response is returned untouched.
    """
    if not policy.active or not isinstance(response, dict):
        return response, []

    choice = _choices(response)
    if choice is None:
        return response, []

    if chat:
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
    else:
        message = None
        content = choice.get("text") or ""
        reasoning = ""

    violations = validate(content, policy)
    if policy.guard_reasoning and reasoning:
        violations = violations + validate(reasoning, policy, is_reasoning=True)

    if violations:
        if message is not None:
            message["content"] = policy.fallback_message
            message.pop("reasoning_content", None)
            choice["message"] = message
        else:
            choice["text"] = policy.fallback_message
        choice["finish_reason"] = "content_filter"
        response["guardrails"] = {
            "status": "rejected",
            "violations": [{"rule": v.rule, "detail": v.detail}
                           for v in violations],
        }
    else:
        response["guardrails"] = {"status": "passed"}

    return response, violations


def aggregate_stream(chunks: Iterable[dict], chat: bool) -> dict:
    """Consume an OpenAI SSE chunk iterator into a single response-shaped dict.

    Preserves reasoning/content separation, ``finish_reason``, ``usage``, and
    the response id/object/created/model metadata. Closes the source iterator.
    """
    agg = {
        "id": None,
        "object": "chat.completion" if chat else "text_completion",
        "created": None,
        "model": None,
        "content": "",
        "reasoning": "",
        "finish_reason": None,
        "usage": None,
        "error": None,
    }
    try:
        for obj in chunks:
            if not isinstance(obj, dict):
                continue
            for key in ("id", "object", "created", "model"):
                if agg.get(key) is None and obj.get(key) is not None:
                    agg[key] = obj[key]
            if obj.get("usage"):
                agg["usage"] = obj["usage"]
            if obj.get("error"):
                agg["error"] = obj["error"]
            for ch in obj.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                if ch.get("finish_reason"):
                    agg["finish_reason"] = ch["finish_reason"]
                delta = ch.get("delta")
                if isinstance(delta, dict):
                    agg["reasoning"] += delta.get("reasoning_content") or ""
                    agg["content"] += delta.get("content") or ""
                elif isinstance(ch.get("message"), dict):
                    msg = ch["message"]
                    agg["reasoning"] += msg.get("reasoning_content") or ""
                    agg["content"] += msg.get("content") or ""
                else:
                    agg["content"] += ch.get("text") or ""
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return agg


def collect_stream(backend, payload: dict, chat: bool) -> dict:
    """Open ``backend.stream`` and fully buffer it (runs in a worker thread).

    ``backend.stream`` checks readiness and opens the upstream connection
    eagerly, so a pre-first-chunk :class:`~backend.BackendError` propagates here
    before any response headers are sent.
    """
    chunks = backend.stream(payload, chat)
    return aggregate_stream(chunks, chat)


def replay_chunks(agg: dict, policy: GuardrailPolicy, chat: bool) -> Iterator[dict]:
    """Yield OpenAI-shaped chunks for a buffered response, or a safe refusal.

    When the buffered output violates the policy the rejected text is not
    emitted at all: only the fallback message (and no reasoning) is replayed,
    with ``finish_reason="content_filter"``.
    """
    base = {k: agg.get(k) for k in ("id", "object", "created", "model")}
    content = agg.get("content") or ""
    reasoning = agg.get("reasoning") or ""
    finish = agg.get("finish_reason")

    violations = validate(content, policy)
    if policy.guard_reasoning and reasoning:
        violations = violations + validate(reasoning, policy, is_reasoning=True)

    if violations:
        content = policy.fallback_message
        reasoning = ""
        finish = "content_filter"

    meta = None
    if violations:
        meta = {
            "status": "rejected",
            "violations": [{"rule": v.rule, "detail": v.detail}
                           for v in violations],
        }
    else:
        meta = {"status": "passed"}

    if chat:
        yield {**base, "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        if reasoning:
            yield {**base, "choices": [
                {"index": 0, "delta": {"reasoning_content": reasoning},
                 "finish_reason": None}]}
        if content:
            yield {**base, "choices": [
                {"index": 0, "delta": {"content": content},
                 "finish_reason": None}]}
        final = {**base, "choices": [
            {"index": 0, "delta": {}, "finish_reason": finish}]}
    else:
        if content:
            yield {**base, "choices": [
                {"index": 0, "text": content, "finish_reason": None}]}
        final = {**base, "choices": [
            {"index": 0, "text": "", "finish_reason": finish}]}

    if agg.get("usage") is not None:
        final["usage"] = agg["usage"]
    final["guardrails"] = meta
    yield final
