"""Legacy backend: HuggingFace ``transformers`` text2text pipeline (fp32 CPU).

This reproduces the original server.py behavior (FLAN-style prompt wrapper,
hard-coded sampling knobs, whitespace-era defaults) so that
``LLM_BACKEND=transformers`` keeps working as a rollback path, with the
documented bug fixes applied:

* correct ``model`` id in responses (was ``"gpt2"`` for chat),
* no character-slicing of chat answers (was ``result[len(prompt):]``),
* ``max_new_tokens`` semantics for chat (was a word-count ``max_length``),
* real token counts in ``usage`` (was whitespace counting),
* single application of stop sequences.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Iterator

from .base import Backend, BackendError


class _Text2TextShim:
    """Pipeline-shaped replacement for the removed ``text2text-generation``
    pipeline task (dropped in transformers v5). Same call contract:
    ``shim(prompt, **gen_kwargs) -> [{"generated_text": ...}]``."""

    def __init__(self, model_name: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        ml = self.tokenizer.model_max_length
        self.max_source_length = ml if isinstance(ml, int) and 0 < ml < 100000 else 512

    def __call__(self, prompt: str, **gen_kwargs):
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.max_source_length)
        out_ids = self.model.generate(**enc, **gen_kwargs)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return [{"generated_text": text}]


class TransformersBackend(Backend):
    def __init__(self, model_id: str, model_name: str,
                 max_tokens_default: int = 256, max_tokens_cap: int = 4096):
        self.model_id = model_id
        self.model_name = model_name
        self.max_tokens_default = max_tokens_default
        self.max_tokens_cap = max_tokens_cap
        self._generator = None
        self._lock = threading.Lock()
        self._state = "stopped"
        self._detail = "not started"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._state in ("loading", "ready"):
                return
            self._state = "loading"
            self._detail = f"loading {self.model_name} (transformers)"
        try:
            print(f"[backend:transformers] loading {self.model_name} ...")
            gen = _Text2TextShim(self.model_name)
            with self._lock:
                self._generator = gen
                self._state = "ready"
                self._detail = f"transformers ready (model {self.model_id})"
            print(f"[backend:transformers] {self._detail}")
        except Exception as e:
            with self._lock:
                self._state = "error"
                self._detail = f"model failed to load: {e}"
            print(f"[backend:transformers] {self._detail}")

    def health(self):
        with self._lock:
            return (self._state == "ready"), self._detail

    def shutdown(self) -> None:
        with self._lock:
            self._generator = None
            self._state = "stopped"
            self._detail = "stopped"

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def _require_ready(self):
        ready, detail = self.health()
        if not ready:
            raise BackendError(f"model not ready: {detail}", status=503)
        return self._generator

    @staticmethod
    def _stop_sequences(payload: dict):
        stop = payload.get("stop")
        if not stop:
            return []
        seqs = stop if isinstance(stop, list) else [stop]
        return [s for s in seqs if isinstance(s, str) and s]

    @staticmethod
    def _apply_stop(text: str, stop_sequences) -> str:
        for seq in stop_sequences:
            if seq and seq in text:
                text = text.split(seq)[0]
        return text.strip()

    def _count(self, gen, text: str) -> int:
        try:
            ids = gen.tokenizer(text, add_special_tokens=False)["input_ids"]
            return len(ids)
        except Exception:
            return len(text.split())

    def _generate(self, effective_prompt: str, gen_kwargs: dict, stop_sequences):
        gen = self._require_ready()
        with self._lock:
            out = gen(effective_prompt, **gen_kwargs)
        text = out[0]["generated_text"].strip()
        text = self._apply_stop(text, stop_sequences)
        return text, gen

    def _limits(self, payload: dict, minimum: int = None):
        """max_tokens with defaults/cap (normalization is normally done by
        server.py; repeated here so the backend is self-contained)."""
        raw = payload.get("max_tokens", self.max_tokens_default)
        try:
            value = int(raw)
        except Exception:
            value = self.max_tokens_default
        value = max(1, min(value, self.max_tokens_cap))
        if minimum is not None:
            value = max(value, max(0, int(minimum)))
        return value

    # -- /v1/completions --------------------------------------------------
    def _completions(self, payload: dict):
        prompt = (payload.get("prompt") or "").strip()
        system = (payload.get("system") or "").strip()
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 1.0))
        min_tokens = payload.get("min_tokens", 12)
        try:
            min_tokens = int(min_tokens)
        except Exception:
            min_tokens = 12
        max_tokens = self._limits(payload)
        min_tokens = max(0, min(min_tokens, max_tokens))

        if system:
            effective = f"Instruction: {system}\nQuestion: {prompt}\nAnswer:".strip()
        else:
            effective = f"Question: {prompt}\nAnswer:".strip()

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "min_new_tokens": min_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": bool(temperature and temperature > 0),
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        }
        stop_sequences = self._stop_sequences(payload)
        text, gen = self._generate(effective, gen_kwargs, stop_sequences)

        prompt_tokens = self._count(gen, effective)
        completion_tokens = self._count(gen, text)
        response = {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return response, text, effective

    # -- /v1/chat/completions ---------------------------------------------
    def _chat(self, payload: dict):
        messages = payload.get("messages") or []
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 1.0))
        max_tokens = self._limits(payload)

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

        gen = self._require_ready()
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": bool(temperature and temperature > 0),
            "pad_token_id": gen.tokenizer.eos_token_id,
        }
        stop_sequences = self._stop_sequences(payload)
        with self._lock:
            out = gen(prompt, **gen_kwargs)
        text = out[0]["generated_text"].strip()
        if "\n" in text:
            text = text.split("\n")[0]
        text = self._apply_stop(text, stop_sequences)

        prompt_tokens = self._count(gen, prompt)
        completion_tokens = self._count(gen, text)
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return response, text, prompt

    # -- interface ---------------------------------------------------------
    def complete(self, payload: dict, chat: bool) -> dict:
        try:
            if chat:
                response, _, _ = self._chat(payload)
            else:
                response, _, _ = self._completions(payload)
            return response
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"transformers generation failed: {e}", status=500)

    def stream(self, payload: dict, chat: bool) -> Iterator[dict]:
        """Generate eagerly (errors surface before any chunk), then yield
        OpenAI-shaped chunks."""
        response = self.complete(payload, chat)
        return self._iter_stream(response, chat)

    def _iter_stream(self, response: dict, chat: bool) -> Iterator[dict]:
        base = {k: response[k] for k in ("id", "object", "created", "model")}
        if chat:
            yield {**base, "choices": [{
                "index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            content = response["choices"][0]["message"]["content"]
            if content:
                yield {**base, "choices": [{
                    "index": 0, "delta": {"content": content}, "finish_reason": None}]}
            yield {**base, "choices": [{
                "index": 0, "delta": {},
                "finish_reason": response["choices"][0]["finish_reason"]}],
                "usage": response["usage"]}
        else:
            text = response["choices"][0]["text"]
            if text:
                yield {**base, "choices": [{
                    "index": 0, "text": text, "finish_reason": None}]}
            yield {**base, "choices": [{
                "index": 0, "text": "",
                "finish_reason": response["choices"][0]["finish_reason"]}],
                "usage": response["usage"]}
