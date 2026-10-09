"""LLM backend interface used by server.py.

A backend owns model loading, health, and text generation for the
OpenAI-compatible ``/v1/completions`` and ``/v1/chat/completions``
endpoints. The job queue, circuit breaker, and worker endpoints in
server.py are backend-agnostic and untouched by this interface.
"""

from __future__ import annotations

import abc
from typing import Iterator, Tuple


class BackendError(Exception):
    """A backend failed to serve a request.

    ``status`` is the HTTP status code the API should respond with
    (defaults to 502 when the upstream runtime failed).
    """

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class Backend(abc.ABC):
    """Text generation behind the OpenAI-compatible endpoints."""

    #: id reported in responses and by /v1/models
    model_id: str = "unknown"

    @abc.abstractmethod
    def start(self) -> None:
        """Begin loading the model.

        May block until loaded (server.py calls this from a background
        thread) or return immediately after spawning a child process.
        Must not raise; failures are reported through :meth:`health`.
        """

    @abc.abstractmethod
    def health(self) -> Tuple[bool, str]:
        """Return ``(ready, detail)``. ``detail`` is surfaced by /health."""

    @abc.abstractmethod
    def complete(self, payload: dict, chat: bool) -> dict:
        """Non-streaming generation.

        :param payload: the (normalized) request body from the API.
        :param chat: True for /v1/chat/completions, False for /v1/completions.
        :return: an OpenAI-shaped response dict.
        """

    @abc.abstractmethod
    def stream(self, payload: dict, chat: bool) -> Iterator[dict]:
        """Streaming generation.

        Yields OpenAI-shaped *chunk* dicts (the ``[DONE]`` sentinel is
        added by server.py).
        """

    def shutdown(self) -> None:
        """Stop child processes, if any."""
