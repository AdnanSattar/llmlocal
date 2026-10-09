"""API-key authentication for external /v1 and job/queue clients.

This module is the integration contract between the FastAPI migration and the
API-key authentication workstreams:

* ``require_api_key`` — FastAPI dependency guarding external API routes.
* ``install_auth(app)`` — registers the 401 error handler on the app.
* ``AuthError`` — raised by the dependency; mapped to a fixed OpenAI-style body.
* ``auth_enabled()`` — whether API-key enforcement is active.

Policy (documented in docs/API_EXAMPLES.md):

* Authentication is enforced only when ``API_KEY`` is set to a non-empty value
  in the environment. When it is unset/empty the service runs in development
  mode (routes open) and a startup notice says so.
* Keys are accepted ONLY from the ``Authorization: Bearer <key>`` header.
  Query-string keys (``?api_key=...``) are never read or accepted.
* A missing key and an invalid key produce the SAME 401 body — no existence
  leak. Keys are compared in constant time (``hmac.compare_digest``) and are
  never logged, echoed in error messages, or included in tracebacks.
* Worker authentication (``WORKER_TOKEN`` on ``/jobs/next`` and
  ``/workers/heartbeat``) is entirely separate and always enforced,
  regardless of ``API_KEY``.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

__all__ = ["AuthError", "require_api_key", "install_auth", "auth_enabled"]


class AuthError(Exception):
    """401 from API-key auth. Carries a fixed body that never echoes credentials."""

    def __init__(self, status_code: int = 401, body: Optional[dict] = None):
        self.status_code = status_code
        self.body = body if body is not None else _error_body()
        # Generic message: safe for tracebacks/logs (no key material).
        super().__init__(f"auth failed ({status_code})")


def _error_body() -> dict:
    # Same body for missing and invalid credentials: no information leak.
    # Built fresh per call so no shared mutable state.
    return {
        "error": {
            "message": "Invalid or missing API key",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }


def auth_enabled() -> bool:
    """True when a non-empty (non-whitespace) API_KEY is configured."""
    return bool((os.environ.get("API_KEY") or "").strip())


def _extract_bearer(authorization: Optional[str]) -> str:
    """Pull the credential out of an Authorization header value.

    Tolerant of the harmless variations clients send, strict about everything
    else:

    * scheme is case-insensitive (``bearer``, ``BEARER``, ``Bearer``)
    * any run of whitespace separates the scheme from the credential
      (``Bearer\\tkey`` and ``Bearer   key`` both work); surrounding
      whitespace on the header value is ignored
    * non-Bearer schemes (``Basic ...``, ``Digest ...``) and malformed
      headers (``Bearer`` with no credential, empty credential, garbage)
      yield ``""`` — treated as a missing key
    * internal whitespace inside the credential is preserved as-is; it will
      simply fail the constant-time comparison
    """
    if not authorization:
        return ""
    parts = authorization.strip().split(None, 1)
    if not parts:
        return ""
    scheme = parts[0]
    if scheme.lower() != "bearer":
        return ""
    if len(parts) < 2:
        return ""  # "Bearer" with no credential
    return parts[1].strip()


_bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description=(
        "API key sent as "
        "'Authorization: Bearer <key>'."
    ),
)


async def require_api_key(
    request: Request,
    _credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency: enforce ``Authorization: Bearer <API_KEY>``.

    Development mode: when ``API_KEY`` is unset/empty the request passes
    through without credentials checks (documented policy).

    No key is ever logged, echoed, or compared except in constant time.
    The only accepted source is the Authorization header — query parameters
    are never consulted.

    The nested ``_credentials`` sub-dependency exists only so the OpenAPI
    schema advertises the ``BearerAuth`` scheme (Swagger UI "Authorize"
    button on /docs); parsing of the actual header stays in
    ``_extract_bearer`` above, unchanged.
    """
    configured = (os.environ.get("API_KEY") or "").strip()
    if not configured:
        # Development mode: API_KEY not configured -> routes stay open.
        return
    provided = _extract_bearer(request.headers.get("Authorization"))
    if not provided:
        raise AuthError(401)
    if not hmac.compare_digest(provided.encode("utf-8"), configured.encode("utf-8")):
        raise AuthError(401)


def auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    """Fixed 401 body for every auth failure; no credential material anywhere."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.body,
        headers={"WWW-Authenticate": "Bearer"},
    )


def install_auth(app) -> None:
    """Register auth error handling and announce the auth mode (no secrets)."""
    app.add_exception_handler(AuthError, auth_error_handler)
    mode = "enabled" if auth_enabled() else "disabled"
    print(f"API key authentication: {mode}")
