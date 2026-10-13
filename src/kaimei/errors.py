"""Exceptions raised by the Kaimei client."""

from __future__ import annotations

from typing import Any, Dict, Optional


class KaimeiError(Exception):
    """Base class for every client-side and API error."""

    def __init__(
        self,
        message: str,
        *,
        type: Optional[str] = None,
        code: Optional[str] = None,
        param: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"[{self.code}]")
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        return " ".join(parts)


class AuthenticationError(KaimeiError):
    """The API key or session token was rejected."""


class PermissionDeniedError(KaimeiError):
    """The caller is authenticated but not allowed to do this."""


PermissionError = PermissionDeniedError


class NotFoundError(KaimeiError):
    """The requested resource does not exist (or is in another workspace)."""


class RateLimitError(KaimeiError):
    """Too many requests."""


class APIConnectionError(KaimeiError):
    """The HTTP request never reached a usable response."""


class APIStatusError(KaimeiError):
    """Any other non-2xx response from the API."""


def error_from_response(status_code: int, body: Any) -> KaimeiError:
    payload = body if isinstance(body, dict) else {}
    raw_err = payload.get("error")
    err: Dict[str, Any] = raw_err if isinstance(raw_err, dict) else {}
    message = err.get("message") or payload.get("message") or f"Request failed with HTTP {status_code}."
    kwargs: Dict[str, Any] = {
        "type": err.get("type"),
        "code": err.get("code"),
        "param": err.get("param"),
        "status_code": status_code,
        "details": err.get("details") or {},
    }
    if status_code == 401 or err.get("type") == "authentication_error":
        return AuthenticationError(message, **kwargs)
    if status_code == 403 or err.get("type") == "permission_error":
        return PermissionDeniedError(message, **kwargs)
    if status_code == 404 or err.get("type") == "not_found_error":
        return NotFoundError(message, **kwargs)
    if status_code == 429 or err.get("type") == "rate_limit_error":
        return RateLimitError(message, **kwargs)
    return APIStatusError(message, **kwargs)
