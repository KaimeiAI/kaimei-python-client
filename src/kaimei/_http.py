"""Low-level HTTP transport."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Mapping, Optional, Set
from urllib.parse import urlencode

import httpx

from .errors import APIConnectionError, error_from_response

DEFAULT_BASE_URL = "https://api.kaimei.ai"
DEFAULT_TIMEOUT = 120.0
RETRYABLE_STATUS: Set[int] = {429, 502, 503, 504}
IDEMPOTENT_METHODS = {"GET", "HEAD", "OPTIONS"}
#: Failures that happen before a request reaches the server, so any method can be retried.
NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class Transport:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        timeout: float,
        user_agent: str,
        extra_headers: Optional[Mapping[str, str]] = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, int(max_retries))
        headers = {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        content: Optional[bytes] = None,
        raw: bool = False,
        expected: Optional[int] = None,
    ) -> Any:
        """Send one request. ``content`` uploads raw bytes; ``raw`` returns the response bytes."""
        query = _clean_params(params)
        url = path if path.startswith("http") else path
        verb = method.upper()
        headers: Optional[Dict[str, str]] = None
        if json_body is not None:
            headers = {"Content-Type": "application/json"}
            content = encode_json(json_body)
        elif content is not None:
            headers = {"Content-Type": "application/octet-stream"}
        if raw:
            headers = {**(headers or {}), "Accept": "application/octet-stream"}
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    verb,
                    url,
                    params=query or None,
                    content=content,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                # A POST that timed out may still be running on the server; sending it again would
                # apply it twice (a batch of predictions logged four times). Retry only requests that
                # are safe to repeat, or that never reached the server.
                if not (verb in IDEMPOTENT_METHODS or isinstance(exc, NOT_SENT_ERRORS)):
                    raise APIConnectionError(
                        f"No response from Kaimei at {self.base_url}: {exc}. The {verb} was not retried "
                        "because the server may already have applied it."
                    ) from exc
                if attempt >= self.max_retries:
                    raise APIConnectionError(f"Could not reach Kaimei at {self.base_url}: {exc}") from exc
                time.sleep(_backoff(attempt))
                continue

            if raw and 200 <= response.status_code < 300:
                return response.content
            if expected is not None and response.status_code == expected:
                return _empty_or_decode(response)
            if 200 <= response.status_code < 300:
                return _empty_or_decode(response)
            if _should_retry(verb, response.status_code) and attempt < self.max_retries:
                time.sleep(_retry_delay(response, attempt))
                continue
            raise error_from_response(response.status_code, _decode(response))

        raise APIConnectionError(f"Could not reach Kaimei at {self.base_url}: {last_error}")


def encode_json(body: Any) -> bytes:
    """Compact JSON, as httpx sends it, that also accepts NumPy arrays and scalars."""
    return json.dumps(body, default=_json_default, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _json_default(value: Any) -> Any:
    # NumPy arrays and scalars (and anything else array-like) serialize as their Python values.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _empty_or_decode(response: httpx.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return None
    return _decode(response)


def _should_retry(method: str, status_code: int) -> bool:
    if status_code == 429:
        return True
    return method in IDEMPOTENT_METHODS and status_code in RETRYABLE_STATUS


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return min(max(float(header), 0.0), 30.0)
        except ValueError:
            pass
    return _backoff(attempt)


def _backoff(attempt: int) -> float:
    return min(0.5 * (2**attempt), 8.0)


def _decode(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"message": response.text}


def _clean_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = "true" if value else "false"
        else:
            cleaned[key] = value
    return cleaned


def qs(params: Mapping[str, Any]) -> str:
    cleaned = _clean_params(params)
    return urlencode(cleaned, doseq=True)
