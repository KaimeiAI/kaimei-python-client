"""Low-level HTTP transport."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlencode

import httpx

from .errors import APIConnectionError, error_from_response

DEFAULT_BASE_URL = "https://api.kaimei.ai"
DEFAULT_TIMEOUT = 60.0


class Transport:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        timeout: float,
        user_agent: str,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
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
        expected: Optional[int] = None,
    ) -> Any:
        query = _clean_params(params)
        url = path if path.startswith("http") else path
        try:
            response = self._client.request(
                method,
                url,
                params=query or None,
                json=json_body,
                headers={"Content-Type": "application/json"} if json_body is not None else None,
            )
        except httpx.HTTPError as exc:
            raise APIConnectionError(f"Could not reach Kaimei at {self.base_url}: {exc}") from exc

        if expected is not None and response.status_code == expected:
            if expected == 204 or not response.content:
                return None
            return _decode(response)
        if 200 <= response.status_code < 300:
            if response.status_code == 204 or not response.content:
                return None
            return _decode(response)
        raise error_from_response(response.status_code, _decode(response))


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
