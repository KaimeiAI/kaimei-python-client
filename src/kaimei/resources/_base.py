from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from .._objects import APIObject, Page, wrap

if TYPE_CHECKING:
    from ..client import Kaimei


class Resource:
    def __init__(self, client: "Kaimei") -> None:
        self._client = client

    def _get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        return wrap(self._client._request("GET", path, params=params))

    def _post(self, path: str, body: Optional[Mapping[str, Any]] = None) -> Any:
        return wrap(self._client._request("POST", path, json_body=body))

    def _patch(self, path: str, body: Optional[Mapping[str, Any]] = None) -> Any:
        return wrap(self._client._request("PATCH", path, json_body=body))

    def _put(self, path: str, body: Optional[Mapping[str, Any]] = None) -> Any:
        return wrap(self._client._request("PUT", path, json_body=body))

    def _delete(self, path: str) -> Any:
        return wrap(self._client._request("DELETE", path))

    def _page(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Page:
        data = self._client._request("GET", path, params=params)
        if isinstance(data, dict):
            return Page(data)
        return Page({"object": "list", "data": data, "total": len(data), "limit": len(data), "offset": 0, "has_more": False})


def compact(body: Optional[Mapping[str, Any]]) -> Optional[dict]:
    if body is None:
        return None
    return {k: v for k, v in body.items() if v is not None}


__all__ = ["Resource", "compact", "APIObject", "Page"]
