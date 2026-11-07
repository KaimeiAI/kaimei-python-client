from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Optional

from .._objects import APIObject, Page, wrap

if TYPE_CHECKING:
    from ..client import Kaimei


class Resource:
    def __init__(self, client: "Kaimei") -> None:
        self._client = client

    def _check_gaps(self, model: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """NaN is how pandas says missing, and it travels as null. The API then gives any feature that can't take a
        gap its baseline, which is a different question from the one asked, so that is refused here instead."""
        gaps = {name for name, value in row.items() if isinstance(value, Real) and value != value}
        if not gaps:
            return row
        features = self._client._request("GET", f"/v1/models/{model}").get("features") or []
        refused = sorted(str(item["name"]) for item in features if item["name"] in gaps and not item.get("allow_missing"))
        if refused:
            raise ValueError(
                f"{', '.join(refused)} is NaN and model {model} doesn't take missing values there. Send a number, "
                "or None to use the feature's baseline."
            )
        return row

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
        params = dict(params or {})
        data = self._client._request("GET", path, params=params)
        if not isinstance(data, dict):
            rows = list(data)
            return Page(
                {"object": "list", "data": rows, "total": len(rows), "limit": len(rows), "offset": 0, "has_more": False}
            )

        def _next() -> Page:
            nxt = dict(params)
            items = data.get("data") or []
            nxt["offset"] = int(nxt.get("offset") or 0) + len(items)
            return self._page(path, nxt)

        return Page(data, _next=_next)

    def _iter(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Iterator[Any]:
        page = self._page(path, params)
        yield from page.auto_paging_iter()


def compact(body: Optional[Mapping[str, Any]]) -> Optional[dict]:
    if body is None:
        return None
    return {k: v for k, v in body.items() if v is not None}


__all__ = ["Resource", "compact", "APIObject", "Page"]
