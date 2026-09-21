"""Attribute-accessible wrappers around JSON objects returned by the API."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping


def wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return APIObject(value)
    if isinstance(value, list):
        return [wrap(item) for item in value]
    return value


class APIObject(Mapping[str, Any]):
    """A JSON object you can read as ``obj.top_features`` or ``obj["top_features"]``."""

    def __init__(self, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))
        for key, value in data.items():
            object.__setattr__(self, key, wrap(value))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __getitem__(self, key: str) -> Any:
        return wrap(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key in self._data:
            return wrap(self._data[key])
        return default

    def __repr__(self) -> str:
        kind = self._data.get("object") or "object"
        ident = self._data.get("id") or self._data.get("slug") or self._data.get("name")
        if ident:
            return f"<Kaimei {kind} {ident}>"
        return f"<Kaimei {kind}>"


class Page(APIObject):
    """A paginated list: ``page.data``, ``page.total``, ``page.has_more``."""

    @property
    def items(self) -> List[Any]:
        return list(self.data)  # type: ignore[arg-type]
