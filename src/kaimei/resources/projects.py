from __future__ import annotations

from typing import Any, Iterator, Optional

from .._objects import APIObject
from ._base import Page, Resource, compact


class Projects(Resource):
    def create(self, *, name: str, slug: Optional[str] = None, description: Optional[str] = None):
        return self._post("/v1/projects", compact({"name": name, "slug": slug, "description": description}))

    def retrieve(self, project: str) -> APIObject:
        return self._get(f"/v1/projects/{project}")

    def list(self, *, q: Optional[str] = None, limit: int = 50, offset: int = 0) -> Page:
        return self._page("/v1/projects", {"q": q, "limit": limit, "offset": offset})

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/projects", params)

    def update(self, project: str, **fields):
        return self._patch(f"/v1/projects/{project}", compact(fields))

    def delete(self, project: str):
        return self._delete(f"/v1/projects/{project}")
