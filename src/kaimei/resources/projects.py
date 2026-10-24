from __future__ import annotations

from typing import Any, Iterator, Optional

from .._objects import APIObject
from ._base import Page, Resource, compact


class Projects(Resource):
    def create(self, *, name: str, slug: Optional[str] = None, description: Optional[str] = None):
        return self._post("/v1/projects", compact({"name": name, "slug": slug, "description": description}))

    def retrieve(self, project: str, *, include: Optional[str] = None) -> APIObject:
        """A project. ``include="summary"`` adds activity, traffic and alert roll-ups."""
        return self._get(f"/v1/projects/{project}", compact({"include": include}) or None)

    def list(
        self, *, q: Optional[str] = None, include: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> Page:
        """Projects. ``include="summary"`` adds each project's roll-up (models, traffic, alerts)."""
        return self._page("/v1/projects", compact({"q": q, "include": include, "limit": limit, "offset": offset}))

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/projects", compact(params))

    def overview(self, project: str) -> APIObject:
        """The project dashboard: 30-day activity, per-model traffic and drift, open alerts, the
        feature × model attribution matrix, and recent analyses and explanations."""
        return self._get(f"/v1/projects/{project}/overview")

    def update(self, project: str, **fields):
        return self._patch(f"/v1/projects/{project}", compact(fields))

    def delete(self, project: str):
        return self._delete(f"/v1/projects/{project}")
