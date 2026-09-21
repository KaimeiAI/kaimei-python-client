from __future__ import annotations

from typing import Any, Mapping, Optional

from ._base import Resource, compact


class Organization(Resource):
    def retrieve(self):
        return self._get("/v1/organization")

    def update(self, *, name: Optional[str] = None, settings: Optional[Mapping[str, Any]] = None):
        return self._patch("/v1/organization", compact({"name": name, "settings": dict(settings) if settings else None}))

    def overview(self):
        return self._get("/v1/organization/overview")
