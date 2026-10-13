from __future__ import annotations

from typing import Any, Iterator, Optional

from .._objects import APIObject
from ._base import Page, Resource


class Insights(Resource):
    def retrieve(self, insight_id: str) -> APIObject:
        return self._get(f"/v1/insights/{insight_id}")

    def list(
        self,
        *,
        model: Optional[str] = None,
        type: Optional[str] = None,
        severity: Optional[str] = None,
        history: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        return self._page(
            "/v1/insights",
            {
                "model": model,
                "type": type,
                "severity": severity,
                "history": history,
                "limit": limit,
                "offset": offset,
            },
        )

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/insights", params)
