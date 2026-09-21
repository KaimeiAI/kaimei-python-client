from __future__ import annotations

from typing import Optional

from ._base import Page, Resource


class Insights(Resource):
    def retrieve(self, insight_id: str):
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
