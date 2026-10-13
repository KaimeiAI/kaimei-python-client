from __future__ import annotations

import time
from typing import Any, Iterator, Mapping, Optional

from ..errors import APIStatusError
from ._base import Page, Resource, compact


class Analyses(Resource):
    def create(self, *, model: str, type: str = "full", options: Optional[Mapping[str, Any]] = None):
        return self._post("/v1/analyses", compact({"model": model, "type": type, "options": dict(options) if options else None}))

    def retrieve(self, analysis_id: str):
        return self._get(f"/v1/analyses/{analysis_id}")

    def list(
        self,
        *,
        model: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Page:
        return self._page(
            "/v1/analyses",
            {"model": model, "status": status, "limit": limit, "offset": offset},
        )

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/analyses", params)

    def wait(
        self,
        analysis_id: str,
        *,
        timeout: float = 300.0,
        interval: float = 1.5,
    ):
        """Poll until the analysis is ``completed`` or ``failed``."""
        deadline = time.monotonic() + timeout
        latest = self.retrieve(analysis_id)
        while getattr(latest, "status", None) in {"queued", "running"}:
            if time.monotonic() > deadline:
                raise APIStatusError(
                    f"Analysis {analysis_id} did not finish within {timeout}s.",
                    code="analysis_timeout",
                    status_code=408,
                )
            time.sleep(interval)
            latest = self.retrieve(analysis_id)
        return latest
