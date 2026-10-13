from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional, Sequence

from .._objects import APIObject
from ._base import Page, Resource, compact


class Predictions(Resource):
    def create(
        self,
        *,
        model: str,
        input: Mapping[str, Any],
        output: Any = None,
        prediction_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **extra: Any,
    ):
        body = {
            "model": model,
            "input": dict(input),
            "output": output,
            "prediction_id": prediction_id,
            "latency_ms": latency_ms,
            "metadata": {"sdk": "python", **(dict(metadata) if metadata else {})},
        }
        body.update(extra)
        return self._post("/v1/predictions", compact(body))

    def create_batch(self, *, model: str, items: Sequence[Mapping[str, Any]]):
        return self._post("/v1/predictions/batch", {"model": model, "items": [dict(i) for i in items]})

    def retrieve(self, prediction_id: str) -> APIObject:
        return self._get(f"/v1/predictions/{prediction_id}")

    def list(
        self,
        *,
        model: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        return self._page(
            "/v1/predictions",
            {"model": model, "source": source, "limit": limit, "offset": offset},
        )

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/predictions", params)
