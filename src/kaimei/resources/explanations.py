from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ._base import Page, Resource, compact


class Explanations(Resource):
    """``POST /v1/explanations`` — feature-level attributions for one prediction."""

    def create(
        self,
        *,
        model: str,
        input: Mapping[str, Any],
        prediction_id: Optional[str] = None,
        method: str = "shapley",
        options: Optional[Mapping[str, Any]] = None,
        attributions: Optional[Mapping[str, float]] = None,
        output: Any = None,
        base_value: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **extra: Any,
    ):
        body: Dict[str, Any] = {
            "model": model,
            "input": dict(input),
            "prediction_id": prediction_id,
            "method": method,
            "options": dict(options) if options else None,
            "attributions": dict(attributions) if attributions else None,
            "output": output,
            "base_value": base_value,
            "metadata": dict(metadata) if metadata else None,
        }
        body.update(extra)
        return self._post("/v1/explanations", compact(body))

    def retrieve(self, explanation_id: str):
        return self._get(f"/v1/explanations/{explanation_id}")

    def list(
        self,
        *,
        model: Optional[str] = None,
        prediction_id: Optional[str] = None,
        method: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Page:
        return self._page(
            "/v1/explanations",
            {
                "model": model,
                "prediction_id": prediction_id,
                "method": method,
                "limit": limit,
                "offset": offset,
            },
        )
