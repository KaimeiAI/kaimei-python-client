from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Optional

from .._objects import APIObject
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
        """Create an explanation.

        ``method`` is ``shap`` (KernelSHAP), ``shapley`` (permutation sampling),
        ``lime`` (tabular features or image patches), ``gradcam`` (image models whose
        convolutional layers were uploaded), ``occlusion`` (image patches), or ``auto``.

        Image options: ``target_class`` (default: the predicted class), ``grid`` (patches per
        side; 4, or 8 for images 64 pixels or larger), and for Grad-CAM ``layer`` (a node id;
        default: the last convolutional feature map). Image explanations return
        ``options["heatmap"]``, a grid in ``[-1, 1]`` (``[0, 1]`` for Grad-CAM) covering the image.
        """
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

    def retrieve(self, explanation_id: str) -> APIObject:
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

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/explanations", params)
