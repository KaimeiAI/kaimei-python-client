from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from .._objects import APIObject, wrap
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
        row = dict(input)
        if attributions is None:
            self._check_gaps(model, row)
        body: Dict[str, Any] = {
            "model": model,
            "input": row,
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

    def compare_methods(
        self,
        model: str,
        input: Mapping[str, Any],
        methods: Optional[Sequence[str]] = None,
        *,
        prediction_id: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        top: int = 5,
    ) -> APIObject:
        """Explain one input with several methods, and say how far they agree.

        ``methods`` defaults to every method the API lists for the model (``explain_methods``, without ``auto``,
        which is one of the others under another name). Each method is one call to :meth:`create`, so one
        explanation of the month's allowance. No prediction is logged for them: pass ``prediction_id`` to attach
        them all to a prediction already in the log. ``options`` go to every call (``target_class``, ``seed``).

        Returns the ``explanations``, one per method in order, and what they say together:

        - ``agreement``: a ``sentence`` that follows the numbers ("All three put credit_score first and agree on
          the top three."), the ``rank_correlation`` (Spearman's, over every input's absolute attribution,
          averaged over the pairs of methods), the input each method puts ``first``, ``same_first``, and the
          inputs they push ``opposed`` ways;
        - ``rows``: one per input any method ranks in its top ``top``, by mean rank, with each method's
          attribution and rank (``pandas.DataFrame(comparison.to_dict()["rows"])`` is the table);
        - ``costs``: each method's ``n_evals``, ``latency_ms`` and whether it is ``exact``, with how in ``detail``.
        """
        from .._methods import comparable, summarize

        if methods is None:
            listed = self._client._request("GET", f"/v1/models/{model}").get("explain_methods") or []
            methods = comparable([str(method) for method in listed])
            if not methods:
                raise ValueError(f"The API lists no explanation method for model {model}, so there is nothing to compare.")
        methods = list(methods)
        if len(methods) < 2:
            raise ValueError(f"Comparing methods needs at least two, and model {model} has {methods}.")
        shared = {"log_prediction": False, **dict(options or {})}
        explanations = [
            self.create(model=model, input=input, prediction_id=prediction_id, method=method, options=shared).to_dict()
            for method in methods
        ]
        return wrap(
            {
                "object": "method_comparison",
                "model": model,
                "methods": [str(item.get("method")) for item in explanations],
                "explanations": explanations,
                **summarize(explanations, top=top),
            }
        )

    def retrieve(self, explanation_id: str) -> APIObject:
        return self._get(f"/v1/explanations/{explanation_id}")

    def delete(self, explanation_id: str) -> APIObject:
        """Delete one stored explanation. The decision it explains stays logged.

        An explanation of a decision under a hold is not deleted. The audit log keeps that it was deleted
        and by whom.
        """
        return self._delete(f"/v1/explanations/{explanation_id}")

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
