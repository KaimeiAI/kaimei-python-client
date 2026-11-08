from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union
from urllib.parse import quote

from .._objects import APIObject, wrap
from ._base import Page, Resource, compact

#: Outcomes the API takes in one request.
ACTUALS_PER_REQUEST = 1000


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
        actual: Any = None,
        **extra: Any,
    ):
        body = {
            "model": model,
            "input": dict(input),
            "output": output,
            "prediction_id": prediction_id,
            "latency_ms": latency_ms,
            "metadata": {"sdk": "python", **(dict(metadata) if metadata else {})},
            "actual": actual,
        }
        body.update(extra)
        return self._post("/v1/predictions", compact(body))

    def create_batch(self, *, model: str, items: Sequence[Mapping[str, Any]]):
        return self._post("/v1/predictions/batch", {"model": model, "items": [dict(i) for i in items]})

    def log_actual(self, prediction_id: str, actual: Any, *, model: Optional[str] = None) -> APIObject:
        """Record what really happened for one logged prediction. ``None`` clears it."""
        return self.log_actuals([{"prediction_id": prediction_id, "actual": actual}], model=model)

    def log_actuals(
        self,
        actuals: Union[Mapping[str, Any], Sequence[Mapping[str, Any]], Any],
        *,
        model: Optional[str] = None,
    ) -> APIObject:
        """Record real outcomes (ground truth) for logged predictions, in requests of up to 1,000.

        ``actuals`` is ``{prediction_id: outcome}`` (a dict or a pandas Series indexed by id), a list of
        ``{"prediction_id": ..., "actual": ...}``, or a DataFrame with those two columns. An outcome is a class name or position, ``True`` / ``False``
        for a binary model, or a number for a regression; ``None`` clears one, and NaN (not known yet) is
        skipped. In a Series, ``None`` is missing as well and is skipped. With ``model``, ids of other
        models' predictions come back in ``not_found``. Each request is checked whole: an outcome that doesn't
        fit the model raises before that request writes anything.
        """
        items = _actual_items(actuals)
        if not items:
            raise ValueError("No outcomes to log.")
        total: Dict[str, Any] = {"object": "actuals", "updated": 0, "cleared": 0, "not_found": []}
        for start in range(0, len(items), ACTUALS_PER_REQUEST):
            body = compact({"model": model, "items": items[start : start + ACTUALS_PER_REQUEST]})
            done = self._client._request("POST", "/v1/predictions/actuals", json_body=body)
            total["updated"] += int(done.get("updated") or 0)
            total["cleared"] += int(done.get("cleared") or 0)
            total["not_found"].extend(done.get("not_found") or [])
        return wrap(total)

    def retrieve(self, prediction_id: str) -> APIObject:
        # An id can hold anything a path can't, such as ? or #.
        return self._get(f"/v1/predictions/{quote(prediction_id, safe='/')}")

    def list(
        self,
        *,
        model: Optional[str] = None,
        source: Optional[str] = None,
        has_actual: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        params: Dict[str, Any] = {"model": model, "source": source, "limit": limit, "offset": offset}
        if has_actual is not None:
            params["has_actual"] = "true" if has_actual else "false"
        return self._page("/v1/predictions", params)

    def iterate(self, **params: Any) -> Iterator[Any]:
        params.pop("offset", None)
        return self._iter("/v1/predictions", params)


def _actual_items(actuals: Any) -> List[Dict[str, Any]]:
    """Request items, leaving out NaN: pandas' "not known yet", which must not clear an outcome already recorded."""
    if hasattr(actuals, "columns") and hasattr(actuals, "isna"):
        # A DataFrame: one row per outcome.
        if "prediction_id" not in actuals.columns or "actual" not in actuals.columns:
            raise ValueError('A DataFrame of outcomes needs "prediction_id" and "actual" columns.')
        known = actuals[~actuals["actual"].isna()]
        return [
            {"prediction_id": str(_plain(pid)), "actual": _plain(value)}
            for pid, value in zip(known["prediction_id"], known["actual"])
        ]
    if hasattr(actuals, "isna") and hasattr(actuals, "items"):
        # A pandas Series: None is missing there too, not a request to clear.
        known = actuals[~actuals.isna()]
        return [{"prediction_id": str(key), "actual": _plain(value)} for key, value in known.items()]
    if hasattr(actuals, "items") and callable(actuals.items) and not isinstance(actuals, (list, tuple)):
        pairs = [(key, _plain(value)) for key, value in actuals.items()]
        return [{"prediction_id": str(key), "actual": value} for key, value in pairs if not _nan(value)]
    items = []
    for index, item in enumerate(actuals):
        if not isinstance(item, Mapping) or "prediction_id" not in item:
            raise ValueError(f'actuals[{index}] needs a "prediction_id" and an "actual".')
        value = _plain(item.get("actual"))
        if not _nan(value):
            items.append({"prediction_id": str(item["prediction_id"]), "actual": value})
    return items


def _plain(value: Any) -> Any:
    """NumPy scalars as Python ones."""
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 0) == 0:
        return value.item()
    return value


def _nan(value: Any) -> bool:
    return isinstance(value, float) and value != value
