from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

from ._base import Page, Resource, compact


class Models(Resource):
    def create(
        self,
        *,
        name: str,
        task: Optional[str] = None,
        features: Optional[Sequence[Mapping[str, Any]]] = None,
        slug: Optional[str] = None,
        version: str = "1",
        description: Optional[str] = None,
        framework: str = "custom",
        outputs: Optional[Mapping[str, Any]] = None,
        inference: Optional[Mapping[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **extra: Any,
    ):
        body = {
            "name": name,
            "task": task,
            "features": [dict(f) for f in features] if features is not None else None,
            "slug": slug,
            "version": version,
            "description": description,
            "framework": framework,
            "outputs": dict(outputs) if outputs else None,
            "inference": dict(inference) if inference else None,
            "tags": list(tags) if tags is not None else None,
            "metadata": dict(metadata) if metadata else None,
        }
        body.update(extra)
        return self._post("/v1/models", compact(body))

    def retrieve(self, model: str):
        return self._get(f"/v1/models/{model}")

    def list(
        self,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        return self._page(
            "/v1/models",
            {"q": q, "status": status, "task": task, "limit": limit, "offset": offset},
        )

    def update(self, model: str, **fields: Any):
        return self._patch(f"/v1/models/{model}", compact(fields))

    def delete(self, model: str):
        return self._delete(f"/v1/models/{model}")

    def predict(self, model: str, *, input: Mapping[str, Any], log: bool = True, metadata: Optional[Mapping[str, Any]] = None):
        return self._post(
            f"/v1/models/{model}/predict",
            compact({"input": dict(input), "log": log, "metadata": dict(metadata) if metadata else None}),
        )

    def stats(self, model: str, *, days: int = 30):
        return self._get(f"/v1/models/{model}/stats", {"days": days})

    def upload_reference_data(
        self,
        model: str,
        *,
        rows: Sequence[Mapping[str, Any]],
        labels: Optional[List[Any]] = None,
        mode: str = "replace",
    ):
        return self._put(
            f"/v1/models/{model}/reference-data",
            compact({"rows": [dict(r) for r in rows], "labels": labels, "mode": mode}),
        )

    def reference_data(self, model: str, *, sample: int = 5):
        return self._get(f"/v1/models/{model}/reference-data", {"sample": sample})

    def delete_reference_data(self, model: str):
        return self._delete(f"/v1/models/{model}/reference-data")
