from __future__ import annotations

import warnings
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence

from .._objects import APIObject
from ..structure import (
    cap_reference_rows,
    detect_framework,
    estimator_to_spec,
    features_from_spec,
    labels_from_target,
    rows_from_inputs,
    spec_is_executable,
    warn_if_large_spec,
)
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
        structure: Optional[Mapping[str, Any]] = None,
        project_id: Optional[str] = None,
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
            "structure": dict(structure) if structure else None,
            "project_id": project_id,
        }
        body.update(extra)
        return self._post("/v1/models", compact(body))

    def retrieve(self, model: str) -> APIObject:
        return self._get(f"/v1/models/{model}")

    def list(
        self,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        task: Optional[str] = None,
        project_id: Optional[str] = None,
        unassigned: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Page:
        return self._page(
            "/v1/models",
            {
                "q": q,
                "status": status,
                "task": task,
                "project_id": project_id,
                "unassigned": unassigned,
                "limit": limit,
                "offset": offset,
            },
        )

    def iterate(self, **params: Any) -> Iterator[Any]:
        """Yield every model, following pagination automatically."""
        params.pop("offset", None)
        return self._iter("/v1/models", params)

    def update(self, model: str, **fields: Any):
        return self._patch(f"/v1/models/{model}", compact(fields))

    def delete(self, model: str):
        return self._delete(f"/v1/models/{model}")

    def assign_project(self, model: str, project_id: Optional[str]):
        """Move a model into a project. Pass ``None`` to leave it ungrouped."""
        return self._patch(f"/v1/models/{model}", {"project_id": project_id})

    def versions(self, model: str) -> Page:
        return self._page(f"/v1/models/{model}/versions")

    def create_version(
        self,
        model: str,
        *,
        version: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        slug: Optional[str] = None,
    ):
        return self._post(
            f"/v1/models/{model}/versions",
            compact({"version": version, "name": name, "description": description, "slug": slug}),
        )

    def delete_version(self, model: str, version: str):
        return self._delete(f"/v1/models/{model}/versions/{version}")

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

    def upload(
        self,
        estimator: Any,
        *,
        name: str,
        feature_names: Optional[Sequence[str]] = None,
        features: Optional[Sequence[Mapping[str, Any]]] = None,
        task: Optional[str] = None,
        framework: Optional[str] = None,
        outputs: Optional[Mapping[str, Any]] = None,
        inference: Optional[Mapping[str, Any]] = None,
        slug: Optional[str] = None,
        version: str = "1",
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        project_id: Optional[str] = None,
        X: Any = None,
        y: Any = None,
    ):
        """Register a fitted estimator and upload its layers and learned weights.

        Call this from a notebook that already holds the trained object. Kaimei
        stores the extracted JSON spec (not the original ``.h5`` / ``.pt`` file)
        so the dashboard can draw every layer and tensor.

        Executable families (linear, MLP, tree, small forest) can be queried for
        explanations and analyses. Summaries and approximate nets need
        ``inference={"type": "remote", "url": "..."}`` or pre-computed attributions.
        """
        spec = estimator_to_spec(estimator, feature_names=feature_names, task=task, framework=framework)
        warn_if_large_spec(spec)
        names = list(feature_names) if feature_names is not None else list(spec.get("feature_names") or [])
        feature_list = [dict(item) for item in features] if features is not None else features_from_spec(spec)
        resolved_task = task or spec.get("task")
        resolved_framework = framework or spec.get("framework") or detect_framework(estimator)
        resolved_outputs = dict(outputs) if outputs else spec.get("outputs")
        resolved_inference = dict(inference) if inference else {"type": "none"}
        extra_meta = {"uploaded_from": "python-client"}
        if metadata:
            extra_meta.update(dict(metadata))
        if resolved_inference.get("type", "none") == "none" and not spec_is_executable(spec):
            warnings.warn(
                "This estimator was stored as a sketch or feature-importance summary. "
                "Kaimei can visualize it, but explain/predict/analyse need a remote "
                "inference URL or pre-computed attributions.",
                UserWarning,
                stacklevel=2,
            )
        created = self.create(
            name=name,
            task=resolved_task,
            features=feature_list or None,
            slug=slug,
            version=version,
            description=description,
            framework=resolved_framework,
            outputs=resolved_outputs,
            inference=resolved_inference,
            tags=tags,
            metadata=extra_meta,
            structure=spec,
            project_id=project_id,
        )
        model_ref = created.get("id") or created.get("slug") or name
        if X is not None:
            rows, _names = rows_from_inputs(X, names or None)
            labels = labels_from_target(y) if y is not None else None
            rows, labels = cap_reference_rows(rows, labels)
            self.upload_reference_data(str(model_ref), rows=rows, labels=labels)
        return created

    def structure(self, model: str):
        return self._get(f"/v1/models/{model}/structure")

    def upload_structure(
        self,
        model: str,
        estimator_or_spec: Any,
        *,
        feature_names: Optional[Sequence[str]] = None,
        task: Optional[str] = None,
        framework: Optional[str] = None,
    ):
        """Attach layers and weights to an existing model.

        Pass a fitted estimator or an already-extracted structure dict.
        """
        if isinstance(estimator_or_spec, Mapping):
            spec = dict(estimator_or_spec)
        else:
            spec = estimator_to_spec(
                estimator_or_spec, feature_names=feature_names, task=task, framework=framework
            )
        return self._put(f"/v1/models/{model}/structure", spec)

    def delete_structure(self, model: str):
        return self._delete(f"/v1/models/{model}/structure")

    def trace_structure(self, model: str, *, input: Mapping[str, Any]):
        return self._post(f"/v1/models/{model}/structure/trace", {"input": dict(input)})

    def structure_parameter(self, model: str, parameter_id: str, *, offset: int = 0, limit: int = 4096):
        return self._get(
            f"/v1/models/{model}/structure/parameters/{parameter_id}",
            {"offset": offset, "limit": limit},
        )
