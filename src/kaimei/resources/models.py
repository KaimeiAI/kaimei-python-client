from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .._objects import APIObject
from ..structure import (
    EXTERNALIZE_ABOVE,
    Extraction,
    cap_reference_rows,
    detect_framework,
    encode_tensor,
    extract_structure,
    features_from_spec,
    labels_from_target,
    rows_from_inputs,
    spec_is_executable,
    warn_if_large_spec,
)
from ._base import Page, Resource, compact

#: Called as ``progress(stage, done, total)`` while a large model uploads.
Progress = Callable[[str, int, int], None]


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
        weights: str = "full",
        externalize_above: Optional[int] = EXTERNALIZE_ABOVE,
        progress: Optional[Progress] = None,
    ):
        """Register a fitted estimator and upload its layers and learned weights.

        Call this from a notebook that already holds the trained object. Kaimei
        stores the extracted spec (not the original ``.h5`` / ``.pt`` file) so the
        dashboard can draw every layer and tensor. Tensors with more than
        ``externalize_above`` values upload as compressed binary blobs instead of JSON,
        so models with millions of parameters work; ``weights="summary"`` sends shapes
        and statistics only and keeps the weights in your environment.

        Executable families (linear, MLP, tree, small forest, dense/conv nets with
        every weight uploaded) can be queried for explanations and analyses.
        Summaries and approximate nets need ``inference={"type": "remote", "url": "..."}``
        or pre-computed attributions. ``progress(stage, done, total)`` is called while
        tensors upload.
        """
        extraction = extract_structure(
            estimator,
            feature_names=feature_names,
            task=task,
            framework=framework,
            weights=weights,
            externalize_above=externalize_above,
        )
        spec = extraction.spec
        warn_if_large_spec(spec)
        names = list(feature_names) if feature_names is not None else list(spec.get("feature_names") or [])
        feature_list = [dict(item) for item in features] if features is not None else features_from_spec(spec)
        resolved_task = task or spec.get("task")
        resolved_framework = framework or spec.get("framework") or detect_framework(estimator)
        resolved_outputs = dict(outputs) if outputs else spec.get("outputs")
        resolved_inference = dict(inference) if inference else {"type": "none"}
        extra_meta: Dict[str, Any] = {"uploaded_from": "python-client"}
        if extraction.externalized:
            extra_meta["binary_tensors"] = len(extraction.tensors)
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
        elif weights == "summary" and resolved_inference.get("type", "none") == "none":
            warnings.warn(
                "weights='summary' uploads shapes and statistics only. Kaimei draws the architecture, "
                "but explain/predict/analyse need a remote inference URL or pre-computed attributions.",
                UserWarning,
                stacklevel=2,
            )
        # A structure that references binary tensors can only be validated once they exist, so a
        # large model is registered from its shapes first and receives the real spec afterwards.
        initial = extraction.summary_spec() if extraction.externalized else spec
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
            structure=initial,
            project_id=project_id,
        )
        model_ref = str(created.get("id") or created.get("slug") or name)
        if extraction.externalized:
            self.upload_tensors(model_ref, extraction.tensors, progress=progress)
            if progress:
                progress("structure", 0, 1)
            self._put(f"/v1/models/{model_ref}/structure", spec)
            if progress:
                progress("structure", 1, 1)
        if X is not None:
            rows, _names = rows_from_inputs(X, names or None)
            labels = labels_from_target(y) if y is not None else None
            rows, labels = cap_reference_rows(rows, labels)
            self.upload_reference_data(model_ref, rows=rows, labels=labels)
        return created

    def structure(
        self,
        model: str,
        *,
        root: Optional[str] = None,
        depth: Optional[int] = None,
        detail: Optional[str] = None,
        include_values: Optional[bool] = None,
        max_inline: Optional[int] = None,
        tree: Optional[int] = None,
    ):
        """The architecture document. ``root`` / ``depth`` cut a huge model to one block."""
        params = {
            "root": root,
            "depth": depth,
            "detail": detail,
            "include_values": include_values,
            "max_inline": max_inline,
            "tree": tree,
        }
        return self._get(f"/v1/models/{model}/structure", compact(params))

    def upload_structure(
        self,
        model: str,
        estimator_or_spec: Any,
        *,
        feature_names: Optional[Sequence[str]] = None,
        task: Optional[str] = None,
        framework: Optional[str] = None,
        weights: str = "full",
        externalize_above: Optional[int] = EXTERNALIZE_ABOVE,
        progress: Optional[Progress] = None,
    ):
        """Attach layers and weights to an existing model.

        Pass a fitted estimator, an :class:`~kaimei.structure.Extraction`, or an
        already-extracted structure dict (whose ``$tensor`` references must already be uploaded).
        """
        if isinstance(estimator_or_spec, Extraction):
            extraction = estimator_or_spec
        elif isinstance(estimator_or_spec, Mapping):
            extraction = Extraction(spec=dict(estimator_or_spec))
        else:
            extraction = extract_structure(
                estimator_or_spec,
                feature_names=feature_names,
                task=task,
                framework=framework,
                weights=weights,
                externalize_above=externalize_above,
            )
        if extraction.externalized:
            self.upload_tensors(model, extraction.tensors, progress=progress)
        return self._put(f"/v1/models/{model}/structure", extraction.spec)

    def upload_tensors(
        self,
        model: str,
        tensors: Mapping[str, Any],
        *,
        dtype: str = "f32",
        progress: Optional[Progress] = None,
    ):
        """Upload weight tensors as compressed binary; the spec references them as ``{"$tensor": key}``."""
        items = list(tensors.items())
        results = []
        for index, (key, array) in enumerate(items):
            if progress:
                progress("tensors", index, len(items))
            shape = ",".join(str(int(dim)) for dim in getattr(array, "shape", ()) or (len(array),))
            results.append(
                self._client._request(
                    "PUT",
                    f"/v1/models/{model}/structure/tensors/{key}",
                    params={"shape": shape, "dtype": dtype, "compression": "zlib"},
                    content=encode_tensor(array, dtype),
                )
            )
        if progress:
            progress("tensors", len(items), len(items))
        return results

    def tensors(self, model: str):
        """Binary tensors uploaded for this model, with shapes, sizes and statistics."""
        return self._get(f"/v1/models/{model}/structure/tensors")

    def delete_tensor(self, model: str, key: str):
        return self._delete(f"/v1/models/{model}/structure/tensors/{key}")

    def delete_structure(self, model: str):
        return self._delete(f"/v1/models/{model}/structure")

    def trace_structure(
        self,
        model: str,
        *,
        input: Mapping[str, Any],
        detail: Optional[str] = None,
        nodes: Optional[Sequence[str]] = None,
        tree: Optional[int] = None,
    ):
        """Replay one prediction. ``detail="summary"`` returns per-node statistics for large models."""
        body = {"input": dict(input), "detail": detail, "nodes": list(nodes) if nodes is not None else None, "tree": tree}
        return self._post(f"/v1/models/{model}/structure/trace", compact(body))

    def structure_parameter(
        self,
        model: str,
        parameter_id: str,
        *,
        offset: int = 0,
        limit: int = 4096,
        stats: bool = False,
        preview: int = 0,
    ):
        return self._get(
            f"/v1/models/{model}/structure/parameters/{parameter_id}",
            compact({"offset": offset, "limit": limit, "stats": stats or None, "preview": preview or None}),
        )

    def structure_parameter_values(self, model: str, parameter_id: str, *, offset: int = 0, limit: int = 1_000_000):
        """A window of one tensor as a float32 numpy array, fetched as binary."""
        import numpy

        data = self._client._request(
            "GET",
            f"/v1/models/{model}/structure/parameters/{parameter_id}",
            params={"format": "f32", "offset": offset, "limit": limit},
            raw=True,
        )
        return numpy.frombuffer(data, dtype="<f4").astype(numpy.float32)
