"""Read a fitted estimator in the user's process, then upload the JSON spec.

The HTTP API never unpickles a checkpoint. A notebook that already holds the
trained object calls :func:`extract_structure` (or ``client.models.upload``)
and sends layers, weights, and biases: small tensors inline as JSON, large ones
as compressed binary uploads referenced from the spec as ``{"$tensor": key}``.
``weights="summary"`` sends shapes and statistics only, so the weights never
leave the user's environment and the dashboard still draws the architecture.

This module is the user-facing copy of ``kaimei_api.engine.structure.extract``.
Keep the conversion rules aligned with that file when you change either side.
"""

from __future__ import annotations

import json
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import KaimeiError

MAX_REFERENCE_ROWS = 5000
#: Image models: the API keeps reference rows × pixel values under this many values.
MAX_REFERENCE_VALUES = 8_000_000
WARN_STRUCTURE_BYTES = 2_000_000
LARGE_STRUCTURE_BYTES = 16_000_000
#: Tensors with more values than this leave the JSON spec and upload as binary blobs.
EXTERNALIZE_ABOVE = 65_536
#: With the default threshold, a spec still over ``WARN_STRUCTURE_BYTES`` lifts tensors above this too.
COMPACT_EXTERNALIZE_ABOVE = 4_096
#: Spec keys that hold learned or fitted tensors.
TENSOR_KEYS = ("weight", "bias", "gamma", "beta", "center", "scale", "mean", "var", "std", "offset")
EXECUTABLE_FAMILIES = {"network", "tree", "ensemble"}

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional until upload
    np = None  # type: ignore[assignment]


class StructureError(KaimeiError):
    """The estimator could not be turned into a Kaimei structure spec."""

    def __init__(self, message: str, *, code: str = "invalid_structure", param: Optional[str] = None) -> None:
        super().__init__(message, type="invalid_request_error", code=code, param=param)
        self.param = param


def _require_numpy() -> Any:
    if np is None:
        raise StructureError(
            "Uploading a trained model requires numpy. Install it in this environment: pip install numpy",
            code="numpy_required",
        )
    return np

_MLP_ACTIVATIONS = {"relu": "relu", "tanh": "tanh", "logistic": "sigmoid", "identity": None}
_OUT_ACTIVATIONS = {"logistic": "sigmoid", "softmax": "softmax", "identity": None}


def spec_is_executable(spec: Mapping[str, Any]) -> bool:
    """True when the API can score the uploaded spec (explain / predict / analyse)."""
    if spec.get("approximate"):
        return False
    return str(spec.get("family") or "") in EXECUTABLE_FAMILIES


def spec_payload_bytes(spec: Mapping[str, Any]) -> int:
    return len(json.dumps(spec, default=str).encode("utf-8"))


def warn_if_large_spec(spec: Mapping[str, Any]) -> int:
    size = spec_payload_bytes(spec)
    if size >= LARGE_STRUCTURE_BYTES:
        warnings.warn(
            f"This structure spec is {size / 1_000_000:.1f} MB of JSON. Reverse proxies may reject it. "
            "Lower externalize_above so more tensors upload as binary, or use weights='summary'.",
            UserWarning,
            stacklevel=3,
        )
    elif size >= WARN_STRUCTURE_BYTES:
        warnings.warn(
            f"This structure spec is {size / 1_000_000:.1f} MB of JSON. Tensors above externalize_above "
            "values upload as compressed binary instead; lower it to shrink the spec.",
            UserWarning,
            stacklevel=3,
        )
    return size


def tensor_stats(array: Any) -> Dict[str, Any]:
    """Summary statistics in the API's schema, for structure-only (``$shape``) uploads."""
    flat = np.asarray(array, dtype=float).ravel()
    if flat.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "l2": 0.0,
            "abs_mean": 0.0,
            "sparsity": 0.0,
            "numel": 0,
            "histogram": {"counts": [], "edges": []},
        }
    counts, edges = np.histogram(flat, bins=min(12, max(int(flat.size), 1)))
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "l2": float(np.linalg.norm(flat)),
        "abs_mean": float(np.mean(np.abs(flat))),
        "sparsity": float(np.mean(np.abs(flat) < 1e-8)),
        "numel": int(flat.size),
        "histogram": {"counts": [int(count) for count in counts], "edges": [float(edge) for edge in edges]},
    }


def encode_tensor(array: Any, dtype: str = "f32") -> bytes:
    """Little-endian, zlib-compressed bytes the tensor endpoint accepts."""
    kinds = {"f16": "<f2", "f32": "<f4", "f64": "<f8"}
    if dtype not in kinds:
        raise StructureError(f"Unsupported tensor dtype '{dtype}'. Use f16, f32 or f64.", code="invalid_tensor")
    return zlib.compress(np.ascontiguousarray(np.asarray(array), dtype=kinds[dtype]).tobytes(), 6)


@dataclass
class Extraction:
    """A JSON-ready spec plus the tensors that were lifted out of it for binary upload."""

    spec: Dict[str, Any]
    tensors: Dict[str, Any] = field(default_factory=dict)

    @property
    def externalized(self) -> bool:
        return bool(self.tensors)

    @property
    def tensor_bytes(self) -> int:
        return int(sum(int(getattr(array, "nbytes", 0)) for array in self.tensors.values()))

    def summary_spec(self) -> Dict[str, Any]:
        """The same architecture with every tensor as a ghost (shape and statistics, no values).

        This is what a large model is registered with before its tensors exist on the API.
        """
        return finalize_spec(self.spec, weights="summary", externalize_above=None, tensors=self.tensors).spec


def finalize_spec(
    spec: Mapping[str, Any],
    *,
    weights: str = "full",
    externalize_above: Optional[int] = EXTERNALIZE_ABOVE,
    tensors: Optional[Mapping[str, Any]] = None,
) -> Extraction:
    """Turn the arrays inside a raw spec into JSON lists, binary references, or ghosts.

    ``weights="summary"`` replaces every learned tensor with ``{"$shape", "stats"}`` so no
    values are sent; existing ``$tensor`` references are summarised from ``tensors`` when
    given. Otherwise tensors with more than ``externalize_above`` values become
    ``{"$tensor": key}`` references and are returned in :attr:`Extraction.tensors`;
    ``None`` keeps everything inline.
    """
    if weights not in {"full", "summary"}:
        raise StructureError("weights must be 'full' or 'summary'.", code="invalid_structure", param="weights")
    numpy = _require_numpy()
    out = dict(spec)
    lifted: Dict[str, Any] = {}
    layers = spec.get("layers")
    if isinstance(layers, list):
        finalized = []
        for raw in layers:
            if not isinstance(raw, Mapping):
                finalized.append(raw)
                continue
            layer = dict(raw)
            layer_id = str(layer.get("id") or layer.get("type") or "layer")
            for key in TENSOR_KEYS:
                value = layer.get(key)
                if value is None:
                    continue
                if isinstance(value, Mapping):
                    if weights == "summary" and "$tensor" in value:
                        known = (tensors or {}).get(str(value["$tensor"]))
                        shape = list(value.get("shape") or getattr(known, "shape", None) or ())
                        ghost: Dict[str, Any] = {"$shape": [int(dim) for dim in shape] or [1], "dtype": "f32"}
                        if known is not None:
                            ghost["stats"] = tensor_stats(known)
                        layer[key] = ghost
                    continue
                array = numpy.asarray(value, dtype=float)
                if weights == "summary" and key in {"weight", "bias", "gamma", "beta"}:
                    layer[key] = {
                        "$shape": [int(dim) for dim in array.shape] or [1],
                        "dtype": "f32",
                        "stats": tensor_stats(array),
                    }
                elif externalize_above is not None and array.size > externalize_above:
                    tensor_key = f"{layer_id}.{key}"
                    lifted[tensor_key] = numpy.ascontiguousarray(array, dtype=numpy.float32)
                    layer[key] = {"$tensor": tensor_key, "shape": [int(dim) for dim in array.shape], "dtype": "f32"}
                else:
                    layer[key] = array.tolist() if array.ndim else float(array)
            finalized.append(layer)
        out["layers"] = finalized
    if hasattr(out.get("importances"), "tolist"):
        out["importances"] = out["importances"].tolist()
    if weights == "summary":
        out["weights"] = "summary"
    else:
        out.pop("weights", None)
    return Extraction(spec=out, tensors=lifted)


def cap_reference_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: Optional[Sequence[Any]] = None,
    *,
    limit: int = MAX_REFERENCE_ROWS,
    values_per_row: int = 1,
) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
    """Keep at most ``limit`` rows; wide rows (images) also stay under the API's value budget."""
    if values_per_row > 1:
        limit = max(1, min(limit, MAX_REFERENCE_VALUES // values_per_row))
    capped = [dict(row) for row in rows]
    resolved_labels = list(labels) if labels is not None else None
    if len(capped) <= limit:
        return capped, resolved_labels
    warnings.warn(
        f"Reference data has {len(capped)} rows; Kaimei keeps the first {limit}. "
        "Pass a representative sample or raise KAIMEI_MAX_REFERENCE_ROWS "
        "(KAIMEI_MAX_REFERENCE_VALUES for images) on the API.",
        UserWarning,
        stacklevel=3,
    )
    if resolved_labels is not None:
        resolved_labels = resolved_labels[:limit]
    return capped[:limit], resolved_labels


_EXACT_FAMILIES = {"network", "tree", "ensemble"}
_WRAPPER_ATTRS = (
    "best_estimator_",
    "estimator_",
    "base_estimator_",
    "final_estimator_",
    "regressor_",
    "classifier_",
    "calibrated_classifiers_",
    "named_estimators_",
    "model_",
    "module_",
    "network_",
    "net_",
    "learner_",
    "booster_",
    "_model",
    "_estimator",
    "model",
    "module",
    "network",
    "net",
    "estimator",
    "base_estimator",
    "regressor",
    "classifier",
)
_SKIP_UNWRAP_ATTRS = {
    "tree_",
    "coef_",
    "coefs_",
    "intercept_",
    "intercepts_",
    "feature_importances_",
    "feature_names_in_",
    "classes_",
    "n_features_in_",
    "steps",
    "layers",
}
_MAX_UNWRAP = 24
_MAX_UNWRAP_DEPTH = 3


def estimator_to_spec(
    estimator: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    input_shape: Optional[Sequence[int]] = None,
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a ``kaimei.structure.v1`` document with every tensor inline as JSON lists.

    Reads a known fitted shape (linear / MLP / tree / forest / Keras / Torch). If this
    object is a wrapper around one of those, the inner estimator is used — no
    vendor-specific package names required. For large models prefer
    :func:`extract_structure`, which lifts big tensors out for binary upload.
    """
    return extract_structure(
        estimator,
        feature_names=feature_names,
        task=task,
        framework=framework,
        externalize_above=None,
        input_shape=input_shape,
        preprocess=preprocess,
    ).spec


def extract_structure(
    estimator: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
    weights: str = "full",
    externalize_above: Optional[int] = EXTERNALIZE_ABOVE,
    input_shape: Optional[Sequence[int]] = None,
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Extraction:
    """Read a fitted estimator into a spec plus the tensors to upload as binary.

    ``weights="summary"`` keeps the weights private: the spec carries shapes and statistics
    only, and the model is drawable but not executable on Kaimei.

    ``input_shape`` is the image size in the framework's own order: ``(channels, height,
    width)`` for PyTorch (required for convolutional modules), ``(height, width, channels)``
    for Keras (read from the model when built). ``preprocess`` prepends a per-channel
    ``normalize`` layer, e.g. ``{"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}``
    for a torchvision model that expects ``Normalize`` on ``[0, 1]`` pixels.
    """
    raw = _extract_raw(estimator, feature_names, task, framework, input_shape, preprocess)
    extraction = finalize_spec(raw, weights=weights, externalize_above=externalize_above)
    if externalize_above == EXTERNALIZE_ABOVE and spec_payload_bytes(extraction.spec) >= WARN_STRUCTURE_BYTES:
        # A CNN holds dozens of kernels under the default threshold (MobileNetV2: 16 MB of JSON).
        # Lift those as well, unless the caller chose a threshold.
        extraction = finalize_spec(raw, weights=weights, externalize_above=COMPACT_EXTERNALIZE_ABOVE)
    return extraction


def _extract_raw(
    estimator: Any,
    feature_names: Optional[Sequence[str]],
    task: Optional[str],
    framework: Optional[str],
    input_shape: Optional[Sequence[int]] = None,
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The spec with tensors still as numpy arrays (see :func:`finalize_spec`)."""
    _require_numpy()
    candidates = list(_iter_estimators(estimator))
    names: List[str]
    if feature_names is not None:
        names = [str(name) for name in feature_names]
    else:
        inferred: Optional[List[str]] = None
        for candidate in candidates:
            try:
                inferred = _infer_names(candidate)
                break
            except StructureError:
                continue
        if inferred is None:
            if any(_is_keras_model(item) or _is_torch_module(item) for item in candidates):
                inferred = []
            else:
                raise StructureError(
                    "Pass feature_names; this estimator does not say how many inputs it has.",
                    code="invalid_structure",
                    param="feature_names",
                )
        names = inferred
    best: Optional[Dict[str, Any]] = None
    best_rank = -1
    best_source: Any = estimator
    for candidate in candidates:
        spec = _direct_spec(candidate, names, task, framework, input_shape, preprocess)
        if spec is None:
            continue
        rank = 2 if spec.get("family") in _EXACT_FAMILIES else 1
        if rank > best_rank:
            best = spec
            best_rank = rank
            best_source = candidate
        if best_rank >= 2:
            break
    if best is None:
        raise StructureError(
            "Cannot read this estimator. Supported shapes: linear coefficients, MLP coefs_, "
            "a decision tree, a forest, a StandardScaler pipeline around those, a Keras / "
            "TensorFlow model, a torch.nn module, or an estimator with feature_importances_. "
            "Wrappers around any of those are also accepted.",
            code="unsupported_estimator",
        )
    if framework is None:
        extracted = detect_framework(best_source)
        if extracted != "custom":
            best["framework"] = extracted
        else:
            outer = detect_framework(estimator)
            if outer != "custom":
                best["framework"] = outer
    return best


def _direct_spec(
    estimator: Any,
    names: List[str],
    task: Optional[str],
    framework: Optional[str],
    input_shape: Optional[Sequence[int]] = None,
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert ``estimator`` if it already has a known fitted shape. No unwrapping."""
    if _is_pipeline(estimator):
        return _pipeline_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "coefs_") and hasattr(estimator, "intercepts_"):
        return _mlp_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "estimators_"):
        try:
            return _forest_to_spec(estimator, names, task, framework)
        except StructureError as exc:
            if exc.code != "unsupported_estimator":
                raise
    if hasattr(estimator, "tree_"):
        return _tree_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        return _linear_to_spec(estimator, names, task, framework)
    if _is_keras_model(estimator):
        try:
            return _keras_to_spec(estimator, names, task, framework, input_shape, preprocess)
        except StructureError as exc:
            if exc.code != "unsupported_estimator":
                raise
            return None
    if _is_torch_module(estimator):
        try:
            return _torch_to_spec(estimator, names, task, framework, input_shape, preprocess)
        except StructureError as exc:
            if exc.code != "unsupported_estimator":
                raise
            return None
    if hasattr(estimator, "feature_importances_"):
        return _importance_to_spec(estimator, names, task, framework)
    return None


def _iter_estimators(root: Any) -> List[Any]:
    """Breadth-first walk of conventional wrapper attributes and fitted children."""
    found: List[Any] = []
    queue: List[tuple] = [(root, 0)]
    seen = {id(root)}
    while queue and len(found) < _MAX_UNWRAP:
        current, depth = queue.pop(0)
        found.append(current)
        if depth >= _MAX_UNWRAP_DEPTH:
            continue
        for child in _child_estimators(current):
            marker = id(child)
            if marker in seen:
                continue
            seen.add(marker)
            queue.append((child, depth + 1))
    return found


def _child_estimators(estimator: Any) -> List[Any]:
    children: List[Any] = []
    seen: set = set()

    def _add(value: Any) -> None:
        for item in _expand_estimator(value):
            marker = id(item)
            if marker in seen or item is estimator:
                continue
            seen.add(marker)
            children.append(item)

    for name in _WRAPPER_ATTRS:
        if name in _SKIP_UNWRAP_ATTRS:
            continue
        _add(_safe_attr(estimator, name))
    values = getattr(estimator, "__dict__", None)
    if isinstance(values, dict):
        for name, value in values.items():
            if name in _SKIP_UNWRAP_ATTRS or name in _WRAPPER_ATTRS:
                continue
            if name.startswith("__"):
                continue
            if name.endswith("_") or name in {"model", "module", "network", "net", "estimator"}:
                _add(value)
    return children


def _expand_estimator(value: Any) -> List[Any]:
    if value is None or value is True or value is False:
        return []
    if isinstance(value, dict):
        return [item for item in value.values() if _is_estimator_candidate(item)]
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)) and item:
                item = item[-1] if _is_estimator_candidate(item[-1]) else item[0]
            if _is_estimator_candidate(item):
                out.append(item)
        return out
    return [value] if _is_estimator_candidate(value) else []


def _is_estimator_candidate(obj: Any) -> bool:
    if obj is None or isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, type)):
        return False
    module = str(getattr(type(obj), "__module__", "") or "")
    if module in {"builtins", "numpy", "pandas"} or module.startswith(("numpy.", "pandas.")):
        return False
    if np is not None and isinstance(obj, np.ndarray):
        return False
    return (
        _has_known_shape(obj)
        or any(_safe_attr(obj, name) is not None and name not in _SKIP_UNWRAP_ATTRS for name in _WRAPPER_ATTRS)
        or hasattr(obj, "predict")
        or _is_keras_model(obj)
        or _is_torch_module(obj)
        or _is_pipeline(obj)
    )


def _has_known_shape(obj: Any) -> bool:
    return (
        (hasattr(obj, "coefs_") and hasattr(obj, "intercepts_"))
        or hasattr(obj, "estimators_")
        or hasattr(obj, "tree_")
        or (hasattr(obj, "coef_") and hasattr(obj, "intercept_"))
        or hasattr(obj, "feature_importances_")
        or _is_keras_model(obj)
        or _is_torch_module(obj)
        or _is_pipeline(obj)
    )


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _infer_names(estimator: Any) -> List[str]:
    existing = getattr(estimator, "feature_names_in_", None)
    if existing is not None:
        return [str(name) for name in list(existing)]
    width = getattr(estimator, "n_features_in_", None)
    if width is None and hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        width = int(coef.shape[-1])
    if width is None and hasattr(estimator, "coefs_"):
        width = int(np.asarray(estimator.coefs_[0]).shape[0])
    if width is None:
        raise StructureError(
            "Pass feature_names; this estimator does not say how many inputs it has.",
            code="invalid_structure",
            param="feature_names",
        )
    return [f"x{i}" for i in range(int(width))]


def _is_pipeline(estimator: Any) -> bool:
    steps = getattr(estimator, "steps", None)
    return isinstance(steps, (list, tuple)) and bool(steps)


def _linear_to_spec(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    weight = np.asarray(estimator.coef_, dtype=float)
    if weight.ndim == 1:
        weight = weight.reshape(1, -1)
    bias = np.asarray(estimator.intercept_, dtype=float).ravel()
    classes = _classes(estimator)
    resolved_task, activation, labels = _supervised_head(classes, task, weight.shape[0])
    if weight.shape[1] != len(names):
        raise StructureError(
            f"coef_ expects {weight.shape[1]} features, got {len(names)} names.",
            code="invalid_structure",
            param="feature_names",
        )
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or "sklearn",
        "architecture": "linear",
        "task": resolved_task,
        "feature_names": names,
        "summary": "Linear layer read from coef_ and intercept_.",
        "layers": [
            {"id": "input", "type": "input", "names": names},
            {
                "id": "linear",
                "type": "dense",
                "units": int(weight.shape[0]),
                "activation": activation,
                "weight": weight,
                "bias": bias,
                "label": "Linear",
                "output_labels": labels,
            },
        ],
    }
    if labels:
        spec["labels"] = labels
        spec["outputs"] = {"labels": labels}
    return spec


def _mlp_to_spec(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Dict[str, Any]:
    coefs = [np.asarray(weight, dtype=float) for weight in estimator.coefs_]
    intercepts = [np.asarray(bias, dtype=float).ravel() for bias in estimator.intercepts_]
    if coefs[0].shape[0] != len(names):
        raise StructureError(
            f"First MLP layer expects {coefs[0].shape[0]} features, got {len(names)} names.",
            code="invalid_structure",
            param="feature_names",
        )
    hidden_name = str(getattr(estimator, "activation", "relu"))
    hidden = _MLP_ACTIVATIONS.get(hidden_name, hidden_name)
    output_activation = _OUT_ACTIVATIONS.get(str(getattr(estimator, "out_activation_", "identity")))
    classes = _classes(estimator)
    resolved = task
    if classes is not None and resolved is None:
        resolved = "binary_classification" if len(classes) == 2 else "multiclass_classification"
    resolved = resolved or "regression"
    layers: List[Dict[str, Any]] = [{"id": "input", "type": "input", "names": names}]
    for index, (weight_in, bias) in enumerate(zip(coefs, intercepts)):
        weight = weight_in.T  # sklearn stores (in, out)
        last = index == len(coefs) - 1
        activation = output_activation if last else hidden
        layer: Dict[str, Any] = {
            "id": f"dense_{index + 1}",
            "type": "dense",
            "units": int(weight.shape[0]),
            "weight": weight,
            "bias": bias,
            "label": "Output" if last else f"Hidden {index + 1}",
        }
        if activation:
            layer["activation"] = activation
        if last and classes is not None:
            layer["output_labels"] = classes
        layers.append(layer)
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or "sklearn",
        "architecture": "mlp",
        "task": resolved,
        "feature_names": names,
        "summary": "Multilayer perceptron read from coefs_ and intercepts_.",
        "layers": layers,
    }
    if classes is not None:
        spec["labels"] = classes
        spec["outputs"] = {"labels": classes}
    return spec


def _tree_to_spec(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], *, id_prefix: str = "n"
) -> Dict[str, Any]:
    tree = estimator.tree_
    classes = _classes(estimator)
    resolved = task
    if classes is not None and resolved is None:
        resolved = "binary_classification" if len(classes) == 2 else "multiclass_classification"
    resolved = resolved or "regression"
    nodes = []
    for index in range(int(tree.node_count)):
        feature = int(tree.feature[index])
        left = int(tree.children_left[index])
        right = int(tree.children_right[index])
        leaf = feature < 0 or left < 0
        raw = np.asarray(tree.value[index], dtype=float)
        while raw.ndim > 1:
            raw = raw[0]
        raw = np.asarray(raw, dtype=float).ravel()
        if classes is not None:
            total = float(raw.sum())
            output = (raw / total).tolist() if total else [1.0 / max(raw.size, 1)] * int(raw.size)
        else:
            output = [float(raw.reshape(-1)[0])]
        node: Dict[str, Any] = {"id": f"{id_prefix}{index}", "leaf": leaf, "output": output}
        if not leaf:
            node["feature"] = feature
            node["threshold"] = float(tree.threshold[index])
            node["left"] = f"{id_prefix}{left}"
            node["right"] = f"{id_prefix}{right}"
        nodes.append(node)
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "tree",
        "framework": framework or "sklearn",
        "architecture": "tree",
        "task": resolved,
        "feature_names": names,
        "root": f"{id_prefix}0",
        "nodes": nodes,
        "summary": f"Decision tree with {len(nodes)} nodes, read from tree_.",
    }
    if classes is not None:
        spec["classes"] = classes
        spec["outputs"] = {"labels": classes}
    return spec


def _forest_to_spec(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    members = []
    for member in estimator.estimators_:
        if hasattr(member, "tree_"):
            members.append(member)
        elif isinstance(member, (list, tuple)) and member and hasattr(member[0], "tree_"):
            members.append(member[0])
        else:
            raise StructureError("Forest member is not a decision tree.", code="unsupported_estimator")
    total = sum(int(member.tree_.node_count) for member in members)
    classes = _classes(estimator)
    resolved = task
    if classes is not None and resolved is None:
        resolved = "binary_classification" if len(classes) == 2 else "multiclass_classification"
    resolved = resolved or "regression"
    if len(members) <= 30 and total <= 4000:
        trees = []
        for index, member in enumerate(members):
            spec = _tree_to_spec(member, names, resolved, framework, id_prefix=f"t{index}_")
            trees.append({"root": spec["root"], "nodes": spec["nodes"]})
        return {
            "format": "kaimei.structure.v1",
            "family": "ensemble",
            "framework": framework or "sklearn",
            "architecture": "ensemble",
            "task": resolved,
            "feature_names": names,
            "classes": classes,
            "outputs": {"labels": classes} if classes else None,
            "trees": trees,
            "summary": f"Forest of {len(members)} trees.",
        }
    importances = np.asarray(getattr(estimator, "feature_importances_", []), dtype=float).ravel()
    return {
        "format": "kaimei.structure.v1",
        "family": "ensemble_summary",
        "framework": framework or "sklearn",
        "task": resolved,
        "feature_names": names,
        "n_estimators": len(members),
        "importances": importances.tolist(),
        "classes": classes,
    }


def _pipeline_to_spec(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    steps = list(estimator.steps)
    if len(steps) < 2:
        return _extract_raw(steps[-1][1], names, task, framework)
    *preprocessors, final = steps
    center = None
    scale = None
    for step_name, step in preprocessors:
        found = _scaler_arrays(step)
        if found is None:
            raise StructureError(
                f"Pipeline step '{step_name}' is not a supported scaler.",
                code="unsupported_estimator",
                param="steps",
            )
        center, scale = found
    assert center is not None and scale is not None
    final_estimator = final[1]
    spec = _extract_raw(final_estimator, names, task, framework)
    if spec.get("family") == "tree":
        _fold_scaler(spec["nodes"], center, scale)
        spec["summary"] = "Standardizer folded into the split thresholds, so the tree reads raw features."
        return spec
    if spec.get("family") not in {None, "network"}:
        raise StructureError(
            "A scaler can precede a linear model, an MLP, or a single tree.",
            code="unsupported_estimator",
        )
    layers = list(spec["layers"])
    scaler_layer = {
        "id": "standardize",
        "type": "standardize",
        "center": np.asarray(center, dtype=float),
        "scale": np.asarray(scale, dtype=float),
        "label": "Standardize",
    }
    # Insert after the input layer.
    insert_at = 1 if layers and layers[0].get("type") == "input" else 0
    layers.insert(insert_at, scaler_layer)
    spec["layers"] = layers
    spec["summary"] = "Pipeline: standardize, then the fitted estimator."
    return spec


def _scaler_arrays(step: Any) -> Optional[tuple]:
    if hasattr(step, "mean_") and hasattr(step, "scale_"):
        return np.asarray(step.mean_, dtype=float), np.asarray(step.scale_, dtype=float)
    if hasattr(step, "data_min_") and hasattr(step, "data_max_"):
        feature_range = tuple(getattr(step, "feature_range", (0, 1)))
        if feature_range != (0, 1):
            raise StructureError(
                "Only MinMaxScaler feature_range (0, 1) can be folded into the structure.",
                code="unsupported_estimator",
            )
        low = np.asarray(step.data_min_, dtype=float)
        span = np.asarray(step.data_max_, dtype=float) - low
        return low, np.where(np.abs(span) < 1e-12, 1.0, span)
    return None


def _fold_scaler(nodes: List[Dict[str, Any]], center: np.ndarray, scale: np.ndarray) -> None:
    for node in nodes:
        if node.get("leaf"):
            continue
        index = int(node["feature"])
        node["threshold"] = float(node["threshold"]) * float(scale[index]) + float(center[index])


def _classes(estimator: Any) -> Optional[List[str]]:
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return None
    return [str(label) for label in list(classes)]


def _supervised_head(classes: Optional[List[str]], task: Optional[str], n_rows: int) -> tuple:
    if classes is not None:
        if len(classes) == 2:
            return task or "binary_classification", "sigmoid", classes
        return task or "multiclass_classification", "softmax", classes
    if task == "binary_classification" and n_rows == 1:
        return task, "sigmoid", ["negative", "positive"]
    if task == "multiclass_classification":
        return task, "softmax", [f"class_{i}" for i in range(n_rows)]
    return task or "regression", None, None


def _is_torch_module(estimator: Any) -> bool:
    module = estimator.__class__.__module__
    return module.startswith("torch") or hasattr(estimator, "named_parameters") and hasattr(estimator, "forward")


class _Unsupported(Exception):
    """A layer or operation the structure spec cannot express exactly."""


_FUSED_TYPES = {"dense", "conv2d", "add", "batchnorm"}


def _out(layer: Dict[str, Any]) -> str:
    """The node that carries a layer's output: ``id__act`` when an activation is fused into it."""
    act = layer.get("activation")
    if act and layer.get("type") in _FUSED_TYPES and str(act).lower() not in ("linear", "identity", "none"):
        return f"{layer['id']}__{act}"
    return str(layer["id"])


_TORCH_ACTIVATIONS = {
    "ReLU": "relu",
    "ReLU6": "relu6",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "GELU": "gelu",
    "SiLU": "silu",
    "Softmax": "softmax",
    "LeakyReLU": "leaky_relu",
    "ELU": "elu",
    "Hardswish": "hardswish",
    "Hardsigmoid": "hardsigmoid",
}
_TORCH_FUNCTIONS = {
    "relu": "relu",
    "relu_": "relu",
    "relu6": "relu6",
    "sigmoid": "sigmoid",
    "sigmoid_": "sigmoid",
    "tanh": "tanh",
    "tanh_": "tanh",
    "gelu": "gelu",
    "silu": "silu",
    "silu_": "silu",
    "softmax": "softmax",
    "leaky_relu": "leaky_relu",
    "leaky_relu_": "leaky_relu",
    "elu": "elu",
    "elu_": "elu",
    "hardswish": "hardswish",
    "hardswish_": "hardswish",
    "hardsigmoid": "hardsigmoid",
    "hardsigmoid_": "hardsigmoid",
}
#: Calls that return their input unchanged at inference.
_TORCH_PASSTHROUGH = {
    "dropout",
    "dropout2d",
    "contiguous",
    "clone",
    "detach",
    "float",
    "to",
    "type_as",
    "alpha_dropout",
}
#: Calls that compute shapes or indices, never activations.
_TORCH_SHAPE_CALLS = {"size", "dim", "getattr", "numel"}
#: Arithmetic call names (operator, torch and Tensor-method spellings) → operation.
_ARITHMETIC = {
    "add": "add",
    "iadd": "add",
    "add_": "add",
    "radd": "add",
    "sub": "sub",
    "isub": "sub",
    "sub_": "sub",
    "mul": "mul",
    "imul": "mul",
    "mul_": "mul",
    "rmul": "mul",
    "truediv": "div",
    "itruediv": "div",
    "div": "div",
    "div_": "div",
}


def _arr(value: Any) -> Any:
    """A tensor or array as a float numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _two(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        items = [int(item) for item in value]
        return items * 2 if len(items) == 1 else items[:2]
    return [int(value), int(value)]


def _hwc(shape: Optional[Sequence[int]]) -> Optional[List[int]]:
    """A torch ``(C, H, W)`` / ``(F,)`` shape as Kaimei's ``[H, W, C]`` / ``[F]``."""
    if shape is None:
        return None
    dims = [int(dim) for dim in shape]
    if len(dims) == 3:
        return [dims[1], dims[2], dims[0]]
    return dims


def _torch_module_layers(name: str, module: Any, in_shape: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    """One leaf module as spec layers (``[]`` when it is the identity at inference).

    ``in_shape`` is the torch shape of the module's input without the batch axis, when known.
    Raises :class:`_Unsupported` for anything the spec cannot reproduce exactly.
    """
    cls = type(module).__name__
    base: Dict[str, Any] = {"id": _safe_id(name), "path": name, "label": name}
    rank = len(in_shape) if in_shape is not None else None
    if cls == "Linear":
        if rank is not None and rank != 1:
            raise _Unsupported(f"{cls} on a {rank + 1}-D tensor")
        weight = _arr(module.weight)
        bias = _arr(module.bias).ravel() if module.bias is not None else np.zeros(weight.shape[0])
        return [{**base, "type": "dense", "units": int(weight.shape[0]), "weight": weight, "bias": bias}]
    if cls == "Conv2d":
        if getattr(module, "padding_mode", "zeros") != "zeros":
            raise _Unsupported(f"Conv2d padding_mode='{module.padding_mode}'")
        weight = _arr(module.weight)
        padding = module.padding if isinstance(module.padding, str) else _two(module.padding)
        return [
            {
                **base,
                "type": "conv2d",
                "weight": weight,
                "bias": _arr(module.bias).ravel() if module.bias is not None else np.zeros(weight.shape[0]),
                "stride": _two(module.stride),
                "padding": padding,
                "dilation": _two(module.dilation),
                "groups": int(module.groups),
            }
        ]
    if cls in ("BatchNorm2d", "BatchNorm1d"):
        if module.running_mean is None or module.running_var is None:
            raise _Unsupported(f"{cls} without running statistics")
        if rank is not None and rank != (3 if cls == "BatchNorm2d" else 1):
            raise _Unsupported(f"{cls} on a {rank + 1}-D tensor")
        layer = {
            **base,
            "type": "batchnorm",
            "mean": _arr(module.running_mean),
            "var": _arr(module.running_var),
            "eps": float(module.eps),
        }
        if module.affine:
            layer["gamma"] = _arr(module.weight)
            layer["beta"] = _arr(module.bias)
        return [layer]
    if cls in ("MaxPool2d", "AvgPool2d"):
        if cls == "MaxPool2d" and _two(module.dilation) != [1, 1]:
            raise _Unsupported("MaxPool2d with dilation")
        if cls == "AvgPool2d" and getattr(module, "divisor_override", None):
            raise _Unsupported("AvgPool2d divisor_override")
        kernel = _two(module.kernel_size)
        layer = {
            **base,
            "type": "maxpool" if cls == "MaxPool2d" else "avgpool",
            "kernel": kernel,
            "stride": _two(module.stride) if module.stride is not None else kernel,
            "padding": _two(module.padding),
            "ceil_mode": bool(module.ceil_mode),
        }
        if cls == "AvgPool2d":
            layer["count_include_pad"] = bool(module.count_include_pad)
        return [layer]
    if cls in ("AdaptiveAvgPool2d", "AdaptiveMaxPool2d"):
        return [
            {
                **base,
                "type": "adaptive_avgpool" if cls == "AdaptiveAvgPool2d" else "adaptive_maxpool",
                "output_size": _adaptive_size(module.output_size, in_shape),
            }
        ]
    if cls == "Flatten":
        if int(module.start_dim) != 1 or int(module.end_dim) != -1:
            raise _Unsupported("Flatten that keeps more than the batch axis")
        if rank == 3:
            return [{**base, "type": "flatten", "order": "chw"}]
        return []
    if cls in _TORCH_ACTIVATIONS:
        act = _TORCH_ACTIVATIONS[cls]
        if cls == "Softmax" and (rank not in (None, 1) or getattr(module, "dim", -1) not in (None, 1, -1)):
            raise _Unsupported("Softmax over a spatial tensor")
        layer = {**base, "type": "activation", "activation": act}
        if cls == "LeakyReLU":
            layer["alpha"] = float(module.negative_slope)
        if cls == "ELU":
            layer["alpha"] = float(module.alpha)
        return [layer]
    if cls in ("Dropout", "Dropout2d", "AlphaDropout"):
        return [{**base, "type": "dropout", "rate": float(getattr(module, "p", 0.0))}]
    if cls == "Identity":
        return []
    if cls == "ZeroPad2d":
        left, right, top, bottom = (int(item) for item in module.padding)
        return [{**base, "type": "pad2d", "padding": [top, bottom, left, right]}]
    if cls == "LayerNorm":
        if rank not in (None, 1):
            raise _Unsupported("LayerNorm over a spatial tensor")
        layer = {**base, "type": "layernorm", "eps": float(module.eps)}
        if module.elementwise_affine:
            layer["gamma"] = _arr(module.weight)
            layer["beta"] = _arr(module.bias)
        return [layer]
    if cls == "Embedding":
        return [{**base, "type": "embedding", "weight": _arr(module.weight)}]
    raise _Unsupported(cls)


def _adaptive_size(output_size: Any, in_shape: Optional[Sequence[int]]) -> List[int]:
    sizes = list(output_size) if isinstance(output_size, (list, tuple)) else [output_size, output_size]
    if len(sizes) == 1:
        sizes = sizes * 2
    resolved = []
    for axis, size in enumerate(sizes[:2]):
        if size is None:
            if in_shape is None or len(in_shape) != 3:
                raise _Unsupported("adaptive pooling with an open output size")
            size = in_shape[1 + axis]
        resolved.append(int(size))
    return resolved


def _torch_image_shape(estimator: Any, input_shape: Optional[Sequence[int]]) -> Optional[List[int]]:
    """``(C, H, W)`` for a convolutional module, ``None`` for a vector model."""
    if input_shape is not None:
        dims = [int(dim) for dim in input_shape]
        if len(dims) == 4:
            dims = dims[1:]
        if len(dims) == 3:
            return dims
        if len(dims) == 1:
            return None
        raise StructureError(
            "input_shape must be (channels, height, width) for a PyTorch image model.",
            code="invalid_structure",
            param="input_shape",
        )
    for _name, module in estimator.named_modules():
        if type(module).__name__ in ("Conv2d", "BatchNorm2d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"):
            raise StructureError(
                "This PyTorch model is convolutional. Pass input_shape=(channels, height, width) so Kaimei "
                "knows the image size, e.g. input_shape=(3, 224, 224).",
                code="invalid_structure",
                param="input_shape",
            )
    return None


def _torch_to_spec(
    estimator: Any,
    names: List[str],
    task: Optional[str],
    framework: Optional[str],
    input_shape: Optional[Sequence[int]] = None,
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise StructureError("torch is not installed.", code="unsupported_estimator") from exc

    image = _torch_image_shape(estimator, input_shape)
    if image is not None:
        return _torch_image_spec(estimator, names, task, framework, image, preprocess, torch)
    resolved = task or "regression"
    leaves = [
        (name, module)
        for name, module in estimator.named_modules()
        if module is not estimator and not list(module.children())
    ]
    if not leaves:
        raise StructureError("Torch module has no leaf layers.", code="unsupported_estimator")
    example = None
    if names:
        example = torch.zeros(1, len(names))
    seen: List[str] = []
    handles = []

    def _hook(name: str):
        def _record(_module: Any, _inputs: Any, _output: Any) -> None:
            seen.append(name)

        return _record

    for name, module in leaves:
        handles.append(module.register_forward_hook(_hook(name)))
    trainable = getattr(estimator, "eval", lambda: estimator)()
    try:
        if example is not None:
            with torch.no_grad():
                trainable(example)
    except Exception:
        seen = [name for name, _module in leaves]
    finally:
        for handle in handles:
            handle.remove()
    if not seen:
        seen = [name for name, _module in leaves]
    modules = dict(leaves)
    layers: List[Dict[str, Any]] = [{"id": "input", "type": "input", "names": names or [f"x{i}" for i in range(1)]}]
    unsupported = False
    for name in seen:
        module = modules.get(name)
        if module is None:
            continue
        try:
            layers.extend(_torch_module_layers(name, module, None))
        except _Unsupported:
            unsupported = True
            layers.append(
                {
                    "id": _safe_id(name),
                    "path": name,
                    "type": "dropout",
                    "label": type(module).__name__,
                    "rate": 0,
                }
            )
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or "pytorch",
        "architecture": "mlp" if not unsupported else "network",
        "task": resolved,
        "feature_names": names,
        "summary": "PyTorch module read in forward-hook order. Module names group the layers into blocks."
        if not unsupported
        else "PyTorch module read in forward-hook order. Unrecognised layers are passed through.",
        "layers": layers,
    }
    if unsupported:
        spec["approximate"] = True
    return spec


def _torch_image_spec(
    estimator: Any,
    names: List[str],
    task: Optional[str],
    framework: Optional[str],
    image: List[int],
    preprocess: Optional[Mapping[str, Any]],
    torch: Any,
) -> Dict[str, Any]:
    channels, height, width = image
    name = names[0] if names else "image"
    estimator = getattr(estimator, "eval", lambda: estimator)()
    example = torch.zeros(1, channels, height, width)
    layers: List[Dict[str, Any]] = [
        {"id": "input", "type": "input", "names": [name], "shape": [height, width, channels], "label": "Image"}
    ]
    entry = _preprocess_layer(layers, preprocess, "input")
    how = "traced"
    try:
        body, approximate, final_shape = _torch_fx_layers(estimator, example, entry, torch)
    except _Untraceable:
        body, approximate, final_shape = _torch_hook_layers(estimator, example, entry, torch)
        how = "hooks"
    layers.extend(body)
    resolved, labels, added = _classification_head(layers, task, final_shape)
    summary = (
        "PyTorch module traced op by op; residual adds, concatenations and pooling are kept."
        if how == "traced"
        else "PyTorch module read in forward-hook order (it could not be traced), so functional "
        "operations such as residual adds are not visible; the graph is drawn but not executed."
    )
    if added:
        summary += f" A {added} turns the final logits into probabilities."
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or "pytorch",
        "architecture": "cnn",
        "task": resolved,
        "feature_names": [name],
        "summary": summary,
        "layers": layers,
    }
    if labels:
        spec["labels"] = labels
        spec["outputs"] = {"labels": labels}
    if approximate or how == "hooks":
        spec["approximate"] = True
    return spec


class _Untraceable(Exception):
    """``torch.fx`` could not trace the module."""


def _torch_fx_layers(estimator: Any, example: Any, entry: str, torch: Any) -> tuple:
    """Layers from a ``torch.fx`` trace: every module call and functional op, in graph order."""
    try:
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp

        traced = symbolic_trace(estimator)
        with torch.no_grad():
            ShapeProp(traced).propagate(example)
    except Exception as exc:
        raise _Untraceable(str(exc)) from exc
    modules = dict(traced.named_modules())
    produced: Dict[str, str] = {}
    shapes: Dict[str, Optional[List[int]]] = {}
    paths: Dict[str, str] = {}
    layers: List[Dict[str, Any]] = []
    taken: set = {"input", entry}
    approximate = False
    problems: List[str] = []
    failed: set = set()
    final: Optional[str] = None
    final_shape: Optional[List[int]] = None

    def tensor_shape(node: Any) -> Optional[List[int]]:
        meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
        shape = getattr(meta, "shape", None)
        return [int(dim) for dim in shape][1:] if shape is not None else None

    def unique(raw: str) -> str:
        base = _safe_id(raw)
        candidate, index = base, 1
        while candidate in taken:
            index += 1
            candidate = f"{base}_{index}"
        taken.add(candidate)
        return candidate

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            if "placeholder" in paths:
                raise StructureError(
                    "PyTorch models with more than one input are not supported.", code="invalid_structure"
                )
            paths["placeholder"] = node.name
            produced[node.name] = entry
            shapes[node.name] = tensor_shape(node)
            continue
        if node.op == "output":
            result = node.args[0]
            if isinstance(result, (list, tuple)):
                result = result[0]
            if hasattr(result, "name") and result.name in failed:
                continue  # the unsupported op upstream is what gets reported
            if isinstance(result, dict) or not hasattr(result, "name") or result.name not in produced:
                raise StructureError("The model must return one tensor.", code="invalid_structure")
            final = produced[result.name]
            final_shape = shapes.get(result.name)
            continue
        if any(arg.name in failed for arg in _fx_args(node)):
            failed.add(node.name)  # downstream of an op already reported
            continue
        out_shape = tensor_shape(node)
        inputs = [arg for arg in _fx_args(node) if hasattr(arg, "name") and arg.name in produced]
        if out_shape is None:
            # sizes, shapes and other bookkeeping values; only tensor nodes become layers
            continue
        in_shapes = [shapes.get(arg.name) for arg in inputs]
        sources = [produced[arg.name] for arg in inputs]
        if node.op == "call_module":
            path = str(node.target)
        else:
            parent = paths.get(inputs[0].name, "") if inputs else ""
            path = f"{parent.rsplit('.', 1)[0]}.{node.name}" if "." in parent else node.name
        try:
            if node.op == "call_module":
                converted = _torch_module_layers(path, modules[node.target], in_shapes[0] if in_shapes else None)
            elif node.op in ("call_function", "call_method"):
                converted = _torch_call_layers(node, in_shapes, out_shape)
            else:
                raise _Unsupported(f"{node.op} {node.target}")
        except _Unsupported as exc:
            if len(sources) == 1 and in_shapes[0] == out_shape:
                approximate = True
                converted = [{"type": "dropout", "rate": 0, "label": f"{exc} (not replayed)"}]
            else:
                problems.append(f"{exc} at {path}")
                failed.add(node.name)
                continue
        if not converted:
            if not sources:
                problems.append(f"{node.target} at {path}")
                failed.add(node.name)
                continue
            produced[node.name] = sources[0]
            shapes[node.name] = out_shape
            paths[node.name] = paths.get(inputs[0].name, path)
            continue
        previous = sources
        for index, layer in enumerate(converted):
            layer = dict(layer)
            layer["id"] = unique(layer.get("id") or (node.name if index == 0 else f"{node.name}_{index}"))
            layer.setdefault("path", path)
            layer.setdefault("label", path)
            layer["inputs"] = list(previous)
            layers.append(layer)
            previous = [_out(layer)]
        produced[node.name] = previous[0]
        shapes[node.name] = out_shape
        paths[node.name] = path
    if problems:
        listed = "; ".join(problems[:6]) + ("; …" if len(problems) > 6 else "")
        raise StructureError(
            "This model uses operations Kaimei cannot replay yet: "
            f"{listed}. Register it with a remote inference endpoint instead; image explanations then use "
            "occlusion and LIME.",
            code="unsupported_layers",
        )
    if final is None:
        raise StructureError("The traced model has no output.", code="invalid_structure")
    if not layers or _out(layers[-1]) != final:
        layers.append({"id": unique("output"), "type": "dropout", "rate": 0, "label": "Output", "inputs": [final]})
    return layers, approximate, _hwc(final_shape)


def _fx_args(node: Any) -> List[Any]:
    found: List[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif hasattr(value, "op") and hasattr(value, "name") and hasattr(value, "meta"):
            found.append(value)

    walk(node.args)
    walk(node.kwargs)
    return found


def _call_name(node: Any) -> str:
    target = node.target
    name = target if isinstance(target, str) else getattr(target, "__name__", str(target))
    return name.strip("_") if name.startswith("__") else name


def _arg(node: Any, index: int, key: str, default: Any = None) -> Any:
    if key in node.kwargs:
        return node.kwargs[key]
    if len(node.args) > index:
        return node.args[index]
    return default


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _torch_call_layers(node: Any, in_shapes: List[Optional[List[int]]], out_shape: List[int]) -> List[Dict[str, Any]]:
    """A ``call_function`` / ``call_method`` node as spec layers (``[]`` when it is the identity)."""
    name = _call_name(node)
    rank = len(in_shapes[0]) if in_shapes and in_shapes[0] is not None else None
    tensors = len(in_shapes)
    if name in _TORCH_PASSTHROUGH:
        return []
    if name == "stochastic_depth" and not _arg(node, 3, "training", False):
        return []  # torchvision StochasticDepth is the identity at inference
    if name in _TORCH_FUNCTIONS:
        act = _TORCH_FUNCTIONS[name]
        layer: Dict[str, Any] = {"type": "activation", "activation": act}
        if act == "softmax" and (rank != 1 or _arg(node, 1, "dim", -1) not in (1, -1)):
            raise _Unsupported("softmax over a spatial tensor")
        if act == "leaky_relu":
            layer["alpha"] = float(_arg(node, 1, "negative_slope", 0.01))
        if act == "elu":
            layer["alpha"] = float(_arg(node, 1, "alpha", 1.0))
        return [layer]
    if name in ("clamp", "clamp_", "hardtanh", "hardtanh_"):
        low = _number(_arg(node, 1, "min", _arg(node, 1, "min_val", None)))
        high = _number(_arg(node, 2, "max", _arg(node, 2, "max_val", None)))
        if low == 0.0 and high is None:
            return [{"type": "activation", "activation": "relu"}]
        if low == 0.0 and high == 6.0:
            return [{"type": "activation", "activation": "relu6"}]
        raise _Unsupported(f"{name}({low}, {high})")
    if name in _ARITHMETIC:
        return _torch_arithmetic(node, name, in_shapes, out_shape)
    if name in ("cat", "concat"):
        dim = int(_arg(node, 1, "dim", 0))
        if rank is None or dim not in (1, -rank):
            raise _Unsupported(f"cat along dim {dim}")
        return [{"type": "concat"}] if tensors > 1 else []
    if name in ("flatten", "view", "reshape", "squeeze", "unsqueeze"):
        if name == "flatten" and int(_arg(node, 1, "start_dim", 0)) != 1:
            raise _Unsupported("flatten that merges the batch axis")
        in_shape = in_shapes[0] if in_shapes else None
        if in_shape is None or int(np.prod(out_shape)) != int(np.prod(in_shape)):
            raise _Unsupported(f"{name} to {tuple(out_shape)}")
        if list(out_shape) == list(in_shape):
            return []
        if len(out_shape) == 1:
            # (C, H, W) → (C·H·W) reads channels first; (C, 1, 1) → (C) keeps the order either way.
            if rank == 3 and int(in_shape[1]) * int(in_shape[2]) != 1:
                return [{"type": "flatten", "order": "chw"}]
            return []
        if rank == 1 and len(out_shape) == 3 and int(out_shape[1]) * int(out_shape[2]) == 1:
            return []  # (C) → (C, 1, 1): a per-channel gate; multiply broadcasts it
        raise _Unsupported(f"{name} to {tuple(out_shape)}")
    if name in ("expand_as", "expand") and len(in_shapes) >= 1:
        return []  # broadcasting is implicit in multiply
    if name in ("adaptive_avg_pool2d", "adaptive_max_pool2d"):
        return [
            {
                "type": "adaptive_avgpool" if "avg" in name else "adaptive_maxpool",
                "output_size": _adaptive_size(_arg(node, 1, "output_size", 1), in_shapes[0] if in_shapes else None),
            }
        ]
    if name in ("max_pool2d", "avg_pool2d"):
        kernel = _two(_arg(node, 1, "kernel_size"))
        stride = _arg(node, 2, "stride", None)
        layer = {
            "type": "maxpool" if name == "max_pool2d" else "avgpool",
            "kernel": kernel,
            "stride": _two(stride) if stride not in (None, [], ()) else kernel,
            "padding": _two(_arg(node, 3, "padding", 0)),
        }
        if name == "max_pool2d":
            if _two(_arg(node, 4, "dilation", 1)) != [1, 1]:
                raise _Unsupported("max_pool2d with dilation")
            layer["ceil_mode"] = bool(_arg(node, 5, "ceil_mode", False))
        else:
            layer["ceil_mode"] = bool(_arg(node, 4, "ceil_mode", False))
            layer["count_include_pad"] = bool(_arg(node, 5, "count_include_pad", True))
            if _arg(node, 6, "divisor_override", None):
                raise _Unsupported("avg_pool2d divisor_override")
        return [layer]
    if name == "mean" and rank == 3:
        dims = _arg(node, 1, "dim", None)
        dims = [dims] if isinstance(dims, int) else list(dims or [])
        if sorted(int(dim) % 4 for dim in dims) == [2, 3]:
            return [{"type": "adaptive_avgpool", "output_size": [1, 1]}]
        raise _Unsupported(f"mean over dims {dims}")
    if name == "pad" and rank == 3:
        pad = list(_arg(node, 1, "pad", []))
        mode = _arg(node, 2, "mode", "constant")
        if len(pad) != 4 or mode != "constant":
            raise _Unsupported(f"pad {pad} mode={mode}")
        left, right, top, bottom = (int(item) for item in pad)
        return [
            {"type": "pad2d", "padding": [top, bottom, left, right], "value": float(_arg(node, 3, "value", 0.0) or 0.0)}
        ]
    if tensors == 0 or name in _TORCH_SHAPE_CALLS:
        raise _Unsupported(name)
    raise _Unsupported(name)


def _torch_arithmetic(
    node: Any, name: str, in_shapes: List[Optional[List[int]]], out_shape: List[int]
) -> List[Dict[str, Any]]:
    """Tensor ± tensor (residual), tensor × tensor (gating), or tensor with a constant (rescaling)."""
    operands = list(node.args[:2])
    tensors = [operand for operand in operands if hasattr(operand, "meta")]
    kind = _ARITHMETIC[name]
    if len(tensors) == 2:
        if node.kwargs.get("alpha", 1) != 1:
            raise _Unsupported(f"{name} with alpha")
        if kind == "add":
            if in_shapes[0] != in_shapes[1]:
                raise _Unsupported("add with broadcasting")
            return [{"type": "add"}]
        if kind == "mul":
            return [{"type": "multiply"}]
        raise _Unsupported(f"{kind} of two tensors")
    constant = next((_number(operand) for operand in operands if not hasattr(operand, "meta")), None)
    if len(tensors) != 1 or constant is None:
        raise _Unsupported(name)
    first_is_tensor = hasattr(operands[0], "meta")
    if kind == "add":
        return [{"type": "normalize", "offset": constant, "label": f"+ {constant:g}"}]
    if kind == "sub" and first_is_tensor:
        return [{"type": "normalize", "offset": -constant, "label": f"− {constant:g}"}]
    if kind == "mul":
        return [{"type": "normalize", "scale": constant, "label": f"× {constant:g}"}]
    if kind == "div" and first_is_tensor and constant != 0:
        return [{"type": "normalize", "scale": 1.0 / constant, "label": f"÷ {constant:g}"}]
    raise _Unsupported(f"{name} with a constant")


def _torch_hook_layers(estimator: Any, example: Any, entry: str, torch: Any) -> tuple:
    """Fallback for untraceable modules: leaf modules in call order, shapes from hooks."""
    calls: List[tuple] = []
    handles = []
    leaves = [
        (name, module)
        for name, module in estimator.named_modules()
        if module is not estimator and not list(module.children())
    ]

    def hook(name: str):
        def record(module: Any, inputs: Any, output: Any) -> None:
            first = inputs[0] if isinstance(inputs, (list, tuple)) and inputs else inputs
            in_shape = list(first.shape)[1:] if hasattr(first, "shape") else None
            out_shape = list(output.shape)[1:] if hasattr(output, "shape") else None
            calls.append((name, module, in_shape, out_shape))

        return record

    for name, module in leaves:
        handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            estimator(example)
    finally:
        for handle in handles:
            handle.remove()
    layers: List[Dict[str, Any]] = []
    previous = entry
    current: Optional[List[int]] = list(example.shape)[1:]
    for name, module, in_shape, out_shape in calls:
        if in_shape is not None and current is not None and len(current) == 3 and len(in_shape) == 1:
            layers.append({"id": f"{_safe_id(name)}__flatten", "type": "flatten", "order": "chw", "inputs": [previous]})
            previous = layers[-1]["id"]
        try:
            converted = _torch_module_layers(name, module, in_shape)
        except _Unsupported:
            converted = [
                {"id": _safe_id(name), "path": name, "type": "dropout", "rate": 0, "label": type(module).__name__}
            ]
        for layer in converted:
            layer = dict(layer)
            layer["inputs"] = [previous]
            layers.append(layer)
            previous = _out(layer)
        current = out_shape
    return layers, True, _hwc(current)


def _preprocess_layer(layers: List[Dict[str, Any]], preprocess: Optional[Mapping[str, Any]], entry: str) -> str:
    """Prepend a ``normalize`` layer (torchvision ``Normalize`` / Keras ``Rescaling``) when asked."""
    if not preprocess:
        return entry
    layer: Dict[str, Any] = {"id": "preprocess", "type": "normalize", "label": "Preprocess", "inputs": [entry]}
    for key in ("mean", "std", "scale", "offset"):
        if preprocess.get(key) is not None:
            value = preprocess[key]
            layer[key] = np.asarray(value, dtype=float) if isinstance(value, (list, tuple)) else float(value)
    unknown = set(preprocess) - {"mean", "std", "scale", "offset"}
    if unknown:
        raise StructureError(
            f"preprocess takes mean, std, scale and offset; got {sorted(unknown)}.",
            code="invalid_structure",
            param="preprocess",
        )
    layers.append(layer)
    return "preprocess"


def _classification_head(layers: List[Dict[str, Any]], task: Optional[str], final_shape: Optional[List[int]]) -> tuple:
    """Resolve the task and, for classifiers that end in logits, append the softmax / sigmoid."""
    last = layers[-1] if layers else {}
    act = last.get("activation") if last.get("type") in ("dense", "conv2d", "activation") else None
    width = int(np.prod(final_shape)) if final_shape else int(last.get("units") or 0)
    resolved = task
    if resolved is None:
        if act == "softmax" and width > 1:
            resolved = "multiclass_classification"
        elif act == "sigmoid" and width <= 1:
            resolved = "binary_classification"
        else:
            resolved = "regression"
    labels = None
    added = None
    if resolved == "multiclass_classification":
        labels = [f"class_{i}" for i in range(width)] if width else None
        if act != "softmax":
            layers.append(
                {
                    "id": "softmax",
                    "type": "activation",
                    "activation": "softmax",
                    "label": "softmax",
                    "inputs": [_out(last)],
                }
            )
            added = "softmax"
    elif resolved == "binary_classification":
        if width == 1 and act != "sigmoid":
            layers.append(
                {
                    "id": "sigmoid",
                    "type": "activation",
                    "activation": "sigmoid",
                    "label": "sigmoid",
                    "inputs": [_out(last)],
                }
            )
            added = "sigmoid"
        elif width == 2 and act != "softmax":
            layers.append(
                {
                    "id": "softmax",
                    "type": "activation",
                    "activation": "softmax",
                    "label": "softmax",
                    "inputs": [_out(last)],
                }
            )
            added = "softmax"
    return resolved, labels, added


def _safe_id(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "_.-" else "_" for char in name)
    return cleaned or "layer"


def detect_framework(estimator: Any) -> str:
    """Label from this class or a core-framework base class, not from vendor package names."""
    for cls in getattr(type(estimator), "__mro__", (type(estimator),)):
        found = _framework_from_module(str(getattr(cls, "__module__", "") or ""))
        if found:
            return found
    if _is_keras_model(estimator):
        return "keras"
    if _is_torch_module(estimator):
        return "pytorch"
    return "custom"


def _framework_from_module(module: str) -> Optional[str]:
    if module.startswith("sklearn") or ".sklearn." in module:
        return "sklearn"
    if module.startswith("torch") or ".torch." in module:
        return "pytorch"
    if module.startswith("xgboost") or "xgboost" in module:
        return "xgboost"
    if module.startswith("lightgbm") or "lightgbm" in module:
        return "lightgbm"
    if module.startswith("catboost") or "catboost" in module:
        return "catboost"
    if module.startswith("keras") or ".keras" in module:
        return "keras"
    if "tensorflow" in module:
        return "tensorflow"
    return None


def features_from_spec(spec: Dict[str, Any], value_range: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    """Build a model feature schema from an extracted structure document.

    ``value_range`` (e.g. ``[0, 255]``) is the pixel units an image model reads; Kaimei's
    default is ``[0, 1]``.
    """
    names = [str(name) for name in spec.get("feature_names") or []]
    shape = image_shape_of(spec)
    if shape and names:
        feature: Dict[str, Any] = {
            "name": names[0],
            "type": "image",
            "height": int(shape[0]),
            "width": int(shape[1]),
            "channels": int(shape[2]),
        }
        if value_range is not None:
            feature["value_range"] = [float(value_range[0]), float(value_range[1])]
        return [feature]
    return [{"name": name, "type": "numeric"} for name in names]


def image_shape_of(spec: Mapping[str, Any]) -> Optional[List[int]]:
    """``[height, width, channels]`` when the spec's input is an image."""
    layers = list(spec.get("layers") or [])
    first = layers[0] if layers else {}
    shape = first.get("shape") if str(first.get("type") or "") == "input" else None
    if shape and len(shape) == 3:
        return [int(dim) for dim in shape]
    return None


def _importance_to_spec(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    importances = np.asarray(getattr(estimator, "feature_importances_", []), dtype=float).ravel()
    if importances.size and importances.size != len(names):
        raise StructureError(
            f"feature_importances_ has {importances.size} entries, got {len(names)} names.",
            code="invalid_structure",
            param="feature_names",
        )
    classes = _classes(estimator)
    resolved = task
    if classes is not None and resolved is None:
        resolved = "binary_classification" if len(classes) == 2 else "multiclass_classification"
    resolved = resolved or "regression"
    return {
        "format": "kaimei.structure.v1",
        "family": "ensemble_summary",
        "framework": framework or detect_framework(estimator) or "custom",
        "task": resolved,
        "feature_names": names,
        "n_estimators": int(getattr(estimator, "n_estimators", 0) or 0),
        "importances": importances.tolist(),
        "classes": classes,
        "outputs": {"labels": classes} if classes else None,
        "summary": "Feature importances read from the fitted estimator. Individual trees were not inlined.",
    }


_KERAS_LAYER_TYPES = {
    "Activation",
    "Add",
    "AveragePooling2D",
    "BatchNormalization",
    "Concatenate",
    "Conv2D",
    "Dense",
    "DepthwiseConv2D",
    "Dropout",
    "ELU",
    "Flatten",
    "GELU",
    "GlobalAveragePooling2D",
    "GlobalMaxPooling2D",
    "InputLayer",
    "LayerNormalization",
    "LeakyReLU",
    "MaxPool2D",
    "MaxPooling2D",
    "Multiply",
    "ReLU",
    "Rescaling",
    "SeparableConv2D",
    "Softmax",
    "ZeroPadding2D",
}
#: Keras activation function names → Kaimei activations (``None`` = linear).
_KERAS_ACTIVATIONS = {
    "linear": None,
    "identity": None,
    "none": None,
    "relu": "relu",
    "relu6": "relu6",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "softmax": "softmax",
    "gelu": "gelu",
    "silu": "silu",
    "swish": "silu",
    "elu": "elu",
    "leaky_relu": "leaky_relu",
    "hard_sigmoid": "hardsigmoid",
    "hard_silu": "hardswish",
    "hard_swish": "hardswish",
}
_KERAS_PASSTHROUGH = {
    "Dropout",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "GaussianNoise",
    "GaussianDropout",
    "AlphaDropout",
    "ActivityRegularization",
}


def _is_keras_model(estimator: Any) -> bool:
    layers = getattr(estimator, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        return False
    module = str(getattr(type(estimator), "__module__", "") or "")
    if module.startswith(("keras", "tensorflow")) or ".keras" in module:
        return True
    return any(type(layer).__name__ in _KERAS_LAYER_TYPES and hasattr(layer, "get_weights") for layer in layers)


def _keras_activation_of(value: Any) -> tuple:
    """``(activation, alpha)`` for a Keras activation function or name; raises for unknown ones."""
    if value is None:
        return None, None
    if isinstance(value, str):
        name = value
    else:
        name = str(getattr(value, "__name__", None) or getattr(value, "name", "") or "")
    name = name.lower()
    if name not in _KERAS_ACTIVATIONS:
        raise _Unsupported(f"activation '{name}'")
    act = _KERAS_ACTIVATIONS[name]
    alpha = {"leaky_relu": 0.2, "elu": 1.0}.get(act or "")
    return act, alpha


def _keras_activation(layer: Any) -> Optional[str]:
    """The activation applied by ``layer`` (a fused one, or an activation layer), for task inference."""
    try:
        if getattr(layer, "activation", None) is not None:
            return _keras_activation_of(layer.activation)[0]
        converted = _keras_layer(layer, type(layer).__name__)
    except (_Unsupported, StructureError):
        return None
    if converted and converted[-1].get("type") == "activation":
        return converted[-1].get("activation")
    return None


def _keras_input_shape(estimator: Any) -> Optional[List[int]]:
    raw = getattr(estimator, "input_shape", None)
    if raw is None and getattr(estimator, "inputs", None):
        first = estimator.inputs[0] if isinstance(estimator.inputs, (list, tuple)) else estimator.inputs
        raw = getattr(first, "shape", None)
    if raw is None:
        layers = list(getattr(estimator, "layers", None) or [])
        if layers and type(layers[0]).__name__ == "InputLayer":
            raw = getattr(layers[0], "batch_shape", None) or getattr(layers[0], "input_shape", None)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    dims = [dim for dim in list(raw) if dim is not None]
    return [int(dim) for dim in dims] if dims else None


def _keras_to_spec(
    estimator: Any,
    names: List[str],
    task: Optional[str],
    framework: Optional[str],
    input_shape: Optional[Sequence[int]] = None,
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    shape = [int(dim) for dim in input_shape] if input_shape is not None else _keras_input_shape(estimator)
    if shape and len(shape) == 4:
        shape = shape[1:]
    if not names:
        if shape and len(shape) == 1:
            names = [f"x{i}" for i in range(shape[0])]
        elif shape and len(shape) >= 3:
            names = ["image"]
        else:
            names = _infer_names(estimator)
    classes = _classes(estimator)
    resolved = task
    if classes is not None and resolved is None:
        resolved = "binary_classification" if len(classes) == 2 else "multiclass_classification"
    if resolved is None:
        last = list(estimator.layers)[-1]
        units = int(getattr(last, "units", 0) or 0)
        act = _keras_activation(last)
        if act == "sigmoid" and units <= 1:
            resolved = "binary_classification"
            classes = classes or ["negative", "positive"]
        elif act == "softmax" and (units > 1 or units == 0):
            resolved = "multiclass_classification"
        else:
            resolved = "regression"
    image = shape is not None and len(shape) >= 3
    layers: List[Dict[str, Any]] = []
    if image and shape is not None:
        layers.append(
            {"id": "input", "type": "input", "names": names[:1] or ["image"], "shape": shape[-3:], "label": "Image"}
        )
        names = names[:1] or ["image"]
    else:
        layers.append({"id": "input", "type": "input", "names": names})
    entry = _preprocess_layer(layers, preprocess, "input")
    state = _KerasState(layers)
    outputs = _keras_emit_model(estimator, [entry], "", state)
    if len(outputs) != 1:
        raise StructureError("Keras models with more than one output are not supported.", code="invalid_structure")
    if state.problems:
        listed = "; ".join(state.problems[:6]) + ("; …" if len(state.problems) > 6 else "")
        raise StructureError(
            f"This model uses layers Kaimei cannot replay yet: {listed}. Register it with a remote "
            "inference endpoint instead; image explanations then use occlusion and LIME.",
            code="unsupported_layers",
        )
    if _out(layers[-1]) != outputs[0]:
        layers.append({"id": "output", "type": "dropout", "rate": 0, "label": "Output", "inputs": [outputs[0]]})
    conv = any(layer.get("type") == "conv2d" for layer in layers)
    added = None
    if image or state.graph:
        width = int(layers[-1].get("units") or 0) or (len(classes) if classes else 0)
        resolved_head, head_labels, added = _classification_head(layers, resolved, [width] if width else None)
        if classes is None and head_labels:
            classes = head_labels
    elif resolved == "multiclass_classification" and classes is None:
        units = int(layers[-1].get("units") or 0)
        classes = [f"class_{i}" for i in range(units)] if units else None
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or detect_framework(estimator) or "keras",
        "architecture": "cnn" if conv else "mlp",
        "task": resolved,
        "feature_names": names,
        "summary": "Keras / TensorFlow model read from layer weights."
        + (f" A {added} turns the final logits into probabilities." if added else ""),
        "layers": layers,
    }
    if classes is not None:
        spec["labels"] = classes
        spec["outputs"] = {"labels": classes}
        if layers[-1].get("type") == "dense":
            layers[-1]["output_labels"] = classes
    if state.approximate:
        spec["approximate"] = True
        spec["summary"] = "Keras / TensorFlow model read from layer weights. Unrecognised layers are passed through."
    return spec


class _KerasState:
    def __init__(self, layers: List[Dict[str, Any]]):
        self.layers = layers
        self.taken = {layer["id"] for layer in layers}
        self.problems: List[str] = []
        self.approximate = False
        self.graph = False

    def unique(self, raw: str) -> str:
        base = _safe_id(raw)
        candidate, index = base, 1
        while candidate in self.taken:
            index += 1
            candidate = f"{base}_{index}"
        self.taken.add(candidate)
        return candidate


def _keras_graph(model: Any) -> Optional[tuple]:
    """``([(layer, inbound layer names)], output names)`` for a functional model, ``None`` if sequential."""
    try:
        config = model.get_config() if hasattr(model, "get_config") else None
    except Exception:
        return None
    entries = config.get("layers") if isinstance(config, dict) else None
    if not isinstance(entries, list) or not any(
        isinstance(item, dict) and item.get("inbound_nodes") for item in entries
    ):
        return None
    by_name = {str(getattr(layer, "name", "")): layer for layer in model.layers}
    order = []
    for item in entries:
        name = str(item.get("name") or (item.get("config") or {}).get("name") or "")
        layer = by_name.get(name)
        if layer is None:
            continue
        nodes = item.get("inbound_nodes") or []
        if len(nodes) > 1:
            raise StructureError(
                f"Keras layer '{name}' is shared between branches, which the structure spec cannot express.",
                code="unsupported_layers",
            )
        order.append((layer, _keras_inbound(nodes)))
    raw_outputs = (config or {}).get("output_layers") or []
    if raw_outputs and isinstance(raw_outputs[0], str):
        raw_outputs = [raw_outputs]  # one output, written flat
    outputs = [_keras_history_name(item) for item in raw_outputs]
    return order, [name for name in outputs if name]


def _keras_history_name(item: Any) -> Optional[str]:
    if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
        return item[0]
    return None


def _keras_inbound(nodes: Any) -> List[str]:
    """Inbound layer names in order, for both the Keras 2 and Keras 3 config formats."""
    found: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            history = (
                (value.get("config") or {}).get("keras_history")
                if value.get("class_name") == "__keras_tensor__"
                else None
            )
            if history:
                found.append(str(history[0]))
                return
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            if (
                len(value) >= 3
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and isinstance(value[2], int)
            ):
                found.append(value[0])
                return
            for item in value:
                walk(item)

    walk(nodes)
    return found


def _keras_emit_model(model: Any, entry: List[str], prefix: str, state: _KerasState) -> List[str]:
    """Emit the layers of a (possibly nested) Keras model; returns the ids of its outputs."""
    graph = _keras_graph(model)
    if graph is None:
        current = list(entry)
        for layer in model.layers:
            if type(layer).__name__ == "InputLayer":
                continue
            current = _keras_emit_layer(layer, current, prefix, state)
        return current
    state.graph = True
    order, outputs = graph
    produced: Dict[str, List[str]] = {}
    for layer, inbound in order:
        name = str(getattr(layer, "name", ""))
        if type(layer).__name__ == "InputLayer":
            produced[name] = list(entry)
            continue
        sources = [source for item in inbound for source in produced.get(item, [])]
        produced[name] = _keras_emit_layer(layer, sources, prefix, state)
    if not outputs:
        return produced[str(getattr(order[-1][0], "name", ""))]
    return [source for name in outputs for source in produced.get(name, [])]


def _keras_emit_layer(layer: Any, sources: List[str], prefix: str, state: _KerasState) -> List[str]:
    kind = type(layer).__name__
    name = str(getattr(layer, "name", kind))
    if _is_keras_model(layer) and hasattr(layer, "get_config"):
        return _keras_emit_model(layer, sources, f"{prefix}{name}.", state)
    try:
        converted = _keras_layer(layer, kind)
    except _Unsupported as exc:
        converted = None
        reason = str(exc)
    else:
        reason = kind
    if converted is None:
        if len(sources) == 1 and _keras_keeps_shape(layer):
            state.approximate = True
            converted = [{"id": name, "type": "dropout", "rate": 0, "label": f"{kind} (not replayed)"}]
        else:
            state.problems.append(f"{reason} ({prefix}{name})")
            return sources[:1]
    if not converted:
        return sources[:1]
    previous = list(sources)
    for layer_spec in converted:
        layer_spec = dict(layer_spec)
        layer_spec["id"] = state.unique(f"{prefix}{layer_spec.get('id') or name}")
        layer_spec["path"] = f"{prefix}{layer_spec.get('path') or name}"
        layer_spec["inputs"] = previous
        state.layers.append(layer_spec)
        previous = [_out(layer_spec)]
    return previous


def _keras_keeps_shape(layer: Any) -> bool:
    try:
        return tuple(layer.input.shape) == tuple(layer.output.shape)
    except Exception:
        return False


def _keras_weights(layer: Any, node_id: str) -> List[Any]:
    weights = list(layer.get_weights())
    if not weights:
        raise StructureError(
            f"Keras layer '{node_id}' has no weights. Fit the model before uploading.",
            code="unsupported_estimator",
        )
    return [np.asarray(weight, dtype=float) for weight in weights]


def _keras_padding(layer: Any) -> str:
    padding = str(getattr(layer, "padding", "valid")).lower()
    if padding not in ("same", "valid"):
        raise _Unsupported(f"padding='{padding}'")
    return padding


def _keras_channels_last(layer: Any) -> None:
    data_format = getattr(layer, "data_format", None)
    if data_format not in (None, "channels_last"):
        raise _Unsupported(f"data_format='{data_format}'")


def _with_activation(payload: Dict[str, Any], layer: Any) -> Dict[str, Any]:
    act, alpha = _keras_activation_of(getattr(layer, "activation", None))
    if act:
        payload["activation"] = act
        if alpha is not None:
            payload["alpha"] = alpha
    return payload


def _keras_layer(layer: Any, kind: str) -> Optional[List[Dict[str, Any]]]:
    """One Keras layer as spec layers (``[]`` for the identity, ``None`` when unknown)."""
    node_id = _safe_id(str(getattr(layer, "name", kind)))
    label = str(getattr(layer, "name", None) or kind)
    base: Dict[str, Any] = {"id": node_id, "path": label, "label": label}
    if kind == "Dense":
        try:
            rank = len(layer.input.shape)
        except Exception:
            rank = None
        if rank is not None and rank > 2:
            raise _Unsupported("Dense on a spatial tensor (add Flatten or global pooling first)")
        weights = _keras_weights(layer, node_id)
        weight = weights[0].T
        bias = weights[1].ravel() if len(weights) > 1 else np.zeros(weight.shape[0])
        return [
            _with_activation(
                {**base, "type": "dense", "units": int(weight.shape[0]), "weight": weight, "bias": bias}, layer
            )
        ]
    if kind == "Conv2D":
        _keras_channels_last(layer)
        weights = _keras_weights(layer, node_id)
        if weights[0].ndim != 4:
            return None
        # Keras (kh, kw, in / groups, out) → Kaimei (out, in / groups, kh, kw)
        weight = np.transpose(weights[0], (3, 2, 0, 1))
        bias = weights[1].ravel() if len(weights) > 1 else np.zeros(weight.shape[0])
        payload = {
            **base,
            "type": "conv2d",
            "weight": weight,
            "bias": bias,
            "padding": _keras_padding(layer),
            "stride": _two(getattr(layer, "strides", 1)),
            "dilation": _two(getattr(layer, "dilation_rate", 1)),
            "groups": int(getattr(layer, "groups", 1) or 1),
        }
        return [_with_activation(payload, layer)]
    if kind in ("DepthwiseConv2D", "SeparableConv2D"):
        _keras_channels_last(layer)
        weights = _keras_weights(layer, node_id)
        depthwise = weights[0]  # (kh, kw, in, multiplier)
        kh, kw, channels, multiplier = depthwise.shape
        # output channel c·m + j reads input channel c with kernel [:, :, c, j]
        dw_weight = np.transpose(depthwise, (2, 3, 0, 1)).reshape(channels * multiplier, 1, kh, kw)
        common = {
            "type": "conv2d",
            "padding": _keras_padding(layer),
            "stride": _two(getattr(layer, "strides", 1)),
            "dilation": _two(getattr(layer, "dilation_rate", 1)),
            "groups": int(channels),
        }
        if kind == "DepthwiseConv2D":
            bias = weights[1].ravel() if len(weights) > 1 else np.zeros(channels * multiplier)
            return [_with_activation({**base, **common, "weight": dw_weight, "bias": bias}, layer)]
        pointwise = weights[1]  # (1, 1, in·multiplier, out)
        pw_weight = np.transpose(pointwise, (3, 2, 0, 1))
        bias = weights[2].ravel() if len(weights) > 2 else np.zeros(pw_weight.shape[0])
        return [
            {
                **base,
                **common,
                "id": f"{node_id}.depthwise",
                "path": f"{label}.depthwise",
                "weight": dw_weight,
                "bias": np.zeros(channels * multiplier),
            },
            _with_activation(
                {
                    **base,
                    "id": f"{node_id}.pointwise",
                    "path": f"{label}.pointwise",
                    "type": "conv2d",
                    "weight": pw_weight,
                    "bias": bias,
                    "padding": "valid",
                },
                layer,
            ),
        ]
    if kind == "BatchNormalization":
        axis = getattr(layer, "axis", -1)
        axis = axis[0] if isinstance(axis, (list, tuple)) and len(axis) == 1 else axis
        try:
            rank = len(layer.input.shape)
        except Exception:
            rank = None
        if axis != -1 and (rank is None or axis != rank - 1):
            raise _Unsupported(f"BatchNormalization over axis {axis}")
        weights = list(layer.get_weights())
        values = [np.asarray(weight, dtype=float).ravel() for weight in weights]
        index = 0
        payload = {**base, "type": "batchnorm", "eps": float(getattr(layer, "epsilon", 1e-3))}
        if getattr(layer, "scale", True):
            payload["gamma"] = values[index]
            index += 1
        if getattr(layer, "center", True):
            payload["beta"] = values[index]
            index += 1
        payload["mean"], payload["var"] = values[index], values[index + 1]
        return [payload]
    if kind in ("MaxPooling2D", "MaxPool2D", "AveragePooling2D", "AvgPool2D"):
        _keras_channels_last(layer)
        pool = _two(getattr(layer, "pool_size", 2))
        strides = getattr(layer, "strides", None)
        padding = _keras_padding(layer)
        payload = {
            **base,
            "type": "maxpool" if kind.startswith("Max") else "avgpool",
            "kernel": pool,
            "stride": _two(strides) if strides is not None else pool,
            "padding": padding,
        }
        if payload["type"] == "avgpool":
            payload["count_include_pad"] = False  # TensorFlow averages only the real pixels
        return [payload]
    if kind in ("GlobalAveragePooling2D", "GlobalAvgPool2D", "GlobalMaxPooling2D", "GlobalMaxPool2D"):
        _keras_channels_last(layer)
        return [{**base, "type": "global_avgpool" if "Av" in kind else "global_maxpool"}]
    if kind == "Flatten":
        return [{**base, "type": "flatten"}]
    if kind == "Reshape":
        target = [int(dim) for dim in getattr(layer, "target_shape", ())]
        if len(target) == 1:
            return [{**base, "type": "flatten"}]
        if len(target) == 3 and target[0] * target[1] == 1:
            return []  # (C) → (1, 1, C): a per-channel gate, already 1×1×C in Kaimei
        raise _Unsupported(f"Reshape to {tuple(target)}")
    if kind in _KERAS_PASSTHROUGH:
        return [{**base, "type": "dropout", "rate": float(getattr(layer, "rate", 0.0) or 0.0)}]
    if kind in ("Activation", "Softmax", "GELU"):
        source = getattr(layer, "activation", None) if kind == "Activation" else kind.lower()
        act, alpha = _keras_activation_of(source)
        if not act:
            return []
        payload = {**base, "type": "activation", "activation": act}
        if alpha is not None:
            payload["alpha"] = alpha
        return [payload]
    if kind == "ReLU":
        max_value = getattr(layer, "max_value", None)
        slope = float(getattr(layer, "negative_slope", 0.0) or 0.0)
        threshold = float(getattr(layer, "threshold", 0.0) or 0.0)
        if threshold != 0.0:
            raise _Unsupported("ReLU with a threshold")
        if slope and max_value is None:
            return [{**base, "type": "activation", "activation": "leaky_relu", "alpha": slope}]
        if not slope and max_value is None:
            return [{**base, "type": "activation", "activation": "relu"}]
        if not slope and max_value is not None and float(max_value) == 6.0:
            return [{**base, "type": "activation", "activation": "relu6"}]
        raise _Unsupported(f"ReLU(max_value={max_value}, negative_slope={slope})")
    if kind == "LeakyReLU":
        negative: Any = getattr(layer, "negative_slope", None)
        if negative is None:
            negative = getattr(layer, "alpha", 0.3)
        return [{**base, "type": "activation", "activation": "leaky_relu", "alpha": float(negative)}]
    if kind == "ELU":
        return [{**base, "type": "activation", "activation": "elu", "alpha": float(getattr(layer, "alpha", 1.0))}]
    if kind == "Add":
        return [{**base, "type": "add"}]
    if kind == "Multiply":
        return [{**base, "type": "multiply"}]
    if kind == "Concatenate":
        if int(getattr(layer, "axis", -1)) not in (-1, 3):
            raise _Unsupported(f"Concatenate along axis {layer.axis}")
        return [{**base, "type": "concat"}]
    if kind == "ZeroPadding2D":
        _keras_channels_last(layer)
        (top, bottom), (left, right) = getattr(layer, "padding", ((1, 1), (1, 1)))
        return [{**base, "type": "pad2d", "padding": [int(top), int(bottom), int(left), int(right)]}]
    if kind == "Rescaling":
        return [
            {
                **base,
                "type": "normalize",
                "scale": _keras_scalar_or_array(getattr(layer, "scale", 1.0)),
                "offset": _keras_scalar_or_array(getattr(layer, "offset", 0.0)),
            }
        ]
    if kind == "Normalization":
        if getattr(layer, "invert", False):
            raise _Unsupported("Normalization(invert=True)")
        mean = np.asarray(layer.mean, dtype=float).ravel()
        variance = np.asarray(layer.variance, dtype=float).ravel()
        return [{**base, "type": "normalize", "mean": mean, "std": np.maximum(np.sqrt(variance), 1e-7)}]
    if kind == "LayerNormalization":
        weights = list(layer.get_weights()) if hasattr(layer, "get_weights") else []
        payload = {**base, "type": "layernorm", "eps": float(getattr(layer, "epsilon", 1e-5))}
        if len(weights) >= 2:
            payload["gamma"] = np.asarray(weights[0], dtype=float)
            payload["beta"] = np.asarray(weights[1], dtype=float)
        return [payload]
    return None


def _keras_scalar_or_array(value: Any) -> Any:
    array = np.asarray(value, dtype=float).ravel()
    return float(array[0]) if array.size == 1 else array


def rows_from_inputs(
    X: Any, names: Optional[Sequence[str]] = None, *, image_shape: Optional[Sequence[int]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Turn a DataFrame, 2D array, or list of dicts into reference-data rows.

    With ``image_shape=[height, width, channels]``, ``X`` may be a batch of images:
    ``(n, h, w, c)``, PyTorch's ``(n, c, h, w)`` (transposed here), or ``(n, h, w)`` grayscale.
    """
    if X is None:
        raise StructureError("X is required.", code="invalid_reference_rows", param="X")
    if image_shape is not None and not (isinstance(X, list) and X and isinstance(X[0], Mapping)):
        return _image_rows(X, names, image_shape)
    if hasattr(X, "to_dict") and hasattr(X, "columns"):
        resolved = [str(name) for name in names] if names is not None else [str(col) for col in list(X.columns)]
        records = X.to_dict(orient="records")
        return [{key: row.get(key) for key in resolved} for row in records], resolved
    if isinstance(X, list) and X and isinstance(X[0], Mapping):
        resolved = [str(name) for name in names] if names is not None else [str(key) for key in X[0].keys()]
        return [{key: dict(row).get(key) for key in resolved} for row in X], resolved
    numpy = _require_numpy()
    array = numpy.asarray(X)
    if array.ndim != 2:
        raise StructureError("X must be a 2D array of rows.", code="invalid_reference_rows", param="X")
    resolved = [str(name) for name in names] if names is not None else [f"x{i}" for i in range(array.shape[1])]
    if array.shape[1] != len(resolved):
        raise StructureError(
            f"X has {array.shape[1]} columns, but {len(resolved)} feature names were given.",
            code="invalid_reference_rows",
            param="X",
        )
    rows = []
    for raw in array:
        rows.append({name: _json_number(value) for name, value in zip(resolved, raw.tolist())})
    return rows, resolved


def _image_rows(
    X: Any, names: Optional[Sequence[str]], image_shape: Sequence[int]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    numpy = _require_numpy()
    height, width, channels = (int(dim) for dim in image_shape)
    array = numpy.asarray(X.detach().cpu().numpy() if hasattr(X, "detach") else X, dtype=float)
    if array.ndim == 3 and channels == 1:
        array = array[..., None]
    if array.ndim != 4:
        raise StructureError(
            f"Image reference data must be a batch of {height}×{width}×{channels} images; got shape {array.shape}.",
            code="invalid_reference_rows",
            param="X",
        )
    if tuple(array.shape[1:]) != (height, width, channels):
        if tuple(array.shape[1:]) == (channels, height, width):
            array = array.transpose(0, 2, 3, 1)
        else:
            raise StructureError(
                f"Images are {tuple(array.shape[1:])}; the model reads {height}×{width}×{channels} "
                f"(or {channels}×{height}×{width} channels first).",
                code="invalid_reference_rows",
                param="X",
            )
    name = str(list(names)[0]) if names else "image"
    return [{name: image.round(6).tolist()} for image in array], [name]


def _json_number(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def labels_from_target(y: Any) -> List[Any]:
    if hasattr(y, "tolist"):
        return [_json_number(value) for value in list(y.tolist())]
    return [_json_number(value) for value in list(y)]
