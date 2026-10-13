"""Read a fitted estimator in the user's process, then upload the JSON spec.

The HTTP API never unpickles a checkpoint. A notebook that already holds the
trained object calls :func:`estimator_to_spec` (or ``client.models.upload``)
and sends layers, weights, and biases as JSON.

This module is the user-facing copy of ``kaimei_api.engine.structure.extract``.
Keep the conversion rules aligned with that file when you change either side.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import KaimeiError

MAX_REFERENCE_ROWS = 5000
WARN_STRUCTURE_BYTES = 2_000_000
LARGE_STRUCTURE_BYTES = 16_000_000
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
            f"This structure payload is {size / 1_000_000:.1f} MB. Reverse proxies and "
            "Postgres JSONB may reject it. Prefer a smaller model, an ensemble summary, "
            "or a remote inference URL.",
            UserWarning,
            stacklevel=3,
        )
    elif size >= WARN_STRUCTURE_BYTES:
        warnings.warn(
            f"This structure payload is {size / 1_000_000:.1f} MB. Large weight tensors "
            "are stored inline as JSON and can be slow to upload.",
            UserWarning,
            stacklevel=3,
        )
    return size


def cap_reference_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: Optional[Sequence[Any]] = None,
    *,
    limit: int = MAX_REFERENCE_ROWS,
) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
    capped = [dict(row) for row in rows]
    resolved_labels = list(labels) if labels is not None else None
    if len(capped) <= limit:
        return capped, resolved_labels
    warnings.warn(
        f"Reference data has {len(capped)} rows; Kaimei keeps the first {limit}. "
        "Pass a representative sample or raise KAIMEI_MAX_REFERENCE_ROWS on the API.",
        UserWarning,
        stacklevel=3,
    )
    if resolved_labels is not None:
        resolved_labels = resolved_labels[:limit]
    return capped[:limit], resolved_labels


def estimator_to_spec(
    estimator: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    framework: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a ``kaimei.structure.v1`` document for ``compile_spec`` / ``PUT /structure``."""
    _require_numpy()
    if feature_names is not None:
        names = [str(name) for name in feature_names]
    else:
        try:
            names = _infer_names(estimator)
        except StructureError:
            if _is_keras_model(estimator) or _is_torch_module(estimator):
                names = []
            else:
                raise
    if _is_pipeline(estimator):
        return _pipeline_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "coefs_") and hasattr(estimator, "intercepts_"):
        return _mlp_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "estimators_") and hasattr(estimator, "feature_importances_"):
        return _forest_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "tree_"):
        return _tree_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        return _linear_to_spec(estimator, names, task, framework)
    if _is_keras_model(estimator):
        return _keras_to_spec(estimator, names, task, framework)
    if _is_torch_module(estimator):
        return _torch_to_spec(estimator, names, task, framework)
    if hasattr(estimator, "feature_importances_"):
        return _importance_to_spec(estimator, names, task, framework)
    raise StructureError(
        "Cannot read this estimator. Supported shapes: linear coefficients, MLP coefs_, "
        "a decision tree, a forest, a StandardScaler pipeline around those, a Keras / "
        "TensorFlow model, a torch.nn module, or an estimator with feature_importances_.",
        code="unsupported_estimator",
    )


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
    return hasattr(estimator, "steps") and isinstance(estimator.steps, (list, tuple))


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
                "weight": weight.tolist(),
                "bias": bias.tolist(),
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
            "weight": weight.tolist(),
            "bias": bias.tolist(),
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
        return estimator_to_spec(steps[-1][1], feature_names=names, task=task, framework=framework)
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
    if hasattr(final_estimator, "tree_") and not hasattr(final_estimator, "coefs_"):
        spec = _tree_to_spec(final_estimator, names, task, framework)
        _fold_scaler(spec["nodes"], center, scale)
        spec["summary"] = "Standardizer folded into the split thresholds, so the tree reads raw features."
        return spec
    spec = estimator_to_spec(final_estimator, feature_names=names, task=task, framework=framework)
    if spec.get("family") not in {None, "network"}:
        raise StructureError(
            "A scaler can precede a linear model, an MLP, or a single tree.",
            code="unsupported_estimator",
        )
    layers = list(spec["layers"])
    scaler_layer = {
        "id": "standardize",
        "type": "standardize",
        "center": np.asarray(center, dtype=float).tolist(),
        "scale": np.asarray(scale, dtype=float).tolist(),
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


def _torch_to_spec(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise StructureError("torch is not installed.", code="unsupported_estimator") from exc

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
        layer = _torch_layer(name, module, torch)
        if layer is None:
            unsupported = True
            layers.append(
                {
                    "id": _safe_id(name),
                    "type": "dropout",
                    "label": type(module).__name__,
                    "rate": 0,
                }
            )
            continue
        layers.append(layer)
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or "pytorch",
        "architecture": "mlp" if not unsupported else "network",
        "task": resolved,
        "feature_names": names,
        "summary": "PyTorch module read in forward-hook order."
        if not unsupported
        else "PyTorch module read in forward-hook order. Unrecognised layers are passed through.",
        "layers": layers,
    }
    if unsupported:
        spec["approximate"] = True
    return spec


def _torch_layer(name: str, module: Any, torch: Any) -> Optional[Dict[str, Any]]:
    cls = type(module).__name__
    node_id = _safe_id(name)
    if cls == "Linear":
        weight = module.weight.detach().cpu().numpy()
        bias = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(weight.shape[0])
        return {
            "id": node_id,
            "type": "dense",
            "units": int(weight.shape[0]),
            "weight": weight.tolist(),
            "bias": np.asarray(bias).ravel().tolist(),
            "label": name,
        }
    activations = {
        "ReLU": "relu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
        "GELU": "gelu",
        "SiLU": "silu",
        "Softmax": "softmax",
    }
    if cls in activations:
        return {"id": node_id, "type": "activation", "activation": activations[cls], "label": name}
    if cls == "Dropout":
        return {"id": node_id, "type": "dropout", "rate": float(getattr(module, "p", 0.0)), "label": name}
    if cls == "LayerNorm":
        layer: Dict[str, Any] = {"id": node_id, "type": "layernorm", "label": name, "eps": float(module.eps)}
        if module.elementwise_affine:
            layer["gamma"] = module.weight.detach().cpu().numpy().tolist()
            layer["beta"] = module.bias.detach().cpu().numpy().tolist()
        return layer
    if cls == "Embedding":
        return {
            "id": node_id,
            "type": "embedding",
            "weight": module.weight.detach().cpu().numpy().tolist(),
            "label": name,
        }
    del torch
    return None


def _safe_id(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "_.-" else "_" for char in name)
    return cleaned or "layer"


def detect_framework(estimator: Any) -> str:
    module = str(getattr(type(estimator), "__module__", "") or "")
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
    if "tensorflow" in module:
        return "tensorflow"
    if module.startswith("keras") or ".keras" in module:
        return "keras"
    if _is_keras_model(estimator):
        return "keras"
    if _is_torch_module(estimator):
        return "pytorch"
    return "custom"


def features_from_spec(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a model feature schema from an extracted structure document."""
    names = [str(name) for name in spec.get("feature_names") or []]
    layers = list(spec.get("layers") or [])
    first = layers[0] if layers else {}
    shape = first.get("shape") if str(first.get("type") or "") == "input" else None
    if shape and len(shape) == 3 and names:
        return [
            {
                "name": names[0],
                "type": "image",
                "height": int(shape[0]),
                "width": int(shape[1]),
                "channels": int(shape[2]),
            }
        ]
    return [{"name": name, "type": "numeric"} for name in names]


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
    "BatchNormalization",
    "Conv2D",
    "Dense",
    "Dropout",
    "ELU",
    "Flatten",
    "GELU",
    "InputLayer",
    "LayerNormalization",
    "LeakyReLU",
    "MaxPool2D",
    "MaxPooling2D",
    "ReLU",
    "Softmax",
}


def _is_keras_model(estimator: Any) -> bool:
    layers = getattr(estimator, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        return False
    module = str(getattr(type(estimator), "__module__", "") or "")
    if module.startswith(("keras", "tensorflow")) or ".keras" in module:
        return True
    return any(type(layer).__name__ in _KERAS_LAYER_TYPES and hasattr(layer, "get_weights") for layer in layers)


def _keras_activation(layer: Any) -> Optional[str]:
    act = getattr(layer, "activation", None)
    if act is None:
        name = type(layer).__name__.lower()
        aliases = {
            "relu": "relu",
            "sigmoid": "sigmoid",
            "tanh": "tanh",
            "softmax": "softmax",
            "gelu": "gelu",
            "elu": "relu",
        }
        return aliases.get(name)
    if isinstance(act, str):
        name = act
    else:
        name = str(getattr(act, "__name__", None) or getattr(act, "name", "") or "")
    name = name.lower()
    if name in {"", "linear", "identity", "none"}:
        return None
    if name == "swish":
        return "silu"
    return name


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
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    shape = _keras_input_shape(estimator)
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
        elif act == "softmax" and units > 1:
            resolved = "multiclass_classification"
            classes = classes or [f"class_{i}" for i in range(units)]
        else:
            resolved = "regression"
    layers: List[Dict[str, Any]] = []
    if shape and len(shape) >= 3:
        layers.append(
            {
                "id": "input",
                "type": "input",
                "names": names[:1] or ["image"],
                "shape": shape[-3:],
                "label": "Image",
            }
        )
        names = names[:1] or ["image"]
    else:
        layers.append({"id": "input", "type": "input", "names": names})
    unsupported = False
    conv = False
    for layer in estimator.layers:
        kind = type(layer).__name__
        if kind == "InputLayer":
            continue
        converted = _keras_layer(layer, kind)
        if converted is None:
            unsupported = True
            layers.append(
                {
                    "id": _safe_id(str(getattr(layer, "name", kind))),
                    "type": "dropout",
                    "rate": 0,
                    "label": kind,
                }
            )
            continue
        if converted.get("type") == "conv2d":
            conv = True
        layers.append(converted)
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or detect_framework(estimator) or "keras",
        "architecture": "cnn" if conv else "mlp",
        "task": resolved,
        "feature_names": names,
        "summary": "Keras / TensorFlow model read from layer weights.",
        "layers": layers,
    }
    if classes is not None:
        spec["labels"] = classes
        spec["outputs"] = {"labels": classes}
        if layers and layers[-1].get("type") == "dense":
            layers[-1]["output_labels"] = classes
    if unsupported:
        spec["approximate"] = True
        spec["summary"] = (
            "Keras / TensorFlow model read from layer weights. "
            "Unrecognised layers are passed through."
        )
    return spec


def _keras_layer(layer: Any, kind: str) -> Optional[Dict[str, Any]]:
    node_id = _safe_id(str(getattr(layer, "name", kind)))
    if kind == "Dense":
        weights = list(layer.get_weights())
        if not weights:
            raise StructureError(
                f"Keras layer '{node_id}' has no weights. Fit the model before uploading.",
                code="unsupported_estimator",
            )
        kernel = np.asarray(weights[0], dtype=float)
        weight = kernel.T
        bias = np.asarray(weights[1], dtype=float).ravel() if len(weights) > 1 else np.zeros(weight.shape[0])
        payload: Dict[str, Any] = {
            "id": node_id,
            "type": "dense",
            "units": int(weight.shape[0]),
            "weight": weight.tolist(),
            "bias": bias.tolist(),
            "label": str(getattr(layer, "name", None) or "Dense"),
        }
        activation = _keras_activation(layer)
        if activation:
            payload["activation"] = activation
        return payload
    if kind == "Conv2D":
        weights = list(layer.get_weights())
        if not weights:
            raise StructureError(
                f"Keras layer '{node_id}' has no weights. Fit the model before uploading.",
                code="unsupported_estimator",
            )
        kernel = np.asarray(weights[0], dtype=float)
        if kernel.ndim != 4:
            return None
        # Keras (kh, kw, in, out) → Kaimei (out, in, kh, kw)
        weight = np.transpose(kernel, (3, 2, 0, 1))
        bias = np.asarray(weights[1], dtype=float).ravel() if len(weights) > 1 else np.zeros(weight.shape[0])
        padding = str(getattr(layer, "padding", "valid")).lower()
        strides = getattr(layer, "strides", (1, 1))
        stride = int(strides[0] if isinstance(strides, (list, tuple)) else strides)
        payload = {
            "id": node_id,
            "type": "conv2d",
            "weight": weight.tolist(),
            "bias": bias.tolist(),
            "padding": int(weight.shape[2] // 2) if padding == "same" else 0,
            "stride": stride,
            "label": str(getattr(layer, "name", None) or "Conv"),
        }
        activation = _keras_activation(layer)
        if activation:
            payload["activation"] = activation
        return payload
    if kind in {"Flatten"}:
        return {"id": node_id, "type": "flatten", "label": str(getattr(layer, "name", None) or "Flatten")}
    if kind in {"MaxPooling2D", "MaxPool2D"}:
        size = getattr(layer, "pool_size", 2)
        pool = int(size[0] if isinstance(size, (list, tuple)) else size)
        strides = getattr(layer, "strides", pool)
        stride = int(strides[0] if isinstance(strides, (list, tuple)) else strides)
        return {
            "id": node_id,
            "type": "maxpool",
            "kernel": pool,
            "stride": stride,
            "label": str(getattr(layer, "name", None) or "Max pool"),
        }
    if kind == "Dropout":
        return {
            "id": node_id,
            "type": "dropout",
            "rate": float(getattr(layer, "rate", 0.0)),
            "label": str(getattr(layer, "name", None) or "Dropout"),
        }
    if kind in {"ReLU", "Softmax", "Activation", "GELU"}:
        activation = _keras_activation(layer)
        if not activation:
            return None
        return {"id": node_id, "type": "activation", "activation": activation, "label": activation}
    if kind == "LayerNormalization":
        weights = list(layer.get_weights()) if hasattr(layer, "get_weights") else []
        payload = {
            "id": node_id,
            "type": "layernorm",
            "label": str(getattr(layer, "name", None) or "LayerNorm"),
            "eps": float(getattr(layer, "epsilon", 1e-5)),
        }
        if len(weights) >= 2:
            payload["gamma"] = np.asarray(weights[0], dtype=float).tolist()
            payload["beta"] = np.asarray(weights[1], dtype=float).tolist()
        return payload
    return None


def rows_from_inputs(X: Any, names: Optional[Sequence[str]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Turn a DataFrame, 2D array, or list of dicts into reference-data rows."""
    if X is None:
        raise StructureError("X is required.", code="invalid_reference_rows", param="X")
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
