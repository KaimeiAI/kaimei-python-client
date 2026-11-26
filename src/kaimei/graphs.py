"""General model graphs: any model as a graph of ONNX operators that the Kaimei API draws and runs.

The readers in :mod:`kaimei.structure` take a model apart layer by layer, and know a fixed set of layers.
A model they cannot read (a recurrent network, a transformer, a model with several inputs, a scikit-learn
estimator with no reader of its own) is written down the way ONNX writes it instead: named tensors, the
operators that read some and produce others, and the constants they use. That is a structure of ``family:
"graph"``, and the API runs each of its operators as ONNX defines it.

Everything here happens on your machine. :func:`via_onnx` converts a live model to ONNX in memory with the
framework's own exporter, :func:`from_onnx` reads an ONNX model, and :func:`finalize_graph` turns the
result into what is sent: operators, attributes, shapes and tensors. The model file is never sent.

A conversion is only used if it is the model. The rows sent with a graph for the API to check itself on
hold the outputs of the original library (PyTorch, scikit-learn), not of the ONNX conversion, so the whole
chain is checked: library to ONNX here, ONNX to Kaimei's engine there. A conversion whose own outputs
differ from the library's is marked ``approximate`` and is drawn, not run.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import inspect
import io
import numbers
import os
import re
import sys
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .ensembles import rows_to_check
from .structure import (
    EXTERNALIZE_ABOVE,
    MIN_VALUES_FOR_STATS,
    TENSOR_DTYPES,
    Extraction,
    StructureError,
    _input_width,
    _is_keras_model,
    _is_torch_module,
    _names_it_was_fitted_on,
    _pieces,
    _require_numpy,
    extraction_metadata,
    label_names,
    said_together,
    tensor_dtype,
    tensor_stats,
    tensor_view,
)

#: A constant up to this many values is written into the spec as numbers; a larger one as base64 bytes.
INLINE_VALUES = 64
#: ``weights="summary"`` keeps whole-number constants up to this many values: the shapes, axes and indices a graph
#: is drawn from. It keeps no float constant, however small (a bias, a class's mean, a scale), except one that
#: only sets the size of what an operator makes (``_SHAPING``).
SUMMARY_KEEPS = 16
#: The inputs, by position, that set the size of what an operator makes: Resize's and Upsample's scales, Range's
#: bounds. A small float constant given only to them is a setting of the model, not one of its learned numbers.
_SHAPING = {"Resize": (2,), "Upsample": (1,), "Range": (0, 1, 2)}
#: Domains whose operators keep their learned numbers in constants. Any other (``ai.onnx.ml``: a converted
#: scikit-learn model's coefficients, thresholds and means) keeps them in its attributes, short lists included.
_PLAIN_DOMAINS = {"", "ai.onnx", "ai.kaimei"}
#: Rows a check holds for a model whose input is a picture, text, a series or a tensor.
CHECK_ROWS = 4
#: Rows a check holds for a model that reads a table, before any of the caller's own.
TABLE_CHECK_ROWS = 16
#: The caller's own rows a check holds at most.
OWN_CHECK_ROWS = 60
#: Input values a check holds at most, over all its rows. One row is always kept.
CHECK_VALUES = 400_000
#: A conversion whose outputs are further than this from the library's, relative to their size, is not the model.
AGREES_WITHIN = 1e-4
#: Operators whose arithmetic runs long enough in single precision to drift: their checks allow a little more.
_LONG_CHAINS = {"Conv", "ConvTranspose", "LSTM", "GRU", "RNN", "Scan", "Loop", "Attention"}
#: What the API computes a stored tensor in when the spec does not say.
_COMPUTED_AS = {
    "f16": "float32",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "bool": "bool",
}
_KEY_CHARS = re.compile(r"[^A-Za-z0-9_.:/\-]")


# --- what is sent: constants as JSON, bytes, or binary tensors ------------------------------


def finalize_graph(
    spec: Mapping[str, Any],
    *,
    weights: str = "full",
    externalize_above: Optional[int] = EXTERNALIZE_ABOVE,
    tensors: Optional[Mapping[str, Any]] = None,
) -> Extraction:
    """Turn the arrays of a ``family: "graph"`` spec into what the API accepts.

    Every constant of the graph, and of the graphs nested in its loops and branches, keeps its own number
    type (``float32`` stays ``float32``, ``int64`` stays ``int64``, flags stay flags) and becomes one of:

    * ``{"dtype", "shape", "data"}``: up to :data:`INLINE_VALUES` values, as numbers.
    * ``{"$b64", "dtype", "shape"}``: its little-endian bytes in base64. Larger constants, and any that
      holds NaN or infinity, which JSON cannot carry.
    * ``{"$tensor": key, "shape", "dtype"}``: more than ``externalize_above`` values, and two at the least. The
      array is returned in :attr:`Extraction.tensors` to upload as binary. ``None`` keeps everything in the spec.

    ``weights="summary"`` sends no learned numbers. Every float constant, whatever its size, becomes
    ``{"$shape", "dtype"}``, with statistics for one of 256 values or more; so does every whole-number constant
    of more than :data:`SUMMARY_KEEPS` values. What stays is what the graph's shapes are worked out from: small
    whole-number constants (shapes, axes, indices), and small float ones given only to an input that sets a
    size (``Resize`` scales, ``Range`` bounds). In the attributes, a list of numbers longer than that is left
    out, and so is every list of numbers of an operator outside the standard domains (``ai.onnx.ml`` keeps a
    model's coefficients and trees there); a single number (an epsilon, a slope) stays. A value an input is
    always given (``inputs={name: {"value": ...}}``) is kept by the rule for constants, except that a single
    number stays; one sent as its shape leaves the reading drawn, not run. The check is dropped,
    because its rows and answers are exactly what a summary keeps at home. ``tensors`` holds the arrays behind
    ``$tensor`` references already in the spec, for their statistics. ``weights="pictures"`` is a summary
    that also sends a small picture of each such tensor (:func:`kaimei.structure.tensor_view`).
    """
    if weights not in {"full", "summary", "pictures"}:
        raise StructureError(
            "weights must be 'full', 'summary' or 'pictures'.", code="invalid_structure", param="weights"
        )
    pictures = weights == "pictures"
    if pictures:
        weights = "summary"
    numpy = _require_numpy()
    out = dict(spec)
    state = _Finalizing(weights, externalize_above, tensors, numpy)
    state.pictures = pictures
    graph = spec.get("graph")
    if isinstance(graph, Mapping):
        out["graph"] = _finalize_body(graph, "", state, top=True)
    if weights == "summary":
        out.pop("check", None)
        out["weights"] = "summary"
        if state.kept_home:
            # With its numbers left out an operator cannot run, and the API must not try it.
            out["approximate"] = True
            note = "Its learned numbers stayed in your environment, so it is drawn, not run."
            if note not in str(out.get("summary") or ""):
                out["summary"] = f"{out.get('summary') or ''} {note}".strip()
    else:
        out.pop("weights", None)
    return Extraction(spec=out, tensors=state.lifted)


class _Finalizing:
    """What one pass of :func:`finalize_graph` carries through a graph and the graphs nested in it."""

    def __init__(self, weights: str, externalize_above: Optional[int], tensors: Optional[Mapping[str, Any]], numpy: Any):
        self.weights = weights
        self.externalize_above = externalize_above
        self.tensors = tensors or {}
        self.numpy = numpy
        self.lifted: Dict[str, Any] = {}
        #: Whether a tensor kept at home also sends a small picture of itself.
        self.pictures = False
        self.keys: set = set()
        self.kept_home = False

    def key(self, wanted: str) -> str:
        """A tensor key the API accepts, and that no other constant of this spec has."""
        base = _KEY_CHARS.sub("_", wanted).lstrip("_.:/-")[:180] or "tensor"
        key, count = base, 1
        while key in self.keys:
            count += 1
            key = f"{base}_{count}"
        self.keys.add(key)
        return key


def _finalize_body(graph: Mapping[str, Any], prefix: str, state: _Finalizing, top: bool = False) -> Dict[str, Any]:
    body = dict(graph)
    constants = graph.get("constants")
    shaping = _only_shaping(graph.get("nodes"))
    if isinstance(constants, Mapping):
        body["constants"] = {
            str(name): _finalize_constant(value, f"{prefix}{name}", state, shaping=str(name) in shaping)
            for name, value in constants.items()
        }
    nodes = graph.get("nodes")
    if isinstance(nodes, list):
        body["nodes"] = [_finalize_node(node, prefix, state) if isinstance(node, Mapping) else node for node in nodes]
    inputs = graph.get("inputs")
    if top and isinstance(inputs, list):
        # An input the graph is always given the same value for carries that value with it.
        body["inputs"] = [_finalize_input(item, state) if isinstance(item, Mapping) else item for item in inputs]
    return body


def _only_shaping(nodes: Any) -> Set[str]:
    """The names given to the nodes of a graph only as inputs that set a size (see ``_SHAPING``)."""
    shaping: Set[str] = set()
    other: Set[str] = set()
    for node in nodes if isinstance(nodes, list) else []:
        if not isinstance(node, Mapping):
            continue
        slots = _SHAPING.get(str(node.get("op"))) if str(node.get("domain") or "") in _PLAIN_DOMAINS else None
        for at, name in enumerate(node.get("inputs") or []):
            (shaping if slots and at in slots else other).add(str(name))
    return shaping - other


def _finalize_input(item: Mapping[str, Any], state: _Finalizing) -> Dict[str, Any]:
    source = item.get("from")
    if not isinstance(source, Mapping) or "value" not in source:
        return dict(item)
    if state.weights == "summary":
        kept = _summarized_value(source["value"], str(item.get("name") or "input"), state)
        return {**item, "from": {**source, "value": kept}}
    if not hasattr(source.get("value"), "dtype"):
        return dict(item)
    array = state.numpy.asarray(source["value"])
    fixed = _packed(array, state.numpy) if _not_finite(array, state.numpy) else _plain(array)
    return {**item, "from": {**source, "value": fixed}}


def _summarized_value(value: Any, name: str, state: _Finalizing) -> Any:
    """The value an input is always given, as ``weights="summary"`` sends it: by the rule for constants.

    Whole numbers stay, up to :data:`SUMMARY_KEEPS` of them, and so does a single number (one value spread over
    the input is a setting, as a single number in an attribute is). Any other array of floats is sent as its
    shape, and the reading is then drawn, not run: the model cannot be given the input without its values.
    """
    numpy = state.numpy
    if isinstance(value, Mapping):
        if "$shape" in value:
            state.kept_home = True
            return dict(value)
        array = _described(value, name, numpy)
    elif isinstance(value, (list, tuple)):
        array = _inferred(value, name, numpy)
    else:
        array = _array(value, name, numpy)
    if array.size <= 1 or array.dtype.kind in "OUS":
        if hasattr(value, "dtype"):
            return _packed(array, numpy) if _not_finite(array, numpy) else _plain(array)
        return dict(value) if isinstance(value, Mapping) else value
    kept = _finalize_constant(array, name, state)
    if isinstance(kept, Mapping) and "$shape" in kept:
        state.kept_home = True
    return kept


def _finalize_node(node: Mapping[str, Any], prefix: str, state: _Finalizing) -> Dict[str, Any]:
    attrs = node.get("attrs")
    if not isinstance(attrs, Mapping):
        return dict(node)
    step = str(node.get("id") or node.get("op") or "step")
    learned = str(node.get("domain") or "") not in _PLAIN_DOMAINS
    kept: Dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, Mapping) and isinstance(value.get("graph"), Mapping):
            kept[str(key)] = {**value, "graph": _finalize_body(value["graph"], f"{prefix}{step}.{key}.", state)}
        elif isinstance(value, Mapping) and isinstance(value.get("graphs"), list):
            kept[str(key)] = {
                **value,
                "graphs": [
                    _finalize_body(item, f"{prefix}{step}.{key}{index}.", state) if isinstance(item, Mapping) else item
                    for index, item in enumerate(value["graphs"])
                ],
            }
        else:
            kept[str(key)] = _finalize_attribute(value, state, learned)
    return {**node, "attrs": kept}


def _finalize_attribute(value: Any, state: _Finalizing, learned: bool = False) -> Any:
    """An attribute as plain JSON. A summary leaves out the lists of numbers that are learned ones: the long
    ones, and every one of an operator that keeps its learned numbers in its attributes (``learned``)."""
    numpy = state.numpy
    if isinstance(value, numpy.ndarray):
        value = _plain(value)
    elif isinstance(value, numpy.generic):
        value = value.item()
    if state.weights != "summary":
        return value
    if isinstance(value, Mapping) and isinstance(value.get("data"), list):
        long = learned or len(value["data"]) > SUMMARY_KEEPS
        if long and value["data"] and str(value.get("dtype")) != "string":
            state.kept_home = True
            return {"$shape": list(value.get("shape") or [len(value["data"])]), "dtype": str(value.get("dtype"))}
    elif isinstance(value, (list, tuple)) and value and (learned or len(value) > SUMMARY_KEEPS):
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            state.kept_home = True
            whole = all(isinstance(item, int) for item in value)
            return {"$shape": [len(value)], "dtype": "int64" if whole else "float32"}
    return value


def _finalize_constant(value: Any, name: str, state: _Finalizing, shaping: bool = False) -> Any:
    """One constant as the spec sends it. ``shaping``: it is given only to inputs that set a size."""
    numpy = state.numpy
    if state.weights != "summary":
        if isinstance(value, Mapping):
            return dict(value)  # already in a form the API reads
        if isinstance(value, (list, tuple)):
            return list(value)
        array = _array(value, name, numpy)
        if array.dtype.kind in "OUS" or array.size == 0:
            return _plain(array)
        # A single value, or one with no axes, always stays in the spec: an uploaded tensor has a shape of one axis
        # at the least, and lifting one number saves nothing.
        liftable = array.ndim >= 1 and array.size >= 2
        if (
            state.externalize_above is not None
            and array.size > state.externalize_above
            and liftable
            and _storable(array, numpy)
        ):
            key = state.key(name)
            stored, own = _stored_as(array)
            state.lifted[key] = numpy.ascontiguousarray(array, dtype=TENSOR_DTYPES[stored]).reshape(array.shape)
            reference: Dict[str, Any] = {"$tensor": key, "shape": [int(dim) for dim in array.shape], "dtype": stored}
            if own:
                reference["as"] = own
            return reference
        if (array.size > INLINE_VALUES or _not_finite(array, numpy)) and _storable(array, numpy):
            return _packed(array, numpy)
        return _plain(array)
    known: Any = None
    if isinstance(value, Mapping):
        if "$shape" in value:
            return dict(value)
        if "$tensor" in value:
            known = state.tensors.get(str(value["$tensor"]))
            dims: Any = value.get("shape")
            if dims is None:
                dims = getattr(known, "shape", ())
            shape = [int(dim) for dim in dims]
            if known is None:
                return {"$shape": shape, "dtype": str(value.get("dtype") or "f32")}
            array = numpy.asarray(known).reshape(shape)
        else:
            array = _described(value, name, numpy)
    elif isinstance(value, (list, tuple)):
        array = _inferred(value, name, numpy)
    else:
        array = _array(value, name, numpy)
    # Shapes, axes and indices stay, and a size's scales. Floats are learned numbers, however few.
    small = array.size <= SUMMARY_KEEPS and (array.dtype.kind in "biu" or shaping)
    if array.dtype.kind in "OUS" or array.size == 0 or small:
        return _packed(array, numpy) if _not_finite(array, numpy) else _plain(array)
    stored, _own = _stored_as(array)
    ghost: Dict[str, Any] = {"$shape": [int(dim) for dim in array.shape], "dtype": stored}
    if array.dtype.kind == "f" and array.size >= MIN_VALUES_FOR_STATS and _all_numbers(array, numpy):
        ghost["stats"] = tensor_stats(array)
        view = tensor_view(array) if state.pictures else None
        if view is not None:
            ghost["view"] = view
    return ghost


def _array(value: Any, name: str, numpy: Any) -> Any:
    array = numpy.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
    if array.dtype.kind == "c" or array.dtype.kind not in "biufOUS":
        raise StructureError(
            f"Constant '{name}' holds {array.dtype.name} values, which Kaimei does not compute with.",
            code="invalid_structure",
            param=name,
        )
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        array = array.astype(numpy.float64)
    return array


def _inferred(value: Any, name: str, numpy: Any) -> Any:
    """A JSON list as the API reads one: whole numbers are ``int64``, other numbers ``float32``."""
    array = _array(value, name, numpy)
    if array.dtype.kind in "iu":
        return array.astype(numpy.int64)
    if array.dtype.kind == "f":
        return array.astype(numpy.float32)
    return array


def _described(value: Mapping[str, Any], name: str, numpy: Any) -> Any:
    """The array behind a constant that is already written as JSON or base64."""
    shape = [int(dim) for dim in value.get("shape") or []]
    try:
        if "$b64" in value:
            stored = str(value.get("dtype") or "f32")
            array = numpy.frombuffer(base64.b64decode(str(value["$b64"])), dtype=TENSOR_DTYPES[stored])
            return array.astype(str(value.get("as") or _COMPUTED_AS[stored])).reshape(shape)
        kind = str(value.get("dtype") or "float32")
        if kind == "string":
            return numpy.array([str(item) for item in value.get("data") or []], dtype=object).reshape(shape or [-1])
        array = numpy.asarray(value.get("data"), dtype=kind)
        return array.reshape(shape) if value.get("shape") is not None else array
    except (KeyError, TypeError, ValueError) as exc:
        raise StructureError(f"Constant '{name}' is not a tensor Kaimei reads.", code="invalid_structure", param=name) from exc


def _not_finite(array: Any, numpy: Any) -> bool:
    return bool(array.dtype.kind == "f" and array.size and not numpy.isfinite(array).all())


def _all_numbers(array: Any, numpy: Any) -> bool:
    """Whether every value is finite, looked at a piece at a time so a large tensor is not copied to find out."""
    return all(bool(numpy.isfinite(piece).all()) for piece in _pieces(array))


def _storable(array: Any, numpy: Any) -> bool:
    """Whether the array's values fit a type the API stores bytes in. Text never does."""
    if array.dtype.kind in "OUS":
        return False
    if array.dtype.name == "uint64" and array.size and int(array.max()) >= 2**63:
        return False
    return True


def _stored_as(array: Any) -> Tuple[str, Optional[str]]:
    """``(the type its bytes are stored in, the type it is computed in when that is not the stored type's own)``."""
    stored = tensor_dtype(array)
    own = "bool" if array.dtype.kind == "b" else str(array.dtype.name)
    return stored, (None if own == _COMPUTED_AS[stored] else own)


def _plain(array: Any) -> Dict[str, Any]:
    """A tensor as numbers: ``{"dtype", "shape", "data"}``."""
    shape = [int(dim) for dim in array.shape]
    if array.dtype.kind in "OUS":
        return {"dtype": "string", "shape": shape, "data": [str(item) for item in array.reshape(-1).tolist()]}
    name = "bool" if array.dtype.kind == "b" else str(array.dtype.name)
    return {"dtype": name, "shape": shape, "data": array.reshape(-1).tolist()}


def _packed(array: Any, numpy: Any) -> Dict[str, Any]:
    """A tensor as its little-endian bytes in base64: ``{"$b64", "dtype", "shape"}``."""
    stored, own = _stored_as(array)
    raw = numpy.ascontiguousarray(array, dtype=TENSOR_DTYPES[stored]).tobytes()
    packed: Dict[str, Any] = {
        "$b64": base64.b64encode(raw).decode("ascii"),
        "dtype": stored,
        "shape": [int(dim) for dim in array.shape],
    }
    if own:
        packed["as"] = own
    return packed


def holds_arrays(spec: Mapping[str, Any]) -> bool:
    """Whether a graph spec still has numpy arrays among its constants, and so needs :func:`finalize_graph`."""
    graph = spec.get("graph")
    constants = graph.get("constants") if isinstance(graph, Mapping) else None
    if not isinstance(constants, Mapping):
        return False
    return any(hasattr(value, "dtype") and hasattr(value, "shape") for value in constants.values())


# --- ONNX in ---------------------------------------------------------------------------------


def is_onnx(obj: Any) -> bool:
    """Whether ``obj`` is an ONNX model: a ``.onnx`` path, the bytes of one, or an ``onnx.ModelProto``."""
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return True
    if isinstance(obj, (str, os.PathLike)):
        try:
            return os.fspath(obj).lower().endswith(".onnx")
        except TypeError:
            return False
    kind = type(obj)
    return kind.__name__ == "ModelProto" and str(getattr(kind, "__module__", "")).startswith("onnx")


def _has(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def _importer() -> Any:
    """The ONNX importer, or an error saying what to install."""
    _require_numpy()
    if not _has("onnx"):
        raise StructureError(
            "Reading a model through ONNX needs the onnx package: pip install onnx (or pip install 'kaimei[onnx]').",
            code="onnx_required",
        )
    from . import onnx_graph

    return onnx_graph


def from_onnx(
    source: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    labels: Optional[Sequence[Any]] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    input_shapes: Any = None,
    samples: Any = None,
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    output: Any = None,
    name: Optional[str] = None,
    tokenizer: Any = None,
    check_rows: Any = None,
    framework: str = "onnx",
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """An ONNX model (a path, bytes or ``onnx.ModelProto``) as a ``family: "graph"`` spec, its constants still arrays.

    ``feature_names`` names the columns of a table-shaped input. ``inputs`` says what any other input is, by
    its name in the file: ``{"pixels": {"feature": "photo", "layout": "nchw"}}``, ``{"input_ids": {"feature":
    "text", "part": "ids"}}``, ``{"state": {"zeros": True}}``. ``input_shapes`` fills in the sizes the file
    leaves open and ``samples`` gives example inputs (both by input name; for a model with one input the
    shape or the array itself will do). ``output`` picks the output the prediction is read from, ``labels``
    names the classes, and ``tokenizer`` turns a token input into a text feature. ``threshold`` says where a
    segmentation's one score for each pixel is cut, in the model's own numbers (the model says, when it is not
    given); it is refused for a mask of several classes or of class numbers, and for any other task.

    A graph's shapes are fixed when it is read, so a model that reads tokens is read for one number of them,
    and that is the most it is ever given on Kaimei. ``max_length`` says how many: it sets the token axis a
    file leaves open, and fills an example up to it (or cuts it down).

    The check sent with the spec holds what the ONNX model itself gives (by ONNX Runtime when it is installed,
    otherwise by ONNX's reference evaluator) on a few rows, with ``check_rows`` among them for a table.
    Pass the result to :func:`kaimei.structure.finalize_spec`, as ``client.models.upload`` does.
    """
    tools = _importer()
    numpy = _require_numpy()
    labels = label_names(labels)
    try:
        model = tools.load_model(source)
    except Exception as exc:  # noqa: BLE001 - a file that is not a model, in any of the ways it can fail to be one
        raise StructureError(f"This is not an ONNX model Kaimei can read: {_brief(exc)}", code="invalid_structure") from exc
    base_dir = os.path.dirname(os.fspath(source)) if isinstance(source, (str, os.PathLike)) else ""
    shapes = _by_input(tools, model, input_shapes, "input_shapes")
    examples = _by_input(tools, model, samples, "samples")
    open_tokens = _open_token_inputs(tools, model)
    if max_length is not None:
        most = max_length = _whole_length(max_length)
        if examples:
            fitted = _fit_tokens(
                {key: numpy.asarray(value) for key, value in examples.items()}, inputs or {}, _pad_token(tokenizer, inputs), most, numpy
            )
            examples = {**examples, **fitted}
        shapes = dict(shapes or {})
        for input_name, fixed in open_tokens.items():
            if input_name in shapes or input_name in (examples or {}):
                continue
            if fixed is not None and fixed != most:
                raise StructureError(
                    f"This file reads exactly {fixed:,} tokens (its input '{input_name}'), and max_length cannot change that.",
                    code="invalid_structure",
                    param="max_length",
                )
            if fixed is None:
                shapes[input_name] = [1, most]
    spec = _read_onnx(
        model,
        library=None,
        feature_names=feature_names,
        task=task,
        labels=labels,
        inputs=inputs,
        input_shapes=shapes or None,
        samples=examples,
        output=output,
        mask_layout=mask_layout,
        threshold=threshold,
        name=name,
        tokenizer=tokenizer,
        check_rows=check_rows,
        framework=framework,
        base_dir=base_dir,
    )
    if any(fixed is None for fixed in open_tokens.values()):
        _say_token_room(spec, None, max_length)
    from .client import __version__

    described = {
        "extracted_with": f"kaimei-python-client {__version__}",
        "model_class": "onnx.ModelProto",
        "route": "onnx",
    }
    full = extraction_metadata(model, spec)
    for key in ("extracted_at", "python"):
        described[key] = full[key]
    spec["metadata"] = {**described, **dict(spec.get("metadata") or {})}
    return spec


def _whole(value: Any) -> bool:
    """Whether ``value`` is a whole number of any kind (Python's, NumPy's), and not a flag."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _whole_length(max_length: Any) -> int:
    if not _whole(max_length) or int(max_length) < 1:
        raise StructureError(
            "max_length is how many tokens the model reads at most: a whole number, 1 or more.",
            code="invalid_structure",
            param="max_length",
        )
    return int(max_length)


def _open_token_inputs(tools: Any, model: Any) -> Dict[str, Optional[int]]:
    """The file's inputs that hold a row of whole numbers, with their length: ``None`` where the file leaves it open."""
    found: Dict[str, Optional[int]] = {}
    for info in tools._input_infos(model):
        kind = info.type.tensor_type
        if int(kind.elem_type) in (1, 10, 11, 14, 15, 16, 8) or not kind.HasField("shape") or len(kind.shape.dim) != 2:
            continue  # floats, text, or not one row of tokens
        axis = kind.shape.dim[1]
        found[str(info.name)] = int(axis.dim_value) if axis.HasField("dim_value") and axis.dim_value > 0 else None
    return found


def _token_part(name: str, hint: Any) -> str:
    """What a whole-number input holds, as the importer tells: what the caller said, or what its name says."""
    if isinstance(hint, Mapping) and hint.get("part"):
        return str(hint["part"])
    lowered = name.lower()
    if "mask" in lowered:
        return "mask"
    if "type" in lowered or "segment" in lowered:
        return "type_ids"
    if "position" in lowered:
        return "positions"
    if "length" in lowered or "len" in lowered:
        return "length"
    return "ids"


def _pad_token(tokenizer: Any, hints: Optional[Mapping[str, Any]] = None) -> int:
    """The id a text shorter than the model reads is filled up with: the caller's, the tokenizer's, or 0."""
    for hint in (hints or {}).values():
        if isinstance(hint, Mapping) and hint.get("pad_id") is not None:
            return int(hint["pad_id"])
    if isinstance(tokenizer, Mapping):
        vocab, pad = tokenizer.get("vocab"), tokenizer.get("pad")
        if isinstance(vocab, (list, tuple)) and pad in vocab:
            return int(list(vocab).index(pad))
        return 0
    for read in (
        lambda: tokenizer.pad_token_id,
        lambda: (tokenizer.padding or {})["pad_id"],
    ):
        try:
            found = read()
        except Exception:  # noqa: BLE001 - a tokenizer without one, in whichever way it says so
            continue
        if isinstance(found, int) and not isinstance(found, bool) and found >= 0:
            return int(found)
    return 0


def _fit_tokens(arrays: Mapping[str, Any], hints: Mapping[str, Any], pad: int, most: int, numpy: Any) -> Dict[str, Any]:
    """Example inputs with their token axis made ``most`` long. Only the inputs that change are returned.

    Token ids are filled up with the padding token, a mask of real tokens with zeros, segment ids with zeros,
    and positions are counted on. An example longer than ``most`` is cut.
    """
    rows = {
        name: numpy.asarray(value)
        for name, value in arrays.items()
        if numpy.asarray(value).ndim == 2 and numpy.asarray(value).dtype.kind in "iub"
    }
    parts = {name: _token_part(name, hints.get(name)) for name in rows}
    ids = next((name for name in rows if parts[name] == "ids"), None)
    if ids is None:
        return {}
    length = int(rows[ids].shape[1])
    fitted: Dict[str, Any] = {}
    for name, value in rows.items():
        if int(value.shape[1]) != length or length == most or parts[name] == "length":
            continue
        if length > most:
            fitted[name] = value[:, :most]
            continue
        if parts[name] == "positions":
            tail = numpy.tile(numpy.arange(length, most), (value.shape[0], 1))
        else:
            tail = numpy.full((value.shape[0], most - length), pad if parts[name] == "ids" else 0)
        fitted[name] = numpy.concatenate([value, tail.astype(value.dtype)], axis=1)
    return fitted


def _context_length(model: Any) -> Optional[int]:
    """How many tokens the model itself takes, where it says: the usual names a configuration has for it."""
    for holder in (getattr(model, "config", None), model):
        for name in ("n_positions", "max_position_embeddings", "n_ctx", "block_size", "context_length", "max_seq_len"):
            try:
                value = getattr(holder, name, None) if holder is not None else None
            except Exception:  # noqa: BLE001 - a property that raises says nothing
                value = None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
    return None


def _say_token_room(spec: Mapping[str, Any], model: Any, asked: Optional[int]) -> None:
    """Say how many tokens a language model was read for, when nobody chose: it stops writing when they are used up."""
    if asked is not None or str(spec.get("task") or "") != "text_generation":
        return
    room = next(
        (
            int(feature["max_length"])
            for feature in spec.get("features") or []
            if str(feature.get("type")) in ("text", "tokens") and feature.get("max_length")
        ),
        None,
    )
    if room is None:
        return
    context = _context_length(model) if model is not None else None
    if context is not None and room >= context:
        return
    more = f" The model itself takes up to {context:,}." if context else ""
    warnings.warn(
        f"This language model is read for {room:,} tokens, the length of the example it was given: a prompt and "
        "what the model writes after it share them, so on Kaimei it stops writing when they are used up. Pass "
        f"max_length= to read it for more.{more}",
        UserWarning,
        stacklevel=2,
    )


def to_onnx(estimator: Any, **options: Any) -> bytes:
    """The ONNX model this client converts ``estimator`` to, as bytes: the file the model's agent can run.

    Takes what :func:`via_onnx` takes (``sample``, ``input_shape``, ``feature_names``, ``preprocess``,
    ``max_length``, ...) and makes the conversion an upload with those arguments makes. An agent started
    with the file (``kaimei.agent.serve(client, model, onnx=...)``) answers predictions with it and, where the
    file's tensors have the names the uploaded structure's steps were read from, says what every step
    computed. PyTorch's exporter names a module's tensors the same way every time. scikit-learn's converter
    numbers some of its tensors anew on each conversion, so a second conversion is not the file a structure
    was read from: for that, keep the file of the upload itself (``save_onnx=`` on ``client.models.upload``, or
    :attr:`kaimei.structure.Extraction.onnx`). An ONNX model given as ``estimator`` is returned as its bytes.
    """
    if is_onnx(estimator):
        if isinstance(estimator, (bytes, bytearray, memoryview)):
            return bytes(estimator)
        if isinstance(estimator, (str, os.PathLike)):
            with open(os.fspath(estimator), "rb") as handle:
                return handle.read()
        return bytes(estimator.SerializeToString())
    kept: Dict[str, Any] = {}
    reasons: List[str] = []
    spec = via_onnx(estimator, keep=kept, reasons=reasons, **options)
    if spec is None or kept.get("onnx") is None:
        why = said_together(reasons) or "no converter takes this kind of model"
        raise StructureError(f"This model could not be converted to ONNX: {why}.", code="unsupported_estimator")
    return bytes(kept["onnx"])


def _by_input(tools: Any, model: Any, given: Any, what: str) -> Optional[Dict[str, Any]]:
    """``given`` by input name. One shape or one array stands for the model's only input."""
    if given is None:
        return None
    if isinstance(given, Mapping):
        return {str(key): value for key, value in given.items()}
    names = [str(info.name) for info in tools._input_infos(model)]
    if len(names) != 1:
        raise StructureError(
            f"This model has {len(names)} inputs ({', '.join(names[:6])}), so {what} has to say which is which: "
            "pass a dict by input name.",
            code="invalid_structure",
            param=what,
        )
    if what == "input_shapes":
        dims = [int(dim) for dim in given]
        declared = len(tools._input_infos(model)[0].type.tensor_type.shape.dim)
        return {names[0]: dims if len(dims) == declared or not declared else [1, *dims]}
    return {names[0]: given}


def _brief(exc: BaseException, limit: int = 300) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _prediction_output(
    model: Any, wanted: Any, task: Optional[str] = None, defer: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """``(the output to ask the importer for, the tensor that holds its numbers)``.

    Which output the prediction is read from when the caller does not say: the first one that holds floats,
    or the last when a classical-ML operator made them (a converted classifier gives its label first and its
    probabilities second). A ``ZipMap`` output is a list of dictionaries; the tensor before it is read.
    """
    producers = {str(name): node for node in model.graph.node for name in node.output if name}

    def tensor_of(name: str) -> str:
        node = producers.get(name)
        return str(node.input[0]) if node is not None and node.op_type == "ZipMap" else name

    outputs = [info for info in model.graph.output]
    if wanted is not None:
        if _whole(wanted):
            wanted = int(wanted)
            if not 0 <= wanted < len(outputs):
                raise StructureError(
                    f"output is {wanted}, and this model has {len(outputs)} outputs.", code="invalid_structure", param="output"
                )
            wanted = str(outputs[wanted].name)
        wanted = str(wanted)
        if wanted not in producers and wanted not in {str(info.name) for info in outputs}:
            listed = ", ".join(str(info.name) for info in outputs[:8])
            raise StructureError(
                f"This model has no output called '{wanted}'. Its outputs are: {listed}.",
                code="invalid_structure",
                param="output",
            )
        return wanted, tensor_of(wanted)
    floats: List[Tuple[str, str]] = []
    for info in outputs:
        name = str(info.name)
        tensor = tensor_of(name)
        if tensor != name:
            floats.append((name, tensor))
        elif info.type.HasField("tensor_type") and int(info.type.tensor_type.elem_type) in (1, 10, 11):
            floats.append((name, tensor))
        elif not info.type.HasField("tensor_type") or not int(info.type.tensor_type.elem_type):
            return None, None  # the file does not say what this output is: the importer finds out by running it
    if task == "segmentation":
        return _mask_output(outputs, tensor_of, defer)
    if not floats:
        return None, None
    classical = any(
        producers.get(tensor) is not None and producers[tensor].domain == "ai.onnx.ml" for _name, tensor in floats
    )
    return floats[-1] if len(floats) > 1 and classical else floats[0]


#: ONNX's number types that hold floats, and those that hold whole numbers or flags.
_FLOAT_TYPES = (1, 10, 11)
_WHOLE_TYPES = (2, 3, 4, 5, 6, 7, 9, 12, 13)


def _declared_mask(info: Any) -> Optional[bool]:
    """Whether an output the file declares can be a segmentation's mask, as the importer's ``can_hold_a_mask`` tells
    of the tensor itself: two or three axes once the row axis is dropped (a height and a width, with or without an
    axis of classes). The row axis is a leading axis of one, or one the file leaves open. ``None`` when the file
    does not say what shape the output has."""
    kind = info.type.tensor_type
    if not kind.HasField("shape"):
        return None
    dims = [int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else None for dim in kind.shape.dim]
    if len(dims) >= 2 and dims[0] in (1, None):
        dims = dims[1:]
    return len(dims) in (2, 3)


def _mask_output(
    outputs: Sequence[Any], tensor_of: Callable[[str], str], defer: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """Which output of a segmentation model its mask is read from, when the caller does not say: the rule of the
    importer's own ``_mask_output``, told from what the file declares.

    The first output of floats that can hold a mask: a model that gives its scores and something more (torchvision's
    FCN and DeepLab give ``out`` and ``aux``) is read from the first. An output of whole numbers or of flags only
    when no output of floats can hold one. A model with one numeric output is read from it whatever its shape, and
    several outputs none of which can hold a mask are refused. ``(None, None)`` leaves the choice to the importer,
    which runs the model: when the file does not say what shape an output has, or has no numeric output.
    """
    numeric = [
        (str(info.name), "f" if int(info.type.tensor_type.elem_type) in _FLOAT_TYPES else "i", _declared_mask(info))
        for info in outputs
        if info.type.HasField("tensor_type") and int(info.type.tensor_type.elem_type) in _FLOAT_TYPES + _WHOLE_TYPES
    ]
    if not numeric or any(able is None for _name, _kind, able in numeric):
        return None, None
    if defer and sum(1 for _name, _kind, able in numeric if able) > 1:
        # Several could hold the mask. The importer runs the model and prefers the one that is the picture's size
        # (then a whole fraction or multiple of it, then the first): what the file declares cannot tell. Only for
        # a caller that reads the importer's choice back (``defer``): a conversion that compares the library's own
        # answer has to know which output before the importer runs, and takes the first, on both sides.
        return None, None
    for wanted in ("f", "i"):
        for name, kind, able in numeric:
            if able and kind == wanted:
                return name, tensor_of(name)
    if len(numeric) == 1:
        return numeric[0][0], tensor_of(numeric[0][0])
    listed = ", ".join(f"'{name}'" for name, _kind, _able in numeric[:6])
    raise StructureError(
        f"None of this model's outputs can hold a mask: {listed}. A mask is a score for each class at each pixel, or "
        "a class number for each pixel. Say which output to read with output=.",
        code="invalid_structure",
        param="output",
    )


def _trailing_empties(graph: Any) -> bool:
    """Whether a node of ``graph`` (a graph or a function), or of a graph inside it, lists inputs left out at its
    end under empty names."""
    for node in graph.node:
        if len(node.input) and node.input[-1] == "":
            return True
        for attribute in node.attribute:
            bodies = [attribute.g] if attribute.HasField("g") else []
            if any(_trailing_empties(body) for body in [*bodies, *attribute.graphs]):
                return True
    return False


def without_trailing_empties(model: Any) -> Any:
    """``model``, or a copy with every input left out at the end of a node's list taken off that list.

    An optional input at the end may be listed under an empty name or not at all, and both mean the same.
    ONNX Runtime 1.32 stops the whole process on some nodes written the first way (an opset-20 ``DFT``), so
    a model is given to it written the second: in its graph, in the branches and loop bodies inside it, and in
    the functions the file defines itself. The model passed in is left as it is.
    """
    functions = list(getattr(model, "functions", None) or [])
    if not (_trailing_empties(model.graph) or any(_trailing_empties(function) for function in functions)):
        return model
    copy = type(model)()
    copy.CopyFrom(model)

    def strip(graph: Any) -> None:
        # A graph or a function: both keep their steps in ``node``.
        for node in graph.node:
            while len(node.input) and node.input[-1] == "":
                del node.input[-1]
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    strip(attribute.g)
                for body in attribute.graphs:
                    strip(body)

    strip(copy.graph)
    for function in copy.functions:
        strip(function)
    return copy


#: What ONNX Runtime says when it has no kernel for a step at the types it is given: its NOT_IMPLEMENTED status
#: ("Could not find an implementation for ..."), and what a kernel says of a type it does not compute (a float64
#: LSTM is "LSTM operator does not support double yet", raised when the step runs, not when the session is made).
#: Not every refusal that calls something unsupported: an input of the wrong type or shape is the input's fault.
_ONNXRUNTIME_LACKS = re.compile(
    r"NOT_IMPLEMENTED|not implemented|could not find an implementation|find kernel|kernel for|operator does not support",
    re.IGNORECASE,
)


def _onnxruntime_lacks_a_kernel(exc: BaseException) -> bool:
    """Whether ONNX Runtime refused a run because it cannot compute a step, not because of the model or the input."""
    kind = type(exc)
    if not str(getattr(kind, "__module__", "")).startswith("onnxruntime"):
        return False
    return kind.__name__ == "NotImplemented" or bool(_ONNXRUNTIME_LACKS.search(str(exc)))


class _OnnxRunner:
    """Runs an ONNX model for one row: ONNX Runtime when it is installed, ONNX's own evaluator otherwise.

    ``reference_only`` uses the evaluator even when ONNX Runtime is there: a model that draws at random draws from
    the evaluator's generator, which is the one Kaimei's kernels share, and from another one in ONNX Runtime.
    The evaluator also takes over when ONNX Runtime cannot make a session of the model, or has no kernel for one
    of its steps at the types it is run with (it runs no float64 LSTM); any other failure of a run is the model's.
    """

    def __init__(self, tools: Any, model: Any, tensor: str, reference_only: bool = False) -> None:
        self.tensor = tensor
        self.session: Any = None
        self.evaluator: Any = None
        self.evaluator_refused = False
        self.engine = "onnx reference evaluator"
        # Both engines are given the model with no inputs left out under empty names (see without_trailing_empties).
        self.model = without_trailing_empties(model)
        onnx = tools._onnx()
        try:
            import onnxruntime as ort
        except ImportError:
            ort = None
        if reference_only:
            ort = None
        if ort is not None:
            try:
                probe = onnx.ModelProto()
                probe.CopyFrom(self.model)
                if tensor not in {info.name for info in probe.graph.output}:
                    probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(tensor))
                options = ort.SessionOptions()
                options.log_severity_level = 3
                self.session = ort.InferenceSession(
                    probe.SerializeToString(), options, providers=["CPUExecutionProvider"]
                )
                self.engine = f"onnxruntime {ort.__version__}"
            except Exception:  # noqa: BLE001 - whatever ONNX Runtime will not load, the evaluator below may
                self.session = None
        if self.session is None:
            self._use_the_evaluator()

    def _use_the_evaluator(self) -> None:
        from onnx.reference import ReferenceEvaluator

        # Built before anything is given up: a model the evaluator cannot take leaves the runner as it was.
        evaluator = ReferenceEvaluator(self.model)
        self.evaluator = evaluator
        self.session = None
        self.engine = "onnx reference evaluator"

    def __call__(self, feeds: Dict[str, Any]) -> Any:
        if self.session is not None:
            try:
                return self.session.run([self.tensor], feeds)[0]
            except Exception as exc:  # noqa: BLE001 - told apart just below
                if self.evaluator_refused or not _onnxruntime_lacks_a_kernel(exc):
                    raise
                try:
                    self._use_the_evaluator()
                except Exception:  # noqa: BLE001 - the evaluator cannot take this model either
                    # ONNX Runtime stays the engine, and its own refusal is the reason this row was not run.
                    self.evaluator_refused = True
                    raise exc from None
        found = self.evaluator.run(None, feeds, intermediate=True)
        return found[self.tensor]


def _read_onnx(
    model: Any,
    *,
    library: Optional[Callable[[Dict[str, Any]], Any]],
    feature_names: Optional[Sequence[str]],
    task: Optional[str],
    labels: Optional[Sequence[Any]],
    inputs: Optional[Mapping[str, Any]],
    input_shapes: Optional[Mapping[str, Any]],
    samples: Optional[Mapping[str, Any]],
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    output: Any,
    name: Optional[str],
    tokenizer: Any,
    check_rows: Any,
    framework: str,
    base_dir: str = "",
    more_samples: Optional[List[Dict[str, Any]]] = None,
    rename: Optional[Mapping[str, str]] = None,
    vocabulary: Optional[int] = None,
    probabilities: bool = False,
    dropped: Sequence[str] = (),
    whole_table: Any = None,
) -> Dict[str, Any]:
    """Import ``model`` and give it a check that holds ``library``'s outputs (or the ONNX model's own).

    ``library(feeds)`` runs the original model on one row of inputs, by ONNX input name, and returns the
    tensor the prediction is read from. ``more_samples`` are further example rows to check on, and ``rename``
    gives features the names the caller knows them by. ``probabilities`` says the library's answer is a
    classifier's probabilities (``predict_proba``), which the head shows as they are. ``inputs`` must name
    inputs of ``model``, or of the original model's inputs that its conversion left out (``dropped``).
    ``whole_table`` is every row of the check of a model that reads a table, as doubles, a column for each
    feature in the order the model reads them: no row is made up here then, and the example's own is left out.
    ``threshold`` is the importer's: where a segmentation's one score for each pixel is cut (its
    ``head.threshold``, which is no classifier's ``outputs.threshold``).
    """
    tools = _importer()
    numpy = _require_numpy()
    if mask_layout is not None and task is None:
        task = "segmentation"
    wanted, tensor = _prediction_output(model, output, task, defer=True)
    hints = {str(key): dict(value) if isinstance(value, Mapping) else value for key, value in (inputs or {}).items()}
    present = [str(info.name) for info in tools._input_infos(model)]
    unknown = [key for key in hints if key not in present and key not in dropped]
    if unknown:
        # The importer would pass over a name it does not know, and the input would be read as if nothing were said.
        raise StructureError(
            f"inputs names '{unknown[0]}', which is not an input of this model ({', '.join(present[:8])}).",
            code="invalid_structure",
            param="inputs",
        )
    # An input the conversion left out (the prediction does not depend on it) needs nothing said of it.
    hints = {key: value for key, value in hints.items() if key in present}
    reference: Dict[str, Any] = {}
    try:
        spec = tools.import_onnx(
            model,
            feature_names=feature_names,
            task=task,
            labels=[str(label) for label in labels] if labels else None,
            inputs=hints,
            input_shapes=dict(input_shapes) if input_shapes else None,
            samples=dict(samples) if samples else None,
            output=wanted,
            mask_layout=mask_layout,
            threshold=threshold,
            framework=framework,
            name=name,
            check=tensor is None,
            base_dir=base_dir,
            reference=reference,
        )
    except StructureError:
        raise
    except Exception as exc:  # noqa: BLE001 - the importer says what it could not do in many ways
        if getattr(exc, "param", None) == "threshold":
            # The caller's own argument, refused as the importer says it: the model was read well enough.
            raise StructureError(str(exc), code="invalid_structure", param="threshold") from exc
        raise StructureError(
            f"This model could not be read as an ONNX graph: {_brief(exc)}",
            code="unsupported_estimator",
            param=getattr(exc, "param", None),
        ) from exc
    _carry_what_json_cannot(spec, numpy)
    head = str(spec["head"]["output"])
    first = reference.get("values", {}).get(head)
    if tensor is None and spec["task"] == "segmentation":
        # The importer chose which output the mask is read from, and says which.
        if reference.get("output"):
            tensor = str(reference["output"])
        elif len(model.graph.output) == 1:
            tensor = str(model.graph.output[0].name)
    if tensor is None or first is None:
        if library is not None:
            # Nothing here says which of the file's outputs the library's answer is, so the two were never compared,
            # and the importer's own check holds the conversion's outputs, not the model's.
            spec.pop("check", None)
            _drawn(spec, NOT_COMPARED + " (which of the conversion's outputs is the model's answer is not known)")
        # Otherwise the file does not say which output is which, so the importer's own check stands as it is.
        return spec
    first = numpy.asarray(first)
    if probabilities and str(spec.get("task") or "") in _CLASSIFIERS:
        # The library's own numbers are probabilities already: the head shows them as they are, never squashed again.
        spec["head"]["link"] = "none"
    originals = [str(info.name) for info in tools._input_infos(model)]
    feeds0 = dict(zip(originals, reference["feeds"].values()))
    named = dict(zip(reference["feeds"].keys(), originals))
    _settle_token_head(spec, first, numpy)
    _name_classes(spec, first)
    for old, new in (rename or {}).items():
        _rename_feature(spec, str(old), str(new))
    _use_tokenizer(spec, tokenizer, hints, named)
    bindings = _bindings(spec, named, feeds0, numpy)
    _expressible(bindings, feeds0, numpy)
    _labels_fit(spec, labels)
    # Whether it draws is decided on the graph the importer read: the file's own functions written out where they
    # are used, so a random step inside one counts, as does one inside a branch or a loop.
    expanded = _expanded(model)
    draws = bool(tools._draws(expanded.graph))
    if draws:
        # A step that draws: the original draws from no seed, so nothing can be compared with it. The check is
        # the model with the seeds the importer gave its steps, run by the evaluator whose draws Kaimei repeats.
        seeded = tools.fold_constants(expanded, base_dir or "")
        tools._pin_seeds(seeded.graph)
        converted = _OnnxRunner(tools, seeded, tensor, reference_only=True)
        library = None
    else:
        converted = _OnnxRunner(tools, model, tensor)
    limits = _vocabulary_limits(hints)
    rows = _check_rows(spec, bindings, feeds0, more_samples or [], check_rows, limits, vocabulary, numpy, whole=whole_table)
    reasons: List[str] = []
    check, difference = _checked_against(spec, bindings, library, converted, rows, first.shape, numpy, why=reasons)
    spec.pop("check", None)
    metadata = spec.setdefault("metadata", {})
    if draws:
        # Said where a reader of the structure finds it: the original model was not, and cannot be, compared.
        metadata["compared_with_original"] = False
        spec["summary"] = f"{spec.get('summary') or ''} {DRAWS_AT_RANDOM}".strip()
    if check is None or difference > AGREES_WITHIN:
        if check is None and library is None:
            note = "Its outputs could not be computed here to check it against"
            if reasons:
                note = f"{note} ({reasons[0]})"
        elif check is None:
            note = NOT_COMPARED + (f" ({reasons[0]})" if reasons else "")
        else:
            note = (
                f"Its ONNX conversion does not reproduce the model's own outputs (they differ by {difference:.1e} "
                "of their size)"
            )
        _drawn(spec, note)
    else:
        spec["check"] = check
        if library is None and not draws:
            _its_label_is_kaimeis_class(spec, tools, model, tensor, bindings, numpy)
    if draws:
        metadata["checked_against"] = f"{converted.engine}, with fixed seeds"
    else:
        metadata["checked_against"] = converted.engine if library is None else str(framework)
    return spec


#: Why a conversion is drawn when nothing could be compared with the model it was converted from.
NOT_COMPARED = "It could not be compared with the model it was converted from"
#: What the check of a model that draws at random holds, said in its summary.
DRAWS_AT_RANDOM = (
    "It draws numbers at random, and its original draws differently on every run, so its check compares Kaimei "
    "with the ONNX model run with fixed seeds, not with the original model's own outputs."
)
_CLASSIFIERS = ("binary_classification", "multiclass_classification", "multilabel_classification")


def _own_label_output(model: Any, tensor: str) -> Optional[str]:
    """The output of an ONNX classifier that holds its own answer for a row, the class it predicts: its one output
    of whole numbers or of text, with one value for a row, besides the scores Kaimei reads. A classical model
    converted to ONNX gives its label beside its probabilities this way. ``None`` when there is no such output, or
    more than one."""
    found: List[str] = []
    for info in model.graph.output:
        kind = info.type.tensor_type if info.type.HasField("tensor_type") else None
        if kind is None or str(info.name) == tensor or int(kind.elem_type) not in (6, 7, 8):
            continue  # not whole numbers (int32, int64) or text
        dims = list(kind.shape.dim) if kind.HasField("shape") else None
        one_for_a_row = dims is None or len(dims) == 1 or (len(dims) == 2 and dims[1].HasField("dim_value") and dims[1].dim_value == 1)
        if one_for_a_row:
            found.append(str(info.name))
    return found[0] if len(found) == 1 else None


def _file_classes(model: Any, count: int) -> Optional[List[str]]:
    """The classes an ONNX file's own classifier step lists, in the order of its scores (``classlabels_ints``,
    ``classlabels_int64s`` or ``classlabels_strings`` of an ``ai.onnx.ml`` operator), as text. ``None`` unless
    the file's steps agree on one list of ``count`` classes."""
    found = set()
    for node in model.graph.node:
        if str(node.domain) != "ai.onnx.ml":
            continue
        for attribute in node.attribute:
            if attribute.name in ("classlabels_ints", "classlabels_int64s") and len(attribute.ints) == count:
                found.add(tuple(str(int(value)) for value in attribute.ints))
            elif attribute.name == "classlabels_strings" and len(attribute.strings) == count:
                found.add(tuple(value.decode("utf-8", "replace") for value in attribute.strings))
    return list(next(iter(found))) if len(found) == 1 else None


def _its_label_is_kaimeis_class(
    spec: Dict[str, Any], tools: Any, model: Any, tensor: str, bindings: Sequence[Mapping[str, Any]], numpy: Any
) -> None:
    """An ONNX file that gives its own label for a row is the model only if Kaimei's class is that label.

    The file's scores are what the check holds, and its label is its own ``predict``: a classifier converted from
    scikit-learn labels a row by its votes, its margin or its threshold, which the highest of its probabilities
    does not always follow. The two are compared on every row of the check, where the file's labels are the
    structure's class names (they are, when the names were read from the file), and a reading whose class
    differs on any of them is drawn, not run.
    """
    if str(spec.get("task") or "") not in ("binary_classification", "multiclass_classification"):
        return
    label = _own_label_output(model, tensor)
    check = spec.get("check") or {}
    names = _class_names(spec)
    if label is None or not names:
        return
    # The file's own classes, in the order of its scores, where a classical classifier's step lists them; the
    # structure's class names otherwise.
    classes = _file_classes(model, len(names))
    known = set(classes if classes is not None else names)
    try:
        rows = _check_rows_of(spec, check, len(check.get("outputs") or []), numpy) or []
        runner = _OnnxRunner(tools, model, label)
        answers: List[str] = []
        for row in rows:
            value = numpy.asarray(runner(_feeds(bindings, row, numpy)), dtype=object).reshape(-1)
            if value.size != 1:
                return
            answers.append(value[0].decode("utf-8", "replace") if isinstance(value[0], bytes) else str(value[0]))
    except Exception:  # noqa: BLE001 - a label that cannot be computed here says nothing either way
        return
    if not answers or not set(answers) <= known:
        return  # the classes were named by the caller: which label is which name is not known here
    differ, count = classes_differ(spec, answers, classes)
    if differ:
        not_its_class(spec, differ, count, "the file's own label for a row is not always the highest of its scores")


def _drawn(spec: Dict[str, Any], why: str) -> None:
    """Mark a reading drawn, not run, and say why in its summary."""
    spec["approximate"] = True
    spec["summary"] = f"{spec.get('summary') or ''} {why}, so it is drawn, not run.".strip()


def _expanded(model: Any) -> Any:
    """``model`` with the operators it defines itself (its local functions) written out where they are used, as the
    importer reads it. ``onnx.inliner`` is imported by its own name there: ``import onnx`` does not load it."""
    from .onnx_reader import expand_functions

    return expand_functions(model)


# --- what the importer leaves to its caller --------------------------------------------------


def _bodies(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """A graph and every graph nested in the attributes of its nodes."""
    found: List[Dict[str, Any]] = []
    stack: List[Any] = [graph]
    while stack:
        body = stack.pop()
        if not isinstance(body, dict):
            continue
        found.append(body)
        for node in body.get("nodes") or []:
            for value in (node.get("attrs") or {}).values() if isinstance(node, dict) else []:
                if isinstance(value, dict) and isinstance(value.get("graph"), dict):
                    stack.append(value["graph"])
                elif isinstance(value, dict) and isinstance(value.get("graphs"), list):
                    stack.extend(value["graphs"])
    return found


def _carry_what_json_cannot(spec: Dict[str, Any], numpy: Any) -> None:
    """Move NaN and infinity out of operator attributes, where JSON cannot carry them, into constants.

    Two operators keep such a value in an attribute: ``ConstantOfShape`` (a tensor filled with ``-inf``, for
    masking) and ``Imputer`` (replace NaN). Each is rewritten as the same arithmetic on a constant, which
    travels as bytes. Any other attribute that holds one is refused, by name.
    """
    for body in _bodies(spec.get("graph") or {}):
        constants = body.setdefault("constants", {})
        taken = set(constants)
        for node in body.get("nodes") or []:
            for name in node.get("outputs") or []:
                taken.add(str(name))
        rewritten: List[Dict[str, Any]] = []
        for node in body.get("nodes") or []:
            attrs = node.get("attrs") or {}
            if not _unsendable(attrs, numpy):
                rewritten.append(node)
                continue
            op, domain = str(node.get("op")), str(node.get("domain") or "")
            value = attrs.get("value")
            if op == "ConstantOfShape" and domain == "" and isinstance(value, dict) and len(value.get("data") or []) == 1:
                fill = _free_name(f"{node.get('id')}.value", taken)
                constants[fill] = numpy.asarray(value["data"][0], dtype=str(value.get("dtype") or "float32"))
                node.update({"op": "Expand", "inputs": [fill, node["inputs"][0]], "attrs": {}})
                rewritten.append(node)
                continue
            replaced = attrs.get("replaced_value_float")
            floats = attrs.get("imputed_value_floats")
            if op == "Imputer" and isinstance(replaced, float) and replaced != replaced and floats and _all_finite(floats):
                fill = _free_name(f"{node.get('id')}.imputed", taken)
                gap = _free_name(f"{node.get('id')}.gap", taken)
                same = _free_name(f"{node.get('id')}.fill", taken)
                constants[fill] = numpy.asarray(floats, dtype=numpy.float32)
                source = node["inputs"][0]
                base = {key: node[key] for key in ("path", "hidden") if key in node}
                rewritten.append(
                    {"id": gap, "op": "IsNaN", "domain": "", "inputs": [source], "outputs": [gap], "attrs": {}, "hidden": True}
                )
                rewritten.append(
                    {
                        "id": same,
                        "op": "CastLike",
                        "domain": "",
                        "inputs": [fill, source],
                        "outputs": [same],
                        "attrs": {},
                        "hidden": True,
                    }
                )
                rewritten.append(
                    {
                        **base,
                        "id": node["id"],
                        "op": "Where",
                        "domain": "",
                        "inputs": [gap, same, source],
                        "outputs": list(node["outputs"]),
                        "attrs": {},
                        "label": "Imputer",
                    }
                )
                continue
            raise StructureError(
                f"Step '{node.get('id')}' ({op}) keeps NaN or infinity in its attributes, which a structure cannot "
                "carry.",
                code="unsupported_layers",
            )
        body["nodes"] = rewritten


def _free_name(wanted: str, taken: set) -> str:
    name, count = wanted, 1
    while name in taken:
        count += 1
        name = f"{wanted}_{count}"
    taken.add(name)
    return name


def _all_finite(values: Any) -> bool:
    return all(isinstance(item, (int, float)) and item == item and abs(item) != float("inf") for item in values)


def _unsendable(value: Any, numpy: Any, depth: int = 0) -> bool:
    """Whether an attribute holds NaN or infinity. Nested graphs are looked at on their own."""
    if isinstance(value, float):
        return value != value or abs(value) == float("inf")
    if isinstance(value, numpy.ndarray):
        return _not_finite(value, numpy)
    if isinstance(value, dict):
        if "graph" in value or "graphs" in value or depth > 4:
            return False
        return any(_unsendable(item, numpy, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_unsendable(item, numpy, depth + 1) for item in value)
    return False


def _settle_token_head(spec: Dict[str, Any], first: Any, numpy: Any) -> None:
    """A language model's scores become probabilities unless they already are.

    The importer decides this for classifiers only; left at "none", the API would report scores as
    probabilities.
    """
    if spec.get("task") not in ("text_generation", "sequence_to_sequence") or str(spec["head"].get("link") or "none") != "none":
        return
    scores = numpy.asarray(first, dtype=float)
    last = scores.reshape(-1, scores.shape[-1])[-1] if scores.ndim else scores.reshape(-1)
    already = bool(last.min() >= -1e-6 and abs(float(last.sum()) - 1.0) <= 1e-4)
    spec["head"]["link"] = "none" if already else "softmax"


def _name_classes(spec: Dict[str, Any], first: Any) -> None:
    """A classifier nobody named the classes of gets the names the native readers give: ``class_0``, ``class_1``…"""
    task = str(spec.get("task") or "")
    if task not in ("binary_classification", "multiclass_classification", "multilabel_classification", "clustering", "segmentation"):
        return
    if (spec.get("outputs") or {}).get("labels"):
        return
    width = int(first.size // max(int(first.shape[0]), 1)) if first.ndim >= 2 else int(first.size)
    if task == "segmentation":
        layout = str(spec["head"].get("layout") or "chw")
        dims = [int(dim) for dim in first.shape]
        dims = dims[1:] if len(dims) == 4 else dims
        if layout == "indices" or len(dims) != 3:
            return  # the classes of a mask of numbers are only known to its owner: they name them with labels=
        count = max(2, dims[0] if layout == "chw" else dims[-1])
        spec["outputs"] = {**dict(spec.get("outputs") or {}), "labels": [f"class_{index}" for index in range(count)]}
        return
    if task == "clustering":
        spec["outputs"] = {**dict(spec.get("outputs") or {}), "labels": [f"cluster {index}" for index in range(width)]}
        return
    if task == "binary_classification":
        if width > 2:
            return  # the API will say a binary classifier cannot have this many outputs
        named = ["negative", "positive"]
    else:
        named = [f"class_{index}" for index in range(width)]
    spec["outputs"] = {**dict(spec.get("outputs") or {}), "labels": named}


def _rename_feature(spec: Dict[str, Any], old: str, new: str) -> None:
    """Call a feature what the caller calls it, everywhere the spec names it."""
    names = [str(feature.get("name")) for feature in spec.get("features") or []]
    if old == new or old not in names or new in names:
        return
    taken = set(spec["graph"].get("constants") or {}) | {str(item.get("name")) for item in spec["graph"].get("inputs") or []}
    for node in spec["graph"].get("nodes") or []:
        taken.update(str(name) for name in node.get("outputs") or [])
    if new in taken:
        return  # a tensor already has that name, and a tensor and a feature cannot share one
    for feature in spec["features"]:
        if str(feature.get("name")) == old:
            feature["name"] = new
    spec["feature_names"] = [new if str(item) == old else str(item) for item in spec.get("feature_names") or []]
    for item in spec["graph"].get("inputs") or []:
        source = item.get("from") or {}
        if source.get("feature") == old:
            source["feature"] = new
        if isinstance(source.get("features"), list):
            source["features"] = [new if entry == old else entry for entry in source["features"]]


def _use_tokenizer(spec: Dict[str, Any], tokenizer: Any, hints: Mapping[str, Any], named: Mapping[str, str]) -> None:
    """Give the model's token input its tokenizer, so the feature takes text.

    A feature that reads tokens is ``text`` when Kaimei has its tokenizer and ``tokens`` when it has not:
    the first takes a sentence, the second a list of ids.
    """
    try:
        if tokenizer is not None:
            _describe_tokenizer(spec, tokenizer, hints, named)
    finally:
        known = spec.get("tokenizers") or {}
        for feature in spec.get("features") or []:
            if str(feature.get("type")) in ("tokens", "text"):
                feature["type"] = "text" if str(feature.get("name")) in known else "tokens"


def _describe_tokenizer(spec: Dict[str, Any], tokenizer: Any, hints: Mapping[str, Any], named: Mapping[str, str]) -> None:
    from .text import tokenizer_spec
    from .tokenizers import Tokenizer

    worded = [feature for feature in spec.get("features") or [] if str(feature.get("type")) in ("tokens", "text")]
    if not worded:
        raise StructureError(
            "A tokenizer was given, but none of this model's inputs takes token ids.",
            code="invalid_structure",
            param="tokenizer",
        )
    by_feature = {str(feature["name"]): feature for feature in worded}
    one_description = isinstance(tokenizer, Mapping) and ("type" in tokenizer or "model" in tokenizer)
    if isinstance(tokenizer, Mapping) and tokenizer and not one_description and set(tokenizer) <= set(by_feature):
        # A tokenizer for each text feature, by the feature's name.
        chosen: Dict[str, Any] = {str(key): value for key, value in tokenizer.items()}
    else:
        chosen = {str(worded[0]["name"]): tokenizer}
    for feature_name, given in chosen.items():
        described = tokenizer_spec(given)
        if described is None:
            continue  # said why in a warning: the feature stays a list of token ids
        built = Tokenizer(described)
        feature = by_feature[feature_name]
        feature["type"] = "text"
        feature["vocab_size"] = int(built.size)
        spec.setdefault("tokenizers", {})[feature_name] = described
        if built.pad is None:
            continue
        for item in spec["graph"].get("inputs") or []:
            source = item.get("from") or {}
            hinted = hints.get(named.get(str(item.get("name")), ""))
            said = isinstance(hinted, Mapping) and "pad_id" in hinted
            if source.get("feature") == feature_name and source.get("part") == "ids" and not said:
                source["pad_id"] = int(built.pad)


# --- the check: rows, and what the model itself gives for them -------------------------------


def _bindings(spec: Mapping[str, Any], named: Mapping[str, str], feeds0: Mapping[str, Any], numpy: Any) -> List[Dict[str, Any]]:
    """How each input tensor of the model is made from the features, as the API will make it."""
    bound: List[Dict[str, Any]] = []
    used = set()
    for item in spec["graph"].get("inputs") or []:
        original = named.get(str(item["name"]), str(item["name"]))
        used.add(original)
        shape = tuple(int(dim) for dim in item["shape"])
        axis = item.get("batch_axis")
        if "batch_axis" not in item:
            axis = 0 if shape and shape[0] == 1 else None
        sample = shape if axis is None else shape[:axis] + shape[axis + 1 :]
        source = dict(item.get("from") or {})
        entry: Dict[str, Any] = {
            "name": original,
            "dtype": numpy.dtype(str(item.get("dtype") or "float32")),
            "shape": shape,
            "sample": sample,
            "axis": axis,
            "source": source,
        }
        if "value" in source:
            entry["fixed"] = numpy.broadcast_to(numpy.asarray(feeds0[original]), shape).astype(entry["dtype"])
        elif source.get("zeros") or not source:
            entry["fixed"] = numpy.zeros(shape, dtype=entry["dtype"])
        bound.append(entry)
    for original, value in feeds0.items():
        if original not in used:
            # An input the prediction does not depend on still has to be given to the model.
            array = numpy.asarray(value)
            bound.append({"name": original, "dtype": array.dtype, "shape": array.shape, "source": {}, "fixed": array})
    return bound


def _feeds(bindings: Sequence[Mapping[str, Any]], row: Mapping[str, Any], numpy: Any, exact: bool = True) -> Dict[str, Any]:
    """The model's input tensors for one row of feature values.

    ``exact`` gives each tensor the number type the graph declares, as the API will. Without it floats stay
    double precision, which is what the original library is asked with: a model trained on doubles is then
    compared on the numbers a caller sends, not on those numbers already rounded.
    """
    feeds: Dict[str, Any] = {}
    for binding in bindings:
        if "fixed" in binding:
            feeds[binding["name"]] = binding["fixed"]
            continue
        source, dtype, sample = binding["source"], binding["dtype"], binding["sample"]
        if "features" in source:
            flat = numpy.array([float(numpy.asarray(row[name]).reshape(-1)[0]) for name in source["features"]])
        else:
            values = numpy.asarray(row[source["feature"]], dtype=float).reshape(-1)
            part = source.get("part")
            if part is not None:
                real = values >= 0
                if part == "ids":
                    flat = numpy.where(real, values, float(source.get("pad_id", 0)))
                elif part == "mask":
                    flat = real.astype(float)
                elif part == "type_ids":
                    flat = numpy.zeros_like(values)
                elif part == "positions":
                    flat = numpy.arange(values.size, dtype=float)
                else:
                    flat = numpy.array([float(real.sum())])
            elif str(source.get("layout") or "").lower() in ("nchw", "chw") and len(sample) == 3:
                channels, height, width = sample
                flat = values.reshape(height, width, channels).transpose(2, 0, 1).reshape(-1)
            else:
                flat = values
        if dtype.kind in "iu":
            tensor = numpy.rint(flat).astype(dtype)
        elif dtype.kind == "b":
            tensor = flat != 0
        else:
            tensor = flat.astype(dtype if exact else numpy.float64)
        feeds[binding["name"]] = tensor.reshape(binding["shape"])
    return feeds


def _labels_fit(spec: Mapping[str, Any], labels: Optional[Sequence[Any]]) -> None:
    """Refuse class names that do not fit the output: one for each value, or two for one value of a yes-or-no model."""
    if not labels:
        return
    task = str(spec.get("task") or "")
    if task not in ("binary_classification", "multiclass_classification", "multilabel_classification", "clustering"):
        return
    head = dict(spec.get("head") or {})
    values = dict((spec.get("graph") or {}).get("values") or {})
    shape = [int(dim) for dim in (values.get(str(head.get("output")), {}) or {}).get("shape") or []]
    if not shape:
        return
    width = _product(shape[1:]) if len(shape) >= 2 and shape[0] == 1 else _product(shape)
    if width == 1 and task in ("binary_classification", "multiclass_classification"):
        # One value for a row is a yes-or-no answer: its two classes are named, the negative one first.
        if task == "binary_classification" and len(labels) == 2:
            return
        also = "" if task == "binary_classification" else " with task='binary_classification'"
        raise StructureError(
            f"labels has {len(labels)} name{'s' if len(labels) != 1 else ''}, and this model gives one value for a "
            f"row: a yes-or-no answer, whose two classes are named negative first, then positive (labels=[negative, "
            f"positive]{also}).",
            code="invalid_structure",
            param="labels",
        )
    if len(labels) != width:
        raise StructureError(
            f"labels has {len(labels)} names, and this model gives {width} value{'s' if width != 1 else ''} for a row: "
            "a name for each value (two for a yes-or-no model that gives one).",
            code="invalid_structure",
            param="labels",
        )


def _expressible(bindings: Sequence[Mapping[str, Any]], feeds0: Mapping[str, Any], numpy: Any) -> None:
    """Refuse bindings that cannot give the model its example input.

    Kaimei makes every input tensor from the features. When the example, turned into feature values and back,
    does not give the inputs it was (two inputs read from one feature where the example gave them different
    values, say), the model would be run on inputs it is never given: its reading would be checked, and
    then run, on something else.
    """
    row = _row_of(bindings, feeds0, numpy)
    if row is None:
        return
    again = _feeds(bindings, row, numpy)
    differ: List[str] = []
    for binding in bindings:
        source = binding["source"]
        if "fixed" in binding or source.get("part") not in (None, "ids"):
            continue
        name = binding["name"]
        given = numpy.asarray(feeds0[name], dtype=float).reshape(-1)
        made = numpy.asarray(again[name], dtype=float).reshape(-1)
        if given.size != made.size:
            continue
        if source.get("part") == "ids":
            real = numpy.asarray(row[str(source["feature"])], dtype=float).reshape(-1) >= 0
            if real.size == given.size:
                given, made = given[real], made[real]
        if not numpy.allclose(given, made, rtol=1e-6, atol=1e-6):
            differ.append(name)
    if differ:

        def read_from(binding: Mapping[str, Any]) -> set:
            source = binding["source"]
            return {str(source["feature"])} if "feature" in source else {str(item) for item in source.get("features") or []}

        shared = set().union(*(read_from(b) for b in bindings if b["name"] in differ))
        sharing = [b["name"] for b in bindings if "fixed" not in b and read_from(b) & shared]
        raise StructureError(
            f"The example input cannot be given to this model the way Kaimei gives it: {' and '.join(sharing)} "
            f"{'is' if len(sharing) == 1 else 'are'} read from {', '.join(sorted(shared))}, so they cannot be given the "
            "different values the example has. Pass inputs={...} to say what each of the model's inputs is.",
            code="invalid_structure",
            param="inputs",
        )


def _row_of(bindings: Sequence[Mapping[str, Any]], feeds: Mapping[str, Any], numpy: Any) -> Optional[Dict[str, Any]]:
    """The feature values that give these input tensors: an example input as a row. ``None`` when it is not one."""
    row: Dict[str, Any] = {}
    lengths: Dict[str, int] = {}
    try:
        for binding in bindings:
            source = binding["source"]
            if "fixed" in binding:
                continue
            tensor = numpy.asarray(feeds[binding["name"]])
            if tensor.size != int(numpy.prod(binding["shape"])):
                return None
            one = tensor.reshape(binding["shape"])
            if binding["axis"] is not None:
                one = numpy.take(one, 0, axis=binding["axis"])
            if "features" in source:
                for column, feature in enumerate(source["features"]):
                    row[str(feature)] = numpy.array([float(one.reshape(-1)[column])])
                continue
            feature = str(source["feature"])
            part = source.get("part")
            if part == "mask":
                real = numpy.asarray(one, dtype=float).reshape(-1) != 0
                count = int(real.sum())
                if count and real[:count].all():
                    lengths[feature] = count
            elif part == "length":
                lengths[feature] = int(numpy.asarray(one).reshape(-1)[0])
            elif part in (None, "ids"):
                if str(source.get("layout") or "").lower() in ("nchw", "chw") and one.ndim == 3:
                    one = numpy.moveaxis(one, 0, -1)
                row[feature] = numpy.asarray(one, dtype=float).reshape(-1).copy()
        for feature, count in lengths.items():
            if feature in row and 0 < count < row[feature].size:
                row[feature][count:] = -1.0
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    wanted = {str(feature["name"]) for feature in _features(bindings)}
    return row if wanted <= set(row) else None


def _features(bindings: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every feature the inputs read, with how many values one row of it has and what kind of values."""
    found: Dict[str, Dict[str, Any]] = {}
    for binding in bindings:
        source = binding["source"]
        if "features" in source:
            for name in source["features"]:
                found.setdefault(str(name), {"name": str(name), "size": 1, "kind": "number"})
        elif "feature" in source:
            name = str(source["feature"])
            part = source.get("part")
            if part in (None, "ids"):
                size = int(_product(binding["sample"]))
                kind = "tokens" if part == "ids" else "numbers"
                found[name] = {"name": name, "size": size, "kind": kind}
            else:
                found.setdefault(name, {"name": name, "size": 0, "kind": "tokens"})
    return list(found.values())


def _product(dims: Sequence[int]) -> int:
    total = 1
    for dim in dims:
        total *= int(dim)
    return total


def _is_table(bindings: Sequence[Mapping[str, Any]]) -> bool:
    features = _features(bindings)
    return bool(features) and all(feature["kind"] == "number" for feature in features)


def _check_rows(
    spec: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    feeds0: Mapping[str, Any],
    more_samples: Sequence[Mapping[str, Any]],
    own: Any,
    limits: Mapping[str, int],
    vocabulary: Optional[int],
    numpy: Any,
    whole: Any = None,
) -> List[Dict[str, Any]]:
    """Rows to compare on: the caller's example, rows made up here, and the caller's own table rows.

    ``whole`` is the whole check of a model that reads a table of numbers, made by its caller (the caller's own
    rows and rows made up where the model's data lies): a column for each feature, in the order the model reads
    them. Those rows are then the check, as the doubles they are.
    """
    features = _features(bindings)
    declared = {str(item.get("name")): item for item in spec.get("features") or [] if isinstance(item, Mapping)}
    per_row = max(sum(int(feature["size"]) for feature in features), 1)
    most = max(1, min(240, CHECK_VALUES // per_row))
    table = _is_table(bindings)
    if whole is not None and table:
        names = [str(feature["name"]) for feature in features]
        lines = numpy.asarray(whole, dtype=float)
        ids = any(_ranged(declared.get(name)) for name in names)  # an id or a category takes the values it has
        if lines.ndim == 2 and lines.shape[0] and lines.shape[1] == len(names) and not ids:
            return [{name: numpy.array([float(value)]) for name, value in zip(names, line)} for line in lines][:most]
    rows: List[Dict[str, Any]] = []
    for feeds in [feeds0, *more_samples]:
        row = _row_of(bindings, feeds, numpy)
        if row is not None:
            rows.append(row)
    generator = numpy.random.default_rng(0)
    for index in range(TABLE_CHECK_ROWS if table else CHECK_ROWS):
        spread = index >= (4 if table else CHECK_ROWS // 2)
        row = {}
        for feature in features:
            size = int(feature["size"])
            if feature["kind"] == "tokens":
                known = declared.get(feature["name"], {}).get("vocab_size") or limits.get(feature["name"])
                limit = int(known or vocabulary or 2)
                count = size if index % 2 == 0 or size < 2 else max(1, size // 2)
                ids = numpy.full(size, -1.0)
                ids[:count] = generator.integers(0, max(limit, 2), size=count)
                row[feature["name"]] = ids
            elif _ranged(declared.get(feature["name"])):
                # An id: a user, an item, a category. Made-up rows hold ids the model has.
                low, high = _ranged(declared[feature["name"]])  # type: ignore[misc]
                row[feature["name"]] = generator.integers(low, high + 1, size=size).astype(float)
            elif spread:
                row[feature["name"]] = generator.normal(size=size).round(6)
            else:
                row[feature["name"]] = generator.uniform(0.0, 1.0, size=size).round(6)
        rows.append(row)
    if table and own is not None:
        names = [str(feature["name"]) for feature in features]
        kept = rows_to_check(own, names, limit=OWN_CHECK_ROWS)
        if kept is not None:
            rows.extend({name: numpy.array([float(value)]) for name, value in zip(names, line)} for line in kept)
    return rows[:most]


def _ranged(feature: Optional[Mapping[str, Any]]) -> Optional[Tuple[int, int]]:
    """``(low, high)`` of an integer feature with both bounds, or the positions of a category's names; ``None`` for
    anything else."""
    if feature and str(feature.get("type")) == "categorical" and feature.get("categories"):
        return 0, len(list(feature["categories"])) - 1
    if not feature or str(feature.get("type")) != "integer":
        return None
    low, high = feature.get("min"), feature.get("max")
    if low is None or high is None or float(high) < float(low):
        return None
    return int(float(low)), int(float(high))


def _checked_against(
    spec: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    library: Optional[Callable[[Dict[str, Any]], Any]],
    converted: Callable[[Dict[str, Any]], Any],
    rows: Sequence[Mapping[str, Any]],
    shape: Sequence[int],
    numpy: Any,
    why: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """``(the check to send, how far the conversion is from the library)``.

    ``library`` runs the original model and ``converted`` its ONNX conversion; with no library the ONNX model
    is the original. The check's outputs are the library's, read the way the model's head reads them: a
    language model's scores at its last real token, a classifier's as probabilities, and a tensor in the
    order the model itself keeps it (a picture written channels first stays channels first; the API turns it
    to the order it shows one in). The distance is the largest difference between the library's output and
    the conversion's on any row, relative to the largest output the library gave. ``None`` for the check
    when no row could be compared, and ``why``, when given a list, is told why.
    """
    said = why if why is not None else []
    kept: List[Mapping[str, Any]] = []
    answers: List[List[float]] = []
    worst, scale = 0.0, 0.0
    refused: Optional[BaseException] = None
    for row in rows:
        if library is None:
            try:
                theirs = numpy.asarray(converted(_feeds(bindings, row, numpy)), dtype=float)
            except Exception as exc:  # noqa: BLE001 - a row the model itself refuses proves nothing
                refused = refused or exc
                continue
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    theirs = numpy.asarray(_plain_array(library(_feeds(bindings, row, numpy, exact=False))), dtype=float)
            except Exception as exc:  # noqa: BLE001 - a row the model itself refuses (a negative count, say) proves nothing
                refused = refused or exc
                continue
        if not theirs.size or not numpy.isfinite(theirs).all():
            continue
        if library is not None:
            try:
                ours = numpy.asarray(converted(_feeds(bindings, row, numpy)), dtype=float)
            except Exception as exc:  # noqa: BLE001 - the conversion cannot run a row the model can
                said.append(f"the conversion could not run a row the model ran: {type(exc).__name__}: {_brief(exc, 160)}")
                return None, float("inf")
            if ours.size != theirs.size:
                said.append(f"the conversion gives {ours.size:,} values for a row, and the model {theirs.size:,}")
                return None, float("inf")
            worst = max(worst, float(numpy.max(numpy.abs(ours.reshape(-1) - theirs.reshape(-1)))))
        scale = max(scale, float(numpy.max(numpy.abs(theirs))))
        read = _as_the_head_reads(spec, theirs, row, shape, numpy)
        if read is None:
            said.append(f"the model gives {theirs.size:,} values for a row, and its reading has room for {_product(shape):,}")
            return None, float("inf")
        kept.append(row)
        answers.append([float(value) for value in read])
    if not kept:
        whose = "the ONNX model" if library is None else "the model"
        found = f": {type(refused).__name__}: {_brief(refused, 160)}" if refused is not None else ""
        said.append(f"{whose} gave no finite answer for any of the {len(rows):,} rows it was given{found}")
        return None, float("inf")
    names = [str(feature["name"]) for feature in _features(bindings)]
    steps = spec["graph"].get("nodes") or []
    long_chain = len(steps) >= 30 or any(str(node.get("op")) in _LONG_CHAINS for node in steps)
    # Every value as a number, as the API encodes it: a category is its position in the feature's categories.
    check = {
        "inputs": {name: [[float(value) for value in numpy.asarray(row[name]).reshape(-1)] for row in kept] for name in names},
        "outputs": answers,
        "tolerance": 1e-3 if long_chain else 1e-4,
    }
    return check, (worst / max(scale, 1e-12) if worst else 0.0)


def _plain_array(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _as_the_head_reads(
    spec: Mapping[str, Any], raw: Any, row: Mapping[str, Any], shape: Sequence[int], numpy: Any
) -> Optional[Any]:
    """One row's output as the API compares it: the last real token's scores, probabilities, or the tensor as it is."""
    task = str(spec.get("task") or "")
    link = str((spec.get("head") or {}).get("link") or "none")
    dims = [int(dim) for dim in shape]
    value = numpy.asarray(raw, dtype=float)
    if value.size != _product(dims):
        return None
    value = value.reshape(dims)
    if task == "sequence_to_sequence":
        # What the decoder writes next: its scores after the last token it has written so far.
        shown = dims[1:] if len(dims) >= 2 and dims[0] == 1 else dims
        table = value.reshape(-1, shown[-1]) if len(shown) >= 2 else value.reshape(1, -1)
        written = str((spec.get("decoder") or {}).get("feature") or "decoder")
        count = int((numpy.asarray(row.get(written, [0]), dtype=float) >= 0).sum()) if written in row else 1
        flat = table[min(max(count, 1), table.shape[0]) - 1]
    elif task == "text_generation":
        shown = dims[1:] if len(dims) >= 2 and dims[0] == 1 else dims
        if len(shown) == 2:
            axis = next((index for index in (1, 0, 2) if index < len(dims) and dims[index] == shown[0]), 1)
            flat = numpy.take(value, _real_tokens(spec, row, shown[0], numpy) - 1, axis=axis).reshape(-1)
        elif len(shown) == 1:
            flat = value.reshape(-1)
        else:
            return None
    else:
        flat = value.reshape(-1)
    if link == "softmax":
        shifted = numpy.exp(flat - flat.max())
        return shifted / shifted.sum()
    if link == "sigmoid":
        return 1.0 / (1.0 + numpy.exp(-flat))
    return flat


def _real_tokens(spec: Mapping[str, Any], row: Mapping[str, Any], positions: int, numpy: Any) -> int:
    """How many tokens of the row's text are real: a language model is read at the last of them."""
    for item in spec["graph"].get("inputs") or []:
        source = item.get("from") or {}
        if source.get("part") is not None and source.get("feature") in row:
            count = int((numpy.asarray(row[source["feature"]], dtype=float) >= 0).sum())
            return max(1, min(count, positions))
    return positions


# --- a live model, through ONNX --------------------------------------------------------------


def via_onnx(
    estimator: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    input_shape: Any = None,
    sample: Any = None,
    inputs: Optional[Mapping[str, Any]] = None,
    check_rows: Any = None,
    labels: Optional[Sequence[Any]] = None,
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    output: Any = None,
    tokenizer: Any = None,
    framework: Optional[str] = None,
    preprocess: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
    reasons: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    keep: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert a live model to ONNX in memory and read that as a ``family: "graph"`` spec.

    A ``torch.nn.Module`` goes through ``torch.onnx.export``, a scikit-learn estimator or pipeline through
    ``skl2onnx``, a Keras or TensorFlow model through ``tf2onnx``. Returns ``None`` when the converter is not
    installed or the conversion fails; ``reasons``, when given a list, is told why.

    The spec's check holds the original library's outputs on a few rows (``check_rows`` among them for a
    table), and a conversion that does not give those outputs is marked ``approximate``: drawn, not run. So is a
    scikit-learn classifier's whose class, as Kaimei would report it from those outputs, is not the class the
    model's own ``predict`` gives on every one of those rows (:func:`classes_differ`); a threshold of the model's
    own is carried in ``outputs.threshold``, and its ``decision_function`` is read where that is what agrees.
    See :func:`kaimei.structure.extract_structure` for ``sample``, ``inputs``, ``output``, ``tokenizer``,
    ``max_length`` and ``threshold``. ``keep``, when given a dict, receives the converted model's bytes under ``"onnx"``
    (:func:`to_onnx` returns them).
    """
    why = reasons if reasons is not None else []
    numpy = _require_numpy()
    labels = label_names(labels)
    if not _has("onnx"):
        why.append("the onnx package is not installed (pip install onnx, or pip install 'kaimei[onnx]')")
        return None
    if max_length is not None:
        max_length = _whole_length(max_length)  # a NumPy whole number goes on as Python's own
    options: Dict[str, Any] = {
        "feature_names": feature_names,
        "task": task,
        "inputs": inputs,
        "check_rows": check_rows,
        "labels": labels,
        "output": output,
        "mask_layout": mask_layout,
        "threshold": threshold,
        "tokenizer": tokenizer,
        "framework": framework,
        "name": name,
        "keep": keep,
    }
    from . import converters, tabular_pipelines, text_pipelines

    try:
        kind = converters.model_file_kind(estimator)
        if kind == "gguf":
            import torch

            model, found = converters.open_gguf(estimator)
            if sample is None:
                sample = converters.gguf_sample(model, found, torch)
                max_length = max_length or converters.GGUF_DEFAULT_TOKENS
            options["tokenizer"] = tokenizer if tokenizer is not None else found
            options["framework"] = framework or "gguf"
            spec = _torch_through_onnx(
                model, why, numpy, input_shape=input_shape, sample=sample, preprocess=preprocess, max_length=max_length,
                **options,
            )
        elif kind is not None:
            spec = converters.tensorflow_file_through_onnx(estimator, kind, why, numpy, sample=sample, **options)
        elif converters.is_jax(estimator, framework):
            spec = converters.jax_through_onnx(estimator, why, numpy, sample=sample, **options)
        elif text_pipelines.text_parts(estimator) is not None:
            spec = text_pipelines.through_graph(estimator, why, numpy, max_length=max_length, **options)
        elif tabular_pipelines.column_parts(estimator) is not None:
            spec = tabular_pipelines.through_graph(estimator, why, numpy, **options)
        elif _is_keras_model(estimator) and not _torch_only(estimator):
            spec = _keras_through_onnx(estimator, why, numpy, sample=sample, **options)
        elif _is_torch_module(estimator):
            spec = _torch_through_onnx(
                estimator,
                why,
                numpy,
                input_shape=input_shape,
                sample=sample,
                preprocess=preprocess,
                max_length=max_length,
                **options,
            )
        elif _is_sklearn(estimator):
            spec = _sklearn_through_onnx(estimator, why, numpy, **options)
        else:
            why.append(f"no ONNX converter is known for {type(estimator).__name__}")
            return None
    except StructureError as exc:
        if exc.param in ("tokenizer", "output", "labels", "inputs", "sample", "max_length", "task", "mask_layout", "threshold"):
            raise  # the caller's own argument is wrong, and saying nothing would hide it
        if exc.param == "feature_names" and feature_names is not None:
            raise  # likewise: the names are the caller's, and a reason among others would lose which argument it was
        why.append(exc.message)
        return None
    except Exception as exc:  # noqa: BLE001 - a converter can fail in any way its framework can
        why.append(f"the conversion failed ({type(exc).__name__}: {_brief(exc, 200)})")
        return None
    if spec is None:
        return None
    described = extraction_metadata(estimator, spec)
    spec["metadata"] = {**described, **dict(spec.get("metadata") or {}), "route": "onnx"}
    return spec


def _torch_only(estimator: Any) -> bool:
    """A plain torch module, which the Keras test would also pass on the torch backend."""
    return str(type(estimator).__module__).startswith("torch")


def _is_sklearn(estimator: Any) -> bool:
    """Something scikit-learn's converter may take: an estimator, a pipeline, or a search around one."""
    return callable(getattr(estimator, "get_params", None)) and (
        callable(getattr(estimator, "predict", None))
        or callable(getattr(estimator, "transform", None))
        or callable(getattr(estimator, "decision_function", None))
    )


# --- PyTorch ---------------------------------------------------------------------------------


def _forward_parameters(module: Any) -> List[str]:
    """The names ``forward`` gives its inputs, in order."""
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]


def _torch_example(
    module: Any, names: List[str], feature_names: Optional[Sequence[str]], input_shape: Any, sample: Any, torch: Any
) -> Dict[str, Any]:
    """Example inputs by ``forward`` parameter name, each with its batch axis, on the module's device.

    A sample that is not a tensor (NumPy, a list) is made one of the module's own float type: NumPy's float64
    would meet float32 weights and the module would refuse it. Whole numbers stay as they are.
    """
    device, floats = _placement(module, torch)

    def tensor(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().to(device)
        made = torch.as_tensor(value)
        if made.dtype.is_floating_point:
            made = made.to(floats)
        return made.to(device)

    if sample is not None:
        if isinstance(sample, Mapping):
            unknown = [key for key in sample if str(key) not in names] if names else []
            if unknown:
                raise StructureError(
                    f"sample names '{unknown[0]}', which is not an input of this model's forward ({', '.join(names)}).",
                    code="invalid_structure",
                    param="sample",
                )
            ordered = [name for name in names if name in sample] if names else [str(key) for key in sample]
            if names and ordered != names[: len(ordered)]:
                raise StructureError(
                    "sample has to give forward's inputs from the first one on, without leaving one out between them.",
                    code="invalid_structure",
                    param="sample",
                )
            return {name: tensor(sample[name]) for name in ordered}
        values = list(sample) if isinstance(sample, (tuple, list)) else [sample]
        if isinstance(sample, list) and sample and not hasattr(sample[0], "shape"):
            values = [sample]  # a nested list of numbers is one input, not several
        labels = names[: len(values)] if len(names) >= len(values) else [f"input_{index}" for index in range(len(values))]
        return {label: tensor(value) for label, value in zip(labels, values)}
    first = names[0] if names else "input"
    kind = floats
    if input_shape is not None:
        if isinstance(input_shape, Mapping):
            return {
                str(key): torch.zeros(1, *[int(dim) for dim in dims], dtype=kind, device=device)
                for key, dims in input_shape.items()
            }
        return {first: torch.zeros(1, *[int(dim) for dim in input_shape], dtype=kind, device=device)}
    width = len(feature_names) if feature_names else None
    if width is None:
        leaves = [leaf for leaf in module.modules() if not list(leaf.children()) and leaf is not module]
        if leaves and type(leaves[0]).__name__ == "Linear":
            width = int(leaves[0].in_features)
    if width is None:
        raise StructureError(
            "it needs an example input to be converted: pass sample=... (what you would call the model with), or "
            "feature_names for a table, or input_shape for an image",
            code="invalid_structure",
        )
    return {first: torch.zeros(1, width, dtype=kind, device=device)}


def _placement(module: Any, torch: Any) -> Tuple[Any, Any]:
    """``(device, float type)`` a module computes on: its first parameter's device (a buffer's, when it has no
    parameters), and the type of its first floating one. The CPU and ``float32`` for a module that holds neither.

    Every tensor a conversion makes to give the module (an example, a decoder's first tokens, the numbers a
    picture is normalized with) is made there, so a module on a GPU or in double precision is called as it runs.
    """
    first = next(iter(module.parameters()), None)
    if first is None:
        first = next(iter(module.buffers()), None)
    device = first.device if first is not None else torch.device("cpu")
    held = (tensor for group in (module.parameters(), module.buffers()) for tensor in group)
    floating = next((tensor for tensor in held if tensor.dtype.is_floating_point), None)
    return device, (floating.dtype if floating is not None else torch.float32)


def _decoder_start(length: int, pad: int, start: int, device: Any, torch: Any) -> Any:
    """What an encoder-decoder's decoder has written before its first token: its start token, then padding."""
    started = torch.full((1, max(int(length), 2)), int(pad), dtype=torch.long, device=device)
    started[0, 0] = int(start)
    return started


def _flat_tensors(value: Any) -> List[Any]:
    """Every tensor of a module's output, in the order an exporter numbers them."""
    if hasattr(value, "detach"):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _flat_tensors(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _flat_tensors(item)]
    return []


class _NoConversion(Exception):
    """A converter could not write the model as ONNX."""


@contextlib.contextmanager
def _hushed() -> Any:
    """Keep an exporter's own output off the caller's screen while it works.

    An export that fails prints the whole graph it was working on, from C++, before the next operator set is
    tried; one that succeeds prints its progress. Neither is the caller's to read: what matters is said in
    the error or the result. Python's streams are turned away, and so are the process's own (which is where
    C++ writes), for as long as the export runs. Where there is no real stream to turn away, nothing is done.
    """
    kept: List[Tuple[int, int]] = []
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - a stream that cannot be flushed has nothing waiting in it
                pass
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            for descriptor in (1, 2):
                kept.append((descriptor, os.dup(descriptor)))
                os.dup2(null, descriptor)
        finally:
            os.close(null)
    except OSError:
        pass
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        _flush_c_streams()
        for descriptor, saved in reversed(kept):
            try:
                os.dup2(saved, descriptor)
            except OSError:
                pass
            finally:
                os.close(saved)


def _flush_c_streams() -> None:
    """What C and C++ code has written but not yet sent on, sent on now: before the streams are given back."""
    try:
        import ctypes

        runtime = ctypes.CDLL("ucrtbase") if os.name == "nt" else ctypes.CDLL(None)
        runtime.fflush(None)
    except Exception:  # noqa: BLE001 - without a C runtime to ask, there is nothing to flush
        pass


def _export_torch(module: Any, example: Dict[str, Any], outputs: List[str], torch: Any) -> bytes:
    """The module as ONNX bytes: the tracing exporter at a few operator sets, then the newer one."""
    args = tuple(example.values())
    axes = {name: {0: "batch"} for name, value in example.items() if value.dim() >= 1}
    failures: List[str] = []
    for opset, dynamo in ((17, False), (20, False), (None, True)):
        buffer = io.BytesIO()
        options: Dict[str, Any] = {"input_names": list(example), "output_names": list(outputs), "verbose": False}
        if opset is not None:
            options["opset_version"] = opset
        if not dynamo:
            options["dynamic_axes"] = axes
        try:
            with warnings.catch_warnings(), _hushed():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    try:
                        produced = torch.onnx.export(module, args, buffer, dynamo=dynamo, **options)
                    except TypeError as exc:
                        if "dynamo" not in str(exc):
                            raise
                        if dynamo:
                            continue  # an older torch has only the tracing exporter
                        produced = torch.onnx.export(module, args, buffer, **options)
            if buffer.getbuffer().nbytes == 0 and produced is not None and hasattr(produced, "save"):
                produced.save(buffer)
            if buffer.getbuffer().nbytes:
                return buffer.getvalue()
            failures.append("the exporter wrote nothing")
        except Exception as exc:  # noqa: BLE001 - each exporter fails in its own ways; the next one may not
            failures.append(f"{'dynamo' if dynamo else f'opset {opset}'}: {type(exc).__name__}: {_brief(exc, 160)}")
    raise _NoConversion("torch.onnx.export could not write it (" + "; ".join(failures[:2]) + ")")


def _preprocessed(module: Any, preprocess: Mapping[str, Any], torch: Any) -> Any:
    """The module behind the normalization its pixels need, so the graph takes ``[0, 1]`` pixels as Kaimei sends them.

    The numbers it normalizes with are kept on the module's own device and in its float type (see :func:`_placement`).
    """
    unknown = set(preprocess) - {"mean", "std", "scale", "offset"}
    if unknown:
        raise StructureError(
            f"preprocess takes mean, std, scale and offset; got {sorted(unknown)}.",
            code="invalid_structure",
            param="preprocess",
        )
    device, floats = _placement(module, torch)

    def per_channel(key: str, default: float) -> Any:
        value = torch.as_tensor(preprocess.get(key, default), dtype=floats, device=device)
        return value.reshape(1, -1, 1, 1) if value.dim() == 1 else value

    class Preprocessed(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner
            for key, default in (("mean", 0.0), ("std", 1.0), ("scale", 1.0), ("offset", 0.0)):
                self.register_buffer(key, per_channel(key, default))

        def forward(self, image: Any) -> Any:
            return self.inner((image - self.mean) / self.std * self.scale + self.offset)

    return Preprocessed(module)


def _torch_through_onnx(
    estimator: Any,
    why: List[str],
    numpy: Any,
    *,
    feature_names: Optional[Sequence[str]],
    task: Optional[str],
    input_shape: Any,
    sample: Any,
    inputs: Optional[Mapping[str, Any]],
    check_rows: Any,
    labels: Optional[Sequence[Any]],
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    output: Any,
    tokenizer: Any,
    framework: Optional[str],
    preprocess: Optional[Mapping[str, Any]],
    name: Optional[str],
    max_length: Optional[int] = None,
    keep: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        import torch
    except ImportError:
        why.append("torch is not installed")
        return None
    module = estimator
    names = _forward_parameters(estimator)
    example = _torch_example(estimator, names, feature_names, input_shape, sample, torch)
    told = hugging_face_hints(estimator) or torchvision_hints(estimator)
    if task is None and told.get("task"):
        task = str(told["task"])
    elif task == "text_generation" and told.get("task") == "sequence_to_sequence":
        task = "sequence_to_sequence"  # an encoder-decoder writes text from its input, never on from a prompt alone
    if not labels and told.get("labels"):
        labels = list(told["labels"])
    decoding = task == "sequence_to_sequence" and "decoder_input_ids" in names
    decoder = dict(told.get("decoder") or {})
    if decoding and "decoder_input_ids" not in example:
        given_ids = next((value for value in example.values() if not value.dtype.is_floating_point and value.dim() >= 2), None)
        length = int(given_ids.shape[-1]) if given_ids is not None else 16
        said_pad, said_start = decoder.get("pad_token"), decoder.get("start_token")
        pad = int(said_pad) if said_pad is not None else 0
        start = int(said_start) if said_start is not None else pad
        example["decoder_input_ids"] = _decoder_start(length, pad, start, _placement(estimator, torch)[0], torch)
    if max_length is not None:
        # The graph's shapes are the example's: its token axis is made as long as the model is to read at most.
        counted = {key: value.detach().cpu().numpy() for key, value in example.items() if not value.dtype.is_floating_point}
        fitted = _fit_tokens(counted, inputs or {}, _pad_token(tokenizer, inputs), int(max_length), numpy)
        for key, value in fitted.items():
            example[key] = torch.as_tensor(value, dtype=example[key].dtype, device=example[key].device)
    wrapped = bool(preprocess) and len(example) == 1 and next(iter(example.values())).dim() == 4
    if preprocess and not wrapped:
        raise StructureError(
            "preprocess normalizes the pixels of a model with one image input, and this model is not one.",
            code="invalid_structure",
            param="preprocess",
        )
    if wrapped:
        module = _preprocessed(estimator, preprocess or {}, torch)
        own_name = next(iter(example))
        example = {"image": next(iter(example.values()))}
        if inputs and own_name in inputs and own_name != "image":
            # The wrapper takes the module's one input as "image": what the caller said of it is said of that.
            inputs = {("image" if key == own_name else key): value for key, value in inputs.items()}
    detecting = task == "object_detection"
    detection = dict(told.get("detection") or {})
    if detecting and isinstance(output, Mapping):
        detection.update({str(key): value for key, value in output.items()})
        output = None
    # Settings of forward an exporter would otherwise write as inputs of the file (logits_to_keep), bound to what
    # calls the model as it is called by default.
    settings = _bound_settings(estimator, example) if not wrapped else {}
    if decoding:
        module = _KeywordCall.make(torch, estimator, list(example), str(told.get("output") or "logits"), settings)
        wrapped_by_name = True
    elif settings and not detecting:
        module = _KeywordCall.make(torch, estimator, list(example), None, settings)
        wrapped_by_name = True
    elif detecting:
        if "boxes" not in detection or "scores" not in detection:
            raise StructureError(
                "A detector is read from its boxes and their scores: pass output={'boxes': <its boxes output>, "
                "'scores': <its scores output>, 'classes': <its classes output, if apart>} (names, or positions "
                "among what forward returns).",
                code="invalid_structure",
                param="output",
            )
        module = _Detections.make(torch, estimator, detection["boxes"], detection["scores"], detection.get("classes"))
        wrapped_by_name = True
    else:
        wrapped_by_name = False
    rows = max(int(value.shape[0]) if value.dim() else 1 for value in example.values())
    one = {key: (value[:1] if value.dim() and value.shape[0] == rows and rows > 1 else value) for key, value in example.items()}
    # Calling the module draws from torch's generators when it draws at random: they are put back afterwards, so
    # converting a model does not change what the caller's own program draws next.
    drawn_from = _random_state(module, torch)
    modes = training_modes(module)
    module.eval()
    # nn.TransformerEncoderLayer and nn.MultiheadAttention take a fused path at inference that no exporter can
    # write. With it off they run the same arithmetic step by step, which is what is exported and compared.
    fused = getattr(torch.backends, "mha", None)
    was_fused = fused.get_fastpath_enabled() if fused is not None else None
    if fused is not None:
        fused.set_fastpath_enabled(False)
    # A language model hands back its cache with its scores (an object no exporter can write); without it,
    # it gives the same scores.
    cached = _cache_off(estimator)
    try:
        with torch.no_grad():
            answered = module(*one.values())
        produced = _flat_tensors(answered)
        if not produced:
            why.append("its forward does not return a tensor")
            return None
        if output is None and told.get("output"):
            output = _named_output(answered, str(told["output"]))
            if output is not None and task is None and told.get("task_if_pooled"):
                task = str(told["task_if_pooled"])
        chosen = _torch_output(produced, output)
        out_names = ["output"] if len(produced) == 1 else [f"output_{index}" for index in range(len(produced))]
        try:
            data = _export_torch(module, one, out_names, torch)
        except _NoConversion as exc:
            why.append(str(exc))
            return None
        tools = _importer()
        proto = tools.load_model(data)
        written = [str(info.name) for info in tools._input_infos(proto)]
        added = [name for name in written if name not in one]
        if added:
            # The file would be given values the module never was, so the two could not be compared on any row.
            why.append(
                f"torch.onnx.export wrote {'an input' if len(added) == 1 else 'inputs'} that forward was not called "
                f"with ({', '.join(added[:4])}), so the conversion is not the module as it was called and cannot be "
                "compared with it"
            )
            return None
        if keep is not None:
            keep["onnx"] = data

        def library(feeds: Dict[str, Any]) -> Any:
            # An input the exporter dropped (the output does not depend on it) is given as the example had it.
            args = [
                torch.as_tensor(feeds[key], dtype=value.dtype, device=value.device) if key in feeds else value
                for key, value in one.items()
            ]
            with torch.no_grad():
                return _flat_tensors(module(*args))[chosen].detach().cpu().numpy()

        more = [
            {key: _plain_array(value[index : index + 1] if value.dim() and value.shape[0] == rows else value) for key, value in example.items()}
            for index in range(1, min(rows, 8))
        ]
        whole = [value for value in one.values() if not value.dtype.is_floating_point]
        vocabulary = max((int(value.max()) + 1 for value in example.values() if not value.dtype.is_floating_point), default=0)
        hints = dict(inputs or {})
        if decoding:
            # What the decoder has written so far is a feature of token ids, filled with its padding token; the API
            # writes into it, token by token, and starts it from the decoder's first token.
            hints.setdefault("decoder_input_ids", {"feature": "decoder", "part": "ids", "pad_id": int(decoder.get("pad_token") or 0)})
            hints["decoder_input_ids"] = _padded_as_the_model_pads(hints["decoder_input_ids"], decoder.get("pad_token"))
            if "decoder_attention_mask" in one:
                # The mask of the same feature, under the name the caller gave it when they gave it one.
                given = hints["decoder_input_ids"]
                called = str(given.get("feature") or "decoder") if isinstance(given, Mapping) else "decoder"
                hints.setdefault("decoder_attention_mask", {"feature": called, "part": "mask"})
        rename: Dict[str, str] = {}
        only = next(iter(one)) if len(one) == 1 else None
        if only is not None and only not in hints:
            value = one[only]
            dims = [int(dim) for dim in value.shape]
            named_one = str(feature_names[0]) if feature_names and len(feature_names) == 1 else None
            said_image = wrapped or (input_shape is not None and not isinstance(input_shape, Mapping) and len(input_shape) == 3)
            pictured = value.dtype.is_floating_point and len(dims) == 4 and dims[1] in (1, 3, 4)
            if pictured and (said_image or min(dims[2], dims[3]) >= 8):
                # A picture, channels first as PyTorch has them, under the name the native reader gives one.
                hints[only] = {"feature": named_one or "image", "layout": "nchw"}
            elif named_one and not (value.dtype.is_floating_point and len(dims) == 2):
                # One name for a model whose one input is not a table: the name is the feature's.
                rename = {only: named_one, "text": named_one}
        spec = _read_onnx(
            proto,
            library=library,
            feature_names=feature_names,
            task=task,
            labels=labels,
            inputs=hints,
            input_shapes=None,
            samples={key: _plain_array(value) for key, value in one.items()},
            output=out_names[chosen],
            mask_layout=mask_layout,
            threshold=threshold,
            name=name,
            tokenizer=tokenizer,
            check_rows=check_rows,
            framework=framework or "pytorch",
            more_samples=more,
            rename=rename,
            vocabulary=vocabulary if whole else None,
            dropped=[key for key in one if key not in written],
        )
    finally:
        restore_modes(modes)
        if fused is not None:
            fused.set_fastpath_enabled(bool(was_fused))
        if cached is not None:
            estimator.config.use_cache = cached
        _put_back_random_state(drawn_from, torch)
    if wrapped or wrapped_by_name:
        _drop_scope(spec, "inner")
    if decoding:
        ids = example["decoder_input_ids"]
        # What the importer wrote when the caller hinted the input {"generated": True} stands: the feature's name
        # and the tokens the hint said.
        said = dict(spec.get("decoder") or {})
        writes_into = str(said.get("feature") or _hinted_feature((inputs or {}).get("decoder_input_ids")) or "decoder")
        start = int(said.get("start_token", ids[0, 0]))
        spec["decoder"] = {
            key: value
            for key, value in {
                "feature": writes_into,
                "start_token": start,
                "eos_token": said.get("eos_token", decoder.get("eos_token")),
                "pad_token": said.get("pad_token", decoder.get("pad_token")),
                "max_length": int(ids.shape[-1]),
            }.items()
            if value is not None
        }
        for feature in spec.get("features") or []:
            if feature.get("name") == writes_into:
                # Not an input a person gives: the API keeps it, starting from the decoder's first token.
                feature["generated"] = True
                feature["baseline"] = [start]
        _decoder_vocabulary(spec, tokenizer)
    if detecting:
        _detection_head(spec, detection)
    _say_token_room(spec, estimator, max_length)
    steps = len(spec["graph"].get("nodes") or [])
    spec["summary"] = (
        f"PyTorch module converted to ONNX and read as a graph of {steps:,} operations. " + str(spec.get("summary") or "")
    ).strip()
    return spec


def training_modes(module: Any) -> List[Tuple[Any, bool]]:
    """Each submodule of a ``torch.nn.Module`` with its own ``training`` flag, to put back after a run in eval mode.

    ``module.train(flag)`` would set one flag for all of them, and a part held in eval mode (a frozen backbone,
    whose batch normalization must not learn) would be put back into training.
    """
    return [(layer, bool(layer.training)) for layer in module.modules()]


def restore_modes(modes: Sequence[Tuple[Any, bool]]) -> None:
    """Put back the flags :func:`training_modes` recorded, each on its own submodule."""
    for layer, training in modes:
        layer.training = training


#: Hugging Face model classes by what they are for, by the end of their names.
_HF_CLASSIFIERS = (
    "ForSequenceClassification",
    "ForImageClassification",
    "ForImageClassificationWithTeacher",
    "ForAudioClassification",
    "ForVideoClassification",
)
_HF_WRITERS = ("ForCausalLM", "LMHeadModel")
_DEFAULT_LABEL = re.compile(r"^LABEL_\d+$")


def hugging_face_hints(module: Any) -> Dict[str, Any]:
    """What a Hugging Face model says about itself: ``task``, ``labels`` and the ``output`` to read, by name.

    A model's class says what it is for (``...ForSequenceClassification`` is a classifier, ``...ForCausalLM``
    writes text, a bare ``...Model`` with a pooled output gives an embedding), and its config says how many
    classes there are, what they are called (``id2label``, when they were named) and whether several can be
    true at once (``problem_type``). Empty for anything that is not a Hugging Face model, or that says nothing.
    """
    config = getattr(module, "config", None)
    if config is None or not str(type(config).__module__).startswith("transformers"):
        return {}
    kind = type(module).__name__
    count = int(getattr(config, "num_labels", 0) or 0)
    found: Dict[str, Any] = {}
    if kind.endswith(_HF_CLASSIFIERS):
        problem = str(getattr(config, "problem_type", None) or "")
        if problem == "multi_label_classification":
            found["task"] = "multilabel_classification"
        elif problem == "regression" or count == 1:
            found["task"] = "regression"
        elif count >= 2:
            found["task"] = "binary_classification" if count == 2 else "multiclass_classification"
        named = getattr(config, "id2label", None) or {}
        try:
            listed = [str(named[index]) for index in range(count)]
        except (KeyError, TypeError):
            listed = []
        if count >= 2 and listed and not all(_DEFAULT_LABEL.match(label) for label in listed):
            found["labels"] = listed
        found["output"] = "logits"
    elif kind.endswith(_HF_WRITERS) and not getattr(config, "is_encoder_decoder", False):
        found["task"] = "text_generation"
        found["output"] = "logits"
    elif kind.endswith("ForObjectDetection"):
        # DETR's kind: a fixed number of queries, each a box (centre and size, as shares of the picture) and the
        # scores of every class. DETR itself ends its classes with "nothing here" and is read through a softmax;
        # its deformable and real-time descendants score each class on its own, through a sigmoid.
        sigmoid = any(word in kind for word in ("Deformable", "RTDetr", "ConditionalDetr", "DFine", "DabDetr"))
        found["task"] = "object_detection"
        found["detection"] = {
            "boxes": "pred_boxes",
            "scores": "logits",
            "box_format": "cxcywh",
            "normalized": True,
            "score_link": "sigmoid" if sigmoid else "softmax",
            **({} if sigmoid else {"no_object": "last"}),
        }
        named = getattr(config, "id2label", None) or {}
        try:
            listed = [str(named[index]) for index in range(count)]
        except (KeyError, TypeError):
            listed = []
        if listed and not all(_DEFAULT_LABEL.match(label) for label in listed):
            found["labels"] = listed
    elif kind.endswith("ForSemanticSegmentation"):
        found["task"] = "segmentation"
        found["output"] = "logits"
        named = getattr(config, "id2label", None) or {}
        try:
            listed = [str(named[index]) for index in range(count)]
        except (KeyError, TypeError):
            listed = []
        if listed and not all(_DEFAULT_LABEL.match(label) for label in listed):
            found["labels"] = listed
    elif getattr(config, "is_encoder_decoder", False) and (
        kind.endswith(("ForConditionalGeneration", "ForSeq2SeqLM")) or hasattr(module, "lm_head")
    ):
        # An encoder-decoder that writes text: by its class's name, or by its language-model head (Marian's class
        # is MarianMTModel).
        generation = getattr(module, "generation_config", None)

        def said(key: str) -> Optional[int]:
            for holder in (generation, config):
                value = getattr(holder, key, None) if holder is not None else None
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                if value is not None:
                    return int(value)
            return None

        found["task"] = "sequence_to_sequence"
        found["output"] = "logits"
        found["decoder"] = {
            "start_token": said("decoder_start_token_id"),
            "eos_token": said("eos_token_id"),
            "pad_token": said("pad_token_id"),
        }
    elif kind.endswith("Model") and not getattr(config, "is_encoder_decoder", False) and hasattr(config, "hidden_size"):
        found["output"] = "pooler_output"
        found["task_if_pooled"] = "embedding"
    return found


def torchvision_hints(module: Any) -> Dict[str, Any]:
    """What a torchvision model is for, by where it comes from: its segmentation models answer with ``out``."""
    where = str(type(module).__module__)
    if where.startswith("torchvision.models.segmentation"):
        return {"task": "segmentation", "output": "out"}
    return {}


class _Detections:
    """Made a ``torch.nn.Module`` on first use: a detector whose boxes, scores and classes come out as one tensor.

    ``[..., 4 + C (+ 1)]``: the four numbers of each box, then its scores, then its class when the model gives one
    apart. One output is what the API's head reads and what the check holds.
    """

    @staticmethod
    def make(torch: Any, inner: Any, boxes: Any, scores: Any, classes: Any = None) -> Any:
        def pick(answered: Any, key: Any) -> Any:
            if isinstance(key, str) and isinstance(answered, Mapping):
                return answered[key]
            return _flat_tensors(answered)[int(key)]

        class Detections(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, *args: Any) -> Any:
                answered = self.inner(*args)
                box = pick(answered, boxes)
                score = pick(answered, scores)
                if score.dim() == box.dim() - 1:
                    score = score.unsqueeze(-1)
                parts = [box, score.to(box.dtype)]
                if classes is not None:
                    found = pick(answered, classes)
                    parts.append(found.unsqueeze(-1).to(box.dtype) if found.dim() == box.dim() - 1 else found.to(box.dtype))
                return torch.cat(parts, dim=-1)

        return Detections()


class _KeywordCall:
    """Made a ``torch.nn.Module`` on first use: calls a model by keyword, in the order given, with ``settings`` as
    well, and returns the output it names (all it returns when ``wanted`` is ``None``). An encoder-decoder takes
    its decoder's input after others (a mask, say) that an example may leave out, so it cannot be called by
    position; a language model is given its settings (:func:`_bound_settings`) so they are not inputs of the file."""

    @staticmethod
    def make(
        torch: Any, inner: Any, names: Sequence[str], wanted: Optional[str], settings: Optional[Mapping[str, Any]] = None
    ) -> Any:
        given = dict(settings or {})

        class KeywordCall(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, *args: Any) -> Any:
                answered = self.inner(**dict(zip(names, args)), **given)
                if wanted is None:
                    return answered
                return answered[wanted] if isinstance(answered, Mapping) else answered[0]

        return KeywordCall()


#: Settings of a language model's ``forward`` that ``torch.onnx.export`` writes as inputs of the file when they are
#: left to their defaults, with the value that keeps the default: scores for every position (0 keeps them all).
_KEEP_EVERY_POSITION = {"logits_to_keep": 0, "num_logits_to_keep": 0}


def _bound_settings(module: Any, given: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings to call a module's ``forward`` with, by keyword, so an export holds no input for them.

    Hugging Face language models take ``logits_to_keep`` (``num_logits_to_keep`` in older ones), and exporting
    OPT or BERT's language-model head with it left to its default writes it as a second input of the file, which
    the model was never given. It is bound to keep the scores of every position, as its default does, and the
    cache is turned off with it. Empty for a module with no such setting, or one the example already gives.
    """
    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return {}
    settings = {name: value for name, value in _KEEP_EVERY_POSITION.items() if name in parameters and name not in given}
    if settings and "use_cache" in parameters and "use_cache" not in given:
        settings["use_cache"] = False
    return settings


def _random_state(module: Any, torch: Any) -> Dict[str, Any]:
    """The state of the generators a module draws from when it runs: the CPU's, and its GPU's when it is on one."""
    device = _placement(module, torch)[0]
    kept: Dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if getattr(device, "type", "") == "cuda":
        kept["cuda"] = (device, torch.cuda.get_rng_state(device))
    return kept


def _put_back_random_state(kept: Mapping[str, Any], torch: Any) -> None:
    """Put back what :func:`_random_state` recorded."""
    torch.random.set_rng_state(kept["cpu"])
    if "cuda" in kept:
        device, state = kept["cuda"]
        torch.cuda.set_rng_state(state, device)


def _cache_off(module: Any) -> Optional[bool]:
    """Turn a Hugging Face model's ``use_cache`` off, and return what it was (``None`` when it has none)."""
    config = getattr(module, "config", None)
    if config is None or not isinstance(getattr(config, "use_cache", None), bool):
        return None
    was = bool(config.use_cache)
    config.use_cache = False
    return was


def _named_output(answered: Any, wanted: str) -> Optional[int]:
    """Where the tensor called ``wanted`` is among a module's outputs, as :func:`_flat_tensors` numbers them."""
    if not isinstance(answered, Mapping):
        return None
    position = 0
    for key, item in answered.items():
        count = len(_flat_tensors(item))
        if str(key) == wanted and count == 1:
            return position
        position += count
    return None


#: What a detection head may say about how its boxes and scores are read, with what it says when nothing does.
DETECTION_DEFAULTS = {"box_format": "xyxy", "normalized": False, "score_link": "none", "score_threshold": 0.3, "max_detections": 100}


def _detection_head(spec: Dict[str, Any], detection: Mapping[str, Any]) -> None:
    """Say, in the head, which columns of the joined output are the box, the scores and the class."""
    head = spec["head"]
    values = dict(spec["graph"].get("values") or {})
    shape = [int(dim) for dim in (values.get(str(head.get("output")), {}) or {}).get("shape") or []]
    width = shape[-1] if shape else 0
    apart = detection.get("classes") is not None
    scores_end = width - 1 if apart else width
    head.update(
        {
            "kind": "detection",
            "link": "none",
            "boxes": [0, 4],
            "scores": [4, scores_end],
            **({"classes": scores_end} if apart else {}),
        }
    )
    for key, default in DETECTION_DEFAULTS.items():
        head[key] = detection.get(key, default)
    if detection.get("no_object"):
        head["no_object"] = str(detection["no_object"])
    if scores_end - 4 < 1 or width < 5:
        raise StructureError(
            f"A detector's boxes and scores come out as {width} numbers each; a box is 4 and there must be a score.",
            code="invalid_structure",
            param="output",
        )
    labels = list((spec.get("outputs") or {}).get("labels") or [])
    classes = scores_end - 4 - (1 if detection.get("no_object") == "last" else 0)
    if not labels or len(labels) < classes:
        spec["outputs"] = {**dict(spec.get("outputs") or {}), "labels": labels + [f"class_{index}" for index in range(len(labels), max(classes, 1))]}


def _decoder_vocabulary(spec: Dict[str, Any], tokenizer: Any) -> None:
    """Say which tokenizer names the tokens the decoder writes: the text feature's, or the one given for it."""
    texts = list((spec.get("tokenizers") or {}).keys())
    if texts:
        spec["decoder"]["tokenizer"] = {"same_as": str(texts[0])}
        return
    if tokenizer is None:
        return
    from .text import tokenizer_spec

    described = tokenizer_spec(tokenizer)
    if described is not None:
        # No text goes in (a recording does): the tokenizer is the decoder feature's own, which names what it writes.
        # The feature is the one the structure's decoder writes into, whatever the caller called it.
        written = str(spec["decoder"].get("feature") or "decoder")
        spec.setdefault("tokenizers", {})[written] = described
        spec["decoder"]["tokenizer"] = {"same_as": written}
        for feature in spec.get("features") or []:
            if feature.get("name") == written:
                feature["type"] = "text"


def _torch_output(produced: Sequence[Any], wanted: Any) -> int:
    """Which of a module's output tensors the prediction is read from: the one asked for, or the first of floats."""
    if wanted is None:
        return next((index for index, tensor in enumerate(produced) if tensor.dtype.is_floating_point), 0)
    if isinstance(wanted, str) and wanted.startswith("output"):
        tail = wanted[len("output") :].lstrip("_")
        wanted = int(tail) if tail.isdigit() else (0 if not tail else wanted)
    if not _whole(wanted) or not 0 <= int(wanted) < len(produced):
        raise StructureError(
            f"output picks one of this model's {len(produced)} output tensors by position (0 is the first); got {wanted!r}.",
            code="invalid_structure",
            param="output",
        )
    return int(wanted)


def _drop_scope(spec: Dict[str, Any], scope: str) -> None:
    """Paths without the wrapper this module put around the model, so they are the model's own module names."""
    for node in spec["graph"].get("nodes") or []:
        path = str(node.get("path") or "")
        if path.startswith(scope + "."):
            node["path"] = path[len(scope) + 1 :]


# --- scikit-learn ----------------------------------------------------------------------------


def _sklearn_through_onnx(
    estimator: Any,
    why: List[str],
    numpy: Any,
    *,
    feature_names: Optional[Sequence[str]],
    task: Optional[str],
    inputs: Optional[Mapping[str, Any]],
    check_rows: Any,
    labels: Optional[Sequence[Any]],
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    output: Any,
    tokenizer: Any,
    framework: Optional[str],
    name: Optional[str],
    keep: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not _has("skl2onnx"):
        why.append("converting a scikit-learn model needs skl2onnx (pip install skl2onnx)")
        return None
    model = estimator
    best = getattr(estimator, "best_estimator_", None)
    if best is not None and type(estimator).__name__.endswith("SearchCV"):
        model = best  # a search predicts with the estimator it found
    width = _input_width(model)
    fitted = _names_it_was_fitted_on(model)
    if feature_names is not None:
        names = [str(item) for item in feature_names]
    elif fitted is not None:
        names = [str(item) for item in fitted]
    elif width is not None:
        names = [f"x{index}" for index in range(width)]
    else:
        why.append("it does not say how many features it reads; pass feature_names")
        return None
    final = model.steps[-1][1] if isinstance(getattr(model, "steps", None), list) and model.steps else model
    found = getattr(model, "classes_", None)
    if found is not None and _per_output(found):
        # classes_ holds the classes of each output: a k-NN, a forest or a MultiOutputClassifier fitted on a table
        # of answers. Its conversion gives each output's probabilities apart, not one row of them for a row.
        each = "a yes or a no for each" if all(len(list(item)) == 2 for item in found) else "a class for each"
        why.append(
            f"it answers {each} of its {len(found)} outputs on its own (a multi-output classifier, whose classes_ is a "
            "list), which this route does not read as one model"
        )
        return None
    classes: List[Any] = list(found) if found is not None and not _is_regressor(final) else []
    classifier = len(classes) >= 2
    multilabel = classifier and _multilabel(model, final, len(names), fitted, numpy)
    clusterer = _is_clusterer(final)
    method, kind = _cluster_answer(model) if clusterer else _sklearn_answer(model, classifier)
    if method is None:
        why.append("it has no predict, predict_proba or decision_function to compare a conversion with")
        return None
    if task is None and clusterer:
        task = "clustering"
    if task is None:
        if multilabel:
            task = "multilabel_classification"
        elif classifier:
            task = "binary_classification" if len(classes) == 2 else "multiclass_classification"
        else:
            task = "regression"
    elif multilabel and task in ("binary_classification", "multiclass_classification"):
        raise StructureError(
            f"This classifier answers each of its {len(classes)} labels with a yes or a no of its own (it was fitted "
            f"on a table of them), so it is not a {task.split('_')[0]} classifier: pass "
            "task='multilabel_classification', or leave task out.",
            code="invalid_structure",
            param="task",
        )
    if classifier and labels is None:
        labels = [str(item.item() if hasattr(item, "item") else item) for item in classes]
    if classifier and kind == "decision_function" and len(classes) > 2 and _pairwise(final):
        # Three classes have three pairs: the scores would fit a head of three classes, and be read as them.
        why.append(
            "its decision_function gives a score for each pair of classes (decision_function_shape='ovo'), not one "
            "for each class, which is what Kaimei reads: set decision_function_shape='ovr' on it (it needs no refit), "
            "or fit it with probability=True"
        )
        return None

    tools = _importer()
    failures: List[str] = []
    # The rows the check holds: the caller's own, as the doubles they are, and rows made up where the data the model
    # was fitted on lies, as far as that is known (its first scaler's statistics, the caller's rows). Rows made up
    # around zero say nothing of a column whose values are far from it (a time in seconds), and the answers they
    # give can be so large that an error on real rows looks small beside them.
    own = rows_to_check(check_rows, names, limit=OWN_CHECK_ROWS)
    ranges, points = _data_ranges(model, own, len(names), numpy)
    made = _made_table(ranges, TABLE_CHECK_ROWS, numpy)
    table = made if own is None else numpy.vstack([own, made])
    unplaced = [name for name, known in zip(names, ranges) if known is None]
    # A column whose range nothing says is made up without knowing where its real values lie, so the rows cannot
    # show which precision gives the model's numbers there. The model is then computed as it computes itself: in
    # doubles when it tells apart numbers that single precision cannot (a tree does not: scikit-learn rounds a
    # tree's inputs to single precision itself). So is a model whose data is placed only by the points it keeps
    # itself (support vectors, a nearest-neighbours model's rows): rows made up among them need not lie where the
    # model is sensitive (a kernel is zero a little way from every support vector, in either precision), so the
    # model is asked at those points themselves.
    asked_at = [made] if points is None else [made, points[:_ASKED_AT_MOST]]
    doubles = (bool(unplaced) or points is not None) and _tells_doubles_apart(model, fitted, asked_at, len(names), numpy)

    def read(method: Callable[[Any], Any], kind: str, raw_scores: bool = False) -> Tuple[Optional[Dict[str, Any]], bytes]:
        """The model read by one of its methods (``kind`` is its name), with the ONNX model it was read from.

        ``raw_scores`` asks the converter for the scores ``decision_function`` gives where it would give the
        probabilities of a classifier that has both.
        """

        def library(feeds: Dict[str, Any]) -> Any:
            rows = numpy.asarray(next(iter(feeds.values())), dtype=float).reshape(1, -1)
            return method(_table(model, rows, fitted))

        kept: Optional[Dict[str, Any]] = None
        kept_from = b""
        for precision in (numpy.float32, numpy.float64):
            try:
                data = _convert_sklearn(model, final, len(names), precision, classifier, numpy, raw_scores=raw_scores)
            except _NoConversion as exc:
                failures.append(str(exc))
                continue
            proto = tools.load_model(data)
            read_from = output
            if classifier and kind == "decision_function" and len(classes) == 2 and output is None and not multilabel:
                proto, read_from = _one_margin(tools, proto, library, len(names), precision, numpy)
            try:
                spec = _read_onnx(
                    proto,
                    library=library,
                    feature_names=names,
                    task=task,
                    labels=labels,
                    inputs=inputs,
                    input_shapes=None,
                    samples=None,
                    output=read_from,
                    mask_layout=mask_layout,
                    threshold=threshold,
                    name=name,
                    tokenizer=tokenizer,
                    check_rows=check_rows,  # read only when the inputs are not one table of numbers, which is rare
                    framework=framework or "sklearn",
                    # predict_proba's numbers are the probabilities: shown as they are, never through a softmax again.
                    probabilities=classifier and kind == "predict_proba",
                    whole_table=table,
                )
            except StructureError as exc:
                if exc.param in ("labels", "inputs", "output", "tokenizer", "threshold"):
                    raise  # the caller's own argument: another precision would not make it right
                failures.append(exc.message)
                continue
            spec.setdefault("metadata", {})["library_method"] = kind
            spec["metadata"]["model_precision"] = "double" if precision is numpy.float64 else "single"
            if kind == "decision_function" and classifier:
                squashed = (
                    "each label's share shown is its own score squashed between 0 and 1"
                    if str(spec.get("task")) == "multilabel_classification"
                    else "the shares shown are those scores squashed to add up to one"
                )
                spec["summary"] = (
                    str(spec.get("summary") or "") + f" This classifier gives scores, not probabilities: {squashed}."
                ).strip()
            if not spec.get("approximate") or kept is None:
                kept = spec
                kept_from = bytes(proto.SerializeToString())
            if not spec.get("approximate") and not (doubles and precision is numpy.float32):
                # Single precision stands when it gives the model's numbers on rows from the data's own range. Where
                # a column's range is not known the rows could not show it, and a model that tells doubles apart is
                # read in doubles when that reading is the model too.
                break
        return kept, kept_from

    kept, kept_from = read(method, kind)
    if kept is None:
        why.append("skl2onnx could not convert it (" + "; ".join(dict.fromkeys(failures)) + ")")
        return None
    if classifier:

        kept_names = [str(column) for column in kept.get("feature_names") or []]

        def own_class(row: Mapping[str, Any]) -> Any:
            """What the model's own ``predict`` gives one row of a check."""
            line = [float(numpy.asarray(row[column]).reshape(-1)[0]) for column in kept_names]
            return model.predict(_table(model, numpy.array([line]), fitted))

        kept, kept_from = _settle_its_class(
            kept, kept_from, kind, model, final, classes, own_class, read, pairwise=len(classes) > 2 and _pairwise(final)
        )
    if keep is not None:
        keep["onnx"] = kept_from
    steps = len(kept["graph"].get("nodes") or [])
    kept["summary"] = (
        f"scikit-learn {type(final).__name__} converted to ONNX and read as a graph of {steps:,} operations. "
        + str(kept.get("summary") or "")
    ).strip()
    if not kept.get("approximate"):
        kept["summary"] += _precision_said(
            (kept.get("metadata") or {}).get("model_precision"), unplaced, asked_the_model=doubles and points is not None
        )
    return kept


def _data_ranges(model: Any, own: Any, width: int, numpy: Any) -> Tuple[List[Any], Optional[Any]]:
    """``(where the values of each column lie in the data a scikit-learn model was fitted on, as far as it is known;
    the points the model itself keeps of that data, when they are what placed a column)``.

    For each column: ``("normal", mean, deviation)`` or ``("between", low, high)`` from the statistics of the scaler
    a pipeline reads its table with first (a ``StandardScaler``'s mean and deviation, a ``MinMaxScaler``'s smallest
    and largest value, a ``RobustScaler``'s median and quartile range); else ``("values", [...])``, the caller's own
    values of the column (``own``, a table of doubles); else what the model keeps of its data
    (:func:`_fitted_points`): the values of a nearest-neighbours model's rows or of its classes' means, or
    ``("between", low, high)`` of a kernel model's support vectors; else ``None``.
    """
    found: List[Any] = [None] * width
    steps = getattr(model, "steps", None)
    first = steps[0][1] if isinstance(steps, list) and len(steps) > 1 and isinstance(steps[0], (list, tuple)) else None

    def of(name: str) -> Optional[Any]:
        try:
            values = numpy.asarray(getattr(first, name, None), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        return values if values.size == width and bool(numpy.isfinite(values).all()) else None

    kind = type(first).__name__ if first is not None else ""
    if kind == "StandardScaler":
        mean, spread = of("mean_"), of("var_")
        if mean is not None and spread is not None:
            found = [("normal", float(a), float(numpy.sqrt(b))) for a, b in zip(mean, spread)]
    elif kind == "MinMaxScaler":
        low, high = of("data_min_"), of("data_max_")
        if low is not None and high is not None:
            found = [("between", float(a), float(b)) for a, b in zip(low, high)]
    elif kind == "RobustScaler":
        centre, spread = of("center_"), of("scale_")
        if centre is not None and spread is not None:
            found = [("between", float(a - b), float(a + b)) for a, b in zip(centre, spread)]
    if own is not None and getattr(own, "ndim", 0) == 2 and own.shape[1] == width and own.shape[0]:
        found = [known if known is not None else ("values", own[:, at].tolist()) for at, known in enumerate(found)]
    points: Optional[Any] = None
    if any(known is None for known in found) and not isinstance(steps, list):
        points, at_the_boundary = _fitted_points(model, width, numpy)
        if points is not None:
            # Support vectors are the rows nearest a boundary, not rows like any other: they say between what
            # values the data lies, and rows are made up anywhere between those.
            kept = [
                ("between", float(points[:, at].min()), float(points[:, at].max())) if at_the_boundary else ("values", points[:, at].tolist())
                for at in range(width)
            ]
            found = [known if known is not None else kept[at] for at, known in enumerate(found)]
    return found, points


#: Where a fitted scikit-learn model that reads the table itself keeps rows of the data it was fitted on, or points
#: among them: a kernel model's support vectors, a nearest-neighbours model's rows, the means of classes or groups.
_FITTED_POINTS = ("support_vectors_", "_fit_X", "X_fit_", "X_train_", "theta_", "means_", "cluster_centers_", "centroids_")
#: Models made of models that are each given the same table: what one of their members keeps of the data is the
#: whole model's. (Not a bag, whose members may read the columns in another order, nor a stack's final model.)
_SAME_TABLE = ("OneVsRestClassifier", "OneVsOneClassifier", "VotingClassifier", "VotingRegressor")
#: Rows of a model's own points it is asked about, at most, when it is asked whether it tells doubles apart.
_ASKED_AT_MOST = 64
#: Rows of them a made-up row's values are drawn from, at most.
_MOST_FITTED_POINTS = 2000


def _fitted_points(model: Any, width: int, numpy: Any, depth: int = 0) -> Tuple[Optional[Any], bool]:
    """``(points of the data a scikit-learn model was fitted on, as the model itself keeps them; whether they are
    support vectors)``: a table of doubles with a column for each feature, or ``None`` when it keeps none this
    reads. A naive Bayes model's class means come with a point a deviation to either side of each. A one-vs-rest
    model, a vote or a calibration is asked through the models it is made of, which are given the same table."""
    members: List[Any] = []
    if depth < 3 and type(model).__name__ in _SAME_TABLE:
        members = list(getattr(model, "estimators_", None) or [])
    elif depth < 3 and type(model).__name__ == "CalibratedClassifierCV":
        members = [getattr(fold, "estimator", None) for fold in getattr(model, "calibrated_classifiers_", None) or []]
    for member in members:
        found, at_the_boundary = _fitted_points(member, width, numpy, depth + 1)
        if found is not None:
            return found, at_the_boundary
    for name in _FITTED_POINTS:
        try:
            kept = getattr(model, name, None)
            if kept is None or hasattr(kept, "toarray"):
                continue  # none, or a sparse table: not made dense for this
            points = numpy.asarray(kept, dtype=float)
            if points.ndim != 2 or points.shape[1] != width or not points.shape[0]:
                continue
            points = points[:_MOST_FITTED_POINTS]
            if name == "theta_":
                spread = numpy.sqrt(numpy.asarray(getattr(model, "var_", None), dtype=float))
                if spread.shape == points.shape:
                    points = numpy.vstack([points, points - spread, points + spread])
            if bool(numpy.isfinite(points).all()):
                return points, name == "support_vectors_"
        except Exception:  # noqa: BLE001 - an attribute of another kind on this model: the next one is tried
            continue
    return None, False


def _made_table(ranges: Sequence[Any], count: int, numpy: Any) -> Any:
    """``count`` made-up rows of a table, as doubles: each column's value drawn where its data lies
    (:func:`_data_ranges`). A column whose range is not known is drawn around zero, as rows were always made up:
    between 0 and 1 in the first four rows, around zero with a spread of one in the rest."""
    generator = numpy.random.default_rng(0)
    table = numpy.empty((count, len(ranges)), dtype=numpy.float64)
    for index in range(count):
        for at, known in enumerate(ranges):
            if known is None:
                drawn = generator.normal(size=1) if index >= 4 else generator.uniform(0.0, 1.0, size=1)
                value = float(drawn.round(6)[0])
            elif known[0] == "normal":
                value = float(generator.normal(known[1], known[2]))
            elif known[0] == "between":
                value = float(generator.uniform(known[1], known[2]))
            else:
                value = float(known[1][int(generator.integers(0, len(known[1])))])
            table[index, at] = value
    return table


#: The methods a scikit-learn model is asked by whether it tells doubles apart: its scores before they are squashed
#: first (probabilities far from a model's data are 0 and 1 whatever the precision), then the rest.
_ASKED_ABOUT_DOUBLES = ("decision_function", "predict_log_proba", "score_samples", "predict_proba", "transform", "predict")


def _tells_doubles_apart(model: Any, fitted: Optional[Sequence[str]], tables: Sequence[Any], width: int, numpy: Any) -> bool:
    """Whether a scikit-learn model answers a number differently from what single precision rounds it to.

    A linear model, a support vector machine or a network does: scikit-learn computes them in doubles, and so must
    Kaimei for a column of large numbers that differ by little (a time in seconds). A tree does not: scikit-learn
    rounds a tree's inputs to single precision itself, and a tree read in doubles would send a number that lies
    between two single-precision ones down another branch than scikit-learn does. So the model is asked, by each
    method it has that gives numbers: about ``tables`` (rows of its check) and about whole numbers single precision
    cannot hold (odd ones past 2**24), each as it is and as single precision rounds it. One answer that differs
    says it does. ``False`` when it cannot be asked.
    """
    odd = 2.0**24 + 1.0 + 2.0 * numpy.random.default_rng(0).integers(0, 2**22, size=(16, width)).astype(numpy.float64)
    for name in _ASKED_ABOUT_DOUBLES:
        try:
            method = getattr(model, name, None)
            if not callable(method):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for probe in (odd, *tables):
                    exact = numpy.asarray(method(_table(model, probe, fitted)), dtype=float)
                    single = numpy.asarray(
                        method(_table(model, probe.astype(numpy.float32).astype(numpy.float64), fitted)), dtype=float
                    )
                    if not numpy.array_equal(exact, single, equal_nan=True):
                        return True
        except Exception:  # noqa: BLE001 - a method that does not apply, or gives no numbers: the next one is asked
            continue
    return False


def _either(names: Sequence[str], limit: int = 6) -> str:
    """Column names as a summary lists them: ``age``, ``age or income``, ``age, income or 3 others``."""
    shown = [str(name) for name in names[:limit]]
    if len(names) > limit:
        shown.append(f"{len(names) - limit} other{'s' if len(names) - limit != 1 else ''}")
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " or " + shown[-1]


def _padded_as_the_model_pads(hint: Any, pad_token: Any) -> Any:
    """A decoder input the caller hints ``{"generated": True}`` without a ``pad_id`` is filled with the model's own
    padding token, as it is when the caller says nothing: the structure then names one padding token, not two."""
    if isinstance(hint, Mapping) and hint.get("generated") and hint.get("pad_id") is None and pad_token is not None:
        return {**hint, "pad_id": int(pad_token)}
    return hint


def _hinted_feature(hint: Any) -> Optional[str]:
    """The feature a hint names for an input: ``"answer"`` or ``{"feature": "answer", ...}``."""
    if isinstance(hint, str):
        return hint
    if isinstance(hint, Mapping) and hint.get("feature"):
        return str(hint["feature"])
    return None


def _vocabulary_limits(hints: Mapping[str, Any]) -> Dict[str, int]:
    """How many tokens each text feature has, where a hint says (``vocab``): the made-up rows of a check draw their
    tokens from it. A hint that only says ``{"generated": True}`` is for the feature the importer calls
    ``decoder``."""
    limits: Dict[str, int] = {}
    for hint in hints.values():
        if not isinstance(hint, Mapping) or not hint.get("vocab"):
            continue
        feature = hint.get("feature") or ("decoder" if hint.get("generated") else None)
        if feature:
            limits[str(feature)] = int(hint["vocab"])
    return limits


def _precision_said(precision: Any, unplaced: Sequence[str], asked_the_model: bool = False) -> str:
    """What a summary says of the precision a scikit-learn model's reading is computed in, and of values it could not
    be checked near. ``asked_the_model`` says double precision was chosen because the model tells doubles apart and
    its data is placed only by the points it keeps itself, not because single precision was seen to fail."""
    note = ""
    if precision == "double" and not unplaced and asked_the_model:
        note += (
            " It is computed in double precision, as scikit-learn computes a model that tells apart numbers single "
            "precision cannot: it was given no rows of its table, so single precision could not be checked on values "
            "like yours (give rows with X= or check_rows= to check it)."
        )
    elif precision == "double" and not unplaced:
        note += (
            " It is computed in double precision: in single precision it did not give the model's own numbers on rows "
            "from the range of its data."
        )
    if unplaced:
        note += (
            f" Nothing says where the values of {_either(unplaced)} lie (the model keeps no such statistics, and it was "
            "given no rows of its table), so it was checked on made-up values of them that may lie far from yours: "
            "give rows of the table (X= or check_rows=) to check it on values like yours."
        )
        if precision == "double":
            note += (
                " It is computed in double precision, as scikit-learn computes a model that tells apart numbers single "
                "precision cannot."
            )
    return note


def _settle_its_class(
    spec: Dict[str, Any],
    read_from: bytes,
    kind: str,
    model: Any,
    final: Any,
    classes: Sequence[Any],
    own_class: Callable[[Mapping[str, Any]], Any],
    read: Callable[..., Tuple[Optional[Dict[str, Any]], bytes]],
    pairwise: bool = False,
) -> Tuple[Dict[str, Any], bytes]:
    """A scikit-learn classifier's reading, settled so that the class Kaimei reports is the model's own.

    A reading whose scores are the model's is still not the model when the class Kaimei takes from those scores is
    not the one ``predict`` gives. So the two are compared on every row of the check (:func:`classes_differ`), after a
    threshold wrapper's own threshold is written into the reading. Where they differ, the model's
    ``decision_function`` is read in place of its probabilities, and kept if its class is the model's on every row.
    When no reading's is, the reading is marked as not the model (:func:`not_its_class`): drawn, not run.

    What is known to differ on rows the check may not hold is still said, and warned of, on a reading whose check
    rows all agree: an SVC that counts one-vs-one votes (the note for its probabilities is
    ``structure.PREDICTED_CLASS_NOTE``, added where the structure is extracted).
    """
    if spec.get("approximate") or "check" not in spec or str(spec.get("task") or "") not in _CLASSIFIERS:
        return spec, read_from
    wrapper = decision_threshold(final)
    if wrapper is not None and wrapper["method"] != kind:
        # Its threshold is on the scores of the method that was not read: that method is read in its place.
        other = getattr(model, str(wrapper["method"]), None)
        found, found_from = read(other, str(wrapper["method"]), True) if callable(other) else (None, b"")
        if found is None or found.get("approximate") or "check" not in found:
            not_its_threshold(
                spec,
                f"its threshold is set on its {wrapper['method']}, which could not be read as a graph, and its "
                f"{kind} is what was read",
            )
            return spec, read_from
        spec, read_from, kind = found, found_from, str(wrapper["method"])
    if wrapper is not None:
        refused = carry_threshold(spec, wrapper, classes, kind)
        if refused is not None:
            not_its_threshold(spec, refused)
            return spec, read_from
    differ, count = classes_differ(spec, own_class, classes)
    if differ and wrapper is None and kind == "predict_proba" and not pairwise:
        other = getattr(model, "decision_function", None)
        found, found_from = read(other, "decision_function", True) if callable(other) else (None, b"")
        if found is not None and not found.get("approximate") and "check" in found:
            again, compared = classes_differ(found, own_class, classes)
            if compared and not again:
                found["summary"] = (
                    f"{found.get('summary') or ''} Its predict() gives another class than the most probable one of "
                    f"its predict_proba on {differ:,} of the {count:,} rows it was checked on, so its "
                    "decision_function is what was read: Kaimei's class for a row is the model's own."
                ).strip()
                spec, read_from, kind, differ = found, found_from, "decision_function", 0
    votes = kind == "decision_function" and _counts_votes(final)
    if differ:
        from .ensembles import predicts_by_its_margin

        if votes:
            reason = (
                "its predict() counts one-vs-one votes, which the highest one-vs-rest score does not always follow; "
                "with break_ties=True on the model, which needs no refit, the two agree"
            )
        elif kind == "predict_proba" and predicts_by_its_margin(model):
            reason = "its predict() goes by the margin, and its probabilities come from a separate fit"
        elif wrapper is not None:
            reason = "with its own threshold carried"
        else:
            reason = ""
        not_its_class(spec, differ, count, reason)
        return spec, read_from
    if votes:
        spec.setdefault("metadata", {})["predicted_class"] = VOTES_NOTE
        spec["summary"] = (
            f"{spec.get('summary') or ''} Its class for a row is the highest one-vs-rest score, which the model's own "
            "predict() can order differently where one-vs-one votes tie."
        ).strip()
        warnings.warn(
            "Kaimei runs this model's decision_function, converted to ONNX, and its class for a row is the class with "
            f"the highest score. {VOTES_NOTE} Set break_ties=True on the model (it needs no refit) for predict() to go "
            "by the same scores.",
            UserWarning,
            stacklevel=4,
        )
    return spec, read_from


#: Said in a structure's metadata, under ``predicted_class`` as ``structure.PREDICTED_CLASS_NOTE`` is, when the class
#: Kaimei gives a row is the highest one-vs-rest score of an SVC or NuSVC, which its own ``predict()`` can order
#: differently.
VOTES_NOTE = (
    "The class with the highest one-vs-rest score of decision_function. The model's own predict() counts one-vs-one "
    "votes instead, so on rows where two classes get as many votes the two can differ (with break_ties=True on the "
    "model they agree)."
)


def _pairwise(final: Any) -> bool:
    """Whether a libsvm classifier's ``decision_function`` gives a score for each pair of classes, not for each class."""
    return str(getattr(final, "decision_function_shape", "") or "") == "ovo"


def _counts_votes(final: Any) -> bool:
    """Whether a classifier's ``predict`` counts one-vs-one votes, which its one-vs-rest scores can order differently
    where votes tie: an SVC or NuSVC of more than two classes, unless ``break_ties`` makes it go by those scores."""
    classes = getattr(final, "classes_", None)
    return (
        hasattr(final, "decision_function_shape")
        and classes is not None
        and len(classes) > 2
        and not _pairwise(final)
        and not bool(getattr(final, "break_ties", False))
    )


#: Two probabilities this close to an even split are one, which the API's engine gives to the first class (its
#: ``reporting.TIE``), as ``predict`` gives an even split in scikit-learn.
EVEN_SPLIT = 1e-15
#: How a summary begins to say that a reading's class is not the model's own, where the reason for a drawing is read.
NOT_ITS_CLASS = "Kaimei's class for a row would differ from the model's own predict()"


def _class_names(spec: Mapping[str, Any]) -> List[str]:
    """The names of a classifier's classes, in the order of its output, as a structure of either kind keeps them."""
    found = (spec.get("outputs") or {}).get("labels") or spec.get("labels") or spec.get("classes") or []
    return [str(label) for label in found]


def reported_class(spec: Mapping[str, Any], values: Any) -> Any:
    """What Kaimei reports for a row of a classifier whose output, as its head shows it, is ``values``.

    The position of the class among the structure's labels: the highest value, and the first of the highest when
    several are. A yes-or-no model gives its positive class (the second, unless ``outputs.positive_label`` names the
    first) when that class's probability is ``outputs.threshold`` or more; with no threshold that is one half, and an
    even split is the first class. One value alone is the second class's probability. A multilabel model gives a
    flag for each label instead: whether its value is the threshold (one half unless ``outputs`` says) or more.

    This is the rule of the API's engine (``binary_choice`` in its ``reporting.py``, and its ``labels_output``), kept
    in step with it by hand: a class is compared with a model's own ``predict`` by the rule that will report it.
    """
    numpy = _require_numpy()
    flat = numpy.asarray(values, dtype=float).reshape(-1)
    outputs = dict(spec.get("outputs") or {})
    cut = 0.5 if outputs.get("threshold") is None else float(outputs["threshold"])
    task = str(spec.get("task") or "")
    if task == "multilabel_classification":
        return flat >= cut
    flat = numpy.clip(flat, 0.0, 1.0)
    if task == "binary_classification" and flat.size <= 2:
        if flat.size == 1:
            flat = numpy.array([1.0 - float(flat[0]), float(flat[0])])
        names = _class_names(spec)
        positive = outputs.get("positive_label")
        at = names.index(str(positive)) if positive is not None and str(positive) in names else 1
        score = float(flat[at])
        if cut == 0.5 and abs(score - 0.5) <= EVEN_SPLIT:
            return 0
        return at if score >= cut else 1 - at
    return int(flat.argmax())


def classes_differ(
    spec: Mapping[str, Any], predict: Any, classes: Optional[Sequence[Any]] = None
) -> Tuple[int, int]:
    """``(on how many rows of the spec's check Kaimei's class is not the model's own, how many rows were compared)``.

    A reading is the model only if, besides its scores, the class Kaimei reports for a row is the class the model's
    own ``predict`` gives it. ``spec`` is a classifier's structure with its check: a graph's, written feature by
    feature (``{"inputs": {feature: [a row's values, ...]}, "outputs": [...]}``), or a table's (``{"rows": [...],
    "outputs": [...]}``, columns in the order of ``feature_names``). Kaimei's class for a row is read off the check's
    own output for it by :func:`reported_class`, so by the spec's task, its labels, and the ``threshold`` and
    ``positive_label`` its ``outputs`` hold at the time of the call.

    ``predict`` is the model's own answer. Either a function, called once for each row of the check with ``{feature
    name: that row's values for the feature, as an array of floats}`` (a column of a table is an array of one value,
    a category its position among the feature's categories, as the check holds it) and returning what the model's
    ``predict`` gives that one row; or the answers themselves, one for each row, in the check's order. An answer is
    one class (a label, or an array holding one), or for a multilabel model a yes or a no for each label.

    ``classes`` are the model's own classes in the order of its outputs (scikit-learn's ``classes_``), for when the
    spec's labels are other names for them; left out, the labels are the classes. A class is compared with an
    answer as text.

    A row is left out of both counts when ``predict`` raises for it, answers in another shape, or answers with
    what is none of the classes by name (a model that writes its class 1 as 1.0). ``(0, 0)`` says
    nothing could be compared: the spec has no check with rows, it is no classifier's, or it is a reading of
    Kaimei's own readers that settles an even split by the margins behind it (``combine.ties``), which the check's
    probabilities cannot show and those readers compare for themselves.
    """
    numpy = _require_numpy()
    if str(spec.get("task") or "") not in _CLASSIFIERS or (spec.get("combine") or {}).get("ties"):
        return 0, 0
    check = spec.get("check")
    if not isinstance(check, Mapping):
        return 0, 0
    try:
        outputs = [numpy.asarray(row, dtype=float).reshape(-1) for row in check.get("outputs") or []]
        rows = _check_rows_of(spec, check, len(outputs), numpy)
    except (TypeError, ValueError, KeyError, IndexError):
        return 0, 0
    if rows is None or not outputs:
        return 0, 0
    if not callable(predict):
        try:
            answers = list(predict)
        except TypeError:
            return 0, 0
        if len(answers) != len(outputs):
            return 0, 0
    known = [str(item) for item in classes] if classes is not None else _class_names(spec)
    multilabel = str(spec.get("task")) == "multilabel_classification"
    differ = compared = 0
    for index, values in enumerate(outputs):
        try:
            if callable(predict):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    answer = predict(rows[index])
            else:
                answer = answers[index]
            own = numpy.asarray(_plain_array(answer), dtype=object).reshape(-1)
            ours = reported_class(spec, values)
        except Exception:  # noqa: BLE001 - a row the model's own predict refuses says nothing either way
            continue
        if multilabel:
            if own.size != ours.size:
                continue
            try:
                same = bool((own.astype(float) != 0).tolist() == ours.tolist())
            except (TypeError, ValueError):
                continue
        else:
            if own.size != 1 or not 0 <= int(ours) < len(known) or str(own[0]) not in known:
                continue  # an answer that is none of the classes by name (1.0 for a class 1) cannot be told apart
            same = known[int(ours)] == str(own[0])
        compared += 1
        differ += 0 if same else 1
    return differ, compared


def _check_rows_of(spec: Mapping[str, Any], check: Mapping[str, Any], count: int, numpy: Any) -> Optional[List[Dict[str, Any]]]:
    """The rows of a check as ``{feature: values}``, whichever way the check is written; ``None`` when it holds none
    (a check of pictures made again from a seed)."""
    inputs = check.get("inputs")
    if isinstance(inputs, Mapping) and inputs:
        if any(len(rows) != count for rows in inputs.values()):
            return None
        return [
            {str(name): numpy.asarray(rows[index], dtype=float).reshape(-1) for name, rows in inputs.items()}
            for index in range(count)
        ]
    table = check.get("rows")
    names = [str(name) for name in spec.get("feature_names") or []]
    if not isinstance(table, (list, tuple)) or len(table) != count or not names or "image_shape" in check:
        return None
    if any(len(row) != len(names) for row in table):
        return None
    return [
        {name: numpy.array([numpy.nan if value is None else float(value)]) for name, value in zip(names, row)}
        for row in table
    ]


def not_its_class(spec: Dict[str, Any], differ: int, count: int, why: str = "") -> None:
    """Mark a reading whose class is not the model's own on ``differ`` of the ``count`` rows of its check: it is not
    the model, so it is drawn, not run, and its summary says on how many rows and for what reason: ``why`` (a
    clause, such as ``"its predict() counts votes"``), or, when the caller knows no more, plainly which of the
    model's numbers its class does not follow (its probabilities, when the head shows them as they are; its
    scores otherwise). Its check goes: nothing is left for the API to run."""
    if not why:
        shown_as_they_are = str((spec.get("head") or {}).get("link") or "none") == "none"
        why = (
            "the class its predict() gives is not always its most probable one"
            if shown_as_they_are
            else "the class its predict() gives is not always the one with its highest score"
        )
    reason = f" ({why})" if why else ""
    _drawn(spec, f"{NOT_ITS_CLASS} on {differ:,} of the {count:,} rows it was checked on{reason}")
    spec.pop("check", None)
    spec.setdefault("metadata", {})["predicted_class"] = (
        f"Not the model's own on {differ:,} of the {count:,} rows it was checked on{reason}."
    )


def not_its_threshold(spec: Dict[str, Any], why: str) -> None:
    """Mark a reading of a model that predicts with a threshold of its own which the reading cannot carry: its class
    would not be the model's, so it is drawn, not run, and its summary says ``why`` (a clause)."""
    _drawn(spec, f"{NOT_ITS_CLASS}: {why}")
    spec.pop("check", None)
    spec.setdefault("metadata", {})["predicted_class"] = f"Not the model's own: {why}."


def _is_threshold_wrapper(final: Any) -> bool:
    """scikit-learn's classifiers that predict with a threshold of their own on another classifier's scores."""
    kind = type(final)
    named = kind.__name__ in ("TunedThresholdClassifierCV", "FixedThresholdClassifier")
    return named and str(getattr(kind, "__module__", "")).startswith("sklearn.")


def _wrapped_by_threshold(final: Any) -> Any:
    """The classifier a threshold wrapper takes its scores from: the one it fitted, or the fitted one it was given."""
    inner = getattr(final, "estimator_", None)
    return inner if inner is not None else getattr(final, "estimator", None)


def decision_threshold(final: Any) -> Optional[Dict[str, Any]]:
    """What a scikit-learn threshold wrapper predicts with, or ``None`` for any other model.

    ``TunedThresholdClassifierCV`` and ``FixedThresholdClassifier`` give a row their positive class when a score of
    the classifier they wrap is their threshold or more. Returns ``{"threshold": the number, "method": the method
    that score is of ("predict_proba" or "decision_function"), "positive": the positive class, or None for the
    second of the two}``.
    """
    if not _is_threshold_wrapper(final):
        return None
    inner = _wrapped_by_threshold(final)
    wanted = str(getattr(final, "response_method", "auto") or "auto")
    methods = ("predict_proba", "decision_function") if wanted == "auto" else (wanted,)
    method = next((name for name in methods if callable(getattr(inner, name, None))), None)
    if inner is None or method is None:
        return None
    positive: Any = None
    if type(final).__name__ == "TunedThresholdClassifierCV":
        threshold = getattr(final, "best_threshold_", None)
        try:
            positive = final._curve_scorer._get_pos_label()  # what its own predict asks, and nothing public says
        except Exception:  # noqa: BLE001 - another scikit-learn's wrapper: the comparison of classes decides
            positive = None
    else:
        threshold = getattr(final, "threshold", "auto")
        if isinstance(threshold, str):
            threshold = 0.5 if method == "predict_proba" else 0.0  # "auto"
        positive = getattr(final, "pos_label", None)
    try:
        number = float(threshold)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return {"threshold": number, "method": method, "positive": positive}


#: A threshold on a decision score further out than this is not carried: squashed, it is too near 0 or 1 to tell
#: scores apart.
_FARTHEST_SCORE_THRESHOLD = 30.0


def carry_threshold(
    spec: Dict[str, Any], found: Mapping[str, Any], classes: Sequence[Any], kind: str
) -> Optional[str]:
    """Write a threshold wrapper's threshold (``found``, from :func:`decision_threshold`) into ``spec["outputs"]``, in
    the units the reading's head reports. Returns ``None``, or why it cannot be carried, as a clause.

    ``kind`` is the method the spec was read by. Read by ``predict_proba``, the head reports the two classes'
    probabilities and the threshold is carried as it is. Read by ``decision_function``, the head reports the one
    margin squashed (the second class's share), and so is the threshold: a margin is the threshold or more exactly
    when its squashed value is the squashed threshold or more. ``outputs.positive_label`` names the first class when
    that is the wrapper's positive one. The model gives its positive class at the threshold itself; Kaimei gives an
    even split at one half to the first class, so a threshold of exactly one half is carried as the largest number
    below it, which no probability lies between.
    """
    numpy = _require_numpy()
    check = spec.get("check") or {}
    width = len((check.get("outputs") or [[]])[0])
    known = [str(item) for item in classes]
    names = _class_names(spec)
    if str(spec.get("task") or "") != "binary_classification" or len(known) != 2 or len(names) != 2:
        return "its threshold decides between two classes, and this reading is not of a yes-or-no model"
    if str(found.get("method")) != kind:
        return f"its threshold is set on its {found.get('method')}, and its {kind} is what was read"
    threshold = float(found["threshold"])
    link = str((spec.get("head") or {}).get("link") or "none")
    if kind == "predict_proba":
        if width != 2 or link != "none":
            return "its probabilities were not read as the two classes' own"
        if not 0.0 <= threshold <= 1.0:
            return f"its threshold on a probability is {threshold:g}, which is no probability"
        cut = threshold
    else:
        if width != 1 or link != "sigmoid":
            return "its decision score was not read as one margin"
        if not abs(threshold) <= _FARTHEST_SCORE_THRESHOLD:
            return (
                f"its threshold on its decision score is {threshold:g}, too far out for the squashed score Kaimei "
                "reports to tell scores apart"
            )
        cut = float(1.0 / (1.0 + numpy.exp(-threshold)))
    if cut == 0.5:
        cut = float(numpy.nextafter(0.5, 0.0))
    outputs = dict(spec.get("outputs") or {})
    positive = found.get("positive")
    if positive is not None and str(positive) == known[0]:
        outputs["positive_label"] = names[0]
    elif positive is not None and str(positive) != known[1]:
        return f"its positive class, {positive!r}, is not one of its two classes"
    outputs["threshold"] = cut
    spec["outputs"] = outputs
    shown = "the model's own decision threshold" if kind == "predict_proba" else "the model's own threshold on its decision score, squashed as the score is"
    spec["summary"] = (
        f"{spec.get('summary') or ''} Its class for a row is {outputs.get('positive_label') or names[1]} when that "
        f"class's share is {cut:.6g} or more: {shown}."
    ).strip()
    return None


def _with_another_end(model: Any, final: Any, other: Any) -> Optional[Any]:
    """``model`` with ``other`` in the place of ``final``, the estimator at its end: ``other`` itself, or a copy of
    the pipeline with its last step swapped. The caller's model is not changed. ``None`` when ``final`` is not at
    the model's end in a way this can swap."""
    if model is final:
        return other
    steps = getattr(model, "steps", None)
    if isinstance(steps, list) and steps and steps[-1][1] is final:
        import copy

        swapped = copy.copy(model)
        swapped.steps = [*steps[:-1], (steps[-1][0], other)]
        return swapped
    return None


def _without_its_threshold(model: Any, final: Any) -> Tuple[Any, Any]:
    """``(model, final)`` with a threshold wrapper at the model's end replaced by the classifier it wraps.

    The wrapper's scores are that classifier's, and the threshold it predicts with travels in the structure's
    ``outputs`` (:func:`carry_threshold`). skl2onnx has no converter for ``FixedThresholdClassifier``, and the label
    its converter for ``TunedThresholdClassifierCV`` gives leaves the threshold out. The model itself is not
    changed: a pipeline is copied, with its last step swapped.
    """
    for _ in range(4):
        inner = _wrapped_by_threshold(final) if _is_threshold_wrapper(final) else None
        if inner is None:
            break
        swapped = _with_another_end(model, final, inner)
        if swapped is None:
            break
        model = swapped
        final = inner.steps[-1][1] if isinstance(getattr(inner, "steps", None), list) and inner.steps else inner
    return model, final


def _has_probabilities_of_its_own(part: Any, numpy: Any) -> bool:
    """An SVC or NuSVC fitted with ``probability=True``: a libsvm classifier that keeps the sigmoid it fitted over
    its margins for ``predict_proba``."""
    from .ensembles import _classes_of

    if not _classes_of(part) & {"SVC", "NuSVC"}:
        return False
    try:
        return int(numpy.asarray(part._probA).size) > 0
    except Exception:  # noqa: BLE001 - another scikit-learn's SVC, which keeps them elsewhere: left as it is
        return False


def _without_its_probabilities(model: Any, final: Any, numpy: Any) -> Tuple[Any, Any]:
    """``(model, final)`` for a conversion that has to give the scores of ``decision_function``, with every SVC or
    NuSVC fitted with ``probability=True`` at the model's end stood in for by a copy that keeps no probabilities.

    skl2onnx converts such a classifier to its probabilities whatever it is asked for (asked for raw scores, it
    gives them negated), so its ``decision_function`` could never be read, and a model whose ``predict`` goes by
    those scores could only be drawn once its probabilities disagreed with it on a row. The scores do not depend
    on the probabilities: a copy that says it was fitted without them converts to the model's own
    ``decision_function``, which the check then compares with the model's as it compares any reading. The same is
    done for the classifiers of a one-vs-rest model, whose ``predict`` goes by their scores. The caller's model is
    not changed.
    """
    import copy

    def plain(part: Any) -> Any:
        made = copy.copy(part)
        made.probability = False
        made._probA = numpy.empty(0)
        made._probB = numpy.empty(0)
        return made

    other: Any = None
    members = getattr(final, "estimators_", None)
    if _has_probabilities_of_its_own(final, numpy):
        other = plain(final)
    elif type(final).__name__ == "OneVsRestClassifier" and isinstance(members, list) and any(
        _has_probabilities_of_its_own(member, numpy) for member in members
    ):
        other = copy.copy(final)
        other.estimators_ = [plain(member) if _has_probabilities_of_its_own(member, numpy) else member for member in members]
    swapped = _with_another_end(model, final, other) if other is not None else None
    return (swapped, other) if swapped is not None else (model, final)


def _per_output(found: Any) -> bool:
    """Whether ``classes_`` lists the classes of each output apart (a list of arrays), as a multi-output fit keeps them."""
    return (
        isinstance(found, (list, tuple))
        and bool(found)
        and all(hasattr(item, "__len__") and not isinstance(item, (str, bytes)) for item in found)
    )


def _multilabel(model: Any, final: Any, width: int, fitted: Optional[Sequence[str]], numpy: Any) -> bool:
    """Whether a scikit-learn classifier answers a row with a yes or a no for each of its labels (it was fitted on a
    table of them, an indicator matrix), rather than with one of its classes.

    Its probabilities are then one for each label, on their own, and do not add up to one. scikit-learn says so in
    ``multilabel_`` (``OneVsRestClassifier``) or in the kind of target its label binarizer saw (an MLP, a ridge
    classifier); any classifier says so by what ``predict`` gives a row: a yes or a no for each label.
    """
    for holder in (final, model):
        try:
            said = getattr(holder, "multilabel_", None)
        except Exception:  # noqa: BLE001 - a property that cannot say, on an estimator that is not fitted
            said = None
        if isinstance(said, (bool, numpy.bool_)) and bool(said):
            return True
        for name in ("label_binarizer_", "_label_binarizer"):
            kind = str(getattr(getattr(holder, name, None), "y_type_", "") or "")
            if kind.startswith("multilabel"):
                return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            answer = numpy.asarray(model.predict(_table(model, numpy.zeros((1, width)), fitted)))
    except Exception:  # noqa: BLE001 - a row of zeros the model refuses says nothing either way
        return False
    return answer.ndim == 2 and answer.shape[0] == 1 and answer.shape[1] > 1


def _is_clusterer(estimator: Any) -> bool:
    """A scikit-learn clusterer (KMeans, a Gaussian mixture): it puts a row in a group, with no labels to learn.

    scikit-learn calls a mixture a density estimator; it is read as a clustering all the same, by the probability
    of each of its components.
    """
    mixture = hasattr(estimator, "n_components") and hasattr(estimator, "weights_") and not hasattr(estimator, "classes_")
    if mixture and callable(getattr(estimator, "predict_proba", None)):
        return True
    return _sklearn_says(estimator, "is_clusterer", "clusterer")


def _cluster_answer(model: Any) -> Tuple[Optional[Callable[[Any], Any]], str]:
    """A clusterer's numbers for a row: its distance to each centre, or the probability of each group."""
    for name in ("transform", "predict_proba"):
        method = getattr(model, name, None)
        if callable(method):
            return method, name
    return None, ""


def _is_regressor(estimator: Any) -> bool:
    return _sklearn_says(estimator, "is_regressor", "regressor")


def _sklearn_says(estimator: Any, function: str, kind: str) -> bool:
    """Whether scikit-learn's own ``sklearn.base.<function>`` says the estimator is of ``kind``.

    scikit-learn 1.6 and later keep what an estimator is in its tags and no longer give it ``_estimator_type``;
    that attribute is only read when scikit-learn cannot say (an older one, or a library's estimator that sets it
    without tags).
    """
    try:
        import sklearn.base

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if getattr(sklearn.base, function)(estimator):
                return True
    except Exception:  # noqa: BLE001 - scikit-learn missing, too old, or unable to read this object's tags
        pass
    try:
        return str(getattr(estimator, "_estimator_type", "") or "") == kind
    except Exception:  # noqa: BLE001 - an attribute that raises says nothing
        return False


def _sklearn_answer(model: Any, classifier: bool) -> Tuple[Optional[Callable[[Any], Any]], str]:
    """The method whose numbers the conversion has to give, and its name."""
    order = ("predict_proba", "decision_function") if classifier else ("predict", "decision_function", "score_samples", "transform")
    if not classifier and not _is_regressor(model) and callable(getattr(model, "decision_function", None)):
        order = ("decision_function", "score_samples", "predict", "transform")  # an outlier detector's score
    for name in order:
        try:
            method = getattr(model, name, None)
        except Exception:  # noqa: BLE001 - scikit-learn raises from the attribute itself when a method does not apply
            method = None
        if callable(method):
            return method, name
    return None, ""


def _table(model: Any, rows: Any, fitted: Optional[Sequence[str]]) -> Any:
    """Rows as the model was fitted on them: a table with its column names, or a bare array."""
    if fitted is None:
        return rows
    try:
        import pandas
    except ImportError:
        return rows
    return pandas.DataFrame(rows, columns=list(fitted))


#: Where a fitted scikit-learn model keeps the models it is made of, besides what ``ensembles._parts_of`` follows:
#: an older bag's, a vote's and a stack's by name, a column transformer's, a feature union's.
_MORE_PARTS = ("base_estimator_", "named_estimators_", "transformers_", "transformer_list", "classifier_")
#: How many objects, at most, are looked into for a tree no conversion returns from: more than any model holds,
#: and an end for an object that makes up a new part each time it is asked for one.
_MOST_PARTS = 1_000_000


def _trees_with_infinite_thresholds(estimator: Any, numpy: Any, depth: int = 0) -> bool:
    """Whether a tree in ``estimator`` (or in any model it is made of) splits at an infinite threshold.

    A tree fitted on rows with missing values learns where they go, and may write that as a threshold of
    infinity. skl2onnx's conversion of such a tree never returns: it searches for a float32 number next to the
    threshold. Such a model is refused before the conversion is tried.

    Every model inside is looked at, wherever scikit-learn keeps it: the steps of a pipeline, the members of a
    bag, a vote or a stack and the model that reads them, the classifier each fold of a calibration fitted, the
    regressor behind a target transform, what a search or a threshold wrapper found.
    """
    from .ensembles import _parts_of

    stack: List[Any] = [estimator]
    seen: Set[int] = set()
    while stack and len(seen) < _MOST_PARTS:
        part = stack.pop()
        if part is None or isinstance(part, (str, bytes, int, float, bool)) or id(part) in seen:
            continue
        seen.add(id(part))
        if isinstance(part, Mapping):
            stack.extend(part.values())
            continue
        if isinstance(part, (list, tuple)):
            stack.extend(part)  # a named member (name, model), or a column transformer's (name, model, columns)
            continue
        if hasattr(part, "dtype") and hasattr(part, "ravel"):
            if getattr(part.dtype, "kind", "") == "O":
                stack.extend(part.ravel().tolist())  # a grid of trees, as gradient boosting keeps its own
            continue
        threshold = getattr(getattr(part, "tree_", None), "threshold", None)
        if threshold is not None:
            try:
                if bool(numpy.isinf(numpy.asarray(threshold, dtype=float)).any()):
                    return True
            except (TypeError, ValueError):
                pass
            continue
        try:
            stack.extend(_parts_of(part))
        except Exception:  # noqa: BLE001 - an attribute that raises on a model that is not fitted that way
            pass
        for attribute in _MORE_PARTS:
            try:
                found = getattr(part, attribute, None)
            except Exception:  # noqa: BLE001 - as above
                found = None
            if found is not None:
                stack.append(found)
    return False


def _convert_sklearn(
    model: Any, final: Any, width: int, precision: Any, classifier: bool, numpy: Any, raw_scores: bool = False
) -> bytes:
    """``model`` converted by skl2onnx, as bytes. ``final`` is the estimator its answer comes from (a pipeline's
    last step), which the converter's options are for. ``raw_scores`` asks for the scores ``decision_function``
    gives where the converter would give a classifier's probabilities (not every converter can)."""
    from skl2onnx import to_onnx

    if _trees_with_infinite_thresholds(model, numpy) or _trees_with_infinite_thresholds(final, numpy):
        raise _NoConversion(
            "a tree in it splits at an infinite threshold, as one fitted on rows with missing values may, and "
            "skl2onnx cannot convert such a tree"
        )
    model, final = _without_its_threshold(model, final)
    if raw_scores:
        model, final = _without_its_probabilities(model, final, numpy)
    example = numpy.zeros((1, width), dtype=precision)
    failures: List[str] = []
    choices: List[Dict[str, Any]] = []
    asked = {"zipmap": False, "raw_scores": True} if raw_scores else {"zipmap": False}
    for opset in ({"": 17, "ai.onnx.ml": 3}, None):
        if classifier:
            choices.append({"options": {id(final): asked}, "target_opset": opset})
        if raw_scores:
            continue  # without the option the conversion gives probabilities, which is not what was asked for
        choices.append({"target_opset": opset})
    for choice in choices:
        keywords = {key: value for key, value in choice.items() if value is not None}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return bytes(to_onnx(model, example, **keywords).SerializeToString())
        except Exception as exc:  # noqa: BLE001 - skl2onnx raises its own errors, and whatever a converter raises
            failures.append(f"{type(exc).__name__}: {_brief(exc, 160)}")
    raise _NoConversion(failures[0] if failures else "no conversion")


def _one_margin(
    tools: Any, proto: Any, library: Callable[[Dict[str, Any]], Any], width: int, precision: Any, numpy: Any
) -> Tuple[Any, Optional[str]]:
    """A two-class classifier with no probabilities gives one margin; its conversion gives two scores.

    Which of the two is the margin, and with which sign, differs between converters (an SVC's are
    ``[margin, -margin]``). The column that equals the model's own ``decision_function`` is found on a few
    rows and made an output of the graph, so the prediction is read from the margin itself: positive is the
    second class, as it is for the model. Returns the model and that output's name. When no column is the
    margin the conversion is returned as it is, with ``None``, and the check will say it does not reproduce
    the model.
    """
    onnx = tools._onnx()
    _wanted, tensor = _prediction_output(proto, None)
    if tensor is None:
        return proto, None
    name = str(tools._input_infos(proto)[0].name)
    runner = _OnnxRunner(tools, proto, tensor)
    rows = numpy.random.default_rng(0).normal(size=(12, width))
    try:
        scores = numpy.stack([numpy.asarray(runner({name: row.reshape(1, -1).astype(precision)}), dtype=float).reshape(-1) for row in rows])
        margin = numpy.array([float(numpy.asarray(library({name: row.reshape(1, -1)})).reshape(-1)[0]) for row in rows])
    except Exception:  # noqa: BLE001 - nothing to line up; the check decides
        return proto, None
    if scores.ndim != 2 or scores.shape[1] != 2:
        return proto, None
    floor = 1e-4 * max(float(numpy.abs(margin).max()), 1e-12)
    for column, sign in ((1, 1.0), (0, 1.0), (1, -1.0), (0, -1.0)):
        if numpy.allclose(sign * scores[:, column], margin, rtol=1e-4, atol=floor):
            break
    else:
        return proto, None
    helper = onnx.helper
    edited = onnx.ModelProto()
    edited.CopyFrom(proto)
    graph = edited.graph
    taken = {str(item) for node in graph.node for item in node.output} | {str(item.name) for item in graph.initializer}
    index = _free_name("kaimei_margin_column", taken)
    picked = _free_name("kaimei_margin_picked", taken)
    margin_name = _free_name("margin", taken)
    graph.initializer.append(onnx.numpy_helper.from_array(numpy.array([column], dtype=numpy.int64), index))
    graph.node.append(helper.make_node("Gather", [tensor, index], [picked if sign < 0 else margin_name], axis=1, name="margin/Gather"))
    if sign < 0:
        graph.node.append(helper.make_node("Neg", [picked], [margin_name], name="margin/Neg"))
    kind = onnx.TensorProto.DOUBLE if numpy.dtype(precision) == numpy.float64 else onnx.TensorProto.FLOAT
    graph.output.append(helper.make_tensor_value_info(margin_name, kind, [None, 1]))
    return edited, margin_name


# --- Keras and TensorFlow --------------------------------------------------------------------


def _keras_through_onnx(
    estimator: Any,
    why: List[str],
    numpy: Any,
    *,
    feature_names: Optional[Sequence[str]],
    task: Optional[str],
    sample: Any,
    inputs: Optional[Mapping[str, Any]],
    check_rows: Any,
    labels: Optional[Sequence[Any]],
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    output: Any,
    tokenizer: Any,
    framework: Optional[str],
    name: Optional[str],
    keep: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not _has("tf2onnx"):
        why.append("converting a Keras or TensorFlow model needs tf2onnx (pip install tf2onnx)")
        return None
    proto = _keras_to_onnx(estimator, sample, why, numpy)
    if proto is None:
        return None
    tools = _importer()
    order = [str(info.name) for info in tools._input_infos(proto)]
    if keep is not None:
        keep["onnx"] = bytes(proto.SerializeToString())
    # The model's answer compared with the conversion is the one the conversion is read at: of a model with several
    # outputs, the one output= picks (or the importer's choice), at its place among the file's outputs.
    wanted, _tensor = _prediction_output(proto, output, task)
    written = [str(info.name) for info in proto.graph.output]
    position = written.index(wanted) if wanted in written else 0

    def library(feeds: Dict[str, Any]) -> Any:
        given = [numpy.asarray(feeds[key]) for key in order]
        result = estimator(given[0] if len(given) == 1 else given, training=False)
        answers = _keras_outputs(result)
        chosen = answers[min(position, len(answers) - 1)]
        return chosen.numpy() if hasattr(chosen, "numpy") else numpy.asarray(chosen)

    samples = None
    if sample is not None:
        values = sample if isinstance(sample, Mapping) else dict(zip(order, sample if isinstance(sample, (list, tuple)) else [sample]))
        samples = {str(key): numpy.asarray(value)[:1] for key, value in values.items()}
    spec = _read_onnx(
        proto,
        library=library,
        feature_names=feature_names,
        task=task,
        labels=labels,
        inputs=inputs,
        input_shapes=None,
        samples=samples,
        output=output,
        mask_layout=mask_layout,
        threshold=threshold,
        name=name,
        tokenizer=tokenizer,
        check_rows=check_rows,
        framework=framework or "keras",
    )
    steps = len(spec["graph"].get("nodes") or [])
    spec["summary"] = (
        f"Keras model converted to ONNX and read as a graph of {steps:,} operations. " + str(spec.get("summary") or "")
    ).strip()
    return spec


def _keras_outputs(result: Any) -> List[Any]:
    """A Keras model's outputs in the order its conversion lists them: TensorFlow's own flattening (a dict by its
    sorted keys), which is how ``tf2onnx`` numbers what the function it converts returns."""
    try:
        import tensorflow as tf

        return list(tf.nest.flatten(result))
    except ImportError:
        if isinstance(result, Mapping):
            return [result[key] for key in sorted(result)]
        return list(result) if isinstance(result, (list, tuple)) else [result]


def _keras_signature(estimator: Any, sample: Any, numpy: Any) -> Optional[List[Any]]:
    """``tf.TensorSpec`` for each input of a Keras model: its own shapes, or the sample's where it leaves one open."""
    import tensorflow as tf

    declared = list(getattr(estimator, "inputs", None) or [])
    if not declared:
        return None
    given: List[Any] = []
    if sample is not None:
        if isinstance(sample, Mapping):
            given = [numpy.asarray(value) for value in sample.values()]
        elif isinstance(sample, (list, tuple)) and sample and hasattr(sample[0], "shape"):
            given = [numpy.asarray(value) for value in sample]
        else:
            given = [numpy.asarray(sample)]
    specs = []
    for index, tensor in enumerate(declared):
        dims = list(getattr(tensor, "shape", None) or [])
        if index < len(given) and given[index].ndim == len(dims):
            dims = [None] + [int(dim) if dim is not None else int(given[index].shape[at]) for at, dim in enumerate(dims) if at]
        name = str(getattr(tensor, "name", "") or "").split(":")[0]
        if not name or name.startswith(("keras_tensor", "input_layer")):
            name = "input" if len(declared) == 1 else f"input_{index}"
        specs.append(tf.TensorSpec([None, *dims[1:]], tf.as_dtype(str(getattr(tensor, "dtype", "float32"))), name=name))
    return specs


def _keras_to_onnx(estimator: Any, sample: Any, why: List[str], numpy: Any) -> Any:
    """A Keras model as an ONNX ``ModelProto``, or ``None`` with the reason added to ``why``.

    Keras 3 models go through ``tf.function`` with the model's input signature, which writes recurrent layers
    that ``tf2onnx.convert.from_keras`` cannot (it stops on Keras 3's tensor names); ``from_keras`` is the
    fallback, for the models of older Keras.
    """
    import tf2onnx

    failures: List[str] = []
    try:
        import tensorflow as tf

        signature = _keras_signature(estimator, sample, numpy)
        if signature is not None:
            def call(*tensors: Any) -> Any:
                return estimator(list(tensors) if len(tensors) > 1 else tensors[0], training=False)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proto, _storage = tf2onnx.convert.from_function(
                    tf.function(call), input_signature=signature, opset=17
                )
            return proto
    except Exception as exc:  # noqa: BLE001 - tf2onnx and TensorFlow raise their own errors
        failures.append(f"{type(exc).__name__}: {_brief(exc, 120)}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proto, _storage = tf2onnx.convert.from_keras(estimator, opset=17)
        return proto
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{type(exc).__name__}: {_brief(exc, 120)}")
    why.append(f"tf2onnx could not convert it ({'; '.join(failures)})")
    return None


# --- the sampling loop of a diffusion model -----------------------------------------------------------------

#: Names a diffusion model's step input goes by.
_STEP_NAMES = re.compile(r"(?:^|_)(?:timesteps?|t|time|step|steps|noise_level|sigma|sigmas)(?:$|_)")


def sampler_spec(sampler: Any, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``sampler`` of a diffusion model's structure, from a Hugging Face ``diffusers`` scheduler or a dict.

    A scheduler is read for its whole noise schedule (``alphas_cumprod``), what the model predicts
    (``prediction_type``) and its clipping. ``sample`` is the model's tensor or picture input and ``timestep``
    its step input (a column called ``timestep``, ``t``, ``time``…, or its only number), unless the dict says.
    """
    if isinstance(sampler, Mapping):
        made: Dict[str, Any] = {str(key): value for key, value in sampler.items()}
    else:
        config = getattr(sampler, "config", None)
        if config is None:
            raise StructureError(
                "sampler is a diffusers scheduler (it has a config) or a dict of the sampler's settings.",
                code="invalid_structure",
                param="sampler",
            )
        predicts = str(getattr(config, "prediction_type", "epsilon") or "epsilon")
        made = {
            "type": "diffusion",
            "prediction": {"epsilon": "epsilon", "v_prediction": "v", "sample": "sample"}.get(predicts, predicts),
        }
        cumulative = getattr(sampler, "alphas_cumprod", None)
        if cumulative is not None:
            values = cumulative.detach().cpu().numpy() if hasattr(cumulative, "detach") else cumulative
            made["alphas_cumprod"] = [float(value) for value in list(values)]
        else:
            made["train_steps"] = int(getattr(config, "num_train_timesteps", 1000) or 1000)
            made["beta_start"] = float(getattr(config, "beta_start", 1e-4) or 1e-4)
            made["beta_end"] = float(getattr(config, "beta_end", 0.02) or 0.02)
            schedule = str(getattr(config, "beta_schedule", "linear") or "linear")
            made["schedule"] = {"squaredcos_cap_v2": "cosine"}.get(schedule, schedule)
        if bool(getattr(config, "clip_sample", False)):
            bound = float(getattr(config, "clip_sample_range", 1.0) or 1.0)
            made["clip"] = [-bound, bound]
    made.setdefault("type", "diffusion")
    features = [dict(item) for item in spec.get("features") or [] if isinstance(item, Mapping)]
    pictures = [item for item in features if str(item.get("type")) in ("image", "tensor", "sequence")]
    numbers = [item for item in features if str(item.get("type") or "numeric") in ("numeric", "integer")]
    if "sample" not in made and pictures:
        largest = max(pictures, key=lambda item: _product(item.get("shape") or [item.get("height") or 1, item.get("width") or 1, item.get("channels") or 1]))
        made["sample"] = str(largest["name"])
        shape = [int(dim) for dim in largest.get("shape") or []]
        if str(largest.get("type")) == "tensor" and len(shape) == 3 and shape[0] in (1, 3, 4) and "layout" not in made:
            made["layout"] = "chw"
    if "timestep" not in made:
        named = [item for item in numbers if _STEP_NAMES.search(str(item["name"]).lower())]
        chosen = named[0] if named else (numbers[0] if len(numbers) == 1 else None)
        if chosen is not None:
            made["timestep"] = str(chosen["name"])
    missing = [key for key in ("sample", "timestep") if key not in made]
    if missing:
        raise StructureError(
            f"The sampler needs to know which feature is the model's {' and '.join(missing)}: pass "
            "sampler={..., 'sample': <feature>, 'timestep': <feature>}.",
            code="invalid_structure",
            param="sampler",
        )
    return made


def unwrap_pipeline(estimator: Any) -> Tuple[Any, Any]:
    """``(model, scheduler)`` for a diffusers pipeline (its denoiser and its scheduler); ``(estimator, None)`` otherwise."""
    unet = getattr(estimator, "unet", None) or getattr(estimator, "transformer", None)
    scheduler = getattr(estimator, "scheduler", None)
    if unet is not None and scheduler is not None and str(type(estimator).__module__).startswith("diffusers"):
        return unet, scheduler
    return estimator, None


# --- why a model is drawn and not run ------------------------------------------------------------------------


def drawn_reason(spec: Mapping[str, Any]) -> Optional[str]:
    """Why Kaimei would draw this structure and not run it, in words that say what to do, when that is known here.

    ``None`` when nothing here says why (the API's own check is then the reason).
    """
    task = str(spec.get("task") or "")
    graph = spec.get("graph") if isinstance(spec.get("graph"), Mapping) else None
    head = dict(spec.get("head") or {})
    if graph is not None and task == "regression":
        values = dict(graph.get("values") or {})
        shape = [int(dim) for dim in (values.get(str(head.get("output")), {}) or {}).get("shape") or []]
        width = _product(shape[1:]) if len(shape) >= 2 else _product(shape)
        if width > 1:
            return (
                f"It gives {width:,} numbers for a row and was read as a regression, which answers with one. "
                "If it is a classifier, pass task='multiclass_classification' (or 'binary_classification', or "
                "'multilabel_classification') and labels=[...]; if its answer is the whole tensor, pass "
                "task='generation' or task='embedding'; or pass output= to read one of its other outputs."
            )
    said = str(spec.get("summary") or "")
    if spec.get("approximate") and any(marker in said for marker in ("does not reproduce", NOT_COMPARED, NOT_ITS_CLASS)):
        # The conversion was compared with the model and differs (in its scores, or in the class taken from them), or
        # could not be compared, and the summary says which.
        return said
    return None


def _drawn_because(spec: Mapping[str, Any]) -> Optional[str]:
    """Why a conversion is drawn and not run, as a clause for a list of reasons: that it differs from the model, that
    it could not be compared with it, or that its outputs could not be computed, each with why when that is known.
    ``None`` when its summary does not say: a list of reasons then says what it can."""
    said = str(spec.get("summary") or "")
    for marker in (NOT_COMPARED, "Its ONNX conversion does not reproduce", "Its outputs could not be computed here", NOT_ITS_CLASS):
        at = said.find(marker)
        if at >= 0:
            clause = said[at:].split(", so it is drawn, not run.")[0]
            return clause[:1].lower() + clause[1:]
    return None


__all__ = [
    "carry_threshold",
    "classes_differ",
    "decision_threshold",
    "drawn_reason",
    "finalize_graph",
    "from_onnx",
    "holds_arrays",
    "hugging_face_hints",
    "is_onnx",
    "not_its_class",
    "not_its_threshold",
    "reported_class",
    "sampler_spec",
    "to_onnx",
    "unwrap_pipeline",
    "via_onnx",
]
