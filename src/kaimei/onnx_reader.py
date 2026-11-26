"""Read an ONNX model into plain Python: its operators, their attributes, its constants and its inputs.

This is the first half of turning a model file into a Kaimei structure. It needs the ``onnx`` package and
runs where the model is, on the uploader's machine: Kaimei is sent the structure this produces, never the
file. Nothing here decides what a model means (which input is a picture, which output a probability);
:mod:`onnx_graph` does that on top of what this returns.

The Kaimei API keeps an identical copy of this module (``kaimei_api/engine/structure/nd/onnx_reader.py``)
so its tests can check each of its kernels against the ONNX reference implementation. Edit the client's
copy, then copy it across.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

#: A tensor attribute up to this many values stays in the node; larger ones are not attributes anyone reads.
MAX_ATTRIBUTE_VALUES = 4096

_DTYPE_NAMES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
}


class OnnxReadError(ValueError):
    """A model file this reader cannot turn into operators and constants."""


def _onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise OnnxReadError(
            "Reading an ONNX model needs the onnx package: pip install onnx (or pip install 'kaimei[onnx]')."
        ) from exc
    return onnx


def load_model(source: Any) -> Any:
    """An ``onnx.ModelProto`` from a path, bytes, or a model that is already loaded."""
    onnx = _onnx()
    if isinstance(source, onnx.ModelProto):
        return source
    if isinstance(source, (bytes, bytearray, memoryview)):
        return onnx.load_model_from_string(bytes(source))
    if isinstance(source, (str, os.PathLike)):
        return onnx.load(os.fspath(source))
    if hasattr(source, "SerializeToString"):
        return onnx.load_model_from_string(source.SerializeToString())
    raise OnnxReadError(f"Cannot read an ONNX model from {type(source).__name__}.")


def expand_functions(model: Any) -> Any:
    """``model`` with the operators the file defines itself (model-local functions) written out where they are used.

    ``import onnx`` does not load ``onnx.inliner``, so it is imported here by its own name.
    """
    if not len(model.functions):
        return model
    try:
        from onnx import inliner
    except ImportError as exc:
        raise OnnxReadError(
            "This model defines its own operators (model-local functions), and the installed onnx package "
            "cannot expand them: it has no onnx.inliner. Install a newer one: pip install -U onnx."
        ) from exc
    try:
        return inliner.inline_local_functions(model)
    except Exception as exc:  # noqa: BLE001 - whatever the inliner cannot do, this reader cannot either
        raise OnnxReadError(f"This model defines its own operators, which could not be expanded: {exc}") from exc


def dtype_name(elem_type: int) -> str:
    name = _DTYPE_NAMES.get(int(elem_type))
    if name is None:
        raise OnnxReadError(f"ONNX tensor type {int(elem_type)} is not supported.")
    return name


def tensor_array(tensor: Any, base_dir: str = "") -> np.ndarray:
    """A ``TensorProto`` as a numpy array. Text tensors become arrays of ``str``."""
    onnx = _onnx()
    array = onnx.numpy_helper.to_array(tensor, base_dir) if base_dir else onnx.numpy_helper.to_array(tensor)
    if array.dtype.kind in "OS":
        flat = [item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item) for item in array.ravel()]
        return np.array(flat, dtype=object).reshape(array.shape)
    return np.asarray(array)


def small_tensor(array: np.ndarray) -> Dict[str, Any]:
    """A tensor small enough to live in a node's attributes, as plain JSON."""
    array = np.asarray(array)
    name = "string" if array.dtype.kind in "OUS" else array.dtype.name
    data = array.ravel().tolist()
    if name == "string":
        data = [str(item) for item in data]
    return {"dtype": name, "shape": [int(dim) for dim in array.shape], "data": data}


def _text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def attribute_value(attr: Any, base_dir: str = "") -> Any:
    """One node attribute as plain Python (numbers, text, lists, a small tensor, or a nested graph)."""
    onnx = _onnx()
    kinds = onnx.AttributeProto
    if attr.type == kinds.INT:
        return int(attr.i)
    if attr.type == kinds.FLOAT:
        return float(attr.f)
    if attr.type == kinds.STRING:
        return _text(attr.s)
    if attr.type == kinds.INTS:
        return [int(item) for item in attr.ints]
    if attr.type == kinds.FLOATS:
        return [float(item) for item in attr.floats]
    if attr.type == kinds.STRINGS:
        return [_text(item) for item in attr.strings]
    if attr.type == kinds.TENSOR:
        array = tensor_array(attr.t, base_dir)
        if array.size > MAX_ATTRIBUTE_VALUES:
            raise OnnxReadError(
                f"Attribute '{attr.name}' holds a tensor of {array.size:,} values, more than an attribute may."
            )
        return small_tensor(array)
    if attr.type == kinds.GRAPH:
        return {"graph": read_graph(attr.g, base_dir=base_dir, nested=True)}
    if attr.type == kinds.GRAPHS:
        return {"graphs": [read_graph(graph, base_dir=base_dir, nested=True) for graph in attr.graphs]}
    if attr.type == kinds.TYPE_PROTO:
        # What kind of value an operator makes when it is given none (an empty optional). Nothing computes with it.
        dtype, shape = _shape_of(attr.tp)
        return {"value_type": {"dtype": dtype, "shape": shape}}
    raise OnnxReadError(f"Attribute '{attr.name}' has a type this reader does not support.")


def node_attributes(node: Any, base_dir: str = "") -> Dict[str, Any]:
    return {str(attr.name): attribute_value(attr, base_dir) for attr in node.attribute}


def _shape_of(type_proto: Any) -> Tuple[Optional[str], Optional[List[Any]]]:
    """``(dtype, shape)`` of a value: sizes are ints, a named unknown size is its name, an unnamed one ``None``."""
    if not type_proto.HasField("tensor_type"):
        return None, None
    tensor = type_proto.tensor_type
    dtype = _DTYPE_NAMES.get(int(tensor.elem_type))
    if not tensor.HasField("shape"):
        return dtype, None
    dims: List[Any] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dtype, dims


#: Where PyTorch's newer exporter records the module a node was written from.
_SCOPE_KEY = "pkg.torch.onnx.name_scopes"


def _scope_of(node: Any) -> Optional[str]:
    """The module a node was written from, when its exporter recorded that beside the node and not in its name.

    PyTorch's older exporter names a node ``/layer1/layer1.0/conv1/Conv``. Its newer one names it
    ``node_conv2d`` and keeps the module in the node's metadata: every scope from the root down, then the
    operation's own name (``['', 'layer1', 'layer1.0', 'layer1.0.conv1', 'conv2d']``). ``None`` when a
    node carries no such record; the empty string for a node at the model's top level.
    """
    for entry in getattr(node, "metadata_props", ()):
        if entry.key != _SCOPE_KEY:
            continue
        text = str(entry.value)
        if len(text) > 8000:
            return None
        try:
            scopes = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
        if not isinstance(scopes, (list, tuple)) or not all(isinstance(item, str) for item in scopes):
            return None
        return str(scopes[-2]) if len(scopes) >= 2 else ""
    return None


def _constant_of(node: Any, base_dir: str) -> Optional[np.ndarray]:
    """The value of a ``Constant`` node, or ``None`` when it is written in a form this reader leaves alone."""
    onnx = _onnx()
    kinds = onnx.AttributeProto
    for attr in node.attribute:
        if attr.name == "value" and attr.type == kinds.TENSOR:
            return tensor_array(attr.t, base_dir)
        if attr.name == "value_float":
            return np.asarray(attr.f, dtype=np.float32)
        if attr.name == "value_int":
            return np.asarray(attr.i, dtype=np.int64)
        if attr.name == "value_floats":
            return np.asarray(list(attr.floats), dtype=np.float32)
        if attr.name == "value_ints":
            return np.asarray(list(attr.ints), dtype=np.int64)
        if attr.name == "value_string":
            return np.array(_text(attr.s), dtype=object)
        if attr.name == "value_strings":
            return np.array([_text(item) for item in attr.strings], dtype=object)
    return None


def read_graph(graph: Any, *, base_dir: str = "", nested: bool = False) -> Dict[str, Any]:
    """One ONNX graph as ``{"inputs", "nodes", "constants", "outputs", "values"}``.

    ``constants`` maps a name to a numpy array: every initializer and every ``Constant`` node. ``inputs``
    leaves out the initializers old exporters also list as inputs. ``values`` holds the type and shape the
    file records for any named tensor. A nested graph (the body of a loop or a branch) keeps the same form.
    """
    constants: Dict[str, np.ndarray] = {}
    for tensor in graph.initializer:
        constants[str(tensor.name)] = tensor_array(tensor, base_dir)
    values: Dict[str, Dict[str, Any]] = {}
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        dtype, shape = _shape_of(info.type)
        if dtype is not None or shape is not None:
            values[str(info.name)] = {"dtype": dtype, "shape": shape}
    inputs: List[Dict[str, Any]] = []
    for info in graph.input:
        name = str(info.name)
        if name in constants:
            continue
        dtype, shape = _shape_of(info.type)
        inputs.append({"name": name, "dtype": dtype, "shape": shape})
    nodes: List[Dict[str, Any]] = []
    taken = set()
    for index, node in enumerate(graph.node):
        domain = "" if node.domain in ("", "ai.onnx") else str(node.domain)
        if node.op_type == "Constant" and domain == "" and len(node.output) == 1:
            value = _constant_of(node, base_dir)
            if value is not None:
                constants[str(node.output[0])] = value
                continue
        node_id = str(node.name) or f"{node.op_type}_{index}"
        if node_id in taken:
            node_id = f"{node_id}#{index}"
        taken.add(node_id)
        entry: Dict[str, Any] = {
            "id": node_id,
            "op": str(node.op_type),
            "domain": domain,
            "inputs": [str(name) for name in node.input],
            "outputs": [str(name) for name in node.output],
            "attrs": node_attributes(node, base_dir),
        }
        scope = _scope_of(node)
        if scope is not None:
            entry["scope"] = scope
        nodes.append(entry)
    out: Dict[str, Any] = {
        "inputs": inputs,
        "nodes": nodes,
        "constants": constants,
        "outputs": [str(info.name) for info in graph.output],
        "values": values,
    }
    if nested:
        out["name"] = str(graph.name)
    return out


def read_model(source: Any, *, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """A whole model: :func:`read_graph` of its main graph, plus the operator sets it was written against.

    ``opsets`` maps a domain to a version (the default ONNX domain is the empty string). Models that keep
    their weights in separate files need ``base_dir``, the folder those files are in; for a path it is
    found automatically.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.fspath(source)) if isinstance(source, (str, os.PathLike)) else ""
    # Model-local functions are operators defined by the file itself; they are expanded before reading.
    model = expand_functions(load_model(source))
    opsets: Dict[str, int] = {}
    for entry in model.opset_import:
        domain = "" if entry.domain in ("", "ai.onnx") else str(entry.domain)
        opsets[domain] = max(int(entry.version), opsets.get(domain, 0))
    out = read_graph(model.graph, base_dir=base_dir)
    out["opsets"] = opsets
    out["producer"] = {
        "name": str(model.producer_name or ""),
        "version": str(model.producer_version or ""),
        "ir_version": int(model.ir_version),
        "doc": str(model.doc_string or "")[:400],
    }
    return out


def single_node_model(
    op: str,
    inputs: Sequence[Tuple[str, Optional[np.ndarray]]],
    outputs: Union[int, Sequence[str]] = 1,
    *,
    domain: str = "",
    opset: int = 21,
    initializers: Sequence[str] = (),
    **attributes: Any,
) -> Any:
    """A model holding one operator, for comparing an implementation of it with a reference.

    ``inputs`` are ``(name, array)`` pairs; a ``None`` array is an input the node leaves out. Names listed
    in ``initializers`` are stored in the model as constants instead of being fed.
    """
    onnx = _onnx()
    helper = onnx.helper
    names = [name if array is not None else "" for name, array in inputs]
    while names and names[-1] == "":
        # An input left out at the end is not listed at all. Listed under an empty name it is the same node, and
        # ONNX Runtime 1.32 stops the process on some of them (an opset-20 DFT with its axis left out).
        names.pop()
    out_names = [f"out{i}" for i in range(outputs)] if isinstance(outputs, int) else list(outputs)
    node = helper.make_node(op, names, out_names, domain=domain, **attributes)
    graph_inputs = []
    graph_initializers = []
    for name, array in inputs:
        if array is None:
            continue
        array = np.asarray(array)
        if name in initializers:
            graph_initializers.append(onnx.numpy_helper.from_array(array, name))
            continue
        elem_type = (
            onnx.TensorProto.STRING if array.dtype.kind in "OUS" else helper.np_dtype_to_tensor_dtype(array.dtype)
        )
        graph_inputs.append(helper.make_tensor_value_info(name, elem_type, list(array.shape)))
    graph_outputs = [helper.make_empty_tensor_value_info(name) for name in out_names if name]
    graph = helper.make_graph([node], "single", graph_inputs, graph_outputs, graph_initializers)
    opset_imports = [helper.make_opsetid("", opset if domain == "" else 21)]
    if domain:
        opset_imports.append(helper.make_opsetid(domain, opset))
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = min(model.ir_version, 10)
    return model
