"""Turn an ONNX model into a Kaimei structure of ``family: "graph"``.

Every mainstream framework can write a model down as ONNX, so this is the way in for any model Kaimei has
no reader of its own for: transformers, recurrent networks, models with several inputs, the scikit-learn
estimators that only exist as ONNX conversions. It runs where the model is. Kaimei is sent what this
returns (operators, attributes, shapes and tensors), never the model file.

What it does to the graph, and why:

* **Folds constants.** Exporters leave arithmetic on constants in the graph. It is done once here.
* **Hides shape plumbing.** ``Shape → Gather → Concat → Reshape`` chains say how big a tensor is, not what
  the model computes. They stay in the graph, so it still runs on any number of rows, but are not drawn.
* **Fuses what exporters take apart.** A layer norm written as nine primitives, GELU written with ``Erf``,
  ``MatMul`` then ``Add`` for a linear layer: each becomes one step again.
* **Names steps by the model's own modules**, so the picture groups the way the code does.
* **Binds inputs to features**: columns of a table, a picture, a token sequence, a series, or any tensor.
* **Runs one example through a reference** to record every tensor's shape, and the model's own outputs
  on a few rows so the API can check that it reproduces them.

The Kaimei API keeps an identical copy of this module (``kaimei_api/engine/structure/nd/onnx_graph.py``).
Edit the client's copy, then copy it across.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # the client and the API keep this module under different packages
    from .onnx_reader import OnnxReadError, expand_functions, load_model, read_model
except ImportError:  # pragma: no cover
    from onnx_reader import (  # type: ignore[no-redef]
        OnnxReadError,
        expand_functions,
        load_model,
        read_model,
    )

#: A folded constant larger than this, and larger than what it was computed from, stays a step instead.
FOLD_MAX_VALUES = 1_000_000
#: Operators that draw numbers at random. A model that holds one is shown with fixed seeds (see ``_pin_seeds``).
_RANDOM = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Multinomial", "Bernoulli"}
#: Operators that are never folded: their result is not a fixed tensor, or ONNX's own evaluator, which does
#: the folding, reads them differently from ONNX Runtime (the three that work on text), and the kernel that
#: follows ONNX Runtime should be the one to run them.
_NEVER_FOLD = {
    *_RANDOM,
    "If",
    "Loop",
    "Scan",
    "StringNormalizer",
    "StringSplit",
    "TfIdfVectorizer",
}
#: Rows of the model's own outputs sent along so the API can check its reading.
CHECK_ROWS = 4
_KAIMEI = "ai.kaimei"


class OnnxImportError(ValueError):
    """An ONNX model this importer cannot turn into a structure, with what to pass to help it.

    ``param`` names the caller's argument at fault, when one is: one that is wrong (``labels``,
    ``feature_names``), or the one that has to be passed to settle what the model does not say
    (``mask_layout`` for a segmentation whose class axis cannot be told).
    """

    def __init__(self, message: str, param: Optional[str] = None) -> None:
        super().__init__(message)
        self.param = param


def _onnx() -> Any:
    import onnx

    return onnx


# --- pinning input shapes ----------------------------------------------------------------


def _input_infos(model: Any) -> List[Any]:
    initializers = {tensor.name for tensor in model.graph.initializer}
    return [info for info in model.graph.input if info.name not in initializers]


def _static_shape(
    info: Any, shapes: Dict[str, Sequence[int]], samples: Dict[str, np.ndarray]
) -> Tuple[List[int], Optional[int]]:
    """``(shape for one row, batch axis)`` of a graph input, from the file, ``shapes`` or a sample."""
    name = str(info.name)
    tensor = info.type.tensor_type
    if name in samples:
        return [int(dim) for dim in np.asarray(samples[name]).shape], _batch_axis_of(info)
    if name in shapes:
        return [int(dim) for dim in shapes[name]], _batch_axis_of(info)
    if not tensor.HasField("shape"):
        raise OnnxImportError(f"Input '{name}' has no shape in the file. Pass input_shapes={{'{name}': [...]}}.")
    dims: List[int] = []
    for index, dim in enumerate(tensor.shape.dim):
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        elif index == 0:
            dims.append(1)  # the batch axis: one row
        else:
            raise OnnxImportError(
                f"Input '{name}' has a size the file leaves open (axis {index}). "
                f"Pass input_shapes={{'{name}': [...]}} with the size to use."
            )
    return dims, _batch_axis_of(info)


def _batch_axis_of(info: Any) -> Optional[int]:
    """Axis 0 when the file leaves it open (any number of rows) or sets it to one."""
    tensor = info.type.tensor_type
    if not tensor.HasField("shape") or not len(tensor.shape.dim):
        return None
    first = tensor.shape.dim[0]
    if not first.HasField("dim_value") or first.dim_value == 1:
        return 0
    return None


# --- constant folding --------------------------------------------------------------------


def _evaluate_node(node: Any, constants: Dict[str, np.ndarray], opsets: Sequence[Any]) -> Optional[List[np.ndarray]]:
    """What ``node`` gives on constant inputs, by the ONNX reference; ``None`` when it cannot be evaluated."""
    onnx = _onnx()
    from onnx.reference import ReferenceEvaluator

    helper = onnx.helper
    initializers = []
    for name in node.input:
        if name:
            initializers.append(onnx.numpy_helper.from_array(np.asarray(constants[name]), name))
    outputs = [helper.make_empty_tensor_value_info(name) for name in node.output if name]
    graph = helper.make_graph([node], "fold", [], outputs, initializers)
    model = helper.make_model(graph, opset_imports=list(opsets))
    model.ir_version = min(model.ir_version, 10)
    try:
        results = ReferenceEvaluator(model).run(None, {})
    except Exception:  # noqa: BLE001 - a step the reference cannot run is simply left in the graph
        return None
    if any(not isinstance(result, np.ndarray) for result in results):
        return None
    return list(results)


def fold_constants(model: Any, base_dir: str = "") -> Any:
    """A copy of ``model`` with every step that only reads constants replaced by its result."""
    onnx = _onnx()
    folded = onnx.ModelProto()
    folded.CopyFrom(model)
    graph = folded.graph
    constants: Dict[str, np.ndarray] = {}
    for tensor in graph.initializer:
        constants[tensor.name] = (
            onnx.numpy_helper.to_array(tensor, base_dir) if base_dir else onnx.numpy_helper.to_array(tensor)
        )
    kept = []
    added: Dict[str, np.ndarray] = {}
    for node in graph.node:
        inputs = [name for name in node.input if name]
        domain = node.domain if node.domain not in ("", "ai.onnx") else ""
        foldable = (
            domain == ""
            and node.op_type not in _NEVER_FOLD
            and node.op_type != "Constant"
            and all(name in constants for name in inputs)
            and not any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attr in node.attribute)
        )
        if not foldable:
            kept.append(node)
            continue
        results = _evaluate_node(node, constants, folded.opset_import)
        names = [name for name in node.output if name]
        if results is None or len(results) != len(names):
            kept.append(node)
            continue
        read = sum(int(np.asarray(constants[name]).size) for name in inputs)
        made = sum(int(result.size) for result in results)
        if made > FOLD_MAX_VALUES and made > 2 * read:
            kept.append(node)  # a small recipe for a big tensor is better left as a recipe
            continue
        for name, result in zip(names, results):
            constants[name] = result
            added[name] = result
    del graph.node[:]
    graph.node.extend(kept)
    used = {name for node in kept for name in node.input} | {info.name for info in graph.output}
    used |= _names_in_subgraphs(kept)
    for name, value in added.items():
        if name in used:
            graph.initializer.append(onnx.numpy_helper.from_array(np.asarray(value), name))
    return folded


def _pin_seeds(graph: Any, count: int = 0) -> int:
    """Give every random step of ``graph`` that has no seed one of its own, and return how many there are now.

    A step with no seed draws from the clock: no two runs agree, in any implementation. Kaimei shows a
    model the same way every time and checks its reading against the model's own answers, so each such
    step is given a seed here (a different one for each, so two of them do not draw the same numbers), and
    both the reading and the answers it is checked against are made with those seeds.
    """
    onnx = _onnx()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                count = _pin_seeds(attr.g, count)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for body in attr.graphs:
                    count = _pin_seeds(body, count)
        if node.op_type in _RANDOM and node.domain in ("", "ai.onnx"):
            if not any(attr.name == "seed" for attr in node.attribute):
                count += 1
                node.attribute.append(onnx.helper.make_attribute("seed", float(count)))
    return count


def _draws(graph: Any) -> bool:
    onnx = _onnx()
    for node in graph.node:
        if node.op_type in _RANDOM and node.domain in ("", "ai.onnx"):
            return True
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH and _draws(attr.g):
                return True
            if attr.type == onnx.AttributeProto.GRAPHS and any(_draws(body) for body in attr.graphs):
                return True
    return False


def _names_in_subgraphs(nodes: Sequence[Any]) -> Set[str]:
    onnx = _onnx()
    found: Set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        for attr in node.attribute:
            graphs = [attr.g] if attr.type == onnx.AttributeProto.GRAPH else list(attr.graphs)
            for graph in graphs:
                for inner in graph.node:
                    found.update(name for name in inner.input if name)
                    stack.append(inner)
                found.update(info.name for info in graph.output)
    return found


# --- running the original, for shapes and for the check -----------------------------------


def _feeds_for(model: Any, shapes: Dict[str, List[int]], seed: int, vocab: Dict[str, int]) -> Dict[str, np.ndarray]:
    onnx = _onnx()
    rng = np.random.default_rng(seed)
    feeds: Dict[str, np.ndarray] = {}
    for info in _input_infos(model):
        name = str(info.name)
        elem = info.type.tensor_type.elem_type
        dtype = onnx.helper.tensor_dtype_to_np_dtype(elem)
        shape = shapes[name]
        if np.issubdtype(dtype, np.integer):
            high = max(int(vocab.get(name, 2)), 2)
            feeds[name] = rng.integers(0, high, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.bool_):
            feeds[name] = np.ones(shape, dtype=bool)
        else:
            feeds[name] = rng.uniform(0.0, 1.0, size=shape).astype(dtype)
    return feeds


def _bodies(model: Any) -> Iterator[Any]:
    """Every list of nodes in ``model``: its graph, its own functions, and every branch and loop body in them."""
    onnx = _onnx()
    waiting: List[Any] = [model.graph, *model.functions]
    while waiting:
        body = waiting.pop()
        yield body
        for node in body.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    waiting.append(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    waiting.extend(attr.graphs)


def _without_trailing_empties(model: Any) -> Any:
    """``model``, or a copy of it with every input left out at the end of a node's list taken off that list.

    An optional input at the end may be listed under an empty name or not at all, and both mean the same, but
    neither engine runs every node written the first way: ONNX Runtime 1.32 stops the whole process on some (an
    opset-20 ``DFT``), and ONNX's evaluator hands such a node ``None`` for each and some fail. A node in a branch,
    a loop's body or a function of the file's own is written the second way as well as one at the top.
    """
    if not _has_trailing_empties(model):
        return model
    copy = _onnx().ModelProto()
    copy.CopyFrom(model)
    _strip_trailing_empties(copy)
    return copy


def _has_trailing_empties(model: Any) -> bool:
    return any(len(node.input) and node.input[-1] == "" for body in _bodies(model) for node in body.node)


def _strip_trailing_empties(model: Any) -> None:
    """Take every input left out at the end of a node's list off that list, in ``model`` itself."""
    for body in _bodies(model):
        for node in body.node:
            while len(node.input) and node.input[-1] == "":
                del node.input[-1]


def run_reference(model: Any, feeds: Dict[str, np.ndarray], wanted: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Every named tensor of ``model`` for ``feeds`` (or just ``wanted``), by ONNX Runtime or the ONNX reference.

    Both engines are given the model with no inputs left out under empty names (see
    :func:`_without_trailing_empties`). At most one copy of the model is made for that and for ONNX Runtime, which
    is asked for more outputs than the file declares: a model can be most of what a machine's memory holds.
    """
    onnx = _onnx()
    names = list(wanted) if wanted is not None else [name for node in model.graph.node for name in node.output if name]
    declared = {info.name for info in model.graph.output}
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    copies: List[Any] = []

    def own() -> Any:
        """A copy of the model with no inputs left out under empty names, made once."""
        if not copies:
            copy = onnx.ModelProto()
            copy.CopyFrom(model)
            _strip_trailing_empties(copy)
            copies.append(copy)
        return copies[0]

    def plain() -> Any:
        """The model as either engine runs it: itself, or its copy when a node lists an input under an empty name."""
        return own() if copies or _has_trailing_empties(model) else model

    if ort is not None and _draws(model.graph):
        # ONNX Runtime draws its random numbers from another generator than ONNX's evaluator and Kaimei's
        # kernels share, so a model that draws is recorded by the evaluator when it can run the model.
        from onnx.reference import ReferenceEvaluator

        try:
            drawn: Any = ReferenceEvaluator(plain()).run(None, feeds, intermediate=True)
            kept = set(names) | declared
            return {name: drawn[name] for name in kept if name in drawn}
        except Exception:  # noqa: BLE001 - the evaluator lacks some operators; ONNX Runtime is tried next
            pass
    if ort is not None:
        probe = own()
        for name in names:
            if name not in declared:
                probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
        options = ort.SessionOptions()
        options.log_severity_level = 4  # a model it cannot run is reported below, not printed by the library
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            session = ort.InferenceSession(probe.SerializeToString(), options, providers=["CPUExecutionProvider"])
            listed = [output.name for output in session.get_outputs()]
            return dict(zip(listed, session.run(None, feeds)))
        except Exception:  # noqa: BLE001 - fall back to the reference implementation below
            del probe.graph.output[len(model.graph.output) :]  # the evaluator runs the model as the file declares it
    from onnx.reference import ReferenceEvaluator

    try:
        results: Any = ReferenceEvaluator(plain()).run(None, feeds, intermediate=True)
    except Exception as exc:  # noqa: BLE001 - whatever it is, the file is not a model that can be run
        # An exporter can write a file that is not a valid model (PyTorch writes an LSTM with projections as a
        # plain LSTM whose weights do not fit). Nothing can be read off a model its own format cannot run.
        raise OnnxImportError(
            "This model's ONNX form cannot be run by ONNX Runtime or by ONNX's own evaluator, so there is nothing "
            f"to check a reading of it against: {type(exc).__name__}: {str(exc)[:300]}"
        ) from exc
    return {name: results[name] for name in set(names) | {info.name for info in model.graph.output} if name in results}


# --- names and paths ---------------------------------------------------------------------

_SAFE = re.compile(r"[^0-9A-Za-z_.\-]+")


def module_path(node_name: str) -> Tuple[str, str]:
    """``(module path, leaf)`` from an exporter's node name.

    PyTorch's exporter writes ``/layer1/layer1.0/conv1/Conv``: each scope repeats the path so far. That
    is the module ``layer1.0.conv1`` and the operator ``Conv`` in it. A name with no scopes is its own leaf.
    """
    text = str(node_name or "").strip()
    if "/" not in text:
        return "", text
    parts = [part for part in text.split("/") if part]
    if not parts:
        return "", text
    leaf = parts[-1]
    scopes: List[str] = []
    for part in parts[:-1]:
        if scopes and part.startswith(scopes[-1] + "."):
            scopes[-1] = part
        elif scopes and "." in part and part.split(".")[0] == scopes[-1].split(".")[-1]:
            scopes[-1] = scopes[-1] + part[len(part.split(".")[0]) :]
        else:
            scopes.append(part)
    return ".".join(scopes), leaf


def node_path(node: Dict[str, Any]) -> Tuple[str, str]:
    """``(module path, leaf)`` of a node as it was read: from the scope its exporter recorded, or from its name."""
    scope = node.get("scope")
    if scope is None:
        return module_path(node["id"])
    leaf = str(node["id"])
    return str(scope), leaf[5:] if leaf.startswith("node_") and len(leaf) > 5 else leaf


def _clean(text: str) -> str:
    return _SAFE.sub("_", text).strip("._") or "step"


class _Names:
    """Unique, readable names for tensors, never one a feature already has."""

    def __init__(self, reserved: Sequence[str]) -> None:
        self.taken: Set[str] = set(reserved)

    def make(self, wanted: str) -> str:
        base = _clean(wanted)[:180]
        name = base
        count = 1
        while name in self.taken:
            count += 1
            name = f"{base}_{count}"
        self.taken.add(name)
        return name


# --- fusing what exporters take apart ----------------------------------------------------


class _Graph:
    """A read graph with the lookups the rewrites below need."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.nodes: List[Dict[str, Any]] = raw["nodes"]
        self.constants: Dict[str, np.ndarray] = raw["constants"]
        self.outputs: List[str] = raw["outputs"]
        self.reindex()

    def reindex(self) -> None:
        self.producer: Dict[str, Dict[str, Any]] = {}
        self.readers: Dict[str, List[Dict[str, Any]]] = {}
        for node in self.nodes:
            for name in node["outputs"]:
                if name:
                    self.producer[name] = node
            for name in node["inputs"]:
                if name:
                    self.readers.setdefault(name, []).append(node)
            # A branch or a loop's body reads tensors of the graph around it by name: its node reads them too, so
            # no rewrite takes away a tensor only a body reads.
            for name in _free_names(node["attrs"]):
                self.readers.setdefault(name, []).append(node)

    def only_reader(self, name: str, op: str) -> Optional[Dict[str, Any]]:
        readers = self.readers.get(name, [])
        if len(readers) != 1 or name in self.outputs:
            return None
        node = readers[0]
        return node if node["op"] == op and node["domain"] == "" else None

    def scalar(self, name: str) -> Optional[float]:
        value = self.constants.get(name)
        if value is None or np.asarray(value).size != 1:
            return None
        return float(np.asarray(value).reshape(-1)[0])

    def remove(self, nodes: Sequence[Dict[str, Any]]) -> None:
        gone = {id(node) for node in nodes}
        self.nodes[:] = [node for node in self.nodes if id(node) not in gone]
        self.reindex()


def _other(node: Dict[str, Any], name: str) -> Optional[str]:
    """The other input of a two-input node."""
    inputs = [item for item in node["inputs"] if item]
    if len(inputs) != 2 or name not in inputs:
        return None
    return inputs[1] if inputs[0] == name else inputs[0]


def _erf_inside(graph: _Graph, erf: Dict[str, Any]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """``x / √2`` under an ``Erf``: ``(x, [the step that divides])``."""
    div = graph.producer.get(erf["inputs"][0])
    if div is None or div["op"] not in ("Div", "Mul") or div["domain"] != "":
        return None
    source = next((name for name in div["inputs"] if name and name not in graph.constants), None)
    factor = next((graph.scalar(name) for name in div["inputs"] if name in graph.constants), None)
    if source is None or factor is None or factor == 0:
        return None
    ratio = factor if div["op"] == "Div" else 1.0 / factor
    if abs(ratio - 2.0**0.5) > 1e-3:
        return None
    return source, [div]


def _tanh_inside(graph: _Graph, tanh: Dict[str, Any]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """``√(2/π) · (x + 0.044715 · x³)`` under a ``Tanh``: ``(x, [the steps that make it])``."""
    scale = graph.producer.get(tanh["inputs"][0])
    if scale is None or scale["op"] != "Mul" or scale["domain"] != "":
        return None
    factor = next((graph.scalar(name) for name in scale["inputs"] if name in graph.constants), None)
    inner = next((name for name in scale["inputs"] if name and name not in graph.constants), None)
    if factor is None or inner is None or abs(factor - (2.0 / np.pi) ** 0.5) > 1e-4:
        return None
    add = graph.producer.get(inner)
    if add is None or add["op"] != "Add" or add["domain"] != "":
        return None
    for source in [name for name in add["inputs"] if name]:
        cube = graph.producer.get(_other(add, source) or "")
        if cube is None or cube["op"] != "Mul" or cube["domain"] != "":
            continue
        coefficient = next((graph.scalar(name) for name in cube["inputs"] if name in graph.constants), None)
        powered = next((name for name in cube["inputs"] if name and name not in graph.constants), None)
        if coefficient is None or powered is None or abs(coefficient - 0.044715) > 1e-6:
            continue
        power = graph.producer.get(powered)
        if power is None or power["op"] != "Pow" or power["domain"] != "" or power["inputs"][0] != source:
            continue
        if graph.scalar(power["inputs"][1]) != 3.0:
            continue
        return source, [power, cube, add, scale]
    return None


def _fuse_gelu(graph: _Graph) -> int:
    """A GELU written out, in any of the orders exporters write it → one ``Gelu``.

    Two forms: ``x · ½ · (1 + erf(x / √2))``, and the approximation GPT-2 and its descendants use,
    ``x · ½ · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))``, which is ``Gelu`` with ``approximate="tanh"``.
    """
    fused = 0
    for core in [node for node in graph.nodes if node["op"] in ("Erf", "Tanh") and node["domain"] == ""]:
        if not any(node is core for node in graph.nodes):
            continue
        found = _erf_inside(graph, core) if core["op"] == "Erf" else _tanh_inside(graph, core)
        if found is None:
            continue
        source, inside = found
        # Every step folded away must be read by the next one only, or something else would lose its input.
        if any(len(graph.readers.get(node["outputs"][0], [])) != 1 for node in inside):
            continue
        add = graph.only_reader(core["outputs"][0], "Add")
        if add is None or graph.scalar(_other(add, core["outputs"][0]) or "") != 1.0:
            continue
        first = graph.only_reader(add["outputs"][0], "Mul")
        if first is None:
            continue
        partner = _other(first, add["outputs"][0])
        chain = [*inside, core, add, first]
        if partner == source:
            second = graph.only_reader(first["outputs"][0], "Mul")
            if second is None or graph.scalar(_other(second, first["outputs"][0]) or "") != 0.5:
                continue
            chain.append(second)
            result = second["outputs"][0]
        elif partner is not None and graph.scalar(partner) == 0.5:
            second = graph.only_reader(first["outputs"][0], "Mul")
            if second is None or _other(second, first["outputs"][0]) != source:
                continue
            chain.append(second)
            result = second["outputs"][0]
        else:
            half = graph.producer.get(partner or "")
            if half is None or half["op"] != "Mul" or half["domain"] != "":
                continue
            if _other(half, source) is None or graph.scalar(_other(half, source) or "") != 0.5:
                continue
            if len(graph.readers.get(half["outputs"][0], [])) != 1:
                continue
            chain.append(half)
            result = first["outputs"][0]
        keep = chain[-1] if chain[-1]["outputs"][0] == result else first
        attrs = {"approximate": "tanh"} if core["op"] == "Tanh" else {}
        keep.update({"op": "Gelu", "domain": "", "inputs": [source], "outputs": [result], "attrs": attrs})
        keep["opset"] = 20
        graph.remove([node for node in chain if node is not keep])
        fused += 1
    return fused


def _fuse_layer_norm(graph: _Graph) -> int:
    """``(x − mean) / √(var + ε) · γ + β`` written out over the last axis → one ``LayerNormalization``."""
    fused = 0
    for mean in [node for node in graph.nodes if node["op"] == "ReduceMean" and node["domain"] == ""]:
        if mean not in graph.nodes:
            continue
        source = mean["inputs"][0]
        axes = mean["attrs"].get("axes")
        if axes is None and len(mean["inputs"]) > 1 and mean["inputs"][1] in graph.constants:
            axes = [int(item) for item in np.asarray(graph.constants[mean["inputs"][1]]).reshape(-1)]
        if axes != [-1] or int(mean["attrs"].get("keepdims", 1)) != 1:
            continue
        subs = [
            node
            for node in graph.readers.get(mean["outputs"][0], [])
            if node["op"] == "Sub" and node["inputs"][0] == source and node["inputs"][1] == mean["outputs"][0]
        ]
        if len(subs) != 1 or len(graph.readers.get(mean["outputs"][0], [])) != 1:
            continue
        sub = subs[0]
        centred = sub["outputs"][0]
        readers = graph.readers.get(centred, [])
        square = next(
            (node for node in readers if node["op"] == "Pow" and graph.scalar(node["inputs"][1]) == 2.0), None
        )
        if square is None:
            square = next(
                (node for node in readers if node["op"] == "Mul" and node["inputs"][0] == centred == node["inputs"][1]),
                None,
            )
        div = next((node for node in readers if node["op"] == "Div" and node["inputs"][0] == centred), None)
        if square is None or div is None or len(readers) != (3 if square["op"] == "Mul" else 2):
            continue
        variance = graph.only_reader(square["outputs"][0], "ReduceMean")
        if variance is None:
            continue
        add = graph.only_reader(variance["outputs"][0], "Add")
        epsilon = graph.scalar(_other(add, variance["outputs"][0]) or "") if add is not None else None
        if add is None or epsilon is None:
            continue
        sqrt = graph.only_reader(add["outputs"][0], "Sqrt")
        if sqrt is None or div["inputs"][1] != sqrt["outputs"][0]:
            continue
        chain = [mean, sub, square, variance, add, sqrt, div]
        result = div["outputs"][0]
        inputs = [source]
        scale = graph.only_reader(result, "Mul")
        gamma = _other(scale, result) if scale is not None else None
        if scale is not None and gamma in graph.constants:
            chain.append(scale)
            result = scale["outputs"][0]
            inputs.append(gamma)
            shift = graph.only_reader(result, "Add")
            beta = _other(shift, result) if shift is not None else None
            if shift is not None and beta in graph.constants:
                chain.append(shift)
                result = shift["outputs"][0]
                inputs.append(beta)
        else:
            continue  # ONNX's LayerNormalization needs a scale; a bare normalisation stays as it is written
        keep = chain[-1]
        keep.update(
            {
                "op": "LayerNormalization",
                "domain": "",
                "inputs": inputs,
                "outputs": [result],
                "attrs": {"axis": -1, "epsilon": float(epsilon)},
            }
        )
        keep["opset"] = 17
        graph.remove([node for node in chain if node is not keep])
        fused += 1
    return fused


def _fuse_linear(graph: _Graph) -> int:
    """``x · W`` then ``+ b`` with a constant matrix and vector → one ``Linear`` step (``y = xW + b``)."""
    fused = 0
    for matmul in [node for node in graph.nodes if node["op"] == "MatMul" and node["domain"] == ""]:
        weight = matmul["inputs"][1]
        source = matmul["inputs"][0]
        if weight not in graph.constants or source in graph.constants:
            continue
        if np.asarray(graph.constants[weight]).ndim != 2:
            continue
        add = graph.only_reader(matmul["outputs"][0], "Add")
        bias = _other(add, matmul["outputs"][0]) if add is not None else None
        if add is not None and bias in graph.constants and np.asarray(graph.constants[bias]).ndim <= 1:
            add.update({"op": "Linear", "domain": _KAIMEI, "inputs": [source, weight, bias], "attrs": {}, "opset": 1})
            if not add.get("id_from_module"):
                add["id"] = matmul["id"]
                if "scope" in matmul:
                    add["scope"] = matmul["scope"]
            graph.remove([matmul])
        else:
            matmul.update({"op": "Linear", "domain": _KAIMEI, "inputs": [source, weight], "attrs": {}, "opset": 1})
        fused += 1
    return fused


def _inherit_scopes(graph: _Graph) -> None:
    """Give a node a module when its exporter recorded one for its neighbours and not for it.

    PyTorch's newer exporter rewrites some operations after it has noted their modules (a split, a
    multiply and an add made one), and what it writes in their place carries no module. Such a node sits
    between the steps it reads from and the steps that read it, so it is placed in the innermost module
    that holds them all: never deeper than what is known.
    """
    if not any("scope" in node for node in graph.nodes):
        return
    for node in graph.nodes:
        if "scope" in node:
            continue
        around: List[List[str]] = []
        for name in node["inputs"]:
            producer = graph.producer.get(name) if name else None
            if producer is not None and "scope" in producer:
                around.append([part for part in str(producer["scope"]).split(".") if part])
        for name in node["outputs"]:
            for reader in graph.readers.get(name, []) if name else []:
                if "scope" in reader:
                    around.append([part for part in str(reader["scope"]).split(".") if part])
        if not around:
            continue
        shared = around[0]
        for parts in around[1:]:
            keep = 0
            while keep < min(len(shared), len(parts)) and shared[keep] == parts[keep]:
                keep += 1
            shared = shared[:keep]
        node["scope"] = ".".join(shared)


def _drop_identities(graph: _Graph) -> int:
    """``Identity`` only renames a tensor: point its readers at the original."""
    dropped = 0
    for node in [node for node in graph.nodes if node["op"] == "Identity" and node["domain"] == ""]:
        source, target = node["inputs"][0], node["outputs"][0]
        if target in graph.outputs or source in graph.constants:
            continue
        if any(target in _free_names(reader["attrs"]) for reader in graph.readers.get(target, [])):
            continue  # a branch or a loop's body reads it by its own name
        for reader in graph.readers.get(target, []):
            reader["inputs"] = [source if name == target else name for name in reader["inputs"]]
        graph.remove([node])
        dropped += 1
    return dropped


def _prune(graph: _Graph, keep: Sequence[str]) -> None:
    """Drop every step the outputs in ``keep`` do not depend on."""
    needed: Set[str] = set(keep)
    alive: List[Dict[str, Any]] = []
    for node in reversed(graph.nodes):
        if any(name in needed for name in node["outputs"] if name):
            alive.append(node)
            needed.update(name for name in node["inputs"] if name)
            needed.update(_free_names(node["attrs"]))
    alive.reverse()
    graph.nodes[:] = alive
    for name in [name for name in graph.constants if name not in needed]:
        del graph.constants[name]
    graph.reindex()


def _free_names(attrs: Dict[str, Any]) -> Set[str]:
    """Every tensor name the nested graphs of ``attrs`` read."""
    found: Set[str] = set()
    for value in attrs.values():
        bodies = []
        if isinstance(value, dict) and isinstance(value.get("graph"), dict):
            bodies = [value["graph"]]
        elif isinstance(value, dict) and isinstance(value.get("graphs"), list):
            bodies = value["graphs"]
        for body in bodies:
            for node in body.get("nodes", []):
                found.update(name for name in node["inputs"] if name)
                found.update(_free_names(node.get("attrs", {})))
            found.update(body.get("outputs", []))
    return found


def _mark_plumbing(graph: _Graph, inputs: Sequence[str]) -> None:
    """Hide the steps that carry no data: those that only compute shapes, and arithmetic on constants.

    A tensor is *shape-like* when it comes from ``Shape`` or ``Size``, or from a step whose only live inputs
    are shape-like. It is *derived* when it is computed from constants alone (a step the fold left in the
    graph). Neither says anything about the input the model was given.
    """
    shapes: Set[str] = set()
    derived: Set[str] = set()
    for node in graph.nodes:
        if node["domain"] != "" or node["op"] in _NEVER_FOLD or _free_names(node["attrs"]):
            continue
        reads = [name for name in node["inputs"] if name and name not in graph.constants]
        live = [name for name in reads if name not in derived]
        outputs = [name for name in node["outputs"] if name]
        if node["op"] in ("Shape", "Size") or (live and all(name in shapes for name in live)):
            shapes.update(outputs)
        elif not live:
            derived.update(outputs)
        else:
            continue
        if not any(name in graph.outputs for name in outputs):
            node["hidden"] = True


# --- features and the head ---------------------------------------------------------------


#: Whole numbers by one of these names are a text's token ids. ``stem`` tells several texts apart
#: (``query_input_ids`` and ``passage_input_ids``); an input called just ``input_ids`` is the model's text.
_TEXT_IDS = re.compile(r"^(?P<stem>.*?)_?(?:input_ids|token_ids|word_ids|ids|tokens|token|words|text)$")
#: The other inputs that come with a text, by what they are called. Each is worked out from the ids.
_TEXT_PARTS = (
    ("mask", re.compile(r"^(?P<stem>.*?)_?(?:attention_mask|input_mask|padding_mask|mask)$")),
    ("type_ids", re.compile(r"^(?P<stem>.*?)_?(?:token_type_ids|type_ids|segment_ids|segments?)$")),
    ("positions", re.compile(r"^(?P<stem>.*?)_?(?:position_ids|positions)$")),
    ("length", re.compile(r"^(?P<stem>.*?)_?(?:lengths?|seq_lens?|sequence_lengths?)$")),
)
#: Names a recording goes by. A long run of numbers called one of them is a sound, not a table.
_AUDIO_NAMES = re.compile(r"(?:^|_)(?:input_values|waveforms?|wave|wav|audio|speech|signal|samples|pcm)(?:$|_)")
#: Names a picture goes by: such an input is a picture even when it is small.
_IMAGE_NAMES = re.compile(r"(?:^|_)(?:pixel_values|pixels?|images?|imgs?|pictures?|photos?)(?:$|_)")
#: Names that say nothing about what an input is.
_GENERIC_NAMES = re.compile(
    r"^(?:x|xs|inp|input|inputs|data|in|input_?\d+|args?_?\d+|x_?\d+|l_\w*|in_\d+|keras_tensor\w*|input_layer\w*|"
    r"serving_default\w*)(?::\d+)?$"
)
#: Names an exporter makes up for an input (Keras, TensorFlow Lite, jax2onnx): its feature is named for its kind.
_MADE_UP_NAMES = re.compile(r"^(?:keras_tensor\w*|input_layer\w*|serving_default\w*|in_\d+|args_\d+)(?::\d+)?$")
#: The name a feature gets for its kind when its input's name was made up.
_KIND_NAMES = {
    "image": "image",
    "sequence": "series",
    "tensor": "tensor",
    "tokens": "text",
    "text": "text",
    "audio": "audio",
}
#: A run of more numbers than this is a signal or an embedding, one tensor, rather than columns of a table.
TABLE_MAX_COLUMNS = 1024
#: A recording is at least this many numbers; a shorter run by such a name is still columns.
AUDIO_MIN_SAMPLES = 256
#: Runs of yes-or-no values up to this long are columns; longer ones are one tensor.
FLAG_MAX_COLUMNS = 64
#: The feature an input becomes when Hugging Face's models all give it the same name.
_CANONICAL_NAMES = {"pixel_values": "image", "input_values": "audio", "input_ids": "text"}
#: Steps an input's values pass through unchanged on their way to an embedding table.
_PASS_THROUGH = {"Cast", "Reshape", "Unsqueeze", "Squeeze", "Flatten", "Identity"}


def _feature_name(name: str, kind: Optional[str] = None) -> str:
    """What the feature an input becomes is called: its own name, the name such inputs are always given, or,
    when an exporter made the input's name up, a name for what it is (``image``, ``series``, ``tensor``)."""
    if kind is not None and _MADE_UP_NAMES.match(name.lower()):
        return _KIND_NAMES.get(kind, kind)
    return _CANONICAL_NAMES.get(name.lower(), _clean(name))


def _text_part(lowered: str) -> Optional[Tuple[str, str]]:
    """``(part, stem)`` when an input's name says it is a text's mask, segments, positions or length."""
    for part, pattern in _TEXT_PARTS:
        matched = pattern.match(lowered)
        if matched:
            return part, matched.group("stem")
    return None


def embedding_tables(graph: _Graph, names: Sequence[str]) -> Dict[str, int]:
    """For each input that indexes one embedding table (``Gather`` on a constant matrix), that table's rows.

    That is how many ids the input can hold: a token vocabulary, or how many users or categories there are.
    """
    found: Dict[str, int] = {}
    for name in names:
        sizes: Set[int] = set()
        frontier, seen = [name], set()
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            for reader in graph.readers.get(current, []):
                inputs = reader["inputs"]
                if reader["op"] == "Gather" and len(inputs) > 1 and inputs[1] == current:
                    table = graph.constants.get(inputs[0])
                    if table is not None and np.asarray(table).ndim == 2:
                        sizes.add(int(np.asarray(table).shape[0]))
                elif reader["op"] in _PASS_THROUGH and inputs and inputs[0] == current:
                    frontier.extend(out for out in reader["outputs"] if out)
        if len(sizes) == 1:
            found[name] = sizes.pop()
    return found


def _plan_inputs(
    free: Sequence[Tuple[str, np.dtype, List[int], Optional[int], List[int]]], tables: Dict[str, int]
) -> Dict[str, Dict[str, Any]]:
    """What each input nobody described is, read off its number type, its shape for one row and its name.

    ``free`` holds ``(name, dtype, shape, batch axis, shape of one row)``. See :func:`_bind_inputs` for the rules.
    """
    plans: Dict[str, Dict[str, Any]] = {}
    lowered = {item[0]: item[0].lower() for item in free}
    whole = [item for item in free if np.issubdtype(item[1], np.integer)]
    runs = [item for item in whole if int(np.prod(item[4])) > 1 and _text_part(lowered[item[0]]) is None]
    texts: Dict[str, str] = {}
    for name, _dtype, _shape, _axis, sample in whole:
        if _text_part(lowered[name]) is not None:
            continue
        matched = _TEXT_IDS.match(lowered[name])
        size = int(np.prod(sample))
        by_name = bool(matched) and (size > 1 or lowered[name].endswith("input_ids"))
        # The only run of whole numbers, looked up in an embedding table, is the model's text.
        alone = len(runs) == 1 and runs[0][0] == name and name in tables
        if not (by_name or alone):
            continue
        stem = matched.group("stem") if matched else ""
        feature = _clean(stem) if stem else "text"
        texts[stem] = feature
        plans[name] = {"kind": "text", "feature": feature, "part": "ids", "size": size, "vocab": tables.get(name)}
    for name, dtype, _shape, _axis, sample in free:
        if name in plans:
            continue
        part = _text_part(lowered[name])
        if part is not None and texts:
            kind, stem = part
            owner = texts.get(stem) or (next(iter(texts.values())) if len(texts) == 1 else None)
            if owner is not None:
                plans[name] = {"kind": "text", "feature": owner, "part": kind}
                continue
        size = int(np.prod(sample))
        dims = [int(dim) for dim in sample if int(dim) != 1]
        if np.issubdtype(dtype, np.integer):
            if size > 1 and name in tables:
                # A list of ids into one table: the items someone looked at, the words of a text without a name.
                plans[name] = {
                    "kind": "text", "feature": _feature_name(name), "part": "ids", "size": size, "vocab": tables[name]
                }
            else:
                plans[name] = {"kind": "columns", "number": "integer", "width": size, "vocab": tables.get(name)}
        elif np.issubdtype(dtype, np.bool_):
            if size <= FLAG_MAX_COLUMNS:
                plans[name] = {"kind": "columns", "number": "binary", "width": size}
            else:
                plans[name] = {"kind": "composite", "feature": _feature_name(name)}
        elif not dims or (len(sample) <= 1 and len(dims) == 1 and size <= TABLE_MAX_COLUMNS and not (
            _AUDIO_NAMES.search(lowered[name]) and size >= AUDIO_MIN_SAMPLES
        )):
            plans[name] = {"kind": "columns", "number": "numeric", "width": size}
        else:
            plans[name] = {"kind": "composite", "feature": _feature_name(name)}
    return plans


def _bind_inputs(
    infos: Sequence[Any],
    shapes: Dict[str, List[int]],
    axes: Dict[str, Optional[int]],
    feature_names: Optional[Sequence[str]],
    hints: Dict[str, Any],
    names: _Names,
    tables: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(graph inputs with their bindings, features of the model)`` from what the file says and the hints.

    An input nobody described is read by its number type, its shape for one row and its name, and gets
    features of its own: two inputs never share a feature unless they are parts of one text.

    * Whole numbers called ``input_ids``, ``ids``, ``tokens`` and the like, or the only run of whole numbers
      the model reads, are a text's token ids. ``attention_mask``, ``token_type_ids``, ``position_ids`` and a
      length beside them are worked out from those ids, as the API works them out.
    * Any other whole number (a user, an item, a category) is an integer column, ranged by the embedding table
      it indexes. A run of them that indexes one table is a list of ids; otherwise it is columns.
    * A single float is a numeric column named after its input (a diffusion model's timestep).
    * A run of floats is the columns of a table, unless it is longer than ``TABLE_MAX_COLUMNS`` or named like a
      recording (``input_values``, ``waveform``, ``audio``), which makes it one tensor.
    * Two axes are a series; three with 1, 3 or 4 channels at one end a picture; anything else a tensor.

    ``feature_names`` names every column, or the float columns, or (for a model without columns) its
    features in order. ``tables`` gives, for an input that indexes an embedding table, that table's rows.
    """
    onnx = _onnx()
    tables = dict(tables or {})
    read: List[Tuple[str, np.dtype, List[int], Optional[int], List[int]]] = []
    for info in infos:
        name = str(info.name)
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type))
        shape = [int(dim) for dim in shapes[name]]
        axis = axes.get(name)
        if axis is None or shape[axis] != 1:
            read.append((name, dtype, shape, None, list(shape)))
        else:
            read.append((name, dtype, shape, axis, shape[:axis] + shape[axis + 1 :]))
    for name, dtype, _shape, _axis, sample in read:
        hint = hints.get(name)
        if (
            isinstance(hint, dict)
            and "feature" in hint
            and not hint.get("layout")
            and not hint.get("part")
            and np.issubdtype(dtype, np.floating)
            and int(np.prod(sample)) == 1
        ):
            # A single number is a column, whatever the hint calls it: a tensor of no size is not a feature.
            hints[name] = {"features": [str(hint["feature"])]}
    plans = _plan_inputs([item for item in read if item[0] not in hints], tables)
    columns = _name_columns(read, plans, feature_names)
    composite = [name for name, *_ in read if plans.get(name, {}).get("kind") in ("text", "composite")]
    renamed: Dict[str, str] = {}
    hinted = any(isinstance(hint, dict) and ("feature" in hint or "features" in hint) for hint in hints.values())
    if feature_names is not None and not columns:
        # No table to name the columns of: the names are the features', in the order the model reads them.
        own = list(dict.fromkeys(str(plans[name]["feature"]) for name in composite))
        given = [str(item) for item in feature_names]
        if own and len(given) == len(own):
            renamed = dict(zip(own, given))
        elif own and not hinted:
            raise OnnxImportError(
                f"feature_names gives {len(given)} names, and the model reads {len(own)} features "
                f"({', '.join(own)}) and no table of columns.",
                param="feature_names",
            )
    bound: List[Dict[str, Any]] = []
    features: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    used: Set[str] = set()
    for name, dtype, shape, axis, sample in read:
        entry: Dict[str, Any] = {"name": name, "dtype": dtype.name, "shape": list(shape), "batch_axis": axis}
        hint = hints.get(name)
        plan = plans.get(name)
        if isinstance(hint, dict) and ("value" in hint or hint.get("zeros")):
            entry["from"] = {"value": hint["value"]} if "value" in hint else {"zeros": True}
        elif isinstance(hint, dict) and "features" in hint:
            entry["from"] = {"features": [str(item) for item in hint["features"]]}
            # One column may say what it is: a category (with its categories), a whole number, a flag.
            described = dict(hint.get("describe") or {}) if len(entry["from"]["features"]) == 1 else {}
            for named in entry["from"]["features"]:
                if named not in seen:
                    seen[named] = {"name": named, "type": "numeric", **described}
                    features.append(seen[named])
        elif isinstance(hint, dict) and "feature" in hint:
            source = {key: hint[key] for key in ("feature", "layout", "part") if key in hint}
            pad = _pad_id(hint, name)
            if pad is not None:
                source["pad_id"] = pad
            entry["from"] = source
            feature = str(hint["feature"])
            if feature not in seen:
                described = dict(hint.get("describe") or {})
                for key in ("type", "sample_rate"):
                    if key in hint:
                        described[key] = hint[key]
                seen[feature] = _composite_feature(feature, sample, source, dtype, described)
                features.append(seen[feature])
        elif plan is not None and plan["kind"] == "columns":
            entry["from"] = {"features": columns[name]}
            for named in columns[name]:
                if named not in seen:
                    seen[named] = _column_feature(named, plan)
                    features.append(seen[named])
        elif plan is not None and plan["kind"] == "text":
            feature = renamed.get(plan["feature"], plan["feature"])
            entry["from"] = {"feature": feature, "part": plan["part"]}
            if plan["part"] == "ids":
                feature = _unique(feature, used, seen)
                entry["from"]["feature"] = feature
                described = {"name": feature, "type": "tokens", "max_length": int(plan["size"])}
                if plan.get("vocab"):
                    described["vocab_size"] = int(plan["vocab"])
                seen[feature] = described
                features.append(described)
                used.add(feature)
        else:
            guessed, described_feature = _guess_composite(name, sample, dtype, names)
            wanted = renamed.get(str(guessed["feature"]), str(guessed["feature"]))
            feature = _unique(wanted, used, seen)
            guessed["feature"] = feature
            described_feature["name"] = feature
            entry["from"] = guessed
            seen[feature] = described_feature
            features.append(described_feature)
            used.add(feature)
        bound.append(entry)
    for entry in bound:
        source = entry.get("from") or {}
        if source.get("part") not in (None, "ids") and source.get("feature") not in seen:
            # A mask whose text was renamed after it was planned: point it at the text's final name.
            known = [feature for feature in seen if seen[feature].get("type") in ("tokens", "text")]
            if len(known) == 1:
                source["feature"] = known[0]
    return bound, features


def _unique(wanted: str, used: Set[str], seen: Dict[str, Dict[str, Any]]) -> str:
    """``wanted``, or ``wanted_2``…: a feature of its own for each input."""
    name, count = wanted, 1
    while name in used or name in seen:
        count += 1
        name = f"{wanted}_{count}"
    return name


def _column_feature(name: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    number = str(plan.get("number") or "numeric")
    feature: Dict[str, Any] = {"name": name, "type": number}
    if number == "integer" and plan.get("vocab"):
        feature["min"] = 0.0
        feature["max"] = float(int(plan["vocab"]) - 1)
    return feature


def _name_columns(
    read: Sequence[Tuple[str, np.dtype, List[int], Optional[int], List[int]]],
    plans: Dict[str, Dict[str, Any]],
    feature_names: Optional[Sequence[str]],
) -> Dict[str, List[str]]:
    """The column names of each input that is columns of a table: the caller's, or made from the input's name."""
    inputs = [(name, plans[name]) for name, *_ in read if plans.get(name, {}).get("kind") == "columns"]
    if not inputs:
        return {}
    floats = [(name, plan) for name, plan in inputs if plan["number"] == "numeric"]
    every = sum(int(plan["width"]) for _, plan in inputs)
    floating = sum(int(plan["width"]) for _, plan in floats)
    given = [str(item) for item in feature_names] if feature_names is not None else None
    named: Dict[str, List[str]] = {}
    if given is not None:
        if len(given) == every:
            chosen = inputs
        elif floats and len(given) == floating:
            chosen = floats
        elif len(inputs) == 1:
            only, plan = inputs[0]
            raise OnnxImportError(
                f"Input '{only}' has {int(plan['width'])} columns, and {len(given)} feature names were given.",
                param="feature_names",
            )
        else:
            raise OnnxImportError(
                "feature_names does not cover every column of the model's inputs.", param="feature_names"
            )
        cursor = 0
        for name, plan in chosen:
            width = int(plan["width"])
            named[name] = given[cursor : cursor + width]
            cursor += width
    one_table = len(floats) == 1
    for name, plan in inputs:
        if name in named:
            continue
        width = int(plan["width"])
        if plan["number"] == "numeric" and one_table and (width > 1 or _GENERIC_NAMES.match(name.lower())):
            named[name] = [f"x{index}" for index in range(width)]
        elif width == 1:
            named[name] = [_feature_name(name)]
        else:
            named[name] = [f"{name}_{index}" for index in range(width)]
    return named


#: The sample rate a recording is taken to have when nothing says (what speech models are trained on).
DEFAULT_SAMPLE_RATE = 16000


def _audio_feature(feature: str, sample: Sequence[int], described: Dict[str, Any]) -> Dict[str, Any]:
    samples = int(np.prod([int(dim) for dim in sample])) if len(sample) else 1
    base: Dict[str, Any] = {"name": feature, "type": "audio", "samples": samples, "sample_rate": DEFAULT_SAMPLE_RATE}
    base.update({key: value for key, value in described.items() if key not in ("type",)})
    return base


def _pad_id(hint: Dict[str, Any], name: str) -> Optional[int]:
    """A hint's ``pad_id``, the token that fills a text after its last one: a whole number, or ``None`` when the
    hint gives none. ``None`` written out says the same (a tokenizer with no padding token says it so), and the
    places are then filled with 0, as the API fills them. One that is not a whole number is refused here, as the
    API would refuse the structure."""
    pad = hint.get("pad_id")
    if pad is None:
        return None
    if isinstance(pad, (float, np.floating)) and float(pad).is_integer():
        pad = int(pad)
    if isinstance(pad, (bool, np.bool_)) or not isinstance(pad, (int, np.integer)):
        raise OnnxImportError(
            f"inputs gives '{name}' a pad_id that is not a whole number: {hint['pad_id']!r}.", param="inputs"
        )
    return int(pad)


def _composite_feature(
    feature: str, sample: Sequence[int], source: Dict[str, Any], dtype: np.dtype, described: Dict[str, Any]
) -> Dict[str, Any]:
    if str(described.get("type") or "") == "audio" or described.get("sample_rate"):
        return _audio_feature(feature, sample, described)
    if source.get("layout"):
        if len(sample) != 3:
            raise OnnxImportError(
                f"inputs gives '{feature}' a picture's layout, and one row of it has {len(sample)} axes "
                f"({' x '.join(str(int(dim)) for dim in sample) or 'none'}), not height, width and channels.",
                param="inputs",
            )
        channels_first = str(source["layout"]).lower() in ("nchw", "chw")
        height, width, channels = (sample[1], sample[2], sample[0]) if channels_first else sample
        base = {"name": feature, "type": "image", "height": int(height), "width": int(width), "channels": int(channels)}
    elif source.get("part"):
        base = {"name": feature, "type": "text", "max_length": int(np.prod(sample))}
    elif np.issubdtype(dtype, np.integer):
        base = {"name": feature, "type": "tokens", "max_length": int(np.prod(sample))}
    elif len(sample) == 2:
        base = {"name": feature, "type": "sequence", "shape": [int(dim) for dim in sample]}
    else:
        base = {"name": feature, "type": "tensor", "shape": [int(dim) for dim in sample] or [1]}
    base.update(described)
    return base


def _guess_composite(
    name: str, sample: Sequence[int], dtype: np.dtype, names: _Names
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """What kind of many-valued input a tensor is, from its type, shape and name, when nobody said."""
    lowered = name.lower()
    feature = _feature_name(name)
    dims = [int(dim) for dim in sample]
    if len(dims) == 3:
        feature = _feature_name(name, "image")
        named = bool(_IMAGE_NAMES.search(lowered))
        least = 2 if named else 8
        first = dims[0] in (1, 3, 4) and min(dims[1], dims[2]) >= least
        last = dims[2] in (1, 3, 4) and min(dims[0], dims[1]) >= least
        if first or last:
            layout = "nchw" if first and (not last or dims[0] <= dims[2]) else "nhwc"
            source = {"feature": feature, "layout": layout}
            return source, _composite_feature(feature, sample, source, dtype, {})
    feature = _feature_name(name, "sequence" if len(dims) == 2 and not np.issubdtype(dtype, np.integer) else "tensor")
    source = {"feature": feature}
    length = int(np.prod(dims)) if dims else 1
    if (
        _AUDIO_NAMES.search(lowered)
        and length >= AUDIO_MIN_SAMPLES
        and not np.issubdtype(dtype, np.integer)
        and len([dim for dim in dims if dim != 1]) == 1
    ):
        # A recording: one run of samples (a channel axis of one before it is allowed).
        return source, _audio_feature(_feature_name(name, "audio"), dims, {})
    if len(dims) == 2 and dims[0] == 1 and dims[1] > TABLE_MAX_COLUMNS:
        # One step of a very long run: a recording kept behind a channel axis, not a series of one step.
        return source, {"name": feature, "type": "tensor", "shape": dims}
    return source, _composite_feature(feature, sample, source, dtype, {})


def _pick_output(
    model: Any,
    raw: Dict[str, Any],
    results: Dict[str, Any],
    task: Optional[str],
    output: Optional[str],
    given: Optional[Sequence[str]] = None,
    pictures: Sequence[Tuple[int, int]] = (),
) -> Tuple[str, str, Optional[str], Optional[List[str]]]:
    """``(tensor, link, task, labels)``: which output is the prediction and how to read it.

    ``pictures`` is the height and width of each picture the model reads (:func:`_picture_sizes`), which says
    which of several outputs a segmentation's mask is (:func:`_mask_output`).

    With no task, the output says what the model is. It is a classifier when its values are probabilities,
    when the file names its classes, when ``given`` (the caller's labels) has a name for each value (or two
    names for one value), or when its one value comes out of a sigmoid; several values out of a sigmoid with a
    label for each are a multi-label classifier. Anything else is a regression, which answers with one number.
    """
    producers = {name: node for node in raw["nodes"] for name in node["outputs"] if name}
    labels: Optional[List[str]] = None

    def listed(node: Dict[str, Any]) -> Optional[List[str]]:
        """The classes a classical step lists (a classifier's, or a ZipMap's), in the order of its scores."""
        attrs = node.get("attrs") or {}
        for key in ("classlabels_strings", "classlabels_int64s", "classlabels_ints"):
            found = attrs.get(key)
            if found is not None and len(found):
                return [item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item) for item in found]
        return None

    def behind(name: str) -> str:
        """The tensor ``name`` holds the values of, behind the steps that only lay them out (a cast, a reshape)."""
        for _step in range(64):
            node = producers.get(name)
            if node is None or node["op"] not in _PASS_THROUGH or str(node.get("domain") or "") not in ("", "ai.onnx"):
                break
            if not node["inputs"] or not node["inputs"][0]:
                break
            name = node["inputs"][0]
        return name

    def voted(scores: str, width: int) -> Optional[List[str]]:
        """The classes of a file whose own label is the highest of the scores ``scores``: one of its outputs is,
        behind steps that only lay it out, a constant table of ``width`` classes looked up at the argmax of those
        scores. That is how skl2onnx writes the label of a classifier it converts to plain steps."""
        constants = raw.get("constants") or {}
        for name in raw["outputs"]:
            looked_up = producers.get(behind(name))
            if looked_up is None or looked_up["op"] not in ("ArrayFeatureExtractor", "Gather"):
                continue
            reads = list(looked_up["inputs"])
            if len(reads) != 2 or reads[0] not in constants or not reads[1]:
                continue
            best = producers.get(behind(reads[1]))
            if best is None or best["op"] != "ArgMax" or not best["inputs"] or behind(best["inputs"][0]) != scores:
                continue
            table = np.asarray(constants[reads[0]]).reshape(-1)
            if table.size == width:
                return [
                    item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item) for item in table.tolist()
                ]
        return None

    def through_zipmap(name: str) -> str:
        node = producers.get(name)
        nonlocal labels
        if node is not None and node["op"] == "ZipMap":
            found = listed(node)
            if found:
                labels = found
            return node["inputs"][0]
        return name

    candidates = [through_zipmap(name) for name in raw["outputs"]]
    # An argmax over a convolution's class channels leaves one class number at every pixel: a segmentation,
    # whatever steps that keep those numbers where they are (a cast, a squeeze) come after it.
    candidate = output if output is not None else candidates[0] if len(candidates) == 1 else None
    if task is None and candidate is not None and can_hold_a_mask(_shape(results.get(candidate))):
        made = _class_axis(raw["nodes"], candidate, results, raw.get("constants"))
        if made is not None and made["kind"] == "indices" and made["picture"] and None not in made["picture"]:
            task = "segmentation"
    if output is not None:
        if output not in producers and output not in raw["outputs"]:
            raise OnnxImportError(f"The model has no output called '{output}'.")
        chosen = through_zipmap(output)
    else:
        floats = [
            name for name in candidates if isinstance(results.get(name), np.ndarray) and results[name].dtype.kind == "f"
        ]
        if task == "segmentation":
            chosen = _mask_output(candidates, results, pictures)
            return chosen, "none", task, None
        if not floats:
            raise OnnxImportError("This model has no numeric output to read a prediction from.")
        # A classifier converted from scikit-learn gives the label first and the probabilities second.
        chosen = (
            floats[-1]
            if len(floats) > 1 and any(producers.get(n, {}).get("domain") == "ai.onnx.ml" for n in floats)
            else floats[0]
        )
    value = np.asarray(results[chosen], dtype=float)
    row = value.reshape(value.shape[0], -1) if value.ndim >= 2 else value.reshape(1, -1)
    width = row.shape[1]
    if labels is None:
        # The file's own classes, where a classical classifier's step lists them. The step the scores come out of
        # says, behind any steps that only lay them out (skl2onnx puts a cast after a classifier it writes without
        # a ZipMap). Otherwise the steps of the file that list a class for each score do, when they agree.
        maker = producers.get(behind(chosen))
        one_of_two = False
        if maker is not None and maker["op"] == "ArrayFeatureExtractor" and width == 1 and len(maker["inputs"]) == 2:
            # A two-class model with no probabilities (a linear SVM) is given one column: its second class's score.
            picked = (raw.get("constants") or {}).get(maker["inputs"][1])
            classifier = producers.get(behind(maker["inputs"][0]))
            if (
                picked is not None
                and np.asarray(picked).size == 1
                and int(np.asarray(picked).reshape(-1)[0]) == 1
                and _gives_margins(classifier)
            ):
                maker, one_of_two = classifier, True
        if maker is not None and str(maker.get("domain") or "") == "ai.onnx.ml":
            found = listed(maker)
            if found is not None and (not one_of_two or len(found) == 2):
                labels = found
        if labels is None and width > 1 and value.ndim == 2:
            labels = voted(behind(chosen), width)
        if labels is None and width > 1:
            lists = {
                tuple(found)
                for node in raw["nodes"]
                if str(node.get("domain") or "") == "ai.onnx.ml"
                for found in [listed(node)]
                if found is not None and len(found) == width
            }
            if len(lists) == 1:
                labels = list(lists.pop())
    link = "none"
    classifying = (None, "binary_classification", "multiclass_classification", "multilabel_classification")
    if task in classifying:
        sums_to_one = width > 1 and np.allclose(row.sum(axis=1), 1.0, atol=1e-4) and row.min() >= -1e-6
        in_unit = row.min() >= -1e-6 and row.max() <= 1.0 + 1e-6
        squashed = bool(in_unit) and producers.get(chosen, {}).get("op") == "Sigmoid"
        if task is None:
            named = [str(item) for item in given] if given else None
            if sums_to_one:
                task = "binary_classification" if width == 2 else "multiclass_classification"
            elif labels:
                task = "binary_classification" if len(labels) == 2 else "multiclass_classification"
            elif named and width > 1 and len(named) == width:
                # Scores for each class (the logits a network ends in), or a yes-or-no answer for each label.
                if squashed:
                    task = "multilabel_classification"
                else:
                    task = "binary_classification" if width == 2 else "multiclass_classification"
            elif named and width == 1 and len(named) == 2:
                task = "binary_classification"
            elif named:
                raise OnnxImportError(
                    f"labels names {len(named)} classes, and the model gives {width} values for a row. Pass a "
                    "label for each value (two for a model that gives one), or output= to read another output.",
                    param="labels",
                )
            elif width == 1 and squashed:
                task = "binary_classification"
            else:
                task = "regression"
        if task == "multilabel_classification":
            link = "none" if squashed else "sigmoid"
        elif task != "regression":
            if width == 1:
                link = "none" if squashed else "sigmoid"
            elif not sums_to_one:
                link = "softmax"
    elif task in ("text_generation", "sequence_to_sequence"):
        # Scores for every token at each position. A model that ends in them leaves the softmax to its loss.
        scores = value.reshape(-1, value.shape[-1]) if value.ndim else value.reshape(1, 1)
        link = (
            "none"
            if scores.shape[1] > 1 and scores.min() >= -1e-6 and np.allclose(scores.sum(axis=1), 1.0, atol=1e-4)
            else "softmax"
        )
    elif task == "clustering":
        # Distances to each cluster's centre (the nearest is the cluster), or each cluster's probability.
        sums_to_one = width > 1 and np.allclose(row.sum(axis=1), 1.0, atol=1e-4) and row.min() >= -1e-6
        link = "argmax" if sums_to_one else "argmin"
    return chosen, link, task, labels


def _token_id(value: Any) -> Optional[int]:
    """``value`` as a token id (a whole number from 0, a float that holds one included), or ``None``."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    if not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**31:
        return None
    return int(value)


#: What a hint cannot say of an input it also calls generated: each is another way to fill the input.
_NOT_GENERATED = ("value", "zeros", "features", "layout")


def _written_so_far(
    hints: Dict[str, Any],
    infos: Sequence[Any],
    shapes: Dict[str, List[int]],
    given: Dict[str, np.ndarray],
    task: Optional[str],
) -> Optional[Dict[str, Any]]:
    """What an input hinted ``{"generated": True}`` is, with its hint rewritten as the feature it is bound to.

    An encoder-decoder reads what its decoder has written so far as an input of token ids. On Kaimei that is a
    feature of token ids of its own, marked ``generated``: a caller never gives it, a prediction starts it from the
    token the decoder starts from, and generation writes into it. The input is bound to that feature's ids
    (``{"feature": ..., "part": "ids", "pad_id": ...}``) and the structure's ``decoder`` says the rest. That is the
    form the API runs, and the one the client writes for a model whose decoder it finds itself.

    The hint may also say ``feature`` (the feature's name: ``decoder``), ``pad_id`` (what fills the input after the
    last token written: 0), ``start_token`` (the token the decoder starts from: the first id of the input's sample
    when one was given, else the padding token) and ``eos_token`` (the token that ends what it writes). An input
    that comes with the ids by its name (``decoder_attention_mask`` beside ``decoder_input_ids``) and has no hint
    of its own is bound to the same feature.

    Returns ``None`` when no input is hinted so, else ``{"input", "feature", "start_token", "pad_id"}`` and, when
    the hint said them, ``"pad_token"`` and ``"eos_token"``. A hint that cannot be written in that form is refused
    here, naming ``inputs``: a structure the API would refuse is never written.
    """
    onnx = _onnx()
    types = {str(info.name): info.type.tensor_type.elem_type for info in infos}
    said = [name for name, hint in hints.items() if isinstance(hint, dict) and hint.get("generated") and name in types]
    if not said:
        return None
    name = said[0]
    hint = hints[name]
    what = f"inputs says '{name}' is what the model has written so far"

    def refuse(why: str) -> OnnxImportError:
        return OnnxImportError(why, param="inputs")

    if len(said) > 1:
        raise refuse(
            f"inputs says {len(said)} inputs are what the model has written so far ({', '.join(said)}), and a "
            "model has one decoder: mark one of them generated."
        )
    if task not in (None, "sequence_to_sequence"):
        raise refuse(f"{what}, which only an encoder-decoder reads: its task is sequence_to_sequence, not {task}.")
    other = next((key for key in _NOT_GENERATED if hint.get(key) is not None and hint.get(key) is not False), None)
    if other is None and hint.get("part") not in (None, "ids"):
        other = "part"
    if other is not None:
        raise refuse(f"{what} and also gives it '{other}': an input is filled one way.")
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(types[name]))
    if not np.issubdtype(dtype, np.integer):
        raise refuse(f"{what}, and it holds {dtype.name} values: what a decoder has written is token ids.")
    if int(np.prod(shapes[name])) < 1:
        raise refuse(f"{what}, and it has no room for a token.")
    feature = hint.get("feature", "decoder")
    if not isinstance(feature, str) or not feature:
        raise refuse(f"inputs gives '{name}' a feature that is not a name: {feature!r}.")
    tokens: Dict[str, Optional[int]] = {}
    for key in ("pad_id", "start_token", "eos_token"):
        tokens[key] = None if hint.get(key) is None else _token_id(hint[key])
        if hint.get(key) is not None and tokens[key] is None:
            raise refuse(
                f"inputs gives '{name}' a {key} that is not a token id (a whole number from 0): {hint[key]!r}."
            )
    pad = tokens["pad_id"] or 0
    start = tokens["start_token"]
    if start is None and name in given and given[name].size:
        begun = given[name].reshape(-1)[0]
        start = _token_id(begun.item() if hasattr(begun, "item") else begun)
        if start is None:
            raise refuse(
                f"{what}, and its sample starts with {begun!r}, which is not a token id (a whole number from 0): "
                "say the token the decoder starts from with start_token."
            )
    if start is None:
        start = pad
    found: Dict[str, Any] = {"input": name, "feature": feature, "start_token": start, "pad_id": pad}
    if tokens["eos_token"] is not None:
        found["eos_token"] = tokens["eos_token"]
    if tokens["pad_id"] is not None:
        found["pad_token"] = pad
    described = dict(hint.get("describe") or {})
    hints[name] = {
        "feature": feature,
        "part": "ids",
        "pad_id": pad,
        "type": "tokens",
        **({"describe": described} if described else {}),
        **({"vocab": hint["vocab"]} if "vocab" in hint else {}),
    }
    matched = _TEXT_IDS.match(name.lower())
    stem = matched.group("stem") if matched else ""
    for beside in types:
        part = _text_part(beside.lower())
        if stem and beside != name and beside not in hints and part is not None and part[1] == stem:
            # The decoder's mask, positions or length: worked out from what it has written, as a text's are.
            hints[beside] = {"feature": feature, "part": part[0]}
    return found


# --- the importer ------------------------------------------------------------------------


def import_onnx(
    source: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    input_shapes: Optional[Dict[str, Sequence[int]]] = None,
    samples: Optional[Dict[str, Any]] = None,
    output: Optional[str] = None,
    framework: str = "onnx",
    name: Optional[str] = None,
    fuse: bool = True,
    check: bool = True,
    base_dir: Optional[str] = None,
    reference: Optional[Dict[str, Any]] = None,
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """An ONNX model as a ``family: "graph"`` structure, its constants still numpy arrays.

    ``feature_names`` names the columns of a table-shaped input. ``inputs`` says what any other input is:
    ``{"pixels": {"feature": "image", "layout": "nchw"}}``, ``{"input_ids": {"feature": "text", "part":
    "ids"}}``, ``{"x": {"features": [...]}}``, or ``{"state": {"zeros": True}}`` for one the model is always
    given. ``{"decoder_input_ids": {"generated": True}}`` says an input is what an encoder-decoder has written so
    far: it becomes a feature of token ids marked ``generated`` (``decoder`` unless the hint names it), the input
    is bound to that feature's ids, and the structure's ``decoder`` says the token it starts from (see
    :func:`_written_so_far` for what else the hint may say). ``input_shapes`` fills in sizes the file leaves open
    (all but the batch axis, which is one row).
    ``samples`` are example inputs (``name -> array`` of one row) used to record shapes and the check;
    without them random values of the right type are used. ``output`` picks the tensor the prediction is
    read from when the model has several.

    For a segmentation, ``mask_layout`` says where its classes are when the graph does not (see
    :func:`segmentation_head`), and ``threshold`` where one score for each pixel is cut, in the model's own
    numbers: a pixel is the foreground when its score is that or more. Without it the cut is where the model's
    own probability is one half: 0.5 for a model that ends in a sigmoid, 0 for one that gives raw scores.

    The result has the keys of every Kaimei structure plus ``graph``; ``features`` lists the model's
    features as the API registers them. Pass it to :func:`kaimei.structure.finalize_spec` (the client does)
    to move large constants into binary tensors.
    """
    import os

    if base_dir is None:
        base_dir = os.path.dirname(os.fspath(source)) if isinstance(source, (str, os.PathLike)) else ""
    model = load_model(source)
    try:
        model = expand_functions(model)
    except OnnxReadError as exc:
        raise OnnxImportError(str(exc)) from exc
    hints = dict(inputs or {})
    if labels is not None:
        # Names in a NumPy array or a pandas Index (an encoder's classes_) are a list of names like any other.
        labels = [str(item) for item in labels] or None
    given = {str(key): np.asarray(value) for key, value in (samples or {}).items()}
    infos = _input_infos(model)
    if not infos:
        raise OnnxImportError("This model takes no inputs.")
    shapes: Dict[str, List[int]] = {}
    axes: Dict[str, Optional[int]] = {}
    for info in infos:
        shapes[str(info.name)], axes[str(info.name)] = _static_shape(info, dict(input_shapes or {}), given)

    # An input the model writes itself (a decoder's): a feature of its own, marked generated. See _written_so_far.
    written = _written_so_far(hints, infos, shapes, given, task)
    if written is not None:
        task = "sequence_to_sequence"
    vocab = {key: int(value.get("vocab", 2)) for key, value in hints.items() if isinstance(value, dict)}
    feeds = _feeds_for(model, shapes, 0, vocab)
    for key, value in given.items():
        if key in feeds:
            feeds[key] = value.astype(feeds[key].dtype).reshape(shapes[key])
    for key, value in hints.items():
        if isinstance(value, dict) and key in feeds and ("value" in value or value.get("zeros")):
            fixed = (
                np.zeros(shapes[key])
                if value.get("zeros")
                else np.broadcast_to(np.asarray(value["value"]), shapes[key])
            )
            feeds[key] = np.array(fixed).astype(feeds[key].dtype)
    if written is not None:
        # Before its first token a decoder has written the token it starts from, and nothing after it.
        reads = written["input"]
        started = np.full(int(np.prod(shapes[reads])), written["pad_id"], dtype=feeds[reads].dtype)
        started[0] = written["start_token"]
        feeds[reads] = started.reshape(shapes[reads])

    folded = fold_constants(model, base_dir or "")
    seeded = _pin_seeds(folded.graph)
    try:
        raw = read_model(folded, base_dir=base_dir or "")
    except OnnxReadError as exc:
        raise OnnxImportError(str(exc)) from exc
    results = run_reference(folded, feeds)

    if mask_layout is not None and task is None:
        task = "segmentation"
    pictures = _picture_sizes(shapes, hints)
    chosen, link, task, found_labels = _pick_output(folded, raw, results, task, output, labels, pictures)
    mask_head = None
    cut_by_values = whole_floats = False
    if task == "segmentation":
        mask_head, cut_by_values, whole_floats = _mask_head(
            results[chosen],
            labels or found_labels,
            mask_layout,
            raw["nodes"],
            chosen,
            results,
            raw.get("constants"),
            threshold,
        )
    elif mask_layout is not None:
        raise OnnxImportError("mask_layout applies to a segmentation model.", param="mask_layout")
    elif threshold is not None:
        raise OnnxImportError(
            "threshold applies to a segmentation that gives one score for each pixel.", param="threshold"
        )
    graph = _Graph(raw)
    graph.outputs = [chosen]
    _prune(graph, [chosen])
    fusions = 0
    if fuse:
        _drop_identities(graph)
        fusions += _fuse_gelu(graph)
        fusions += _fuse_layer_norm(graph)
        fusions += _fuse_linear(graph)
        _prune(graph, [chosen])
    _inherit_scopes(graph)
    _mark_plumbing(graph, [str(info.name) for info in infos])

    reserved = list(feature_names or [])
    names = _Names(reserved)
    used_inputs = {name for node in graph.nodes for name in node["inputs"] if name} | _all_free(graph.nodes)
    read = [info for info in infos if str(info.name) in used_inputs or str(info.name) == chosen]
    if not read:
        raise OnnxImportError("The model's output does not depend on any of its inputs.")
    tables = embedding_tables(graph, [str(info.name) for info in read])
    bound, features = _bind_inputs(read, shapes, axes, feature_names, hints, names, tables)
    decoder: Optional[Dict[str, Any]] = None
    if written is not None:
        decoder = _decoder_of(written, bound, features)
    for feature in features:
        names.taken.add(str(feature["name"]))

    rename: Dict[str, str] = {}
    for entry in bound:
        rename[entry["name"]] = names.make(f"in.{entry['name']}")
    values: Dict[str, Any] = {}
    nodes: List[Dict[str, Any]] = []
    for node in graph.nodes:
        path, leaf = node_path(node)
        step_path = f"{path}.{_clean(leaf)}" if path else _clean(leaf or node["op"])
        outputs: List[str] = []
        for position, tensor in enumerate(node["outputs"]):
            if not tensor:
                outputs.append("")
                continue
            wanted = step_path if position == 0 else f"{step_path}.{position}"
            rename[tensor] = names.make(wanted)
            outputs.append(rename[tensor])
            computed = results.get(tensor)
            if isinstance(computed, np.ndarray):
                values[rename[tensor]] = {
                    "shape": [int(dim) for dim in computed.shape],
                    "dtype": "string" if computed.dtype.kind in "OUS" else computed.dtype.name,
                    # What the model file calls this tensor: whoever runs the file can say what it held.
                    "source": str(tensor)[:240],
                }
        entry = {
            "id": outputs[0] or names.make(step_path),
            "op": node["op"],
            "domain": node["domain"],
            "inputs": list(node["inputs"]),
            "outputs": outputs,
            "attrs": node["attrs"],
            "path": step_path,
            "label": path.rsplit(".", 1)[-1] if path and _is_layer(node) else "",
        }
        if node.get("hidden"):
            entry["hidden"] = True
        nodes.append(entry)
    constants: Dict[str, Any] = {}
    for key, value in graph.constants.items():
        rename[key] = names.make(f"const.{key}")
        constants[rename[key]] = value
    for entry in nodes:
        entry["inputs"] = [rename.get(item, item) if item else "" for item in entry["inputs"]]
        _rename_bodies(entry["attrs"], rename)
        if not entry["label"]:
            del entry["label"]
    for entry in bound:
        entry["name"] = rename[entry["name"]]

    head: Dict[str, Any] = {"output": rename.get(chosen, chosen), "link": link}
    if mask_head is not None:
        head.update(mask_head)
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "graph",
        "framework": framework,
        "task": task,
        "feature_names": [str(feature["name"]) for feature in features],
        "features": features,
        "graph": {
            "opsets": dict(raw["opsets"]),
            "inputs": bound,
            "constants": constants,
            "nodes": nodes,
            "outputs": [rename.get(chosen, chosen)],
            "values": values,
        },
        "head": head,
        "reader": 2,
    }
    if name:
        spec["name"] = name
    if decoder is not None:
        spec["decoder"] = decoder
    final_labels = [str(item) for item in labels] if labels else found_labels
    if final_labels and task != "regression":
        spec["outputs"] = {"labels": final_labels}
    spec["metadata"] = {
        "imported_from": "onnx",
        "producer": f"{raw['producer']['name']} {raw['producer']['version']}".strip(),
        "opset": int(raw["opsets"].get("", 0)),
        "fused_steps": fusions,
    }
    if seeded:
        # Steps that draw at random were given seeds, so the model is shown the same way every time.
        spec["metadata"]["seeded_steps"] = seeded
    if check or cut_by_values or whole_floats:
        # Made-up ids stay inside the tables they index, so every check row is one the model can be given.
        ranged = {**{name: size for name, size in tables.items()}, **vocab}
        rows = _check_rows(folded, bound, rename, shapes, axes, feeds, chosen, link, ranged, str(task), decoder)
        if check:
            spec["check"] = rows
        # What no step says of a mask, the model's own values do: on every check row (the first of them is the
        # row the model was read on), not on that one alone.
        every = rows["outputs"] or [results[chosen]]
        if whole_floats and all(_all_whole(value) for value in every):
            raise _numbers_or_scores(results[chosen].shape, len(every))
        if cut_by_values:
            head["threshold"], head["score_kind"] = _cut_by_values(every)
    if reference is not None:
        # For checking an engine against the reference tensor by tensor: every value the reference
        # computed for the first row, under the names the structure uses.
        reference["values"] = {rename[key]: value for key, value in results.items() if key in rename}
        reference["feeds"] = {rename.get(key, key): value for key, value in feeds.items()}
        # The tensor the prediction is read from, by the file's own name for it: which of several outputs it is
        # was decided here, with the model run and its graph read.
        reference["output"] = chosen
    return spec


def _decoder_of(
    written: Dict[str, Any], bound: Sequence[Dict[str, Any]], features: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Mark the feature a decoder writes into as generated, and return the structure's ``decoder``.

    ``written`` is what :func:`_written_so_far` found. The feature starts as the one token the decoder starts from
    (its ``baseline``), which is all a caller's prediction ever gives it.
    """
    name = str(written["feature"])
    feature = next((item for item in features if str(item.get("name")) == name), None)
    readers = [
        entry for entry in bound
        if (entry.get("from") or {}).get("feature") == name and (entry.get("from") or {}).get("part") == "ids"
    ]  # fmt: skip
    if feature is None or not any(entry["name"] == written["input"] for entry in readers):
        raise OnnxImportError(
            f"inputs says '{written['input']}' is what the model has written so far, and the model's output does "
            "not depend on it.",
            param="inputs",
        )
    if len(readers) != 1:
        raise OnnxImportError(
            f"inputs gives what the model has written so far the feature '{name}', which another input reads as "
            "well: give it a feature of its own.",
            param="inputs",
        )
    feature["generated"] = True
    feature["baseline"] = [int(written["start_token"])]
    decoder: Dict[str, Any] = {"feature": name, "start_token": int(written["start_token"])}
    for key in ("eos_token", "pad_token"):
        if key in written:
            decoder[key] = int(written[key])
    decoder["max_length"] = int(feature["max_length"])
    return decoder


# --- a segmentation's mask ----------------------------------------------------------------

#: Steps that keep every axis of their first input where it is: one value for each value, or the same axes at
#: other sizes (a resize, a pad, a crop, a pooling).
_KEEPS_AXES = frozenset(
    {
        "Identity", "Cast", "CastLike", "Dropout", "QuantizeLinear", "DequantizeLinear",
        "Relu", "LeakyRelu", "Sigmoid", "HardSigmoid", "Tanh", "Softmax", "LogSoftmax", "Hardmax", "Softplus",
        "Softsign", "Elu", "Selu", "Celu", "Gelu", "Mish", "Swish", "HardSwish", "ThresholdedRelu", "Shrink", "Clip",
        "Neg", "Abs", "Exp", "Log", "Sqrt", "Reciprocal", "Erf", "Floor", "Ceil", "Round", "Sign", "Not",
        "BatchNormalization", "InstanceNormalization", "GroupNormalization", "LayerNormalization",
        "LpNormalization", "MeanVarianceNormalization", "LRN",
        "Resize", "Upsample", "Pad", "Slice", "Tile", "CumSum", "DepthToSpace", "SpaceToDepth", "GridSample",
        "MaxPool", "AveragePool", "LpPool", "GlobalMaxPool", "GlobalAveragePool", "GlobalLpPool",
    }
)  # fmt: skip
#: Steps worked out value by value on inputs fitted to one another: an input's axes are the output's last ones.
_BROADCASTS = frozenset(
    {
        "Add", "Sub", "Mul", "Div", "Pow", "Mod", "Min", "Max", "Sum", "Mean", "PRelu", "And", "Or", "Xor", "Where",
        "Expand",
    }
)  # fmt: skip
#: Convolutions: what one makes is rows, channels, and then the picture.
_CONVOLUTIONS = frozenset({"Conv", "ConvTranspose", "QLinearConv", "ConvInteger"})
#: How many steps are looked at for a segmentation's classes before the search gives up.
_CLASS_AXIS_STEPS = 4096
#: What became of one axis of a tensor on its way to the output: the output's axis it is now (``None`` once it is
#: gone), and how it went. ``""`` is as it was; ``"picked"``, an argmax put class numbers in its place; ``"one"``,
#: it was dropped while it was one long; ``"folded"``, a reshape merged it into a longer axis that the output still
#: has; ``"lost"``, a reshape took it apart some other way.
_Fate = Tuple[Optional[int], str]
_Ways = List[Tuple[str, Tuple[_Fate, ...]]]


def _shape(value: Any) -> Optional[Tuple[int, ...]]:
    return tuple(int(dim) for dim in value.shape) if isinstance(value, np.ndarray) else None


def _sizes(shape: Sequence[int]) -> str:
    return " x ".join(str(int(dim)) for dim in shape) or "one value"


def can_hold_a_mask(shape: Optional[Sequence[int]]) -> bool:
    """Whether an output of this shape can be a segmentation's mask: it has two or three axes once a leading
    axis of one, the row axis, is dropped (a height and a width, with or without an axis of classes)."""
    if shape is None:
        return False
    dims = [int(dim) for dim in shape]
    if len(dims) >= 2 and dims[0] == 1:
        dims = dims[1:]
    return len(dims) in (2, 3)


def _picture_sizes(shapes: Dict[str, List[int]], hints: Dict[str, Any]) -> List[Tuple[int, int]]:
    """The height and width of each picture the model reads.

    A picture is an input of three axes after a row axis of one: channels and then the picture, or the picture and
    then channels. A hint's ``layout`` says which; without one, the end that holds 1, 3 or 4 values is the channels
    (both readings are kept when both ends do). An input the model is always given, or one read as columns, is no
    picture.
    """
    found: List[Tuple[int, int]] = []
    for name, shape in shapes.items():
        dims = [int(dim) for dim in shape]
        if len(dims) == 4 and dims[0] == 1:
            dims = dims[1:]
        hint = hints.get(name)
        said = hint if isinstance(hint, dict) else {}
        if len(dims) != 3 or "value" in said or said.get("zeros") or "features" in said:
            continue
        layout = str(said.get("layout") or "").lower()
        first, last = (dims[1], dims[2]), (dims[0], dims[1])
        if layout in ("nchw", "chw"):
            found.append(first)
        elif layout in ("nhwc", "hwc"):
            found.append(last)
        else:
            found.extend(size for size, channels in ((first, dims[0]), (last, dims[2])) if channels in (1, 3, 4))
    return found


def _gives_margins(node: Optional[Dict[str, Any]]) -> bool:
    """Whether a classical classifier's step gives raw margins, which a sigmoid makes a probability: a linear model or
    a support vector machine with no transform of its scores and no probabilities of its own. A step that gives
    probabilities already (a transform, a tree ensemble's votes, an SVM's own probabilities) does not, and one
    column of those is read as the number it is."""
    if node is None or str(node.get("op")) not in ("LinearClassifier", "SVMClassifier"):
        return False
    attrs = node.get("attrs") or {}
    transform = attrs.get("post_transform", "NONE")
    transform = transform.decode("utf-8", "replace") if isinstance(transform, bytes) else str(transform)
    has_probabilities = attrs.get("prob_a") is not None and np.asarray(attrs.get("prob_a")).size > 0
    return transform.upper() == "NONE" and not has_probabilities


def _picture_fit(shape: Sequence[int], pictures: Sequence[Tuple[int, int]]) -> int:
    """How an output's height and width stand to the pictures the model reads: 0 when they are a picture's own, 1
    when they are a whole fraction or a whole multiple of one's (a mask at a quarter of the size, or at twice it),
    and 2 otherwise. The output's height and width are its last two axes or, with classes last, the two before."""
    dims = [int(dim) for dim in shape]
    if len(dims) >= 2 and dims[0] == 1:
        dims = dims[1:]
    if len(dims) == 2:
        sizes = [(dims[0], dims[1])]
    elif len(dims) == 3:
        sizes = [(dims[1], dims[2]), (dims[0], dims[1])]
    else:
        return 2
    best = 2
    for height, width in sizes:
        for tall, wide in pictures:
            if (height, width) == (tall, wide):
                return 0
            if min(height, width, tall, wide) < 1:
                continue
            smaller = tall % height == 0 and wide % width == 0 and tall // height == wide // width
            larger = height % tall == 0 and width % wide == 0 and height // tall == width // wide
            if smaller or larger:
                best = 1
    return best


def _mask_output(
    candidates: Sequence[str], results: Dict[str, Any], pictures: Sequence[Tuple[int, int]] = ()
) -> str:
    """Which of a segmentation model's outputs its mask is read from, when the caller does not say.

    Of the outputs that can hold a mask (:func:`can_hold_a_mask`), those whose height and width are the picture's
    own that the model reads come first, then those at a whole fraction or multiple of it, then the rest
    (:func:`_picture_fit`; ``pictures`` are the model's): a score for each class of each of a hundred queries is
    not taken for the mask because it comes first. Among those, the first output of floats: a model that gives
    its scores and something more (torchvision's FCN and DeepLab give ``out`` and ``aux``) is read from the
    first. An output of whole numbers or of yes-or-no values only when no output of floats fits as well. A
    model with a single numeric output is read from it whatever its shape (:func:`segmentation_head` says what is
    wrong with one that is no mask), and several outputs none of which can hold a mask are refused.
    """
    numeric = [
        name for name in candidates if isinstance(results.get(name), np.ndarray) and results[name].dtype.kind in "fiub"
    ]
    able = [name for name in numeric if can_hold_a_mask(results[name].shape)]
    if able:
        # The first of the best: by the picture, and then floats before whole numbers.
        fits = {name: _picture_fit(results[name].shape, pictures) for name in able}
        return min(able, key=lambda name: (fits[name], results[name].dtype.kind != "f"))
    if len(numeric) == 1:
        return numeric[0]
    if not numeric:
        raise OnnxImportError("This model has no numeric output to read a mask from.")
    listed = ", ".join(f"'{name}' ({_sizes(results[name].shape)})" for name in numeric[:6])
    raise OnnxImportError(
        f"None of this model's outputs can hold a mask: {listed}. A mask is a score for each class at each pixel, "
        "or a class number for each pixel. Say which output to read with output=.",
        param="output",
    )


def _whole_axes(
    op: str, source: Sequence[int], made: Sequence[int], given: Optional[Sequence[int]]
) -> Optional[Dict[int, int]]:
    """``{axis of source: axis of made}`` for every axis a squeeze, an unsqueeze or a reshape leaves whole.

    ``given`` is the ``axes`` of a squeeze or an unsqueeze (``None`` when a squeeze has none: it drops every axis
    of one). A reshape leaves an axis whole when it is as long as before and as many values come before it. Its
    axes of one are told apart only where it leaves as many of them in the same place.
    """
    if op == "Squeeze":
        if given is None:
            gone = {axis for axis, size in enumerate(source) if size == 1}
        else:
            gone = {int(axis) % len(source) for axis in given} if len(source) else set()
        left = [axis for axis in range(len(source)) if axis not in gone]
        return dict(zip(left, range(len(made)))) if len(left) == len(made) else None
    if op == "Unsqueeze":
        if given is None or not len(made):
            return None
        added = {int(axis) % len(made) for axis in given}
        slots = [axis for axis in range(len(made)) if axis not in added]
        return dict(zip(range(len(source)), slots)) if len(slots) == len(source) else None
    if 0 in source or 0 in made:
        return None
    read = [int(np.prod(source[:axis], dtype=np.int64)) for axis in range(len(source))]
    write = [int(np.prod(made[:axis], dtype=np.int64)) for axis in range(len(made))]
    whole: Dict[int, int] = {}
    for axis, size in enumerate(source):
        if size == 1:
            continue
        same = [other for other, length in enumerate(made) if length == size and write[other] == read[axis]]
        if same:
            whole[axis] = same[0]
    for count in set(read):
        ones_read = [axis for axis, size in enumerate(source) if size == 1 and read[axis] == count]
        ones_made = [axis for axis, size in enumerate(made) if size == 1 and write[axis] == count]
        if len(ones_read) == len(ones_made):
            whole.update(zip(ones_read, ones_made))
    return whole


def _dropped(axis: int, source: Sequence[int], made: Sequence[int], fates: Sequence[_Fate]) -> str:
    """How an axis of ``source`` went that a squeeze or a reshape into ``made`` did not leave whole (see ``_Fate``).

    One long, it was dropped. Longer, it is ``"folded"`` when its values lie along one axis of ``made`` that is
    longer than it and that the output still has as it was (classes and rows of pixels sharing an axis), and
    ``"lost"`` otherwise: split over several axes, or folded into an axis that a later step took apart again.
    ``fates`` are those of ``made``'s axes.
    """
    if source[axis] == 1:
        return "one"
    if 0 in source or 0 in made:
        return "lost"
    step = int(np.prod(source[axis + 1 :], dtype=np.int64))  # how far apart its values lie
    span = step * int(source[axis])
    for other, size in enumerate(made):
        stride = int(np.prod(made[other + 1 :], dtype=np.int64))
        if stride <= step and span <= stride * size and step % stride == 0 and (stride * size) % span == 0:
            where, how = fates[other]
            return "folded" if size > source[axis] and where is not None and how == "" else "lost"
    return "lost"


def _branches(node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The bodies of an ``If``: ``{"then_branch": graph, "else_branch": graph}`` as the reader keeps them."""
    found: Dict[str, Dict[str, Any]] = {}
    for key, value in (node.get("attrs") or {}).items():
        if isinstance(value, dict) and isinstance(value.get("graph"), dict):
            found[str(key)] = value["graph"]
    return found


def _branch_taken(node: Dict[str, Any], values: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """``(the one step, its constants)`` of the branch an ``If`` ran for the row in ``values``, or ``None``.

    PyTorch writes ``squeeze(axis)`` of an axis whose size the file leaves open (the rows, as a rule) as a branch:
    a ``Squeeze`` when the axis is one, an ``Identity`` when it is not. For the one row a model is read on, the
    branch taken is a single step reading a tensor of the graph around it, and is followed like any other step.
    A branch of more than one step is not followed.
    """
    bodies = _branches(node)
    inputs = list(node.get("inputs") or [])
    said = values.get(inputs[0]) if inputs else None
    if set(bodies) != {"then_branch", "else_branch"} or not isinstance(said, np.ndarray) or said.size != 1:
        return None
    body = bodies["then_branch" if bool(said.reshape(-1)[0]) else "else_branch"]
    steps, outputs = list(body.get("nodes") or []), list(body.get("outputs") or [])
    if len(steps) != 1 or len(outputs) != 1 or list(steps[0].get("outputs") or []) != outputs:
        return None
    return steps[0], dict(body.get("constants") or {})


def _through_branches(node: Dict[str, Any]) -> Optional[str]:
    """The tensor both branches of an ``If`` hand on with its scores as they are, or ``None``.

    Each branch may only move values (a squeeze, an identity: the steps of ``_KEEPS_SCORES``) from one tensor of
    the graph around it, and both from the same one: whichever is taken, a probability stays a probability.
    """
    sources: Set[Optional[str]] = set()
    bodies = _branches(node)
    for body in bodies.values():
        made = {name: step for step in body.get("nodes") or [] for name in step.get("outputs") or [] if name}
        outputs = list(body.get("outputs") or [])
        name: Optional[str] = str(outputs[0]) if len(outputs) == 1 else None
        for _step in range(64):
            step = made.get(name) if name is not None else None
            if step is None:
                break
            reads = list(step.get("inputs") or [])
            whole = step.get("op") == "Cast" and int((step.get("attrs") or {}).get("to", 1)) not in _FLOAT_TYPES
            plain = str(step.get("domain") or "") in ("", "ai.onnx") and step.get("op") in _KEEPS_SCORES
            name = str(reads[0]) if plain and not whole and reads and reads[0] else None
        if name is not None and (name in made or name in (body.get("constants") or {})):
            name = None
        sources.add(name)
    return sources.pop() if len(bodies) == 2 and len(sources) == 1 else None


def _class_axis(
    nodes: Sequence[Dict[str, Any]],
    output: str,
    results: Optional[Dict[str, Any]],
    constants: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Where the classes of the tensor ``output`` lie, found by following its steps back to the one that makes them.

    Classes are made by a convolution (its channels: the second of its four axes) or by a product with a constant
    matrix (a linear layer: its last axis). Between that step and the output, a step either keeps every axis
    where it is (anything worked out value by value, a resize, a pad, a crop), moves axes in a way that can be
    read off it (a transpose, a squeeze, an unsqueeze, a reshape that leaves the axis whole), or puts the number
    of the best class in the place of the classes (an argmax). A step that picks one class out (a ``Gather`` of
    one place along an axis) drops that axis as a squeeze of one class does. Any other step ends the search.

    ``results`` holds every tensor's value for one row, which is where the shapes come from; ``constants`` the
    graph's constants. Returns ``None`` when the steps cannot be followed, else ``{"kind", "axis", "picture"}``:

    * ``"scores"``: ``axis`` of the output holds a score for each class;
    * ``"indices"``: the output holds class numbers, and ``axis`` is the axis of one an argmax left where the
      classes were, or ``None`` when it left none;
    * ``"one"``: one score for each pixel (the axis of classes was dropped while it was one long: a single
      channel, or one class picked out of several);
    * ``"folded"``: a reshape merged the classes into a longer axis that the output still has, so nothing reads
      them apart.

    A reshape that takes the axis of classes apart some other way (a pixel shuffle written as reshapes, a flatten
    that is undone later) proves nothing of where the classes are: the search then has no answer, like one that
    met a step it does not follow.

    ``picture`` is the output's axes that hold the height and the width (``None`` for one that is gone), when a
    convolution says which they are.
    """
    if not nodes or not output or results is None:
        return None
    values: Dict[str, Any] = results
    known = constants or {}
    producers = {name: node for node in nodes for name in node.get("outputs", []) if name}

    def shape_of(name: str) -> Optional[Tuple[int, ...]]:
        found = _shape(values.get(name))
        return found if found is not None else _shape(known.get(name))

    def numbers(name: str, local: Dict[str, Any]) -> Optional[List[int]]:
        value = local[name] if name in local else known[name] if name in known else values.get(name)
        return None if value is None else [int(item) for item in np.asarray(value).reshape(-1)]

    def verdict(
        fates: Tuple[_Fate, ...], made: int, shape: Sequence[int], picture: Sequence[int] = ()
    ) -> Optional[Dict[str, Any]]:
        where, how = fates[made]
        if how == "picked":
            kind = "indices"
        elif where is not None:
            kind = "scores"
        elif how == "one" or shape[made] == 1:
            kind = "one"  # by its size when it was dropped, not by how many channels the step made
        elif how == "folded":
            kind = "folded"
        else:
            return None  # taken apart by a reshape: where the classes went cannot be read off it
        return {
            "kind": kind,
            "axis": where,
            "picture": [fates[axis][0] for axis in picture] or None,
            "classes": int(shape[made]),  # how many the step made: more than one behind "one" is one picked out
        }

    def look(name: str, fates: Tuple[_Fate, ...]) -> Tuple[Optional[Dict[str, Any]], _Ways, Optional[_Fate]]:
        """One step back from ``name``: ``(what is found, the tensors to look at next, the place an argmax took)``.

        ``fates`` says, for each axis of ``name``, which axis of the output it became (``None`` when it is gone)
        and how it went (see ``_Fate``).
        """
        nowhere: _Ways = []
        node = producers.get(name)
        shape = shape_of(name)
        if node is None or shape is None or len(shape) != len(fates):
            return None, nowhere, None
        first = node["outputs"][0] == name
        local: Dict[str, Any] = {}
        if str(node["op"]) == "If" and first:
            # The branch run for this row is the step that made the tensor (see _branch_taken).
            taken = _branch_taken(node, values)
            if taken is None:
                return None, nowhere, None
            node, local = taken
        op, inputs, attrs = str(node["op"]), list(node.get("inputs") or []), node.get("attrs") or {}
        domain = str(node.get("domain") or "")
        rank = len(shape)
        if domain == _KAIMEI:
            return (verdict(fates, rank - 1, shape) if op == "Linear" and first and rank else None), nowhere, None
        if domain not in ("", "ai.onnx") or not inputs or not inputs[0]:
            return None, nowhere, None
        if op in _CONVOLUTIONS:
            # Rows, channels, height, width: a convolution over two axes makes a picture's classes.
            return (verdict(fates, 1, shape, (2, 3)) if first and rank == 4 else None), nowhere, None
        if op == "Gemm":
            return (verdict(fates, rank - 1, shape) if first and rank else None), nowhere, None
        if op == "MatMul":
            weights = _shape(known.get(inputs[1])) if len(inputs) == 2 else None
            if first and rank and inputs[0] not in known and weights is not None and len(weights) >= 2:
                return verdict(fates, rank - 1, shape), nowhere, None  # x · W: one value for each column of W
            return None, nowhere, None
        source = shape_of(inputs[0])
        if op == "Split" or (op in _KEEPS_AXES and first):
            return None, ([(inputs[0], fates)] if source is not None and len(source) == rank else nowhere), None
        if not first:
            return None, nowhere, None
        if op == "Concat" or op in _BROADCASTS:
            reads = inputs[1:3] if op == "Where" else inputs[:1] if op == "Expand" else inputs
            fitted = [(item, shape_of(item)) for item in reads if item and item not in known]
            # The input that has the output's shape already is the one the others are fitted to.
            fitted.sort(key=lambda pair: pair[1] != shape)
            return (
                None,
                [
                    (item, fates[rank - len(held) :])
                    for item, held in fitted
                    if held is not None and len(held) <= rank and (op != "Concat" or len(held) == rank)
                ],
                None,
            )
        if source is None:
            return None, nowhere, None
        if op == "Transpose":
            perm = [int(axis) for axis in (attrs.get("perm") or reversed(range(rank)))]
            if len(source) != rank or sorted(perm) != list(range(rank)):
                return None, nowhere, None
            moved: List[_Fate] = [(None, "lost")] * rank
            for made, read in enumerate(perm):
                moved[read] = fates[made]
            return None, [(inputs[0], tuple(moved))], None
        if op in ("ArgMax", "ArgMin"):
            keeps = bool(int(attrs.get("keepdims", 1)))
            if not len(source) or len(source) != rank + (0 if keeps else 1):
                return None, nowhere, None
            axis = int(attrs.get("axis", 0)) % len(source)
            place: _Fate = (fates[axis][0], "picked") if keeps else (None, "picked")
            return None, [(inputs[0], (*fates[:axis], place, *fates[axis + (1 if keeps else 0) :]))], place
        if op == "Gather":
            # x[:, 1]: one class picked out leaves its axis out, as a squeeze of one class would. A list of
            # them picked keeps the axis.
            picked = shape_of(inputs[1]) if len(inputs) > 1 and inputs[1] else None
            if picked is None or not len(source) or inputs[0] in known:
                return None, nowhere, None
            axis = int(attrs.get("axis", 0)) % len(source)
            if len(picked) == 0 and len(source) == rank + 1:
                one: _Fate = (None, "one")
                return None, [(inputs[0], (*fates[:axis], one, *fates[axis:]))], None
            if len(picked) == 1 and len(source) == rank:
                return None, [(inputs[0], fates)], None
            return None, nowhere, None
        if op in ("Squeeze", "Unsqueeze", "Reshape", "Flatten"):
            given: Optional[List[int]] = None
            if op in ("Squeeze", "Unsqueeze"):
                if attrs.get("axes") is not None:
                    given = [int(axis) for axis in attrs["axes"]]
                elif len(inputs) > 1 and inputs[1]:
                    given = numbers(inputs[1], local)
                    if given is None:
                        return None, nowhere, None
            whole = _whole_axes(op, source, shape, given)
            if whole is None:
                return None, nowhere, None
            kept: Tuple[_Fate, ...] = tuple(
                fates[whole[axis]] if axis in whole else (None, _dropped(axis, source, shape, fates))
                for axis in range(len(source))
            )
            return None, [(inputs[0], kept)], None
        return None, nowhere, None

    start = shape_of(output)
    if start is None:
        return None
    # Depth first, the first input of a step before its others, with a list of what is still to look at: a model's
    # steps can follow one another for longer than Python lets a function call itself.
    as_it_is: Tuple[_Fate, ...] = tuple((axis, "") for axis in range(len(start)))
    waiting: _Ways = [(output, as_it_is)]
    seen: Set[Tuple[str, Tuple[_Fate, ...]]] = set()
    numbers_anyway: Optional[Dict[str, Any]] = None
    for _step in range(_CLASS_AXIS_STEPS):
        if not waiting:
            break
        here = waiting.pop()
        if here in seen:
            continue
        seen.add(here)
        found, further, place = look(*here)
        if found is not None:
            return found
        if place is not None and numbers_anyway is None:
            # Whatever made the scores, what an argmax leaves is the number of the best of them.
            numbers_anyway = {"kind": "indices", "axis": place[0], "picture": None}
        waiting.extend(reversed(further))
    return numbers_anyway


def _index_axes(shape: Sequence[int], kept: Optional[int]) -> Optional[Tuple[int, int]]:
    """The two axes a picture of class numbers is read from, as the API reads one (``_mask_reading`` in its
    ``nd/build.py``): the last two after row axes of one, or the middle two of ``[1, height, width, 1]``, where an
    argmax over the last axis left that axis in place. ``kept`` is the axis an argmax left, when one is known."""
    rank = len(shape)
    if rank == 4 and shape[0] == shape[3] == 1 and (shape[1] != 1 or kept == 3):
        return (1, 2)
    if rank >= 2 and all(size == 1 for size in shape[: rank - 2]):
        return (rank - 2, rank - 1)
    return None


def _not_a_mask(shape: Sequence[int], why: str) -> OnnxImportError:
    return OnnxImportError(
        f"This segmentation output ({_sizes(shape)}) {why}. Kaimei reads a mask as class scores that come first "
        "(chw) or last (hwc), or as a class number for each pixel (indices), after any row axis of one. Read "
        "another output with output=, or end the model with a step that gives one of those."
    )


def _mask_layout(shape: Sequence[int], whole: bool, named: Sequence[str], found: Optional[Dict[str, Any]]) -> str:
    """The layout of a segmentation output nobody gave one for. See :func:`segmentation_head` for the order.

    ``whole`` says the output's type holds whole numbers or flags. A height and a width of those is a class number
    for each pixel; of floats, one score for each pixel (whether floats that are all whole numbers are scores is
    for :func:`_mask_head` to ask, which sees their values).
    """
    rank = len(shape)
    lead = rank - 3
    rows_of_one = rank >= 3 and all(size == 1 for size in shape[:lead])
    if found is not None:
        kind, axis, picture = found["kind"], found["axis"], found["picture"]
        if kind == "folded":
            raise _not_a_mask(shape, "has its classes folded into another axis by a reshape")
        if kind == "scores":
            if rows_of_one and axis - lead in (0, 2):
                return "chw" if axis == lead else "hwc"
            if rank < 3:
                raise _not_a_mask(shape, "holds a score for each class and no picture")
            where = f"axis {axis}, which is neither first nor last after a row axis of one"
            raise _not_a_mask(shape, f"holds its class scores along {where}")
        if kind == "indices":
            if picture is not None:
                truth = sorted(picture) if None not in picture else None
            else:
                rest = [index for index in range(rank) if index != axis]
                while len(rest) > 2 and shape[rest[0]] == 1:
                    rest = rest[1:]
                truth = rest if len(rest) == 2 else None
            read = _index_axes(shape, axis)
            if truth is not None and read is not None and list(read) == truth:
                return "indices"
            raise _not_a_mask(shape, "holds class numbers, with its height and width not where a mask keeps them")
        # One score for each pixel: the single class channel is gone, and what is left is the picture. With two
        # axes it is read as one channel of scores all the same (``chw`` with no axis for its one class), so that
        # it is cut where the same scores are cut behind an axis of one, and never rounded like class numbers.
        first, last = rows_of_one and shape[lead] == 1, rows_of_one and shape[-1] == 1
        if picture is not None and None not in picture:
            if rank == 2:
                return "chw"
            if first and sorted(picture) == [rank - 2, rank - 1]:
                return "chw"  # the row axis of one stands where the channel was
            if last and sorted(picture) == [rank - 3, rank - 2]:
                return "hwc"
        elif picture is None and (rank == 2 or first != last):
            return "chw" if rank == 2 or first else "hwc"
        misplaced = "with its height and width not where a mask keeps them"
        raise _not_a_mask(shape, f"holds one score for each pixel, {misplaced}")
    if rank == 2:
        # A type of whole numbers holds a class number for each pixel; floats, one score for each pixel.
        return "indices" if whole else "chw"
    if whole and rank == 3 and shape[0] == 1:
        return "indices"  # a number for each pixel, after the row axis
    if rows_of_one and not (whole and 1 in shape[lead:]):
        # Whole numbers behind an axis of one may be class numbers or one channel of scores: those are not guessed.
        dims = list(shape[lead:])
        for width in [len(named), 1] if len(named) == 2 else [len(named)] if named else []:
            fits = [index for index, size in enumerate(dims) if size == width]
            if len(fits) == 1 and fits[0] != 1:
                return "chw" if fits[0] == 0 else "hwc"  # the one axis as long as the labels
            if fits:
                break
        if dims[1] == dims[2] != dims[0]:
            return "chw"  # the one axis whose size differs from the two equal others
        if dims[0] == dims[1] != dims[2]:
            return "hwc"
    said = f"its {len(named)} labels fit no one axis, " if named else ""
    raise OnnxImportError(
        f"Kaimei can't tell which axis of this segmentation output ({_sizes(shape)}) holds its classes: no step it "
        f"can follow makes them, {said}and its shape does not say. Pass mask_layout='chw' (class scores, then "
        "height and width), 'hwc' (height, width, then class scores) or 'indices' (a class number for each pixel). "
        "In a structure, that is head.layout.",
        param="mask_layout",
    )


#: Steps that leave a probability a probability and a raw score a raw score: they move, pick or resample values.
_KEEPS_SCORES = frozenset(
    {
        "Identity", "Cast", "CastLike", "Dropout", "Resize", "Upsample", "Transpose", "Squeeze", "Unsqueeze",
        "Reshape", "Flatten", "Slice", "Gather", "Split", "Pad", "DepthToSpace", "SpaceToDepth",
        "MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool",
    }
)  # fmt: skip
#: Steps whose values are probabilities.
_MAKES_PROBABILITIES = frozenset({"Sigmoid", "HardSigmoid", "Softmax"})
#: Steps whose values are raw scores: a probability only once they are squashed.
_MAKES_RAW_SCORES = frozenset(
    {
        "Conv", "ConvTranspose", "QLinearConv", "ConvInteger", "MatMul", "Gemm", "BatchNormalization",
        "InstanceNormalization", "GroupNormalization", "LayerNormalization",
    }
)  # fmt: skip
#: ONNX's element types that hold floats (a cast to any other type no longer holds a score).
_FLOAT_TYPES = (1, 10, 11, 16)
#: A probability of one half, as a probability and as the logarithm of one.
_HALF = 0.5
_LOG_HALF = float(np.log(0.5))


def _clip_bound(
    node: Dict[str, Any], place: int, name: str, known: Dict[str, Any], open_end: float
) -> Optional[float]:
    """One end of a ``Clip``: its attribute (``min``, ``max``) or its constant input, ``open_end`` when it has
    neither, and ``None`` when the end is not a constant."""
    said = (node.get("attrs") or {}).get(name)
    inputs = list(node.get("inputs") or [])
    if len(inputs) > place and inputs[place]:
        if inputs[place] not in known:
            return None
        said = known[inputs[place]]
    if said is None:
        return open_end
    value = np.asarray(said)
    return float(value.reshape(-1)[0]) if value.size == 1 and value.dtype.kind in "fiu" else None


def _score_cut(
    nodes: Sequence[Dict[str, Any]], output: str, constants: Optional[Dict[str, Any]] = None
) -> Optional[Tuple[float, Optional[str]]]:
    """Where one score for each pixel is cut to be the model's own probability of one half, read off the graph.

    The output is followed back through the steps that leave a score what it is (a resize, a transpose, a squeeze,
    a slice, a cast to floats, a pooling). A sigmoid or a softmax on the way means the scores are probabilities,
    cut at one half (a log-softmax: at its logarithm). A convolution, a linear layer, a normalisation or a sum of
    those means they are raw scores, which a sigmoid would make a probability of one half at nought.

    Between those and the output, three steps are followed where the cut behind them is certain. A product or a
    quotient with one constant above nought moves the cut with it (a sigmoid times 255 is cut at 127.5). A ``Relu``
    and a ``Clip`` between constants leave the cut where it is as long as it lies inside what they keep. Where it
    does not (a ``Relu`` over raw scores keeps nothing below nought, so it holds no sign of where their probability
    was one half), no reading of the values could find it either, and the model is refused, naming ``threshold``.

    Returns ``(threshold, what the scores are)``, the second ``"probability"``, ``"raw"`` or ``None`` (a
    probability that was scaled is neither), or ``None`` when a step that says nothing comes first (the model's
    own values then say: see :func:`_cut_by_values`).
    """
    known = constants or {}
    producers = {name: node for node in nodes for name in node.get("outputs", []) if name}
    found: Set[float] = set()
    waiting: List[Tuple[str, bool]] = [(output, False)]  # a tensor, and whether a sum takes it in
    seen: Set[str] = set()
    # The steps between the scores and the output that scale them (by the second number) or keep only what lies
    # between two numbers, the last of them first.
    after: List[Tuple[str, float, float]] = []
    for _step in range(_CLASS_AXIS_STEPS):
        if not waiting:
            break
        name, summed = waiting.pop()
        if name in seen:
            continue
        seen.add(name)
        node = producers.get(name)
        if node is None:
            return None  # one of the model's own inputs: nothing says what its values are
        op, inputs = str(node["op"]), [item for item in node.get("inputs") or []]
        domain = str(node.get("domain") or "")
        if domain == _KAIMEI and op == "Linear":
            found.add(0.0)
        elif domain not in ("", "ai.onnx") or not inputs or not inputs[0]:
            return None
        elif op in _MAKES_RAW_SCORES:
            found.add(0.0)
        elif op in _MAKES_PROBABILITIES or op == "LogSoftmax":
            if summed:
                return None  # a sum of probabilities is neither a probability nor a raw score
            found.add(_LOG_HALF if op == "LogSoftmax" else _HALF)
        elif op in ("Add", "Sum"):
            # A sum of raw scores (a bias, a skip connection) is a raw score, if everything it takes in is one.
            reads = [item for item in inputs if item and item not in known]
            if not reads:
                return None
            if after and len(reads) < len([item for item in inputs if item]):
                # A constant added before a clip or a scale (a hard sigmoid written out: relu6(x + 3) / 6) moves
                # where the cut lies behind them. The model's own values say what the scores are.
                return None
            waiting.extend((item, True) for item in reads)
        elif op in ("Relu", "Clip", "Mul", "Div"):
            if summed:
                return None  # inside a sum, what one part was scaled or cut to says nothing certain of the whole
            scores = inputs[0]
            if op == "Relu":
                after.append(("clip", 0.0, float("inf")))
            elif op == "Clip":
                low = _clip_bound(node, 1, "min", known, float("-inf"))
                high = _clip_bound(node, 2, "max", known, float("inf"))
                if low is None or high is None or np.isnan(low) or np.isnan(high):
                    return None
                after.append(("clip", low, high))
            else:
                # By one constant above nought: the scores keep their order, and the cut is scaled with them.
                if len(inputs) != 2 or not inputs[1]:
                    return None
                constant = inputs[1]
                if op == "Mul" and scores in known:
                    scores, constant = constant, scores
                if scores in known or constant not in known:
                    return None
                factor = np.asarray(known[constant])
                if factor.size != 1 or factor.dtype.kind not in "fiu":
                    return None
                by = float(factor.reshape(-1)[0])
                if not np.isfinite(by) or by <= 0.0:
                    return None
                after.append(("scale", by if op == "Mul" else 1.0 / by, 0.0))
            waiting.append((scores, False))
        elif op in _KEEPS_SCORES:
            if op == "Cast" and int((node.get("attrs") or {}).get("to", 1)) not in _FLOAT_TYPES:
                return None
            waiting.append((inputs[0], summed))
        elif op == "If":
            # How PyTorch writes squeeze() of an axis whose size the file leaves open: followed when both of its
            # branches only move the scores.
            behind = _through_branches(node)
            if behind is None:
                return None
            waiting.append((behind, summed))
        else:
            return None
    if waiting or len(found) != 1:
        return None
    threshold = found.pop()
    kind: Optional[str] = "probability" if threshold == _HALF else "raw"
    made = "probabilities" if threshold == _HALF else "raw scores" if threshold == 0.0 else "log-probabilities"
    for what, low, high in reversed(after):  # from the scores on to the output
        if what == "scale":
            threshold *= low
            if low != 1.0 and kind == "probability":
                kind = None
        elif not low < threshold <= high:
            if high == float("inf"):
                kept = f"nothing below {low:g}"
            elif low == float("-inf"):
                kept = f"nothing above {high:g}"
            else:
                kept = f"only what lies from {low:g} to {high:g}"
            raise OnnxImportError(
                f"Nothing says where to cut this model's one score for each pixel. Its {made} pass through a step "
                f"that keeps {kept}, and the model's own probability is one half at {threshold:g}: what the step "
                "leaves does not show where that was. Pass threshold= to say where to cut, in the model's own "
                "numbers.",
                param="threshold",
            )
    return threshold + 0.0, kind


def _cut_by_values(values: Sequence[Any]) -> Tuple[float, str]:
    """Where one score for each pixel is cut when no step says what the scores are: read off the model's own
    values, on every row there is.

    Scores that go below nought are raw scores, cut at nought. Scores that all lie from 0 to 1 read as
    probabilities, cut at one half. Scores that never go below nought and do go above 1 are neither (raw scores
    that passed a step which keeps only what is above nought, or probabilities that were scaled): nothing says
    where to cut them, and the model is refused, naming ``threshold``.
    """
    lowest, highest, rows = float("inf"), float("-inf"), 0
    for value in values:
        array = np.asarray(value, dtype=float)
        array = array[np.isfinite(array)]
        if array.size:
            lowest, highest, rows = min(lowest, float(array.min())), max(highest, float(array.max())), rows + 1
    if lowest < 0.0:
        return 0.0, "raw"
    if highest <= 1.0:
        return _HALF, "probability"
    raise OnnxImportError(
        "Nothing says where to cut this model's one score for each pixel. No step Kaimei can follow says what "
        f"the scores are, and on the {rows} row{'s' if rows != 1 else ''} tried they run from {lowest:g} to "
        f"{highest:g}: never below 0, as raw scores cut at 0 go, and above 1, where no probability is. Pass "
        "threshold= to say where to cut, in the model's own numbers.",
        param="threshold",
    )


def _all_whole(value: Any) -> bool:
    """Whether every value is a whole number."""
    array = np.asarray(value, dtype=float)
    return bool(np.all(np.isfinite(array)) and np.all(array == np.rint(array)))


def _numbers_or_scores(shape: Sequence[int], rows: int) -> OnnxImportError:
    """The refusal of a height and a width of floats, all of them whole numbers, that no step explains."""
    return OnnxImportError(
        f"Kaimei can't tell what this segmentation output ({_sizes(shape)}) holds. It is a height and a width of "
        f"floats that no step Kaimei can follow explains, and every value on the {rows} row"
        f"{'s' if rows != 1 else ''} tried is a whole number: class numbers are, and scores can happen to be. Pass "
        "mask_layout='indices' (a class number for each pixel) or mask_layout='chw' (one score for each pixel). In "
        "a structure, that is head.layout.",
        param="mask_layout",
    )


def _mask_head(
    value: np.ndarray,
    labels: Optional[Sequence[str]],
    layout: Optional[str],
    nodes: Sequence[Dict[str, Any]],
    output: str,
    results: Optional[Dict[str, Any]],
    constants: Optional[Dict[str, Any]],
    threshold: Optional[float],
) -> Tuple[Dict[str, Any], bool, bool]:
    """``(head, cut by the values, whole floats)``: :func:`segmentation_head` before the model's values are asked.

    The second says that no step says where one score for each pixel is cut, so the head has no ``threshold``
    yet: the values say (:func:`_cut_by_values`), every check row of them. The third says the output is a height
    and a width of floats that no step explains and that are all whole numbers on this row: one score for each
    pixel if any other row holds a number that is not whole, and otherwise not told from class numbers
    (:func:`_numbers_or_scores`). The importer asks every check row; :func:`segmentation_head` has only this one.
    """
    value = np.asarray(value)
    shape = tuple(int(dim) for dim in value.shape)
    named = [str(label) for label in labels] if labels is not None and len(labels) else []
    told = layout is not None
    if told and layout not in ("chw", "hwc", "indices"):
        raise OnnxImportError("mask_layout must be chw, hwc or indices.", param="mask_layout")
    if threshold is not None:
        number = isinstance(threshold, (int, float, np.integer, np.floating))
        try:
            finite = number and not isinstance(threshold, (bool, np.bool_)) and bool(np.isfinite(float(threshold)))
        except OverflowError:  # a whole number no float holds
            finite = False
        if not finite:
            raise OnnxImportError(
                "threshold must be a number: where one score for each pixel is cut, in the model's own numbers.",
                param="threshold",
            )
    whole_floats = False
    layout_found: Optional[Dict[str, Any]] = None
    if layout is None:
        whole = value.dtype.kind in "iub"
        found = _class_axis(nodes, output, results, constants)
        layout_found = found if found is not None and found.get("kind") == "one" else None
        layout = _mask_layout(shape, whole, named, found)
        whole_floats = found is None and not whole and len(shape) == 2 and _all_whole(value)
    rank = 2 if layout == "indices" else 3
    # An argmax over the last axis that keeps it leaves [1, height, width, 1].
    kept_last = layout == "indices" and len(shape) == 4 and shape[0] == shape[-1] == 1
    # One score for each pixel and no axis for its one class: a height and a width, read as one channel.
    no_class_axis = layout != "indices" and len(shape) == 2
    if not kept_last and not no_class_axis and (len(shape) < rank or any(size != 1 for size in shape[:-rank])):
        if told:
            raise OnnxImportError(f"Output {shape} does not fit mask_layout={layout!r}.", param="mask_layout")
        raise _not_a_mask(shape, "has more rows or axes than a mask")
    width = 0 if layout == "indices" else 1 if no_class_axis else shape[-3] if layout == "chw" else shape[-1]
    if layout != "indices" and named:
        if len(named) != width and (width, len(named)) != (1, 2):
            raise OnnxImportError(f"The mask has {width} class channels and {len(named)} labels.", param="labels")
    head: Dict[str, Any] = {"kind": "segmentation", "layout": layout}
    if width != 1:
        if threshold is not None:
            holds = "a class number for each pixel" if layout == "indices" else f"scores of {width} classes"
            raise OnnxImportError(
                f"threshold says where one score for each pixel is cut, and this output ({_sizes(shape)}) holds "
                f"{holds}.",
                param="threshold",
            )
        return head, False, False
    if threshold is not None:
        head["threshold"] = float(threshold)
        return head, False, False  # one score for each pixel, as the caller says: nothing is left to tell apart
    cut = _score_cut(nodes, output, constants)
    if cut is not None and cut[1] == "raw" and layout_found is not None and int(layout_found.get("classes") or 1) > 1:
        raise OnnxImportError(
            f"Nothing says where to cut this model's one score for each pixel. It is the raw score of one class "
            f"picked out of {int(layout_found['classes'])}: a pixel is that class where its score beats the others', "
            "which no single cut of this one score says. Read every class's scores (leave the picking out of the "
            "model, or name that output with output=), or pass threshold= to say where to cut, in the model's own "
            "numbers.",
            param="threshold",
        )
    if cut is None:
        return head, True, whole_floats
    head["threshold"] = cut[0]
    if cut[1] is not None:
        head["score_kind"] = cut[1]
    return head, False, whole_floats


def segmentation_head(
    value: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    layout: Optional[str] = None,
    nodes: Sequence[Dict[str, Any]] = (),
    output: str = "",
    results: Optional[Dict[str, Any]] = None,
    constants: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """How a segmentation's output ``value`` holds its mask: ``{"kind": "segmentation", "layout": ...}``.

    The layout is ``chw`` (class scores, then height and width), ``hwc`` (height, width, then class scores) or
    ``indices`` (a class number for each pixel), after any row axis of one. It is, in this order:

    1. the caller's (``layout``: the client's ``mask_layout``);
    2. what the graph says, by following the scores back to the step that makes the classes
       (:func:`_class_axis`: ``nodes`` are the graph's steps, ``results`` every tensor's value for one row);
    3. the one axis that is as long as ``labels`` (for two labels, the one axis of one: a single score);
    4. the one axis, of the three after any row axis, whose size differs from two equal others.

    An output none of them settles is refused with how to say it, rather than read along an axis that may not
    be its classes. A height and a width with no other axis is class numbers when its type holds whole numbers,
    and one score for each pixel when it holds floats that are not all whole. Floats that are all whole numbers,
    which no step explains, may be either: they are refused, naming ``mask_layout``.

    A mask of one score for each pixel (one class channel, or a height and a width with no axis of classes, which
    is ``chw`` all the same) also has a ``threshold``: a pixel is the foreground when its score is that or more.
    It is the caller's ``threshold``, in the model's own numbers; else where the model's own probability is one
    half, by the graph (:func:`_score_cut`: 0.5 after a sigmoid, 0 for raw scores) or, when no step says, by
    ``value`` itself (:func:`_cut_by_values`), which refuses scores that are neither. ``score_kind``
    (``probability`` or ``raw``) says which the scores were taken for, when the caller did not give the threshold
    and they are one of the two.
    """
    head, by_values, whole_floats = _mask_head(value, labels, layout, nodes, output, results, constants, threshold)
    if whole_floats:
        raise _numbers_or_scores(np.asarray(value).shape, 1)
    if by_values:
        head["threshold"], head["score_kind"] = _cut_by_values([value])
    return head


def _is_layer(node: Dict[str, Any]) -> bool:
    return node["op"] in (
        "Conv",
        "ConvTranspose",
        "Gemm",
        "Linear",
        "LSTM",
        "GRU",
        "RNN",
        "BatchNormalization",
        "LayerNormalization",
    )


def _all_free(nodes: Sequence[Dict[str, Any]]) -> Set[str]:
    found: Set[str] = set()
    for node in nodes:
        found.update(_free_names(node["attrs"]))
    return found


def _rename_bodies(attrs: Dict[str, Any], rename: Dict[str, str]) -> None:
    """Nested graphs read outer tensors by name: follow the renames into them."""
    for value in attrs.values():
        bodies = []
        if isinstance(value, dict) and isinstance(value.get("graph"), dict):
            bodies = [value["graph"]]
        elif isinstance(value, dict) and isinstance(value.get("graphs"), list):
            bodies = value["graphs"]
        for body in bodies:
            local = {name for node in body.get("nodes", []) for name in node["outputs"] if name}
            local |= {str(item["name"]) for item in body.get("inputs", [])}
            local |= set(body.get("constants", {}))
            for node in body.get("nodes", []):
                node["inputs"] = [
                    rename.get(item, item) if item and item not in local else item for item in node["inputs"]
                ]
                _rename_bodies(node.get("attrs", {}), {k: v for k, v in rename.items() if k not in local})
            body["outputs"] = [
                rename.get(item, item) if item not in local else item for item in body.get("outputs", [])
            ]
            body.pop("values", None)


def _check_rows(
    model: Any,
    bound: Sequence[Dict[str, Any]],
    rename: Dict[str, str],
    shapes: Dict[str, List[int]],
    axes: Dict[str, Optional[int]],
    feeds: Dict[str, np.ndarray],
    chosen: str,
    link: str,
    vocab: Dict[str, int],
    task: str = "",
    decoder: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A few rows and the model's own outputs for them, so the API can check that its reading reproduces them.

    An output is written the way the model keeps it (a picture channels first stays channels first; the
    API knows how it shows one). A language model's is what it says comes next: its scores at the last
    position, which is the last real token of a row that has no padding. An encoder-decoder's is what its decoder
    writes next: its scores at the last token the feature it writes into holds (``decoder`` names that feature, as
    the structure's ``decoder`` does; without one it is the feature called ``decoder``, as on the API). A feature
    the importer marked generated holds the token the decoder starts from and nothing more, which is what a
    prediction gives it, so the check is of the first token written.
    """
    original = {new: old for old, new in rename.items()}
    writes = str((decoder or {}).get("feature") or "decoder")
    started: Dict[str, int] = {}
    if decoder is not None and decoder.get("start_token") is not None:
        started[writes] = int(decoder["start_token"])
    rows: Dict[str, List[List[float]]] = {}
    outputs: List[List[float]] = []
    for index in range(CHECK_ROWS):
        if index == 0:
            row_feeds = dict(feeds)
        else:
            row_feeds = _feeds_for(model, shapes, index, vocab)
            for entry in bound:
                source = entry.get("from") or {}
                if "value" in source or source.get("zeros"):
                    row_feeds[original[entry["name"]]] = feeds[original[entry["name"]]]
        for entry in bound:
            source = entry.get("from") or {}
            if source.get("part") == "ids" and source.get("feature") in started:
                ids = np.array(row_feeds[original[entry["name"]]])
                ids.reshape(-1)[0] = started[source["feature"]]
                row_feeds[original[entry["name"]]] = ids
        lengths = _settle_text(row_feeds, bound, original, keep_mask=index == 0, counts={key: 1 for key in started})
        value = np.asarray(run_reference(model, row_feeds, [chosen])[chosen], dtype=float)
        if task == "sequence_to_sequence" and value.ndim >= 2:
            # What the decoder writes next is said at the last token it has written, where the API reads it.
            table = value.reshape(-1, value.shape[-1])
            value = table[min(max(lengths.get(writes, 1) - 1, 0), table.shape[0] - 1)]
        elif task == "text_generation" and value.ndim >= 2:
            # What comes next is said at the last token the row has, not at the last position there is room for.
            table = value.reshape(-1, value.shape[-1])
            last = (next(iter(lengths.values())) if lengths else table.shape[0]) - 1
            value = table[min(max(last, 0), table.shape[0] - 1)]
        value = value.reshape(-1)
        if link == "softmax":
            shifted = np.exp(value - value.max())
            value = shifted / shifted.sum()
        elif link == "sigmoid":
            value = 1.0 / (1.0 + np.exp(-value))
        outputs.append([float(item) for item in value])
        for entry in bound:
            source = entry.get("from") or {}
            tensor = np.asarray(row_feeds[original[entry["name"]]])
            if "features" in source:
                flat = tensor.reshape(-1)
                for column, feature in enumerate(source["features"]):
                    rows.setdefault(feature, []).append([float(flat[column])])
            elif "feature" in source:
                feature = source["feature"]
                if source.get("part") not in (None, "ids"):
                    continue
                flat = _to_feature_order(tensor, entry, source)
                if feature in lengths:
                    flat[lengths[feature] :] = -1.0  # no token: the API pads the ids, and masks them, itself
                if len(rows.get(feature, [])) <= index:
                    rows.setdefault(feature, []).append([float(item) for item in flat])
    return {"inputs": rows, "outputs": outputs, "tolerance": 1e-4}


def _settle_text(
    row_feeds: Dict[str, np.ndarray],
    bound: Sequence[Dict[str, Any]],
    original: Dict[str, str],
    keep_mask: bool,
    counts: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Make what a model reads off a text (its mask, its positions, its length) what the API will read off it.

    The API is given a text's token ids and works the rest out: a mask of ones over the tokens there are,
    zeros as segment ids, positions counted from nought, the number of tokens. A check row whose mask was
    drawn at random, or kept apart from its ids, would hold the model to inputs it is never given. The
    sample's own mask is kept when it is ones and then zeros (a text shorter than the model reads): the
    row then has that many tokens. ``counts`` gives the token count of the features whose count is known (what a
    decoder has written before its first token is one token). Returns each text feature's token count.
    """
    by_feature: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in bound:
        source = entry.get("from") or {}
        if "feature" in source and source.get("part"):
            by_feature.setdefault(str(source["feature"]), {})[str(source["part"])] = entry
    lengths: Dict[str, int] = {}
    for feature, parts in by_feature.items():
        ids_entry = parts.get("ids")
        if ids_entry is None:
            continue
        ids_name = original[ids_entry["name"]]
        ids = np.asarray(row_feeds[ids_name])
        total = int(ids.size)
        length = min(int((counts or {}).get(feature, total)), total)
        mask_entry = parts.get("mask")
        if keep_mask and mask_entry is not None and feature not in (counts or {}):
            given = np.asarray(row_feeds[original[mask_entry["name"]]]).reshape(-1) != 0
            count = int(given.sum())
            if 0 < count < total and given.size == total and bool(given[:count].all()):
                length = count
        if length < total:
            padded = ids.reshape(-1).copy()
            padded[length:] = int((ids_entry.get("from") or {}).get("pad_id") or 0)
            row_feeds[ids_name] = padded.reshape(ids.shape)
        for part, entry in parts.items():
            if part == "ids":
                continue
            name = original[entry["name"]]
            held = np.asarray(row_feeds[name])
            if part == "mask":
                derived = (np.arange(total) < length).astype(float)
            elif part == "type_ids":
                derived = np.zeros(total)
            elif part == "positions":
                derived = np.arange(total, dtype=float)
            else:
                derived = np.array([float(length)])
            if derived.size == held.size:
                row_feeds[name] = derived.astype(held.dtype).reshape(held.shape)
        lengths[feature] = length
    return lengths


def _to_feature_order(tensor: np.ndarray, entry: Dict[str, Any], source: Dict[str, Any]) -> np.ndarray:
    """One row of an input tensor in the order its feature keeps its values (a picture is height × width × channels)."""
    axis = entry.get("batch_axis")
    sample = tensor if axis is None else np.take(tensor, 0, axis=axis)
    if str(source.get("layout") or "").lower() in ("nchw", "chw"):
        sample = np.moveaxis(sample, 0, -1)
    return np.asarray(sample, dtype=float).reshape(-1)
