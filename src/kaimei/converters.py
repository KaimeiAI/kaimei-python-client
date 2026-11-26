"""Models from more frameworks, read through ONNX by way of each framework's own converter.

* **TensorFlow SavedModel** (a directory holding ``saved_model.pb``) and **TensorFlow Lite** (a ``.tflite``
  file): ``tf2onnx`` writes them down as ONNX, and TensorFlow itself runs them for the check.
* **JAX**: a function, a jitted function or a Flax NNX module (a plain Python function needs
  ``framework="jax"``), with ``sample=``. ``jax2onnx`` writes it down, and JAX runs it for the check.
* **GGUF** (a ``.gguf`` file, llama.cpp's format): Hugging Face ``transformers`` opens it as a PyTorch model,
  with the tokenizer the file holds, and it is then read as any PyTorch model is. A model too large for the
  API to run is better registered with ``weights="summary"`` and run by its agent.

Each conversion is checked, like every other, against the original framework's own answers on a few rows: a
conversion that does not reproduce them is drawn, not run.
"""

from __future__ import annotations

import inspect
import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .structure import StructureError

#: Operator set the converters are asked to write: one Kaimei's kernels all cover.
OPSET = 17
#: Token positions a GGUF language model is read for when no sample says otherwise.
GGUF_DEFAULT_TOKENS = 32
#: The packages a JAX model's class comes from, by their top-level name: ``jaxon_models`` is not one of them.
JAX_PACKAGES = ("jax", "jaxlib", "flax")


def _has(package: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def model_file_kind(obj: Any) -> Optional[str]:
    """``"saved_model"``, ``"tflite"`` or ``"gguf"`` for a path to such a model; ``None`` for anything else."""
    if not isinstance(obj, (str, os.PathLike)):
        return None
    path = os.fspath(obj)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "saved_model.pb")):
        return "saved_model"
    lowered = path.lower()
    if lowered.endswith(".tflite") and os.path.isfile(path):
        return "tflite"
    if lowered.endswith(".gguf") and os.path.isfile(path):
        return "gguf"
    return None


def is_jax(obj: Any, framework: Optional[str] = None) -> bool:
    """A JAX model: anything callable when ``framework="jax"``, or a jitted function or a Flax NNX module."""
    if not callable(obj):
        return False
    if str(framework or "").lower() in ("jax", "flax"):
        return True
    # A jitted function's type is JAX's own; a Flax NNX module is the user's class, made from Flax's. A class is
    # JAX's when it comes from one of JAX's packages or their modules, not from any package whose name starts so.
    return any(
        str(getattr(kind, "__module__", "") or "").split(".")[0] in JAX_PACKAGES
        for kind in type(obj).__mro__
        if kind is not object
    )


def _samples(sample: Any) -> List[Any]:
    import numpy

    if sample is None:
        return []
    if isinstance(sample, Mapping):
        return [numpy.asarray(value) for value in sample.values()]
    if isinstance(sample, (list, tuple)) and sample and hasattr(sample[0], "shape"):
        return [numpy.asarray(value) for value in sample]
    return [numpy.asarray(sample)]


def _one_row(value: Any) -> Any:
    """The first row of an example input (its batch axis kept, one long); a value with no axes is left as it is."""
    return value[:1] if getattr(value, "ndim", 0) else value


def _pick(names: Sequence[str], wanted: Optional[str]) -> int:
    """The position of the output the prediction is read from (the importer's choice), among ``names``."""
    if wanted is not None and wanted in names:
        return list(names).index(wanted)
    return 0


# --- TensorFlow files ----------------------------------------------------------------------------------


def tensorflow_file_through_onnx(
    path: Any,
    kind: str,
    why: List[str],
    numpy: Any,
    *,
    sample: Any = None,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    check_rows: Any = None,
    labels: Optional[Sequence[Any]] = None,
    output: Any = None,
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    tokenizer: Any = None,
    framework: Optional[str] = None,
    name: Optional[str] = None,
    keep: Optional[Dict[str, Any]] = None,
    **_unused: Any,
) -> Optional[Dict[str, Any]]:
    """A TensorFlow SavedModel directory or a TensorFlow Lite file, read as a graph and checked against TensorFlow."""
    if not (_has("tensorflow") and _has("tf2onnx")):
        why.append("a TensorFlow model needs tensorflow and tf2onnx (pip install tensorflow tf2onnx)")
        return None
    from .graphs import _importer, _prediction_output, _read_onnx

    location = os.fspath(path)
    try:
        if kind == "saved_model":
            proto, library = _saved_model(location, numpy)
        else:
            proto, library = _tflite(location, numpy)
    except StructureError:
        raise
    except Exception as exc:  # noqa: BLE001 - TensorFlow and tf2onnx raise their own errors
        why.append(f"tf2onnx could not convert it ({type(exc).__name__}: {str(exc)[:200]})")
        return None
    if keep is not None:
        keep["onnx"] = bytes(proto.SerializeToString())
    tools = _importer()
    order = [str(info.name) for info in tools._input_infos(proto)]
    wanted, _tensor = _prediction_output(proto, output, task)
    position = _pick([str(info.name) for info in proto.graph.output], wanted)
    given = _samples(sample)
    samples = {key: value[:1] for key, value in zip(order, given)} if given else None
    spec = _read_onnx(
        proto,
        library=lambda feeds: library(feeds, position),
        feature_names=feature_names,
        task=task,
        labels=labels,
        inputs=inputs,
        input_shapes=None,
        samples=samples,
        output=wanted,
        mask_layout=mask_layout,
        threshold=threshold,
        name=name,
        tokenizer=tokenizer,
        check_rows=check_rows,
        framework=framework or ("tensorflow" if kind == "saved_model" else "tflite"),
    )
    steps = len(spec["graph"].get("nodes") or [])
    what = "TensorFlow SavedModel" if kind == "saved_model" else "TensorFlow Lite model"
    spec["summary"] = f"{what} converted to ONNX and read as a graph of {steps:,} operations. {spec.get('summary') or ''}".strip()
    return spec


def _saved_model(location: str, numpy: Any) -> Tuple[Any, Any]:
    import tensorflow as tf
    import tf2onnx

    loaded = tf.saved_model.load(location)
    signatures = dict(getattr(loaded, "signatures", {}) or {})
    if not signatures:
        raise StructureError(
            "This SavedModel has no serving signature to call it by. Save it with one (tf.saved_model.save with "
            "signatures=, or model.export for a Keras model).",
            code="unsupported_estimator",
        )
    chosen = "serving_default" if "serving_default" in signatures else sorted(signatures)[0]
    function = signatures[chosen]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph_def, inputs, outputs = tf2onnx.tf_loader.from_saved_model(location, None, None, signatures=[chosen])
        proto, _storage = tf2onnx.convert.from_graph_def(graph_def, input_names=inputs, output_names=outputs, opset=OPSET)
    declared = dict(function.structured_input_signature[1])
    arguments = {str(spec.name): str(key) for key, spec in declared.items()}
    # A signature answers with a dict; its tensors are the function's outputs, in the order of its sorted keys.
    keys = sorted(str(key) for key in dict(function.structured_outputs))
    by_tensor = dict(zip((str(tensor.name) for tensor in function.outputs), keys))
    ordered = [by_tensor.get(str(name), keys[min(index, len(keys) - 1)]) for index, name in enumerate(outputs)]

    def library(feeds: Mapping[str, Any], position: int) -> Any:
        given = {}
        for onnx_name, value in feeds.items():
            plain = str(onnx_name).split(":")[0]
            key = arguments.get(plain, plain)
            # The check asks in double precision; the signature takes its own number type.
            kind = declared[key].dtype if key in declared else None
            given[key] = tf.constant(numpy.asarray(value), dtype=kind)
        answered = function(**given)
        return numpy.asarray(answered[ordered[position]])

    return proto, library


def _tflite(location: str, numpy: Any) -> Tuple[Any, Any]:
    import tensorflow as tf
    import tf2onnx

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proto, _storage = tf2onnx.convert.from_tflite(location, opset=OPSET)
    interpreter = tf.lite.Interpreter(model_path=location)
    interpreter.allocate_tensors()
    ins = {str(item["name"]): item for item in interpreter.get_input_details()}
    outs = interpreter.get_output_details()

    def library(feeds: Mapping[str, Any], position: int) -> Any:
        for onnx_name, value in feeds.items():
            item = ins.get(str(onnx_name))
            if item is None:
                continue
            interpreter.set_tensor(item["index"], numpy.asarray(value).astype(item["dtype"]))
        interpreter.invoke()
        return numpy.asarray(interpreter.get_tensor(outs[min(position, len(outs) - 1)]["index"]))

    return proto, library


# --- JAX ---------------------------------------------------------------------------------------------------


def jax_through_onnx(
    function: Any,
    why: List[str],
    numpy: Any,
    *,
    sample: Any = None,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    check_rows: Any = None,
    labels: Optional[Sequence[Any]] = None,
    output: Any = None,
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    tokenizer: Any = None,
    framework: Optional[str] = None,
    name: Optional[str] = None,
    keep: Optional[Dict[str, Any]] = None,
    **_unused: Any,
) -> Optional[Dict[str, Any]]:
    """A JAX function or Flax NNX module, written down by ``jax2onnx`` and checked against JAX itself."""
    if not (_has("jax") and _has("jax2onnx")):
        why.append("a JAX model needs jax and jax2onnx (pip install jax2onnx)")
        return None
    given = _samples(sample)
    if not given:
        raise StructureError(
            "A JAX model is converted for the shapes of an example input: pass sample=... (what you would call it with).",
            code="invalid_structure",
            param="sample",
        )
    import jax
    import jax2onnx

    from .graphs import _importer, _prediction_output, _read_onnx

    names = _parameter_names(function, len(given))
    # The model is read and checked one row at a time, so it is converted for one row: a sample of eight rows would
    # otherwise fix its inputs at eight, and the reading would be fed one.
    given = [_one_row(value) for value in given]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proto = jax2onnx.to_onnx(
                function,
                [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in given],
                opset=OPSET,
                input_names=names,
                model_name=str(name or "jax_model"),
            )
    except Exception as exc:  # noqa: BLE001 - jax2onnx raises its own errors
        why.append(f"jax2onnx could not convert it ({type(exc).__name__}: {str(exc)[:200]})")
        return None
    if keep is not None:
        keep["onnx"] = bytes(proto.SerializeToString())
    tools = _importer()
    order = [str(info.name) for info in tools._input_infos(proto)]
    wanted, _tensor = _prediction_output(proto, output, task)
    position = _pick([str(info.name) for info in proto.graph.output], wanted)

    def library(feeds: Mapping[str, Any]) -> Any:
        arrays = [numpy.asarray(feeds[key]) for key in order]
        answered = function(*arrays)
        leaves = jax.tree_util.tree_leaves(answered)
        return numpy.asarray(leaves[min(position, len(leaves) - 1)])

    spec = _read_onnx(
        proto,
        library=library,
        feature_names=feature_names,
        task=task,
        labels=labels,
        inputs=inputs,
        input_shapes=None,
        samples=dict(zip(order, given)),
        output=wanted,
        mask_layout=mask_layout,
        threshold=threshold,
        name=name,
        tokenizer=tokenizer,
        check_rows=check_rows,
        framework=framework or "jax",
    )
    steps = len(spec["graph"].get("nodes") or [])
    spec["summary"] = f"JAX model converted to ONNX and read as a graph of {steps:,} operations. {spec.get('summary') or ''}".strip()
    return spec


def _parameter_names(function: Any, count: int) -> List[str]:
    """What the function calls its inputs, for the importer to read: ``input_ids`` is text, ``x`` says nothing."""
    try:
        target = function.__call__ if not inspect.isfunction(function) and hasattr(function, "__call__") else function
        parameters = [
            item.name
            for item in inspect.signature(target).parameters.values()
            if item.kind in (item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        parameters = []
    names = [str(item) for item in parameters[:count]]
    while len(names) < count:
        names.append(f"input_{len(names)}")
    return names


# --- GGUF --------------------------------------------------------------------------------------------------


def open_gguf(path: Any) -> Tuple[Any, Any]:
    """``(PyTorch model, tokenizer or None)`` for a GGUF file, opened by Hugging Face ``transformers``."""
    missing = [package for package in ("torch", "transformers", "gguf", "accelerate") if not _has(package)]
    if missing:
        raise StructureError(
            f"A GGUF file is opened by Hugging Face transformers, which needs {', '.join(missing)} "
            f"(pip install {' '.join(missing)}).",
            code="unsupported_estimator",
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    folder, file = os.path.split(os.path.abspath(os.fspath(path)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = AutoModelForCausalLM.from_pretrained(folder, gguf_file=file)
        except Exception as exc:  # noqa: BLE001 - transformers reads only the architectures it knows
            raise StructureError(
                f"transformers could not open this GGUF file ({type(exc).__name__}: {str(exc)[:200]}). It reads "
                "the architectures it knows (Llama, Mistral, Qwen2 and others); register any other with an agent.",
                code="unsupported_estimator",
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(folder, gguf_file=file)
        except Exception:  # noqa: BLE001 - a file without a tokenizer still makes a model of token ids
            tokenizer = None
    return model.eval(), tokenizer


def gguf_sample(model: Any, tokenizer: Any, torch: Any) -> Any:
    """An example input for a GGUF language model: its first token, to be filled up to ``GGUF_DEFAULT_TOKENS``."""
    first = getattr(getattr(model, "config", None), "bos_token_id", None)
    if first is None and tokenizer is not None:
        first = getattr(tokenizer, "bos_token_id", None)
    return torch.tensor([[int(first or 0)]])


__all__ = [
    "GGUF_DEFAULT_TOKENS",
    "gguf_sample",
    "is_jax",
    "jax_through_onnx",
    "model_file_kind",
    "open_gguf",
    "tensorflow_file_through_onnx",
]
