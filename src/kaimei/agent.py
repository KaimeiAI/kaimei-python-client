"""An agent: your model run where it lives, answering for itself on Kaimei.

Some models are not run by Kaimei: they are too large, they need a GPU, their code is your own, or their
weights may not leave your network. An agent is a small loop, started beside the model, that connects *out*
to the Kaimei API and waits for work. When a prediction, an explanation or a look inside the model is asked
for, the API hands the agent a job; the agent runs the real model and sends the answer back. Nothing is
opened on your side and no weights are uploaded.

Register the model as one an agent runs, then start the agent next to it::

    from kaimei import Kaimei
    from kaimei.agent import predict_with, serve

    client = Kaimei()
    client.models.create(
        name="Churn (in our VPC)", task="binary_classification",
        features=[{"name": "tenure", "type": "numeric"}, {"name": "seats", "type": "numeric"}],
        outputs={"labels": ["stays", "leaves"]},
        inference={"type": "agent"},
    )
    serve(client, "churn-in-our-vpc", predict=predict_with(estimator, ["tenure", "seats"]))   # blocks

``predict(rows)`` is any function from a list of rows to an array of ``(rows,)`` values or ``(rows, classes)``
probabilities. A row is ``{feature: value}``: a number, flag or category for a column of a table; a list of
token ids for a text feature (Kaimei has already split the text with the model's tokenizer); a ``float32``
numpy array for an image (height by width by channels, in the feature's own units), a series or a tensor.

The agent can also run the model itself, and then it answers *trace* and *activations* requests too, so the
model's page shows live values for a structure uploaded with ``weights="summary"`` (nothing Kaimei could
run). There are two ways:

* ``module=`` a ``torch.nn.Module``. The agent runs it once with forward hooks and reports what the modules
  returned. It reports a step only where a module's output is that step's, and nothing for the steps in
  between (the arithmetic written in a ``forward``, the inside of an attention layer).
* ``onnx=`` the model's ONNX file (a path, bytes or ``onnx.ModelProto``). The agent runs it with ONNX
  Runtime and can ask for any tensor in it, so it reports every step of a structure that was read from that
  file. ``save_onnx=`` on ``client.models.upload`` writes the file an upload of a live model was read from.

Either way the agent sends the model's own output and says so; Kaimei reads it as the model's structure
does (a classifier's scores through its softmax, a language model at the last token it was given). An
encoder-decoder's are its scores for the next token: the ones after the last token its decoder has written
so far, which Kaimei keeps in a feature of its own (marked ``generated``) and sends with every row.

What a look inside sends is what the model computed from the input it was given, and nothing else. The
structure that names the steps comes from Kaimei, and the agent does not take its word for what a step is:

* A step is sent only where its values are computed from the row. A weight, a constant, and what is worked
  out from those alone (or from the input's shape alone, like the embedding of each position) are the
  model's own numbers, and are never sent, whatever step the structure says they are. In a file that is
  read off its graph; a module is followed as it runs, one operation at a time (PyTorch 1.13 or later;
  before that only what the modules' hooks can tell is kept back: a parameter or a buffer given back in
  whole or in part, and what a module gives that was called with nothing else).
* A step's values are laid out only as its tensor is: the shape the structure gives must be the tensor's
  own, and the order of its axes an order of them.
* An agent started with the structure that was uploaded (``structure=``) says nothing of a step that
  structure does not hold, and reads a file's steps and its prediction from the tensors that structure names.

A step kept back is left out of the answer, which names it and says why under ``withheld``; the job is
answered all the same. What stays visible is the model at work: whoever can ask it for predictions and looks
inside sees what its steps compute for the inputs they choose (a layer's bias for a row of zeros, a row of an
embedding table for each token), and from enough of those a model can be worked out. That comes with
answering at all.

With ``tokenizer=`` set to the model's tokenizer the agent also splits text. Kaimei runs a WordPiece or a
byte-level BPE tokenizer itself; any other (SentencePiece's, tiktoken's, your own) it only has the
vocabulary of, so a text sent for the model is first handed to the agent as a *tokenize* job and the ids it
answers with are what the model is then given. Without a tokenizer the agent does not offer to, and such a
model takes token ids.

From a shell: ``python -m kaimei.agent --model churn-in-our-vpc --predict mypackage.scoring:predict`` or
``python -m kaimei.agent --model my-model --onnx model.onnx``, with the API key in ``KAIMEI_API_KEY`` (and
``KAIMEI_BASE_URL`` when it is not the default). Ctrl+C or SIGTERM stops it, and its session is closed.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import os
import platform
import re
import sys
import threading
import time
import warnings
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .errors import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    KaimeiError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

#: What an agent can be asked for.
KINDS = ("predict", "trace", "activations", "tokenize")
#: Seconds one request for work is held open by the API (it allows up to 25).
WAIT_SECONDS = 20.0
#: A request for work that comes back empty sooner than this is followed by a pause to the end of it, so an
#: agent told not to wait (``wait=0``), or a client that gives up on a request early, does not ask non-stop.
MIN_POLL_SECONDS = 1.0
#: What the API takes in one answer: values (every number, string, list and object counts as one) and bytes.
MAX_ANSWER_VALUES = 4_000_000
MAX_ANSWER_BYTES = 128 * 1024 * 1024
#: Values of a step sent with a trace, and how many more are sampled along it.
FRAME_VALUES = 48
FRAME_SAMPLE = 256
#: A step up to this many values sends them all with its activations.
VALUE_LIMIT = 4096
#: Steps whose maps or values are sent in full when a request names none: maps, then vectors.
DETAILED_MAPS = 24
DETAILED_VECTORS = 8
#: Steps an answer names as kept back (``withheld``), at most: a structure can hold tens of thousands.
WITHHELD_SAID = 64
#: The longest pause between attempts to reach the API again.
MAX_BACKOFF = 30.0
#: Tensors asked of a model's ONNX file in one run, and values they may hold together. More steps than that
#: are read in several runs, so a look inside a large model does not hold all of it in memory at once. How much
#: a tensor holds is what the file says it does, never the size Kaimei's structure gives its step, and what
#: one run read is made into its frames or stages and let go before the next run.
FETCH_STEPS = 256
FETCH_VALUES = 1 << 24
#: ONNX Runtime sessions kept ready: the file's own outputs (a prediction), which is always kept, and those of
#: looks inside. A look that reads its steps in several runs makes a session for each run, and they are kept by
#: most recent use, so the next look makes none of them again, as far as the agent's ``session_memory`` allows
#: (``SESSION_BYTES`` unless it says): a session holds about three copies of the file's weights. A look never
#: lets one of its own sessions go to make room for another of them: what does not fit is made again each time,
#: and only that. ``SESSIONS_KEPT`` are kept however little the memory, and ``MOST_SESSIONS`` at most.
SESSIONS_KEPT = 3
SESSION_BYTES = 2 * 1024 * 1024 * 1024
MOST_SESSIONS = 64
#: Feature types that are one value in a table, and those that are token ids.
_COLUMN_TYPES = {"numeric", "integer", "binary", "categorical"}
_TEXT_TYPES = {"text", "tokens"}
#: The API's own lists (its ``default_detail``): steps that only resample a map, and activations.
_RESAMPLING = {
    "maxpool", "avgpool", "adaptive_avgpool", "adaptive_maxpool", "pad2d", "dropout", "identity",
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "LpPool", "Pad", "Dropout", "Identity",
    "Resize", "Upsample",
}  # fmt: skip
_ACTIVATIONS = {
    "relu", "relu6", "leaky_relu", "elu", "gelu", "silu", "hardswish", "hardsigmoid", "sigmoid", "tanh", "softmax",
}  # fmt: skip
#: How the importer turns a name in a model file into a name in a structure.
_UNSAFE = re.compile(r"[^0-9A-Za-z_.\-]+")
#: A module's later calls, as PyTorch's exporter names their scope: ``relu_1`` is the second call of ``relu``.
_CALL_SCOPE = re.compile(r"^(.+)_(\d+)$")

Predict = Callable[[List[Dict[str, Any]]], Any]


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - numpy comes with every model library
        raise KaimeiError("A Kaimei agent needs numpy: pip install numpy") from exc
    return numpy


def _is_torch(obj: Any) -> bool:
    return hasattr(obj, "named_modules") and hasattr(obj, "register_forward_hook") and callable(getattr(obj, "forward", None))


def _product(dims: Iterable[Any]) -> int:
    total = 1
    for dim in dims:
        total *= int(dim)
    return total


def _clean(text: str) -> str:
    return _UNSAFE.sub("_", text).strip("._") or "step"


def decode_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A row as the API sends it to an agent, with its packed values as numpy arrays.

    A picture, a series or a tensor arrives as ``{"$b64": ..., "dtype": "f32", "shape": [...]}``: its numbers as
    little-endian ``float32`` bytes in base64. Everything else (numbers, flags, categories, lists of token
    ids) is already what it is.
    """
    numpy = _numpy()
    decoded: Dict[str, Any] = {}
    for name, value in row.items():
        if isinstance(value, Mapping) and "$b64" in value:
            kind = {"f32": "<f4", "f64": "<f8", "f16": "<f2"}.get(str(value.get("dtype") or "f32"), "<f4")
            array = numpy.frombuffer(base64.b64decode(str(value["$b64"])), dtype=kind).astype(numpy.float32)
            shape = [int(dim) for dim in value.get("shape") or [array.size]]
            decoded[str(name)] = array.reshape(shape)
        else:
            decoded[str(name)] = value
    return decoded


def encoder_of(tokenizer: Any) -> Callable[[str, Optional[int]], List[int]]:
    """``encode(text, most)`` from whatever splits text: the ids the model reads, special tokens included.

    A ``transformers`` tokenizer is called as ``tokenizer(text, truncation=..., max_length=most)``, so it cuts
    a long text the way it was set up to. Anything with an ``encode`` method (a ``tokenizers.Tokenizer``, a
    SentencePiece processor, a tiktoken encoding) is asked through it, and any other callable is called with
    the text; their ids are cut to ``most`` from the end.
    """
    if callable(tokenizer) and callable(getattr(tokenizer, "convert_ids_to_tokens", None)):

        def by_transformers(text: str, most: Optional[int]) -> List[int]:
            return _ids(tokenizer(text, truncation=most is not None, max_length=most), None)

        return by_transformers
    method = getattr(tokenizer, "encode", None)
    if callable(method):
        return lambda text, most: _ids(method(text), most)
    if callable(tokenizer):
        return lambda text, most: _ids(tokenizer(text), most)
    raise TypeError(
        f"{type(tokenizer).__name__} cannot split text: give the model's tokenizer, or a function from a text to its ids."
    )


def _ids(found: Any, most: Optional[int]) -> List[int]:
    """Token ids as a list of whole numbers, from the several shapes tokenizers answer in."""
    if isinstance(found, Mapping) or hasattr(found, "input_ids"):
        found = found["input_ids"] if isinstance(found, Mapping) else found.input_ids
    found = getattr(found, "ids", found)
    if hasattr(found, "tolist"):
        found = found.tolist()
    ids = [int(item) for item in found]
    return ids if most is None else ids[: max(int(most), 0)]


def predict_with(estimator: Any, feature_names: Optional[Sequence[str]] = None) -> Predict:
    """A ``predict(rows)`` for a scikit-learn-style estimator or a ``torch.nn.Module``.

    Rows become a table with one column for each name in ``feature_names``, in that order (the estimator's own
    ``feature_names_in_``, or the order of the first row's keys, when it is left out). An estimator fitted on
    named columns gets each where it was fitted to find it, however ``feature_names`` orders the same names;
    other names are taken as its columns in order, and a mix of the two is refused. An estimator answers
    with ``predict_proba`` when it has one. A classifier without it answers with ``decision_function``, and
    anything else with ``predict``.

    A module is called on the table as a float tensor. When the rows hold one feature that is itself an array
    the module is called on those arrays stacked: token ids as whole numbers, padded with zeros to the longest
    row; a picture channels first, as PyTorch has them, when the module's first convolution takes as many
    channels as the picture's last axis has. Write your own function for anything else.

    What a module gives, and what ``decision_function`` gives, are the model's own output. The function
    returned says so (its ``raw`` attribute is true), the agent passes that on, and Kaimei reads the numbers
    the way the model's structure does. Its ``link`` attribute says what they are where that is known:
    ``"none"`` for probabilities (``predict_proba``, a module that ends in a softmax or a sigmoid),
    ``"softmax"`` for a module that ends in a log-softmax, ``"scores"`` for a ``decision_function``. A
    function of your own is taken to return probabilities, or the value.
    """
    names = [str(name) for name in feature_names] if feature_names is not None else None
    if _is_torch(estimator):

        def predict_module(rows: List[Dict[str, Any]]) -> Any:
            import torch

            from .graphs import restore_modes, training_modes

            given = _module_input(estimator, rows, names, torch)
            modes = training_modes(estimator)
            estimator.eval()
            try:
                with torch.no_grad():
                    produced = estimator(given)
            finally:
                restore_modes(modes)  # each part as it was: a frozen one stays frozen, a training one trains on
            first = produced[0] if isinstance(produced, (tuple, list)) else produced
            return first.detach().cpu().numpy()

        predict_module.raw = True  # type: ignore[attr-defined]
        ends = _traced_ends(estimator)
        if ends is not None:
            predict_module.link = ends  # type: ignore[attr-defined]
        return predict_module
    scores = False
    link: Optional[str] = None
    method = getattr(estimator, "predict_proba", None)
    if callable(method):
        link = "none"  # probabilities already
    if not callable(method) and getattr(estimator, "classes_", None) is not None:
        method = getattr(estimator, "decision_function", None)
        scores = callable(method)
        link = "scores" if scores else None
    if not callable(method):
        method = getattr(estimator, "predict", None)
    if not callable(method):
        raise TypeError(f"{type(estimator).__name__} has no predict_proba or predict to answer with.")
    fitted = getattr(estimator, "feature_names_in_", None)
    fitted_names = [str(name) for name in fitted] if fitted is not None else None
    if names and fitted_names and set(names) == set(fitted_names):
        # The same columns: each is put where the estimator was fitted to find it, whatever order they were named in.
        names = fitted_names
    elif names and fitted_names and set(names) & set(fitted_names):
        raise ValueError(
            f"feature_names names some of the columns {type(estimator).__name__} was fitted on "
            f"({', '.join(fitted_names)}) and not others. Name exactly those, or leave feature_names out."
        )

    def predict_estimator(rows: List[Dict[str, Any]]) -> Any:
        numpy = _numpy()
        columns = names or fitted_names or [str(key) for key in rows[0]]
        table: Any = numpy.asarray([[row[name] for name in columns] for row in rows], dtype=float)
        if fitted is not None:
            try:
                import pandas

                table = pandas.DataFrame(table, columns=list(fitted))
            except ImportError:
                pass
        return numpy.asarray(method(table), dtype=float)

    predict_estimator.raw = scores  # type: ignore[attr-defined]
    if link is not None:
        predict_estimator.link = link  # type: ignore[attr-defined]
    return predict_estimator


# --- what a model's own output is: scores, or probabilities already ---------------------------

#: What a model's last step makes of its scores, and the ``link`` the API is told: a softmax or a sigmoid
#: gives probabilities ("none": nothing more to apply); a log-softmax gives their logarithms, which a softmax
#: turns back into them exactly. ``"scores"`` is said to no one: it is how the API reads an output anyway.
_ENDS = {"softmax": "none", "sigmoid": "none", "logsoftmax": "softmax", "log_softmax": "softmax"}
#: The links an answer may name, as the API reads them.
_LINKS = ("none", "softmax", "sigmoid")
#: Steps after the last one that only lay its values out anew, looked through to find it.
_LAYOUT = {
    "flatten", "reshape", "view", "squeeze", "unsqueeze", "contiguous", "identity", "dropout", "float", "to",
    "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Identity", "Dropout", "Cast",
}  # fmt: skip
#: What a classical classifier (``ai.onnx.ml``) does to its scores to make them probabilities.
_PROBABILITY_TRANSFORMS = (b"SOFTMAX", b"LOGISTIC", b"SOFTMAX_ZERO")
#: Classical classifiers: their first output is the class, their second the probabilities or the scores.
_CLASSIFIERS = {"LinearClassifier", "SVMClassifier", "TreeEnsembleClassifier"}
#: Operators that compute scores from a model's weights. A look back from what a converted classifier gives,
#: for the step that made its numbers, stops at them: what comes out of them is scores.
_WEIGHED = {
    "MatMul", "Gemm", "Conv", "ConvTranspose", "Einsum", "MatMulInteger", "ConvInteger", "QLinearMatMul",
    "QLinearConv", "RNN", "GRU", "LSTM", "Attention",
}  # fmt: skip
#: Element types of a tensor that holds a class (ONNX's numbers): whole numbers of any size, and strings.
_LABEL_TYPES = {2, 3, 4, 5, 6, 7, 8, 12, 13}
#: Operators that read only the shape of what they are given: what they give says nothing of its values.
_SHAPE_READERS = {"Shape", "Size", "EyeLike", "RandomNormalLike", "RandomUniformLike"}
#: Parts of a text that an input of a file is given the same for every text of that length: segment ids of
#: zeros, and positions counted from nought. Nothing computed from them alone is computed from the text.
_SAME_FOR_EVERY_ROW = {"type_ids", "positions"}
#: Why nothing is said of a step that Kaimei's structure names and the agent could read. An agent sends what a
#: model computes from the input it is given, never the model itself: its weights stay where they are.
_NOT_HELD = "It is not a step of the structure this agent was started with."
_NOT_COMPUTED = (
    "Its values are not computed from the model's input: they are the model's own numbers (weights, constants, or "
    "what is worked out from those alone), and those stay where the model is."
)
#: Operations of PyTorch that read only the shape and the kind of the tensor they are given: what they give
#: holds no number of it.
_SHAPE_ONLY = {
    "zeros_like", "ones_like", "full_like", "empty_like", "rand_like", "randn_like", "randint_like",
    "new_zeros", "new_ones", "new_full", "new_empty", "new_empty_strided",
}  # fmt: skip
#: What a tensor computed from the model's own numbers alone is marked with while a module's run is followed.
_OWN_MARK = "_kaimei_own"
#: Why no step of a module is sent when the agent could not make sense of an operation it ran.
_NOT_FOLLOWED = (
    "The agent could not follow what this model computes its steps from, so it cannot tell the model's own numbers "
    "from what the input went into, and sends no step of it."
)
#: What a module is called with that holds no tensor, whatever it says.
_PLAIN = (type(None), bool, int, float, complex, str, bytes)
#: The largest number there is: a statistic too large for one is sent as this.
_LARGEST = sys.float_info.max


def _traced_ends(module: Any) -> Optional[str]:
    """What a module's last step is, read from its code with ``torch.fx``: ``"none"`` or ``"softmax"`` (see
    ``_ENDS``), ``"scores"`` for anything else, or ``None`` when its code cannot be traced."""
    try:
        import torch.fx

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traced = torch.fx.symbolic_trace(module)
    except Exception:  # noqa: BLE001 - a forward torch.fx cannot follow (control flow on values, say)
        return None
    node: Any = next((item for item in traced.graph.nodes if item.op == "output"), None)
    node = node.args[0] if node is not None and node.args else None
    for _hop in range(8):
        while isinstance(node, (tuple, list)) and node:
            node = node[0]
        if isinstance(node, Mapping) and node:
            node = next(iter(node.values()))
        if not isinstance(node, torch.fx.Node):
            return "scores"
        if node.op == "call_module":
            name = type(traced.get_submodule(str(node.target))).__name__.lower()
        elif node.op == "call_function":
            name = str(getattr(node.target, "__name__", "")).lower()
        elif node.op == "call_method":
            name = str(node.target).lower()
        else:
            return "scores"
        if name in _ENDS:
            return _ENDS[name]
        if name not in _LAYOUT or not node.args:
            return "scores"
        node = node.args[0]
    return "scores"


def _ends_by_values(values: Any) -> Optional[str]:
    """What a model's output is, from the numbers themselves, for a model whose code or file does not say.

    Only rows of several numbers say it: rows that are never negative and each add up to one are
    probabilities, and rows whose exponentials each add up to one are their logarithms. Scores do neither.
    """
    numpy = _numpy()
    array = numpy.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2 or not array.size or not numpy.isfinite(array).all():
        return None
    if (array >= -1e-6).all() and numpy.allclose(array.sum(axis=1), 1.0, atol=1e-4):
        return "none"
    if (array <= 1e-6).all() and numpy.allclose(numpy.exp(array).sum(axis=1), 1.0, atol=1e-4):
        return "softmax"
    return None


# --- a classifier converted to ONNX: its class, and what its numbers beside it are --------------


def _attribute(node: Any, name: str) -> Any:
    """A node's attribute called ``name``, or ``None``."""
    return next((item for item in node.attribute if item.name == name), None)


def _classlabels(node: Any) -> Optional[List[Any]]:
    """The classes a classical classifier or a ``ZipMap`` names, in the order of its numbers."""
    for item in node.attribute:
        if item.name in ("classlabels_ints", "classlabels_int64s") and len(item.ints):
            return [int(value) for value in item.ints]
        if item.name == "classlabels_strings" and len(item.strings):
            return [value.decode("utf-8", "replace") for value in item.strings]
    return None


def _what_it_makes(node: Any) -> Optional[str]:
    """What a step of a converted classifier makes of its scores, when it says (``None`` when it does not):
    ``"none"`` for probabilities (a softmax, a sigmoid, a classical classifier's own transform, or a support
    vector machine's probability model), ``"softmax"`` for their logarithms, and ``"scores"`` for a linear or
    support vector classifier without them. A tree ensemble's numbers are probabilities or scores as it was
    trained, so they say it themselves."""
    domain = str(node.domain)
    if domain in ("", "ai.onnx"):
        if node.op_type in ("Softmax", "Sigmoid"):
            return "none"
        return "softmax" if node.op_type == "LogSoftmax" else None
    if domain != "ai.onnx.ml" or node.op_type not in _CLASSIFIERS:
        return None
    transform = _attribute(node, "post_transform")
    platt = _attribute(node, "prob_a")
    if (transform is not None and transform.s in _PROBABILITY_TRANSFORMS) or (platt is not None and len(platt.floats)):
        return "none"
    return "scores" if node.op_type in ("LinearClassifier", "SVMClassifier") else None


def _class_key(value: Any) -> Any:
    """A class as one value, whatever it is held in: a whole number, or a string (bytes are decoded)."""
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (bytes, str)):
        value = item()  # a numpy number or string
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (bool, int)):
        return int(value)
    return value if isinstance(value, str) else str(value)


def _class_indices(labels: Sequence[Any], classes: Sequence[Any]) -> Optional[Any]:
    """Where each row's class is among ``classes``, or ``None`` when one row's is not among them."""
    numpy = _numpy()
    keys = [_class_key(item) for item in classes]
    found = []
    for label in labels:
        flat = numpy.asarray(label, dtype=object).reshape(-1)
        key = _class_key(flat[0]) if flat.size == 1 else None
        if key is None or key not in keys:
            return None
        found.append(keys.index(key))
    return numpy.asarray(found, dtype=int)


def _margin_of(values: Any, sign: float, paired: bool) -> Any:
    """A two-class model's scores as one margin, a column for each row, turned by ``sign`` to be positive for
    its second class. One score is the margin. Two are a margin and its negative (``paired``: an SVC's
    ``[margin, -margin]``, a linear model's ``[-margin, margin]``), of which the second is taken; or two scores
    of their own, whose difference is the margin a softmax of the two reads."""
    numpy = _numpy()
    array = numpy.asarray(values, dtype=float)
    if array.shape[1] == 1:
        one = array[:, 0]
    else:
        one = array[:, 1] if paired else array[:, 1] - array[:, 0]
    return (float(sign) * one).reshape(-1, 1)


def _class_read(values: Any, binary: bool, positive: int) -> Any:
    """The class Kaimei reads from a classifier's own numbers sent with no link (``AgentRunner.read_raw``).

    One number is the second class's score, through a sigmoid (for a model of two classes, whichever is the
    positive one); several are every class's, through a softmax. Two classes are then split at one half, an
    even split going to the first; more go to the highest, the first of equal ones.
    """
    numpy = _numpy()
    array = numpy.asarray(values, dtype=float)
    if array.shape[1] == 1:
        second = 1.0 / (1.0 + numpy.exp(-numpy.clip(array[:, 0], -700.0, 700.0)))
        # Not a model of two classes (a multiclass one with two labels): its one number is the positive class's.
        first = not binary and positive == 0
        probabilities = numpy.column_stack([second, 1.0 - second] if first else [1.0 - second, second])
    else:
        shifted = numpy.exp(array - array.max(axis=1, keepdims=True))
        probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    if binary and probabilities.shape[1] == 2:
        score = probabilities[:, positive]
        chosen = numpy.where(score >= 0.5, positive, 1 - positive)
        return numpy.where(numpy.abs(score - 0.5) <= 1e-15, 0, chosen)
    return probabilities.argmax(axis=1)


def _first_channels(module: Any) -> Optional[int]:
    """How many channels the module's first convolution takes, when it has one."""
    for layer in module.modules():
        if type(layer).__name__ in ("Conv2d", "Conv1d", "Conv3d") and hasattr(layer, "in_channels"):
            return int(layer.in_channels)
    return None


def _module_input(
    module: Any,
    rows: Sequence[Mapping[str, Any]],
    names: Optional[Sequence[str]],
    torch: Any,
    kinds: Optional[Mapping[str, str]] = None,
    lengths: Optional[Mapping[str, int]] = None,
    pad_id: int = 0,
) -> Any:
    """Rows as the tensor a module is called on: a float table, or one array-valued feature stacked.

    ``kinds`` says what each feature is when the model on Kaimei is known (``image``, ``text``…), and
    ``lengths`` how many tokens a text feature holds; without them the values themselves are looked at.
    """
    numpy = _numpy()
    columns = list(names) if names else [str(key) for key in rows[0]]
    weight = next(iter(module.parameters()), None)
    kind = weight.dtype if weight is not None and weight.dtype.is_floating_point else torch.float32
    device = weight.device if weight is not None else None
    first = rows[0][columns[0]]
    composite = isinstance(first, (list, tuple)) or hasattr(first, "shape")
    if len(columns) > 1 and any(isinstance(rows[0][name], (list, tuple)) or hasattr(rows[0][name], "shape") for name in columns):
        raise ValueError(
            "This model has more than one input that is not a single number. Give the agent prepare=..., a function "
            "from one row to what the module is called with."
        )
    if not composite:
        table = numpy.asarray([[row[name] for name in columns] for row in rows], dtype=float)
        return torch.as_tensor(table, dtype=kind, device=device)
    name = columns[0]
    declared = str((kinds or {}).get(name) or "")
    values = [row[name] for row in rows]
    tokens = declared in _TEXT_TYPES or (
        not declared and isinstance(first, (list, tuple)) and all(isinstance(item, int) for item in first)
    )
    if tokens:
        longest = int((lengths or {}).get(name) or max(len(item) for item in values))
        ids = numpy.full((len(values), longest), int(pad_id), dtype=numpy.int64)
        for at, item in enumerate(values):
            ids[at, : min(len(item), longest)] = [int(token) for token in item[:longest]]
        return torch.as_tensor(ids, dtype=torch.long, device=device)
    stacked = numpy.stack([numpy.asarray(item, dtype=numpy.float32) for item in values])
    pictured = declared == "image" or (not declared and stacked.ndim == 4 and _first_channels(module) == stacked.shape[-1])
    if pictured and stacked.ndim == 4 and _first_channels(module) != stacked.shape[1]:
        stacked = stacked.transpose(0, 3, 1, 2)  # height, width, channels as Kaimei sends a picture -> channels first
    return torch.as_tensor(numpy.ascontiguousarray(stacked), dtype=kind, device=device)


# --- what a look inside is reported as: the API's own forms -----------------------------------


def _sendable(values: Any, numpy: Any) -> Any:
    """Statistics as numbers an answer can hold. JSON has no infinity, so one too large for a float is sent as
    the largest float there is; what is not a number is nought, which is how the API reads one too."""
    return numpy.nan_to_num(numpy.asarray(values, dtype=float), nan=0.0, posinf=_LARGEST, neginf=-_LARGEST)


def _mean(values: Any, axis: Any, numpy: Any) -> Any:
    """The mean along ``axis``. Values whose sum is too large for a float are divided before they are added."""
    with numpy.errstate(over="ignore", invalid="ignore"):
        mean = numpy.asarray(values.mean(axis=axis))
        if not numpy.isfinite(mean).all():
            mean = numpy.asarray((values / (values.size / max(mean.size, 1))).sum(axis=axis))
    return mean


def _summary(flat: Any, numpy: Any) -> Dict[str, Any]:
    """What a trace says of one step's values, in the API's own terms.

    A sum of squares overflows for values above about 1e154, long before the values themselves do: their
    length is then worked out at a size where it does not, and one that is itself too large for a float is sent
    as the largest float there is.
    """
    if not flat.size:
        return {"size": 0, "max_abs": 0.0, "mean_abs": 0.0, "l2": 0.0, "active": 0, "argmax": 0}
    magnitude = numpy.abs(flat)
    largest = float(magnitude.max())
    with numpy.errstate(over="ignore", invalid="ignore"):
        mean, length = float(magnitude.mean()), float(numpy.linalg.norm(flat))
        if not (numpy.isfinite(mean) and numpy.isfinite(length)) and numpy.isfinite(largest) and largest > 0:
            scaled = magnitude / largest
            mean, length = largest * float(scaled.mean()), largest * float(numpy.linalg.norm(scaled))
    return {
        "size": int(flat.size),
        "max_abs": float(_sendable(largest, numpy)),
        "mean_abs": float(_sendable(mean, numpy)),
        "l2": float(_sendable(length, numpy)),
        "active": int(numpy.count_nonzero(flat)),
        "argmax": int(numpy.argmax(magnitude)),
    }


def _picture(values: Any, numpy: Any) -> Dict[str, Any]:
    """Values as a byte each with their range: ``value = range[0] + byte / 255 * (range[1] - range[0])``."""
    array = numpy.asarray(values, dtype=float)
    finite = numpy.where(numpy.isfinite(array), array, 0.0)
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    if not numpy.isfinite(high - low):
        # A range wider than the largest float: the bytes are worked out at half the size, where it fits.
        data = numpy.round((finite / 2.0 - low / 2.0) / (high / 2.0 - low / 2.0) * 255.0).astype(numpy.uint8)
    elif high - low > 1e-12:
        data = numpy.round((finite - low) / (high - low) * 255.0).astype(numpy.uint8)
    else:
        data = numpy.zeros(finite.shape, dtype=numpy.uint8)
    return {
        "shape": [int(dim) for dim in array.shape],
        "range": [round(low, 8), round(high, 8)],
        "encoding": "u8",
        "data": base64.b64encode(data.tobytes(order="C")).decode("ascii"),
    }


def _shrunk(map2d: Any, max_side: int, numpy: Any) -> Any:
    """A map averaged over areas until its long side is at most ``max_side`` cells, as the API shrinks one."""
    height, width = map2d.shape
    longest = max(height, width)
    if longest <= max_side:
        return map2d
    largest = float(numpy.abs(map2d).max()) if map2d.size else 0.0
    if numpy.isfinite(largest) and largest > 1e150:
        # Values this large overflow when they are added up: they are averaged at a size where they do not.
        return _shrunk(numpy.asarray(map2d, dtype=float) / largest, max_side, numpy) * largest
    out_h = max(1, round(height * max_side / longest))
    out_w = max(1, round(width * max_side / longest))
    ys = numpy.linspace(0, height, out_h + 1).round().astype(int)
    xs = numpy.linspace(0, width, out_w + 1).round().astype(int)
    ys[1:] = numpy.maximum(ys[1:], ys[:-1] + 1)
    xs[1:] = numpy.maximum(xs[1:], xs[:-1] + 1)
    ys, xs = numpy.minimum(ys, height), numpy.minimum(xs, width)
    table = numpy.zeros((height + 1, width + 1), dtype=float)
    table[1:, 1:] = numpy.asarray(map2d, dtype=float).cumsum(axis=0).cumsum(axis=1)
    high, low = table[ys[1:], :], table[ys[:-1], :]
    sums = high[:, xs[1:]] - low[:, xs[1:]] - high[:, xs[:-1]] + low[:, xs[:-1]]
    counts = numpy.diff(ys)[:, None] * numpy.diff(xs)[None, :]
    return sums / numpy.maximum(counts, 1)


def _shrunk_table(table: Any, most: int, numpy: Any) -> Any:
    """A table with neighbouring rows and columns averaged until it fits ``most`` by ``most``."""
    for axis in (0, 1):
        size = table.shape[axis]
        if size <= most:
            continue
        step = int(numpy.ceil(size / most))
        pad = (-size) % step
        if pad:
            widths = [(0, 0), (0, 0)]
            widths[axis] = (0, pad)
            table = numpy.pad(table, widths, mode="edge")
        if axis == 0:
            table = table.reshape(table.shape[0] // step, step, table.shape[1]).mean(axis=1)
        else:
            table = table.reshape(table.shape[0], table.shape[1] // step, step).mean(axis=2)
    return table


def _spatial(step: Mapping[str, Any]) -> Optional[Tuple[int, int, int]]:
    """``(height, width, channels)`` when the step is shown as a map with more than one position."""
    shown = (step.get("args") or {}).get("shape")
    if not shown or len(shown) != 3:
        return None
    height, width, channels = (int(dim) for dim in shown)
    if height * width <= 1 or height * width * channels != int(step.get("units") or 0):
        return None
    return height, width, channels


def _count(value: Any) -> Optional[int]:
    """A size or an axis as a whole number that is not negative, or ``None`` when ``value`` is not one. What a
    structure says comes from Kaimei and is read, not trusted."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    try:
        number = int(value.__index__())  # a whole number of any kind, numpy's included
    except (AttributeError, TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _counts(value: Any) -> Optional[List[int]]:
    """A shape, or an order of axes, as a list of whole numbers; ``None`` when ``value`` is not one."""
    if not isinstance(value, (list, tuple)):
        return None
    found = [_count(item) for item in value]
    return None if any(item is None for item in found) else [int(item) for item in found if item is not None]


def _step_read(node: Any) -> Optional[Dict[str, Any]]:
    """A step of Kaimei's structure as the agent reads it, or ``None`` for what is not a step.

    The structure comes from Kaimei and is read, not trusted. A step has an id. Its size is a whole number
    (nought when it is not one), what it says of itself (``args``) is a mapping, the shape it is shown in is
    a list of whole numbers (left out when it is not one), and the steps it reads are a list of ids. Nothing
    a structure says can then make a look inside fail: at most a step is left out.
    """
    if not isinstance(node, Mapping) or isinstance(node.get("id"), bool) or not isinstance(node.get("id"), (str, int)):
        return None
    step = dict(node)
    step["id"] = str(node["id"])
    args = dict(node["args"]) if isinstance(node.get("args"), Mapping) else {}
    if "shape" in args:
        shape = _counts(args["shape"])
        if shape is None:
            del args["shape"]
        else:
            args["shape"] = shape
    step["args"] = args
    step["units"] = _count(node.get("units")) or 0
    sources = node.get("inputs")
    step["inputs"] = [str(item) for item in sources] if isinstance(sources, (list, tuple)) else []
    for key in ("op", "kind"):
        if not isinstance(step.get(key), str):
            step.pop(key, None)
    return step


def _laid_out(step: Mapping[str, Any], tensor: Any, exact: bool) -> Tuple[Optional[Any], str]:
    """``(a step's tensor as flat values in the order the structure shows the step in, "")``, or ``(None, why)``.

    ``why`` is empty when the tensor is simply not the step's (it is not numbers, or it does not hold as many
    values), and says what is wrong when the structure lays the step out in a way that does not fit its tensor:
    a shape that is not the tensor's own (with ``exact``), or an order of axes that is not an order of them.
    Such a step is left out; nothing a structure says raises here.
    """
    numpy = _numpy()
    array = numpy.asarray(tensor)
    units = _count(step.get("units")) or 0
    if array.dtype.kind not in "fiub" or not units or array.size != units:
        return None, ""
    args = step.get("args")
    args = args if isinstance(args, Mapping) else {}
    described = args.get("tensor")
    if isinstance(described, Mapping) and described.get("shape") is not None:
        native = _counts(described["shape"])
        own = [int(dim) for dim in array.shape]
        if native is None:
            return None, "The shape the structure gives its tensor is not a shape."
        if _product(native) != array.size:
            return None, ""
        if exact and own != native:
            return None, f"The structure gives its tensor the shape {native}, and the tensor in the model has {own}."
        array = array.reshape(native)
        perm = described.get("perm")
        if perm:
            order = _counts(perm)
            if order is None or sorted(order) != list(range(len(native))):
                return None, "The order the structure takes its tensor's axes in is not an order of those axes."
            array = array.transpose(order)
    else:
        shown = _counts(args.get("shape")) or []
        channels_last = array.ndim >= 3 and shown == [*array.shape[2:], array.shape[1]]
        if channels_last and (array.ndim == 4 or shown != list(array.shape[1:])):
            array = numpy.moveaxis(array, 1, -1)  # channels last, as the step is shown
    return numpy.ascontiguousarray(array, dtype=float).reshape(-1), ""


def shown_values(step: Mapping[str, Any], tensor: Any, *, exact: bool = False) -> Optional[Any]:
    """A step's tensor as flat values in the order the structure shows the step in.

    ``None`` when the tensor is not that step's: it does not hold as many values or, with ``exact``, it has
    another shape than the structure says the step's tensor has. ``None`` too when the order of axes the
    structure gives is not an order of the tensor's axes.

    A step of a model graph says how its tensor is laid out: ``args.tensor`` has the tensor's own shape, batch
    axis included, and the order to take its axes in (``perm``). That turns a map written channels first
    into height by width by channels, and attention weights into queries by keys by heads. A step of any
    other structure shows a map channels last, where PyTorch has it channels first.
    """
    return _laid_out(step, tensor, exact)[0]


def stage_of(
    step: Mapping[str, Any],
    values: Any,
    *,
    detailed: bool = False,
    units: int = 16,
    channels: int = 8,
    max_side: int = 56,
    channel_start: Optional[int] = None,
) -> Dict[str, Any]:
    """One step's activations in the form the API gives its own stages (it keeps nothing in any other form).

    ``step`` is the step as the structure document has it (its ``id`` and the shape it is shown in, under
    ``args``); ``values`` are what it computed, in that shown order. A map (height by width by channels)
    reports its strongest channels and, when ``detailed``, their maps and a saliency map; anything else its
    strongest units and, when ``detailed``, every value (up to 4,096) and, for a table of positions by
    features, how strongly each position fires and the table itself.

    The heads of an attention layer (queries by keys by heads) all add up to the same, so they are ranked by
    their focus instead: the largest weight each query gives, averaged over the queries. ``channel_start``
    sends ``channels`` maps in their own order from that one on, where the strongest are sent without it.
    """
    numpy = _numpy()
    flat = numpy.asarray(values, dtype=float).reshape(-1)
    flat = numpy.where(numpy.isfinite(flat), flat, 0.0)
    shown = _counts((step.get("args") or {}).get("shape")) or []
    top = max(1, min(int(units), 64))
    stage: Dict[str, Any] = {
        "node_id": str(step.get("id")),
        "summary": _summary(flat, numpy),
        "active_fraction": round(float(numpy.count_nonzero(flat > 0)) / flat.size, 4) if flat.size else 0.0,
        "detailed": bool(detailed),
    }
    spatial = len(shown) == 3 and shown[0] * shown[1] > 1 and shown[0] * shown[1] * shown[2] == flat.size
    role = (step.get("args") or {}).get("role")
    if spatial:
        height, width, count = shown
        stage["shape"] = [height, width, count]
        grid = flat.reshape(height, width, count)
        magnitude = numpy.abs(grid)
        with numpy.errstate(over="ignore", invalid="ignore"):
            energy = _sendable(magnitude.sum(axis=(0, 1)), numpy)  # a sum too large for a float is the largest one
        attention = role == "attention"
        strength = _sendable(_mean(magnitude.max(axis=1), 0, numpy), numpy) if attention else energy
        order = numpy.argsort(-strength, kind="stable")
        stage["top_channels"] = [int(index) for index in order[:top]]
        if detailed:
            kept = max(1, min(int(channels), 64))
            stage["energy"] = [round(float(item), 6) for item in energy[:4096]]
            if attention:
                stage["focus"] = [round(float(item), 6) for item in strength[:4096]]
            stage["top_channels"] = [int(index) for index in order[: max(kept, top)][:64]]
            if channel_start is None:
                chosen = [int(channel) for channel in order[:kept]]
            else:
                first = max(0, min(int(channel_start), count))
                chosen = list(range(first, min(first + kept, count)))
                stage["channel_start"] = first
                stage["channel_count"] = int(count)
            stage["maps"] = [
                {
                    "channel": channel,
                    "energy": round(float(energy[channel]), 6),
                    "peak": round(float(magnitude[:, :, channel].max()), 6),
                    **({"focus": round(float(strength[channel]), 6)} if attention else {}),
                    **_picture(_shrunk(grid[:, :, channel], int(max_side), numpy), numpy),
                }
                for channel in chosen
            ]
            stage["saliency"] = _picture(_shrunk(_mean(magnitude, 2, numpy), int(max_side), numpy), numpy)
    else:
        order = numpy.argsort(-numpy.abs(flat), kind="stable")[:top]
        stage["top_units"] = [{"index": int(index), "value": round(float(flat[index]), 6)} for index in order]
        table = len(shown) == 2 and shown[0] >= 2 and shown[1] >= 1 and shown[0] * shown[1] == flat.size
        if table:
            stage["shape"] = [shown[0], shown[1]]
        if detailed:
            if flat.size <= VALUE_LIMIT:
                stage["values"] = _picture(flat, numpy)
            if table:
                grid = flat.reshape(shown[0], shown[1])
                with numpy.errstate(over="ignore", invalid="ignore"):
                    firing = numpy.sqrt(numpy.mean(grid * grid, axis=1))
                    largest = float(numpy.abs(grid).max()) if grid.size else 0.0
                    if not numpy.isfinite(firing).all() and largest > 0:
                        # Squares too large for a float: taken at a size where they fit.
                        firing = largest * numpy.sqrt(numpy.mean((grid / largest) ** 2, axis=1))
                stage["rows"] = [round(float(item), 6) for item in _sendable(firing, numpy)[:4096]]
                stage["matrix"] = _picture(_shrunk_table(grid, max(int(max_side), 8) * 4, numpy), numpy)
    if isinstance(role, str) and role:
        stage["role"] = role
    return stage


def _answer_size(value: Any) -> Tuple[int, int]:
    """``(values, most bytes)`` of an answer: values as the API counts them (every number, string, list and
    object is one), and at most how many bytes of JSON it is (it is never more; usually it is less)."""
    count, most, stack = 0, 0, [value]
    while stack:
        item = stack.pop()
        count += 1
        if isinstance(item, Mapping):
            children: Iterable[Any] = item.values()
            most += 2 + sum(6 * len(str(key)) + 4 for key in item)
        elif isinstance(item, (list, tuple)):
            children = item
            most += 2
        else:
            most += 6 * len(item) + 3 if isinstance(item, str) else 26
            continue
        for child in children:
            if isinstance(child, (Mapping, list, tuple, str)):
                stack.append(child)
            else:
                count += 1
                most += 26  # a number with its sign, 17 digits, a point, an exponent, and a comma
    return count, most


def _too_large(body: Mapping[str, Any], job: Mapping[str, Any]) -> Optional[str]:
    """Why the API would refuse this answer for its size, or ``None`` when it fits."""
    from ._http import encode_json

    result = body.get("result")
    values, most = _answer_size(result)
    rows = (result or {}).get("predictions") if isinstance(result, Mapping) else None
    per_row = ""
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        per_row = f" ({len(rows):,} rows of {len(rows[0]):,} numbers)"
    kind = str(job.get("kind"))
    fewer = "Ask for fewer rows at a time." if kind == "predict" else "Ask for fewer steps at a time (nodes=[...])."
    if values > MAX_ANSWER_VALUES:
        return (
            f"This answer to a {kind} job holds {values:,} values{per_row}, and Kaimei takes at most "
            f"{MAX_ANSWER_VALUES:,} in an agent's answer. {fewer}"
        )
    if most > MAX_ANSWER_BYTES:
        size = len(encode_json(body))
        if size > MAX_ANSWER_BYTES:
            return (
                f"This answer to a {kind} job is {size / 2**20:,.0f} MiB of JSON{per_row}, and Kaimei takes at most "
                f"{MAX_ANSWER_BYTES / 2**20:,.0f} MiB in an agent's answer. {fewer}"
            )
    return None


def _fixed_value(value: Any, name: str) -> Any:
    """An input's fixed value as an array: as given, or as an upload writes one (``{dtype, shape, data}``, ``$b64``).

    ``name`` is what the model file calls the input, which is what ``Agent(inputs=...)`` is keyed by; the
    structure's own name for it (``in.<name>``) is not.
    """
    numpy = _numpy()
    if isinstance(value, Mapping):
        if "$shape" in value and "data" not in value and "$b64" not in value:
            raise ValueError(
                f"The file's input '{name}' is always given a value that the upload kept at home "
                "(weights=\"summary\" sends its shape only), so it cannot be read from the structure. Upload "
                f"the structure with weights=\"full\", or give the agent the value: inputs={{'{name}': lambda row: "
                "the array}."
            )
        from .graphs import _described

        return _described(value, name, numpy)
    return numpy.asarray(value)


def _reason(exc: BaseException) -> str:
    """What went wrong, short enough for an answer and written so that it can always be sent."""
    try:
        text = f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001 - an error that cannot say what it is still has a name
        text = type(exc).__name__
    return text.encode("utf-8", "replace").decode("utf-8")[:2000]


def _whole_or_none(value: Any) -> Optional[int]:
    """A whole number that is not negative, or ``None``: what a job says is read, not trusted."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _thinned(items: Sequence[str], limit: int) -> List[str]:
    """Evenly spaced members, always keeping the last one."""
    if len(items) <= limit:
        return list(items)
    numpy = _numpy()
    return [items[int(index)] for index in numpy.unique(numpy.linspace(0, len(items) - 1, limit).round().astype(int))]


def default_detail(nodes: Sequence[Mapping[str, Any]], head: Optional[str] = None) -> List[str]:
    """The steps sent in full when a request names none: the API's own choice for a model it runs.

    Maps that compute features (not pools, pads or pass-throughs), leaving out a convolution whose only
    reader is its own normalization or activation (that reader has the finished map), thinned evenly to 24;
    the vectors, thinned to eight; and the output.
    """
    consumers: Dict[str, List[Mapping[str, Any]]] = {}
    for node in nodes:
        for source in node.get("inputs") or []:
            consumers.setdefault(str(source), []).append(node)
    order = [node for node in nodes if node.get("op") != "input"]
    maps: List[str] = []
    for node in order:
        if _spatial(node) is None or node.get("op") in _RESAMPLING:
            continue
        readers = consumers.get(str(node.get("id")), [])
        if (
            len(readers) == 1
            and (
                readers[0].get("op") in ("batchnorm", "BatchNormalization")
                or readers[0].get("op") in _ACTIVATIONS
                or readers[0].get("kind") == "activation"
            )
            and _spatial(readers[0])
        ):
            continue
        maps.append(str(node.get("id")))
    vectors = [
        str(node.get("id"))
        for node in order
        if _spatial(node) is None and node.get("op") not in _RESAMPLING and node.get("kind") != "reshape"
    ]
    chosen = _thinned(maps, DETAILED_MAPS) + _thinned(vectors, DETAILED_VECTORS)
    if head and head not in chosen and any(str(node.get("id")) == head for node in order):
        chosen.append(head)
    return chosen


def hook_plan(
    steps: Sequence[Mapping[str, Any]], modules: Iterable[str]
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    """Which steps of a structure a module's calls can speak for: ``(own, under)``, both by module name.

    A forward hook sees what a module returns, each time it is called. That is one step's values only
    where the step is the module:

    * ``own[name]``: the steps whose path is the module itself, in the structure's order. The module's
      i-th call is the i-th of them, when it was called exactly as many times as there are such steps.
    * ``under[name]``: for a module with no step of its own, every step anywhere below its path, in order.
      A module called once speaks for the last of them, the step that produces what the module returns. A
      module called as many times as it has steps (an activation used twice in a block) speaks for one
      of them each time. PyTorch's exporter writes a module's second call under ``name_1``, its third
      under ``name_2``: the steps then come as one list for each call, and each call speaks for the last
      step of its own list.

    No other step is given a module's values: the arithmetic a ``forward`` writes between its modules has
    no hook to see it, and a wrong step's values under a step's name are worse than none.
    """
    known = {str(name) for name in modules if name}
    own: Dict[str, List[str]] = {}
    scopes: Dict[str, Dict[int, List[str]]] = {}
    for step in steps:
        path = str(step.get("path") or step.get("id") or "")
        if not path:
            continue
        step_id = str(step["id"])
        if path in known:
            own.setdefault(path, []).append(step_id)
        parts = path.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            name, call = ".".join(parts[:cut]), 0
            if name not in known:
                numbered = _CALL_SCOPE.match(name)
                if numbered is None or numbered.group(1) not in known:
                    continue
                name, call = numbered.group(1), int(numbered.group(2))
            scopes.setdefault(name, {}).setdefault(call, []).append(step_id)
    under: Dict[str, List[List[str]]] = {}
    for name, by_call in scopes.items():
        # A module with steps of its own is told by those; one whose calls are not all there is not told at all.
        if name not in own and sorted(by_call) == list(range(len(by_call))):
            under[name] = [by_call[call] for call in range(len(by_call))]
    return own, under


# --- what a module's hooks may pass on: never the model's own numbers ---------------------------


def _kept_at(tensor: Any) -> int:
    """Where the memory a tensor is kept in starts, whichever part of it the tensor is; 0 when it has none."""
    try:
        return int(tensor.data_ptr()) - int(tensor.storage_offset()) * int(tensor.element_size())
    except Exception:  # noqa: BLE001 - a tensor with no memory of its own kind (a sparse one)
        return 0


def _written(tensor: Any) -> int:
    """How many times a tensor has been written in place, as PyTorch counts; -1 where it does not count."""
    try:
        return int(tensor._version)
    except Exception:  # noqa: BLE001 - a tensor made in inference mode keeps no count
        return -1


def _tensors_in(value: Any, most: int = 64) -> Optional[List[Any]]:
    """Every tensor in what a module was called with or gave: itself, or in tuples, lists and mappings.

    ``None`` when it cannot be told: it holds an object of another kind, which may carry anything, or more
    than ``most`` tensors.
    """
    found: List[Any] = []
    stack, seen = [value], 0
    while stack:
        item = stack.pop()
        seen += 1
        if seen > 16 * most:
            return None
        if hasattr(item, "data_ptr") and hasattr(item, "detach"):
            found.append(item)
            if len(found) > most:
                return None
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif not isinstance(item, _PLAIN):
            return None
    return found


def _listen(layer: Any, hook: Callable[[Any, Any, Any, Any], None]) -> Any:
    """A forward hook on ``layer`` called with ``(layer, what it was called with, by name too, what it gave)``.

    PyTorch before 2.0 does not say what a module was called with by name: the hook is then given ``None``
    there, and takes it as not known.
    """
    try:
        return layer.register_forward_hook(hook, with_kwargs=True)
    except TypeError:

        def unnamed(module: Any, inputs: Any, output: Any) -> None:
            hook(module, inputs, None, output)

        return layer.register_forward_hook(unnamed)


#: The mode a module is run under to follow its operations, once made: ``[the class]``, or ``[None]`` for a
#: PyTorch that has none.
_FOLLOWING: List[Any] = []


def _following() -> Any:
    """A mode of PyTorch's under which every operation it runs is shown to an :class:`_OwnNumbers` as it
    happens (a ``TorchDispatchMode``, entered this way since PyTorch 1.13), or ``None`` where there is no such thing.

    The operation itself runs as it would have: the mode looks at what went in and what came out. Should
    that look ever fail, the run goes on and the :class:`_OwnNumbers` is told it lost track.
    """
    if not _FOLLOWING:
        try:
            from torch.utils._python_dispatch import TorchDispatchMode

            class Following(TorchDispatchMode):
                own: Any = None

                def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
                    out = func(*args, **(kwargs or {}))
                    try:
                        self.own._saw(func, args, kwargs or {}, out)
                    except Exception:  # noqa: BLE001 - the model's run goes on; what it gave is not told apart
                        if self.own is not None:
                            self.own.lost = True
                    return out

            with Following():
                pass
            _FOLLOWING.append(Following)
        except Exception:  # noqa: BLE001 - a PyTorch from before operations could be followed this way
            _FOLLOWING.append(None)
    return _FOLLOWING[0]


def _kept_by(module: Any, torch: Any) -> List[Any]:
    """Every tensor a module, or a module registered inside it, keeps as a plain attribute: a table made when
    it was built, or something worked out from its weights once and kept for the next call. The attribute is
    the tensor, or a list, a tuple or a dict that holds it (one level, no deeper).

    A tensor kept anywhere else is not found: deeper than that, in a module that is not registered, in a
    variable a function closes over, in a global, or in an object of another kind.
    """
    found: List[Any] = []
    for part in module.modules():
        for value in vars(part).values():
            inside = value.values() if isinstance(value, Mapping) else value if isinstance(value, (list, tuple)) else (value,)
            found.extend(item for item in inside if isinstance(item, torch.Tensor))
    return found


class _OwnNumbers:
    """Which tensors of one run of a module hold only the model's own numbers, and nothing computed from the
    input: what an agent must not send, whichever step Kaimei's structure says they are.

    The model's own numbers are what it keeps: its parameters and buffers, and any tensor it keeps as a
    plain attribute (:func:`_kept_by`: a table made when it was built, something worked out from its weights
    at an earlier look and kept). What the module is called with for the row is never among them
    (:meth:`fed`), though a module may have kept that very tensor. A tensor that is any part of one of those (a
    table of positions cut to the length of a text) is kept in the same memory. From there every operation
    PyTorch runs is followed (see :meth:`following`): what an operation gives is the model's own when every
    tensor it reads is, and so is a tensor made from nothing (``torch.arange(length)``) or from the shape of
    another alone (``torch.zeros_like(ids)``). So the embedding of each position, a weight made by a
    parametrization and a mask worked out from a length are the model's own, however many steps they went
    through, and whatever reads the input, even once, is not. Memory of the model's own that is written
    with something else (``state.copy_(x)``) stops being the model's own. Each such tensor is marked, and
    the marks go when the run is over; none is kept alive for it, so a run takes the memory it always took.
    (What the model kept when the run began is held until it ends, so that nothing made in the run can be
    kept where one of those was.)

    What is followed is tensors. A number taken out of one and put into another by hand (``x.max().item()``)
    is not, so a tensor made from such a number alone counts as the model's own.

    A PyTorch that cannot be followed (before 1.13) is listened to through the modules' hooks alone
    (:meth:`called_with`): what a module gives is then the model's own when it was called with nothing but
    the model's own, or is itself a part of a parameter or a buffer. That cannot tell a tensor made on the
    way from one made from the input, so it keeps back less.
    """

    def __init__(self, module: Any, torch: Any) -> None:
        self._parameter = torch.nn.Parameter
        self._tensor = torch.Tensor
        #: Where what the model keeps is kept: the start of each piece of memory, with the tensor kept in it
        #: and how often that had been written when this was made. Listened to through hooks alone, also what a
        #: module gave from the model's own numbers. Each is held for the run, so the memory stays that tensor's.
        self._own: Dict[int, Tuple[Any, int]] = {}
        #: Memory that held the model's own numbers until something else was written into it.
        self._spoiled: Set[int] = set()
        for tensor in [*module.parameters(), *module.buffers(), *_kept_by(module, torch)]:
            self._keep(tensor)
        #: What a followed run marks each tensor computed from the model's own numbers with, and those it
        #: marked (weakly: they live as long as the model keeps them, and no longer), also by where each is
        #: kept: while a marked tensor lives, the memory it is kept in is its own.
        self._token = object()
        self._marked: List[Any] = []
        self._places: Dict[int, Any] = {}
        #: What is known of each operation met: its name, the arguments it only writes its answer into, and
        #: every argument it writes into, each by its place and by its name.
        self._plans: Dict[Any, Tuple[str, Any, Any]] = {}
        #: Whether every operation of the run is followed, and whether one could not be made sense of.
        self.followed = False
        self.lost = False
        kind = _following()
        self._mode = kind() if kind is not None else None
        if self._mode is not None:
            self._mode.own = self
        #: Whether this PyTorch lets a run be followed at all.
        self.can_follow = self._mode is not None

    def _keep(self, tensor: Any) -> None:
        where = _kept_at(tensor)
        if where and where not in self._own:
            self._own[where] = (tensor, _written(tensor))
            self._spoiled.discard(where)

    def _spoil(self, tensor: Any) -> None:
        where = _kept_at(tensor)
        if where:
            self._own.pop(where, None)
            self._spoiled.add(where)

    def fed(self, given: Any) -> None:
        """What the module is called with for the row is the input, wherever else it is kept: a module that
        keeps what it was last called with (``self.last = x``) has not made it the model's own."""
        for tensor in self._tensors(given, []):
            self._spoil(tensor)

    def holds(self, tensor: Any) -> bool:
        """Whether ``tensor`` is known to hold only the model's own numbers."""
        where = _kept_at(tensor)
        if where in self._spoiled:
            return False
        if getattr(tensor, _OWN_MARK, None) is self._token:
            return True
        known = self._own.get(where)
        if known is not None:
            # Followed, a write is seen as it happens. Listened to, it shows in how often the memory was written.
            return self.followed or _written(known[0]) == known[1]
        marked = self._places.get(where)
        if marked is not None and marked() is not None:
            return True  # kept where a marked tensor is: another name for the same numbers (``marked.data``)
        return isinstance(tensor, self._parameter)

    def settle(self, output: Any) -> None:
        """What a module gave from the model's own numbers alone is the model's own from here on (for a run
        that is listened to through hooks; a followed run marks what each operation gives)."""
        for item in _tensors_in(output) or []:
            self._keep(item)

    def called_with(self, inputs: Any, named: Any) -> bool:
        """Whether a module was called with nothing computed from the input: with nothing at all, or with
        tensors that are all the model's own. ``named`` is ``None`` when it is not known what the module was
        called with by name; it is then taken to be the model's own only when every tensor it was called with
        in order is, and there is one. This is what is gone by when the run cannot be followed."""
        if named is not None and not inputs and not named:
            return True
        given = _tensors_in(inputs if named is None else (inputs, named))
        return bool(given) and all(self.holds(item) for item in given or [])

    # --- every operation of the run, followed ---------------------------------------------------

    def following(self) -> Any:
        """What the module is run under: every operation is then seen as it happens (``followed``). With a
        PyTorch that has no way to, nothing: the hooks are then all there is.

        What was written into the model's own memory since this was made (a row put into a buffer before the
        run) is not the model's own.
        """
        for tensor, written in list(self._own.values()):
            if _written(tensor) != written:
                self._spoil(tensor)
        self.followed = self._mode is not None
        return self._mode if self._mode is not None else contextlib.nullcontext()

    def done(self) -> None:
        """The run is over: the marks it left on tensors go with it, and nothing the model kept is held on to."""
        for held in self._marked:
            tensor = held()
            if tensor is not None:
                try:
                    delattr(tensor, _OWN_MARK)
                except Exception:  # noqa: BLE001 - a mark that cannot be taken off is one no later run goes by
                    pass
        self._marked, self._places, self._own = [], {}, {}

    def _mark(self, tensor: Any) -> None:
        """``tensor`` holds the model's own numbers: an operation that read nothing else gave it, or it is a
        part of one that does."""
        where = _kept_at(tensor)
        if where:
            self._spoiled.discard(where)
            if where in self._own:
                return  # a part of a parameter or a buffer: known by where it is kept
        if getattr(tensor, _OWN_MARK, None) is not self._token:
            setattr(tensor, _OWN_MARK, self._token)
            self._marked.append(weakref.ref(tensor))
            if where:
                self._places[where] = self._marked[-1]

    def _tensors(self, value: Any, found: List[Any]) -> List[Any]:
        """The tensors among an operation's arguments, or in what it gave: themselves, or in lists of them."""
        if isinstance(value, self._tensor):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._tensors(item, found)
        elif isinstance(value, Mapping):
            for item in value.values():
                self._tensors(item, found)
        return found

    def _plan(self, func: Any) -> Tuple[str, Any, Any]:
        plan = self._plans.get(func)
        if plan is None:
            schema = getattr(func, "_schema", None)
            name = str(getattr(schema, "name", "") or getattr(func, "__name__", "") or func).split("::")[-1].split(".")[0]
            out_only: Set[Any] = set()
            written: Set[Any] = set()
            for index, argument in enumerate(getattr(schema, "arguments", None) or []):
                alias = getattr(argument, "alias_info", None)
                if alias is None or not getattr(alias, "is_write", False):
                    continue
                places = {str(argument.name)} if getattr(argument, "kwarg_only", False) else {str(argument.name), index}
                written |= places
                if getattr(argument, "is_out", False) or str(argument.name) == "out":
                    out_only |= places
            if schema is None:
                # Not known what it writes into: an answer asked for in ``out``, and itself when its name says so.
                written |= {"out"} | ({0} if name.endswith("_") else set())
                out_only.add("out")
            plan = self._plans[func] = (name, frozenset(out_only), frozenset(written))
        return plan

    def _saw(self, func: Any, args: Any, kwargs: Mapping[str, Any], out: Any) -> None:
        """One operation, as it happened. What it gave is the model's own when all it read was, or when it read
        only a shape. Otherwise the model's own memory it wrote into no longer is, and of what it gave only a
        part of the model's own that it cut out without changing it (``table.expand_as(x)``) still is."""
        name, out_only, written = self._plan(func)
        read: List[Any] = []
        wrote: List[Any] = []
        for place, value in [*enumerate(args), *kwargs.items()]:
            tensors = self._tensors(value, [])
            if place in written:
                wrote.extend(tensors)
            if place not in out_only:
                read.extend(tensors)
        gave = self._tensors(out, [])
        if name in _SHAPE_ONLY or all(self.holds(item) for item in read):
            for item in gave:
                self._mark(item)
            return
        for item in wrote:
            if self.holds(item):
                self._spoil(item)
        kept = {_kept_at(item) for item in read if self.holds(item)} - {0}
        for item in gave if kept else []:
            if _kept_at(item) in kept:
                self._mark(item)


# --- the model's ONNX file, run here ----------------------------------------------------------


def _read_from_outside(body: Any) -> Set[str]:
    """The tensors a branch or a loop's body (a graph inside a step of an ONNX file) reads and does not make."""
    own = {str(info.name) for info in body.input} | {str(item.name) for item in body.initializer}
    reads = {str(info.name) for info in body.output}
    for node in body.node:
        own.update(str(name) for name in node.output if name)
        reads.update(str(name) for name in node.input if name)
        for attribute in node.attribute:
            bodies = [attribute.g] if attribute.HasField("g") else []
            for inner in [*bodies, *attribute.graphs]:
                reads |= _read_from_outside(inner)
    return reads - own


class _OnnxFile:
    """A model's ONNX file run by ONNX Runtime: its outputs and, when asked, any tensor inside it.

    ``memory`` is what the sessions kept ready may take, in bytes (``SESSION_BYTES`` when it is ``None``).
    """

    def __init__(self, source: Any, providers: Optional[Sequence[Any]] = None, memory: Optional[int] = None) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "An agent that runs a model's ONNX file needs ONNX Runtime: pip install onnxruntime "
                "(onnxruntime-gpu on a machine with a GPU)."
            ) from exc
        try:
            import onnx
        except ImportError as exc:
            raise ImportError(
                "An agent that runs a model's ONNX file needs the onnx package as well: pip install onnx "
                "(or pip install 'kaimei[onnx]')."
            ) from exc
        self._runtime = onnxruntime
        self._onnx = onnx
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.path: Optional[str] = None
        if isinstance(source, (str, os.PathLike)):
            self.path = os.fspath(source)
            # A large model keeps its weights in files beside it: they stay there, and ONNX Runtime reads them.
            self.model = onnx.load(self.path, load_external_data=False)
        elif isinstance(source, (bytes, bytearray, memoryview)):
            self.model = onnx.load_model_from_string(bytes(source))
        elif type(source).__name__ == "ModelProto":
            self.model = source
        else:
            raise TypeError("onnx is the model's ONNX file: its path, its bytes, or an onnx.ModelProto.")
        from .graphs import without_trailing_empties

        # Inputs left out at the end of a node under empty names stop ONNX Runtime on some nodes; a copy without them.
        written = without_trailing_empties(self.model)
        self._rewritten = written is not self.model
        self.model = written
        graph = self.model.graph
        numpy = _numpy()
        constants = {str(item.name) for item in graph.initializer}
        self.inputs: List[Tuple[str, Any, List[Optional[int]]]] = []
        for info in graph.input:
            if str(info.name) in constants:
                continue
            kind = info.type.tensor_type
            dims: List[Optional[int]] = [
                int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else None for dim in kind.shape.dim
            ]
            dtype = numpy.dtype(onnx.helper.tensor_dtype_to_np_dtype(int(kind.elem_type)))
            self.inputs.append((str(info.name), dtype, dims))
        self.outputs = [str(info.name) for info in graph.output]
        self.floats = [
            str(info.name) for info in graph.output if int(info.type.tensor_type.elem_type) in (1, 10, 11)
        ]
        self.tensors: Set[str] = {str(name) for node in graph.node for name in node.output if name} | set(self.outputs)
        self._producers: Dict[str, Any] = {str(name): node for node in graph.node for name in node.output if name}
        self.prediction = self._prediction_tensor()
        #: The file's class, as a converted classifier gives it beside its scores (its ``predict``): an output of
        #: whole numbers or strings. ``None`` for a file without one.
        self.label: Optional[str] = next(
            (
                str(info.name)
                for info in graph.output
                if info.type.HasField("tensor_type") and int(info.type.tensor_type.elem_type) in _LABEL_TYPES
            ),
            None,
        )
        self._outside = any(int(item.data_location) == 1 for item in graph.initializer)
        self.weight_bytes = self._weight_bytes()
        self.memory = SESSION_BYTES if memory is None else int(memory)
        #: Sessions by most recent use (the last used last), and those of the look being answered.
        self._sessions: "OrderedDict[Tuple[str, ...], Any]" = OrderedDict()
        self._planned: Set[Tuple[str, ...]] = set()
        self._kept = SESSIONS_KEPT
        #: The shape the file says each of its tensors has (see ``_declared_shapes``), once read; and how many
        #: values each tensor was found to hold when it was read, by the shapes of the inputs and its name.
        self._declared: Optional[Dict[str, List[Any]]] = None
        self._measured: Dict[Tuple[Any, str], int] = {}

    def _prediction_tensor(self) -> Optional[str]:
        """The tensor a prediction is read from when no structure says: as the importer reads the file.

        The first output that holds floats, or the last when a classical-ML operator made them (a converted
        classifier gives its label first and its probabilities second). An output that is a list of
        dictionaries (scikit-learn's ``ZipMap``) is read in the tensor it was made from. ``None`` when no
        output holds floats: a class label or a token id is not a score.
        """
        from .graphs import _prediction_output

        tensor = _prediction_output(self.model, None)[1]
        if tensor is not None:
            return str(tensor)
        untyped = [
            str(info.name)
            for info in self.model.graph.output
            if not info.type.HasField("tensor_type") or not int(info.type.tensor_type.elem_type)
        ]
        # An output the file says nothing of may be anything: running it says.
        found = self.floats or untyped
        return found[0] if found else None

    def _weight_bytes(self) -> int:
        """What the file's weights take in memory, wherever they are kept."""
        numpy = _numpy()
        total = 0
        for item in self.model.graph.initializer:
            try:
                size = numpy.dtype(self._onnx.helper.tensor_dtype_to_np_dtype(int(item.data_type))).itemsize
            except (KeyError, TypeError, ValueError):
                size = 8
            total += _product(item.dims) * int(size)
        return total

    def ends(self, tensor: Optional[str]) -> Optional[str]:
        """What the node that makes ``tensor`` makes of the model's scores: ``"none"`` (probabilities: a
        ``Softmax``, a ``Sigmoid``, a classical classifier's own transform) or ``"softmax"`` (a ``LogSoftmax``),
        ``"scores"`` for any other node, and ``None`` when the file does not say.

        What a converted classifier gives beside its class (what a ``ZipMap`` pairs with the classes) is its
        probabilities or, for one without them, its scores: the steps that made it are looked back through for
        one that says which (``_what_it_makes``). When none does (a forest's averaged votes) its numbers say.
        """
        if tensor is None:
            return None
        if self.classifier_output(tensor):
            for node in self._made_by(tensor):
                said = _what_it_makes(node)
                if said is not None:
                    return said
            return None
        name = tensor
        for _hop in range(8):
            node = self._producers.get(name)
            if node is None:
                return None  # an input of the file, or a tensor it does not make
            if node.op_type in ("Softmax", "Sigmoid"):
                return "none"
            if node.op_type == "LogSoftmax":
                return "softmax"
            if node.op_type in _LAYOUT and node.input:
                name = str(node.input[0])
                continue
            if str(node.domain) == "ai.onnx.ml":
                transform = next((item.s for item in node.attribute if item.name == "post_transform"), b"NONE")
                return "none" if transform in _PROBABILITY_TRANSFORMS else None
            return "scores"
        return None

    def classifier_output(self, tensor: Optional[str]) -> bool:
        """Whether ``tensor`` is what a converted classifier gives beside its class: what a ``ZipMap`` pairs with
        the classes, or the scores of a file of classical-ML operators that also gives its label."""
        if tensor is None:
            return False
        graph = self.model.graph
        if any(node.op_type == "ZipMap" and node.input and str(node.input[0]) == tensor for node in graph.node):
            return True
        return self.label is not None and any(str(node.domain) == "ai.onnx.ml" for node in graph.node)

    def _made_by(self, tensor: str) -> List[Any]:
        """The steps a converted classifier's ``tensor`` was made by, the nearest first. The look back stops at a
        classical classifier and at an operator that computes scores from weights: those made the scores."""
        found: List[Any] = []
        seen: Set[str] = set()
        queue = [tensor]
        while queue and len(found) < 512:
            name = queue.pop(0)
            node = self._producers.get(name)
            if name in seen or node is None or any(node is item for item in found):
                seen.add(name)
                continue
            seen.add(name)
            found.append(node)
            if node.op_type in _WEIGHED or (str(node.domain) == "ai.onnx.ml" and node.op_type in _CLASSIFIERS):
                continue
            queue.extend(str(item) for item in node.input if item)
        return found

    def classes(self, tensor: Optional[str]) -> Optional[List[Any]]:
        """The classes the columns of a converted classifier's ``tensor`` are for, in order, as the file names them
        (its ``ZipMap``, its classifier, or the list its label is looked up in). ``None`` when it does not say."""
        if tensor is None:
            return None
        for node in self.model.graph.node:
            if node.op_type == "ZipMap" and node.input and str(node.input[0]) == tensor:
                return _classlabels(node)
        for node in self._made_by(tensor):
            if str(node.domain) == "ai.onnx.ml" and node.op_type in _CLASSIFIERS:
                return _classlabels(node)
        constants = {str(item.name): item for item in self.model.graph.initializer}
        name = self.label
        for _hop in range(8):
            node = self._producers.get(str(name))
            if node is None:
                return None
            if node.op_type == "ArrayFeatureExtractor" and node.input and str(node.input[0]) in constants:
                from onnx import numpy_helper

                listed = numpy_helper.to_array(constants[str(node.input[0])]).reshape(-1)
                return list(listed.tolist())  # what a label made from the highest score is looked up in
            if str(node.domain) == "ai.onnx.ml" and node.op_type in _CLASSIFIERS:
                return _classlabels(node)
            if node.op_type not in _LAYOUT or not node.input:
                return None
            name = str(node.input[0])
        return None

    def votes(self, tensor: Optional[str]) -> bool:
        """Whether a converted classifier's class is not its highest score: a support vector machine without
        probabilities for three classes or more, which takes the class with the most votes of its pairs of
        classes. Votes can tie, and the tie goes to the first class, where the highest score may be another."""
        if tensor is None:
            return False
        for node in self._made_by(tensor):
            if str(node.domain) == "ai.onnx.ml" and node.op_type in _CLASSIFIERS:
                return node.op_type == "SVMClassifier" and _what_it_makes(node) == "scores" and len(_classlabels(node) or []) > 2
        return False

    def computed_from(self, inputs: Iterable[str]) -> Set[str]:
        """The tensors the file computes from what these inputs of it hold: what a step gives that reads one of
        them, or a tensor computed from one, itself or in a branch or a loop's body inside it.

        Everything else in the file is the model's own: an initializer (its weights), a ``Constant``, and
        whatever is computed from those alone. So is what follows from an input's shape alone: a step that
        reads only a shape (``Shape``, ``Size``) computes nothing from the values, and position embeddings cut
        to a text's length are the same rows of the table whatever the text says.
        """
        reached = {str(name) for name in inputs}
        found: Set[str] = set()
        steps = []
        for node in self.model.graph.node:
            if str(node.domain) in ("", "ai.onnx") and node.op_type in _SHAPE_READERS:
                continue
            reads = {str(name) for name in node.input if name}
            for attribute in node.attribute:
                bodies = [attribute.g] if attribute.HasField("g") else []
                for body in [*bodies, *attribute.graphs]:
                    reads |= _read_from_outside(body)
            steps.append((reads, [str(name) for name in node.output if name]))
        changed = True
        while changed:  # a file lists its steps in the order they run, so one pass finds them all; a second confirms it
            changed = False
            waiting = []
            for reads, gives in steps:
                if reads & reached:
                    reached.update(gives)
                    found.update(gives)
                    changed = True
                else:
                    waiting.append((reads, gives))
            steps = waiting
        return found

    def key(self, names: Sequence[str]) -> Tuple[str, ...]:
        """The session that gives these tensors: the tensors among them that are not outputs of the file."""
        wanted = dict.fromkeys(str(name) for name in names)
        return tuple(name for name in wanted if name not in self.outputs)

    def keep(self, count: int) -> None:
        """Keep ``count`` sessions ready from now on, the file's own outputs' among them, as memory allows."""
        room = self.memory // (3 * self.weight_bytes + (16 << 20))
        self._kept = max(self._kept, min(int(count), int(room), MOST_SESSIONS))

    def plan(self, keys: Iterable[Tuple[str, ...]]) -> None:
        """The sessions the look being answered uses: none of them goes to make room for another of them while
        it runs. Nothing (``()``) when the look is done."""
        self._planned = set(keys)

    def _room_for(self, extra: Tuple[str, ...]) -> bool:
        """Whether the session just made for ``extra`` is kept, the one used longest ago going to make room for it.

        The plain session (the file's own outputs) is always kept. One the look being answered uses does not go
        for another of its own: when those are all there is room for, the new one is used for its run and let go,
        so a look larger than the memory makes again what does not fit, and only that.
        """
        while len(self._sessions) >= max(SESSIONS_KEPT, self._kept):
            stale = next((key for key in self._sessions if key != () and key not in self._planned), None)
            if stale is None:
                return extra == ()
            del self._sessions[stale]
        return True

    def _session(self, extra: Tuple[str, ...]) -> Any:
        found = self._sessions.get(extra)
        if found is not None:
            self._sessions.move_to_end(extra)
            return found
        # A session that reads tensors inside the file is for a look inside, and several are kept at once. Without
        # ONNX Runtime's arena each gives back what a run took when the run ends, and keeps only its weights.
        if not extra and self.path is not None and not self._rewritten:
            made = self._runtime.InferenceSession(self.path, self._options(), providers=self.providers)
        else:
            made = self._made(lambda: self._with_outputs(extra), bool(extra))
        if self._room_for(extra):
            self._sessions[extra] = made
        return made

    def _made(self, written: Callable[[], bytes], lean: bool, sure: bool = True) -> Any:
        """A session for the model as ``written()`` gives it. ``sure`` is for a model that is known to load: should
        an ONNX Runtime too old to be told where the weights are refuse it, the weights are read into memory."""
        if self._outside and self.path is not None:
            # The weights are in files beside the model, and a model given as bytes has no folder of its own.
            folder = os.path.dirname(os.path.abspath(self.path))
            try:
                options = self._options(lean)
                options.add_session_config_entry("session.model_external_initializers_file_folder_path", folder)
                return self._runtime.InferenceSession(written(), options, providers=self.providers)
            except Exception:  # noqa: BLE001 - an ONNX Runtime too old to be told the folder
                if not sure:
                    raise
                from onnx.external_data_helper import load_external_data_for_model

                load_external_data_for_model(self.model, folder)  # read into memory, once
                self._outside = False
        return self._runtime.InferenceSession(written(), self._options(lean), providers=self.providers)

    def _options(self, lean: bool = False) -> Any:
        options = self._runtime.SessionOptions()
        options.log_severity_level = 4  # what goes wrong is raised, and said by whoever catches it
        if lean:
            options.enable_cpu_mem_arena = False
        return options

    def _with_outputs(self, extra: Tuple[str, ...]) -> bytes:
        """The model with these tensors among its outputs, as bytes. The model itself is left as it was."""
        outputs = self.model.graph.output
        before = len(outputs)
        outputs.extend(self._onnx.helper.make_empty_tensor_value_info(name) for name in extra)
        try:
            return bytes(self.model.SerializeToString())
        finally:
            del outputs[before:]

    def run(self, feeds: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Any]:
        """The tensors called ``names`` for these inputs: outputs of the file, or tensors inside it."""
        wanted = list(dict.fromkeys(str(name) for name in names))
        session = self._session(self.key(wanted))
        return dict(zip(wanted, session.run(wanted, dict(feeds))))

    # --- how much a tensor holds: what the file says, never what a structure claims ------------------

    def sizes(self, feeds: Mapping[str, Any], names: Iterable[str]) -> Dict[str, int]:
        """How many values each of these tensors holds for these inputs, by what the file itself says.

        As it was found to hold the last time it was read for inputs of these shapes (see :meth:`learn`); else
        as the file declares its shape, with the sizes these inputs give the ones it leaves open; else as one
        run that asks for the shapes of the rest, and for none of the tensors, finds them (:meth:`shapes`).
        A tensor none of these tells of is left out: how much it holds is not known until it is read.
        """
        shaped = self._shapes_of(feeds)
        found: Dict[str, int] = {}
        unknown: List[str] = []
        declared: Optional[Dict[str, int]] = None
        for name in dict.fromkeys(str(item) for item in names):
            size = self._measured.get((shaped, name))
            if size is None:
                if declared is None:
                    declared = self._declared_sizes(feeds)
                size = declared.get(name)
            if size is None:
                unknown.append(name)
            else:
                found[name] = size
        for name, shape in (self.shapes(feeds, unknown) if unknown else {}).items():
            found[name] = self._measured[(shaped, name)] = _product(shape)
        return found

    def learn(self, feeds: Mapping[str, Any], read: Mapping[str, Any]) -> None:
        """Tensors as they were read for these inputs: how much each holds is known from here on."""
        if len(self._measured) > 1_000_000:
            self._measured.clear()  # inputs of ever new shapes: what is known of the old ones is let go
        shaped = self._shapes_of(feeds)
        for name, value in read.items():
            size = getattr(value, "size", None)
            if isinstance(size, int) and hasattr(value, "shape"):
                self._measured[(shaped, str(name))] = size

    @staticmethod
    def _shapes_of(feeds: Mapping[str, Any]) -> Tuple[Any, ...]:
        numpy = _numpy()
        return tuple(sorted((str(name), tuple(int(dim) for dim in numpy.shape(value))) for name, value in feeds.items()))

    def _declared_shapes(self) -> Dict[str, List[Any]]:
        """The shape of each tensor as the file says it: written in it (most exporters write them), and worked
        out from its steps where it is not (ONNX's own shape inference, over a copy without the weights). A
        size the file leaves open is its name there (``"batch"``), or ``None`` when it has none."""
        if self._declared is None:
            graph = self.model.graph
            found: Dict[str, List[Any]] = {}

            def read(infos: Iterable[Any]) -> None:
                for info in infos:
                    kind = info.type
                    if str(info.name) in found or not kind.HasField("tensor_type") or not kind.tensor_type.HasField("shape"):
                        continue
                    found[str(info.name)] = [
                        int(dim.dim_value)
                        if dim.HasField("dim_value") and dim.dim_value > 0
                        else (str(dim.dim_param) if dim.HasField("dim_param") and dim.dim_param else None)
                        for dim in kind.tensor_type.shape.dim
                    ]

            read([*graph.value_info, *graph.output])
            if any(name not in found for name in self._producers):
                try:
                    from onnx import shape_inference

                    light = self._without_weights()
                    try:
                        inferred = shape_inference.infer_shapes(light, strict_mode=False, data_prop=True)
                    except TypeError:  # an onnx from before shape inference took these
                        inferred = shape_inference.infer_shapes(light)
                    read(inferred.graph.value_info)
                except Exception:  # noqa: BLE001 - a file its own library cannot work the shapes of: they are asked for
                    pass
            self._declared = found
        return self._declared

    def _without_weights(self) -> Any:
        """A copy of the model to work its shapes out from: its steps, and of its weights only those small enough
        to be a shape or an axis. Every other weight is there by its type and shape alone."""
        graph = self.model.graph
        if sum(node.ByteSize() for node in graph.node) > (64 << 20):
            raise ValueError("The file keeps its weights in its steps: too much to copy for this.")
        light = self._onnx.ModelProto()
        light.ir_version = self.model.ir_version
        light.opset_import.extend(self.model.opset_import)
        light.functions.extend(getattr(self.model, "functions", []))
        light.graph.name = graph.name
        light.graph.node.extend(graph.node)
        light.graph.input.extend(graph.input)
        light.graph.output.extend(graph.output)
        light.graph.value_info.extend(graph.value_info)
        listed = {str(info.name) for info in graph.input}
        for item in graph.initializer:
            if _product(item.dims) <= 64 and int(item.data_location) != 1:
                light.graph.initializer.append(item)
            elif str(item.name) not in listed:
                described = self._onnx.helper.make_tensor_value_info(
                    str(item.name), int(item.data_type), [int(dim) for dim in item.dims]
                )
                light.graph.input.append(described)
        return light

    def _declared_sizes(self, feeds: Mapping[str, Any]) -> Dict[str, int]:
        """How many values each tensor holds as the file declares it, for inputs of these shapes: a size the
        file leaves open under a name is what the inputs give that name."""
        numpy = _numpy()
        opened: Dict[str, int] = {}
        unsure: Set[str] = set()
        for info in self.model.graph.input:
            value = feeds.get(str(info.name))
            dims = info.type.tensor_type.shape.dim
            real = numpy.shape(value) if value is not None else ()
            if value is None or len(dims) != len(real):
                continue
            for dim, size in zip(dims, real):
                if dim.HasField("dim_param") and dim.dim_param:
                    if opened.setdefault(str(dim.dim_param), int(size)) != int(size):
                        unsure.add(str(dim.dim_param))  # one name, two sizes: the name says nothing
        sizes: Dict[str, int] = {}
        for name, dims in self._declared_shapes().items():
            total = 1
            for dim in dims:
                size = (None if dim in unsure else opened.get(dim)) if isinstance(dim, str) else dim
                if size is None:
                    break
                total *= int(size)
            else:
                sizes[name] = total
        return sizes

    def shapes(self, feeds: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Tuple[int, ...]]:
        """The shape each of these tensors has for these inputs, as the file makes it: found by one run that
        asks for their shapes and for none of them, so nothing is held to find it out.

        ``{}`` when that cannot be asked: one of them is not a tensor (a list, a map of classes), or the file
        cannot be run on these inputs, which the run that reads the tensors then says.
        """
        graph = self.model.graph
        asked = [name for name in dict.fromkeys(str(item) for item in names) if name in self.tensors]
        if not asked:
            return {}
        taken = self.tensors | {str(info.name) for info in graph.input} | {str(item.name) for item in graph.initializer}
        probes = []
        for index in range(len(asked)):
            probe = f"kaimei.shape.{index}"
            while probe in taken:
                probe += "_"
            probes.append(probe)

        def written() -> bytes:
            nodes, outputs = len(graph.node), len(graph.output)
            graph.node.extend(self._onnx.helper.make_node("Shape", [name], [probe]) for name, probe in zip(asked, probes))
            graph.output.extend(self._onnx.helper.make_empty_tensor_value_info(probe) for probe in probes)
            try:
                return bytes(self.model.SerializeToString())
            finally:
                del graph.node[nodes:]
                del graph.output[outputs:]

        try:
            found = self._made(written, True, sure=False).run(probes, dict(feeds))
        except Exception:  # noqa: BLE001 - not asked this way: each is read by itself, and its size is then known
            return {}
        return {name: tuple(int(dim) for dim in shape) for name, shape in zip(asked, found)}


def _part_of(name: str) -> str:
    """What a whole-number input holds, from its name: the importer's own guess."""
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


def bind_inputs(
    file_inputs: Sequence[Tuple[str, Any, Sequence[Optional[int]]]],
    features: Sequence[Mapping[str, Any]],
    given: Optional[Mapping[str, Any]] = None,
    pad_id: int = 0,
) -> List[Dict[str, Any]]:
    """How each input of a model file is made from the model's features, when no structure is at hand to say.

    ``file_inputs`` are ``(name, numpy dtype, shape with None where the file leaves a size open)``. ``given``
    says what an input is, by its name in the file: a feature's name, a list of names for the columns of a
    table, the same description an upload takes (``{"feature": "review", "part": "mask"}``), or a function
    from a row to the array. An input nobody describes is the obvious thing:

    * floats in two axes: the model's number, flag and category features, in order, as the columns;
    * whole numbers: the model's text feature. Its token ids, or by the input's name a mask of the real
      tokens (``mask``), segment ids of zeros (``type``), positions or the length. With several texts (an
      encoder-decoder has two: the text given, and what its decoder has written so far, the feature marked
      ``generated``) an input with ``decoder`` in its name reads the one text that is generated and any other
      the one that is given, and only when that leaves each text's token ids read by exactly one input
      (``input_ids`` and ``decoder_input_ids``, their masks beside them). Otherwise nothing is chosen between
      two texts: which one an input reads is asked for;
    * floats in four axes: the model's image, turned channels first when that is how the file has it;
    * anything else: the series or tensor feature of that name, or the only one.
    """
    numpy = _numpy()
    by_name = {str(feature["name"]): feature for feature in features}
    kinds = {name: str(feature.get("type") or "numeric") for name, feature in by_name.items()}
    scalars = [name for name in by_name if kinds[name] in _COLUMN_TYPES]
    texts = [name for name in by_name if kinds[name] in _TEXT_TYPES]
    # One text: every input of whole numbers reads it. Several: whose an input is goes by ``decoder`` in its name,
    # and only when every text's token ids are then read by exactly one input. Otherwise it is asked for.
    reads: Dict[str, Dict[str, Any]] = {}
    token_ids: List[str] = []
    for name, dtype, _declared in file_inputs if len(texts) > 1 else ():
        said = (given or {}).get(name)
        said = {"feature": said} if isinstance(said, str) else said
        if isinstance(said, Mapping) and str(said.get("feature")) in texts:
            token_ids += [str(said["feature"])] if str(said.get("part") or _part_of(name)) == "ids" else []
        elif said is None and dtype.kind in "iub":
            whose = [text for text in texts if bool(by_name[text].get("generated")) == ("decoder" in name.lower())]
            reads[name] = {"feature": whose[0] if len(whose) == 1 else "", "part": _part_of(name)}
            token_ids += [str(reads[name]["feature"])] if reads[name]["part"] == "ids" else []
    # An input the caller feeds themselves (a function, a value fixed beforehand) is theirs to have matched to a
    # text: the others still go by their names, as long as no text's token ids are then read twice.
    fed = any(
        callable(said) or (isinstance(said, Mapping) and ("value" in said or said.get("zeros")))
        for said in ((given or {}).get(name) for name, _dtype, _declared in file_inputs)
    )
    settled = sorted(token_ids) == sorted(texts) or (fed and len(set(token_ids)) == len(token_ids))
    if not settled or not all(read["feature"] for read in reads.values()):
        reads = {}
    images = [name for name in by_name if kinds[name] == "image"]
    others = [name for name in by_name if kinds[name] in ("sequence", "tensor")]
    column = 0
    bound: List[Dict[str, Any]] = []
    for name, dtype, declared in file_inputs:
        dims = list(declared)
        rank = len(dims)
        said = (given or {}).get(name)
        if callable(said):
            bound.append({"name": name, "dtype": dtype, "call": said})
            continue
        source: Dict[str, Any]
        if said is None:
            source = {}
        elif isinstance(said, str):
            source = {"feature": said}
        elif isinstance(said, (list, tuple)):
            source = {"features": [str(item) for item in said]}
        elif isinstance(said, Mapping):
            source = {str(key): value for key, value in said.items()}
        else:
            raise TypeError(f"inputs['{name}'] is a feature's name, a list of them, a description or a function.")
        if "value" in source or source.get("zeros"):
            shape = tuple(1 if dim is None else int(dim) for dim in dims)
            fixed = numpy.zeros(shape) if source.get("zeros") else numpy.broadcast_to(numpy.asarray(source["value"]), shape)
            bound.append({"name": name, "dtype": dtype, "shape": shape, "source": {}, "fixed": numpy.array(fixed).astype(dtype)})
            continue
        if "features" not in source and "feature" not in source:
            if dtype.kind in "iub":
                if len(texts) == 1:
                    source = {"feature": texts[0], "part": _part_of(name)}
                elif name in reads:
                    source = dict(reads[name])
                else:
                    raise ValueError(
                        f"The file's input '{name}' takes whole numbers, and this model has {len(texts)} text features. "
                        f"Say which it reads: inputs={{'{name}': {{'feature': ..., 'part': 'ids'}}}}."
                    )
            elif rank == 2 and scalars[column:]:
                width = int(dims[1]) if dims[1] is not None else len(scalars) - column
                source = {"features": scalars[column : column + width]}
                column += width
            elif rank in (3, 4) and len(images) == 1:
                source = {"feature": images[0]}
            else:
                match = next((item for item in others if item in (name, _clean(name))), None)
                if match is None and len(others) == 1:
                    match = others[0]
                if match is None:
                    raise ValueError(
                        f"It is not clear which feature the file's input '{name}' is made from. "
                        f"Say so: inputs={{'{name}': 'feature name'}}, or a function from a row to the array."
                    )
                source = {"feature": match}
        if "features" in source:
            columns = [str(item) for item in source["features"]]
            unknown = [item for item in columns if item not in by_name]
            if unknown:
                raise ValueError(f"The model has no feature called '{unknown[0]}' (for the file's input '{name}').")
            batched = rank != 1
            sample: Tuple[int, ...] = (len(columns),)
            source = {"features": columns}
        else:
            feature = by_name.get(str(source["feature"]))
            if feature is None:
                raise ValueError(f"The model has no feature called '{source['feature']}' (for the file's input '{name}').")
            kind = kinds[str(feature["name"])]
            if kind in _TEXT_TYPES:
                source.setdefault("part", _part_of(name))
                if source["part"] == "ids":
                    source.setdefault("pad_id", int(pad_id))
                length = int(feature.get("max_length") or 0) or int(dims[-1] or 1)
                batched = rank != 1
                sample = (1,) if source["part"] == "length" else (length,)
            elif kind == "image":
                height, width, channels = (int(feature.get(key) or 1) for key in ("height", "width", "channels"))
                body = dims[-3:]
                layout = str(source.get("layout") or "").lower()
                if not layout:
                    last = len(body) == 3 and body[2] == channels and body[0] != channels
                    layout = "nhwc" if last else "nchw"
                source["layout"] = layout
                sample = (channels, height, width) if layout in ("nchw", "chw") else (height, width, channels)
                batched = rank != 3
            elif kind in ("sequence", "tensor"):
                sample = tuple(int(dim) for dim in feature.get("shape") or [])
                batched = rank != len(sample)
            else:
                source = {"features": [str(feature["name"])]}
                sample = (1,)
                batched = rank != 1
        shape = (1, *sample) if batched else sample
        for at, dim in enumerate(dims if len(dims) == len(shape) else []):
            if dim is not None and int(dim) != int(shape[at]):
                raise ValueError(
                    f"The file's input '{name}' has shape {list(dims)}, and the model's features would fill "
                    f"{list(shape)}. Say how it is made: inputs={{'{name}': a function from a row to the array}}."
                )
        bound.append(
            {"name": name, "dtype": dtype, "shape": shape, "sample": sample, "axis": 0 if batched else None, "source": source}
        )
    return bound


class Agent:
    """One model's agent. :meth:`serve` runs it until it is stopped; :func:`serve` is the short way to do both.

    ``predict`` answers prediction jobs (see :func:`predict_with` and the module docstring for what a row
    is). ``module`` is the ``torch.nn.Module`` behind the model, and ``onnx`` the model's ONNX file (a path,
    its bytes or an ``onnx.ModelProto``): with either the agent answers trace and activations jobs too, and
    predictions when ``predict`` is left out. A module is watched with forward hooks, which see what each
    module returns; a file is run with ONNX Runtime, which gives every step of a structure read from that
    file. ``tokenizer`` is the model's tokenizer (a ``transformers`` or ``tokenizers`` tokenizer, a
    SentencePiece processor, a tiktoken encoding, or any function from a text to its ids; a dict of them by
    feature for a model with several text features): with it the agent splits text for a model whose
    tokenizer Kaimei cannot run. ``label`` names the agent on the model's page. ``capabilities`` narrows what
    it answers (``"predict"``, ``"trace"``, ``"activations"``, ``"tokenize"``).

    For a module: ``prepare(row)`` turns one row into what the module is called with (a tensor, or a tuple
    or dict of them), for a module that does not take a plain table or one array; ``feature_names`` is the
    order of a table's columns, read from the model on Kaimei when left out.

    For a file: ``structure`` is what was uploaded for the model (the :class:`~kaimei.structure.Extraction`
    or its spec), which says exactly how the file's inputs are made from the model's features. Without it
    ``inputs`` says so by input name (see :func:`bind_inputs`), and what nobody describes is the obvious
    thing: the table's columns in order, the text's token ids and mask, the image, and for an
    encoder-decoder what its decoder has written so far. Beside a structure, ``inputs`` says it for the
    inputs it names and those only (a value the upload kept at home, given as a function of the row); every
    other input is fed as the structure says. It goes by the file's own names for its inputs, and a name
    that is none of them is refused when the agent is made. The structure is also
    what the agent then goes by for the model's steps, with a module as with a file: a step that Kaimei's
    structure names and this one does not hold is not answered, and a file's steps and its prediction are
    read from the tensors this one names (see the module's notes on what a look inside sends). ``providers`` is passed to
    ONNX Runtime (``["CUDAExecutionProvider", "CPUExecutionProvider"]`` to use a GPU; the CPU otherwise).
    ``session_memory`` is how many bytes the ONNX Runtime sessions kept ready may take (2 GiB unless it says):
    a look inside a large model reads its steps in several runs, each with a session of its own, and as many
    of those are kept as this allows, so the next look does not make them again. The session that gives the
    file's own outputs, a prediction, is always kept.

    A classifier converted from scikit-learn gives its class beside its probabilities or, without them, its
    scores. With no structure on Kaimei to say how the file is read, its probabilities are sent as they are, and
    its scores so that Kaimei reads the class the file gives from them: a model of two classes as one margin,
    one of more as its scores, or the class itself where its scores do not give it (a support vector machine of
    three classes or more takes the class with the most votes, and votes can tie).

    ``pad_id`` is the token a text shorter than the model reads is filled up with. ``wait`` is how long one
    request for work is held open (up to 25 seconds, and less when the client's own timeout is shorter); one
    that comes back empty sooner than a second is followed by a pause to the end of it. ``log`` is called
    with a line about each thing the agent does.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        predict: Optional[Predict] = None,
        *,
        module: Any = None,
        onnx: Any = None,
        tokenizer: Any = None,
        label: Optional[str] = None,
        capabilities: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
        prepare: Optional[Callable[[Dict[str, Any]], Any]] = None,
        inputs: Optional[Mapping[str, Any]] = None,
        structure: Any = None,
        providers: Optional[Sequence[Any]] = None,
        session_memory: Optional[int] = None,
        pad_id: int = 0,
        wait: float = WAIT_SECONDS,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        if session_memory is not None and (isinstance(session_memory, bool) or int(session_memory) < 0):
            raise ValueError("session_memory is how many bytes the ONNX Runtime sessions kept ready may take: 0 or more.")
        if module is not None and not _is_torch(module):
            raise TypeError("module is the torch.nn.Module behind the model; pass anything else through predict.")
        if module is not None and onnx is not None:
            raise TypeError("Give the agent the model as a module or as its ONNX file, not both: one of them runs it.")
        if predict is None and module is None and onnx is None and tokenizer is None:
            raise TypeError(
                "An agent needs something to answer with: predict (a function from rows to predictions), module, "
                "onnx, tokenizer, or several of them."
            )
        self._encoders: Dict[str, Callable[[str, Optional[int]], List[int]]] = {}
        if isinstance(tokenizer, Mapping) and tokenizer and not callable(tokenizer):
            # One for each text feature, by the feature's name.
            self._encoders = {str(name): encoder_of(item) for name, item in tokenizer.items()}
        elif tokenizer is not None:
            self._encoders = {"": encoder_of(tokenizer)}
        self.client = client
        self.model = str(model)
        self.module = module
        self._file: Optional[_OnnxFile] = (
            _OnnxFile(onnx, providers, None if session_memory is None else int(session_memory)) if onnx is not None else None
        )
        self.label = str(label or platform.node() or "kaimei agent")[:120]
        self.wait = max(0.0, min(float(wait), 25.0))
        self.pad_id = int(pad_id)
        self._predict = predict
        self._prepare = prepare
        self._names = [str(name) for name in feature_names] if feature_names is not None else None
        self._inputs = dict(inputs) if inputs else {}
        self._spec: Optional[Mapping[str, Any]] = getattr(structure, "spec", structure)
        if self._spec is not None and not isinstance(self._spec, Mapping):
            raise TypeError("structure is what was uploaded for the model: the Extraction, or its spec.")
        if self._file is not None and self._inputs:
            self._check_inputs()
        self._log = log
        self._record: Optional[Dict[str, Any]] = None
        self._scene: Optional[Dict[str, Any]] = None
        self._bound: Optional[List[Dict[str, Any]]] = None
        self._tensors: Optional[Dict[str, str]] = None
        #: The file's tensors that are computed from what a row holds, once known (see ``_from_the_row``).
        self._computed: Optional[Set[str]] = None
        #: Steps Kaimei's structure names whose tensors the agent keeps back, each with why, and those already
        #: said in the log.
        self._kept_back: Dict[str, str] = {}
        self._told: Set[str] = set()
        self._plan: Optional[Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]] = None
        #: What the model's own output is (see ``_ENDS``), once known: a property of the model, not of a row.
        self._ends: Optional[str] = None
        #: How a converted classifier's numbers are sent (see ``_as_classes``), once known: ``("plain", 1.0)`` as
        #: they are, ``("margin", sign)`` or ``("paired", sign)`` as one margin, ``("scores", 1.0)`` as they are
        #: and checked, ``("labels", 1.0)`` as the class itself; ``("pending", 1.0)`` until a first answer says.
        self._form: Optional[Tuple[str, float]] = None
        self._stop = threading.Event()
        self.session_id: Optional[str] = None
        self.answered = 0
        runs = module is not None or self._file is not None
        able = ["predict"] if predict is not None or runs else []
        able += ["trace", "activations"] if runs else []
        able += ["tokenize"] if self._encoders else []
        if capabilities is not None:
            unknown = [item for item in capabilities if item not in KINDS]
            if unknown:
                raise ValueError(f"An agent answers {', '.join(KINDS)}; got {unknown[0]!r}.")
            able = [item for item in able if item in capabilities]
            if not able:
                raise ValueError("None of the capabilities asked for is one this agent can answer.")
        self.capabilities = able

    # --- the loop ----------------------------------------------------------------------------

    def serve(self, stop: Optional[threading.Event] = None) -> None:
        """Answer jobs until ``stop`` is set, :meth:`stop` is called, or the process is interrupted.

        A job that fails is answered with its error and the loop goes on. When the API cannot be reached the
        agent waits and tries again, a little longer each time; when the API no longer knows its session it
        opens another, at once the first time and, should the new one be lost too, after the same growing pause.
        A rejected key, or a model that is gone, ends it with that error.
        """
        if stop is not None:
            self._stop = stop
        delay = 1.0
        lost = 0  # sessions the API has lost in a row, with no work asked for in between
        try:
            while not self._stop.is_set():
                try:
                    if self.session_id is None:
                        self._open()
                    asked = time.monotonic()
                    job = self._poll()
                    delay, lost = 1.0, 0
                    early = MIN_POLL_SECONDS - (time.monotonic() - asked)
                    if job is None and early > 0 and not self._stop.is_set():
                        self._stop.wait(early)  # nothing to do, and the API did not wait: do not ask again at once
                except (AuthenticationError, PermissionDeniedError):
                    raise
                except NotFoundError as exc:
                    if exc.code != "agent_session_not_found":
                        raise
                    self.session_id = None  # the API closed or lost the session: open another
                    lost += 1
                    if lost >= 2:
                        self._say(f"Kaimei keeps losing this agent's session; opening another in {delay:.0f} s.")
                        self._stop.wait(delay)
                        delay = min(delay * 2.0, MAX_BACKOFF)
                    continue
                except (APIConnectionError, RateLimitError, APIStatusError) as exc:
                    if isinstance(exc, APIStatusError) and (exc.status_code or 500) < 500:
                        raise
                    self._say(f"Kaimei could not be reached ({exc.message}); trying again in {delay:.0f} s.")
                    self._stop.wait(delay)
                    delay = min(delay * 2.0, MAX_BACKOFF)
                    continue
                if job is not None:
                    try:
                        self._answer(job)
                    except Exception as exc:  # noqa: BLE001 - nothing a job does ends the loop
                        self._note(f"A job went wrong in a way the agent has no answer for ({_reason(exc)}); on to the next.")
        except KeyboardInterrupt:
            self._say("Stopping.")
        finally:
            self.close()

    def stop(self) -> None:
        """Ask :meth:`serve` to return. It does when the request for work it is waiting on comes back."""
        self._stop.set()

    def close(self) -> None:
        """Tell the API this agent is going away. Safe to call twice."""
        session, self.session_id = self.session_id, None
        if session is None:
            return
        try:
            self.client._request("DELETE", f"/v1/models/{self.model}/agent/sessions/{session}")
        except KaimeiError:
            pass  # the session ends by itself when it stops asking for work

    def _open(self) -> None:
        from .client import __version__

        info = {"client": f"kaimei-python {__version__}", "python": platform.python_version()}
        if self.module is not None:
            info["module"] = type(self.module).__name__
            try:
                import torch

                info["framework"] = f"pytorch {torch.__version__}"
            except ImportError:  # pragma: no cover - a module without torch cannot exist
                pass
        if self._file is not None:
            info["framework"] = f"onnxruntime {self._file._runtime.__version__}"
            info["providers"] = ", ".join(str(item if isinstance(item, str) else item[0]) for item in self._file.providers)[:200]
        opened = self.client._request(
            "POST",
            f"/v1/models/{self.model}/agent/sessions",
            json_body={"label": self.label, "capabilities": list(self.capabilities), "info": info},
        )
        self.session_id = str(opened["id"])
        self._say(f"Connected for {self.model}: answering {', '.join(self.capabilities)}.")

    def _poll(self) -> Optional[Dict[str, Any]]:
        work = self.client._request(
            "POST", f"/v1/models/{self.model}/agent/sessions/{self.session_id}/poll", json_body={"wait": self.poll_wait()}
        )
        job = (work or {}).get("job")
        return dict(job) if isinstance(job, dict) else None

    def poll_wait(self) -> float:
        """How long one request for work is held open: ``wait``, or less when the client gives up on a request
        sooner (``Kaimei(timeout=...)``). A request held open past the client's timeout would fail every time."""
        timeout = getattr(getattr(getattr(self.client, "_transport", None), "_client", None), "timeout", None)
        read = getattr(timeout, "read", None)
        if isinstance(read, (int, float)) and not isinstance(read, bool) and read > 0:
            return round(max(0.0, min(self.wait, float(read) - max(2.0, float(read) / 4.0))), 3)
        return self.wait

    def _answer(self, job: Dict[str, Any]) -> None:
        """Run one job and send its answer. Nothing a job does ends the loop.

        An answer larger than the API takes is not sent: the job fails, saying how large it was. One that does
        not get through for any other reason than the job being gone (refused, the connection lost on the
        way, or an answer that cannot be written out at all) is followed by a short failure, so whoever asked
        is told why rather than left to wait. Should the answer have arrived after all, the API refuses that
        failure: a job is answered once. A failure that cannot be sent either is said in the log.
        """
        started = time.perf_counter()
        try:
            result = self.run(job)
            body: Dict[str, Any] = {"ok": True, "result": result}
            too_large = _too_large(body, job)
            if too_large is not None:
                body = {"ok": False, "error": too_large[:2000]}
        except Exception as exc:  # noqa: BLE001 - whatever the model raises is the answer to this job
            body = {"ok": False, "error": _reason(exc)}
        path = f"/v1/models/{self.model}/agent/jobs/{job.get('id')}/result"
        outcome = "answered" if body["ok"] else f"failed: {body.get('error', '')}"
        try:
            self._send(path, body)
        except Exception as exc:  # noqa: BLE001 - an answer that does not get through is not the end of the agent
            status = exc.status_code if isinstance(exc, KaimeiError) else None
            why = exc.message if isinstance(exc, KaimeiError) else _reason(exc)
            if status in (404, 409) or (isinstance(exc, KaimeiError) and not body["ok"]):
                # The job is gone: whoever asked has stopped waiting, or it was answered already. On to the next one.
                outcome = f"not taken ({why})"
            elif not body["ok"]:
                outcome += f"; and whoever asked could not be told ({why})"
            else:
                refused = status is not None and 400 <= status < 500
                said = "was refused" if refused else "could not be sent"
                outcome = f"{'refused by Kaimei' if refused else 'not sent'}: {why}"
                try:
                    self._send(path, {"ok": False, "error": f"The agent's answer {said}: {why}"[:2000]})
                except Exception as again:  # noqa: BLE001 - said here, and the agent goes on to the next job
                    outcome += f"; and whoever asked could not be told ({_reason(again)})"
        self.answered += 1
        took = (time.perf_counter() - started) * 1000.0
        self._note(f"{job.get('kind')} {outcome} ({took:.0f} ms)")

    def _send(self, path: str, body: Dict[str, Any]) -> None:
        """Post an answer. One that did not get through for lack of a connection is sent once more."""
        try:
            self.client._request("POST", path, json_body=body)
        except APIConnectionError:
            self._stop.wait(1.0)
            self.client._request("POST", path, json_body=body)

    def _say(self, message: str) -> None:
        if self._log is not None:
            self._log(message)

    def _note(self, message: str) -> None:
        """A line for the log from where nothing may go wrong: a log that fails does not end the loop."""
        try:
            self._say(message)
        except Exception:  # noqa: BLE001 - there is nowhere left to say it
            pass

    # --- the work ----------------------------------------------------------------------------

    def run(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        """The answer to one job, given as the API hands it out: ``{"kind": ..., "payload": {...}}``.

        Numbers that are the model's own output (from ``module`` or ``onnx``, or from a ``predict`` that says
        so) are marked ``"raw": true``: Kaimei then reads them as the model's structure does. Numbers from a
        ``predict`` function of your own are taken as they are: probabilities, or the value.
        """
        kind = str(job.get("kind"))
        payload = job.get("payload") or {}
        if kind not in self.capabilities:
            raise ValueError(f"This agent does not answer '{kind}' jobs.")
        if kind == "predict":
            rows = [decode_row(row) for row in payload.get("rows") or []]
            predictions, raw = self._predictions(rows)
            answer: Dict[str, Any] = {"predictions": predictions}
            if raw:
                answer["raw"] = True
            link = self._link(predictions, raw)
            if link is not None:
                answer["link"] = link
            return answer
        if kind == "tokenize":
            return {"ids": self._tokenized(payload)}
        row = decode_row(payload.get("input") or {})
        if kind == "trace":
            return self._trace(row, payload.get("nodes"))
        return self._activations(
            row,
            payload.get("nodes"),
            units=int(payload.get("units") or 16),
            channels=int(payload.get("channels") or 8),
            max_side=int(payload.get("max_side") or 56),
            channel_start=_whole_or_none(payload.get("channel_start")),
        )

    def _tokenized(self, payload: Mapping[str, Any]) -> List[int]:
        """The ids the model reads for a text: its own tokenizer's, special tokens included, cut to what it reads."""
        feature = str(payload.get("feature") or "")
        encode = self._encoders.get(feature) or self._encoders.get("")
        if encode is None:
            raise ValueError(f"This agent has no tokenizer for the feature '{feature}'.")
        text = payload.get("text")
        if not isinstance(text, str):
            raise ValueError("A tokenize job carries the text to split.")
        most = payload.get("max_length")
        return encode(text, int(most) if most is not None else None)

    def _predictions(self, rows: List[Dict[str, Any]]) -> Tuple[List[List[float]], bool]:
        """``(a list of numbers for each row, whether they are the model's own output)``."""
        numpy = _numpy()
        if not rows:
            return [], False
        if self._predict is not None:
            raw = bool(getattr(self._predict, "raw", False))
            values = numpy.asarray(self._predict(rows), dtype=float)
        elif self._file is not None:
            raw = True
            if self._output_tensor() is None:
                raise self._no_prediction()
            runs = [self._run_file(row, []) for row in rows]
            values = numpy.stack([self._read_output(output, row) for (_found, output), row in zip(runs, rows)])
            label = self._label_tensor()
            if label is not None:
                values = self._as_classes(values.reshape(len(rows), -1), [found[label] for found, _output in runs])
        else:
            raw = True
            values = self._from_module(rows)
        if values.ndim == 0 or values.shape[0] != len(rows):
            raise ValueError(
                f"predict returned {values.shape[0] if values.ndim else 1} predictions for {len(rows)} rows."
            )
        values = values.reshape(len(rows), -1)
        if not numpy.isfinite(values).all():
            raise ValueError("The model's prediction holds a value that is not a finite number.")
        return [[float(item) for item in line] for line in values], raw

    def _link(self, values: Any, raw: bool) -> Optional[str]:
        """The ``link`` an answer names: what the model's own output is, when it is known and is not scores.

        Without a structure to say, Kaimei reads a model's own output as scores and puts it through a softmax
        or a sigmoid, which would be done twice to a model that ends in one. So the agent says so: ``"none"``
        for probabilities, ``"softmax"`` for their logarithms (a log-softmax), which a softmax turns back
        into them exactly. A function of your own says it with its ``link`` attribute (``predict_with`` sets
        one); a module is read with ``torch.fx``, or by which of its modules gave the output; a file by the
        node that makes the prediction. Where none of them says, rows of several numbers that are
        probabilities, or their logarithms, say it, once for the model. A converted classifier's scores sent
        as one margin, or as they are, are scores; its class sent as itself is a probability of one.
        """
        declared = getattr(self._predict, "link", None) if self._predict is not None else None
        if not raw:
            return str(declared) if declared in _LINKS else None
        if self._predict is None and self._form is not None and self._form[0] not in ("plain", "pending"):
            return "none" if self._form[0] == "labels" else None
        if self._ends is None:
            if declared is not None:
                found: Optional[str] = str(declared)
            elif self._predict is not None:
                found = None
            elif self._file is not None:
                found = self._file.ends(self._output_tensor())
            else:
                found = _traced_ends(self.module)
            if found is None:
                numpy = _numpy()
                array = numpy.asarray(values, dtype=float)
                if not array.size:
                    return None  # nothing to tell by yet
                found = _ends_by_values(array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(1, -1))
            self._ends = found or "scores"
        return self._ends if self._ends in _LINKS else None

    # --- what the model on Kaimei says of itself ---------------------------------------------

    def _model_record(self) -> Dict[str, Any]:
        if self._record is None:
            self._record = dict(self.client._request("GET", f"/v1/models/{self.model}") or {})
        return self._record

    def _features(self) -> List[Dict[str, Any]]:
        return [dict(feature) for feature in self._model_record().get("features") or [] if isinstance(feature, Mapping)]

    def _document(self) -> Dict[str, Any]:
        """The model's structure on Kaimei: its steps, in the order they run, and what its prediction is read from.

        ``{}`` for a model with no structure: it has predictions to give, and no steps to look inside.
        """
        if self._scene is None:
            try:
                document = self.client._request(
                    "GET", f"/v1/models/{self.model}/structure", params={"detail": "summary", "max_inline": 0}
                )
            except NotFoundError:
                document = None
            found = dict(document) if isinstance(document, Mapping) else {}
            scene = found.get("scene")
            listed = scene.get("nodes") if isinstance(scene, Mapping) else None
            read = [_step_read(node) for node in (listed if isinstance(listed, (list, tuple)) else [])]
            nodes = [node for node in read if node is not None]
            # A tensor's slab is drawn beside its step: it is not a step.
            nodes = [node for node in nodes if node.get("op") != "tensor" and node.get("kind") != "weights"]
            if nodes and all(isinstance(node.get("flow_index"), int) for node in nodes):
                nodes.sort(key=lambda node: int(node["flow_index"]))
            head = found.get("head") if isinstance(found.get("head"), Mapping) else {}
            self._scene = {"nodes": nodes, "head": dict(head or {})}
        return self._scene

    def _nodes(self) -> List[Dict[str, Any]]:
        return list(self._document().get("nodes") or [])

    def _structure_steps(self) -> List[Dict[str, Any]]:
        """The steps of the model's structure on Kaimei, without its inputs: what a look inside is reported against."""
        return [node for node in self._nodes() if node.get("op") != "input"]

    def _held(self) -> Optional[Dict[str, Mapping[str, Any]]]:
        """The steps of the structure this agent was started with (``structure=``), by id, each with what that
        structure says of its tensor: what the model file calls it (``source``) and its shape.

        ``None`` for an agent started with no structure, or with one that does not list its tensors (a model
        graph's does; one written by hand to say how the inputs are fed need not): the steps Kaimei names
        are then all there is to go by.
        """
        graph = self._spec.get("graph") if isinstance(self._spec, Mapping) else None
        values = graph.get("values") if isinstance(graph, Mapping) else None
        if not isinstance(values, Mapping):
            return None
        return {str(name): item for name, item in values.items() if isinstance(item, Mapping)}

    def _own_steps(self) -> List[Dict[str, Any]]:
        """The steps a look inside speaks of: the structure's on Kaimei or, for an agent started with a
        structure that lists its own, only those of them that structure holds."""
        held = self._held()
        steps = self._structure_steps()
        return steps if held is None else [step for step in steps if str(step["id"]) in held]

    def _withheld(self, kept_back: Mapping[str, str]) -> List[Dict[str, str]]:
        """What an answer says of the steps kept back, the first ``WITHHELD_SAID`` of them: each with why. The
        log is told of each once."""
        for step_id, why in kept_back.items():
            if step_id not in self._told and len(self._told) < 4 * WITHHELD_SAID:
                self._told.add(step_id)
                self._note(f"Nothing is sent of the step '{step_id[:120]}'. {why}")
        return [{"node_id": step_id, "reason": why} for step_id, why in list(kept_back.items())[:WITHHELD_SAID]]

    def _head_step(self, fits: Callable[[Mapping[str, Any]], bool]) -> Optional[Dict[str, Any]]:
        """The step the prediction is read from: the head's, or the nearest one before it that ``fits``.

        Kaimei adds a step or two of its own after the model's last (the softmax of a classifier's scores, a
        language model's last position). Those are not the model's, so the search walks back past them.
        """
        by_id = {str(node.get("id")): node for node in self._nodes()}
        node = by_id.get(str((self._document().get("head") or {}).get("vector_node") or ""))
        for _hop in range(4):
            if node is None:
                return None
            if fits(node):
                return node
            sources = node.get("inputs") or []
            node = by_id.get(str(sources[0])) if sources else None
        return None

    def _token_count(self, row: Mapping[str, Any], written: bool = False) -> int:
        """How many tokens this row gave the model: a language model's prediction is read at the last of them.

        ``written`` counts what an encoder-decoder has written so far instead (the feature marked
        ``generated``, which holds the token its decoder starts from before it has written any): its
        prediction is read after the last of those, not after the text its encoder read.
        """
        for feature in self._features():
            if str(feature.get("type")) in _TEXT_TYPES and bool(feature.get("generated")) == written:
                value = row.get(str(feature["name"]))
                if value is None and written and isinstance(feature.get("baseline"), (list, tuple)):
                    value = feature["baseline"]
                if isinstance(value, (list, tuple)):
                    most = int(feature.get("max_length") or 0)
                    return min(len(value), most) if most else len(value)
        return 0

    def _read_output(self, output: Any, row: Mapping[str, Any]) -> Any:
        """One row's output of the model as the numbers its prediction is read from. Nothing is squashed here.

        A language model gives scores for every position: the ones at the last token the row gave are its
        prediction. An encoder-decoder gives them for every position of what it has written: the ones after
        the last token written so far are the next token's, which is its prediction (the first token it
        writes, for a row that holds only the token its decoder starts from). Any other output is sent whole,
        in the order the structure shows it in.
        """
        numpy = _numpy()
        array = numpy.asarray(output)
        task = str(self._model_record().get("task") or "")
        if task in ("text_generation", "sequence_to_sequence"):
            body = array[0] if array.ndim == 3 and array.shape[0] == 1 else array
            if body.ndim == 2:
                decoding = task == "sequence_to_sequence"
                count = self._token_count(row, written=decoding)
                if decoding and not count:
                    raise ValueError(
                        "This model writes its answer token by token, and its prediction is read after the last token "
                        "it has written so far. The row holds none: the model on Kaimei needs a feature of token ids "
                        "marked \"generated\" (it starts as the token the decoder starts from), which Kaimei sends "
                        "with every row."
                    )
                count = count or int(body.shape[0])
                return numpy.asarray(body[max(1, min(count, int(body.shape[0]))) - 1], dtype=float).reshape(-1)
            return numpy.asarray(body, dtype=float).reshape(-1)
        step = self._head_step(lambda node: shown_values(node, array) is not None)
        if step is not None:
            return shown_values(step, array)
        return numpy.asarray(array, dtype=float).reshape(-1)

    def _input_values(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """What the model's input nodes hold for this row, as the API holds it: a text's ids end in -1."""
        numpy = _numpy()

        def number(value: Any) -> float:
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return float(value)
            raise TypeError("not a number")

        found: Dict[str, Any] = {}
        for node in self._nodes():
            if node.get("op") != "input":
                continue
            args = node.get("args") or {}
            units = int(node.get("units") or 0)
            try:
                names = args.get("features")
                if units > 1 and names:
                    flat = numpy.array([number(row[str(name)]) for name in names], dtype=float)
                else:
                    value = row[str(node.get("feature") or node.get("id"))]
                    if str(args.get("feature_type")) in _TEXT_TYPES:
                        flat = numpy.full(units, -1.0)
                        ids = [float(item) for item in value][:units]
                        flat[: len(ids)] = ids
                    elif hasattr(value, "shape") or isinstance(value, (list, tuple)):
                        flat = numpy.asarray(value, dtype=float).reshape(-1)
                    else:
                        flat = numpy.array([number(value)], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if flat.size == units:
                found[str(node["id"])] = flat
        return found

    # --- a module, watched with hooks --------------------------------------------------------

    def _from_module(self, rows: List[Dict[str, Any]]) -> Any:
        """The module's own output for these rows, when no ``predict`` was given."""
        import torch

        from .graphs import restore_modes, training_modes

        numpy = _numpy()
        module = self.module
        modes = training_modes(module)
        module.eval()
        handles, last = self._watch_ending(module)
        try:
            with torch.no_grad():
                if self._prepare is not None:
                    firsts = [self._first_tensor(self._call(self._prepare(row))) for row in rows]
                    self._heard_ending(last, firsts[0] if firsts else None)
                    outputs = [self._first(item) for item in firsts]
                    return numpy.stack([self._read_output(item, row) for item, row in zip(outputs, rows)])
                first = self._first_tensor(self._call(self._given(rows, torch)))
                self._heard_ending(last, first)
                produced = self._first(first)
            if produced.ndim == 0 or produced.shape[0] != len(rows):
                raise ValueError(f"The module returned {produced.shape[0] if produced.ndim else 1} outputs for {len(rows)} rows.")
            return numpy.stack([self._read_output(produced[at : at + 1], row) for at, row in enumerate(rows)])
        finally:
            for handle in handles:
                handle.remove()
            restore_modes(modes)

    def _watch_ending(self, module: Any) -> Tuple[List[Any], List[Any]]:
        """Hooks that say whether the module's output is what a softmax, a sigmoid or a log-softmax module gave.

        Only while that is not known yet, and only for a module whose code ``torch.fx`` cannot read. The last
        such output is held until the call ends, so its memory cannot be taken by another tensor meanwhile.
        """
        if self._ends is not None:
            return [], []
        traced = _traced_ends(module)
        if traced is not None:
            self._ends = traced
            return [], []
        last: List[Any] = [None, None]

        def heard(kind: str) -> Callable[[Any, Any, Any], None]:
            def record(_layer: Any, _inputs: Any, output: Any) -> None:
                last[:] = [kind, output if hasattr(output, "data_ptr") else None]

            return record

        handles = [
            layer.register_forward_hook(heard(type(layer).__name__.lower()))
            for layer in module.modules()
            if type(layer).__name__.lower() in _ENDS and not list(layer.children())
        ]
        return handles, last

    def _heard_ending(self, last: List[Any], first: Any) -> None:
        """The module's output is the last such module's own, or a view of it: the model ends in that step."""
        if not last or last[1] is None or first is None or self._ends is not None:
            return
        ending = last[1]
        if int(ending.data_ptr()) == int(first.data_ptr()) and int(ending.numel()) == int(first.numel()):
            self._ends = _ENDS[str(last[0])]

    @staticmethod
    def _first_tensor(produced: Any) -> Any:
        first = produced
        while isinstance(first, (tuple, list)) and first:
            first = first[0]
        if isinstance(first, Mapping) and first:
            first = next(iter(first.values()))
        return first

    def _call(self, given: Any) -> Any:
        if isinstance(given, Mapping):
            return self.module(**given)
        if isinstance(given, (tuple, list)):
            return self.module(*given)
        return self.module(given)

    @staticmethod
    def _first(produced: Any) -> Any:
        first = Agent._first_tensor(produced)
        return _numpy().asarray(first.detach().cpu().numpy(), dtype=float)

    def _given(self, rows: Sequence[Mapping[str, Any]], torch: Any) -> Any:
        """Rows as the module's input, by what the model on Kaimei says each feature is."""
        features = self._features()
        kinds = {str(feature["name"]): str(feature.get("type") or "numeric") for feature in features}
        lengths = {str(feature["name"]): int(feature["max_length"]) for feature in features if feature.get("max_length")}
        names = self._names or [str(feature["name"]) for feature in features]
        return _module_input(self.module, rows, names, torch, kinds=kinds, lengths=lengths, pad_id=self.pad_id)

    def _watch_module(self, row: Dict[str, Any], wanted: Optional[Set[str]]) -> Tuple[Dict[str, Any], Any, Dict[str, str]]:
        """Run the module once on ``row`` and return ``(values by step id, the module's output, the steps kept
        back and why)``.

        Every call of every module that can speak for a step is recorded (see :func:`hook_plan`), as a copy:
        a later step that works in place, an in-place activation or a ``+=``, must not change it. A step is
        then given a call's output only when that output has the very shape the structure says the step's
        tensor has; where a structure does not say, when it holds as many values.

        A module that ran and has no step in the structure was taken into another step when the model was
        converted. The one case that is known is a normalization folded into the convolution it follows:
        that convolution's step then holds what the normalization returned, and is given it. A step that
        took in anything else is given nothing.

        A hook sees what a module gave, not what it was computed from, and the structure that says which
        module a step is comes from Kaimei. So the run is followed, one operation at a time (see
        :class:`_OwnNumbers`), and a call's output is not sent where it holds only the model's own numbers:
        a parameter or a buffer given back as it is or in part, and what was computed from those alone (a
        parametrization that makes a layer's weight, the embedding of each position). An agent started with a
        structure that lists its steps (``structure=``) says nothing of a step that structure does not hold.
        """
        import torch

        module = self.module
        modules = {str(name): item for name, item in module.named_modules() if name}
        steps = {str(step["id"]): step for step in self._own_steps()}
        kept_back: Dict[str, str] = {}
        for step in self._structure_steps() if self._held() is not None else []:
            step_id = str(step["id"])
            parts = str(step.get("path") or step_id).split(".")
            if step_id not in steps and (wanted is None or step_id in wanted):
                if any(".".join(parts[:cut]) in modules for cut in range(len(parts), 0, -1)):
                    kept_back[step_id] = _NOT_HELD  # a module could have spoken for it
        if self._plan is None:
            self._plan = hook_plan(list(steps.values()), modules)
        own, under = self._plan
        speaks = {**{name: [item for ids in lists for item in ids] for name, lists in under.items()}, **own}
        heard = [name for name, ids in speaks.items() if wanted is None or any(item in wanted for item in ids)]
        unseen = [
            name for name, item in modules.items() if name not in own and name not in under and not list(item.children())
        ]
        calls: Dict[str, List[Optional[List[Any]]]] = {name: [] for name in heard}
        handles = []
        own_numbers = _OwnNumbers(module, torch)
        #: Calls some of whose output was the model's own numbers and was not recorded: ``(module, which call)``.
        unsaid: Set[Tuple[str, int]] = set()
        mark = "_kaimei_call"
        # The tensors marked, held weakly: a tensor that outlives the run (a parameter or a buffer a module returns
        # as it is) loses its mark when the run ends, so the next run does not take it for one of its own calls.
        marked: List[Any] = []

        def marking(item: Any, source: Tuple[str, int]) -> None:
            try:
                setattr(item, mark, source)
            except Exception:  # noqa: BLE001 - a tensor that takes no mark is simply not followed
                return
            try:
                marked.append(weakref.ref(item))
            except TypeError:  # pragma: no cover - torch tensors take weak references
                marked.append(lambda item=item: item)

        def tap(name: str) -> Callable[[Any, Any, Any, Any], None]:
            def record(_module: Any, inputs: Any, named: Any, output: Any) -> None:
                found: List[Any] = []
                stack = [output]
                # Where the run is not followed: called with nothing computed from the input, what a module
                # gives is the model's own, all of it.
                own_call = not own_numbers.followed and own_numbers.called_with(inputs, named)
                if own_call:
                    own_numbers.settle(output)
                kept = 0
                while stack and len(found) + kept < 8:
                    item = stack.pop(0)
                    if hasattr(item, "detach"):
                        if own_call or own_numbers.holds(item):
                            kept += 1  # the model's own numbers: not recorded, so never sent
                            continue
                        if not found and not kept:
                            marking(item, (name, len(calls[name])))  # whoever reads it next knows whose it is
                        found.append(item.detach().clone())
                    elif isinstance(item, (tuple, list)):
                        stack = list(item) + stack
                    elif isinstance(item, Mapping):
                        stack = list(item.values()) + stack
                if kept:
                    unsaid.add((name, len(calls[name])))
                calls[name].append(found)

            return record

        def weigh(_module: Any, inputs: Any, named: Any, output: Any) -> None:
            # A module no step is read from still passes on what it was called with: the model's own numbers stay so.
            if own_numbers.called_with(inputs, named):
                own_numbers.settle(output)

        def follow(name: str) -> Callable[[Any, Any, Any], None]:
            def absorbed(layer: Any, inputs: Any, output: Any) -> None:
                taken: Any = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else None
                source = getattr(taken, mark, None)
                if not isinstance(source, tuple) or len(source) != 2 or not hasattr(output, "detach"):
                    return
                made = calls.get(str(source[0]))
                if made is None:
                    return  # a mark of a module this run does not listen to
                # A module that only passes its input on, or lays it out anew (a dropout at rest, a flatten), took nothing in.
                same = output is taken or (
                    output.numel() == taken.numel() and bool(torch.equal(output.reshape(-1), taken.reshape(-1)))
                )
                folded = type(layer).__name__.startswith("BatchNorm") and type(modules.get(source[0])).__name__ in (
                    "Conv1d",
                    "Conv2d",
                    "Conv3d",
                )
                if same or folded:
                    if folded and source[1] < len(made):
                        made[source[1]] = [output.detach().clone()]
                    marking(output, source)
                elif source[1] < len(made):
                    made[source[1]] = None  # what this step took in is not known: nothing is said of it

            return absorbed

        from .graphs import restore_modes, training_modes

        # The input first: a row that cannot be prepared fails here, before any hook is on the module.
        given = self._prepare(row) if self._prepare is not None else self._given([row], torch)
        own_numbers.fed(given)
        modes = training_modes(module)
        try:
            for name in heard:
                handles.append(_listen(modules[name], tap(name)))
            for name, layer in modules.items() if not own_numbers.can_follow else ():
                if name not in calls:
                    handles.append(_listen(layer, weigh))  # hooks alone: every module is listened to
            for name in unseen:
                handles.append(modules[name].register_forward_hook(follow(name)))
            module.eval()
            with torch.no_grad(), own_numbers.following():
                produced = self._call(given)
        finally:
            for handle in handles:
                handle.remove()
            own_numbers.done()
            for held in marked:
                still = held()
                if still is not None and getattr(still, mark, None) is not None:
                    try:
                        delattr(still, mark)
                    except Exception:  # noqa: BLE001 - a mark that cannot be taken off is one no later run can use
                        pass
            restore_modes(modes)
        offered: Dict[str, List[List[Any]]] = {}
        own_steps: Set[str] = set()  # steps a call spoke for with the model's own numbers

        def offer(step_id: str, tensors: Optional[List[Any]], name: str, call: int) -> None:
            if (name, call) in unsaid:
                own_steps.add(step_id)
            if tensors:
                offered.setdefault(step_id, []).append(tensors)

        for name, ids in own.items():
            made = calls.get(name) or []
            if len(made) == len(ids):
                for call, (step_id, tensors) in enumerate(zip(ids, made)):
                    offer(step_id, tensors, name, call)
        for name in sorted(under, key=lambda item: -item.count(".")):
            lists, made = under[name], calls.get(name) or []
            if len(lists) > 1:
                if len(made) == len(lists):
                    for call, (ids, tensors) in enumerate(zip(lists, made)):
                        offer(ids[-1], tensors, name, call)
            elif len(made) == 1:
                offer(lists[0][-1], made[0], name, 0)
            elif len(made) == len(lists[0]):
                for call, (step_id, tensors) in enumerate(zip(lists[0], made)):
                    offer(step_id, tensors, name, call)
        values: Dict[str, Any] = {}
        for step_id, candidates in offered.items():
            step = steps[step_id]
            if wanted is not None and step_id not in wanted:
                continue
            says_shape = isinstance((step.get("args") or {}).get("tensor"), Mapping)
            for tensors in candidates:
                arrays = [tensor.cpu().numpy() for tensor in (tensors if says_shape else tensors[:1])]
                fits = [flat for flat in (shown_values(step, array, exact=says_shape) for array in arrays) if flat is not None]
                if len(fits) == 1:
                    values[step_id] = fits[0]
                    break
        for step_id in own_steps:
            if step_id not in values and (wanted is None or step_id in wanted):
                kept_back[step_id] = _NOT_COMPUTED
        if own_numbers.lost:
            # An operation the agent could not make sense of: nothing it ran after can be told apart, so nothing is sent.
            kept_back.update({step_id: _NOT_FOLLOWED for step_id in values})
            values = {}
        return values, self._first(produced), kept_back

    # --- an ONNX file, run with ONNX Runtime -------------------------------------------------

    def _structure_inputs(self) -> Optional[Dict[str, str]]:
        """Which input of the file each input of the structure this agent was started with is, by the
        structure's own name for it (``in.<name>``). ``None`` for an agent started with no structure, or with
        one that lists no inputs."""
        assert self._file is not None
        graph = self._spec.get("graph") if isinstance(self._spec, Mapping) else None
        listed = graph.get("inputs") if isinstance(graph, Mapping) else None
        if not listed:
            return None
        free = [name for name, _dtype, _dims in self._file.inputs]
        named: Dict[str, str] = {}
        for item in listed:
            wanted = str(item["name"])
            match = next((name for name in free if _clean(f"in.{name}")[:180] == wanted), None)
            if match is None:
                match = next((name for name in free if wanted.startswith(_clean(f"in.{name}")[:180])), None)
            if match is None:
                raise ValueError(
                    f"The structure's input '{wanted}' is not an input of this file ({', '.join(free) or 'none left'}). "
                    "Is this the file the structure was read from? Otherwise say how the file is fed: inputs={...}."
                )
            free.remove(match)
            named[wanted] = match
        return named

    def _check_inputs(self) -> None:
        """Refuse, when the agent is made, an ``inputs=`` that names what is not an input of the model's file.

        ``inputs=`` goes by the file's own names for its inputs. A name that is none of them would feed
        nothing, and the input meant would be fed some other way without a word; a structure's own name for
        an input (``in.<name>``) is the likeliest slip, and is answered with the name that was meant.
        """
        assert self._file is not None
        there = [name for name, _dtype, _dims in self._file.inputs]
        unknown = [str(name) for name in self._inputs if str(name) not in there]
        if not unknown:
            return
        try:
            named = self._structure_inputs() or {}
        except ValueError:
            named = {}  # a structure that is not this file's: said when the file is first fed
        meant = named.get(unknown[0])
        fed = f" The structure this agent was started with feeds: {', '.join(named.values())}." if named else ""
        raise ValueError(
            f"inputs names '{unknown[0]}', which is not an input of this model's file. Its inputs are: "
            f"{', '.join(there) or 'none'}.{fed}"
            + (f" inputs= goes by the file's own names: '{meant}' is what the structure calls '{unknown[0]}'." if meant else "")
        )

    def _written_feature(self) -> Optional[str]:
        """The feature that holds what the model has written so far (an encoder-decoder's decoder tokens, which
        Kaimei keeps and sends with every row): the one the structure's ``decoder`` names, or the model's one
        feature of token ids marked ``generated``. ``None`` when there is none, or nothing says which of several."""
        texts = {
            str(feature["name"]): feature for feature in self._features() if str(feature.get("type")) in _TEXT_TYPES
        }
        decoder = self._spec.get("decoder") if isinstance(self._spec, Mapping) else None
        named = str(decoder.get("feature") or "") if isinstance(decoder, Mapping) else ""
        if named in texts:
            return named
        generated = [name for name, feature in texts.items() if feature.get("generated")]
        return generated[0] if len(generated) == 1 else None

    def _bindings(self) -> List[Dict[str, Any]]:
        """How the file's inputs are made from the model's features: from the structure, or the obvious way.

        With a structure, each input is made as the structure says. An input the caller names in ``inputs=``
        is made as the caller says instead, and that input only: the structure's word still holds for every
        other one. An input the structure says is what the model has written so far (``"generated"``) is the
        token ids of the feature that holds it, as it is without a structure.
        """
        if self._bound is not None:
            return self._bound
        assert self._file is not None
        numpy = _numpy()
        file_inputs = self._file.inputs
        named = self._structure_inputs()
        spec = self._spec
        if named is None or not isinstance(spec, Mapping):
            self._bound = bind_inputs(file_inputs, self._features(), self._inputs, self.pad_id)
            return self._bound
        from .graphs import _bindings

        mine = {str(name) for name in self._inputs}
        kept = []
        for item in spec["graph"]["inputs"]:
            original = named[str(item["name"])]
            source = item.get("from") or {}
            if original in mine:
                continue  # fed as the caller says: what the structure says of it is not gone by, nor needed
            if source.get("generated") and "feature" not in source and "value" not in source:
                feature = self._written_feature()
                if feature is None:
                    raise ValueError(
                        f"The structure's input '{item['name']}' is given what the model has written so far, and the "
                        "model on Kaimei has no feature that holds it (a feature of token ids marked \"generated\"). "
                        f"Say how the file's input '{original}' is fed: inputs={{'{original}': a function from a row "
                        "to the array}."
                    )
                item = {**item, "from": {"feature": feature, "part": "ids", "pad_id": int(source.get("pad_id") or 0)}}
            kept.append(item)
        # A value the input is always given is written as the upload sends it: numbers, or bytes in base64.
        # One that cannot be read is said by the file's own name for the input, which is what inputs= goes by.
        given = {
            named[str(item["name"])]: _fixed_value((item.get("from") or {})["value"], named[str(item["name"])])
            for item in kept
            if "value" in (item.get("from") or {})
        }
        fed = set(named.values()) | mine
        for name, dtype, dims in file_inputs:
            if name not in fed:
                # An input the prediction does not depend on still has to be given to the file.
                given[name] = numpy.zeros([1 if dim is None else int(dim) for dim in dims], dtype=dtype)
        bound = _bindings({**spec, "graph": {**spec["graph"], "inputs": kept}}, named, given, numpy)
        if mine:
            said = [entry for entry in file_inputs if entry[0] in mine]
            bound = [*bound, *bind_inputs(said, self._features(), self._inputs, self.pad_id)]
        self._bound = bound
        return self._bound

    def _encoded(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """A row's values as the numbers a structure's inputs are made from: a text's ids end in -1."""
        numpy = _numpy()
        encoded: Dict[str, Any] = {}
        for feature in self._features():
            name = str(feature["name"])
            kind = str(feature.get("type") or "numeric")
            started = feature.get("baseline") if kind in _TEXT_TYPES and feature.get("generated") else None
            if name not in row and not isinstance(started, (list, tuple)):
                continue
            # What a model has written so far is never given by a caller: before it has written anything it is
            # the token its decoder starts from (the feature's baseline), as a prediction on Kaimei has it.
            value = row[name] if name in row else list(started or [])
            if kind in _TEXT_TYPES:
                size = int(feature.get("max_length") or 0) or len(value)
                flat = numpy.full(size, -1.0)
                ids = [float(item) for item in value][:size]
                flat[: len(ids)] = ids
                encoded[name] = flat
            elif hasattr(value, "shape") or isinstance(value, (list, tuple)):
                encoded[name] = numpy.asarray(value, dtype=float).reshape(-1)
            elif isinstance(value, str):
                categories = [str(item) for item in feature.get("categories") or []]
                if value not in categories:
                    raise ValueError(
                        f"The feature '{name}' is the category '{value}', and a model file takes numbers. "
                        "Say how it is fed: inputs={input name: a function from a row to the array}."
                    )
                encoded[name] = numpy.array([float(categories.index(value))])
            elif value is None:
                encoded[name] = numpy.array([float("nan")])
            else:
                encoded[name] = numpy.array([float(value)])
        return encoded

    def _feeds(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        from .graphs import _feeds

        numpy = _numpy()
        bound = self._bindings()
        try:
            feeds = _feeds([item for item in bound if "call" not in item], self._encoded(row), numpy)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "?"
            raise ValueError(
                f"The row has no value for the feature '{missing}', which an input of the model's file is made from."
            ) from exc
        for item in bound:
            if "call" in item:
                feeds[item["name"]] = numpy.asarray(item["call"](dict(row)), dtype=item["dtype"])
        return feeds

    def _file_tensors(self) -> Dict[str, str]:
        """The tensor of the file each step was read from, by step id.

        A step says what the file calls its tensor (``args.source``) unless that is the step's own id. A step
        Kaimei added after the model's own, the softmax that reads a prediction, has no tensor in the file,
        even where the file happens to call one of its tensors by that step's name: a converted classifier
        calls its scores ``probabilities``, and so does Kaimei the step that squashes them.

        The structure comes from Kaimei, and a step of it may name any tensor of the file. A step is read
        only where its tensor is computed from what a row holds (see :meth:`_from_the_row`): never the model's
        weights, a constant, or what is worked out from those alone. An agent started with the structure that
        was uploaded (``structure=``) goes by that structure: a step it does not hold is not read, and a step
        it holds is read from the tensor it names. What is kept back is in ``_kept_back``, each with why.
        """
        assert self._file is not None
        if self._tensors is None:
            steps = self._structure_steps()
            held = self._held()
            said = {str(step["args"]["source"]) for step in steps if (step.get("args") or {}).get("source")}
            found: Dict[str, str] = {}
            kept_back: Dict[str, str] = {}
            for step in steps:
                step_id = str(step["id"])
                source = (step.get("args") or {}).get("source")
                name = str(source or step_id)
                named = name in self._file.tensors and bool(source or name not in said)
                if held is not None:
                    mine = held.get(step_id)
                    if mine is None:
                        if named:
                            kept_back[step_id] = _NOT_HELD
                        continue
                    name = str(mine.get("source") or step_id)
                    named = name in self._file.tensors
                if not named:
                    continue
                if name in self._from_the_row():
                    found[step_id] = name
                else:
                    kept_back[step_id] = _NOT_COMPUTED
            self._tensors, self._kept_back = found, kept_back
        return self._tensors

    def _from_the_row(self) -> Set[str]:
        """The file's tensors that are computed from what a row holds: every tensor a step gives that reads,
        directly or through other steps, an input of the file that is made from the row.

        An input given the same value for every row is not one: a value fixed when the model was uploaded,
        zeros, a text's segment ids and its positions. So a table looked up by those alone (the embedding of
        each position) is the model's own numbers, like any weight, whatever the row says. An input fed by a
        function of your own (``inputs={name: function}``) is taken to be made from the row.
        """
        assert self._file is not None
        if self._computed is None:
            made = [
                str(item["name"])
                for item in self._bindings()
                if "call" in item or ("fixed" not in item and (item.get("source") or {}).get("part") not in _SAME_FOR_EVERY_ROW)
            ]
            self._computed = self._file.computed_from(made)
        return self._computed

    def _output_tensor(self) -> Optional[str]:
        """The tensor of the file the prediction is read from: the one the structure's head reads or, with no
        structure to say, the one the importer reads (see :meth:`_OnnxFile._prediction_tensor`). ``None`` when
        the file has no output a prediction can be read from.

        An agent started with the structure that was uploaded reads the prediction where that structure's own
        head does, whatever step the structure on Kaimei points at. And a prediction, like a step, is read only
        from a tensor that is computed from the row: an output of the file that holds the model's own numbers
        (a table it also gives out) is passed over for the first output of floats that does not.
        """
        assert self._file is not None
        named = self._file_tensors()
        held = self._held()
        if held is not None and isinstance(self._spec, Mapping):
            head = self._spec.get("head")
            own = str(head.get("output") or "") if isinstance(head, Mapping) else ""
            tensor = str((held.get(own) or {}).get("source") or own)
            if own in held and tensor in self._file.tensors and tensor in self._from_the_row():
                return tensor
        step = self._head_step(lambda node: str(node.get("id")) in named)
        if step is not None:
            return named[str(step["id"])]
        found = self._file.prediction
        if found is None or found in self._from_the_row():
            return found
        return next((name for name in self._file.floats if name in self._from_the_row()), None)

    def _label_tensor(self) -> Optional[str]:
        """The file's label, asked for beside its output when that is a converted classifier's numbers, sent by
        the class the label says (see :meth:`_as_classes`). ``None`` otherwise."""
        if self._file is None or self._predict is not None or self._file.label is None:
            return None
        if self._form is None:
            self._form = ("pending", 1.0) if self._reads_classes() else ("plain", 1.0)
        return self._file.label if self._form[0] != "plain" else None

    def _reads_classes(self) -> bool:
        """Whether the prediction is a converted classifier's numbers that Kaimei reads as they come: the model on
        Kaimei gives one class a row, and no structure there names the tensor and says how it is read."""
        assert self._file is not None
        if str(self._model_record().get("task") or "") not in ("binary_classification", "multiclass_classification"):
            return False
        tensor = self._output_tensor()
        named = self._file_tensors()
        if tensor is None or self._head_step(lambda node: str(node.get("id")) in named) is not None:
            return False
        return self._file.classifier_output(tensor) and bool(self._file.classes(tensor))

    def _as_classes(self, values: Any, labels: Sequence[Any]) -> Any:
        """A converted classifier's numbers for some rows, sent so that Kaimei reads from them the class the file
        gives (its ``predict``): ``labels`` holds the file's class for each row.

        Probabilities, or their logarithms, go as they are. Scores are read by Kaimei through a sigmoid (one
        number, the second class's) or a softmax (several), so they go in the form whose reading is the file's
        class: for two classes one margin, positive for the second (an SVC's two scores are the other way round
        from a linear model's); for more the scores as they are, when the highest is the class. Where that is not
        so, a support vector machine of three classes or more whose votes decide or scores that stop giving the
        class on some row, the class itself is sent, as a probability of one, from then on.
        """
        numpy = _numpy()
        assert self._file is not None
        if self._form is None or self._form[0] == "plain":
            return values
        classes = self._file.classes(self._output_tensor()) or []
        indices = _class_indices(labels, classes)
        if indices is None:
            self._form = ("plain", 1.0)  # a class the file does not name: its numbers go as they are
            return values
        form = self._form
        if form[0] == "pending":
            form, settled = self._chosen_form(values, indices, len(classes))
            if settled:
                self._form = form
        kind, sign = form
        if kind == "plain":
            return values
        if kind != "labels":
            sent = values if kind == "scores" else _margin_of(values, sign, kind == "paired")
            if numpy.array_equal(self._classes_read(sent), indices):
                return sent
            self._form = ("labels", 1.0)
            self._say("This model's scores did not give the class its file gives on every row: its class is sent from now on.")
        return numpy.eye(len(classes))[indices]

    def _chosen_form(self, values: Any, indices: Any, count: int) -> Tuple[Tuple[str, float], bool]:
        """``(how a converted classifier's numbers are sent, whether that is settled)`` (see :meth:`_as_classes`), by
        what the file says they are and by a first answer: ``indices`` is each row's class among the ``count``.

        A margin is turned the way that gives each row's class. Rows whose margins are all nought give their class
        either way (and the numbers sent are the same): the way is then settled by an answer that says it.
        """
        numpy = _numpy()
        assert self._file is not None
        tensor = self._output_tensor()
        array = numpy.asarray(values, dtype=float)
        said = self._file.ends(tensor)
        if said is None:
            said = _ends_by_values(array)
        if said in ("none", "softmax"):
            return ("plain", 1.0), True  # probabilities, or their logarithms: the model's own, read as they are
        declared = str((self._model_record().get("outputs") or {}).get("kind") or "").lower()
        if self._file.votes(tensor) or declared in ("probabilities", "probability"):
            return ("labels", 1.0), True  # its votes decide, or Kaimei is told its numbers are probabilities
        width = array.shape[1]
        if count == 2 and width in (1, 2):
            paired = width == 2 and bool(numpy.allclose(array[:, 0], -array[:, 1], rtol=1e-5, atol=1e-6))
            kind = "paired" if paired else "margin"
            fits = [
                sign
                for sign in (1.0, -1.0)
                if numpy.array_equal(self._classes_read(_margin_of(array, sign, paired)), indices)
            ]
            if not fits:
                return ("labels", 1.0), True
            return (kind, fits[0]), len(fits) == 1
        if width == count and numpy.array_equal(self._classes_read(array), indices):
            return ("scores", 1.0), True
        return ("labels", 1.0), True

    def _classes_read(self, values: Any) -> Any:
        """The class Kaimei reads from these numbers of the model's own, sent with no link (see ``_class_read``)."""
        binary = str(self._model_record().get("task") or "") == "binary_classification"
        return _class_read(values, binary, self._positive_index())

    def _positive_index(self) -> int:
        """The class Kaimei counts as the positive one, found as it finds it (``ModelRunner.positive_index``)."""
        record = self._model_record()
        outputs = record.get("outputs") or {}
        labels = [str(item) for item in outputs.get("labels") or []]
        positive = outputs.get("positive_label")
        if positive is not None and str(positive) in labels:
            return labels.index(str(positive))
        return 1 if len(labels) == 2 or str(record.get("task") or "") == "binary_classification" else 0

    def _run_file(
        self, row: Mapping[str, Any], names: Sequence[str], feeds: Optional[Mapping[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Any]:
        """``(the tensors called names, the model's output)`` for one row. The output is ``None`` when the file
        has none a prediction is read from, and the tensors inside it still are. A converted classifier's label is
        asked for beside the output when the output is sent by it (an output of the file: no session more).
        ``feeds`` is what the row gives the file's inputs, when that has been made already."""
        assert self._file is not None
        output = self._output_tensor()
        label = self._label_tensor()
        asked = [*([output] if output is not None else []), *([label] if label is not None else []), *names]
        found = self._file.run(self._feeds(row) if feeds is None else feeds, asked)
        return found, (found[output] if output is not None else None)

    def _no_prediction(self) -> ValueError:
        assert self._file is not None
        graph = self._file.model.graph
        kinds = []
        for info in graph.output:
            kind = info.type.tensor_type
            try:
                said = str(self._file._onnx.helper.tensor_dtype_to_np_dtype(int(kind.elem_type)))
            except (KeyError, TypeError, ValueError):
                said = "not a tensor"
            kinds.append(f"{info.name} ({said})")
        if self._file.prediction is not None:
            return ValueError(
                f"No output of this ONNX file that holds scores is computed from its input: {', '.join(kinds)}. What "
                "does not depend on the input is the model's own numbers, and an agent sends none of those. Is every "
                "input of the file fed from the row (inputs={...})?"
            )
        return ValueError(
            f"No output of this ONNX file holds scores to read a prediction from: {', '.join(kinds) or 'it has none'}. "
            "A class label is not a score. Convert the model with its probabilities or scores among its outputs, or "
            "upload the structure read from this file, which says the tensor to read."
        )

    def _watch_file(
        self, row: Dict[str, Any], wanted: Optional[Set[str]], told: Callable[[Mapping[str, Any], Any], Any]
    ) -> Tuple[Dict[str, Any], Any, Any, Dict[str, str]]:
        """Run the file on ``row`` and return ``(what told made of each step's values by step id, the model's
        output, the file's label, the steps kept back and why)``. The label is ``None`` unless the output is sent
        by it (see :meth:`_as_classes`).

        A step read from this file names its tensor in it (``args.source``, or the step's own id). Those
        tensors are asked for beside the file's outputs, a few hundred at a time for a model with very many
        steps, and laid out as the structure shows each step. A step is laid out only as its tensor is: the
        shape the structure gives must be the tensor's own, and the order of its axes an order of them, so
        nothing Kaimei's structure says can regroup the values of one. Steps whose tensors are not computed
        from the row are never asked for (see :meth:`_file_tensors`).

        How much one run reads goes by what the file says each tensor holds (:meth:`_OnnxFile.sizes`), never by
        the size Kaimei's structure gives a step: at most ``FETCH_STEPS`` tensors and ``FETCH_VALUES`` values a
        run, a tensor named by several steps read once, and one whose size the file cannot say read by itself.
        What a run read is let go before the next run: each step's values are given to ``told`` as soon as they
        are laid out, and only what it makes of them (a frame, a stage) is kept.
        """
        assert self._file is not None
        tensors = self._file_tensors()
        kept_back = {name: why for name, why in self._kept_back.items() if wanted is None or name in wanted}
        #: The steps read from each tensor, the tensors in the order their first steps come.
        reads: Dict[str, List[Dict[str, Any]]] = {}
        for step in self._structure_steps():
            if wanted is not None and str(step["id"]) not in wanted:
                continue
            tensor = tensors.get(str(step["id"]))
            if tensor is not None:
                reads.setdefault(tensor, []).append(step)
        feeds = self._feeds(row)
        sizes = self._file.sizes(feeds, reads) if reads else {}
        lots: List[List[str]] = []
        lot: List[str] = []
        held = 0
        for tensor in reads:
            size = sizes.get(tensor)
            if lot and (size is None or len(lot) >= FETCH_STEPS or held + size > FETCH_VALUES):
                lots.append(lot)
                lot, held = [], 0
            lot.append(tensor)
            if size is None:
                lots.append(lot)  # how much it holds is not known until it is read: by itself, then
                lot = []
            else:
                held += size
        if lot or not lots:
            lots.append(lot)
        # Each lot of steps is read by a session of its own. They are all kept, as memory allows, so the next
        # look inside makes none of them again; while this one runs, none of them goes to make room for another.
        output_tensor = self._output_tensor()
        first = [output_tensor] if output_tensor is not None else []
        keys = {self._file.key([*first, *lot]) for lot in lots}
        self._file.keep(len(keys | {()}))
        self._file.plan(keys)
        made: Dict[str, Any] = {}
        output: Any = None
        label = self._label_tensor()
        label_read: Any = None
        flat: Any = None
        try:
            for lot in lots:
                found, output = self._run_file(row, lot, feeds)
                self._file.learn(feeds, found)
                if label is not None:
                    label_read = found.get(label)
                for tensor in lot:
                    for step in reads[tensor]:
                        flat, why = _laid_out(step, found[tensor], True)
                        if flat is not None:
                            made[str(step["id"])] = told(step, flat)
                        elif why:
                            kept_back[str(step["id"])] = why
                found.clear()  # nothing a run read is held while the next run is made
                flat = None
        finally:
            self._file.plan(())
        return made, output, label_read, kept_back

    # --- a look inside -------------------------------------------------------------------------

    def _watch(
        self, row: Dict[str, Any], nodes: Any, told: Callable[[Mapping[str, Any], Any], Any]
    ) -> Tuple[Dict[str, Any], Any, Any, Dict[str, str]]:
        """``(what told made of each step's values by step id, the model's output, the file's label or None, the
        steps kept back and why)`` for one row.

        ``told(step, values)`` is called with each step's values as soon as they are at hand, and what it gives
        (a frame, a stage) is kept in their place: a look inside a file never holds every step's values at once.
        """
        wanted = {str(item) for item in nodes} if nodes else None
        label: Any = None
        if self._file is not None:
            made, output, label, kept_back = self._watch_file(row, wanted, told)
        else:
            values, output, kept_back = self._watch_module(row, wanted)
            steps = {str(step["id"]): step for step in self._structure_steps()}
            made = {}
            for step_id in list(values):
                made[step_id] = told(steps[step_id], values.pop(step_id))
        given = {str(node["id"]): node for node in self._nodes() if node.get("op") == "input"}
        for node_id, flat in self._input_values(row).items():
            if wanted is None or node_id in wanted:
                made[node_id] = told(given[node_id], flat)
        return made, output, label, kept_back

    def _prediction(self, output: Any, row: Dict[str, Any], label: Any = None) -> Tuple[Optional[List[float]], bool]:
        """``(the model's numbers for this row, whether they are its own output)``. The API names them. ``label`` is
        the file's class for the row, when the numbers are sent by it (see :meth:`_as_classes`)."""
        numpy = _numpy()
        values, raw = None, True
        if self._predict is not None:
            try:
                values = numpy.asarray(self._predict([row]), dtype=float).reshape(-1)
                raw = bool(getattr(self._predict, "raw", False))
            except Exception:  # noqa: BLE001 - the look inside still stands without its headline
                values = None
        if values is None and output is None:
            return None, False  # a file with no output to read a prediction from
        if values is None:
            try:
                values, raw = self._read_output(output, row), True
                if label is not None:
                    values = numpy.asarray(self._as_classes(values.reshape(1, -1), [label]), dtype=float).reshape(-1)
            except Exception:  # noqa: BLE001 - an output that cannot be read leaves the look inside as it is
                return None, False
        if values is None or not values.size or not numpy.isfinite(values).all():
            return None, False
        return [float(item) for item in values], raw

    def _trace(self, row: Dict[str, Any], nodes: Any) -> Dict[str, Any]:
        numpy = _numpy()

        def frame_of(step: Mapping[str, Any], flat: Any) -> Dict[str, Any]:
            flat = numpy.where(numpy.isfinite(flat), flat, 0.0)
            frame: Dict[str, Any] = {
                "node_id": str(step["id"]),
                "value": [float(item) for item in flat[:FRAME_VALUES]],
                "summary": _summary(flat, numpy),
            }
            # A sample is sent with every frame: the API makes one of its own, a single zero, for a frame without.
            index = numpy.linspace(0, flat.size - 1, min(FRAME_SAMPLE, flat.size)).round().astype(int)
            frame["sample"] = [float(item) for item in flat[index]]
            return frame

        made, output, label, kept_back = self._watch(row, nodes, frame_of)
        frames = [made[str(step["id"])] for step in self._nodes() if str(step["id"]) in made]
        answer: Dict[str, Any] = {"frames": frames}
        if kept_back:
            # Said, not raised: the steps the agent keeps to itself are left out like any step it has no values for.
            answer["withheld"] = self._withheld(kept_back)
        prediction, raw = self._prediction(output, row, label)
        if prediction is not None:
            answer["prediction"] = prediction
            if raw:
                answer["raw"] = True
            link = self._link([prediction], raw)
            if link is not None:
                answer["link"] = link
        return answer

    def _activations(
        self,
        row: Dict[str, Any],
        nodes: Any,
        *,
        units: int,
        channels: int,
        max_side: int,
        channel_start: Optional[int] = None,
    ) -> Dict[str, Any]:
        if nodes:
            detailed = {str(item) for item in nodes}
        else:
            head = str((self._document().get("head") or {}).get("vector_node") or "")
            detailed = set(default_detail(self._nodes(), head or None))
        described = [0.0]  # seconds spent making stages, which are not the model's

        def stage(step: Mapping[str, Any], flat: Any) -> Dict[str, Any]:
            began = time.perf_counter()
            found = stage_of(
                step,
                flat,
                detailed=str(step["id"]) in detailed and step.get("op") != "input",
                units=units,
                channels=channels,
                max_side=max_side,
                channel_start=channel_start,
            )
            described[0] += time.perf_counter() - began
            return found

        started = time.perf_counter()
        made, output, label, kept_back = self._watch(row, None, stage)
        took = max(time.perf_counter() - started - described[0], 0.0) * 1000.0
        stages = [made[str(step["id"])] for step in self._nodes() if str(step["id"]) in made]
        answer: Dict[str, Any] = {"stages": stages, "forward_ms": took}
        if kept_back:
            answer["withheld"] = self._withheld(kept_back)
        prediction, raw = self._prediction(output, row, label)
        if prediction is not None:
            answer["prediction"] = prediction
            if raw:
                answer["raw"] = True
            link = self._link([prediction], raw)
            if link is not None:
                answer["link"] = link
        return answer


def serve(
    client: Any,
    model: str,
    predict: Optional[Predict] = None,
    *,
    module: Any = None,
    onnx: Any = None,
    tokenizer: Any = None,
    label: Optional[str] = None,
    capabilities: Optional[Sequence[str]] = None,
    stop: Optional[threading.Event] = None,
    **options: Any,
) -> Agent:
    """Run an agent for ``model`` until ``stop`` is set or the process is interrupted. Blocks.

    See :class:`Agent` for the arguments. Returns the agent, whose session is closed by then. To keep working
    while it runs, start it on a thread and set ``stop`` when you are done::

        stop = threading.Event()
        threading.Thread(target=serve, args=(client, "my-model"), kwargs={"predict": fn, "stop": stop}).start()
    """
    agent = Agent(
        client,
        model,
        predict,
        module=module,
        onnx=onnx,
        tokenizer=tokenizer,
        label=label,
        capabilities=capabilities,
        **options,
    )
    agent.serve(stop)
    return agent


def _load(target: str) -> Any:
    """``package.module:name`` as the object it names."""
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise SystemExit(f"Expected package.module:name, got {target!r}.")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    found: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        found = getattr(found, part)
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kaimei.agent",
        description="Run a model beside Kaimei: connect out, wait for work, and answer it with the model.",
    )
    parser.add_argument("--model", required=True, help="the model's slug or id on Kaimei")
    parser.add_argument(
        "--predict",
        default=None,
        help="package.module:function, a function from a list of {feature: value} rows to predictions; or an "
        "estimator or torch module object, which is asked through predict_proba / predict / forward",
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help="the model's ONNX file: the agent runs it with ONNX Runtime, for predictions and for every step "
        "of a structure that was read from it",
    )
    parser.add_argument(
        "--providers",
        default=None,
        help="ONNX Runtime execution providers, in order, separated by commas "
        "(CUDAExecutionProvider,CPUExecutionProvider to use a GPU; the CPU when left out)",
    )
    parser.add_argument(
        "--session-memory",
        type=int,
        default=None,
        metavar="MIB",
        help="memory, in MiB, that the ONNX Runtime sessions kept ready for --onnx may take (default "
        f"{SESSION_BYTES >> 20}): a look inside a large model reads it in several runs, each with a session of its "
        "own, and as many as fit are kept for the next look",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="package.module:object, the model's tokenizer or a function from a text to its token ids, for a "
        "model whose tokenizer Kaimei cannot run itself",
    )
    parser.add_argument("--label", default=None, help="what the agent is called on the model's page")
    parser.add_argument("--api-key", default=None, help="default: the KAIMEI_API_KEY environment variable")
    parser.add_argument("--base-url", default=None, help="default: KAIMEI_BASE_URL, or https://api.kaimei.ai")
    parser.add_argument("--quiet", action="store_true", help="do not print a line for each job")
    args = parser.parse_args(argv)
    if not (args.predict or args.onnx or args.tokenizer):
        parser.error("give the agent something to answer with: --predict, --onnx or --tokenizer")
    if args.session_memory is not None and args.session_memory < 0:
        parser.error("--session-memory is a number of MiB: 0 or more")
    from .client import Kaimei

    module = None
    predict: Optional[Predict] = None
    if args.predict:
        target = _load(args.predict)
        if _is_torch(target) and not args.onnx:
            module = target
        elif callable(target) and not hasattr(target, "predict") and not _is_torch(target):
            predict = target
        else:
            predict = predict_with(target)
    providers = [item.strip() for item in args.providers.split(",") if item.strip()] if args.providers else None
    client = Kaimei(api_key=args.api_key, base_url=args.base_url)

    def say(message: str) -> None:
        print(time.strftime("%H:%M:%S"), message, flush=True)

    tokenizer = _load(args.tokenizer) if args.tokenizer else None
    agent = Agent(
        client,
        args.model,
        predict,
        module=module,
        onnx=args.onnx,
        providers=providers,
        session_memory=args.session_memory << 20 if args.session_memory is not None else None,
        tokenizer=tokenizer,
        label=args.label,
        log=None if args.quiet else say,
    )
    print(f"Kaimei agent for {args.model} at {client.base_url}. Ctrl+C stops it.", flush=True)
    restore = _stop_on_sigterm()
    try:
        agent.serve()
    except KaimeiError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        restore()
        client.close()
    return 0


def _stop_on_sigterm() -> Callable[[], None]:
    """Stop on SIGTERM as on Ctrl+C: a container is stopped with SIGTERM, and the session is then closed, so no
    job is queued for an agent that is gone. Returns what puts the previous handler back."""
    import signal

    def stopped(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt  # what Agent.serve stops on, closing its session

    try:
        previous = signal.signal(signal.SIGTERM, stopped)
    except (AttributeError, OSError, ValueError):  # not the main thread, or no SIGTERM here
        return lambda: None

    def restore() -> None:
        try:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
        except (OSError, ValueError):
            pass

    return restore


if __name__ == "__main__":  # pragma: no cover - the command line
    raise SystemExit(main())
