"""Tables with columns of words: a ``ColumnTransformer`` of encoders, imputers and scalers in front of a model, read
exactly.

The most common scikit-learn pipeline for real tables fills in its missing values, one-hot encodes its columns of
words (``city``, ``plan``) and scales its numbers before the model, each kind of column in a ``Pipeline`` of its own
inside a ``ColumnTransformer``. ``skl2onnx`` writes the encoders with string tensors (and its one-hot converter cannot
read string categories in recent scikit-learn), so Kaimei writes the front of such a pipeline itself and leaves only
the model to the converter:

* a column an encoder reads becomes a ``categorical`` feature with the encoder's categories. The API sends the model a
  category's position in that list, and the graph gives for it what the encoder gives: the encoder itself is asked,
  category by category, so a dropped category (``drop``) and infrequent ones sharing a column (``min_frequency``,
  ``max_categories``) come out as they do in scikit-learn. A category that is a number or a yes or no is named by its
  text (``"3"``, ``"True"``) and given to the pipeline as the value it is;
* a column of numbers is scaled as ``StandardScaler``, ``MinMaxScaler``, ``RobustScaler`` or ``MaxAbsScaler`` scales
  it, or passed through; a scaler after an encoder scales what the encoder gives;
* a ``SimpleImputer`` first in a part says what a missing value becomes: the feature's ``baseline``, which the API
  gives the model for a value sent as ``null`` (or left out);
* the columns are joined in the order the transformer joins them, and the rest of the pipeline (the model, and
  anything between) is converted by ``skl2onnx`` from a table of numbers, then joined on.

A pipeline that starts with an encoder (with an imputer before it or not) and no ``ColumnTransformer`` is read the same
way, as one part over all of its columns.

The columns are computed in double precision, as scikit-learn computes them: a caller's number is a double, and a
time in seconds (1.7e9) is not what single precision rounds it to. The model after them is read in single precision
when that gives the pipeline's own numbers on rows from the range of the data it was fitted on, and in doubles
otherwise.

The check holds the pipeline's own answers on the caller's rows, as the doubles they are, and on rows made up where
the fitted data lies (a scaler's statistics, the encoders' categories, the caller's own values), category names
included. A classifier's reading stands only if the class Kaimei reports for every one of those rows is the class
the pipeline's own ``predict`` gives it (:func:`_settle_class`).

A pipeline fitted on an array names its columns ``x0``, ``x1``, ... or as the caller's ``feature_names`` say, and
its spec says which column of that array each feature is read from (``metadata.table_columns``): the features are
listed in the order the parts read them, and the columns the pipeline drops are left out.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

#: Encoders, scalers and imputers whose arithmetic this writes down.
_ENCODERS = ("OneHotEncoder", "OrdinalEncoder")
_SCALERS = ("StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler")
_IMPUTERS = ("SimpleImputer",)
#: What a refusal says this module writes down.
_WRITES = (
    "OneHotEncoder, OrdinalEncoder, SimpleImputer, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler and "
    "passthrough columns"
)
#: The most values of an encoder's output asked for at once, so a wide one-hot encoding stays small in memory.
_PROBE_VALUES = 2_000_000
#: Rows of the caller's table the check holds.
_OWN_ROWS = 40
#: Rows made up from the range of the data the pipeline was fitted on, which the check holds beside them.
_MADE_ROWS = 24
#: Why a classifier that scores each pair of its classes is not read, as ``graphs`` says it of a model on its own.
_PAIRS_NOT_CLASSES = (
    "its decision_function gives a score for each pair of classes (decision_function_shape='ovo'), not one "
    "for each class, which is what Kaimei reads: set decision_function_shape='ovr' on it (it needs no refit), "
    "or fit it with probability=True"
)


class _NotRead(Exception):
    """A pipeline this module does not write down, and why. ``final`` says the reason is the model's own, so the
    conversion of the whole pipeline by ``skl2onnx`` would not read it either."""

    def __init__(self, reason: str, final: bool = False) -> None:
        super().__init__(reason)
        self.final = final


class _Part(NamedTuple):
    """One transformer of the front: its name, the steps its columns go through, the columns, and its weight."""

    label: str
    imputer: Any
    encoder: Any
    scalers: List[Any]
    columns: List[str]
    weight: float


class _Front(NamedTuple):
    """What comes before the model: every column the pipeline is given (in order), the parts that read them, and how
    many of the pipeline's steps they are."""

    names: List[str]
    named: bool
    parts: List[_Part]
    title: str
    steps: int = 1


class _Refused(NamedTuple):
    """A pipeline only this module could read, and the reason it does not."""

    reason: str


def _kind(part: Any) -> str:
    """What a fitted transformer does: an identity ``FunctionTransformer`` (or ``None``) is a passthrough."""
    if part is None:
        return "passthrough"
    if isinstance(part, str):
        return part
    name = type(part).__name__
    if name == "FunctionTransformer" and getattr(part, "func", None) is None:
        return "passthrough"
    return name


def _flat(part: Any) -> List[Any]:
    """The steps a part takes its columns through: a ``Pipeline``'s (nested ones too), passthroughs left out."""
    steps = getattr(part, "steps", None)
    if isinstance(steps, list) and type(part).__name__ == "Pipeline":
        return [inner for _name, step in steps for inner in _flat(step)]
    return [] if _kind(part) == "passthrough" else [part]


def _chain(label: str, part: Any) -> Tuple[Any, Any, List[Any]]:
    """``(imputer, encoder, scalers)`` a part takes its columns through, in that order; ``_NotRead`` for anything else."""
    imputer: Any = None
    encoder: Any = None
    scalers: List[Any] = []
    for step in _flat(part):
        kind = _kind(step)
        if kind in _IMPUTERS and imputer is None and encoder is None and not scalers:
            imputer = step
        elif kind in _ENCODERS and encoder is None and not scalers:
            encoder = step
        elif kind in _SCALERS:
            scalers.append(step)
        elif kind in _IMPUTERS + _ENCODERS:
            raise _NotRead(
                f"its part '{label}' has a {kind} after another step; Kaimei reads a SimpleImputer first, then an "
                "encoder, then scalers"
            )
        else:
            raise _NotRead(
                f"its part '{label}' holds a {kind}, which Kaimei does not write down (it writes {_WRITES}). Replace "
                "that step, or register the model with an agent (kaimei.agent) to run it where it is"
            )
    return imputer, encoder, scalers


def _input_names(step: Any, given: Optional[Sequence[Any]] = None) -> Tuple[List[str], bool]:
    """The columns ``step`` was fitted on, and whether they had names (a table) or not (an array).

    A table's own names are kept. An array's columns take the caller's names (``given``, one for each column of the
    array, in its order), or ``x0``, ``x1``, ... when the caller gave none.
    """
    known = getattr(step, "feature_names_in_", None)
    if known is not None:
        return [str(item) for item in known], True
    count = getattr(step, "n_features_in_", None)
    if count is None:
        raise _NotRead("it does not say how many columns it reads")
    if given is None:
        return [f"x{index}" for index in range(int(count))], False
    names = [str(item) for item in given]
    if len(names) != int(count) or len(set(names)) != len(names):
        from .structure import StructureError

        what = (
            f"feature_names has {len(names)} name{'s' if len(names) != 1 else ''}, and this pipeline was fitted on a "
            f"table of {int(count)} column{'s' if int(count) != 1 else ''}"
            if len(names) != int(count)
            else "feature_names gives two columns the same name"
        )
        raise StructureError(
            f"{what}: give one name for each column of that table, in its order (the columns the pipeline drops too).",
            code="invalid_structure",
            param="feature_names",
        )
    return names, False


def _columns_of(names: List[str], columns: Any) -> List[str]:
    """The names of the columns a part reads, however they were chosen: names, positions, a mask or a slice."""
    import numpy

    if isinstance(columns, slice):
        if isinstance(columns.start, str) or isinstance(columns.stop, str):
            # A slice of names takes its last name too, as pandas' .loc does.
            start = names.index(str(columns.start)) if columns.start is not None else 0
            stop = names.index(str(columns.stop)) + 1 if columns.stop is not None else len(names)
            return names[start : stop : columns.step]
        return names[columns]
    if isinstance(columns, (str, int, numpy.integer)):
        columns = [columns]
    listed = list(columns)
    if not listed:
        return []
    if all(isinstance(item, (bool, numpy.bool_)) for item in listed):
        if len(listed) != len(names):
            raise ValueError("a mask of another length")
        return [name for name, keep in zip(names, listed) if keep]
    if all(isinstance(item, (int, numpy.integer)) for item in listed):
        return [names[int(index)] for index in listed]
    return [str(item) for item in listed]


def _by_name(columns: Any) -> bool:
    """Whether a part chooses its columns by name, which skl2onnx's conversion of a whole pipeline cannot read."""
    if isinstance(columns, str):
        return True
    if isinstance(columns, slice):
        return isinstance(columns.start, str) or isinstance(columns.stop, str)
    try:
        return any(isinstance(item, str) for item in list(columns))
    except TypeError:
        return False


def _front_of(estimator: Any, feature_names: Optional[Sequence[Any]] = None) -> Optional[Tuple[_Front, Any]]:
    """``(the front, the rest of the pipeline)``, ``None`` for a pipeline of another kind, ``_NotRead`` for one whose
    front has a step this module does not write down. ``feature_names`` names the columns of a pipeline fitted on
    an array (see :func:`_input_names`)."""
    steps = getattr(estimator, "steps", None)
    if not isinstance(steps, list) or len(steps) < 2:
        return None
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
    except ImportError:
        return None
    first = steps[0][1]
    parts: List[_Part] = []
    if isinstance(first, ColumnTransformer):
        if not hasattr(first, "transformers_"):
            return None
        names, named = _input_names(first, feature_names)
        weights = dict(getattr(first, "transformer_weights", None) or {})
        for label, part, columns in first.transformers_:
            kind = _kind(part)
            if kind == "drop":
                continue
            try:
                picked = _columns_of(names, columns)
            except (ValueError, IndexError) as exc:
                raise _NotRead(f"the columns its part '{label}' reads are not among those it was fitted on") from exc
            if not picked:
                continue
            imputer, encoder, scalers = (None, None, []) if kind == "passthrough" else _chain(str(label), part)
            parts.append(_Part(str(label), imputer, encoder, scalers, picked, float(weights.get(label, 1.0))))
        title, after = "ColumnTransformer", steps[1:]
    else:
        # An encoder at the start, an imputer before it or not: one part over every column.
        lead: List[Any] = []
        at: Optional[int] = None
        for index, (_name, step) in enumerate(steps[:-1]):
            kind = _kind(step)
            if kind == "passthrough":
                continue
            if kind in _IMPUTERS and not lead:
                lead.append(step)
                continue
            if kind in _ENCODERS:
                lead.append(step)
                at = index
            break
        if at is None:
            return None
        names, named = _input_names(lead[0], feature_names)
        encoder = lead[-1]
        parts.append(_Part(type(encoder).__name__, lead[0] if len(lead) == 2 else None, encoder, [], names, 1.0))
        title, after = type(encoder).__name__, steps[at + 1 :]
    if not after:
        return None
    rest = Pipeline(after) if len(after) > 1 else after[0][1]
    return _Front(names, named, parts, title, len(steps) - len(after)), rest


def column_parts(estimator: Any, feature_names: Optional[Sequence[Any]] = None) -> Optional[Tuple[Any, Any]]:
    """``(front, the rest of the pipeline)`` when this module reads ``estimator``, else ``None``.

    It reads a pipeline that starts with a fitted ``ColumnTransformer`` whose parts are all steps it writes down, or
    with an encoder. A ``ColumnTransformer`` with a part it does not write down is left to the other routes when its
    columns are chosen by position (``skl2onnx`` converts such a pipeline whole); one that chooses any by name, which
    no other route reads, is taken here to be refused with the reason. ``feature_names`` names the columns of a
    pipeline fitted on an array.
    """
    try:
        return _front_of(estimator, feature_names)
    except _NotRead as exc:
        if _chosen_by_name(estimator):
            return _Refused(str(exc)), None
        return None


def _chosen_by_name(estimator: Any) -> bool:
    """Whether a part of the pipeline's ``ColumnTransformer`` chooses its columns by name: only this module reads
    such a pipeline, where ``skl2onnx`` converts one whose columns are all chosen by position."""
    first = estimator.steps[0][1]
    transformers = getattr(first, "transformers_", None) or []
    return any(_by_name(columns) for _label, _part, columns in transformers)


def _is_gap(value: Any) -> bool:
    """Whether an imputer's ``missing_values`` is a gap no number is (NaN, pandas' NA, ``None``)."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        return True  # pandas.NA, which is neither equal nor unequal to anything


def _name_of(value: Any) -> str:
    """A category as the API names it: its text, as the value prints in Python."""
    item = value.item() if hasattr(value, "item") and not isinstance(value, (str, bytes)) else value
    return str(item)


def _key(value: Any) -> Any:
    """``value`` as a dictionary key that finds the category it equals (``3`` is ``3.0``; every gap is one)."""
    if _is_gap(value):
        return ("kaimei-gap",)
    item = value.item() if hasattr(value, "item") and not isinstance(value, (str, bytes)) else value
    try:
        hash(item)
    except TypeError:
        return ("kaimei-unhashable", repr(item))
    return item


def _probe(encoder: Any, values: List[List[Any]], dtypes: List[Any], numpy: Any) -> Any:
    """Rows of the given categories, one column per column the encoder reads, as it was fitted on them."""
    arrays = [numpy.asarray(column, dtype=dtype) for column, dtype in zip(values, dtypes)]
    fitted = getattr(encoder, "feature_names_in_", None)
    if fitted is not None:
        import pandas

        return pandas.DataFrame({str(name): array for name, array in zip(fitted, arrays)})
    if len({array.dtype for array in arrays}) == 1 and arrays[0].dtype != object:
        return numpy.column_stack(arrays)
    table = numpy.empty((len(arrays[0]), len(arrays)), dtype=object)
    for index, array in enumerate(arrays):
        table[:, index] = array
    return table


def _encoding(label: str, encoder: Any, numpy: Any) -> List[Dict[str, Any]]:
    """What a fitted encoder gives for each category of each column it reads, asked of the encoder itself.

    For each column: ``values``, the categories a caller can send (in the encoder's order), and either ``hot`` (a
    one-hot encoding: the column of this input's block each category sets, ``-1`` for none) with the block's
    ``width``, or ``codes`` (an ordinal encoding: the number each category is written as). A category an ordinal
    encoder writes as NaN (its missing value) cannot be sent, and is left out.
    """
    kind = type(encoder).__name__
    categories = [numpy.asarray(item) for item in getattr(encoder, "categories_", [])]
    if not categories or min(len(item) for item in categories) < 1:
        raise _NotRead(f"the {kind} of its part '{label}' has a column with no categories")
    one_hot = kind == "OneHotEncoder"
    if one_hot:
        widths = getattr(encoder, "_n_features_outs", None)
        if widths is None:  # a scikit-learn from before infrequent categories: only a dropped one narrows a block
            dropped = getattr(encoder, "drop_idx_", None)
            widths = [len(item) - (dropped is not None and dropped[at] is not None) for at, item in enumerate(categories)]
        starts = numpy.cumsum([0, *[int(width) for width in widths]])
    total = int(starts[-1]) if one_hot else len(categories)
    count = max(len(item) for item in categories)
    step = max(1, min(count, _PROBE_VALUES // max(total, 1)))
    found: List[List[float]] = [[] for _ in categories]
    for start in range(0, count, step):
        lines = range(start, min(count, start + step))
        probe = _probe(encoder, [[item[line % len(item)] for line in lines] for item in categories], [item.dtype for item in categories], numpy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = encoder.transform(probe)
        if tuple(out.shape) != (len(lines), total):
            raise _NotRead(f"the {kind} of its part '{label}' gives {out.shape[-1]} columns where Kaimei counts {total}")
        if not one_hot:
            dense = numpy.asarray(out, dtype=float)
            for column in range(len(categories)):
                found[column].extend(float(value) for value in dense[:, column])
            continue
        if hasattr(out, "tocoo"):
            sparse = out.tocoo()
            rows, places, values = sparse.row, sparse.col, numpy.asarray(sparse.data, dtype=float)
        else:
            dense = numpy.asarray(out, dtype=float)
            rows, places = numpy.nonzero(dense)
            values = dense[rows, places]
        hot = numpy.full((len(lines), len(categories)), -1, dtype=numpy.int64)
        for row, place, value in zip(rows, places, values):
            if value == 0.0:
                continue
            column = int(numpy.searchsorted(starts, place, side="right") - 1)
            if value != 1.0 or hot[row, column] != -1:
                raise _NotRead(f"the {kind} of its part '{label}' sets more than one column, or a value other than 1, for one category")
            hot[row, column] = int(place - starts[column])
        for column in range(len(categories)):
            found[column].extend(float(value) for value in hot[:, column])
    encoded: List[Dict[str, Any]] = []
    for column, item in enumerate(categories):
        given, size = numpy.asarray(found[column], dtype=float), len(item)
        if not numpy.array_equal(given, given[numpy.arange(len(given)) % size], equal_nan=True):
            raise _NotRead(f"the {kind} of its part '{label}' does not give one category the same numbers every time")
        given = given.tolist()
        if one_hot:
            encoded.append({"values": list(item), "hot": [int(value) for value in given[:size]], "width": int(starts[column + 1] - starts[column])})
            continue
        kept = [at for at in range(size) if numpy.isfinite(given[at])]
        encoded.append({"values": [item[at] for at in kept], "codes": [given[at] for at in kept]})
    return encoded


def _fills(part: _Part) -> List[Optional[Tuple[Any, Any]]]:
    """For each column of ``part``: ``(the value its imputer takes as missing, what it fills in)``, or ``None``."""
    imputer = part.imputer
    if imputer is None:
        return [None] * len(part.columns)
    if getattr(imputer, "add_indicator", False):
        raise _NotRead(
            f"the SimpleImputer of its part '{part.label}' adds columns that say which values were missing "
            "(add_indicator=True), which Kaimei cannot give the model: a missing value reaches it as its baseline"
        )
    found = getattr(imputer, "statistics_", None)
    stats = list(found) if found is not None else []
    if len(stats) != len(part.columns) or any(_is_gap(value) for value in stats):
        raise _NotRead(f"the SimpleImputer of its part '{part.label}' leaves out a column it saw no values of")
    return [(imputer.missing_values, value) for value in stats]


def _index_of(values: Sequence[Any], wanted: Any) -> Optional[int]:
    """The position of ``wanted`` among ``values``, or ``None``."""
    key = _key(wanted)
    for index, value in enumerate(values):
        if _key(value) == key:
            return index
    return None


def _layout(front: _Front, numpy: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """``(parts, columns)``: each part as the graph writes it, and each column it reads as a feature.

    A column of ``columns`` holds its ``feature`` (``type``, ``categories``, ``baseline``), the values its categories
    are (``values``, which the pipeline is given), and what its imputer takes as missing (``gap``) and fills in
    (``fill``). A column of numbers whose imputer takes a number as missing has that number as ``marker``.
    """
    columns: Dict[str, Dict[str, Any]] = {}

    def declare(name: str, entry: Dict[str, Any]) -> None:
        earlier = columns.get(name)
        if earlier is None:
            columns[name] = entry
            return
        plain = all(item["feature"]["type"] == "numeric" and "gap" not in item for item in (earlier, entry))
        if not plain:
            raise _NotRead(
                f"its column '{name}' is read by two of its parts, as a category or with an imputer in one of "
                "them; Kaimei reads such a column once"
            )

    parts: List[Dict[str, Any]] = []
    for part in front.parts:
        fills = _fills(part)
        entry: Dict[str, Any] = {"part": part}
        if part.encoder is not None:
            encoded = _encoding(part.label, part.encoder, numpy)
            if len(encoded) != len(part.columns):
                raise _NotRead(f"the encoder of its part '{part.label}' reads {len(encoded)} columns of the {len(part.columns)} it is given")
            for position, name in enumerate(part.columns):
                piece = encoded[position]
                listed = [_name_of(value) for value in piece["values"]]
                if len(set(listed)) != len(listed):
                    raise _NotRead(f"two categories of its column '{name}' print as the same text, so a caller could not tell them apart")
                column: Dict[str, Any] = {"feature": {"type": "categorical", "categories": listed}, "values": piece["values"]}
                filling = fills[position]
                if filling is not None:
                    column["gap"], column["fill"] = filling
                    at = _index_of(piece["values"], column["fill"])
                    if at is not None:
                        column["feature"]["baseline"] = listed[at]
                else:
                    # An encoder that met missing values keeps a category for them (NaN, None): a null is that category.
                    at = next((index for index, value in enumerate(piece["values"]) if _is_gap(value)), None)
                    if at is not None:
                        column["feature"]["baseline"] = listed[at]
                        column["missing"] = True
                declare(name, column)
                piece["column"] = name
            entry["encoded"] = encoded
            entry["width"] = sum(piece["width"] if "hot" in piece else 1 for piece in encoded)
        else:
            for position, name in enumerate(part.columns):
                column = {"feature": {"type": "numeric"}}
                filling = fills[position]
                if filling is not None:
                    gap, fill = filling
                    try:
                        number = float(fill)
                    except (TypeError, ValueError):
                        number = float("nan")
                    if not numpy.isfinite(number):
                        raise _NotRead(f"its part '{part.label}' fills the column '{name}' with {fill!r}, and no encoder reads it after")
                    column.update(gap=gap, fill=number)
                    column["feature"]["baseline"] = number
                    if not _is_gap(gap):
                        column["marker"] = float(gap)
                declare(name, column)
            entry["width"] = len(part.columns)
        for scaler in part.scalers:
            seen = int(getattr(scaler, "n_features_in_", entry["width"]))
            if seen != entry["width"]:
                raise _NotRead(f"the {type(scaler).__name__} of its part '{part.label}' was fitted on {seen} columns, and the part gives {entry['width']}")
        parts.append(entry)
    return parts, columns


def _front_graph(
    layout: List[Dict[str, Any]], columns: Dict[str, Dict[str, Any]], precision: Any, opset: int, numpy: Any
) -> Tuple[Any, List[str]]:
    """``(the front as an ONNX model, the columns in the order it reads them)``. Its output, ``kaimei_table``, is the
    table of numbers the model is given, in the precision the model was converted in (``precision``).

    The front itself is computed in double precision whatever the model's: a caller's number is a double, and
    scikit-learn scales it as one. Read in single precision, a time in seconds (1.7e9, which single precision holds
    to the nearest 128) would be rounded before the scaler ever saw it. What the front gives is rounded to single
    precision only as it is handed to a model converted in single precision.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    real = TensorProto.DOUBLE
    given = TensorProto.DOUBLE if numpy.dtype(precision) == numpy.float64 else TensorProto.FLOAT
    nodes: List[Any] = []
    constants: List[Any] = []
    pieces: List[str] = []
    read: List[str] = []

    def constant(name: str, value: Any, dtype: Any = numpy.float64) -> str:
        constants.append(numpy_helper.from_array(numpy.asarray(value, dtype=dtype), name))
        return name

    def source(name: str) -> str:
        if name not in read:
            read.append(name)
        return name

    def node(op: str, inputs: List[str], output: str, label: str, **attrs: Any) -> str:
        nodes.append(helper.make_node(op, inputs, [output], name=label, **attrs))
        return output

    width = 0
    for index, entry in enumerate(layout):
        part: _Part = entry["part"]
        label, tag = part.label, f"kaimei_t{index}"
        outputs: List[str] = []
        if "encoded" in entry:
            for position, piece in enumerate(entry["encoded"]):
                name, at = piece["column"], f"{tag}_{position}"
                codes = source(name)
                if "hot" not in piece:
                    if piece["codes"] == [float(value) for value in range(len(piece["codes"]))]:
                        outputs.append(node("Cast", [codes], f"{at}_ordinal", f"{label}/{name}/ordinal", to=real))
                    else:
                        table = constant(f"{at}_codes", piece["codes"])
                        outputs.append(node("Gather", [table, codes], f"{at}_ordinal", f"{label}/{name}/ordinal", axis=0))
                    continue
                owners: Dict[int, List[int]] = {}
                for category, column in enumerate(piece["hot"]):
                    if column >= 0:
                        owners.setdefault(column, []).append(category)
                if all(len(owners.get(column, [])) == 1 for column in range(piece["width"])):
                    # Each column of the block is one category's: the category's position is compared with each.
                    compared, keys = codes, [owners[column][0] for column in range(piece["width"])]
                else:
                    # Infrequent categories share a column: which column each category sets is looked up first.
                    table = constant(f"{at}_columns", piece["hot"], numpy.int64)
                    compared = node("Gather", [table, codes], f"{at}_column", f"{label}/{name}/group", axis=0)
                    keys = list(range(piece["width"]))
                equal = node("Equal", [compared, constant(f"{at}_keys", [keys], numpy.int64)], f"{at}_equal", f"{label}/{name}/compare")
                outputs.append(node("Cast", [equal], f"{at}_hot", f"{label}/{name}/one_hot", to=real))
        else:
            for position, name in enumerate(part.columns):
                current, at = source(name), f"{tag}_{position}"
                marker = columns[name].get("marker") if "gap" in columns[name] else None
                if marker is not None:
                    # The imputer takes a number as missing: that number becomes what it fills in.
                    gap = node("Equal", [current, constant(f"{at}_marker", [[marker]])], f"{at}_gap", f"{label}/{name}/missing")
                    current = node("Where", [gap, constant(f"{at}_fill", [[columns[name]["fill"]]]), current], f"{at}_filled", f"{label}/{name}/fill")
                outputs.append(current)
        current = outputs[0] if len(outputs) == 1 else node("Concat", outputs, f"{tag}_joined", f"{label}/join", axis=1)
        for step, scaler in enumerate(part.scalers):
            current = _scaled(scaler, current, f"{tag}_s{step}", label, node, constant, numpy)
        if part.weight != 1.0:
            current = node("Mul", [current, constant(f"{tag}_weight", [[part.weight]])], f"{tag}_weighted", f"{label}/weight")
        pieces.append(current)
        width += int(entry["width"])
    if given == real:
        node("Concat", pieces, "kaimei_table", "columns/join", axis=1)
    else:
        joined = node("Concat", pieces, "kaimei_table_double", "columns/join", axis=1)
        node("Cast", [joined], "kaimei_table", "columns/single_precision", to=given)
    graph_inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT64 if columns[name]["feature"]["type"] == "categorical" else real, [1, 1])
        for name in read
    ]
    graph = helper.make_graph(nodes, "columns", graph_inputs, [helper.make_tensor_value_info("kaimei_table", given, [1, width])], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", int(opset))], producer_name="kaimei")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model, read


def _scaled(scaler: Any, current: str, tag: str, label: str, node: Callable[..., str], constant: Callable[..., str], numpy: Any) -> str:
    """``current`` scaled as ``scaler`` scales each of its columns."""
    kind = type(scaler).__name__
    path = f"{label}/{kind}"

    def row(values: Any) -> List[Any]:
        return [numpy.asarray(values, dtype=float).reshape(-1)]

    if kind == "StandardScaler":
        if getattr(scaler, "with_mean", True) and getattr(scaler, "mean_", None) is not None:
            current = node("Sub", [current, constant(f"{tag}_mean", row(scaler.mean_))], f"{tag}_centred", f"{path}/centre")
        if getattr(scaler, "with_std", True) and getattr(scaler, "scale_", None) is not None:
            current = node("Div", [current, constant(f"{tag}_scale", row(scaler.scale_))], f"{tag}_scaled", f"{path}/scale")
    elif kind == "MinMaxScaler":
        current = node("Mul", [current, constant(f"{tag}_scale", row(scaler.scale_))], f"{tag}_times", f"{path}/scale")
        current = node("Add", [current, constant(f"{tag}_min", row(scaler.min_))], f"{tag}_shifted", f"{path}/shift")
        if getattr(scaler, "clip", False):
            low, high = (float(value) for value in scaler.feature_range)
            current = node("Clip", [current, constant(f"{tag}_low", low), constant(f"{tag}_high", high)], f"{tag}_clipped", f"{path}/clip")
    elif kind == "RobustScaler":
        if getattr(scaler, "center_", None) is not None:
            current = node("Sub", [current, constant(f"{tag}_centre", row(scaler.center_))], f"{tag}_centred", f"{path}/centre")
        if getattr(scaler, "scale_", None) is not None:
            current = node("Div", [current, constant(f"{tag}_scale", row(scaler.scale_))], f"{tag}_scaled", f"{path}/scale")
    else:  # MaxAbsScaler
        current = node("Div", [current, constant(f"{tag}_scale", row(scaler.scale_))], f"{tag}_scaled", f"{path}/scale")
    return current


def _dtype_of(values: Sequence[Any], numpy: Any) -> Any:
    """The dtype a column of these categories had: their own, with text as Python objects (as pandas keeps it)."""
    dtype = numpy.asarray(values).dtype
    return object if dtype.kind in "OUS" else dtype


def _own_rows(check_rows: Any, front: _Front, columns: Dict[str, Dict[str, Any]], numpy: Any) -> List[Dict[str, Any]]:
    """The caller's rows as the model's inputs: a category as its position, a number as the double it is (the
    pipeline is asked about the caller's own numbers, not about what single precision rounds them to). A row with a
    value no feature can be sent (a category the encoder does not know, a gap) is left out.

    A table that has the pipeline's columns by name is read by name. Any other table, and an array, is read by
    position when it has as many columns as the pipeline was fitted on."""
    if check_rows is None:
        return []
    names = list(columns)
    values: Dict[str, List[Any]] = {}
    framed = hasattr(check_rows, "iloc") and hasattr(check_rows, "columns")
    present = {str(column): column for column in check_rows.columns} if framed else {}
    if framed and all(name in present for name in names):
        taken = check_rows.head(_OWN_ROWS)
        values = {name: list(taken[present[name]]) for name in names}
    elif framed and front.named:
        return []
    else:
        table = check_rows.to_numpy(dtype=object) if hasattr(check_rows, "to_numpy") else check_rows
        try:
            table = numpy.asarray(table, dtype=object)[:_OWN_ROWS]
        except (TypeError, ValueError):
            return []
        if table.ndim != 2 or table.shape[1] != len(front.names):
            return []
        values = {name: list(table[:, front.names.index(name)]) for name in names}
    count = len(next(iter(values.values()))) if values else 0
    lookups = {
        name: {_key(value): at for at, value in reversed(list(enumerate(entry["values"])))}
        for name, entry in columns.items()
        if entry["feature"]["type"] == "categorical"
    }
    rows: List[Dict[str, Any]] = []
    for line in range(count):
        feeds: Dict[str, Any] = {}
        for name in names:
            value = values[name][line]
            if name in lookups:
                at = lookups[name].get(_key(value))
                if at is None:
                    break
                feeds[name] = numpy.asarray([[at]], dtype=numpy.int64)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                break
            if not numpy.isfinite(number):
                break
            feeds[name] = numpy.asarray([[number]], dtype=numpy.float64)
        else:
            rows.append(feeds)
    return rows


def _ranges(front: _Front, columns: Dict[str, Dict[str, Any]], given: Sequence[Mapping[str, Any]], numpy: Any) -> Dict[str, Any]:
    """Where the values of each column of numbers lie in the data the pipeline was fitted on, as far as it is known.

    For each: ``("normal", mean, deviation)`` or ``("between", low, high)`` from the statistics of the scaler that
    reads the column first (a ``StandardScaler``'s mean and deviation, a ``MinMaxScaler``'s smallest and largest
    value, a ``RobustScaler``'s median and quartile range); else ``("values", [...])``, the caller's own values of
    the column; else ``None``. A ``MaxAbsScaler`` and an imputer know how large the values are and not how far
    apart, which is what decides whether single precision can tell them apart: they leave a column unknown.
    """
    found: Dict[str, Any] = {}
    for part in front.parts:
        if part.encoder is not None or not part.scalers:
            continue
        scaler = part.scalers[0]
        kind = type(scaler).__name__

        def of(name: str, at: int, scaler: Any = scaler) -> Optional[float]:
            listed = getattr(scaler, name, None)
            try:
                value = float(numpy.asarray(listed, dtype=float).reshape(-1)[at]) if listed is not None else None
            except (TypeError, ValueError, IndexError):
                value = None
            return value if value is not None and numpy.isfinite(value) else None

        for at, column in enumerate(part.columns):
            if column in found or columns.get(column, {}).get("feature", {}).get("type") != "numeric":
                continue
            if kind == "StandardScaler":
                mean, spread = of("mean_", at), of("var_", at)
                if mean is not None and spread is not None:
                    found[column] = ("normal", mean, float(numpy.sqrt(spread)))
            elif kind == "MinMaxScaler":
                low, high = of("data_min_", at), of("data_max_", at)
                if low is not None and high is not None:
                    found[column] = ("between", low, high)
            elif kind == "RobustScaler":
                centre, spread = of("center_", at), of("scale_", at)
                if centre is not None and spread is not None:
                    found[column] = ("between", centre - spread, centre + spread)
    for column, entry in columns.items():
        if entry["feature"]["type"] != "numeric":
            continue
        if column not in found:
            held = [float(numpy.asarray(row[column]).reshape(-1)[0]) for row in given]
            found[column] = ("values", held) if held else None
    return found


def _made_rows(
    front: _Front, columns: Dict[str, Dict[str, Any]], ranges: Mapping[str, Any], count: int, numpy: Any
) -> List[Dict[str, Any]]:
    """Made-up rows from the range of the data the pipeline was fitted on (see :func:`_ranges`), as the model's
    inputs: each category in turn, and for a column of numbers a value drawn where its data lies. Rows made up
    around zero would say nothing of a column whose values are far from it, and the answers they give (a prediction
    for the year 1970, say) can be so large that any error on real rows looks small beside them.

    A column whose range is not known is drawn as near to its data as the pipeline says: around what its imputer
    fills in, within its ``MaxAbsScaler``'s largest value, or else around zero.
    """
    generator = numpy.random.default_rng(0)
    rough: Dict[str, Tuple[float, float]] = {}
    for part in front.parts:
        if part.encoder is not None:
            continue
        for at, column in enumerate(part.columns):
            if ranges.get(column) is not None or column in rough or column not in columns:
                continue
            largest = next((getattr(step, "max_abs_", None) for step in part.scalers[:1] if type(step).__name__ == "MaxAbsScaler"), None)
            if largest is not None and numpy.isfinite(float(largest[at])) and float(largest[at]) > 0:
                rough[column] = (-float(largest[at]), float(largest[at]))
            elif "fill" in columns[column]:
                fill = float(columns[column]["fill"])
                rough[column] = (fill - max(abs(fill), 1.0) * 0.5, fill + max(abs(fill), 1.0) * 0.5)
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        row: Dict[str, Any] = {}
        for place, (column, entry) in enumerate(columns.items()):
            if entry["feature"]["type"] == "categorical":
                size = len(entry["values"])
                # Each category in turn at first (two columns out of step with each other), then at random.
                at = (index + place) % size if index < size else int(generator.integers(0, size))
                row[column] = numpy.asarray([[at]], dtype=numpy.int64)
                continue
            known = ranges.get(column)
            if known is None:
                low, high = rough.get(column, (None, None))
                value = float(generator.uniform(low, high)) if low is not None else float(generator.normal())
            elif known[0] == "normal":
                value = float(generator.normal(known[1], known[2]))
            elif known[0] == "between":
                value = float(generator.uniform(known[1], known[2]))
            else:
                value = float(known[1][int(generator.integers(0, len(known[1])))])
            row[column] = numpy.asarray([[value]], dtype=numpy.float64)
        rows.append(row)
    return rows


def _tells_doubles_apart(rest: Any, tables: Sequence[Any], width: int, numpy: Any) -> bool:
    """Whether the model after the columns answers a number differently from what single precision rounds it to.

    A linear model, a support vector machine or a network does: scikit-learn computes them in doubles, and so must
    Kaimei for a column of large numbers that differ by little (a time in seconds). A tree does not: scikit-learn
    rounds a tree's inputs to single precision itself, and a tree read in doubles would send a number that lies
    between two single-precision ones down another branch than scikit-learn does.

    The model is asked about ``tables`` (what the columns give it for the rows of the check) and about whole numbers
    single precision cannot hold (odd ones past 2**24), each as it is and as single precision rounds it, by each
    method it has that gives numbers. One answer that differs says it does. ``False`` when the model cannot be asked.
    """
    from . import graphs

    # Every method the model has that gives numbers is asked (its scores before they are squashed first), as a bare
    # model is: a naive Bayes model's probabilities are 0 and 1 far from its data in either precision, and its
    # log-probabilities are not.
    fits = [table for table in tables if getattr(table, "ndim", 0) == 2 and table.shape[1] == width]
    return graphs._tells_doubles_apart(rest, None, fits, width, numpy)


def _given_to_the_model(estimator: Any, front: _Front, tables: Sequence[Any], numpy: Any) -> List[Any]:
    """What the pipeline's own first steps (its columns) give the model for each of ``tables``, as one array of
    doubles; nothing when they cannot be asked."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = [estimator[: front.steps].transform(table) for table in tables]
        rows = [numpy.asarray(item.toarray() if hasattr(item, "toarray") else item, dtype=numpy.float64) for item in found]
        return [numpy.vstack(rows)] if rows else []
    except Exception:  # noqa: BLE001 - a pipeline that cannot be cut there: the model is asked about other numbers
        return []


def _fills_hold(
    columns: Dict[str, Dict[str, Any]], table: Callable[[Dict[str, Any]], Any], method: Callable[[Any], Any], base: Dict[str, Any], numpy: Any
) -> None:
    """Refuse an imputer whose missing value does not give what its fill gives, which a baseline would hide."""
    for name, column in columns.items():
        if "baseline" not in column["feature"] or "gap" not in column:
            continue
        missing, filled = dict(base), dict(base)
        missing[name] = column["gap"]
        filled[name] = column["fill"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                same = numpy.allclose(
                    numpy.asarray(method(table(missing)), dtype=float), numpy.asarray(method(table(filled)), dtype=float), rtol=1e-9, atol=1e-12
                )
        except Exception:  # noqa: BLE001 - whatever the pipeline raises, the fill could not be shown to hold
            same = False
        if not same:
            raise _NotRead(
                f"a missing {name} does not give what the SimpleImputer's fill ({column['fill']!r}) gives, so Kaimei "
                "could not give a missing value as the pipeline reads it"
            )


def through_graph(
    estimator: Any,
    why: List[str],
    numpy: Any,
    *,
    feature_names: Optional[Sequence[Any]] = None,
    task: Optional[str] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    check_rows: Any = None,
    labels: Optional[Sequence[Any]] = None,
    output: Any = None,
    tokenizer: Any = None,
    framework: Optional[str] = None,
    name: Optional[str] = None,
    keep: Optional[Dict[str, Any]] = None,
    **_unused: Any,
) -> Optional[Dict[str, Any]]:
    """A pipeline with a column transformer in front (see the module docstring) as a ``family: "graph"`` spec.

    ``feature_names`` names the columns of a pipeline fitted on an array, one for each column of that array in its
    order; a pipeline fitted on a table keeps the table's own names. When this module does not write the pipeline
    down and its columns are all chosen by position, ``skl2onnx`` is given the whole pipeline to convert (with
    ``inputs``, ``output`` and ``tokenizer``, which only that conversion reads), as it was before this module read
    any.
    """
    parts = column_parts(estimator, feature_names)
    if parts is None:
        why.append("it does not start with a ColumnTransformer of encoders, imputers and scalers, or with an encoder")
        return None
    front, rest = parts
    if isinstance(front, _Refused):
        why.append(front.reason)
        return None
    try:
        import onnx  # noqa: F401
        import skl2onnx  # noqa: F401
    except ImportError:
        why.append("reading a table with columns of words needs onnx and skl2onnx (pip install skl2onnx)")
        return None
    try:
        return _read(estimator, front, rest, numpy, task=task, mask_layout=mask_layout, threshold=threshold,
                     check_rows=check_rows, labels=labels, framework=framework, name=name, keep=keep)
    except _NotRead as exc:
        why.append(str(exc))
        if exc.final or _chosen_by_name(estimator):
            return None
    # Its columns are all chosen by position: skl2onnx converts such a pipeline whole, as it did before this module
    # wrote any of them down, and says why when it cannot.
    from . import graphs

    return graphs._sklearn_through_onnx(
        estimator, why, numpy, feature_names=feature_names, task=task, inputs=inputs, check_rows=check_rows,
        labels=labels, mask_layout=mask_layout, threshold=threshold, output=output, tokenizer=tokenizer,
        framework=framework, name=name, keep=keep,
    )


def _read(
    estimator: Any,
    front: _Front,
    rest: Any,
    numpy: Any,
    *,
    task: Optional[str],
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    check_rows: Any,
    labels: Optional[Sequence[Any]],
    framework: Optional[str],
    name: Optional[str],
    keep: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    from onnx import compose

    from . import graphs
    from .structure import StructureError

    layout, columns = _layout(front, numpy)
    width = sum(int(entry["width"]) for entry in layout)
    if not width:
        raise _NotRead("its ColumnTransformer gives the model no columns")
    final = rest.steps[-1][1] if hasattr(rest, "steps") else rest
    found = getattr(final, "classes_", None)
    if found is not None and graphs._per_output(found):
        raise _NotRead(_per_output_reason(found), final=True)
    classes = list(found) if found is not None and not graphs._is_regressor(final) else []
    classifier = len(classes) >= 2
    # A yes or a no for each label (fitted on a table of them): its probabilities are each label's own.
    multilabel = classifier and graphs._multilabel(rest, final, width, None, numpy)
    clusterer = not classifier and graphs._is_clusterer(final)
    method, kind = graphs._cluster_answer(estimator) if clusterer else graphs._sklearn_answer(estimator, classifier)
    if method is None:
        raise _NotRead("it has no predict, predict_proba or decision_function to compare a conversion with", final=True)
    if classifier and kind == "decision_function" and len(classes) > 2 and graphs._pairwise(final):
        # Three classes have three pairs: the scores would fit a head of three classes, and be read as them.
        raise _NotRead(_PAIRS_NOT_CLASSES, final=True)
    task = _task_of(task, classes, classifier, multilabel, clusterer)
    # The caller's names may be an array (an encoder's classes_), which has no truth value of its own.
    given_labels = list(labels) if labels is not None else []
    named = given_labels or ([_name_of(item) for item in classes] if classifier else None)
    dtypes = {
        column: _dtype_of(entry["values"], numpy) if entry["feature"]["type"] == "categorical" else numpy.float64
        for column, entry in columns.items()
    }

    def table(row: Dict[str, Any]) -> Any:
        """One row as the pipeline was fitted on its rows: a table with every column it was given, or an array."""
        if not front.named:
            line = numpy.empty((1, len(front.names)), dtype=object)
            for index, column in enumerate(front.names):
                line[0, index] = row.get(column, numpy.nan)
            return line
        import pandas

        data: Dict[str, Any] = {}
        for column in front.names:
            value = row.get(column, numpy.nan)
            try:
                data[column] = pandas.Series([value], dtype=dtypes.get(column))
            except (TypeError, ValueError):
                data[column] = pandas.Series([value])
        return pandas.DataFrame(data)

    def values_of(held: Mapping[str, Any]) -> Dict[str, Any]:
        """The model's inputs for one row (or a row of a check) as the pipeline's own values: a category by the
        position it is held as, a number as the double it is."""
        return {column: _value_of(columns, column, numpy.asarray(held[column])) for column in columns}

    def place(held: Mapping[str, Any]) -> Tuple[float, ...]:
        return tuple(float(numpy.asarray(held[column]).reshape(-1)[0]) for column in columns)

    def rest_answer(feeds: Dict[str, Any]) -> Any:
        rows = numpy.asarray(next(iter(feeds.values())), dtype=float).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rest.decision_function(rows)

    hints = {column: {"features": [column], "describe": dict(entry["feature"])} for column, entry in columns.items()}
    given = _own_rows(check_rows, front, columns, numpy)
    _fills_hold(columns, table, method, values_of(given[0]) if given else _a_row(columns), numpy)
    # The rows the check holds: the caller's own, as the doubles they are, and rows made up where the fitted data
    # lies. Single precision is kept only when it gives the pipeline's numbers on them.
    ranges = _ranges(front, columns, given, numpy)
    unplaced = [column for column, known in ranges.items() if known is None]
    made = _made_rows(front, columns, ranges, _MADE_ROWS, numpy)
    # A column whose range nothing says is made up without knowing where its real values lie, so the rows cannot
    # show which precision gives the pipeline's numbers there. The model is then computed as it computes itself: in
    # doubles when it tells doubles apart.
    doubles = bool(unplaced) and _tells_doubles_apart(
        rest, _given_to_the_model(estimator, front, [table(values_of(row)) for row in made], numpy), width, numpy
    )
    rows = [*given, *made]
    ours = {place(row) for row in rows}
    tools = graphs._importer()
    failures: List[str] = []

    def read(kind: str, method: Callable[[Any], Any], raw_scores: bool = False) -> Optional[Tuple[Dict[str, Any], Any]]:
        """``(spec, the ONNX model it was read from)`` of the pipeline read by one of its methods (``kind`` is its
        name), or ``None`` with the reason in ``failures``. ``raw_scores`` asks skl2onnx for a classifier's scores
        where it would give its probabilities."""

        def library(feeds: Dict[str, Any]) -> Any:
            if place(feeds) not in ours:
                # A row made up elsewhere, around zero whatever the column holds: far from this table's data it
                # proves nothing, and the size of its answer would hide an error on the rows that are near it.
                raise ValueError("not a row from the range of this table's data")
            return numpy.asarray(method(table(values_of(feeds))), dtype=float).reshape(1, -1)

        best: Optional[Tuple[Dict[str, Any], Any]] = None
        for precision in (numpy.float32, numpy.float64):
            if precision is numpy.float64 and best is None:
                # Doubles mend a reading single precision came close to, not one it could not make; and skl2onnx
                # never returns from converting in doubles a tree with an infinite threshold (one that split on
                # missing values).
                break
            try:
                data = graphs._convert_sklearn(rest, final, width, precision, classifier, numpy, raw_scores=raw_scores)
            except graphs._NoConversion as exc:
                failures.append(str(exc))
                continue
            back = tools.load_model(data)
            read_from: Optional[str] = None
            if classifier and kind == "decision_function" and len(classes) == 2 and not multilabel:
                # One margin, where the conversion gives two scores: the column that is the margin is the output.
                back, read_from = graphs._one_margin(tools, back, rest_answer, width, precision, numpy)
            opset = max([17, *[int(item.version) for item in back.opset_import if item.domain in ("", "ai.onnx")]])
            model, order = _front_graph(layout, columns, precision, opset, numpy)
            back_input = str(back.graph.input[0].name)
            back = compose.add_prefix(back, prefix="model/")
            model.ir_version = back.ir_version = max(int(model.ir_version), int(back.ir_version))
            joined = compose.merge_models(model, back, io_map=[("kaimei_table", f"model/{back_input}")])
            try:
                spec = graphs._read_onnx(
                    joined,
                    library=library,
                    feature_names=None,
                    task=task,
                    labels=named,
                    inputs={column: hints[column] for column in order},
                    input_shapes=None,
                    samples=rows[0],
                    output=f"model/{read_from}" if read_from else None,
                    mask_layout=mask_layout,
                    threshold=threshold,
                    name=name,
                    tokenizer=None,
                    check_rows=None,
                    framework=framework or "sklearn",
                    more_samples=rows[1:],
                    # predict_proba's numbers are the probabilities: shown as they are, never through a softmax again.
                    probabilities=classifier and kind == "predict_proba",
                )
            except StructureError as exc:
                if exc.param in ("labels", "inputs", "output", "tokenizer", "threshold"):
                    raise  # the caller's own argument: another precision would not make it right
                failures.append(exc.message)
                continue
            spec.setdefault("metadata", {})["model_precision"] = "double" if precision is numpy.float64 else "single"
            if best is None or not spec.get("approximate"):
                best = (spec, joined)
            if not spec.get("approximate") and not (doubles and precision is numpy.float32):
                # Single precision stands when it gives the pipeline's numbers on rows from the data's own range.
                # Where a column's range is not known the rows could not show it, and a model that tells doubles
                # apart is read in doubles when that reading is the pipeline too.
                break
        return best

    def own_class(row: Mapping[str, Any]) -> Any:
        """What the pipeline's own ``predict`` gives one row of a check."""
        return estimator.predict(table(values_of(row)))

    def read_again(kind: str) -> Optional[Tuple[Dict[str, Any], Any]]:
        other = getattr(estimator, kind, None)
        return read(kind, other, True) if callable(other) else None

    got = read(kind, method)
    if got is None:
        raise _NotRead(
            "skl2onnx could not convert the model after the columns (" + "; ".join(dict.fromkeys(failures)) + "); "
            "replace that step, or register the model with an agent (kaimei.agent) to run it where it is"
        )
    if classifier:
        got, kind = _settle_class(got, kind, final, classes, own_class, read_again)
    spec, joined = got
    if keep is not None:
        keep["onnx"] = bytes(joined.SerializeToString())
    steps = len(spec["graph"].get("nodes") or [])
    used = sorted({type(step).__name__ for part in front.parts for step in (part.imputer, part.encoder, *part.scalers) if step is not None})
    listed = f" ({', '.join(used)})" if used and front.title == "ColumnTransformer" else ""
    categorical = [column for column, entry in columns.items() if entry["feature"]["type"] == "categorical"]
    spec["summary"] = (
        f"scikit-learn {front.title}{listed} and {type(final).__name__} read as a graph of {steps:,} operations, with "
        f"{len(categorical)} column{'s' if len(categorical) != 1 else ''} of categories. "
        f"{_missing_note(columns)}{_unknown_note(front, columns)}{spec.get('summary') or ''}"
    ).strip()
    metadata = spec.setdefault("metadata", {})
    metadata["library_method"] = kind
    if not front.named:
        # Fitted on an array: which column of that array each feature is read from, for whoever pairs a table's
        # columns with the features by position (the reference rows of an upload).
        metadata["table_columns"] = [front.names.index(str(column)) for column in spec.get("feature_names") or []]
    if kind == "decision_function" and classifier:
        spec["summary"] += _scores_note(multilabel)
    if not spec.get("approximate"):
        spec["summary"] += _precision_note(metadata.get("model_precision"), unplaced)
    return spec


def _either(names: Sequence[str], limit: int = 6) -> str:
    """Column names as a summary lists them: ``age``, ``age or income``, ``age, income or 3 others``."""
    shown = [str(name) for name in names[:limit]]
    if len(names) > limit:
        shown.append(f"{len(names) - limit} other{'s' if len(names) - limit != 1 else ''}")
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " or " + shown[-1]


def _precision_note(precision: Any, unplaced: Sequence[str]) -> str:
    """What the summary says of the precision a reading is computed in, and of rows it could not be checked near."""
    note = ""
    if precision == "double" and not unplaced:
        note += (
            " The model after the columns is computed in double precision: in single precision it did not give the "
            "pipeline's own numbers on rows from the range of its data."
        )
    if unplaced:
        note += (
            f" Nothing in the pipeline says where the values of {_either(unplaced)} lie, and it was given no rows of its "
            "table to tell, so it was checked on made-up values of them that may lie far from yours: give rows of the "
            "table (X= or check_rows=) to check it on values like yours."
        )
        if precision == "double":
            note += (
                " The model after the columns is computed in double precision, as scikit-learn computes a model that "
                "tells apart numbers single precision cannot."
            )
    return note


def _unknown_note(front: _Front, columns: Dict[str, Dict[str, Any]]) -> str:
    """What the summary says of a category the encoder never saw, where the pipeline's own encoder would take it."""
    lenient = [
        column
        for part in front.parts
        if part.encoder is not None and str(getattr(part.encoder, "handle_unknown", "error")) != "error"
        for column in part.columns
        if column in columns and columns[column]["feature"]["type"] == "categorical"
    ]
    if not lenient:
        return ""
    return (
        f"A value of {_either(lenient)} that its encoder never saw is refused: Kaimei takes the categories the encoder "
        "learned, where the pipeline's own encoder would let such a value pass (handle_unknown). "
    )


def _settle_class(
    read: Any,
    kind: str,
    final: Any,
    classes: Sequence[Any],
    own_class: Callable[[Mapping[str, Any]], Any],
    read_again: Callable[[str], Any],
) -> Tuple[Any, str]:
    """A classifier pipeline's reading, settled so that the class Kaimei reports for a row is the pipeline's own.

    ``read`` is a reading as its route keeps it, a tuple whose first item is the spec; ``kind`` the method it was
    read by; ``own_class`` gives the pipeline's own ``predict`` for one row of a check; ``read_again(kind)`` reads the
    pipeline by another of its methods (``None`` when it cannot). Returns the reading that stands and its method.

    This is the rule ``graphs`` applies to a scikit-learn model on its own, with its functions. A threshold wrapper's
    threshold is carried first (``graphs.decision_threshold``, ``graphs.carry_threshold``). Then Kaimei's class and
    the pipeline's are compared on every row of the check (``graphs.classes_differ``). Where they differ and the
    probabilities were read, the scores of ``decision_function`` are read instead, and kept if their class is the
    pipeline's on every row. When no reading's is, the reading is not the pipeline: drawn, not run, with the
    number of rows said (``graphs.not_its_class``). A reading whose rows all agree still says, and warns, when its
    rule is known to differ elsewhere: an SVC that counts one-vs-one votes (``graphs.VOTES_NOTE``).
    """
    from . import graphs

    spec = read[0]
    if spec.get("approximate") or "check" not in spec or str(spec.get("task") or "") not in graphs._CLASSIFIERS:
        return read, kind
    wrapper = graphs.decision_threshold(final)
    if wrapper is not None and str(wrapper["method"]) != kind:
        # Its threshold is on the scores of the method that was not read: that method is read in its place.
        found = read_again(str(wrapper["method"]))
        if found is None or found[0].get("approximate") or "check" not in found[0]:
            _not_its_threshold(
                spec,
                f"its threshold is set on its {wrapper['method']}, which could not be read as a graph, and its {kind} "
                "is what was read",
            )
            return read, kind
        read, kind, spec = found, str(wrapper["method"]), found[0]
    if wrapper is not None:
        refused = graphs.carry_threshold(spec, wrapper, classes, kind)
        if refused is not None:
            _not_its_threshold(spec, refused)
            return read, kind
    differ, count = graphs.classes_differ(spec, own_class, classes)
    pairwise = len(classes) > 2 and graphs._pairwise(final)
    if differ and wrapper is None and kind == "predict_proba" and not pairwise:
        found = read_again("decision_function")
        if found is not None and not found[0].get("approximate") and "check" in found[0]:
            again, compared = graphs.classes_differ(found[0], own_class, classes)
            if compared and not again:
                found[0]["summary"] = (
                    f"{found[0].get('summary') or ''} Its predict() gives another class than the most probable one of "
                    f"its predict_proba on {differ:,} of the {count:,} rows it was checked on, so its "
                    "decision_function is what was read: Kaimei's class for a row is the pipeline's own."
                ).strip()
                read, kind, spec, differ = found, "decision_function", found[0], 0
    votes = kind == "decision_function" and graphs._counts_votes(final)
    if differ:
        why = "its predict() counts one-vs-one votes, which the highest one-vs-rest score does not always follow" if votes else ""
        graphs.not_its_class(spec, differ, count, why)
        return read, kind
    if votes:
        spec.setdefault("metadata", {})["predicted_class"] = graphs.VOTES_NOTE
        spec["summary"] = (
            f"{spec.get('summary') or ''} Its class for a row is the highest one-vs-rest score, which the model's own "
            "predict() can order differently where one-vs-one votes tie."
        ).strip()
        warnings.warn(
            "Kaimei runs this pipeline's decision_function, read as a graph, and its class for a row is the class with "
            f"the highest score. {graphs.VOTES_NOTE} Set break_ties=True on the model (it needs no refit) for predict() "
            "to go by the same scores.",
            UserWarning,
            stacklevel=5,
        )
    return read, kind


def _not_its_threshold(spec: Dict[str, Any], why: str) -> None:
    """Mark a reading of a pipeline that predicts with a threshold of its own which the reading cannot carry: its
    class would not be the pipeline's, so it is drawn, not run, and its summary says ``why`` (a clause)."""
    from . import graphs

    graphs._drawn(spec, f"{graphs.NOT_ITS_CLASS}: {why}")
    spec.pop("check", None)
    spec.setdefault("metadata", {})["predicted_class"] = f"Not the model's own: {why}."


def _per_output_reason(found: Any) -> str:
    """Why a model that answers each of several outputs on its own is not read here."""
    each = "a yes or a no for each" if all(len(list(item)) == 2 for item in found) else "a class for each"
    return (
        f"it ends in a model that answers {each} of its {len(found)} outputs on its own (a multi-output classifier, "
        "whose classes_ is a list), which Kaimei does not read as one model; register the model with an agent "
        "(kaimei.agent) to run it where it is"
    )


def _task_of(task: Optional[str], classes: Sequence[Any], classifier: bool, multilabel: bool, clusterer: bool) -> str:
    """The task the model is read for: the caller's, or what the model is. A multi-label model is not another kind."""
    from .structure import StructureError

    if task is None:
        if clusterer:
            return "clustering"
        if multilabel:
            return "multilabel_classification"
        if classifier:
            return "binary_classification" if len(classes) == 2 else "multiclass_classification"
        return "regression"
    if multilabel and task in ("binary_classification", "multiclass_classification"):
        raise StructureError(
            f"This classifier answers each of its {len(classes)} labels with a yes or a no of its own (it was fitted "
            f"on a table of them), so it is not a {task.split('_')[0]} classifier: pass "
            "task='multilabel_classification', or leave task out.",
            code="invalid_structure",
            param="task",
        )
    return task


def _scores_note(multilabel: bool) -> str:
    """What the summary says of a classifier that gives scores, not probabilities."""
    shown = (
        "each label's share shown is its own score squashed between 0 and 1"
        if multilabel
        else "the shares shown are those scores squashed to add up to one"
    )
    return f" This classifier gives scores, not probabilities: {shown}."


def _value_of(columns: Dict[str, Dict[str, Any]], column: str, feed: Any) -> Any:
    """What a feed of the model's input is as a value of the pipeline's column."""
    value = feed.reshape(-1)[0]
    entry = columns[column]
    return entry["values"][int(value)] if entry["feature"]["type"] == "categorical" else float(value)


def _a_row(columns: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A row of the pipeline's columns when the caller gave none: each column's first category, or its fill, or 0."""
    row: Dict[str, Any] = {}
    for column, entry in columns.items():
        if entry["feature"]["type"] == "categorical":
            row[column] = entry["values"][0]
        else:
            row[column] = float(entry.get("fill", 0.0))
    return row


def _missing_note(columns: Dict[str, Dict[str, Any]]) -> str:
    """What the summary says about missing values: how to send one, and for which columns none can be sent."""
    filled = [column for column, entry in columns.items() if "gap" in entry and "baseline" in entry["feature"]]
    kept = [column for column, entry in columns.items() if entry.get("missing")]
    unknown = [column for column, entry in columns.items() if "gap" in entry and "baseline" not in entry["feature"]]
    def either(names: List[str]) -> str:
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]

    note = ""
    if filled:
        note += (
            f"A missing {either(filled)} is sent as null (or left out): the model is given what the pipeline's "
            "SimpleImputer fills in, the feature's baseline. "
        )
    if kept:
        note += (
            f"A missing {either(kept)} is sent as null (or left out): it is the category its encoder keeps for a "
            "missing value. "
        )
    if unknown:
        note += (
            f"A missing {either(unknown)} cannot be sent: the SimpleImputer fills in a category the encoder after "
            "it does not know. "
        )
    return note


__all__ = ["column_parts", "through_graph"]
