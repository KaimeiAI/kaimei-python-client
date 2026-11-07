"""Reading tree ensembles and models made of other models, and checking what was read.

Tree models of any size become flat arrays plus a rule for combining their leaves. Meta-estimators become a
``composite`` of their members plus what they do with their outputs. :func:`verify` then scores the result
against the fitted estimator, so a spec is only called exact when it gives the estimator's own numbers.

The API keeps an identical copy of this file at ``kaimei_api/engine/structure/ensembles.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - the structure module reports this
    np = None  # type: ignore[assignment]

#: Which reading rules produced a spec. Absent means a client from before 0.7.2, which could store a part as the whole.
READER = 2
#: Single trees above this many nodes travel as flat arrays rather than a list of node objects.
FLAT_TREE_NODES = 2000
#: Forests within both of these keep the readable ``trees`` form that older APIs understand.
INLINE_TREES = 30
INLINE_NODES = 4000
#: Stands in for an infinite threshold, the split that only separates missing values. JSON has no infinity.
NO_LIMIT = sys.float_info.max
_SAMPLE_ROWS = 64
#: Row-by-row functions a target transform can be undone with, by the name the API runs them under.
_INVERSE_NAMES = {"exp", "expm1", "square", "log", "log1p", "sqrt"}

Extract = Callable[[Any, List[str], Optional[str], Optional[str]], Dict[str, Any]]


class Unreadable(Exception):
    """This estimator can't be expressed exactly. The caller falls back to a summary or refuses."""


def _classes_of(estimator: Any) -> set:
    return {cls.__name__ for cls in type(estimator).__mro__}


def _from_library(estimator: Any, library: str) -> bool:
    """Whether this is that library's class, or someone's own subclass of one."""
    return any(str(cls.__module__ or "").startswith(library) for cls in type(estimator).__mro__)


def _labels(estimator: Any) -> Optional[List[str]]:
    # A CatBoost regressor has classes_ too, and it's empty.
    classes = _safe(estimator, "classes_")
    return None if classes is None else [str(label) for label in list(classes)] or None


def _task(labels: Optional[Sequence[str]], task: Optional[str]) -> str:
    if labels is not None and task is None:
        return "binary_classification" if len(labels) == 2 else "multiclass_classification"
    return task or "regression"


def _safe(estimator: Any, name: str) -> Any:
    try:
        return getattr(estimator, name)
    except Exception:
        return None


# ---------- flat trees ----------


def _empty_forest() -> Dict[str, List[Any]]:
    return {"roots": [], "feature": [], "threshold": [], "left": [], "right": [], "value": [], "missing_left": []}


def _concat(parts: Dict[str, List[Any]], *, strict: bool = False, float32: bool = False) -> Dict[str, Any]:
    forest: Dict[str, Any] = {
        "roots": [int(root) for root in parts["roots"]],
        "feature": np.concatenate(parts["feature"]).astype(np.int64),
        "threshold": np.clip(np.concatenate(parts["threshold"]).astype(np.float64), -NO_LIMIT, NO_LIMIT),
        "left": np.concatenate(parts["left"]).astype(np.int64),
        "right": np.concatenate(parts["right"]).astype(np.int64),
        "value": np.concatenate(parts["value"]).astype(np.float64),
        "strict": strict,
        "float32": float32,
    }
    if parts["missing_left"] and all(item is not None for item in parts["missing_left"]):
        forest["missing_left"] = np.concatenate(parts["missing_left"]).astype(np.int64)
    return forest


def _add_tree(
    parts: Dict[str, List[Any]],
    feature: Any,
    threshold: Any,
    left: Any,
    right: Any,
    value: Any,
    missing_left: Any = None,
) -> None:
    offset = sum(item.size for item in parts["feature"])
    leaf = np.asarray(feature) < 0
    parts["roots"].append(offset)
    parts["feature"].append(np.where(leaf, -1, feature))
    parts["threshold"].append(np.where(leaf, 0.0, threshold))
    parts["left"].append(np.where(leaf, -1, np.asarray(left) + offset))
    parts["right"].append(np.where(leaf, -1, np.asarray(right) + offset))
    # Only a leaf's value is ever read. Zeros for the rest compress to nothing.
    values = np.array(value, dtype=float)
    values[~leaf] = 0.0
    parts["value"].append(values)
    parts["missing_left"].append(missing_left)


def _sklearn_trees(
    members: Sequence[Any],
    *,
    classes: int = 0,
    columns: Optional[Sequence[Any]] = None,
    positions: Optional[Sequence[Any]] = None,
    one_hot: bool = False,
) -> Dict[str, Any]:
    """sklearn trees as one flat forest. ``classes`` is 0 for values, else the width of a leaf's distribution."""
    parts = _empty_forest()
    for index, member in enumerate(members):
        tree = getattr(member, "tree_", None)
        if tree is None:
            raise Unreadable("A member is not a decision tree.")
        raw = np.asarray(tree.value, dtype=float)
        if raw.ndim != 3 or raw.shape[1] != 1:
            raise Unreadable("Multi-output trees can't be run.")
        raw = raw[:, 0, :]
        feature = np.asarray(tree.feature, dtype=np.int64)
        if columns is not None:
            mapped = np.asarray(columns[index], dtype=np.int64)
            feature = np.where(feature >= 0, mapped[np.maximum(feature, 0)], -1)
        value: Any
        if classes:
            totals = raw.sum(axis=1, keepdims=True)
            shares = np.divide(raw, totals, out=np.full_like(raw, 1.0 / raw.shape[1]), where=totals != 0)
            where = list(positions[index]) if positions is not None else list(range(raw.shape[1]))
            if one_hot:
                # A vote: SAMME scores the class a tree picks at 1, and every other at -1 / (K - 1).
                value = np.full((raw.shape[0], classes), -1.0 / max(classes - 1, 1), dtype=float)
                value[np.arange(raw.shape[0]), np.asarray(where)[shares.argmax(axis=1)]] = 1.0
            else:
                value = np.zeros((raw.shape[0], classes), dtype=float)
                value[:, where] = shares
        else:
            value = raw[:, :1]
        missing = getattr(tree, "missing_go_to_left", None)
        _add_tree(
            parts,
            feature,
            tree.threshold,
            tree.children_left,
            tree.children_right,
            value,
            None if missing is None else np.asarray(missing, dtype=np.int64),
        )
    return _concat(parts, float32=True)


def _forest_spec(
    forest: Dict[str, Any],
    combine: Dict[str, Any],
    estimator: Any,
    names: List[str],
    task: Optional[str],
    framework: Optional[str],
    summary: str,
    labels: Optional[List[str]] = None,
    default_framework: str = "sklearn",
) -> Dict[str, Any]:
    labels = _labels(estimator) if labels is None else labels
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "ensemble",
        "framework": framework or default_framework,
        "architecture": "ensemble",
        "task": _task(labels, task),
        "feature_names": names,
        "classes": labels,
        "outputs": {"labels": labels} if labels else None,
        "forest": forest,
        "combine": combine,
        "n_estimators": len(forest["roots"]),
        "summary": summary,
    }
    importances = _safe(estimator, "feature_importances_")
    if importances is not None and np.size(importances) == len(names):
        spec["importances"] = np.asarray(importances, dtype=float)
    return spec


def flat_forest_spec(
    estimator: Any, members: Sequence[Any], names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    """A large average of trees, such as a forest of any size."""
    labels = _labels(estimator)
    forest = _sklearn_trees(members, classes=len(labels) if labels else 0)
    count = len(members)
    return _forest_spec(
        forest, {"method": "mean"}, estimator, names, task, framework, f"Ensemble of {count:,} trees, averaged."
    )


def flat_tree_spec(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Dict[str, Any]:
    """One tree too large to send as a list of nodes."""
    labels = _labels(estimator)
    forest = _sklearn_trees([estimator], classes=len(labels) if labels else 0)
    summary = f"Decision tree with {forest['feature'].size:,} nodes, read from tree_."
    return _forest_spec(forest, {"method": "mean"}, estimator, names, task, framework, summary)


def _margin_of(estimator: Any, rows: Any) -> Optional[Any]:
    """The estimator's raw score for ``rows``, from whichever public method gives it."""
    for name in ("decision_function", "predict"):
        method = _safe(estimator, name)
        if callable(method):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.asarray(method(rows), dtype=float)
    return None


def _gradient_boosting(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    stages = np.asarray(estimator.estimators_, dtype=object)
    if stages.ndim != 2:
        raise Unreadable("Unexpected gradient boosting stages.")
    per_stage = stages.shape[1]
    members = [stages[stage, klass] for stage in range(stages.shape[0]) for klass in range(per_stage)]
    tree_class = [klass for _ in range(stages.shape[0]) for klass in range(per_stage)]
    labels = _labels(estimator)
    forest = _sklearn_trees(members)
    combine: Dict[str, Any] = {"method": "sum", "scale": float(estimator.learning_rate), "tree_class": tree_class}
    if labels:
        combine["link"] = "softmax" if per_stage > 1 else "sigmoid"
        if getattr(estimator, "loss", None) == "exponential":
            combine["link_scale"] = 2.0
    combine["base"] = _base_score(forest, combine, estimator, names)
    return _forest_spec(
        forest,
        combine,
        estimator,
        names,
        task,
        framework,
        f"Gradient boosting: {stages.shape[0]:,} rounds of trees added to a starting score.",
    )


def _base_score(forest: Dict[str, Any], combine: Mapping[str, Any], estimator: Any, names: List[str]) -> List[float]:
    """The starting score: what the estimator's raw output has that its trees don't explain."""
    rows = np.zeros((1, len(names)), dtype=float)
    margin = _margin_of(estimator, _frame_for(estimator, rows))
    if margin is None:
        raise Unreadable("This estimator doesn't report a raw score.")
    raw = _score_forest(forest, {**combine, "base": None, "link": "identity"}, rows)
    return [float(value) for value in (margin.reshape(1, -1) - raw).ravel()]


def _hist_gradient_boosting(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    if np.any(_safe(estimator, "is_categorical_") if _safe(estimator, "is_categorical_") is not None else False):
        raise Unreadable("Categorical splits can't be run.")
    predictors = estimator._predictors
    parts = _empty_forest()
    tree_class: List[int] = []
    for stage in predictors:
        for klass, predictor in enumerate(stage):
            nodes = predictor.nodes
            leaf = nodes["is_leaf"].astype(bool)
            _add_tree(
                parts,
                np.where(leaf, -1, nodes["feature_idx"].astype(np.int64)),
                nodes["num_threshold"],
                nodes["left"].astype(np.int64),
                nodes["right"].astype(np.int64),
                nodes["value"].reshape(-1, 1),
                nodes["missing_go_to_left"].astype(np.int64),
            )
            tree_class.append(klass)
    labels = _labels(estimator)
    width = len(predictors[0]) if predictors else 1
    link_name = type(estimator._loss.link).__name__ if hasattr(estimator, "_loss") else ""
    combine: Dict[str, Any] = {
        "method": "sum",
        "tree_class": tree_class,
        "base": [float(v) for v in np.asarray(estimator._baseline_prediction).ravel()],
    }
    if labels:
        combine["link"] = "softmax" if width > 1 else "sigmoid"
    elif link_name == "LogLink":
        combine["link"] = "exp"
    elif link_name != "IdentityLink":
        raise Unreadable(f"Unknown link {link_name}.")
    return _forest_spec(
        _concat(parts),
        combine,
        estimator,
        names,
        task,
        framework,
        f"Histogram gradient boosting: {len(predictors):,} rounds of trees.",
    )


def _adaboost(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Dict[str, Any]:
    members = list(estimator.estimators_)
    if not all(hasattr(member, "tree_") for member in members):
        raise Unreadable("Boosting over something other than trees can't be run.")
    weights = np.asarray(estimator.estimator_weights_, dtype=float)[: len(members)]
    labels = _labels(estimator)
    if labels is None:
        forest = _sklearn_trees(members)
        combine: Dict[str, Any] = {"method": "median", "weights": weights}
        return _forest_spec(
            forest,
            combine,
            estimator,
            names,
            task,
            framework,
            f"AdaBoost: the weighted median of {len(members):,} trees.",
        )
    if str(getattr(estimator, "algorithm", "SAMME")) == "SAMME.R":
        raise Unreadable("Only SAMME boosting can be run.")
    classes = np.asarray(estimator.classes_)
    positions = [np.searchsorted(classes, member.classes_) for member in members]
    forest = _sklearn_trees(members, classes=len(labels), positions=positions, one_hot=True)
    scale = 1.0 / (float(weights.sum()) * max(len(labels) - 1, 1))
    combine = {"method": "sum", "weights": weights, "scale": scale, "link": "softmax"}
    return _forest_spec(
        forest, combine, estimator, names, task, framework, f"AdaBoost: a weighted vote of {len(members):,} trees."
    )


# ---------- XGBoost, LightGBM, CatBoost ----------


def _renumbered(left: Sequence[int], right: Sequence[int], is_leaf: Sequence[bool]) -> List[int]:
    """Nodes reachable from the root, parents first, so each child comes after its parent."""
    order: List[int] = []
    stack = [0]
    while stack:
        node = stack.pop()
        order.append(node)
        if not is_leaf[node]:
            stack.extend((int(right[node]), int(left[node])))
    return order


def _xgboost(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Dict[str, Any]:
    booster = estimator.get_booster() if hasattr(estimator, "get_booster") else estimator
    raw = json.loads(bytes(booster.save_raw(raw_format="json")))
    learner = raw["learner"]
    gradient = learner["gradient_booster"]
    # DART keeps ordinary trees, and at prediction time scales each by the weight it ended training with.
    # xgboost 2 wraps a gbtree and keeps them beside it. xgboost 3 keeps them on the model, as ones for plain boosting.
    outer = gradient
    if gradient.get("name") == "dart":
        gradient = gradient["gbtree"]
    if gradient.get("name") != "gbtree":
        raise Unreadable(f"Only tree boosters can be run, not {gradient.get('name')}.")
    model = gradient["model"]
    scaled = outer.get("weight_drop", model.get("weight_drop"))
    dropped = [float(weight) for weight in scaled] if scaled and any(float(w) != 1.0 for w in scaled) else None
    objective = str(learner["objective"]["name"])
    trees = list(model["trees"])
    tree_info = [int(item) for item in model["tree_info"]]
    width = max(tree_info) + 1 if tree_info else 1
    best = _safe(estimator, "best_iteration")
    if isinstance(best, int) and best >= 0 and _xgboost_stops_early(estimator):
        per_round = int(model["gbtree_model_param"].get("num_parallel_tree", 1) or 1) * width
        keep = (best + 1) * per_round
        trees, tree_info = trees[:keep], tree_info[:keep]
        dropped = dropped[:keep] if dropped is not None else None
    parts = _empty_forest()
    for tree in trees:
        if any(int(kind) != 0 for kind in tree.get("split_type", [])):
            raise Unreadable("Categorical splits can't be run.")
        left = [int(v) for v in tree["left_children"]]
        right = [int(v) for v in tree["right_children"]]
        leaf = [v == -1 for v in left]
        order = _renumbered(left, right, leaf)
        position = {node: index for index, node in enumerate(order)}
        conditions = np.asarray(tree["split_conditions"], dtype=np.float32).astype(np.float64)
        _add_tree(
            parts,
            [(-1 if leaf[n] else int(tree["split_indices"][n])) for n in order],
            [(0.0 if leaf[n] else float(conditions[n])) for n in order],
            [(0 if leaf[n] else position[left[n]]) for n in order],
            [(0 if leaf[n] else position[right[n]]) for n in order],
            np.asarray([(float(conditions[n]) if leaf[n] else 0.0) for n in order]).reshape(-1, 1),
            np.asarray([int(tree["default_left"][n]) for n in order]),
        )
    # xgboost compares single-precision values with "<".
    forest = _concat(parts, strict=True, float32=True)
    marker = _safe(estimator, "missing")
    if marker is not None and not np.isnan(float(marker)):
        if not np.isfinite(float(marker)):
            raise Unreadable("Treating infinity as missing can't be run.")
        forest["missing_value"] = float(marker)
    labels = _labels(estimator)
    native = "Booster" in _classes_of(estimator)
    combine: Dict[str, Any] = {"method": "sum", "tree_class": tree_info}
    if dropped is not None:
        combine["weights"] = dropped
    if objective == "binary:logistic" or objective == "reg:logistic":
        combine["link"] = "sigmoid"
        if labels is None and native and objective == "binary:logistic":
            labels = ["0", "1"]
    elif objective in {"multi:softprob", "multi:softmax"}:
        combine["link"] = "softmax"
        if labels is None and native:
            labels = [str(index) for index in range(width)]
    elif objective in {"count:poisson", "reg:gamma", "reg:tweedie", "survival:cox"}:
        combine["link"] = "exp"
    elif objective not in {
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:absoluteerror",
        "reg:pseudohubererror",
        "reg:linear",
    }:
        raise Unreadable(f"The objective {objective} can't be run.")
    rows = np.zeros((1, len(names)), dtype=float)
    margin = _xgboost_margin(estimator, booster, rows, names)
    forest_raw = _score_forest(forest, {**combine, "base": None, "link": "identity"}, rows)
    combine["base"] = [float(value) for value in (margin.reshape(1, -1) - forest_raw).ravel()]
    return _forest_spec(
        forest,
        combine,
        estimator,
        names,
        task,
        framework,
        f"XGBoost: {len(trees):,} trees added to a starting score.",
        labels,
        "xgboost",
    )


def _xgboost_stops_early(estimator: Any) -> bool:
    """The sklearn wrappers predict with the best round. A bare Booster predicts with every tree it has."""
    return "Booster" not in _classes_of(estimator)


def _xgboost_margin(estimator: Any, booster: Any, rows: Any, names: List[str]) -> Any:
    import xgboost

    known = list(booster.feature_names) if booster.feature_names else None
    marker = _safe(estimator, "missing")
    matrix = xgboost.DMatrix(rows, feature_names=known, missing=None if marker is None else float(marker))
    limit = _safe(estimator, "best_iteration")
    stops = isinstance(limit, int) and limit >= 0 and _xgboost_stops_early(estimator)
    window = (0, int(limit) + 1) if stops else (0, 0)
    return np.asarray(booster.predict(matrix, output_margin=True, iteration_range=window), dtype=float)


def _lightgbm_tree(root: Mapping[str, Any]) -> tuple:
    """One LightGBM tree as arrays, parents before children, from its nested form."""
    feature: List[int] = []
    threshold: List[float] = []
    left: List[int] = []
    right: List[int] = []
    value: List[float] = []
    missing: List[int] = []

    waiting: List[tuple] = [(root, -1, left)]
    while waiting:
        node, parent, side = waiting.pop()
        at = len(feature)
        if parent >= 0:
            side[parent] = at
        for column in (feature, left, right, missing):
            column.append(0)
        threshold.append(0.0)
        value.append(0.0)
        if "split_feature" not in node:
            feature[at] = -1
            value[at] = float(node.get("leaf_value", 0.0))
            continue
        if str(node.get("decision_type")) != "<=":
            raise Unreadable("Categorical splits can't be run.")
        if str(node.get("missing_type")) == "Zero":
            raise Unreadable("Treating zero as missing can't be run.")
        feature[at] = int(node["split_feature"])
        threshold[at] = float(node["threshold"])
        # A split that saw no missing values in training reads one as zero.
        routes = str(node.get("missing_type")) == "NaN"
        missing[at] = 1 if (node.get("default_left") if routes else 0.0 <= threshold[at]) else 0
        waiting.append((node["right_child"], at, right))
        waiting.append((node["left_child"], at, left))
    return (
        np.asarray(feature),
        np.asarray(threshold),
        np.asarray(left),
        np.asarray(right),
        np.asarray(value).reshape(-1, 1),
        np.asarray(missing),
    )


def _lightgbm(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Dict[str, Any]:
    booster = estimator.booster_ if hasattr(estimator, "booster_") else estimator
    dump = booster.dump_model()
    if not dump.get("objective"):
        # A custom objective: what its scores mean, and any link over them, lives in the customer's own function.
        raise Unreadable("LightGBM with a custom objective can't be read exactly.")
    objective = str(dump.get("objective"))
    width = int(dump.get("num_tree_per_iteration") or 1)
    parts = _empty_forest()
    tree_class: List[int] = []
    for index, info in enumerate(dump["tree_info"]):
        _add_tree(parts, *_lightgbm_tree(info["tree_structure"]))
        tree_class.append(index % width)
    labels = _labels(estimator)
    averaged = bool(dump.get("average_output"))
    combine: Dict[str, Any] = {"method": "mean" if averaged else "sum", "tree_class": tree_class}
    if averaged:
        combine["scale"] = float(width)
    head = objective.split(" ")[0]
    if labels is None and "Booster" in _classes_of(estimator) and head in {"binary", "multiclass", "multiclassova"}:
        labels = ["0", "1"] if head == "binary" else [str(index) for index in range(width)]
    if head in {"binary", "cross_entropy"}:
        combine["link"] = "sigmoid"
        for piece in objective.split(" ")[1:]:
            if piece.startswith("sigmoid:"):
                combine["link_scale"] = float(piece.split(":", 1)[1])
    elif head == "multiclass":
        combine["link"] = "softmax"
    elif head in {"poisson", "gamma", "tweedie"}:
        combine["link"] = "exp"
    elif head not in {"regression", "regression_l1", "huber", "fair", "quantile", "mape"}:
        raise Unreadable(f"The objective {objective} can't be run.")
    if head == "regression" and "sqrt" in objective:
        raise Unreadable("A square-root target can't be run.")
    trees = len(dump["tree_info"])
    return _forest_spec(
        _concat(parts),
        combine,
        estimator,
        names,
        task,
        framework,
        f"LightGBM: {trees:,} trees added together.",
        labels,
        "lightgbm",
    )


def _catboost(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Dict[str, Any]:
    handle, path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        estimator.save_model(path, format="json")
        with open(path, encoding="utf-8") as f:
            model = json.load(f)
    finally:
        os.unlink(path)
    if model.get("features_info", {}).get("categorical_features") or model.get("features_info", {}).get("ctrs"):
        raise Unreadable("Categorical features can't be run.")
    if "oblivious_trees" not in model:
        raise Unreadable("Only symmetric trees can be run.")
    flat = {
        int(item["feature_index"]): int(item.get("flat_feature_index", item["feature_index"]))
        for item in model["features_info"].get("float_features", [])
    }
    # "value > border" is simply false for a missing value, unless training decided that it counts as above.
    missing_goes_right = {
        int(item["feature_index"])
        for item in model["features_info"].get("float_features", [])
        if item.get("nan_value_treatment") == "AsTrue"
    }
    labels = _labels(estimator)
    params = model.get("model_info", {}).get("params") or "{}"
    params = json.loads(params) if isinstance(params, str) else params
    loss = str(((params.get("loss_function") or {}).get("type")) or "RMSE")
    parts = _empty_forest()
    first = model["oblivious_trees"][0]["leaf_values"] if model["oblivious_trees"] else []
    first_depth = len(model["oblivious_trees"][0]["splits"]) if model["oblivious_trees"] else 0
    width = max(1, len(first) // max(1, 2**first_depth))
    for tree in model["oblivious_trees"]:
        splits = tree["splits"]
        depth = len(splits)
        if any(split.get("split_type") != "FloatFeature" for split in splits):
            raise Unreadable("Only numeric splits can be run.")
        if depth > 16:
            raise Unreadable("Trees this deep can't be expanded.")
        leaves = np.asarray(tree["leaf_values"], dtype=float).reshape(-1, width)
        internal = 2**depth - 1
        count = internal + 2**depth
        feature = np.full(count, -1, dtype=np.int64)
        threshold = np.zeros(count, dtype=float)
        left = np.zeros(count, dtype=np.int64)
        right = np.zeros(count, dtype=np.int64)
        value = np.zeros((count, width), dtype=float)
        missing_left = np.ones(count, dtype=np.int64)
        # Heap layout. CatBoost's first split is the lowest bit of the leaf index, so level d reads split d.
        for level in range(depth):
            split = splits[level]
            for node in range(2**level - 1, 2 ** (level + 1) - 1):
                feature[node] = flat[int(split["float_feature_index"])]
                threshold[node] = float(np.float32(split["border"]))
                missing_left[node] = 0 if int(split["float_feature_index"]) in missing_goes_right else 1
                left[node] = 2 * node + 1
                right[node] = 2 * node + 2
        for leaf in range(2**depth):
            index = sum(((leaf >> (depth - 1 - bit)) & 1) << bit for bit in range(depth))
            value[internal + leaf] = leaves[index]
        _add_tree(parts, feature, threshold, left, right, value, missing_left)
    # CatBoost's sign is "value > border goes right", and it reads features as single precision.
    forest = _concat(parts, float32=True)
    scale_and_bias = model.get("scale_and_bias") or [1.0, [0.0]]
    combine: Dict[str, Any] = {
        "method": "sum",
        "scale": float(scale_and_bias[0]),
        "base": [float(v) for v in scale_and_bias[1]],
    }
    if labels and width == 1:
        combine["link"] = "sigmoid"
    elif labels:
        combine["link"] = "softmax"
    elif loss in {"Poisson", "Tweedie"}:
        combine["link"] = "exp"
    elif loss not in {"RMSE", "MAE", "Quantile", "MultiRMSE", "Huber", "LogCosh", "Lq", "MAPE", "Expectile"}:
        raise Unreadable(f"The loss {loss} can't be run.")
    count = len(model["oblivious_trees"])
    return _forest_spec(
        forest,
        combine,
        estimator,
        names,
        task,
        framework,
        f"CatBoost: {count:,} symmetric trees added together.",
        labels,
        "catboost",
    )


# ---------- models made of other models ----------


def _composite(
    estimator: Any,
    names: List[str],
    task: Optional[str],
    framework: Optional[str],
    members: List[Dict[str, Any]],
    combine: Dict[str, Any],
    summary: str,
) -> Dict[str, Any]:
    labels = _labels(estimator)
    if combine.get("op") != "vote":
        # Only a hard vote takes a member's label. Anything else takes its probabilities, and a tie stays a tie.
        members = [_without_tie_break(member) for member in members]
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "composite",
        "framework": framework or next((str(m.get("framework")) for m in members if m.get("framework")), "sklearn"),
        "architecture": "composite",
        "task": _task(labels, task),
        "feature_names": names,
        "classes": labels,
        "outputs": {"labels": labels} if labels else None,
        "members": members,
        "combine": combine,
        "summary": summary,
    }
    if any(member.get("approximate") or member.get("family") == "ensemble_summary" for member in members):
        spec["approximate"] = True
    return spec


def _without_tie_break(member: Dict[str, Any]) -> Dict[str, Any]:
    combine = member.get("combine") or {}
    if combine.get("op") != "ovr" or not combine.get("ties"):
        return member
    return {**member, "combine": {key: value for key, value in combine.items() if key != "ties"}}


def _linear_member(
    weight: Any,
    bias: Any,
    names: List[str],
    framework: Optional[str],
    activation: Optional[str],
    task: str,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    weight = np.asarray(weight, dtype=float)
    weight = weight.reshape(1, -1) if weight.ndim == 1 else weight
    layer: Dict[str, Any] = {
        "id": "linear",
        "type": "dense",
        "units": int(weight.shape[0]),
        "weight": weight,
        "bias": np.asarray(bias, dtype=float).ravel(),
        "label": "Linear",
    }
    if activation:
        layer["activation"] = activation
    if labels:
        layer["output_labels"] = labels
    spec: Dict[str, Any] = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "framework": framework or "sklearn",
        "architecture": "linear",
        "task": task,
        "feature_names": names,
        "summary": "Linear layer read from coef_ and intercept_.",
        "layers": [{"id": "input", "type": "input", "names": names}, layer],
    }
    if labels:
        spec["labels"] = labels
        spec["outputs"] = {"labels": labels}
    return spec


def _glm(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Optional[Dict[str, Any]]:
    """Linear coefficients under a log link: the output is the exponential of the linear score."""
    link = str(getattr(estimator, "link", "auto"))
    power = float(getattr(estimator, "power", 1.0))
    logged = "TweedieRegressor" not in _classes_of(estimator) or link == "log" or (link == "auto" and power > 0)
    if not logged:
        return None
    spec = _linear_member(estimator.coef_, [float(estimator.intercept_)], names, framework, "exp", task or "regression")
    spec["summary"] = "Linear score under a log link, read from coef_ and intercept_."
    return spec


def _platt(estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]) -> Optional[Dict[str, Any]]:
    """A binary linear SVM's probabilities: a sigmoid fitted over its margin (libsvm's sign, so flipped)."""
    labels = _labels(estimator)
    if labels is None or len(labels) != 2 or np.size(estimator.probA_) != 1:
        return None
    margin = _linear_member(estimator.coef_, estimator.intercept_, names, framework, None, "regression")
    fold = {
        "member": 0,
        "calibrators": [
            {"kind": "libsvm", "a": float(np.ravel(estimator.probA_)[0]), "b": float(np.ravel(estimator.probB_)[0])}
        ],
    }
    config = {"op": "calibrated", "folds": [fold], "classes": 2}
    return _composite(
        estimator,
        names,
        task,
        framework,
        [margin],
        config,
        "A linear SVM's margin, through the sigmoid fitted over it for probabilities.",
    )


def one_vs_rest_shares(scores: Any, config: Mapping[str, Any]) -> Any:
    """Each class's share of what one model per class gives it, read the way the library reads them.

    With a ``link`` those are margins. ``none`` says what a row every member rejects gets: an even split, or nothing.
    ``ties`` gives a tie in the shares to the larger margin, since two margins' sigmoids can both round to 1. The winner
    gets the smallest amount a number can differ, so only the decision moves.
    """
    margins = scores
    if config.get("link"):
        with np.errstate(over="ignore"):
            scores = 1.0 / (1.0 + np.exp(-margins))
    totals = scores.sum(axis=1, keepdims=True)
    empty = 0.0 if config.get("none") == "zero" else 1.0 / scores.shape[1]
    shares = np.divide(scores, totals, out=np.full_like(scores, empty), where=totals != 0)
    if config.get("ties"):
        tied = np.flatnonzero((shares == shares.max(axis=1, keepdims=True)).sum(axis=1) > 1)
        winners = margins[tied].argmax(axis=1)
        shares[tied, winners] = np.nextafter(shares[tied, winners], 2.0)
    return shares


def _linear_one_vs_rest(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str]
) -> Dict[str, Any]:
    """Multiclass probabilities normalised from one sigmoid per class, rather than a softmax."""
    weight = np.asarray(estimator.coef_, dtype=float)
    bias = np.asarray(estimator.intercept_, dtype=float).ravel()
    labels = _labels(estimator) or []
    specs = [
        _linear_member(weight[index], bias[index : index + 1], names, framework, None, "regression")
        for index in range(len(labels))
    ]
    config: Dict[str, Any] = {"op": "ovr", "members": list(range(len(specs))), "link": "sigmoid", "ties": "margin"}
    return _composite(
        estimator,
        names,
        task,
        framework,
        specs,
        config,
        f"One linear model per class ({len(specs)}), normalised. The scene draws the first; a prediction uses all.",
    )


def _is_tree(member: Any) -> bool:
    return hasattr(member, "tree_")


def _voting(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], extract: Extract
) -> Dict[str, Any]:
    members = list(estimator.estimators_)
    weights = _safe(estimator, "_weights_not_none")
    labels = _labels(estimator)
    hard = getattr(estimator, "voting", "soft") == "hard"
    if labels is not None and not hard and all(_is_tree(member) for member in members):
        forest = _sklearn_trees(members, classes=len(labels))
        combine: Dict[str, Any] = (
            {"method": "mean"} if weights is None else {"method": "mean", "weights": np.asarray(weights, dtype=float)}
        )
        return _forest_spec(
            forest, combine, estimator, names, task, framework, f"A vote of {len(members)} trees, averaged."
        )
    specs = [extract(member, names, None, framework) for member in members]
    config: Dict[str, Any] = {"op": "vote" if hard else "mean", "members": list(range(len(specs)))}
    if weights is not None:
        config["weights"] = [float(value) for value in weights]
    what = "hard vote" if hard else "weighted average" if weights is not None else "average"
    return _composite(
        estimator,
        names,
        task,
        framework,
        specs,
        config,
        f"A {what} of {len(specs)} models. The scene draws the first; a prediction uses all of them.",
    )


def _bagging(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], extract: Extract
) -> Dict[str, Any]:
    members = list(estimator.estimators_)
    columns = [np.asarray(item, dtype=np.int64) for item in estimator.estimators_features_]
    if any(np.unique(item).size != item.size for item in columns):
        raise Unreadable("Bagging that repeats a column can't be run.")
    labels = _labels(estimator)
    count = len(labels) if labels else 0
    if all(_is_tree(member) for member in members):
        positions = [np.asarray(member.classes_, dtype=np.int64) for member in members] if labels else None
        forest = _sklearn_trees(members, classes=count, columns=columns, positions=positions)
        return _forest_spec(
            forest, {"method": "mean"}, estimator, names, task, framework, f"Bagging: {len(members):,} trees, averaged."
        )
    specs = [
        extract(member, [names[int(i)] for i in column], None, framework) for member, column in zip(members, columns)
    ]
    config: Dict[str, Any] = {"op": "mean", "members": list(range(len(specs)))}
    if labels:
        if not all(callable(_safe(member, "predict_proba")) for member in members):
            config["op"] = "vote"
        config["positions"] = [[int(c) for c in member.classes_] for member in members]
        config["classes"] = count
    return _composite(
        estimator,
        names,
        task,
        framework,
        specs,
        config,
        f"Bagging: {len(specs)} models, averaged. The scene draws the first; a prediction uses all of them.",
    )


def margin_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The same model, stopped before its output becomes probabilities: what ``decision_function`` returns."""
    out = {**spec, "task": "regression", "classes": None, "labels": None, "outputs": None}
    out.pop("uncalibrated", None)
    if spec.get("forest") is not None and (spec.get("combine") or {}).get("method") == "sum":
        out["combine"] = {**spec["combine"], "link": "identity"}
        return out
    layers = list(spec.get("layers") or [])
    if layers and layers[-1].get("type") == "dense":
        last = {key: value for key, value in layers[-1].items() if key not in {"activation", "output_labels"}}
        out["layers"] = layers[:-1] + [last]
        return out
    raise Unreadable("This model's raw score can't be read.")


def _response_member(
    member: Any, names: List[str], framework: Optional[str], extract: Extract, prefer: Sequence[str]
) -> Dict[str, Any]:
    """A member as sklearn consumes it inside a stack or calibrator: its raw score or its probabilities."""
    spec = extract(member, names, None, framework)
    for method in prefer:
        if method == "decision_function" and callable(_safe(member, "decision_function")):
            margin = margin_spec(spec)
            if "AdaBoostClassifier" in _classes_of(member):
                # predict_proba divides its score by K - 1, and with two classes it holds half the difference.
                classes = len(_labels(member) or [])
                margin["combine"] = {**margin["combine"], "scale": margin["combine"]["scale"] * max(classes - 1, 2)}
            return margin
        if method == "predict_proba" and callable(_safe(member, "predict_proba")):
            return spec
        if method == "predict" and callable(_safe(member, "predict")) and _labels(member) is None:
            return spec
    raise Unreadable("This member has no output that can be read.")


def _stacking(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], extract: Extract
) -> Dict[str, Any]:
    members = list(estimator.estimators_)
    methods = list(getattr(estimator, "stack_method_", []) or ["predict"] * len(members))
    labels = _labels(estimator)
    specs: List[Dict[str, Any]] = []
    inputs: List[Dict[str, Any]] = []
    final_inputs: List[str] = []
    for index, (member, method) in enumerate(zip(members, methods)):
        spec = _response_member(member, names, framework, extract, [str(method)])
        specs.append(spec)
        drop_first = method == "predict_proba" and labels is not None and len(labels) == 2
        inputs.append({"member": index, "take": "positive" if drop_first else "all"})
        if labels is None or method != "predict_proba" and len(labels) == 2:
            width = 1
        elif drop_first:
            width = 1
        else:
            width = len(labels)
        final_inputs.extend(f"member_{index}_{column}" for column in range(width))
    passthrough = bool(getattr(estimator, "passthrough", False))
    final_names = final_inputs + (list(names) if passthrough else [])
    final = extract(estimator.final_estimator_, final_names, None, framework)
    specs.append(final)
    config = {
        "op": "stack",
        "inputs": inputs,
        "final": len(specs) - 1,
        "final_inputs": final_names,
        "passthrough": list(names) if passthrough else [],
    }
    return _composite(
        estimator,
        names,
        task,
        framework,
        specs,
        config,
        f"Stacking: {len(members)} models feed a final model. The scene draws the first; a prediction uses all.",
    )


def _calibrator_config(calibrator: Any) -> Dict[str, Any]:
    if hasattr(calibrator, "a_") and hasattr(calibrator, "b_"):
        return {"kind": "sigmoid", "a": float(calibrator.a_), "b": float(calibrator.b_)}
    if hasattr(calibrator, "X_thresholds_") and hasattr(calibrator, "y_thresholds_"):
        return {
            "kind": "isotonic",
            "x": [float(v) for v in calibrator.X_thresholds_],
            "y": [float(v) for v in calibrator.y_thresholds_],
        }
    raise Unreadable(f"Unknown calibrator {type(calibrator).__name__}.")


def _calibrated(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], extract: Extract
) -> Dict[str, Any]:
    specs: List[Dict[str, Any]] = []
    folds: List[Dict[str, Any]] = []
    for index, fold in enumerate(estimator.calibrated_classifiers_):
        inner: Any = getattr(fold, "estimator", None)
        if type(inner).__name__ == "FrozenEstimator":
            inner = inner.estimator
        specs.append(_response_member(inner, names, framework, extract, ["decision_function", "predict_proba"]))
        folds.append({"member": index, "calibrators": [_calibrator_config(item) for item in fold.calibrators]})
    labels = _labels(estimator) or []
    config = {"op": "calibrated", "folds": folds, "classes": len(labels)}
    return _composite(
        estimator,
        names,
        task,
        framework,
        specs,
        config,
        f"Calibrated across {len(specs)} folds. The scene draws the first; a prediction uses all of them.",
    )


def _read_to_its_margin(member: Any, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A member read up to the score it decides by, or ``None`` when that isn't what its last sigmoid is given."""
    layers = spec.get("layers") or []
    last = layers[-1] if layers else {}
    readable = (
        spec.get("family") == "network"
        and not spec.get("approximate")
        and last.get("type") == "dense"
        and last.get("activation") == "sigmoid"
        and int(last.get("units") or 0) == 1
    )
    if not readable or not callable(_safe(member, "decision_function")):
        return None
    if _classes_of(member) & {"SVC", "NuSVC"} and callable(_safe(member, "predict_proba")):
        return None
    return margin_spec(spec)


def _one_vs_rest(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], extract: Extract
) -> Dict[str, Any]:
    if bool(_safe(estimator, "multilabel_")):
        raise Unreadable("Multi-label models can't be run.")
    members = list(estimator.estimators_)
    specs = [extract(member, names, None, framework) for member in members]
    # scikit-learn gives a row every member rejects no probabilities at all.
    config: Dict[str, Any] = {"op": "ovr", "members": list(range(len(specs))), "none": "zero"}
    margins = [_read_to_its_margin(member, spec) for member, spec in zip(members, specs)]
    if len(members) > 1 and all(margin is not None for margin in margins):
        specs = [margin for margin in margins if margin is not None]
        config.update(link="sigmoid", ties="margin")
    spec = _composite(
        estimator,
        names,
        task,
        framework,
        specs,
        config,
        f"One model per class ({len(specs)}), normalised. The scene draws the first; a prediction uses all of them.",
    )
    if not callable(_safe(estimator, "predict_proba")):
        spec["uncalibrated"] = True
    return spec


def _inverse_name(function: Any) -> Optional[str]:
    if function is None:
        return "identity"
    name = getattr(function, "__name__", "")
    module = str(getattr(function, "__module__", "") or "")
    return (
        name if name in _INVERSE_NAMES and (type(function).__name__ == "ufunc" or module.startswith("numpy")) else None
    )


def _target(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], extract: Extract
) -> Dict[str, Any]:
    transformer = estimator.transformer_
    config: Dict[str, Any] = {"op": "target", "member": 0, "inverse": "identity"}
    if type(transformer).__name__ == "FunctionTransformer":
        inverse = _inverse_name(_safe(transformer, "inverse_func"))
        if inverse is None:
            raise Unreadable("This target transform is custom code, which can't be stored.")
        config["inverse"] = inverse
    elif hasattr(transformer, "mean_") and hasattr(transformer, "scale_"):
        config["center"] = float(np.ravel(transformer.mean_)[0]) if getattr(transformer, "with_mean", True) else 0.0
        config["scale"] = float(np.ravel(transformer.scale_)[0]) if getattr(transformer, "with_std", True) else 1.0
    elif hasattr(transformer, "center_") and hasattr(transformer, "scale_"):  # RobustScaler
        config["center"] = float(np.ravel(transformer.center_)[0]) if transformer.center_ is not None else 0.0
        config["scale"] = float(np.ravel(transformer.scale_)[0]) if transformer.scale_ is not None else 1.0
    elif hasattr(transformer, "data_min_") and hasattr(transformer, "min_"):  # MinMaxScaler
        config["center"] = -float(np.ravel(transformer.min_)[0]) / float(np.ravel(transformer.scale_)[0])
        config["scale"] = 1.0 / float(np.ravel(transformer.scale_)[0])
    else:
        raise Unreadable("This target transform can't be run.")
    spec = extract(estimator.regressor_, names, None, framework)
    return _composite(
        estimator,
        names,
        "regression",
        framework,
        [spec],
        config,
        "A regressor on a transformed target, with the transform undone.",
    )


def ensemble_spec(
    estimator: Any, names: List[str], task: Optional[str], framework: Optional[str], extract: Extract
) -> Optional[Dict[str, Any]]:
    """The spec for an ensemble or meta-estimator, or ``None`` when it is neither. Raises :class:`Unreadable`."""
    classes = _classes_of(estimator)
    if classes & {"GradientBoostingClassifier", "GradientBoostingRegressor"}:
        return _gradient_boosting(estimator, names, task, framework)
    if classes & {"HistGradientBoostingClassifier", "HistGradientBoostingRegressor"}:
        return _hist_gradient_boosting(estimator, names, task, framework)
    if classes & {"AdaBoostClassifier", "AdaBoostRegressor"}:
        return _adaboost(estimator, names, task, framework)
    if classes & {"BaggingClassifier", "BaggingRegressor"}:
        return _bagging(estimator, names, task, framework, extract)
    if classes & {"VotingClassifier", "VotingRegressor"}:
        return _voting(estimator, names, task, framework, extract)
    if classes & {"StackingClassifier", "StackingRegressor"}:
        return _stacking(estimator, names, task, framework, extract)
    if "CalibratedClassifierCV" in classes:
        return _calibrated(estimator, names, task, framework, extract)
    if "OneVsRestClassifier" in classes:
        return _one_vs_rest(estimator, names, task, framework, extract)
    if "TransformedTargetRegressor" in classes:
        return _target(estimator, names, task, framework, extract)
    if classes & {"PoissonRegressor", "GammaRegressor", "TweedieRegressor"}:
        return _glm(estimator, names, task, framework)
    if (
        classes & {"SVC", "NuSVC"}
        and getattr(estimator, "probability", False)
        and getattr(estimator, "kernel", "") == "linear"
    ):
        return _platt(estimator, names, task, framework)
    many = len(_labels(estimator) or []) > 2
    if many and "SGDClassifier" in classes and getattr(estimator, "loss", "") == "log_loss":
        return _linear_one_vs_rest(estimator, names, task, framework)
    if many and "LogisticRegression" in classes and getattr(estimator, "multi_class", "") == "ovr":
        return _linear_one_vs_rest(estimator, names, task, framework)
    if _from_library(estimator, "xgboost") and (hasattr(estimator, "get_booster") or "Booster" in classes):
        return _xgboost(estimator, names, task, framework)
    if _from_library(estimator, "lightgbm") and (hasattr(estimator, "booster_") or "Booster" in classes):
        return _lightgbm(estimator, names, task, framework)
    if _from_library(estimator, "catboost") and hasattr(estimator, "save_model"):
        return _catboost(estimator, names, task, framework)
    return None


# ---------- checking a spec against the estimator ----------


def through_scalers(steps: Sequence[Any], width: int) -> Any:
    """What a pipeline's fitted scalers make of one column's numbers, by their own arithmetic."""

    def scaled(column: int, values: Any) -> Any:
        for step in steps:
            table = np.zeros((values.size, width))
            table[:, column] = values
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")  # Fitted on a table with names, and handed an array.
                values = np.asarray(step.transform(table), dtype=float)[:, column]
        return values

    return scaled


def moved_thresholds(
    feature: Any, threshold: Any, center: Any, scale: Any, scaled: Any, *, single: bool = False, strict: bool = False
) -> Any:
    """Each split's threshold in the input's units, at the number where the library's own comparison turns over.

    Multiplying one back out lands a rounding error away, and a row sitting there would go the other way. A scaler's
    arithmetic only ever rises with its input, as does rounding to single precision, so that number can be searched
    for, and then every input goes the way the library sends it.
    """
    feature = np.asarray(feature, dtype=np.int64)
    threshold = np.asarray(threshold, dtype=float)
    moved = threshold * scale[feature] + center[feature]
    for column in np.unique(feature):
        at = np.flatnonzero(feature == column)
        moved[at] = _turning_points(moved[at], threshold[at], int(column), scaled, single, strict)
    return moved


def _turning_points(guess: Any, limits: Any, column: int, scaled: Any, single: bool, strict: bool) -> Any:
    def goes_left(values: Any, where: Any) -> Any:
        seen = scaled(column, values)
        if single:
            with np.errstate(all="ignore"):
                seen = seen.astype(np.float32).astype(np.float64)
        return seen < limits[where] if strict else seen <= limits[where]

    guess = np.clip(guess, -NO_LIMIT, NO_LIMIT)
    width = np.spacing(np.abs(guess))
    everywhere = np.arange(guess.size)
    low, high = np.clip(guess - width, -NO_LIMIT, NO_LIMIT), np.clip(guess + width, -NO_LIMIT, NO_LIMIT)
    while True:
        short = (low > -NO_LIMIT) & ~goes_left(low, everywhere)
        over = (high < NO_LIMIT) & goes_left(high, everywhere)
        if not (short.any() or over.any()):
            break
        low[short] = np.maximum(low[short] - width[short], -NO_LIMIT)
        high[over] = np.minimum(high[over] + width[over], NO_LIMIT)
        width = np.minimum(width * 2, NO_LIMIT)
    while True:
        apart = np.flatnonzero(np.nextafter(low, high) < high)
        if not apart.size:
            # A strict comparison needs the first number that doesn't go left, and the other the last that does.
            return high if strict else low
        middle = low[apart] / 2 + high[apart] / 2
        # Halving the very smallest numbers can round back onto an end of the range.
        stuck = (middle <= low[apart]) | (middle >= high[apart])
        middle[stuck] = np.nextafter(low[apart][stuck], high[apart][stuck])
        left = goes_left(middle, apart)
        low[apart[left]] = middle[left]
        high[apart[~left]] = middle[~left]


def _score_forest(forest: Mapping[str, Any], combine: Mapping[str, Any], X: Any) -> Any:
    feature = np.asarray(forest["feature"], dtype=np.int64)
    threshold = np.asarray(forest["threshold"], dtype=float)
    left = np.asarray(forest["left"], dtype=np.int64)
    right = np.asarray(forest["right"], dtype=np.int64)
    value = np.asarray(forest["value"], dtype=float).reshape(feature.size, -1)
    roots = np.asarray(forest["roots"], dtype=np.int64)
    missing = forest.get("missing_left")
    missing = None if missing is None else np.asarray(missing) > 0.5
    view = X.astype(np.float32) if forest.get("float32") else X
    if forest.get("missing_value") is not None:
        view = np.where(view == float(forest["missing_value"]), np.nan, view)
    cursor = np.broadcast_to(roots, (X.shape[0], roots.size)).copy()
    while True:
        split = feature[cursor]
        rows, trees = np.nonzero(split >= 0)
        if not rows.size:
            break
        at = cursor[rows, trees]
        seen = view[rows, split[rows, trees]]
        go_left = seen < threshold[at] if forest.get("strict") else seen <= threshold[at]
        if missing is not None:
            go_left = np.where(np.isnan(seen), missing[at], go_left)
        cursor[rows, trees] = np.where(go_left, left[at], right[at])
    leaves = value[cursor]
    method = str(combine.get("method") or "mean")
    weights = combine.get("weights")
    weights = None if weights is None else np.asarray(weights, dtype=float)
    if method == "median":
        predictions = leaves[:, :, 0]
        if weights is None:
            weights = np.ones(roots.size, dtype=float)
        order = np.argsort(predictions, axis=1)
        cdf = np.cumsum(weights[order], axis=1)
        pick = (cdf >= 0.5 * cdf[:, -1:]).argmax(axis=1)
        chosen = order[np.arange(X.shape[0]), pick]
        return predictions[np.arange(X.shape[0]), chosen].reshape(-1, 1)
    if weights is not None:
        leaves = leaves * weights[None, :, None]
    tree_class = combine.get("tree_class")
    if tree_class is not None:
        tree_class = np.asarray(tree_class, dtype=np.int64)
        total = np.zeros((X.shape[0], int(tree_class.max()) + 1), dtype=float)
        for klass in np.unique(tree_class):
            total[:, klass] = leaves[:, tree_class == klass, 0].sum(axis=1)
    else:
        total = leaves.sum(axis=1)
    if method == "mean":
        total = total / (float(weights.sum()) if weights is not None else roots.size)
    base = combine.get("base")
    total = float(combine.get("scale", 1.0)) * total
    if base is not None:
        total = np.asarray(base, dtype=float).ravel() + total
    return _link(total, str(combine.get("link") or "identity"), float(combine.get("link_scale", 1.0)))


def _link(total: Any, link: str, link_scale: float = 1.0) -> Any:
    if link == "sigmoid":
        positive = 1.0 / (1.0 + np.exp(np.clip(-link_scale * total[:, 0], -700.0, 700.0)))
        return np.stack([1.0 - positive, positive], axis=1)
    if link == "softmax":
        shifted = np.exp(total - total.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)
    if link == "exp":
        return _exp(total)
    return total


def _exp(values: Any) -> Any:
    with np.errstate(over="ignore"):
        return np.exp(values)


_ACTIVATIONS: Dict[str, Callable[[Any], Any]] = {
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": lambda x: np.tanh(x),
    "identity": lambda x: x,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(np.clip(-x, -700.0, 700.0))),
    "softmax": lambda x: _link(x, "softmax"),
    "exp": _exp,
    # What the API runs for each of these, so that a library's different formula shows up as a difference.
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
    "silu": lambda x: x / (1.0 + np.exp(np.clip(-x, -700.0, 700.0))),
    "relu6": lambda x: np.clip(x, 0.0, 6.0),
    "hardsigmoid": lambda x: np.clip(x + 3.0, 0.0, 6.0) / 6.0,
    "hardswish": lambda x: x * np.clip(x + 3.0, 0.0, 6.0) / 6.0,
}
_ALPHA_ACTIVATIONS: Dict[str, Callable[[Any, float], Any]] = {
    "leaky_relu": lambda x, alpha: np.where(x >= 0, x, alpha * x),
    "elu": lambda x, alpha: np.where(x > 0, x, alpha * np.expm1(np.minimum(x, 0.0))),
}
_ALPHA_DEFAULTS = {"leaky_relu": 0.01, "elu": 1.0}


def _activate(name: str, out: Any, layer: Mapping[str, Any]) -> Optional[Any]:
    """A layer's activation, or ``None`` when this file can't score it."""
    if name in _ALPHA_ACTIVATIONS:
        return _ALPHA_ACTIVATIONS[name](out, float(layer.get("alpha", _ALPHA_DEFAULTS[name])))
    return _ACTIVATIONS[name](out) if name in _ACTIVATIONS else None


def _evaluate(spec: Mapping[str, Any], rows: Mapping[str, Any], n: int) -> Optional[Any]:
    """The spec's output for a batch, ``(n, width)``, or ``None`` for a kind this can't score."""
    family = str(spec.get("family") or "")
    names = [str(name) for name in spec.get("feature_names") or []]
    if family == "composite":
        return _evaluate_composite(spec, rows, n)
    if any(name not in rows for name in names):
        return None
    X = (
        np.concatenate([np.asarray(rows[name], dtype=float).reshape(n, -1) for name in names], axis=1)
        if names
        else np.zeros((n, 0))
    )
    if spec.get("forest") is not None:
        combine = spec.get("combine") or {}
        scored = _score_forest(spec["forest"], combine, X)
        # A sigmoid read as one number, as XGBoost's reg:logistic, rather than as two classes.
        one_number = combine.get("link") == "sigmoid" and str(spec.get("task")) == "regression"
        return scored[:, 1:] if one_number else scored
    view = X.astype(np.float32).astype(np.float64) if spec.get("float32") else X
    if family == "ensemble" and spec.get("trees"):
        total: Any = None
        for tree in spec["trees"]:
            scored = _walk(tree, view)
            total = scored if total is None else total + scored
        return total / len(spec["trees"])
    if family == "tree":
        return _walk(spec, view)
    if family != "network":
        return None
    out: Any = X
    previous = ""
    for layer in spec.get("layers") or []:
        kind = str(layer.get("type"))
        sources = [str(source).split("__")[0] for source in layer.get("inputs") or []]
        if sources and sources != [previous]:
            return None  # A branch. This only scores layers one after another, and the API's engine scores any.
        previous = str(layer.get("id"))
        if kind in {"input", "dropout"}:
            continue
        if kind == "activation":
            out = _activate(str(layer.get("activation")), out, layer)
            if out is None:
                return None
        elif kind == "batchnorm":
            out = (out - np.asarray(layer["mean"], dtype=float)) / np.sqrt(
                np.asarray(layer["var"], dtype=float) + float(layer.get("eps") or 0.0)
            )
            if layer.get("gamma") is not None:
                out = out * np.asarray(layer["gamma"], dtype=float)
            if layer.get("beta") is not None:
                out = out + np.asarray(layer["beta"], dtype=float)
        elif kind == "standardize":
            scale = np.asarray(layer["scale"], dtype=float)
            out = (out - np.asarray(layer["center"], dtype=float)) / np.where(scale == 0, 1.0, scale)
        elif kind == "dense":
            out = out @ np.asarray(layer["weight"], dtype=float).T + np.asarray(layer["bias"], dtype=float)
            activation = layer.get("activation")
            if activation:
                out = _activate(str(activation), out, layer)
                if out is None:
                    return None
        else:
            return None
    if out.shape[1] == 1 and str(spec.get("task")) == "binary_classification":
        return np.concatenate([1.0 - out, out], axis=1)
    return out


def _walk(tree: Mapping[str, Any], X: Any) -> Any:
    catalog = {str(node["id"]): node for node in tree["nodes"]}
    rows = []
    for row in X:
        node = catalog[str(tree["root"])]
        while not node.get("leaf"):
            seen = row[int(node["feature"])]
            left = node.get("missing_left", False) if np.isnan(seen) else seen <= float(node["threshold"])
            node = catalog[str(node["left"] if left else node["right"])]
        rows.append(np.asarray(node["output"], dtype=float))
    return np.stack(rows, axis=0)


def _evaluate_composite(spec: Mapping[str, Any], rows: Mapping[str, Any], n: int) -> Optional[Any]:
    members = list(spec["members"])
    config = spec["combine"]
    op = str(config["op"])

    def run(index: int, batch: Mapping[str, Any]) -> Any:
        scored = _evaluate(members[index], batch, n)
        if scored is None:
            raise Unreadable("A member can't be scored here.")
        return scored

    try:
        if op in {"mean", "vote"}:
            positions = config.get("positions")
            classes = int(config.get("classes") or 0)
            order = list(config["members"])
            weights = (
                np.asarray(config["weights"], dtype=float) if config.get("weights") is not None else np.ones(len(order))
            )
            total: Any = None
            for slot, member in enumerate(order):
                scored = run(int(member), rows)
                if positions is not None:
                    full = np.zeros((n, classes), dtype=float)
                    full[:, positions[slot]] = scored
                    scored = full
                if op == "vote":
                    shares = np.zeros_like(scored)
                    shares[np.arange(n), scored.argmax(axis=1)] = 1.0
                    scored = shares
                total = weights[slot] * scored if total is None else total + weights[slot] * scored
            return total / weights.sum()
        if op == "ovr":
            order = list(config["members"])
            if len(order) == 1:
                positive = run(int(order[0]), rows)[:, -1]
                return np.stack([1.0 - positive, positive], axis=1)
            scores = np.stack([run(int(member), rows)[:, -1] for member in order], axis=1)
            return one_vs_rest_shares(scores, config)
        if op == "calibrated":
            return _evaluate_calibrated(config, run, rows, n)
        if op == "stack":
            columns = []
            for source in config["inputs"]:
                scored = run(int(source["member"]), rows)
                columns.append(scored[:, -1:] if source.get("take") == "positive" else scored)
            for name in config.get("passthrough") or []:
                columns.append(np.asarray(rows[str(name)], dtype=float).reshape(n, -1))
            stacked = np.concatenate(columns, axis=1)
            inputs = {name: stacked[:, i : i + 1] for i, name in enumerate(config["final_inputs"])}
            return run(int(config["final"]), inputs)
        if op == "target":
            scored = run(int(config["member"]), rows)
            unscaled = float(config.get("scale", 1.0)) * scored + float(config.get("center", 0.0))
            inverse = str(config.get("inverse") or "identity")
            return unscaled if inverse == "identity" else getattr(np, inverse)(unscaled)
    except Unreadable:
        return None
    return None


def _libsvm_two_class(first: Any) -> Any:
    """libsvm's probability for class 1 from its pairwise estimate for class 0. It solves for a pair the same way
    as for many classes, stopping within a tolerance, so the answer is close to the estimate but not equal."""
    first = np.clip(first, 1e-7, 1.0 - 1e-7)
    second = 1.0 - first
    out = np.empty_like(first)
    for row, (r01, r10) in enumerate(zip(first.tolist(), second.tolist())):
        q = [[r10 * r10, -r10 * r01], [-r01 * r10, r01 * r01]]
        p = [0.5, 0.5]
        for _ in range(100):
            qp = [q[0][0] * p[0] + q[0][1] * p[1], q[1][0] * p[0] + q[1][1] * p[1]]
            pqp = p[0] * qp[0] + p[1] * qp[1]
            if max(abs(qp[0] - pqp), abs(qp[1] - pqp)) < 0.0025:
                break
            for t in range(2):
                diff = (-qp[t] + pqp) / q[t][t]
                p[t] += diff
                pqp = (pqp + diff * (diff * q[t][t] + 2 * qp[t])) / (1 + diff) / (1 + diff)
                for j in range(2):
                    qp[j] = (qp[j] + diff * q[t][j]) / (1 + diff)
                    p[j] /= 1 + diff
        out[row] = p[1]
    return out


def _evaluate_calibrated(
    config: Mapping[str, Any], run: Callable[[int, Mapping[str, Any]], Any], rows: Mapping[str, Any], n: int
) -> Any:
    classes = int(config["classes"])

    def apply(calibrator: Mapping[str, Any], response: Any) -> Any:
        if calibrator["kind"] == "sigmoid":
            return 1.0 / (
                1.0 + np.exp(np.clip(float(calibrator["a"]) * response + float(calibrator["b"]), -700.0, 700.0))
            )
        if calibrator["kind"] == "libsvm":
            return _libsvm_two_class(
                1.0
                / (1.0 + np.exp(np.clip(-float(calibrator["a"]) * response + float(calibrator["b"]), -700.0, 700.0)))
            )
        return np.interp(response, np.asarray(calibrator["x"], dtype=float), np.asarray(calibrator["y"], dtype=float))

    total = np.zeros((n, classes), dtype=float)
    for fold in config["folds"]:
        response = run(int(fold["member"]), rows)
        proba = np.zeros((n, classes), dtype=float)
        if classes == 2:
            proba[:, 1] = apply(fold["calibrators"][0], response[:, -1])
            proba[:, 0] = 1.0 - proba[:, 1]
        else:
            for index, calibrator in enumerate(fold["calibrators"]):
                proba[:, index] = apply(calibrator, response[:, index])
            totals = proba.sum(axis=1, keepdims=True)
            proba = np.divide(proba, totals, out=np.full_like(proba, 1.0 / classes), where=totals != 0)
        proba[(proba > 1.0) & (proba <= 1.0 + 1e-5)] = 1.0
        total += proba
    return total / len(config["folds"])


def _thresholds_of(
    spec: Mapping[str, Any], names: Sequence[str], found: Dict[str, List[float]], ties: Dict[str, List[float]]
) -> None:
    own = [str(name) for name in spec.get("feature_names") or []]
    for member in spec.get("members") or []:
        _thresholds_of(member, names, found, ties)
    forest = spec.get("forest")
    if forest is not None:
        feature = np.asarray(forest["feature"], dtype=np.int64)
        threshold = np.asarray(forest["threshold"], dtype=float)
        for column in np.unique(feature[feature >= 0]):
            if int(column) < len(own):
                cuts = threshold[feature == column][:256]
                # A row can't sit near a split that only separates missing values: that threshold is infinite.
                cuts = cuts[np.abs(cuts) < 1e30].tolist()
                found.setdefault(own[int(column)], []).extend(cuts)
                if forest.get("float32"):
                    # Training rows sit exactly on a single-precision split, so check those too.
                    ties.setdefault(own[int(column)], []).extend(cuts)
    for tree in list(spec.get("trees") or []) + ([spec] if spec.get("nodes") else []):
        for node in tree.get("nodes") or []:
            if not node.get("leaf") and node.get("feature") is not None and int(node["feature"]) < len(own):
                if abs(float(node["threshold"])) < 1e30:
                    found.setdefault(own[int(node["feature"])], []).append(float(node["threshold"]))


def _missing_markers(spec: Mapping[str, Any]) -> List[float]:
    """Values a library may treat specially: zero, and any it was told means missing."""
    found = [0.0]
    marker = (spec.get("forest") or {}).get("missing_value")
    if marker is not None:
        found.append(float(marker))
    for member in spec.get("members") or []:
        found += _missing_markers(member)
    return found


def _sample_rows(spec: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Rows that land on both sides of the splits the spec actually has."""
    generator = np.random.default_rng(0)
    rows = generator.normal(size=(_SAMPLE_ROWS, len(names)))
    found: Dict[str, List[float]] = {}
    ties: Dict[str, List[float]] = {}
    _thresholds_of(spec, names, found, ties)
    for index, name in enumerate(names):
        cuts = found.get(name)
        if cuts:
            chosen = generator.choice(np.asarray(cuts, dtype=float), size=_SAMPLE_ROWS)
            rows[:, index] = chosen * (1.0 + generator.choice([-1e-3, 1e-3], size=_SAMPLE_ROWS)) + generator.choice(
                [-1e-3, 1e-3], size=_SAMPLE_ROWS
            )
        if ties.get(name):
            rows[: _SAMPLE_ROWS // 2, index] = generator.choice(
                np.asarray(ties[name], dtype=float), size=_SAMPLE_ROWS // 2
            )
    for marker in set(_missing_markers(spec)):
        tail = rows[-_SAMPLE_ROWS // 4 :]
        tail[generator.random(tail.shape) < 0.3] = marker
    return rows


def _frame_for(estimator: Any, rows: Any, names: Optional[Sequence[str]] = None) -> Any:
    """Rows in the shape the estimator was fitted on: some refuse a bare array after fitting on a table."""
    columns = _safe(estimator, "feature_names_in_")
    if columns is None and names is not None and not _takes_an_array(estimator, rows):
        columns = names
    if columns is None:
        return rows
    try:
        import pandas
    except ImportError:
        return rows
    return pandas.DataFrame(rows, columns=list(columns))


def _takes_an_array(estimator: Any, rows: Any) -> bool:
    """Someone's own wrapper may only take a table, and has no attribute that says so."""
    method = _safe(estimator, "predict")
    if not callable(method):
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            method(rows[:1])
    except Exception:
        return False
    return True


def _booster_output(estimator: Any, rows: Any) -> Any:
    """What a library's bare Booster predicts: not sklearn's ``predict_proba``, and for xgboost, not on an array."""
    if _from_library(estimator, "xgboost"):
        import xgboost

        known = list(estimator.feature_names) if estimator.feature_names else None
        return np.asarray(estimator.predict(xgboost.DMatrix(rows, feature_names=known)), dtype=float)
    return np.asarray(estimator.predict(rows), dtype=float)


def several_outputs(spec: Mapping[str, Any], names: Sequence[str]) -> bool:
    """A prediction is one number or one class, so a spec with more than one number and no classes can't be run."""
    if np is None or not names or spec.get("classes") or spec.get("labels"):
        return False
    rows = np.zeros((1, len(names)), dtype=float)
    try:
        ours = _evaluate(spec, {str(name): rows[:, index : index + 1] for index, name in enumerate(names)}, 1)
    except Exception:
        return False
    return ours is not None and ours.size > 1


def _refuses_gaps(spec: Mapping[str, Any]) -> bool:
    """A reading that left out an imputer. Where a gap goes in it has nothing to do with where the imputer sends it."""
    return spec.get("missing_values") is False or any(_refuses_gaps(member) for member in spec.get("members") or [])


def handles_missing(spec: Mapping[str, Any], estimator: Any, names: Sequence[str]) -> bool:
    """Whether the library scores rows with missing values, and this reading gives the same numbers for them."""
    if np is None or not names or spec.get("family") not in {"tree", "ensemble", "composite"} or _refuses_gaps(spec):
        return False
    rows = _sample_rows(spec, names)
    rows[np.random.default_rng(1).random(rows.shape) < 0.3] = np.nan
    return verify(spec, estimator, names, rows) is True


def verify(spec: Mapping[str, Any], estimator: Any, names: Sequence[str], rows: Any = None) -> Optional[bool]:
    """Whether running ``spec`` reproduces the estimator. ``None`` when it can't be compared here.

    An estimator that answers in another shape, or refuses the rows, isn't shown to match. One with no way of being
    asked, a bare set of numbers say, has nothing to compare with.
    """
    if np is None or not names or spec.get("family") == "ensemble_summary":
        return None
    if _counts_an_svcs_margin(estimator):
        return False
    if _classes_of(estimator) & {"SVC", "NuSVC"}:
        if len(_labels(estimator) or []) > 2 and not spec.get("members"):
            # Its coef_ holds a row per pair of classes and it picks by votes among them. Rows far from the training
            # data can agree anyway, because every vote goes the same way.
            return False
        if callable(_safe(estimator, "predict_proba")):
            # Its probabilities come from a separate fit, and predict() goes by the margin, so the two can disagree
            # on rows no sample is sure to hit.
            return False
    rows = _sample_rows(spec, names) if rows is None else rows
    try:
        ours = _evaluate(
            spec, {str(name): rows[:, index : index + 1] for index, name in enumerate(names)}, rows.shape[0]
        )
    except Exception:
        return False  # The parts don't fit together, so this reading can't be what the model does.
    if ours is None:
        return None
    labels = spec.get("classes") or spec.get("labels")
    table = _frame_for(estimator, rows, names)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if "Booster" in _classes_of(estimator):
                theirs = _booster_output(estimator, rows)
                if labels and theirs.ndim == 1 and ours.shape[1] > 2:
                    # multi:softmax answers with the class, not its probability.
                    return bool(np.array_equal(theirs.astype(np.int64), ours.argmax(axis=1)))
                if labels and theirs.ndim == 1:
                    ours = ours[:, -1:]
            elif labels:
                method = _safe(estimator, "predict_proba")
                if callable(method):
                    theirs = np.asarray(method(table), dtype=float)
                else:
                    method = _safe(estimator, "predict")
                    if not callable(method):
                        return None
                    picked = [str(label) for label in np.asarray(method(table)).ravel()]
                    return picked == [str(labels[int(index)]) for index in ours.argmax(axis=1)]
            else:
                method = _safe(estimator, "predict")
                if not callable(method):
                    return None
                theirs = np.asarray(method(table), dtype=float)
    except Exception:
        return False
    if theirs.size != ours.size:
        return False
    if not _matches(theirs.reshape(ours.shape), ours):
        return False
    # A reading that breaks ties in its probabilities does as ``predict`` does, and they can't show that themselves.
    breaks_ties = (spec.get("combine") or {}).get("ties")
    return _same_decisions(estimator, table, ours if breaks_ties else theirs.reshape(ours.shape), labels)


def _same_decisions(estimator: Any, table: Any, probabilities: Any, labels: Any) -> bool:
    """Whether ``predict`` picks the class its own probabilities do, and an even split the first of them.

    Kaimei goes by the probabilities, and an SVC or a saturated SGD classifier decides from a margin they can't show.
    """
    method = None if not labels or "Booster" in _classes_of(estimator) else _safe(estimator, "predict")
    if not callable(method) or probabilities.shape[1] < 2:
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            picked = np.asarray(method(table))
    except Exception:
        return True
    if picked.ndim > 1 and picked.shape[1] > 1:
        # Several labels per row, and a prediction reports one.
        return False
    names = [str(label) for label in labels]
    theirs = [str(label) for label in picked.ravel()]
    if not set(theirs) <= set(names):
        return True
    return theirs == [names[int(index)] for index in probabilities.argmax(axis=1)]


def rows_to_check(X: Any, names: Sequence[str], limit: int = 200) -> Optional[Any]:
    """The customer's own rows as numbers, in ``names`` order, or ``None`` when they aren't a plain table of numbers."""
    if np is None or X is None or not names:
        return None
    try:
        columns = getattr(X, "columns", None)
        if columns is not None and set(names) <= {str(column) for column in columns}:
            X = X[list(names)]
        rows = np.asarray(X, dtype=float)
    except (TypeError, ValueError, KeyError):
        return None
    if rows.ndim != 2 or rows.shape[1] != len(names):
        return None
    rows = rows[np.isfinite(rows).all(axis=1)][:limit]
    return rows if rows.shape[0] else None


#: Libraries whose own ``predict`` is what a reading of their learned numbers describes.
_READ_LIBRARIES = {"sklearn", "xgboost", "lightgbm", "catboost"}


#: Where a fitted model keeps the models and steps it's made of.
_FITTED_PARTS = ("estimators_", "estimator_", "best_estimator_", "final_estimator_", "regressor_", "estimator")
_MAX_PART_DEPTH = 8


def _parts_of(estimator: Any) -> List[Any]:
    found: List[Any] = [step for _, step in _safe(estimator, "steps") or []]
    found.extend(_safe(estimator, "calibrated_classifiers_") or [])
    for name in _FITTED_PARTS:
        value = _safe(estimator, name)
        if value is None:
            continue
        if hasattr(value, "ravel"):
            found.extend(value.ravel().tolist())
        elif isinstance(value, (list, tuple)):
            found.extend(value)
        else:
            found.append(value)
    return [part for part in found if part is not None and not isinstance(part, str)]


def changes_how_it_predicts(estimator: Any, depth: int = 0) -> bool:
    """A subclass with its own ``predict``, say a cap or a threshold, does something its numbers can't show.

    Nor can someone's own class among a model's parts. At the top the upload is compared with it, but a part is only
    ever seen through the whole, and rows around zero needn't reach what it does.
    """
    ancestry = type(estimator).__mro__
    if any(_library_of(klass) in _READ_LIBRARIES for klass in ancestry):
        for name in ("predict", "predict_proba", "decision_function", "transform"):
            for klass in ancestry:
                if name in vars(klass):
                    if _library_of(klass) not in _READ_LIBRARIES:
                        return True
                    break
    elif depth:
        return True
    if depth >= _MAX_PART_DEPTH:
        return False
    return any(changes_how_it_predicts(part, depth + 1) for part in _parts_of(estimator))


def _holds_an_svc_with_probabilities(part: Any, depth: int = 0) -> bool:
    if _classes_of(part) & {"SVC", "NuSVC"} and callable(_safe(part, "predict_proba")):
        return True
    if depth >= _MAX_PART_DEPTH:
        return False
    return any(_holds_an_svc_with_probabilities(inner, depth + 1) for inner in _parts_of(part))


def _counts_an_svcs_margin(estimator: Any, depth: int = 0) -> bool:
    """A hard vote or one-vs-rest goes by each member's decision, and an SVC's isn't the probabilities read of it."""
    classes = _classes_of(estimator)
    if "OneVsRestClassifier" in classes or ("VotingClassifier" in classes and _safe(estimator, "voting") == "hard"):
        return any(_holds_an_svc_with_probabilities(part) for part in _parts_of(estimator))
    return depth < _MAX_PART_DEPTH and any(_counts_an_svcs_margin(part, depth + 1) for part in _parts_of(estimator))


def _library_of(klass: type) -> str:
    return str(klass.__module__).split(".")[0]


def _matches(theirs: Any, ours: Any) -> bool:
    # Relative to the size of these outputs: a fixed floor would pass anything for a model whose outputs are all tiny.
    finite = np.abs(theirs[np.isfinite(theirs)])
    floor = max(1e-5 * float(finite.max(initial=0.0)), 1e-12)
    return bool(np.allclose(theirs, ours, rtol=1e-5, atol=floor))


def far_rows(names: Sequence[str]) -> List[Any]:
    """Rows far from zero, one size at a time. A cap or a clip does nothing to the rows around zero."""
    generator = np.random.default_rng(1)
    return [generator.normal(size=(_SAMPLE_ROWS // 8, len(names))) * scale for scale in (10.0, 1000.0)]


def _library_answer(estimator: Any, rows: Any, labels: Any, names: Optional[Sequence[str]] = None) -> Optional[Any]:
    """What the model's own library gives for these rows, as numbers. ``None`` when it only gives labels."""
    if _from_library(estimator, "torch") or _from_library(estimator, "keras"):
        return _network_output(estimator, rows)
    table = _frame_for(estimator, rows, names)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if "Booster" in _classes_of(estimator):
                answer = _booster_output(estimator, rows)
                return None if labels and answer.ndim == 1 and len(labels) > 2 else answer
            method = _safe(estimator, "predict_proba") if labels else _safe(estimator, "predict")
            return np.asarray(method(table), dtype=float) if callable(method) else None
    except Exception:
        return None


def upload_check(
    spec: Mapping[str, Any], estimator: Any, names: Sequence[str], own: Any = None
) -> Optional[Dict[str, Any]]:
    """Rows and the library's answers, for the API to compare with its real engine when the model arrives.

    Each group is compared at its own scale: outputs a thousand times larger would excuse any error on small ones.
    """
    if np is None or not names or spec.get("family") == "ensemble_summary":
        return None
    labels = spec.get("classes") or spec.get("labels")
    # A very wide model's rows would outweigh the model, and it has few enough ways to go wrong per row.
    keep = max(2, min(_SAMPLE_ROWS, 200_000 // max(3 * len(names), 1)))
    groups = [_sample_rows(spec, names)[:keep], *(rows[:keep] for rows in far_rows(names))]
    if own is not None:
        groups.append(own[:keep])
    rows, outputs, sizes = [], [], []
    for group in groups:
        answer = _library_answer(estimator, group, labels, names)
        if answer is None:
            return None
        finite = np.isfinite(answer).reshape(group.shape[0], -1).all(axis=1) & ~np.isinf(group).any(axis=1)
        if not finite.any():
            continue
        rows.append(group[finite])
        outputs.append(np.asarray(answer).reshape(group.shape[0], -1)[finite])
        sizes.append(int(finite.sum()))
    if not rows:
        return None
    return {
        "rows": [[None if np.isnan(value) else float(value) for value in row] for row in np.concatenate(rows).tolist()],
        "outputs": np.concatenate(outputs).tolist(),
        "groups": sizes,
    }


def image_check(
    estimator: Any, shape: Sequence[int], preprocess: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """The library's answers for a seeded noise image, which is too large to send but easy to make again."""
    if np is None:
        return None
    height, width, channels = (int(dim) for dim in shape)
    count, seed = 2, 0
    rows = np.random.default_rng(seed).random((count, height, width, channels))
    if preprocess:
        mean, std = (np.asarray(preprocess.get(key, base), dtype=float) for key, base in (("mean", 0.0), ("std", 1.0)))
        scale = np.asarray(preprocess.get("scale", 1.0), dtype=float)
        offset = np.asarray(preprocess.get("offset", 0.0), dtype=float)
        rows = (rows - mean) / std * scale + offset
    keras = _from_library(estimator, "keras")
    answer = _network_output(estimator, rows if keras else rows.transpose(0, 3, 1, 2))
    if answer is None:
        return None
    # Thousands of float32 steps drift further than a small net's do, and a wrong step moves far more than this.
    return {
        "image_shape": [height, width, channels],
        "count": count,
        "seed": seed,
        "outputs": answer.tolist(),
        "tolerance": 1e-3,
    }


def _network_output(estimator: Any, rows: Any) -> Optional[Any]:
    """What a PyTorch or Keras model itself gives for these rows, or ``None`` when it won't say here."""
    try:
        # Keras on the torch backend is a torch module too, and only Keras is told to turn Dropout off.
        if _from_library(estimator, "torch") and not _from_library(estimator, "keras"):
            import torch

            training = bool(getattr(estimator, "training", False))
            estimator.eval()
            try:
                weight = next(estimator.parameters(), None)
                kind = {"dtype": weight.dtype, "device": weight.device} if weight is not None else {}
                with torch.no_grad():
                    output = estimator(torch.as_tensor(rows, **{"dtype": torch.float32, **kind}))
            finally:
                estimator.train(training)
        else:
            output = estimator(rows.astype(np.float32), training=False)
        output = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
        return np.asarray(output, dtype=float)
    except Exception:
        return None


def network_differs(spec: Mapping[str, Any], estimator: Any, names: Sequence[str], rows: Any = None) -> bool:
    """Whether a PyTorch or Keras model answers differently from this reading. ``False`` when it can't be compared.

    Layers read one after another say nothing about what ``forward()`` does between them, such as ``F.relu`` or a skip
    connection. Only layers this file can score are compared, which leaves out convolutional networks.
    """
    if np is None or not names:
        return False
    rows = _sample_rows(spec, names) if rows is None else rows
    columns = {str(name): rows[:, index : index + 1] for index, name in enumerate(names)}
    try:
        ours = _evaluate(spec, columns, rows.shape[0])
    except ValueError:
        return True  # The layers don't line up one after another, so this reading can't be what it does.
    if ours is None:
        return False
    theirs = _network_output(estimator, rows)
    if theirs is None:
        return False
    if not (spec.get("classes") or spec.get("labels")) and theirs.reshape(rows.shape[0], -1).shape[1] > 1:
        return True  # Several outputs with no classes, and a prediction reports one number.
    if theirs.size == rows.shape[0] and ours.shape[1] == 2:
        ours = ours[:, 1:]
    if theirs.size != ours.size:
        return True
    theirs = theirs.reshape(ours.shape)
    if _matches(theirs, ours):
        return False
    # The reader adds the sigmoid or softmax a classifier that ends in logits leaves to its loss.
    layers = spec.get("layers") or []
    last = str((layers[-1] if layers else {}).get("activation"))
    return not (last in {"sigmoid", "softmax"} and _matches(_ACTIVATIONS[last](theirs), ours))
