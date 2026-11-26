"""Which route reads a model: Kaimei's own readers, ONNX, or whichever of the two gives the model back.

These need no converter: the ONNX route is stood in for, so only the choosing is tested.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from kaimei import StructureError, estimator_to_spec, graphs  # noqa: E402
from kaimei.structure import (  # noqa: E402
    ROUTES,
    _better_through_onnx,
    _only_onnx_can_feed,
    _with_labels,
    extract_structure,
)

NAMES = ["income", "score"]


class Linear:
    coef_ = np.array([[0.25, -0.5]])
    intercept_ = np.array([0.1])
    classes_ = np.array(["no", "yes"])
    n_features_in_ = 2

    def predict_proba(self, rows):
        positive = 1.0 / (1.0 + np.exp(-(np.asarray(rows, dtype=float) @ self.coef_.T + self.intercept_)))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, rows):
        return self.classes_[(self.predict_proba(rows)[:, 1] > 0.5).astype(int)]


class Unknown:
    """Nothing the native readers know: no coefficients, no trees, no layers."""

    n_features_in_ = 2

    def predict(self, rows):
        return np.zeros(len(rows))


class OnlyImportances:
    """What the native readers can only summarise."""

    n_features_in_ = 2
    feature_importances_ = np.array([0.7, 0.3])
    n_estimators = 3

    def predict(self, rows):
        return np.zeros(len(rows))


def graph(**more):
    spec = {"format": "kaimei.structure.v1", "family": "graph", "task": "regression", "feature_names": NAMES, "graph": {"constants": {}, "nodes": [], "outputs": ["y"]}}
    spec.update(more)
    return spec


@pytest.fixture
def route(monkeypatch):
    """Stands in for the ONNX route and records how it was called."""
    calls = []
    answer = {"spec": None, "reasons": []}

    def via_onnx(estimator, *, reasons=None, **options):
        calls.append(options)
        if reasons is not None:
            reasons.extend(answer["reasons"])
        if isinstance(answer["spec"], Exception):
            raise answer["spec"]
        return answer["spec"]

    monkeypatch.setattr(graphs, "via_onnx", via_onnx)
    return calls, answer


def test_the_routes_are_auto_native_and_onnx():
    assert ROUTES == ("auto", "native", "onnx")
    with pytest.raises(StructureError) as caught:
        extract_structure(Linear(), feature_names=NAMES, via="tensorflow")
    assert caught.value.param == "via"


def test_a_model_the_native_readers_read_never_reaches_the_conversion(route):
    calls, answer = route
    answer["spec"] = graph()
    for via in ("auto", "native"):
        spec = extract_structure(Linear(), feature_names=NAMES, via=via).spec
        assert spec["family"] == "network" and spec["architecture"] == "linear" and "check" in spec
    assert calls == []
    assert estimator_to_spec(Linear(), feature_names=NAMES)["family"] == "network"
    assert calls == []


def test_onnx_is_tried_for_a_model_the_native_readers_refuse(route):
    calls, answer = route
    answer["spec"] = graph()
    spec = extract_structure(Unknown(), feature_names=NAMES, task="regression", check_rows=[[1.0, 2.0]]).spec
    assert spec["family"] == "graph"
    assert calls[0]["feature_names"] == NAMES and calls[0]["task"] == "regression" and calls[0]["check_rows"] == [[1.0, 2.0]]
    assert set(calls[0]) == {
        "feature_names", "task", "framework", "input_shape", "preprocess", "check_rows", "sample", "inputs", "output",
        "tokenizer", "labels", "max_length", "keep",
    }


def test_the_converted_file_is_kept_with_a_reading_made_from_it_and_with_no_other(route, monkeypatch):
    calls, answer = route

    def with_file(estimator, *, reasons=None, keep=None, **options):
        calls.append(options)
        keep["onnx"] = b"the file"
        return answer["spec"]

    monkeypatch.setattr(graphs, "via_onnx", with_file)
    answer["spec"] = graph()
    read = extract_structure(Unknown(), feature_names=NAMES, max_length=64)
    assert read.spec["family"] == "graph" and read.onnx == b"the file" and calls[-1]["max_length"] == 64
    assert "onnx" not in read.spec and b"the file" not in json.dumps(read.spec).encode()  # kept here, never sent
    # A conversion that was tried and is not the model: the native reading stands, and it was read from no file.
    answer["spec"] = graph(approximate=True)
    assert extract_structure(OnlyImportances(), feature_names=NAMES).onnx is None
    assert extract_structure(Linear(), feature_names=NAMES).onnx is None


def test_native_never_tries_the_conversion(route):
    calls, answer = route
    answer["spec"] = graph()
    with pytest.raises(StructureError) as caught:
        extract_structure(Unknown(), feature_names=NAMES, via="native")
    assert caught.value.code == "unsupported_estimator" and calls == []
    assert "Converting it to ONNX" not in caught.value.message


def test_the_original_error_says_why_the_conversion_was_not_used(route):
    calls, answer = route
    answer["reasons"] = ["converting a scikit-learn model needs skl2onnx (pip install skl2onnx)"]
    with pytest.raises(StructureError) as caught:
        extract_structure(Unknown(), feature_names=NAMES)
    assert caught.value.code == "unsupported_estimator"
    assert caught.value.message.startswith("Cannot read this estimator.")
    assert "Converting it to ONNX was tried as well" in caught.value.message and "pip install skl2onnx" in caught.value.message
    # With nothing to add, the error is the one the native readers gave.
    answer["reasons"] = []
    with pytest.raises(StructureError) as caught:
        extract_structure(Unknown(), feature_names=NAMES)
    assert "Converting it to ONNX" not in caught.value.message


def test_a_conversion_that_is_not_the_model_is_not_used(route):
    calls, answer = route
    answer["spec"] = graph(approximate=True)
    with pytest.raises(StructureError) as caught:
        extract_structure(Unknown(), feature_names=NAMES)
    assert "does not reproduce" in caught.value.message
    # A summary is what the native readers gave, and a conversion that is not the model does not replace it.
    summary = extract_structure(OnlyImportances(), feature_names=NAMES).spec
    assert summary["family"] == "ensemble_summary"
    assert "does not reproduce" in summary["metadata"]["onnx_route"]
    assert len(calls) == 2


def test_a_summary_or_a_drawn_reading_is_replaced_by_a_conversion_that_runs(route):
    calls, answer = route
    answer["spec"] = graph()
    assert extract_structure(OnlyImportances(), feature_names=NAMES).spec["family"] == "graph"
    answer["spec"] = None
    answer["reasons"] = ["the onnx package is not installed"]
    kept = extract_structure(OnlyImportances(), feature_names=NAMES).spec
    assert kept["family"] == "ensemble_summary"
    assert kept["metadata"]["onnx_route"] == "not used: the onnx package is not installed"
    assert kept["metadata"]["model_class"].endswith("OnlyImportances")


def test_a_failure_while_trying_the_conversion_never_takes_the_native_reading_away(route):
    calls, answer = route
    answer["spec"] = StructureError("the exporter gave up", code="unsupported_estimator")
    kept = extract_structure(OnlyImportances(), feature_names=NAMES).spec
    assert kept["family"] == "ensemble_summary" and "the exporter gave up" in kept["metadata"]["onnx_route"]


def test_a_mistake_in_the_names_is_not_hidden_by_another_route(route):
    calls, answer = route
    answer["spec"] = graph()

    class Fitted(Linear):
        feature_names_in_ = np.array(["income", "score"])

    for via in ("auto", "onnx"):
        with pytest.raises(StructureError) as caught:
            extract_structure(Fitted(), feature_names=["score", "income"], via=via)
        assert caught.value.param == "feature_names" and "different order" in caught.value.message
        with pytest.raises(StructureError) as caught:
            extract_structure(Fitted(), feature_names=["income"], via=via)
        assert caught.value.param == "feature_names" and "reads 2 features" in caught.value.message
    assert calls == []


def test_onnx_is_forced_and_says_why_it_could_not_be(route):
    calls, answer = route
    answer["spec"] = graph()
    assert extract_structure(Linear(), feature_names=NAMES, via="onnx").spec["family"] == "graph"
    assert len(calls) == 1
    answer["spec"] = None
    answer["reasons"] = ["torch is not installed", "torch is not installed"]
    with pytest.raises(StructureError) as caught:
        extract_structure(Linear(), feature_names=NAMES, via="onnx")
    assert caught.value.code == "unsupported_estimator"
    assert caught.value.message == "This model could not be converted to ONNX: torch is not installed."
    answer["reasons"] = []
    with pytest.raises(StructureError) as caught:
        extract_structure(Linear(), feature_names=NAMES, via="onnx")
    assert "no converter takes this kind of model" in caught.value.message


def test_metadata_and_on_unsupported_apply_to_whichever_route_was_taken(route):
    calls, answer = route
    answer["spec"] = graph(metadata={"route": "onnx"})
    spec = extract_structure(Unknown(), feature_names=NAMES, metadata={"run": "exp-9"}, on_unsupported="error").spec
    assert spec["metadata"] == {"route": "onnx", "run": "exp-9"}


def test_what_makes_a_native_reading_worth_converting():
    assert not _better_through_onnx({"family": "network", "layers": [{"id": "fc", "type": "dense"}], "check": {}})
    assert _better_through_onnx({"family": "network", "approximate": True})
    assert _better_through_onnx({"family": "ensemble_summary"})
    assert _better_through_onnx({"family": "network", "layers": [{"id": "attn", "type": "opaque", "op": "Attention"}]})
    # A network no check was made for is only suspect when it came from a module that could not be called.
    unchecked = {"family": "network", "layers": [{"id": "fc", "type": "dense"}]}
    assert not _better_through_onnx(unchecked) and not _better_through_onnx(unchecked, Linear())

    class Module:
        __module__ = "torch.nn.modules.container"

    assert _better_through_onnx(unchecked, Module())
    assert not _better_through_onnx({**unchecked, "check": {"rows": []}}, Module())


def test_a_sample_shows_when_only_the_conversion_can_call_a_module():
    class Module:
        __module__ = "torch.nn.modules.container"

    module = Module()
    table, picture = np.zeros((1, 5), dtype=np.float32), np.zeros((1, 3, 8, 8), dtype=np.float32)
    assert not _only_onnx_can_feed(module, None)
    assert not _only_onnx_can_feed(module, table) and not _only_onnx_can_feed(module, picture)
    assert not _only_onnx_can_feed(module, (table,)) and not _only_onnx_can_feed(module, {"x": table})
    assert _only_onnx_can_feed(module, np.zeros((1, 12, 3), dtype=np.float32))  # a series
    assert _only_onnx_can_feed(module, np.zeros((1, 16), dtype=np.int64))  # token ids
    assert _only_onnx_can_feed(module, (table, table)) and _only_onnx_can_feed(module, {"a": table, "b": table})
    assert not _only_onnx_can_feed(Linear(), np.zeros((1, 16), dtype=np.int64))  # not a module: its own readers first


def test_onnx_goes_first_for_a_module_only_it_can_call(route, monkeypatch):
    calls, answer = route
    answer["spec"] = graph()

    class Module:
        __module__ = "torch.nn.modules.container"

    def native(*args, **kwargs):
        raise AssertionError("the native readers were asked about a model they cannot call")

    monkeypatch.setattr("kaimei.structure._extract_raw", native)
    spec = extract_structure(Module(), task="regression", sample=np.zeros((1, 16), dtype=np.int64)).spec
    assert spec["family"] == "graph" and len(calls) == 1


def test_labels_name_the_classes_of_a_native_reading():
    spec = extract_structure(Linear(), feature_names=NAMES, labels=["stays", "leaves"]).spec
    assert spec["outputs"]["labels"] == ["stays", "leaves"]
    with pytest.raises(StructureError) as caught:
        extract_structure(Linear(), feature_names=NAMES, labels=["a", "b", "c"])
    assert caught.value.param == "labels"
    assert _with_labels({"family": "network"}, None) == {"family": "network"}
    named = _with_labels({"family": "network", "labels": ["class_0", "class_1"]}, ["no", "yes"])
    assert named["labels"] == ["no", "yes"] and named["outputs"] == {"labels": ["no", "yes"]}
