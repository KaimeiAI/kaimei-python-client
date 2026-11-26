"""Round 3 fixes to the conversion, the reading routes and the upload (graphs.py, structure.py, resources/models.py).

The first part is the rule every classifier's reading now answers to: a reading is the model only when, besides its
scores, the class Kaimei reports for a row is the class the model's own ``predict`` gives it, on every row of its
check. The rest are smaller fixes, each with the call that used to fail.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import warnings

import httpx
import pytest
import respx

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper  # noqa: E402

from kaimei import APIStatusError, Kaimei, StructureError, graphs, structure  # noqa: E402
from kaimei.resources import models as models_module  # noqa: E402
from kaimei.structure import extract_structure, finalize_spec, spec_is_executable  # noqa: E402

BASE = "https://api.kaimei.ai"
RNG = np.random.default_rng(0)
X = RNG.normal(size=(150, 4))
NAMES = ["f0", "f1", "f2", "f3"]
YES_NO = (X[:, 0] + 0.5 * X[:, 1] + RNG.normal(scale=0.5, size=150) > 0.6).astype(int)
THREE = np.digitize(X[:, 0] + 0.3 * X[:, 1], [-0.5, 0.5])


@pytest.fixture(scope="module")
def sklearn():
    pytest.importorskip("skl2onnx")
    return pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def client():
    made = Kaimei(api_key="km_test_x", base_url=BASE, max_retries=0)
    yield made
    made.close()


@pytest.fixture
def api():
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


def asked_of(model):
    """What the model's own ``predict`` gives one row of a check, as ``classes_differ`` asks for it."""
    return lambda row: model.predict(np.array([[row[name][0] for name in NAMES]]))


def check_table(spec):
    check = spec["check"]
    return np.array([[check["inputs"][name][row][0] for name in NAMES] for row in range(len(check["outputs"]))])


def kaimei_classes(spec, classes):
    """The class Kaimei reports for every row of the spec's check, by the rule the API reports one with."""
    return [classes[graphs.reported_class(spec, row)] for row in spec["check"]["outputs"]]


# --- 1. the class Kaimei reports is the model's own ---------------------------------------------------------------


def test_the_rule_a_class_is_reported_by():
    two = {"task": "binary_classification", "outputs": {"labels": ["no", "yes"]}}
    picked = [graphs.reported_class(two, row) for row in ([0.4, 0.6], [0.6, 0.4], [0.5, 0.5], [0.7], [0.5], [0.2])]
    assert picked == [1, 0, 0, 1, 0, 0]  # an even split is the first class; one value is the second class's share
    # Within rounding of an even split is an even split, as the API's engine takes it.
    assert graphs.reported_class(two, [0.4999999999999999, 0.5000000000000001]) == 0
    cut = {"task": "binary_classification", "outputs": {"labels": ["no", "yes"], "threshold": 0.3}}
    assert [graphs.reported_class(cut, row) for row in ([0.69, 0.31], [0.71, 0.29], [0.7, 0.3])] == [1, 0, 1]
    first = {"task": "binary_classification", "outputs": {"labels": ["no", "yes"], "threshold": 0.3, "positive_label": "no"}}
    assert [graphs.reported_class(first, row) for row in ([0.31, 0.69], [0.29, 0.71])] == [0, 1]
    assert graphs.reported_class({"task": "multiclass_classification"}, [0.2, 0.4, 0.4]) == 1  # the first of the highest
    many = {"task": "multilabel_classification", "outputs": {"labels": ["a", "b", "c"]}}
    assert graphs.reported_class(many, [0.5, 0.4, 0.9]).tolist() == [True, False, True]


def test_classes_are_compared_on_every_row_of_either_kind_of_check():
    graph = {
        "task": "multiclass_classification",
        "outputs": {"labels": ["a", "b", "c"]},
        "check": {
            "inputs": {"x": [[0.0], [1.0], [2.0], [3.0]], "w": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]},
            "outputs": [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]],
        },
    }
    seen = []

    def predict(row):
        seen.append((float(row["x"][0]), row["w"].tolist()))
        if row["x"][0] == 2.0:
            raise ValueError("a row this model refuses")
        return np.array(["a", "b", "c", "b"][int(row["x"][0])])

    # Row 3 is an even split, which Kaimei gives the first class and this model its second; row 2 could not be asked.
    assert graphs.classes_differ(graph, predict) == (1, 3)
    assert seen[1] == (1.0, [3.0, 4.0])  # each row's values, by feature
    assert graphs.classes_differ(graph, ["a", "b", "c", "a"]) == (0, 4)  # the answers themselves, in the check's order
    assert graphs.classes_differ(graph, [0, 1, 2, 0], classes=[0, 1, 2]) == (0, 4)  # the model's own classes
    assert graphs.classes_differ(graph, ["a", "b"]) == (0, 0)  # not an answer for each row: nothing is compared
    table = {
        "task": "binary_classification",
        "classes": ["no", "yes"],
        "feature_names": ["p", "q"],
        "check": {"rows": [[1.0, None], [2.0, 3.0]], "outputs": [[0.9, 0.1], [0.2, 0.8]], "groups": [2]},
    }
    rows = []
    assert graphs.classes_differ(table, lambda row: rows.append(row) or "yes") == (1, 2)
    assert rows[1]["q"].tolist() == [3.0] and np.isnan(rows[0]["q"][0])
    labels = {"task": "multilabel_classification", "check": {"inputs": {"x": [[0.0], [1.0]]}, "outputs": [[0.9, 0.2], [0.6, 0.5]]}}
    assert graphs.classes_differ(labels, [[1, 0], [1, 0]]) == (1, 2)  # one half is a yes
    # Nothing to compare: no check, no classifier, or a reading that settles an even split by its margins.
    assert graphs.classes_differ({"task": "regression", "check": graph["check"]}, predict) == (0, 0)
    assert graphs.classes_differ({"task": "multiclass_classification"}, predict) == (0, 0)
    assert graphs.classes_differ({**table, "combine": {"op": "ovr", "ties": "margin"}}, ["yes", "yes"]) == (0, 0)


def test_a_tuned_threshold_is_carried_so_the_class_is_the_models(sklearn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TunedThresholdClassifierCV

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TunedThresholdClassifierCV(LogisticRegression(), scoring="recall").fit(X, YES_NO)
    assert model.best_threshold_ < 0.1  # nowhere near the half a plain reading would cut at
    spec = extract_structure(model, feature_names=NAMES, check_rows=X[:40]).spec
    assert spec["family"] == "graph" and not spec.get("approximate") and spec_is_executable(spec)
    assert spec["outputs"]["threshold"] == pytest.approx(float(model.best_threshold_), abs=0)
    assert "the model's own decision threshold" in spec["summary"]
    rows = check_table(spec)
    assert len(rows) >= 40 and kaimei_classes(spec, model.classes_.tolist()) == model.predict(rows).tolist()
    assert graphs.classes_differ(spec, asked_of(model), model.classes_)[0] == 0
    # Read at one half, as it was before, Kaimei's class was another on most of these rows, and nothing said so.
    plain = {**spec, "outputs": {"labels": spec["outputs"]["labels"]}}
    assert graphs.classes_differ(plain, asked_of(model), model.classes_)[0] > len(rows) // 3


def test_a_fixed_threshold_classifier_is_read_with_its_threshold(sklearn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import FixedThresholdClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    named = np.where(YES_NO == 1, "yes", "no")
    cases = {
        "on a probability": (FixedThresholdClassifier(LogisticRegression(), threshold=0.3).fit(X, named), 0.3, None),
        "the first class positive": (
            FixedThresholdClassifier(LogisticRegression(), threshold=0.3, pos_label="no").fit(X, named), 0.3, "no",
        ),
        # A threshold on a decision score is carried squashed, as the score Kaimei reports is.
        "on a margin": (FixedThresholdClassifier(LinearSVC(), threshold=-0.4).fit(X, named), 1 / (1 + np.exp(0.4)), None),
        "on the margin of a model with probabilities": (
            FixedThresholdClassifier(LogisticRegression(), threshold=0.2, response_method="decision_function").fit(X, named),
            1 / (1 + np.exp(-0.2)),
            None,
        ),
        "in a pipeline": (
            make_pipeline(StandardScaler(), FixedThresholdClassifier(LogisticRegression(), threshold=0.7)).fit(X, named), 0.7, None,
        ),
    }
    for title, (model, threshold, positive) in cases.items():
        spec = graphs.via_onnx(model, feature_names=NAMES, check_rows=X[:60], reasons=(why := []))
        assert spec is not None and not spec.get("approximate"), (title, why, spec and spec["summary"])
        assert spec["outputs"]["threshold"] == pytest.approx(threshold, rel=1e-12), title
        assert spec["outputs"].get("positive_label") == positive, title
        assert kaimei_classes(spec, ["no", "yes"]) == model.predict(check_table(spec)).tolist(), title
    # At exactly one half the wrapper gives its positive class and Kaimei's even split is the first class: the
    # threshold is carried as the largest number below one half, and a tree's even leaves go the model's way.
    tree = FixedThresholdClassifier(DecisionTreeClassifier(max_depth=2, random_state=0)).fit(X, YES_NO)
    spec = graphs.via_onnx(tree, feature_names=NAMES)
    assert spec["outputs"]["threshold"] == float(np.nextafter(0.5, 0.0)) and not spec.get("approximate")
    assert graphs.reported_class(spec, [0.5, 0.5]) == 1 and tree.classes_.tolist() == [0, 1]
    # The model converted is the caller's own, left as it was.
    assert type(cases["in a pipeline"][0].steps[-1][1]).__name__ == "FixedThresholdClassifier"


def test_a_threshold_that_cannot_be_carried_is_not_exact_and_says_why(sklearn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import FixedThresholdClassifier
    from sklearn.svm import LinearSVC

    for model, why in (
        (FixedThresholdClassifier(LogisticRegression(), threshold=1.5).fit(X, YES_NO), "is 1.5, which is no probability"),
        (FixedThresholdClassifier(LinearSVC(), threshold=80.0).fit(X, YES_NO), "is 80, too far out"),
    ):
        spec = graphs.via_onnx(model, feature_names=NAMES)
        assert spec["approximate"] is True and "check" not in spec and "threshold" not in spec["outputs"]
        assert graphs.NOT_ITS_CLASS in spec["summary"] and why in spec["summary"]
        assert spec["summary"].rstrip().endswith("so it is drawn, not run.") or "drawn, not run" in spec["summary"]
        assert graphs.drawn_reason(spec) == spec["summary"]  # an upload's warning says why
        # Nothing else reads it, so the default route uploads the same drawing, and "error" refuses it with the reason.
        assert extract_structure(model, feature_names=NAMES).spec["approximate"] is True
        with pytest.raises(StructureError, match="no probability|too far out") as caught:
            extract_structure(model, feature_names=NAMES, on_unsupported="error")
        assert caught.value.code == "unsupported_layers"


def test_a_classifier_whose_probabilities_tie_is_read_by_the_scores_predict_goes_by(sklearn):
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier

    for model in (
        SGDClassifier(loss="modified_huber", random_state=0).fit(X * 2, THREE),
        OneVsRestClassifier(SGDClassifier(loss="modified_huber", random_state=0)).fit(X * 2, THREE),
    ):
        rows = X[:60] * 2
        ties = model.classes_[model.predict_proba(rows).argmax(axis=1)] != model.predict(rows)
        assert ties.sum() >= 5  # its clipped probabilities tie where its scores do not
        spec = extract_structure(model, feature_names=NAMES, check_rows=rows).spec
        assert spec["family"] == "graph" and not spec.get("approximate") and "check" in spec
        assert spec["metadata"]["library_method"] == "decision_function" and spec["head"]["link"] == "softmax"
        assert "so its decision_function is what was read" in spec["summary"]
        assert kaimei_classes(spec, model.classes_.tolist()) == model.predict(check_table(spec)).tolist()
    # Two classes have one margin and no ties to break: the probabilities stay what is read.
    two = SGDClassifier(loss="modified_huber", random_state=0).fit(X * 2, YES_NO)
    spec = graphs.via_onnx(two, feature_names=NAMES, check_rows=X[:60] * 2)
    assert spec["metadata"]["library_method"] == "predict_proba" and not spec.get("approximate")


def test_a_reading_whose_class_is_not_the_models_on_a_check_row_is_not_exact(sklearn):
    from sklearn.svm import SVC

    four = np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.8, 0.0, 0.8])
    model = SVC(probability=True, random_state=0).fit(X, four)
    probe = np.random.default_rng(3).normal(size=(4000, 4))
    apart = probe[model.classes_[model.predict_proba(probe).argmax(axis=1)] != model.predict(probe)][:2]
    wide = probe * 1.5
    tied = wide[model.classes_[model.decision_function(wide).argmax(axis=1)] != model.predict(wide)][:2]
    # Rows whose most probable class is not the one predict() gives, and rows on which its votes tie, so that the
    # highest of its scores is not either: no reading of this model gives its own class on all of them.
    assert len(apart) == 2 and len(tied) == 2
    both = np.vstack([apart, tied])
    spec = graphs.via_onnx(model, feature_names=NAMES, check_rows=both)
    assert spec["approximate"] is True and "check" not in spec
    said = re.search(re.escape(graphs.NOT_ITS_CLASS) + r" on (\d+) of the (\d+) rows it was checked on \(", spec["summary"])
    assert said is not None and 2 <= int(said.group(1)) <= int(said.group(2)) == 4 + graphs.TABLE_CHECK_ROWS
    assert "goes by the margin" in spec["summary"] and spec["summary"].endswith("so it is drawn, not run.")
    assert spec["metadata"]["predicted_class"].startswith("Not the model's own on ")
    # The default route uploads that drawing (it used to upload the model as exact), and says so when it is uploaded.
    drawn = extract_structure(model, feature_names=NAMES, check_rows=both).spec
    assert drawn["family"] == "graph" and drawn["approximate"] is True and not spec_is_executable(drawn)
    assert drawn["metadata"]["predicted_class"] != structure.PREDICTED_CLASS_NOTE
    # Described with an argument only a conversion reads, it is the same drawing, not "could not be converted".
    described = extract_structure(model, check_rows=both, inputs={"X": {"features": NAMES}}).spec
    assert described["approximate"] is True and graphs.NOT_ITS_CLASS in described["summary"]
    # A model checked on rows where its most probable class is its own runs by its probabilities, with the note
    # that the two can differ elsewhere. (Rows deep inside either class of a model that goes by one column: the
    # rows made up from their values are as deep.)
    sure = SVC(kernel="linear", probability=True, random_state=0).fit(X, (X[:, 0] > 0).astype(int))
    deep = X[np.abs(X[:, 0]) > 1.5][:12]
    with pytest.warns(UserWarning, match="most probable class"):
        agreed = extract_structure(sure, feature_names=NAMES, check_rows=deep).spec
    assert not agreed.get("approximate") and agreed["metadata"]["library_method"] == "predict_proba"
    assert agreed["metadata"]["predicted_class"] == structure.PREDICTED_CLASS_NOTE


def test_a_reading_with_no_reason_of_its_own_says_plainly_what_its_class_does_not_follow():
    shown = {"task": "multiclass_classification", "summary": "A reading.", "head": {"output": "y", "link": "none"}, "check": {}}
    graphs.not_its_class(shown, 2, 9)
    assert shown["summary"] == (
        "A reading. Kaimei's class for a row would differ from the model's own predict() on 2 of the 9 rows it was "
        "checked on (the class its predict() gives is not always its most probable one), so it is drawn, not run."
    )
    scored = {"task": "multiclass_classification", "head": {"output": "y", "link": "softmax"}, "check": {}}
    graphs.not_its_class(scored, 1, 9)
    assert "(the class its predict() gives is not always the one with its highest score)" in scored["summary"]
    told = {"task": "multiclass_classification", "head": {"output": "y", "link": "none"}}
    graphs.not_its_class(told, 1, 9, "its predict() counts votes")
    assert "(its predict() counts votes)" in told["summary"] and "most probable" not in told["summary"]


# --- 13. the scores an SVC with probabilities predicts by ---------------------------------------------------------


def test_the_scores_of_an_svc_fitted_with_probabilities_can_be_read(sklearn):
    pytest.importorskip("onnxruntime")
    import onnxruntime
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, NuSVC

    probe = np.random.default_rng(3).normal(size=(4000, 4))
    for title, model, target in (
        ("two classes", SVC(probability=True, random_state=0), YES_NO),
        ("three classes", NuSVC(probability=True, random_state=0), THREE),
        ("one against the rest", OneVsRestClassifier(SVC(probability=True, random_state=0)), THREE),
        ("in a pipeline", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)), YES_NO),
    ):
        model.fit(X, target)
        final = model.steps[-1][1] if hasattr(model, "steps") else model
        # Asked for its scores, the conversion gave the probabilities (negated, for two classes), so no second
        # reading of such a model was ever its decision_function.
        converted = graphs._convert_sklearn(model, final, 4, np.float32, True, np, raw_scores=True)
        session = onnxruntime.InferenceSession(converted, providers=["CPUExecutionProvider"])
        scores = session.run(None, {session.get_inputs()[0].name: probe[:8].astype(np.float32)})[-1]
        own = np.asarray(model.decision_function(probe[:8]), dtype=float)
        if own.ndim == 1:
            assert any(np.allclose(sign * scores[:, column], own, atol=1e-4) for sign in (1.0, -1.0) for column in (0, 1)), title
        else:
            np.testing.assert_allclose(scores, own, atol=1e-4, err_msg=title)
        # The caller's model keeps its probabilities.
        assert model.predict_proba(probe[:2]).shape[0] == 2, title
        # So where a row's most probable class is not the one predict() gives, the model is read by its scores,
        # and runs with its own class, where it could only be drawn.
        apart = probe[model.classes_[model.predict_proba(probe).argmax(axis=1)] != model.predict(probe)][:2]
        assert len(apart) == 2, title
        spec = graphs.via_onnx(model, feature_names=NAMES, check_rows=apart)
        assert not spec.get("approximate") and spec["metadata"]["library_method"] == "decision_function", (title, spec["summary"])
        assert "so its decision_function is what was read" in spec["summary"], title
        assert kaimei_classes(spec, model.classes_.tolist()) == model.predict(check_table(spec)).tolist(), title


class _Coin:
    """A model whose predict settles an even split the other way from Kaimei."""

    classes_ = np.array([0, 1])
    n_features_in_ = 2

    def __init__(self, answers):
        self.answers = answers

    def predict(self, rows):
        return np.array(self.answers[: len(rows)])


def test_kaimeis_own_readers_answer_to_the_same_rule():
    def native(outputs):
        return {
            "family": "network",
            "task": "binary_classification",
            "classes": ["0", "1"],
            "feature_names": ["p", "q"],
            "summary": "A reading.",
            "check": {"rows": [[0.0, 1.0], [2.0, 3.0]], "outputs": outputs, "groups": [2]},
        }

    # Probabilities an even split apart by rounding: the engine reports the first class, and this model the second.
    split = native([[0.9, 0.1], [0.4999999999999999, 0.5000000000000001]])
    structure._its_class_is_the_models(split, _Coin([0, 1]), ["p", "q"])
    assert split["approximate"] is True and "check" not in split
    assert f"{graphs.NOT_ITS_CLASS} on 1 of the 2 rows it was checked on" in split["summary"]
    same = native([[0.9, 0.1], [0.4999999999999999, 0.5000000000000001]])
    structure._its_class_is_the_models(same, _Coin([0, 0]), ["p", "q"])
    assert not same.get("approximate") and "check" in same and same["summary"] == "A reading."


def test_an_onnx_file_that_labels_a_row_itself_answers_to_its_own_label(sklearn):
    # A file whose own label for a row is its lowest score, beside the scores Kaimei reads the highest of.
    nodes = [
        helper.make_node("Softmax", ["x"], ["scores"], axis=1),
        helper.make_node("ArgMin", ["x"], ["label"], axis=1, keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]), helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 2])],
    )
    file = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    file.ir_version = 9
    spec = graphs.from_onnx(file, feature_names=["a", "b"], labels=["0", "1"])
    assert spec["approximate"] is True and "check" not in spec
    assert graphs.NOT_ITS_CLASS in spec["summary"] and "the file's own label" in spec["summary"]
    # Classes the caller named otherwise cannot be matched with the file's labels, and nothing is claimed of them.
    renamed = graphs.from_onnx(file, feature_names=["a", "b"], labels=["no", "yes"])
    assert not renamed.get("approximate") and "check" in renamed
    # A converted classifier labels a row by its votes, where Kaimei takes the most probable class.
    import skl2onnx
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agreeing = graphs.to_onnx(LogisticRegression().fit(X, THREE), feature_names=NAMES)
        model = SVC(probability=True, random_state=0).fit(X, THREE)
        # The file skl2onnx itself writes for it: its label by votes, beside its probabilities.
        voting = skl2onnx.to_onnx(
            model, X[:1].astype(np.float32), options={id(model): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3}
        ).SerializeToString()
    read = graphs.from_onnx(agreeing, feature_names=NAMES, check_rows=X[:40])
    assert not read.get("approximate") and len(read["check"]["outputs"]) >= 40  # its label is its most probable class
    probe = np.random.default_rng(3).normal(size=(4000, 4))
    apart = probe[model.classes_[model.predict_proba(probe).argmax(axis=1)] != model.predict(probe)][:2]
    # Whatever the classes are called here, the file's own classifier step says which score is which class.
    for named in (None, ["ant", "bee", "cat"]):
        drawn = graphs.from_onnx(voting, feature_names=NAMES, check_rows=apart, labels=named)
        assert drawn["approximate"] is True and "on 2 of the " in drawn["summary"], named
        assert "the file's own label" in drawn["summary"]
    assert not graphs.from_onnx(voting, feature_names=NAMES).get("approximate")  # rows on which the two agree


def binary_graph(**outputs):
    """A raw graph of a yes-or-no model, as a conversion leaves it, carrying what ``outputs`` says."""
    return {
        "format": "kaimei.structure.v1",
        "family": "graph",
        "framework": "sklearn",
        "task": "binary_classification",
        "feature_names": ["age", "income"],
        "features": [{"name": "age", "type": "numeric"}, {"name": "income", "type": "numeric"}],
        "outputs": {"labels": ["no", "yes"], **outputs},
        "graph": {
            "opsets": {"": 17},
            "inputs": [{"name": "x", "dtype": "float32", "shape": [1, 2], "from": {"features": ["age", "income"]}}],
            "constants": {"w": np.ones((2, 2), dtype=np.float32), "b": np.zeros(2, dtype=np.float32)},
            "nodes": [{"id": "fc", "op": "Gemm", "inputs": ["x", "w", "b"], "outputs": ["y"], "attrs": {}}],
            "outputs": ["y"],
            "values": {"y": {"shape": [1, 2], "dtype": "float32"}},
        },
        "head": {"output": "y", "link": "none"},
        "check": {"inputs": {"age": [[0.0]], "income": [[0.0]]}, "outputs": [[0.0, 0.0]], "tolerance": 1e-4},
        "metadata": {"route": "onnx"},
    }


def stand_in(monkeypatch, spec, onnx_bytes=None):
    """``extract_structure`` answering with a spec made here, so the upload is tested without a converter."""

    def extract(estimator, **options):
        made = finalize_spec(spec, weights=options.get("weights", "full"), externalize_above=None)
        made.onnx = onnx_bytes
        return made

    monkeypatch.setattr(models_module, "extract_structure", extract)


def made(request):
    return httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "m"})


RUNS = httpx.Response(200, json={"runnable": True, "summary": ""})


class RuleApi:
    """The API's one model, as far as these tests need it (made, read, updated, given a structure), with the rule by
    which a model's registered decision follows its structure's:

    - a structure carries a decision when its ``outputs`` hold a ``threshold`` from 0 to 1 and, for two classes,
      optionally a ``positive_label`` naming one of its own class names; the positive label is translated to the
      model's registered name for the class at the same place;
    - ``threshold`` and ``positive_label`` are one decision, followed together or not at all;
    - the registered decision follows the structure's when it is the default (0.5 on the second label) or what the
      structure being replaced carried; a structure that carries none puts the default back in that same case; any
      other registered decision is the customer's own and stays;
    - when the model is made, a ``threshold`` or a ``positive_label`` the caller sends is the caller's own, kept,
      both, even 0.5.
    """

    def __init__(self, router):
        self.created, self.patches, self.reads = [], [], 0
        self.model, self.structure = None, None
        router.post("/v1/models").mock(side_effect=self.make)
        router.get("/v1/models/mdl_1").mock(side_effect=self.read)
        router.patch("/v1/models/mdl_1").mock(side_effect=self.update)
        router.put("/v1/models/mdl_1/structure").mock(side_effect=self.restructure)
        router.get("/v1/models/mdl_1/structure").mock(return_value=httpx.Response(200, json={"runnable": True, "summary": ""}))
        router.put("/v1/models/mdl_1/reference-data").mock(return_value=httpx.Response(200, json={"rows": 1, "rejected": 0}))

    @staticmethod
    def carried(structure):
        """``(threshold, place of the positive class)``; ``(0.5, 1)`` for a structure that says none; ``None`` when
        what it says cannot be kept (then nothing is followed), as ``services/structure._carried_decision`` has it."""
        outputs = (structure or {}).get("outputs") or {}
        threshold, positive = outputs.get("threshold"), outputs.get("positive_label")
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
            return None
        names = [str(name) for name in outputs.get("labels") or []]
        if positive is not None and (len(names) != 2 or str(positive) not in names):
            return None
        return (0.5 if threshold is None else float(threshold)), (1 if positive is None else names.index(str(positive)))

    def follow(self, new, was):
        """The API's ``_follow_decision``, its tolerance included: thresholds within 1e-12 are one threshold."""

        def same(one, other):
            return abs(one[0] - other[0]) < 1e-12 and one[1] == other[1]

        now, earlier = self.carried(new), self.carried(was) if was is not None else None
        if now is None:
            return
        followed = same(self.decision, (0.5, 1)) or (earlier is not None and same(self.decision, earlier))
        if followed and not same(self.decision, now):
            self.decide(now)

    @property
    def decision(self):
        outputs = self.model["outputs"]
        return float(outputs["threshold"]), outputs["labels"].index(outputs["positive_label"])

    def decide(self, decision):
        self.model["outputs"].update(threshold=decision[0], positive_label=self.model["outputs"]["labels"][decision[1]])

    def make(self, request):
        body = json.loads(request.content)
        self.created.append(body)
        said, structure = dict(body.get("outputs") or {}), body.get("structure") or {}
        labels = [str(name) for name in said.get("labels") or (structure.get("outputs") or {}).get("labels") or ["negative", "positive"]]
        self.model = {"object": "model", "id": "mdl_1", "slug": "m", "outputs": {**said, "labels": labels}}
        own = said.get("threshold") is not None or said.get("positive_label") is not None
        self.model["outputs"].setdefault("threshold", 0.5)
        self.model["outputs"].setdefault("positive_label", labels[1])
        if not own:
            self.follow(structure, None)
        self.structure = structure
        return httpx.Response(201, json=self.model)

    def restructure(self, request):
        new = json.loads(request.content)
        self.follow(new, self.structure)
        self.structure = new
        return httpx.Response(200, json={"object": "model.structure"})

    def read(self, request):
        self.reads += 1
        return httpx.Response(200, json=self.model)

    def update(self, request):
        self.patches.append(json.loads(request.content))
        return httpx.Response(200, json=self.model)


def quietly_about_decisions(caught):
    """No warning about a decision or a threshold among those caught."""
    return not [item for item in caught if any(word in str(item.message) for word in ("threshold", "predicts", "decision"))]


@pytest.fixture(scope="module")
def decided(sklearn):
    """A plain yes-or-no model, and one that predicts its FIRST class at a probability of 0.3 or more."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import FixedThresholdClassifier

    plain = LogisticRegression().fit(X, YES_NO)
    wrapped = FixedThresholdClassifier(LogisticRegression(), threshold=0.3, pos_label=0).fit(X, YES_NO)
    return plain, wrapped


def test_the_models_own_decision_is_the_apis_to_register_under_the_models_class_names(client, api, decided):
    _plain, wrapped = decided
    fake = RuleApi(api)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.models.upload(wrapped, name="Churn", feature_names=NAMES, X=X[:40], outputs={"labels": ["neg", "pos"]})
    # What the caller said, and nothing of the structure's decision beside it. (The structure's threshold and its
    # positive_label '0' were merged in: '0' is no label of this model, so the threshold landed on the second class.)
    assert fake.created[-1]["outputs"] == {"labels": ["neg", "pos"]} and quietly_about_decisions(caught)
    carried = fake.created[-1]["structure"]["outputs"]
    assert carried == {"labels": ["0", "1"], "positive_label": "0", "threshold": 0.3}  # it travels in the structure
    assert fake.model["outputs"] == {"labels": ["neg", "pos"], "threshold": 0.3, "positive_label": "neg"}
    # By the decision the API ends with, in the model's own label names, Kaimei's class is the model's on every row.
    check = fake.structure["check"]
    rows = np.array([[check["inputs"][name][at][0] for name in NAMES] for at in range(len(check["outputs"]))])
    registered = {"task": "binary_classification", "outputs": fake.model["outputs"]}
    picked = [graphs.reported_class(registered, values) for values in check["outputs"]]
    assert len(rows) >= 40 and [int(wrapped.classes_[at]) for at in picked] == wrapped.predict(rows).tolist()
    assert 0 in picked and 1 in picked and not fake.patches
    # With nothing said of its outputs, the model takes its class names from the structure, and its decision too.
    client.models.upload(wrapped, name="Churn", feature_names=NAMES)
    assert fake.created[-1]["outputs"] == {"labels": ["0", "1"]}
    assert fake.model["outputs"] == {"labels": ["0", "1"], "threshold": 0.3, "positive_label": "0"}


def test_a_decision_of_the_callers_own_is_sent_as_it_is_and_said_not_to_be_the_models(client, api, decided):
    _plain, wrapped = decided
    fake = RuleApi(api)
    # A threshold, or a positive class, is a decision of the caller's own: kept whole, never mixed with the model's.
    for said, worded in (
        ({"threshold": 0.6}, r"outputs says this model predicts '1' at 0\.6 or more, and the model itself predicts '0' at 0\.3 or more"),
        ({"positive_label": "1"}, r"predicts '1' at 0\.5 or more, and the model itself predicts '0' at 0\.3 or more"),
        ({"labels": ["neg", "pos"], "threshold": 0.3}, r"predicts 'pos' at 0\.3 or more, and the model itself predicts 'neg' at 0\.3"),
    ):
        with pytest.warns(UserWarning, match=worded) as caught:
            client.models.upload(wrapped, name="Churn", feature_names=NAMES, X=X[:40], outputs=dict(said))
        assert fake.created[-1]["outputs"] == said
        told = next(str(item.message) for item in caught if "outputs says" in str(item.message))
        # How far that is from the model's own class is counted on the rows the structure was checked on.
        count = re.search(r"not always the class predict\(\) gives \(on (\d+) of the (\d+) rows it was checked on\)", told)
        assert count is not None and 0 < int(count.group(1)) <= int(count.group(2)) and int(count.group(2)) >= 40
        assert "Leave threshold and positive_label out of outputs" in told
        assert fake.decision != (0.3, 0)  # the API kept the caller's
    # The model's own decision, said by the caller in the model's own terms, is no other decision.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.models.upload(wrapped, name="Churn", feature_names=NAMES, outputs={"labels": ["neg", "pos"], "threshold": 0.3, "positive_label": "neg"})
    assert quietly_about_decisions(caught) and fake.decision == (0.3, 0)
    # And a threshold on a model that carries none is the caller's operating point, as it always was.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.models.upload(decided[0], name="Churn", feature_names=NAMES, outputs={"threshold": 0.3})
    assert quietly_about_decisions(caught) and fake.created[-1]["outputs"] == {"threshold": 0.3}


def test_a_structure_given_to_a_model_leaves_its_registered_decision_to_the_api(client, api, decided):
    plain, wrapped = decided
    fake = RuleApi(api)
    client.models.upload(plain, name="Churn", feature_names=NAMES, outputs={"labels": ["neg", "pos"]})
    assert fake.decision == (0.5, 1)
    # The model starts with the decision every model starts with, so it follows the new structure's: the API's
    # doing. The client reads the model back, finds the structure's decision under the model's own class names,
    # and writes nothing. (It wrote a threshold itself, beside whatever positive label was registered.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.models.upload_structure("mdl_1", wrapped, feature_names=NAMES)
    assert fake.model["outputs"] == {"labels": ["neg", "pos"], "threshold": 0.3, "positive_label": "neg"}
    assert fake.reads == 1 and not fake.patches and quietly_about_decisions(caught)
    # A decision registered as the customer's own stays, and the caller is told Kaimei's class is by it.
    for own in ({"threshold": 0.8, "positive_label": "pos"}, {"threshold": 0.5, "positive_label": "neg"}):
        fake.model["outputs"].update(own)
        with pytest.warns(UserWarning, match=r"Model mdl_1 is registered to predict .* at 0\.[58] or more") as caught:
            client.models.upload_structure("mdl_1", wrapped, feature_names=NAMES)
        told = next(str(item.message) for item in caught if "is registered to predict" in str(item.message))
        assert "the model this structure was read from predicts 'neg' at 0.3 or more" in told
        # What registers the model's own names its positive class as the model does, never beside a stale one.
        assert "client.models.update('mdl_1', outputs={'threshold': 0.3, 'positive_label': 'neg'})" in told
        assert {key: fake.model["outputs"][key] for key in own} == own and not fake.patches
    assert fake.reads == 3


def test_a_structure_with_no_decision_of_its_own_puts_the_default_back_by_the_apis_rule(client, api, decided):
    plain, wrapped = decided
    fake = RuleApi(api)
    client.models.upload(wrapped, name="Churn", feature_names=NAMES)
    assert fake.decision == (0.3, 0)  # registered from the structure, by the API
    # Replaced by a model that predicts at one half: the decision the old structure brought goes with it. The
    # client asks nothing of the model and writes nothing; a threshold it had written itself would have stayed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.models.upload_structure("mdl_1", plain, feature_names=NAMES)
    assert fake.decision == (0.5, 1) and fake.reads == 0 and not fake.patches and quietly_about_decisions(caught)
    assert "threshold" not in fake.structure["outputs"]
    # The same when the decision came with a structure given to a model that was there already: it is the API that
    # registered it, so it is the API that knows it was the structure's, and takes it away with the structure. (The
    # client wrote the threshold itself then, and nothing ever took it away.)
    client.models.upload(plain, name="Churn", feature_names=NAMES, outputs={"labels": ["neg", "pos"]})
    client.models.upload_structure("mdl_1", wrapped, feature_names=NAMES)
    assert fake.model["outputs"] == {"labels": ["neg", "pos"], "threshold": 0.3, "positive_label": "neg"}
    assert fake.reads == 1 and not fake.patches
    client.models.upload_structure("mdl_1", plain, feature_names=NAMES)
    assert fake.model["outputs"] == {"labels": ["neg", "pos"], "threshold": 0.5, "positive_label": "pos"}
    assert fake.reads == 1 and not fake.patches
    # A decision of the customer's own is not one a structure brought, so it stays through either kind of structure.
    fake.model["outputs"].update(threshold=0.7)
    client.models.upload_structure("mdl_1", plain, feature_names=NAMES)
    assert fake.decision == (0.7, 1) and not fake.patches


def test_a_model_that_gives_an_even_split_to_its_positive_class_is_said_when_one_half_is_registered(client, api, sklearn):
    from sklearn.model_selection import FixedThresholdClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    # "At one half or more" is carried as the largest number below one half: at one half itself Kaimei gives an even
    # split to the first class. The API's rule takes thresholds within 1e-12 for one threshold, so it registers one
    # half, and four neighbours split two against two often. Nothing said so, and the model was uploaded as exact.
    fake = RuleApi(api)
    evenly = FixedThresholdClassifier(KNeighborsClassifier(4)).fit(X, YES_NO)
    with pytest.warns(UserWarning, match=r"is registered to predict '1' at 0\.5 or more") as caught:
        client.models.upload(evenly, name="Churn", feature_names=NAMES, X=X[:60])
    assert fake.created[-1]["structure"]["outputs"]["threshold"] == float(np.nextafter(0.5, 0.0)) and fake.decision == (0.5, 1)
    told = next(str(item.message) for item in caught if "is registered to predict" in str(item.message))
    assert "the model this structure was read from predicts '1' at 0.49999999999999994 or more" in told
    count = re.search(r"\(on (\d+) of the (\d+) rows it was checked on\)", told)
    assert count is not None and 0 < int(count.group(1)) < int(count.group(2))
    assert "outputs={'threshold': 0.49999999999999994, 'positive_label': '1'}" in told
    # A model none of whose checked rows is split evenly predicts the same by either, and nothing is said of it.
    never = FixedThresholdClassifier(DecisionTreeClassifier(max_depth=1)).fit(X, YES_NO)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.models.upload(never, name="Churn", feature_names=NAMES, X=X[:60])
    assert quietly_about_decisions(caught) and fake.decision == (0.5, 1)


# --- 2. trees with infinite thresholds inside wrappers --------------------------------------------------------------

WRAPPED_TREES = textwrap.dedent(
    """
    import sys
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, VotingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.tree import DecisionTreeClassifier
    from kaimei import graphs
    from kaimei.structure import StructureError, extract_structure

    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 3))
    X[rng.random(240) < 0.2, 0] = np.nan
    y = np.isnan(X[:, 0]).astype(int)
    tree = lambda: DecisionTreeClassifier(max_depth=3, random_state=0)
    models = {
        "calibrated tree": CalibratedClassifierCV(tree(), cv=3),
        "calibrated forest": CalibratedClassifierCV(RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0), cv=3),
        "target transform": TransformedTargetRegressor(regressor=RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0)),
        "stack read by a tree": StackingClassifier([("a", tree())], final_estimator=tree(), passthrough=True, cv=3),
        "vote": VotingClassifier([("a", tree()), ("b", tree())], voting="soft"),
        "pipeline of a calibration": make_pipeline(FunctionTransformer(), CalibratedClassifierCV(tree(), cv=3)),
    }
    only = sys.argv[1:]
    for title, model in models.items():
        if only and title not in only:
            continue
        model.fit(X, y.astype(float) if title == "target transform" else y)
        print(title, "| guard:", graphs._trees_with_infinite_thresholds(model, np), flush=True)
        if "--guard-only" in only:
            continue
        for call in (
            lambda: extract_structure(model, via="onnx", feature_names=["a", "b", "c"]),
            lambda: graphs.to_onnx(model, feature_names=["a", "b", "c"]),
        ):
            try:
                call()
                print(title, "| read", flush=True)
            except StructureError as exc:
                print(title, "| refused:", " ".join(exc.message.split()), flush=True)
    """
)


def run_script(script, *args, timeout=240):
    import kaimei

    source = os.path.dirname(os.path.dirname(os.path.abspath(kaimei.__file__)))  # the kaimei these tests import
    env = {**os.environ, "PYTHONPATH": source, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, "-W", "ignore", "-c", script, *args], capture_output=True, text=True, timeout=timeout, env=env)


def test_a_tree_with_an_infinite_threshold_inside_a_wrapper_is_refused_not_hung(sklearn):
    # Each of these conversions never returned: the tree sat where the guard did not look.
    done = run_script(WRAPPED_TREES)  # a hang ends in TimeoutExpired, not in a hung test run
    assert done.returncode == 0, done.stderr[-2000:]
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    titles = ["calibrated tree", "calibrated forest", "target transform", "stack read by a tree", "vote", "pipeline of a calibration"]
    for title in titles:
        mine = [line for line in lines if line.startswith(title + " |")]
        assert mine[0].endswith("guard: True"), mine
        assert len(mine) == 3 and all("refused:" in line and "infinite threshold" in line for line in mine[1:]), mine


def test_models_without_such_a_tree_are_still_converted(sklearn):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    forest = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0)
    for model in (
        CalibratedClassifierCV(forest, cv=3).fit(X, YES_NO),
        make_pipeline(ColumnTransformer([("s", StandardScaler(), [0, 1, 2, 3])]), forest).fit(X, YES_NO),
    ):
        assert graphs._trees_with_infinite_thresholds(model, np) is False
    assert graphs._trees_with_infinite_thresholds({"a": [object(), "name", 3, None, slice(0, 2), np.arange(4)]}, np) is False
    assert len(graphs.to_onnx(CalibratedClassifierCV(forest, cv=3).fit(X, YES_NO), feature_names=NAMES)) > 100


# --- 3. labels= as an array ---------------------------------------------------------------------------------------


def test_class_names_given_as_an_array_or_an_index_name_the_classes(sklearn, client, api, monkeypatch):
    pandas = pytest.importorskip("pandas")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    logistic = LogisticRegression().fit(X, THREE)
    for given in (np.array(["ant", "bee", "cat"]), pandas.Index(["ant", "bee", "cat"]), ("ant", "bee", "cat")):
        # "The truth value of an array is ambiguous", on every route.
        for via in ("native", "onnx", "auto"):
            spec = extract_structure(logistic, feature_names=NAMES, labels=given, via=via).spec
            assert spec["outputs"]["labels"] == ["ant", "bee", "cat"], (type(given).__name__, via)
        assert graphs.via_onnx(SVC().fit(X, YES_NO), feature_names=NAMES, labels=given[:2])["outputs"]["labels"] == ["ant", "bee"]
    node = helper.make_node("Softmax", ["x"], ["y"], axis=1)
    graph = helper.make_graph(
        [node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])],
    )
    file = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    file.ir_version = 9
    read = graphs.from_onnx(file, feature_names=["a", "b", "c"], labels=np.array(["ant", "bee", "cat"]))
    assert read["outputs"]["labels"] == ["ant", "bee", "cat"]
    assert extract_structure(logistic, feature_names=NAMES, labels=np.array([])).spec["outputs"]["labels"] == ["0", "1", "2"]
    for wrong in ("yes", {"a": 1}, 3):
        with pytest.raises(StructureError) as caught:
            graphs.via_onnx(logistic, feature_names=NAMES, labels=wrong)
        assert caught.value.param == "labels"
    # The upload itself asked an array whether it was empty.
    stand_in(monkeypatch, binary_graph())
    create = api.post("/v1/models").mock(side_effect=made)
    api.get("/v1/models/mdl_1/structure").mock(return_value=RUNS)
    client.models.upload(object(), name="Named", labels=np.array(["stays", "leaves"]))
    assert json.loads(create.calls.last.request.content)["outputs"]["labels"] == ["stays", "leaves"]


# --- 4. a segmenter with a second output ----------------------------------------------------------------------------


def outputs_model(*outputs):
    """A model with these outputs, each ``(name, type, shape)``, all made from one input."""
    nodes, infos = [], []
    for name, kind, shape in outputs:
        nodes.append(helper.make_node("Identity", ["x"], [name]))
        infos.append(helper.make_tensor_value_info(name, kind, shape))
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])], infos)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_a_segmenters_mask_is_its_first_output_that_can_hold_one():
    pick = lambda *outputs: graphs._prediction_output(outputs_model(*outputs), None, "segmentation")[0]  # noqa: E731
    floats, whole = TensorProto.FLOAT, TensorProto.INT64
    # Several outputs that could hold the mask (torchvision's FCN and DeepLab give "out" and "aux"): the first, for
    # a conversion that has to know which before the importer runs (it compares the library's own answer).
    assert pick(("out", floats, [None, 3, 4, 4]), ("aux", floats, [None, 3, 4, 4])) == "out"
    assert pick(("classes", whole, [None, 4, 4]), ("scores", floats, [None, 3, 4, 4])) == "scores"  # floats first
    # Reading a file, the choice is left to the importer, which runs the model and prefers the output of the
    # picture's size, then the first (see the next test); what it chose is read back.
    left = lambda *outputs: graphs._prediction_output(outputs_model(*outputs), None, "segmentation", defer=True)  # noqa: E731
    assert left(("out", floats, [None, 3, 4, 4]), ("aux", floats, [None, 3, 4, 4])) == (None, None)
    assert left(("score", floats, [None, 5]), ("mask", floats, [1, 3, 4, 4]))[0] == "mask"  # one that can: no choice
    assert pick(("score", floats, [None, 5]), ("mask", floats, [1, 3, 4, 4])) == "mask"  # a row of scores holds no mask
    assert pick(("score", floats, [None, 5]), ("classes", whole, [None, 4, 4])) == "classes"  # whole numbers when no floats can
    assert pick(("only", floats, [None, 5])) == "only"  # one output is read whatever its shape: its reading says what is wrong
    with pytest.raises(StructureError, match="None of this model's outputs can hold a mask: 'a', 'b'") as caught:
        pick(("a", floats, [None, 5]), ("b", whole, [None, 7]))
    assert caught.value.param == "output"
    # What the caller says still decides, and a file that leaves a shape out is left to the importer, which runs it.
    assert graphs._prediction_output(outputs_model(("a", floats, [None, 5]), ("b", floats, [None, 3, 4, 4])), "a", "segmentation")[0] == "a"
    unsaid = outputs_model(("out", floats, [None, 3, 4, 4]), ("aux", floats, [None, 3, 4, 4]))
    unsaid.graph.output[1].type.tensor_type.ClearField("shape")
    assert graphs._prediction_output(unsaid, None, "segmentation") == (None, None)


def test_an_onnx_segmenter_with_two_outputs_is_read_from_the_first(torch):
    nn = torch.nn

    class TwoHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.out = nn.Conv2d(3, 4, 3, padding=1)
            self.aux = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            return self.out(x), self.aux(x)

    torch.manual_seed(0)
    import io

    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            TwoHeads().eval(), (torch.zeros(1, 3, 8, 8),), buffer, input_names=["pixels"], output_names=["out", "aux"],
            opset_version=17, dynamo=False,
        )
    spec = graphs.from_onnx(buffer.getvalue(), task="segmentation", inputs={"pixels": {"feature": "photo", "layout": "nchw"}})
    assert spec["task"] == "segmentation" and not spec.get("approximate"), spec.get("summary")
    assert spec["metadata"]["checked_against"]  # it was compared with the file's own "out"


# --- 5. save_onnx to a path that cannot be written ------------------------------------------------------------------


def test_a_file_that_cannot_be_written_is_refused_before_anything_is_made(client, api, monkeypatch, tmp_path):
    stand_in(monkeypatch, binary_graph(), onnx_bytes=b"onnx bytes")
    create = api.post("/v1/models").mock(side_effect=made)
    stored = api.put("/v1/models/m1/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    (tmp_path / "folder").mkdir()
    for path, why in ((tmp_path / "missing" / "m.onnx", "a folder that does not exist"), (tmp_path / "folder", "is a folder, not a file")):
        # The model used to be made first, and left with no reference rows when the write failed.
        with pytest.raises(StructureError, match=why) as caught:
            client.models.upload(object(), name="m", X=[[1.0, 2.0]], save_onnx=path)
        assert caught.value.param == "save_onnx" and "Nothing was uploaded" in caught.value.message
        with pytest.raises(StructureError, match=why):
            client.models.upload_structure("m1", object(), save_onnx=str(path))
    assert not create.called and not stored.called and len(api.calls) == 0


def test_a_write_that_fails_after_the_upload_is_a_warning_that_names_the_model(client, api, monkeypatch, tmp_path):
    stand_in(monkeypatch, binary_graph(), onnx_bytes=b"onnx bytes")
    api.post("/v1/models").mock(side_effect=made)
    api.get("/v1/models/mdl_1/structure").mock(return_value=RUNS)
    rows = api.put("/v1/models/mdl_1/reference-data").mock(return_value=httpx.Response(200, json={"rows": 1, "rejected": 0}))
    stored = api.put("/v1/models/m1/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    gone = tmp_path / "taken away" / "m.onnx"
    monkeypatch.setattr(models_module, "_check_onnx_path", lambda path: None)  # the folder went after it was looked at
    with pytest.warns(UserWarning, match=r"save_onnx could not write .*m\.onnx.*model mdl_1 is uploaded all the same"):
        model = client.models.upload(object(), name="m", X=[[1.0, 2.0]], save_onnx=gone)
    assert model["id"] == "mdl_1" and rows.called  # the model's id comes back, and its reference rows went up
    with pytest.warns(UserWarning, match=r"model m1 is uploaded all the same"):
        assert client.models.upload_structure("m1", object(), save_onnx=gone)["object"] == "model.structure"
    assert stored.called


# --- 6. the caller's random generators ---------------------------------------------------------------------------


def test_reading_a_module_that_draws_at_random_leaves_the_callers_generators_as_they_were(torch):
    nn = torch.nn

    class Noisy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x) + torch.randn_like(x[:, :2])

    class Refuses(Noisy):
        def forward(self, x):
            torch.rand(3)
            raise RuntimeError("no")

    sample = torch.zeros(2, 4)
    noisy, refuses = Noisy().eval(), Refuses().eval()  # made first: making a layer draws its weights
    for via in ("auto", "native", "onnx"):
        torch.manual_seed(123)
        state = torch.random.get_rng_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                extract_structure(noisy, sample=sample, feature_names=NAMES, task="regression", via=via)
            except StructureError:
                pass  # whether it is read or refused, nothing the caller will draw has been used up
        assert torch.equal(torch.random.get_rng_state(), state), via
    torch.manual_seed(7)
    state = torch.random.get_rng_state()
    with pytest.raises(StructureError):
        extract_structure(refuses, sample=sample, feature_names=NAMES)
    assert torch.equal(torch.random.get_rng_state(), state)


# --- 7. inputs= for forward(self, x, *rest) -----------------------------------------------------------------------


def test_the_inputs_of_a_forward_that_takes_more_values_are_hinted_by_the_names_they_are_exported_under(torch):
    nn = torch.nn

    class Rest(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x, *rest):
            return self.fc(x) + rest[0].sum(dim=1, keepdim=True)

    module = Rest().eval()
    sample = (torch.zeros(1, 4), torch.zeros(1, 3))
    options = dict(feature_names=None, task="regression", input_shape=None, preprocess=None, sample=sample, output=None, tokenizer=None, labels=None, max_length=None)
    # 'input_1' is what the export calls the second value: it was refused as "not an input of this model's forward (x)".
    structure._check_arguments(module, inputs={"input_1": {"feature": "extra"}}, **options)
    for unknown in ("x", "rest", "nope"):
        with pytest.raises(StructureError, match=r"not an input of this model's forward \(input_0, input_1\)") as caught:
            structure._check_arguments(module, inputs={unknown: {"feature": "extra"}}, **options)
        assert caught.value.param == "inputs"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(module, sample=sample, inputs={"input_1": {"feature": "extra"}}, task="regression").spec
    assert "extra" in spec["feature_names"]
    # A forward that names its inputs is still hinted by those names, and by no others.
    plain = nn.Linear(4, 2).eval()
    structure._check_arguments(plain, inputs={"input": {"features": NAMES}}, **{**options, "sample": torch.zeros(1, 4)})
    with pytest.raises(StructureError, match=r"forward \(input\)"):
        structure._check_arguments(plain, inputs={"input_0": {"features": NAMES}}, **{**options, "sample": torch.zeros(1, 4)})


# --- 8. a sample for a pipeline that picks its columns by name ----------------------------------------------------


def test_a_sample_for_a_pipeline_that_names_its_columns_holds_those_columns(sklearn):
    pandas = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frame = pandas.DataFrame(RNG.normal(size=(60, 6)), columns=["a", "b", "c", "d", "e", "f"])
    target = (frame["a"] > 0).astype(int)
    named = Pipeline([("columns", ColumnTransformer([("scaled", StandardScaler(), ["a", "c"]), ("kept", "passthrough", ["e"])])), ("model", LogisticRegression())]).fit(frame, target)
    assert named.n_features_in_ == 6
    # A table of the columns it picks out is its input: this was "sample has 3 columns, and this model reads 6 features".
    structure._check_table_sample(named, frame[["e", "a", "c"]].head(3))
    structure._check_table_sample(GridSearchCV(named, {"model__C": [1.0]}, cv=2).fit(frame, target), frame[["a", "c", "e"]].head(3))
    assert structure._columns_chosen_by_name(named) == ["a", "c", "e"]
    with pytest.raises(StructureError, match="sample has no column 'c', which this model reads by name") as caught:
        structure._check_table_sample(named, frame[["a", "e"]].head(3))
    assert caught.value.param == "sample"
    # Columns chosen by their place are still counted.
    placed = Pipeline([("columns", ColumnTransformer([("scaled", StandardScaler(), [0, 2])])), ("model", LogisticRegression())]).fit(frame.to_numpy(), target)
    assert structure._columns_chosen_by_name(placed) is None
    with pytest.raises(StructureError, match="sample has 3 columns, and this model reads 6 features"):
        structure._check_table_sample(placed, frame.to_numpy()[:3, :3])


# --- 9. what an older API is said to need -------------------------------------------------------------------------


def refused(status, code, message, param=None):
    return httpx.Response(status, json={"error": {"type": "invalid_request_error", "code": code, "message": message, "param": param}})


NO_GRAPHS = refused(400, "invalid_structure", "A network structure needs a layers array.", "layers")


def older(api, version="1.2.0"):
    return api.get("/health").mock(return_value=httpx.Response(200, json={"status": "ok", "version": version}))


def test_an_older_api_is_not_said_to_take_what_this_client_would_refuse(client, api, monkeypatch):
    routes = []

    def extract(estimator, **options):
        routes.append(options.get("via"))
        return finalize_spec(binary_graph(), weights="full", externalize_above=None)

    monkeypatch.setattr(models_module, "extract_structure", extract)
    create = api.post("/v1/models").mock(return_value=NO_GRAPHS)
    older(api)
    # inputs= is read only from a graph: Kaimei's own readers were asked all the same, refused it, and the error told
    # the caller to pass via='auto', which is what they had passed.
    with pytest.raises(APIStatusError) as caught:
        client.models.upload(object(), name="m", inputs={"x": {"features": ["age", "income"]}})
    said = caught.value.message
    assert routes == ["auto"] and create.call_count == 1
    assert "Kaimei API 1.2.0 cannot take this model" in said and "needs API 1.3.0 or later. Upgrade the API." in said
    assert "inputs is read only from a model converted to a graph" in said and "leave inputs out" in said
    assert "pass via='auto'" not in said
    with pytest.raises(APIStatusError) as caught:
        client.models.upload(object(), name="m", tokenizer={"type": "words"}, mask_layout="chw", via="onnx")
    assert "tokenizer and mask_layout are read only from a model converted to a graph" in caught.value.message
    # An ONNX model is a graph whatever via says: via='native' would be refused the same.
    for via in ("auto", "onnx"):
        with pytest.raises(APIStatusError) as caught:
            client.models.upload("model.onnx", name="m", via=via)
        said = caught.value.message
        assert "an ONNX model is always read as a graph, whatever via says" in said and "via='native'" not in said
        assert "needs API 1.3.0 or later" in said
    # A live model read as a graph on purpose can still be read the older way.
    with pytest.raises(APIStatusError, match=r"upload with via='native' to read it as clients before 0\.12\.0 did"):
        client.models.upload(object(), name="m", via="onnx")


def test_a_structure_an_older_api_refuses_says_what_it_does_not_know(client, api, monkeypatch):
    stand_in(monkeypatch, {**binary_graph(), "task": "clustering"})
    older(api, "1.1.0")
    put = api.put("/v1/models/m1/structure")
    put.mock(return_value=refused(422, "validation_error", "Input should be 'binary_classification', 'multiclass_classification' or 'regression'", "task"))
    # This said the model "was read as a graph" and to upload with via='native', which the same API refuses the same way.
    with pytest.raises(APIStatusError, match=r"Kaimei API 1\.1\.0 does not know the task 'clustering'.*need API 1\.3\.0 or later") as caught:
        client.models.upload_structure("m1", object())
    assert "via='native'" not in caught.value.message
    put.mock(return_value=NO_GRAPHS)
    with pytest.raises(APIStatusError, match=r"cannot take this model.*or upload with via='native'"):
        client.models.upload_structure("m1", object())
    with pytest.raises(APIStatusError) as caught:  # a structure handed over already read: no route reads it again
        client.models.upload_structure("m1", finalize_spec(binary_graph(), weights="full", externalize_above=None))
    assert "this structure is a graph already" in caught.value.message and "via='native'" not in caught.value.message


# --- 10. the evaluator that takes over from ONNX Runtime ----------------------------------------------------------


def onnxruntime_error(name, message):
    return type(name, (Exception,), {"__module__": "onnxruntime.capi.onnxruntime_pybind11_state"})(message)


def test_only_a_missing_kernel_hands_a_run_to_the_evaluator():
    lacks = graphs._onnxruntime_lacks_a_kernel
    assert lacks(onnxruntime_error("RuntimeException", "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : LSTM operator does not support double yet"))
    assert lacks(onnxruntime_error("NotImplemented", "[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Foo(1)"))
    assert lacks(onnxruntime_error("Fail", "[ONNXRuntimeError] : 1 : FAIL : Failed to find kernel for Foo(1)"))
    # A refusal of the input or the model that happens to call something unsupported is not a missing kernel.
    for said in (
        "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unsupported input type tensor(string) for input x",
        "[ONNXRuntimeError] : 1 : FAIL : Negative axis is not supported for this shape",
        "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: x",
    ):
        assert not lacks(onnxruntime_error("InvalidArgument", said)), said
    assert not lacks(type("NotImplemented", (Exception,), {"__module__": "somewhere.else"})("NOT_IMPLEMENTED"))


def test_an_evaluator_that_cannot_be_built_leaves_the_runner_and_the_reason(monkeypatch):
    pytest.importorskip("onnxruntime")
    import onnx.reference

    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    runner = graphs._OnnxRunner(graphs._importer(), model, "y")
    assert runner.engine.startswith("onnxruntime")
    lacking = onnxruntime_error("NotImplemented", "[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Relu(14)")

    class Session:
        runs = 0

        def run(self, names, feeds):
            Session.runs += 1
            raise lacking

    builds = []

    def cannot_build(model):
        builds.append(1)
        raise RuntimeError("the evaluator has no Relu either")

    session = runner.session = Session()
    monkeypatch.setattr(onnx.reference, "ReferenceEvaluator", cannot_build)
    feeds = {"x": np.zeros((1, 3), dtype=np.float32)}
    for _ in range(2):
        # This left the runner with no engine: the next row was "'NoneType' object has no attribute 'run'".
        with pytest.raises(Exception) as caught:
            runner(feeds)
        assert caught.value is lacking and runner.session is session and runner.engine.startswith("onnxruntime")
    assert builds == [1] and Session.runs == 2  # the evaluator is not built again for every row
    monkeypatch.undo()
    taken_over = graphs._OnnxRunner(graphs._importer(), model, "y")
    taken_over.session = Session()
    assert taken_over(feeds).tolist() == [[0.0, 0.0, 0.0]] and taken_over.session is None  # one that builds does take over
    assert taken_over.engine == "onnx reference evaluator"


# --- 11. reference rows of a pipeline fitted on an array ----------------------------------------------------------


def test_reference_rows_are_read_from_the_columns_the_features_are_read_from(client, api, monkeypatch):
    pandas = pytest.importorskip("pandas")
    spec = binary_graph()
    spec["feature_names"] = ["x3", "x0"]
    spec["features"] = [{"name": "x3", "type": "numeric"}, {"name": "x0", "type": "numeric"}]
    spec["graph"]["inputs"][0]["from"] = {"features": ["x3", "x0"]}
    spec["check"]["inputs"] = {"x3": [[0.0]], "x0": [[0.0]]}
    spec["metadata"] = {"route": "onnx", "table_columns": [3, 0]}  # each feature's column in the table it was fitted on
    stand_in(monkeypatch, spec)
    api.post("/v1/models").mock(side_effect=made)
    api.get("/v1/models/mdl_1/structure").mock(return_value=RUNS)
    rows = api.put("/v1/models/mdl_1/reference-data").mock(return_value=httpx.Response(200, json={"rows": 2, "rejected": 0}))
    table = np.array([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])
    sent = lambda: json.loads(rows.calls.last.request.content)["rows"]  # noqa: E731
    # X[:, 0] was stored under x3 and X[:, 1] under x0 (or the upload failed on the number of columns).
    client.models.upload(object(), name="m", X=table)
    assert sent() == [{"x3": 13.0, "x0": 10.0}, {"x3": 23.0, "x0": 20.0}]
    client.models.upload(object(), name="m", X=pandas.DataFrame(table, columns=["w", "x", "y", "z"]))  # other names: by place
    assert sent() == [{"x3": 13.0, "x0": 10.0}, {"x3": 23.0, "x0": 20.0}]
    client.models.upload(object(), name="m", X=pandas.DataFrame({"x0": [1.0], "x3": [2.0], "more": [3.0]}))  # its names: by name
    assert sent() == [{"x3": 2.0, "x0": 1.0}]
    client.models.upload(object(), name="m", X=[{"x3": 5.0, "x0": 6.0}])
    assert sent() == [{"x3": 5.0, "x0": 6.0}]
    calls = len(api.calls)
    with pytest.raises(StructureError, match=r"X has 3 columns, and this model reads its feature 'x3' from column 3") as caught:
        client.models.upload(object(), name="m", X=table[:, :3])
    assert caught.value.param == "X" and len(api.calls) == calls  # refused before a model is made
    # A structure that does not say reads an array as it always was read: a column for each feature, in order.
    spec["metadata"] = {"route": "onnx"}
    client.models.upload(object(), name="m", X=table[:, :2])
    assert sent() == [{"x3": 10.0, "x0": 11.0}, {"x3": 20.0, "x0": 21.0}]


# --- 12. the precision a scikit-learn model is computed in --------------------------------------------------------


def times(count, seed):
    """A table with a time in seconds and an amount far from zero, whose values differ by little beside their size."""
    rng = np.random.default_rng(seed)
    table = np.column_stack(
        [1_700_000_000 + rng.uniform(0, 86_400, count), rng.normal(40, 12, count), 1e11 + rng.normal(0, 5e4, count).round(0), rng.normal(size=count)]
    )
    return table, (table[:, 0] - 1_700_000_000) / 8640.0 + (table[:, 2] - 1e11) / 5e4


def as_converted(data, rows):
    """What the ONNX model a structure was read from gives rows, given to it in the number type it reads."""
    import onnxruntime

    session = onnxruntime.InferenceSession(data, providers=["CPUExecutionProvider"])
    reads = session.get_inputs()[0]
    kind = np.float64 if "double" in reads.type else np.float32
    return np.asarray(session.run(None, {reads.name: rows.astype(kind)})[-1], dtype=float).reshape(len(rows), -1)


def test_a_model_of_values_far_from_zero_is_computed_in_the_precision_that_gives_its_numbers(sklearn):
    pytest.importorskip("onnxruntime")
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    table, answers = times(300, 0)
    fresh, _ = times(200, 1)
    for title, model in (
        ("a scaler", make_pipeline(StandardScaler(), LinearRegression())),
        ("another scaler", make_pipeline(MinMaxScaler(), LinearRegression())),
        ("no scaler", LinearRegression()),
    ):
        model.fit(table, answers)
        own = model.predict(fresh)
        assert -5 < own.min() and own.max() < 15
        for rows in (table, None):  # with the caller's rows, and with what the model itself says of its data
            kept: dict = {}
            spec = graphs.via_onnx(model, feature_names=NAMES, check_rows=rows, keep=kept)
            assert not spec.get("approximate"), (title, spec["summary"])
            # Called exact in single precision, it was 0.09 off on answers like these: rows made up around zero gave
            # answers so large that the error on real rows looked small beside them.
            assert np.abs(as_converted(kept["onnx"], fresh)[:, 0] - own).max() < 1e-6, (title, rows is None)
            assert spec["metadata"]["model_precision"] == "double" and spec["graph"]["inputs"][0]["dtype"] == "float64"
            assert "computed in double precision" in spec["summary"]
            held = check_table(spec)
            if rows is not None:
                # The caller's rows as the doubles they are, then rows made up from where their values lie.
                np.testing.assert_array_equal(held[: graphs.OWN_CHECK_ROWS], table[: graphs.OWN_CHECK_ROWS])
                assert len(held) == graphs.OWN_CHECK_ROWS + graphs.TABLE_CHECK_ROWS
            if rows is not None or title != "no scaler":
                assert 1_699_000_000 < held[:, 0].min() and held[:, 0].max() < 1_701_000_000, (title, rows is None)
                assert np.abs(np.asarray(spec["check"]["outputs"])).max() < 100  # answers the size of real ones
                assert "Nothing says where" not in spec["summary"]
            else:
                # Nothing says where this model's data lies: it is computed as scikit-learn computes it, and says so.
                assert "Nothing says where the values of f0, f1, f2 or f3 lie" in spec["summary"]
    # A tree is asked, not assumed: scikit-learn rounds a tree's inputs to single precision itself, so single
    # precision is the tree, with or without rows.
    tree = DecisionTreeRegressor(max_depth=6, random_state=0).fit(table, answers)
    for rows in (table, None):
        kept = {}
        spec = graphs.via_onnx(tree, feature_names=NAMES, check_rows=rows, keep=kept)
        assert not spec.get("approximate") and spec["metadata"]["model_precision"] == "single"
        np.testing.assert_allclose(as_converted(kept["onnx"], fresh)[:, 0], tree.predict(fresh), rtol=1e-6)


def test_single_precision_stands_where_it_gives_the_models_numbers(sklearn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = LogisticRegression().fit(X, YES_NO)
    with_rows = graphs.via_onnx(model, feature_names=NAMES, check_rows=X)
    assert with_rows["metadata"]["model_precision"] == "single" and with_rows["graph"]["inputs"][0]["dtype"] == "float32"
    assert "precision" not in with_rows["summary"] and "Nothing says" not in with_rows["summary"]
    scaled = graphs.via_onnx(make_pipeline(StandardScaler(), LogisticRegression()).fit(X, YES_NO), feature_names=NAMES)
    assert scaled["metadata"]["model_precision"] == "single" and "Nothing says" not in scaled["summary"]
    # With no rows and no scaler nothing says where its data lies, and a model that tells doubles apart is computed
    # in doubles, as scikit-learn computes it.
    alone = graphs.via_onnx(model, feature_names=NAMES)
    assert alone["metadata"]["model_precision"] == "double" and not alone.get("approximate")
    assert "give rows of the table (X= or check_rows=)" in alone["summary"]


# --- 14. a wrong feature_names on the conversion route ------------------------------------------------------------


def test_a_wrong_feature_names_stays_the_callers_argument_and_reasons_end_in_one_full_stop(sklearn, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    said = "feature_names has 2 names, and this model's table has 4 columns."

    def refuses(estimator, why, numpy, **options):
        raise StructureError(said, code="invalid_structure", param="feature_names")

    monkeypatch.setattr(graphs, "_sklearn_through_onnx", refuses)
    model = LogisticRegression().fit(X, YES_NO)
    # It arrived as "This model could not be converted to ONNX: feature_names has 2 names ... .." with no param.
    with pytest.raises(StructureError) as caught:
        graphs.via_onnx(model, feature_names=["a", "b"])
    assert caught.value.param == "feature_names" and caught.value.message == said
    # Names the caller did not give are not theirs to fix: a reason among others, in a sentence with one full stop.
    why: list = []
    assert graphs.via_onnx(model, reasons=why) is None and why == [said]
    with pytest.raises(StructureError) as caught:
        extract_structure(model, via="onnx")
    assert caught.value.message == "This model could not be converted to ONNX: " + said
    with pytest.raises(StructureError) as caught:
        graphs.to_onnx(model)
    assert caught.value.message == "This model could not be converted to ONNX: " + said
    assert structure.said_together(["a reason.", "another. ", "a reason.", ""]) == "a reason; another"


# --- 15. where a mask's one score for each pixel is cut -----------------------------------------------------------

PIXELS = {"pixels": {"feature": "photo", "layout": "nchw"}}


def segmenter(channels=1, last=None):
    """An ONNX model: a convolution over a 3 by 6 by 6 picture that gives ``channels`` scores a pixel, then ``last``
    (a Sigmoid, an ArgMax over the channels, or nothing)."""
    from onnx import numpy_helper

    weights = np.random.default_rng(0).normal(size=(channels, 3, 3, 3)).astype(np.float32)
    nodes = [helper.make_node("Conv", ["pixels", "w"], ["scores" if last else "mask"], pads=[1, 1, 1, 1])]
    kind, shape = TensorProto.FLOAT, [1, channels, 6, 6]
    if last == "Sigmoid":
        nodes.append(helper.make_node("Sigmoid", ["scores"], ["mask"]))
    elif last == "ArgMax":
        nodes.append(helper.make_node("ArgMax", ["scores"], ["mask"], axis=1, keepdims=0))
        kind, shape = TensorProto.INT64, [1, 6, 6]
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 3, 6, 6])],
        [helper.make_tensor_value_info("mask", kind, shape)], [numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model.SerializeToString()


def cut_of(spec):
    """``(where a mask's score is cut, what kind of score the importer took it for when it chose the cut itself)``."""
    spec = getattr(spec, "spec", spec)
    return spec["head"].get("threshold"), spec["head"].get("score_kind")


def test_a_masks_threshold_reaches_the_importer_from_every_call(client, api):
    one, squashed = segmenter(), segmenter(last="Sigmoid")
    # Left out, the importer's own choice stands: zero for raw scores, one half for probabilities.
    assert cut_of(graphs.from_onnx(one, task="segmentation", inputs=PIXELS)) == (0.0, "raw")
    assert cut_of(graphs.from_onnx(squashed, task="segmentation", inputs=PIXELS)) == (0.5, "probability")
    # Given, it is the caller's, and nothing says what kind of score it was taken for.
    assert cut_of(graphs.from_onnx(one, task="segmentation", inputs=PIXELS, threshold=0.25)) == (0.25, None)
    assert cut_of(graphs.from_onnx(squashed, task="segmentation", inputs=PIXELS, threshold=0.7)) == (0.7, None)
    for via in ("auto", "onnx", "native"):  # an ONNX model is a graph whatever via says
        made_by = extract_structure(one, task="segmentation", inputs=PIXELS, threshold=0.0, via=via)
        assert cut_of(made_by) == (0.0, None) and not made_by.spec.get("approximate"), via
        assert "threshold" not in made_by.spec["outputs"]  # a mask's cut is its head's, not a classifier's
    create = api.post("/v1/models").mock(side_effect=made)
    api.get("/v1/models/mdl_1/structure").mock(return_value=RUNS)
    client.models.upload(one, name="Mask", task="segmentation", inputs=PIXELS, threshold=0.25)
    sent = json.loads(create.calls.last.request.content)
    assert sent["structure"]["head"]["threshold"] == 0.25 and "score_kind" not in sent["structure"]["head"]
    assert "threshold" not in sent["outputs"]  # and the model is registered with no classifier's threshold
    client.models.upload(one, name="Mask", task="segmentation", inputs=PIXELS)
    assert cut_of(json.loads(create.calls.last.request.content)["structure"]) == (0.0, "raw")
    stored = api.put("/v1/models/m1/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    shown = api.get("/v1/models/m1").mock(return_value=httpx.Response(200, json={"id": "m1", "outputs": {"threshold": 0.5}}))
    client.models.upload_structure("m1", squashed, task="segmentation", inputs=PIXELS, threshold=0.7)
    assert cut_of(json.loads(stored.calls.last.request.content)) == (0.7, None)
    assert not shown.called  # nothing of the model's registered outputs follows a mask's cut


def test_a_torch_segmenters_threshold_is_carried_through_its_conversion(torch):
    nn = torch.nn
    torch.manual_seed(0)
    raw = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1)).eval()
    squashed = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.Sigmoid()).eval()
    several = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1)).eval()
    options = dict(input_shape=(3, 6, 6), task="segmentation")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for via in ("auto", "onnx"):
            spec = extract_structure(raw, threshold=0.1, via=via, **options).spec
            assert spec["family"] == "graph" and cut_of(spec) == (0.1, None) and not spec.get("approximate"), via
        assert cut_of(graphs.via_onnx(squashed, threshold=0.7, **options)) == (0.7, None)
        assert cut_of(extract_structure(raw, via="onnx", **options)) == (0.0, "raw")
        assert cut_of(extract_structure(squashed, via="onnx", **options)) == (0.5, "probability")
        # Refused as the importer refuses it, by name, on the routes that convert; Kaimei's own readers, which
        # would drop it, refuse it as they refuse every argument only a conversion reads.
        for via in ("auto", "onnx", "native"):
            for model, value in ((several, 0.5), (raw, "high"), (raw, float("nan")), (raw, True)):
                with pytest.raises(StructureError) as caught:
                    extract_structure(model, threshold=value, via=via, **options)
                assert caught.value.param == "threshold", (via, value)
                if via != "native":
                    assert caught.value.message.startswith("threshold "), caught.value.message
        with pytest.raises(StructureError, match="via='native' reads this model with Kaimei's own readers") as caught:
            extract_structure(raw, threshold=0.1, via="native", **options)
        assert caught.value.param == "threshold"


def test_a_threshold_that_does_not_apply_is_refused_by_name_on_every_route(sklearn):
    from sklearn.linear_model import LogisticRegression

    one = segmenter()
    cases = (
        (segmenter(3), "segmentation", 0.5, "this output (1 x 3 x 6 x 6) holds scores of 3 classes."),
        (segmenter(3, "ArgMax"), "segmentation", 0.5, "this output (1 x 6 x 6) holds a class number for each pixel."),
        (one, "generation", 0.5, "threshold applies to a segmentation that gives one score for each pixel."),
        (one, "segmentation", "high", "threshold must be a number: where one score for each pixel is cut, in the model's own numbers."),
    )
    for via in ("auto", "onnx", "native"):
        for model, task, value, said in cases:
            with pytest.raises(StructureError) as caught:
                extract_structure(model, task=task, inputs=PIXELS, threshold=value, via=via)
            # The importer's own words, not "This model could not be read as an ONNX graph".
            assert caught.value.param == "threshold" and caught.value.message.endswith(said), (via, caught.value.message)
            assert caught.value.message.startswith("threshold ") and caught.value.code == "invalid_structure"
    # A classifier has no mask to cut: its own threshold is its registered outputs', and this one is refused,
    # not dropped, whether the model is converted or read by Kaimei's own readers.
    logistic = LogisticRegression().fit(X, YES_NO)
    for via in ("auto", "onnx", "native"):
        with pytest.raises(StructureError) as caught:
            extract_structure(logistic, feature_names=NAMES, threshold=0.3, via=via)
        assert caught.value.param == "threshold", via
    with pytest.raises(StructureError) as caught:
        graphs.via_onnx(logistic, feature_names=NAMES, threshold=0.3)
    assert caught.value.message == "threshold applies to a segmentation that gives one score for each pixel."


def test_a_threshold_is_not_dropped_by_the_pipeline_readers(sklearn):
    pandas = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(0)
    frame = pandas.DataFrame({"plan": rng.choice(["free", "pro"], 80), "age": rng.normal(40, 12, 80)})
    target = (frame["age"] > 40).astype(int)
    table = make_pipeline(ColumnTransformer([("c", OneHotEncoder(), ["plan"])], remainder="passthrough"), LogisticRegression()).fit(frame, target)
    texts = ["good film", "bad film", "great plot", "awful plot", "fine music", "poor acting"] * 6
    words = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, [1, 0, 1, 0, 1, 0] * 6)
    for title, pipeline in (("a table", table), ("a text", words)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert extract_structure(pipeline).spec["family"] == "graph", title  # read, when nothing is wrong
            for via in ("auto", "onnx"):
                # Their readers took any argument they did not know and said nothing of it.
                with pytest.raises(StructureError) as caught:
                    extract_structure(pipeline, threshold=0.5, via=via)
                assert caught.value.param == "threshold", (title, via)
                assert caught.value.message == "threshold applies to a segmentation that gives one score for each pixel."


def test_a_classifiers_threshold_and_a_masks_are_never_taken_for_each_other(client, api, sklearn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import FixedThresholdClassifier

    model = LogisticRegression().fit(X, YES_NO)
    create = api.post("/v1/models").mock(side_effect=made)
    api.get("/v1/models/mdl_1/structure").mock(return_value=RUNS)
    # outputs={"threshold": ...} is the class threshold a yes-or-no model is registered with, as it always was.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.models.upload(model, name="Churn", feature_names=NAMES, outputs={"threshold": 0.3})
    sent = json.loads(create.calls.last.request.content)
    assert sent["outputs"] == {"threshold": 0.3} and not any("threshold" in str(item.message) for item in caught)
    assert "threshold" not in (sent["structure"].get("head") or {})
    converted = extract_structure(model, feature_names=NAMES, via="onnx").spec
    assert "threshold" not in converted["head"] and "threshold" not in converted["outputs"]
    # A classifier's own threshold travels in its outputs, never in its head.
    fixed = graphs.via_onnx(FixedThresholdClassifier(LogisticRegression(), threshold=0.3).fit(X, YES_NO), feature_names=NAMES)
    assert fixed["outputs"]["threshold"] == 0.3 and "threshold" not in fixed["head"]
    assert models_module._carried_decision(fixed) == (0.3, 1)
    # A mask's cut is its head's: nothing is registered for it, and no class is compared by it.
    mask = graphs.from_onnx(segmenter(), task="segmentation", inputs=PIXELS, threshold=0.25)
    assert mask["head"]["threshold"] == 0.25 and "threshold" not in mask["outputs"]
    assert models_module._carried_decision(mask) is None
    assert models_module._carried_decision({**mask, "outputs": {**mask["outputs"], "threshold": 0.9}}) is None
    assert models_module._undecided(mask["outputs"]) == mask["outputs"]
    assert graphs.classes_differ(mask, lambda row: 0) == (0, 0)


def test_an_older_api_is_told_a_masks_threshold_needs_a_graph(client, api, monkeypatch):
    stand_in(monkeypatch, binary_graph())
    create = api.post("/v1/models").mock(return_value=NO_GRAPHS)
    older(api)
    with pytest.raises(APIStatusError) as caught:
        client.models.upload(object(), name="m", threshold=0.5)
    assert "threshold is read only from a model converted to a graph" in caught.value.message
    assert create.call_count == 1  # Kaimei's own readers, which refuse it, were not asked


# --- round 4 --------------------------------------------------------------------------------------------------------


def test_class_names_given_as_a_generator_are_read_once_and_kept(sklearn, client, api, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def names():
        return (name for name in ["no", "yes"])

    logistic = LogisticRegression().fit(X, YES_NO)
    for via in ("native", "onnx", "auto"):
        # Checking the names read them all, and what read them next found none: ['0', '1'], or ['negative', 'positive'].
        spec = extract_structure(logistic, feature_names=NAMES, labels=names(), via=via).spec
        assert spec["outputs"]["labels"] == ["no", "yes"], via
    assert graphs.via_onnx(logistic, feature_names=NAMES, labels=names())["outputs"]["labels"] == ["no", "yes"]
    node = helper.make_node("Softmax", ["x"], ["y"], axis=1)
    graph = helper.make_graph(
        [node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])],
    )
    file = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    file.ir_version = 9
    assert graphs.from_onnx(file, feature_names=["a", "b"], labels=names())["outputs"]["labels"] == ["no", "yes"]
    stored = api.put("/v1/models/m1/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    client.models.upload_structure("m1", logistic, feature_names=NAMES, labels=names())
    assert json.loads(stored.calls.last.request.content)["outputs"]["labels"] == ["no", "yes"]
    # An upload asked again by another route (an API from before graphs) asks with the names, not with what is left
    # of a generator that has been read.
    asked_with = []

    def extract(estimator, **options):
        asked_with.append((options["via"], options["labels"]))
        return finalize_spec(binary_graph(), weights="full", externalize_above=None)

    monkeypatch.setattr(models_module, "extract_structure", extract)
    answers = iter([NO_GRAPHS, httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "m"})])
    create = api.post("/v1/models").mock(side_effect=lambda request: next(answers))
    api.get("/v1/models/mdl_1/structure").mock(return_value=RUNS)
    older(api)
    with pytest.warns(UserWarning, match="does not run model graphs"):
        client.models.upload(object(), name="m", labels=names())
    assert asked_with == [("auto", ["no", "yes"]), ("native", ["no", "yes"])]
    assert json.loads(create.calls.last.request.content)["outputs"]["labels"] == ["no", "yes"]


def test_a_kernel_model_of_values_far_from_zero_is_not_called_exact_in_single_precision(sklearn):
    pytest.importorskip("onnxruntime")
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVC, SVR, NuSVC

    rng = np.random.default_rng(5)
    near = rng.normal(size=(300, 4))
    far, fresh = near + 1e7, rng.normal(size=(200, 4)) + 1e7
    number = 2.0 * near[:, 0] - near[:, 1] + 0.5 * near[:, 2] * near[:, 3]
    yes_no = (near[:, 0] + 0.5 * near[:, 1] > 0).astype(int)
    for title, model, target in (
        ("SVR", SVR(C=10.0), number), ("SVC", SVC(), yes_no), ("NuSVC", NuSVC(), yes_no),
        ("one against the rest", OneVsRestClassifier(SVC()), yes_no), ("neighbours", KNeighborsRegressor(3), number),
    ):
        model.fit(far, target)
        answer = model.predict if title in ("SVR", "neighbours") else model.decision_function
        own = np.asarray(answer(fresh), dtype=float)
        for rows in (None, far):
            kept: dict = {}
            spec = graphs.via_onnx(model, feature_names=NAMES, check_rows=rows, keep=kept)
            if spec.get("approximate"):
                # A reading that does not give the model's numbers near its data is drawn, and says so. (ONNX keeps
                # a support vector in single precision, so no reading of these gives them.)
                assert "so it is drawn, not run" in spec["summary"] and "check" not in spec, (title, spec["summary"])
                if title in ("SVR", "SVC", "NuSVC"):
                    assert "does not reproduce the model's own outputs" in spec["summary"], (title, spec["summary"])
                continue
            # A reading that runs gives the model's own numbers on rows like the model's. Read with no rows, an SVR
            # was called exact in single precision and was up to 22 off: a kernel is zero around zero and near 2**24
            # in either precision, so asking the model there said nothing.
            converted = as_converted(kept["onnx"], fresh)
            ours = converted[:, 0] if own.ndim == 1 and converted.shape[1] == 1 else converted
            if own.ndim == 1 and converted.shape[1] == 2:
                ours = converted[:, 1]
            assert np.abs(ours - own).max() < 1e-3 * (own.max() - own.min()), (title, rows is None)
            assert spec["metadata"]["model_precision"] == "double", (title, rows is None)
        if title == "neighbours":
            assert not spec.get("approximate")  # doubles do give a nearest-neighbours model's numbers
    # Where the data lies is read off the support vectors: between their smallest and largest values, not at them.
    svc = SVC().fit(X, YES_NO)
    ranges, points = graphs._data_ranges(svc, None, 4, np)
    assert points is not None and np.array_equal(points, svc.support_vectors_)
    assert ranges == [("between", float(points[:, at].min()), float(points[:, at].max())) for at in range(4)]
    wrapped_points, at_the_boundary = graphs._fitted_points(OneVsRestClassifier(SVC()).fit(X, THREE), 4, np)
    assert at_the_boundary and wrapped_points.shape[1] == 4
    spec = graphs.via_onnx(svc, feature_names=NAMES)
    held = check_table(spec)
    assert not spec.get("approximate") and (held >= points.min(axis=0)).all() and (held <= points.max(axis=0)).all()
    assert "Nothing says where" not in spec["summary"]  # its support vectors say


def test_a_table_of_another_width_is_said_to_be_x_not_names_the_caller_never_gave(sklearn, client, api):
    pandas = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(ColumnTransformer([("s", StandardScaler(), [0, 1, 2, 3])]), LogisticRegression()).fit(X, YES_NO)
    wide = pandas.DataFrame(np.column_stack([X[:20], np.arange(20)]), columns=["a", "b", "c", "d", "row"])
    for model in (pipeline, LogisticRegression().fit(X, YES_NO)):
        # "feature_names has 5 names, and this pipeline was fitted on a table of 4 columns", of names taken from X.
        with pytest.raises(StructureError, match="X has 5 columns, and this model was fitted on a table of 4") as caught:
            client.models.upload(model, name="m", X=wide)
        assert caught.value.param == "X" and "feature_names has" not in caught.value.message
    assert len(api.calls) == 0  # refused before anything is made
    create = api.post("/v1/models").mock(side_effect=made)
    api.get("/v1/models/mdl_1/structure").mock(return_value=RUNS)
    api.put("/v1/models/mdl_1/reference-data").mock(return_value=httpx.Response(200, json={"rows": 20, "rejected": 0}))
    client.models.upload(LogisticRegression().fit(X, YES_NO), name="m", X=wide[["a", "b", "c", "d"]])
    assert [item["name"] for item in json.loads(create.calls.last.request.content)["features"]] == ["a", "b", "c", "d"]


def test_a_file_that_is_there_and_can_be_written_is_taken_whatever_its_folder_allows(monkeypatch, tmp_path):
    there = tmp_path / "net.onnx"
    there.write_bytes(b"old")
    folder = os.path.abspath(str(tmp_path))
    allowed = {os.path.abspath(str(there)): True, folder: False}
    monkeypatch.setattr(models_module.os, "access", lambda path, mode: allowed.get(os.path.abspath(os.fspath(path)), True))
    # Its folder cannot be written to, and the file can: writing it works, so it was refused for nothing.
    models_module._check_onnx_path(there)
    with pytest.raises(StructureError, match="is in a folder that cannot be written to") as caught:
        models_module._check_onnx_path(tmp_path / "new.onnx")
    assert caught.value.param == "save_onnx"
    allowed[os.path.abspath(str(there))] = False
    with pytest.raises(StructureError, match="net.onnx', which cannot be written to"):
        models_module._check_onnx_path(there)


def test_a_decoder_input_hinted_as_generated_keeps_the_name_and_tokens_the_hint_gave(torch):
    nn = torch.nn
    vocabulary, room = 12, 4
    torch.manual_seed(3)

    class Writer(nn.Module):
        def __init__(self):
            super().__init__()
            self.read = nn.Embedding(vocabulary, 6)
            self.write = nn.Embedding(vocabulary, 6)
            self.place = nn.Parameter(torch.randn(room, 6))
            self.head = nn.Linear(6, vocabulary)
            self.register_buffer("before", torch.tril(torch.ones(room, room)) / torch.arange(1, room + 1).unsqueeze(1))

        def forward(self, input_ids, decoder_input_ids):
            memory = self.read(input_ids).mean(dim=1, keepdim=True)
            seen = torch.matmul(self.before, self.write(decoder_input_ids) + self.place)
            return self.head(seen + memory) * 3.0

    sample = {"input_ids": torch.tensor([[3, 5, 7, 2, 1]]), "decoder_input_ids": torch.tensor([[0, 0, 0, 0]])}
    cases = (
        (None, "decoder", 0, None),
        ({"generated": True}, "decoder", 0, None),
        ({"generated": True, "pad_id": 0}, "decoder", 0, 0),
        # The structure said its decoder writes into 'decoder' where the feature was 'answer', and the API refused it.
        ({"generated": True, "feature": "answer"}, "answer", 0, None),
        ({"generated": True, "start_token": 5}, "decoder", 5, None),
    )
    for hint, name, start, pad in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = extract_structure(
                Writer().eval(), sample=sample, task="sequence_to_sequence", via="onnx",
                **({} if hint is None else {"inputs": {"decoder_input_ids": dict(hint)}}),
            ).spec
        assert not spec.get("approximate") and "check" in spec, (hint, spec.get("summary"))
        assert spec["decoder"]["feature"] == name and spec["decoder"]["start_token"] == start, hint
        assert spec["decoder"].get("pad_token") == pad and spec["decoder"]["max_length"] == room, hint
        written = [item["from"] for item in spec["graph"]["inputs"] if item["from"].get("feature") == name]
        assert written and written[0].get("part") == "ids", hint  # the decoder's own input is read from that feature
        feature = next(item for item in spec["features"] if item["name"] == name)
        assert feature["generated"] is True and feature["baseline"] == [start], hint
        assert [item["name"] for item in spec["features"]] == ["text", name], hint
    # The tokenizer of a decoder nothing else names is kept under the feature it writes into, whatever it is called.
    named = {"decoder": {"feature": "answer"}, "features": [{"name": "answer", "type": "tokens"}, {"name": "decoder", "type": "tokens"}]}
    graphs._decoder_vocabulary(named, {"type": "word", "vocab": ["<unk>", "a", "film"], "unk": "<unk>"})
    assert list(named["tokenizers"]) == ["answer"] and named["decoder"]["tokenizer"] == {"same_as": "answer"}
    assert [item["type"] for item in named["features"]] == ["text", "tokens"]
