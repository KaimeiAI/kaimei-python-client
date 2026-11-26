"""The agent and the upload flow, as a pre-release review and an audit found them: each test failed before its fix.

A converted classifier's class, ONNX Runtime sessions kept across looks inside, a module's marks, an agent whose
session keeps being lost, an API from before model graphs, ``on_unsupported="error"``, ``save_onnx`` and the
weights warning.

How the API reads an agent's answer is mirrored here (``api_reads``). With ``KAIMEI_API_SRC`` set to a Kaimei API
checkout's ``src``, the same answers are also read by the API's own ``AgentRunner``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import warnings

import httpx
import pytest
import respx

np = pytest.importorskip("numpy")

from kaimei import Kaimei  # noqa: E402
from kaimei import agent as agents  # noqa: E402
from kaimei.agent import Agent, main, serve  # noqa: E402
from kaimei.errors import APIStatusError, NotFoundError  # noqa: E402
from kaimei.resources import models as models_module  # noqa: E402
from kaimei.structure import StructureError, finalize_spec  # noqa: E402

BASE = "https://api.kaimei.ai"
NAMES = ["a", "b", "c"]
FEATURES = [{"name": name, "type": "numeric"} for name in NAMES]


@pytest.fixture
def client():
    made = Kaimei(api_key="km_test_x", base_url=BASE, max_retries=0)
    yield made
    made.close()


@pytest.fixture
def api():
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


class NoStructure:
    """A model on Kaimei with no structure: its record says its task, features and classes."""

    def __init__(self, task, labels, positive=None, features=FEATURES):
        outputs = {"labels": [str(label) for label in labels], **({"positive_label": str(positive)} if positive is not None else {})}
        self.record = {"task": task, "features": features, "outputs": outputs}

    def _request(self, method, path, **kwargs):
        if path.endswith("/structure"):
            raise NotFoundError("This model has no structure.", status_code=404)
        return self.record


def positive_index(task, labels, positive=None):
    labels = [str(label) for label in labels]
    if positive is not None and str(positive) in labels:
        return labels.index(str(positive))
    return 1 if len(labels) == 2 or task == "binary_classification" else 0


def api_reads(answer, task, labels, positive=None):
    """``(probabilities, class of each row)`` as Kaimei reads an agent's answer: ``AgentRunner.read_raw`` for one
    marked raw (a softmax of scores, a sigmoid of one, the second label's however positive is set), then
    ``_normalise`` and the class a prediction reports (the threshold of one half for two classes)."""
    values = np.asarray(answer["predictions"], dtype=float)
    pos = positive_index(task, labels, positive)
    binary = task == "binary_classification"
    if answer.get("raw") is True:
        if answer.get("link") != "none":
            if values.shape[1] == 1 or answer.get("link") == "sigmoid":
                values = 1.0 / (1.0 + np.exp(-values))
            else:
                values = np.exp(values - values.max(axis=1, keepdims=True))
                values = values / values.sum(axis=1, keepdims=True)
        if values.shape[1] == 1 and binary and pos == 0:
            values = 1.0 - values  # one number is the second label's, whichever class is positive
    if values.shape[1] == 1:
        p = values[:, 0]
        values = np.column_stack([1.0 - p, p]) if pos == 1 else np.column_stack([p, 1.0 - p])
    values = np.clip(values, 0.0, 1.0)
    if binary and values.shape[1] == 2:
        score = values[:, pos]
        chosen = np.where(score >= 0.5, pos, 1 - pos)
        return values, np.where(np.abs(score - 0.5) <= 1e-15, 0, chosen)
    return values, values.argmax(axis=1)


def api_runners():
    """The API's own runners, when ``KAIMEI_API_SRC`` names a Kaimei API checkout's ``src``."""
    source = os.environ.get("KAIMEI_API_SRC")
    if not source or not os.path.isdir(source):
        pytest.skip("KAIMEI_API_SRC does not name the Kaimei API's src; the reading is checked against its mirror")
    if source not in sys.path:
        sys.path.append(source)
    return pytest.importorskip("kaimei_api.engine.runners")


# --- item 1: a converted classifier leads Kaimei to the class its model predicts --------------


def problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    two = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    three = np.digitize(X[:, 0] + 0.3 * rng.normal(size=300), [-0.5, 0.5])
    rows = (rng.normal(size=(160, 3)) * 1.3).astype(np.float32)
    return X, two, three, rows


def converted(model, X, zipmap=True):
    skl2onnx = pytest.importorskip("skl2onnx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if zipmap:
            return skl2onnx.to_onnx(model, X[:1])
        return skl2onnx.to_onnx(model, X[:1], options={id(model): {"zipmap": False}})


def as_rows(table):
    return [{name: float(value) for name, value in zip(NAMES, line)} for line in table]


def classifiers():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import (
        LogisticRegression,
        PassiveAggressiveClassifier,
        Perceptron,
        RidgeClassifier,
        SGDClassifier,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC, LinearSVC, NuSVC

    return [
        # Without probabilities: their files give scores, which a ZipMap pairs with the classes all the same.
        ("SVC", lambda: SVC(), 2, True),
        ("SVC", lambda: SVC(), 3, True),
        ("SVC ovo", lambda: SVC(decision_function_shape="ovo"), 3, True),
        ("NuSVC", lambda: NuSVC(), 2, True),
        ("SGD hinge", lambda: SGDClassifier(random_state=0), 2, True),
        ("SGD hinge", lambda: SGDClassifier(random_state=0), 3, True),
        ("Perceptron", lambda: Perceptron(), 2, True),
        ("PassiveAggressive", lambda: PassiveAggressiveClassifier(random_state=0), 3, True),
        ("LinearSVC", lambda: LinearSVC(), 2, True),
        ("Ridge", lambda: RidgeClassifier(), 3, True),
        ("SVC", lambda: SVC(), 2, False),
        ("SVC", lambda: SVC(), 3, False),
        ("SGD hinge", lambda: SGDClassifier(random_state=0), 2, False),
        ("Ridge", lambda: RidgeClassifier(), 2, False),
        # With probabilities.
        ("LogisticRegression", lambda: LogisticRegression(), 2, True),
        ("LogisticRegression", lambda: LogisticRegression(), 3, True),
        ("RandomForest", lambda: RandomForestClassifier(5, random_state=0), 2, True),
        ("SGD log", lambda: SGDClassifier(loss="log_loss", random_state=0), 2, False),
        ("GaussianNB", lambda: GaussianNB(), 3, False),
    ]


@pytest.mark.parametrize("name, make, count, zipmap", classifiers(), ids=lambda item: str(item) if not callable(item) else "")
def test_a_converted_classifier_leads_kaimei_to_the_class_its_model_predicts(name, make, count, zipmap):
    pytest.importorskip("onnxruntime")
    X, two, three, table = problem()
    model = make().fit(X, two if count == 2 else three)
    file = converted(model, X, zipmap)
    task = "binary_classification" if count == 2 else "multiclass_classification"
    labels = [str(item) for item in model.classes_]
    want = np.searchsorted(model.classes_, model.predict(table))
    rows = as_rows(table)
    for positive in ([None, labels[0]] if count == 2 else [None]):
        agent = Agent(NoStructure(task, labels, positive), "m", onnx=file)
        answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
        probabilities, chosen = api_reads(answer, task, labels, positive)
        assert (chosen == want).all(), (name, count, zipmap, positive, int((chosen != want).sum()))
        if hasattr(model, "predict_proba"):
            # Its probabilities are sent as they are, read as they are: never squashed a second time.
            np.testing.assert_allclose(probabilities, model.predict_proba(table), atol=1e-5)
        for kind in ("trace", "activations"):
            looked = agent.run({"kind": kind, "payload": {"input": rows[0]}})
            assert api_reads({**looked, "predictions": [looked["prediction"]]}, task, labels, positive)[1][0] == want[0]


def test_the_reported_svc_is_read_as_the_class_it_predicts_not_the_opposite():
    pytest.importorskip("onnxruntime")
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = SVC().fit(X, y)
    rows = [{"a": 0.3, "b": -1.2, "c": 0.8}, {"a": 1.1, "b": 0.4, "c": -0.2}, {"a": -0.6, "b": 0.1, "c": 0.5}]
    table = np.array([[row[name] for name in NAMES] for row in rows], dtype=np.float32)
    agent = Agent(NoStructure("binary_classification", ["no", "yes"]), "m", onnx=converted(model, X))
    answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
    # Its one margin: the model's own decision function, positive for its second class.
    assert answer["raw"] is True and "link" not in answer
    np.testing.assert_allclose(np.asarray(answer["predictions"]).reshape(-1), model.decision_function(table), rtol=1e-4, atol=1e-5)
    assert api_reads(answer, "binary_classification", ["no", "yes"])[1].tolist() == model.predict(table).tolist() == [0, 1, 0]


def test_what_a_converted_file_holds_is_read_from_the_steps_that_made_it():
    pytest.importorskip("onnxruntime")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC

    X, two, three, _table = problem()
    cases = (
        (SVC().fit(X, two), True, "scores"),  # a ZipMap pairs scores with the classes too
        (SVC(probability=True, random_state=0).fit(X, two), True, "none"),  # its probability model
        (LogisticRegression().fit(X, three), True, "none"),
        (SGDClassifier(loss="log_loss", random_state=0).fit(X, two), False, "none"),  # a Sigmoid, then a Concat
        (RandomForestClassifier(3, random_state=0).fit(X, two), True, None),  # its averaged votes say it themselves
    )
    for model, zipmap, said in cases:
        agent = Agent(NoStructure("binary_classification", ["0", "1"]), "m", onnx=converted(model, X, zipmap))
        assert agent._file.ends(agent._output_tensor()) == said, type(model).__name__
    assert Agent(NoStructure("multiclass_classification", ["0", "1", "2"]), "m", onnx=converted(SVC().fit(X, three), X))._file.votes(
        "ovr_output"
    )


def test_scores_that_stop_giving_the_files_class_are_followed_by_the_class_itself():
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    # Its class is the highest of its scores after a ReLU: the highest score, unless they are all below zero.
    weights = numpy_helper.from_array(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32), "W")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["scores"]),
            helper.make_node("Relu", ["scores"], ["kept"]),
            helper.make_node("ArgMax", ["kept"], ["label"], axis=1, keepdims=0),
            helper.make_node("ZipMap", ["scores"], ["output_probability"], domain="ai.onnx.ml", classlabels_int64s=[0, 1, 2]),
        ],
        "votes",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, [1]),
            helper.make_value_info(
                "output_probability",
                helper.make_sequence_type_proto(
                    helper.make_map_type_proto(TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, []))
                ),
            ),
        ],
        [weights],
    )
    file = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 1)])
    file.ir_version = 8
    features = [{"name": "a", "type": "numeric"}, {"name": "b", "type": "numeric"}]
    told: list = []
    agent = Agent(NoStructure("multiclass_classification", ["0", "1", "2"], features=features), "m", onnx=file, log=told.append)
    first = agent.run({"kind": "predict", "payload": {"rows": [{"a": 2.0, "b": 1.0}]}})
    assert first["predictions"] == [[2.0, 1.0, 1.5]] and "link" not in first  # its scores give its class
    below = agent.run({"kind": "predict", "payload": {"rows": [{"a": -2.0, "b": -1.0}]}})
    assert below == {"predictions": [[1.0, 0.0, 0.0]], "raw": True, "link": "none"}  # its class: 0, where -1.0 is highest
    assert any("its class is sent from now on" in line for line in told)
    again = agent.run({"kind": "predict", "payload": {"rows": [{"a": 2.0, "b": 1.0}]}})
    assert again["predictions"] == [[1.0, 0.0, 0.0]] and again["link"] == "none"  # one way for the model from then on


def test_a_first_answer_whose_margins_are_nought_does_not_settle_which_way_they_turn():
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    # Two scores, [margin, -margin], with the class second when the margin is above nought: as an SVC writes them.
    weight = numpy_helper.from_array(np.array([[1.0]], dtype=np.float32), "W")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["d"]),
            helper.make_node("Neg", ["d"], ["minus"]),
            helper.make_node("Concat", ["d", "minus"], ["scores"], axis=1),
            helper.make_node("Concat", ["minus", "d"], ["turned"], axis=1),
            helper.make_node("ArgMax", ["turned"], ["label"], axis=1, keepdims=0),
            helper.make_node("ZipMap", ["scores"], ["output_probability"], domain="ai.onnx.ml", classlabels_int64s=[0, 1]),
        ],
        "margins",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, [1]),
            helper.make_value_info(
                "output_probability",
                helper.make_sequence_type_proto(
                    helper.make_map_type_proto(TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, []))
                ),
            ),
        ],
        [weight],
    )
    file = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 1)])
    file.ir_version = 8
    agent = Agent(NoStructure("binary_classification", ["0", "1"], features=[{"name": "a", "type": "numeric"}]), "m", onnx=file)
    nought = agent.run({"kind": "predict", "payload": {"rows": [{"a": 0.0}]}})
    assert nought["predictions"] == [[0.0]] and agent._form == ("pending", 1.0)  # either way gives its class
    for value, margin in ((1.0, 1.0), (-2.0, -2.0)):
        answer = agent.run({"kind": "predict", "payload": {"rows": [{"a": value}]}})
        assert answer["predictions"] == [[margin]] and "link" not in answer  # was its class itself, for good
    assert agent._form == ("paired", -1.0)


def test_a_structure_that_names_the_output_is_read_as_it_says_and_nothing_is_turned():
    pytest.importorskip("onnxruntime")
    from sklearn.svm import SVC

    X, two, _three, table = problem()
    file = converted(SVC().fit(X, two), X, zipmap=False)
    scene = [
        {"id": "in", "op": "input", "units": 3, "flow_index": 0, "args": {"features": NAMES}},
        {"id": "scores", "op": "Cast", "units": 2, "flow_index": 1, "inputs": ["in"], "args": {"shape": [2], "source": "probabilities"}},
    ]

    class WithStructure(NoStructure):
        def _request(self, method, path, **kwargs):
            if path.endswith("/structure"):
                return {"scene": {"nodes": scene}, "head": {"vector_node": "scores"}}
            return self.record

    agent = Agent(WithStructure("binary_classification", ["0", "1"]), "m", onnx=file)
    answer = agent.run({"kind": "predict", "payload": {"rows": as_rows(table[:3])}})
    assert np.asarray(answer["predictions"]).shape == (3, 2) and agent._form == ("plain", 1.0)


def test_through_the_apis_own_reading_every_row_is_the_class_its_model_predicts():
    runners = api_runners()
    pytest.importorskip("onnxruntime")
    from kaimei_api.engine.features import FeatureCodec
    from kaimei_api.engine.reporting import decisions
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC

    X, two, three, table = problem()
    for model, count, zipmap in (
        (SVC(), 2, True),
        (SVC(), 2, False),
        (SVC(), 3, True),
        (SGDClassifier(random_state=0), 3, True),
        (LogisticRegression(), 2, True),
    ):
        model.fit(X, two if count == 2 else three)
        labels = [str(item) for item in model.classes_]
        task = "binary_classification" if count == 2 else "multiclass_classification"
        want = np.searchsorted(model.classes_, model.predict(table))
        for positive in ([None, labels[0]] if count == 2 else [None]):
            agent = Agent(NoStructure(task, labels, positive), "m", onnx=converted(model, X, zipmap))
            outputs = {"labels": labels, **({"positive_label": positive} if positive is not None else {})}

            def ask(kind, payload, agent=agent):
                return agent.run({"kind": kind, "payload": payload})

            runner = runners.AgentRunner(FeatureCodec.from_raw(FEATURES, None, task), task, outputs, ask)
            P = runner.predict(runner.codec.encode_many(as_rows(table), strict=True))
            chosen = decisions(P, binary=task == "binary_classification", positive=runner.positive_index)
            assert (np.asarray(chosen) == want).all(), (type(model).__name__, count, zipmap, positive)
            assert [runner.describe(row)["class_index"] for row in P[:20]] == want[:20].tolist()


# --- item 2: the sessions of a look inside are kept, within a budget the agent is given ----------


def chain(steps=40, width=4):
    from onnx import TensorProto, helper

    nodes, previous, names = [], "x", []
    for index in range(steps):
        out = f"t{index}"
        nodes.append(helper.make_node("Relu" if index % 2 else "Neg", [previous], [out], name=f"n{index}"))
        names.append(out)
        previous = out
    graph = helper.make_graph(
        nodes,
        "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, width])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    features = [{"name": f"c{i}", "type": "numeric"} for i in range(width)]
    scene = [{"id": "in", "op": "input", "units": width, "flow_index": 0, "args": {"features": [f["name"] for f in features]}}]
    for at, name in enumerate(names):
        scene.append({"id": name, "op": "Relu", "units": width, "flow_index": at + 1, "inputs": [scene[-1]["id"]], "args": {"shape": [width]}})

    class OnKaimei:
        def _request(self, method, path, **kwargs):
            if path.endswith("/structure"):
                return {"scene": {"nodes": scene}, "head": {"vector_node": names[-1]}}
            return {"features": features, "task": "regression"}

    return model, OnKaimei(), {f"c{i}": float(i) - 1.5 for i in range(width)}


@pytest.fixture
def made(monkeypatch):
    onnxruntime = pytest.importorskip("onnxruntime")
    real = onnxruntime.InferenceSession
    sessions: list = []

    class Counting(real):  # type: ignore[misc, valid-type]
        def __init__(self, source, options=None, *args, **kwargs):
            sessions.append(1)
            super().__init__(source, options, *args, **kwargs)

    monkeypatch.setattr(onnxruntime, "InferenceSession", Counting)
    return sessions


def sessions_made(made, agent, jobs):
    counts = []
    for job in jobs:
        before = len(made)
        agent.run(job)
        counts.append(len(made) - before)
    return counts


def test_a_prediction_between_looks_no_longer_makes_every_session_of_a_look_again(made, monkeypatch):
    monkeypatch.setattr(agents, "FETCH_STEPS", 10)  # 40 steps: a look in four runs, as a model of 1,024 steps has
    model, on_kaimei, row = chain()
    look = {"kind": "activations", "payload": {"input": row}}
    predict = {"kind": "predict", "payload": {"rows": [row]}}
    agent = Agent(on_kaimei, "m", onnx=model)
    agent._file.weight_bytes = 150 << 20  # room for four sessions in 2 GiB: a look's, or the plain one and three
    assert sessions_made(made, agent, [look, look, predict, look, look]) == [4, 0, 1, 1, 1]  # was [4, 0, 1, 4, 4]
    assert () in agent._file._sessions  # the plain session is kept
    # Given the memory for them, every session stays.
    roomy = Agent(on_kaimei, "m", onnx=model, session_memory=8 << 30)
    roomy._file.weight_bytes = 150 << 20
    assert sessions_made(made, roomy, [look, predict, look, predict]) == [4, 1, 0, 0]


def test_a_look_larger_than_the_memory_makes_again_only_what_does_not_fit(made, monkeypatch):
    monkeypatch.setattr(agents, "FETCH_STEPS", 10)
    model, on_kaimei, row = chain()
    look = {"kind": "activations", "payload": {"input": row}}
    agent = Agent(on_kaimei, "m", onnx=model)
    agent._file.weight_bytes = 250 << 20  # room for two: as before, three are kept however little the memory
    assert sessions_made(made, agent, [look, look, look]) == [4, 1, 1]  # was [4, 4, 4]
    assert len(agent._file._sessions) <= agents.SESSIONS_KEPT
    # Given no memory, the plain session is still kept between looks, and a prediction makes it only once.
    bare = Agent(on_kaimei, "m", onnx=model, session_memory=0)
    predict = {"kind": "predict", "payload": {"rows": [row]}}
    assert sessions_made(made, bare, [predict, look, predict, look, predict])[::2] == [1, 0, 0]
    assert () in bare._file._sessions


def test_the_session_memory_is_an_argument_of_the_agent_and_its_command_line(monkeypatch):
    pytest.importorskip("onnxruntime")
    model, on_kaimei, _row = chain(steps=2)
    assert Agent(on_kaimei, "m", onnx=model)._file.memory == agents.SESSION_BYTES
    assert Agent(on_kaimei, "m", onnx=model, session_memory=3 << 30)._file.memory == 3 << 30
    for wrong in (-1, True):
        with pytest.raises(ValueError, match="session_memory"):
            Agent(on_kaimei, "m", onnx=model, session_memory=wrong)
    seen: dict = {}

    class Recording:
        def __init__(self, client, model, predict=None, **options):
            seen.update(options)

        def serve(self):
            return None

    import kaimei.client

    class Shell:
        base_url = BASE

        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(agents, "Agent", Recording)
    monkeypatch.setattr(kaimei.client, "Kaimei", Shell)
    assert main(["--model", "m", "--onnx", "model.onnx", "--session-memory", "4096", "--quiet"]) == 0
    assert seen["session_memory"] == 4096 << 20 and seen["onnx"] == "model.onnx"
    assert main(["--model", "m", "--onnx", "model.onnx", "--quiet"]) == 0 and seen["session_memory"] is None


# --- item 3: a mark left on a tensor a module returns as it is ------------------------------------


@pytest.mark.parametrize("kept", ["a parameter", "an attribute", "outside the module"])
def test_a_module_that_returns_the_same_tensor_every_time_does_not_break_a_later_look(kept):
    torch = pytest.importorskip("torch")
    outside = torch.ones(1, 4)  # in a variable the module's code closes over: nothing says whose it is

    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # A tensor that is kept between calls: a parameter, a plain attribute, or nothing of the module's at all.
            self.value = torch.nn.Parameter(torch.ones(1, 4)) if kept == "a parameter" else torch.ones(1, 4)

        def forward(self, x):
            return outside if kept == "outside the module" else self.value  # the same tensor on every call

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.constant = Constant()
            self.act = torch.nn.ReLU()  # no step of its own: it is followed
            self.head = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.act(self.constant(x)) + x)

    net = Net()
    document = {
        "scene": {
            "nodes": [
                {"id": "constant", "op": "dense", "path": "constant", "units": 4, "flow_index": 0, "args": {"shape": [4]}},
                {"id": "head", "op": "dense", "path": "head", "units": 2, "flow_index": 1, "args": {"shape": [2]}},
            ]
        },
        "head": {"vector_node": "head"},
    }

    class OnKaimei:
        def _request(self, method, path, **kwargs):
            if path.endswith("/structure"):
                return document
            return {"features": [{"name": f"f{i}", "type": "numeric"} for i in range(4)], "task": "regression"}

    agent = Agent(OnKaimei(), "m", module=net)
    row = {f"f{i}": float(i) for i in range(4)}
    first = agent.run({"kind": "trace", "payload": {"input": row}})
    if kept == "outside the module":
        # Kept where an agent cannot find it, and never seen being made: it is taken for what the module computed.
        assert [frame["node_id"] for frame in first["frames"]] == ["constant", "head"]
        assert "withheld" not in first
    else:
        # What a module keeps, a parameter or a plain attribute, holds the model's own numbers: nothing of that
        # step is sent.
        assert [frame["node_id"] for frame in first["frames"]] == ["head"]
        assert [item["node_id"] for item in first["withheld"]] == ["constant"]
    for tensor in (net.constant.value, outside):
        assert not hasattr(tensor, "_kaimei_call") and not hasattr(tensor, "_kaimei_own")  # the marks go with the run
    # A look that does not listen to the module that returns its parameter: it used to fail with a KeyError.
    only = agent.run({"kind": "trace", "payload": {"input": row, "nodes": ["head"]}})
    assert [frame["node_id"] for frame in only["frames"]] == ["head"]


# --- item 4: an agent whose session keeps being lost -------------------------------------------


class Recorded(threading.Event):
    """A stop event that does not sleep, remembers how long it was asked to wait, and stops after ``most`` waits."""

    def __init__(self, most=1000):
        super().__init__()
        self.waits: list = []
        self.most = most

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if len(self.waits) >= self.most:
            self.set()
        return self.is_set()


def lost_session():
    return httpx.Response(404, json={"error": {"type": "not_found_error", "code": "agent_session_not_found", "message": "No such session."}})


def test_a_session_lost_again_and_again_is_opened_after_a_growing_pause(client, api):
    opened = api.post("/v1/models/churn/agent/sessions").mock(side_effect=lambda request: httpx.Response(201, json={"id": "s1"}))
    stop = Recorded(most=4)

    def poll(request):
        if len(polls.calls) > 50:
            stop.set()  # an agent that does not pause is stopped here, not left asking for ever
        return lost_session()

    polls = api.post("/v1/models/churn/agent/sessions/s1/poll").mock(side_effect=poll)
    serve(client, "churn", lambda rows: [0.5], stop=stop)
    assert stop.waits == [1.0, 2.0, 4.0, 8.0]  # none after the first loss; was no pause at all
    assert opened.call_count == 5 and polls.call_count == 5


def test_a_session_lost_once_is_opened_again_at_once(client, api):
    api.post("/v1/models/churn/agent/sessions").mock(side_effect=lambda request: httpx.Response(201, json={"id": "s1"}))
    stop = Recorded()
    answers = [lost_session(), httpx.Response(200, json={"job": None}), lost_session()]

    def poll(request):
        if not answers:
            stop.set()
            return httpx.Response(200, json={"job": None})
        return answers.pop(0)

    api.post("/v1/models/churn/agent/sessions/s1/poll").mock(side_effect=poll)
    api.delete("/v1/models/churn/agent/sessions/s1").mock(return_value=httpx.Response(200, json={"deleted": True}))
    serve(client, "churn", lambda rows: [0.5], stop=stop, wait=0)
    assert all(wait <= agents.MIN_POLL_SECONDS for wait in stop.waits)  # only the pauses between empty polls


# --- items 5, 6, 8, 9: the upload --------------------------------------------------------------


def graph_spec(task="regression", features=None):
    features = features or [{"name": "a", "type": "numeric"}, {"name": "b", "type": "numeric"}]
    names = [str(feature["name"]) for feature in features]
    return {
        "format": "kaimei.structure.v1",
        "family": "graph",
        "framework": "sklearn",
        "task": task,
        "feature_names": names,
        "features": features,
        "graph": {
            "opsets": {"": 17},
            "inputs": [{"name": "x", "dtype": "float32", "shape": [1, len(names)], "from": {"features": names}}],
            "constants": {"w": np.ones((len(names), 1), dtype=np.float32)},
            "nodes": [{"id": "fc", "op": "MatMul", "inputs": ["x", "w"], "outputs": ["y"], "attrs": {}}],
            "outputs": ["y"],
            "values": {"y": {"shape": [1, 1], "dtype": "float32"}},
        },
        "head": {"output": "y", "link": "none"},
        "check": {"inputs": {name: [[0.0]] for name in names}, "outputs": [[0.0]], "tolerance": 1e-4},
        "metadata": {"route": "onnx"},
    }


NATIVE = {
    "format": "kaimei.structure.v1",
    "family": "network",
    "framework": "sklearn",
    "task": "binary_classification",
    "feature_names": ["a", "b"],
    "layers": [{"id": "fc", "type": "dense", "units": 1, "weight": [[0.5], [-0.5]], "bias": [0.1]}],
}


def stand_in(monkeypatch, by_route, onnx=None):
    """``extract_structure`` answering for each route (``via``) with a spec made here, or refusing."""
    routes: list = []

    def extract(estimator, **options):
        routes.append(options.get("via"))
        made = by_route[options.get("via")]
        if isinstance(made, Exception):
            raise made
        done = finalize_spec(made, weights=options.get("weights", "full"), externalize_above=None)
        if onnx is not None and made.get("family") == "graph":
            done.onnx = onnx
        return done

    monkeypatch.setattr(models_module, "extract_structure", extract)
    return routes


def older(api, version="1.1.0"):
    return api.get("/health").mock(return_value=httpx.Response(200, json={"status": "ok", "version": version}))


def refused(status, code, message, param):
    return httpx.Response(status, json={"error": {"type": "invalid_request_error", "code": code, "message": message, "param": param}})


OLD_TASK = refused(422, "validation_error", "Input should be 'binary_classification', 'multiclass_classification' or 'regression'", "task")


def test_a_graph_with_a_task_an_older_api_does_not_know_falls_back_as_0_10_0_did(client, api, monkeypatch, tmp_path):
    routes = stand_in(monkeypatch, {"auto": graph_spec(task="multilabel_classification"), "native": NATIVE}, onnx=b"onnx")

    def create(request):
        body = json.loads(request.content)
        if body["task"] not in ("binary_classification", "multiclass_classification", "regression"):
            return OLD_TASK
        return httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "m"})

    made = api.post("/v1/models").mock(side_effect=create)
    older(api)
    with pytest.warns(UserWarning, match=r"Kaimei API 1\.1\.0 does not run model graphs.*save_onnx wrote no file"):
        model = client.models.upload(object(), name="Labels", save_onnx=tmp_path / "m.onnx")
    assert model["id"] == "mdl_1" and routes == ["auto", "native"] and made.call_count == 2  # was the raw 422
    assert not (tmp_path / "m.onnx").exists()  # the file is not what was uploaded


def test_one_kaimeis_own_readers_refuse_is_refused_saying_the_api_it_needs(client, api, monkeypatch):
    stand_in(monkeypatch, {"auto": graph_spec(task="clustering"), "native": StructureError("Cannot read this estimator.")})
    api.post("/v1/models").mock(return_value=OLD_TASK)
    older(api, "1.2.0")
    with pytest.raises(StructureError, match=r"Kaimei API 1\.2\.0 does not run model graphs.*API 1\.3\.0 runs it.*Cannot read"):
        client.models.upload(object(), name="Groups")


def test_a_feature_type_or_a_task_an_older_api_does_not_know_says_which_api_it_needs(client, api, monkeypatch):
    window = [{"name": "window", "type": "sequence", "shape": [24, 3]}]
    stand_in(monkeypatch, {"onnx": graph_spec(features=window), "native": dict(NATIVE, task="multilabel_classification")})
    route = api.post("/v1/models").mock(
        return_value=refused(400, "invalid_feature_schema", "Unsupported feature type 'sequence' for 'window'", "features")
    )
    older(api)
    with pytest.raises(APIStatusError, match=r"Kaimei API 1\.1\.0 does not know the feature type 'sequence'.*API 1\.3\.0 or later") as caught:
        client.models.upload(object(), name="Forecast", via="onnx")
    assert caught.value.status_code == 400 and caught.value.code == "invalid_feature_schema" and route.call_count == 1
    # A native reading with a task that came with graphs: no reading of Kaimei's own is older than it.
    route.mock(return_value=OLD_TASK)
    with pytest.raises(APIStatusError, match=r"does not know the task 'multilabel_classification'.*API 1\.3\.0 or later"):
        client.models.upload(object(), name="Labels", via="native")
    # An API that knows them refuses for another reason: that is passed on as it is.
    older(api, "1.3.0")
    with pytest.raises(APIStatusError) as caught:
        client.models.upload(object(), name="Labels", via="native")
    assert caught.value.message.startswith("Input should be")


def test_an_agent_an_older_api_has_no_place_for_says_so_and_nothing_is_tried_again(client, api, monkeypatch):
    routes = stand_in(monkeypatch, {"auto": graph_spec(), "native": NATIVE})
    route = api.post("/v1/models").mock(return_value=refused(422, "validation_error", "Input should be 'remote', 'builtin' or 'none'", "inference.type"))
    older(api)
    with pytest.raises(APIStatusError, match=r"Kaimei API 1\.1\.0 has no agents.*needs API 1\.3\.0"):
        client.models.upload(object(), name="Agent", inference={"type": "agent"})
    assert route.call_count == 1 and routes == ["auto"]


def test_a_model_on_unsupported_error_refuses_is_deleted_before_the_error(client, api, monkeypatch, tmp_path):
    stand_in(monkeypatch, {"auto": graph_spec()}, onnx=b"onnx")
    api.post("/v1/models").mock(return_value=httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "m"}))
    api.get(url__regex=r".*/v1/models/mdl_1/structure.*").mock(
        return_value=httpx.Response(200, json={"runnable": False, "summary": "Kaimei draws but does not run: Foo (ai.example)."})
    )
    reference = api.put("/v1/models/mdl_1/reference-data").mock(return_value=httpx.Response(200, json={"rows": 2}))
    deleted = api.delete("/v1/models/mdl_1").mock(return_value=httpx.Response(200, json={"deleted": True}))
    with pytest.raises(StructureError, match=r"only draws this model.*Foo.*so the model was deleted again \(mdl_1\)") as caught:
        client.models.upload(object(), name="m", on_unsupported="error", X=[[1.0, 2.0], [3.0, 4.0]], save_onnx=tmp_path / "m.onnx")
    assert caught.value.code == "unsupported_layers" and deleted.call_count == 1  # was left registered
    assert not reference.called and not (tmp_path / "m.onnx").exists()  # nothing more for a model that is not kept
    deleted.mock(return_value=httpx.Response(500, json={"error": {"type": "api_error", "message": "Down."}}))
    with pytest.raises(StructureError, match=r"deleting it again failed: delete mdl_1 yourself"):
        client.models.upload(object(), name="m", on_unsupported="error")


def test_upload_structure_with_metadata_keeps_the_onnx_file_to_write(client, api, tmp_path):
    stored = api.put("/v1/models/m1/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    extraction = finalize_spec(graph_spec(), externalize_above=None)
    extraction.onnx = b"onnx bytes"
    client.models.upload_structure("m1", extraction, metadata={"run": "42"}, save_onnx=tmp_path / "m.onnx")
    assert (tmp_path / "m.onnx").read_bytes() == b"onnx bytes"  # was no file, and a warning that none was written
    assert json.loads(stored.calls.last.request.content)["metadata"] == {"route": "onnx", "run": "42"}
    # Refused, the structure is not the model's: no file is written for it.
    stored.mock(return_value=httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "No."}}))
    with pytest.raises(APIStatusError):
        client.models.upload_structure("m1", extraction, save_onnx=tmp_path / "refused.onnx")
    assert not (tmp_path / "refused.onnx").exists()


def test_the_weights_warning_says_none_were_sent_and_an_agent_can_run_the_model(client, api, monkeypatch):
    stand_in(monkeypatch, {"auto": graph_spec()})
    api.post("/v1/models").mock(return_value=httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "m"}))
    for weights in ("summary", "pictures"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.models.upload(object(), name="m", weights=weights)
        said = [str(item.message) for item in caught]
        assert any(f"weights='{weights}' sent none of this model's weights" in line and "kaimei.agent" in line for line in said)
        assert not any("sketch or feature-importance summary" in line for line in said)  # what it said before
    # A graph drawn because its conversion does not reproduce it says so, not that it is a sketch.
    drawn = dict(graph_spec(), approximate=True, summary="Its ONNX conversion does not reproduce the model's own outputs.")
    drawn.pop("check")
    stand_in(monkeypatch, {"auto": drawn})
    with pytest.warns(UserWarning, match=r"draws this model and does not run it: Its ONNX conversion does not reproduce") as caught:
        client.models.upload(object(), name="m")
    assert not any("sketch" in str(item.message) for item in caught)
