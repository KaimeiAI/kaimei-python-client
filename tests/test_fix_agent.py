"""The agent and the upload, as a pre-release review found them: each test here failed before its fix.

A converted classifier's ZipMap, ONNX Runtime sessions kept across looks inside, answers too large for the
API, SIGTERM, polls that do not wait, and an API from before model graphs.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import types
import warnings

import httpx
import pytest
import respx

np = pytest.importorskip("numpy")

from kaimei import AuthenticationError, Kaimei, KaimeiError  # noqa: E402
from kaimei import agent as agents  # noqa: E402
from kaimei.agent import Agent, main, serve  # noqa: E402
from kaimei.errors import APIStatusError, NotFoundError  # noqa: E402
from kaimei.structure import StructureError  # noqa: E402

BASE = "https://api.kaimei.ai"
MODEL = "/v1/models/churn"


@pytest.fixture
def client():
    made = Kaimei(api_key="km_test_x", base_url=BASE, max_retries=0)
    yield made
    made.close()


@pytest.fixture
def api():
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


class Recorded(threading.Event):
    """A stop event that does not sleep, remembers how long it was asked to, and stops after ``most`` waits."""

    def __init__(self, most: int = 1000) -> None:
        super().__init__()
        self.waits: list = []
        self.most = most

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if len(self.waits) >= self.most:
            self.set()
        return self.is_set()


def working(api, jobs, stop, session="s1"):
    """An API that hands out ``jobs`` one poll at a time, then asks the agent to stop."""
    queue = list(jobs)
    api.post(f"{MODEL}/agent/sessions").mock(
        return_value=httpx.Response(201, json={"object": "model.agent.session", "id": session, "model_id": "m", "capabilities": ["predict"]})
    )

    def poll(request):
        if queue:
            return httpx.Response(200, json={"job": queue.pop(0)})
        stop.set()
        return httpx.Response(200, json={"job": None})

    polled = api.post(f"{MODEL}/agent/sessions/{session}/poll").mock(side_effect=poll)
    answers = api.post(url__regex=rf".*{MODEL}/agent/jobs/[^/]+/result").mock(return_value=httpx.Response(200, json={"job": None}))
    closed = api.delete(f"{MODEL}/agent/sessions/{session}").mock(return_value=httpx.Response(200, json={"deleted": True}))
    return polled, answers, closed


def sent(answers):
    return [json.loads(call.request.content) for call in answers.calls]


class OnKaimei:
    """What the model on Kaimei says of itself, without HTTP: its record and its structure (or none)."""

    def __init__(self, features, task="regression", document=None, labels=None):
        self.record = {"features": features, "task": task, **({"outputs": {"labels": labels}} if labels else {})}
        self.document = document

    def _request(self, method, path, **kwargs):
        if path.endswith("/structure"):
            if self.document is None:
                raise NotFoundError("This model has no structure.", status_code=404)
            return self.document
        return self.record


# --- a scikit-learn classifier as skl2onnx writes it: its probabilities in a ZipMap ---------------


def test_a_converted_classifier_answers_with_its_probabilities_not_its_label():
    pytest.importorskip("onnxruntime")
    skl2onnx = pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    data = skl2onnx.to_onnx(model, X[:1]).SerializeToString()  # outputs: the label (int64), then a ZipMap
    features = [{"name": f"f{i}", "type": "numeric"} for i in range(3)]
    agent = Agent(OnKaimei(features, "binary_classification", labels=["0", "1"]), "m", onnx=data)
    assert agent._file.floats == []  # no output of the file is a tensor of floats
    rows = [{f"f{i}": float(X[j, i]) for i in range(3)} for j in range(5)]
    answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
    assert answer["raw"] is True
    np.testing.assert_allclose(answer["predictions"], model.predict_proba(X[:5]), atol=1e-6)


def test_a_file_whose_outputs_hold_no_scores_is_refused_rather_than_answered_with_labels():
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(np.eye(3, 2, dtype=np.float32), "W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["scores"]), helper.make_node("ArgMax", ["scores"], ["label"], axis=1)],
        "labels_only",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [1, 1])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    features = [{"name": name, "type": "numeric"} for name in "abc"]
    agent = Agent(OnKaimei(features, "binary_classification"), "m", onnx=model)
    with pytest.raises(ValueError, match=r"No output of this ONNX file holds scores.*label \(int64\)"):
        agent.run({"kind": "predict", "payload": {"rows": [{"a": 1.0, "b": 0.0, "c": 2.0}]}})


# --- a look inside a model with more steps than one run fetches ------------------------------


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
    document = {"scene": {"nodes": scene}, "head": {"vector_node": names[-1]}}
    return model, OnKaimei(features, document=document), {f"c{i}": float(i) - 1.5 for i in range(width)}


@pytest.fixture
def made(monkeypatch):
    """Every ONNX Runtime session made, with whether it keeps ONNX Runtime's memory arena."""
    onnxruntime = pytest.importorskip("onnxruntime")
    real = onnxruntime.InferenceSession
    sessions: list = []

    class Counting(real):  # type: ignore[misc, valid-type]
        def __init__(self, source, options=None, *args, **kwargs):
            sessions.append(bool(getattr(options, "enable_cpu_mem_arena", True)))
            super().__init__(source, options, *args, **kwargs)

    monkeypatch.setattr(onnxruntime, "InferenceSession", Counting)
    return sessions


def test_the_sessions_of_a_look_in_several_runs_are_kept_for_the_next_look(made, monkeypatch):
    monkeypatch.setattr(agents, "FETCH_STEPS", 10)  # 40 steps: four runs, as a model of 1,024 steps has
    model, on_kaimei, row = chain()
    agent = Agent(on_kaimei, "m", onnx=model)
    job = {"kind": "activations", "payload": {"input": row}}
    built = []
    for _ in range(3):
        before = len(made)
        assert len(agent.run(job)["stages"]) == 41
        built.append(len(made) - before)
    assert built == [4, 0, 0]
    assert made == [False] * 4  # a session for a look inside gives back what a run took
    before = len(made)
    agent.run({"kind": "predict", "payload": {"rows": [row]}})
    agent.run(job)
    agent.run({"kind": "trace", "payload": {"input": row}})
    assert len(made) - before == 1  # the file's own outputs, made once; the looks inside made nothing again
    assert made[-1] is True  # and that one keeps its arena: predictions are what it is for


def test_sessions_kept_stay_within_the_memory_they_may_hold(made, monkeypatch):
    monkeypatch.setattr(agents, "FETCH_STEPS", 10)
    # No room: the few kept however little the memory hold the look's first sessions, and only the last is made again.
    monkeypatch.setattr(agents, "SESSION_BYTES", 0)
    model, on_kaimei, row = chain()
    agent = Agent(on_kaimei, "m", onnx=model)
    job = {"kind": "activations", "payload": {"input": row}}
    agent.run(job)
    before = len(made)
    agent.run(job)
    assert len(made) - before == 1 and len(agent._file._sessions) <= agents.SESSIONS_KEPT


# --- answers too large for the API, and answers that do not get through ----------------------


def test_an_answer_refused_for_its_size_is_reported_as_a_failure(client, api):
    stop = threading.Event()
    polled, answers, closed = working(api, [{"id": "big", "kind": "predict", "payload": {"rows": [{"x": 1}]}}], stop)

    def reply(request):
        if json.loads(request.content)["ok"]:
            return httpx.Response(413, json={"error": {"code": "payload_too_large", "message": "This request's body is too large."}})
        return httpx.Response(200, json={"job": None})

    answers.mock(side_effect=reply)
    lines: list = []
    serve(client, "churn", lambda rows: [1.0], stop=stop, log=lines.append)
    bodies = sent(answers)
    assert bodies[0]["ok"] is True
    assert bodies[1] == {"ok": False, "error": "The agent's answer was refused: This request's body is too large."}
    assert any("refused by Kaimei" in line for line in lines)


def test_an_answer_lost_on_the_way_is_followed_by_a_failure(client, api):
    stop = Recorded()  # the second before the answer is sent once more is not really waited
    polled, answers, closed = working(api, [{"id": "lost", "kind": "predict", "payload": {"rows": [{"x": 1}]}}], stop)

    def reply(request):
        if json.loads(request.content)["ok"]:
            raise httpx.ReadError("connection reset")
        return httpx.Response(200, json={"job": None})

    answers.mock(side_effect=reply)
    serve(client, "churn", lambda rows: [1.0], stop=stop)
    bodies = sent(answers)
    assert [body["ok"] for body in bodies] == [True, True, False]  # sent, sent once more, then the failure
    assert bodies[2]["error"].startswith("The agent's answer could not be sent: No response from Kaimei")
    assert stop.waits == [1.0]


def test_an_answer_with_more_values_than_the_api_takes_is_not_sent(client, api, monkeypatch):
    monkeypatch.setattr(agents, "MAX_ANSWER_VALUES", 10)
    stop = threading.Event()
    polled, answers, closed = working(api, [{"id": "wide", "kind": "predict", "payload": {"rows": [{"x": 1}] * 4}}], stop)
    serve(client, "churn", lambda rows: np.zeros((len(rows), 3)), stop=stop)
    assert sent(answers) == [
        {
            "ok": False,
            "error": "This answer to a predict job holds 18 values (4 rows of 3 numbers), and Kaimei takes at most 10 in "
            "an agent's answer. Ask for fewer rows at a time.",
        }
    ]
    # Counted as the API counts them: the answer, its list, each row and each number.
    assert agents._answer_size({"predictions": [[0.0] * 3] * 4})[0] == 18


def test_an_answer_with_more_bytes_than_the_api_takes_is_not_sent(client, api, monkeypatch):
    monkeypatch.setattr(agents, "MAX_ANSWER_BYTES", 300)
    stop = threading.Event()
    polled, answers, closed = working(api, [{"id": "long", "kind": "predict", "payload": {"rows": [{"x": 1}] * 2}}], stop)
    serve(client, "churn", lambda rows: np.full((len(rows), 20), 1 / 3), stop=stop)
    (body,) = sent(answers)
    assert body["ok"] is False and "MiB of JSON (2 rows of 20 numbers)" in body["error"]
    # One that fits goes as it is.
    values, most = agents._answer_size({"predictions": [[1 / 3] * 2]})
    assert values == 5 and most >= len(json.dumps({"predictions": [[1 / 3] * 2]}))


# --- the command line, stopped as a container is ---------------------------------------------


class Shell:
    """``Kaimei`` as the command line makes it: an API that answers polls, and raises SIGTERM on the first."""

    made: list = []

    def __init__(self, **kwargs):
        self.base_url = BASE
        self.calls: list = []
        Shell.made.append(self)

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path))
        if path.endswith("/agent/sessions"):
            return {"id": "s1"}
        if path.endswith("/poll"):
            polls = sum(1 for _method, called in self.calls if called.endswith("/poll"))
            if polls == 1:
                signal.raise_signal(signal.SIGTERM)
            if polls > 3:  # without a handler of its own the agent would ask for ever
                raise AuthenticationError("stopped by the test", status_code=401)
            return {"job": None}
        return {}

    def close(self):
        pass


def test_sigterm_stops_the_command_line_and_closes_its_session(monkeypatch, capsys):
    import kaimei.client

    module = types.ModuleType("kaimei_fix_agent_scoring")
    module.predict = lambda rows: [0.5] * len(rows)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaimei_fix_agent_scoring", module)
    monkeypatch.setattr(kaimei.client, "Kaimei", Shell)
    monkeypatch.setattr(agents, "MIN_POLL_SECONDS", 0.0)
    heard: list = []

    def theirs(signum, frame):
        heard.append(signum)

    previous = signal.signal(signal.SIGTERM, theirs)
    try:
        code = main(["--model", "churn", "--predict", "kaimei_fix_agent_scoring:predict", "--quiet"])
        assert signal.getsignal(signal.SIGTERM) is theirs  # put back as it was
    finally:
        signal.signal(signal.SIGTERM, previous)
    calls = Shell.made[-1].calls
    assert code == 0 and heard == []
    assert calls[-1] == ("DELETE", "/v1/models/churn/agent/sessions/s1")
    assert sum(1 for _method, path in calls if path.endswith("/poll")) == 1


# --- how often an agent asks for work ----------------------------------------------------------


def test_an_agent_that_does_not_wait_still_pauses_between_empty_polls(client, api):
    api.post(f"{MODEL}/agent/sessions").mock(return_value=httpx.Response(201, json={"id": "s1"}))
    stop = Recorded(most=3)

    def poll(request):
        if len(polled.calls) >= 10:
            stop.set()  # an agent that asks without pausing is stopped here, not left asking for ever
        return httpx.Response(200, json={"job": None})

    polled = api.post(f"{MODEL}/agent/sessions/s1/poll").mock(side_effect=poll)
    api.delete(f"{MODEL}/agent/sessions/s1").mock(return_value=httpx.Response(200, json={"deleted": True}))
    serve(client, "churn", lambda rows: [0.5], stop=stop, wait=0)
    assert polled.call_count == 3 and json.loads(polled.calls.last.request.content) == {"wait": 0.0}
    assert len(stop.waits) == 3 and all(0 < wait <= agents.MIN_POLL_SECONDS for wait in stop.waits)


def test_a_poll_is_held_open_for_less_than_the_clients_timeout(api):
    api.post(f"{MODEL}/agent/sessions").mock(return_value=httpx.Response(201, json={"id": "s1"}))
    stop = threading.Event()

    def poll(request):
        stop.set()
        return httpx.Response(200, json={"job": None})

    polled = api.post(f"{MODEL}/agent/sessions/s1/poll").mock(side_effect=poll)
    api.delete(f"{MODEL}/agent/sessions/s1").mock(return_value=httpx.Response(200, json={"deleted": True}))
    with Kaimei(api_key="km_test_x", base_url=BASE, max_retries=0, timeout=8) as short:
        serve(short, "churn", lambda rows: [0.5], stop=stop)
        assert json.loads(polled.calls.last.request.content) == {"wait": 6.0}
        assert Agent(short, "churn", lambda rows: [0.5], wait=3).poll_wait() == 3.0
    with Kaimei(api_key="km_test_x", base_url=BASE, timeout=2) as shorter:
        assert Agent(shorter, "churn", lambda rows: [0.5]).poll_wait() == 0.0
    with Kaimei(api_key="km_test_x", base_url=BASE) as default:
        assert Agent(default, "churn", lambda rows: [0.5]).poll_wait() == agents.WAIT_SECONDS
    assert Agent(OnKaimei([]), "churn", lambda rows: [0.5]).poll_wait() == agents.WAIT_SECONDS  # no timeout to read


# --- an API from before model graphs --------------------------------------------------------


def api_version(api, version):
    return api.get("/health").mock(return_value=httpx.Response(200, json={"status": "ok", "version": version}))


def refused(message, code="invalid_structure"):
    return httpx.Response(400, json={"error": {"type": "invalid_request_error", "code": code, "message": message}})


def onnx_mlp():
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(3)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["X", "W", "B"], ["y"])],
        "mlp",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
        [numpy_helper.from_array(rng.normal(size=(3, 1)).astype(np.float32), "W"), numpy_helper.from_array(np.zeros(1, np.float32), "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_whole_number_tensors_refused_by_an_older_api_say_which_api_takes_them(client, api):
    api.put(url__regex=r".*/structure/tensors/.+").mock(
        return_value=refused("Unsupported tensor dtype 'i64'. Use f16, f32 or f64.", code="invalid_tensor")
    )
    api_version(api, "1.1.0")
    with pytest.raises(APIStatusError, match=r"Kaimei API 1\.1\.0 .*\(i64\) need API 1\.3\.0 .*dtype='f32'.*The API said: Unsupported") as caught:
        client.models.upload_tensors("m1", {"idx": np.arange(10, dtype=np.int64)})
    assert caught.value.code == "invalid_tensor" and caught.value.status_code == 400
    api_version(api, "1.3.0")  # an API that takes them and refuses for another reason is passed on as it is
    with pytest.raises(APIStatusError) as caught:
        client.models.upload_tensors("m1", {"idx": np.arange(10, dtype=np.int64)})
    assert caught.value.message == "Unsupported tensor dtype 'i64'. Use f16, f32 or f64."


def test_a_graph_refused_by_an_older_api_says_which_api_it_needs_and_leaves_nothing(client, api):
    pytest.importorskip("onnx")
    create = api.post("/v1/models").mock(return_value=refused("Structure spec needs a layers array."))
    api_version(api, "1.1.0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(APIStatusError, match=r"Kaimei API 1\.1\.0 cannot take this model: .*needs API 1\.3\.0"):
            client.models.upload(onnx_mlp(), name="From a file", feature_names=["a", "b", "c"], task="regression")
    assert create.call_count == 1  # an ONNX file has no other route: nothing is tried again
    api_version(api, "1.3.0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(APIStatusError) as caught:
            client.models.upload(onnx_mlp(), name="From a file", feature_names=["a", "b", "c"], task="regression")
    assert caught.value.message == "Structure spec needs a layers array."


def test_with_via_auto_an_older_api_gets_the_model_as_earlier_clients_uploaded_it(client, api):
    pytest.importorskip("skl2onnx")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] - X[:, 2] > 0).astype(int)
    names = ["a", "b", "c", "d"]

    def create(request):
        spec = json.loads(request.content).get("structure") or {}
        if spec.get("family") == "graph":
            return refused("Structure spec needs a layers array.")
        return httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "drawn"})

    route = api.post("/v1/models").mock(side_effect=create)
    api_version(api, "1.1.0")
    api.get(url__regex=r".*/v1/models/mdl_1/structure.*").mock(return_value=httpx.Response(200, json={"runnable": False}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        made_model = client.models.upload(SVC(kernel="linear", probability=True).fit(X, y), name="Drawn", feature_names=names)
    assert made_model["id"] == "mdl_1" and route.call_count == 2
    second = json.loads(route.calls.last.request.content)["structure"]
    assert second.get("family") != "graph"
    assert any("does not run model graphs" in str(item.message) and "as clients before 0.12.0 did" in str(item.message) for item in caught)
    # One Kaimei's own readers refuse is refused as clients before 0.12.0 refused it, saying what the API lacks.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(StructureError, match=r"Kaimei API 1\.1\.0 does not run model graphs.*own readers refuse it"):
            client.models.upload(KNeighborsClassifier(3).fit(X, y), name="Neighbours", feature_names=names)


def test_upload_structure_refused_by_an_older_api_says_which_api_it_needs(client, api):
    pytest.importorskip("onnx")
    api.put("/v1/models/m1/structure").mock(return_value=refused("Structure spec needs a layers array."))
    api_version(api, "1.2.0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(KaimeiError, match=r"Kaimei API 1\.2\.0 cannot take this model: .*needs API 1\.3\.0"):
            client.models.upload_structure("m1", onnx_mlp(), feature_names=["a", "b", "c"], task="regression")


# --- what the API takes that the calls did not send -------------------------------------------


def test_generate_sends_seeds_and_draw_only_when_given(client, api):
    route = api.post("/v1/models/painter/generate").mock(
        return_value=httpx.Response(200, json={"object": "model.generation", "model_id": "m", "variations": []})
    )
    client.models.generate("painter", input={"label": 3})
    assert json.loads(route.calls.last.request.content) == {"input": {"label": 3}, "temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0}
    client.models.generate("painter", seeds=(4, np.int64(9)), draw=["latent"])
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["seeds"] == [4, 9] and sent_body["draw"] == ["latent"]
    for wrong in ([], list(range(17)), "12"):
        with pytest.raises(ValueError, match="seeds"):
            client.models.generate("painter", seeds=wrong)
    with pytest.raises(ValueError, match="draw is a list"):
        client.models.generate("painter", draw="latent")
    assert route.call_count == 2  # nothing was sent for the ones refused here


def test_activations_sends_channel_start_only_when_given(client, api):
    route = api.post("/v1/models/encoder/structure/activations").mock(
        return_value=httpx.Response(200, json={"object": "model.activations", "stages": []})
    )
    client.models.activations("encoder", input={"review": "fine"}, channels=4)
    assert "channel_start" not in json.loads(route.calls.last.request.content)
    client.models.activations("encoder", input={"review": "fine"}, nodes=["attn.weights"], channels=4, channel_start=8)
    assert json.loads(route.calls.last.request.content)["channel_start"] == 8
    client.models.activations("encoder", input={"review": "fine"}, channel_start=0)  # the first page is a page too
    assert json.loads(route.calls.last.request.content)["channel_start"] == 0
    for wrong in (-1, True, 1_000_001):
        with pytest.raises(ValueError, match="channel_start"):
            client.models.activations("encoder", input={"review": "fine"}, channel_start=wrong)
    assert route.call_count == 3


# --- what a model's own output is, said to the API ---------------------------------------------
#
# Without a structure, the API reads an agent's raw output as scores and puts it through a softmax (or a
# sigmoid, for one number), unless the answer names its ``link``: "none" is read as it is, "softmax" through
# a softmax. A model that ends in a softmax must say "none", or its probabilities are squashed a second time.


def module_answer(module, task="multiclass_classification", width=3, kind="predict"):
    features = [{"name": f"f{i}", "type": "numeric"} for i in range(width)]
    agent = Agent(OnKaimei(features, task), "m", module=module)
    rows = [{f"f{i}": float(i + j) / 3 - 0.5 for i in range(width)} for j in range(4)]
    if kind == "predict":
        return agent.run({"kind": "predict", "payload": {"rows": rows}})
    return agent.run({"kind": kind, "payload": {"input": rows[0]}})


def test_a_module_that_ends_in_a_softmax_or_a_sigmoid_says_its_output_is_probabilities():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(0)

    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 4)

        def forward(self, x):
            return torch.softmax(self.fc(x), dim=-1)

    answer = module_answer(nn.Sequential(nn.Linear(3, 4), nn.Softmax(dim=1)))
    assert answer["raw"] is True and answer["link"] == "none"
    np.testing.assert_allclose(np.sum(answer["predictions"], axis=1), 1.0, atol=1e-6)
    assert module_answer(Functional())["link"] == "none"
    binary = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid(), nn.Flatten())
    assert module_answer(binary, "binary_classification")["link"] == "none"
    # Log-probabilities: a softmax of them is the probabilities exactly, so that is what the API is told to apply.
    assert module_answer(nn.Sequential(nn.Linear(3, 4), nn.LogSoftmax(dim=1)))["link"] == "softmax"
    # Scores are what the API reads an output as anyway: nothing is said.
    assert "link" not in module_answer(nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4)))
    # A trace says it beside its prediction too.
    traced = module_answer(nn.Sequential(nn.Linear(3, 4), nn.Softmax(dim=1)), kind="trace")
    assert traced["raw"] is True and traced["link"] == "none"


def test_a_module_torch_fx_cannot_read_is_told_by_its_last_module_or_its_numbers():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(0)

    class Branching(nn.Module):
        """A forward that branches on values: torch.fx cannot trace it."""

        def __init__(self, last):
            super().__init__()
            self.fc = nn.Linear(3, 4)
            self.last = last

        def forward(self, x):
            hidden = self.fc(x)
            if float(hidden.sum()) > 1e9:
                hidden = hidden * 0
            return self.last(hidden)

    assert agents._traced_ends(Branching(nn.Softmax(dim=1))) is None
    assert module_answer(Branching(nn.Softmax(dim=1)))["link"] == "none"  # the softmax module gave the output
    assert module_answer(Branching(lambda h: torch.softmax(h, dim=1)))["link"] == "none"  # rows of probabilities
    assert module_answer(Branching(lambda h: torch.log_softmax(h, dim=1)))["link"] == "softmax"
    assert "link" not in module_answer(Branching(nn.Identity()))


def test_predict_with_says_what_an_estimator_answers_with(client):
    torch = pytest.importorskip("torch")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    row = {"kind": "predict", "payload": {"rows": [{"a": 1.0, "b": 0.0}]}}
    probabilities = agents.predict_with(LogisticRegression().fit(X, y), ["a", "b"])
    answer = Agent(client, "churn", probabilities).run(row)
    assert probabilities.link == "none" and probabilities.raw is False
    assert answer["link"] == "none" and "raw" not in answer  # taken as they are, and said to be probabilities
    margins = agents.predict_with(LinearSVC().fit(X, y), ["a", "b"])
    answer = Agent(client, "churn", margins).run(row)
    assert margins.raw is True and answer["raw"] is True and "link" not in answer  # scores
    module = agents.predict_with(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Softmax(dim=1)), ["a", "b"])
    answer = Agent(client, "churn", module).run(row)
    assert module.link == "none" and answer["raw"] is True and answer["link"] == "none"


def onnx_classifier(*ops):
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(2)
    nodes, previous = [helper.make_node("Gemm", ["X", "W", "B"], ["scores"])], "scores"
    for at, (op, attributes) in enumerate(ops):
        out = f"t{at}"
        extra = ["shape"] if op == "Reshape" else []
        nodes.append(helper.make_node(op, [previous, *extra], [out], **attributes))
        previous = out
    constants = [
        numpy_helper.from_array(rng.normal(size=(3, 4)).astype(np.float32), "W"),
        numpy_helper.from_array(np.zeros(4, np.float32), "B"),
        numpy_helper.from_array(np.array([1, 4], dtype=np.int64), "shape"),
    ]
    graph = helper.make_graph(
        nodes,
        "classifier",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, 4])],
        constants,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def file_answer(model):
    features = [{"name": name, "type": "numeric"} for name in ("a", "b", "c")]
    agent = Agent(OnKaimei(features, "multiclass_classification"), "m", onnx=model)
    rows = [{"a": 0.5, "b": -1.0, "c": 2.0}, {"a": 1.0, "b": 0.0, "c": 0.0}]
    return agent.run({"kind": "predict", "payload": {"rows": rows}})


def test_a_file_whose_prediction_comes_from_a_softmax_says_its_output_is_probabilities():
    pytest.importorskip("onnxruntime")
    assert file_answer(onnx_classifier(("Softmax", {"axis": 1})))["link"] == "none"
    assert file_answer(onnx_classifier(("Softmax", {"axis": 1}), ("Reshape", {})))["link"] == "none"
    assert file_answer(onnx_classifier(("LogSoftmax", {"axis": 1})))["link"] == "softmax"
    assert "link" not in file_answer(onnx_classifier())  # a Gemm's scores
    assert "link" not in file_answer(onnx_classifier(("Relu", {})))


def test_a_converted_scikit_learn_classifier_says_its_output_is_probabilities():
    pytest.importorskip("onnxruntime")
    skl2onnx = pytest.importorskip("skl2onnx")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    rows = [{name: float(X[j, i]) for i, name in enumerate("abc")} for j in range(3)]
    features = [{"name": name, "type": "numeric"} for name in "abc"]
    zipped = skl2onnx.to_onnx(LogisticRegression().fit(X, y), X[:1])  # its probabilities behind a ZipMap
    agent = Agent(OnKaimei(features, "binary_classification"), "m", onnx=zipped)
    assert agent.run({"kind": "predict", "payload": {"rows": rows}})["link"] == "none"
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    plain = skl2onnx.to_onnx(forest, X[:1], options={id(forest): {"zipmap": False}})
    agent = Agent(OnKaimei(features, "binary_classification"), "m", onnx=plain)
    assert agent._file.ends(agent._output_tensor()) is None  # the file does not say: its numbers do
    answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
    np.testing.assert_allclose(answer["predictions"], forest.predict_proba(X[:3]), atol=1e-6)
    assert answer["link"] == "none"


# --- a summary sends no learned number, however few ------------------------------------------


def sent_floats(spec):
    """Every float a spec carries as a number, in its constants and its attributes, nested graphs included."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            if "data" in value and str(value.get("dtype", "")).startswith("float"):
                found.extend(value["data"])
            if "$b64" in value and str(value.get("dtype", "")).startswith("f"):
                found.append(value["$b64"])
            for key, item in value.items():
                if key not in ("stats", "check", "inputs", "outputs", "features", "head", "metadata", "data"):
                    walk(item)
        elif isinstance(value, list):
            if value and all(isinstance(item, float) for item in value):
                found.extend(value)
            for item in value:
                walk(item)

    walk(spec["graph"])
    return found


def test_a_summary_of_a_small_model_sends_none_of_its_learned_numbers():
    pytest.importorskip("skl2onnx")
    torch = pytest.importorskip("torch")
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    from kaimei.structure import extract_structure

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)) * [1, 5, 10, 50] + [0, 100, -3, 7]
    y = (X[:, 0] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bayes = extract_structure(GaussianNB().fit(X, y), feature_names=list("abcd"), weights="summary").spec
        # Converted, a linear model keeps its four coefficients in a LinearClassifier's attributes (ai.onnx.ml).
        linear = extract_structure(
            LogisticRegression().fit(X, y), feature_names=list("abcd"), weights="summary", via="onnx"
        ).spec

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.GRU(2, 2, batch_first=True)
                self.head = torch.nn.Linear(2, 3)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1])

        torch.manual_seed(0)
        net = Tiny().eval()
        tiny = extract_structure(net, task="multiclass_classification", sample=torch.zeros(1, 5, 2), weights="summary").spec
    fitted_bayes = GaussianNB().fit(X, y)
    fitted_linear = LogisticRegression().fit(X, y)
    learned = {
        "bayes": [*fitted_bayes.theta_.ravel(), *fitted_bayes.var_.ravel()],
        "linear": [*fitted_linear.coef_.ravel(), *fitted_linear.intercept_.ravel()],
        "tiny": [float(value) for parameter in net.parameters() for value in parameter.detach().numpy().ravel()],
    }
    for name, spec in (("bayes", bayes), ("linear", linear), ("tiny", tiny)):
        assert spec["family"] == "graph" and spec["weights"] == "summary"
        sent = {round(float(value), 4) for value in sent_floats(spec) if isinstance(value, float)}
        assert not sent & {round(float(value), 4) for value in learned[name]}, name  # none of them, however few
        assert not [value for value in sent_floats(spec) if isinstance(value, str)], name  # nor any as bytes
    assert sent_floats(bayes) == [] and sent_floats(linear) == []
    assert sent_floats(tiny) == [0.0]  # what a ConstantOfShape fills a GRU's first state with: a setting
    # What the graph's shapes are worked out from stays: the whole numbers of its reshapes and axes.
    kept = [value for value in bayes["graph"]["constants"].values() if isinstance(value, dict) and "data" in value]
    assert kept and all(value["dtype"].startswith("int") for value in kept)


def test_a_summary_keeps_shapes_axes_scales_and_settings_and_nothing_learned():
    from kaimei.graphs import finalize_graph

    spec = {
        "format": "kaimei.structure.v1",
        "family": "graph",
        "task": "regression",
        "feature_names": ["a", "b"],
        "graph": {
            "inputs": [{"name": "x", "dtype": "float32", "shape": [1, 2], "from": {"features": ["a", "b"]}}],
            "constants": {
                "w": np.array([[0.5], [1.5]], dtype=np.float32),
                "bias": np.array(0.25, dtype=np.float32),  # one learned number is still a learned number
                "shape": np.array([1, -1], dtype=np.int64),
                "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
                "roi": np.zeros(0, dtype=np.float32),
            },
            "nodes": [
                {"id": "fc", "op": "Gemm", "domain": "", "inputs": ["x", "w", "bias"], "outputs": ["y"], "attrs": {"alpha": 1.0}},
                {"id": "up", "op": "Resize", "domain": "", "inputs": ["y", "roi", "scales"], "outputs": ["z"], "attrs": {"mode": "nearest"}},
                {"id": "flat", "op": "Reshape", "domain": "", "inputs": ["z", "shape"], "outputs": ["f"], "attrs": {}},
                {
                    "id": "lin",
                    "op": "LinearRegressor",
                    "domain": "ai.onnx.ml",
                    "inputs": ["x"],
                    "outputs": ["r"],
                    "attrs": {"coefficients": [0.5, -1.25], "intercepts": [3.0], "targets": 1, "post_transform": "NONE"},
                },
            ],
            "outputs": ["f"],
        },
    }
    done = finalize_graph(spec, weights="summary").spec
    constants = done["graph"]["constants"]
    assert constants["w"] == {"$shape": [2, 1], "dtype": "f32"}
    assert constants["bias"] == {"$shape": [], "dtype": "f32"}
    assert constants["shape"] == {"dtype": "int64", "shape": [2], "data": [1, -1]}
    assert constants["scales"] == {"dtype": "float32", "shape": [4], "data": [1.0, 1.0, 2.0, 2.0]}  # sets a size
    assert constants["roi"]["shape"] == [0]
    nodes = {node["id"]: node for node in done["graph"]["nodes"]}
    assert nodes["fc"]["attrs"] == {"alpha": 1.0}  # a setting of the operator
    learned = nodes["lin"]["attrs"]
    assert learned["coefficients"] == {"$shape": [2], "dtype": "float32"} and learned["intercepts"] == {"$shape": [1], "dtype": "float32"}
    assert learned["targets"] == 1 and learned["post_transform"] == "NONE"
    assert done["approximate"] is True and sent_floats(done) == [1.0, 1.0, 2.0, 2.0]
    # Sent in full, everything goes as before.
    full = finalize_graph(spec).spec["graph"]
    assert full["constants"]["bias"]["data"] == [0.25] and full["nodes"][3]["attrs"]["coefficients"] == [0.5, -1.25]


# --- a file ONNX Runtime would stop the process on --------------------------------------------

_DFT_RUN = r"""
import sys, warnings
import numpy as np
from onnx import TensorProto, helper
from kaimei import graphs
from kaimei.agent import Agent
from kaimei.errors import NotFoundError

warnings.simplefilter("ignore")
node = helper.make_node("DFT", ["x", "", ""], ["y"], name="dft")  # its last inputs left out under empty names
graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
model.ir_version = 9
spec = graphs.from_onnx(model, task="generation")
print("read", spec["metadata"].get("checked_against"), flush=True)


class OnKaimei:
    def _request(self, method, path, **kwargs):
        if path.endswith("/structure"):
            raise NotFoundError("none", status_code=404)
        return {"features": [{"name": "s", "type": "tensor", "shape": [1, 8, 1]}], "task": "generation"}


answer = Agent(OnKaimei(), "m", onnx=model).run({"kind": "predict", "payload": {"rows": [{"s": np.ones((1, 8, 1), np.float32)}]}})
print("answered", len(answer["predictions"][0]), list(model.graph.node[0].input), flush=True)
"""


def test_a_node_with_its_last_inputs_left_out_under_empty_names_does_not_stop_the_process(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import os
    import subprocess

    script = tmp_path / "dft.py"
    script.write_text(_DFT_RUN, encoding="utf-8")
    import kaimei

    source = os.path.dirname(os.path.dirname(os.path.abspath(kaimei.__file__)))  # the kaimei these tests import
    env = {**os.environ, "PYTHONPATH": source, "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr[-2000:])  # was killed by ONNX Runtime
    assert "read onnxruntime" in done.stdout
    assert "answered 16 ['x', '', '']" in done.stdout  # the caller's model is left as it was


# --- what the agent leaves on a module: no hooks, and each part's own mode ---------------------


def frozen_net(torch):
    class Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            self.head = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.body(x))

    model = Frozen()
    model.train()
    model.body.eval()  # a frozen backbone, the usual way to fine-tune
    return model


TRACED = {
    "scene": {
        "nodes": [
            {"id": "body.0", "op": "dense", "path": "body.0", "units": 4, "flow_index": 0, "args": {"shape": [4]}},
            {"id": "head", "op": "dense", "path": "head", "units": 2, "flow_index": 1, "args": {"shape": [2]}},
        ]
    },
    "head": {"vector_node": "head"},
}
ROW = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}


def test_a_trace_whose_row_cannot_be_prepared_leaves_no_hook_on_the_module():
    torch = pytest.importorskip("torch")
    net = frozen_net(torch)
    agent = Agent(OnKaimei([{"name": name, "type": "numeric"} for name in "abcd"], document=TRACED), "m", module=net)

    def hooks():
        return sum(len(layer._forward_hooks) for layer in net.modules())

    for _ in range(3):
        with pytest.raises(KeyError):
            agent.run({"kind": "trace", "payload": {"input": {"a": 1.0, "b": 2.0, "c": 3.0}}})  # no 'd'
        assert hooks() == 0
    assert len(agent.run({"kind": "trace", "payload": {"input": ROW}})["frames"]) == 2 and hooks() == 0


def test_running_a_module_leaves_each_of_its_parts_in_its_own_mode():
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from kaimei import graphs

    def modes(model):
        return {name: layer.training for name, layer in model.named_modules()}

    net = frozen_net(torch)
    before = modes(net)
    assert before["body"] is False and before["head"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert graphs.via_onnx(net, feature_names=list("abcd")) is not None
    assert modes(net) == before  # the frozen backbone is not put back into training
    agents.predict_with(net, list("abcd"))([ROW, ROW])
    assert modes(net) == before
    agent = Agent(OnKaimei([{"name": name, "type": "numeric"} for name in "abcd"], document=TRACED), "m", module=net)
    agent.run({"kind": "predict", "payload": {"rows": [ROW, ROW]}})
    agent.run({"kind": "trace", "payload": {"input": ROW}})
    assert modes(net) == before


# --- an input the upload gave a fixed value ----------------------------------------------------


def test_an_agent_given_the_uploaded_structure_feeds_an_input_its_fixed_value():
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    from kaimei import graphs
    from kaimei.structure import finalize_spec

    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "scale"], ["scaled"]), helper.make_node("MatMul", ["scaled", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4]), helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])],
        [numpy_helper.from_array(np.ones((4, 2), dtype=np.float32), "w")],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    proto.ir_version = 9
    features = [{"name": name, "type": "numeric"} for name in "abcd"]
    fixed = np.array([[2.0, 0.5, -1.0, 3.0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = graphs.from_onnx(proto, task="regression", feature_names=list("abcd"), inputs={"scale": {"value": fixed}})
    extraction = finalize_spec(raw)
    written = extraction.spec["graph"]["inputs"][1]["from"]["value"]
    assert written == {"dtype": "float32", "shape": [1, 4], "data": [2.0, 0.5, -1.0, 3.0]}  # as the upload writes it
    packed = json.loads(json.dumps(extraction.spec))
    packed["graph"]["inputs"][1]["from"]["value"] = graphs._packed(fixed, np)  # and as it writes one that is not finite
    for structure in (extraction, packed):
        agent = Agent(OnKaimei(features), "m", onnx=proto, structure=structure)
        answer = agent.run({"kind": "predict", "payload": {"rows": [{"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}]}})
        expected = (np.array([[1.0, 2.0, 3.0, 4.0]]) * fixed) @ np.ones((4, 2))
        np.testing.assert_allclose(answer["predictions"], expected)


# --- predict_with and the order of a fitted estimator's columns --------------------------------


def test_predict_with_puts_each_column_where_the_estimator_was_fitted_to_find_it():
    pd = pytest.importorskip("pandas")
    from sklearn.linear_model import LinearRegression

    table = pd.DataFrame({"tenure": [1.0, 2.0, 3.0, 4.0], "seats": [1.0, 3.0, 2.0, 5.0]})
    model = LinearRegression().fit(table, 100 * table["tenure"] + table["seats"])
    row = {"tenure": 2.0, "seats": 7.0}
    for names in (None, ["tenure", "seats"], ["seats", "tenure"]):
        np.testing.assert_allclose(agents.predict_with(model, names)([row]), [207.0])
    # Other names are the fitted columns in order, as before; a mix of the two cannot be told apart.
    np.testing.assert_allclose(agents.predict_with(model, ["t", "s"])([{"t": 2.0, "s": 7.0}]), [207.0])
    with pytest.raises(ValueError, match="names some of the columns"):
        agents.predict_with(model, ["seats", "plan"])
