"""The agent: the loop that asks Kaimei for work, and the answers it gives in the API's own forms."""

from __future__ import annotations

import base64
import json
import threading

import httpx
import pytest
import respx

np = pytest.importorskip("numpy")

from kaimei import AuthenticationError, Kaimei, NotFoundError  # noqa: E402
from kaimei.activations import decode  # noqa: E402
from kaimei.agent import (  # noqa: E402
    KINDS,
    Agent,
    _load,
    bind_inputs,
    decode_row,
    default_detail,
    encoder_of,
    hook_plan,
    main,
    predict_with,
    serve,
    shown_values,
    stage_of,
)

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


class NoWaiting(threading.Event):
    """A stop event that does not sleep, and remembers how long it was asked to."""

    def __init__(self) -> None:
        super().__init__()
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.is_set()


def working(api, jobs, stop, session="s1"):
    """An API that hands out ``jobs`` one poll at a time, then asks the agent to stop."""
    queue = list(jobs)
    opened = api.post(f"{MODEL}/agent/sessions").mock(
        side_effect=lambda request: httpx.Response(201, json={"object": "model.agent.session", "id": session, "model_id": "mdl_1", "capabilities": json.loads(request.content)["capabilities"]})
    )

    def poll(request):
        if queue:
            return httpx.Response(200, json={"object": "model.agent.work", "job": queue.pop(0)})
        stop.set()
        return httpx.Response(200, json={"object": "model.agent.work", "job": None})

    polled = api.post(f"{MODEL}/agent/sessions/{session}/poll").mock(side_effect=poll)
    answers = api.post(url__regex=rf".*{MODEL}/agent/jobs/[^/]+/result").mock(return_value=httpx.Response(200, json={"job": None}))
    closed = api.delete(f"{MODEL}/agent/sessions/{session}").mock(return_value=httpx.Response(200, json={"deleted": True}))
    return opened, polled, answers, closed


def sent(answers):
    return [(call.request.url.path.split("/")[-2], json.loads(call.request.content)) for call in answers.calls]


def test_a_row_arrives_with_its_pictures_series_and_tensors_packed_and_is_unpacked():
    photo = (np.arange(12, dtype=np.float32).reshape(2, 2, 3) * 20).astype("<f4")
    signal = np.array([[1.5, -2.0], [3.0, 4.25], [5.0, 6.0]], dtype="<f4")
    row = decode_row(
        {
            "review": [7, 2, 1, 6],
            "photo": {"$b64": base64.b64encode(photo.tobytes()).decode(), "dtype": "f32", "shape": [2, 2, 3]},
            "signal": {"$b64": base64.b64encode(signal.tobytes()).decode(), "dtype": "f32", "shape": [3, 2]},
            "age": 41,
            "plan": "team",
            "member": True,
            "gap": None,
        }
    )
    assert row["review"] == [7, 2, 1, 6] and row["age"] == 41 and row["plan"] == "team" and row["member"] is True
    assert row["gap"] is None
    assert row["photo"].dtype == np.float32 and row["photo"].shape == (2, 2, 3)
    np.testing.assert_array_equal(row["photo"], photo)
    np.testing.assert_array_equal(row["signal"], signal)


def test_the_loop_opens_a_session_answers_each_job_and_closes(client, api):
    stop = threading.Event()
    jobs = [
        {"id": "job_1", "kind": "predict", "payload": {"rows": [{"x": 2}, {"x": 6}]}},
        {"id": "job_2", "kind": "predict", "payload": {"rows": [{"x": 4}]}},
    ]
    opened, polled, answers, closed = working(api, jobs, stop)
    seen = []

    def predict(rows):
        seen.append(rows)
        return np.array([[1 - row["x"] / 8, row["x"] / 8] for row in rows])

    lines = []
    agent = serve(client, "churn", predict, label="test agent", stop=stop, wait=3, log=lines.append)
    assert json.loads(opened.calls.last.request.content)["capabilities"] == ["predict"]
    assert json.loads(opened.calls.last.request.content)["label"] == "test agent"
    assert json.loads(opened.calls.last.request.content)["info"]["client"].startswith("kaimei-python ")
    assert json.loads(polled.calls.last.request.content) == {"wait": 3.0}
    assert seen == [[{"x": 2}, {"x": 6}], [{"x": 4}]]
    assert sent(answers) == [
        ("job_1", {"ok": True, "result": {"predictions": [[0.75, 0.25], [0.25, 0.75]]}}),
        ("job_2", {"ok": True, "result": {"predictions": [[0.5, 0.5]]}}),
    ]
    assert closed.called and agent.session_id is None and agent.answered == 2
    assert lines[0].startswith("Connected for churn") and "predict answered" in lines[1]


def test_values_for_each_row_are_sent_as_one_number_a_row(client, api):
    stop = threading.Event()
    opened, polled, answers, closed = working(api, [{"id": "j", "kind": "predict", "payload": {"rows": [{"x": 2}, {"x": 3}]}}], stop)
    serve(client, "churn", lambda rows: [row["x"] * 1.5 for row in rows], stop=stop)
    assert sent(answers) == [("j", {"ok": True, "result": {"predictions": [[3.0], [4.5]]}})]


def test_a_job_that_fails_is_answered_with_its_error_and_the_loop_goes_on(client, api):
    stop = threading.Event()
    jobs = [
        {"id": "boom", "kind": "predict", "payload": {"rows": [{"x": 1}]}},
        {"id": "short", "kind": "predict", "payload": {"rows": [{"x": 2}, {"x": 3}]}},
        {"id": "nan", "kind": "predict", "payload": {"rows": [{"x": 4}]}},
        {"id": "trace", "kind": "trace", "payload": {"input": {"x": 1}}},
        {"id": "fine", "kind": "predict", "payload": {"rows": [{"x": 5}]}},
    ]
    opened, polled, answers, closed = working(api, jobs, stop)

    def predict(rows):
        if rows[0]["x"] == 1:
            raise RuntimeError("the model fell over")
        if rows[0]["x"] == 2:
            return [0.5]
        if rows[0]["x"] == 4:
            return [float("nan")]
        return [0.25]

    serve(client, "churn", predict, stop=stop)
    answered = dict(sent(answers))
    assert answered["boom"] == {"ok": False, "error": "RuntimeError: the model fell over"}
    assert answered["short"]["ok"] is False and "1 predictions for 2 rows" in answered["short"]["error"]
    assert answered["nan"]["ok"] is False and "not a finite number" in answered["nan"]["error"]
    assert answered["trace"]["ok"] is False and "does not answer 'trace'" in answered["trace"]["error"]
    assert answered["fine"] == {"ok": True, "result": {"predictions": [[0.25]]}}
    assert closed.called


def test_a_job_that_is_gone_when_its_answer_arrives_does_not_stop_the_agent(client, api):
    stop = threading.Event()
    jobs = [{"id": f"job_{index}", "kind": "predict", "payload": {"rows": [{"x": index}]}} for index in range(3)]
    opened, polled, answers, closed = working(api, jobs, stop)
    gone = {
        "job_0": httpx.Response(404, json={"error": {"type": "not_found_error", "code": "agent_job_not_found", "message": "No such job for this model."}}),
        "job_1": httpx.Response(409, json={"error": {"type": "invalid_request_error", "code": "agent_job_not_running", "message": "This job is not waiting for an answer."}}),
    }
    answers.mock(side_effect=lambda request: gone.get(request.url.path.split("/")[-2], httpx.Response(200, json={"job": None})))
    lines = []
    agent = serve(client, "churn", lambda rows: [1.0] * len(rows), stop=stop, log=lines.append)
    assert [name for name, _body in sent(answers)] == ["job_0", "job_1", "job_2"]  # each sent once, none again
    assert agent.answered == 3 and closed.called
    assert any("not taken (No such job" in line for line in lines) and any("not taken (This job is not waiting" in line for line in lines)


def test_an_answer_the_api_refuses_becomes_a_failure_the_caller_is_told_of(client, api):
    stop = threading.Event()
    opened, polled, answers, closed = working(api, [{"id": "big", "kind": "predict", "payload": {"rows": [{"x": 1}]}}], stop)
    replies = [
        httpx.Response(400, json={"error": {"code": "agent_answer_too_large", "message": "This answer is larger than an agent's answer may be."}}),
        httpx.Response(200, json={"job": None}),
    ]
    answers.mock(side_effect=lambda request: replies.pop(0))
    serve(client, "churn", lambda rows: [1.0], stop=stop)
    bodies = [body for _name, body in sent(answers)]
    assert bodies[0]["ok"] is True
    assert bodies[1] == {"ok": False, "error": "The agent's answer was refused: This answer is larger than an agent's answer may be."}


def test_a_session_the_api_no_longer_knows_is_opened_again(client, api):
    stop = threading.Event()
    opened, polled, answers, closed = working(api, [], stop)
    replies = [
        httpx.Response(404, json={"error": {"type": "not_found_error", "code": "agent_session_not_found", "message": "No such agent session for this model."}}),
        httpx.Response(200, json={"job": {"id": "after", "kind": "predict", "payload": {"rows": [{"x": 1}]}}}),
    ]

    def poll(request):
        if replies:
            return replies.pop(0)
        stop.set()
        return httpx.Response(200, json={"job": None})

    polled.mock(side_effect=poll)
    serve(client, "churn", lambda rows: [0.5], stop=stop)
    assert opened.call_count == 2 and [name for name, _ in sent(answers)] == ["after"]


def test_when_kaimei_cannot_be_reached_the_agent_waits_longer_each_time_and_carries_on(client, api):
    stop = NoWaiting()
    opened, polled, answers, closed = working(api, [], stop)
    replies = [
        httpx.Response(503, json={"error": {"message": "unavailable"}}),
        httpx.ConnectError("refused"),
        httpx.Response(502, json={"error": {"message": "bad gateway"}}),
        httpx.Response(200, json={"job": {"id": "later", "kind": "predict", "payload": {"rows": [{"x": 1}]}}}),
    ]

    def poll(request):
        if replies:
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        stop.set()
        return httpx.Response(200, json={"job": None})

    polled.mock(side_effect=poll)
    lines = []
    serve(client, "churn", lambda rows: [0.5], stop=stop, log=lines.append)
    assert stop.waits == [1.0, 2.0, 4.0]
    assert [name for name, _ in sent(answers)] == ["later"]
    assert sum("could not be reached" in line for line in lines) == 3 and opened.call_count == 1


def test_a_rejected_key_or_a_model_that_is_gone_ends_the_agent_with_that_error(client, api):
    api.post(f"{MODEL}/agent/sessions").mock(
        return_value=httpx.Response(401, json={"error": {"type": "authentication_error", "message": "Invalid API key."}})
    )
    with pytest.raises(AuthenticationError):
        serve(client, "churn", lambda rows: [0.5])
    api.post(f"{MODEL}/agent/sessions").mock(
        return_value=httpx.Response(404, json={"error": {"type": "not_found_error", "code": "model_not_found", "message": "No such model."}})
    )
    with pytest.raises(NotFoundError):
        serve(client, "churn", lambda rows: [0.5])


def test_stop_and_an_interrupt_both_close_the_session(client, api):
    stop = threading.Event()
    opened, polled, answers, closed = working(api, [], stop)
    agent = Agent(client, "churn", lambda rows: [0.5])

    def poll(request):
        agent.stop()
        return httpx.Response(200, json={"job": None})

    polled.mock(side_effect=poll)
    agent.serve()
    assert closed.call_count == 1 and agent.session_id is None
    agent.close()  # twice is fine
    assert closed.call_count == 1

    def interrupted(request):
        raise KeyboardInterrupt

    polled.mock(side_effect=interrupted)
    again = Agent(client, "churn", lambda rows: [0.5])
    again.serve()
    assert closed.call_count == 2


def test_what_an_agent_needs_and_what_it_says_it_answers(client):
    assert KINDS == ("predict", "trace", "activations", "tokenize")
    with pytest.raises(TypeError):
        Agent(client, "churn")
    with pytest.raises(TypeError):
        Agent(client, "churn", module=object())
    agent = Agent(client, "churn", lambda rows: [0.5], wait=90)
    assert agent.capabilities == ["predict"] and agent.wait == 25.0  # no tokenizer: it does not offer to split text
    with pytest.raises(ValueError):
        Agent(client, "churn", lambda rows: [0.5], capabilities=["explain"])
    with pytest.raises(ValueError):
        Agent(client, "churn", lambda rows: [0.5], capabilities=["trace"])
    assert Agent(client, "churn", lambda rows: [0.5], tokenizer=str.split).capabilities == ["predict", "tokenize"]
    assert Agent(client, "churn", tokenizer=str.split).capabilities == ["tokenize"]
    assert Agent(client, "churn", lambda rows: [0.5], tokenizer=str.split, capabilities=["predict"]).capabilities == ["predict"]
    with pytest.raises(TypeError):
        Agent(client, "churn", lambda rows: [0.5], tokenizer=7)


class Pieces:
    """A SentencePiece model as transformers wraps one: called with a text, it answers with the ids the model reads."""

    words = ["<pad>", "</s>", "<unk>", "the", "film", "was", "good"]

    def convert_ids_to_tokens(self, ids):
        return [self.words[index] for index in ids]

    def __call__(self, text, truncation=False, max_length=None):
        ids = [self.words.index(word) if word in self.words else 2 for word in text.lower().split()] + [1]
        return {"input_ids": ids[:max_length] if truncation else ids, "attention_mask": [1] * len(ids)}


def test_a_tokenizer_is_asked_in_whichever_way_it_answers():
    by_call = encoder_of(Pieces())
    assert by_call("The film was good", None) == [3, 4, 5, 6, 1]
    assert by_call("The film was good", 3) == [3, 4, 5]  # cut by the tokenizer itself, as it was set up to

    class Encoded:
        ids = np.array([5, 6, 7, 8])

    class Fast:
        def encode(self, text):
            return Encoded()

    assert encoder_of(Fast())("anything", None) == [5, 6, 7, 8] and encoder_of(Fast())("anything", 2) == [5, 6]
    assert all(type(item) is int for item in encoder_of(Fast())("anything", None))

    class Processor:
        def encode(self, text):
            return [len(word) for word in text.split()]

    assert encoder_of(Processor())("a film", None) == [1, 4]
    assert encoder_of(lambda text: [ord(char) for char in text])("ab", None) == [97, 98]
    assert encoder_of(lambda text: {"input_ids": [1, 2, 3]})("ab", 2) == [1, 2]
    with pytest.raises(TypeError):
        encoder_of(object())


def test_text_is_split_by_the_agent_for_a_tokenizer_kaimei_cannot_run(client, api):
    stop = threading.Event()
    jobs = [
        {"id": "t1", "kind": "tokenize", "payload": {"feature": "review", "text": "The film was good", "max_length": 12}},
        {"id": "t2", "kind": "tokenize", "payload": {"feature": "review", "text": "the film was zzz and long", "max_length": 4}},
        {"id": "t3", "kind": "tokenize", "payload": {"feature": "review", "text": 7, "max_length": None}},
        {"id": "p1", "kind": "predict", "payload": {"rows": [{"review": [3, 4, 5, 6, 1]}]}},
    ]
    opened, polled, answers, closed = working(api, jobs, stop)
    seen = []

    def predict(rows):
        seen.extend(rows)
        return [[0.2, 0.8] for _row in rows]

    serve(client, "churn", predict, tokenizer=Pieces(), stop=stop)
    assert json.loads(opened.calls.last.request.content)["capabilities"] == ["predict", "tokenize"]
    answered = dict(sent(answers))
    assert answered["t1"] == {"ok": True, "result": {"ids": [3, 4, 5, 6, 1]}}
    assert answered["t2"] == {"ok": True, "result": {"ids": [3, 4, 5, 2]}}
    assert answered["t3"]["ok"] is False and "the text to split" in answered["t3"]["error"]
    assert answered["p1"] == {"ok": True, "result": {"predictions": [[0.2, 0.8]]}}
    assert seen == [{"review": [3, 4, 5, 6, 1]}]  # the model is given the ids, as before


def test_a_tokenizer_for_each_text_feature_and_none_for_another(client):
    agent = Agent(client, "churn", lambda rows: [0.5], tokenizer={"question": lambda text: [1, 2], "answer": lambda text: [3]})
    assert agent.run({"kind": "tokenize", "payload": {"feature": "answer", "text": "x", "max_length": None}}) == {"ids": [3]}
    assert agent.run({"kind": "tokenize", "payload": {"feature": "question", "text": "x", "max_length": 1}}) == {"ids": [1]}
    with pytest.raises(ValueError, match="no tokenizer for the feature 'title'"):
        agent.run({"kind": "tokenize", "payload": {"feature": "title", "text": "x"}})
    plain = Agent(client, "churn", lambda rows: [0.5])
    with pytest.raises(ValueError, match="does not answer 'tokenize'"):
        plain.run({"kind": "tokenize", "payload": {"feature": "review", "text": "x"}})


def test_predict_with_asks_an_estimator_in_the_order_of_its_features():
    class Classifier:
        def predict_proba(self, table):
            self.seen = np.asarray(table)
            return np.column_stack([1 - self.seen[:, 0] / 10, self.seen[:, 0] / 10])

    class Regressor:
        feature_names_in_ = np.array(["b", "a"])

        def predict(self, table):
            self.seen = np.asarray(table)
            return self.seen[:, 0] * 2

    classifier = Classifier()
    rows = [{"b": 5, "a": 2}, {"b": 7, "a": 4}]
    np.testing.assert_allclose(predict_with(classifier, ["a", "b"])(rows), [[0.8, 0.2], [0.6, 0.4]])
    np.testing.assert_array_equal(classifier.seen, [[2, 5], [4, 7]])
    predict_with(classifier)(rows)
    np.testing.assert_array_equal(classifier.seen, [[5, 2], [7, 4]])  # the first row's own order
    regressor = Regressor()
    np.testing.assert_allclose(predict_with(regressor)([{"a": 2, "b": 5}]), [10.0])  # the order it was fitted on
    assert predict_with(classifier).raw is False and predict_with(regressor).raw is False
    with pytest.raises(TypeError):
        predict_with(object())


def test_a_classifier_without_probabilities_answers_with_its_scores_and_says_so():
    class Margins:
        classes_ = np.array(["no", "yes"])

        def decision_function(self, table):
            return np.asarray(table)[:, 0] - 1.0

        def predict(self, table):
            return np.where(self.decision_function(table) > 0, "yes", "no")

    asked = predict_with(Margins(), ["a"])
    np.testing.assert_allclose(asked([{"a": 3.0}, {"a": 0.5}]), [2.0, -0.5])
    assert asked.raw is True  # scores, which Kaimei squashes the way the model's structure does


def test_a_predict_that_gives_the_models_own_scores_is_passed_on_as_such(client):
    def scores(rows):
        return [[2.0, -1.0]] * len(rows)

    plain = Agent(client, "churn", scores)
    assert plain.run({"kind": "predict", "payload": {"rows": [{"a": 1}]}}) == {"predictions": [[2.0, -1.0]]}
    scores.raw = True
    assert plain.run({"kind": "predict", "payload": {"rows": [{"a": 1}]}}) == {"predictions": [[2.0, -1.0]], "raw": True}


# --- a stage, in the forms the API keeps ------------------------------------------------------


def picture_ok(packed, cells=None):
    """What the API checks of a picture before it keeps one."""
    count = int(np.prod(packed["shape"]))
    assert packed["encoding"] == "u8" and len(packed["range"]) == 2 and packed["range"][0] <= packed["range"][1]
    assert len(packed["data"]) == 4 * ((count + 2) // 3) and all(1 <= dim <= 4096 for dim in packed["shape"])
    assert len(base64.b64decode(packed["data"])) == count
    if cells is not None:
        assert count == cells


def test_a_vector_stage_reports_its_strongest_units_and_every_value():
    values = np.array([0.5, -3.0, 0.0, 2.0, np.nan, 1.0])
    brief = stage_of({"id": "fc", "args": {"shape": [6]}}, values, units=2)
    assert brief["node_id"] == "fc" and brief["detailed"] is False
    assert brief["top_units"] == [{"index": 1, "value": -3.0}, {"index": 3, "value": 2.0}]
    assert brief["summary"] == {"size": 6, "max_abs": 3.0, "mean_abs": pytest.approx(6.5 / 6), "l2": pytest.approx(np.sqrt(14.25)), "active": 4, "argmax": 1}
    assert brief["active_fraction"] == 0.5 and "values" not in brief and "shape" not in brief
    full = stage_of({"id": "fc", "args": {"shape": [6]}}, values, detailed=True)
    assert full["detailed"] is True
    picture_ok(full["values"], 6)
    np.testing.assert_allclose(decode(full["values"]), [0.5, -3.0, 0.0, 2.0, 0.0, 1.0], atol=5 / 255)
    wide = stage_of({"id": "wide"}, np.ones(5000), detailed=True)
    assert "values" not in wide and len(wide["top_units"]) == 16
    json.dumps(full, allow_nan=False)


def test_a_table_stage_says_how_strongly_each_position_fires():
    table = np.arange(12, dtype=float).reshape(4, 3) - 5
    stage = stage_of({"id": "embedded", "args": {"shape": [4, 3], "role": "attention"}}, table, detailed=True)
    assert stage["shape"] == [4, 3] and stage["role"] == "attention"
    np.testing.assert_allclose(stage["rows"], np.sqrt((table**2).mean(axis=1)), atol=1e-6)
    picture_ok(stage["matrix"], 12)
    np.testing.assert_allclose(decode(stage["matrix"]), table, atol=11 / 255)
    tall = stage_of({"id": "tall", "args": {"shape": [1000, 3]}}, np.ones(3000), detailed=True, max_side=8)
    assert tall["matrix"]["shape"] == [32, 3] and len(tall["rows"]) == 1000 and "values" in tall
    picture_ok(tall["matrix"])


def test_a_map_stage_sends_its_strongest_channels_and_a_saliency_map():
    rng = np.random.default_rng(0)
    maps = rng.normal(size=(6, 5, 4))
    maps[:, :, 2] *= 10
    brief = stage_of({"id": "conv", "args": {"shape": [6, 5, 4]}}, maps, units=2)
    assert brief["shape"] == [6, 5, 4] and brief["top_channels"][0] == 2 and len(brief["top_channels"]) == 2
    assert "maps" not in brief and "top_units" not in brief
    full = stage_of({"id": "conv", "args": {"shape": [6, 5, 4]}}, maps, detailed=True, channels=3, max_side=4)
    assert len(full["energy"]) == 4 and full["energy"][2] == pytest.approx(float(np.abs(maps[:, :, 2]).sum()), abs=1e-5)
    assert [item["channel"] for item in full["maps"]][0] == 2 and len(full["maps"]) == 3
    first = full["maps"][0]
    assert first["shape"] == [4, 3] and first["peak"] == pytest.approx(float(np.abs(maps[:, :, 2]).max()), abs=1e-5)
    picture_ok(first)
    picture_ok(full["saliency"], 12)
    whole = stage_of({"id": "conv", "args": {"shape": [6, 5, 4]}}, maps, detailed=True, max_side=56)
    np.testing.assert_allclose(decode(whole["maps"][0]), maps[:, :, 2], atol=np.ptp(maps[:, :, 2]) / 255)
    np.testing.assert_allclose(decode(whole["saliency"]), np.abs(maps).mean(axis=2), atol=np.ptp(np.abs(maps).mean(axis=2)) / 255)
    # One position is not a map: its channels are units.
    assert "top_units" in stage_of({"id": "pooled", "args": {"shape": [1, 1, 4]}}, np.ones(4))
    # Maps are sent strongest first, or in their own order from where a caller says.
    assert "channel_start" not in full and "focus" not in full and "focus" not in full["maps"][0]
    paged = stage_of(
        {"id": "conv", "args": {"shape": [6, 5, 4]}}, maps, detailed=True, channels=2, max_side=4, channel_start=1
    )
    assert [item["channel"] for item in paged["maps"]] == [1, 2]
    assert paged["channel_start"] == 1 and paged["channel_count"] == 4
    last = stage_of({"id": "conv", "args": {"shape": [6, 5, 4]}}, maps, detailed=True, channels=8, channel_start=3)
    assert [item["channel"] for item in last["maps"]] == [3]
    past = stage_of({"id": "conv", "args": {"shape": [6, 5, 4]}}, maps, detailed=True, channels=8, channel_start=9)
    assert past["maps"] == [] and past["channel_start"] == 4


def test_the_heads_of_an_attention_layer_are_ranked_by_how_sharply_they_look():
    """Every query's weights add up to one, so every head adds up to the same. What differs is their focus."""
    weights = np.zeros((3, 5, 2))
    weights[:, :, 0] = 0.2  # the first head looks at every key alike
    weights[:, :, 1] = [[0.9, 0.1, 0, 0, 0], [0, 0.8, 0.2, 0, 0], [0, 0, 0, 0.3, 0.7]]
    step = {"id": "attn", "args": {"shape": [3, 5, 2], "role": "attention"}}
    brief = stage_of(step, weights)
    assert brief["top_channels"] == [1, 0] and brief["role"] == "attention" and "focus" not in brief
    full = stage_of(step, weights, detailed=True, channels=2)
    assert full["energy"] == pytest.approx([3.0, 3.0]) and full["focus"] == pytest.approx([0.2, 0.8])
    assert [item["channel"] for item in full["maps"]] == [1, 0]
    assert full["maps"][0]["focus"] == pytest.approx(0.8) and full["maps"][1]["focus"] == pytest.approx(0.2)
    in_order = stage_of(step, weights, detailed=True, channels=64, channel_start=0)
    assert [item["channel"] for item in in_order["maps"]] == [0, 1] and in_order["channel_count"] == 2
    # The same numbers without the role are a feature map: ranked by what they hold, and no focus is claimed.
    plain = stage_of({"id": "maps", "args": {"shape": [3, 5, 2]}}, weights, detailed=True)
    assert "focus" not in plain and plain["top_channels"] == [0, 1]


# --- which steps a module's calls can speak for ------------------------------------------------


def test_a_module_speaks_for_a_step_only_where_the_step_is_the_module():
    steps = [
        {"id": "conv1", "path": "block.conv1"},
        {"id": "relu", "path": "block.relu"},  # one activation, used twice: a step for each call
        {"id": "conv2", "path": "block.conv2"},
        {"id": "add", "path": "block.add"},  # written in forward: no hook sees it
        {"id": "relu_2", "path": "block.relu"},
        {"id": "pool.GlobalAveragePool", "path": "pool.GlobalAveragePool"},
        {"id": "head.Flatten", "path": "head.Flatten"},
        {"id": "head.Gemm", "path": "head.Gemm"},
        {"id": "Softmax", "path": "Softmax"},
    ]
    own, under = hook_plan(steps, ["", "block", "block.conv1", "block.relu", "block.conv2", "pool", "head", "unused"])
    assert own == {"block.conv1": ["conv1"], "block.relu": ["relu", "relu_2"], "block.conv2": ["conv2"]}
    # Every step below a module's path counts, so that the last of them is the one that makes its output.
    assert under == {
        "block": [["conv1", "relu", "conv2", "add", "relu_2"]],
        "pool": [["pool.GlobalAveragePool"]],
        "head": [["head.Flatten", "head.Gemm"]],
    }


def test_a_modules_later_calls_are_told_apart_by_the_scope_the_exporter_gives_them():
    steps = [
        {"id": "a", "path": "block.relu.Relu"},
        {"id": "b", "path": "block.conv.Conv"},
        {"id": "c", "path": "block.relu_1.Relu"},
        {"id": "d", "path": "tail_2.Relu"},  # a third call with no second: nothing can be told of this module
        {"id": "e", "path": "relu_1.Relu"},  # a module that is itself called relu_1
    ]
    own, under = hook_plan(steps, ["block", "block.relu", "block.conv", "tail", "relu_1"])
    assert own == {}
    assert under == {"block": [["a", "b", "c"]], "block.relu": [["a"], ["c"]], "block.conv": [["b"]], "relu_1": [["e"]]}


# --- a torch module: predictions and a look inside ---------------------------------------------


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def module_api(api, features, nodes, task="multiclass_classification", labels=("low", "mid", "high"), head=None):
    api.get(MODEL).mock(return_value=httpx.Response(200, json={"object": "model", "id": "mdl_1", "task": task, "features": features, "outputs": {"labels": list(labels)}}))
    document = {"object": "model.structure", "scene": {"nodes": nodes}, "head": head or {}}
    return api.get(f"{MODEL}/structure").mock(return_value=httpx.Response(200, json=document))


def test_a_module_answers_predictions_traces_and_activations(client, api, torch):
    nn = torch.nn
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 5), nn.Tanh())
            self.head = nn.Linear(5, 3)

        def forward(self, x):
            return self.head(self.body(x)) * 2.0

    net = Net()
    features = [{"name": name, "type": "numeric"} for name in ("c", "a", "b")]
    nodes = [
        {"id": "c", "op": "input", "kind": "input", "units": 1, "feature": "c"},
        {"id": "a", "op": "input", "kind": "input", "units": 1, "feature": "a"},
        {"id": "body.0", "op": "dense", "kind": "linear", "units": 8, "path": "body.0", "args": {}},
        {"id": "body.0__relu", "op": "activation", "kind": "activation", "units": 8, "path": "body.1", "args": {}},
        {"id": "body.2.Gemm", "op": "Gemm", "kind": "linear", "units": 5, "path": "body.2.Gemm", "args": {"shape": [5]}},
        {"id": "reshaped", "op": "Reshape", "kind": "reshape", "units": 99, "path": "body.3.Reshape", "args": {}},
        {"id": "body.3", "op": "Tanh", "kind": "activation", "units": 5, "path": "body.3", "args": {}},
        {"id": "head.Gemm", "op": "Gemm", "kind": "linear", "units": 3, "path": "head.Gemm", "args": {}},
        {"id": "doubled", "op": "Mul", "kind": "combine", "units": 3, "path": "Mul", "args": {}},
        {"id": "body.0.weight::tensor", "op": "tensor", "kind": "weights", "units": 24, "path": "body.0"},
    ]
    structure = module_api(api, features, nodes, head={"vector_node": "doubled", "link": "softmax"})
    agent = Agent(client, "churn", module=net, pad_id=0)
    assert agent.capabilities == ["predict", "trace", "activations"]
    row = {"a": 0.5, "b": -1.0, "c": 2.0}
    given = torch.tensor([[2.0, 0.5, -1.0]])  # the model's own order of features: c, a, b
    net.eval()
    with torch.no_grad():
        first, fired = net.body[0](given), net.body[1](net.body[0](given))
        third = net.body[2](fired)
        scores = net.head(net.body(given)).numpy().reshape(-1)
        own = net(given).numpy().reshape(-1)
    predicted = agent.run({"kind": "predict", "payload": {"rows": [row, row]}})
    np.testing.assert_allclose(predicted["predictions"], [own, own], rtol=1e-6)
    assert predicted["raw"] is True  # the module's own scores: Kaimei reads them as the structure's head does
    traced = agent.run({"kind": "trace", "payload": {"input": row, "nodes": None}})
    frames = {frame["node_id"]: frame for frame in traced["frames"]}
    # The inputs are what the row held. A step is the module at its path, or the last step below a module's
    # path; one that has another size than the module's output, and one written in forward, are left out.
    assert list(frames) == ["c", "a", "body.0", "body.0__relu", "body.2.Gemm", "body.3", "head.Gemm"]
    assert frames["c"]["value"] == [2.0] and frames["a"]["value"] == [0.5]
    np.testing.assert_allclose(frames["body.0"]["value"], first.numpy().reshape(-1), rtol=1e-6)
    np.testing.assert_allclose(frames["body.0__relu"]["value"], fired.numpy().reshape(-1), rtol=1e-6)
    np.testing.assert_allclose(frames["body.2.Gemm"]["value"], third.numpy().reshape(-1), rtol=1e-6)
    np.testing.assert_allclose(frames["head.Gemm"]["value"], scores, rtol=1e-6)
    assert frames["body.3"]["summary"]["size"] == 5 and frames["body.3"]["sample"] == frames["body.3"]["value"]
    np.testing.assert_allclose(traced["prediction"], own, rtol=1e-6)
    assert traced["raw"] is True and "output" not in traced  # the API names the prediction itself
    only = agent.run({"kind": "trace", "payload": {"input": row, "nodes": ["body.3", "a"]}})
    assert [frame["node_id"] for frame in only["frames"]] == ["a", "body.3"]
    watched = agent.run({"kind": "activations", "payload": {"input": row, "nodes": ["body.3", "c"], "units": 3, "channels": 8, "max_side": 56}})
    stages = {stage["node_id"]: stage for stage in watched["stages"]}
    assert list(stages) == list(frames) and watched["forward_ms"] >= 0 and watched["raw"] is True
    assert stages["body.3"]["detailed"] is True and stages["body.0"]["detailed"] is False
    assert stages["c"]["detailed"] is False  # an input is never sent in full: it is what was sent
    assert len(stages["body.0"]["top_units"]) == 3 and "values" not in stages["body.0"]
    np.testing.assert_allclose(decode(stages["body.3"]["values"]), np.tanh(third.numpy().reshape(-1)), atol=2 / 255)
    chosen = agent.run({"kind": "activations", "payload": {"input": row, "nodes": None}})
    assert [stage["node_id"] for stage in chosen["stages"] if stage["detailed"]] == ["body.0", "body.0__relu", "body.2.Gemm", "body.3", "head.Gemm"]
    assert structure.call_count == 1  # the structure is read once
    assert dict(structure.calls.last.request.url.params) == {"detail": "summary", "max_inline": "0"}
    assert net.training is False  # the module is left in the mode it was found in
    net.train()
    agent.run({"kind": "trace", "payload": {"input": row}})
    assert net.training is True
    json.dumps(watched, allow_nan=False)
    # A function of your own answers predictions in its own terms: probabilities, and no flag.
    mine = Agent(client, "churn", lambda rows: [[0.2, 0.3, 0.5]] * len(rows), module=net)
    assert "raw" not in mine.run({"kind": "predict", "payload": {"rows": [row]}})
    looked = mine.run({"kind": "trace", "payload": {"input": row}})
    assert looked["prediction"] == [0.2, 0.3, 0.5] and "raw" not in looked


def residual(torch):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(4)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(4)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += x
            return self.relu(out)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 4, 1)
            self.block = Block()
            self.drop = nn.Dropout(0.5)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.drop(self.block(self.stem(x))).mean(dim=(2, 3)))

    torch.manual_seed(4)
    net = Net()
    with torch.no_grad():
        for norm in (net.block.bn1, net.block.bn2):
            norm.running_mean.uniform_(-0.5, 0.5)
            norm.running_var.uniform_(0.5, 2.0)
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.3, 0.3)
    return net.eval()


def by_hand(net, torch, picture):
    """What every step of the block computes, written out without its in-place arithmetic."""
    block = net.block
    with torch.no_grad():
        x = net.stem(torch.tensor(picture.transpose(2, 0, 1)[None]))
        conv1 = block.conv1(x)
        bn1 = block.bn1(conv1)
        relu1 = torch.relu(bn1)
        conv2 = block.conv2(relu1)
        bn2 = block.bn2(conv2)
        added = bn2 + x
        relu2 = torch.relu(added)
        scores = net.head(relu2.mean(dim=(2, 3)))
    last = {name: value.numpy()[0].transpose(1, 2, 0).reshape(-1) for name, value in {"stem": x, "conv1": conv1, "bn1": bn1, "relu1": relu1, "conv2": conv2, "bn2": bn2, "add": added, "relu2": relu2}.items()}
    return last, scores.numpy().reshape(-1)


def test_hooks_never_report_another_steps_values_as_kaimeis_own_reader_names_the_steps(client, api, torch):
    net = residual(torch)
    shown = {"shape": [5, 6, 4]}
    nodes = [
        {"id": "image", "op": "input", "kind": "input", "units": 90, "feature": "image", "args": {"shape": [5, 6, 3]}},
        {"id": "stem", "op": "conv2d", "units": 120, "path": "stem", "args": shown},
        {"id": "block.conv1", "op": "conv2d", "units": 120, "path": "block.conv1", "args": shown},
        {"id": "block.bn1", "op": "batchnorm", "units": 120, "path": "block.bn1", "args": shown},
        {"id": "block.relu", "op": "relu", "units": 120, "path": "block.relu", "args": shown},
        {"id": "block.conv2", "op": "conv2d", "units": 120, "path": "block.conv2", "args": shown},
        {"id": "block.bn2", "op": "batchnorm", "units": 120, "path": "block.bn2", "args": shown},
        {"id": "add", "op": "add", "units": 120, "path": "block.add", "args": shown},
        {"id": "block.relu_2", "op": "relu", "units": 120, "path": "block.relu", "args": shown},
        {"id": "mean", "op": "mean", "units": 4, "path": "mean", "args": {}},
        {"id": "head", "op": "dense", "units": 2, "path": "head", "args": {}},
        {"id": "softmax", "op": "softmax", "kind": "output", "units": 2, "path": "softmax", "args": {}},
    ]
    features = [{"name": "image", "type": "image", "height": 5, "width": 6, "channels": 3}]
    module_api(api, features, nodes, task="binary_classification", labels=("no", "yes"), head={"vector_node": "softmax"})
    picture = np.random.default_rng(2).uniform(0, 1, size=(5, 6, 3)).astype("<f4")
    packed = {"image": {"$b64": base64.b64encode(picture.tobytes()).decode(), "dtype": "f32", "shape": [5, 6, 3]}}
    want, scores = by_hand(net, torch, picture)
    agent = Agent(client, "churn", module=net)
    traced = agent.run({"kind": "trace", "payload": {"input": packed}})
    frames = {frame["node_id"]: frame for frame in traced["frames"]}
    # The residual add and the mean are written in forward: nothing is said of them. Kaimei's own softmax neither.
    assert list(frames) == ["image", "stem", "block.conv1", "block.bn1", "block.relu", "block.conv2", "block.bn2", "block.relu_2", "head"]
    for step, name in (("stem", "stem"), ("block.conv1", "conv1"), ("block.bn1", "bn1"), ("block.relu", "relu1"), ("block.conv2", "conv2"), ("block.bn2", "bn2"), ("block.relu_2", "relu2")):
        # In-place steps (the activation, the +=) came after: each value is the step's own, not what was written over it.
        np.testing.assert_allclose(frames[step]["value"], want[name][:48], rtol=1e-5, atol=1e-6, err_msg=step)
    np.testing.assert_allclose(frames["image"]["value"], picture.reshape(-1)[:48], rtol=1e-6)
    np.testing.assert_allclose(traced["prediction"], scores, rtol=1e-5)
    assert traced["raw"] is True
    # One step is missing for the activation's two calls: neither call can be told to be it, so neither is sent.
    module_api(api, features, [node for node in nodes if node["id"] != "block.relu_2"], task="binary_classification", labels=("no", "yes"))
    fewer = Agent(client, "churn", module=net).run({"kind": "trace", "payload": {"input": packed}})
    assert "block.relu" not in [frame["node_id"] for frame in fewer["frames"]]


def test_hooks_on_a_structure_read_from_onnx_follow_what_the_export_did_to_the_modules(client, api, torch):
    net = residual(torch)
    native = {"shape": [1, 4, 5, 6], "dtype": "float32", "perm": [0, 2, 3, 1]}
    args = {"shape": [5, 6, 4], "tensor": native}
    nodes = [
        {"id": "stem.Conv", "op": "Conv", "units": 120, "path": "stem.Conv", "args": args},
        # The exporter folds each normalization into the convolution before it: these steps hold what bn1 and bn2 return.
        {"id": "block.conv1.Conv", "op": "Conv", "units": 120, "path": "block.conv1.Conv", "args": args},
        {"id": "block.relu.Relu", "op": "Relu", "units": 120, "path": "block.relu.Relu", "args": args},
        {"id": "block.conv2.Conv", "op": "Conv", "units": 120, "path": "block.conv2.Conv", "args": args},
        {"id": "block.Add", "op": "Add", "units": 120, "path": "block.Add", "args": args},
        {"id": "block.relu_1.Relu", "op": "Relu", "units": 120, "path": "block.relu_1.Relu", "args": args},
        {"id": "ReduceMean", "op": "ReduceMean", "units": 4, "path": "ReduceMean", "args": {"shape": [4], "tensor": {"shape": [1, 4], "dtype": "float32"}}},
        {"id": "head.Gemm", "op": "Gemm", "units": 2, "path": "head.Gemm", "args": {"shape": [2], "tensor": {"shape": [1, 2], "dtype": "float32"}}},
        {"id": "probabilities", "op": "Softmax", "kind": "output", "units": 2, "path": "probabilities", "inputs": ["head.Gemm"], "args": {"shape": [2], "tensor": {"shape": [1, 2], "dtype": "float32"}}},
    ]
    features = [{"name": "image", "type": "image", "height": 5, "width": 6, "channels": 3}]
    module_api(api, features, nodes, task="binary_classification", labels=("no", "yes"), head={"vector_node": "probabilities", "link": "softmax"})
    picture = np.random.default_rng(3).uniform(0, 1, size=(5, 6, 3)).astype("<f4")
    want, scores = by_hand(net, torch, picture)
    agent = Agent(client, "churn", module=net)
    traced = agent.run({"kind": "trace", "payload": {"input": {"image": picture}}})
    frames = {frame["node_id"]: frame for frame in traced["frames"]}
    assert list(frames) == ["stem.Conv", "block.conv1.Conv", "block.relu.Relu", "block.conv2.Conv", "block.relu_1.Relu", "head.Gemm"]
    for step, name in (("stem.Conv", "stem"), ("block.conv1.Conv", "bn1"), ("block.relu.Relu", "relu1"), ("block.conv2.Conv", "bn2"), ("block.relu_1.Relu", "relu2")):
        np.testing.assert_allclose(frames[step]["value"], want[name][:48], rtol=1e-5, atol=1e-6, err_msg=step)
    np.testing.assert_allclose(traced["prediction"], scores, rtol=1e-5)
    # A tensor of another shape than the structure gives the step is not the step's, whatever its size.
    turned = [dict(node, args={"shape": [5, 6, 4], "tensor": {"shape": [1, 5, 6, 4], "dtype": "float32"}}) if node["id"] == "stem.Conv" else node for node in nodes]
    module_api(api, features, turned, task="binary_classification", labels=("no", "yes"))
    again = Agent(client, "churn", module=net).run({"kind": "trace", "payload": {"input": {"image": picture}}})
    assert "stem.Conv" not in [frame["node_id"] for frame in again["frames"]]


def test_a_step_that_took_in_a_module_nobody_can_account_for_is_given_nothing(client, api, torch):
    nn = torch.nn
    torch.manual_seed(5)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.scale = nn.Hardtanh(-0.1, 0.1)  # it changes the values, and the structure has no step for it
            self.last = nn.Linear(4, 2)

        def forward(self, x):
            return self.last(self.scale(self.first(x)))

    net = Net().eval()
    nodes = [
        {"id": "first.Gemm", "op": "Gemm", "units": 4, "path": "first.Gemm", "args": {"shape": [4], "tensor": {"shape": [1, 4], "dtype": "float32"}}},
        {"id": "last.Gemm", "op": "Gemm", "units": 2, "path": "last.Gemm", "args": {"shape": [2], "tensor": {"shape": [1, 2], "dtype": "float32"}}},
    ]
    module_api(api, [{"name": name, "type": "numeric"} for name in "abc"], nodes, task="binary_classification", labels=("no", "yes"))
    traced = Agent(client, "churn", module=net).run({"kind": "trace", "payload": {"input": {"a": 1.0, "b": 2.0, "c": 3.0}}})
    assert [frame["node_id"] for frame in traced["frames"]] == ["last.Gemm"]


def test_a_steps_tensor_is_laid_out_as_the_structure_shows_the_step():
    attention = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)  # heads, queries, keys
    step = {"id": "attn", "units": 24, "args": {"shape": [3, 4, 2], "tensor": {"shape": [1, 2, 3, 4], "dtype": "float32", "perm": [0, 2, 3, 1]}}}
    np.testing.assert_array_equal(shown_values(step, attention), attention[0].transpose(1, 2, 0).reshape(-1))
    assert shown_values(step, attention.reshape(1, 4, 3, 2), exact=True) is None  # as many values, another tensor
    assert shown_values(step, attention[:, :1]) is None and shown_values(step, np.array(["a"] * 24)) is None
    plain = {"id": "fc", "units": 6, "args": {"shape": [6], "tensor": {"shape": [1, 6], "dtype": "float32"}}}
    np.testing.assert_array_equal(shown_values(plain, np.arange(6).reshape(1, 6)), np.arange(6.0))
    # A structure that does not describe its tensors shows a map channels last, where PyTorch has it channels first.
    cube = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    np.testing.assert_array_equal(shown_values({"id": "conv", "units": 8, "args": {"shape": [2, 2, 2]}}, cube), cube[0].transpose(1, 2, 0).reshape(-1))
    table = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    np.testing.assert_array_equal(shown_values({"id": "seq", "units": 6, "args": {"shape": [2, 3]}}, table), np.arange(6.0))
    np.testing.assert_array_equal(shown_values({"id": "c1", "units": 6, "args": {"shape": [3, 2]}}, table), table[0].T.reshape(-1))


def test_the_steps_sent_in_full_are_the_ones_the_api_would_choose():
    def node(name, op, shape, kind="linear", inputs=()):
        return {"id": name, "op": op, "kind": kind, "units": int(np.prod(shape)), "inputs": list(inputs), "args": {"shape": list(shape)}}

    nodes = [
        node("image", "input", [8, 8, 3], kind="input"),
        node("conv", "Conv", [8, 8, 4], inputs=["image"]),  # only its activation reads it: the activation has the finished map
        node("relu", "Relu", [8, 8, 4], kind="activation", inputs=["conv"]),
        node("pool", "MaxPool", [4, 4, 4], inputs=["relu"]),  # only resamples
        node("conv2", "Conv", [4, 4, 4], inputs=["pool"]),
        node("flat", "Flatten", [64], kind="reshape", inputs=["conv2"]),
        node("fc", "Gemm", [3], inputs=["flat"]),
        node("probabilities", "Softmax", [3], kind="output", inputs=["fc"]),
    ]
    assert default_detail(nodes, "probabilities") == ["relu", "conv2", "fc", "probabilities"]
    many = [node(f"v{index}", "Gemm", [4]) for index in range(30)]
    chosen = default_detail(many, None)
    assert len(chosen) == 8 and chosen[0] == "v0" and chosen[-1] == "v29"


def test_a_picture_reaches_a_convolution_channels_first_and_its_maps_come_back_channels_last(client, api, torch):
    nn = torch.nn
    torch.manual_seed(1)
    conv = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    features = [{"name": "photo", "type": "image", "height": 6, "width": 5, "channels": 3}]
    nodes = [
        {"id": "photo", "op": "input", "kind": "input", "units": 90, "feature": "photo", "args": {"shape": [6, 5, 3]}},
        {"id": "0.Conv", "op": "Conv", "kind": "conv", "units": 120, "path": "0.Conv", "args": {"shape": [6, 5, 4]}},
        {"id": "1.Relu", "op": "Relu", "kind": "activation", "units": 120, "path": "1.Relu", "args": {"shape": [6, 5, 4]}},
        {"id": "2.GlobalAveragePool", "op": "GlobalAveragePool", "kind": "pool", "units": 4, "path": "2.GlobalAveragePool", "args": {"shape": [1, 1, 4]}},
        {"id": "4.Gemm", "op": "Gemm", "kind": "linear", "units": 2, "path": "4.Gemm", "args": {"shape": [2]}},
    ]
    module_api(api, features, nodes, task="binary_classification", labels=("no", "yes"))
    picture = np.random.default_rng(2).uniform(0, 1, size=(6, 5, 3)).astype("<f4")
    packed = {"photo": {"$b64": base64.b64encode(picture.tobytes()).decode(), "dtype": "f32", "shape": [6, 5, 3]}}
    conv.eval()
    with torch.no_grad():
        maps = conv[1](conv[0](torch.tensor(picture.transpose(2, 0, 1)[None]))).numpy()[0]
        own = conv(torch.tensor(picture.transpose(2, 0, 1)[None])).numpy().reshape(-1)
    agent = Agent(client, "churn", module=conv)
    np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": [packed]}})["predictions"][0], own, rtol=1e-5)
    watched = agent.run({"kind": "activations", "payload": {"input": packed, "nodes": None, "units": 16, "channels": 8, "max_side": 56}})
    stages = {stage["node_id"]: stage for stage in watched["stages"]}
    assert stages["photo"]["shape"] == [6, 5, 3] and stages["photo"]["detailed"] is False
    relu = stages["1.Relu"]
    assert relu["shape"] == [6, 5, 4] and relu["detailed"] is True and len(relu["maps"]) == 4
    strongest = relu["maps"][0]
    np.testing.assert_allclose(decode(strongest), maps[strongest["channel"]], atol=np.ptp(maps[strongest["channel"]]) / 255 + 1e-6)
    traced = agent.run({"kind": "trace", "payload": {"input": packed}})
    frame = next(item for item in traced["frames"] if item["node_id"] == "1.Relu")
    np.testing.assert_allclose(frame["value"], maps.transpose(1, 2, 0).reshape(-1)[:48], rtol=1e-5, atol=1e-6)
    assert len(frame["sample"]) == 120  # fewer than 256 values: all of them


def test_token_ids_are_filled_up_to_what_the_model_reads_and_prepare_takes_over_when_given(client, api, torch):
    nn = torch.nn
    torch.manual_seed(2)

    class Reviews(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.head = nn.Embedding(9, 4), nn.Linear(4, 2)
            self.position = nn.Parameter(torch.randn(1, 6, 4))

        def forward(self, ids):
            return self.head((self.embed(ids) + self.position).mean(dim=1))

    reviews = Reviews().eval()
    features = [{"name": "review", "type": "text", "max_length": 6, "vocab_size": 9}]
    nodes = [
        {"id": "review", "op": "input", "kind": "input", "units": 6, "feature": "review", "args": {"feature_type": "text"}},
        {"id": "head.Gemm", "op": "Gemm", "kind": "linear", "units": 2, "path": "head.Gemm", "args": {}},
    ]
    module_api(api, features, nodes, task="binary_classification", labels=("no", "yes"))
    with torch.no_grad():
        own = reviews(torch.tensor([[2, 6, 7, 7, 7, 7]])).numpy().reshape(-1)
    agent = Agent(client, "churn", module=reviews, pad_id=7)
    np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": [{"review": [2, 6]}]}})["predictions"][0], own, rtol=1e-6)
    traced = agent.run({"kind": "trace", "payload": {"input": {"review": [2, 6]}}})
    assert traced["frames"][0]["value"] == [2.0, 6.0, -1.0, -1.0, -1.0, -1.0]  # a text's ids, as the API holds them
    np.testing.assert_allclose(traced["frames"][1]["value"], own, rtol=1e-6)
    seen = []

    def prepare(row):
        seen.append(row)
        return (torch.tensor([row["review"] + [0] * (6 - len(row["review"]))]),)

    prepared = Agent(client, "churn", module=reviews, prepare=prepare)
    with torch.no_grad():
        zero_padded = reviews(torch.tensor([[2, 6, 0, 0, 0, 0]])).numpy().reshape(-1)
    np.testing.assert_allclose(prepared.run({"kind": "predict", "payload": {"rows": [{"review": [2, 6]}]}})["predictions"][0], zero_padded, rtol=1e-6)
    assert seen == [{"review": [2, 6]}]


def test_a_language_model_answers_with_its_scores_at_the_last_token_the_row_gave(client, api, torch):
    nn = torch.nn
    torch.manual_seed(6)

    class Writer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.head = nn.Embedding(7, 5), nn.Linear(5, 7)

        def forward(self, ids):
            return self.head(torch.cumsum(self.embed(ids), dim=1))  # every position's scores: (rows, positions, vocabulary)

    writer = Writer().eval()
    features = [{"name": "prompt", "type": "text", "max_length": 6, "vocab_size": 7}]
    nodes = [{"id": "head.MatMul", "op": "Linear", "units": 42, "path": "head.MatMul", "args": {"shape": [6, 7], "tensor": {"shape": [1, 6, 7], "dtype": "float32"}}}]
    module_api(api, features, nodes, task="text_generation", labels=())
    rows = [{"prompt": [3, 1, 4]}, {"prompt": [2]}, {"prompt": [1, 2, 3, 4, 5, 6, 1, 2]}]
    with torch.no_grad():
        scores = writer(torch.tensor([[3, 1, 4, 0, 0, 0], [2, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]])).numpy()
    agent = Agent(client, "churn", module=writer)
    answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
    assert answer["raw"] is True and [len(line) for line in answer["predictions"]] == [7, 7, 7]
    np.testing.assert_allclose(answer["predictions"][0], scores[0, 2], rtol=1e-5)  # the third token is the last one given
    np.testing.assert_allclose(answer["predictions"][1], scores[1, 0], rtol=1e-5)
    np.testing.assert_allclose(answer["predictions"][2], scores[2, 5], rtol=1e-5)  # a longer text is cut to what the model reads
    traced = agent.run({"kind": "trace", "payload": {"input": rows[0]}})
    np.testing.assert_allclose(traced["prediction"], scores[0, 2], rtol=1e-5)
    assert traced["raw"] is True and traced["frames"][0]["summary"]["size"] == 42


def test_predict_with_a_module_stacks_what_the_rows_hold(torch):
    nn = torch.nn
    table = nn.Linear(2, 1)
    with torch.no_grad():
        table.weight.copy_(torch.tensor([[2.0, -1.0]]))
        table.bias.zero_()
    asked = predict_with(table, ["a", "b"])
    np.testing.assert_allclose(asked([{"b": 1.0, "a": 3.0}, {"a": 0.0, "b": 4.0}]), [[5.0], [-4.0]])
    assert asked.raw is True  # a module's own output
    series = nn.Sequential(nn.Flatten(), nn.Linear(6, 1))
    rows = [{"signal": np.ones((3, 2), dtype=np.float32)}, {"signal": np.zeros((3, 2), dtype=np.float32)}]
    assert predict_with(series)(rows).shape == (2, 1)
    conv = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    assert predict_with(conv)([{"photo": np.zeros((5, 4, 3), dtype=np.float32)}]).shape == (1, 2)
    embed = nn.Sequential(nn.Embedding(5, 3), nn.Flatten(), nn.Linear(9, 1))
    assert predict_with(embed)([{"ids": [1, 2, 3]}, {"ids": [4]}]).shape == (2, 1)
    with pytest.raises(ValueError):
        predict_with(table)([{"a": [1.0], "b": [2.0]}])


# --- how a model file's inputs are made from the model's features -------------------------------


def test_a_files_inputs_nobody_describes_are_the_obvious_thing():
    f4, i8 = np.dtype("float32"), np.dtype("int64")
    table = bind_inputs(
        [("X", f4, [None, 2]), ("more", f4, [None, None])],
        [{"name": "a", "type": "numeric"}, {"name": "b", "type": "binary"}, {"name": "c", "type": "categorical"}],
    )
    assert [(item["source"], item["shape"]) for item in table] == [({"features": ["a", "b"]}, (1, 2)), ({"features": ["c"]}, (1, 1))]
    text = bind_inputs(
        [("input_ids", i8, [None, None]), ("attention_mask", i8, [None, None]), ("token_type_ids", i8, [None, None])],
        [{"name": "review", "type": "text", "max_length": 16}],
        pad_id=3,
    )
    assert [item["source"] for item in text] == [
        {"feature": "review", "part": "ids", "pad_id": 3},
        {"feature": "review", "part": "mask"},
        {"feature": "review", "part": "type_ids"},
    ]
    assert all(item["shape"] == (1, 16) and item["sample"] == (16,) and item["axis"] == 0 for item in text)
    photo = {"name": "photo", "type": "image", "height": 8, "width": 6, "channels": 3}
    first = bind_inputs([("pixels", f4, [None, 3, 8, 6])], [photo])[0]
    last = bind_inputs([("pixels", f4, [1, 8, 6, 3])], [photo])[0]
    alone = bind_inputs([("pixels", f4, [3, 8, 6])], [photo])[0]
    assert (first["source"]["layout"], first["shape"]) == ("nchw", (1, 3, 8, 6))
    assert (last["source"]["layout"], last["shape"]) == ("nhwc", (1, 8, 6, 3))
    assert (alone["shape"], alone["sample"], alone["axis"]) == ((3, 8, 6), (3, 8, 6), None)
    series = bind_inputs(
        [("signal", f4, [None, None, 2]), ("x", i8, [None, None])],
        [{"name": "signal", "type": "sequence", "shape": [24, 2]}, {"name": "note", "type": "tokens", "max_length": 5}],
    )
    assert series[0]["source"] == {"feature": "signal"} and series[0]["shape"] == (1, 24, 2)
    assert series[1]["source"] == {"feature": "note", "part": "ids", "pad_id": 0} and series[1]["shape"] == (1, 5)


def test_a_files_inputs_can_be_described_and_what_cannot_be_told_is_asked_for():
    f4 = np.dtype("float32")
    features = [{"name": "wave", "type": "tensor", "shape": [4]}, {"name": "echo", "type": "tensor", "shape": [4]}, {"name": "gain", "type": "numeric"}]

    def twice(row):
        return [[2.0 * row["gain"]]]

    bound = bind_inputs(
        [("a", f4, [None, 4]), ("b", f4, [None, 4]), ("state", f4, [1, 3]), ("g", f4, [None, 1]), ("h", f4, [None, 1])],
        features,
        {"a": "echo", "b": {"feature": "wave"}, "state": {"zeros": True}, "g": twice, "h": ["gain"]},
    )
    assert [item.get("source") for item in bound] == [{"feature": "echo"}, {"feature": "wave"}, {}, None, {"features": ["gain"]}]
    assert bound[2]["fixed"].shape == (1, 3) and bound[3]["call"] is twice
    with pytest.raises(ValueError, match="not clear which feature"):
        bind_inputs([("a", f4, [None, 4])], features[:2])
    with pytest.raises(ValueError, match="no feature called 'hum'"):
        bind_inputs([("a", f4, [None, 4])], features, {"a": "hum"})
    with pytest.raises(TypeError):
        bind_inputs([("a", f4, [None, 4])], features, {"a": 4})


# --- the command line -------------------------------------------------------------------------


def test_the_command_line_loads_the_function_it_is_given(monkeypatch, capsys):
    assert _load("json:dumps") is json.dumps
    assert _load("os.path:join").__name__ == "join"
    for wrong in ("json", ":dumps", "json:"):
        with pytest.raises(SystemExit):
            _load(wrong)
    with pytest.raises(SystemExit):
        main(["--model", "churn"])  # --predict is required
    served = {}

    def serve_once(self, stop=None):
        served["agent"] = self

    monkeypatch.setattr(Agent, "serve", serve_once)
    monkeypatch.setenv("KAIMEI_API_KEY", "km_test_cli")
    monkeypatch.setenv("KAIMEI_BASE_URL", "https://kaimei.example")
    assert main(["--model", "churn", "--predict", "json:dumps", "--label", "gpu box", "--quiet"]) == 0
    agent = served["agent"]
    assert agent.model == "churn" and agent.label == "gpu box" and agent._predict is json.dumps and agent._log is None
    assert agent.capabilities == ["predict"]
    assert main(["--model", "churn", "--predict", "json:dumps", "--tokenizer", "builtins:list", "--quiet"]) == 0
    assert served["agent"].capabilities == ["predict", "tokenize"]
    assert agent.client.base_url == "https://kaimei.example"
    assert "Kaimei agent for churn at https://kaimei.example" in capsys.readouterr().out
