"""The agent, as a third review found it: each fix has tests here that failed before it.

The structure that names a model's steps comes from Kaimei. An agent took its word for what a step is, so a
step could name the model's weights and have them sent back; an answer that could not be written as JSON ended
the agent; and a value kept at home was asked for under a name the agent does not go by. A few tests beside
those pin what must not change: an ordinary look inside, and what a model's steps show of it by answering.
"""

from __future__ import annotations

import base64
import json
import threading
import warnings

import httpx
import pytest
import respx

np = pytest.importorskip("numpy")

from kaimei import Kaimei  # noqa: E402
from kaimei import agent as agents  # noqa: E402
from kaimei.agent import Agent, serve, shown_values, stage_of  # noqa: E402

BASE = "https://api.kaimei.ai"
MODEL = "/v1/models/churn"
RNG = np.random.default_rng(7)
#: The model's own numbers: nothing of these may reach an answer unless the input's values went into it.
W = RNG.normal(size=(4, 4)).astype(np.float32)
B = RNG.normal(size=4).astype(np.float32)
C = RNG.normal(size=(2, 3)).astype(np.float32)
FEATURES = [{"name": f"f{i}", "type": "numeric"} for i in range(4)]
ROW = {"f0": 0.3, "f1": -1.2, "f2": 0.8, "f3": 2.0}
ZERO = {f"f{i}": 0.0 for i in range(4)}


class OnKaimei:
    """What Kaimei says of the model, without HTTP: its record, and a structure the server wrote."""

    def __init__(self, document, features=FEATURES, task="regression"):
        self.document, self.features, self.task = document, features, task

    def _request(self, method, path, **kwargs):
        if path.endswith("/structure"):
            return self.document
        return {"task": self.task, "features": self.features}


def numbers(value, found=None):
    """Every number an answer holds, with the bytes of its pictures turned back into values."""
    found = [] if found is None else found
    if isinstance(value, dict):
        if value.get("encoding") == "u8" and isinstance(value.get("data"), str):
            low, high = value["range"]
            raw = np.frombuffer(base64.b64decode(value["data"]), dtype=np.uint8).astype(float)
            found.extend((low + raw / 255.0 * (high - low)).tolist())
        for key, item in value.items():
            if key != "data":
                numbers(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            numbers(item, found)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found.append(float(value))
    return found


def told(answer, secret):
    """How many of ``secret``'s numbers are in an answer (to five decimal places)."""
    held = np.array(numbers(answer))
    return sum(1 for value in np.asarray(secret, dtype=float).reshape(-1) if held.size and np.min(np.abs(held - value)) < 5e-6)


#: A trace asked for one step alone says nothing of the others; every other look speaks of them all.
ALONE = "trace of the step"


def looks(agent, row, step):
    """Every way a look inside can be asked for one step: its answers by what was asked."""
    return {
        ALONE: agent.run({"kind": "trace", "payload": {"input": row, "nodes": [step]}}),
        "trace": agent.run({"kind": "trace", "payload": {"input": row}}),
        "activations of the step": agent.run({"kind": "activations", "payload": {"input": row, "nodes": [step], "units": 64, "max_side": 4096}}),
        "activations": agent.run({"kind": "activations", "payload": {"input": row, "units": 64}}),
    }


def steps_in(answer):
    return [item["node_id"] for item in answer.get("frames", answer.get("stages", []))]


def sent_of(answer, step):
    return [item for item in answer.get("frames", answer.get("stages", [])) if item["node_id"] == step]


# --- item 1: a file's weights -------------------------------------------------------------------

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def dense_file():
    """``y = relu(x @ W.T + B)``, with every way a file can hold its weights under a tensor's name."""
    nodes = [
        helper.make_node("Transpose", ["W"], ["Wt"], perm=[1, 0], name="t"),  # folded from an initializer
        helper.make_node("Identity", ["W"], ["Wid"], name="id"),  # an identity fed by a weight
        helper.make_node("Mul", ["W", "two"], ["W2"], name="twice"),  # computed from constants alone
        helper.make_node("Constant", [], ["Cc"], value=numpy_helper.from_array(C, "c"), name="const"),
        helper.make_node("Shape", ["x"], ["shape"], name="shape"),
        helper.make_node("Expand", ["B", "shape"], ["Bexp"], name="expand"),  # a weight, through the input's shape
        helper.make_node("MatMul", ["x", "Wt"], ["h"], name="fc/MatMul"),
        helper.make_node("Add", ["h", "B"], ["z"], name="fc/Add"),
        helper.make_node("Relu", ["z"], ["y"], name="act/Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        # A file may even give a weight among its outputs.
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4]), helper.make_tensor_value_info("Wid", TensorProto.FLOAT, [4, 4])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B"), numpy_helper.from_array(np.float32(2.0), "two")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def by_hand(row):
    x = np.array([[row[f"f{i}"] for i in range(4)]], dtype=np.float32)
    hidden = x @ W.T
    return hidden.reshape(-1), (hidden + B).reshape(-1), np.maximum(hidden + B, 0).reshape(-1)


def honest_steps():
    return [
        {"id": "in", "op": "input", "units": 4, "args": {"features": ["f0", "f1", "f2", "f3"]}},
        {"id": "z", "op": "Add", "units": 4, "inputs": ["in"], "args": {"shape": [4], "tensor": {"shape": [1, 4]}}},
        {"id": "y", "op": "Relu", "units": 4, "inputs": ["z"], "args": {"shape": [4], "tensor": {"shape": [1, 4]}}},
    ]


def dense_agent(extra=(), head="y", **options):
    document = {"scene": {"nodes": [*honest_steps(), *extra]}, "head": {"vector_node": head}}
    return Agent(OnKaimei(document), "m", onnx=dense_file().SerializeToString(), **options)


def leak(source, units, shape, perm=None, name="leak"):
    described = {"shape": shape, **({"perm": perm} if perm else {})}
    return {"id": name, "op": "MatMul", "units": units, "args": {"source": source, "tensor": described}}


@pytest.mark.parametrize(
    "step, secret",
    [
        (leak("Wt", 16, [16]), W),  # the review's: a constant folded from an initializer
        (leak("Wt", 16, [4, 4], [1, 0]), W),
        (leak("Wt", 16, [2, 8], [1, 0]), W),  # regrouped, to read another 48 values each time
        (leak("Wid", 16, [4, 4]), W),  # an identity fed by a weight, and an output of the file
        (leak("W2", 16, [4, 4]), W * 2),  # computed from constants alone
        (leak("Cc", 6, [2, 3]), C),  # a Constant
        (leak("Bexp", 4, [1, 4]), B),  # a weight, through the shape of the input alone
        ({"id": "Wt", "op": "MatMul", "units": 16, "args": {"tensor": {"shape": [4, 4]}}}, W),  # by its own id
    ],
)
def test_a_step_that_names_the_models_own_numbers_is_never_read(step, secret):
    agent = dense_agent([step])
    name = step["id"]
    for asked, answer in looks(agent, ROW, name).items():
        assert sent_of(answer, name) == [], asked
        assert told(answer, secret) == 0, asked
        # Said, not raised: the step is left out and named, and the rest of the answer is what it always was.
        assert [(item["node_id"], item["reason"]) for item in answer["withheld"]] == [(name, agents._NOT_COMPUTED)], asked
        np.testing.assert_allclose(answer["prediction"], by_hand(ROW)[2], rtol=1e-5)
        json.dumps(answer, allow_nan=False)
    traced = agent.run({"kind": "trace", "payload": {"input": ROW}})
    frames = {frame["node_id"]: frame["value"] for frame in traced["frames"]}
    assert list(frames) == ["in", "z", "y"]
    np.testing.assert_allclose(frames["z"], by_hand(ROW)[1], rtol=1e-5)


def test_an_initializer_is_not_a_tensor_a_step_can_name():
    agent = dense_agent([leak("W", 16, [4, 4]), leak("B", 4, [4], name="bias")])
    for asked, answer in looks(agent, ROW, "leak").items():
        assert sent_of(answer, "leak") == sent_of(answer, "bias") == [] and told(answer, W) == 0, asked


def test_a_prediction_is_never_read_from_the_models_own_numbers():
    # The head of the structure says which step a prediction is read from, and it comes from Kaimei too.
    agent = dense_agent([leak("Wt", 16, [16])], head="leak")
    answer = agent.run({"kind": "predict", "payload": {"rows": [ROW, ZERO]}})
    np.testing.assert_allclose(answer["predictions"], [by_hand(ROW)[2], by_hand(ZERO)[2]], rtol=1e-5, atol=1e-6)
    assert told(answer, W) == 0 and agent._output_tensor() == "y"  # the file's own output, as with no head to say
    traced = agent.run({"kind": "trace", "payload": {"input": ROW}})
    np.testing.assert_allclose(traced["prediction"], by_hand(ROW)[2], rtol=1e-5)


def test_a_file_that_gives_out_its_own_numbers_first_is_still_asked_for_what_it_computes():
    # With no structure to say, a prediction is read from the file's first output of floats. Here that is a weight.
    model = dense_file()
    outputs = list(model.graph.output)
    del model.graph.output[:]
    model.graph.output.extend(reversed(outputs))
    assert [info.name for info in model.graph.output] == ["Wid", "y"]
    for document in (None, {"scene": {"nodes": honest_steps()}, "head": {"vector_node": "nowhere"}}):
        agent = Agent(OnKaimei(document), "m", onnx=model)
        answer = agent.run({"kind": "predict", "payload": {"rows": [ROW]}})
        np.testing.assert_allclose(answer["predictions"], [by_hand(ROW)[2]], rtol=1e-5)
        assert told(answer, W) == 0 and agent._output_tensor() == "y"
    # A file all of whose outputs are its own numbers has no prediction to give, and says why.
    del model.graph.output[1:]
    with pytest.raises(ValueError, match="No output of this ONNX file that holds scores is computed from its input: Wid"):
        Agent(OnKaimei(None), "m", onnx=model).run({"kind": "predict", "payload": {"rows": [ROW]}})


def test_many_steps_that_name_the_weights_make_an_answer_no_longer_than_it_may_be():
    lines = []
    extra = [leak("Wt", 16, [16], name=f"leak{index}") for index in range(1000)]
    agent = dense_agent(extra, log=lines.append)
    answer = agent.run({"kind": "trace", "payload": {"input": ROW}})
    assert [frame["node_id"] for frame in answer["frames"]] == ["in", "z", "y"] and told(answer, W) == 0
    assert len(answer["withheld"]) == agents.WITHHELD_SAID
    agent.run({"kind": "trace", "payload": {"input": ROW}})
    assert len(lines) == 4 * agents.WITHHELD_SAID  # each is said once in the log, and not without end
    assert lines[0].startswith("Nothing is sent of the step 'leak0'. Its values are not computed from the model's input")


def test_what_the_input_goes_into_is_still_sent_whatever_it_tells_of_the_model():
    # A model's steps computed from the input are what a look inside is: for inputs chosen for it, they hold
    # the model's own numbers (a column of W for a one-hot row, the bias for a row of zeros). That is the model
    # answering, and it is not kept back.
    agent = dense_agent([leak("h", 4, [1, 4])])
    one_hot = {f"f{i}": float(i == 2) for i in range(4)}
    traced = agent.run({"kind": "trace", "payload": {"input": one_hot}})
    frames = {frame["node_id"]: frame["value"] for frame in traced["frames"]}
    np.testing.assert_allclose(frames["leak"], W[:, 2], rtol=1e-6)
    assert "withheld" not in traced
    zeros = agent.run({"kind": "trace", "payload": {"input": ZERO}})
    np.testing.assert_allclose({frame["node_id"]: frame["value"] for frame in zeros["frames"]}["z"], B, rtol=1e-6)


def test_a_step_is_laid_out_only_as_its_tensor_is():
    hidden = by_hand(ROW)[0]
    # The tensor's own shape, and a true order of its axes: honoured.
    agent = dense_agent([leak("h", 4, [1, 4], [1, 0])])
    frames = {frame["node_id"]: frame["value"] for frame in agent.run({"kind": "trace", "payload": {"input": ROW}})["frames"]}
    np.testing.assert_allclose(frames["leak"], hidden, rtol=1e-5)
    # Another shape of as many values would regroup them, and an order that is not one is nothing at all.
    wrong = [
        leak("h", 4, [2, 2], [1, 0], name="regrouped"),
        leak("h", 4, [4, 1], name="turned"),
        leak("h", 4, [1, 4], [0, 0], name="twice"),
        leak("h", 4, [1, 4], [0, 5], name="beyond"),
        leak("h", 4, [1, 4], [-1, 0], name="backwards"),
        leak("h", 4, [1, 4], [0], name="short"),
        leak("h", 4, [1, 4], "10", name="text"),
        {"id": "worded", "op": "MatMul", "units": 4, "args": {"source": "h", "tensor": {"shape": "1x4"}}},
        {"id": "sized", "op": "MatMul", "units": "four", "args": {"source": "h", "tensor": {"shape": [1, 4]}}},
        {"id": "listed", "op": ["MatMul"], "units": 4, "args": ["source", "h"]},
        {"op": "MatMul", "units": 4},
        "not a step",
    ]
    agent = dense_agent(wrong)
    for asked, answer in looks(agent, ROW, "regrouped").items():
        reasons = {item["node_id"]: item["reason"] for item in answer["withheld"]}
        assert "the shape [2, 2], and the tensor in the model has [1, 4]" in reasons["regrouped"], asked
        if asked == ALONE:
            assert steps_in(answer) == [] and list(reasons) == ["regrouped"]
            continue
        assert steps_in(answer) == ["in", "z", "y"], asked  # nothing raised, and every step that is one is there
        assert "the shape [4, 1]" in reasons["turned"]
        for name in ("twice", "beyond", "backwards", "short", "text"):
            assert reasons[name] == "The order the structure takes its tensor's axes in is not an order of those axes.", name
        assert reasons["worded"] == "The shape the structure gives its tensor is not a shape."
        assert "sized" not in reasons and "listed" not in reasons  # not this tensor's step at all
    # The same for whoever lays a tensor out by hand: nothing a structure says raises.
    tensor = np.arange(4.0).reshape(1, 4)
    step = {"id": "s", "units": 4, "args": {"tensor": {"shape": [2, 2], "perm": [1, 0]}}}
    np.testing.assert_array_equal(shown_values(step, tensor), [0.0, 2.0, 1.0, 3.0])  # as many values: laid out as said
    assert shown_values(step, tensor, exact=True) is None  # and not when it must be the tensor's own shape
    for perm in ([0, 0], [2, 0], [0], "01", [0.5, 1]):
        assert shown_values({"id": "s", "units": 4, "args": {"tensor": {"shape": [1, 4], "perm": perm}}}, tensor) is None
    assert shown_values({"id": "s", "units": "4", "args": None}, tensor) is None
    np.testing.assert_array_equal(shown_values({"id": "s", "units": 4.0, "args": {"shape": "?"}}, tensor), np.arange(4.0))


def test_the_reviews_stub_server_reads_nothing_of_the_weights():
    # review r3, finding 5: y = x @ Transpose(W), and a step the server adds over Transpose(W), on a row of zeros.
    secret = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.125 + 1.0
    graph = helper.make_graph(
        [helper.make_node("Transpose", ["W"], ["Wt"], perm=[1, 0], name="t"), helper.make_node("MatMul", ["x", "Wt"], ["y"], name="mm")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(secret, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    for shape, perm in (([16], [0]), ([4, 4], [1, 0]), ([2, 8], [1, 0])):
        document = {
            "scene": {
                "nodes": [
                    {"id": "in", "op": "input", "units": 4, "args": {"features": ["f0", "f1", "f2", "f3"]}},
                    {"id": "y", "op": "MatMul", "units": 4, "inputs": ["in"]},
                    {"id": "leak", "op": "MatMul", "units": 16, "args": {"source": "Wt", "tensor": {"shape": shape, "perm": perm}}},
                ]
            },
            "head": {"vector_node": "y"},
        }
        agent = Agent(OnKaimei(document), "m", onnx=model.SerializeToString())
        fractions = secret[secret % 1 != 0]  # a whole number in an answer may be a count; 1.125 is a weight
        for asked, answer in looks(agent, ZERO, "leak").items():
            assert sent_of(answer, "leak") == [] and told(answer, fractions) == 0, (shape, asked)
            assert steps_in(answer) == ([] if asked == ALONE else ["in", "y"]), (shape, asked)
            assert answer["prediction"] == [0.0, 0.0, 0.0, 0.0]
            assert [item["node_id"] for item in answer["withheld"]] == ["leak"]


def text_file():
    """Token, segment and position embeddings added up, as a transformer's first step has them."""
    rng = np.random.default_rng(11)
    tables = {
        "E": rng.normal(size=(9, 4)).astype(np.float32),
        "T": rng.normal(size=(2, 4)).astype(np.float32),
        "P": rng.normal(size=(6, 4)).astype(np.float32),
        "OUT": rng.normal(size=(4, 2)).astype(np.float32),
    }
    whole = {"one": np.array(1, dtype=np.int64), "zero": np.array(0, dtype=np.int64), "last": np.array([-1], dtype=np.int64), "along": np.array([1], dtype=np.int64)}
    nodes = [
        helper.make_node("Gather", ["E", "input_ids"], ["tok"], name="tok"),
        helper.make_node("Gather", ["T", "token_type_ids"], ["typ"], name="typ"),
        # Positions counted from the length of the text: its shape, and nothing it says.
        helper.make_node("Shape", ["input_ids"], ["shape"], name="shape"),
        helper.make_node("Gather", ["shape", "one"], ["length"], name="length"),
        helper.make_node("Range", ["zero", "length", "one"], ["positions"], name="positions"),
        helper.make_node("Gather", ["P", "positions"], ["pos"], name="pos"),
        helper.make_node("Add", ["tok", "typ"], ["both"], name="both"),
        helper.make_node("Add", ["both", "pos"], ["summed"], name="summed"),
        helper.make_node("Cast", ["attention_mask"], ["real"], to=TensorProto.FLOAT, name="real"),
        helper.make_node("Unsqueeze", ["real", "last"], ["real3"], name="real3"),
        helper.make_node("Mul", ["summed", "real3"], ["kept"], name="kept"),
        helper.make_node("ReduceSum", ["kept", "along"], ["pooled"], keepdims=0, name="pooled"),
        helper.make_node("MatMul", ["pooled", "OUT"], ["logits"], name="logits"),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", 6]) for name in ("input_ids", "attention_mask", "token_type_ids")],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        [numpy_helper.from_array(value, name) for name, value in {**tables, **whole}.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model, tables


def test_what_follows_from_the_inputs_shape_or_from_an_input_that_never_changes_is_the_models_own():
    model, tables = text_file()

    def step(name, shape):
        return {"id": name, "op": "x", "units": int(np.prod(shape)), "args": {"tensor": {"shape": shape}}}

    nodes = [
        {"id": "review", "op": "input", "units": 6, "feature": "review", "args": {"feature_type": "text"}},
        step("tok", [1, 6, 4]),
        step("typ", [1, 6, 4]),  # segment ids are zeros for every text: the first row of that table, six times
        step("pos", [6, 4]),  # the table of positions, cut to the text's length
        step("positions", [6]),
        step("summed", [1, 6, 4]),
        step("real", [1, 6]),
        step("logits", [1, 2]),
    ]
    features = [{"name": "review", "type": "text", "max_length": 6, "vocab_size": 9}]
    agent = Agent(OnKaimei({"scene": {"nodes": nodes}, "head": {"vector_node": "logits"}}, features, "binary_classification"), "m", onnx=model)
    assert agent._from_the_row() == {"tok", "both", "summed", "real", "real3", "kept", "pooled", "logits"}
    ids = [3, 1, 4]
    for asked, answer in looks(agent, {"review": ids}, "pos").items():
        alone = asked == ALONE
        assert steps_in(answer) == ([] if alone else ["review", "tok", "summed", "real", "logits"]), asked
        assert sorted(item["node_id"] for item in answer["withheld"]) == (["pos"] if alone else ["pos", "positions", "typ"])
        # Neither table is in any answer; the embedding of the tokens that were sent is, as it must be.
        assert told(answer, tables["P"]) == 0 and told(answer, tables["T"]) == 0, asked
    traced = agent.run({"kind": "trace", "payload": {"input": {"review": ids}}})
    frames = {frame["node_id"]: frame for frame in traced["frames"]}
    np.testing.assert_allclose(frames["tok"]["value"][:12], tables["E"][ids].reshape(-1), rtol=1e-6)
    assert frames["real"]["value"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    padded = tables["E"][[3, 1, 4, 0, 0, 0]] + tables["T"][0] + tables["P"]
    np.testing.assert_allclose(traced["prediction"], padded[:3].sum(axis=0) @ tables["OUT"], rtol=1e-4)


def test_an_input_given_a_value_fixed_beforehand_is_not_the_row():
    scale = np.array([[2.0, 0.5, -1.0, 3.0]], dtype=np.float32)
    gain = RNG.normal(size=(1, 4)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["scale", "gain"], ["own"], name="own"),  # a fixed value times a weight
            helper.make_node("Mul", ["x", "own"], ["scaled"], name="scaled"),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4]), helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [None, 4])],
        [numpy_helper.from_array(gain, "gain")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    nodes = [
        {"id": "own", "op": "Mul", "units": 4, "args": {"tensor": {"shape": [1, 4]}}},
        {"id": "scaled", "op": "Mul", "units": 4, "args": {"tensor": {"shape": [1, 4]}}},
    ]
    document = {"scene": {"nodes": nodes}, "head": {"vector_node": "scaled"}}
    agent = Agent(OnKaimei(document), "m", onnx=model, inputs={"scale": {"value": scale}})
    traced = agent.run({"kind": "trace", "payload": {"input": ROW}})
    assert [frame["node_id"] for frame in traced["frames"]] == ["scaled"] and told(traced, gain) == 0
    assert [item["node_id"] for item in traced["withheld"]] == ["own"]
    x = np.array([[ROW[f"f{i}"] for i in range(4)]], dtype=np.float32)
    np.testing.assert_allclose(traced["prediction"], (x * scale * gain).reshape(-1), rtol=1e-5)
    # An input fed by a function of your own is taken to be made from the row.
    fed = Agent(OnKaimei(document), "m", onnx=model, inputs={"scale": lambda row: scale * row["f0"]})
    assert [frame["node_id"] for frame in fed.run({"kind": "trace", "payload": {"input": ROW}})["frames"]] == ["own", "scaled"]


def test_a_branch_reads_what_it_reads_from_outside_itself():
    def branch(name, output):
        return helper.make_graph([helper.make_node("Identity", [output], [f"{name}_out"])], name, [], [helper.make_tensor_value_info(f"{name}_out", TensorProto.FLOAT, None)])

    def bare(name, output):  # a branch with no step of its own: what it gives is a tensor from outside
        return helper.make_graph([], name, [], [helper.make_tensor_value_info(output, TensorProto.FLOAT, None)])

    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["h"], name="h"),
        helper.make_node("ReduceSum", ["x"], ["total"], keepdims=0, name="total"),
        helper.make_node("Greater", ["total", "nought"], ["positive"], name="positive"),
        helper.make_node("If", ["always"], ["own"], then_branch=branch("a", "W"), else_branch=bare("b", "B"), name="own"),
        helper.make_node("If", ["always"], ["mixed"], then_branch=branch("c", "W"), else_branch=bare("d", "h"), name="mixed"),
        helper.make_node("If", ["positive"], ["chosen"], then_branch=branch("e", "W"), else_branch=branch("f", "W"), name="chosen"),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B"), numpy_helper.from_array(np.array(True), "always"), numpy_helper.from_array(np.float32(0), "nought")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    file = agents._OnnxFile(model)
    # A branch that reads the weights gives the weights; one that reads what the input went into does not.
    assert file.computed_from(["x"]) == {"h", "total", "positive", "mixed", "chosen"}
    assert file.computed_from([]) == set()
    # Steps written out of the order they run in are found all the same.
    del model.graph.node[:]
    model.graph.node.extend(reversed(nodes))
    assert agents._OnnxFile(model).computed_from(["x"]) == {"h", "total", "positive", "mixed", "chosen"}


def uploaded():
    from kaimei import graphs
    from kaimei.structure import finalize_spec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = graphs.from_onnx(dense_file(), task="regression", feature_names=[f"f{i}" for i in range(4)], output="y")
    return finalize_spec(raw)


def on_kaimei_as_uploaded(extra=(), head="act.Relu", **changed):
    nodes = [
        {"id": "in", "op": "input", "units": 4, "args": {"features": ["f0", "f1", "f2", "f3"]}},
        {"id": "fc.MatMul", "op": "Linear", "units": 4, "inputs": ["in"], "args": {"source": "z", "shape": [4], "tensor": {"shape": [1, 4]}}},
        {"id": "act.Relu", "op": "Relu", "units": 4, "inputs": ["fc.MatMul"], "args": {"source": "y", "shape": [4], "tensor": {"shape": [1, 4]}}},
        *extra,
    ]
    nodes = [dict(node, **changed.get(node["id"], {})) for node in nodes]
    return OnKaimei({"scene": {"nodes": nodes}, "head": {"vector_node": head}})


def test_an_agent_started_with_the_uploaded_structure_reads_only_its_steps():
    extraction = uploaded()
    assert sorted(extraction.spec["graph"]["values"]) == ["act.Relu", "fc.MatMul"] and extraction.spec["head"]["output"] == "act.Relu"
    hidden, summed, fired = by_hand(ROW)
    # The structure on Kaimei as it was uploaded: every step, and nothing kept back.
    agent = Agent(on_kaimei_as_uploaded(), "m", onnx=dense_file(), structure=extraction)
    traced = agent.run({"kind": "trace", "payload": {"input": ROW}})
    frames = {frame["node_id"]: frame["value"] for frame in traced["frames"]}
    assert list(frames) == ["in", "fc.MatMul", "act.Relu"] and "withheld" not in traced
    np.testing.assert_allclose(frames["fc.MatMul"], summed, rtol=1e-5)
    # A step added on Kaimei: the uploaded structure joined the MatMul and the Add, and does not hold "h".
    added = [leak("h", 4, [1, 4])]
    for structure in (extraction, json.loads(json.dumps(extraction.spec))):
        agent = Agent(on_kaimei_as_uploaded(added), "m", onnx=dense_file(), structure=structure)
        for asked, answer in looks(agent, ROW, "leak").items():
            assert sent_of(answer, "leak") == [], asked
            assert [(item["node_id"], item["reason"]) for item in answer["withheld"]] == [("leak", agents._NOT_HELD)], asked
    # Without the structure the same step is one Kaimei may name: its tensor is computed from the row.
    plain = Agent(on_kaimei_as_uploaded(added), "m", onnx=dense_file())
    np.testing.assert_allclose(sent_of(plain.run({"kind": "trace", "payload": {"input": ROW}}), "leak")[0]["value"], hidden, rtol=1e-5)
    # A step it holds is read from the tensor it names itself, whatever Kaimei's structure calls its source.
    moved = on_kaimei_as_uploaded(**{"fc.MatMul": {"args": {"source": "h", "shape": [4], "tensor": {"shape": [1, 4]}}}})
    agent = Agent(moved, "m", onnx=dense_file(), structure=extraction)
    frames = {frame["node_id"]: frame["value"] for frame in agent.run({"kind": "trace", "payload": {"input": ROW}})["frames"]}
    np.testing.assert_allclose(frames["fc.MatMul"], summed, rtol=1e-5)
    # And the prediction is read where its own head reads it, whichever step Kaimei's head points at.
    agent = Agent(on_kaimei_as_uploaded(head="fc.MatMul"), "m", onnx=dense_file(), structure=extraction)
    assert agent._output_tensor() == "y"
    np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": [ROW]}})["predictions"], [fired], rtol=1e-5)
    assert Agent(on_kaimei_as_uploaded(head="fc.MatMul"), "m", onnx=dense_file())._output_tensor() == "z"  # as before, without it
    # A structure written by hand to say how the inputs are fed lists no steps: Kaimei's are gone by.
    by_inputs = {"family": "graph", "graph": {"inputs": extraction.spec["graph"]["inputs"]}}
    agent = Agent(on_kaimei_as_uploaded(added), "m", onnx=dense_file(), structure=by_inputs)
    assert [frame["node_id"] for frame in agent.run({"kind": "trace", "payload": {"input": ROW}})["frames"]] == ["in", "fc.MatMul", "act.Relu", "leak"]


def test_an_ordinary_look_inside_a_file_is_what_it_was():
    agent = dense_agent()
    hidden, summed, fired = by_hand(ROW)
    traced = agent.run({"kind": "trace", "payload": {"input": ROW, "nodes": None}})
    assert sorted(traced) == ["frames", "prediction", "raw"]
    frames = {frame["node_id"]: frame for frame in traced["frames"]}
    assert list(frames) == ["in", "z", "y"] and sorted(frames["z"]) == ["node_id", "sample", "summary", "value"]
    np.testing.assert_allclose(frames["z"]["value"], summed, rtol=1e-5)
    assert frames["z"]["sample"] == frames["z"]["value"]
    values = np.array(frames["z"]["value"])
    assert frames["z"]["summary"] == {
        "size": 4,
        "max_abs": float(np.abs(values).max()),
        "mean_abs": float(np.abs(values).mean()),
        "l2": float(np.linalg.norm(values)),
        "active": 4,
        "argmax": int(np.argmax(np.abs(summed))),
    }
    watched = agent.run({"kind": "activations", "payload": {"input": ROW, "nodes": None, "units": 2}})
    assert sorted(watched) == ["forward_ms", "prediction", "raw", "stages"]
    stages = {stage["node_id"]: stage for stage in watched["stages"]}
    assert list(stages) == ["in", "z", "y"] and [name for name, stage in stages.items() if stage["detailed"]] == ["z", "y"]
    assert [unit["index"] for unit in stages["y"]["top_units"]] == list(np.argsort(-np.abs(fired), kind="stable")[:2])
    np.testing.assert_allclose(watched["prediction"], fired, rtol=1e-5)


# --- item 1: a module's weights -----------------------------------------------------------------


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def module_agent(net, nodes, features=FEATURES, task="regression", head="", **options):
    return Agent(OnKaimei({"scene": {"nodes": nodes}, "head": {"vector_node": head}}, features, task), "m", module=net, **options)


@pytest.fixture(params=["followed", "hooks alone"])
def telling(request, monkeypatch):
    """How the model's own numbers are told from what the input went into: by following every operation
    PyTorch runs, or, with a PyTorch that cannot be followed, by what the modules' hooks see."""
    if request.param == "hooks alone":
        monkeypatch.setattr(agents, "_following", lambda: None)
    return request.param


def test_a_parametrization_that_makes_a_layers_weight_is_not_a_step(torch, telling):
    nn = torch.nn

    class Twice(nn.Module):
        def forward(self, learned):
            return learned * 2.0  # the layer's weight is made from what was learned, each time it is used

    torch.manual_seed(3)
    net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    nn.utils.parametrize.register_parametrization(net[0], "weight", Twice())
    weight = net[0].weight.detach().numpy().copy()
    nodes = [
        {"id": "first", "op": "dense", "units": 3, "path": "0", "args": {}},
        {"id": "fired", "op": "relu", "units": 3, "path": "1", "args": {}},
        {"id": "last", "op": "dense", "units": 2, "path": "2", "args": {}},
        # Kaimei's structure says the module that makes the weight is a step, by every name it can be reached.
        {"id": "leak", "op": "dense", "units": 12, "path": "0.parametrizations.weight", "args": {}},
        {"id": "inner", "op": "dense", "units": 12, "path": "0.parametrizations.weight.0", "args": {"tensor": {"shape": [3, 4]}}},
        {"id": "below", "op": "dense", "units": 12, "path": "0.parametrizations.weight.0.Mul", "args": {"tensor": {"shape": [3, 4], "perm": [1, 0]}}},
    ]
    agent = module_agent(net, nodes, head="last")
    x = torch.tensor([[ROW[f"f{i}"] for i in range(4)]])
    with torch.no_grad():
        first, last = net[0](x).numpy().reshape(-1), net(x).numpy().reshape(-1)
    for asked, answer in looks(agent, ROW, "leak").items():
        assert steps_in(answer) == ([] if asked == ALONE else ["first", "fired", "last"]), asked
        assert told(answer, weight) == 0 and told(answer, weight / 2.0) == 0, asked
        kept = {item["node_id"]: item["reason"] for item in answer["withheld"]}
        assert kept == ({"leak": agents._NOT_COMPUTED} if asked == ALONE else {"leak": agents._NOT_COMPUTED, "inner": agents._NOT_COMPUTED}), asked
    frames = {frame["node_id"]: frame["value"] for frame in agent.run({"kind": "trace", "payload": {"input": ROW}})["frames"]}
    np.testing.assert_allclose(frames["first"], first, rtol=1e-5)
    np.testing.assert_allclose(frames["last"], last, rtol=1e-5)
    assert sum(len(layer._forward_hooks) for layer in net.modules()) == 0


def test_a_module_that_gives_back_the_models_own_numbers_is_not_a_step(torch, telling):
    nn = torch.nn

    class Places(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.randn(1, 6, 4))

        def forward(self, tokens):
            return self.table[:, : tokens.shape[1]]  # a part of what it learned, as long as the text

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.words = nn.Embedding(9, 4)
            self.places = Places()
            self.segments = nn.Embedding(2, 4)
            self.register_buffer("segment_ids", torch.zeros(1, 6, dtype=torch.long))
            self.drop = nn.Dropout(0.1)
            self.passed = nn.Identity()
            self.head = nn.Linear(4, 2)

        def forward(self, ids):
            words = self.words(ids)
            segments = self.passed(self.drop(self.segments(self.segment_ids[:, : ids.shape[1]])))  # looked up in a buffer
            return self.head((words + self.places(words) + segments).mean(dim=1))

    torch.manual_seed(5)
    net = Net().eval()
    nodes = [
        {"id": "review", "op": "input", "units": 6, "feature": "review", "args": {"feature_type": "text"}},
        {"id": "words", "op": "embedding", "units": 24, "path": "words", "args": {"shape": [6, 4]}},
        {"id": "places", "op": "embedding", "units": 24, "path": "places", "args": {"shape": [6, 4]}},
        {"id": "segments", "op": "embedding", "units": 24, "path": "segments", "args": {"shape": [6, 4]}},
        {"id": "dropped", "op": "dropout", "units": 24, "path": "drop", "args": {"shape": [6, 4]}},
        {"id": "passed", "op": "identity", "units": 24, "path": "passed", "args": {"shape": [6, 4]}},
        {"id": "head", "op": "dense", "units": 2, "path": "head", "args": {}},
    ]
    features = [{"name": "review", "type": "text", "max_length": 6, "vocab_size": 9}]
    agent = module_agent(net, nodes, features, "binary_classification", head="head")
    ids = [3, 1, 4, 1, 5, 2]
    for asked, answer in looks(agent, {"review": ids}, "places").items():
        alone = asked == ALONE
        assert steps_in(answer) == ([] if alone else ["review", "words", "head"]), asked
        assert sorted(item["node_id"] for item in answer["withheld"]) == (["places"] if alone else ["dropped", "passed", "places", "segments"]), asked
        assert told(answer, net.places.table.detach().numpy()) == 0 and told(answer, net.segments.weight.detach().numpy()) == 0
    traced = agent.run({"kind": "trace", "payload": {"input": {"review": ids}}})
    words = {frame["node_id"]: frame["value"] for frame in traced["frames"]}["words"]
    np.testing.assert_allclose(words, net.words.weight.detach().numpy()[ids].reshape(-1), rtol=1e-6)  # what the row chose


def test_what_is_made_on_the_way_from_no_input_is_the_models_own_too(torch, telling):
    nn = torch.nn

    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.randn(1, 6, 4))
            self.scale = nn.Parameter(torch.tensor(1.5))

        def forward(self, tokens):
            return self.table[:, : tokens.shape[1]] * self.scale  # called with the input, computed from none of it

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.words = nn.Embedding(9, 4)
            self.places = nn.Embedding(6, 4)
            self.kinds = nn.Embedding(2, 4)
            self.scaled = Scaled()
            self.mixed = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, ids):
            words = self.words(ids)
            places = self.places(torch.arange(ids.shape[1]))  # positions made on the way, as GPT-2 has them
            kinds = self.kinds(torch.zeros_like(ids))  # zeros in the shape of the ids: nothing the ids say
            state = places.clone()
            state += words[0]  # the input written into what was the model's own: it no longer is
            return self.head(self.mixed(state) + kinds + self.scaled(words)).mean(dim=1)

    torch.manual_seed(9)
    net = Net().eval()
    nodes = [
        {"id": "review", "op": "input", "units": 6, "feature": "review", "args": {"feature_type": "text"}},
        {"id": "words", "op": "embedding", "units": 24, "path": "words", "args": {"shape": [6, 4]}},
        {"id": "places", "op": "embedding", "units": 24, "path": "places", "args": {"shape": [6, 4]}},
        {"id": "kinds", "op": "embedding", "units": 24, "path": "kinds", "args": {"shape": [6, 4]}},
        {"id": "scaled", "op": "mul", "units": 24, "path": "scaled", "args": {"shape": [6, 4]}},
        {"id": "mixed", "op": "dense", "units": 24, "path": "mixed", "args": {"shape": [6, 4]}},
        {"id": "head", "op": "dense", "units": 12, "path": "head", "args": {"shape": [6, 2]}},
    ]
    features = [{"name": "review", "type": "text", "max_length": 6, "vocab_size": 9}]
    agent = module_agent(net, nodes, features, "binary_classification")
    ids = [3, 1, 4, 1, 5, 2]
    with torch.no_grad():
        expected = net(torch.tensor([ids])).numpy().reshape(-1)
    for asked, answer in looks(agent, {"review": ids}, "places").items():
        np.testing.assert_array_equal(np.float32(answer["prediction"]), expected)  # the run is what it would have been
        if telling == "hooks alone":
            # A hook sees what a module gave, not what it was made from: every step that is computed is there.
            assert asked == ALONE or {"words", "mixed", "head"} <= set(steps_in(answer)), asked
            continue
        alone = asked == ALONE
        assert steps_in(answer) == ([] if alone else ["review", "words", "mixed", "head"]), asked
        assert sorted(item["node_id"] for item in answer["withheld"]) == (["places"] if alone else ["kinds", "places", "scaled"]), asked
        for own in (net.places.weight, net.kinds.weight, net.scaled.table, net.scaled.table * net.scaled.scale):
            assert told(answer, own.detach().numpy()) == 0, asked
    assert sum(len(layer._forward_hooks) for layer in net.modules()) == 0


def test_following_a_run_tells_the_models_own_numbers_one_operation_at_a_time(torch):
    import gc
    import weakref

    if agents._following() is None:
        pytest.skip("this PyTorch cannot be followed")
    nn = torch.nn
    torch.manual_seed(12)
    net = nn.Linear(4, 3)
    net.register_buffer("places", torch.arange(6).reshape(1, 6))
    x = torch.randn(2, 4)
    own = agents._OwnNumbers(net, torch)
    with torch.no_grad(), own.following():
        weight, places = net.weight, net.places
        # Made from nothing, from a shape alone, or from the model's own numbers alone: the model's own.
        made = {
            "arange": torch.arange(x.shape[1]),
            "zeros like": torch.zeros_like(x),
            "new zeros": x.new_zeros(3),
            "a part": places[:, :3],
            "doubled": weight * 2.0,
            "looked up": weight.t()[torch.arange(2)],
            "cut to a shape": net.bias.expand_as(x[:, :3]),
            "another name": (weight * 2.0).data,
        }
        made["written by its own"] = made["doubled"].clone()
        made["written by its own"] += weight
        made["given out into"] = torch.add(weight, weight, out=torch.empty(3, 4))
        for name, tensor in made.items():
            assert own.holds(tensor), name
        # Whatever read the input, even once, is not; nor is memory the input was written into.
        mixed = {
            "the input": x,
            "a layer": net(x),
            "added": weight[:2] + x,
            "chosen by the input": torch.where(x[:, :3] > 0, net.bias, torch.zeros(3)),
            "put in by hand": torch.tensor([1.0, 2.0]),
        }
        written = weight.clone()
        written += x[0]
        filled = torch.zeros(2, 4)
        filled[:1] = x[:1]
        part = filled[1:]  # the same memory as what was written into
        net.places[:, :1] = x[:1, :1].long()  # into a buffer of the model itself
        mixed.update({"written with the input": written, "filled from the input": filled, "beside what was filled": part, "the buffer": net.places, "a part of the buffer": places[:, 3:]})
        for name, tensor in mixed.items():
            assert not own.holds(tensor), name
        assert own.holds(weight) and own.holds(net.bias) and not own.lost
        gone = weakref.ref(weight * 3.0)
        kept = weight * 3.0
    gc.collect()
    assert gone() is None  # followed, and not kept alive for it
    assert hasattr(kept, agents._OWN_MARK)
    own.done()
    assert not hasattr(kept, agents._OWN_MARK) and not any(hasattr(tensor, agents._OWN_MARK) for tensor in made.values())


def test_a_run_that_cannot_be_followed_to_its_end_sends_no_step(torch, monkeypatch):
    nn = torch.nn
    torch.manual_seed(10)
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2)).eval()
    nodes = [
        {"id": "first", "op": "dense", "units": 3, "path": "0", "args": {}},
        {"id": "last", "op": "dense", "units": 2, "path": "1", "args": {}},
    ]
    agent = module_agent(net, nodes, head="last")
    assert steps_in(agent.run({"kind": "trace", "payload": {"input": ROW}})) == ["first", "last"]

    def confused(self, func, args, kwargs, out):
        raise RuntimeError("an operation of a kind never seen")

    monkeypatch.setattr(agents._OwnNumbers, "_saw", confused)
    answer = agent.run({"kind": "trace", "payload": {"input": ROW}})
    # The model ran as it would have, and its prediction is its own; of its steps, nothing can be told apart.
    assert answer["frames"] == [] and {item["node_id"]: item["reason"] for item in answer["withheld"]} == {"first": agents._NOT_FOLLOWED, "last": agents._NOT_FOLLOWED}
    with torch.no_grad():
        np.testing.assert_allclose(answer["prediction"], net(torch.tensor([[ROW[f"f{i}"] for i in range(4)]])).numpy().reshape(-1), rtol=1e-6)


def test_what_a_run_writes_into_the_models_own_memory_is_no_longer_the_models_own(torch, telling):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.zeros(1, 4))
            self.body = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            self.state.copy_(x)  # the input, kept in a buffer of the module
            return self.head(self.body(self.state))

    torch.manual_seed(6)
    net = Net().eval()
    nodes = [
        {"id": "body", "op": "dense", "units": 4, "path": "body", "args": {}},
        {"id": "head", "op": "dense", "units": 2, "path": "head", "args": {}},
    ]
    agent = module_agent(net, nodes, head="head")
    traced = agent.run({"kind": "trace", "payload": {"input": ROW}})
    assert [frame["node_id"] for frame in traced["frames"]] == ["body", "head"] and "withheld" not in traced
    # What is known of a call when PyTorch does not say what a module was called with by name (before 2.0).
    own = agents._OwnNumbers(net, torch)
    weight, fresh = net.body.weight, torch.ones(1, 4)
    assert own.called_with((), {}) and own.called_with((weight,), {}) and own.called_with((weight[:2], 3), {"scale": 2.0})
    assert not own.called_with((weight, fresh), {}) and not own.called_with((weight,), {"mask": fresh})
    assert not own.called_with((3,), {}) and not own.called_with((object(),), {})  # nothing to tell by
    assert own.called_with((weight,), None) and not own.called_with((), None) and not own.called_with((fresh,), None)
    assert own.holds(net.state)  # nothing has written into it since this was made
    own.settle((fresh,))
    assert own.holds(fresh) and own.called_with((fresh, weight), {})
    fresh += 1.0  # written in place since: no longer known to be the model's own
    assert not own.holds(fresh)


# --- what a module keeps beside its parameters and buffers, and from one look to the next (a later review) ---

TEXT_OF_SIX = [{"name": "review", "type": "text", "max_length": 6, "vocab_size": 9}]
IDS = [3, 1, 4, 1, 5, 2]


def kept_net(torch, places):
    """Word vectors with what ``places`` gives added to them: ``places`` is called with the input every time."""
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.words = nn.Embedding(9, 4)
            self.places = places
            self.head = nn.Linear(4, 2)

        def forward(self, ids):
            words = self.words(ids)
            given = self.places(words)
            for one in given if isinstance(given, tuple) else (given,):
                words = words + one
            return self.head(words.mean(dim=1))

    return Net().eval()


def kept_nodes(*places):
    return [
        {"id": "review", "op": "input", "units": 6, "feature": "review", "args": {"feature_type": "text"}},
        {"id": "words", "op": "embedding", "units": 24, "path": "words", "args": {"shape": [6, 4]}},
        *[{"id": name, "op": "embedding", "units": 24, "path": f"places.{name}" if len(places) > 1 else "places", "args": {"shape": [6, 4]}} for name in places],
        {"id": "head", "op": "dense", "units": 2, "path": "head", "args": {}},
    ]


def test_what_a_module_worked_out_from_its_weights_once_and_kept_is_its_own_at_every_look(torch, telling):
    nn = torch.nn

    class Places(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.randn(1, 6, 4))
            self.cache = None

        def forward(self, words):
            if self.cache is None:
                self.cache = self.table * 2.0  # worked out from what it learned, the first time, and kept
            return self.cache

    torch.manual_seed(14)
    net = kept_net(torch, Places())
    agent = module_agent(net, kept_nodes("places"), TEXT_OF_SIX, "binary_classification", head="head")
    if telling == "hooks alone":
        # Through hooks alone, a module that is called with the input and makes something there is taken to have
        # computed it from the input: the look that makes the table sends it. Every look after it does not.
        agent.run({"kind": "trace", "payload": {"input": {"review": IDS}}})
    table = net.places.table.detach().numpy()
    for look in range(3):
        for asked, answer in looks(agent, {"review": IDS}, "places").items():
            alone = asked == ALONE
            assert steps_in(answer) == ([] if alone else ["review", "words", "head"]), (look, asked)
            assert {item["node_id"]: item["reason"] for item in answer["withheld"]} == {"places": agents._NOT_COMPUTED}, (look, asked)
            assert told(answer, table) == 0 and told(answer, table * 2.0) == 0, (look, asked)
    assert not hasattr(net.places.cache, agents._OWN_MARK)


def test_a_tensor_a_module_keeps_as_a_plain_attribute_is_its_own_at_every_look(torch, telling):
    nn = torch.nn

    class Kept(nn.Module):
        def __init__(self, give):
            super().__init__()
            self.give = give
            self.table = torch.randn(1, 6, 4)  # made when the module was built: no parameter, and no buffer
            self.more = {"scales": torch.randn(1, 6, 4)}  # and one in a dict
            self.listed = [torch.randn(1, 6, 4)]  # and one in a list

        def forward(self, words):
            if self.give == "whole":
                return self.table
            if self.give == "part":
                return self.table[:, : words.shape[1]]
            return self.listed[0] * self.more["scales"]

    class Each(nn.Module):
        def __init__(self):
            super().__init__()
            for give in GIVEN:
                setattr(self, give, Kept(give))

        def forward(self, words):
            return tuple(getattr(self, give)(words) for give in GIVEN)

    GIVEN = ("whole", "part", "made")
    torch.manual_seed(15)
    net = kept_net(torch, Each())
    agent = module_agent(net, kept_nodes(*GIVEN), TEXT_OF_SIX, "binary_classification", head="head")
    each = net.places
    secrets = {"whole": each.whole.table, "part": each.part.table, "made": each.made.listed[0] * each.made.more["scales"]}
    # Through hooks alone only what a module gives as it keeps it can be told: what it makes of it on the way cannot.
    kept_back = ["whole", "part"] if telling == "hooks alone" else list(GIVEN)
    for look in range(3):
        for asked, answer in looks(agent, {"review": IDS}, "whole").items():
            withheld = {item["node_id"]: item["reason"] for item in answer["withheld"]}
            assert withheld == {name: agents._NOT_COMPUTED for name in (["whole"] if asked == ALONE else kept_back)}, (look, asked)
            assert not set(steps_in(answer)) & set(kept_back), (look, asked)
            for name in kept_back:
                assert told(answer, secrets[name].detach().numpy()) == 0, (look, asked, name)
    with torch.no_grad():
        np.testing.assert_allclose(agent.run({"kind": "trace", "payload": {"input": {"review": IDS}}})["prediction"], net(torch.tensor([IDS])).numpy().reshape(-1), rtol=1e-6)


def test_what_a_module_is_called_with_is_the_input_though_the_module_kept_it(torch, telling):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)
            self.last = None

        def forward(self, x):
            self.last = x  # what it was last called with, as a plain attribute
            return self.head(self.body(x))

    given = {}

    def prepare(row):
        # The same tensor for the same row, every time it is asked for.
        return given.setdefault(tuple(sorted(row.items())), torch.tensor([[row[f"f{i}"] for i in range(4)]]))

    torch.manual_seed(17)
    net = Net().eval()
    nodes = [
        {"id": "body", "op": "dense", "units": 4, "path": "body", "args": {}},
        {"id": "head", "op": "dense", "units": 2, "path": "head", "args": {}},
    ]
    agent = module_agent(net, nodes, head="head", prepare=prepare)
    with torch.no_grad():
        body = net.body(prepare(ROW)).numpy().reshape(-1)
    for look in range(3):
        traced = agent.run({"kind": "trace", "payload": {"input": ROW}})
        assert steps_in(traced) == ["body", "head"] and "withheld" not in traced, look
        np.testing.assert_allclose(traced["frames"][0]["value"], body, rtol=1e-6)
    assert net.last is prepare(ROW)


def test_a_module_agent_started_with_a_structure_reads_only_its_steps(torch):
    nn = torch.nn
    torch.manual_seed(8)
    net = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.Linear(3, 2)).eval()
    nodes = [
        {"id": "1.Gemm", "op": "Gemm", "units": 3, "path": "1.Gemm", "args": {"tensor": {"shape": [1, 3]}}},
        {"id": "2.Gemm", "op": "Gemm", "units": 2, "path": "2.Gemm", "args": {"tensor": {"shape": [1, 2]}}},
        {"id": "extra", "op": "Flatten", "units": 4, "path": "0.Flatten", "args": {"tensor": {"shape": [1, 4]}}},
        {"id": "probabilities", "op": "Softmax", "units": 2, "path": "probabilities", "args": {}},  # Kaimei's own
    ]
    spec = {"family": "graph", "graph": {"values": {"1.Gemm": {"shape": [1, 3]}, "2.Gemm": {"shape": [1, 2]}}}}
    answer = module_agent(net, nodes, head="2.Gemm", structure=spec).run({"kind": "trace", "payload": {"input": ROW}})
    assert [frame["node_id"] for frame in answer["frames"]] == ["1.Gemm", "2.Gemm"]
    assert [(item["node_id"], item["reason"]) for item in answer["withheld"]] == [("extra", agents._NOT_HELD)]
    plain = module_agent(net, nodes, head="2.Gemm").run({"kind": "trace", "payload": {"input": ROW}})
    assert [frame["node_id"] for frame in plain["frames"]] == ["1.Gemm", "2.Gemm", "extra"] and "withheld" not in plain


# --- item 2: an answer that cannot be sent ------------------------------------------------------


@pytest.fixture
def client():
    made = Kaimei(api_key="km_test_x", base_url=BASE, max_retries=0)
    yield made
    made.close()


@pytest.fixture
def api():
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


def working(api, jobs, stop, features, document=None, task="regression"):
    """An API that hands out ``jobs`` one poll at a time, then asks the agent to stop."""
    queue = list(jobs)
    api.get(MODEL).mock(return_value=httpx.Response(200, json={"object": "model", "id": "m", "task": task, "features": features}))
    reply = httpx.Response(200, json=document) if document else httpx.Response(404, json={"error": {"type": "not_found_error", "message": "none"}})
    api.get(f"{MODEL}/structure").mock(return_value=reply)
    api.post(f"{MODEL}/agent/sessions").mock(return_value=httpx.Response(201, json={"id": "s1", "model_id": "m", "capabilities": ["predict"]}))

    def poll(request):
        if queue:
            return httpx.Response(200, json={"job": queue.pop(0)})
        stop.set()
        return httpx.Response(200, json={"job": None})

    api.post(f"{MODEL}/agent/sessions/s1/poll").mock(side_effect=poll)
    answers = api.post(url__regex=rf".*{MODEL}/agent/jobs/[^/]+/result").mock(return_value=httpx.Response(200, json={"job": None}))
    api.delete(f"{MODEL}/agent/sessions/s1").mock(return_value=httpx.Response(200, json={"deleted": True}))
    return answers


def sent(answers):
    return [(call.request.url.path.split("/")[-2], json.loads(call.request.content)) for call in answers.calls]


def test_a_trace_of_values_too_large_to_square_is_answered_and_the_next_job_too(client, api):
    # review r3, finding 4: y = Exp(x) in double precision, traced at x = 400. Its length overflowed to infinity,
    # which JSON cannot hold, and the agent ended with the next job never asked for.
    graph = helper.make_graph(
        [helper.make_node("Exp", ["x"], ["y"], name="exp")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    document = {
        "scene": {"nodes": [{"id": "in", "op": "input", "units": 1, "args": {"features": ["f0"]}}, {"id": "y", "op": "Exp", "units": 1, "inputs": ["in"]}]},
        "head": {"vector_node": "y"},
    }
    jobs = [
        {"id": "trace", "kind": "trace", "payload": {"input": {"f0": 400.0}}},
        {"id": "watch", "kind": "activations", "payload": {"input": {"f0": 400.0}, "nodes": ["y"]}},
        {"id": "next", "kind": "predict", "payload": {"rows": [{"f0": 1.0}]}},
    ]
    stop = threading.Event()
    answers = working(api, jobs, stop, [{"name": "f0", "type": "numeric"}], document)
    agent = serve(client, "churn", onnx=model.SerializeToString(), stop=stop)
    answered = dict(sent(answers))
    assert list(answered) == ["trace", "watch", "next"] and agent.answered == 3
    assert all(body["ok"] is True for body in answered.values())
    frame = {item["node_id"]: item for item in answered["trace"]["result"]["frames"]}["y"]
    assert frame["value"] == [pytest.approx(np.exp(400.0))]
    assert frame["summary"]["l2"] == pytest.approx(np.exp(400.0)) and frame["summary"]["max_abs"] == pytest.approx(np.exp(400.0))
    assert answered["next"]["result"]["predictions"] == [[pytest.approx(np.e)]]


def test_statistics_of_values_near_the_largest_float_are_numbers_an_answer_can_hold():
    huge = np.array([3e307, -1.2e308, 9e307, 1.7e308, -4e307, 2e306])
    summary = agents._summary(huge, np)
    assert summary["max_abs"] == 1.7e308 and summary["mean_abs"] == pytest.approx(np.abs(huge / 1e300).mean() * 1e300)
    assert summary["l2"] == agents._LARGEST  # longer than any float: the largest one is sent
    assert agents._summary(np.array([3e200, 4e200]), np)["l2"] == pytest.approx(5e200)
    assert agents._summary(np.array([float("nan"), float("inf"), 1.0]), np)["max_abs"] == 0.0  # not a number: nought
    ordinary = np.array([0.5, -2.0, 1.5])
    assert agents._summary(ordinary, np) == {"size": 3, "max_abs": 2.0, "mean_abs": float(np.abs(ordinary).mean()), "l2": float(np.linalg.norm(ordinary)), "active": 3, "argmax": 1}
    # Maps, a table of positions and a plain vector, in full: every statistic of each is a number.
    cube = np.resize(huge, (6, 4, 3))
    for step in (
        {"id": "map", "units": cube.size, "args": {"shape": [6, 4, 3]}},
        {"id": "attention", "units": cube.size, "args": {"shape": [6, 4, 3], "role": "attention"}},
        {"id": "table", "units": cube.size, "args": {"shape": [12, 6]}},
        {"id": "vector", "units": cube.size, "args": {}},
    ):
        stage = stage_of(step, cube, detailed=True, units=64, channels=3, max_side=2)
        json.dumps(stage, allow_nan=False)
        assert stage["summary"]["max_abs"] == 1.7e308
    table = stage_of({"id": "table", "units": 4, "args": {"shape": [2, 2]}}, np.array([3e200, 4e200, 0.0, 1e200]), detailed=True)
    assert table["rows"] == [pytest.approx(np.sqrt(12.5) * 1e200), pytest.approx(np.sqrt(0.5) * 1e200)]
    wide = agents._picture(np.array([-1.5e308, 0.0, 1.5e308]), np)
    assert list(base64.b64decode(wide["data"])) == [0, 128, 255] and wide["range"] == [-1.5e308, 1.5e308]
    shrunk = agents._shrunk(np.full((4, 4), 1.5e308), 2, np)
    assert shrunk.shape == (2, 2) and np.allclose(shrunk, 1.5e308)


def test_an_answer_that_cannot_be_written_out_is_a_failure_and_the_next_job_is_answered(client, api, monkeypatch):
    jobs = [{"id": "first", "kind": "predict", "payload": {"rows": [{"x": 1}]}}, {"id": "second", "kind": "predict", "payload": {"rows": [{"x": 2}]}}]
    stop = threading.Event()
    answers = working(api, jobs, stop, [{"name": "x", "type": "numeric"}])
    lines = []
    agent = Agent(client, "churn", lambda rows: [0.5] * len(rows), log=lines.append)
    run = agent.run
    # Whatever puts it there, a number JSON cannot hold must not end the agent.
    monkeypatch.setattr(agent, "run", lambda job: {"predictions": [[float("inf")]]} if job["id"] == "first" else run(job))
    agent.serve(stop)
    answered = sent(answers)
    assert [name for name, _body in answered] == ["first", "second"] and agent.answered == 2
    assert answered[0][1]["ok"] is False and answered[0][1]["error"].startswith("The agent's answer could not be sent: ValueError: Infinity")
    assert answered[1][1] == {"ok": True, "result": {"predictions": [[0.5]]}}
    assert any(line.startswith("predict not sent: ValueError: Infinity") for line in lines)


def test_nothing_that_goes_wrong_with_an_answer_ends_the_loop(client, api, monkeypatch):
    # Four jobs, each going wrong in its own way, and a fifth that is answered as if nothing had happened.
    jobs = [{"id": name, "kind": "predict", "payload": {"rows": [{"x": at}]}} for at, name in enumerate(("lost", "garbled", "unlogged", "failed", "fine"))]
    stop = threading.Event()
    answers = working(api, jobs, stop, [{"name": "x", "type": "numeric"}])
    lines = []

    def log(line):
        lines.append(line)
        if sum(item.startswith("predict") for item in lines) == 3:
            raise OSError("the log is full")  # the line about the third job cannot be written

    def predict(rows):
        if rows[0]["x"] == 1:
            raise RuntimeError("it cannot say \ud83d what went wrong")  # text that cannot be written as UTF-8
        if rows[0]["x"] == 3:
            raise RuntimeError("the model fell over")
        return [0.5] * len(rows)

    agent = Agent(client, "churn", predict, log=log)
    send = agent._send

    def failing(path, body):
        if path.split("/")[-2] in ("lost", "failed"):
            raise RuntimeError("the wire is cut")  # not an error of Kaimei's: one like it used to end the agent
        return send(path, body)

    monkeypatch.setattr(agent, "_send", failing)
    agent.serve(stop)
    assert agent.answered == 5 and stop.is_set()
    answered = dict(sent(answers))
    assert list(answered) == ["garbled", "unlogged", "fine"]
    assert answered["garbled"] == {"ok": False, "error": "RuntimeError: it cannot say ? what went wrong"}
    assert answered["unlogged"] == answered["fine"] == {"ok": True, "result": {"predictions": [[0.5]]}}
    said = [line for line in lines if line.startswith("predict")]
    # An answer that could not be sent, and then its failure neither; a failure that could not be sent.
    assert said[0].startswith("predict not sent: RuntimeError: the wire is cut; and whoever asked could not be told (RuntimeError: the wire is cut)")
    assert said[3].startswith("predict failed: RuntimeError: the model fell over; and whoever asked could not be told (RuntimeError: the wire is cut)")


# --- item 3: the name a value kept at home is asked for by ----------------------------------------


def test_a_value_kept_at_home_is_asked_for_by_the_files_own_name_for_the_input():
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "state"], ["y"], name="mul")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4]), helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    # As a summary upload writes it: the structure's own names for the inputs, and a fixed value as its shape only.
    spec = {
        "family": "graph",
        "graph": {
            "inputs": [
                {"name": "in.x", "dtype": "float32", "shape": [1, 4], "batch_axis": 0, "from": {"features": ["f0", "f1", "f2", "f3"]}},
                {"name": "in.state", "dtype": "float32", "shape": [1, 4], "from": {"value": {"$shape": [1, 4], "dtype": "float32"}}},
            ]
        },
    }
    agent = Agent(OnKaimei(None), "m", onnx=model, structure=spec)
    with pytest.raises(ValueError) as caught:
        agent.run({"kind": "predict", "payload": {"rows": [ROW]}})
    said = str(caught.value)
    assert "The file's input 'state' is always given a value that the upload kept at home" in said
    assert "inputs={'state': lambda row:" in said and "in.state" not in said
    # What it says to do works: inputs= goes by the file's own names.
    state = np.array([[2.0, 0.5, -1.0, 3.0]], dtype=np.float32)
    fed = Agent(OnKaimei(None), "m", onnx=model, structure=spec, inputs={"state": lambda row: state})
    x = np.array([ROW[f"f{i}"] for i in range(4)], dtype=np.float32)
    np.testing.assert_allclose(fed.run({"kind": "predict", "payload": {"rows": [ROW]}})["predictions"], [x * state[0]], rtol=1e-6)


# --- item 4: an encoder-decoder, whose decoder reads what it has written so far ---------------------

SEQ = np.random.default_rng(21)
READ, WRITE, WORDS = (SEQ.normal(size=shape).astype(np.float32) for shape in ((12, 4), (12, 4), (4, 12)))
SEQ_FEATURES = [
    {"name": "text", "type": "tokens", "max_length": 5, "vocab_size": 12},
    # What the decoder has written so far: Kaimei keeps it, and it starts as the token the decoder starts from.
    {"name": "decoder", "type": "tokens", "max_length": 4, "generated": True, "baseline": [0]},
]


def seq2seq_file(reads="input_ids", writes="decoder_input_ids", length=5):
    """A stand-in for an encoder-decoder: what the encoder read, pooled, added to each token the decoder has written.

    ``reads`` and ``writes`` are what the file calls its two inputs, and ``length`` how many tokens the first holds.
    """
    nodes = [
        helper.make_node("Gather", ["READ", reads], ["read"], name="encoder/embed"),
        helper.make_node("ReduceMean", ["read"], ["memory"], axes=[1], keepdims=1, name="encoder/pool"),
        helper.make_node("Gather", ["WRITE", writes], ["written"], name="decoder/embed"),
        helper.make_node("Add", ["written", "memory"], ["state"], name="decoder/attend"),
        helper.make_node("MatMul", ["state", "WORDS"], ["logits"], name="head"),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(reads, TensorProto.INT64, ["batch", length]), helper.make_tensor_value_info(writes, TensorProto.INT64, ["batch", 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4, 12])],
        [numpy_helper.from_array(value, name) for name, value in (("READ", READ), ("WRITE", WRITE), ("WORDS", WORDS))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    return model


def next_token_scores(text, decoded, length=5):
    """The model by hand: its scores for the token after the last one the decoder has written."""
    ids = np.zeros(length, dtype=int)
    ids[: len(text)] = text
    so_far = np.zeros(4, dtype=int)
    so_far[: len(decoded)] = decoded
    state = WRITE[so_far] + READ[ids].mean(axis=0, keepdims=True)
    return (state @ WORDS)[len(decoded) - 1]


def seq2seq_structure(decoder_input):
    """The structure the importer reads from the file, its decoder's input described as ``decoder_input`` says."""
    from kaimei import graphs
    from kaimei.structure import finalize_spec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = graphs.from_onnx(
            seq2seq_file(), task="sequence_to_sequence", inputs={"input_ids": {"feature": "text", "part": "ids"}, "decoder_input_ids": decoder_input}
        )
    return finalize_spec(raw)


def generated_by_hand():
    """The stand-in's structure with its decoder's input written ``"from": {"generated": true}`` and no feature for
    it: a form the importer no longer writes (the API refuses it), which an agent may still be handed."""
    extraction = seq2seq_structure({"feature": "decoder", "part": "ids", "pad_id": 0})
    spec = extraction.spec
    for item in spec["graph"]["inputs"]:
        if item["from"].get("feature") == "decoder":
            item["from"] = {"generated": True, "pad_id": 0}
    spec["features"] = [feature for feature in spec["features"] if feature["name"] != "decoder"]
    spec["feature_names"] = [name for name in spec.get("feature_names") or [] if name != "decoder"]
    spec.pop("check", None)
    return extraction


#: The steps of the stand-in, as the importer names them, in the order they run.
SEQ_STEPS = ["encoder.embed", "encoder.pool", "decoder.embed", "decoder.attend", "head"]


def seq2seq_agents(features=SEQ_FEATURES):
    as_a_feature = seq2seq_structure({"feature": "decoder", "part": "ids", "pad_id": 0})
    # Hinted as generated, the importer writes the same binding, marks the feature and says the decoder.
    hinted = seq2seq_structure({"generated": True, "pad_id": 0})
    assert [item["from"] for item in hinted.spec["graph"]["inputs"]][1] == {"feature": "decoder", "part": "ids", "pad_id": 0}
    assert hinted.spec["features"][1] == SEQ_FEATURES[1]
    assert hinted.spec["decoder"] == {"feature": "decoder", "start_token": 0, "pad_token": 0, "max_length": 4}
    as_generated = generated_by_hand()
    assert [item["from"] for item in as_generated.spec["graph"]["inputs"]][1] == {"generated": True, "pad_id": 0}
    values = as_a_feature.spec["graph"]["values"]
    assert list(values) == SEQ_STEPS == list(as_generated.spec["graph"]["values"]) == list(hinted.spec["graph"]["values"])
    # The structure as Kaimei shows it: the model's two texts, and each step under the name the upload gave it.
    nodes = [
        {"id": "text", "op": "input", "units": 5, "feature": "text", "args": {"feature_type": "tokens"}},
        {"id": "decoder", "op": "input", "units": 4, "feature": "decoder", "args": {"feature_type": "tokens"}},
        *[
            {"id": name, "op": "x", "units": int(np.prod(item["shape"])), "args": {"source": item["source"], "tensor": {"shape": item["shape"]}}}
            for name, item in values.items()
        ],
    ]
    document = {"scene": {"nodes": nodes}, "head": {"vector_node": "head", "link": "softmax"}}

    def started(**options):
        return Agent(OnKaimei(document, features, "sequence_to_sequence"), "m", onnx=seq2seq_file(), **options)

    return {
        "without a structure": started(),
        "with the structure, its decoder's input a feature": started(structure=as_a_feature),
        "with the structure, its decoder's input hinted as written by the model": started(structure=hinted),
        'with the structure, its decoder\'s input {"generated": true}': started(structure=as_generated),
    }


def test_an_encoder_decoder_is_fed_what_it_has_written_with_a_structure_as_without_one():
    agents = seq2seq_agents()
    rows = [{"text": [5, 9, 3, 7, 1], "decoder": [0]}, {"text": [7, 3], "decoder": [0, 11, 4]}, {"text": [2, 2, 2, 2], "decoder": [0, 5, 5, 8]}]
    want = [next_token_scores(row["text"], row["decoder"]) for row in rows]
    answers = {}
    for name, agent in agents.items():
        said = answers[name] = {"predict": agent.run({"kind": "predict", "payload": {"rows": rows}})}
        # A prediction is the scores of the next token, one for each word: not every position's, laid end to end.
        assert said["predict"]["raw"] is True and [len(line) for line in said["predict"]["predictions"]] == [12, 12, 12], name
        np.testing.assert_allclose(said["predict"]["predictions"], want, rtol=1e-5, atol=1e-6, err_msg=name)
        for kind in ("trace", "activations"):
            said[kind] = [agent.run({"kind": kind, "payload": {"input": row}}) for row in rows]
            for row, answer, scores in zip(rows, said[kind], want):
                answer.pop("forward_ms", None)
                assert steps_in(answer) == ["text", "decoder", *SEQ_STEPS] and "withheld" not in answer, (name, kind)
                np.testing.assert_allclose(answer["prediction"], scores, rtol=1e-5, atol=1e-6, err_msg=name)
        frames = {frame["node_id"]: frame["value"] for frame in said["trace"][1]["frames"]}
        assert frames["decoder"] == [0.0, 11.0, 4.0, -1.0]
        np.testing.assert_allclose(frames["decoder.embed"][:12], WRITE[[0, 11, 4]].reshape(-1), rtol=1e-6, err_msg=name)
        # Generation is Kaimei asking for one prediction after another, the decoder's feature a token longer each time.
        decoded = [0]
        for _step in range(3):
            scores = agent.run({"kind": "predict", "payload": {"rows": [{"text": [5, 9, 3], "decoder": decoded}]}})["predictions"][0]
            assert int(np.argmax(scores)) == int(np.argmax(next_token_scores([5, 9, 3], decoded))), name
            decoded = [*decoded, int(np.argmax(scores))]
        said["generation"] = decoded
        # A row that does not hold the feature yet is the decoder's first token, as a prediction on Kaimei has it.
        first = agent.run({"kind": "predict", "payload": {"rows": [{"text": [5, 9, 3]}]}})["predictions"]
        np.testing.assert_allclose(first, [next_token_scores([5, 9, 3], [0])], rtol=1e-5, atol=1e-6, err_msg=name)
    assert len(set(tuple(said["generation"]) for said in answers.values())) == 1 and len(set(answers["without a structure"]["generation"])) > 1
    # Step for step, the same values: every answer of an agent started with a structure is the one started without's.
    plain = answers["without a structure"]
    for name, said in answers.items():
        assert json.dumps(said, sort_keys=True) == json.dumps(plain, sort_keys=True), name


def test_an_encoder_decoder_that_cannot_be_answered_for_says_why():
    # Nothing on Kaimei holds what the decoder has written: no feature of the model is marked as generated.
    bare = [SEQ_FEATURES[0], {"name": "decoder", "type": "tokens", "max_length": 4}]
    row = {"text": [5, 9, 3], "decoder": [0]}
    for name, agent in seq2seq_agents(bare).items():
        for kind, payload in (("predict", {"rows": [row]}), ("trace", {"input": row}), ("activations", {"input": row})):
            if name == "without a structure":
                # Which of the model's two texts each input reads is said nowhere: it is asked for, in plain words.
                with pytest.raises(ValueError, match="The file's input 'input_ids' takes whole numbers, and this model has 2 text features"):
                    agent.run({"kind": kind, "payload": payload})
            elif "generated" in name:
                with pytest.raises(ValueError, match="is given what the model has written so far") as caught:
                    agent.run({"kind": kind, "payload": payload})
                assert "inputs={'decoder_input_ids':" in str(caught.value), (name, kind)
            elif kind == "predict":
                # The structure names the feature, so the file is fed. Which token is the next cannot be told.
                with pytest.raises(ValueError, match="writes its answer token by token"):
                    agent.run({"kind": kind, "payload": payload})
            else:
                answer = agent.run({"kind": kind, "payload": payload})
                assert "prediction" not in answer and steps_in(answer) == ["text", "decoder", *SEQ_STEPS]
    # A row whose feature holds no token at all has no last token to read after.
    agent = seq2seq_agents()["without a structure"]
    with pytest.raises(ValueError, match="The row holds none"):
        agent.run({"kind": "predict", "payload": {"rows": [{"text": [5, 9, 3], "decoder": []}]}})
    # And a row that lacks a feature an input is made from is said to, by that feature's name.
    with pytest.raises(ValueError, match="The row has no value for the feature 'text'"):
        agent.run({"kind": "predict", "payload": {"rows": [{"decoder": [0]}]}})


def read_by(names, features, given=None):
    """Which feature, and which part of it, each whole-number input of a file is made from when no structure says."""
    bound = agents.bind_inputs([(name, np.dtype("int64"), [None, None]) for name in names], features, given)
    return {item["name"]: (item["source"]["feature"], item["source"]["part"]) for item in bound}


def test_two_texts_are_never_chosen_between_for_an_input_whose_name_does_not_say():
    # Nothing in ``src`` or ``tgt`` says which reads the text given and which what the decoder has written.
    with pytest.raises(ValueError) as caught:
        read_by(["src", "tgt"], SEQ_FEATURES)
    assert str(caught.value) == (
        "The file's input 'src' takes whole numbers, and this model has 2 text features. "
        "Say which it reads: inputs={'src': {'feature': ..., 'part': 'ids'}}."
    )
    refused = [
        (["input_ids", "tgt"], None, "input_ids"),  # two inputs for the token ids of the text given
        (["src", "tgt"], {"src": "text"}, "tgt"),  # the same, one of them said by the caller
        (["input_ids", "tokens", "decoder_input_ids"], None, "input_ids"),
        (["input_ids", "decoder_input_ids"], {"input_ids": "decoder"}, "decoder_input_ids"),  # two for what was written
        (["tgt"], None, "tgt"),  # one input and two texts: it is not given the one its name does not rule out
    ]
    for names, given, asked in refused:
        with pytest.raises(ValueError, match=f"The file's input '{asked}' takes whole numbers, and this model has 2 text features"):
            read_by(names, SEQ_FEATURES, given)
    # Two texts neither of which the model writes: ``decoder`` in a name says nothing of either.
    with pytest.raises(ValueError, match="The file's input 'input_ids' takes whole numbers"):
        read_by(["input_ids", "decoder_input_ids"], [SEQ_FEATURES[0], {"name": "decoder", "type": "tokens", "max_length": 4}])
    # Said for each, they are read as said.
    told_which = {"src": "text", "tgt": {"feature": "decoder", "part": "ids"}}
    assert read_by(["src", "tgt"], SEQ_FEATURES, told_which) == {"src": ("text", "ids"), "tgt": ("decoder", "ids")}


def test_an_agent_does_not_feed_a_decoder_the_text_when_the_files_inputs_do_not_say_whose_they_are():
    # Both inputs hold four whole numbers, so nothing but their names could tell them apart, and those say nothing.
    features = [{**SEQ_FEATURES[0], "max_length": 4}, SEQ_FEATURES[1]]
    row = {"text": [5, 9, 3], "decoder": [0, 11]}

    def started(**options):
        return Agent(OnKaimei(None, features, "sequence_to_sequence"), "m", onnx=seq2seq_file("src", "tgt", length=4), **options)

    agent = started()
    for kind, payload in (("predict", {"rows": [row]}), ("trace", {"input": row}), ("activations", {"input": row})):
        with pytest.raises(ValueError, match="The file's input 'src' takes whole numbers, and this model has 2 text features") as caught:
            agent.run({"kind": kind, "payload": payload})
        assert "inputs={'src':" in str(caught.value), kind
    # What it says to do works, and the decoder is fed what it has written.
    told_which = started(inputs={"src": "text", "tgt": "decoder"})
    scores = told_which.run({"kind": "predict", "payload": {"rows": [row]}})["predictions"]
    np.testing.assert_allclose(scores, [next_token_scores([5, 9, 3], [0, 11], length=4)], rtol=1e-5, atol=1e-6)


def test_a_model_whose_only_text_is_what_it_has_written_is_read_by_its_input_whatever_that_is_called():
    written = SEQ_FEATURES[1]
    assert read_by(["tokens"], [written]) == {"tokens": ("decoder", "ids")}
    assert read_by(["tokens", "mask"], [written]) == {"tokens": ("decoder", "ids"), "mask": ("decoder", "mask")}
    graph = helper.make_graph(
        [helper.make_node("Gather", ["WRITE", "tokens"], ["written"], name="embed"), helper.make_node("MatMul", ["written", "WORDS"], ["logits"], name="head")],
        "g",
        [helper.make_tensor_value_info("tokens", TensorProto.INT64, ["batch", 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4, 12])],
        [numpy_helper.from_array(WRITE, "WRITE"), numpy_helper.from_array(WORDS, "WORDS")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    agent = Agent(OnKaimei(None, [written], "sequence_to_sequence"), "m", onnx=model)
    rows = [{"decoder": [0]}, {"decoder": [0, 11, 4]}]
    scores = agent.run({"kind": "predict", "payload": {"rows": rows}})["predictions"]
    np.testing.assert_allclose(scores, [(WRITE[row["decoder"]] @ WORDS)[-1] for row in rows], rtol=1e-5, atol=1e-6)


def masked_file():
    """An encoder-decoder with the four inputs a T5 has: the masks say which tokens of each text are real."""
    nodes = [
        helper.make_node("Gather", ["READ", "input_ids"], ["read"], name="encoder/embed"),
        helper.make_node("Cast", ["attention_mask"], ["real"], to=TensorProto.FLOAT, name="encoder/real"),
        helper.make_node("Unsqueeze", ["real", "LAST"], ["real3"], name="encoder/real3"),
        helper.make_node("Mul", ["read", "real3"], ["counted"], name="encoder/counted"),
        helper.make_node("ReduceSum", ["counted", "ALONG"], ["memory"], keepdims=1, name="encoder/pool"),
        helper.make_node("Gather", ["WRITE", "decoder_input_ids"], ["written"], name="decoder/embed"),
        helper.make_node("Cast", ["decoder_attention_mask"], ["shown"], to=TensorProto.FLOAT, name="decoder/shown"),
        helper.make_node("Unsqueeze", ["shown", "LAST"], ["shown3"], name="decoder/shown3"),
        helper.make_node("Mul", ["written", "shown3"], ["seen"], name="decoder/seen"),
        helper.make_node("Add", ["seen", "memory"], ["state"], name="decoder/attend"),
        helper.make_node("MatMul", ["state", "WORDS"], ["logits"], name="head"),
    ]
    whole = [("input_ids", 5), ("attention_mask", 5), ("decoder_input_ids", 4), ("decoder_attention_mask", 4)]
    kept = (("READ", READ), ("WRITE", WRITE), ("WORDS", WORDS), ("LAST", np.array([2], dtype=np.int64)), ("ALONG", np.array([1], dtype=np.int64)))
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", length]) for name, length in whole],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4, 12])],
        [numpy_helper.from_array(value, name) for name, value in kept],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    return model


def test_inputs_named_as_an_encoder_decoders_are_still_bound_by_their_names():
    t5 = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"]
    assert read_by(t5, SEQ_FEATURES) == {
        "input_ids": ("text", "ids"),
        "attention_mask": ("text", "mask"),
        "decoder_input_ids": ("decoder", "ids"),
        "decoder_attention_mask": ("decoder", "mask"),
    }
    assert read_by(["encoder_input_ids", "encoder_attention_mask", "decoder_input_ids"], SEQ_FEATURES) == {
        "encoder_input_ids": ("text", "ids"),
        "encoder_attention_mask": ("text", "mask"),
        "decoder_input_ids": ("decoder", "ids"),
    }
    # And an agent run on such a file feeds each input its own text, and each mask its own text's real tokens.
    agent = Agent(OnKaimei(None, SEQ_FEATURES, "sequence_to_sequence"), "m", onnx=masked_file())
    rows = [{"text": [5, 9, 3, 7, 1], "decoder": [0]}, {"text": [7, 3], "decoder": [0, 11, 4]}, {"text": [2, 2, 2, 2], "decoder": [0, 5, 5, 8]}]
    want = [(WRITE[row["decoder"][-1]] + READ[row["text"]].sum(axis=0)) @ WORDS for row in rows]
    np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": rows}})["predictions"], want, rtol=1e-5, atol=1e-6)


def test_an_encoder_decoder_that_cannot_be_answered_for_is_a_failed_job_not_the_end_of_the_agent(client, api):
    spec = generated_by_hand()
    row = {"text": [5, 9, 3], "decoder": [0]}
    jobs = [{"id": kind, "kind": kind, "payload": {"rows": [row]} if kind == "predict" else {"input": row}} for kind in ("predict", "trace", "activations")]
    stop = threading.Event()
    answers = working(api, jobs, stop, [SEQ_FEATURES[0]], task="sequence_to_sequence")
    agent = Agent(client, "churn", onnx=seq2seq_file(), structure=spec)
    agent.serve(stop)
    answered = dict(sent(answers))
    assert list(answered) == ["predict", "trace", "activations"] and agent.answered == 3
    for kind, body in answered.items():
        assert body["ok"] is False and body["error"].startswith("ValueError: The structure's input 'in.decoder_input_ids' is given what the model has written so far"), kind
        assert "KeyError" not in body["error"] and "inputs={'decoder_input_ids':" in body["error"]


def test_an_encoder_decoder_behind_a_module_answers_with_the_next_tokens_scores(torch):
    nn = torch.nn

    class Writer(nn.Module):
        def __init__(self):
            super().__init__()
            self.read = nn.Embedding.from_pretrained(torch.tensor(READ), freeze=True)
            self.write = nn.Embedding.from_pretrained(torch.tensor(WRITE), freeze=True)
            self.words = nn.Linear(4, 12, bias=False)
            with torch.no_grad():
                self.words.weight.copy_(torch.tensor(WORDS.T))

        def forward(self, input_ids, decoder_input_ids):
            return self.words(self.write(decoder_input_ids) + self.read(input_ids).mean(dim=1, keepdim=True))

    def tensors(row):
        ids = torch.zeros(1, 5, dtype=torch.long)
        ids[0, : len(row["text"])] = torch.tensor(row["text"])
        so_far = torch.zeros(1, 4, dtype=torch.long)
        so_far[0, : len(row["decoder"])] = torch.tensor(row["decoder"])
        return ids, so_far

    agent = Agent(OnKaimei(None, SEQ_FEATURES, "sequence_to_sequence"), "m", module=Writer().eval(), prepare=tensors)
    rows = [{"text": [5, 9, 3, 7, 1], "decoder": [0]}, {"text": [7, 3], "decoder": [0, 11, 4]}]
    answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
    assert [len(line) for line in answer["predictions"]] == [12, 12]  # not four positions of twelve, end to end
    np.testing.assert_allclose(answer["predictions"], [next_token_scores(row["text"], row["decoder"]) for row in rows], rtol=1e-5, atol=1e-6)


def test_an_encoder_decoder_run_by_an_agent_is_the_one_kaimei_runs_itself():
    """Against the API's own engine, when ``KAIMEI_API_SRC`` names its ``src``: its predictions, and what it writes."""
    import os
    import sys

    source = os.environ.get("KAIMEI_API_SRC")
    if not source or not os.path.isdir(source):
        pytest.skip("KAIMEI_API_SRC does not name the Kaimei API's src")
    if source not in sys.path:
        sys.path.append(source)
    runners = pytest.importorskip("kaimei_api.engine.runners")
    from kaimei_api.engine.structure import compile_spec
    from kaimei_api.schemas.structure import GenerateIn
    from kaimei_api.services import generate as generation

    from kaimei._http import encode_json

    # The structure as the client's converter writes an encoder-decoder's: a feature for what the decoder has
    # written, marked as generated, and what its decoder starts from.
    extraction = seq2seq_structure({"feature": "decoder", "part": "ids", "pad_id": 0})
    spec = json.loads(json.dumps(extraction.spec, default=lambda value: value.tolist()))
    spec["decoder"] = {"feature": "decoder", "start_token": 0, "pad_token": 0, "max_length": 4}
    for feature in spec["features"]:
        if feature["name"] == "decoder":
            feature.update(generated=True, baseline=[0])
    graph = compile_spec(spec)
    engine = runners.build_runner(spec["features"], spec["task"], spec.get("outputs"), {}, structure=spec, graph=graph)
    rows = [{"text": [5, 9, 3, 7, 1], "decoder": [0]}, {"text": [7, 3], "decoder": [0, 11, 4]}, {"text": [2, 2, 2, 2], "decoder": [0, 5, 5, 8]}]
    X = engine.codec.encode_many(rows)
    want = engine.predict(X)
    assert want.shape == (3, 12)

    def written(runner):
        payload = GenerateIn(input={"text": [5, 9, 3]}, steps=3, temperature=1.0)
        told = generation._write(runner, runner.codec, payload, np.random.default_rng(5), 60.0, decoding=True, decoder=spec["decoder"])
        return told["tokens"], [step["probability"] for step in told["steps"]]

    tokens, chances = written(engine)
    for name, agent in seq2seq_agents(spec["features"]).items():

        def ask(kind, payload, agent=agent):
            return json.loads(encode_json(agent.run(json.loads(encode_json({"kind": kind, "payload": payload})))))

        runner = runners.build_runner(spec["features"], spec["task"], spec.get("outputs"), {"type": "agent"}, structure=spec, graph=graph, agent=ask)
        np.testing.assert_allclose(runner.predict(X), want, rtol=1e-5, atol=1e-7, err_msg=name)
        got, got_chances = written(runner)
        assert got == tokens, name
        np.testing.assert_allclose(got_chances, chances, rtol=1e-5, atol=1e-7, err_msg=name)


# --- item 5: inputs= beside structure= ----------------------------------------------------------------


def two_fixed_file():
    """``y = x * state + shift``: one input made from the row, and two that are always given the same values."""
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "state"], ["scaled"], name="mul"), helper.make_node("Add", ["scaled", "shift"], ["y"], name="add")],
        "g",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [None if name == "x" else 1, 4]) for name in ("x", "state", "shift")],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


STATE = np.array([[2.0, 0.5, -1.0, 3.0]], dtype=np.float32)
SHIFT = np.array([[0.25, -4.0, 1.5, 10.0]], dtype=np.float32)
#: The columns in an order of the structure's own, which is not the order of the model's features.
TURNED = ["f3", "f1", "f0", "f2"]


def two_fixed_structure(state):
    return {
        "family": "graph",
        "graph": {
            "inputs": [
                {"name": "in.x", "dtype": "float32", "shape": [1, 4], "batch_axis": 0, "from": {"features": TURNED}},
                {"name": "in.state", "dtype": "float32", "shape": [1, 4], "from": {"value": state}},
                {"name": "in.shift", "dtype": "float32", "shape": [1, 4], "from": {"value": {"dtype": "float32", "shape": [1, 4], "data": SHIFT.reshape(-1).tolist()}}},
            ]
        },
    }


def test_an_input_the_caller_names_is_fed_as_the_caller_says_and_every_other_as_the_structure_says():
    full = two_fixed_structure({"dtype": "float32", "shape": [1, 4], "data": STATE.reshape(-1).tolist()})
    summary = two_fixed_structure({"$shape": [1, 4], "dtype": "float32"})  # as weights="summary" sends a fixed value
    nodes = [
        {"id": "scaled", "op": "Mul", "units": 4, "args": {"tensor": {"shape": [1, 4]}}},
        {"id": "y", "op": "Add", "units": 4, "args": {"tensor": {"shape": [1, 4]}}},
    ]
    kaimei = OnKaimei({"scene": {"nodes": nodes}, "head": {"vector_node": "y"}})
    rows = [ROW, ZERO, {"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0}]
    want = [np.array([row[name] for name in TURNED], dtype=np.float32) * STATE[0] + SHIFT[0] for row in rows]

    def answers(agent):
        found = [agent.run({"kind": "predict", "payload": {"rows": rows}})]
        for kind in ("trace", "activations"):
            found += [agent.run({"kind": kind, "payload": {"input": row}}) for row in rows]
        for answer in found:
            answer.pop("forward_ms", None)
        return found

    whole = answers(Agent(kaimei, "m", onnx=two_fixed_file(), structure=full))
    np.testing.assert_allclose(whole[0]["predictions"], want, rtol=1e-6)
    # The structure of a summary upload cannot feed 'state', and says what to do about it.
    with pytest.raises(ValueError, match=r"inputs=\{'state': lambda row:"):
        Agent(kaimei, "m", onnx=two_fixed_file(), structure=summary).run({"kind": "predict", "payload": {"rows": rows}})
    # Doing it feeds that input, and leaves the other two as the structure has them: the columns in its order, and
    # the value it always gives 'shift'. Every answer is the one an agent with the whole structure gives.
    fed = Agent(kaimei, "m", onnx=two_fixed_file(), structure=summary, inputs={"state": lambda row: STATE})
    assert [item["name"] for item in fed._bindings()] == ["x", "shift", "state"]
    assert fed._bindings()[0]["source"] == {"features": TURNED} and "call" in fed._bindings()[2]
    told = answers(fed)
    assert json.dumps(told, sort_keys=True) == json.dumps(whole, sort_keys=True)
    assert steps_in(told[1]) == ["scaled", "y"] and "withheld" not in told[1]
    # Any way an input can be described is taken for that input alone.
    described = Agent(kaimei, "m", onnx=two_fixed_file(), structure=full, inputs={"x": [f"f{i}" for i in range(4)], "shift": {"zeros": True}})
    plain = [np.array([row[f"f{i}"] for i in range(4)], dtype=np.float32) * STATE[0] for row in rows]
    np.testing.assert_allclose(described.run({"kind": "predict", "payload": {"rows": rows}})["predictions"], plain, rtol=1e-6)


def test_an_input_named_that_is_not_one_is_refused_when_the_agent_is_made():
    full = two_fixed_structure({"dtype": "float32", "shape": [1, 4], "data": STATE.reshape(-1).tolist()})
    kaimei = OnKaimei(None)
    with pytest.raises(ValueError) as caught:
        Agent(kaimei, "m", onnx=two_fixed_file(), structure=full, inputs={"stat": lambda row: STATE})
    assert "inputs names 'stat', which is not an input of this model's file. Its inputs are: x, state, shift." in str(caught.value)
    assert "The structure this agent was started with feeds: x, state, shift." in str(caught.value)
    # The structure's own name for an input is the likeliest slip: the name that was meant is said.
    with pytest.raises(ValueError, match="inputs= goes by the file's own names: 'state' is what the structure calls 'in.state'"):
        Agent(kaimei, "m", onnx=two_fixed_file(), structure=full, inputs={"in.state": lambda row: STATE})
    # Without a structure a name that is no input fed nothing either, and said nothing.
    with pytest.raises(ValueError, match="inputs names 'X', which is not an input of this model's file. Its inputs are: x, state, shift."):
        Agent(kaimei, "m", onnx=two_fixed_file(), inputs={"X": [f"f{i}" for i in range(4)]})
    # An input of the file that the structure leaves out may be fed: it is the file's, and the file is what runs.
    partly = {"family": "graph", "graph": {"inputs": full["graph"]["inputs"][:2]}}
    agent = Agent(kaimei, "m", onnx=two_fixed_file(), structure=partly, inputs={"shift": lambda row: SHIFT})
    turned = np.array([ROW[name] for name in TURNED], dtype=np.float32)
    np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": [ROW]}})["predictions"], [turned * STATE[0] + SHIFT[0]], rtol=1e-6)


# --- item 7: why a body cannot be written as JSON ------------------------------------------------------


def test_what_cannot_be_written_as_json_is_said_for_what_it_is():
    from kaimei._http import encode_json

    # A number that is not finite: said as before. A missing value is still written as null.
    for body in ({"x": float("inf")}, {"x": float("nan"), "y": float("-inf")}, {"values": np.array([1.0, np.inf])}):
        with pytest.raises(ValueError, match="^Infinity can't be sent to Kaimei, because JSON can't hold it"):
            encode_json(body)
    assert json.loads(encode_json({"x": float("nan"), "y": np.float32(1.5)})) == {"x": None, "y": 1.5}
    # Text that is not text: half of a surrogate pair, which UTF-8 cannot hold. Said for what it is, and where.
    for body in ({"error": "it cannot say \ud83d what went wrong"}, {"rows": [{"name": "ok"}, {"name": "caf\udce9"}], "x": float("nan")}):
        with pytest.raises(ValueError) as caught:
            encode_json(body)
        said = str(caught.value)
        assert "Infinity" not in said and "holds a character that cannot be written as UTF-8" in said, said
        assert type(caught.value) is ValueError  # what was refused is refused as it was: with a ValueError
        said.encode("utf-8")  # and the message itself can be sent
    assert "U+D83D" in str(caught_for({"error": "it cannot say \ud83d what"})) and "right after '{\"error\":\"it cannot say '" in str(caught_for({"error": "it cannot say \ud83d what"}))
    assert "U+DCE9" in str(caught_for({"name": "caf\udce9"})) and "caf" in str(caught_for({"name": "caf\udce9"}))
    # A list that holds itself.
    loop: list = [1]
    loop.append(loop)
    assert "it holds itself" in str(caught_for({"rows": loop})) and "Infinity" not in str(caught_for({"rows": loop}))
    # What is accepted has not changed: a whole pair is text, and so is everything that was.
    assert json.loads(encode_json({"text": "\U0001f600 café", "n": 3, "items": np.arange(3)})) == {"text": "\U0001f600 café", "n": 3, "items": [0, 1, 2]}
    with pytest.raises(TypeError):
        encode_json({"x": object()})


def caught_for(body):
    from kaimei._http import encode_json

    with pytest.raises(ValueError) as caught:
        encode_json(body)
    return caught.value


# --- item 6: how much one run reads, by what the file says and not by what the structure claims -----------

MISSING = object()


def chain_file(steps=24, width=8):
    nodes, previous = [], "x"
    for index in range(steps):
        nodes.append(helper.make_node("Relu" if index % 2 else "Neg", [previous], [f"t{index}"], name=f"n{index}"))
        previous = f"t{index}"
    graph = helper.make_graph(
        nodes,
        "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", width])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, ["batch", width])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def chain_agent(units, steps=24, width=8, named=None):
    """An agent for a chain of steps of ``width`` values each, on Kaimei as a structure that says each holds ``units``."""
    features = [{"name": f"c{i}", "type": "numeric"} for i in range(width)]
    scene = [{"id": "in", "op": "input", "units": width, "args": {"features": [feature["name"] for feature in features]}}]
    for index in range(steps):
        step = {"id": f"t{index}", "op": "Relu", "args": {"tensor": {"shape": [1, width]}}}
        if named is not None:
            step["args"]["source"] = named
        if units is not MISSING:
            step["units"] = units
        scene.append(step)
    document = {"scene": {"nodes": scene}, "head": {"vector_node": f"t{steps - 1}"}}
    return Agent(OnKaimei(document, features), "m", onnx=chain_file(steps, width)), {f"c{i}": float(i) - 3.5 for i in range(width)}


def runs_of(agent, monkeypatch):
    """The tensors each run of the file is asked for from here on, the file's own output aside."""
    asked: list = []
    run = agent._file.run
    monkeypatch.setattr(agent._file, "run", lambda feeds, names: asked.append([name for name in dict.fromkeys(names) if name != "t23"]) or run(feeds, names))
    return asked


def test_one_run_reads_what_the_file_says_fits_whatever_the_structure_claims(monkeypatch):
    monkeypatch.setattr(agents, "FETCH_VALUES", 40)  # five tensors of eight values a run
    lots = [[f"t{index}" for index in range(start, min(start + 5, 23))] for start in range(0, 24, 5)]
    agent, row = chain_agent(8)
    # The file says how much each tensor holds: its inputs' open size is the row's, and nothing has to be asked.
    monkeypatch.setattr(agent._file, "shapes", lambda feeds, names: pytest.fail("the file's own shapes were enough"))
    asked = runs_of(agent, monkeypatch)
    honest = agent.run({"kind": "trace", "payload": {"input": row}})
    assert asked == lots and len(honest["frames"]) == 25
    # A structure can say anything of a step's size. What one run holds does not go by it.
    for claim in (0, -5, 10**15, "a great many", None, 3.5, [8], MISSING):
        agent, row = chain_agent(claim)
        asked = runs_of(agent, monkeypatch)
        answer = agent.run({"kind": "trace", "payload": {"input": row}})
        assert asked == lots, claim  # not one run of them all (a size of nought), nor a run for each (a huge one)
        assert steps_in(answer) == ["in"] and "withheld" not in answer, claim  # a step of another size is no step of the file's
        np.testing.assert_allclose(answer["prediction"], honest["prediction"])
    # A tensor that many steps name is read once, however many they are and whatever they say it holds.
    agent, row = chain_agent(10**15, named="t0")
    asked = runs_of(agent, monkeypatch)
    agent.run({"kind": "activations", "payload": {"input": row}})
    assert asked == [["t0"]]


def test_what_a_run_read_is_made_into_frames_before_the_next_run_is_made(monkeypatch):
    monkeypatch.setattr(agents, "FETCH_VALUES", 40)
    agent, row = chain_agent(8)
    summed = []
    summary = agents._summary
    monkeypatch.setattr(agents, "_summary", lambda flat, numpy: summed.append(1) or summary(flat, numpy))
    made_by_then = []
    run = agent._file.run
    monkeypatch.setattr(agent._file, "run", lambda feeds, names: made_by_then.append(len(summed)) or run(feeds, names))
    answer = agent.run({"kind": "trace", "payload": {"input": row}})
    # Five steps a run: each run's are frames, and their values let go, by the time the next run is asked for.
    assert made_by_then == [0, 5, 10, 15, 20] and len(answer["frames"]) == 25 == len(summed)


def test_a_tensor_whose_size_the_file_cannot_say_is_asked_for_its_shape_or_read_by_itself(monkeypatch):
    graph = helper.make_graph(
        [
            helper.make_node("NonZero", ["x"], ["where"], name="where"),  # as many columns as the row has values that are not nought
            helper.make_node("Cast", ["where"], ["spots"], to=TensorProto.FLOAT, name="spots"),
            helper.make_node("Relu", ["x"], ["y"], name="y"),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    features = [{"name": f"c{i}", "type": "numeric"} for i in range(8)]
    nodes = [
        {"id": "where", "op": "NonZero", "units": 0, "args": {"tensor": {"shape": [2, 6]}}},
        {"id": "spots", "op": "Cast", "units": 12, "args": {"tensor": {"shape": [2, 6]}}},
        {"id": "y", "op": "Relu", "units": 10**9, "args": {"tensor": {"shape": [1, 8]}}},
    ]
    document = {"scene": {"nodes": nodes}, "head": {"vector_node": "y"}}
    row = {f"c{i}": float(i % 4) for i in range(8)}  # two of the eight are nought
    monkeypatch.setattr(agents, "FETCH_VALUES", 20)

    def asked_of(agent):
        asked: list = []
        run = agent._file.run
        monkeypatch.setattr(agent._file, "run", lambda feeds, names: asked.append([name for name in dict.fromkeys(names) if name != "y"]) or run(feeds, names))
        return asked

    agent = Agent(OnKaimei(document, features), "m", onnx=model)
    feeds = agent._feeds(row)
    assert agent._file._declared_sizes(feeds) == {"y": 8}  # the file cannot say how many are not nought
    before = (len(model.graph.node), len(model.graph.output), len(agent._file._sessions))
    assert agent._file.shapes(feeds, ["where", "spots", "nothing of the file's"]) == {"where": (2, 6), "spots": (2, 6)}
    assert (len(agent._file.model.graph.node), len(agent._file.model.graph.output), len(agent._file._sessions)) == before
    asked = asked_of(agent)
    answer = agent.run({"kind": "trace", "payload": {"input": row}})
    # Twelve values and twelve are more than a run of twenty holds; twelve and eight are not.
    assert asked == [["where"], ["spots"]] and steps_in(answer) == ["spots"]
    np.testing.assert_array_equal(answer["frames"][0]["value"], np.array(np.nonzero(np.array([[row[f"c{i}"] for i in range(8)]]))).reshape(-1))
    # Where the shapes cannot be asked for, a tensor of a size nobody knows is read by itself: then it is known.
    agent = Agent(OnKaimei(document, features), "m", onnx=model)
    probes = []
    monkeypatch.setattr(agent._file, "shapes", lambda feeds, names: probes.append(list(names)) or {})
    asked = asked_of(agent)
    first = agent.run({"kind": "trace", "payload": {"input": row}})
    assert asked == [["where"], ["spots"], []] and probes == [["where", "spots"]] and first == answer
    asked.clear()
    assert agent.run({"kind": "trace", "payload": {"input": row}}) == answer
    assert asked == [["where"], ["spots"]] and len(probes) == 1  # as it was found to be, the next time
