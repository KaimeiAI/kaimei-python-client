"""An agent that runs the model's ONNX file: predictions, and every step of a structure read from that file."""

from __future__ import annotations

import json
import sys

import httpx
import pytest
import respx

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from kaimei import Kaimei  # noqa: E402
from kaimei import agent as agents  # noqa: E402
from kaimei.activations import decode  # noqa: E402
from kaimei.agent import Agent, main  # noqa: E402

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


def on_kaimei(api, features, nodes, task="multiclass_classification", head=None):
    api.get(MODEL).mock(return_value=httpx.Response(200, json={"object": "model", "id": "mdl_1", "task": task, "features": features}))
    document = {"object": "model.structure", "scene": {"nodes": nodes}, "head": head or {}}
    return api.get(f"{MODEL}/structure").mock(return_value=httpx.Response(200, json=document))


def file_of(nodes, inputs, outputs, constants):
    graph = helper.make_graph(nodes, "g", inputs, outputs, [numpy_helper.from_array(value, name) for name, value in constants.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def step(name, op, shape, native, source=None, **more):
    args = {"shape": list(shape), "tensor": {"shape": list(native), "dtype": "float32"}}
    if source:
        args["source"] = source
    args.update(more.pop("args", {}))
    return {"id": name, "op": op, "kind": more.pop("kind", "linear"), "units": int(np.prod(native)), "path": name, "args": args, **more}


# --- a table ---------------------------------------------------------------------------------

RNG = np.random.default_rng(0)
W1, B1 = RNG.normal(size=(3, 4)).astype(np.float32), RNG.normal(size=4).astype(np.float32)
W2, B2 = RNG.normal(size=(4, 2)).astype(np.float32), RNG.normal(size=2).astype(np.float32)


def table_file():
    return file_of(
        [
            helper.make_node("Gemm", ["X", "W1", "B1"], ["h"], name="fc1/Gemm"),
            helper.make_node("Relu", ["h"], ["r"], name="act/Relu"),
            helper.make_node("Gemm", ["r", "W2", "B2"], ["scores"], name="head/Gemm"),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", 2])],
        {"W1": W1, "B1": B1, "W2": W2, "B2": B2},
    )


TABLE_FEATURES = [{"name": name, "type": "numeric"} for name in ("tenure", "seats", "spend")]
TABLE_NODES = [
    *[{"id": name, "op": "input", "kind": "input", "units": 1, "feature": name} for name in ("tenure", "seats", "spend")],
    step("fc1.Gemm", "Gemm", [4], [1, 4], source="h", inputs=["tenure", "seats", "spend"]),
    step("r", "Relu", [4], [1, 4], kind="activation", inputs=["fc1.Gemm"]),  # its tensor's name in the file is its own
    step("head.Gemm", "Gemm", [2], [1, 2], source="scores", inputs=["r"]),
    # The softmax Kaimei adds to read the prediction: not a tensor of the file, so nothing is said of it.
    step("probabilities", "Softmax", [2], [1, 2], kind="output", inputs=["head.Gemm"]),
    {"id": "fc1.Gemm.weight::tensor", "op": "tensor", "kind": "weights", "units": 12},
]
TABLE_HEAD = {"vector_node": "probabilities", "link": "softmax"}


def by_hand(row):
    x = np.array([[row["tenure"], row["seats"], row["spend"]]], dtype=np.float32)
    h = x @ W1 + B1
    r = np.maximum(h, 0)
    return h.reshape(-1), r.reshape(-1), (r @ W2 + B2).reshape(-1)


def test_a_file_answers_predictions_with_the_models_own_scores(client, api):
    on_kaimei(api, TABLE_FEATURES, TABLE_NODES, head=TABLE_HEAD)
    agent = Agent(client, "churn", onnx=table_file())
    assert agent.capabilities == ["predict", "trace", "activations"]
    rows = [{"spend": 2.0, "tenure": 0.5, "seats": -1.0}, {"tenure": 3.0, "seats": 1.0, "spend": 0.0}]
    answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
    assert answer["raw"] is True  # scores: Kaimei applies the softmax its structure has, the agent never does
    for got, row in zip(answer["predictions"], rows):
        np.testing.assert_allclose(got, by_hand(row)[2], rtol=1e-5)
    # A function of your own still answers predictions when one is given; the file answers the looks inside.
    both = Agent(client, "churn", lambda given: [[0.25, 0.75]] * len(given), onnx=table_file())
    assert both.run({"kind": "predict", "payload": {"rows": rows[:1]}}) == {"predictions": [[0.25, 0.75]]}
    looked = both.run({"kind": "trace", "payload": {"input": rows[0]}})
    assert looked["prediction"] == [0.25, 0.75] and "raw" not in looked


def test_a_file_says_what_every_step_computed(client, api):
    structure = on_kaimei(api, TABLE_FEATURES, TABLE_NODES, head=TABLE_HEAD)
    agent = Agent(client, "churn", onnx=table_file().SerializeToString())
    row = {"tenure": 0.5, "seats": -1.0, "spend": 2.0}
    hidden, fired, scores = by_hand(row)
    traced = agent.run({"kind": "trace", "payload": {"input": row, "nodes": None}})
    frames = {frame["node_id"]: frame for frame in traced["frames"]}
    assert list(frames) == ["tenure", "seats", "spend", "fc1.Gemm", "r", "head.Gemm"]
    assert frames["seats"]["value"] == [-1.0]
    np.testing.assert_allclose(frames["fc1.Gemm"]["value"], hidden, rtol=1e-5)
    np.testing.assert_allclose(frames["r"]["value"], fired, rtol=1e-5)
    np.testing.assert_allclose(frames["head.Gemm"]["value"], scores, rtol=1e-5)
    np.testing.assert_allclose(traced["prediction"], scores, rtol=1e-5)
    assert traced["raw"] is True
    only = agent.run({"kind": "trace", "payload": {"input": row, "nodes": ["r", "probabilities", "tenure"]}})
    assert [frame["node_id"] for frame in only["frames"]] == ["tenure", "r"]
    watched = agent.run({"kind": "activations", "payload": {"input": row, "nodes": None, "units": 2}})
    stages = {stage["node_id"]: stage for stage in watched["stages"]}
    assert list(stages) == list(frames) and watched["raw"] is True and watched["forward_ms"] >= 0
    assert [name for name, stage in stages.items() if stage["detailed"]] == ["fc1.Gemm", "r", "head.Gemm"]
    np.testing.assert_allclose(decode(stages["r"]["values"]), fired, atol=np.ptp(fired) / 255 + 1e-6)
    assert [unit["index"] for unit in stages["fc1.Gemm"]["top_units"]] == list(np.argsort(-np.abs(hidden), kind="stable")[:2])
    assert structure.call_count == 1
    json.dumps([traced, watched], allow_nan=False)


def test_many_steps_are_read_in_several_runs_and_sessions_are_kept(client, api, monkeypatch):
    on_kaimei(api, TABLE_FEATURES, TABLE_NODES, head=TABLE_HEAD)
    agent = Agent(client, "churn", onnx=table_file())
    row = {"tenure": 0.5, "seats": -1.0, "spend": 2.0}
    asked = []
    run = agent._file.run
    monkeypatch.setattr(agent._file, "run", lambda feeds, names: asked.append(list(names)) or run(feeds, names))
    whole = agent.run({"kind": "trace", "payload": {"input": row}})
    assert asked == [["scores", "h", "r", "scores"]]  # one run: the output and every step's tensor
    asked.clear()
    monkeypatch.setattr(agents, "FETCH_STEPS", 1)
    apart = agent.run({"kind": "trace", "payload": {"input": row}})
    assert asked == [["scores", "h"], ["scores", "r"], ["scores", "scores"]]
    assert apart["frames"] == whole["frames"]
    monkeypatch.setattr(agents, "FETCH_STEPS", 256)
    monkeypatch.setattr(agents, "FETCH_VALUES", 5)  # values, not steps, are what fills memory
    asked.clear()
    agent.run({"kind": "trace", "payload": {"input": row}})
    assert asked == [["scores", "h"], ["scores", "r"], ["scores", "scores"]]
    # The file's own outputs stay ready, and the last few sets of steps asked for.
    assert () in agent._file._sessions or len(agent._file._sessions) <= agents.SESSIONS_KEPT
    assert len(agent._file._sessions) <= agents.SESSIONS_KEPT
    # A prediction asks for nothing inside the file.
    asked.clear()
    agent.run({"kind": "predict", "payload": {"rows": [row]}})
    assert asked == [["scores"]]


def test_how_a_files_inputs_are_made_can_be_said_in_several_ways(client, api):
    on_kaimei(api, TABLE_FEATURES, TABLE_NODES, head=TABLE_HEAD)
    row = {"tenure": 0.5, "seats": -1.0, "spend": 2.0}
    want = by_hand(row)[2]
    turned = by_hand({"tenure": 2.0, "seats": -1.0, "spend": 0.5})[2]
    for inputs, expected in (
        (None, want),
        ({"X": ["spend", "seats", "tenure"]}, turned),
        ({"X": {"features": ["spend", "seats", "tenure"]}}, turned),
        ({"X": lambda given: [[given["tenure"], given["seats"], given["spend"]]]}, want),
    ):
        agent = Agent(client, "churn", onnx=table_file(), inputs=inputs)
        np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": [row]}})["predictions"][0], expected, rtol=1e-5)
    with pytest.raises(ValueError, match="no feature called 'age'"):
        Agent(client, "churn", onnx=table_file(), inputs={"X": ["age", "seats", "spend"]}).run({"kind": "predict", "payload": {"rows": [row]}})
    with pytest.raises(ValueError, match=r"has shape \[None, 3\]"):
        Agent(client, "churn", onnx=table_file(), inputs={"X": ["seats", "spend"]}).run({"kind": "predict", "payload": {"rows": [row]}})


def test_the_structure_that_was_uploaded_says_exactly_how_the_file_is_fed(client, api):
    on_kaimei(api, TABLE_FEATURES, TABLE_NODES, head=TABLE_HEAD)
    spec = {
        "family": "graph",
        "graph": {"inputs": [{"name": "in.X", "dtype": "float32", "shape": [1, 3], "batch_axis": 0, "from": {"features": ["spend", "tenure", "seats"]}}]},
    }
    row = {"tenure": 0.5, "seats": -1.0, "spend": 2.0}
    agent = Agent(client, "churn", onnx=table_file(), structure=spec)
    want = by_hand({"tenure": 2.0, "seats": 0.5, "spend": -1.0})[2]  # spend, tenure, seats as the three columns
    np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": [row]}})["predictions"][0], want, rtol=1e-5)

    class Extraction:
        pass

    held = Extraction()
    held.spec = spec
    np.testing.assert_allclose(Agent(client, "churn", onnx=table_file(), structure=held)._feeds(row)["X"], [[2.0, 0.5, -1.0]])
    other = {"family": "graph", "graph": {"inputs": [{"name": "in.pixels", "dtype": "float32", "shape": [1, 3], "from": {"features": ["spend", "tenure", "seats"]}}]}}
    with pytest.raises(ValueError, match="not an input of this file"):
        Agent(client, "churn", onnx=table_file(), structure=other)._feeds(row)
    with pytest.raises(TypeError):
        Agent(client, "churn", onnx=table_file(), structure=[1, 2])


# --- text, a picture, a language model -------------------------------------------------------

EMBED = RNG.normal(size=(9, 4)).astype(np.float32)
OUT = RNG.normal(size=(4, 2)).astype(np.float32)


def text_file():
    return file_of(
        [
            helper.make_node("Gather", ["E", "input_ids"], ["embedded"], name="embed/Gather"),
            helper.make_node("Cast", ["attention_mask"], ["real"], to=TensorProto.FLOAT, name="Cast"),
            helper.make_node("Unsqueeze", ["real", "last"], ["real3"], name="Unsqueeze"),
            helper.make_node("Mul", ["embedded", "real3"], ["kept"], name="Mul"),
            helper.make_node("ReduceSum", ["kept", "along"], ["pooled"], keepdims=0, name="ReduceSum"),
            helper.make_node("MatMul", ["pooled", "OUT"], ["logits"], name="head/MatMul"),
        ],
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        {"E": EMBED, "OUT": OUT, "last": np.array([-1], dtype=np.int64), "along": np.array([1], dtype=np.int64)},
    )


def test_a_text_reaches_the_file_as_ids_filled_up_and_a_mask_of_the_real_ones(client, api):
    features = [{"name": "review", "type": "text", "max_length": 6, "vocab_size": 9}]
    nodes = [
        {"id": "review", "op": "input", "kind": "input", "units": 6, "feature": "review", "args": {"feature_type": "text"}},
        step("embed.Gather", "Gather", [6, 4], [1, 6, 4], source="embedded"),
        step("ReduceSum", "ReduceSum", [4], [1, 4], source="pooled"),
        step("head.MatMul", "MatMul", [2], [1, 2], source="logits"),
    ]
    on_kaimei(api, features, nodes, task="binary_classification", head={"vector_node": "head.MatMul", "link": "softmax"})
    agent = Agent(client, "churn", onnx=text_file(), pad_id=7)
    feeds = agent._feeds({"review": [2, 6, 1]})
    assert feeds["input_ids"].dtype == np.int64 and feeds["input_ids"].tolist() == [[2, 6, 1, 7, 7, 7]]
    assert feeds["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0]]
    want = EMBED[[2, 6, 1]].sum(axis=0) @ OUT
    traced = agent.run({"kind": "trace", "payload": {"input": {"review": [2, 6, 1]}}})
    frames = {frame["node_id"]: frame for frame in traced["frames"]}
    assert frames["review"]["value"] == [2.0, 6.0, 1.0, -1.0, -1.0, -1.0]
    np.testing.assert_allclose(frames["embed.Gather"]["value"][:4], EMBED[2], rtol=1e-6)
    assert frames["embed.Gather"]["summary"]["size"] == 24
    np.testing.assert_allclose(traced["prediction"], want, rtol=1e-5)
    # Said by hand: the mask is always ones, so the padding counts too.
    ones = Agent(client, "churn", onnx=text_file(), pad_id=7, inputs={"attention_mask": lambda row: np.ones((1, 6))})
    padded = EMBED[[2, 6, 1, 7, 7, 7]].sum(axis=0) @ OUT
    np.testing.assert_allclose(ones.run({"kind": "predict", "payload": {"rows": [{"review": [2, 6, 1]}]}})["predictions"][0], padded, rtol=1e-5)
    on_kaimei(api, [*features, {"name": "title", "type": "text", "max_length": 6}], nodes, task="binary_classification")
    with pytest.raises(ValueError, match="2 text features"):
        Agent(client, "churn", onnx=text_file())._feeds({"review": [2]})


def test_a_picture_is_turned_channels_first_for_the_file_and_its_maps_come_back_as_shown(client, api):
    kernel = RNG.normal(size=(2, 3, 1, 1)).astype(np.float32)
    file = file_of(
        [
            helper.make_node("Conv", ["pixels", "K"], ["maps"], name="conv/Conv"),
            helper.make_node("GlobalAveragePool", ["maps"], ["pooled"], name="pool/GlobalAveragePool"),
            helper.make_node("Flatten", ["pooled"], ["out"], name="Flatten"),
        ],
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["batch", 3, 4, 5])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 2])],
        {"K": kernel},
    )
    features = [{"name": "photo", "type": "image", "height": 4, "width": 5, "channels": 3}]
    nodes = [
        {"id": "photo", "op": "input", "kind": "input", "units": 60, "feature": "photo", "args": {"shape": [4, 5, 3]}},
        step("conv.Conv", "Conv", [4, 5, 2], [1, 2, 4, 5], source="maps", args={"tensor": {"shape": [1, 2, 4, 5], "dtype": "float32", "perm": [0, 2, 3, 1]}}),
        step("Flatten", "Flatten", [2], [1, 2], source="out", kind="reshape"),
    ]
    on_kaimei(api, features, nodes, task="regression", head={"vector_node": "Flatten", "link": "none"})
    picture = RNG.uniform(0, 1, size=(4, 5, 3)).astype(np.float32)
    maps = np.einsum("hwc,kc->hwk", picture, kernel[:, :, 0, 0])
    agent = Agent(client, "churn", onnx=file)
    assert agent._bindings()[0]["source"] == {"feature": "photo", "layout": "nchw"}
    watched = agent.run({"kind": "activations", "payload": {"input": {"photo": picture}, "nodes": ["conv.Conv"], "channels": 2}})
    stages = {stage["node_id"]: stage for stage in watched["stages"]}
    conv = stages["conv.Conv"]
    assert conv["shape"] == [4, 5, 2] and conv["detailed"] is True and stages["photo"]["shape"] == [4, 5, 3]
    for item in conv["maps"]:
        np.testing.assert_allclose(decode(item), maps[:, :, item["channel"]], atol=np.ptp(maps[:, :, item["channel"]]) / 255 + 1e-6)
    np.testing.assert_allclose(watched["prediction"], maps.mean(axis=(0, 1)), rtol=1e-5)


def test_a_language_models_file_answers_at_the_last_token_the_row_gave(client, api):
    words = RNG.normal(size=(4, 9)).astype(np.float32)
    file = file_of(
        [
            helper.make_node("Gather", ["E", "ids"], ["embedded"], name="embed/Gather"),
            helper.make_node("CumSum", ["embedded", "axis"], ["state"], name="CumSum"),
            helper.make_node("MatMul", ["state", "WORDS"], ["scores"], name="head/MatMul"),
        ],
        [helper.make_tensor_value_info("ids", TensorProto.INT64, ["batch", 6])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", 6, 9])],
        {"E": EMBED, "WORDS": words, "axis": np.array(1, dtype=np.int64)},
    )
    features = [{"name": "prompt", "type": "text", "max_length": 6, "vocab_size": 9}]
    nodes = [
        step("head.MatMul", "MatMul", [6, 9], [1, 6, 9], source="scores"),
        step("last_position", "LastToken", [9], [1, 9], kind="output", inputs=["head.MatMul"]),
        step("next_token", "Softmax", [9], [1, 9], kind="output", inputs=["last_position"]),
    ]
    on_kaimei(api, features, nodes, task="text_generation", head={"vector_node": "next_token", "link": "softmax"})
    agent = Agent(client, "churn", onnx=file)
    rows = [{"prompt": [3, 1, 4]}, {"prompt": [2]}]
    answer = agent.run({"kind": "predict", "payload": {"rows": rows}})
    assert answer["raw"] is True and [len(line) for line in answer["predictions"]] == [9, 9]
    np.testing.assert_allclose(answer["predictions"][0], EMBED[[3, 1, 4]].sum(axis=0) @ words, rtol=1e-4)
    np.testing.assert_allclose(answer["predictions"][1], EMBED[2] @ words, rtol=1e-4)
    traced = agent.run({"kind": "trace", "payload": {"input": rows[0]}})
    assert [frame["node_id"] for frame in traced["frames"]] == ["head.MatMul"] and len(traced["prediction"]) == 9


# --- where the file is, and what is missing ----------------------------------------------------


def test_a_step_kaimei_added_is_not_a_tensor_of_the_file_even_under_the_same_name(client, api):
    # A converted classifier calls its scores "probabilities", and that is what Kaimei calls the step that
    # squashes them. The step is Kaimei's own: the file's tensor of that name belongs to the step before it.
    pick = np.array([1], dtype=np.int64)
    file = file_of(
        [
            helper.make_node("Gemm", ["X", "W1", "B1"], ["h"], name="fc1/Gemm"),
            helper.make_node("Gemm", ["h", "W2", "B2"], ["probabilities"], name="head/Gemm"),
            helper.make_node("Gather", ["probabilities", "pick"], ["margin"], axis=1, name="margin/Gather"),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 2]), helper.make_tensor_value_info("margin", TensorProto.FLOAT, ["batch", 1])],
        {"W1": W1, "B1": B1, "W2": W2, "B2": B2, "pick": pick},
    )
    row = {"tenure": 0.5, "seats": -1.0, "spend": 2.0}
    scores = by_hand(row)[2] - np.maximum(by_hand(row)[0], 0) @ W2 + by_hand(row)[0] @ W2  # no activation in this file
    steps = [
        step("fc1.Gemm", "Gemm", [4], [1, 4], source="h"),
        step("head.Gemm", "Gemm", [2], [1, 2], source="probabilities"),
        step("margin.Gather", "Gather", [1], [1, 1], source="margin", inputs=["head.Gemm"]),
    ]
    one = [*steps, step("probabilities", "Sigmoid", [1], [1, 1], kind="output", inputs=["margin.Gather"])]
    on_kaimei(api, TABLE_FEATURES, one, task="binary_classification", head={"vector_node": "probabilities", "link": "sigmoid"})
    agent = Agent(client, "churn", onnx=file)
    assert agent._output_tensor() == "margin"
    answer = agent.run({"kind": "predict", "payload": {"rows": [row]}})
    np.testing.assert_allclose(answer["predictions"], [[scores[1]]], rtol=1e-5)  # the margin, for Kaimei to squash
    assert answer["raw"] is True
    traced = agent.run({"kind": "trace", "payload": {"input": row}})
    frames = {frame["node_id"]: frame["value"] for frame in traced["frames"]}
    assert list(frames) == ["fc1.Gemm", "head.Gemm", "margin.Gather"]
    np.testing.assert_allclose(frames["head.Gemm"], scores, rtol=1e-5)
    # The same with a softmax over both scores: as many values as the file's tensor, and still not that tensor.
    both = [*steps[:2], step("probabilities", "Softmax", [2], [1, 2], kind="output", inputs=["head.Gemm"])]
    on_kaimei(api, TABLE_FEATURES, both, task="binary_classification", head={"vector_node": "probabilities", "link": "softmax"})
    agent = Agent(client, "churn", onnx=file)
    assert agent._output_tensor() == "probabilities"  # the file's tensor, through the step that was read from it
    traced = agent.run({"kind": "trace", "payload": {"input": row}})
    assert [frame["node_id"] for frame in traced["frames"]] == ["fc1.Gemm", "head.Gemm"]
    np.testing.assert_allclose(traced["prediction"], scores, rtol=1e-5)


def test_a_file_on_disk_is_run_from_there_with_its_weights_beside_it(client, api, tmp_path):
    on_kaimei(api, TABLE_FEATURES, TABLE_NODES, head=TABLE_HEAD)
    path = tmp_path / "model.onnx"
    onnx.save_model(table_file(), str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
    assert (tmp_path / "weights.bin").stat().st_size > 0
    agent = Agent(client, "churn", onnx=str(path))
    row = {"tenure": 0.5, "seats": -1.0, "spend": 2.0}
    np.testing.assert_allclose(agent.run({"kind": "predict", "payload": {"rows": [row]}})["predictions"][0], by_hand(row)[2], rtol=1e-5)
    traced = agent.run({"kind": "trace", "payload": {"input": row}})
    np.testing.assert_allclose(next(frame for frame in traced["frames"] if frame["node_id"] == "r")["value"], by_hand(row)[1], rtol=1e-5)
    assert Agent(client, "churn", onnx=path).capabilities == ["predict", "trace", "activations"]  # a Path will do


def test_what_an_onnx_agent_needs(client, monkeypatch):
    with pytest.raises(TypeError, match="not both"):
        Agent(client, "churn", module=type("M", (), {"named_modules": 1, "register_forward_hook": 1, "forward": lambda self: 0})(), onnx=table_file())
    with pytest.raises(TypeError, match="its path, its bytes"):
        Agent(client, "churn", onnx=42)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match="pip install onnxruntime"):
        Agent(client, "churn", onnx=table_file())


def test_the_command_line_takes_the_file(monkeypatch, tmp_path, capsys):
    path = tmp_path / "model.onnx"
    onnx.save_model(table_file(), str(path))
    served = {}
    monkeypatch.setattr(Agent, "serve", lambda self, stop=None: served.update(agent=self))
    monkeypatch.setenv("KAIMEI_API_KEY", "km_test_cli")
    monkeypatch.setenv("KAIMEI_BASE_URL", "https://kaimei.example")
    assert main(["--model", "churn", "--onnx", str(path), "--providers", "CPUExecutionProvider", "--quiet"]) == 0
    agent = served["agent"]
    assert agent.capabilities == ["predict", "trace", "activations"] and agent._predict is None
    assert agent._file.providers == ["CPUExecutionProvider"] and agent._file.path == str(path)
    assert main(["--model", "churn", "--onnx", str(path), "--predict", "json:dumps", "--quiet"]) == 0
    assert served["agent"]._predict is json.dumps and served["agent"]._file is not None
    capsys.readouterr()
