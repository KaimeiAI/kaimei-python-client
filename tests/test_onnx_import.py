"""ONNX models read as graph structures: what is declared, what is checked, and what the client settles itself."""

from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from kaimei import StructureError, graphs  # noqa: E402
from kaimei.graphs import from_onnx  # noqa: E402
from kaimei.structure import extract_structure, finalize_spec, spec_is_executable  # noqa: E402

RNG = np.random.default_rng(11)
W1 = RNG.normal(size=(4, 3)).astype(np.float32)
B1 = RNG.normal(size=4).astype(np.float32)
W2 = RNG.normal(size=(2, 4)).astype(np.float32)
B2 = RNG.normal(size=2).astype(np.float32)
NAMES = ["age", "income", "tenure"]
WORDS = ["<pad>", "<unk>", "good", "bad", "great", "awful", "film", "a", "not", "very"]
TABLE = RNG.normal(size=(len(WORDS), 4)).astype(np.float32)


def model_of(nodes, inputs, outputs, initializers, extra_opsets=()):
    graph = helper.make_graph(nodes, "test", inputs, outputs, initializers)
    made = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), *extra_opsets])
    made.ir_version = 9
    return made


def softmax(values):
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def mlp_model():
    return model_of(
        [
            helper.make_node("Gemm", ["rows", "w1", "b1"], ["hidden"], transB=1, name="/body/fc1/Gemm"),
            helper.make_node("Relu", ["hidden"], ["fired"], name="/body/act/Relu"),
            helper.make_node("Gemm", ["fired", "w2", "b2"], ["logits"], transB=1, name="/head/Gemm"),
        ],
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        [numpy_helper.from_array(W1, "w1"), numpy_helper.from_array(B1, "b1"), numpy_helper.from_array(W2, "w2"), numpy_helper.from_array(B2, "b2")],
    )


def mlp_numpy(rows):
    hidden = np.maximum(np.asarray(rows, dtype=np.float32) @ W1.T + B1, 0)
    return softmax(hidden @ W2.T + B2)


def check_table(spec):
    """The check's rows as a table, in the order of the spec's features."""
    return np.array([[spec["check"]["inputs"][name][row][0] for name in spec["feature_names"]] for row in range(len(spec["check"]["outputs"]))])


def test_a_table_model_is_read_with_its_features_head_and_a_check_of_its_own_outputs():
    spec = from_onnx(mlp_model(), feature_names=NAMES, task="binary_classification", labels=["stays", "leaves"])
    assert spec["family"] == "graph" and spec["task"] == "binary_classification" and spec["reader"] == 2
    assert spec["feature_names"] == NAMES
    assert spec["features"] == [{"name": name, "type": "numeric"} for name in NAMES]
    assert spec["head"]["link"] == "softmax" and spec["outputs"] == {"labels": ["stays", "leaves"]}
    assert spec["graph"]["inputs"][0]["from"] == {"features": NAMES}
    assert [node["op"] for node in spec["graph"]["nodes"]] == ["Gemm", "Relu", "Gemm"]
    assert [node["path"] for node in spec["graph"]["nodes"]] == ["body.fc1.Gemm", "body.act.Relu", "head.Gemm"]
    assert not spec.get("approximate") and spec_is_executable(spec)
    rows = check_table(spec)
    # The row the model's shapes were recorded on, then rows made up near zero and further from it.
    assert rows.shape == (1 + graphs.TABLE_CHECK_ROWS, 3) and rows[:5].min() >= 0 and rows.min() < 0
    np.testing.assert_allclose(spec["check"]["outputs"], mlp_numpy(rows), rtol=1e-5, atol=1e-6)
    assert spec["check"]["tolerance"] == 1e-4
    meta = spec["metadata"]
    assert meta["route"] == "onnx" and meta["imported_from"] == "onnx" and meta["model_class"] == "onnx.ModelProto"
    assert meta["extracted_with"].startswith("kaimei-python-client") and meta["checked_against"].startswith("onnx")
    done = finalize_spec(spec)
    json.dumps(done.spec, allow_nan=False)
    assert done.spec["graph"]["constants"]["const.w1"]["dtype"] == "float32"


def test_the_callers_own_rows_join_the_check_of_a_table_model():
    own = RNG.normal(size=(7, 3)) * 50
    spec = from_onnx(mlp_model(), feature_names=NAMES, task="binary_classification", check_rows=own)
    rows = check_table(spec)
    assert rows.shape[0] == 1 + graphs.TABLE_CHECK_ROWS + 7
    np.testing.assert_allclose(rows[-7:], own)
    np.testing.assert_allclose(spec["check"]["outputs"], mlp_numpy(rows), rtol=1e-4, atol=1e-5)


def test_classes_nobody_named_get_the_names_the_native_readers_give():
    assert from_onnx(mlp_model(), task="multiclass_classification")["outputs"] == {"labels": ["class_0", "class_1"]}
    assert from_onnx(mlp_model(), task="binary_classification")["outputs"] == {"labels": ["negative", "positive"]}
    regression = from_onnx(mlp_model(), task="embedding")
    assert "outputs" not in regression and regression["feature_names"] == ["x0", "x1", "x2"]


def test_a_path_or_bytes_are_read_by_extract_structure(tmp_path):
    path = tmp_path / "churn.onnx"
    onnx.save(mlp_model(), str(path))
    for source in (str(path), path, path.read_bytes(), mlp_model()):
        extraction = extract_structure(source, feature_names=NAMES, task="binary_classification")
        assert extraction.spec["family"] == "graph" and extraction.spec["feature_names"] == NAMES
        assert extraction.spec["metadata"]["route"] == "onnx" and extraction.spec["framework"] == "onnx"
        json.dumps(extraction.spec, allow_nan=False)
    named = extract_structure(str(path), feature_names=NAMES, task="binary_classification", metadata={"run": "exp-7"})
    assert named.spec["metadata"]["run"] == "exp-7"
    with pytest.raises(StructureError) as caught:
        extract_structure(str(tmp_path / "absent.onnx"), feature_names=NAMES)
    assert "not an ONNX model" in caught.value.message
    with pytest.raises(StructureError):
        extract_structure(b"this is not a model", feature_names=NAMES)


def test_the_wrong_number_of_names_is_refused():
    with pytest.raises(StructureError) as caught:
        from_onnx(mlp_model(), feature_names=["a", "b"])
    assert "3 columns" in caught.value.message and caught.value.code == "unsupported_estimator"


def masked_model():
    """Token ids and their mask, averaged over the real tokens, then two classes."""
    return model_of(
        [
            helper.make_node("Gather", ["table", "input_ids"], ["embedded"], name="/embed/Gather"),
            helper.make_node("Cast", ["attention_mask"], ["real"], to=TensorProto.FLOAT, name="/Cast"),
            helper.make_node("Unsqueeze", ["real", "last"], ["real3"], name="/Unsqueeze"),
            helper.make_node("Mul", ["embedded", "real3"], ["kept"], name="/pool/Mul"),
            helper.make_node("ReduceSum", ["kept", "one"], ["summed"], keepdims=0, name="/pool/ReduceSum"),
            helper.make_node("ReduceSum", ["real", "one"], ["count"], keepdims=1, name="/pool/ReduceSum_1"),
            helper.make_node("Div", ["summed", "count"], ["pooled"], name="/pool/Div"),
            helper.make_node("Gemm", ["pooled", "w2", "b2"], ["logits"], transB=1, name="/head/Gemm"),
        ],
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", 6]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", 6]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        [
            numpy_helper.from_array(TABLE, "table"),
            numpy_helper.from_array(W2, "w2"),
            numpy_helper.from_array(B2, "b2"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "last"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
        ],
    )


def masked_numpy(row, pad_id):
    row = np.asarray(row)
    real = row >= 0
    ids = np.where(real, row, pad_id).astype(int)
    pooled = (TABLE[ids] * real[:, None]).sum(axis=0) / real.sum()
    return softmax(pooled @ W2.T + B2)


def test_a_text_model_gets_its_tokenizer_and_a_check_the_api_can_repeat():
    tokenizer = {"type": "word", "vocab": WORDS, "unk": "<unk>", "pad": "<pad>"}
    spec = from_onnx(masked_model(), task="binary_classification", tokenizer=tokenizer)
    assert spec["features"] == [{"name": "text", "type": "text", "max_length": 6, "vocab_size": len(WORDS)}]
    assert spec["tokenizers"] == {"text": tokenizer}
    by_part = {item["from"]["part"]: item for item in spec["graph"]["inputs"]}
    assert set(by_part) == {"ids", "mask"} and by_part["ids"]["from"]["pad_id"] == 0
    assert all(item["from"]["feature"] == "text" for item in spec["graph"]["inputs"])
    rows = spec["check"]["inputs"]["text"]
    assert len(rows) >= 4 and all(len(row) == 6 for row in rows)
    # Some rows are shorter than the model reads, as real texts are: the rest is -1, which the API pads.
    assert any(-1.0 in row for row in rows) and any(-1.0 not in row for row in rows)
    assert max(max(row) for row in rows) < len(WORDS)
    # The mask the API derives from those rows is the mask the model was run with here.
    for row, answer in zip(rows, spec["check"]["outputs"]):
        np.testing.assert_allclose(answer, masked_numpy(row, 0), rtol=1e-5, atol=1e-6)


def test_without_a_tokenizer_the_feature_takes_token_ids():
    spec = from_onnx(masked_model(), task="binary_classification", inputs={"input_ids": {"feature": "review", "part": "ids", "vocab": 10, "pad_id": 3}, "attention_mask": {"feature": "review", "part": "mask"}})
    assert spec["features"] == [{"name": "review", "type": "tokens", "max_length": 6}]
    assert "tokenizers" not in spec
    rows = spec["check"]["inputs"]["review"]
    assert max(max(row) for row in rows) > 1  # ids are drawn from the ten the hint says there are
    for row, answer in zip(rows, spec["check"]["outputs"]):
        np.testing.assert_allclose(answer, masked_numpy(row, 3), rtol=1e-5, atol=1e-6)
    with pytest.raises(StructureError) as caught:
        from_onnx(mlp_model(), feature_names=NAMES, tokenizer={"type": "char", "vocab": ["a"]})
    assert caught.value.param == "tokenizer"


def test_a_tokenizers_own_padding_token_fills_a_short_text_unless_the_caller_says():
    padded = {"type": "word", "vocab": WORDS[1:] + ["<pad>"], "unk": "<unk>", "pad": "<pad>"}
    spec = from_onnx(masked_model(), task="binary_classification", tokenizer=padded)
    ids = next(item for item in spec["graph"]["inputs"] if item["from"]["part"] == "ids")
    assert ids["from"]["pad_id"] == len(WORDS) - 1
    said = from_onnx(
        masked_model(), task="binary_classification", tokenizer=padded,
        inputs={"input_ids": {"feature": "text", "part": "ids", "pad_id": 2}, "attention_mask": {"feature": "text", "part": "mask"}},
    )
    ids = next(item for item in said["graph"]["inputs"] if item["from"]["part"] == "ids")
    assert ids["from"]["pad_id"] == 2


class SentencePieces:
    """A tokenizer Kaimei cannot run itself: it names its pieces, and only it knows how a text is split into them."""

    pieces = ["<unk>", "▁good", "▁bad", "▁great", "▁awful", "▁film", "s", "▁a", "▁not", "<pad>"]
    unk_token, pad_token, eos_token, mask_token = "<unk>", "<pad>", None, None

    def __len__(self):
        return len(self.pieces)

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[index] for index in ids]


def test_a_tokenizer_kaimei_cannot_run_still_makes_the_feature_text():
    with pytest.warns(UserWarning, match="its vocabulary alone is sent"):
        spec = from_onnx(masked_model(), task="binary_classification", tokenizer=SentencePieces())
    # The feature is text all the same: its agent splits it, and Kaimei names the tokens.
    assert spec["features"] == [{"name": "text", "type": "text", "max_length": 6, "vocab_size": 10}]
    assert spec["tokenizers"]["text"] == {
        "type": "pieces", "vocab": SentencePieces.pieces, "space": "▁", "unk": "<unk>", "pad": "<pad>"
    }
    ids = next(item for item in spec["graph"]["inputs"] if item["from"]["part"] == "ids")
    assert ids["from"]["pad_id"] == 9  # short texts are filled up with the tokenizer's own padding piece
    for row, answer in zip(spec["check"]["inputs"]["text"], spec["check"]["outputs"]):
        np.testing.assert_allclose(answer, masked_numpy(row, 9), rtol=1e-5, atol=1e-6)
    json.dumps(finalize_spec(spec).spec, allow_nan=False)
    # Something with no vocabulary to send leaves a feature of token ids.
    with pytest.warns(UserWarning, match="token ids instead of text"):
        bare = from_onnx(masked_model(), task="binary_classification", tokenizer=lambda text: [1, 2])
    # Its embedding table says how many ids there are, so an id past the vocabulary is refused before it runs.
    assert bare["features"] == [{"name": "text", "type": "tokens", "max_length": 6, "vocab_size": 10}] and "tokenizers" not in bare


LETTERS = list("abcde")
NEXT = RNG.normal(size=(5, 5)).astype(np.float32)


def language_model():
    return model_of(
        [helper.make_node("Gather", ["next", "ids"], ["scores"], name="/lookup/Gather")],
        [helper.make_tensor_value_info("ids", TensorProto.INT64, ["batch", 7])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", 7, 5])],
        [numpy_helper.from_array(NEXT, "next")],
    )


def test_a_language_model_is_read_at_its_last_real_token_as_probabilities():
    spec = from_onnx(language_model(), task="text_generation", tokenizer={"type": "char", "vocab": LETTERS})
    assert spec["task"] == "text_generation" and spec["head"]["link"] == "softmax"
    assert spec["features"] == [{"name": "text", "type": "text", "max_length": 7, "vocab_size": 5}]
    rows = spec["check"]["inputs"]["text"]
    assert any(-1.0 in row for row in rows)
    for row, answer in zip(rows, spec["check"]["outputs"]):
        real = [int(item) for item in row if item >= 0]
        assert len(answer) == 5  # one position's scores, not all seven
        np.testing.assert_allclose(answer, softmax(NEXT[real[-1]]), rtol=1e-5, atol=1e-6)


def test_scores_that_are_already_probabilities_are_left_alone():
    already = model_of(
        [
            helper.make_node("Gather", ["next", "ids"], ["scores"], name="/lookup/Gather"),
            helper.make_node("Softmax", ["scores"], ["shares"], axis=-1, name="/Softmax"),
        ],
        [helper.make_tensor_value_info("ids", TensorProto.INT64, ["batch", 7])],
        [helper.make_tensor_value_info("shares", TensorProto.FLOAT, ["batch", 7, 5])],
        [numpy_helper.from_array(NEXT, "next")],
    )
    spec = from_onnx(already, task="text_generation")
    assert spec["head"]["link"] == "none"
    for row, answer in zip(spec["check"]["inputs"]["text"], spec["check"]["outputs"]):
        real = [int(item) for item in row if item >= 0]
        np.testing.assert_allclose(answer, softmax(NEXT[real[-1]]), rtol=1e-5, atol=1e-6)


KERNEL = RNG.normal(size=(2, 1, 3, 3)).astype(np.float32)


def painter_model():
    return model_of(
        [
            helper.make_node("Conv", ["x", "kernel"], ["maps"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="/conv/Conv"),
            helper.make_node("Sigmoid", ["maps"], ["picture"], name="/Sigmoid"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 1, 4, 4])],
        [helper.make_tensor_value_info("picture", TensorProto.FLOAT, ["batch", 2, 4, 4])],
        [numpy_helper.from_array(KERNEL, "kernel")],
    )


def painter_numpy(picture):
    """The model on one height x width x 1 picture, as the model keeps its answer: 2 x height x width."""
    padded = np.pad(np.asarray(picture, dtype=np.float32).reshape(4, 4), 1)
    out = np.zeros((2, 4, 4), dtype=np.float32)
    for channel in range(2):
        for y in range(4):
            for x in range(4):
                out[channel, y, x] = float((padded[y : y + 3, x : x + 3] * KERNEL[channel, 0]).sum())
    return 1.0 / (1.0 + np.exp(-out))


def test_a_model_that_draws_is_checked_on_its_picture_in_the_order_it_keeps_it():
    spec = from_onnx(painter_model(), task="generation", inputs={"x": {"feature": "sketch", "layout": "nchw"}})
    assert spec["features"] == [{"name": "sketch", "type": "image", "height": 4, "width": 4, "channels": 1}]
    head = spec["head"]["output"]
    # How a picture is shown is the API's to say: the client writes down the tensor's shape, and the name the
    # file gives it (which an agent that runs the file reads it by), and nothing more.
    assert spec["graph"]["values"][head] == {"shape": [1, 2, 4, 4], "dtype": "float32", "source": "picture"}
    for row, answer in zip(spec["check"]["inputs"]["sketch"], spec["check"]["outputs"]):
        np.testing.assert_allclose(answer, painter_numpy(row).reshape(-1), rtol=1e-4, atol=1e-5)
    assert spec["check"]["tolerance"] == 1e-3  # a convolution's arithmetic is long enough to drift
    series = model_of(
        [helper.make_node("Mul", ["x", "two"], ["out"], name="/Mul")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 5, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 5, 3])],
        [numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two")],
    )
    plain = from_onnx(series, task="generation")
    assert "view" not in plain["graph"]["values"][plain["head"]["output"]]
    assert plain["features"] == [{"name": "x", "type": "sequence", "shape": [5, 3]}]
    for row, answer in zip(plain["check"]["inputs"]["x"], plain["check"]["outputs"]):
        np.testing.assert_allclose(answer, 2 * np.asarray(row), rtol=1e-6)


def test_infinity_in_an_attribute_moves_to_a_constant_that_can_be_sent():
    floor = helper.make_tensor("value", TensorProto.FLOAT, [1], [float("-inf")])
    model = model_of(
        [
            helper.make_node("Shape", ["rows"], ["size"], name="/Shape"),
            helper.make_node("ConstantOfShape", ["size"], ["floor"], value=floor, name="/ConstantOfShape"),
            helper.make_node("Max", ["rows", "floor"], ["kept"], name="/Max"),
            helper.make_node("Gemm", ["kept", "w1", "b1"], ["out"], transB=1, name="/fc/Gemm"),
        ],
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 4])],
        [numpy_helper.from_array(W1, "w1"), numpy_helper.from_array(B1, "b1")],
    )
    spec = from_onnx(model, feature_names=NAMES, task="embedding")
    ops = [node["op"] for node in spec["graph"]["nodes"]]
    assert "ConstantOfShape" not in ops and "Expand" in ops
    expand = next(node for node in spec["graph"]["nodes"] if node["op"] == "Expand")
    fill = spec["graph"]["constants"][expand["inputs"][0]]
    assert fill.shape == () and np.isneginf(fill) and expand["attrs"] == {} and expand.get("hidden") is True
    done = finalize_spec(spec)
    sent = json.dumps(done.spec, allow_nan=False)
    assert "$b64" in sent and "Infinity" not in sent
    rows = check_table(spec)
    np.testing.assert_allclose(spec["check"]["outputs"], rows.astype(np.float32) @ W1.T + B1, rtol=1e-5, atol=1e-6)


def test_an_imputer_of_nan_becomes_arithmetic_on_a_constant():
    model = model_of(
        [
            helper.make_node(
                "Imputer", ["rows"], ["filled"], domain="ai.onnx.ml", name="/imputer/Imputer",
                imputed_value_floats=[1.5, -2.0, 0.25], replaced_value_float=float("nan"),
            ),
            helper.make_node("Gemm", ["filled", "w1", "b1"], ["out"], transB=1, name="/fc/Gemm"),
        ],
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 4])],
        [numpy_helper.from_array(W1, "w1"), numpy_helper.from_array(B1, "b1")],
        extra_opsets=[helper.make_opsetid("ai.onnx.ml", 3)],
    )
    spec = from_onnx(model, feature_names=NAMES, task="embedding")
    nodes = spec["graph"]["nodes"]
    assert [node["op"] for node in nodes] == ["IsNaN", "CastLike", "Where", "Gemm"]
    where = nodes[2]
    assert where["inputs"] == [nodes[0]["outputs"][0], nodes[1]["outputs"][0], "in.rows"] and where["label"] == "Imputer"
    np.testing.assert_array_equal(spec["graph"]["constants"][nodes[1]["inputs"][0]], np.array([1.5, -2.0, 0.25], dtype=np.float32))
    json.dumps(finalize_spec(spec).spec, allow_nan=False)
    # Any other attribute that holds one is refused by name rather than sent as null.
    odd = model_of(
        [helper.make_node("LeakyRelu", ["rows"], ["out"], alpha=float("inf"), name="/odd/LeakyRelu")],
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 3])],
        [],
    )
    with pytest.raises(StructureError) as caught:
        from_onnx(odd, feature_names=NAMES, task="embedding")
    assert caught.value.code == "unsupported_layers" and "LeakyRelu" in caught.value.message


def two_outputs():
    return model_of(
        [
            helper.make_node("Gemm", ["rows", "w1", "b1"], ["first"], transB=1, name="/a/Gemm"),
            helper.make_node("ReduceSum", ["first", "one"], ["second"], keepdims=1, name="/b/ReduceSum"),
            helper.make_node("ArgMax", ["first"], ["which"], axis=1, keepdims=0, name="/ArgMax"),
        ],
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["batch", 3])],
        [
            helper.make_tensor_value_info("which", TensorProto.INT64, ["batch"]),
            helper.make_tensor_value_info("first", TensorProto.FLOAT, ["batch", 4]),
            helper.make_tensor_value_info("second", TensorProto.FLOAT, ["batch", 1]),
        ],
        [numpy_helper.from_array(W1, "w1"), numpy_helper.from_array(B1, "b1"), numpy_helper.from_array(np.array([1], dtype=np.int64), "one")],
    )


def test_the_output_the_prediction_is_read_from_can_be_named_or_counted():
    model = two_outputs()
    assert graphs._prediction_output(model, None) == ("first", "first")  # the first output that holds floats
    assert graphs._prediction_output(model, 2) == ("second", "second")
    assert graphs._prediction_output(model, "second") == ("second", "second")
    default = from_onnx(model, feature_names=NAMES, task="embedding")
    assert len(default["check"]["outputs"][0]) == 4
    chosen = from_onnx(model, feature_names=NAMES, task="regression", output="second")
    rows = check_table(chosen)
    np.testing.assert_allclose(np.ravel(chosen["check"]["outputs"]), (rows.astype(np.float32) @ W1.T + B1).sum(axis=1), rtol=1e-4, atol=1e-5)
    for wrong in ("third", 7):
        with pytest.raises(StructureError) as caught:
            from_onnx(model, feature_names=NAMES, output=wrong)
        assert caught.value.param == "output"


def test_a_conversion_that_does_not_give_the_librarys_outputs_is_drawn_not_run():
    model = mlp_model()

    def honest(feeds):
        return np.asarray(feeds["rows"], dtype=np.float32) @ W1.T + B1  # the hidden layer: not this model's output

    def exact(feeds):
        hidden = np.maximum(np.asarray(feeds["rows"], dtype=np.float32) @ W1.T + B1, 0)
        return hidden @ W2.T + B2

    def close(feeds):
        return exact(feeds) * (1 + 5e-4)

    def refuses(feeds):
        raise ValueError("no")

    common = dict(
        feature_names=NAMES, task="binary_classification", labels=None, inputs=None, input_shapes=None, samples=None,
        output=None, name=None, tokenizer=None, check_rows=None, framework="sklearn",
    )
    agreed = graphs._read_onnx(model, library=exact, **common)
    assert not agreed.get("approximate") and agreed["metadata"]["checked_against"] == "sklearn"
    np.testing.assert_allclose(agreed["check"]["outputs"], mlp_numpy(check_table(agreed)), rtol=1e-5, atol=1e-6)
    for library, words in ((honest, "could not be compared"), (close, "does not reproduce"), (refuses, "could not be compared")):
        spec = graphs._read_onnx(model, library=library, **common)
        assert spec["approximate"] is True and "check" not in spec, words
        assert words in spec["summary"] and "drawn, not run" in spec["summary"]
        assert not spec_is_executable(spec)


def test_the_check_is_the_librarys_numbers_not_the_conversions():
    """A library a hair away from its conversion: the check holds the library's outputs."""

    def library(feeds):
        hidden = np.maximum(np.asarray(feeds["rows"], dtype=np.float64) @ W1.T.astype(np.float64) + B1, 0)
        return (hidden @ W2.T.astype(np.float64) + B2) * (1 + 2e-5)

    spec = graphs._read_onnx(
        mlp_model(), library=library, feature_names=NAMES, task="regression", labels=None, inputs=None, input_shapes=None,
        samples=None, output=None, name=None, tokenizer=None, check_rows=None, framework="sklearn",
    )
    assert not spec.get("approximate")
    rows = check_table(spec)
    expected = np.stack([library({"rows": row.reshape(1, -1)}).reshape(-1) for row in rows])
    np.testing.assert_allclose(spec["check"]["outputs"], expected, rtol=1e-12)
    onnx_own = np.maximum(rows.astype(np.float32) @ W1.T + B1, 0) @ W2.T + B2
    assert np.abs(np.asarray(spec["check"]["outputs"]) - onnx_own).max() > 1e-6


def test_sizes_the_file_leaves_open_come_from_the_caller():
    open_model = model_of(
        [helper.make_node("ReduceMean", ["series"], ["mean"], axes=[1], keepdims=0, name="/ReduceMean")],
        [helper.make_tensor_value_info("series", TensorProto.FLOAT, ["batch", "steps", 2])],
        [helper.make_tensor_value_info("mean", TensorProto.FLOAT, ["batch", 2])],
        [],
    )
    with pytest.raises(StructureError) as caught:
        from_onnx(open_model, task="embedding")
    assert "input_shapes" in caught.value.message
    for given in ({"series": [1, 6, 2]}, [1, 6, 2], [6, 2]):
        spec = from_onnx(open_model, task="embedding", input_shapes=given)
        assert spec["features"] == [{"name": "series", "type": "sequence", "shape": [6, 2]}]
    sample = RNG.normal(size=(1, 9, 2)).astype(np.float32)
    sampled = from_onnx(open_model, task="embedding", samples=sample)
    assert sampled["features"][0]["shape"] == [9, 2]
    # The caller's example is the first row of the check.
    np.testing.assert_allclose(sampled["check"]["inputs"]["series"][0], sample.reshape(-1), rtol=1e-6)
    np.testing.assert_allclose(sampled["check"]["outputs"][0], sample[0].mean(axis=0), rtol=1e-5)
    with pytest.raises(StructureError) as caught:
        from_onnx(masked_model(), samples=np.zeros((1, 6)))
    assert caught.value.param == "samples"


def test_a_token_model_is_read_for_as_many_tokens_as_it_is_told():
    def writer(tokens):
        return model_of(
            [helper.make_node("Gather", ["next", "ids"], ["scores"], name="/lookup/Gather")],
            [helper.make_tensor_value_info("ids", TensorProto.INT64, ["batch", tokens])],
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", tokens, 5])],
            [numpy_helper.from_array(NEXT, "next")],
        )

    # The file leaves the number of tokens open: it is read for as many as max_length says.
    spec = from_onnx(writer("tokens"), task="text_generation", tokenizer={"type": "char", "vocab": LETTERS}, max_length=24)
    assert spec["features"][0]["max_length"] == 24 and spec["graph"]["inputs"][0]["shape"] == [1, 24]
    assert "check" in spec and not spec.get("approximate")
    # An example is filled up with the padding token, or cut down; a mask and segment ids follow it.
    padded = graphs._fit_tokens(
        {
            "input_ids": np.array([[5, 6, 7]]),
            "attention_mask": np.array([[1, 1, 1]]),
            "token_type_ids": np.array([[0, 1, 1]]),
            "position_ids": np.array([[0, 1, 2]]),
            "length": np.array([[3]]),
            "weights": np.ones((1, 3), dtype=np.float32),
        },
        {},
        9,
        5,
        np,
    )
    assert padded["input_ids"].tolist() == [[5, 6, 7, 9, 9]] and padded["attention_mask"].tolist() == [[1, 1, 1, 0, 0]]
    assert padded["token_type_ids"].tolist() == [[0, 1, 1, 0, 0]] and padded["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert set(padded) == {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    assert graphs._fit_tokens({"ids": np.array([[5, 6, 7]])}, {}, 0, 2, np)["ids"].tolist() == [[5, 6]]
    assert graphs._fit_tokens({"ids": np.array([[5, 6, 7]])}, {}, 0, 3, np) == {}
    sampled = from_onnx(writer("tokens"), task="text_generation", samples=np.array([[1, 2, 3]]), max_length=6)
    assert sampled["graph"]["inputs"][0]["shape"] == [1, 6]
    # With no max_length a language model is read for its example's length, and says so: it stops writing there.
    with pytest.warns(UserWarning, match="read for 3 tokens"):
        short = from_onnx(writer("tokens"), task="text_generation", samples=np.array([[1, 2, 3]]))
    assert short["features"][0]["max_length"] == 3
    # A file that fixes the number of tokens cannot be read for another, and warns of nothing it could not change.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert from_onnx(writer(7), task="text_generation")["features"][0]["max_length"] == 7
        assert from_onnx(writer(7), task="text_generation", max_length=7)["features"][0]["max_length"] == 7
    with pytest.raises(StructureError) as caught:
        from_onnx(writer(7), task="text_generation", max_length=12)
    assert caught.value.param == "max_length" and "exactly 7 tokens" in caught.value.message
    for wrong in (0, -3, 2.5, True):
        with pytest.raises(StructureError) as caught:
            from_onnx(writer("tokens"), task="text_generation", max_length=wrong)
        assert caught.value.param == "max_length"
    assert graphs._pad_token({"type": "word", "vocab": ["a", "<pad>"], "pad": "<pad>"}) == 1
    assert graphs._pad_token(None) == 0 and graphs._pad_token(None, {"ids": {"feature": "t", "pad_id": 4}}) == 4
    assert graphs._pad_token(type("Tok", (), {"pad_token_id": 3})()) == 3
    assert graphs._pad_token(type("Tok", (), {"padding": {"pad_id": 8}})()) == 8


def test_an_onnx_model_is_its_own_file():
    model = mlp_model()
    data = model.SerializeToString()
    assert graphs.to_onnx(model) == data and graphs.to_onnx(data) == data and graphs.to_onnx(bytearray(data)) == data
    with pytest.raises(StructureError) as caught:
        graphs.to_onnx(object())
    assert caught.value.code == "unsupported_estimator"
    assert extract_structure(model, feature_names=NAMES).onnx is None  # the caller has the file already


def test_onnx_has_to_be_installed_to_read_one(monkeypatch):
    monkeypatch.setattr(graphs, "_has", lambda package: False)
    with pytest.raises(StructureError) as caught:
        from_onnx(b"anything")
    assert caught.value.code == "onnx_required" and "pip install" in caught.value.message
    reasons = []
    assert graphs.via_onnx(object(), reasons=reasons) is None
    assert "onnx" in reasons[0]


def test_no_warning_is_raised_by_an_ordinary_import():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_onnx(mlp_model(), feature_names=NAMES, task="binary_classification")


@pytest.mark.parametrize("kind", ["scores", "indices", "keep_first", "keep_last"])
def test_segmentation_channel_axis_and_integer_masks(kind):
    indices = kind != "scores"
    nodes = [helper.make_node("Conv", ["pixels", "w", "b"], ["scores"])]
    if indices:
        if kind == "keep_last":
            nodes.append(helper.make_node("Transpose", ["scores"], ["nhwc"], perm=[0, 2, 3, 1]))
        nodes.append(helper.make_node("ArgMax", ["nhwc" if kind == "keep_last" else "scores"], ["mask"],
                                     axis=-1 if kind == "keep_last" else 1, keepdims=int(kind.startswith("keep"))))
    else:
        nodes.append(helper.make_node("Transpose", ["scores"], ["mask"], perm=[0, 2, 3, 1]))
    model = model_of(nodes, [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("mask", TensorProto.INT64 if indices else TensorProto.FLOAT,
                                       {"indices": [1, 16, 16], "scores": [1, 16, 16, 21],
                                        "keep_first": [1, 1, 16, 16], "keep_last": [1, 16, 16, 1]}[kind])],
        [numpy_helper.from_array(np.zeros((21, 3, 1, 1), dtype=np.float32), "w"),
         numpy_helper.from_array(np.arange(21, dtype=np.float32), "b")])
    spec = extract_structure(model, task=None if indices else "segmentation").spec
    assert spec["task"] == "segmentation"
    assert spec["head"]["layout"] == ("indices" if indices else "hwc")
    assert spec_is_executable(spec) and not spec.get("approximate")
    actual = np.asarray(spec["check"]["outputs"][0])
    if not indices:
        actual = np.argmax(actual.reshape(16, 16, 21), axis=2)
    np.testing.assert_array_equal(actual.reshape(16, 16), np.full((16, 16), 20))


def test_ambiguous_segmentation_axes_require_labels_or_an_explicit_layout():
    model = model_of([helper.make_node("Relu", ["pixels"], ["scores"])],
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 3, 4, 5])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3, 4, 5])], [])
    with pytest.raises(StructureError, match="mask_layout") as caught:
        extract_structure(model, task="segmentation")
    assert caught.value.param == "mask_layout"
    assert from_onnx(model, task="segmentation", labels=["a", "b", "c"])["head"]["layout"] == "chw"
    forced = extract_structure(model, mask_layout="hwc").spec
    assert forced["head"]["layout"] == "hwc" and forced["task"] == "segmentation"
