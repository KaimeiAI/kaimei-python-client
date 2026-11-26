"""Fixes to the conversion through ONNX (graphs.py): what an upload claims is what the model does, and the model is
left as it was found."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from kaimei import StructureError, graphs  # noqa: E402
from kaimei.structure import extract_structure, finalize_spec  # noqa: E402

RNG = np.random.default_rng(0)
X = RNG.normal(size=(150, 4))
NAMES = ["f0", "f1", "f2", "f3"]


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def sklearn():
    pytest.importorskip("skl2onnx")
    return pytest.importorskip("sklearn")


def check_rows(spec):
    count = len(spec["check"]["outputs"])
    return np.array([[spec["check"]["inputs"][name][row][0] for name in spec["feature_names"]] for row in range(count)])


def finished(nodes, inputs, outputs, initializers=(), functions=(), opset=17, domains=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, list(initializers))
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset), *[helper.make_opsetid(domain, 1) for domain in domains]],
        functions=list(functions),
    )
    model.ir_version = 9
    return model


def table(name, width):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [None, width])


# --- 1. a multi-label scikit-learn classifier -----------------------------------------------------------------


def labels_of(rows):
    return np.stack([rows[:, 0] > 0, rows[:, 1] + rows[:, 2] > 0, rows[:, 3] > 0.5], axis=1).astype(int)


def test_a_multilabel_classifier_is_read_with_its_own_probabilities(sklearn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neural_network import MLPClassifier

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = [
            OneVsRestClassifier(LogisticRegression()).fit(X, labels_of(X)),
            MLPClassifier(hidden_layer_sizes=(5,), max_iter=300, random_state=0).fit(X, labels_of(X)),
        ]
    for model in models:
        spec = graphs.via_onnx(model, feature_names=NAMES, reasons=(why := []))
        assert spec is not None, why
        # A yes or a no for each label, each with its own probability: not one class among three, through a softmax.
        assert spec["task"] == "multilabel_classification" and spec["head"]["link"] == "none", type(model).__name__
        assert not spec.get("approximate") and spec["outputs"] == {"labels": ["0", "1", "2"]}
        rows = check_rows(spec)
        np.testing.assert_allclose(spec["check"]["outputs"], model.predict_proba(rows), rtol=1e-4, atol=1e-5)
        assert spec["metadata"]["library_method"] == "predict_proba"
    routed = extract_structure(models[0], feature_names=NAMES).spec  # the native readers refuse it; the conversion reads it
    assert routed["family"] == "graph" and routed["task"] == "multilabel_classification"


def test_a_multilabel_classifier_with_scores_squashes_each_on_its_own(sklearn):
    from sklearn.linear_model import RidgeClassifier

    ridge = RidgeClassifier().fit(X, labels_of(X))
    spec = graphs.via_onnx(ridge, feature_names=NAMES)
    assert spec["task"] == "multilabel_classification" and spec["head"]["link"] == "sigmoid"
    rows = check_rows(spec)
    np.testing.assert_allclose(spec["check"]["outputs"], 1 / (1 + np.exp(-ridge.decision_function(rows))), rtol=1e-4, atol=1e-5)
    assert "each label's share shown is its own score squashed between 0 and 1" in spec["summary"]


def test_a_multilabel_classifier_is_not_read_as_one_class_among_several(sklearn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    ovr = OneVsRestClassifier(LogisticRegression()).fit(X, labels_of(X))
    # The caller's own task is wrong for this model: said so, naming the argument, on every route.
    with pytest.raises(StructureError) as caught:
        graphs.via_onnx(ovr, feature_names=NAMES, task="multiclass_classification")
    assert caught.value.param == "task" and "task='multilabel_classification'" in caught.value.message
    for via in ("onnx", "auto"):
        with pytest.raises(StructureError) as caught:
            extract_structure(ovr, feature_names=NAMES, task="multiclass_classification", via=via)
        assert caught.value.param == "task" and "a yes or a no of its own" in caught.value.message, via


def test_a_multi_output_classifier_is_refused_with_its_reason(sklearn):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(3).fit(X, labels_of(X))  # classes_ is a list: the classes of each output
    why = []
    assert graphs.via_onnx(knn, feature_names=NAMES, reasons=why) is None
    assert "multi-output classifier" in why[0] and "can only convert" not in why[0]


def test_the_probabilities_of_a_converted_classifier_are_never_squashed_again(sklearn):
    from sklearn.linear_model import LogisticRegression

    # Probabilities read where the file leaves them, whatever the importer would make of their sums.
    model = LogisticRegression().fit(X, np.digitize(X[:, 0], [-0.5, 0.5]))
    spec = graphs.via_onnx(model, feature_names=NAMES)
    assert spec["task"] == "multiclass_classification" and spec["head"]["link"] == "none"
    np.testing.assert_allclose(spec["check"]["outputs"], model.predict_proba(check_rows(spec)), rtol=1e-4, atol=1e-5)


# --- 2. a Keras model with several outputs ---------------------------------------------------------------------


class TwoHeads:
    """Stands in for a Keras model with two outputs: called as Keras calls one, it answers with a list of both."""

    def __init__(self):
        self.left = np.array([[1.0], [2.0], [0.5]])
        self.right = np.array([[-1.0], [0.25], [3.0]])

    def __call__(self, x, training=False):
        rows = np.asarray(x, dtype=np.float64)
        return [rows @ self.left, rows @ self.right]


def two_head_file(heads):
    left = numpy_helper.from_array(heads.left.astype(np.float32), "left")
    right = numpy_helper.from_array(heads.right.astype(np.float32), "right")
    return finished(
        [helper.make_node("MatMul", ["x", "left"], ["first"]), helper.make_node("MatMul", ["x", "right"], ["second"])],
        [table("x", 3)],
        [table("first", 1), table("second", 1)],
        [left, right],
    )


def test_a_keras_model_is_compared_with_the_output_it_is_read_at(monkeypatch):
    heads = TwoHeads()
    proto = two_head_file(heads)
    monkeypatch.setattr(graphs, "_has", lambda package: True)
    monkeypatch.setattr(graphs, "_keras_to_onnx", lambda estimator, sample, why, numpy: proto)
    options = {
        "feature_names": ["a", "b", "c"], "task": "regression", "sample": None, "inputs": None, "check_rows": None,
        "labels": None, "tokenizer": None, "framework": None, "name": None,
    }
    for output, weights in (("second", heads.right), (1, heads.right), ("first", heads.left)):
        spec = graphs._keras_through_onnx(heads, [], np, output=output, **options)
        assert not spec.get("approximate"), (output, spec.get("summary"))
        np.testing.assert_allclose(spec["check"]["outputs"], check_rows(spec) @ weights, rtol=1e-5, atol=1e-6)


def test_a_real_keras_model_with_two_outputs_is_read_at_the_one_asked_for():
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tf2onnx")
    keras = tf.keras
    given = keras.Input((3,), name="x")
    first = keras.layers.Dense(2, name="first")(given)
    second = keras.layers.Dense(1, name="second")(keras.layers.Dense(4, activation="tanh")(given))
    model = keras.Model(given, [first, second])
    sample = RNG.normal(size=(1, 3)).astype(np.float32)
    spec = graphs.via_onnx(model, feature_names=["a", "b", "c"], task="regression", sample=sample, output=1, reasons=(why := []))
    assert spec is not None and not spec.get("approximate"), (why, spec and spec.get("summary"))
    own = np.asarray(model(check_rows(spec).astype(np.float32), training=False)[1])
    np.testing.assert_allclose(spec["check"]["outputs"], own, rtol=1e-4, atol=1e-5)


# --- 3. externalize_above=0 -------------------------------------------------------------------------------------


def test_a_single_value_is_never_lifted_into_the_tensor_upload():
    spec = {
        "family": "graph",
        "graph": {
            "constants": {
                "scale": np.array(2.0, dtype=np.float32),  # no axes
                "one": np.array([0.5], dtype=np.float32),  # one value
                "w": np.arange(6, dtype=np.float32).reshape(3, 2),
            },
            "nodes": [],
        },
    }
    done = graphs.finalize_graph(spec, externalize_above=0)
    assert set(done.tensors) == {"w"}
    constants = done.spec["graph"]["constants"]
    assert constants["scale"] == {"dtype": "float32", "shape": [], "data": [2.0]}
    assert constants["one"] == {"dtype": "float32", "shape": [1], "data": [0.5]}
    for array in done.tensors.values():
        assert ",".join(str(int(dim)) for dim in array.shape) == "3,2"  # the shape the upload sends with it


# --- 4. the module's device and float type ----------------------------------------------------------------------


def test_what_a_conversion_gives_a_module_is_made_on_its_device_and_in_its_float_type(torch):
    nn = torch.nn
    double = nn.Sequential(nn.Conv2d(3, 2, 3), nn.Flatten()).double()
    wrapper = graphs._preprocessed(double, {"mean": [0.4, 0.5, 0.6], "std": [0.2, 0.2, 0.2]}, torch)
    assert {buffer.dtype for buffer in wrapper.buffers() if buffer is not None} == {torch.float64}
    assert graphs._placement(double, torch) == (torch.device("cpu"), torch.float64)
    # No GPU here: a module on the meta device stands for one on another device than the CPU.
    elsewhere = nn.Sequential(nn.Conv2d(3, 2, 3), nn.Flatten()).to("meta")
    moved = graphs._preprocessed(elsewhere, {"mean": [0.4, 0.5, 0.6]}, torch)
    assert {buffer.device.type for buffer in [moved.mean, moved.std, moved.scale, moved.offset]} == {"meta"}
    started = graphs._decoder_start(5, 0, 2, graphs._placement(elsewhere, torch)[0], torch)
    assert started.device.type == "meta" and started.dtype == torch.long and tuple(started.shape) == (1, 5)
    example = graphs._torch_example(elsewhere, ["x"], None, (3, 8, 8), None, torch)
    assert example["x"].device.type == "meta"


def test_a_double_precision_image_model_with_preprocessing_is_read_in_double_precision(torch):
    nn = torch.nn
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).double().eval()
    spec = graphs.via_onnx(
        model, input_shape=(3, 12, 12), task="multiclass_classification", preprocess={"mean": [0.5, 0.4, 0.3], "std": [0.2, 0.3, 0.25]},
        reasons=(why := []),
    )
    assert spec is not None and not spec.get("approximate"), (why, spec and spec.get("summary"))
    assert [item["dtype"] for item in spec["graph"]["inputs"]] == ["float64"]


# --- 5. ONNX Runtime without a kernel for a step ----------------------------------------------------------------


class _Refusal(Exception):
    pass


def refusal(module, message):
    kind = type("RuntimeException", (_Refusal,), {"__module__": module})
    return kind(message)


def test_only_onnx_runtimes_missing_kernels_hand_a_run_to_the_evaluator():
    lacks = graphs._onnxruntime_lacks_a_kernel
    said = "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : ... Status Message: LSTM operator does not support double yet"
    assert lacks(refusal("onnxruntime.capi.onnxruntime_pybind11_state", said))
    assert lacks(type("NotImplemented", (Exception,), {"__module__": "onnxruntime.capi.onnxruntime_pybind11_state"})("x"))
    assert not lacks(refusal("onnxruntime.capi.onnxruntime_pybind11_state", "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions"))
    assert not lacks(refusal("somewhere.else", "operator does not support double yet"))


def test_a_run_onnx_runtime_refuses_for_another_reason_is_not_handed_on():
    pytest.importorskip("onnxruntime")
    model = finished([helper.make_node("Relu", ["x"], ["y"])], [table("x", 3)], [table("y", 3)])
    runner = graphs._OnnxRunner(graphs._importer(), model, "y")
    assert runner.engine.startswith("onnxruntime")
    with pytest.raises(Exception) as caught:
        runner({"x": np.zeros((1, 5), dtype=np.float32)})  # the wrong width: the input's fault, not a missing kernel
    assert not graphs._onnxruntime_lacks_a_kernel(caught.value) and runner.engine.startswith("onnxruntime")


def test_a_double_precision_lstm_is_compared_with_its_module(torch):
    pytest.importorskip("onnxruntime")
    nn = torch.nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm, self.head = nn.LSTM(3, 8, batch_first=True), nn.Linear(8, 1)

        def forward(self, window):
            out, _ = self.lstm(window)
            return self.head(out[:, -1])

    torch.manual_seed(0)
    model = Forecaster().double().eval()
    sample = RNG.normal(size=(1, 24, 3))
    spec = graphs.via_onnx(model, sample=sample, task="regression", reasons=(why := []))
    assert spec is not None and not spec.get("approximate"), (why, spec and spec.get("summary"))
    assert spec["metadata"]["checked_against"] == "pytorch" and spec["graph"]["inputs"][0]["dtype"] == "float64"
    rows = np.asarray(spec["check"]["inputs"]["window"], dtype=np.float64).reshape(-1, 24, 3)
    with torch.no_grad():
        own = model(torch.tensor(rows)).numpy()
    np.testing.assert_allclose(spec["check"]["outputs"], own, rtol=1e-6, atol=1e-8)


# --- 6. whole numbers of NumPy's ---------------------------------------------------------------------------------


def test_numpy_whole_numbers_say_how_many_tokens_and_which_output(torch):
    assert graphs._whole_length(np.int64(8)) == 8 and type(graphs._whole_length(np.int32(8))) is int
    for wrong in (True, np.bool_(True), 2.0, np.float64(3.0), 0, np.int64(0)):
        with pytest.raises(StructureError):
            graphs._whole_length(wrong)
    assert graphs._torch_output([torch.zeros(1), torch.zeros(2)], np.int64(1)) == 1
    with pytest.raises(StructureError):
        graphs._torch_output([torch.zeros(1), torch.zeros(2)], np.bool_(True))
    proto = two_head_file(TwoHeads())
    assert graphs._prediction_output(proto, np.int64(1)) == ("second", "second")

    class Reviews(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.head = torch.nn.Embedding(9, 6), torch.nn.Linear(6, 2)

        def forward(self, ids):
            return self.head(self.embed(ids).mean(dim=1))

    lengths = np.array([3, 6, 5])
    spec = graphs.via_onnx(Reviews().eval(), sample=torch.tensor([[2, 6, 3]]), max_length=lengths.max(), task="binary_classification")
    assert spec is not None and spec["features"][0]["max_length"] == 6


# --- 7. what scikit-learn says an estimator is ------------------------------------------------------------------


def test_a_regressor_and_a_clusterer_are_known_on_current_scikit_learn(sklearn):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression

    assert graphs._is_regressor(LinearRegression().fit(X, X[:, 0])) and not graphs._is_regressor(LogisticRegression())
    assert graphs._is_clusterer(KMeans(2, n_init=1).fit(X)) and not graphs._is_clusterer(LinearRegression())

    class Older:
        _estimator_type = "regressor"  # what estimators of older scikit-learn, and some libraries', say

    assert graphs._is_regressor(Older())

    class Scored(LinearRegression):
        def decision_function(self, rows):
            return self.predict(rows)

    # A regressor with a decision_function is still a regressor: its answer is predict, not an outlier score.
    method, name = graphs._sklearn_answer(Scored().fit(X, X[:, 0]), False)
    assert name == "predict"


# --- 8. a random step inside a function the file defines --------------------------------------------------------


def noisy_function():
    return helper.make_function(
        "local", "Noisy", ["a"], ["b"],
        [helper.make_node("RandomNormalLike", ["a"], ["n"]), helper.make_node("Add", ["a", "n"], ["b"])],
        [helper.make_opsetid("", 17)],
    )


def weights():
    return numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0, "w")


def seeded_answers(model, spec):
    """What the model gives on the check's rows with the seeds the importer gives its random steps, by ONNX's evaluator."""
    from onnx.reference import ReferenceEvaluator

    from kaimei.onnx_reader import expand_functions

    tools = graphs._importer()
    seeded = tools.fold_constants(expand_functions(model))
    tools._pin_seeds(seeded.graph)
    rows = check_rows(spec).astype(np.float32)
    return np.stack([ReferenceEvaluator(seeded).run(None, {"x": row.reshape(1, -1)})[0].reshape(-1) for row in rows])


def test_a_random_step_inside_a_local_function_is_checked_with_fixed_seeds():
    model = finished(
        [helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Noisy", ["h"], ["y"], domain="local")],
        [table("x", 4)], [table("y", 3)], [weights()], functions=[noisy_function()], domains=["local"],
    )
    spec = graphs.from_onnx(model, task="generation", feature_names=NAMES)
    again = graphs.from_onnx(model, task="generation", feature_names=NAMES)
    assert spec["metadata"]["seeded_steps"] == 1 and not spec.get("approximate")
    assert spec["metadata"]["checked_against"].startswith("onnx reference evaluator")  # not ONNX Runtime's own draws
    np.testing.assert_allclose(spec["check"]["outputs"], again["check"]["outputs"])
    np.testing.assert_allclose(spec["check"]["outputs"], seeded_answers(model, spec), rtol=1e-6, atol=1e-6)


def test_a_random_step_in_a_function_called_from_a_branch_is_checked_with_fixed_seeds():
    then_branch = helper.make_graph([helper.make_node("Noisy", ["h"], ["t"], domain="local")], "then", [], [table("t", 3)])
    else_branch = helper.make_graph([helper.make_node("Identity", ["h"], ["e"])], "else", [], [table("e", 3)])
    model = finished(
        [
            helper.make_node("MatMul", ["x", "w"], ["h"]),
            helper.make_node("ReduceSum", ["h"], ["total"], keepdims=0),
            helper.make_node("Greater", ["total", "zero"], ["positive"]),
            helper.make_node("If", ["positive"], ["y"], then_branch=then_branch, else_branch=else_branch),
        ],
        [table("x", 4)], [table("y", 3)], [weights(), helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])],
        functions=[noisy_function()], domains=["local"],
    )
    spec = graphs.from_onnx(model, task="generation", feature_names=NAMES)
    assert spec["metadata"]["checked_against"].startswith("onnx reference evaluator") and not spec.get("approximate")
    np.testing.assert_allclose(spec["check"]["outputs"], seeded_answers(model, spec), rtol=1e-6, atol=1e-6)


# --- 9. one value for a row, one name -----------------------------------------------------------------------------


def test_a_classifier_that_gives_one_value_needs_two_labels(sklearn, torch):
    from sklearn.svm import SVC

    binary = (X[:, 0] > 0).astype(int)
    with pytest.raises(StructureError) as caught:
        extract_structure(SVC().fit(X, binary), feature_names=NAMES, labels=["yes"])
    assert caught.value.param == "labels" and "labels=[negative, positive]" in caught.value.message
    named = extract_structure(SVC().fit(X, binary), feature_names=NAMES, labels=["no", "yes"]).spec
    assert named["outputs"] == {"labels": ["no", "yes"]}
    torch.manual_seed(0)
    one = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)).eval()
    for task in ("binary_classification", "multiclass_classification"):
        with pytest.raises(StructureError) as caught:
            extract_structure(one, feature_names=NAMES, labels=["yes"], task=task, via="onnx")
        assert caught.value.param == "labels", task
    read = extract_structure(one, feature_names=NAMES, labels=["no", "yes"], task="binary_classification", via="onnx").spec
    assert read["outputs"] == {"labels": ["no", "yes"]} and read["head"]["link"] == "sigmoid"


# --- 10. a fixed input under weights="summary" ------------------------------------------------------------------


def scaled_model():
    return finished(
        [helper.make_node("Mul", ["x", "scale"], ["scaled"]), helper.make_node("MatMul", ["scaled", "w"], ["y"])],
        [table("x", 4), helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1, 4])],
        [table("y", 2)],
        [numpy_helper.from_array(np.ones((4, 2), dtype=np.float32), "w")],
    )


def fixed_value(spec):
    return next(item["from"]["value"] for item in spec["graph"]["inputs"] if "value" in (item.get("from") or {}))


def test_a_summary_sends_no_float_values_an_input_is_always_given():
    learned = np.array([[0.1234, 5.6789, -2.468, 9.1011]], dtype=np.float32)
    for given in (learned, learned.tolist()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = graphs.from_onnx(scaled_model(), task="generation", feature_names=NAMES, inputs={"scale": {"value": given}})
        summary = finalize_spec(raw, weights="summary").spec
        assert fixed_value(summary) == {"$shape": [1, 4], "dtype": "f32"}
        assert "0.1234" not in json.dumps(summary) and summary["approximate"] is True
        assert "stayed in your environment" in summary["summary"]
        full = finalize_spec(raw).spec  # sent in full, the value goes as before, and the model runs
        assert np.allclose(np.asarray(fixed_value(full)["data"] if isinstance(fixed_value(full), dict) else fixed_value(full)).reshape(-1), learned.reshape(-1))
        assert not full.get("approximate")
    for kept in (0.5, np.array([[1, 2, 3, 4]], dtype=np.int64)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = graphs.from_onnx(scaled_model(), task="generation", feature_names=NAMES, inputs={"scale": {"value": kept}})
        summary = finalize_spec(raw, weights="summary").spec
        # A single number is a setting, and whole numbers are shapes and indices: both stay, and say nothing learned.
        assert "$shape" not in json.dumps(fixed_value(summary)) and not summary.get("approximate")


# --- 11. a language model whose forward takes logits_to_keep ----------------------------------------------------


def test_a_language_model_whose_forward_takes_logits_to_keep_is_converted_and_compared(torch):
    transformers = pytest.importorskip("transformers")
    config = transformers.OPTConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, ffn_dim=32, max_position_embeddings=16,
        word_embed_proj_dim=16,
    )
    torch.manual_seed(4)
    model = transformers.OPTForCausalLM(config).eval()
    assert graphs._bound_settings(model, {"input_ids": None}) == {"logits_to_keep": 0, "use_cache": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(model, sample=torch.tensor([[5, 7, 9, 11]]), max_length=8).spec
    assert spec["task"] == "text_generation" and not spec.get("approximate") and "check" in spec
    assert [item["name"] for item in spec["graph"]["inputs"]] == ["in.input_ids"]  # logits_to_keep is no input
    assert not any(str(node.get("path", "")).startswith("inner.") for node in spec["graph"]["nodes"])
    assert model.config.use_cache is True  # put back as it was


def test_an_export_that_adds_an_input_is_refused_with_why(torch, monkeypatch):
    nn = torch.nn
    net = nn.Sequential(nn.Linear(4, 3)).eval()
    added = finished(
        [helper.make_node("Gemm", ["input", "w", "b"], ["h"], transB=1), helper.make_node("Add", ["h", "extra"], ["output"])],
        [table("input", 4), helper.make_tensor_value_info("extra", TensorProto.FLOAT, [])],
        [table("output", 3)],
        [numpy_helper.from_array(np.ones((3, 4), np.float32), "w"), numpy_helper.from_array(np.zeros(3, np.float32), "b")],
    )
    monkeypatch.setattr(graphs, "_export_torch", lambda module, example, outputs, torch: added.SerializeToString())
    why = []
    assert graphs.via_onnx(net, feature_names=NAMES, task="generation", reasons=why) is None
    assert "forward was not called with (extra)" in why[0]


def test_a_conversion_that_could_not_be_compared_says_why():
    model = finished([helper.make_node("Relu", ["x"], ["y"])], [table("x", 4)], [table("y", 4)])
    options = {
        "feature_names": NAMES, "task": "generation", "labels": None, "inputs": None, "input_shapes": None, "samples": None,
        "output": None, "name": None, "tokenizer": None, "check_rows": None, "framework": "test",
    }
    spec = graphs._read_onnx(model, library=lambda feeds: np.zeros((1, 6)), **options)
    assert spec["approximate"] and "check" not in spec
    assert "could not be compared with the model it was converted from (the conversion gives 4 values for a row, and the model 6)" in spec["summary"]
    assert graphs.drawn_reason(spec) == spec["summary"]
    # As a reason in a list, for a refusal to say what happened rather than that a comparison failed.
    assert graphs._drawn_because(spec) == (
        "it could not be compared with the model it was converted from (the conversion gives 4 values for a row, and the model 6)"
    )
    differs = {"approximate": True, "summary": "X. Its ONNX conversion does not reproduce the model's own outputs (they differ by 1.0e-01 of their size), so it is drawn, not run."}
    assert graphs._drawn_because(differs) == "its ONNX conversion does not reproduce the model's own outputs (they differ by 1.0e-01 of their size)"
    assert graphs._drawn_because({"summary": "A graph of 3 operations."}) is None


def test_a_conversion_whose_outputs_cannot_be_told_apart_is_not_checked_against_itself():
    model = finished(
        [helper.make_node("Relu", ["x"], ["y"])], [table("x", 4)], [helper.make_empty_tensor_value_info("y")]
    )
    options = {
        "feature_names": NAMES, "task": "generation", "labels": None, "inputs": None, "input_shapes": None, "samples": None,
        "output": None, "name": None, "tokenizer": None, "check_rows": None, "framework": "test",
    }
    # With a model to compare with, the importer's own check (the conversion against itself) is not enough.
    spec = graphs._read_onnx(model, library=lambda feeds: np.maximum(feeds["x"], 0), **options)
    assert spec["approximate"] and "check" not in spec and "could not be compared" in spec["summary"]
    alone = graphs._read_onnx(model, library=None, **options)  # an ONNX file is its own original
    assert "check" in alone and not alone.get("approximate")


# --- 12. a model that draws at random ---------------------------------------------------------------------------


def test_a_model_that_draws_says_its_check_is_not_against_the_original(torch):
    nn = torch.nn

    class Noisy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x) + torch.randn_like(x[:, :2])

    torch.manual_seed(0)
    net = Noisy().eval()
    torch.manual_seed(123)
    state = torch.random.get_rng_state().clone()
    spec = graphs.via_onnx(net, feature_names=NAMES, task="generation", reasons=(why := []))
    assert spec is not None, why
    assert torch.equal(torch.random.get_rng_state(), state)  # converting it drew nothing the caller's program will miss
    meta = spec["metadata"]
    assert meta["compared_with_original"] is False and meta["seeded_steps"] == 1
    assert "pytorch" not in meta["checked_against"] and "fixed seeds" in meta["checked_against"]
    assert graphs.DRAWS_AT_RANDOM in spec["summary"] and "not with the original model's own outputs" in spec["summary"]
    plain = graphs.via_onnx(nn.Sequential(nn.Linear(4, 2)).eval(), feature_names=NAMES, task="generation")
    assert "compared_with_original" not in plain["metadata"] and graphs.DRAWS_AT_RANDOM not in plain["summary"]


# --- 14. an SVC that counts votes -------------------------------------------------------------------------------


def four_classes(rows):
    return np.digitize(rows[:, 0] + 0.5 * rows[:, 1], [-0.8, 0.0, 0.8])


def test_an_svc_that_counts_votes_says_its_class_can_differ_from_predict(sklearn):
    from sklearn.svm import SVC

    model = SVC().fit(X, four_classes(X))
    with pytest.warns(UserWarning, match="counts one-vs-one votes"):
        spec = graphs.via_onnx(model, feature_names=NAMES)
    # The scores are the model's, and run: what is said is how the class is chosen from them.
    assert not spec.get("approximate") and "check" in spec
    assert spec["metadata"]["predicted_class"] == graphs.VOTES_NOTE
    assert "can order differently where one-vs-one votes tie" in spec["summary"]
    probe = RNG.normal(size=(4000, 4)) * 1.5
    tied = probe[model.classes_[model.decision_function(probe).argmax(axis=1)] != model.predict(probe)][:3]
    assert len(tied) == 3  # rows where the votes tie, and predict() takes another class than the highest score
    # Checked on such rows, the reading's class is not the model's own on them: it is not the model, and says so.
    checked = graphs.via_onnx(model, feature_names=NAMES, check_rows=tied)
    assert checked["approximate"] is True and "check" not in checked
    said = re.search(re.escape(graphs.NOT_ITS_CLASS) + r" on (\d+) of the (\d+) rows", checked["summary"])
    # The three rows given, and any of the rows made up from their values on which the votes tie as well.
    assert said is not None and 3 <= int(said.group(1)) <= int(said.group(2)) == 3 + graphs.TABLE_CHECK_ROWS
    assert "counts one-vs-one votes" in checked["summary"] and "break_ties=True" in checked["summary"]
    agreeing = SVC(break_ties=True).fit(X, four_classes(X))  # predict() goes by the same scores
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = graphs.via_onnx(agreeing, feature_names=NAMES)
    assert "predicted_class" not in quiet["metadata"]


def test_the_pairwise_scores_of_an_svc_are_not_read_as_class_scores(sklearn):
    from sklearn.svm import SVC

    for count, labels in ((3, np.digitize(X[:, 0], [-0.5, 0.5])), (4, four_classes(X))):
        model = SVC(decision_function_shape="ovo").fit(X, labels)  # three classes have three pairs, as many as classes
        why = []
        assert graphs.via_onnx(model, feature_names=NAMES, reasons=why) is None, count
        assert "decision_function_shape='ovo'" in why[0] and "labels has" not in why[0]


# --- 15. inputs= that names no input of the model ---------------------------------------------------------------


def test_inputs_that_names_no_input_of_the_model_is_refused(sklearn):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(X, (X[:, 0] > 0).astype(int))
    for call in (
        lambda: graphs.via_onnx(model, feature_names=NAMES, inputs={"nope": {"features": NAMES}}),
        lambda: extract_structure(model, feature_names=NAMES, inputs={"nope": {"zeros": True}}, via="onnx"),
    ):
        with pytest.raises(StructureError) as caught:
            call()
        assert caught.value.param == "inputs" and "'nope'" in caught.value.message and "(X)" in caught.value.message
    named = graphs.via_onnx(model, feature_names=NAMES, inputs={"X": {"features": ["p", "q", "r", "s"]}})  # its own input
    assert named["feature_names"] == ["p", "q", "r", "s"] and not named.get("approximate")
    relu = finished([helper.make_node("Relu", ["x"], ["y"])], [table("x", 4)], [table("y", 4)])
    with pytest.raises(StructureError) as caught:
        graphs.from_onnx(relu, task="generation", inputs={"nope": {"features": NAMES}})
    assert caught.value.param == "inputs"
    assert graphs.from_onnx(relu, task="generation", inputs={"x": {"features": NAMES}})["feature_names"] == NAMES


def test_a_hint_for_a_modules_own_input_still_reaches_it_when_preprocessing_wraps_it(torch):
    nn = torch.nn
    torch.manual_seed(0)
    cnn = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).eval()
    for key in ("input", "image"):  # forward's own name for it, or the wrapper's
        spec = graphs.via_onnx(
            cnn, input_shape=(3, 12, 12), task="multiclass_classification", preprocess={"mean": [0.5, 0.5, 0.5]},
            inputs={key: {"feature": "photo", "layout": "nchw"}},
        )
        assert spec["feature_names"] == ["photo"] and not spec.get("approximate"), key
    with pytest.raises(StructureError) as caught:
        graphs.via_onnx(cnn, input_shape=(3, 12, 12), preprocess={"mean": [0.5, 0.5, 0.5]}, inputs={"pixels": {"feature": "photo"}})
    assert caught.value.param == "inputs"

    class Unread(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x, mask):
            return self.fc(x)  # mask is never read, and the exporter leaves it out of the file

    # A hint for an input the exporter left out is no mistake of the caller's: there is nothing to say of it.
    kept = graphs.via_onnx(Unread().eval(), sample=(torch.zeros(1, 4), torch.ones(1, 4)), task="generation", inputs={"mask": {"zeros": True}})
    assert kept is not None and not kept.get("approximate")


# --- 13. inputs left out under empty names inside a function the file defines ---------------------------------

_FUNCTION_DFT = r"""
import warnings
import numpy as np
from onnx import TensorProto, helper
from kaimei import graphs, onnx_graph

warnings.simplefilter("ignore")
spectrum = helper.make_function("local", "Spectrum", ["a"], ["b"], [helper.make_node("DFT", ["a", "", ""], ["b"])],
                                [helper.make_opsetid("", 20)])
graph = helper.make_graph([helper.make_node("Spectrum", ["x"], ["y"], domain="local")], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("local", 1)], functions=[spectrum])
model.ir_version = 9
runner = graphs._OnnxRunner(onnx_graph, model, "y")
print("ran", runner(dict(x=np.arange(8, dtype=np.float32).reshape(1, 8, 1))).shape, flush=True)
spec = graphs.from_onnx(model, task="generation")
print("read", spec.get("approximate"), spec["metadata"].get("checked_against"), flush=True)
print("left", list(model.functions[0].node[0].input), flush=True)
"""


def test_a_function_with_inputs_left_out_under_empty_names_does_not_stop_the_process(tmp_path):
    pytest.importorskip("onnxruntime")
    script = tmp_path / "function_dft.py"
    script.write_text(_FUNCTION_DFT, encoding="utf-8")
    import kaimei

    source = os.path.dirname(os.path.dirname(os.path.abspath(kaimei.__file__)))  # the kaimei these tests import
    env = {**os.environ, "PYTHONPATH": source, "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr[-2000:])  # was killed by ONNX Runtime
    assert "ran (1, 8, 2)" in done.stdout and "read None onnxruntime" in done.stdout
    assert "left ['a', '', '']" in done.stdout  # the caller's model is left as it was
