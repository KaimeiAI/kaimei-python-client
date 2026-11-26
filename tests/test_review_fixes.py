"""Fixes for the review of the ONNX route (items 4, 10, 12, 13 and 14 of the general-models review)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from kaimei import StructureError, graphs  # noqa: E402
from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_a_model_that_draws_is_checked_against_its_seeded_reading():
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.eye(3, dtype=np.float32) * 2, "w")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["y"]),
                helper.make_node("RandomNormalLike", ["y"], ["noise"]),
                helper.make_node("Add", ["y", "noise"], ["out"]),
            ],
            "draws",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 3])],
            [weight],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    spec = graphs.from_onnx(model, task="generation")
    assert spec["metadata"]["seeded_steps"] == 1 and not spec.get("approximate") and "check" in spec
    # The check is what the seeded model gives, on ONNX's evaluator: run again, the same numbers come out.
    from onnx.reference import ReferenceEvaluator

    tools = graphs._importer()
    seeded = tools.fold_constants(model)
    tools._pin_seeds(seeded.graph)
    names = [name for name in spec["feature_names"]]
    for index, answer in enumerate(spec["check"]["outputs"]):
        row = np.array([[spec["check"]["inputs"][name][index][0] for name in names]], dtype=np.float32)
        again = ReferenceEvaluator(seeded).run(None, {"x": row})[0]
        np.testing.assert_allclose(answer, again.reshape(-1), rtol=1e-5, atol=1e-5)


def test_a_numpy_sample_is_given_to_the_module_in_its_own_float_type(torch):
    nn = torch.nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm, self.head = nn.LSTM(3, 8, batch_first=True), nn.Linear(8, 1)

        def forward(self, series):
            out, _ = self.lstm(series)
            return self.head(out[:, -1])

    why = []
    spec = graphs.via_onnx(Forecaster().eval(), sample=np.zeros((1, 24, 3)), task="regression", reasons=why)
    assert spec is not None and not why and "check" in spec
    extracted = extract_structure(Forecaster().eval(), sample=np.random.default_rng(0).normal(size=(1, 24, 3)), task="regression").spec
    assert extracted["family"] == "graph" and spec_is_executable(extracted)


def test_on_unsupported_error_refuses_a_graph_that_would_only_be_drawn(torch, monkeypatch):
    nn = torch.nn
    net = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)).eval()
    real = graphs._checked_against

    def far(*args, **kwargs):
        check, _difference = real(*args, **kwargs)
        return check, 1.0  # as if the conversion were nowhere near the module

    monkeypatch.setattr(graphs, "_checked_against", far)
    drawn = extract_structure(net, sample=torch.randn(1, 4), labels=["a", "b"], via="onnx").spec
    assert drawn.get("approximate") and "drawn, not run" in drawn["summary"]
    with pytest.raises(StructureError) as caught:
        extract_structure(net, sample=torch.randn(1, 4), labels=["a", "b"], via="onnx", on_unsupported="error")
    assert caught.value.code == "unsupported_layers" and "on_unsupported='error'" in caught.value.message


def test_labels_of_the_wrong_count_are_refused_on_the_onnx_route():
    pytest.importorskip("skl2onnx")
    from sklearn.neighbors import KNeighborsClassifier

    rows = np.random.default_rng(0).normal(size=(40, 3))
    knn = KNeighborsClassifier(3).fit(rows, (rows[:, 0] > 0).astype(int))
    for via in ("onnx", "auto"):
        with pytest.raises(StructureError) as caught:
            extract_structure(knn, feature_names=list("abc"), labels=["x", "y", "z"], via=via)
        assert caught.value.param == "labels", via
    named = extract_structure(knn, feature_names=list("abc"), labels=["no", "yes"]).spec
    assert named["outputs"] == {"labels": ["no", "yes"]}


def test_auto_does_not_hide_a_wrong_argument_behind_a_native_reading(torch):
    nn = torch.nn

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(4, 2, batch_first=True)

        def forward(self, x):
            out, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
            return out.squeeze(1).sum(-1, keepdim=True)

    with pytest.raises(StructureError) as caught:
        extract_structure(Attention().eval(), feature_names=list("abcd"), output=7)
    assert caught.value.param == "output"
