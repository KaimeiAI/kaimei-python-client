"""How a model is read: by which route, what each route promises, and what reading it leaves behind.

Regression tests for the reading routes (``kaimei.structure``), the converters for TensorFlow, JAX and GGUF files
(``kaimei.converters``) and the reader of composite models (``kaimei.ensembles``). Unlike the older modules listed in
``conftest.py``, this one runs with whatever converters are installed, so the default route is tested as it runs.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest

np = pytest.importorskip("numpy")

from kaimei import StructureError, converters, ensembles, graphs, structure  # noqa: E402
from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402

NAMES = ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def two_classes(rows=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 4))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


def mlp(torch):
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()


def attention(torch):
    """A module Kaimei's own readers can only draw."""
    nn = torch.nn

    class Attends(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(4, 2, batch_first=True)
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            tokens = x.unsqueeze(1)
            mixed, _ = self.attn(tokens, tokens, tokens)
            return self.fc(mixed[:, 0])

    torch.manual_seed(0)
    return Attends().eval()


def two_heads(torch):
    nn = torch.nn

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.value, self.code = nn.Linear(4, 6), nn.Linear(6, 1), nn.Linear(6, 3)

        def forward(self, x):
            hidden = torch.tanh(self.body(x))
            return self.value(hidden), self.code(hidden)

    torch.manual_seed(3)
    return Heads().eval()


def branches(torch):
    """Takes one of two paths by the values it is given, so no reading of one path is the model."""
    nn = torch.nn

    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.up, self.down = nn.Linear(4, 2), nn.Linear(4, 2)

        def forward(self, x):
            if x.sum() > 1.0:
                return self.up(x)
            return self.down(x)

    torch.manual_seed(0)
    return Branches().eval()


def unavailable(monkeypatch, reason="the onnx package is not installed (pip install onnx, or pip install 'kaimei[onnx]')"):
    """The ONNX route stood in for by one that cannot convert anything, saying why."""

    def via_onnx(estimator, *, reasons=None, **options):
        if reasons is not None:
            reasons.append(reason)
        return None

    monkeypatch.setattr(graphs, "via_onnx", via_onnx)


# --- an SVC whose label and probabilities can disagree (R-CL-6) ---------------------------------------------------


def test_an_svc_with_probabilities_runs_by_default_and_says_its_class_can_differ_from_predict():
    pytest.importorskip("skl2onnx")
    from sklearn.svm import SVC

    X, y = two_classes()
    model = SVC(kernel="linear", probability=True, random_state=0).fit(X, y)
    for via in ("auto", "onnx"):
        with pytest.warns(UserWarning, match=r"most probable class of those probabilities.*can differ from predict\(\)"):
            spec = extract_structure(model, feature_names=NAMES, via=via).spec
        assert spec["family"] == "graph" and "check" in spec and spec_is_executable(spec)
        assert spec["metadata"]["library_method"] == "predict_proba"
        assert spec["metadata"]["predicted_class"] == structure.PREDICTED_CLASS_NOTE
    # Kaimei's own readers draw it, for that very reason, and need no note.
    drawn = extract_structure(model, feature_names=NAMES, via="native").spec
    assert drawn["approximate"] is True and "predicted_class" not in drawn["metadata"]


def test_a_pipeline_ending_in_a_nusvc_with_probabilities_carries_the_note_and_a_plain_svc_does_not():
    pytest.importorskip("skl2onnx")
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, NuSVC

    X, y = two_classes()
    pipeline = make_pipeline(StandardScaler(), NuSVC(probability=True, random_state=0)).fit(X, y)
    with pytest.warns(UserWarning, match="most probable class"):
        spec = extract_structure(pipeline, feature_names=NAMES).spec
    assert spec["family"] == "graph" and not spec.get("approximate")
    assert spec["metadata"]["predicted_class"] == structure.PREDICTED_CLASS_NOTE
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = extract_structure(SVC(kernel="linear").fit(X, y), feature_names=NAMES).spec
    assert "predicted_class" not in plain["metadata"]
    assert not any("most probable class" in str(item.message) for item in caught)


def test_which_models_predict_by_an_svcs_margin():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import BaggingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, NuSVC

    X, y = two_classes(120)
    three = np.digitize(X[:, 0], [-0.5, 0.5])
    margin = ensembles.predicts_by_its_margin
    scaled = make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))
    assert margin(SVC(probability=True, random_state=0).fit(X, y))
    assert margin(NuSVC(probability=True, random_state=0).fit(X, y))
    assert margin(make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)).fit(X, y))
    assert margin(GridSearchCV(scaled, {"svc__C": [1.0]}, cv=2).fit(X, y))
    assert margin(OneVsRestClassifier(SVC(probability=True, random_state=0)).fit(X, three))
    # These go by probabilities, or have none for a class to disagree with.
    assert not margin(SVC().fit(X, y))
    assert not margin(LogisticRegression().fit(X, y))
    soft = VotingClassifier([("svc", SVC(probability=True, random_state=0)), ("logit", LogisticRegression())], voting="soft")
    assert not margin(soft.fit(X, y))
    assert not margin(BaggingClassifier(SVC(probability=True, random_state=0), n_estimators=2, random_state=0).fit(X, y))


# --- a Keras model only a conversion runs, without tf2onnx (R-CL-10) ------------------------------------------------


class _SeriesInput:
    shape = (None, 10, 3)
    dtype = "float32"
    name = "series"


class _Layer:
    def get_weights(self):
        return []


class LSTM(_Layer):
    units = 4
    name = "lstm"


class Dense(_Layer):
    units = 1
    name = "dense"
    activation = None

    def get_weights(self):
        return [np.full((4, 1), 0.5), np.zeros(1)]


class SeriesModel:
    """Stands in for ``keras.Sequential([Input((10, 3)), LSTM(4), Dense(1)])``: a Keras model of a series."""

    def __init__(self):
        self.layers = [LSTM(), Dense()]
        self.inputs = [_SeriesInput()]


SeriesModel.__module__ = "keras.src.models.sequential"


def without(monkeypatch, *packages):
    real = graphs._has
    monkeypatch.setattr(graphs, "_has", lambda package: package not in packages and real(package))


def test_a_keras_model_only_a_conversion_runs_is_drawn_as_before_when_tf2onnx_is_missing(monkeypatch):
    # Exercise the missing converter independently of whether this test environment installed ONNX.
    monkeypatch.setattr(graphs, "_has", lambda package: package == "onnx")
    native = extract_structure(SeriesModel(), feature_names=["series"], via="native").spec
    with pytest.warns(UserWarning, match=r"converted by tf2onnx.*needs tf2onnx \(pip install tf2onnx\).*drawn, not run"):
        spec = extract_structure(SeriesModel(), feature_names=["series"]).spec
    assert spec["family"] == "network" and spec["approximate"] is True
    assert [layer["type"] for layer in spec["layers"]] == [layer["type"] for layer in native["layers"]] == ["input", "opaque", "dense"]
    assert "pip install tf2onnx" in spec["metadata"]["onnx_route"]
    with pytest.raises(StructureError) as caught:
        extract_structure(SeriesModel(), feature_names=["series"], on_unsupported="error")
    assert caught.value.code == "unsupported_layers"


def test_a_keras_conversion_that_fails_leaves_the_drawing_and_a_mistake_of_the_callers_does_not(monkeypatch):
    def fails(estimator, *, reasons=None, **options):
        reasons.append("tf2onnx could not convert it (ValueError: a test said so)")
        return None

    monkeypatch.setattr(graphs, "via_onnx", fails)
    with pytest.warns(UserWarning, match="a test said so"):
        spec = extract_structure(SeriesModel(), feature_names=["series"]).spec
    assert spec["approximate"] is True and "a test said so" in spec["metadata"]["onnx_route"]

    def refuses(estimator, *, reasons=None, **options):
        raise StructureError("sample is not what this model takes (a test said so).", param="sample")

    monkeypatch.setattr(graphs, "via_onnx", refuses)
    with pytest.raises(StructureError) as caught:
        extract_structure(SeriesModel(), feature_names=["series"])
    assert caught.value.param == "sample"
    # What only a conversion reads is never left out by falling back to the drawing.
    monkeypatch.setattr(graphs, "via_onnx", fails)
    with pytest.raises(StructureError) as caught:
        extract_structure(SeriesModel(), feature_names=["series"], output=1)
    assert caught.value.code == "unsupported_estimator" and "output is read only from a model converted to ONNX" in caught.value.message


def test_a_real_keras_lstm_is_drawn_without_tf2onnx_and_runs_with_it(monkeypatch):
    keras = pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((10, 3)), keras.layers.LSTM(4), keras.layers.Dense(1)])
    native = extract_structure(model, feature_names=["series"], via="native").spec
    without(monkeypatch, "tf2onnx")
    with pytest.warns(UserWarning, match="tf2onnx"):
        drawn = extract_structure(model, feature_names=["series"]).spec
    assert drawn["approximate"] is True and [layer["id"] for layer in drawn["layers"]] == [layer["id"] for layer in native["layers"]]
    with pytest.raises(StructureError) as caught:
        extract_structure(model, feature_names=["series"], sample=np.zeros((1, 7, 2), dtype=np.float32), via="native")
    assert caught.value.param == "sample"
    monkeypatch.undo()
    if graphs._has("tf2onnx"):
        converted = extract_structure(model, feature_names=["series"]).spec
        assert converted["family"] == "graph" and spec_is_executable(converted)


# --- training flags on the native route (audit new 7) ------------------------------------------------------------


def frozen(torch):
    nn = torch.nn

    class Frozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.body(x))

    torch.manual_seed(0)
    model = Frozen()
    model.train()
    model.body.eval()  # a frozen backbone, its batch normalization not learning
    return model


def modes(model):
    return {name or "(top)": part.training for name, part in model.named_modules()}


@pytest.mark.parametrize("via", ["auto", "native"])
def test_reading_a_module_leaves_each_part_in_its_own_training_mode(torch, via):
    model = frozen(torch)
    before = modes(model)
    statistics = model.body[1].running_mean.clone(), int(model.body[1].num_batches_tracked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extract_structure(model, feature_names=NAMES, via=via)
        extract_structure(model, feature_names=NAMES, via=via, weights="summary")
        extract_structure(model, feature_names=NAMES, via=via, task="binary_classification", output=0)
    assert before["(top)"] and before["head"] and not before["body"]
    assert modes(model) == before
    assert torch.equal(model.body[1].running_mean, statistics[0]) and int(model.body[1].num_batches_tracked) == statistics[1]


def test_a_picture_model_and_one_fx_cannot_trace_keep_their_modes_too(torch, monkeypatch):
    unavailable(monkeypatch)
    nn = torch.nn

    class Picture(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.norm, self.head = nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Linear(4, 2)

        def forward(self, x):
            return self.head(torch.relu(self.norm(self.conv(x))).mean(dim=(2, 3)))

    class Gated(nn.Module):
        """fx cannot trace a branch on the input's values: the hooks read it."""

        def __init__(self):
            super().__init__()
            self.body, self.head = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)), nn.Linear(4, 2)

        def forward(self, x):
            if x.sum() > 1e9:
                raise RuntimeError("too large")
            return self.head(self.body(x))

    torch.manual_seed(0)
    for model, options in ((Picture(), {"input_shape": (3, 8, 8)}), (Gated(), {"feature_names": NAMES})):
        model.train()
        next(part for name, part in model.named_modules() if name in ("norm", "body")).eval()
        before = modes(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extract_structure(model, via="native", **options)
        assert modes(model) == before, type(model).__name__


def test_comparing_a_reading_with_its_module_puts_each_flag_back(torch):
    model = frozen(torch)
    before = modes(model)
    answer = ensembles._network_output(model, np.zeros((3, 4)))
    assert answer.shape == (3, 2) and modes(model) == before


# --- on_unsupported="error" on every route (audit C12) ------------------------------------------------------------


@pytest.mark.parametrize("via", ["auto", "native", "onnx"])
def test_on_unsupported_error_refuses_any_reading_that_would_only_be_drawn(torch, via):
    if via != "native":
        pytest.importorskip("onnx")
    net = branches(torch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drawn = extract_structure(net, feature_names=NAMES, via=via).spec
        assert drawn["approximate"] is True
        with pytest.raises(StructureError) as caught:
            extract_structure(net, feature_names=NAMES, via=via, on_unsupported="error")
    assert caught.value.code == "unsupported_layers" and "on_unsupported='error' refuses it" in caught.value.message
    assert "drawn, not run" in caught.value.message


def test_on_unsupported_error_refuses_a_summary_of_importances_and_not_weights_kept_home(monkeypatch):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    unavailable(monkeypatch)

    class OnlyImportances:
        n_features_in_ = 4
        feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        n_estimators = 3

        def predict(self, rows):
            return np.zeros(len(rows))

    assert extract_structure(OnlyImportances(), feature_names=NAMES).spec["family"] == "ensemble_summary"
    with pytest.raises(StructureError) as caught:
        extract_structure(OnlyImportances(), feature_names=NAMES, on_unsupported="error")
    assert caught.value.code == "unsupported_layers" and "Kaimei would draw this model, not run it" in caught.value.message
    X, y = two_classes(80)
    kept = extract_structure(LogisticRegression().fit(X, y), feature_names=NAMES, weights="summary", on_unsupported="error")
    assert kept.spec["weights"] == "summary"


# --- the caller's own arguments, on every route (audit C14) ---------------------------------------------------------


@pytest.mark.parametrize("via", ["auto", "native", "onnx"])
def test_a_sample_the_module_cannot_be_called_with_is_an_error_on_every_route(torch, via):
    for model in (attention(torch), mlp(torch)):
        with pytest.raises(StructureError) as caught:
            extract_structure(model, sample=torch.zeros(1, 7), via=via)
        assert caught.value.param == "sample" and "could not be called with sample" in caught.value.message
        with pytest.raises(StructureError) as caught:
            extract_structure(model, feature_names=NAMES, task="binary_classification", sample=np.zeros((1, 7)), via=via)
        assert caught.value.param == "sample"


@pytest.mark.parametrize("via", ["auto", "native", "onnx"])
def test_inputs_for_an_input_the_module_lacks_or_a_picture_that_is_not_one_are_errors(torch, via):
    model = attention(torch)
    with pytest.raises(StructureError) as caught:
        extract_structure(model, feature_names=NAMES, inputs={"nope": {"feature": "f"}}, via=via)
    assert caught.value.param == "inputs" and "'nope', which is not an input of this model's forward (x)" in caught.value.message
    with pytest.raises(StructureError) as caught:
        extract_structure(model, feature_names=NAMES, inputs={"x": {"feature": "f", "layout": "nchw"}}, via=via)
    assert caught.value.param == "inputs" and "is a picture" in caught.value.message
    with pytest.raises(StructureError) as caught:
        extract_structure(model, feature_names=NAMES, inputs={"x": "f"}, via=via)
    assert caught.value.param == "inputs"


@pytest.mark.parametrize("via", ["auto", "native", "onnx"])
def test_an_output_the_module_does_not_give_is_an_error_on_every_route(torch, via):
    with pytest.raises(StructureError) as caught:
        extract_structure(mlp(torch), feature_names=NAMES, task="binary_classification", output=7, via=via)
    assert caught.value.param == "output" and "1 output tensors" in caught.value.message
    for wrong in (-1, 1.5, [1], True):
        with pytest.raises(StructureError) as caught:
            extract_structure(mlp(torch), feature_names=NAMES, task="binary_classification", output=wrong, via=via)
        assert caught.value.param == "output"


@pytest.mark.parametrize("via", ["auto", "native", "onnx"])
def test_a_max_length_that_is_not_a_whole_number_is_an_error_for_any_model(torch, via):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X, y = two_classes(80)
    for model in (mlp(torch), LogisticRegression().fit(X, y)):
        for wrong in (0, -3, 2.5, "64", True):
            with pytest.raises(StructureError) as caught:
                extract_structure(model, feature_names=NAMES, task="binary_classification", max_length=wrong, via=via)
            assert caught.value.param == "max_length"


@pytest.mark.parametrize("via", ["auto", "native", "onnx"])
def test_a_tokenizer_for_a_model_that_takes_no_token_ids_is_an_error_on_every_route(torch, via):
    pytest.importorskip("onnx")
    tokenizer = {"type": "char", "vocab": ["a", "b"]}
    for options in ({"sample": torch.zeros(1, 4)}, {"feature_names": NAMES}):
        with pytest.raises(StructureError) as caught:
            extract_structure(mlp(torch), task="binary_classification", tokenizer=tokenizer, via=via, **options)
        assert caught.value.param == "tokenizer", options


def test_labels_are_a_list_of_names(torch):
    for wrong in ("yes", {"0": "no", "1": "yes"}, 3):
        with pytest.raises(StructureError) as caught:
            extract_structure(mlp(torch), feature_names=NAMES, task="binary_classification", labels=wrong)
        assert caught.value.param == "labels"


def test_a_scikit_learn_models_arguments_are_checked_although_its_own_readers_read_it():
    pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LogisticRegression

    X, y = two_classes(80)
    logit = LogisticRegression().fit(X, y)
    assert extract_structure(logit, feature_names=NAMES).spec["family"] == "network"
    with pytest.raises(StructureError) as caught:
        extract_structure(logit, feature_names=NAMES, output=7)
    assert caught.value.param == "output"
    with pytest.raises(StructureError) as caught:
        extract_structure(logit, feature_names=NAMES, tokenizer={"type": "char", "vocab": ["a", "b"]})
    assert caught.value.param == "tokenizer"
    for via in ("auto", "native", "onnx"):
        with pytest.raises(StructureError) as caught:
            extract_structure(logit, feature_names=NAMES, sample=np.zeros((1, 7)), via=via)
        assert caught.value.param == "sample" and "7 columns" in caught.value.message
    # An output it does give is read where the caller said, through ONNX.
    read = extract_structure(logit, feature_names=NAMES, output="probabilities").spec
    assert read["family"] == "graph" and spec_is_executable(read)


def test_what_only_a_conversion_reads_is_honoured_or_refused_and_never_dropped(torch, monkeypatch):
    pytest.importorskip("onnx")
    model = mlp(torch)
    assert extract_structure(model, feature_names=NAMES, task="binary_classification").spec["family"] == "network"
    told = extract_structure(
        model, feature_names=NAMES, task="binary_classification", inputs={"input": {"features": ["w", "x", "y", "z"]}}
    ).spec
    assert told["family"] == "graph" and [feature["name"] for feature in told["features"]] == ["w", "x", "y", "z"]
    # A module's first output is what its own readers read: naming it changes nothing.
    assert extract_structure(model, feature_names=NAMES, task="binary_classification", output=0).spec["family"] == "network"
    for name, value in (("inputs", {"input": {"features": NAMES}}), ("tokenizer", {"type": "char", "vocab": ["a"]})):
        with pytest.raises(StructureError) as caught:
            extract_structure(model, feature_names=NAMES, task="binary_classification", via="native", **{name: value})
        assert caught.value.param == name and "via='native'" in caught.value.message
    unavailable(monkeypatch)
    cases = (
        (model, "inputs", {"input": {"features": NAMES}}),
        (model, "tokenizer", {"type": "char", "vocab": ["a"]}),
        (two_heads(torch), "output", 1),
    )
    for module, name, value in cases:
        with pytest.raises(StructureError) as caught:
            extract_structure(module, feature_names=NAMES, task="binary_classification", **{name: value})
        assert caught.value.code == "unsupported_estimator", name
        assert f"{name} is read only from a model converted to ONNX" in caught.value.message
        assert "pip install onnx" in caught.value.message


# --- an ONNX model with via="native" (R-CL-26) -----------------------------------------------------------------------


def onnx_table_model():
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.array([[0.5], [-0.25], [1.0], [0.1]], dtype=np.float32), "w")
    bias = numpy_helper.from_array(np.array([0.2], dtype=np.float32), "b")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["rows", "w", "b"], ["y"], name="/head/Gemm")],
        "table",
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 1])],
        [weight, bias],
    )
    made = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    made.ir_version = 9
    return made


def test_an_onnx_model_is_read_as_a_graph_whatever_via_says(tmp_path):
    model = onnx_table_model()
    path = tmp_path / "table.onnx"
    path.write_bytes(model.SerializeToString())
    auto = extract_structure(str(path), feature_names=NAMES, task="regression").spec
    assert auto["family"] == "graph" and spec_is_executable(auto)
    for via in ("native", "onnx"):
        for source in (str(path), model.SerializeToString(), model):
            spec = extract_structure(source, feature_names=NAMES, task="regression", via=via).spec
            assert spec["family"] == "graph" and spec["graph"]["nodes"] == auto["graph"]["nodes"], (via, type(source))


# --- JAX: the sample's rows, and which classes are JAX's (R-CL-13, R-CL-19) ------------------------------------------


WEIGHTS = np.array([[0.5], [-0.25], [1.0], [0.1]], dtype=np.float32)


def stand_in_jax(monkeypatch, exported):
    """jax and jax2onnx stood in for: ``to_onnx`` writes the inputs at the shapes of its ShapeDtypeStructs, as
    jax2onnx does, and records them."""
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    jax = types.ModuleType("jax")

    class ShapeDtypeStruct:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = tuple(shape), dtype

    jax.ShapeDtypeStruct = ShapeDtypeStruct  # type: ignore[attr-defined]
    jax.tree_util = types.SimpleNamespace(tree_leaves=lambda value: [value])  # type: ignore[attr-defined]
    jax2onnx = types.ModuleType("jax2onnx")

    def to_onnx(function, shapes, opset, input_names, model_name):
        exported.append([tuple(item.shape) for item in shapes])
        (shape,) = [list(item.shape) for item in shapes]
        graph = helper.make_graph(
            [helper.make_node("MatMul", [input_names[0], "w"], ["y"], name="/score/MatMul")],
            model_name,
            [helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [shape[0], 1])],
            [numpy_helper.from_array(WEIGHTS, "w")],
        )
        made = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        made.ir_version = 9
        return made

    jax2onnx.to_onnx = to_onnx  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax2onnx", jax2onnx)
    real = converters._has
    monkeypatch.setattr(converters, "_has", lambda package: package in ("jax", "jax2onnx") or real(package))


def test_a_jax_model_given_several_rows_is_converted_for_the_one_it_is_read_and_checked_with(monkeypatch):
    exported: list = []
    stand_in_jax(monkeypatch, exported)

    def score(x):
        return np.asarray(x, dtype=np.float32) @ WEIGHTS

    rows = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    spec = extract_structure(score, framework="jax", sample=rows, task="regression").spec
    assert exported == [[(1, 4)]]
    assert spec["framework"] == "jax" and "check" in spec and spec_is_executable(spec)
    assert spec["graph"]["inputs"][0]["shape"] == [1, 4]


def test_only_jaxs_own_packages_make_a_model_a_jax_model(torch):
    def made_in(module):
        kind = type("Model", (), {"__call__": lambda self, *args: None})
        kind.__module__ = module
        return kind()

    for module in ("jax", "jax._src.pjit", "jaxlib.xla_extension", "flax", "flax.nnx.module"):
        assert converters.is_jax(made_in(module)), module
    for module in ("jaxon_models", "jaxon_models.nets", "flaxen.layers", "jaxlibrary", "my_jax", "__main__"):
        assert not converters.is_jax(made_in(module)), module
    assert converters.is_jax(made_in("__main__"), "jax")  # said to be one

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 1)

        def forward(self, x):
            return self.fc(x)

    Net.__module__ = "jaxon_models.nets"
    spec = extract_structure(Net().eval(), feature_names=NAMES, task="regression").spec
    assert spec["family"] == "network" and spec["framework"] == "pytorch"
