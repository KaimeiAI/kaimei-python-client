"""Live models through ONNX: PyTorch modules and scikit-learn estimators, with checks that hold their own outputs."""

from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnx")

from kaimei import StructureError, graphs  # noqa: E402
from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402

NAMES = ["f0", "f1", "f2", "f3"]


def softmax(values):
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def table_of(spec):
    count = len(spec["check"]["outputs"])
    return np.array([[spec["check"]["inputs"][name][row][0] for name in spec["feature_names"]] for row in range(count)])


# --- PyTorch ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_a_recurrent_module_is_converted_and_checked_against_itself(torch):
    nn = torch.nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(2, 5, batch_first=True)
            self.head = nn.Linear(5, 1)

        def forward(self, series):
            states, _ = self.rnn(series)
            return self.head(states[:, -1])

    torch.manual_seed(0)
    model = Forecaster()
    model.train()
    example = torch.randn(1, 6, 2)
    spec = extract_structure(model, task="regression", sample=example).spec
    assert model.training  # left as it was found
    assert spec["family"] == "graph" and spec["framework"] == "pytorch" and spec_is_executable(spec)
    assert spec["features"] == [{"name": "series", "type": "sequence", "shape": [6, 2]}]
    assert "LSTM" in {node["op"] for node in spec["graph"]["nodes"]}
    assert spec["metadata"]["route"] == "onnx" and spec["metadata"]["checked_against"] == "pytorch"
    assert spec["metadata"]["model_class"].endswith("Forecaster") and spec["metadata"]["framework_version"] == torch.__version__
    assert spec["summary"].startswith("PyTorch module converted to ONNX and read as a graph of")
    rows = np.asarray(spec["check"]["inputs"]["series"], dtype=np.float32)
    np.testing.assert_allclose(rows[0], example.numpy().reshape(-1), rtol=1e-6)  # the caller's example leads the check
    model.eval()
    with torch.no_grad():
        own = model(torch.tensor(rows.reshape(-1, 6, 2))).numpy()
    np.testing.assert_allclose(spec["check"]["outputs"], own, rtol=1e-6, atol=1e-7)
    assert spec["check"]["tolerance"] == 1e-3
    json.dumps(spec["check"], allow_nan=False)


def test_a_module_the_native_readers_read_stays_with_them_unless_onnx_is_asked_for(torch):
    nn = torch.nn
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
    native = extract_structure(model, feature_names=NAMES, task="multiclass_classification").spec
    assert native["family"] == "network" and "route" not in native["metadata"]
    forced = extract_structure(model, feature_names=NAMES, task="multiclass_classification", via="onnx").spec
    assert forced["family"] == "graph" and forced["feature_names"] == NAMES
    assert forced["head"]["link"] == "softmax" and forced["outputs"] == {"labels": ["class_0", "class_1", "class_2"]}
    rows = table_of(forced)
    with torch.no_grad():
        own = model(torch.tensor(rows, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(forced["check"]["outputs"], softmax(own), rtol=1e-5, atol=1e-6)


def test_a_module_with_two_inputs_and_one_that_takes_token_ids(torch):
    nn = torch.nn

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.left, self.right, self.head = nn.Linear(3, 4), nn.Linear(2, 4), nn.Linear(8, 1)

        def forward(self, account, usage):
            return self.head(torch.cat([torch.relu(self.left(account)), torch.tanh(self.right(usage))], dim=1))

    torch.manual_seed(1)
    two = Two().eval()
    names = ["tenure", "plan", "seats", "calls", "logins"]
    spec = extract_structure(two, feature_names=names, task="regression", sample=(torch.zeros(1, 3), torch.zeros(1, 2))).spec
    assert spec["family"] == "graph" and spec["feature_names"] == names
    assert [item["from"]["features"] for item in spec["graph"]["inputs"]] == [names[:3], names[3:]]
    rows = table_of(spec)
    with torch.no_grad():
        own = two(torch.tensor(rows[:, :3], dtype=torch.float32), torch.tensor(rows[:, 3:], dtype=torch.float32)).numpy()
    np.testing.assert_allclose(spec["check"]["outputs"], own, rtol=1e-5, atol=1e-6)

    class Reviews(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.head = nn.Embedding(9, 6), nn.Linear(6, 2)

        def forward(self, ids):
            return self.head(self.embed(ids).mean(dim=1))

    reviews = Reviews().eval()
    vocab = ["<pad>", "<unk>", "good", "bad", "great", "awful", "film", "a", "not"]
    text = extract_structure(
        reviews, task="binary_classification", sample=torch.tensor([[2, 6, 0, 0, 0]]), feature_names=["review"],
        tokenizer={"type": "word", "vocab": vocab, "unk": "<unk>", "pad": "<pad>"}, labels=["negative", "positive"],
    ).spec
    assert text["features"] == [{"name": "review", "type": "text", "max_length": 5, "vocab_size": 9}]
    assert text["tokenizers"]["review"]["vocab"] == vocab and text["outputs"] == {"labels": ["negative", "positive"]}
    assert text["graph"]["inputs"][0]["from"] == {"feature": "review", "part": "ids", "pad_id": 0}
    for row, answer in zip(text["check"]["inputs"]["review"], text["check"]["outputs"]):
        ids = [int(item) if item >= 0 else 0 for item in row]
        with torch.no_grad():
            own = reviews(torch.tensor([ids])).numpy()[0]
        np.testing.assert_allclose(answer, softmax(own), rtol=1e-5, atol=1e-6)
    assert max(max(row) for row in text["check"]["inputs"]["review"]) < 9
    # Without a tokenizer the same model takes token ids.
    ids_only = extract_structure(reviews, task="binary_classification", sample=torch.tensor([[2, 6, 0, 0, 0]])).spec
    assert ids_only["features"] == [{"name": "text", "type": "tokens", "max_length": 5, "vocab_size": 9}] and "tokenizers" not in ids_only


def test_pictures_series_and_what_an_example_is_needed_for(torch):
    nn = torch.nn
    torch.manual_seed(2)
    conv = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).eval()
    picture = extract_structure(conv, input_shape=(3, 8, 8), task="binary_classification", via="onnx").spec
    assert picture["features"] == [{"name": "image", "type": "image", "height": 8, "width": 8, "channels": 3}]
    assert picture["graph"]["inputs"][0]["from"] == {"feature": "image", "layout": "nchw"}
    row = np.asarray(picture["check"]["inputs"]["image"][1], dtype=np.float32).reshape(8, 8, 3)
    with torch.no_grad():
        own = conv(torch.tensor(row.transpose(2, 0, 1)[None])).numpy()[0]
    np.testing.assert_allclose(picture["check"]["outputs"][1], softmax(own), rtol=1e-4, atol=1e-5)
    named = extract_structure(conv, input_shape=(3, 8, 8), feature_names=["photo"], task="binary_classification", via="onnx").spec
    assert named["feature_names"] == ["photo"]
    # The pixel normalization a caller declares is part of the graph, and the paths are still the module's own.
    scaled = extract_structure(
        conv, input_shape=(3, 8, 8), task="binary_classification", via="onnx",
        preprocess={"mean": [0.5, 0.4, 0.3], "std": [0.2, 0.2, 0.2]},
    ).spec
    row = np.asarray(scaled["check"]["inputs"]["image"][1], dtype=np.float32).reshape(8, 8, 3)
    normal = (row - np.array([0.5, 0.4, 0.3], dtype=np.float32)) / np.float32(0.2)
    with torch.no_grad():
        own = conv(torch.tensor(normal.transpose(2, 0, 1)[None])).numpy()[0]
    np.testing.assert_allclose(scaled["check"]["outputs"][1], softmax(own), rtol=1e-4, atol=1e-5)
    assert not any(str(node.get("path", "")).startswith("inner.") for node in scaled["graph"]["nodes"])
    # A convolutional module with no shape and no example cannot be converted, and the error says what to pass.
    with pytest.raises(StructureError) as caught:
        extract_structure(conv, task="binary_classification", via="onnx")
    assert "sample=" in caught.value.message
    series = nn.Sequential(nn.Conv1d(2, 3, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(3, 1)).eval()
    read = extract_structure(series, input_shape=(2, 9), task="regression", feature_names=["signal"]).spec
    assert read["family"] == "graph" and read["features"] == [{"name": "signal", "type": "sequence", "shape": [2, 9]}]


def test_the_output_read_and_what_a_module_gives_for_a_tensor_task(torch):
    nn = torch.nn

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.value, self.code = nn.Linear(4, 6), nn.Linear(6, 1), nn.Linear(6, 3)

        def forward(self, x):
            hidden = torch.tanh(self.body(x))
            return self.value(hidden), self.code(hidden)

    torch.manual_seed(3)
    heads = Heads().eval()
    first = extract_structure(heads, feature_names=NAMES, task="regression", via="onnx").spec
    assert len(first["check"]["outputs"][0]) == 1
    second = extract_structure(heads, feature_names=NAMES, task="embedding", via="onnx", output=1).spec
    rows = table_of(second)
    with torch.no_grad():
        own = heads(torch.tensor(rows, dtype=torch.float32))[1].numpy()
    np.testing.assert_allclose(second["check"]["outputs"], own, rtol=1e-5, atol=1e-6)
    assert extract_structure(heads, feature_names=NAMES, task="embedding", via="onnx", output="output_1").spec["head"] == second["head"]
    with pytest.raises(StructureError) as caught:
        extract_structure(heads, feature_names=NAMES, via="onnx", output=5)
    assert caught.value.param == "output"
    painter = nn.Sequential(nn.Linear(4, 2 * 3 * 3), nn.Unflatten(1, (2, 3, 3)), nn.ConvTranspose2d(2, 1, 2, stride=2), nn.Sigmoid()).eval()
    drawn = extract_structure(painter, feature_names=NAMES, task="generation", via="onnx").spec
    head = drawn["head"]["output"]
    assert drawn["graph"]["values"][head]["shape"] == [1, 1, 6, 6] and "view" not in drawn["graph"]["values"][head]
    rows = table_of(drawn)
    with torch.no_grad():
        own = painter(torch.tensor(rows, dtype=torch.float32)).numpy()
    # A picture stays as the module gives it, channels first: the API turns it to the order it shows one in.
    np.testing.assert_allclose(drawn["check"]["outputs"], own.reshape(len(rows), -1), rtol=1e-4, atol=1e-5)


def test_a_language_model_is_converted_for_as_many_tokens_as_it_is_told_and_its_file_is_kept(torch):
    onnx = pytest.importorskip("onnx")
    nn = torch.nn

    class Writer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.position, self.head = nn.Embedding(11, 6), nn.Embedding(16, 6), nn.Linear(6, 11)
            self.config = type("Config", (), {"n_positions": 16})()

        def forward(self, ids):
            return self.head(torch.cumsum(self.embed(ids) + self.position(torch.arange(ids.shape[1])), dim=1))

    torch.manual_seed(7)
    writer = Writer().eval()
    sample = torch.tensor([[3, 1, 4, 1]])
    with pytest.warns(UserWarning, match=r"read for 4 tokens.*takes up to 16"):
        short = extract_structure(writer, task="text_generation", sample=sample)
    assert short.spec["features"][0]["max_length"] == 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        whole = extract_structure(writer, task="text_generation", sample=sample, max_length=16)
    spec = whole.spec
    assert spec["features"][0]["max_length"] == 16 and spec["graph"]["inputs"][0]["shape"] == [1, 16]
    assert "check" in spec and not spec.get("approximate") and spec_is_executable(spec)
    assert sample.shape == (1, 4)  # the caller's example is left as it was
    with pytest.raises(StructureError) as caught:
        extract_structure(writer, task="text_generation", sample=sample, max_length=0)
    assert caught.value.param == "max_length"
    # The file the structure was read from stays here, for the model's agent: every step names its tensor in it.
    assert isinstance(whole.onnx, bytes)
    file = onnx.load_model_from_string(whole.onnx)
    inside = {name for node in file.graph.node for name in node.output} | {item.name for item in file.graph.output}
    sources = [value["source"] for value in spec["graph"]["values"].values() if "source" in value]
    assert sources and set(sources) <= inside
    again = onnx.load_model_from_string(graphs.to_onnx(writer, task="text_generation", sample=sample, max_length=16))
    assert [list(node.output) for node in again.graph.node] == [list(node.output) for node in file.graph.node]
    kept_home = extract_structure(writer, task="text_generation", sample=sample, max_length=16, weights="summary")
    assert kept_home.onnx == whole.onnx


def test_what_cannot_be_exported_is_said_and_the_native_reading_stands(torch, monkeypatch):
    nn = torch.nn
    torch.manual_seed(4)

    class Attends(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(4, 8)
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
            self.head = nn.Linear(8, 3)

        def forward(self, x):
            tokens = self.embed(x).unsqueeze(1)
            mixed, _ = self.attn(tokens, tokens, tokens)
            return self.head(mixed.squeeze(1))

    model = Attends().eval()
    converted = extract_structure(model, feature_names=NAMES, task="multiclass_classification").spec
    assert converted["family"] == "graph" and not converted.get("approximate")  # the readers could only draw it

    def refuses(*args, **kwargs):
        raise RuntimeError("this exporter cannot write that operator")

    monkeypatch.setattr(torch.onnx, "export", refuses)
    drawn = extract_structure(model, feature_names=NAMES, task="multiclass_classification").spec
    assert drawn["family"] == "network" and drawn["approximate"] is True
    assert "torch.onnx.export could not write it" in drawn["metadata"]["onnx_route"]
    with pytest.raises(StructureError) as caught:
        extract_structure(model, feature_names=NAMES, via="onnx")
    assert "torch.onnx.export could not write it" in caught.value.message and "cannot write that operator" in caught.value.message


# --- scikit-learn ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def problem():
    pytest.importorskip("skl2onnx")
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=0)
    return X, y


def test_an_estimator_with_no_reader_of_its_own_is_converted_and_checked(problem):
    from sklearn.naive_bayes import GaussianNB

    X, y = problem
    model = GaussianNB().fit(X, y)
    spec = extract_structure(model, feature_names=NAMES, check_rows=X).spec
    assert spec["family"] == "graph" and spec["framework"] == "sklearn" and spec_is_executable(spec)
    assert spec["task"] == "multiclass_classification" and spec["outputs"] == {"labels": ["0", "1", "2"]}
    assert spec["head"]["link"] == "none" and spec["metadata"]["library_method"] == "predict_proba"
    assert spec["metadata"]["route"] == "onnx" and spec["metadata"]["model_class"].endswith("GaussianNB")
    assert spec["summary"].startswith("scikit-learn GaussianNB converted to ONNX")
    rows = table_of(spec)
    # The caller's own rows, as the doubles they are, then rows made up from the values those rows hold: nothing
    # made up around zero, whatever range the caller's data lies in.
    assert rows.shape[0] == graphs.OWN_CHECK_ROWS + graphs.TABLE_CHECK_ROWS
    np.testing.assert_array_equal(rows[: graphs.OWN_CHECK_ROWS], X[: graphs.OWN_CHECK_ROWS])
    for column in range(rows.shape[1]):
        assert set(rows[graphs.OWN_CHECK_ROWS :, column]) <= set(X[: graphs.OWN_CHECK_ROWS, column])
    np.testing.assert_allclose(spec["check"]["outputs"], model.predict_proba(rows), rtol=1e-12)
    json.dumps(extract_structure(model, feature_names=NAMES).spec, allow_nan=False)


def test_an_estimator_the_native_readers_read_stays_with_them(problem):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = problem
    for model in (LogisticRegression(max_iter=500).fit(X, y), RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)):
        spec = extract_structure(model, feature_names=NAMES, check_rows=X).spec
        assert spec["family"] in ("network", "ensemble") and "route" not in spec["metadata"]


def test_a_classifier_with_no_probabilities_is_read_through_its_scores(problem):
    from sklearn.svm import SVC

    X, y = problem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        several = SVC(kernel="poly").fit(X, y)
        two = SVC(kernel="poly").fit(X, (y > 0).astype(int))
    spec = extract_structure(several, feature_names=NAMES).spec
    assert spec["head"]["link"] == "softmax" and spec["metadata"]["library_method"] == "decision_function"
    assert "scores, not probabilities" in spec["summary"]
    rows = table_of(spec)
    np.testing.assert_allclose(spec["check"]["outputs"], softmax(several.decision_function(rows)), rtol=1e-9)
    # Two classes: one margin, positive for the second class as it is for the model.
    binary = extract_structure(two, feature_names=NAMES).spec
    assert not binary.get("approximate") and binary["head"]["link"] == "sigmoid"
    assert binary["task"] == "binary_classification" and binary["outputs"] == {"labels": ["0", "1"]}
    rows = table_of(binary)
    margin = two.decision_function(rows)
    np.testing.assert_allclose(np.ravel(binary["check"]["outputs"]), 1.0 / (1.0 + np.exp(-margin)), rtol=1e-9)
    assert ((np.ravel(binary["check"]["outputs"]) > 0.5).astype(int) == two.predict(rows)).all()
    assert [node["op"] for node in binary["graph"]["nodes"]][-1] in ("Gather", "Neg")


def test_regressors_pipelines_and_searches(problem):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVR

    X, y = problem
    target = X @ np.array([1.0, -2.0, 0.5, 3.0])
    pipeline = make_pipeline(PCA(3), SVR()).fit(X, target)
    spec = extract_structure(pipeline, feature_names=NAMES).spec
    assert spec["family"] == "graph" and spec["task"] == "regression" and spec["metadata"]["library_method"] == "predict"
    rows = table_of(spec)
    np.testing.assert_allclose(np.ravel(spec["check"]["outputs"]), pipeline.predict(rows), rtol=1e-12)
    search = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": [2, 3]}, cv=2).fit(X, target)
    found = extract_structure(search, feature_names=NAMES).spec
    assert found["family"] == "graph"
    np.testing.assert_allclose(np.ravel(found["check"]["outputs"]), search.predict(table_of(found)), rtol=1e-12)
    # A Scan holds the training rows in its body, and they are sent with it, each in the type the graph has it in.
    done = extract_structure(search, feature_names=NAMES, externalize_above=100)
    kinds = {str(array.dtype) for array in done.tensors.values()}
    assert done.tensors and "float32" in kinds and kinds <= {"float32", "float64", "int64"}
    assert any(array.shape == (200, 4) for array in done.tensors.values())


def test_single_precision_is_given_up_for_doubles_when_only_they_reproduce_the_model(problem):
    from sklearn.neighbors import KNeighborsRegressor

    rng = np.random.default_rng(0)
    # Rows a hair apart around a million: single precision cannot tell the neighbours from each other.
    X = 1e6 + rng.normal(size=(120, 4)) * 0.01
    target = rng.normal(size=120) * 100
    model = KNeighborsRegressor(1).fit(X, target)
    spec = extract_structure(model, feature_names=NAMES, check_rows=X).spec
    assert spec["family"] == "graph" and not spec.get("approximate")
    assert spec["graph"]["inputs"][0]["dtype"] == "float64"
    rows = table_of(spec)
    np.testing.assert_allclose(np.ravel(spec["check"]["outputs"]), model.predict(rows), rtol=1e-12)


def test_a_model_that_keeps_its_weights_at_home_is_drawn_from_shapes(problem):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    X, y = problem
    summary = extract_structure(GaussianNB().fit(X, y), feature_names=NAMES, weights="summary").spec
    assert summary["weights"] == "summary" and "check" not in summary and not spec_is_executable(summary)
    held = extract_structure(KNeighborsClassifier(3).fit(X, y), feature_names=NAMES, weights="summary")
    assert held.tensors == {}
    training = [value for value in _every_constant(held.spec["graph"]) if "$shape" in value and value["$shape"][-1] == 4]
    assert training and all("data" not in value and "$b64" not in value for value in training)


def _every_constant(graph):
    for value in (graph.get("constants") or {}).values():
        if isinstance(value, dict):
            yield value
    for node in graph.get("nodes") or []:
        for attr in (node.get("attrs") or {}).values():
            if isinstance(attr, dict) and isinstance(attr.get("graph"), dict):
                yield from _every_constant(attr["graph"])


def test_the_file_an_estimator_was_converted_to_is_kept_for_its_agent(problem):
    onnx = pytest.importorskip("onnx")
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    X, y = problem[0], problem[1]
    model = KNeighborsClassifier(3).fit(X, y)
    read = extract_structure(model, feature_names=NAMES)
    assert read.spec["family"] == "graph" and isinstance(read.onnx, bytes)
    file = onnx.load_model_from_string(read.onnx)
    inside = {name for node in file.graph.node for name in node.output}
    sources = [value["source"] for value in read.spec["graph"]["values"].values() if "source" in value]
    assert sources and set(sources) <= inside
    # Converting again gives the model again, but scikit-learn's converter numbers some tensors anew each time:
    # the file to keep for a structure is the one it was read from.
    again = onnx.load_model_from_string(graphs.to_onnx(model, feature_names=NAMES))
    assert [node.op_type for node in again.graph.node] == [node.op_type for node in file.graph.node]
    # A model Kaimei's own readers read was read from no file.
    assert extract_structure(LogisticRegression().fit(X, y), feature_names=NAMES).onnx is None


def test_a_missing_converter_is_named(problem, monkeypatch):
    from sklearn.naive_bayes import GaussianNB

    X, y = problem
    model = GaussianNB().fit(X, y)
    real = graphs._has
    monkeypatch.setattr(graphs, "_has", lambda package: package != "skl2onnx" and real(package))
    with pytest.raises(StructureError) as caught:
        extract_structure(model, feature_names=NAMES)
    assert caught.value.code == "unsupported_estimator" and "pip install skl2onnx" in caught.value.message
    with pytest.raises(StructureError) as caught:
        extract_structure(model, feature_names=NAMES, via="onnx")
    assert "pip install skl2onnx" in caught.value.message
