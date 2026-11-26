"""What a model is, worked out from the model: its task, its inputs, the output to read and how to run it.

Each test uploads a model the way someone would who has not read Kaimei's documentation: the model and an
example input, and at most the names of its classes. The structure has to say what the model is, and its
check has to hold the model's own answers on rows Kaimei can actually give it.
"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnx")

from kaimei import StructureError, graphs  # noqa: E402
from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def softmax(values):
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def features_of(spec):
    return {feature["name"]: feature for feature in spec["features"]}


def inputs_of(spec):
    return {item["name"].split(".", 1)[-1]: item["from"] for item in spec["graph"]["inputs"]}


def runs(spec):
    return "check" in spec and not spec.get("approximate") and spec_is_executable(spec)


# --- the task, from the output and the labels ---------------------------------------------------


def test_logits_with_labels_are_a_classifier(torch):
    nn = torch.nn
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).eval()
    spec = extract_structure(net, sample=torch.rand(1, 3, 16, 16), labels=["cat", "dog", "bird"]).spec
    assert spec["task"] == "multiclass_classification" and spec["head"]["link"] == "softmax"
    assert spec["outputs"] == {"labels": ["cat", "dog", "bird"]} and runs(spec)
    two = extract_structure(nn.Sequential(nn.Linear(4, 2)).eval(), sample=torch.rand(1, 4), labels=["no", "yes"]).spec
    assert two["task"] == "binary_classification" and two["head"]["link"] == "softmax" and runs(two)
    one = extract_structure(nn.Sequential(nn.Linear(4, 1)).eval(), sample=torch.rand(1, 4), labels=["no", "yes"]).spec
    assert one["task"] == "binary_classification" and one["head"]["link"] == "sigmoid" and runs(one)


def test_labels_that_do_not_fit_the_output_are_refused(torch):
    nn = torch.nn
    with pytest.raises(StructureError) as caught:
        extract_structure(nn.Sequential(nn.Linear(4, 3)).eval(), sample=torch.rand(1, 4), labels=["a", "b"])
    assert "labels names 2 classes" in caught.value.message and caught.value.param == "labels"


def test_one_value_out_of_a_sigmoid_is_a_yes_or_no_answer(torch):
    nn = torch.nn
    spec = extract_structure(nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()).eval(), sample=torch.rand(1, 4)).spec
    assert spec["task"] == "binary_classification" and spec["head"]["link"] == "none" and runs(spec)


def test_a_sigmoid_for_each_label_is_a_multilabel_classifier(torch):
    nn = torch.nn
    torch.manual_seed(1)
    tags = ["indoor", "people", "night", "text", "animal"]
    squashed = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 5), nn.Sigmoid()).eval()
    spec = extract_structure(squashed, sample=torch.rand(1, 6), labels=tags).spec
    assert spec["task"] == "multilabel_classification" and spec["head"]["link"] == "none"
    assert spec["outputs"] == {"labels": tags} and "check" in spec
    rows = np.array([[spec["check"]["inputs"][f"x{i}"][row][0] for i in range(6)] for row in range(len(spec["check"]["outputs"]))])
    with torch.no_grad():
        own = squashed(torch.tensor(rows, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(spec["check"]["outputs"], own, rtol=1e-5, atol=1e-6)
    # Logits of the same kind of model, said to be multi-label: read through a sigmoid each.
    logits = nn.Sequential(nn.Linear(6, 5)).eval()
    told = extract_structure(logits, sample=torch.rand(1, 6), labels=tags, task="multilabel_classification").spec
    assert told["head"]["link"] == "sigmoid"
    rows = np.array([[told["check"]["inputs"][f"x{i}"][row][0] for i in range(6)] for row in range(len(told["check"]["outputs"]))])
    with torch.no_grad():
        own = 1 / (1 + np.exp(-logits(torch.tensor(rows, dtype=torch.float32)).numpy()))
    np.testing.assert_allclose(told["check"]["outputs"], own, rtol=1e-5, atol=1e-6)


def test_several_numbers_without_labels_stay_a_regression_and_say_what_to_pass(torch):
    nn = torch.nn
    spec = extract_structure(nn.Sequential(nn.Linear(4, 12)).eval(), sample=torch.rand(1, 4)).spec
    assert spec["task"] == "regression"
    reason = graphs.drawn_reason(spec)
    assert reason and "12 numbers" in reason and "task='generation'" in reason and "labels=" in reason


# --- inputs: each its own feature ----------------------------------------------------------------


def test_two_ids_are_two_integer_columns_ranged_by_their_tables(torch):
    nn = torch.nn

    class Recommender(nn.Module):
        def __init__(self):
            super().__init__()
            self.users, self.items = nn.Embedding(50, 8), nn.Embedding(80, 8)

        def forward(self, user, item):
            return torch.sigmoid((self.users(user) * self.items(item)).sum(-1))

    torch.manual_seed(2)
    model = Recommender().eval()
    spec = extract_structure(model, sample=(torch.tensor([[3]]), torch.tensor([[7]]))).spec
    assert spec["task"] == "binary_classification" and runs(spec)
    assert features_of(spec) == {
        "user": {"name": "user", "type": "integer", "min": 0.0, "max": 49.0},
        "item": {"name": "item", "type": "integer", "min": 0.0, "max": 79.0},
    }
    assert inputs_of(spec) == {"user": {"features": ["user"]}, "item": {"features": ["item"]}}
    users, items = spec["check"]["inputs"]["user"], spec["check"]["inputs"]["item"]
    assert any(u != i for u, i in zip(users, items)), "the check holds rows whose user and item differ"
    for user, item, answer in zip(users, items, spec["check"]["outputs"]):
        with torch.no_grad():
            own = float(model(torch.tensor([[int(user[0])]]), torch.tensor([[int(item[0])]]))[0, 0])
        assert answer == pytest.approx([own], abs=1e-6)


def test_a_category_id_beside_numbers_is_a_column_of_its_own(torch):
    nn = torch.nn

    class Table(nn.Module):
        def __init__(self):
            super().__init__()
            self.plan, self.head = nn.Embedding(12, 4), nn.Linear(9, 2)

        def forward(self, numbers, plan):
            return self.head(torch.cat([numbers, self.plan(plan).squeeze(1)], dim=1))

    spec = extract_structure(Table().eval(), sample=(torch.randn(1, 5), torch.tensor([[4]])), labels=["stays", "leaves"]).spec
    assert spec["feature_names"] == ["x0", "x1", "x2", "x3", "x4", "plan"] and runs(spec)
    assert features_of(spec)["plan"] == {"name": "plan", "type": "integer", "min": 0.0, "max": 11.0}
    named = extract_structure(
        Table().eval(), sample=(torch.randn(1, 5), torch.tensor([[4]])), labels=["stays", "leaves"],
        feature_names=["age", "income", "tenure", "seats", "calls", "plan_kind"],
    ).spec
    assert named["feature_names"] == ["age", "income", "tenure", "seats", "calls", "plan_kind"]


def test_a_recording_is_one_tensor_and_a_short_vector_is_columns(torch):
    nn = torch.nn

    class Listener(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.head = nn.Conv1d(1, 4, 64, stride=16), nn.Linear(4, 3)

        def forward(self, waveform):
            return self.head(torch.relu(self.conv(waveform.unsqueeze(1))).mean(-1))

    spec = extract_structure(Listener().eval(), sample=torch.randn(1, 4000), labels=["yes", "no", "stop"]).spec
    assert features_of(spec) == {"waveform": {"name": "waveform", "type": "audio", "samples": 4000, "sample_rate": 16000}}
    assert runs(spec)

    class Long(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2048, 2)

        def forward(self, x):
            return self.head(x)

    long = extract_structure(Long().eval(), sample=torch.randn(1, 2048), labels=["a", "b"]).spec
    assert [feature["type"] for feature in long["features"]] == ["tensor"]
    short = extract_structure(nn.Sequential(nn.Linear(40, 2)).eval(), sample=torch.randn(1, 40), labels=["a", "b"]).spec
    assert short["feature_names"][:2] == ["x0", "x1"] and len(short["feature_names"]) == 40


def test_a_single_number_beside_a_picture_is_a_column(torch):
    nn = torch.nn

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, sample, timestep):
            return self.conv(sample) * timestep.reshape(-1, 1, 1, 1) / 1000

    spec = extract_structure(Denoiser().eval(), sample=(torch.randn(1, 3, 8, 8), torch.tensor([10.0])), task="generation").spec
    assert features_of(spec)["timestep"] == {"name": "timestep", "type": "numeric"}
    assert inputs_of(spec)["timestep"] == {"features": ["timestep"]} and runs(spec)


def test_names_for_a_model_without_columns_name_its_features(torch):
    nn = torch.nn

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.left, self.right = nn.LSTM(3, 4, batch_first=True), nn.Linear(4, 2)

        def forward(self, series, other):
            out, _ = self.left(series + other)
            return self.right(out[:, -1])

    sample = (torch.randn(1, 6, 3), torch.randn(1, 6, 3))
    spec = extract_structure(Pair().eval(), sample=sample, labels=["a", "b"], feature_names=["load", "weather"]).spec
    assert sorted(features_of(spec)) == ["load", "weather"]
    with pytest.raises(StructureError) as caught:
        extract_structure(Pair().eval(), sample=sample, labels=["a", "b"], feature_names=["only"])
    assert "2 features" in caught.value.message


def test_an_example_the_features_cannot_give_is_refused(torch):
    nn = torch.nn

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(20, 4)

        def forward(self, left, right):
            return (self.table(left) - self.table(right)).sum(-1)

    with pytest.raises(StructureError) as caught:
        extract_structure(
            Two().eval(), sample=(torch.tensor([[3]]), torch.tensor([[7]])), task="regression",
            inputs={"left": {"features": ["id"]}, "right": {"features": ["id"]}},
        )
    assert caught.value.param == "inputs" and "left and right are read from id" in caught.value.message


# --- Hugging Face models say what they are --------------------------------------------------------


@pytest.fixture(scope="module")
def transformers():
    return pytest.importorskip("transformers")


def bert_config(transformers, **more):
    return transformers.BertConfig(
        vocab_size=60, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=32, **more,
    )


def encoded(torch):
    ids = torch.tensor([[2, 15, 9, 31, 3, 0, 0, 0]])
    return {"input_ids": ids, "attention_mask": (ids != 0).long(), "token_type_ids": torch.zeros_like(ids)}


def test_a_hugging_face_classifier_says_its_task_and_classes(torch, transformers):
    torch.manual_seed(3)
    config = bert_config(transformers, num_labels=3, id2label={0: "negative", 1: "neutral", 2: "positive"},
                         label2id={"negative": 0, "neutral": 1, "positive": 2})
    model = transformers.BertForSequenceClassification(config).eval()
    spec = extract_structure(model, sample=encoded(torch), feature_names=["review"]).spec
    assert spec["task"] == "multiclass_classification" and spec["outputs"] == {"labels": ["negative", "neutral", "positive"]}
    assert list(features_of(spec)) == ["review"] and features_of(spec)["review"]["type"] == "tokens"
    assert {source["part"] for source in inputs_of(spec).values()} == {"ids", "mask", "type_ids"} and runs(spec)
    tagged = transformers.BertForSequenceClassification(
        bert_config(transformers, num_labels=4, problem_type="multi_label_classification")
    ).eval()
    multi = extract_structure(tagged, sample=encoded(torch)).spec
    assert multi["task"] == "multilabel_classification" and multi["head"]["link"] == "sigmoid"
    assert multi["outputs"] == {"labels": ["class_0", "class_1", "class_2", "class_3"]}


def test_a_hugging_face_language_model_converts_with_its_cache_on(torch, transformers):
    torch.manual_seed(4)
    config = transformers.GPT2Config(vocab_size=50, n_embd=16, n_layer=1, n_head=2, n_positions=16)
    model = transformers.GPT2LMHeadModel(config).eval()
    assert model.config.use_cache is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(model, sample=torch.tensor([[5, 7, 9, 11]]), max_length=8).spec
    assert spec["task"] == "text_generation" and runs(spec) and spec["features"][0]["max_length"] == 8
    assert model.config.use_cache is True  # left as it was


def test_a_hugging_face_encoder_gives_its_pooled_embedding(torch, transformers):
    model = transformers.BertModel(bert_config(transformers)).eval()
    spec = extract_structure(model, sample=encoded(torch)).spec
    assert spec["task"] == "embedding" and runs(spec)
    assert graphs.hugging_face_hints(model)["output"] == "pooler_output"
    assert graphs.hugging_face_hints(torch.nn.Linear(2, 2)) == {}


# --- a model that needed converting, and could not be ------------------------------------------------


def test_a_module_that_only_converts_is_refused_with_why_and_not_sketched(torch, monkeypatch):
    nn = torch.nn

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.lstm, self.head = nn.Embedding(10, 4), nn.LSTM(4, 4, batch_first=True), nn.Linear(4, 2)

        def forward(self, ids):
            out, _ = self.lstm(self.embed(ids))
            return self.head(out[:, -1])

    def refuse(*args, **kwargs):
        raise graphs._NoConversion("torch.onnx.export could not write it (a test said so)")

    monkeypatch.setattr(graphs, "_export_torch", refuse)
    with pytest.raises(StructureError) as caught:
        extract_structure(Reader().eval(), sample=torch.tensor([[1, 2, 3]]), task="binary_classification")
    assert caught.value.code == "unsupported_estimator" and "a test said so" in caught.value.message
    assert "via='native'" in caught.value.message


# --- a diffusion model brings its sampling loop -----------------------------------------------------------


def test_a_diffusers_pipeline_brings_its_scheduler(torch):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(5)
    unet = diffusers.UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3, layers_per_block=1, block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "UpBlock2D"), norm_num_groups=4,
    ).eval()
    scheduler = diffusers.DDPMScheduler(num_train_timesteps=200, beta_schedule="squaredcos_cap_v2", clip_sample=True)
    pipeline = diffusers.DDPMPipeline(unet=unet, scheduler=scheduler)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(pipeline, sample=(torch.randn(1, 3, 8, 8), torch.tensor([10.0]))).spec
    assert spec["task"] == "generation" and runs(spec)
    sampler = spec["sampler"]
    assert sampler["type"] == "diffusion" and sampler["sample"] == "sample" and sampler["timestep"] == "timestep"
    assert sampler["prediction"] == "epsilon" and sampler["clip"] == [-1.0, 1.0] and sampler["layout"] == "chw"
    np.testing.assert_allclose(sampler["alphas_cumprod"], scheduler.alphas_cumprod.numpy(), rtol=1e-6)
    assert features_of(spec)["sample"]["type"] == "tensor"
    with pytest.warns(UserWarning, match="without its scheduler"):
        extract_structure(unet, sample=(torch.randn(1, 3, 8, 8), torch.tensor([10.0])))


def test_a_sampler_can_be_written_out(torch):
    spec = {"features": [{"name": "noise", "type": "tensor", "shape": [3, 8, 8]}, {"name": "t", "type": "numeric"}]}
    made = graphs.sampler_spec({"steps": 20}, spec)
    assert made == {"steps": 20, "type": "diffusion", "sample": "noise", "layout": "chw", "timestep": "t"}
    with pytest.raises(StructureError) as caught:
        graphs.sampler_spec({}, {"features": [{"name": "noise", "type": "tensor", "shape": [4]}]})
    assert caught.value.param == "sampler"


# --- Keras 3 recurrent layers ------------------------------------------------------------------------------


def test_a_keras_recurrent_model_converts_through_its_own_signature():
    keras = pytest.importorskip("keras")
    pytest.importorskip("tf2onnx")
    rng = np.random.default_rng(0)
    series = rng.normal(size=(16, 10, 3)).astype("float32")
    model = keras.Sequential([keras.Input((10, 3)), keras.layers.LSTM(4), keras.layers.Dense(2, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(series, (series[:, -1, 0] > 0).astype(int), epochs=1, verbose=0)
    spec = extract_structure(model).spec
    assert spec["family"] == "graph" and spec["task"] == "binary_classification" and runs(spec)
    assert [feature["type"] for feature in spec["features"]] == ["sequence"]
