"""Structures for the tasks that are not a class or a number: clusters, masks, text written from text, sounds.

These are what the client sends; the API's side of each (how it reports a prediction) has its own tests there.
"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnx")

from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402


def runs(spec):
    return "check" in spec and not spec.get("approximate") and spec_is_executable(spec)


def table_rows(spec):
    names = spec["feature_names"]
    return np.array([[spec["check"]["inputs"][name][row][0] for name in names] for row in range(len(spec["check"]["outputs"]))])


# --- clusters -----------------------------------------------------------------------------------------------------


def test_kmeans_is_a_clustering_read_by_its_distances():
    pytest.importorskip("skl2onnx")
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    rows = np.random.default_rng(0).normal(size=(120, 4))
    names = ["a", "b", "c", "d"]
    kmeans = KMeans(3, n_init=3, random_state=0).fit(rows)
    spec = extract_structure(kmeans, feature_names=names, check_rows=rows[:10]).spec
    assert spec["task"] == "clustering" and spec["head"]["link"] == "argmin" and runs(spec)
    assert spec["outputs"] == {"labels": ["cluster 0", "cluster 1", "cluster 2"]}
    np.testing.assert_allclose(spec["check"]["outputs"], kmeans.transform(table_rows(spec)), rtol=1e-4, atol=1e-4)
    mixture = GaussianMixture(3, random_state=0).fit(rows)
    soft = extract_structure(mixture, feature_names=names).spec
    assert soft["task"] == "clustering" and soft["head"]["link"] == "argmax" and runs(soft)


# --- masks --------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_a_segmentation_model_says_how_its_classes_are_held(torch):
    nn = torch.nn

    class Painter(nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.head = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 4, 1)

        def forward(self, image):
            return self.head(torch.relu(self.body(image)))

    torch.manual_seed(0)
    spec = extract_structure(Painter().eval(), sample=torch.rand(1, 3, 16, 16), task="segmentation").spec
    assert spec["head"]["kind"] == "segmentation" and spec["head"]["layout"] == "chw" and spec["head"]["link"] == "none"
    assert spec["outputs"] == {"labels": ["class_0", "class_1", "class_2", "class_3"]} and runs(spec)
    named = extract_structure(Painter().eval(), sample=torch.rand(1, 3, 16, 16), task="segmentation", labels=["sky", "road", "car", "tree"]).spec
    assert named["outputs"] == {"labels": ["sky", "road", "car", "tree"]}


def test_a_hugging_face_segmenter_says_its_task_and_classes(torch):
    transformers = pytest.importorskip("transformers")
    config = transformers.SegformerConfig(
        num_channels=3, num_encoder_blocks=1, depths=[1], sr_ratios=[1], hidden_sizes=[8], patch_sizes=[3],
        strides=[2], num_attention_heads=[1], mlp_ratios=[2], decoder_hidden_size=8, num_labels=3,
        id2label={0: "background", 1: "cat", 2: "dog"}, label2id={"background": 0, "cat": 1, "dog": 2},
    )
    model = transformers.SegformerForSemanticSegmentation(config).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(model, sample=torch.rand(1, 3, 32, 32)).spec
    assert spec["task"] == "segmentation" and spec["head"]["kind"] == "segmentation"
    assert spec["outputs"] == {"labels": ["background", "cat", "dog"]} and runs(spec)


# --- text written from text -----------------------------------------------------------------------------------------


def test_an_encoder_decoder_writes_from_its_own_input(torch):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(1)
    config = transformers.T5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_decoder_layers=1,
                                   num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    model = transformers.T5ForConditionalGeneration(config).eval()
    ids = torch.tensor([[5, 9, 13, 21, 1, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(model, sample={"input_ids": ids, "attention_mask": (ids != 0).long()}).spec
    assert spec["task"] == "sequence_to_sequence" and spec["head"]["link"] == "softmax" and runs(spec)
    bound = {item["name"].split(".", 1)[-1]: item["from"] for item in spec["graph"]["inputs"]}
    assert bound["decoder_input_ids"] == {"feature": "decoder", "part": "ids", "pad_id": 0}
    assert bound["input_ids"] == {"feature": "text", "part": "ids"} and bound["attention_mask"]["part"] == "mask"
    assert spec["decoder"] == {"feature": "decoder", "start_token": 0, "eos_token": 1, "pad_token": 0, "max_length": 6}
    written = next(feature for feature in spec["features"] if feature["name"] == "decoder")
    assert written["generated"] is True and written["baseline"] == [0] and written["type"] == "tokens"
    # The check holds what the decoder writes next, after the tokens each row says it has written.
    for text, decoded, answer in zip(spec["check"]["inputs"]["text"], spec["check"]["inputs"]["decoder"], spec["check"]["outputs"]):
        count = sum(1 for item in decoded if item >= 0)
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor([[int(item) if item >= 0 else 0 for item in text]]),
                attention_mask=(torch.tensor([text]) >= 0).long(),
                decoder_input_ids=torch.tensor([[int(item) if item >= 0 else 0 for item in decoded]]),
            ).logits[0, count - 1]
        np.testing.assert_allclose(answer, torch.softmax(logits, -1).numpy(), rtol=1e-4, atol=1e-6)


def test_an_encoder_decoder_is_known_by_its_language_model_head_too(torch):
    # Marian's class is MarianMTModel, not ...ForConditionalGeneration.
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(3)
    config = transformers.MarianConfig(
        vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32,
        pad_token_id=63, eos_token_id=0, decoder_start_token_id=63,
    )
    model = transformers.MarianMTModel(config).eval()
    ids = torch.tensor([[5, 9, 13, 21, 0, 63]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(model, sample={"input_ids": ids, "attention_mask": (ids != 63).long()}).spec
    assert spec["task"] == "sequence_to_sequence" and runs(spec)
    assert spec["decoder"] == {"feature": "decoder", "start_token": 63, "eos_token": 0, "pad_token": 63, "max_length": 6}


# --- sounds -----------------------------------------------------------------------------------------------------------


def test_a_recording_is_an_audio_feature(torch):
    nn = torch.nn

    class Listener(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.head = nn.Conv1d(1, 4, 64, stride=16), nn.Linear(4, 3)

        def forward(self, input_values):
            return self.head(torch.relu(self.conv(input_values.unsqueeze(1))).mean(-1))

    spec = extract_structure(Listener().eval(), sample=torch.randn(1, 3200), labels=["yes", "no", "stop"]).spec
    assert spec["features"] == [{"name": "audio", "type": "audio", "samples": 3200, "sample_rate": 16000}] and runs(spec)
    told = extract_structure(
        Listener().eval(), sample=torch.randn(1, 3200), labels=["yes", "no", "stop"],
        inputs={"input_values": {"feature": "speech", "sample_rate": 8000}},
    ).spec
    assert told["features"] == [{"name": "speech", "type": "audio", "samples": 3200, "sample_rate": 8000}]


# --- boxes in a picture ---------------------------------------------------------------------------------------------


def test_a_detector_comes_out_as_one_tensor_of_boxes_and_scores(torch):
    nn = torch.nn

    class Finder(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.box, self.score = nn.Linear(8, 5 * 4), nn.Linear(8, 5 * 3)

        def forward(self, image):
            found = self.body(image)
            return {"boxes": torch.sigmoid(self.box(found)).view(-1, 5, 4), "scores": self.score(found).view(-1, 5, 3)}

    torch.manual_seed(3)
    finder = Finder().eval()
    spec = extract_structure(
        finder, sample=torch.rand(1, 3, 16, 16), task="object_detection", labels=["cat", "dog", "bird"],
        output={"boxes": "boxes", "scores": "scores", "box_format": "cxcywh", "normalized": True, "score_link": "softmax"},
    ).spec
    head = spec["head"]
    assert head["kind"] == "detection" and head["boxes"] == [0, 4] and head["scores"] == [4, 7] and head["link"] == "none"
    assert head["box_format"] == "cxcywh" and head["normalized"] is True and head["score_link"] == "softmax"
    assert spec["outputs"] == {"labels": ["cat", "dog", "bird"]} and runs(spec)
    assert len(spec["check"]["outputs"][0]) == 5 * 7  # five boxes of four numbers and three scores


def test_a_hugging_face_detector_says_how_its_boxes_are_read(torch):
    transformers = pytest.importorskip("transformers")
    config = transformers.YolosConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, image_size=[32, 32],
        patch_size=8, num_detection_tokens=4, num_labels=2, id2label={0: "cat", 1: "dog"}, label2id={"cat": 0, "dog": 1},
    )
    torch.manual_seed(4)
    model = transformers.YolosForObjectDetection(config).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(model, sample=torch.rand(1, 3, 32, 32)).spec
    head = spec["head"]
    assert spec["task"] == "object_detection" and head["kind"] == "detection" and head["no_object"] == "last"
    assert head["box_format"] == "cxcywh" and head["normalized"] is True and head["score_link"] == "softmax"
    assert head["scores"] == [4, 7] and spec["outputs"] == {"labels": ["cat", "dog"]} and runs(spec)
