"""What the review of the last fixes found in the client, each as the case it was found on."""

from __future__ import annotations

import numpy as np
import pytest

from kaimei import Kaimei, onnx_graph
from kaimei.agent import bind_inputs
from kaimei.onnx_graph import segmentation_head

INT = np.dtype("int64")


def test_a_fraction_of_the_picture_is_the_same_fraction_down_and_across():
    """A hundred boxes of four numbers were taken for a mask of an 800 by 1200 picture: each side divides it."""
    picture = [(800, 1200)]
    assert onnx_graph._picture_fit([1, 100, 4], picture) == 2  # boxes: no mask of this picture
    assert onnx_graph._picture_fit([1, 100, 200, 300], picture) == 1  # masks at a quarter of the size
    assert onnx_graph._picture_fit([1, 3, 800, 1200], picture) == 0
    assert onnx_graph._picture_fit([1, 3, 1600, 2400], picture) == 1  # at twice the size
    assert onnx_graph._picture_fit([1, 3, 1600, 1200], picture) == 2


def test_one_column_of_a_classifier_is_its_margin_only_where_the_classifier_gives_margins():
    """One column picked from a classifier that gives probabilities was put through a sigmoid again."""
    linear = {"op": "LinearClassifier", "attrs": {}}
    assert onnx_graph._gives_margins(linear)
    assert onnx_graph._gives_margins({"op": "SVMClassifier", "attrs": {"post_transform": b"NONE"}})
    assert not onnx_graph._gives_margins({"op": "LinearClassifier", "attrs": {"post_transform": "LOGISTIC"}})
    assert not onnx_graph._gives_margins({"op": "SVMClassifier", "attrs": {"prob_a": [0.1], "prob_b": [0.2]}})
    assert not onnx_graph._gives_margins({"op": "TreeEnsembleClassifier", "attrs": {}})
    assert not onnx_graph._gives_margins(None)


def test_a_constant_added_before_a_clip_leaves_the_cut_to_the_values():
    """A hard sigmoid written out, relu6(x + 3) / 6, was refused: the walk took the sum for raw scores cut at 0."""
    constants = {
        "k": np.ones((1, 1, 1, 1), np.float32),
        "three": np.float32(3.0),
        "low": np.float32(0.0),
        "high": np.float32(6.0),
        "six": np.float32(6.0),
        "bias": np.float32(0.25),
    }
    hard_sigmoid = [
        {"op": "Conv", "inputs": ["x", "k"], "outputs": ["scores"]},
        {"op": "Add", "inputs": ["scores", "three"], "outputs": ["shifted"]},
        {"op": "Clip", "inputs": ["shifted", "low", "high"], "outputs": ["kept"]},
        {"op": "Div", "inputs": ["kept", "six"], "outputs": ["out"]},
    ]
    assert onnx_graph._score_cut(hard_sigmoid, "out", constants) is None
    # Values from 0 to 1 then read as probabilities, cut at one half.
    assert onnx_graph._cut_by_values([np.array([[0.0, 0.4], [0.6, 1.0]])]) == (0.5, "probability")
    # A bias added to raw scores, with nothing clipped or scaled after it, is still raw scores cut at nought.
    biased = [
        {"op": "Conv", "inputs": ["x", "k"], "outputs": ["scores"]},
        {"op": "Add", "inputs": ["scores", "bias"], "outputs": ["out"]},
    ]
    assert onnx_graph._score_cut(biased, "out", constants) == (0.0, "raw")


def test_a_threshold_settles_a_height_and_a_width_of_whole_floats():
    """A hard mask written as floats (0.0 and 1.0) was refused as not told from class numbers, threshold or no."""
    hard = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    head = segmentation_head(hard, threshold=0.5)
    assert head["threshold"] == 0.5 and head["layout"] != "indices"
    with pytest.raises(onnx_graph.OnnxImportError) as refused:
        segmentation_head(hard)
    assert refused.value.param == "mask_layout"


def _encoder_decoder():
    features = [{"name": "text", "type": "text"}, {"name": "decoder", "type": "tokens", "generated": True}]
    file_inputs = [("input_ids", INT, [1, 8]), ("attention_mask", INT, [1, 8]), ("decoder_input_ids", INT, [1, 8])]
    return features, file_inputs


def test_feeding_one_text_yourself_leaves_the_others_bound_by_their_names():
    """An agent with no structure refused an encoder-decoder whose decoder input the caller feeds with a function."""
    features, file_inputs = _encoder_decoder()
    bound = bind_inputs(file_inputs, features, {"decoder_input_ids": lambda row: np.zeros((1, 8), dtype="int64")})
    by_name = {item["name"]: item for item in bound}
    assert callable(by_name["decoder_input_ids"].get("call"))
    assert by_name["input_ids"]["source"]["feature"] == "text" and by_name["input_ids"]["source"]["part"] == "ids"
    assert by_name["attention_mask"]["source"]["feature"] == "text"
    # With nothing fed by the caller the names still settle it by themselves.
    plain = {item["name"]: item for item in bind_inputs(file_inputs, features, None)}
    assert plain["decoder_input_ids"]["source"]["feature"] == "decoder"


def test_two_inputs_that_would_read_the_same_text_are_still_asked_about():
    features, _ = _encoder_decoder()
    file_inputs = [("src", INT, [1, 8]), ("tgt", INT, [1, 8]), ("extra", np.dtype("float32"), [1, 4])]
    for given in (None, {"extra": lambda row: np.zeros((1, 4), dtype="float32")}):
        with pytest.raises(ValueError):
            bind_inputs(file_inputs, features, given)


def test_a_networks_own_num_features_is_not_taken_for_the_width_of_its_table():
    """A module with ``num_features = 64`` (a hidden width) uploaded with a ten-column table was refused as X."""
    pd = pytest.importorskip("pandas")

    class Net:
        num_features = 64

        def named_parameters(self):
            return iter(())

        def forward(self, x):
            return x

    client = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    frame = pd.DataFrame(np.zeros((3, 10)), columns=[f"c{index}" for index in range(10)])
    with pytest.raises(Exception) as refused:  # it is no model Kaimei reads: refused for that, not for its table
        client.models.upload(Net(), name="net", X=frame)
    assert "was fitted on a table of" not in str(refused.value)


def test_a_generated_hint_without_a_padding_token_takes_the_models():
    """The structure named two padding tokens: 0 on the input, the model's own in ``decoder``."""
    from kaimei import graphs

    assert graphs._padded_as_the_model_pads({"generated": True}, 58100) == {"generated": True, "pad_id": 58100}
    assert graphs._padded_as_the_model_pads({"generated": True, "pad_id": 3}, 58100)["pad_id"] == 3
    plain = {"feature": "decoder", "part": "ids", "pad_id": 0}
    assert graphs._padded_as_the_model_pads(plain, 58100) is plain
    assert graphs._padded_as_the_model_pads({"generated": True}, None) == {"generated": True}


def test_the_feature_a_decoder_writes_into_is_the_one_its_hint_names():
    """With an ordinary hint (``{"feature": "answer", "part": "ids"}``) the structure's decoder pointed at a
    feature called ``decoder``, which did not exist."""
    from kaimei import graphs

    assert graphs._hinted_feature({"feature": "answer", "part": "ids"}) == "answer"
    assert graphs._hinted_feature("answer") == "answer"
    assert graphs._hinted_feature({"generated": True}) is None and graphs._hinted_feature(None) is None


def test_the_vocabulary_of_a_generated_hint_is_for_the_feature_the_importer_binds():
    """``{"generated": True, "vocab": 32000}`` names no feature, so its vocabulary was left out of the check rows."""
    from kaimei import graphs

    hints = {
        "input_ids": {"feature": "text", "vocab": 1000},
        "decoder_input_ids": {"generated": True, "vocab": 32000},
        "attention_mask": {"feature": "text", "part": "mask"},
        "pixels": "photo",
    }
    assert graphs._vocabulary_limits(hints) == {"text": 1000, "decoder": 32000}
    assert graphs._vocabulary_limits({"d": {"generated": True, "feature": "answer", "vocab": 7}}) == {"answer": 7}


def test_double_precision_chosen_by_asking_the_model_is_not_said_to_be_a_failure_of_single():
    from kaimei import graphs

    asked = graphs._precision_said("double", [], asked_the_model=True)
    assert "as scikit-learn computes" in asked and "did not give" not in asked
    failed = graphs._precision_said("double", [], asked_the_model=False)
    assert "did not give the model's own numbers" in failed
    assert graphs._precision_said("single", []) == ""


def test_a_torch_segmenter_given_only_its_input_shape_is_converted_not_read_natively():
    """With ``input_shape`` and no ``sample`` a segmenter went to Kaimei's own readers: called exact, marked as
    running, and unable to answer one prediction (its head said nothing of how a mask is read)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from kaimei.structure import StructureError, extract_structure

    class Segmenter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Conv2d(1, 4, 3, padding=1)
            self.head = torch.nn.Conv2d(4, 1, 1)

        def forward(self, x):
            return self.head(torch.relu(self.body(x)))

    torch.manual_seed(0)
    model = Segmenter().eval()
    spec = extract_structure(model, task="segmentation", input_shape=(1, 8, 8)).spec
    assert spec["family"] == "graph", spec.get("family")
    head = spec["head"]
    assert head.get("layout") == "chw" and head.get("threshold") == 0.0, head  # raw scores, cut at nought
    # Asked for Kaimei's own readers by name, it is refused, naming the task.
    with pytest.raises(StructureError) as refused:
        extract_structure(model, task="segmentation", input_shape=(1, 8, 8), via="native")
    assert refused.value.param == "task"
    # A classifier's task changes nothing: its own readers still read it.
    from kaimei.structure import _onnx_only_arguments

    assert _onnx_only_arguments(model, None, None, None, "multiclass_classification") == []
    assert _onnx_only_arguments(model, None, None, None, "object_detection") == ["task"]
