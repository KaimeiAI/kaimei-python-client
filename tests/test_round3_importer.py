"""How a segmentation model's mask is read off its ONNX graph, and three smaller readings of the importer.

Each test stands for a model that was read wrongly or refused for no reason:

* a segmenter whose scores pass through any step but a handful after the convolution that makes them (a
  log-softmax, a skip sum, a crop, a squeeze, a linear layer on channels-last features) was refused, where its
  class axis can be followed through those steps;
* a segmenter with a second output (torchvision's ``out`` and ``aux``) was refused, where its first output of
  floats holds the mask;
* a reference run copied the model twice;
* a byte-pair tokenizer whose merges name pieces its vocabulary lacks read every word as unknown tokens;
* an input hinted ``{"generated": True}`` (what an encoder-decoder has written so far) was written in a form the
  API refuses, where the form it runs is a feature marked ``generated`` that the input is bound to.

The models are real ONNX graphs with random weights, so every class wins somewhere and a mask read along the
wrong axis is another mask. The API's ``tests/test_round3_masks.py`` runs the same kinds of model through its
engine and compares each mask with ONNX Runtime's, pixel for pixel.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from kaimei import StructureError, onnx_graph  # noqa: E402
from kaimei.onnx_graph import OnnxImportError, import_onnx, segmentation_head  # noqa: E402
from kaimei.text import tokenizer_spec  # noqa: E402
from kaimei.tokenizers import TokenizerError, build_tokenizer  # noqa: E402

F, I64 = TensorProto.FLOAT, TensorProto.INT64
HINT = {"x": {"feature": "photo", "layout": "nchw"}}
CLASSES = 5


def const(name, value, dtype=np.float32):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def conv(name, classes, rng, source="x"):
    """A 1x1 convolution of a picture's three colours into ``classes`` scores.

    The weights are random and the bias centres the scores on mid-grey, so which class is strongest changes from
    pixel to pixel: a mask read along another axis than the classes' is then another mask.
    """
    node = helper.make_node("Conv", [source, f"{name}.w", f"{name}.b"], [name], name=name)
    weight = rng.normal(size=(classes, 3, 1, 1)) * 4.0
    bias = -weight.reshape(classes, 3).sum(axis=1) / 2
    return node, [const(f"{name}.w", weight), const(f"{name}.b", bias)]


def model_of(nodes, outputs, initializers, height, width, opset=17):
    """A model reading one picture ``x`` of ``[1, 3, height, width]``; ``outputs`` are ``(name, element type)``."""
    graph = helper.make_graph(
        list(nodes),
        "segmenter",
        [helper.make_tensor_value_info("x", F, [1, 3, height, width])],
        [helper.make_tensor_value_info(name, kind, None) for name, kind in outputs],
        list(initializers),
    )
    made = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    made.ir_version = 8
    return made


#: A step of :func:`after_conv` that is not one operator: ``.squeeze(0)`` as PyTorch writes it.
ROWS_SQUEEZED = ("squeeze(0), as PyTorch writes it", {}, [])


def rows_squeezed(source, made):
    """``source.squeeze(0)`` as PyTorch writes it when the file leaves the number of rows open: a branch that
    squeezes the axis when it is one, and hands the tensor on as it is when it is not."""
    squeeze = helper.make_node("Squeeze", [source, f"{made}.axes"], [f"{made}.then"])
    then = helper.make_graph([squeeze], "then", [], [helper.make_tensor_value_info(f"{made}.then", F, None)],
                             [const(f"{made}.axes", [0], np.int64)])
    keep = helper.make_node("Identity", [source], [f"{made}.else"])
    other = helper.make_graph([keep], "else", [], [helper.make_tensor_value_info(f"{made}.else", F, None)])
    nodes = [
        helper.make_node("Shape", [source], [f"{made}.shape"]),
        helper.make_node("Gather", [f"{made}.shape", f"{made}.at"], [f"{made}.size"], axis=0),
        helper.make_node("Equal", [f"{made}.size", f"{made}.one"], [f"{made}.is_one"]),
        helper.make_node("If", [f"{made}.is_one"], [made], then_branch=then, else_branch=other),
    ]
    return nodes, [const(f"{made}.at", 0, np.int64), const(f"{made}.one", 1, np.int64)]


def after_conv(*steps, classes=CLASSES, kind=F, opset=17):
    """A model builder: the convolution, then ``steps``, each ``(operator, attributes, constant inputs)``."""

    def build(height, width):
        rng = np.random.default_rng(7)
        node, initializers = conv("scores", classes, rng)
        nodes, current = [node], "scores"
        for index, (op, attrs, constants) in enumerate(steps):
            if op == ROWS_SQUEEZED[0]:
                branch, more = rows_squeezed(current, f"t{index}")
                nodes += branch
                initializers += more
                current = f"t{index}"
                continue
            names = []
            for name, value, dtype in constants:
                value = value(height, width, classes) if callable(value) else value
                initializers.append(const(f"s{index}.{name}", value, dtype))
                names.append(f"s{index}.{name}")
            nodes.append(helper.make_node(op, [current, *names], [f"t{index}"], **attrs))
            current = f"t{index}"
        return model_of(nodes, [(current, kind)], initializers, height, width, opset)

    return build


def skip_sum(height, width):
    rng = np.random.default_rng(7)
    first, a = conv("a", CLASSES, rng)
    second, b = conv("b", CLASSES, rng)
    add = helper.make_node("Add", ["a", "b"], ["sum"])
    return model_of([first, second, add], [("sum", F)], [*a, *b], height, width)


def linear_after_permute(height, width):
    """Features moved to channels last, then a linear layer (a MatMul and an Add): classes last."""
    rng = np.random.default_rng(7)
    features, initializers = conv("features", 8, rng)
    nodes = [
        features,
        helper.make_node("Transpose", ["features"], ["nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", ["nhwc", "fc.w"], ["fc"]),
        helper.make_node("Add", ["fc", "fc.b"], ["logits"]),
    ]
    initializers += [const("fc.w", rng.normal(size=(8, CLASSES))), const("fc.b", rng.normal(size=CLASSES))]
    return model_of(nodes, [("logits", F)], initializers, height, width)


def gemm_for_each_pixel(height, width):
    """Pixels as the rows of a table through a Gemm, and back to a picture: classes last."""
    rng = np.random.default_rng(7)
    features, initializers = conv("features", 8, rng)
    nodes = [
        features,
        helper.make_node("Transpose", ["features"], ["nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node("Reshape", ["nhwc", "rows"], ["table"]),
        helper.make_node("Gemm", ["table", "fc.w", "fc.b"], ["fc"]),
        helper.make_node("Reshape", ["fc", "back"], ["logits"]),
    ]
    initializers += [
        const("rows", [-1, 8], np.int64),
        const("fc.w", rng.normal(size=(8, CLASSES))),
        const("fc.b", rng.normal(size=CLASSES)),
        const("back", [1, height, width, CLASSES], np.int64),
    ]
    return model_of(nodes, [("logits", F)], initializers, height, width)


def by_einsum(equation, classes=CLASSES):
    """Scores made by an Einsum with a constant: a step the importer does not follow."""

    def build(height, width):
        mix = const("mix", np.random.default_rng(7).normal(size=(classes, 3)))
        return model_of([helper.make_node("Einsum", ["x", "mix"], ["scores"], equation=equation)], [("scores", F)],
                        [mix], height, width)

    return build


def own_output(spec):
    """The model's own output for the first check row, in the shape the model gives it."""
    said = spec["graph"]["values"][spec["head"]["output"]]
    return np.asarray(spec["check"]["outputs"][0]).reshape(said["shape"])


def mask_by(layout, value, threshold=0.5):
    """The mask Kaimei reads off ``value`` under ``layout``: as ``mask_output`` in the API's task_outputs.py.

    One channel of scores, or a height and a width of them with no axis of classes, is the foreground at
    ``threshold`` or more.
    """
    grid = value
    if layout == "indices":
        if grid.ndim == 4 and grid.shape[0] == grid.shape[-1] == 1 and grid.shape[1] != 1:
            grid = grid[0, :, :, 0]
        return np.rint(grid.reshape(grid.shape[-2:])).astype(int)
    if grid.ndim == 2:
        return (grid >= threshold).astype(int)
    while grid.ndim > 3:
        grid = grid[0]
    scores = grid if layout == "chw" else np.moveaxis(grid, -1, 0)
    return (scores[0] >= threshold).astype(int) if scores.shape[0] == 1 else np.argmax(scores, axis=0)


AXES = [("axes", [1], np.int64)]
ROW = [("axes", [0], np.int64)]
FOLLOWED = {
    # name: (builder, the layout a right reading gives, the axis of the output that truly holds the classes)
    "log-softmax": (after_conv(("LogSoftmax", {"axis": 1}, [])), "chw", 1),
    "add a constant": (after_conv(("Add", {}, [("b", lambda h, w, c: np.arange(c).reshape(1, c, 1, 1), np.float32)])),
                       "chw", 1),
    "skip sum of two convolutions": (skip_sum, "chw", 1),
    "multiply by a constant": (after_conv(("Mul", {}, [("k", 0.5, np.float32)])), "chw", 1),
    "divide by a constant": (after_conv(("Div", {}, [("k", 4.0, np.float32)])), "chw", 1),
    "clip": (after_conv(("Clip", {}, [("low", -1.0, np.float32), ("high", 1.0, np.float32)])), "chw", 1),
    "leaky relu": (after_conv(("LeakyRelu", {"alpha": 0.1}, [])), "chw", 1),
    "tanh": (after_conv(("Tanh", {}, [])), "chw", 1),
    "resize": (after_conv(("Resize", {"mode": "nearest"}, [("roi", [], np.float32),
                                                             ("scales", [1, 1, 2, 2], np.float32)])), "chw", 1),
    "pad": (after_conv(("Pad", {}, [("pads", [0, 0, 1, 2, 0, 0, 1, 2], np.int64)])), "chw", 1),
    "crop (slice)": (after_conv(("Slice", {}, [("from", [1, 1], np.int64), ("to", [-1, -1], np.int64),
                                               ("axes", [2, 3], np.int64)])), "chw", 1),
    "row axis squeezed away": (after_conv(("Squeeze", {}, ROW)), "chw", 0),
    "row axis reshaped away": (after_conv(("Reshape", {}, [("to", lambda h, w, c: [c, h, w], np.int64)])), "chw", 0),
    "moved to channels last": (after_conv(("Transpose", {"perm": [0, 2, 3, 1]}, [])), "hwc", 3),
    "channels last, then softmax": (after_conv(("Transpose", {"perm": [0, 2, 3, 1]}, []),
                                               ("Softmax", {"axis": -1}, [])), "hwc", 3),
    "channels last, row axis squeezed": (after_conv(("Transpose", {"perm": [0, 2, 3, 1]}, []),
                                                    ("Squeeze", {}, ROW)), "hwc", 2),
    "linear layer after a permute": (linear_after_permute, "hwc", 3),
    "gemm for each pixel": (gemm_for_each_pixel, "hwc", 3),
    "rows squeezed as PyTorch writes it": (after_conv(ROWS_SQUEEZED), "chw", 0),
    "channels last, rows squeezed as PyTorch writes it": (
        after_conv(("Transpose", {"perm": [0, 2, 3, 1]}, []), ROWS_SQUEEZED), "hwc", 2),
}


@pytest.mark.parametrize("name", sorted(FOLLOWED))
@pytest.mark.parametrize("height, width", [(12, 16), (5, 5)])
def test_a_segmenters_class_axis_is_followed_back_to_the_step_that_makes_its_classes(name, height, width):
    """Neither picture says by its shape where five classes lie (12 x 16 x 5, 5 x 5 x 5): the graph does."""
    build, expected, axis = FOLLOWED[name]
    spec = import_onnx(build(height, width), task="segmentation", inputs=dict(HINT))
    assert spec["head"]["kind"] == "segmentation" and spec["head"]["layout"] == expected
    value = own_output(spec)
    truth = np.argmax(value, axis=axis)
    truth = truth.reshape(truth.shape[-2:])
    assert len(np.unique(truth)) > 1  # the classes differ from pixel to pixel, so the axis matters
    np.testing.assert_array_equal(mask_by(expected, value), truth)


def test_scores_upsampled_by_an_old_exporter_are_followed_too():
    pytest.importorskip("onnxruntime")  # ONNX's own evaluator no longer runs Upsample
    build = after_conv(("Upsample", {"mode": "nearest"}, [("scales", [1, 1, 2, 2], np.float32)]), opset=9)
    spec = import_onnx(build(12, 16), task="segmentation", inputs=dict(HINT))
    assert spec["head"]["layout"] == "chw" and own_output(spec).shape == (1, CLASSES, 24, 32)


@pytest.mark.parametrize("squeezed", [1, 0])
def test_one_score_for_each_pixel_is_read_with_its_class_axis_squeezed_away(squeezed):
    """``sigmoid(conv(x)).squeeze(1)``: [1, height, width], one foreground score a pixel."""
    build = after_conv(("Sigmoid", {}, []), ("Squeeze", {}, [("axes", [squeezed], np.int64)]), classes=1)
    spec = import_onnx(build(12, 16), task="segmentation", inputs=dict(HINT))
    value = own_output(spec)
    assert spec["head"]["layout"] == "chw" and value.shape == (1, 12, 16)
    np.testing.assert_array_equal(mask_by("chw", value), (value[0] >= 0.5).astype(int))
    assert 0 < mask_by("chw", value).mean() < 1


NUMBERS = {
    # name: (argmax axis, keepdims, classes moved last first, steps after the argmax, the output's shape)
    "argmax": (1, 0, False, (), (1, 12, 16)),
    "argmax that keeps its axis": (1, 1, False, (), (1, 1, 12, 16)),
    "argmax, then a cast": (1, 1, False, (("Cast", {"to": TensorProto.INT32}),), (1, 1, 12, 16)),
    "channels last argmax that keeps its axis": (-1, 1, True, (), (1, 12, 16, 1)),
    "channels last argmax, a cast and an identity": (
        -1, 1, True, (("Cast", {"to": TensorProto.INT32}), ("Identity", {})), (1, 12, 16, 1)),
    "argmax, then a cast to floats": (1, 0, False, (("Cast", {"to": F}),), (1, 12, 16)),
    "argmax, then moved to channels last": (1, 1, False, (("Transpose", {"perm": [0, 2, 3, 1]}),), (1, 12, 16, 1)),
}  # fmt: skip


def numbers_model(name, height=12, width=16):
    axis, keep, last, after, _shape = NUMBERS[name]
    rng = np.random.default_rng(7)
    node, initializers = conv("scores", CLASSES, rng)
    nodes, current = [node], "scores"
    if last:
        nodes.append(helper.make_node("Transpose", ["scores"], ["nhwc"], perm=[0, 2, 3, 1]))
        current = "nhwc"
    nodes.append(helper.make_node("ArgMax", [current], ["picked"], axis=axis, keepdims=keep))
    current = "picked"
    for index, (op, attrs) in enumerate(after):
        nodes.append(helper.make_node(op, [current], [f"a{index}"], **attrs))
        current = f"a{index}"
    kind = F if any(attrs.get("to") == F for _op, attrs in after) else I64
    return model_of(nodes, [(current, kind)], initializers, height, width)


@pytest.mark.parametrize("name", sorted(NUMBERS))
@pytest.mark.parametrize("task", ["segmentation", None])
def test_class_numbers_are_known_by_the_argmax_that_makes_them_whatever_comes_after_it(name, task):
    """A cast after the argmax made the output unreadable: only the output's own step was looked at."""
    spec = import_onnx(numbers_model(name), task=task, inputs=dict(HINT))
    assert spec["task"] == "segmentation" and spec["head"]["layout"] == "indices"
    value = own_output(spec)
    assert value.shape == NUMBERS[name][-1] and len(np.unique(value)) > 1
    assert mask_by("indices", value).shape == (12, 16)


def test_a_long_run_of_steps_after_the_convolution_is_followed_to_its_end():
    """Fifteen hundred steps one after another: more than Python lets a function call itself."""
    nodes = [{"op": "Conv", "domain": "", "inputs": ["x", "w"], "outputs": ["t0"], "attrs": {}}]
    nodes += [
        {"op": "Relu", "domain": "", "inputs": [f"t{index}"], "outputs": [f"t{index + 1}"], "attrs": {}}
        for index in range(1500)
    ]
    scores = np.zeros((1, 5, 3, 4), dtype=np.float32)
    results = {f"t{index}": scores for index in range(1501)}
    head = segmentation_head(scores, nodes=nodes, output="t1500", results=results)
    assert head == {"kind": "segmentation", "layout": "chw"}
    moved = [*nodes, {"op": "Transpose", "domain": "", "inputs": ["t1500"], "outputs": ["last"],
                      "attrs": {"perm": [0, 2, 3, 1]}}]
    results["last"] = np.zeros((1, 3, 4, 5), dtype=np.float32)
    assert segmentation_head(results["last"], nodes=moved, output="last", results=results)["layout"] == "hwc"


def test_scores_alone_and_a_classifiers_class_number_are_not_taken_for_a_segmentation():
    scores = import_onnx(after_conv()(12, 16), inputs=dict(HINT))
    assert scores["task"] == "regression"
    rng = np.random.default_rng(7)
    node, initializers = conv("scores", CLASSES, rng)
    nodes = [
        node,
        helper.make_node("GlobalAveragePool", ["scores"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"]),
        helper.make_node("ArgMax", ["flat"], ["label"], axis=1, keepdims=0),
        helper.make_node("Cast", ["label"], ["answer"], to=F),
    ]
    label = import_onnx(model_of(nodes, [("answer", F)], initializers, 12, 16), inputs=dict(HINT))
    assert label["task"] == "regression"


# --- when the graph does not say ----------------------------------------------------------------------------


def test_the_labels_and_then_the_shape_say_where_the_classes_are_when_no_step_does():
    first, last = by_einsum("nchw,kc->nkhw"), by_einsum("nchw,kc->nhwk")
    names = [f"class {index}" for index in range(CLASSES)]
    # The one axis as long as the labels.
    assert import_onnx(first(12, 16), task="segmentation", inputs=dict(HINT), labels=names)["head"]["layout"] == "chw"
    assert import_onnx(last(12, 16), task="segmentation", inputs=dict(HINT), labels=names)["head"]["layout"] == "hwc"
    # Names in a NumPy array are a list of names like any other.
    array = import_onnx(last(12, 16), task="segmentation", inputs=dict(HINT), labels=np.array(names))
    assert array["head"]["layout"] == "hwc" and array["outputs"] == {"labels": names}
    # The one axis whose size differs from the two equal others.
    assert import_onnx(first(16, 16), task="segmentation", inputs=dict(HINT))["head"]["layout"] == "chw"
    assert import_onnx(last(16, 16), task="segmentation", inputs=dict(HINT))["head"]["layout"] == "hwc"


def test_a_16_by_16_picture_of_21_classes_kept_last_is_never_read_classes_first():
    """The wrong reading this work started from: 16 x 16 x 21 taken for 16 classes of a 16 x 21 picture."""
    for build in (by_einsum("nchw,kc->nhwk", 21), after_conv(("Transpose", {"perm": [0, 2, 3, 1]}, []), classes=21)):
        spec = import_onnx(build(16, 16), task="segmentation", inputs=dict(HINT))
        value = own_output(spec)
        assert value.shape == (1, 16, 16, 21) and spec["head"]["layout"] == "hwc"
        np.testing.assert_array_equal(mask_by("hwc", value), np.argmax(value[0], axis=-1))


@pytest.mark.parametrize(
    "equation, height, width, labels",
    [
        ("nchw,kc->nkhw", 12, 16, None),  # three different sizes
        ("nchw,kc->nhwk", 12, 16, None),
        ("nchw,kc->nkhw", 5, 5, None),  # three equal sizes
        ("nchw,kc->nhwk", 5, 12, 5),  # the labels fit two axes
        ("nchw,kc->nkhw", 12, 16, 4),  # the labels fit none
    ],
)
def test_a_class_axis_that_cannot_be_told_is_refused_with_how_to_say_it(equation, height, width, labels):
    model = by_einsum(equation)(height, width)
    names = [f"class {index}" for index in range(labels)] if labels else None
    with pytest.raises(OnnxImportError) as caught:
        import_onnx(model, task="segmentation", inputs=dict(HINT), labels=names)
    said = str(caught.value)
    assert caught.value.param == "mask_layout"  # the argument that settles it
    assert "can't tell which axis" in said and "mask_layout='chw'" in said and "head.layout" in said
    assert "does not fit" not in said  # nothing is said of a layout the caller never gave
    # Said, it is read as said, and the mask is the model's own.
    layout = "chw" if "nkhw" in equation else "hwc"
    spec = import_onnx(model, task="segmentation", inputs=dict(HINT), mask_layout=layout)
    value = own_output(spec)
    truth = np.argmax(value[0], axis=0 if layout == "chw" else -1)
    np.testing.assert_array_equal(mask_by(spec["head"]["layout"], value), truth)


def test_labels_that_do_not_count_the_classes_and_a_layout_that_does_not_fit_name_the_argument_at_fault():
    names = [f"class {index}" for index in range(4)]
    with pytest.raises(OnnxImportError, match="5 class channels and 4 labels") as caught:
        import_onnx(after_conv()(12, 16), task="segmentation", inputs=dict(HINT), labels=names)
    assert caught.value.param == "labels"
    with pytest.raises(OnnxImportError, match="does not fit mask_layout='indices'") as caught:
        import_onnx(after_conv()(12, 16), task="segmentation", inputs=dict(HINT), mask_layout="indices")
    assert caught.value.param == "mask_layout"


@pytest.mark.parametrize(
    "steps, why",
    [
        ((("Transpose", {"perm": [0, 2, 1, 3]}, []),), "neither first nor last"),
        ((("Reshape", {}, [("to", lambda h, w, c: [1, c * h, w], np.int64)]),), "folded into another axis"),
        ((("GlobalAveragePool", {}, []), ("Flatten", {}, [])), "no picture"),
    ],
)
def test_an_output_whose_classes_the_graph_puts_where_no_mask_keeps_them_is_refused_not_guessed(steps, why):
    """The shape of each of these could be read as a mask. The graph says it is not one."""
    with pytest.raises(OnnxImportError, match=why) as caught:
        import_onnx(after_conv(*steps)(5, 5), task="segmentation", inputs=dict(HINT))
    assert caught.value.param is None


def test_whole_numbers_behind_an_axis_of_one_are_not_guessed_to_be_class_numbers_or_scores():
    """[1, 1, 12, 12] of whole numbers that no argmax made: class numbers, or one channel of quantized scores."""
    value = np.zeros((1, 1, 12, 12), dtype=np.int64)
    with pytest.raises(OnnxImportError, match="can't tell which axis"):
        segmentation_head(value)
    assert segmentation_head(value, layout="indices")["layout"] == "indices"
    assert segmentation_head(np.zeros((1, 12, 12), dtype=np.int64))["layout"] == "indices"  # after the row axis
    assert segmentation_head(np.zeros((1, 12, 12), dtype=np.float32))["layout"] == "chw"  # one score a pixel
    assert segmentation_head(np.zeros((21, 32, 32), dtype=np.float32))["layout"] == "chw"  # no row axis
    assert segmentation_head(np.zeros((32, 32, 21), dtype=np.float32))["layout"] == "hwc"


# --- one score for each pixel -----------------------------------------------------------------------------------

SIGMOID = ("Sigmoid", {}, [])
LAST = ("Transpose", {"perm": [0, 2, 3, 1]}, [])
BOTH = [("axes", [0, 1], np.int64)]
TWICE = ("Resize", {"mode": "nearest"}, [("roi", [], np.float32), ("scales", [1, 1, 2, 2], np.float32)])
BATCH_NORM = ("BatchNormalization", {}, [("s", [1.5], np.float32), ("b", [0.25], np.float32),
                                         ("m", [0.1], np.float32), ("v", [2.0], np.float32)])
ONE_SCORE = {
    # name: (the steps after a convolution of one channel, layout, threshold, what the scores are, picture axes)
    "raw [1, 1, H, W]": ((), "chw", 0.0, "raw", (2, 3)),
    "raw [1, H, W]": ((("Squeeze", {}, AXES),), "chw", 0.0, "raw", (1, 2)),
    "raw [H, W]": ((("Squeeze", {}, BOTH),), "chw", 0.0, "raw", (0, 1)),
    "raw [H, W, 1]": ((LAST, ("Squeeze", {}, ROW)), "hwc", 0.0, "raw", (0, 1)),
    "sigmoid [1, 1, H, W]": ((SIGMOID,), "chw", 0.5, "probability", (2, 3)),
    "sigmoid [1, H, W]": ((SIGMOID, ("Squeeze", {}, AXES)), "chw", 0.5, "probability", (1, 2)),
    "sigmoid [H, W]": ((SIGMOID, ("Squeeze", {}, BOTH)), "chw", 0.5, "probability", (0, 1)),
    "sigmoid [H, W, 1]": ((SIGMOID, LAST, ("Squeeze", {}, ROW)), "hwc", 0.5, "probability", (0, 1)),
    "raw, a bias added": ((("Add", {}, [("b", 0.3, np.float32)]),), "chw", 0.0, "raw", (2, 3)),
    "raw, batch norm": ((BATCH_NORM,), "chw", 0.0, "raw", (2, 3)),
    "raw, resized, [1, H, W]": ((TWICE, ("Squeeze", {}, AXES)), "chw", 0.0, "raw", (1, 2)),
    "sigmoid, resized": ((SIGMOID, TWICE), "chw", 0.5, "probability", (2, 3)),
    "hard sigmoid, pooled": ((("HardSigmoid", {}, []), ("MaxPool", {"kernel_shape": [1, 1]}, [])), "chw", 0.5,
                             "probability", (2, 3)),
    # After a step that says neither, the model's own values on its check rows say.
    "raw, clipped to -3..3": ((("Clip", {}, [("low", -3.0, np.float32), ("high", 3.0, np.float32)]),), "chw", 0.0,
                              "raw", (2, 3)),
    "sigmoid, clipped, [H, W]": (
        (SIGMOID, ("Clip", {}, [("low", 0.001, np.float32), ("high", 0.999, np.float32)]), ("Squeeze", {}, BOTH)),
        "chw", 0.5, "probability", (0, 1)),
    # PyTorch writes .squeeze(0) of rows the file leaves open as a branch: the scores are followed through it.
    "raw [H, W], as PyTorch squeezes it": ((("Squeeze", {}, AXES), ROWS_SQUEEZED), "chw", 0.0, "raw", (0, 1)),
    "sigmoid [H, W], as PyTorch squeezes it": ((SIGMOID, ("Squeeze", {}, AXES), ROWS_SQUEEZED), "chw", 0.5,
                                               "probability", (0, 1)),
    "raw [H, W, 1], as PyTorch squeezes it": ((LAST, ROWS_SQUEEZED), "hwc", 0.0, "raw", (0, 1)),
}  # fmt: skip


def half(value, picture, raw):
    """One score a pixel cut where the model's own probability is one half: the mask every reading must give."""
    grid = np.asarray(value, dtype=float).reshape(value.shape[picture[0]], value.shape[picture[1]])
    return ((1.0 / (1.0 + np.exp(-grid)) if raw else grid) >= 0.5).astype(int)


@pytest.mark.parametrize("name", sorted(ONE_SCORE))
@pytest.mark.parametrize("height, width", [(12, 16), (5, 5), (1, 7)])
def test_one_score_a_pixel_is_cut_where_the_models_own_probability_is_one_half(name, height, width):
    """Raw scores (no sigmoid in the file) were cut at 0.5, which is the model's probability 0.62, and a height
    and a width of them with no axis of one were rounded as if they were class numbers."""
    steps, layout, threshold, kind, picture = ONE_SCORE[name]
    spec = import_onnx(after_conv(*steps, classes=1)(height, width), task="segmentation", inputs=dict(HINT))
    head = spec["head"]
    assert (head["layout"], head["threshold"], head["score_kind"]) == (layout, threshold, kind)
    shape = spec["graph"]["values"][head["output"]]["shape"]
    sides = set()
    for row in spec["check"]["outputs"]:
        value = np.asarray(row).reshape(shape)
        truth = half(value, picture, kind == "raw")
        np.testing.assert_array_equal(mask_by(layout, value, head["threshold"]), truth)
        sides.update(int(item) for item in np.unique(truth))
    assert sides == {0, 1}  # scores on both sides of the cut


def test_a_softmax_of_two_classes_cut_down_to_one_is_a_probability_and_its_logarithm_is_cut_at_that_of_one_half():
    one = [("from", [1], np.int64), ("to", [2], np.int64), ("axes", [1], np.int64)]
    spec = import_onnx(after_conv(("Softmax", {"axis": 1}, []), ("Slice", {}, one), classes=2)(12, 16),
                       task="segmentation", inputs=dict(HINT))
    assert (spec["head"]["threshold"], spec["head"]["score_kind"]) == (0.5, "probability")
    logs = import_onnx(after_conv(("LogSoftmax", {"axis": 1}, []), ("Slice", {}, one), classes=2)(12, 16),
                       task="segmentation", inputs=dict(HINT))
    assert logs["head"]["threshold"] == pytest.approx(np.log(0.5)) and logs["head"]["score_kind"] == "raw"
    likely, logged = own_output(spec), own_output(logs)
    np.testing.assert_array_equal(mask_by("chw", logged, logs["head"]["threshold"]), mask_by("chw", likely, 0.5))
    assert 0 < mask_by("chw", likely, 0.5).mean() < 1


def test_what_the_scores_are_is_read_off_the_steps_that_make_them_or_left_to_their_values():
    def step(op, inputs, output, **attrs):
        return {"op": op, "domain": "", "inputs": list(inputs), "outputs": [output], "attrs": attrs}

    weights = {
        "w": np.zeros((2, 2), dtype=np.float32), "b": np.zeros(2, dtype=np.float32),
        "by": np.array(255.0, dtype=np.float32), "minus": np.array(-1.0, dtype=np.float32),
        "low": np.array(0.001, dtype=np.float32), "high": np.array(0.999, dtype=np.float32),
        "nought": np.array(0.0, dtype=np.float32), "six": np.array(6.0, dtype=np.float32),
        "most": np.array(100.0, dtype=np.float32),
    }  # fmt: skip
    conv, squash = step("Conv", ["x", "k"], "a"), step("Sigmoid", ["a"], "p")
    for nodes, output, found in (
        ([conv], "a", (0.0, "raw")),
        ([conv, squash], "p", (0.5, "probability")),
        ([conv, squash, step("Resize", ["p"], "r"), step("Squeeze", ["r"], "s"), step("Cast", ["s"], "c", to=11)],
         "c", (0.5, "probability")),
        ([step("MatMul", ["x", "w"], "m"), step("Add", ["m", "b"], "y")], "y", (0.0, "raw")),  # a linear layer
        ([conv, step("Conv", ["x", "k2"], "a2"), step("Add", ["a", "a2"], "y")], "y", (0.0, "raw")),  # a skip sum
        ([conv, step("BatchNormalization", ["a", "s", "b", "m", "v"], "n")], "n", (0.0, "raw")),
        # A sum of probabilities is neither, a cast to whole numbers holds no score, and other steps say nothing.
        ([conv, squash, step("Conv", ["x", "k2"], "a2"), step("Sigmoid", ["a2"], "p2"), step("Add", ["p", "p2"], "y")],
         "y", None),
        ([conv, squash, step("Cast", ["p"], "c", to=7)], "c", None),
        ([step("Relu", ["x"], "r")], "r", None),
        ([conv, step("Tanh", ["a"], "t")], "t", None),
        # A scale by one constant above nought moves the cut with it, and scaled probabilities are no longer
        # probabilities. By a constant that is not above nought, or by another tensor, nothing is certain.
        ([conv, squash, step("Mul", ["p", "by"], "y")], "y", (127.5, None)),
        ([conv, squash, step("Mul", ["by", "p"], "y")], "y", (127.5, None)),
        ([conv, squash, step("Div", ["p", "by"], "y")], "y", (0.5 / 255.0, None)),
        ([conv, step("Mul", ["a", "by"], "y")], "y", (0.0, "raw")),
        ([conv, squash, step("Mul", ["p", "minus"], "y")], "y", None),
        ([conv, squash, step("Mul", ["p", "a"], "y")], "y", None),
        ([conv, squash, step("Div", ["by", "p"], "y")], "y", None),
        # A Relu or a Clip leaves the cut where it is when the cut lies inside what the step keeps.
        ([conv, squash, step("Relu", ["p"], "r")], "r", (0.5, "probability")),
        ([conv, squash, step("Clip", ["p", "low", "high"], "c")], "c", (0.5, "probability")),
        ([conv, step("Clip", ["a"], "c", min=-6.0, max=6.0)], "c", (0.0, "raw")),
        ([conv, squash, step("Mul", ["p", "by"], "y"), step("Clip", ["y", "nought", ""], "c")], "c", (127.5, None)),
        ([conv, step("Clip", ["a", "x"], "c")], "c", None),  # an end that is no constant
    ):
        assert onnx_graph._score_cut(nodes, output, weights) == found, (output, [node["op"] for node in nodes])
    # Where the step leaves no place for the cut, no reading of the values could find it either: refused.
    for nodes, output, kept in (
        ([conv, step("Relu", ["a"], "r")], "r", "raw scores pass through a step that keeps nothing below 0"),
        ([conv, step("Clip", ["a", "nought", "six"], "c")], "c", "keeps only what lies from 0 to 6"),
        ([conv, squash, step("Clip", ["p", "", "low"], "c")], "c", "keeps nothing above 0.001"),
        ([conv, squash, step("Mul", ["p", "by"], "y"), step("Clip", ["y", "nought", "most"], "c")], "c",
         "one half at 127.5"),
    ):
        with pytest.raises(OnnxImportError, match=kept) as caught:
            onnx_graph._score_cut(nodes, output, weights)
        assert caught.value.param == "threshold" and "threshold=" in str(caught.value)
    # Left to the values, on every row: below nought somewhere, raw scores; all from 0 to 1, probabilities; never
    # below nought and above 1 somewhere, neither, and nothing says where to cut them.
    assert onnx_graph._cut_by_values([np.array([0.0, 0.4, 1.0])]) == (0.5, "probability")
    assert onnx_graph._cut_by_values([np.array([0.2, 0.9]), np.array([0.5, -1.5])]) == (0.0, "raw")
    assert onnx_graph._cut_by_values([np.array([0.2, -0.1])]) == (0.0, "raw")
    assert onnx_graph._cut_by_values([np.array([0.2, 1.5]), np.array([0.5, -0.1])]) == (0.0, "raw")
    with pytest.raises(OnnxImportError, match="on the 2 rows tried they run from 0.2 to 1.5: never below 0") as caught:
        onnx_graph._cut_by_values([np.array([0.2, 0.9]), np.array([0.5, 1.5])])
    assert caught.value.param == "threshold"


def test_a_height_and_a_width_no_step_explains_is_class_numbers_by_its_type_and_never_by_one_rows_values():
    """Floats that happen to be whole on the one row looked at (all 0.0 after a Relu) were taken for class
    numbers, so a later score of 3.7 would have been reported as class 4."""
    assert segmentation_head(np.array([[0, 2], [1, 1]])) == {"kind": "segmentation", "layout": "indices"}
    for value in (np.array([[0.0, 2.0], [1.0, 1.0]]), np.zeros((4, 7), dtype=np.float32)):
        with pytest.raises(OnnxImportError, match="every value on the 1 row tried is a whole number") as caught:
            segmentation_head(value)
        assert caught.value.param == "mask_layout" and "mask_layout='indices'" in str(caught.value)
        assert segmentation_head(value, layout="indices") == {"kind": "segmentation", "layout": "indices"}
        assert segmentation_head(value, layout="chw", threshold=0.5)["layout"] == "chw"
    scores = segmentation_head(np.array([[0.2, 1.5], [2.0, -1.0]]))
    assert scores == {"kind": "segmentation", "layout": "chw", "threshold": 0.0, "score_kind": "raw"}
    likely = segmentation_head(np.array([[0.2, 0.5], [0.75, 0.0]]))
    assert likely == {"kind": "segmentation", "layout": "chw", "threshold": 0.5, "score_kind": "probability"}
    # Said, a height and a width of scores is one channel of them; class numbers are still class numbers.
    assert segmentation_head(np.array([[0.2, 1.5], [2.0, -1.0]]), layout="hwc")["layout"] == "hwc"
    assert "threshold" not in segmentation_head(np.array([[0.2, 1.5], [2.0, -1.0]]), layout="indices")


def test_the_callers_threshold_is_where_one_score_a_pixel_is_cut_and_is_refused_where_nothing_is_cut():
    raw = after_conv(("Squeeze", {}, BOTH), classes=1)(12, 16)
    spec = import_onnx(raw, task="segmentation", inputs=dict(HINT), threshold=1.25)
    assert spec["head"] == {**spec["head"], "layout": "chw", "threshold": 1.25} and "score_kind" not in spec["head"]
    whole = import_onnx(raw, task="segmentation", inputs=dict(HINT), threshold=np.int64(2))
    assert whole["head"]["threshold"] == 2.0 and isinstance(whole["head"]["threshold"], float)
    for build, options in (
        (after_conv(), {"task": "segmentation"}),  # five classes: nothing to cut
        (lambda height, width: numbers_model("argmax", height, width), {"task": "segmentation"}),  # class numbers
        (after_conv(), {"task": "regression"}),
        (after_conv(("Squeeze", {}, BOTH), classes=1), {"task": "segmentation", "threshold": "high"}),
        (after_conv(("Squeeze", {}, BOTH), classes=1), {"task": "segmentation", "threshold": True}),
        (after_conv(("Squeeze", {}, BOTH), classes=1), {"task": "segmentation", "threshold": float("inf")}),
    ):
        with pytest.raises(OnnxImportError) as caught:
            import_onnx(build(12, 16), inputs=dict(HINT), **{"threshold": 0.3, **options})
        assert caught.value.param == "threshold", options


# --- several outputs --------------------------------------------------------------------------------------------


def pair(kind, height=12, width=16):
    rng = np.random.default_rng(7)
    main, initializers = conv("out", CLASSES, rng)
    pick = helper.make_node("ArgMax", ["out"], ["mask"], axis=1, keepdims=0)
    pool = [helper.make_node("GlobalAveragePool", ["out"], ["pooled"]), helper.make_node("Flatten", ["pooled"], ["label"])]
    if kind == "out and aux":
        aux, more = conv("aux", CLASSES, rng)
        return model_of([main, aux], [("out", F), ("aux", F)], [*initializers, *more], height, width)
    if kind == "scores and mask":
        return model_of([main, pick], [("out", F), ("mask", I64)], initializers, height, width)
    if kind == "mask and scores":
        return model_of([main, pick], [("mask", I64), ("out", F)], initializers, height, width)
    if kind == "label and scores":
        return model_of([main, *pool], [("label", F), ("out", F)], initializers, height, width)
    if kind == "label and mask":
        return model_of([main, pick, *pool], [("label", F), ("mask", I64)], initializers, height, width)
    assert kind == "two labels"
    return model_of([main, *pool, helper.make_node("Neg", ["label"], ["other"])], [("label", F), ("other", F)],
                    initializers, height, width)


@pytest.mark.parametrize(
    "kind, read, layout",
    [
        ("out and aux", "out", "chw"),  # torchvision's FCN and DeepLab
        ("scores and mask", "out", "chw"),
        ("mask and scores", "out", "chw"),  # floats before whole numbers, whichever comes first
        ("label and scores", "out", "chw"),  # a row of scores cannot hold a mask
        ("label and mask", "mask", "indices"),  # whole numbers when no floats can hold one
    ],
)
def test_a_segmenter_with_several_outputs_is_read_from_the_first_that_can_hold_a_mask(kind, read, layout):
    """Every one of these was refused with "Choose the segmentation output with output=."."""
    spec = import_onnx(pair(kind), task="segmentation", inputs=dict(HINT))
    assert spec["graph"]["values"][spec["head"]["output"]]["source"] == read
    assert spec["head"]["layout"] == layout and len(spec["graph"]["outputs"]) == 1
    chosen = import_onnx(pair("out and aux"), task="segmentation", inputs=dict(HINT), output="aux")
    assert chosen["graph"]["values"][chosen["head"]["output"]]["source"] == "aux"


def test_outputs_none_of_which_can_hold_a_mask_are_refused_and_named():
    with pytest.raises(OnnxImportError) as caught:
        import_onnx(pair("two labels"), task="segmentation", inputs=dict(HINT))
    assert caught.value.param == "output" and "'label' (1 x 5)" in str(caught.value) and "output=" in str(caught.value)
    assert onnx_graph.can_hold_a_mask((1, 5, 12, 16)) and onnx_graph.can_hold_a_mask((12, 16))
    assert onnx_graph.can_hold_a_mask((1, 12, 16)) and onnx_graph.can_hold_a_mask((1, 12, 16, 1))
    assert not onnx_graph.can_hold_a_mask((1, 5)) and not onnx_graph.can_hold_a_mask((2, 3, 5, 12, 16))
    assert not onnx_graph.can_hold_a_mask(None) and not onnx_graph.can_hold_a_mask(())


# --- a reference run copies the model once ----------------------------------------------------------------------


def test_a_reference_run_copies_a_model_with_inputs_left_out_under_empty_names_once(monkeypatch):
    """Stripping the empty names made one copy and asking ONNX Runtime for every tensor a second."""
    pytest.importorskip("onnxruntime")
    real = onnx_graph._onnx()
    made = []

    class Counting:
        def __getattr__(self, name):
            return getattr(real, name)

        def ModelProto(self):  # noqa: N802 - the name onnx gives it
            made.append(1)
            return real.ModelProto()

    clip = helper.make_node("Clip", ["x", "low", ""], ["y"])  # no upper bound, written as an empty name
    model = model_of([clip], [("y", F)], [const("low", 0.25)], 2, 3)
    feeds = {"x": np.linspace(0, 1, 18, dtype=np.float32).reshape(1, 3, 2, 3)}
    monkeypatch.setattr(onnx_graph, "_onnx", lambda: Counting())
    found = onnx_graph.run_reference(model, feeds)
    np.testing.assert_allclose(found["y"], np.maximum(feeds["x"], 0.25))
    assert len(made) == 1
    assert list(model.graph.node[0].input) == ["x", "low", ""] and len(model.graph.output) == 1  # the caller's own


# --- byte-pair merges that name tokens the vocabulary lacks -------------------------------------------------------


def test_merges_that_name_a_piece_the_vocabulary_lacks_are_refused_as_hugging_face_refuses_them():
    """Read with them, every word came back as unknown tokens: 'hello' as five of them."""
    described = {
        "type": "bpe",
        "byte_level": False,
        "vocab": ["<unk>", "he", "llo", "hello", "ll"],
        "merges": [["h", "e"], ["l", "l"], ["ll", "o"], ["he", "llo"]],
        "unk": "<unk>",
    }
    with pytest.raises(TokenizerError, match="'h', which is not a token of the vocabulary"):
        build_tokenizer(described)
    with pytest.raises(StructureError, match="needs 'h'") as caught:
        tokenizer_spec(described)
    assert caught.value.code == "invalid_tokenizer" and caught.value.param == "tokenizer"
    # The piece a merge makes has to be a token as well, and the first piece that is missing is the one named.
    for vocab, missing in ((["<unk>", "b", "ab"], "'a'"), (["<unk>", "a", "ab"], "'b'"), (["<unk>", "a", "b"], "'ab'")):
        with pytest.raises(TokenizerError, match=f"needs {missing}"):
            build_tokenizer({"type": "bpe", "byte_level": False, "vocab": vocab, "merges": ["a b"], "unk": "<unk>"})
    # A token added beside the vocabulary of pieces is not one a merge can join.
    added = {
        "type": "bpe", "byte_level": False, "vocab": ["<unk>", "a", "b", "ab"], "merges": ["a b"], "unk": "<unk>",
        "added_tokens": [{"content": "ab", "piece": False}],
    }  # fmt: skip
    with pytest.raises(TokenizerError, match="needs 'ab'"):
        build_tokenizer(added)


def test_a_byte_pair_tokenizer_written_by_hand_with_every_piece_in_its_vocabulary_still_reads_text():
    vocab = ["<unk>", "h", "e", "l", "o", "he", "ll", "llo", "hello", "w</w>", "o</w>", "lo</w>"]
    plain = build_tokenizer({
        "type": "bpe", "byte_level": False, "vocab": vocab[:9], "unk": "<unk>",
        "merges": [["h", "e"], ["l", "l"], ["ll", "o"], ["he", "llo"]],
    })  # fmt: skip
    assert plain.encode("hello hell x") == [8, 5, 6, 0]
    marked = build_tokenizer({
        "type": "bpe", "byte_level": False, "vocab": vocab, "unk": "<unk>", "end_of_word": "</w>",
        "merges": [["l", "o</w>"], ["h", "e"]],
    })  # fmt: skip
    assert [marked.tokens[index] for index in marked.encode("hello")] == ["he", "l", "lo</w>"]


# --- what an encoder-decoder has written so far -------------------------------------------------------------------

VOCAB, READS, ROOM = 12, 5, 4
WRITTEN = "decoder_input_ids"


def writer(masked=False, floats=False, unread=False):
    """An encoder that pools its text, and a decoder that sees each token it has written, where it stands, the
    tokens before it and, faintly, what stands after: its scores differ at every position, and with whatever fills
    the places it has not written yet.

    ``masked`` adds the decoder's mask as an input, ``floats`` takes what it has written as floats, and ``unread``
    leaves what it has written out of the scores altogether.
    """
    rng = np.random.default_rng(9)
    before = np.tril(np.ones((ROOM, ROOM))) / np.arange(1, ROOM + 1)[:, None] + 0.05
    tensors = [
        const("E", rng.normal(size=(VOCAB, 6))),
        const("D", rng.normal(size=(VOCAB, 6))),
        const("P", rng.normal(size=(ROOM, 6))),
        const("BEFORE", before),
        const("OUT", rng.normal(size=(6, VOCAB)) * 2.0),
    ]
    inputs = [
        helper.make_tensor_value_info("input_ids", I64, ["batch", READS]),
        helper.make_tensor_value_info(WRITTEN, F if floats else I64, ["batch", ROOM]),
    ]
    nodes = [
        helper.make_node("Gather", ["E", "input_ids"], ["read"], name="encoder/embed"),
        helper.make_node("ReduceMean", ["read"], ["memory"], axes=[1], keepdims=1, name="encoder/pool"),
    ]
    if floats:
        nodes.append(helper.make_node("Cast", [WRITTEN], ["whole"], to=I64, name="decoder/whole"))
    nodes += [
        helper.make_node("Gather", ["D", "whole" if floats else WRITTEN], ["written"], name="decoder/embed"),
        helper.make_node("Add", ["written", "P"], ["placed"], name="decoder/place"),
    ]
    placed = "placed"
    if masked:
        inputs.append(helper.make_tensor_value_info("decoder_attention_mask", I64, ["batch", ROOM]))
        tensors.append(const("last", [2], np.int64))
        nodes += [
            helper.make_node("Cast", ["decoder_attention_mask"], ["kept"], to=F, name="decoder/kept"),
            helper.make_node("Unsqueeze", ["kept", "last"], ["kept_each"], name="decoder/kept_each"),
            helper.make_node("Mul", ["placed", "kept_each"], ["shown"], name="decoder/shown"),
        ]
        placed = "shown"
    nodes += [
        helper.make_node("MatMul", ["BEFORE", placed], ["seen"], name="decoder/before"),
        helper.make_node("Add", ["seen", "memory"], ["state"], name="decoder/attend"),
        helper.make_node("MatMul", ["memory" if unread else "state", "OUT"], ["logits"], name="head"),
    ]
    scores = helper.make_tensor_value_info("logits", F, ["batch", 1 if unread else ROOM, VOCAB])
    made = helper.make_model(
        helper.make_graph(nodes, "writer", inputs, [scores], tensors), opset_imports=[helper.make_opsetid("", 13)]
    )
    made.ir_version = 8
    return made


def own_next(model, text, written, pad=0):
    """The model's own scores for the token after ``written``, as probabilities, by ONNX's reference evaluator:
    the places not written yet hold ``pad``, and its mask (when it reads one) covers what is written."""
    from onnx.reference import ReferenceEvaluator

    ids = np.full((1, ROOM), pad, dtype=np.int64)
    ids[0, : len(written)] = written
    feeds = {"input_ids": np.asarray([text], dtype=np.int64), WRITTEN: ids}
    if any(item.name == "decoder_attention_mask" for item in model.graph.input):
        feeds["decoder_attention_mask"] = (np.arange(ROOM) < len(written)).astype(np.int64)[None]
    scores = ReferenceEvaluator(model).run(None, feeds)[0][0, len(written) - 1].astype(float)
    shifted = np.exp(scores - scores.max())
    return shifted / shifted.sum()


def bound(spec):
    """Each input's ``from`` by the input's name in the file, every one in a form the API reads."""
    found = {}
    for item in spec["graph"]["inputs"]:
        source = item["from"]
        assert "generated" not in source and any(key in source for key in ("feature", "features", "value", "zeros"))
        found[str(item["name"])[len("in.") :]] = source
    return found


def checked_at_the_first_token(model, spec, feature, start, pad):
    """The check gives the decoder what a prediction gives it (its first token and nothing after), and holds the
    model's own scores for the token after that one."""
    check = spec["check"]
    assert check["inputs"][feature] == [[float(start)] + [-1.0] * (ROOM - 1)] * len(check["outputs"])
    assert len({tuple(row) for row in check["inputs"]["text"]}) > 1
    for text, scores in zip(check["inputs"]["text"], check["outputs"]):
        np.testing.assert_allclose(scores, own_next(model, [int(item) for item in text], [start], pad), atol=1e-6)


def test_an_input_hinted_generated_is_the_feature_the_decoder_writes_into():
    """It was written as ``"from": {"generated": true}`` with no feature, a structure the API refuses."""
    model = writer()
    spec = import_onnx(model, inputs={WRITTEN: {"generated": True, "pad_id": 0}})
    assert spec["task"] == "sequence_to_sequence"
    assert bound(spec) == {
        "input_ids": {"feature": "text", "part": "ids"},
        WRITTEN: {"feature": "decoder", "part": "ids", "pad_id": 0},
    }
    assert [feature["name"] for feature in spec["features"]] == ["text", "decoder"]
    assert spec["features"][1] == {
        "name": "decoder", "type": "tokens", "max_length": ROOM, "generated": True, "baseline": [0],
    }  # fmt: skip
    assert spec["decoder"] == {"feature": "decoder", "start_token": 0, "pad_token": 0, "max_length": ROOM}
    checked_at_the_first_token(model, spec, "decoder", 0, 0)


@pytest.mark.parametrize(
    "hint, sample, start, pad, decoder",
    [
        ({"generated": True}, None, 0, 0, {"feature": "decoder", "start_token": 0, "max_length": ROOM}),
        # Its first token is the first of the example it was given, as the client's converter reads it.
        ({"generated": True}, [[7, 0, 0, 0]], 7, 0, {"feature": "decoder", "start_token": 7, "max_length": ROOM}),
        ({"generated": True, "pad_id": 9.0}, [[7.0, 9.0, 9.0, 9.0]], 7, 9,
         {"feature": "decoder", "start_token": 7, "pad_token": 9, "max_length": ROOM}),
        ({"generated": True, "feature": "answer", "pad_id": 9, "start_token": 2, "eos_token": 1}, [[7, 0, 0, 0]], 2, 9,
         {"feature": "answer", "start_token": 2, "eos_token": 1, "pad_token": 9, "max_length": ROOM}),
    ],
)  # fmt: skip
def test_the_hint_says_where_the_decoder_starts_what_fills_the_rest_and_where_it_ends(hint, sample, start, pad, decoder):
    model = writer()
    given = dict(hint)
    spec = import_onnx(model, inputs={WRITTEN: given}, samples=None if sample is None else {WRITTEN: sample})
    assert given == hint  # the caller's own hint is left as it was given
    name = decoder["feature"]
    assert bound(spec)[WRITTEN] == {"feature": name, "part": "ids", "pad_id": pad}
    assert spec["decoder"] == decoder and list(spec["decoder"]) == list(decoder)
    assert spec["features"][1] == {
        "name": name, "type": "tokens", "max_length": ROOM, "generated": True, "baseline": [start],
    }  # fmt: skip
    # The scores depend on what fills the places not written yet, so the check also holds the padding it was told.
    checked_at_the_first_token(model, spec, name, start, pad)


def test_the_mask_that_comes_with_the_decoders_ids_is_worked_out_from_what_it_has_written():
    """Left to its name alone it was bound to the encoder's text, a mask of another length."""
    model = writer(masked=True)
    spec = import_onnx(model, inputs={WRITTEN: {"generated": True}})
    assert bound(spec)["decoder_attention_mask"] == {"feature": "decoder", "part": "mask"}
    checked_at_the_first_token(model, spec, "decoder", 0, 0)
    # What the caller says of the mask itself stands.
    said = import_onnx(model, inputs={WRITTEN: {"generated": True}, "decoder_attention_mask": {"value": 1}})
    assert bound(said)["decoder_attention_mask"] == {"value": 1}


@pytest.mark.parametrize(
    "options, said",
    [
        ({"inputs": {WRITTEN: {"generated": True, "value": 0}}}, "also gives it 'value'"),
        ({"inputs": {WRITTEN: {"generated": True, "zeros": True}}}, "also gives it 'zeros'"),
        ({"inputs": {WRITTEN: {"generated": True, "features": ["a", "b", "c", "d"]}}}, "also gives it 'features'"),
        ({"inputs": {WRITTEN: {"generated": True, "layout": "nchw"}}}, "also gives it 'layout'"),
        ({"inputs": {WRITTEN: {"generated": True, "part": "mask"}}}, "also gives it 'part'"),
        ({"inputs": {WRITTEN: {"generated": True, "pad_id": -1}}}, "a pad_id that is not a token id"),
        ({"inputs": {WRITTEN: {"generated": True, "pad_id": 1.5}}}, "a pad_id that is not a token id"),
        ({"inputs": {WRITTEN: {"generated": True, "pad_id": True}}}, "a pad_id that is not a token id"),
        ({"inputs": {WRITTEN: {"generated": True, "start_token": "2"}}}, "a start_token that is not a token id"),
        ({"inputs": {WRITTEN: {"generated": True, "eos_token": 2**31}}}, "an? eos_token that is not a token id"),
        ({"inputs": {WRITTEN: {"generated": True, "feature": ""}}}, "a feature that is not a name"),
        ({"inputs": {WRITTEN: {"generated": True, "feature": 3}}}, "a feature that is not a name"),
        ({"inputs": {WRITTEN: {"generated": True}}, "samples": {WRITTEN: [[-100, 0, 0, 0]]}}, "its sample starts with"),
        ({"inputs": {WRITTEN: {"generated": True}, "input_ids": {"generated": True}}}, "a model has one decoder"),
        ({"inputs": {WRITTEN: {"generated": True}}, "task": "text_generation"}, "not text_generation"),
        ({"inputs": {WRITTEN: {"generated": True}}, "task": "regression"}, "not regression"),
        ({"inputs": {WRITTEN: {"generated": True, "feature": "text"}}}, "which another input reads as well"),
    ],
)
def test_a_generated_hint_that_cannot_be_written_as_a_feature_is_refused_naming_inputs(options, said):
    """Never a structure the API would refuse, and never a reading of another kind made without a word."""
    with pytest.raises(OnnxImportError, match=said) as caught:
        import_onnx(writer(), **options)
    assert caught.value.param == "inputs"


def test_what_a_decoder_has_written_is_whole_numbers_that_its_scores_depend_on():
    with pytest.raises(OnnxImportError, match="holds float32 values: what a decoder has written is token ids") as caught:
        import_onnx(writer(floats=True), inputs={WRITTEN: {"generated": True}})
    assert caught.value.param == "inputs"
    with pytest.raises(OnnxImportError, match="the model's output does not depend on it") as caught:
        import_onnx(writer(unread=True), inputs={WRITTEN: {"generated": True}})
    assert caught.value.param == "inputs"


def test_a_decoder_given_as_a_feature_is_checked_at_the_last_token_it_holds():
    """The importer's own check held the scores after the first token for a row of four, which is not where the
    API reads them: a structure that was right did not pass its own check."""
    model = writer()
    as_feature = {WRITTEN: {"feature": "decoder", "part": "ids", "pad_id": 0, "type": "tokens"}}
    spec = import_onnx(model, task="sequence_to_sequence", inputs=as_feature, samples={WRITTEN: [[4, 6, 1, 3]]})
    check = spec["check"]
    assert check["inputs"]["decoder"][0] == [4.0, 6.0, 1.0, 3.0]
    for text, written, scores in zip(check["inputs"]["text"], check["inputs"]["decoder"], check["outputs"]):
        assert min(written) >= 0  # four tokens: the scores are those after the fourth
        own = own_next(model, [int(item) for item in text], [int(item) for item in written])
        np.testing.assert_allclose(scores, own, atol=1e-6)


def test_the_clients_from_onnx_writes_the_same_form_and_refuses_as_a_structure_error():
    from kaimei import graphs
    from kaimei.structure import finalize_spec

    raw = graphs.from_onnx(writer(), inputs={WRITTEN: {"generated": True, "pad_id": 0, "eos_token": 1}})
    spec = finalize_spec(raw).spec
    assert bound(spec)[WRITTEN] == {"feature": "decoder", "part": "ids", "pad_id": 0}
    assert spec["features"][1] == {
        "name": "decoder", "type": "tokens", "max_length": ROOM, "generated": True, "baseline": [0],
    }  # fmt: skip
    assert spec["decoder"] == {
        "feature": "decoder", "start_token": 0, "eos_token": 1, "pad_token": 0, "max_length": ROOM,
    }  # fmt: skip
    assert spec["task"] == "sequence_to_sequence" and not spec.get("approximate")
    # The client's own check: what the decoder has written in each row, and the file's scores after the last of it
    # (a place with no token, −1, is given the padding token, as the API gives it).
    model, check = writer(), spec["check"]
    for text, written, scores in zip(check["inputs"]["text"], check["inputs"]["decoder"], check["outputs"]):
        kept = [int(item) for item in written if item >= 0]
        own = own_next(model, [max(int(item), 0) for item in text], kept)
        np.testing.assert_allclose(scores, own, atol=1e-5)
    with pytest.raises(StructureError, match="what a decoder has written is token ids") as caught:
        graphs.from_onnx(writer(floats=True), inputs={WRITTEN: {"generated": True}})
    assert caught.value.param == "inputs"


def test_the_importer_writes_for_a_generated_input_what_the_converter_writes_for_a_decoder():
    """One module, read by the client's converter (which finds the decoder itself) and from the ONNX file the
    converter made, its decoder's input hinted generated: the same structure, key for key."""
    torch = pytest.importorskip("torch")
    from kaimei import graphs
    from kaimei.structure import extract_structure, finalize_spec

    class Writer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.read = torch.nn.Embedding(VOCAB, 6)
            self.write = torch.nn.Embedding(VOCAB, 6)
            self.place = torch.nn.Parameter(torch.randn(ROOM, 6))
            self.head = torch.nn.Linear(6, VOCAB)
            ahead = torch.tril(torch.ones(ROOM, ROOM)) / torch.arange(1, ROOM + 1).unsqueeze(1)
            self.register_buffer("before", ahead)

        def forward(self, input_ids, decoder_input_ids):
            memory = self.read(input_ids).mean(dim=1, keepdim=True)
            seen = torch.matmul(self.before, self.write(decoder_input_ids) + self.place)
            return self.head(seen + memory)

    def steps(spec):
        # Where a step lies in the module is the converter's to say (it drops the wrapper it exported through).
        return [{key: value for key, value in node.items() if key != "path"} for node in spec["graph"]["nodes"]]

    for start in (0, 7):
        sample = {
            "input_ids": torch.tensor([[3, 5, 7, 2, 1]]),
            "decoder_input_ids": torch.tensor([[start, 0, 0, 0]]),
        }
        done = extract_structure(Writer().eval(), sample=sample, task="sequence_to_sequence", via="onnx")
        converted = done.spec
        raw = graphs.from_onnx(
            done.onnx,
            inputs={"decoder_input_ids": {"generated": True}},
            samples={key: value.numpy() for key, value in sample.items()},
        )
        imported = finalize_spec(raw).spec
        assert converted["decoder"] == {"feature": "decoder", "start_token": start, "max_length": ROOM}
        for key in ("task", "decoder", "features", "feature_names", "head"):
            assert imported[key] == converted[key], key
        assert list(imported["decoder"]) == list(converted["decoder"])
        for key in ("inputs", "outputs", "opsets", "values"):
            assert imported["graph"][key] == converted["graph"][key], key
        assert sorted(imported["graph"]["constants"]) == sorted(converted["graph"]["constants"])
        assert steps(imported) == steps(converted)
        assert bound(imported)["decoder_input_ids"] == {"feature": "decoder", "part": "ids", "pad_id": 0}


# --- round 4 ------------------------------------------------------------------------------------------------------

SECOND = [("from", [1], np.int64), ("to", [2], np.int64), ("axes", [1], np.int64)]


def split_and_squeezed(height, width):
    """Two classes split apart, and the second one's axis squeezed away."""
    rng = np.random.default_rng(7)
    node, initializers = conv("scores", 2, rng)
    nodes = [
        node,
        helper.make_node("Split", ["scores"], ["bg", "fg"], axis=1),
        helper.make_node("Squeeze", ["fg", "axes"], ["out"]),
    ]
    return model_of(nodes, [("out", F)], [*initializers, const("axes", [1], np.int64)], height, width)


PICKED = {
    # name: (builder, the output's picture axes, threshold, what the scores are)
    "sliced and squeezed": (after_conv(("Slice", {}, SECOND), ("Squeeze", {}, AXES), classes=2), (1, 2), 0.0, "raw"),
    "sliced, both squeezed": (after_conv(("Slice", {}, SECOND), ("Squeeze", {}, BOTH), classes=2), (0, 1), 0.0, "raw"),
    "split and squeezed": (split_and_squeezed, (1, 2), 0.0, "raw"),
    "one gathered": (after_conv(("Gather", {"axis": 1}, [("at", 1, np.int64)]), classes=2), (1, 2), 0.0, "raw"),
    "softmax, one gathered": (
        after_conv(("Softmax", {"axis": 1}, []), ("Gather", {"axis": 1}, [("at", 1, np.int64)]), classes=2),
        (1, 2), 0.5, "probability"),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(PICKED))
@pytest.mark.parametrize("labels", [None, ["bg", "fg"]])
def test_one_class_picked_out_and_its_axis_dropped_is_one_score_for_each_pixel(name, labels):
    """Refused as "classes folded into another axis by a reshape": judged by the convolution's two channels, not
    by the one the axis held when it was dropped. One class's probability is cut at one half. One class's raw
    score out of several has no cut of its own (the class wins where its score beats the others'), so it is read
    where the caller says, and refused naming ``threshold`` when the caller does not."""
    build, picture, threshold, kind = PICKED[name]
    if kind == "raw":
        with pytest.raises(OnnxImportError) as refused:
            import_onnx(build(8, 12), task="segmentation", inputs=dict(HINT), labels=labels)
        assert refused.value.param == "threshold" and "picked out of" in str(refused.value)
        spec = import_onnx(build(8, 12), task="segmentation", inputs=dict(HINT), labels=labels, threshold=threshold)
        assert (spec["head"]["layout"], spec["head"]["threshold"]) == ("chw", threshold)
        assert "score_kind" not in spec["head"]
    else:
        spec = import_onnx(build(8, 12), task="segmentation", inputs=dict(HINT), labels=labels)
        assert (spec["head"]["layout"], spec["head"]["threshold"], spec["head"]["score_kind"]) == ("chw", threshold, kind)
    head = spec["head"]
    shape = spec["graph"]["values"][head["output"]]["shape"]
    sides = set()
    for row in spec["check"]["outputs"]:
        value = np.asarray(row).reshape(shape)
        truth = half(value, picture, kind == "raw")
        np.testing.assert_array_equal(mask_by("chw", value, threshold), truth)
        sides.update(int(item) for item in np.unique(truth))
    assert sides == {0, 1}


def test_a_reshape_that_takes_the_classes_apart_proves_nothing_and_one_that_folds_them_is_still_refused():
    names = [f"class {index}" for index in range(CLASSES)]
    undone = after_conv(
        ("Reshape", {}, [("to", lambda h, w, c: [1, c * h * w], np.int64)]),
        ("Reshape", {}, [("to", lambda h, w, c: [1, c, h, w], np.int64)]),
    )
    spec = import_onnx(undone(12, 16), task="segmentation", inputs=dict(HINT), labels=names)  # was "folded"
    assert spec["head"]["layout"] == "chw"
    value = own_output(spec)
    np.testing.assert_array_equal(mask_by("chw", value), np.argmax(value, axis=1)[0])
    with pytest.raises(OnnxImportError, match="can't tell which axis") as caught:
        import_onnx(undone(12, 16), task="segmentation", inputs=dict(HINT))
    assert caught.value.param == "mask_layout"
    folded = after_conv(("Reshape", {}, [("to", lambda h, w, c: [1, c * h, w], np.int64)]))
    with pytest.raises(OnnxImportError, match="folded into another axis"):
        import_onnx(folded(5, 5), task="segmentation", inputs=dict(HINT), labels=names)


def test_scores_that_are_never_negative_and_go_above_one_are_not_cut_at_nought():
    """Conv, Slice, Relu and a sigmoid times 255 were both cut at 0, which made every pixel the foreground."""
    kept = after_conv(("Slice", {}, SECOND), ("Relu", {}, []), classes=2)
    with pytest.raises(OnnxImportError, match="keeps nothing below 0") as caught:
        import_onnx(kept(8, 8), task="segmentation", inputs=dict(HINT))
    assert caught.value.param == "threshold"
    told = import_onnx(kept(8, 8), task="segmentation", inputs=dict(HINT), threshold=0.75)
    assert told["head"]["threshold"] == 0.75 and "score_kind" not in told["head"]
    scaled = import_onnx(after_conv(SIGMOID, ("Mul", {}, [("k", 255.0, np.float32)]), classes=1)(8, 12),
                         task="segmentation", inputs=dict(HINT))
    assert scaled["head"]["threshold"] == 127.5 and "score_kind" not in scaled["head"]
    shape = scaled["graph"]["values"][scaled["head"]["output"]]["shape"]
    sides = set()
    for row in scaled["check"]["outputs"]:
        value = np.asarray(row).reshape(shape)
        truth = half(value / 255.0, (2, 3), False)
        np.testing.assert_array_equal(mask_by("chw", value, 127.5), truth)
        sides.update(int(item) for item in np.unique(truth))
    assert sides == {0, 1}
    # No step says what they are, and they are never below 0 and go above 1: refused with what was seen.
    with pytest.raises(OnnxImportError, match="rows tried they run from .* never below 0") as caught:
        import_onnx(after_conv(("Exp", {}, []), classes=1)(8, 8), task="segmentation", inputs=dict(HINT))
    assert caught.value.param == "threshold"


def queries(order, height=16, width=16):
    """A score for each of 11 classes of each of 7 queries, and each query's mask at a quarter of the picture."""
    rng = np.random.default_rng(7)
    features, initializers = conv("features", 8, rng)
    nodes = [
        features,
        helper.make_node("GlobalAveragePool", ["features"], ["pooled"]),
        helper.make_node("Reshape", ["pooled", "one"], ["token"]),
        helper.make_node("Add", ["token", "each"], ["tokens"]),
        helper.make_node("MatMul", ["tokens", "classify"], ["class_queries_logits"]),
        helper.make_node("AveragePool", ["features"], ["small"], kernel_shape=[4, 4], strides=[4, 4]),
        helper.make_node("Einsum", ["tokens", "small"], ["masks_queries_logits"], equation="nqc,nchw->nqhw"),
    ]
    initializers += [
        const("one", [1, 1, 8], np.int64),
        const("each", rng.normal(size=(1, 7, 8))),
        const("classify", rng.normal(size=(8, 11))),
    ]
    return model_of(nodes, [(name, F) for name in order], initializers, height, width)


def test_with_several_outputs_the_one_the_size_of_the_picture_is_the_mask():
    """The first output of floats with two or three axes was read: 7 queries by 11 classes as a 1 x 7 mask."""
    for order in (("class_queries_logits", "masks_queries_logits"), ("masks_queries_logits", "class_queries_logits")):
        spec = import_onnx(queries(order), task="segmentation", inputs=dict(HINT))
        assert spec["graph"]["values"][spec["head"]["output"]]["source"] == "masks_queries_logits", order
        assert spec["head"]["layout"] == "chw"
    assert onnx_graph._picture_sizes({"x": [1, 3, 16, 24], "ids": [1, 8]}, dict(HINT)) == [(16, 24)]
    assert onnx_graph._picture_fit((1, 5, 16, 24), [(16, 24)]) == 0 and onnx_graph._picture_fit((16, 24), [(16, 24)]) == 0
    assert onnx_graph._picture_fit((1, 7, 4, 6), [(16, 24)]) == 1 and onnx_graph._picture_fit((1, 7, 11), [(16, 24)]) == 2
    # An output the size of the picture before one at half of it, whichever comes first; with no picture to go
    # by, the order the file gives, as before.
    values = {"half": np.zeros((1, 5, 8, 12), np.float32), "out": np.zeros((1, 5, 16, 24), np.float32)}
    assert onnx_graph._mask_output(["half", "out"], values, [(16, 24)]) == "out"
    assert onnx_graph._mask_output(["half", "out"], values) == "half"


def test_a_pad_id_of_none_is_no_padding_token_and_one_that_is_no_whole_number_is_refused():
    said = {"input_ids": {"feature": "text", "part": "ids", "pad_id": None}, WRITTEN: {"generated": True}}
    assert bound(import_onnx(writer(), inputs=said))["input_ids"] == {"feature": "text", "part": "ids"}
    with pytest.raises(OnnxImportError, match="a pad_id that is not a whole number") as caught:
        import_onnx(writer(), inputs={"input_ids": {"feature": "text", "part": "ids", "pad_id": "1"}})
    assert caught.value.param == "inputs"


@pytest.mark.parametrize("classes", [["no", "yes"], [3, 5, 7]])
def test_a_classical_classifier_written_without_a_zipmap_is_read_as_its_zipmap_file_is(classes):
    """Its scores do not add up to one and a cast stands after the classifier: it was read as a regression."""
    count = len(classes)
    rng = np.random.default_rng(3)
    named = {"classlabels_strings": classes} if isinstance(classes[0], str) else {"classlabels_ints": classes}
    classify = helper.make_node(
        "LinearClassifier", ["x"], ["its_label", "its_scores"], domain="ai.onnx.ml", post_transform="NONE",
        coefficients=[float(item) for item in rng.normal(size=count * 4)],
        intercepts=[float(item) for item in rng.normal(size=count)], **named,
    )  # fmt: skip
    kind = TensorProto.STRING if isinstance(classes[0], str) else I64
    told = {}
    for zipmap in (False, True):
        if zipmap:
            last = helper.make_node(
                "ZipMap", ["its_scores"], ["probabilities"], domain="ai.onnx.ml",
                **{"classlabels_strings" if kind == TensorProto.STRING else "classlabels_int64s": classes},
            )  # fmt: skip
            map_of = helper.make_map_type_proto(kind, helper.make_tensor_type_proto(F, []))
            scores = helper.make_value_info("probabilities", helper.make_sequence_type_proto(map_of))
        else:
            last = helper.make_node("Cast", ["its_scores"], ["probabilities"], to=F)
            scores = helper.make_tensor_value_info("probabilities", F, [1, count])
        graph = helper.make_graph(
            [classify, helper.make_node("Identity", ["its_label"], ["label"]), last],
            "classical",
            [helper.make_tensor_value_info("x", F, [1, 4])],
            [helper.make_tensor_value_info("label", kind, [1]), scores],
        )
        file = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)])
        file.ir_version = 8
        spec = import_onnx(file, check=False)
        told[zipmap] = (spec["task"], spec["outputs"]["labels"], spec["head"]["link"])
    task = "binary_classification" if count == 2 else "multiclass_classification"
    assert told[False] == told[True] == (task, [str(item) for item in classes], "softmax")
