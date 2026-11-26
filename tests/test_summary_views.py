"""What a summary says of a tensor it keeps at home: statistics, and a small picture, made a piece at a time."""

from __future__ import annotations

import base64
import json

import pytest

np = pytest.importorskip("numpy")

import kaimei.structure as structure  # noqa: E402
from kaimei.activations import decode  # noqa: E402
from kaimei.structure import VIEW_SIDE, finalize_spec, tensor_stats, tensor_view  # noqa: E402

RNG = np.random.default_rng(9)


def matrix_dims(shape):
    """The API's own (engine/structure/tensors.py): first axis down, the rest across, leading axes of one dropped."""
    dims = tuple(int(dim) for dim in shape)
    while len(dims) > 1 and dims[0] == 1:
        dims = dims[1:]
    if len(dims) == 0:
        return (1, 1)
    if len(dims) == 1:
        return (1, dims[0])
    return (dims[0], int(np.prod(dims[1:])))


def strongest_blocks(array):
    """The picture as the API makes it for a tensor it holds: blocks with their edges repeated, strongest value kept."""
    height = matrix_dims(array.shape)[0]
    matrix = np.asarray(array, dtype=float).reshape(height, -1)
    step_r, step_c = -(-matrix.shape[0] // VIEW_SIDE), -(-matrix.shape[1] // VIEW_SIDE)
    pad_r, pad_c = (-matrix.shape[0]) % step_r, (-matrix.shape[1]) % step_c
    matrix = np.pad(matrix, ((0, pad_r), (0, pad_c)), mode="edge")
    blocks = matrix.reshape(matrix.shape[0] // step_r, step_r, matrix.shape[1] // step_c, step_c)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(blocks.shape[0], blocks.shape[2], step_r * step_c)
    return np.take_along_axis(blocks, np.abs(blocks).argmax(axis=2)[..., None], axis=2)[..., 0]


def accepted_by_the_api(view, shape):
    """The checks the API makes of a picture before it keeps one (engine/structure/tensors.py, parse_view)."""
    rows, cols = view["shape"]
    height, width = matrix_dims(shape)
    assert 1 <= rows <= 64 and 1 <= cols <= 64 and rows <= height and cols <= width
    step_r, step_c = -(-height // rows), -(-width // cols)
    assert -(-height // step_r) == rows and -(-width // step_c) == cols
    assert view["encoding"] == "u8" and view["pool"] == "strongest"
    assert len(base64.b64decode(view["data"], validate=True)) == rows * cols
    low, high = view["range"]
    assert np.isfinite(low) and np.isfinite(high) and low <= high
    json.dumps(view, allow_nan=False)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (5,),
        (64,),
        (65,),
        (300,),
        (4097,),
        (3, 4),
        (64, 64),
        (65, 65),
        (100, 3),
        (129, 257),
        (7, 1000),
        (16, 3, 3, 3),
        (130, 4, 5),
        (1, 96, 24),
        (1, 1, 300),
        (1, 640),
        (2, 1, 70),
    ],
)
def test_the_picture_is_the_one_the_api_would_make(shape):
    array = RNG.normal(size=shape).astype(np.float32)
    if array.size > 8:
        flat = array.reshape(-1)
        flat[3] = -flat[2]  # a tie in size with the other sign: the first of them is kept
    view = tensor_view(array)
    accepted_by_the_api(view, shape)
    cells = strongest_blocks(array)
    assert view["shape"] == list(cells.shape)
    assert view["range"] == [float(cells.min()), float(cells.max())]
    spread = float(cells.max() - cells.min())
    np.testing.assert_allclose(decode(view), cells, atol=spread / 255 / 2 + 1e-12)


def test_a_small_tensor_has_a_cell_for_each_value_and_a_large_one_keeps_its_strongest():
    small = np.array([[0.0, -2.0, 1.0], [0.5, 2.0, -1.0]], dtype=np.float32)
    view = tensor_view(small)
    assert view["shape"] == [2, 3] and view["range"] == [-2.0, 2.0]
    assert list(base64.b64decode(view["data"])) == [128, 0, 191, 159, 255, 64]
    wide = np.zeros((1, 640), dtype=np.float32)
    wide[0, 15] = -7.0  # the only value of its block of ten
    wide[0, 21], wide[0, 25] = 3.0, -3.0  # a tie: the first is kept
    cells = decode(tensor_view(wide))
    assert cells.shape == (1, 64) and cells[0, 1] == pytest.approx(-7.0) and cells[0, 2] == pytest.approx(3.0)
    assert tensor_view(np.zeros((0, 4))) is None and tensor_view(np.array(["a"])) is None
    flat = tensor_view(np.full((10, 10), 0.25))
    assert set(base64.b64decode(flat["data"])) == {0} and flat["range"] == [0.25, 0.25]
    holes = np.array([[np.nan, 1.0], [np.inf, -1.0]])
    seen = decode(tensor_view(holes))
    assert np.isfinite(seen).all() and seen[0, 0] == pytest.approx(0.0, abs=1 / 255)


def test_leading_axes_of_one_are_not_rows():
    """A recurrent layer keeps its weights as (directions, 4 × units, inputs). One direction is not one row."""
    weights = RNG.normal(size=(1, 96, 24)).astype(np.float32)
    view = tensor_view(weights)
    assert view["shape"] == [48, 24] and view == tensor_view(weights[0])
    assert tensor_view(np.ones((1, 1, 1, 5), dtype=np.float32))["shape"] == [1, 5]
    assert tensor_view(np.ones((1, 1), dtype=np.float32))["shape"] == [1, 1]


def test_a_tensor_is_read_a_piece_at_a_time(monkeypatch):
    array = RNG.normal(size=(333, 77)).astype(np.float32)
    array[5, 5] = 0.0
    whole_view, whole_stats = tensor_view(array), tensor_stats(array)
    monkeypatch.setattr(structure, "SUMMARY_PIECE_VALUES", 1000)
    pieces = list(structure._pieces(array, 1000))
    assert len(pieces) == 26 and all(piece.base is not None for piece in pieces)  # views, not copies
    assert tensor_view(array) == whole_view
    in_pieces = tensor_stats(array)
    for key in ("min", "max", "mean", "std", "l2", "abs_mean", "sparsity"):
        assert in_pieces[key] == pytest.approx(whole_stats[key], rel=1e-9, abs=1e-12), key
    assert in_pieces["numel"] == 333 * 77 and in_pieces["histogram"]["counts"] == whole_stats["histogram"]["counts"]
    np.testing.assert_allclose(in_pieces["histogram"]["edges"], whole_stats["histogram"]["edges"])
    assert sum(in_pieces["histogram"]["counts"]) == array.size
    # One that is not laid out in order in memory gives the same picture.
    turned = np.ascontiguousarray(array).T
    assert not turned.flags.c_contiguous
    assert tensor_view(turned) == tensor_view(np.ascontiguousarray(turned))
    assert tensor_stats(turned)["mean"] == pytest.approx(whole_stats["mean"], rel=1e-9)


def network(weight, **more):
    return {
        "format": "kaimei.structure.v1",
        "family": "network",
        "task": "regression",
        "feature_names": [f"x{index}" for index in range(weight.shape[1])],
        "layers": [
            {"id": "input", "type": "input", "names": [f"x{index}" for index in range(weight.shape[1])]},
            {"id": "scale", "type": "normalize", "center": RNG.normal(size=(weight.shape[1],)), "scale": np.ones(weight.shape[1])},
            {"id": "fc", "type": "dense", "units": int(weight.shape[0]), "weight": weight, "bias": np.zeros(weight.shape[0]), **more},
        ],
    }


def test_pictures_are_asked_for_and_a_summary_alone_sends_none():
    """A summary says what it said before pictures existed: shapes and statistics. A picture of a small tensor is
    the tensor, so pictures are something a caller asks for by name."""
    weight = RNG.normal(size=(20, 300))
    plain = finalize_spec(network(weight), weights="summary").spec
    assert set(plain["layers"][2]["weight"]) == {"$shape", "dtype", "stats"} and "view" not in json.dumps(plain)
    asked = finalize_spec(network(weight), weights="pictures").spec
    assert asked["weights"] == "summary" and "check" not in asked  # everything a summary is, and the pictures
    assert asked["layers"][2]["weight"]["stats"] == plain["layers"][2]["weight"]["stats"]
    with pytest.raises(structure.StructureError, match="'full', 'summary' or 'pictures'"):
        finalize_spec(network(weight), weights="sketch")


def test_a_summary_of_a_network_sends_a_picture_where_it_sends_statistics():
    weight = RNG.normal(size=(20, 300))
    spec = finalize_spec(network(weight), weights="pictures").spec
    ghost = spec["layers"][2]["weight"]
    assert set(ghost) == {"$shape", "dtype", "stats", "view"} and ghost["$shape"] == [20, 300]
    accepted_by_the_api(ghost["view"], (20, 300))
    assert ghost["view"]["shape"] == [20, 60]
    np.testing.assert_allclose(decode(ghost["view"]), strongest_blocks(weight), atol=float(np.ptp(weight)) / 255)
    # A bias of a few values, and what a scaler holds, say nothing of themselves: they would be the values, or the data.
    assert spec["layers"][2]["bias"] == {"$shape": [20], "dtype": "f32"}
    assert spec["layers"][1]["center"] == {"$shape": [300], "dtype": "f32"}
    assert "view" not in json.dumps(finalize_spec(network(RNG.normal(size=(5, 4))), weights="pictures").spec)


def test_a_summary_made_from_tensors_already_lifted_out_has_the_same_picture():
    weight = RNG.normal(size=(40, 50))
    extraction = finalize_spec(network(weight), externalize_above=1000)
    assert set(extraction.tensors) == {"fc.weight"}
    # What a model is registered with while its tensors are on their way has no pictures: the tensors follow.
    assert "view" not in extraction.summary_spec()["layers"][2]["weight"]
    ghost = finalize_spec(extraction.spec, weights="pictures", externalize_above=None, tensors=extraction.tensors).spec[
        "layers"
    ][2]["weight"]
    direct = finalize_spec(network(weight), weights="pictures").spec["layers"][2]["weight"]
    assert ghost["$shape"] == [40, 50] and ghost["view"]["shape"] == [40, 50]
    # Lifted out in single precision, so the picture is of those values.
    np.testing.assert_allclose(decode(ghost["view"]), decode(direct["view"]), atol=1e-6)
    assert ghost["stats"]["numel"] == 2000


def test_summarising_does_not_copy_a_tensor_in_double_precision(monkeypatch):
    monkeypatch.setattr(structure, "SUMMARY_PIECE_VALUES", 4096)
    weight = RNG.normal(size=(300, 200)).astype(np.float32)
    converted = []
    real = np.asarray

    def watching(value, *args, **kwargs):
        made = real(value, *args, **kwargs)
        wanted = kwargs.get("dtype", args[0] if args else None)
        if getattr(value, "size", 0) == weight.size and wanted in (float, np.float64) and made is not value:
            converted.append(made.shape)
        return made

    monkeypatch.setattr(structure.np, "asarray", watching)
    ghost = finalize_spec(network(weight), weights="pictures").spec["layers"][2]["weight"]
    assert converted == [] and ghost["stats"]["numel"] == 60000 and ghost["view"]["shape"] == [60, 50]
    assert ghost["stats"]["mean"] == pytest.approx(float(weight.astype(np.float64).mean()), abs=1e-9)
