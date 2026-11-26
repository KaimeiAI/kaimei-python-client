"""General model graphs: how their constants are sent, and what a summary keeps at home."""

from __future__ import annotations

import base64
import json
import re

import pytest

np = pytest.importorskip("numpy")

from kaimei import StructureError  # noqa: E402
from kaimei.graphs import (  # noqa: E402
    INLINE_VALUES,
    SUMMARY_KEEPS,
    finalize_graph,
    holds_arrays,
    is_onnx,
)
from kaimei.structure import (  # noqa: E402
    EXECUTABLE_FAMILIES,
    TENSOR_DTYPES,
    Extraction,
    encode_tensor,
    features_from_spec,
    finalize_spec,
    image_shape_of,
    spec_is_executable,
    tensor_dtype,
)

RNG = np.random.default_rng(3)
#: Tensor keys the API accepts (routers/structure.py).
KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/\-]{0,199}$")


def graph_spec(**constants):
    """A two-step graph over whatever constants a test wants to see finalized."""
    return {
        "format": "kaimei.structure.v1",
        "family": "graph",
        "task": "regression",
        "feature_names": ["a", "b"],
        "features": [{"name": "a", "type": "numeric"}, {"name": "b", "type": "numeric"}],
        "graph": {
            "opsets": {"": 17},
            "inputs": [{"name": "x", "dtype": "float32", "shape": [1, 2], "from": {"features": ["a", "b"]}}],
            "constants": dict(constants),
            "nodes": [{"id": "fc", "op": "Gemm", "inputs": ["x", "w", "b"], "outputs": ["y"], "attrs": {"transB": 1}}],
            "outputs": ["y"],
        },
        "head": {"output": "y", "link": "none"},
        "check": {"inputs": {"a": [[0.5]], "b": [[1.0]]}, "outputs": [[2.0]], "tolerance": 1e-4},
    }


def unpacked(value):
    """A ``$b64`` constant back as the array the API will read."""
    raw = base64.b64decode(value["$b64"])
    array = np.frombuffer(raw, dtype=TENSOR_DTYPES[value["dtype"]]).reshape(value["shape"])
    return array.astype(value["as"]) if value.get("as") else array


def test_a_graph_is_a_family_the_api_runs():
    assert "graph" in EXECUTABLE_FAMILIES
    assert spec_is_executable({"family": "graph"})
    assert not spec_is_executable({"family": "graph", "approximate": True})
    assert not spec_is_executable({"family": "graph", "weights": "summary"})


def test_small_constants_are_numbers_in_their_own_type():
    spec = graph_spec(
        w=np.array([[0.25, -0.5]], dtype=np.float32),
        b=np.array([0.1], dtype=np.float64),
        axes=np.array([0, 2], dtype=np.int64),
        flag=np.array(True),
        scalar=np.float32(2.5),
        empty=np.zeros((0,), dtype=np.float32),
        names=np.array(["no", "yes"], dtype=object),
    )
    done = finalize_graph(spec)
    constants = done.spec["graph"]["constants"]
    assert constants["w"] == {"dtype": "float32", "shape": [1, 2], "data": [0.25, -0.5]}
    assert constants["b"] == {"dtype": "float64", "shape": [1], "data": [0.1]}
    assert constants["axes"] == {"dtype": "int64", "shape": [2], "data": [0, 2]}
    assert constants["flag"] == {"dtype": "bool", "shape": [], "data": [True]}
    assert constants["scalar"] == {"dtype": "float32", "shape": [], "data": [2.5]}
    assert constants["empty"] == {"dtype": "float32", "shape": [0], "data": []}
    assert constants["names"] == {"dtype": "string", "shape": [2], "data": ["no", "yes"]}
    assert done.tensors == {} and not done.externalized
    json.dumps(done.spec, allow_nan=False)  # plain JSON, with nothing left for a default= to rescue


def test_the_spec_given_is_not_changed_and_keeps_what_the_api_reads():
    weight = RNG.normal(size=(40, 30)).astype(np.float32)
    spec = graph_spec(w=weight, b=np.zeros(40, dtype=np.float32))
    done = finalize_graph(spec, externalize_above=100)
    assert spec["graph"]["constants"]["w"] is weight and "check" in spec
    assert done.spec["features"] == spec["features"] and done.spec["check"] == spec["check"]
    assert done.spec["head"] == spec["head"] and "weights" not in done.spec
    assert done.spec["graph"]["nodes"] == spec["graph"]["nodes"]


def test_larger_constants_and_any_holding_nan_or_infinity_travel_as_bytes():
    table = RNG.normal(size=(INLINE_VALUES + 1,)).astype(np.float32)
    mask = np.array([0.0, -np.inf, np.nan], dtype=np.float32)
    doubles = RNG.normal(size=(9, 9))
    spec = graph_spec(w=table, b=mask, d=doubles)
    constants = finalize_graph(spec).spec["graph"]["constants"]
    assert set(constants["w"]) == {"$b64", "dtype", "shape"} and constants["w"]["dtype"] == "f32"
    np.testing.assert_array_equal(unpacked(constants["w"]), table)
    assert constants["b"]["dtype"] == "f32" and constants["b"]["shape"] == [3]
    back = unpacked(constants["b"])
    assert back[0] == 0.0 and np.isneginf(back[1]) and np.isnan(back[2])
    assert constants["d"]["dtype"] == "f64"
    np.testing.assert_array_equal(unpacked(constants["d"]), doubles)
    json.dumps(constants, allow_nan=False)


def test_types_the_api_stores_no_bytes_of_are_widened_and_say_what_they_are():
    spec = graph_spec(
        half=RNG.normal(size=(100,)).astype(np.float16),
        counts=np.arange(70, dtype=np.uint16),
        wide=np.arange(70, dtype=np.uint32),
        bytes_=np.arange(70, dtype=np.uint8),
        small=np.arange(70, dtype=np.int8),
    )
    constants = finalize_graph(spec).spec["graph"]["constants"]
    assert (constants["half"]["dtype"], constants["half"]["as"]) == ("f16", "float16")
    assert (constants["counts"]["dtype"], constants["counts"]["as"]) == ("i32", "uint16")
    assert (constants["wide"]["dtype"], constants["wide"]["as"]) == ("i64", "uint32")
    assert constants["bytes_"]["dtype"] == "u8" and "as" not in constants["bytes_"]
    assert constants["small"]["dtype"] == "i8" and "as" not in constants["small"]
    for name in ("half", "counts", "wide", "bytes_", "small"):
        np.testing.assert_array_equal(unpacked(constants[name]), spec["graph"]["constants"][name])
    with pytest.raises(StructureError):
        finalize_graph(graph_spec(w=np.zeros(4, dtype=np.complex64)))


def test_constants_above_the_threshold_leave_the_spec_in_their_own_number_type():
    spec = graph_spec(
        w=RNG.normal(size=(40, 30)).astype(np.float32),
        b=RNG.normal(size=(1500,)),
        ids=np.arange(1300, dtype=np.int64),
        keep=RNG.random(1300) > 0.5,
        small=np.arange(12, dtype=np.int64),
    )
    done = finalize_graph(spec, externalize_above=1000)
    constants = done.spec["graph"]["constants"]
    assert constants["w"] == {"$tensor": "w", "shape": [40, 30], "dtype": "f32"}
    assert constants["b"] == {"$tensor": "b", "shape": [1500], "dtype": "f64"}
    assert constants["ids"] == {"$tensor": "ids", "shape": [1300], "dtype": "i64"}
    assert constants["keep"] == {"$tensor": "keep", "shape": [1300], "dtype": "bool"}
    assert constants["small"]["data"] == list(range(12))
    assert {key: str(array.dtype) for key, array in done.tensors.items()} == {
        "w": "float32", "b": "float64", "ids": "int64", "keep": "bool"
    }
    np.testing.assert_array_equal(done.tensors["ids"], spec["graph"]["constants"]["ids"])
    assert done.externalized and done.tensor_bytes == 1200 * 4 + 1500 * 8 + 1300 * 8 + 1300
    everything = finalize_graph(spec, externalize_above=None)
    assert everything.tensors == {} and "$b64" in everything.spec["graph"]["constants"]["w"]


def test_tensor_keys_are_ones_the_api_accepts_and_no_two_are_the_same():
    odd = {
        "const./layer 1/weight:0": RNG.normal(size=(2000,)).astype(np.float32),
        "const._layer_1_weight:0": RNG.normal(size=(2000,)).astype(np.float32),
        "/starts/with/a/slash": RNG.normal(size=(2000,)).astype(np.float32),
        "x" * 230: RNG.normal(size=(2000,)).astype(np.float32),
        "ünïcödé": RNG.normal(size=(2000,)).astype(np.float32),
    }
    done = finalize_graph(graph_spec(**odd), externalize_above=1000)
    keys = [value["$tensor"] for value in done.spec["graph"]["constants"].values()]
    assert len(set(keys)) == len(odd) == len(done.tensors)
    assert all(KEY.match(key) for key in keys), keys
    for name, value in done.spec["graph"]["constants"].items():
        np.testing.assert_array_equal(done.tensors[value["$tensor"]], odd[name])


def nested_spec():
    """A Scan whose body holds its own constants, as a converted k-nearest-neighbours model does."""
    spec = graph_spec(w=np.ones((1, 2), dtype=np.float32), b=np.zeros(1, dtype=np.float32))
    spec["graph"]["nodes"].insert(
        0,
        {
            "id": "scan",
            "op": "Scan",
            "inputs": ["x"],
            "outputs": ["z"],
            "attrs": {
                "num_scan_inputs": 1,
                "body": {
                    "graph": {
                        "inputs": [{"name": "item", "dtype": "float32", "shape": [None, 2]}],
                        "constants": {
                            "train": RNG.normal(size=(600, 2)).astype(np.float32),
                            "two": np.array(2.0, dtype=np.float32),
                        },
                        "nodes": [
                            {"id": "sub", "op": "Sub", "inputs": ["item", "train"], "outputs": ["d"], "attrs": {}},
                            {
                                "id": "inner",
                                "op": "If",
                                "inputs": ["flag"],
                                "outputs": ["e"],
                                "attrs": {
                                    "branches": {
                                        "graphs": [
                                            {"constants": {"deep": RNG.normal(size=(700,))}, "nodes": [], "outputs": ["deep"]}
                                        ]
                                    }
                                },
                            },
                        ],
                        "outputs": ["d"],
                    }
                },
            },
        },
    )
    return spec


def test_constants_of_nested_graphs_are_finalized_too():
    spec = nested_spec()
    done = finalize_graph(spec, externalize_above=500)
    body = done.spec["graph"]["nodes"][0]["attrs"]["body"]["graph"]
    assert body["constants"]["two"] == {"dtype": "float32", "shape": [], "data": [2.0]}
    train = body["constants"]["train"]
    assert train["shape"] == [600, 2] and train["dtype"] == "f32" and KEY.match(train["$tensor"])
    deep = body["nodes"][1]["attrs"]["branches"]["graphs"][0]["constants"]["deep"]
    assert deep["dtype"] == "f64" and deep["$tensor"] != train["$tensor"]
    assert set(done.tensors) == {train["$tensor"], deep["$tensor"]}
    np.testing.assert_array_equal(done.tensors[deep["$tensor"]], spec["graph"]["nodes"][0]["attrs"]["body"]["graph"]["nodes"][1]["attrs"]["branches"]["graphs"][0]["constants"]["deep"])
    assert body["inputs"] == [{"name": "item", "dtype": "float32", "shape": [None, 2]}]
    json.dumps(done.spec, allow_nan=False)
    assert not holds_arrays(done.spec) and holds_arrays(spec)


def test_an_input_that_is_always_given_a_value_carries_it_as_json():
    spec = graph_spec(w=np.ones((1, 2), dtype=np.float32), b=np.zeros(1, dtype=np.float32))
    spec["graph"]["inputs"].append(
        {"name": "state", "dtype": "float32", "shape": [1, 3], "from": {"value": np.array([[0.0, 1.0, np.inf]], dtype=np.float32)}}
    )
    spec["graph"]["inputs"].append({"name": "zeros", "dtype": "float32", "shape": [1, 3], "from": {"zeros": True}})
    inputs = finalize_graph(spec).spec["graph"]["inputs"]
    assert np.isinf(unpacked(inputs[1]["from"]["value"])[0, 2])
    assert inputs[2]["from"] == {"zeros": True}
    spec["graph"]["inputs"][1]["from"]["value"] = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    assert finalize_graph(spec).spec["graph"]["inputs"][1]["from"]["value"]["data"] == [0.0, 1.0, 2.0]


def test_a_summary_sends_shapes_statistics_and_a_small_picture_and_nothing_learned():
    weight = RNG.normal(size=(40, 30)).astype(np.float32)
    spec = graph_spec(
        w=weight,
        b=RNG.normal(size=(40,)).astype(np.float32),
        shape=np.array([1, -1], dtype=np.int64),
        scale=np.array(0.5, dtype=np.float32),
        table=np.arange(300, dtype=np.int64),
        edge=RNG.normal(size=(SUMMARY_KEEPS + 1,)).astype(np.float32),
        fits=RNG.normal(size=(SUMMARY_KEEPS,)).astype(np.float32),
    )
    plain = finalize_graph(spec, weights="summary").spec["graph"]["constants"]
    assert set(plain["w"]) == {"$shape", "dtype", "stats"}  # a summary alone: no picture unless one is asked for
    done = finalize_graph(spec, weights="pictures")
    constants = done.spec["graph"]["constants"]
    assert done.tensors == {} and done.spec["weights"] == "summary" and "check" not in done.spec
    assert not spec_is_executable(done.spec)
    assert set(constants["w"]) == {"$shape", "dtype", "stats", "view"}
    assert constants["w"]["$shape"] == [40, 30] and constants["w"]["dtype"] == "f32"
    assert constants["w"]["stats"]["numel"] == 1200
    assert constants["w"]["stats"]["max"] == pytest.approx(float(weight.max()))
    assert constants["w"]["view"]["shape"] == [40, 30] and constants["w"]["view"]["pool"] == "strongest"
    assert constants["b"] == {"$shape": [40], "dtype": "f32"}  # too few values for statistics to say little
    assert constants["table"] == {"$shape": [300], "dtype": "i64"}  # whole numbers have no statistics to give
    assert constants["edge"] == {"$shape": [SUMMARY_KEEPS + 1], "dtype": "f32"}
    # What the graph is drawn from stays: shapes and axes. A float the model learned does not, however few values.
    assert constants["shape"] == {"dtype": "int64", "shape": [2], "data": [1, -1]}
    assert constants["scale"] == {"$shape": [], "dtype": "f32"}
    assert constants["fits"] == {"$shape": [SUMMARY_KEEPS], "dtype": "f32"}
    assert done.spec["features"] == spec["features"]
    assert "approximate" not in done.spec  # the ghost tensors alone tell the API it cannot run this
    json.dumps(done.spec, allow_nan=False)


def test_a_summary_keeps_the_numbers_an_operator_holds_in_its_attributes_at_home():
    spec = graph_spec(w=np.ones((1, 2), dtype=np.float32), b=np.zeros(1, dtype=np.float32))
    spec["summary"] = "A converted forest."
    spec["graph"]["nodes"].append(
        {
            "id": "trees",
            "op": "TreeEnsembleRegressor",
            "domain": "ai.onnx.ml",
            "inputs": ["x"],
            "outputs": ["t"],
            "attrs": {
                "nodes_values": [0.1 * index for index in range(40)],
                "nodes_featureids": list(range(40)),
                "nodes_modes": ["BRANCH_LEQ"] * 40,
                "n_targets": 1,
                "post_transform": "NONE",
                "short": [1.0, 2.0],
                "tensor": {"dtype": "float32", "shape": [20], "data": [0.5] * 20},
            },
        }
    )
    done = finalize_graph(spec, weights="summary")
    attrs = done.spec["graph"]["nodes"][1]["attrs"]
    assert attrs["nodes_values"] == {"$shape": [40], "dtype": "float32"}
    assert attrs["nodes_featureids"] == {"$shape": [40], "dtype": "int64"}
    assert attrs["tensor"] == {"$shape": [20], "dtype": "float32"}
    assert attrs["nodes_modes"] == ["BRANCH_LEQ"] * 40 and attrs["n_targets"] == 1
    # A short list of a learned operator's numbers is still its numbers: left out too.
    assert attrs["short"] == {"$shape": [2], "dtype": "float32"}
    # With its numbers left out an operator cannot run, and the API is told so.
    assert done.spec["approximate"] is True
    assert done.spec["summary"].startswith("A converted forest.") and "stayed in your environment" in done.spec["summary"]
    full = finalize_graph(spec)
    assert full.spec["graph"]["nodes"][1]["attrs"]["nodes_values"] == spec["graph"]["nodes"][1]["attrs"]["nodes_values"]
    assert "approximate" not in full.spec


def test_a_spec_whose_tensors_were_lifted_out_is_summarised_from_them():
    weight = RNG.normal(size=(40, 30)).astype(np.float32)
    packed = RNG.normal(size=(20, 10)).astype(np.float32)
    spec = graph_spec(w=weight, b=RNG.normal(size=(8,)).astype(np.float32), p=packed)
    done = finalize_spec(spec, externalize_above=1000)
    assert isinstance(done, Extraction) and set(done.tensors) == {"w"}
    ghost = done.summary_spec()
    constants = ghost["graph"]["constants"]
    assert constants["w"]["$shape"] == [40, 30] and constants["w"]["stats"]["numel"] == 1200
    assert constants["w"]["stats"]["mean"] == pytest.approx(float(weight.mean()), abs=1e-6)
    assert "view" not in constants["w"]  # its tensors are on their way: a picture would be sent for nothing
    assert constants["p"] == {"$shape": [20, 10], "dtype": "f32"}  # written as bytes before; a ghost now
    assert constants["b"] == {"$shape": [8], "dtype": "f32"}  # learned, so left out however small
    assert ghost["weights"] == "summary" and "check" not in ghost
    # A reference with no array behind it still becomes a shape.
    alone = finalize_graph(done.spec, weights="summary")
    assert alone.spec["graph"]["constants"]["w"] == {"$shape": [40, 30], "dtype": "f32"}


def test_json_lists_are_left_as_they_are_and_summarised_as_the_api_reads_them():
    spec = graph_spec(w=[[0.5, 1.5]], b=[0.25], ids=list(range(40)), halves=[0.5] * 40)
    constants = finalize_graph(spec).spec["graph"]["constants"]
    assert constants["w"] == [[0.5, 1.5]] and constants["ids"] == list(range(40))
    ghosts = finalize_graph(spec, weights="summary").spec["graph"]["constants"]
    assert ghosts["ids"] == {"$shape": [40], "dtype": "i64"} and ghosts["halves"] == {"$shape": [40], "dtype": "f32"}
    assert ghosts["w"] == {"$shape": [1, 2], "dtype": "f32"}


def test_weights_must_be_full_or_summary():
    with pytest.raises(StructureError) as caught:
        finalize_graph(graph_spec(), weights="some")
    assert caught.value.param == "weights"


def test_features_come_from_what_a_graph_declares():
    declared = [
        {"name": "review", "type": "text", "max_length": 16},
        {"name": "photo", "type": "image", "height": 8, "width": 8, "channels": 3},
        {"name": "load", "type": "sequence", "shape": [48, 1]},
        {"name": "age", "type": "numeric"},
    ]
    spec = {"family": "graph", "feature_names": ["review", "photo", "load", "age"], "features": declared}
    features = features_from_spec(spec, value_range=[0, 255])
    assert features[0] == declared[0] and features[2] == declared[2] and features[3] == declared[3]
    assert features[1] == {**declared[1], "value_range": [0.0, 255.0]}
    assert "value_range" not in declared[1]  # the spec's own list is not written on
    assert image_shape_of(spec) is None  # more than a picture
    assert image_shape_of({"features": [declared[1]]}) == [8, 8, 3]
    assert image_shape_of({"features": [declared[2]]}) is None
    # A spec that declares nothing is read as before.
    assert features_from_spec({"feature_names": ["a", "b"]}) == [{"name": "a", "type": "numeric"}, {"name": "b", "type": "numeric"}]


def test_tensors_upload_in_their_own_number_type():
    assert tensor_dtype(np.zeros(3, dtype=np.float32)) == "f32"
    assert tensor_dtype(np.zeros(3, dtype=np.float64)) == "f64"
    assert tensor_dtype(np.zeros(3, dtype=np.float16)) == "f16"
    assert tensor_dtype(np.zeros(3, dtype=np.int64)) == "i64"
    assert tensor_dtype(np.zeros(3, dtype=np.int32)) == "i32"
    assert tensor_dtype(np.zeros(3, dtype=np.int16)) == "i16"
    assert tensor_dtype(np.zeros(3, dtype=np.int8)) == "i8"
    assert tensor_dtype(np.zeros(3, dtype=np.uint8)) == "u8"
    assert tensor_dtype(np.zeros(3, dtype=np.uint16)) == "i32"
    assert tensor_dtype(np.zeros(3, dtype=np.uint64)) == "i64"
    assert tensor_dtype(np.zeros(3, dtype=bool)) == "bool"
    assert tensor_dtype([1.0, 2.0]) == "f32"  # not an array: as before, single precision
    import zlib

    ids = np.array([3, -1, 2**40], dtype=np.int64)
    assert zlib.decompress(encode_tensor(ids, "i64")) == ids.astype("<i8").tobytes()
    flags = np.array([True, False, True])
    assert zlib.decompress(encode_tensor(flags, "bool")) == b"\x01\x00\x01"
    assert zlib.decompress(encode_tensor(np.array([1.5, -2.0]), "f16")) == np.array([1.5, -2.0], dtype="<f2").tobytes()
    assert set(TENSOR_DTYPES) == {"f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "bool"}
    with pytest.raises(StructureError):
        encode_tensor(ids, "u64")


def test_what_counts_as_an_onnx_model(tmp_path):
    assert is_onnx("model.onnx") and is_onnx("C:/models/Net.ONNX") and is_onnx(tmp_path / "net.onnx")
    assert is_onnx(b"\x08\x07") and is_onnx(bytearray(b"\x08\x07"))
    assert not is_onnx("model.pt") and not is_onnx("onnx") and not is_onnx(None) and not is_onnx(object())
    assert not is_onnx({"family": "graph"}) and not is_onnx(3)

    class ModelProto:
        pass

    assert not is_onnx(ModelProto())  # the name alone is not enough: it has to be onnx's
    onnx = pytest.importorskip("onnx")
    assert is_onnx(onnx.ModelProto())
