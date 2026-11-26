"""Uploading a general model graph, and the calls that go with models of the newer kinds."""

from __future__ import annotations

import json
import warnings
import zlib

import httpx
import pytest
import respx

np = pytest.importorskip("numpy")

from kaimei import Kaimei  # noqa: E402
from kaimei.resources import models as models_module  # noqa: E402
from kaimei.structure import finalize_spec  # noqa: E402

BASE = "https://api.kaimei.ai"
RNG = np.random.default_rng(5)


@pytest.fixture
def client():
    made = Kaimei(api_key="km_test_x", base_url=BASE, max_retries=0)
    yield made
    made.close()


@pytest.fixture
def api():
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


def raw_graph(features, inputs, **more):
    spec = {
        "format": "kaimei.structure.v1",
        "family": "graph",
        "framework": "pytorch",
        "task": "regression",
        "feature_names": [str(feature["name"]) for feature in features],
        "features": features,
        "graph": {
            "opsets": {"": 17},
            "inputs": inputs,
            "constants": {
                "w": RNG.normal(size=(40, 30)).astype(np.float32),
                "ids": np.arange(1500, dtype=np.int64),
                "keep": RNG.random(1200) > 0.5,
                "b": np.zeros(2, dtype=np.float32),
            },
            "nodes": [{"id": "fc", "op": "Gemm", "inputs": ["x", "w", "b"], "outputs": ["y"], "attrs": {}}],
            "outputs": ["y"],
            "values": {"y": {"shape": [1, 2], "dtype": "float32"}},
        },
        "head": {"output": "y", "link": "none"},
        "check": {"inputs": {features[0]["name"]: [[0.0]]}, "outputs": [[1.0]], "tolerance": 1e-4},
        "metadata": {"route": "onnx"},
    }
    spec.update(more)
    return spec


TABLE = [{"name": "age", "type": "numeric"}, {"name": "income", "type": "numeric"}]
TABLE_INPUT = [{"name": "x", "dtype": "float32", "shape": [1, 2], "from": {"features": ["age", "income"]}}]


def stand_in(monkeypatch, spec, externalize_above=1000):
    """extract_structure answering with a spec made here, so no converter is needed to test the upload."""
    seen = {}

    def extract(estimator, **options):
        seen.update(options)
        return finalize_spec(spec, weights=options.get("weights", "full"), externalize_above=externalize_above)

    monkeypatch.setattr(models_module, "extract_structure", extract)
    return seen


def created(request, **more):
    body = json.loads(request.content)
    return httpx.Response(201, json={"object": "model", "id": "mdl_g", "slug": "g", "name": body["name"], **more})


def test_a_graph_uploads_its_tensors_in_their_own_types_then_its_structure(client, api, monkeypatch):
    spec = raw_graph(TABLE, TABLE_INPUT, task="multiclass_classification")
    seen = stand_in(monkeypatch, spec)
    create = api.post("/v1/models").mock(side_effect=created)
    tensors = api.put(url__regex=r".*/v1/models/mdl_g/structure/tensors/.+").mock(
        side_effect=lambda request: httpx.Response(200, json={"object": "model.structure.tensor", "key": request.url.path.rsplit("/", 1)[-1]})
    )
    structure = api.put("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    api.get("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure", "runnable": True, "summary": ""}))
    reference = api.put("/v1/models/mdl_g/reference-data").mock(return_value=httpx.Response(200, json={"object": "reference_data", "rows": 3}))
    stages = []
    model = client.models.upload(
        object(),
        name="Any model",
        task="multiclass_classification",
        X=[[31, 52000.0], [44, 61000.0], [27, 39000.0]],
        y=["a", "b", "a"],
        sample=[[0.0, 0.0]],
        inputs={"x": {"features": ["age", "income"]}},
        output=0,
        tokenizer=None,
        labels=["a", "b"],
        via="onnx",
        progress=lambda stage, done, total: stages.append((stage, done, total)),
    )
    assert model.id == "mdl_g"
    assert seen["via"] == "onnx" and seen["sample"] == [[0.0, 0.0]] and seen["output"] == 0 and seen["labels"] == ["a", "b"]
    assert seen["inputs"] == {"x": {"features": ["age", "income"]}} and seen["tokenizer"] is None
    assert seen["check_rows"] == [[31, 52000.0], [44, 61000.0], [27, 39000.0]]
    # The model is registered from shapes first: its tensors do not exist on the API yet.
    body = json.loads(create.calls.last.request.content)
    assert body["features"] == TABLE and body["task"] == "multiclass_classification"
    assert body["outputs"] == {"labels": ["a", "b"]} and body["framework"] == "pytorch"
    assert body["metadata"] == {"uploaded_from": "python-client", "binary_tensors": 3}
    first = body["structure"]
    assert first["family"] == "graph" and first["weights"] == "summary" and "check" not in first
    assert first["graph"]["constants"]["w"]["$shape"] == [40, 30] and "view" not in first["graph"]["constants"]["w"]
    assert first["graph"]["constants"]["ids"] == {"$shape": [1500], "dtype": "i64"}
    # Each tensor goes up in its own number type.
    sent = {call.request.url.path.rsplit("/", 1)[-1]: call.request for call in tensors.calls}
    assert set(sent) == {"w", "ids", "keep"}
    assert dict(sent["w"].url.params) == {"shape": "40,30", "dtype": "f32", "compression": "zlib"}
    assert dict(sent["ids"].url.params) == {"shape": "1500", "dtype": "i64", "compression": "zlib"}
    assert dict(sent["keep"].url.params) == {"shape": "1200", "dtype": "bool", "compression": "zlib"}
    assert zlib.decompress(sent["ids"].content) == spec["graph"]["constants"]["ids"].astype("<i8").tobytes()
    assert zlib.decompress(sent["w"].content) == spec["graph"]["constants"]["w"].astype("<f4").tobytes()
    assert zlib.decompress(sent["keep"].content) == spec["graph"]["constants"]["keep"].tobytes()
    # Then the structure that refers to them, with its check.
    full = json.loads(structure.calls.last.request.content)
    assert full["graph"]["constants"]["w"] == {"$tensor": "w", "shape": [40, 30], "dtype": "f32"}
    assert full["check"]["tolerance"] == 1e-4 and "weights" not in full
    rows = json.loads(reference.calls.last.request.content)
    assert rows["rows"][0] == {"age": 31, "income": 52000.0} and rows["labels"] == ["a", "b", "a"]
    assert ("tensors", 3, 3) in stages and ("structure", 1, 1) in stages


def test_a_small_graph_goes_up_in_one_request(client, api, monkeypatch):
    spec = raw_graph(TABLE, TABLE_INPUT)
    stand_in(monkeypatch, spec, externalize_above=None)
    create = api.post("/v1/models").mock(side_effect=created)
    api.get("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"runnable": True, "summary": ""}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client.models.upload(object(), name="Small")
    body = json.loads(create.calls.last.request.content)
    assert body["structure"]["check"] == spec["check"] and "$b64" in body["structure"]["graph"]["constants"]["w"]
    assert body["task"] == "regression" and body["inference"] == {"type": "none"}
    assert len(api.calls) == 2  # create, then one look at whether it runs


def test_the_file_a_structure_was_read_from_is_written_where_asked_and_the_token_room_is_passed_on(client, api, monkeypatch, tmp_path):
    spec = raw_graph(TABLE, TABLE_INPUT)
    seen = {}

    def extract(estimator, **options):
        seen.update(options)
        made = finalize_spec(spec, weights=options.get("weights", "full"), externalize_above=None)
        made.onnx = seen.get("file")
        return made

    monkeypatch.setattr(models_module, "extract_structure", extract)
    create = api.post("/v1/models").mock(side_effect=created)
    api.get("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"runnable": True, "summary": ""}))
    stored = api.put("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    seen["file"] = b"onnx bytes"
    client.models.upload(object(), name="Kept", max_length=128, save_onnx=tmp_path / "kept.onnx")
    assert (tmp_path / "kept.onnx").read_bytes() == b"onnx bytes" and seen["max_length"] == 128
    assert b"onnx bytes" not in create.calls.last.request.content  # the file stays here
    client.models.upload_structure("mdl_g", object(), max_length=64, save_onnx=str(tmp_path / "again.onnx"))
    assert (tmp_path / "again.onnx").read_bytes() == b"onnx bytes" and seen["max_length"] == 64 and stored.called
    seen["file"] = None
    with pytest.warns(UserWarning, match="save_onnx wrote no file"):
        client.models.upload(object(), name="Native", save_onnx=tmp_path / "none.onnx")
    assert not (tmp_path / "none.onnx").exists()


def test_operators_the_api_cannot_run_are_named_in_a_warning(client, api, monkeypatch):
    stand_in(monkeypatch, raw_graph(TABLE, TABLE_INPUT), externalize_above=None)
    api.post("/v1/models").mock(side_effect=created)
    said = "Model graph of 3 operations. Kaimei draws but does not run: STFT, com.example.FancyNewOp."
    api.get("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"runnable": False, "summary": said}))
    with pytest.warns(UserWarning, match="STFT, com.example.FancyNewOp") as caught:
        client.models.upload(object(), name="Drawn")
    assert "kaimei.agent" in str(caught[0].message)
    api.get("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"runnable": False, "summary": "It differs."}))
    # A regression that gives two numbers is drawn for that reason, and the warning says what to pass instead.
    with pytest.warns(UserWarning, match="gives 2 numbers for a row and was read as a regression") as told:
        client.models.upload(object(), name="Two numbers")
    assert "task='generation'" in str(told[0].message)
    one = raw_graph(TABLE, TABLE_INPUT)
    one["graph"]["values"]["y"]["shape"] = [1, 1]
    stand_in(monkeypatch, one, externalize_above=None)
    with pytest.warns(UserWarning, match="didn't give the model's own outputs"):
        client.models.upload(object(), name="Differs")


def test_text_series_and_tensors_register_as_what_they_are_and_keep_no_reference_rows(client, api, monkeypatch):
    features = [{"name": "review", "type": "text", "max_length": 12, "vocab_size": 9}]
    inputs = [{"name": "x", "dtype": "int64", "shape": [1, 12], "from": {"feature": "review", "part": "ids", "pad_id": 0}}]
    tokenizers = {"review": {"type": "word", "vocab": ["<pad>", "<unk>", "good"], "unk": "<unk>", "pad": "<pad>"}}
    stand_in(monkeypatch, raw_graph(features, inputs, task="text_generation", tokenizers=tokenizers), externalize_above=None)
    create = api.post("/v1/models").mock(side_effect=created)
    api.get("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"runnable": True, "summary": ""}))
    reference = api.put("/v1/models/mdl_g/reference-data").mock(return_value=httpx.Response(200, json={}))
    with pytest.warns(UserWarning, match="reference rows are stored for tables and images"):
        client.models.upload(object(), name="Writer", X=["a good film"], tokenizer=tokenizers["review"])
    body = json.loads(create.calls.last.request.content)
    assert body["features"] == features and body["task"] == "text_generation"
    assert body["structure"]["tokenizers"] == tokenizers
    assert not reference.called
    for kind, shape in (("sequence", [48, 1]), ("tensor", [6])):
        declared = [{"name": "signal", "type": kind, "shape": shape}]
        bound = [{"name": "x", "dtype": "float32", "shape": [1, *shape], "from": {"feature": "signal"}}]
        stand_in(monkeypatch, raw_graph(declared, bound, task="generation"), externalize_above=None)
        with pytest.warns(UserWarning, match="was not uploaded as reference data"):
            client.models.upload(object(), name="Series", X=np.zeros((2, *shape)))
        assert json.loads(create.calls.last.request.content)["features"] == declared
    assert not reference.called


def test_a_picture_model_keeps_reference_pictures_and_its_pixel_units(client, api, monkeypatch):
    features = [{"name": "photo", "type": "image", "height": 4, "width": 5, "channels": 3}]
    inputs = [{"name": "x", "dtype": "float32", "shape": [1, 3, 4, 5], "from": {"feature": "photo", "layout": "nchw"}}]
    stand_in(monkeypatch, raw_graph(features, inputs), externalize_above=None)
    create = api.post("/v1/models").mock(side_effect=created)
    api.get("/v1/models/mdl_g/structure").mock(return_value=httpx.Response(200, json={"runnable": True, "summary": ""}))
    reference = api.put("/v1/models/mdl_g/reference-data").mock(return_value=httpx.Response(200, json={}))
    batch = RNG.random((2, 3, 4, 5))  # channels first, as PyTorch has them
    client.models.upload(object(), name="Photos", X=batch, value_range=[0, 255])
    body = json.loads(create.calls.last.request.content)
    assert body["features"] == [{**features[0], "value_range": [0.0, 255.0]}]
    rows = json.loads(reference.calls.last.request.content)["rows"]
    assert len(rows) == 2 and np.asarray(rows[0]["photo"]).shape == (4, 5, 3)
    np.testing.assert_allclose(rows[1]["photo"], batch[1].transpose(1, 2, 0), atol=1e-6)


def test_upload_structure_finalizes_a_graph_that_still_holds_arrays(client, api, monkeypatch):
    tensors = api.put(url__regex=r".*/v1/models/m1/structure/tensors/.+").mock(return_value=httpx.Response(200, json={}))
    structure = api.put("/v1/models/m1/structure").mock(return_value=httpx.Response(200, json={"object": "model.structure"}))
    client.models.upload_structure("m1", raw_graph(TABLE, TABLE_INPUT), externalize_above=1000, metadata={"run": "7"})
    assert {call.request.url.path.rsplit("/", 1)[-1] for call in tensors.calls} == {"w", "ids", "keep"}
    sent = json.loads(structure.calls.last.request.content)
    assert sent["graph"]["constants"]["ids"]["$tensor"] == "ids" and sent["metadata"] == {"route": "onnx", "run": "7"}
    # A spec that is already JSON goes as it is.
    ready = finalize_spec(raw_graph(TABLE, TABLE_INPUT), externalize_above=None).spec
    client.models.upload_structure("m1", ready)
    assert json.loads(structure.calls.last.request.content) == ready
    # Anything else is read, with what the caller says about its inputs.
    seen = stand_in(monkeypatch, raw_graph(TABLE, TABLE_INPUT), externalize_above=None)
    client.models.upload_structure("m1", object(), via="onnx", sample="s", inputs={"x": {}}, output=1, tokenizer="t", labels=["a"])
    assert (seen["via"], seen["sample"], seen["inputs"], seen["output"], seen["tokenizer"], seen["labels"]) == ("onnx", "s", {"x": {}}, 1, "t", ["a"])


def test_upload_tensors_names_each_tensors_type(client, api):
    sent = api.put(url__regex=r".*/v1/models/m1/structure/tensors/.+").mock(return_value=httpx.Response(200, json={}))
    client.models.upload_tensors(
        "m1",
        {"a": np.zeros((2, 3), dtype=np.float32), "b": np.zeros(4), "c": np.zeros(4, dtype=np.int32), "d": np.zeros(4, dtype=bool), "e": np.zeros(4, dtype=np.float16)},
    )
    assert [call.request.url.params["dtype"] for call in sent.calls] == ["f32", "f64", "i32", "bool", "f16"]
    client.models.upload_tensors("m1", {"a": np.arange(4, dtype=np.int64)}, dtype="f32")
    assert sent.calls.last.request.url.params["dtype"] == "f32"
    assert zlib.decompress(sent.calls.last.request.content) == np.arange(4, dtype="<f4").tobytes()


def test_generate_runs_a_models_loop(client, api):
    route = api.post("/v1/models/writer/generate").mock(
        return_value=httpx.Response(200, json={"object": "model.generation", "model_id": "m", "kind": "text", "generated": "cd", "steps": [{"token": "c"}, {"token": "d"}]})
    )
    out = client.models.generate("writer", input={"prompt": "ab"}, steps=2, temperature=0.7, top_k=5, top_p=0.9, stop_tokens=[3, 4], seed=11)
    assert out.generated == "cd" and len(out.steps) == 2
    assert json.loads(route.calls.last.request.content) == {
        "input": {"prompt": "ab"}, "steps": 2, "temperature": 0.7, "top_k": 5, "top_p": 0.9, "stop_tokens": [3, 4], "seed": 11
    }
    # A model that draws needs no input: its noise comes from the seed.
    client.models.generate("writer")
    assert json.loads(route.calls.last.request.content) == {"input": {}, "temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0}


def test_tokenize_splits_text_or_names_ids(client, api):
    route = api.post("/v1/models/m1/tokenize").mock(
        return_value=httpx.Response(200, json={"object": "model.tokens", "model_id": "m", "feature": "review", "tokens": [{"token_id": 2, "token": "good"}], "count": 1})
    )
    assert client.models.tokenize("m1", text="good").tokens[0]["token"] == "good"
    assert json.loads(route.calls.last.request.content) == {"text": "good"}
    client.models.tokenize("m1", ids=np.array([2, 3]), feature="review")
    assert json.loads(route.calls.last.request.content) == {"ids": [2, 3], "feature": "review"}
    for wrong in ({}, {"text": "a", "ids": [1]}):
        with pytest.raises(ValueError):
            client.models.tokenize("m1", **wrong)


def test_parameter_view_asks_for_a_picture_of_a_window(client, api):
    route = api.get("/v1/models/m1/structure/parameters/embedded.weight").mock(
        return_value=httpx.Response(200, json={"object": "model.structure.parameter.view", "matrix": [9, 6], "view": {"shape": [2, 2], "range": [0, 1], "encoding": "u8", "data": "AAEC/w=="}})
    )
    out = client.models.parameter_view("m1", "embedded.weight")
    assert dict(route.calls.last.request.url.params) == {"side": "64", "pool": "strongest"}
    from kaimei.activations import decode

    np.testing.assert_allclose(decode(out.view), np.array([[0, 1], [2, 255]]) / 255.0)
    client.models.parameter_view("m1", "embedded.weight", side=8, rows=(2, 4), cols=(1, None), pool="mean")
    assert dict(route.calls.last.request.url.params) == {"side": "8", "rows": "2:4", "cols": "1:", "pool": "mean"}
    for wrong in ({"side": 0}, {"side": 600}, {"rows": (4, 2)}, {"cols": (-1, 3)}, {"rows": "2:4"}, {"rows": (1, 2, 3)}):
        with pytest.raises(ValueError):
            client.models.parameter_view("m1", "embedded.weight", **wrong)


def test_agent_status_and_registering_a_model_an_agent_runs(client, api):
    api.get("/v1/models/m1/agent").mock(
        return_value=httpx.Response(200, json={"object": "model.agent", "model_id": "m", "connected": True, "capabilities": ["predict", "trace"], "sessions": [{"id": "s1"}]})
    )
    status = client.models.agent_status("m1")
    assert status.connected is True and status.capabilities == ["predict", "trace"]
    create = api.post("/v1/models").mock(side_effect=created)
    client.models.create(name="Elsewhere", task="embedding", features=[{"name": "x", "type": "tensor", "shape": [4]}], inference={"type": "agent"})
    body = json.loads(create.calls.last.request.content)
    assert body["inference"] == {"type": "agent"} and body["task"] == "embedding"
