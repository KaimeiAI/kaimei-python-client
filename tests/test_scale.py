"""Large models: tensors leave the JSON spec, upload as binary, and structure-only uploads stay private."""

from __future__ import annotations

import json
import zlib

import httpx
import numpy as np
import pytest

from kaimei import Kaimei
from kaimei.structure import EXTERNALIZE_ABOVE, encode_tensor, extract_structure, finalize_spec


class _Mlp:
    """Looks like a fitted sklearn MLPRegressor with one wide layer."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.coefs_ = [rng.standard_normal((4, 20_000)), rng.standard_normal((20_000, 1))]
        self.intercepts_ = [np.zeros(20_000), np.zeros(1)]
        self.activation = "relu"
        self.out_activation_ = "identity"
        self.n_features_in_ = 4


def test_large_tensors_are_externalized():
    extraction = extract_structure(_Mlp(np.random.default_rng(0)), feature_names=["a", "b", "c", "d"])
    assert extraction.externalized
    # 20,000 × 4 = 80,000 values leaves the JSON; the 1 × 20,000 output layer stays inline.
    assert set(extraction.tensors) == {"dense_1.weight"}
    assert extraction.tensors["dense_1.weight"].dtype == np.float32
    assert extraction.tensors["dense_1.weight"].shape == (20_000, 4)
    layer = extraction.spec["layers"][1]
    assert layer["weight"] == {"$tensor": "dense_1.weight", "shape": [20_000, 4], "dtype": "f32"}
    assert isinstance(layer["bias"], list) and len(layer["bias"]) == 20_000
    assert len(extraction.spec["layers"][2]["weight"][0]) == 20_000
    ghost = extraction.summary_spec()
    assert ghost["layers"][1]["weight"]["$shape"] == [20_000, 4]
    assert ghost["layers"][1]["weight"]["stats"]["numel"] == 80_000
    assert ghost["layers"][2]["weight"]["$shape"] == [1, 20_000]
    # The spec itself stays small JSON.
    assert len(json.dumps(extraction.spec)) < 600_000
    inline = extract_structure(_Mlp(np.random.default_rng(0)), feature_names=["a", "b", "c", "d"], externalize_above=None)
    assert not inline.externalized
    assert len(inline.spec["layers"][1]["weight"]) == 20_000


def test_summary_weights_become_ghosts():
    extraction = extract_structure(_Mlp(np.random.default_rng(1)), feature_names=["a", "b", "c", "d"], weights="summary")
    assert not extraction.externalized
    layer = extraction.spec["layers"][1]
    assert layer["weight"]["$shape"] == [20_000, 4]
    assert layer["weight"]["stats"]["numel"] == 80_000
    assert layer["bias"]["$shape"] == [20_000]
    assert extraction.spec["weights"] == "summary"


def test_finalize_spec_keeps_existing_references():
    spec = {"layers": [{"id": "fc", "type": "dense", "weight": {"$tensor": "fc.weight"}, "bias": np.zeros(3)}]}
    out = finalize_spec(spec).spec
    assert out["layers"][0]["weight"] == {"$tensor": "fc.weight"}
    assert out["layers"][0]["bias"] == [0.0, 0.0, 0.0]


def test_encode_tensor_round_trips():
    array = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    raw = zlib.decompress(encode_tensor(array))
    np.testing.assert_allclose(np.frombuffer(raw, dtype="<f4").reshape(3, 4), array)


def _recorder():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append((request.method, path, dict(request.url.params)))
        if path == "/v1/models" and request.method == "POST":
            body = json.loads(request.content)
            # The first registration carries shapes only: tensors are not uploaded yet.
            assert body["structure"]["weights"] == "summary"
            assert body["structure"]["layers"][1]["weight"]["$shape"] == [20_000, 4]
            return httpx.Response(201, json={"object": "model", "id": "mdl_big", "slug": "big", "name": body["name"]})
        if path.startswith("/v1/models/mdl_big/structure/tensors/") and request.method == "PUT":
            assert request.headers["content-type"] == "application/octet-stream"
            key = path.rsplit("/", 1)[-1]
            shape = [int(dim) for dim in request.url.params["shape"].split(",")]
            raw = zlib.decompress(request.content)
            assert len(raw) == 4 * int(np.prod(shape))
            return httpx.Response(200, json={"object": "model.structure.tensor", "key": key, "shape": shape})
        if path == "/v1/models/mdl_big/structure" and request.method == "PUT":
            body = json.loads(request.content)
            assert body["layers"][1]["weight"]["$tensor"] == "dense_1.weight"
            assert "weights" not in body
            return httpx.Response(200, json={"object": "model.structure", "model_id": "mdl_big"})
        if path == "/v1/models/mdl_big/structure/parameters/dense_1.weight" and request.method == "GET":
            assert request.url.params["format"] == "f32"
            return httpx.Response(200, content=np.arange(8, dtype="<f4").tobytes(), headers={"content-type": "application/octet-stream"})
        return httpx.Response(404, json={"error": {"type": "not_found_error", "message": path}})

    return calls, handler


@pytest.fixture
def client():
    calls, handler = _recorder()
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c, calls


def test_upload_registers_then_streams_tensors_then_puts_structure(client):
    kaimei, calls = client
    seen = []
    model = kaimei.models.upload(
        _Mlp(np.random.default_rng(2)),
        name="Big net",
        feature_names=["a", "b", "c", "d"],
        progress=lambda stage, done, total: seen.append((stage, done, total)),
    )
    assert model.id == "mdl_big"
    methods = [(method, path) for method, path, _params in calls]
    assert methods == [
        ("POST", "/v1/models"),
        ("PUT", "/v1/models/mdl_big/structure/tensors/dense_1.weight"),
        ("PUT", "/v1/models/mdl_big/structure"),
    ]
    assert seen[0] == ("tensors", 0, 1) and seen[-1] == ("structure", 1, 1)
    values = kaimei.models.structure_parameter_values("mdl_big", "dense_1.weight", limit=8)
    assert values.dtype == np.float32 and values.tolist() == list(range(8))


def test_externalize_threshold_is_sane():
    assert EXTERNALIZE_ABOVE == 65_536


def test_many_mid_sized_tensors_are_lifted_when_the_spec_would_be_large():
    torch = pytest.importorskip("torch")
    # Twelve 40,000-value layers: each under the default threshold, together ~10 MB of JSON.
    net = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(12)], torch.nn.Linear(200, 1)).eval()
    names = [f"f{i}" for i in range(200)]
    compact = extract_structure(net, feature_names=names, task="regression")
    assert len(compact.tensors) == 12
    assert len(json.dumps(compact.spec)) < 400_000
    # A threshold the caller chose is kept as given.
    chosen = extract_structure(net, feature_names=names, task="regression", externalize_above=50_000)
    assert not chosen.externalized and len(json.dumps(chosen.spec)) > 2_000_000
