"""Models with steps Kaimei cannot replay are drawn, not refused."""

from __future__ import annotations

import numpy as np
import pytest

from kaimei import StructureError
from kaimei.structure import (
    _out,
    drawn_steps,
    extract_structure,
    features_from_spec,
    finalize_spec,
    spec_is_executable,
)

torch = pytest.importorskip("torch")
nn = torch.nn

NAMES = ["a", "b", "c", "d"]


class TinyAttention(nn.Module):
    """Tabular rows through one attention step: unsqueeze, attention and squeeze cannot be replayed."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.head = nn.Linear(8, 3)

    def forward(self, x):  # type: ignore[no-untyped-def]
        tokens = self.embed(x).unsqueeze(1)
        mixed, _weights = self.attn(tokens, tokens, tokens)
        return self.head(mixed.squeeze(1))


class Gate(nn.Module):
    """A module of someone's own, which no table of layers knows."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = 2.0
        self.register_buffer("bias", torch.full((8,), 0.5))

    def forward(self, x):  # type: ignore[no-untyped-def]
        return torch.sigmoid(x / self.temperature) + self.bias


class Untraceable(nn.Module):
    """Data-dependent control flow: torch.fx gives up, so the layers are read in forward-hook order."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 8)
        self.gate = Gate()
        self.last = nn.Linear(8, 1)

    def forward(self, x):  # type: ignore[no-untyped-def]
        h = self.first(x)
        if h.abs().sum() >= 0:  # always taken, but a tracer cannot know that
            h = self.gate(h)
        return self.last(h)


def _opaque(spec, op):
    return [layer for layer in spec["layers"] if layer.get("type") == "opaque" and layer.get("op") == op]


def _inputs(spec, layer):
    """What a layer reads: named outright, or the layer before it."""
    if "inputs" in layer:
        return layer["inputs"]
    return [_out(spec["layers"][spec["layers"].index(layer) - 1])]


def test_an_attention_model_is_drawn_step_by_step():
    torch.manual_seed(0)
    spec = extract_structure(TinyAttention().eval(), feature_names=NAMES, task="multiclass_classification").spec
    assert spec["approximate"] is True
    assert not spec_is_executable(spec)
    ops = {layer["op"] for layer in spec["layers"] if layer.get("type") == "opaque"}
    assert ops == {"unsqueeze", "MultiheadAttention", "getitem"}

    (attn,) = _opaque(spec, "MultiheadAttention")
    assert attn["path"] == "attn" and attn["shape"] == [1, 8]
    assert attn["attrs"]["num_heads"] == 2 and attn["attrs"]["batch_first"] is True
    assert attn["reason"] == "MultiheadAttention"
    assert np.shape(attn["params"]["in_proj_weight"]) == (24, 8)
    assert np.shape(attn["params"]["out_proj.weight"]) == (8, 8)
    # Self-attention reads the same tokens three times; that is one input, not three.
    assert _inputs(spec, attn) == [_opaque(spec, "unsqueeze")[0]["id"]]

    # The attention output and its weights are two steps read off the tuple; the squeeze back to a
    # vector is a no-op Kaimei can follow, so the dense head reads the attention output directly.
    taken, weights = _opaque(spec, "getitem")
    assert _inputs(spec, taken) == [attn["id"]] and taken["attrs"]["args"] == [f"${attn['id']}", 0]
    assert _inputs(spec, weights) == [attn["id"]]
    assert weights["attrs"]["args"] == [f"${attn['id']}", 1] and weights["shape"] == [1, 1]
    head = next(layer for layer in spec["layers"] if layer.get("id") == "head")
    assert head["type"] == "dense" and _inputs(spec, head) == [taken["id"]]
    assert spec["labels"] == ["class_0", "class_1", "class_2"]
    assert "4 steps Kaimei cannot replay" in spec["summary"] and "drawn, not run" in spec["summary"]
    assert [feature["name"] for feature in features_from_spec(spec)] == NAMES


def test_a_model_ending_in_a_drawn_step_gets_no_class_names_made_up():
    class Scores(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(4, 6)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.dense(x).view(-1, 2, 3)

    spec = extract_structure(Scores().eval(), feature_names=NAMES, task="multiclass_classification").spec
    (view,) = _opaque(spec, "view")
    assert view["shape"] == [2, 3] and view["attrs"]["args"][1:] == [-1, 2, 3]
    assert spec["layers"][-1]["id"] == view["id"]
    assert "labels" not in spec and spec["task"] == "multiclass_classification"


def test_an_untraceable_model_draws_its_own_modules():
    spec = extract_structure(Untraceable().eval(), feature_names=NAMES, task="regression").spec
    assert spec["approximate"] is True
    (gate,) = _opaque(spec, "Gate")
    assert gate["path"] == "gate" and gate["shape"] == [8]  # read off the forward pass, so the next dense lines up
    assert gate["attrs"]["temperature"] == 2.0
    assert np.allclose(gate["params"]["bias"], 0.5)
    assert gate["reason"] == "Gate"
    assert [layer["type"] for layer in spec["layers"]] == ["input", "dense", "opaque", "dense"]


def test_refusing_drawn_steps_is_still_possible():
    with pytest.raises(StructureError) as excinfo:
        extract_structure(Untraceable().eval(), feature_names=NAMES, task="regression", on_unsupported="error")
    assert excinfo.value.code == "unsupported_layers"
    assert "Gate at gate" in str(excinfo.value)
    with pytest.raises(StructureError):
        extract_structure(Untraceable().eval(), feature_names=NAMES, on_unsupported="maybe")


def test_drawn_tensors_travel_like_any_other():
    raw = {
        "format": "kaimei.structure.v1",
        "family": "network",
        "task": "regression",
        "feature_names": ["a"],
        "layers": [
            {"id": "input", "type": "input", "names": ["a"]},
            {
                "id": "attn",
                "type": "opaque",
                "op": "Attention",
                "shape": [1, 300],
                "params": {"in_proj_weight": np.ones((300, 300)), "in_proj_bias": np.zeros(300)},
            },
        ],
    }
    assert drawn_steps(raw) == ["Attention at attn"]

    lifted = finalize_spec(raw, externalize_above=1000)
    attn = lifted.spec["layers"][1]
    assert attn["params"]["in_proj_weight"] == {"$tensor": "attn.in_proj_weight", "shape": [300, 300], "dtype": "f32"}
    assert attn["params"]["in_proj_bias"] == [0.0] * 300
    assert lifted.tensors["attn.in_proj_weight"].shape == (300, 300)

    private = finalize_spec(raw, weights="summary")
    ghost = private.spec["layers"][1]["params"]["in_proj_weight"]
    assert ghost["$shape"] == [300, 300] and ghost["stats"]["mean"] == pytest.approx(1.0)
    bias = private.spec["layers"][1]["params"]["in_proj_bias"]
    assert bias["$shape"] == [300] and bias["stats"]["mean"] == 0.0
    assert not private.tensors


def test_a_spec_says_where_it_came_from():
    import kaimei

    spec = extract_structure(Untraceable().eval(), feature_names=NAMES, task="regression", metadata={"run": "exp-42"}).spec
    meta = spec["metadata"]
    assert meta["extracted_with"] == f"kaimei-python-client {kaimei.__version__}"
    assert meta["framework_version"] == torch.__version__ and meta["python"].count(".") == 2
    assert meta["model_class"].endswith(".Untraceable") and meta["extracted_at"].endswith("Z")
    assert meta["description"].startswith("Data-dependent control flow")
    assert meta["run"] == "exp-42"


class Scaled(nn.Module):
    """Reads a stored buffer and a raw parameter, and returns a scalar beside its prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.register_buffer("scale", torch.full((8,), 2.0))
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, x):  # type: ignore[no-untyped-def]
        y = self.fc(x) * self.scale + self.bias
        return y, y.sum()


def test_stored_tensors_and_scalars_keep_their_own_shapes():
    spec = extract_structure(Scaled().eval(), feature_names=NAMES, task="regression").spec
    constants = _opaque(spec, "constant")
    assert sorted(c["shape"] for c in constants) == [[8], [8]]
    assert all(np.shape(c["params"]["value"]) == (8,) for c in constants)
    assert {c["id"] for c in constants} == {"scale", "bias"}
    (total,) = _opaque(spec, "sum")
    assert total["shape"] == [1]
    assert all(layer["shape"] for layer in spec["layers"] if layer["type"] == "opaque")


def test_a_cast_or_a_same_shape_view_does_not_make_a_model_drawn():
    class Cast(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, x):  # type: ignore[no-untyped-def]
            y = self.net(x.double().cpu())
            return torch.nan_to_num(y.view_as(y))

    spec = extract_structure(Cast().double().eval(), feature_names=NAMES, task="regression").spec
    assert "approximate" not in spec
    assert [layer["type"] for layer in spec["layers"]] == ["input", "dense", "activation", "dense"]


def test_keras_attention_keeps_every_weight_under_its_own_name():
    import os

    if os.environ.get("KERAS_BACKEND") is None:
        pytest.skip("set KERAS_BACKEND to run the Keras converter test")
    keras = pytest.importorskip("keras")
    from kaimei.structure import _keras_weight_names, _opaque_keras

    tokens = keras.Input((6, 8))
    layer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, name="mha")
    layer(tokens, tokens)
    names = _keras_weight_names(layer)
    assert len(names) == len(set(names)) == len(layer.weights) == 8
    assert {"query.kernel", "query.bias", "key.kernel", "value.kernel", "attention_output.kernel", "attention_output.bias"} <= set(names)
    drawn = _opaque_keras(layer, "MultiHeadAttention", "MultiHeadAttention")
    assert len(drawn["params"]) == 8 and np.shape(drawn["params"]["query.kernel"]) == (8, 2, 4)
    assert drawn["attrs"]["num_heads"] == 2


def test_upload_structure_adds_metadata_to_a_ready_made_spec():
    import json

    import httpx

    from kaimei import Kaimei

    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"object": "model.structure", "model_id": "mdl_x"})

    client = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer km_test_x"})
    spec = {"format": "kaimei.structure.v1", "family": "network", "task": "regression", "feature_names": ["a"], "layers": [{"id": "out", "type": "dense", "units": 1, "weight": [[1.0]], "bias": [0.0]}], "metadata": {"run": "exp-1"}}
    client.models.upload_structure("mdl_x", spec, metadata={"owner": "me"})
    assert sent["metadata"] == {"run": "exp-1", "owner": "me"}
