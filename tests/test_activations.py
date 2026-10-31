"""Activations: the request the client sends and the helpers that decode the byte-packed maps."""

from __future__ import annotations

import base64
import json

import httpx
import numpy as np
import pytest

from kaimei import Kaimei
from kaimei.activations import (
    class_attention,
    decode,
    detailed_stages,
    feature_maps,
    saliency,
    spatial_stages,
    stage,
    unit_values,
)


def _packed(array: np.ndarray) -> dict:
    lo, hi = float(array.min()), float(array.max())
    scaled = np.round((array - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros(array.shape, np.uint8)
    return {"shape": list(array.shape), "range": [lo, hi], "encoding": "u8", "data": base64.b64encode(scaled.tobytes()).decode()}


def _response(cam: bool = False) -> dict:
    rng = np.random.default_rng(0)
    relu = rng.normal(size=(4, 4, 3))
    relu[:, :, 1] = np.maximum(relu[:, :, 1], 0)
    energy = np.abs(relu).sum(axis=(0, 1))
    order = np.argsort(-energy)
    stage_relu = {
        "index": 2,
        "node_id": "conv__relu",
        "label": "Edges",
        "op": "relu",
        "kind": "activation",
        "units": 48,
        "intensity": 0.7,
        "summary": {"size": 48, "max_abs": 2.0, "mean_abs": 0.8, "l2": 6.0, "active": 30, "argmax": 5},
        "active_fraction": 0.625,
        "detailed": True,
        "shape": [4, 4, 3],
        "top_channels": [int(i) for i in order],
        "energy": [float(e) for e in energy],
        "maps": [{"channel": int(c), "energy": float(energy[c]), "peak": float(np.abs(relu[:, :, c]).max()), **_packed(relu[:, :, c])} for c in order[:2]],
        "saliency": _packed(np.abs(relu).mean(axis=2)),
    }
    if cam:
        stage_relu["cam"] = {"available": True, "layer": "conv__relu", "layer_label": "Edges", "map_shape": [4, 4, 3], "score": 1.2, "seconds": 0.01, **_packed(np.linspace(0, 1, 64).reshape(8, 8))}
    head = np.array([0.2, 0.8])
    return {
        "object": "model.structure.activations",
        "model_id": "mdl_1",
        "runnable": True,
        "output": {"label": "dog", "probability": 0.8, "probabilities": {"cat": 0.2, "dog": 0.8}},
        "target": {"index": 1, "label": "dog"},
        "input": {"node_id": "photo", "shape": [8, 8, 3], "feature": "photo"},
        "stages": [
            {"index": 0, "node_id": "photo", "label": "Image", "op": "input", "kind": "input", "units": 192, "intensity": 0.5, "summary": {"size": 192, "max_abs": 1, "mean_abs": 0.5, "l2": 8, "active": 192, "argmax": 0}, "active_fraction": 1.0, "detailed": False, "shape": [8, 8, 3], "top_channels": [0, 1, 2]},
            {"index": 1, "node_id": "conv", "label": "Edges", "op": "conv2d", "kind": "linear", "units": 48, "intensity": 0.7, "summary": {"size": 48, "max_abs": 2.0, "mean_abs": 0.8, "l2": 6.0, "active": 30, "argmax": 5}, "active_fraction": 0.6, "detailed": False, "shape": [4, 4, 3], "top_channels": [1, 0, 2]},
            stage_relu,
            {"index": 3, "node_id": "head__softmax", "label": "Output", "op": "softmax", "kind": "output", "units": 2, "intensity": 0.8, "summary": {"size": 2, "max_abs": 0.8, "mean_abs": 0.5, "l2": 0.82, "active": 2, "argmax": 1}, "active_fraction": 1.0, "detailed": True, "top_units": [{"index": 1, "value": 0.8, "label": "dog"}, {"index": 0, "value": 0.2, "label": "cat"}], "values": _packed(head)},
        ],
        "detailed": ["conv__relu", "head__softmax"],
        "channels": 2,
        "max_side": 56,
        "forward_ms": 1.5,
        "_relu": relu.tolist(),
    }


def _client(seen: list, payload: dict) -> Kaimei:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=payload)

    client = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return client


def test_activations_sends_only_what_was_asked():
    seen: list = []
    client = _client(seen, _response())
    client.models.activations("cats-dogs", input={"photo": [[[0.1, 0.2, 0.3]]]})
    assert seen[-1][:2] == ("POST", "/v1/models/cats-dogs/structure/activations")
    assert seen[-1][2] == {"input": {"photo": [[[0.1, 0.2, 0.3]]]}}
    client.models.activations("cats-dogs", input={"photo": []}, nodes=["conv__relu"], cam=["conv__relu"], channels=4, max_side=28, units=8, target="dog")
    assert seen[-1][2] == {
        "input": {"photo": []},
        "nodes": ["conv__relu"],
        "cam": ["conv__relu"],
        "channels": 4,
        "max_side": 28,
        "units": 8,
        "target": "dog",
    }


def test_helpers_decode_maps_within_half_a_step():
    payload = _response(cam=True)
    relu = np.asarray(payload.pop("_relu"))
    result = _client([], payload).models.activations("cats-dogs", input={"photo": []})
    assert [s["node_id"] for s in detailed_stages(result)] == ["conv__relu", "head__softmax"]
    assert [s["node_id"] for s in spatial_stages(result)] == ["conv", "conv__relu"]
    last = stage(result)
    assert last["node_id"] == "conv__relu"
    assert stage(result, "head__softmax")["op"] == "softmax"
    with pytest.raises(KeyError):
        stage(result, "nope")

    maps = feature_maps(last)
    assert list(maps) == last["top_channels"][:2]
    for channel, grid in maps.items():
        step = (np.abs(relu[:, :, channel]).max() * 2) / 255
        assert np.abs(grid - relu[:, :, channel]).max() <= step / 2 + 1e-9
    where = saliency(last)
    assert where.shape == (4, 4)
    assert np.abs(where - np.abs(relu).mean(axis=2)).max() <= np.abs(relu).mean(axis=2).max() / 255 / 2 + 1e-9
    heat = class_attention(last)
    assert heat.shape == (8, 8) and heat.min() == pytest.approx(0.0) and heat.max() == pytest.approx(1.0)
    assert class_attention(stage(result, "conv")) is None
    values = unit_values(stage(result, "head__softmax"))
    assert values == pytest.approx([0.2, 0.8], abs=1 / 255)
    assert unit_values(last) is None
    flat = decode({"shape": [3], "range": [2.0, 2.0], "encoding": "u8", "data": base64.b64encode(bytes(3)).decode()})
    assert flat.tolist() == [2.0, 2.0, 2.0]


def test_plot_activations_draws_a_strip():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from kaimei.activations import plot_activations

    payload = _response()
    payload.pop("_relu")
    result = _client([], payload).models.activations("cats-dogs", input={"photo": []})
    figure = plot_activations(result, image=np.full((8, 8, 3), 0.5), channels=2)
    # Input column + one detailed map column; one saliency row + two channel rows.
    assert len(figure.axes) == 2 * 3
    assert figure._suptitle.get_text() == "dog · P = 0.800"
    plain = plot_activations(result)
    assert len(plain.axes) == 1 * 3
