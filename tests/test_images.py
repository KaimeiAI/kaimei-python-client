from __future__ import annotations

import json
import os
import warnings

import httpx
import numpy as np
import pytest

from kaimei import Kaimei, StructureError, estimator_to_spec
from kaimei.structure import (
    MAX_REFERENCE_VALUES,
    cap_reference_rows,
    features_from_spec,
    finalize_spec,
    image_shape_of,
    rows_from_inputs,
)

IMAGE_SPEC = {
    "format": "kaimei.structure.v1",
    "family": "network",
    "task": "binary_classification",
    "feature_names": ["photo"],
    "layers": [{"id": "input", "type": "input", "names": ["photo"], "shape": [4, 5, 3]}],
}


def test_image_reference_rows_accept_channels_last_and_first():
    rng = np.random.default_rng(0)
    hwc = rng.random((2, 4, 5, 3))
    rows, names = rows_from_inputs(hwc, ["photo"], image_shape=[4, 5, 3])
    assert names == ["photo"]
    np.testing.assert_allclose(np.asarray(rows[1]["photo"]), hwc[1].round(6))
    chw_rows, _ = rows_from_inputs(hwc.transpose(0, 3, 1, 2), ["photo"], image_shape=[4, 5, 3])
    assert chw_rows == rows
    gray, _ = rows_from_inputs(rng.random((3, 4, 5)), ["digit"], image_shape=[4, 5, 1])
    assert np.asarray(gray[0]["digit"]).shape == (4, 5, 1)
    with pytest.raises(StructureError, match="4×5×3"):
        rows_from_inputs(rng.random((2, 6, 6, 3)), ["photo"], image_shape=[4, 5, 3])


def test_image_reference_rows_are_capped_by_value_budget():
    rows = [{"photo": [[0.0]]}] * 20
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capped, _ = cap_reference_rows(rows, values_per_row=MAX_REFERENCE_VALUES // 8)
    assert len(capped) == 8
    assert caught


def test_features_from_spec_reads_the_image_input():
    assert image_shape_of(IMAGE_SPEC) == [4, 5, 3]
    feature = features_from_spec(IMAGE_SPEC, value_range=[0, 255])[0]
    assert feature == {"name": "photo", "type": "image", "height": 4, "width": 5, "channels": 3, "value_range": [0.0, 255.0]}


def test_finalize_spec_serializes_normalization_tensors():
    spec = {
        "layers": [
            {"id": "bn", "type": "batchnorm", "mean": np.zeros(3), "var": np.ones(3), "gamma": np.ones(3)},
            {"id": "norm", "type": "normalize", "scale": 1 / 255, "mean": np.array([0.5, 0.5, 0.5])},
        ]
    }
    out = finalize_spec(spec).spec
    json.dumps(out)
    assert out["layers"][0]["var"] == [1.0, 1.0, 1.0]
    assert out["layers"][1]["scale"] == pytest.approx(1 / 255)


def _recording_client(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append((request.method, request.url.path, body))
        if request.url.path == "/v1/models" and request.method == "POST":
            return httpx.Response(201, json={"object": "model", "id": "mdl_cnn", "name": body["name"]})
        return httpx.Response(200, json={"object": "ok"})

    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_torch_cnn_uploads_as_an_image_model():
    torch = pytest.importorskip("torch")
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, stride=2, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.head = nn.Linear(4 * 4 * 4, 3)

        def forward(self, x):
            x = torch.relu(self.bn(self.conv(x)))
            return self.head(torch.flatten(x, 1))

    seen: list = []
    client = _recording_client(seen)
    images = np.random.default_rng(0).random((5, 3, 8, 8))
    client.models.upload(
        Net().eval(),
        name="Tiny CNN",
        task="multiclass_classification",
        input_shape=(3, 8, 8),
        preprocess={"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        X=images,
    )
    created = next(body for method, path, body in seen if path == "/v1/models")
    types = [layer["type"] for layer in created["structure"]["layers"]]
    assert types[:3] == ["input", "normalize", "conv2d"]
    assert "batchnorm" in types and "flatten" in types and types[-1] == "activation"
    assert created["features"] == [{"name": "image", "type": "image", "height": 8, "width": 8, "channels": 3}]
    reference = next(body for method, path, body in seen if path.endswith("/reference-data"))
    assert np.asarray(reference["rows"][0]["image"]).shape == (8, 8, 3)


def test_torch_cnn_needs_an_input_shape_and_refuses_unknown_ops():
    torch = pytest.importorskip("torch")
    from torch import nn

    class Shuffle(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 1)

        def forward(self, x):
            n, c, h, w = x.shape
            x = self.conv(x).view(n, 2, c // 2, h, w).transpose(1, 2).reshape(n, c, h, w)
            return torch.flatten(x, 1)

    with pytest.raises(StructureError, match="input_shape"):
        estimator_to_spec(Shuffle())
    with pytest.raises(StructureError) as caught:
        estimator_to_spec(Shuffle(), input_shape=(4, 6, 6))
    assert caught.value.code == "unsupported_layers"


def test_keras_cnn_spec_keeps_residual_inputs():
    if os.environ.get("KERAS_BACKEND") is None:
        pytest.skip("set KERAS_BACKEND to run the Keras converter test")
    keras = pytest.importorskip("keras")
    layers = keras.layers
    inputs = keras.Input((12, 12, 3))
    x = layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    y = layers.Conv2D(4, 3, padding="same")(x)
    x = layers.Add()([x, y])
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    spec = estimator_to_spec(keras.Model(inputs, outputs))
    add = next(layer for layer in spec["layers"] if layer["type"] == "add")
    assert len(add["inputs"]) == 2
    assert add["inputs"][0].endswith("__relu")
