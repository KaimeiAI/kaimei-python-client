from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from kaimei import Kaimei, StructureError, estimator_to_spec


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/models" and request.method == "POST":
        body = json.loads(request.content)
        assert body["structure"]["layers"]
        return httpx.Response(
            201,
            json={
                "object": "model",
                "id": "mdl_up",
                "slug": body.get("slug") or "churn-net",
                "name": body["name"],
                "framework": body.get("framework") or "sklearn",
                "task": body.get("task") or "binary_classification",
                "features": body.get("features") or [],
            },
        )
    if path == "/v1/models/mdl_up/structure" and request.method == "PUT":
        return httpx.Response(
            200,
            json={"object": "model.structure", "source": "upload", "fidelity": "exact", "model_id": "mdl_up"},
        )
    if path == "/v1/models/mdl_up/reference-data" and request.method == "PUT":
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"object": "reference_data", "model_id": "mdl_up", "rows": len(body["rows"])},
        )
    return httpx.Response(404, json={"error": {"type": "not_found_error", "message": path}})


@pytest.fixture
def client() -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(_handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_estimator_to_spec_reads_linear_weights():
    class Linear:
        coef_ = np.array([[0.25, -0.5]])
        intercept_ = np.array([0.1])
        classes_ = np.array(["no", "yes"])
        n_features_in_ = 2

    spec = estimator_to_spec(Linear(), feature_names=["income", "score"])
    assert spec["task"] == "binary_classification"
    assert spec["layers"][1]["weight"] == [[0.25, -0.5]]
    assert spec["layers"][1]["bias"] == [0.1]


def test_estimator_to_spec_reads_keras_dense():
    class Dense:
        name = "hidden"
        units = 1
        activation = type("act", (), {"__name__": "sigmoid"})()

        def get_weights(self):
            return [np.array([[0.5], [-1.0]]), np.array([0.1])]

    class Sequential:
        layers = [Dense()]
        input_shape = (None, 2)

    spec = estimator_to_spec(Sequential(), feature_names=["a", "b"])
    assert spec["architecture"] == "mlp"
    assert spec["layers"][1]["weight"] == [[0.5, -1.0]]


def test_upload_sends_structure_and_reference_rows(client: Kaimei):
    class Linear:
        coef_ = np.array([[0.25, -0.5]])
        intercept_ = np.array([0.1])
        classes_ = np.array(["no", "yes"])
        feature_names_in_ = np.array(["income", "score"])

    model = client.models.upload(
        Linear(),
        name="Churn net",
        X=[[72000, 681], [41000, 540]],
        y=["no", "yes"],
    )
    assert model.id == "mdl_up"
    assert model.framework == "sklearn"


def test_unknown_estimator_raises():
    class Unknown:
        n_features_in_ = 2

    with pytest.raises(StructureError) as exc:
        estimator_to_spec(Unknown(), feature_names=["a", "b"])
    assert exc.value.code == "unsupported_estimator"
