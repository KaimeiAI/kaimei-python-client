from __future__ import annotations

import json
import warnings

import httpx
import numpy as np
import pytest

from kaimei import Kaimei, StructureError, estimator_to_spec
from kaimei.structure import MAX_REFERENCE_ROWS, cap_reference_rows, spec_is_executable


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
        raise AssertionError("upload() must not PUT structure after POST already included it")
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
    assert spec_is_executable(spec)


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


def test_cap_reference_rows_warns():
    rows = [{"x": i} for i in range(MAX_REFERENCE_ROWS + 10)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capped, labels = cap_reference_rows(rows, list(range(len(rows))))
    assert len(capped) == MAX_REFERENCE_ROWS
    assert labels is not None and len(labels) == MAX_REFERENCE_ROWS
    assert any("Reference data" in str(item.message) for item in caught)


def test_summary_spec_is_not_executable():
    assert not spec_is_executable({"family": "ensemble_summary", "importances": [0.5, 0.5]})
    assert not spec_is_executable({"family": "network", "approximate": True})


def test_sklearn_logistic_to_spec():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    del sklearn
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    spec = estimator_to_spec(clf, feature_names=["a", "b"])
    assert spec["family"] == "network"
    assert spec["layers"][1]["type"] == "dense"
    assert spec_is_executable(spec)
