"""Optional live-API checks. Skipped unless KAIMEI_API_KEY is set."""

from __future__ import annotations

import os

import numpy as np
import pytest

from kaimei import Kaimei

pytestmark = pytest.mark.integration


@pytest.fixture
def live_client() -> Kaimei:
    key = os.environ.get("KAIMEI_API_KEY")
    if not key:
        pytest.skip("KAIMEI_API_KEY is not set")
    return Kaimei(api_key=key, base_url=os.environ.get("KAIMEI_BASE_URL"))


def test_live_health(live_client: Kaimei) -> None:
    assert live_client.health().status == "ok"


def test_live_upload_linear(live_client: Kaimei) -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    del sklearn
    rng = np.random.default_rng(1)
    X = rng.normal(size=(32, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    model = live_client.models.upload(
        clf,
        name="Client live linear",
        feature_names=["a", "b"],
        X=X[:10],
        y=y[:10],
    )
    assert model.id
    explanation = live_client.explanations.create(
        model=model.id,
        input={"a": float(X[0, 0]), "b": float(X[0, 1])},
        method="shapley",
    )
    assert explanation.output
    live_client.models.delete(model.id)
