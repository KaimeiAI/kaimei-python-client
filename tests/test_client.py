from kaimei import Kaimei, NotFoundError
from kaimei._objects import APIObject
import httpx
import pytest

from kaimei._http import Transport


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/explanations" and request.method == "POST":
        return httpx.Response(
            201,
            json={
                "object": "explanation",
                "id": "exp_1",
                "model_id": "mdl_1",
                "output": {"label": "decline", "score": 0.81},
                "top_features": [{"feature": "credit_score", "attribution": -0.22}],
                "attributions": {"credit_score": -0.22, "income": 0.05},
            },
        )
    if path == "/v1/models" and request.method == "GET":
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "model", "id": "mdl_1", "slug": "credit-risk-v3", "name": "Credit risk"}],
                "total": 1,
                "limit": 50,
                "offset": 0,
                "has_more": False,
            },
        )
    if path == "/v1/models/missing":
        return httpx.Response(
            404,
            json={"error": {"type": "not_found_error", "code": "model_not_found", "message": "Model not found."}},
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


def test_explain_matches_website_snippet(client: Kaimei) -> None:
    explanation = client.explanations.create(
        model="credit-risk-v3",
        prediction_id="pred_8f2a19",
        input={"income": 72000, "credit_score": 681, "utilization": 0.41},
    )
    assert isinstance(explanation, APIObject)
    assert explanation.output.score == 0.81
    assert explanation.top_features[0].feature == "credit_score"


def test_list_models(client: Kaimei) -> None:
    page = client.models.list()
    assert page.total == 1
    assert page.data[0].slug == "credit-risk-v3"


def test_not_found(client: Kaimei) -> None:
    with pytest.raises(NotFoundError) as exc:
        client.models.retrieve("missing")
    assert exc.value.code == "model_not_found"


def test_transport_default() -> None:
    t = Transport(api_key="k", base_url="https://api.kaimei.ai", timeout=5, user_agent="t")
    assert t.base_url == "https://api.kaimei.ai"
    t.close()
