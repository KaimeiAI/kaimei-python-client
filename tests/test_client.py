import json

import httpx
import pytest

from kaimei import (
    APIConnectionError,
    Kaimei,
    NotFoundError,
    PermissionDeniedError,
    PermissionError,
    __version__,
)
from kaimei._http import Transport
from kaimei._objects import APIObject


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
    if path == "/health":
        return httpx.Response(200, json={"status": "ok", "service": "kaimei-api"})
    if path == "/v1/auth/signup":
        body = json.loads(request.content)
        assert "turnstile_token" in body
        return httpx.Response(201, json={"object": "session", "access_token": "jwt", "user": {"email": body["email"]}})
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


def test_version_is_030() -> None:
    assert __version__ == "0.6.3"


def test_permission_denied_alias() -> None:
    assert PermissionError is PermissionDeniedError
    err = PermissionDeniedError("no", code="forbidden", status_code=403)
    assert "no" in str(err)


def test_health(client: Kaimei) -> None:
    assert client.health().status == "ok"


def test_signup_sends_turnstile(client: Kaimei) -> None:
    session = client.auth.signup(email="ada@example.com", password="correct-horse", turnstile_token="tok")
    assert session.access_token == "jwt"


def test_retries_on_429() -> None:
    hits = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] < 3:
            return httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "slow down"}}, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"object": "model", "id": "mdl_ok"})

    client = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai", max_retries=3)
    client._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(flaky),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    model = client.models.retrieve("ok")
    assert model.id == "mdl_ok"
    assert hits["n"] == 3
    client.close()


def _flaky_client(error: Exception, hits: dict) -> Kaimei:
    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            raise error
        return httpx.Response(200, json={"object": "list", "data": [], "count": 0})

    client = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai", max_retries=3)
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(handler))
    return client


def test_a_post_that_timed_out_is_not_sent_again() -> None:
    """Regression: a slow batch POST timed out client-side and was resent three times; all four
    copies were logged."""
    hits = {"n": 0}
    client = _flaky_client(httpx.ReadTimeout("slow"), hits)
    with pytest.raises(APIConnectionError, match="not retried"):
        client.predictions.create_batch(model="m", items=[{"input": {"x": 1}}])
    assert hits["n"] == 1
    client.close()


def test_requests_that_never_left_or_are_safe_to_repeat_are_retried() -> None:
    hits = {"n": 0}
    client = _flaky_client(httpx.ConnectError("refused"), hits)
    client.predictions.create_batch(model="m", items=[{"input": {"x": 1}}])
    assert hits["n"] == 2
    client.close()

    hits = {"n": 0}
    client = _flaky_client(httpx.ReadTimeout("slow"), hits)
    client.models.list()
    assert hits["n"] == 2
    client.close()


def test_auto_paging() -> None:
    def pages(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset") or 0)
        if offset == 0:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "model", "id": "mdl_a"}],
                    "total": 2,
                    "limit": 1,
                    "offset": 0,
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "model", "id": "mdl_b"}],
                "total": 2,
                "limit": 1,
                "offset": 1,
                "has_more": False,
            },
        )

    client = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(pages),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    ids = [item.id for item in client.models.iterate(limit=1)]
    assert ids == ["mdl_a", "mdl_b"]
    client.close()
