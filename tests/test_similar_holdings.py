import json

import httpx
import pytest

from kaimei import Kaimei, display


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


SIMILAR = {
    "object": "similar_cases",
    "model_id": "mdl_1",
    "source": "reference",
    "k": 2,
    "considered": 600,
    "prediction_id": None,
    "input": {"income": 52000, "home": "rent"},
    "spread": {"from": "reference", "rows": 600, "widths": {"income": 105177.0}},
    "cases": [
        {
            "id": 251,
            "distance": 0.0102,
            "similarity": 0.9898,
            "inputs": {"income": 49865.5, "home": "rent"},
            "gaps": {"income": 0.0203, "home": 0.0},
            "differs": [{"feature": "income", "value": 49865.5, "asked": 52000, "distance": 0.0203}],
            "label": "approve",
        },
        {
            "id": 377,
            "distance": 0.5379,
            "similarity": 0.4621,
            "inputs": {"income": 44023.9, "home": "own"},
            "gaps": {"income": 0.0758, "home": 1.0},
            "differs": [
                {"feature": "home", "value": "own", "asked": "rent", "distance": 1.0},
                {"feature": "income", "value": 44023.9, "asked": 52000, "distance": 0.0758},
            ],
            "label": "decline",
        },
    ],
    "summary": {
        "cases": 2,
        "labels": {"known": 2, "counts": {"approve": 1, "decline": 1}, "mean": None, "min": None, "max": None},
        "predicted": None,
        "outcomes": None,
        "sentence": "Of the 2 reference rows most like this case, 1 was labelled approve and 1 decline.",
    },
}


def test_similar_sends_the_case_and_reads_as_one_row_per_case():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=SIMILAR)

    client = _client(handler)
    answer = client.models.similar("credit", {"income": 52000, "home": "rent", "age": float("nan")})
    assert seen[-1] == (
        "POST",
        "/v1/models/credit/similar",
        {"input": {"income": 52000, "home": "rent", "age": None}, "k": 10, "source": "reference"},
    )
    client.models.similar("credit", prediction_id="pred_9", k=3, source="predictions")
    assert seen[-1][2] == {"prediction_id": "pred_9", "k": 3, "source": "predictions"}
    assert answer.summary.sentence.startswith("Of the 2 reference rows")
    assert answer.cases[0].similarity == 0.9898 and answer.cases[1].differs[0].feature == "home"
    table, columns = display.rows(answer)
    assert columns == ["id", "similarity", "distance", "label", "differs"]
    assert table[0] == {
        "id": 251, "similarity": 0.9898, "distance": 0.0102, "label": "approve", "differs": "income",
        "input:income": 49865.5, "input:home": "rent",
    }  # fmt: skip
    assert table[1]["differs"] == "home, income" and table[1]["input:home"] == "own"


def test_similar_needs_one_case_and_a_logged_prediction_has_its_own_columns():
    client = _client(lambda request: httpx.Response(500))
    with pytest.raises(ValueError):
        client.models.similar("credit")
    with pytest.raises(ValueError):
        client.models.similar("credit", {"income": 1}, "pred_9")
    logged = dict(
        SIMILAR,
        source="predictions",
        cases=[
            {
                "id": "pred_7",
                "distance": 0.1,
                "similarity": 0.9,
                "inputs": {"income": 50000},
                "differs": [],
                "time": "2026-11-25T10:00:00Z",
                "output": {"label": "decline"},
                "predicted": "decline",
                "value": 0.81,
                "outcome": None,
            }
        ],
    )
    table, columns = display.rows(logged)
    assert columns == ["id", "similarity", "distance", "predicted", "value", "outcome", "time", "differs"]
    assert table == [
        {
            "id": "pred_7", "similarity": 0.9, "distance": 0.1, "predicted": "decline", "value": 0.81, "outcome": None,
            "time": "2026-11-25T10:00:00Z", "differs": "", "input:income": 50000,
        }
    ]  # fmt: skip
    assert display.rows(dict(SIMILAR, cases=[]))[0] == []


def test_holdings_reads_as_one_row_per_item():
    seen = []
    holdings = {
        "object": "model_holdings",
        "model_id": "mdl_1",
        "as_of": "2026-11-26T12:00:00Z",
        "runs": {"by": "your_endpoint", "sentence": "Your endpoint at https://models.example.com runs this model."},
        "items": [
            {"key": "structure", "label": "Weights and structure", "held": "no", "count": 0, "unit": None, "bytes": None, "detail": "No structure and no weights have been uploaded."},
            {"key": "reference_rows", "label": "Reference rows", "held": "yes", "count": 600, "unit": "row", "bytes": 157690, "detail": "600 rows of example inputs, each with its label."},
            {"key": "credential", "label": "Credential for your endpoint", "held": "yes", "count": 1, "unit": "header", "bytes": None, "detail": "1 request header stored sealed."},
        ],
        "crosses": [{"when": "predict", "what": "The row you send to Kaimei is posted on to your endpoint."}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json=holdings)

    answer = _client(handler).models.holdings("credit")
    assert seen == [("GET", "/v1/models/credit/holdings")]
    assert answer.runs.by == "your_endpoint" and answer.items[1].count == 600 and answer.crosses[0].when == "predict"
    table, columns = display.rows(answer)
    assert columns == ["key", "label", "held", "count", "unit", "bytes", "detail"]
    assert [row["key"] for row in table] == ["structure", "reference_rows", "credential"]
    assert table[1] == {
        "key": "reference_rows", "label": "Reference rows", "held": "yes", "count": 600, "unit": "row", "bytes": 157690,
        "detail": "600 rows of example inputs, each with its label.",
    }  # fmt: skip
