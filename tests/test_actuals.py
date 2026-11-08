import json
import math

import httpx
import pytest

from kaimei import Kaimei, display
from kaimei._html import render


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_log_actuals_splits_into_requests_and_adds_up():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/predictions/actuals" and request.method == "POST"
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(
            200,
            json={"object": "actuals", "updated": len(body["items"]) - 1, "cleared": 0, "not_found": [body["items"][0]["prediction_id"]]},
        )

    client = _client(handler)
    result = client.predictions.log_actuals({f"pred_{i}": i % 2 for i in range(2500)}, model="credit")
    assert [len(b["items"]) for b in bodies] == [1000, 1000, 500]
    assert all(b["model"] == "credit" for b in bodies)
    assert result.updated == 2497 and result.not_found == ["pred_0", "pred_1000", "pred_2000"]


def test_unknown_outcomes_are_skipped_and_none_clears():
    np = pytest.importorskip("numpy")
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"object": "actuals", "updated": 2, "cleared": 1, "not_found": []})

    client = _client(handler)
    client.predictions.log_actuals(
        [
            {"prediction_id": "a", "actual": np.int64(1)},
            {"prediction_id": "b", "actual": float("nan")},
            {"prediction_id": "c", "actual": None},
            {"prediction_id": "d", "actual": np.float64(2.5)},
        ]
    )
    assert "model" not in sent
    assert sent["items"] == [
        {"prediction_id": "a", "actual": 1},
        {"prediction_id": "c", "actual": None},
        {"prediction_id": "d", "actual": 2.5},
    ]
    with pytest.raises(ValueError, match="prediction_id"):
        client.predictions.log_actuals([{"actual": 1}])
    with pytest.raises(ValueError, match="No outcomes"):
        client.predictions.log_actuals({"b": math.nan})


def test_series_of_outcomes_and_one_outcome():
    pd = pytest.importorskip("pandas")
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"object": "actuals", "updated": 1, "cleared": 0, "not_found": []})

    client = _client(handler)
    client.predictions.log_actuals(pd.Series(["survived", None, float("nan")], index=["p1", "p2", "p3"]))
    assert sent[-1]["items"] == [{"prediction_id": "p1", "actual": "survived"}]
    client.predictions.log_actual("p9", True)
    assert sent[-1]["items"] == [{"prediction_id": "p9", "actual": True}]
    frame = pd.DataFrame({"prediction_id": ["p1", "p2", "p3"], "actual": [1.5, float("nan"), 0.0]})
    client.predictions.log_actuals(frame)
    assert sent[-1]["items"] == [{"prediction_id": "p1", "actual": 1.5}, {"prediction_id": "p3", "actual": 0.0}]
    with pytest.raises(ValueError, match="columns"):
        client.predictions.log_actuals(pd.DataFrame({"id": ["p1"], "outcome": [1]}))


def test_performance_and_filters_send_their_parameters():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/performance"):
            return httpx.Response(200, json=PERFORMANCE)
        return httpx.Response(200, json={"object": "list", "data": [], "total": 0, "limit": 50, "offset": 0, "has_more": False})

    client = _client(handler)
    report = client.models.performance("credit", window="month", days=180)
    assert seen[-1] == ("/v1/models/credit/performance", {"window": "month", "days": "180"})
    assert report.overall.auc == 0.81
    client.predictions.list(model="credit", has_actual=False)
    assert seen[-1][1]["has_actual"] == "false"
    client.predictions.create(model="credit", input={"x": 1}, actual="decline")


PERFORMANCE = {
    "object": "performance",
    "model_id": "mdl_1",
    "task": "binary_classification",
    "window": "week",
    "primary_metric": "auc",
    "overall": {"n": 300, "auc": 0.81, "accuracy": 0.76},
    "series": [
        {"start": "2026-10-19", "end": "2026-10-26", "n": 150, "auc": 0.84, "accuracy": 0.79},
        {"start": "2026-10-26", "end": "2026-11-02", "n": 150, "auc": 0.78, "accuracy": 0.73},
    ],
}

INSIGHT = {
    "object": "insight",
    "id": "ins_1",
    "model_id": "mdl_1",
    "type": "performance",
    "title": "Performance against outcomes",
    "summary": "AUC on the latest 200 outcomes is 0.702, down from 0.861 on the 700 outcomes before them.",
    "severity": "high",
    "payload": {
        "status": "degraded",
        "metric": "auc",
        "comparison": {"z": -5.2},
        "current": {"n": 200, "auc": 0.702},
        "baseline": {"n": 700, "auc": 0.861},
        "series": PERFORMANCE["series"],
    },
}


def test_performance_records_display_as_tables_and_cards():
    table, columns = display.rows(PERFORMANCE)
    assert [row["start"] for row in table] == ["2026-10-19", "2026-10-26"] and columns[:3] == ["start", "end", "n"]
    table, _ = display.rows(INSIGHT)
    assert len(table) == 2 and table[1]["auc"] == 0.78
    card = render(INSIGHT)
    assert "AUC 0.702" in card and "baseline 0.861" in card and "degraded" in card and "<svg" in card
    prediction = render(
        {"object": "prediction", "id": "p1", "output": {"label": "decline"}, "actual": {"label": "approve"}, "input": {}}
    )
    assert "actual approve" in prediction
