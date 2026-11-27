import json

import httpx
import pytest

from kaimei import Kaimei, KaimeiError, threshold_for
from kaimei._html import render

CURVE = {
    "threshold": 0.5,
    "rows": 100,
    "prevalence": 0.1,
    "points": [
        {"threshold": 0.0, "tp": 10, "fp": 90, "fn": 0, "tn": 0, "precision": 0.1, "recall": 1.0, "f1": 0.18},
        {"threshold": 0.3, "tp": 9, "fp": 21, "fn": 1, "tn": 69, "precision": 0.3, "recall": 0.9, "f1": 0.45},
        {"threshold": 0.4, "tp": 9, "fp": 11, "fn": 1, "tn": 79, "precision": 0.45, "recall": 0.9, "f1": 0.6},
        {"threshold": 0.5, "tp": 6, "fp": 4, "fn": 4, "tn": 86, "precision": 0.6, "recall": 0.6, "f1": 0.6},
        {"threshold": 0.8, "tp": 2, "fp": 0, "fn": 8, "tn": 90, "precision": 1.0, "recall": 0.2, "f1": 0.33},
        {"threshold": 0.9, "tp": 0, "fp": 0, "fn": 10, "tn": 90, "precision": None, "recall": 0.0, "f1": 0.0},
    ],
}


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_threshold_for_meets_one_target():
    # The most recall while 45% of positive answers are right; at 0.9 recall the higher threshold wins the tie.
    assert threshold_for(CURVE, min_precision=0.45)["threshold"] == 0.4
    assert threshold_for(CURVE, min_recall=0.9)["threshold"] == 0.4
    assert threshold_for(CURVE, min_recall=0.5)["threshold"] == 0.5
    assert threshold_for(CURVE, min_precision=1.0)["threshold"] == 0.8
    assert threshold_for(CURVE, min_recall=1.01) is None
    assert threshold_for(None, min_recall=0.5) is None
    with pytest.raises(ValueError):
        threshold_for(CURVE)
    with pytest.raises(ValueError):
        threshold_for(CURVE, min_precision=0.5, min_recall=0.5)


def test_pr_curve_and_the_metric_the_checks_watch():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, dict(request.url.params), body))
        if request.url.path.endswith("/performance"):
            return httpx.Response(200, json={"object": "performance", "overall": {"n": 100, "pr_curve": CURVE}})
        return httpx.Response(200, json={"object": "model", "id": "mdl_1", "slug": "fraud", "performance_metric": (body or {}).get("performance_metric")})

    client = _client(handler)
    curve = client.models.pr_curve("fraud", days=30)
    assert seen[-1][:3] == ("GET", "/v1/models/fraud/performance", {"window": "week", "days": "30"})
    assert curve["threshold"] == 0.5 and threshold_for(curve, min_recall=0.9)["threshold"] == 0.4

    assert client.models.set_performance_metric("fraud", "recall")["performance_metric"] == "recall"
    assert seen[-1][0] == "PATCH" and seen[-1][3] == {"performance_metric": "recall"}
    # None is sent, not dropped: it goes back to AUC or accuracy.
    client.models.set_performance_metric("fraud", None)
    assert seen[-1][3] == {"performance_metric": None}
    client.models.create(name="Fraud", task="binary_classification", features=[{"name": "x"}], performance_metric="precision")
    assert seen[-1][1] == "/v1/models" and seen[-1][3]["performance_metric"] == "precision"


def test_a_check_watching_recall_shows_as_a_share_beside_precision():
    card = render(
        {
            "object": "insight",
            "id": "ins_2",
            "type": "performance",
            "title": "Performance against outcomes",
            "summary": "Recall on the latest 300 outcomes is 42.0%, down from 71.0%.",
            "severity": "high",
            "payload": {
                "status": "degraded",
                "metric": "recall",
                "comparison": {"z": -7.1},
                "current": {"n": 300, "recall": 0.42, "precision": 0.8, "auc": 0.8},
                "baseline": {"n": 600, "recall": 0.71},
                "series": [],
            },
        }
    )
    assert "recall 42.0%" in card and "baseline 71.0%" in card and "precision 80.0%" in card


def test_an_api_that_ignores_the_metric_is_said_to_have_changed_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "model", "id": "mdl_1", "slug": "fraud"})

    with pytest.raises(KaimeiError, match="nothing changed"):
        _client(handler).models.set_performance_metric("fraud", "recall")
