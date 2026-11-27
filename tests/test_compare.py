import httpx

from kaimei import Kaimei, display


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


COMPARISON = {
    "object": "comparison",
    "models": [
        {"id": "mdl_1", "slug": "loan-forest-v1", "name": "Loan forest", "version": "1", "positive": "decline", "threshold": 0.5},
        {"id": "mdl_2", "slug": "loan-scorecard-v1", "name": "Loan scorecard", "version": "1", "positive": "decline", "threshold": 0.5},
    ],
    "source": "reference",
    "available": 500,
    "compared": 3,
    "output": {"kind": "probability", "name": "P(decline)", "classes": ["approve", "decline"], "positive": "decline", "threshold": 0.5},
    "agreement": {"rows": 3, "rate": 0.666667, "same": 2, "different": 1, "correlation": 0.91, "mean_gap": 0.1, "max_gap": 0.38},
    "scores": [{"n": 3, "accuracy": 0.738, "auc": 0.81}, {"n": 3, "accuracy": 0.746, "auc": 0.83}],
    "rows": {
        "id": [0, 1, 204],
        "values": [[0.428627, 0.396763, 0.22029], [0.267818, 0.336154, 0.601714]],
        "decisions": [[0, 0, 0], [0, 0, 1]],
        "actual": [1, 0, None],
        "inputs": {"income": [84.8, 70.5, 41.0], "credit_score": [680, 738, 601]},
    },
    "inputs_complete": True,
    "differ_most": [2, 0, 1],
    "importance": {"features": [{"feature": "credit_score", "shares": [0.42, 0.48]}], "target_labels": [None, None], "rows": [400, 400]},
}


def test_compare_sends_the_other_model_and_reads_as_one_row_per_row_compared():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json=COMPARISON)

    client = _client(handler)
    both = client.models.compare("loan-forest-v1", "loan-scorecard-v1")
    assert seen[-1] == ("GET", "/v1/models/loan-forest-v1/comparison", {"with": "loan-scorecard-v1", "source": "reference", "limit": "500"})
    client.models.compare("loan-forest-v1", "mdl_2", source="predictions", limit=120)
    assert seen[-1][2] == {"with": "mdl_2", "source": "predictions", "limit": "120"}
    assert both.agreement.rate == 0.666667 and both.agreement.different == 1
    assert [model.name for model in both.models] == ["Loan forest", "Loan scorecard"]
    assert [score.auc for score in both.scores] == [0.81, 0.83]
    assert both.importance.features[0].shares == [0.42, 0.48]

    table, columns = display.rows(both)
    assert columns == ["id", "value", "other_value", "gap", "decision", "other_decision", "agree", "actual"]
    assert table[0] == {
        "id": 0, "value": 0.428627, "other_value": 0.267818, "gap": abs(0.267818 - 0.428627),
        "decision": "approve", "other_decision": "approve", "agree": True, "actual": "decline",
        "input:income": 84.8, "input:credit_score": 680,
    }
    assert table[2]["other_decision"] == "decline" and table[2]["agree"] is False and table[2]["actual"] is None
    # The row the two are furthest apart on is the one the API lists first.
    assert max(range(3), key=lambda index: table[index]["gap"]) == both.differ_most[0]


def test_a_comparison_of_two_regressions_has_numbers_and_no_decisions():
    numbers = dict(
        COMPARISON,
        output={"kind": "value", "name": "load", "unit": "MW", "classes": [], "positive": None},
        rows=dict(COMPARISON["rows"], decisions=None, actual=[3.5, None, 2]),
    )
    table, _columns = display.rows(numbers)
    assert [row["actual"] for row in table] == [3.5, None, 2]
    assert all(row["decision"] is None and row["other_decision"] is None and row["agree"] is None for row in table)
    assert table[1]["gap"] == abs(0.336154 - 0.396763)
    # Nothing compared yet reads as no rows, not as an error.
    assert display.rows(dict(COMPARISON, rows={}))[0] == []


def test_a_comparison_is_a_frame_with_the_inputs_as_columns():
    pd = __import__("pytest").importorskip("pandas")
    frame = _client(lambda request: httpx.Response(200, json=COMPARISON)).models.compare("a", "b").to_frame()
    assert list(frame.columns[:4]) == ["id", "value", "other_value", "gap"] and len(frame) == 3
    assert frame.sort_values("gap", ascending=False).iloc[0]["id"] == 204
    assert isinstance(frame, pd.DataFrame) and frame["input:credit_score"].tolist() == [680, 738, 601]
