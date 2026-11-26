"""The reference-data summary carries each sampled row's stored label through to the caller."""

import httpx

from kaimei import Kaimei


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


SUMMARY = {
    "model_id": "mdl_1",
    "rows": 300,
    "has_labels": True,
    "sample": [
        {"income": 61.2, "credit_score": 702},
        {"income": 48.0, "credit_score": 611},
        {"income": 90.5, "credit_score": 745},
    ],
    "sample_labels": ["approve", None, "decline"],
    "columns": ["income", "credit_score"],
    "rejected": 0,
    "limit": None,
}


def test_sampled_reference_rows_come_with_their_labels():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json=SUMMARY)

    client = _client(handler)
    summary = client.models.reference_data("loan-approval-forest-v1", sample=3)
    assert seen == [("GET", "/v1/models/loan-approval-forest-v1/reference-data", {"sample": "3"})]
    assert summary.rows == 300 and summary.has_labels is True
    # One label per sampled row, in the rows' order; a row stored without one has None.
    assert list(summary.sample_labels) == ["approve", None, "decline"]
    paired = [(row.credit_score, label) for row, label in zip(summary.sample, summary.sample_labels)]
    assert paired == [(702, "approve"), (611, None), (745, "decline")]


def test_an_older_api_sends_no_labels_and_nothing_breaks():
    older = {key: value for key, value in SUMMARY.items() if key != "sample_labels"}
    summary = _client(lambda request: httpx.Response(200, json=older)).models.reference_data("credit")
    assert len(summary.sample) == 3 and summary.get("sample_labels") is None
