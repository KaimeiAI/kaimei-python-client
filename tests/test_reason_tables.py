"""Reason tables: drafting the approved wording, reading the version in force, and a record's draft reasons.

``fixtures/reason_table.json`` is version 1 of the demo credit model's table as the API returned it, and
``fixtures/decision_record.json`` a declined decision whose reasons were drawn with it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from kaimei import APIObject, Kaimei, PermissionDeniedError, display

FIXTURES = Path(__file__).parent / "fixtures"
TABLE = json.loads((FIXTURES / "reason_table.json").read_text(encoding="utf-8"))
RECORD = json.loads((FIXTURES / "decision_record.json").read_text(encoding="utf-8"))
DRAFT = {
    **TABLE,
    "id": "rt_draft",
    "version": None,
    "status": "draft",
    "approved_at": None,
    "approved_by": None,
}


@pytest.fixture
def calls() -> List[httpx.Request]:
    return []


@pytest.fixture
def client(calls: List[httpx.Request]) -> Kaimei:
    def page(items: List[Dict[str, Any]]) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": items,
                "total": len(items),
                "limit": 50,
                "offset": 0,
                "has_more": False,
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path, query = request.url.path, dict(request.url.params)
        if path == "/v1/reason-tables" and request.method == "GET":
            if query.get("model") == "no-table":
                return page([])
            status = query.get("status")
            return page([t for t in (TABLE, DRAFT) if status is None or t["status"] == status])
        if path == "/v1/reason-tables/draft" and request.method == "PUT":
            return httpx.Response(200, json=DRAFT)
        if path == "/v1/reason-tables/rt_draft/approve":
            return httpx.Response(
                403,
                json={
                    "error": {
                        "type": "permission_error",
                        "code": "user_session_required",
                        "message": "Takes a person.",
                    }
                },
            )
        if path == "/v1/reason-tables/rt_draft" and request.method == "DELETE":
            return httpx.Response(
                200, json={"id": "rt_draft", "object": "reason_table", "deleted": True}
            )
        if path == f"/v1/reason-tables/{TABLE['id']}":
            return httpx.Response(200, json=TABLE)
        if path == "/v1/decisions/APP-2026-00417":
            return httpx.Response(200, json=RECORD)
        return httpx.Response(404, json={"error": {"type": "not_found_error", "message": path}})

    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def _sent(request: httpx.Request) -> Dict[str, Any]:
    return json.loads(request.content)


def test_a_draft_is_sent_as_the_api_stores_it(client: Kaimei, calls: List[httpx.Request]) -> None:
    draft = client.reason_tables.save_draft(
        "credit-risk-v3",
        adverse_outcome="decline",
        entries={
            "debt_to_income": "Excessive obligations in relation to income",
            "num_delinquencies": {
                "reason": "Delinquent past or present credit obligations with others",
                "code": "D2",
            },
        },
        never_state=["age"],
        note="First wording.",
    )
    assert draft.status == "draft" and draft.version is None
    assert calls[-1].method == "PUT" and calls[-1].url.path == "/v1/reason-tables/draft"
    assert _sent(calls[-1]) == {
        "model": "credit-risk-v3",
        "adverse": {"outcome": "decline"},
        "max_reasons": 4,
        "entries": {
            "debt_to_income": {"reason": "Excessive obligations in relation to income"},
            "num_delinquencies": {
                "reason": "Delinquent past or present credit obligations with others",
                "code": "D2",
            },
        },
        "never_state": {"age": ""},
        "note": "First wording.",
    }
    # A model that predicts a number names a direction; a word on why an input is never stated is kept.
    client.reason_tables.save_draft(
        "energy",
        adverse_direction="higher",
        entries={"temperature": "Outdoor temperature"},
        never_state={"zone": "policy"},
    )
    assert _sent(calls[-1])["adverse"] == {"direction": "higher"} and _sent(calls[-1])[
        "never_state"
    ] == {"zone": "policy"}
    assert "note" not in _sent(calls[-1])
    for wrong in ({}, {"adverse_outcome": "decline", "adverse_direction": "higher"}):
        with pytest.raises(ValueError, match="adverse_outcome"):
            client.reason_tables.save_draft("credit-risk-v3", entries={"a": "b"}, **wrong)


def test_the_version_in_force_and_the_draft_are_read_by_model(
    client: Kaimei, calls: List[httpx.Request]
) -> None:
    in_force = client.reason_tables.in_force("credit-risk-v3")
    assert dict(calls[-1].url.params) == {
        "model": "credit-risk-v3",
        "status": "approved",
        "limit": "1",
        "offset": "0",
    }
    assert in_force is not None and in_force.version == 1 and in_force.adverse.outcome == "decline"
    assert in_force.coverage.unmapped == ["income", "utilization", "loan_amount", "home_ownership"]
    draft = client.reason_tables.draft("credit-risk-v3")
    assert draft is not None and draft.id == "rt_draft"
    assert (
        client.reason_tables.in_force("no-table") is None
        and client.reason_tables.draft("no-table") is None
    )
    assert [table.status for table in client.reason_tables.list().data] == ["approved", "draft"]
    assert client.reason_tables.retrieve(TABLE["id"]).content_hash == TABLE["content_hash"]


def test_approving_takes_a_person_and_a_draft_can_be_discarded(
    client: Kaimei, calls: List[httpx.Request]
) -> None:
    with pytest.raises(PermissionDeniedError) as refused:
        client.reason_tables.approve("rt_draft")
    assert refused.value.code == "user_session_required"
    assert calls[-1].method == "POST" and calls[-1].url.path == "/v1/reason-tables/rt_draft/approve"
    assert client.reason_tables.discard("rt_draft").deleted is True
    assert calls[-1].method == "DELETE"


def test_a_record_can_be_asked_for_a_drafts_wording(
    client: Kaimei, calls: List[httpx.Request]
) -> None:
    record = client.decisions.retrieve("APP-2026-00417", reason_table="draft")
    assert dict(calls[-1].url.params) == {"reason_table": "draft"}
    client.decisions.retrieve("APP-2026-00417", reason_table=2)
    assert dict(calls[-1].url.params) == {"reason_table": "2"}
    reasons = record.reasons
    assert reasons.applies is True and reasons.table.version == 1
    assert [item.reason for item in reasons.reasons][:2] == [
        "Excessive obligations in relation to income",
        "Delinquent past or present credit obligations with others",
    ]
    assert {item.feature for item in reasons.held} == {"home_ownership", "age"}


def test_a_records_card_states_its_reasons_and_what_is_held() -> None:
    card = display.html(RECORD).split('class="km-raw"')[0]
    assert "Draft reasons" in card
    assert "4 draft reasons for decline, strongest first. 2 inputs are held for review" in card
    assert "<b>Excessive obligations in relation to income</b>" in card
    assert (
        "Housing status pushed toward decline and has no approved reason. Hold for review." in card
    )
    assert "must never be given as a reason" in card
    assert "reason table version 1" in card
    # A record of a model with no table has no such section.
    assert (
        "Draft reasons" not in display.html(dict(RECORD, reasons=None)).split('class="km-raw"')[0]
    )


def test_a_reason_table_shows_as_a_card_and_a_table() -> None:
    card = display.html(TABLE)
    assert "Reason table" in card and "Version 1" in card and "reasons for decline" in card
    assert "Excessive obligations in relation to income" in card and "never state" in card
    assert "4 inputs undecided" in card and "home_ownership" in card
    assert "/models/credit-risk-v3/decisions?view=reasons" in card
    assert "A draft changes nothing" in display.html(DRAFT)
    rows, columns = display.rows(TABLE)
    assert columns == ["input", "treatment", "reason", "code"]
    by_input = {row["input"]: row for row in rows}
    assert by_input["debt_to_income"]["treatment"] == "reason"
    assert (
        by_input["age"]["treatment"] == "never state"
        and by_input["income"]["treatment"] == "undecided"
    )
    assert len(rows) == 9
    assert isinstance(APIObject(TABLE)._repr_mimebundle_()[0]["text/html"], str)
