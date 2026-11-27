"""The audit log, retention and holds, and what was decided beside the model.

The fixtures are what the API returned for the demo workspace: a decision whose record states a cut-off and a
person's review (``decision_record_decided.json``), a page of the audit log, the retention schedule, and holds.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from kaimei import APIObject, Kaimei, display

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


DECIDED = _fixture("decision_record_decided.json")
AUDIT = _fixture("audit_log.json")
RETENTION = _fixture("retention.json")
HOLDS = _fixture("holds.json")
RECORD_HOLDS = _fixture("decision_holds.json")
CSV = "﻿time (UTC),who,kind,action,what happened\r\n".encode("utf-8")


@pytest.fixture
def calls() -> List[httpx.Request]:
    return []


@pytest.fixture
def client(calls: List[httpx.Request]) -> Kaimei:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path, query = request.url.path, dict(request.url.params)
        if path == "/v1/audit-log":
            if query.get("format") == "csv":
                return httpx.Response(
                    200, content=CSV, headers={"Content-Type": "text/csv; charset=utf-8"}
                )
            return httpx.Response(200, json=AUDIT)
        if path == "/v1/retention/run":
            return httpx.Response(
                200,
                json={
                    "object": "retention_run",
                    "dry_run": query.get("dry_run") == "true",
                    "ran_at": "2026-11-26T12:00:00Z",
                    "decisions": 3,
                    "explanations": 0,
                    "models": [],
                },
            )
        if path == "/v1/retention/holds" and request.method == "GET":
            return httpx.Response(200, json=HOLDS)
        if path == "/v1/retention/holds":
            return httpx.Response(201, json=HOLDS["data"][0])
        if path.startswith("/v1/retention/holds/"):
            return httpx.Response(200, json={**HOLDS["data"][0], "active": False})
        if path.startswith("/v1/retention"):
            return httpx.Response(200, json=RETENTION)
        if path == "/v1/decisions/APP-2026-00419" and request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "object": "decision",
                    "id": "APP-2026-00419",
                    "deleted": True,
                    "explanations": 1,
                },
            )
        if path.startswith("/v1/decisions/"):
            return httpx.Response(200, json=DECIDED)
        if path.startswith("/v1/explanations/"):
            return httpx.Response(
                200, json={"object": "explanation", "id": path.rsplit("/", 1)[1], "deleted": True}
            )
        return httpx.Response(404, json={"error": {"type": "not_found_error", "message": path}})

    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def _body(request: httpx.Request) -> Dict[str, Any]:
    return json.loads(request.content.decode("utf-8"))


# --- what was decided ------------------------------------------------------------------------


def test_what_was_decided_is_recorded_beside_a_decision(
    client: Kaimei, calls: List[httpx.Request]
) -> None:
    rules = [
        {
            "name": "Score cut-off",
            "kind": "cutoff",
            "detail": "Decline at 50% or more",
            "effect": "declined",
        },
        {"name": "Recent bankruptcy", "kind": "overlay", "applied": False},
    ]
    record = client.decisions.record_policy("LOAN/2026/0042", outcome="declined", rules=rules)
    sent = calls[-1]
    assert (sent.method, sent.url.raw_path.decode()) == (
        "PUT",
        "/v1/decisions/LOAN/2026/0042/policy",
    )
    assert _body(sent) == {"outcome": "declined", "rules": rules}
    assert isinstance(record, APIObject) and record.policy.outcome == "declined"
    assert record.policy.decided_by == "person" and record.policy.rules[1].kind == "manual"
    client.decisions.record_policy(
        "APP-1", outcome="approved", decided_by="person", note="Seen by committee"
    )
    assert _body(calls[-1]) == {
        "outcome": "approved",
        "decided_by": "person",
        "note": "Seen by committee",
    }
    client.decisions.remove_policy("APP-1")
    assert (calls[-1].method, calls[-1].url.path) == ("DELETE", "/v1/decisions/APP-1/policy")


def test_a_decision_and_an_explanation_can_be_deleted(
    client: Kaimei, calls: List[httpx.Request]
) -> None:
    gone = client.decisions.delete("APP-2026-00419", reason="Erasure request ER-12")
    assert calls[-1].method == "DELETE" and dict(calls[-1].url.params) == {
        "reason": "Erasure request ER-12"
    }
    assert gone.deleted is True and gone.explanations == 1
    client.decisions.delete("APP-2026-00419")
    assert not calls[-1].url.params
    with pytest.raises(ValueError, match="decision id"):
        client.decisions.delete("  ")
    assert client.explanations.delete("exp_123").deleted is True
    assert (calls[-1].method, calls[-1].url.path) == ("DELETE", "/v1/explanations/exp_123")


def test_the_record_card_states_what_was_decided_and_its_custody() -> None:
    card = display.html({**DECIDED, "holds": RECORD_HOLDS})
    shown = card.split('class="km-raw"')[0]
    assert "What was decided" in shown and "Custody of this record" in shown
    for sentence in DECIDED["policy"]["sentences"]:
        assert sentence.split("“")[0] in shown
    assert (
        "Delinquent past or present credit obligations with others" in shown
        and "reason given by a rule" in shown
    )
    assert DECIDED["policy"]["stated"][:40] in shown
    assert (
        "on hold" in shown
        and "CASE-2026-0117" in shown
        and "Customer complaint under review" in shown
    )
    assert DECIDED["history"][0]["summary"].split("“")[0] in shown
    # A record from an API older than these fields shows as it did.
    old = {k: v for k, v in DECIDED.items() if k not in ("policy", "holds", "history")}
    plain = display.html(old).split('class="km-raw"')[0]
    assert "What was decided" not in plain and "Custody of this record" not in plain


# --- the audit log ---------------------------------------------------------------------------


def test_the_audit_log_is_listed_and_narrowed(client: Kaimei, calls: List[httpx.Request]) -> None:
    page = client.audit_log.list(action="decision", actor="demo", limit=8)
    assert dict(calls[-1].url.params) == {
        "action": "decision",
        "actor": "demo",
        "limit": "8",
        "offset": "0",
    }
    assert page.total == 41 and page.has_more and page.data[0].object == "audit_event"
    assert {entry.action for entry in page.data} >= {
        "decision.deleted",
        "hold.placed",
        "retention.applied",
    }
    applied = next(entry for entry in page.data if entry.action == "retention.applied")
    assert applied.actor.type in ("system", "user") and applied.detail.decisions == 3
    # A day means its first instant, and a time without a zone is UTC.
    client.audit_log.list(since=date(2026, 11, 1), until=datetime(2026, 12, 1, 9, 30))
    sent = dict(calls[-1].url.params)
    assert (
        sent["since"] == "2026-11-01T00:00:00+00:00"
        and sent["until"] == "2026-12-01T09:30:00+00:00"
    )
    aware = datetime(2026, 11, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    client.audit_log.list(target_type="decision", target_id="APP-2026-00417", since=aware)
    sent = dict(calls[-1].url.params)
    assert sent["target_id"] == "APP-2026-00417" and sent["since"] == "2026-11-01T12:00:00+02:00"


def test_the_audit_log_shows_as_a_table_and_saves_as_a_spreadsheet(
    client: Kaimei, calls: List[httpx.Request], tmp_path: Path
) -> None:
    card = display.html(AUDIT)
    for entry in AUDIT["data"]:
        assert entry["action"] in card and entry["actor"]["name"] in card
    assert "What happened" in card and "41 audit events" in card
    rows, columns = display.rows(AUDIT)
    assert columns[:4] == ["at", "who", "action", "summary"] and len(rows) == len(AUDIT["data"])
    assert (
        rows[0]["who"] == AUDIT["data"][0]["actor"]["name"]
        and rows[0]["summary"] == AUDIT["data"][0]["summary"]
    )
    assert client.audit_log.csv(action="hold") == CSV
    assert dict(calls[-1].url.params) == {"action": "hold", "format": "csv"}
    saved = client.audit_log.save_csv(tmp_path, since="2026-11-01T00:00:00Z")
    assert saved == tmp_path / "audit-log.csv" and saved.read_bytes() == CSV
    assert dict(calls[-1].url.params) == {"since": "2026-11-01T00:00:00Z", "format": "csv"}


# --- retention and holds ---------------------------------------------------------------------


def test_the_schedule_is_read_set_and_applied(client: Kaimei, calls: List[httpx.Request]) -> None:
    schedule = client.retention.retrieve()
    assert (
        schedule.object == "retention"
        and schedule.default.keep_days == 760
        and schedule.minimum_days == 30
    )
    credit = next(item for item in schedule.models if item.model.slug == "credit-risk-v3")
    assert (
        credit.source == "organization" and credit.due.decisions == 3 and credit.held.decisions == 1
    )
    client.retention.set(760)
    assert (calls[-1].method, calls[-1].url.path, _body(calls[-1])) == (
        "PUT",
        "/v1/retention",
        {"keep_days": 760},
    )
    client.retention.set(None)
    assert _body(calls[-1]) == {
        "keep_days": None
    }  # sent, not left out: it is what turns the schedule off
    client.retention.set_for_model("credit-risk-v3", 2557)
    assert (calls[-1].url.path, _body(calls[-1])) == (
        "/v1/retention/models/credit-risk-v3",
        {"keep_days": 2557},
    )
    client.retention.clear_for_model("credit-risk-v3")
    assert (calls[-1].method, calls[-1].url.path) == (
        "DELETE",
        "/v1/retention/models/credit-risk-v3",
    )
    preview = client.retention.run(dry_run=True)
    assert (
        dict(calls[-1].url.params) == {"dry_run": "true"}
        and preview.dry_run is True
        and preview.decisions == 3
    )
    client.retention.run()
    assert calls[-1].method == "POST" and not calls[-1].url.params


def test_holds_are_placed_listed_and_released(client: Kaimei, calls: List[httpx.Request]) -> None:
    hold = client.retention.place_hold(
        decision="LOAN/2026/0042", reason="Customer complaint", matter="CASE-2026-0117"
    )
    assert _body(calls[-1]) == {
        "decision": "LOAN/2026/0042",
        "reason": "Customer complaint",
        "matter": "CASE-2026-0117",
    }
    assert hold.object == "hold"
    client.retention.place_hold(model="credit-risk-v3", reason="Supervisory review")
    assert _body(calls[-1]) == {"model": "credit-risk-v3", "reason": "Supervisory review"}
    for wrong in ({}, {"decision": "A", "model": "m"}):
        with pytest.raises(ValueError, match="Name what to hold"):
            client.retention.place_hold(reason="x y z", **wrong)
    listed = client.retention.holds(active=True, decision="LOAN/2026/0042")
    assert dict(calls[-1].url.params) == {
        "active": "true",
        "decision": "LOAN/2026/0042",
        "limit": "50",
        "offset": "0",
    }
    assert listed.data[0].reason
    released = client.retention.release_hold(hold.id)
    assert (
        calls[-1].url.path == f"/v1/retention/holds/{hold.id}/release" and released.active is False
    )


def test_the_schedule_and_holds_show_as_cards() -> None:
    card = display.html(RETENTION)
    shown = card.split('class="km-raw"')[0]
    assert "Retention" in shown and "760 days" in shown and "Credit risk scorer" in shown
    assert "3 decisions and 0 explanations due" in shown and "in force" in shown
    assert RETENTION["covers"] in shown and "/retention" in card
    rows, columns = display.rows(RETENTION)
    assert columns[:3] == ["model", "name", "keep_days"] and len(rows) == len(RETENTION["models"])
    credit = next(row for row in rows if row["model"] == "credit-risk-v3")
    assert credit["due_decisions"] == 3 and credit["held_decisions"] == 1
    held = display.html(HOLDS)
    assert "Matter" in held and HOLDS["data"][0]["reason"] in held
    assert ("in force" in held) == any(item["active"] for item in HOLDS["data"])
