"""Decision records: finding a decision by its id, explaining it, and saving its record.

The record in ``fixtures/decision_record.json`` is one the API returned for the documentation's applicant
(a 59.5% chance of decline, explained with KernelSHAP, seed 0, age held fixed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from kaimei import APIObject, Kaimei, NotFoundError, display
from kaimei.resources.decisions import record_filename

RECORD = json.loads((Path(__file__).parent / "fixtures" / "decision_record.json").read_text(encoding="utf-8"))
PDF = b"%PDF-1.4\n% a decision record\n%%EOF\n"


@pytest.fixture
def calls() -> List[httpx.Request]:
    return []


@pytest.fixture
def client(calls: List[httpx.Request]) -> Kaimei:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path, query = request.url.raw_path.decode().partition("?")[0], dict(request.url.params)
        if path == "/v1/decisions" and request.method == "GET":
            item = {
                "object": "decision",
                "id": RECORD["id"],
                "model_id": "mdl_credit",
                "decided_at": RECORD["decided_at"],
                "source": "api",
                "outcome": {"label": "decline", "probability": 0.5947, "value": 0.5947},
                "explained": True,
                "explanations": 1,
                "explained_at": RECORD["explanation"]["computed_at"],
                "has_actual": False,
            }
            return httpx.Response(200, json={"object": "list", "data": [item], "total": 1, "limit": 20, "offset": 0, "has_more": False})
        if path.endswith("/explanation") and request.method == "POST":
            return httpx.Response(201, json=RECORD)
        if path in ("/v1/decisions/APP-2026-00417", "/v1/decisions/LOAN/2026/0042", "/v1/decisions/a%20b%3Fc") and request.method == "GET":
            if query.get("format") == "pdf":
                return httpx.Response(200, content=PDF, headers={"Content-Type": "application/pdf"})
            return httpx.Response(200, json=RECORD)
        return httpx.Response(
            404, json={"error": {"type": "not_found_error", "code": "decision_not_found", "message": "No decision with that id."}}
        )

    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def _sent(request: httpx.Request) -> Dict[str, Any]:
    return json.loads(request.content) if request.content else {}


def test_a_record_is_read_by_the_id_the_decision_was_logged_under(client: Kaimei, calls: List[httpx.Request]) -> None:
    record = client.decisions.retrieve("APP-2026-00417")
    assert isinstance(record, APIObject)
    assert record.status == "explained"
    assert record.outcome.sentence.startswith("The model gave this case a 59.5% chance of decline.")
    assert record.explanation.method_name == "KernelSHAP" and record.explanation.reproducible is True
    assert record.factors[0].sentence.endswith("raised the chance of decline by 8.0 percentage points.")
    assert len(record.fingerprint.value) == 64
    assert calls[-1].url.path == "/v1/decisions/APP-2026-00417" and not calls[-1].url.params

    client.decisions.retrieve("APP-2026-00417", explanation="exp_9")
    assert dict(calls[-1].url.params) == {"explanation": "exp_9"}


def test_an_id_keeps_its_slashes_and_has_everything_else_escaped(client: Kaimei, calls: List[httpx.Request]) -> None:
    client.decisions.retrieve("LOAN/2026/0042")
    assert calls[-1].url.raw_path == b"/v1/decisions/LOAN/2026/0042"
    client.decisions.retrieve("a b?c")
    assert calls[-1].url.raw_path == b"/v1/decisions/a%20b%3Fc"
    with pytest.raises(ValueError, match="decision id"):
        client.decisions.retrieve("  ")
    with pytest.raises(NotFoundError):
        client.decisions.retrieve("NO-SUCH-ID")


def test_decisions_are_listed_with_their_filters(client: Kaimei, calls: List[httpx.Request]) -> None:
    page = client.decisions.list(model="credit-risk-v3", q="APP-2026", explained=False, limit=5)
    assert dict(calls[-1].url.params) == {"model": "credit-risk-v3", "q": "APP-2026", "explained": "false", "limit": "5", "offset": "0"}
    assert page.data[0].id == "APP-2026-00417" and page.data[0].explained is True
    assert [item.id for item in client.decisions.iterate(explained=True)] == ["APP-2026-00417"]
    assert "explained" in display.html(page) and "APP-2026-00417" in display.html(page)


def test_explaining_sends_only_what_was_chosen(client: Kaimei, calls: List[httpx.Request]) -> None:
    record = client.decisions.explain("APP-2026-00417")
    assert calls[-1].method == "POST" and calls[-1].url.path == "/v1/decisions/APP-2026-00417/explanation"
    assert calls[-1].content == b""  # nothing chosen: the API explains it its recommended way
    assert record.object == "decision_record"

    client.decisions.explain("LOAN/2026/0042", method="shapley", options={"seed": 7, "frozen_features": ["age"]})
    assert calls[-1].url.raw_path == b"/v1/decisions/LOAN/2026/0042/explanation"
    assert _sent(calls[-1]) == {"method": "shapley", "options": {"seed": 7, "frozen_features": ["age"]}}


def test_the_document_comes_back_as_bytes_and_saves_under_the_api_name(client: Kaimei, calls: List[httpx.Request], tmp_path: Path) -> None:
    assert client.decisions.pdf("APP-2026-00417") == PDF
    assert dict(calls[-1].url.params) == {"format": "pdf", "paper": "letter"}
    client.decisions.pdf("APP-2026-00417", explanation="exp_9", paper="a4")
    assert dict(calls[-1].url.params) == {"explanation": "exp_9", "format": "pdf", "paper": "a4"}
    with pytest.raises(ValueError, match="letter"):
        client.decisions.pdf("APP-2026-00417", paper="legal")

    saved = client.decisions.save_pdf("LOAN/2026/0042", tmp_path)
    assert saved == tmp_path / "decision-record-LOAN-2026-0042.pdf" and saved.read_bytes() == PDF
    named = client.decisions.save_pdf("APP-2026-00417", tmp_path / "file.pdf", paper="a4")
    assert named.name == "file.pdf" and named.read_bytes() == PDF
    assert record_filename("///") == "decision-record-decision.pdf"


def test_a_record_shows_as_a_card_with_the_api_sentences() -> None:
    record = APIObject(RECORD)
    card = display.html(record)
    assert "Decision record" in card and "APP-2026-00417" in card
    assert "The model gave this case a 59.5% chance of decline." in card
    assert "raised the chance of decline by 8.0 percentage points." in card
    assert "Held fixed in the search: applicant age." in card
    assert "What this record does not show" in card and "It is not a fairness test." in card
    assert f"sha256:{RECORD['fingerprint']['value'][:16]}" in card
    assert "/decisions/APP-2026-00417" in card
    assert "KernelSHAP" in card and "reproducible" in card
    bundle, _ = record._repr_mimebundle_()
    assert "text/html" in bundle


def test_a_card_escapes_what_a_record_holds() -> None:
    unsafe = json.loads(json.dumps(RECORD))
    unsafe["id"] = 'A/<b>"1'
    unsafe["outcome"]["sentence"] = "<script>alert(1)</script>"
    unsafe["factors"][0]["label"] = "<img src=x>"
    card = display.html(unsafe)
    assert "<script>alert(1)</script>" not in card and "<img src=x>" not in card
    assert "/decisions/A/%3Cb%3E%221" in card


def test_a_record_that_is_not_explained_says_so_and_lists_its_inputs() -> None:
    pending = json.loads(json.dumps(RECORD))
    pending.update(status="not_explained", explanation=None, factors=[], counterfactual=None, notes=["This decision is logged but has not been explained."])
    card = display.html(pending)
    assert "not explained" in card and "has not been explained" in card
    assert "How the model reached it" not in card
    table, columns = display.rows(pending)
    assert [row["name"] for row in table] == [item["name"] for item in RECORD["inputs"]]
    assert columns[:2] == ["name", "label"]


def test_a_record_is_a_table_of_its_effects() -> None:
    table, columns = display.rows(RECORD)
    assert [row["rank"] for row in table] == list(range(1, 10))
    assert columns[:5] == ["rank", "feature", "label", "display", "attribution"]
    assert table[0]["feature"] == "debt_to_income" and table[0]["direction"] == "raises"


def test_a_record_is_a_frame_of_its_effects() -> None:
    pd = pytest.importorskip("pandas")
    frame = APIObject(RECORD).to_frame()
    assert isinstance(frame, pd.DataFrame) and list(frame.columns[:3]) == ["rank", "feature", "label"]
    assert frame["attribution"].abs().is_monotonic_decreasing
