"""A model's evidence by requirement, and its documentation report.

``fixtures/evidence.json`` and ``fixtures/model_report.json`` are what the API returned for the demo credit model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import httpx
import pytest

from kaimei import APIObject, Kaimei, display

FIXTURES = Path(__file__).parent / "fixtures"
EVIDENCE = json.loads((FIXTURES / "evidence.json").read_text(encoding="utf-8"))
REPORT = json.loads((FIXTURES / "model_report.json").read_text(encoding="utf-8"))
PDF = b"%PDF-1.4\n% model documentation\n%%EOF\n"


@pytest.fixture
def calls() -> List[httpx.Request]:
    return []


@pytest.fixture
def client(calls: List[httpx.Request]) -> Kaimei:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path, query = request.url.path, dict(request.url.params)
        if path == "/v1/models/credit-risk-v3/evidence":
            return httpx.Response(200, json=EVIDENCE)
        if path == "/v1/models/credit-risk-v3/report":
            if query.get("format") == "pdf":
                return httpx.Response(200, content=PDF, headers={"Content-Type": "application/pdf"})
            return httpx.Response(200, json=REPORT)
        return httpx.Response(404, json={"error": {"type": "not_found_error", "message": path}})

    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_evidence_is_read_requirement_by_requirement(
    client: Kaimei, calls: List[httpx.Request]
) -> None:
    evidence = client.models.evidence("credit-risk-v3")
    assert calls[-1].url.path == "/v1/models/credit-risk-v3/evidence" and not calls[-1].url.params
    assert isinstance(evidence, APIObject) and evidence.object == "evidence"
    assert [rule.id for rule in evidence.frameworks] == [
        "eu-ai-act",
        "regulation-b",
        "model-risk",
        "gdpr",
    ]
    keeping = next(r for r in evidence.frameworks[0].requirements if r.id == "article-12")
    assert keeping.reference == "Articles 12, 19 and 26" and keeping.status == "held"
    assert [part.key for part in keeping.evidence] == ["predictions", "explanations"]
    assert evidence.todo and all(entry.todo for entry in evidence.todo)
    client.models.evidence("credit-risk-v3", fresh_days=30)
    assert dict(calls[-1].url.params) == {"fresh_days": "30"}


def test_evidence_shows_as_a_card_and_a_table_and_never_says_compliant() -> None:
    card = display.html(EVIDENCE)
    shown = card.split('class="km-raw"')[0]
    assert (
        "requirements with evidence held" in shown
        and "EU AI Act" in shown
        and "Model risk management" in shown
    )
    assert EVIDENCE["frameworks"][0]["summary"]["sentence"] in shown
    assert "Record-keeping" in shown and "Articles 12, 19 and 26" in shown
    assert (
        "What would add evidence" in shown and "/models/credit-risk-v3/compliance" in card
    )  # the foot of the card
    assert "compliant" not in card.lower() and "not legal advice" in card
    rows, columns = display.rows(EVIDENCE)
    assert columns[:4] == ["rule", "requirement", "reference", "status"]
    assert len(rows) == sum(len(rule["requirements"]) for rule in EVIDENCE["frameworks"])
    first = rows[0]
    assert (
        first["rule"] == "EU AI Act"
        and first["requirement"] == "Risk management"
        and "Robustness" in first["held"]
    )


def test_the_report_is_read_as_data_and_as_a_document(
    client: Kaimei, calls: List[httpx.Request], tmp_path: Path
) -> None:
    report = client.models.report("credit-risk-v3")
    assert report.object == "model_report" and report.model.task_words == "a yes-or-no classifier"
    assert report.findings[0].name == "Global importance" and len(report.fingerprint.value) == 64
    assert client.models.report_pdf("credit-risk-v3") == PDF
    assert dict(calls[-1].url.params) == {"format": "pdf", "paper": "letter"}
    client.models.report_pdf("credit-risk-v3", paper="a4", fresh_days=30)
    assert dict(calls[-1].url.params) == {"format": "pdf", "paper": "a4", "fresh_days": "30"}
    with pytest.raises(ValueError, match="letter"):
        client.models.report_pdf("credit-risk-v3", paper="legal")
    saved = client.models.save_report_pdf("credit-risk-v3", tmp_path)
    assert (
        saved == tmp_path / "model-documentation-credit-risk-v3.pdf" and saved.read_bytes() == PDF
    )
    named = client.models.save_report_pdf("credit-risk-v3", tmp_path / "file.pdf")
    assert named.name == "file.pdf" and named.read_bytes() == PDF


def test_the_report_shows_as_a_card_with_its_findings() -> None:
    card = display.html(REPORT)
    shown = card.split('class="km-raw"')[0]
    assert (
        "Model documentation" in shown
        and "Credit risk scorer" in shown
        and "a yes-or-no classifier" in shown
    )
    assert "It answers approve or decline" in shown and "What the analyses found" in shown
    for finding in REPORT["findings"]:
        assert f"<b>{finding['name']}.</b>" in shown
    assert "Evidence held, by rule" in shown and "This is not a validation report" in shown
    assert (
        f"sha256:{REPORT['fingerprint']['value'][:16]}" in card and "compliant" not in card.lower()
    )
    rows, columns = display.rows(REPORT)
    assert columns[:2] == ["name", "summary"] and [row["name"] for row in rows] == [
        f["name"] for f in REPORT["findings"]
    ]
