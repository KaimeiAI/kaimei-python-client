"""A model's report: the JSON, the Markdown, one row per section, and the notebook card."""

import httpx

from kaimei import Kaimei, display
from kaimei._html import render

REPORT = {
    "object": "model_report",
    "model": {"id": "mdl_1", "slug": "credit", "name": "Credit <risk> scorer", "version": "3", "task": "binary_classification"},
    "generated_at": "2026-11-26T19:19:47Z",
    "stale_after_days": 30,
    "headline": "A yes-or-no classifier that reads 9 inputs and decides between `approve` and `decline`.",
    "facts": [
        {"key": "reference", "label": "Reference rows", "value": "600", "note": "with labels"},
        {"key": "outcomes", "label": "Outcomes recorded", "value": "None", "note": None},
    ],
    "sections": [
        {
            "key": "drivers",
            "title": "What drives its answers",
            "summary": "`credit_score` moves the model's answer most: 25% of the measured importance.",
            "as_of": "2026-10-01T12:00:00Z",
            "status": "stale",
            "applies": True,
            "insight_id": "ins_1",
            "data": {"type": "global_importance", "payload": {"features": [{"feature": "credit_score", "share": 0.25}]}},
            "table": {"columns": ["Input", "Share of importance"], "rows": [["credit_score", "25%"]]},
        },
        {
            "key": "interactions",
            "title": "Inputs that work together",
            "summary": "This study reads a model's inputs one by one.",
            "as_of": None,
            "status": "missing",
            "applies": False,
            "insight_id": None,
            "data": None,
            "table": None,
        },
        {
            "key": "performance",
            "title": "How it does against outcomes",
            "summary": "No outcome has been recorded for a logged prediction.",
            "as_of": None,
            "status": "missing",
            "applies": True,
            "insight_id": None,
            "data": None,
            "table": None,
        },
    ],
    "gaps": [{"key": "outcomes", "title": "No outcomes recorded", "how": "Attach them with `POST /v1/predictions/actuals`.", "sections": ["performance"]}],
}
MARKDOWN = "# Credit risk scorer: model report\n\nA yes-or-no classifier.\n\n| Fact | Value | Note |\n| --- | --- | --- |\n| Median | 3,218 MW | −2 °C |\n"


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_report_reads_the_json_and_report_markdown_the_document():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.params.get("format") == "markdown":
            return httpx.Response(200, content=MARKDOWN.encode("utf-8"), headers={"content-type": "text/markdown; charset=utf-8"})
        return httpx.Response(200, json=REPORT)

    client = _client(handler)
    report = client.models.report("credit")
    assert seen[-1] == ("GET", "/v1/models/credit/report", {})
    assert report.headline.startswith("A yes-or-no classifier") and report.model.slug == "credit"
    assert [section.key for section in report.sections] == ["drivers", "interactions", "performance"]
    assert report.sections[0].status == "stale" and report.sections[0].data.payload.features[0].share == 0.25
    assert report.sections[1].applies is False and report.gaps[0].sections == ["performance"]
    text = client.models.report_markdown("credit")
    assert seen[-1] == ("GET", "/v1/models/credit/report", {"format": "markdown"})
    # The document comes back as text, with what is not ASCII in it intact.
    assert isinstance(text, str) and text == MARKDOWN and "−2 °C" in text


def test_a_report_reads_as_one_row_per_section():
    table, columns = display.rows(REPORT)
    assert columns == ["key", "title", "status", "applies", "as_of", "summary"]
    assert [row["key"] for row in table] == ["drivers", "interactions", "performance"]
    assert table[0] == {
        "key": "drivers",
        "title": "What drives its answers",
        "status": "stale",
        "applies": True,
        "as_of": "2026-10-01T12:00:00Z",
        "summary": "`credit_score` moves the model's answer most: 25% of the measured importance.",
    }
    assert display.rows({"object": "model_report", "sections": None}) == ([], columns)


def test_the_notebook_card_says_what_each_section_says_and_what_is_missing():
    html = render(REPORT)
    assert "Model report" in html and "Credit &lt;risk&gt; scorer" in html and "<risk>" not in html
    # Names are set in backticks in the API's sentences; a card shows them as plain words.
    assert "decides between approve and decline." in html and "`" not in html.split("</style>")[-1]
    assert "Reference rows" in html and ">600<" in html and "with labels" in html
    for title in ("What drives its answers", "Inputs that work together", "How it does against outcomes"):
        assert title in html
    assert "credit_score moves the model&#x27;s answer most" in html
    assert ">stale<" in html and ">does not apply<" in html and ">missing<" in html
    assert "What is not on file (1)" in html and "<b>No outcomes recorded.</b> Attach them with POST /v1/predictions/actuals." in html
    assert "/models/credit/report" in html
    # A report with nothing missing has no list of gaps, and one with no sections still draws.
    assert "What is not on file" not in render(dict(REPORT, gaps=[]))
    assert "Model report" in render({"object": "model_report", "model": {}, "sections": [], "facts": [], "gaps": []})
