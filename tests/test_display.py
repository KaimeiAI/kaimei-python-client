"""Notebook display: the themed cards, the raw mode and the DataFrame / JSON exports."""

from __future__ import annotations

import json
import re
from html import escape

import pytest

import kaimei
from kaimei import APIObject, Page, display
from kaimei import _html as cards
from kaimei._html import (
    compact,
    fmt_number,
    fmt_signed,
    fmt_value,
    long_date,
    output_parts,
    pct,
    short_date,
)
from kaimei._methods import summarize
from test_case_study import REPORT
from test_compare import COMPARISON
from test_compare_methods import APPLICANT
from test_compare_methods import _explanation as _method_explanation
from test_similar_holdings import SIMILAR
from test_tabular import FLIPPED, INSIGHTS, MODEL, PASSENGER, SHORT, _explanation

TOP = [
    {"feature": "Sex", "attribution": -0.091, "value": "male", "share": 0.5112, "direction": "negative", "rank": 1},
    {"feature": "Pclass", "attribution": -0.042, "value": 3, "share": 0.236, "direction": "negative", "rank": 2},
]


def _unstyled(html):
    """A card without the inline copies of its styles, to check its content and structure."""
    return re.sub(r' style="[^"]*"', "", html)


def _card(record):
    return _unstyled(display.html(record))


def _record(**overrides):
    fields = dict(counterfactual=SHORT, top_features=TOP, n_evals=128, latency_ms=41.2, prediction_id="pred_7b",
                  created_at="2026-10-25T15:04:05.123456+00:00")
    fields.update(overrides)
    return _explanation(**fields)


@pytest.fixture(autouse=True)
def _rich_mode(monkeypatch):
    monkeypatch.delenv("KAIMEI_DISPLAY", raising=False)
    monkeypatch.setattr(display, "_mode", None)


# --- the dashboard's formatting -----------------------------------------------------------------


def test_numbers_format_as_the_dashboard_formats_them():
    assert fmt_number(1234.5, 3) == "1,234.5"
    assert fmt_number(0.252, 3) == "0.252"
    assert fmt_number(2.5e-8, 3) == "2.5e-8"
    assert fmt_number(0.4, 0) == "0"
    assert fmt_number(0.0, 3) == "0"
    assert fmt_number(None) == "—"
    assert compact(12345) == "12.3K"
    assert compact(12000) == "12K"
    assert compact(2_500_000) == "2.5M"
    assert compact(999) == "999"
    assert pct(0.926, 1) == "92.6%"
    assert fmt_signed(0.091) == "+0.091"
    assert fmt_signed(-0.0912) == "−0.091"
    assert fmt_signed(0.0) == "0.000"
    assert fmt_value(True) == "yes"
    assert fmt_value(72000) == "72,000"
    assert fmt_value(0.41) == "0.41"
    assert fmt_value([1, 2, 3]) == "[3 values]"
    assert fmt_value({"a": 1}) == "{…}"
    assert fmt_value(7.25, "GBP") == "7.25 GBP"
    assert output_parts({"label": "decline", "probability": 0.81}) == ("decline", "81.0%")
    assert output_parts({"label": "decline", "score": 0.5}) == ("decline", "50.0%")
    assert output_parts({"value": 12.3456, "unit": "GBP"}) == ("12.346", "GBP")
    assert output_parts({"score": 0.5}) == ("0.5", None)
    assert output_parts(None) == ("—", None)
    assert long_date("2026-10-25T15:04:05Z") == "Oct 25, 2026, 15:04 UTC"
    assert short_date("2026-10-05T09:04:05Z") == "Oct 5, 09:04"
    assert long_date("not a date") == "not a date"


# --- what a cell shows --------------------------------------------------------------------------


def test_explanation_card_reads_like_the_dashboard():
    bundle, metadata = _record()._repr_mimebundle_()
    assert set(bundle) == {"text/html", "text/plain"} and metadata == {}
    html = _unstyled(bundle["text/html"])
    assert html.startswith("<style>") and 'class="km"' in html
    assert "kai<i>mei</i>" in html and "<svg" in html  # the mark and the wordmark
    assert "Explanation" in html and "expl_1" in html and "Oct 25, 2026, 15:04 UTC" in html
    assert 'class="km-value">died<' in html and "P(died) = 92.6%" in html
    assert "KernelSHAP" in html and "explains survived" in html
    assert "base 0.252" in html and "128 evals" in html and "41 ms" in html and "pred_7b" in html
    # The attribution bars: the top_features rows first, then the rest of ``attributions``.
    assert "pushes toward survived" in html and "pushes away" in html
    assert html.index("<b>Sex</b>") < html.index("<b>Pclass</b>") < html.index("<b>Fare</b>") < html.index("<b>IsAlone</b>")
    assert "−0.091" in html and "−0.042" in html and "+0.000" in html and "· male" in html
    assert 'class="km-fill neg"' in html and 'class="km-fill pos"' in html
    # The summary sentence, the probabilities, the counterfactual and the input.
    assert "The model predicts died: P(survived) = 7.4%" in html
    assert "92.6%" in html and "7.4%" in html
    assert "Closest the counterfactual search got" in html
    assert "Set <b>Fare</b> from <b>7.25</b>" in html and "<b>73.5</b>" in html
    assert "→ P(survived) 22.9% · died" in html and "22.9% died" not in html  # names what 22.9% is
    assert "P(survived) 7.4% → 35.9%" in html and "Decision threshold 50.0%" in html
    assert "5 features" in html and "<td class=\"mono muted\">Age</td><td class=\"num mono\">22</td>" in html
    assert "/explanations/expl_1" in html and "Open in dashboard" in html
    assert bundle["text/plain"] == "<Kaimei explanation expl_1>"


def test_flipped_counterfactual_and_regression_outputs():
    html = _card(_record(counterfactual=FLIPPED))
    assert "Smallest change that flips the decision" in html and "P(survived) 54.1% · survived" in html and "54.1% survived" not in html
    regression = APIObject(
        {
            "object": "explanation", "id": "expl_r", "model_id": "m", "method": "shap",
            "input": {"sqft": 1850, "garage": True}, "output": {"value": 412350.5, "name": "price", "unit": "USD"},
            "attributions": {"sqft": 61250.0, "garage": -7900.0}, "base_value": 364000.0, "top_features": [],
            "counterfactual": {"found": True, "changes": [{"feature": "sqft", "from": 1850, "to": 2100, "output_after": 440000.0}],
                               "output_before": 412350.5, "output_after": 440000.0, "target": {"kind": "reach_value", "target_value": 430000.0}},
            "options": {}, "metadata": {}, "created_at": "2026-10-25T15:04:05Z",
        }
    )
    html = _card(regression)
    assert 'class="km-value">412,350.5<' in html and 'class="km-sub">USD<' in html
    assert "pushes higher" in html and "pushes lower" in html and "+61250.000" in html
    assert "Output 412,350.5 → 440,000" in html and "Target value 430,000" in html and "· yes" in html


def test_cards_keep_their_layout_where_style_blocks_are_dropped():
    """GitHub's notebook viewer removes <style> elements but keeps style attributes."""
    html = display.html(_record(counterfactual=FLIPPED))
    css, body = html.split("</style>", 1)
    assert css.startswith("<style>.km{") and "<style" not in body
    root = re.match(r'<div style="([^"]*)" class="km">', body).group(1)
    assert "max-width:780px" in root and "background:var(--bg,#ffffff)" in root and "'Segoe UI'" in root
    assert re.search(r'<svg style="width:18px;height:18px;[^"]*" width="18" height="18" viewBox="0 0 1024 1024"', body)  # the logo keeps its size
    styles = re.findall(r' style="([^"]*)"', body)
    assert len(styles) > 50 and not any(re.search(r"(^|;)--", style) for style in styles)  # themes stay in the stylesheet
    fill = re.search(r'<span class="km-fill neg" style="([^"]*)"', body).group(1)
    assert fill.startswith("position:absolute") and "linear-gradient" in fill and re.search(r";left:[0-9.]+%;width:[0-9.]+%$", fill)
    row = re.search(r'<div style="([^"]*)" class="km-row"', body).group(1)
    assert "grid-template-columns" in row and "background" not in row  # its hover highlight stays in the stylesheet
    summary = re.search(r'<summary style="([^"]*)"', body).group(1)
    assert "padding" in summary and "display" not in summary and "list-style" not in summary  # keeps its marker
    cell = re.search(r'<td style="([^"]*)"', body).group(1)
    assert "padding:5px 8px" in cell and "border" not in cell  # the last row's border is a :last-child rule
    word = re.search(r'<i( style="[^"]*")?>mei</i>', body).group(1) or ""
    assert "color" not in word  # the gradient text comes from an @supports rule
    # GitHub also drops <details> and <summary>, keeping their contents: the JSON dump stays hidden
    # there, while the stylesheet shows it again everywhere else.
    github = re.sub(r"</?(details|summary)[^>]*>", "", body)
    hidden = github.index('<div class="km-raw" style="display:none">')
    assert github.index('class="km-json"') > hidden and ".km .km-raw{display:block!important}" in css
    assert github.index("Input") < hidden  # the input table still shows


def test_feature_names_and_values_are_escaped():
    record = _record(attributions={"<script>alert(1)</script>": 0.5}, top_features=[], input={"<script>alert(1)</script>": "<b>"})
    html = _card(record)
    assert "<script>" not in html and "&lt;script&gt;" in html and "&lt;b&gt;" in html


def test_raw_mode_shows_the_json_and_restores_itself(monkeypatch):
    record = _record()
    with display.using("raw"):
        bundle, metadata = record._repr_mimebundle_()
        assert bundle["application/json"] == record.to_dict()
        assert json.loads(bundle["text/plain"]) == record.to_dict()
        assert metadata == {"application/json": {"expanded": True, "root": "explanation"}}
        assert kaimei.display_mode() == "raw"
    assert kaimei.display_mode() == "rich"
    kaimei.set_display("raw")
    assert "text/html" not in record._repr_mimebundle_()[0]
    kaimei.set_display("rich")
    assert "text/html" in record._repr_mimebundle_()[0]
    with pytest.raises(ValueError):
        kaimei.set_display("fancy")
    monkeypatch.setattr(display, "_mode", None)
    monkeypatch.setenv("KAIMEI_DISPLAY", "raw")
    assert kaimei.display_mode() == "raw"
    monkeypatch.setenv("KAIMEI_DISPLAY", "nonsense")
    assert kaimei.display_mode() == "rich"


def test_nested_values_keep_their_plain_repr():
    record = _record()
    assert record.output._repr_mimebundle_() is None
    assert record.top_features[0]._repr_mimebundle_() is None
    assert repr(record.output).startswith("{'label': 'died'")


def test_to_json_is_the_api_record():
    record = _record()
    assert json.loads(record.to_json()) == record.to_dict()
    assert record.to_json(indent=None).startswith('{"object": "explanation"')


# --- tables -------------------------------------------------------------------------------------


def test_explanation_frame_has_the_dashboard_numbers():
    pd = pytest.importorskip("pandas")
    frame = _record().to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["rank", "feature", "value", "attribution", "share", "direction"]
    assert frame["feature"].tolist() == ["Sex", "Pclass", "Fare", "Age", "IsAlone"]
    assert frame["rank"].tolist() == [1, 2, 3, 4, 5]
    # The API's rows are kept as sent; the rest get a share by the API's formula.
    assert frame["share"].tolist()[:2] == [0.5112, 0.236]
    total = 0.042 + 0.091 + 0.008 + 0.037 + 0.0004
    assert frame["share"].tolist()[2] == round(0.037 / total, 4)
    assert frame["direction"].tolist() == ["negative"] * 4 + ["positive"]
    assert frame["value"].tolist() == ["male", 3, 7.25, 22.0, False]


def test_rows_without_pandas():
    table, columns = display.rows(_record())
    assert [row["feature"] for row in table] == ["Sex", "Pclass", "Fare", "Age", "IsAlone"]
    assert columns[:2] == ["rank", "feature"]
    table, _ = display.rows({"object": "explanation", "attributions": {"a": 1.0, "b": -3.0}, "input": {"a": 1, "b": 2}})
    assert [(r["feature"], r["share"], r["rank"]) for r in table] == [("b", 0.75, 1), ("a", 0.25, 2)]


def test_insight_cards_and_tables():
    pytest.importorskip("pandas")
    by_type = {str(i.type): i for i in INSIGHTS}
    html = _card(by_type["drift"])
    assert "PSI 0.31" in html and 'class="km-badge bad">significant' in html and "891 reference vs 418 recent rows" in html
    assert 'class="km-fill significant"' in html and 'class="km-fill stable"' in html and "0.03" in html
    assert by_type["drift"].to_frame()["psi"].tolist() == [0.03, 0.31]
    html = _card(by_type["global_importance"])
    assert "27%" in html and "15%" in html and "accuracy 0.85" in html
    html = _card(by_type["behavior"])
    assert "<svg" in html and "threshold 0.5" in html and "survived 40.0%" in html and "died 60.0%" in html
    assert by_type["behavior"].to_frame().columns.tolist() == ["bin_start", "bin_end", "count"]
    html = _card(by_type["partial_dependence"])
    assert html.count('stroke="var(--series-1)"') == 3 and "decreasing" in html and "non monotonic" in html
    curves = by_type["partial_dependence"].to_frame()
    assert curves.columns.tolist() == ["feature", "type", "grid", "pd", "monotonicity"] and len(curves) == 8
    html = _card(by_type["interactions"])
    assert "Sex × Pclass" in html and "12%" in html
    assert by_type["interactions"].to_frame().iloc[0].to_dict() == {"feature_a": "Sex", "feature_b": "Pclass", "strength": 0.12}
    html = _card(by_type["robustness"])
    assert "ε 0.05" in html and "<b>Fare</b>" in html and "<b>Sex</b>" not in html  # a zero sensitivity is not a bar
    html = _card(by_type["learned_rules"])
    assert "Sex = female pushes the probability of survived highest." in html and "`" not in html.split("km-rules")[1].split("</div></div>")[0]
    assert by_type["learned_rules"].to_frame()["statement"].tolist() == ["`Sex` = female pushes the probability of survived highest."]


def test_model_prediction_analysis_and_generic_cards():
    pytest.importorskip("pandas")
    model = APIObject({**MODEL.to_dict(), "id": "mdl_1", "name": "Titanic forest", "version": "2", "framework": "sklearn",
                       "status": "active", "reference_rows": 891, "predictions_count": 12480, "tags": ["demo"],
                       "outputs": {"labels": ["died", "survived"], "positive_label": "survived", "threshold": 0.5},
                       "created_at": "2026-10-20T09:00:00Z", "updated_at": "2026-10-25T15:06:40Z"})
    html = _card(model)
    assert "Titanic forest" in html and ">v2<" in html and ">Binary<" in html and "Active" in html and "#demo" in html
    assert "12.5K" in html and "891" in html and "female, male" in html and ">years<" in html
    assert "threshold 0.5" in html and "positive survived" in html and "/models/titanic-forest" in html
    assert model.to_frame()["name"].tolist() == ["Pclass", "Sex", "Age", "Fare", "IsAlone"]

    prediction = APIObject({"object": "prediction", "id": "pred_1", "model_id": "mdl_1", "input": dict(PASSENGER),
                            "output": {"label": "died", "probability": 0.926, "probabilities": {"died": 0.926, "survived": 0.074}},
                            "latency_ms": 3.2, "source": "api", "metadata": {}, "created_at": "2026-10-25T15:04:05Z"})
    html = _card(prediction)
    assert "P(died) = 92.6%" in html and ">api<" in html and "3 ms" in html and "92.6%" in html and "7.4%" in html
    assert prediction.to_frame().loc[0, "output.probability"] == 0.926

    analysis = APIObject({"object": "analysis", "id": "ana_1", "model_id": "mdl_1", "type": "full", "status": "running",
                          "progress": 0.4, "options": {}, "error": None, "insight_ids": ["ins_1", "ins_2"], "n_evals": 12480,
                          "created_at": "2026-10-25T15:04:05Z", "started_at": "2026-10-25T15:04:06Z", "completed_at": None})
    html = _card(analysis)
    assert "Running" in html and "40% done" in html and "12.5K" in html and "ins_2" in html
    finished = APIObject({**analysis.to_dict(), "status": "completed", "progress": 1.0, "completed_at": "2026-10-25T15:06:40Z", "error": "boom"})
    html = _card(finished)
    assert "Completed" in html and "2m 34s" in html and "boom" in html and "% done" not in html

    organization = APIObject({"object": "organization", "id": "org_1", "name": "Acme", "slug": "acme", "plan": "pro",
                              "settings": {"a": 1}, "quotas": {"limits": {"x": 1}}, "created_at": "2026-10-25T15:04:05Z"})
    html = _card(organization)
    assert "Organization" in html and "Acme" in html and ">pro<" in html and "Quotas" in html


def test_page_card_and_frame():
    pytest.importorskip("pandas")
    page = Page({"object": "list", "data": [_record().to_dict(), _record(id="expl_2").to_dict()], "total": 57, "limit": 2,
                 "offset": 10, "has_more": True})
    assert re.match(r'<style>.*</style><div style="[^"]*max-width:1080px[^"]*" class="km wide">', display.html(page), re.S)
    html = _card(page)
    assert "57 explanations" in html and "showing 11–12" in html and "auto_paging_iter" in html
    assert "expl_2" in html and "Sex −0.09" in html and "KernelSHAP" in html and "92.6%" in html
    frame = page.to_frame()
    assert frame.columns.tolist()[:3] == ["id", "object", "model_id"]
    assert frame["output.label"].tolist() == ["died", "died"]
    assert frame["attributions.Sex"].tolist() == [-0.091, -0.091]
    assert "top_features" not in frame.columns  # tables are not flattened into a row
    empty = Page({"object": "list", "data": [], "total": 0, "limit": 20, "offset": 0, "has_more": False})
    assert "Nothing on this page" in _card(empty)
    assert empty.to_frame().empty
    insights = Page({"object": "list", "data": [i.to_dict() for i in INSIGHTS], "total": 7, "limit": 50, "offset": 0, "has_more": False})
    assert "7 insights" in _card(insights)
    assert len(insights.to_frame()) == 7


def test_image_explanation_card_lists_regions_and_draws_the_photo():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    photo = np.full((8, 8, 3), 0.3)
    photo[:4, 4:] = [0.9, 0.8, 0.5]
    regions = [
        {"id": 1, "name": "upper right", "attribution": 0.9, "area": 0.25, "centroid": [0.75, 0.25], "bbox": [0.5, 0, 1, 0.5], "anchor": [0.75, 0.25]},
        {"id": 2, "name": "lower left", "attribution": -0.4, "area": 0.25, "centroid": [0.25, 0.75], "bbox": [0, 0.5, 0.5, 1], "anchor": [0.25, 0.75]},
    ]
    record = APIObject(
        {
            "object": "explanation", "id": "expl_img", "model_id": "mdl_cnn", "method": "occlusion",
            "input": {"photo": photo.tolist()},
            "output": {"label": "cat", "probability": 0.97, "probabilities": {"cat": 0.97, "dog": 0.03}, "target_label": "cat", "value": 0.97},
            "attributions": {r["name"]: r["attribution"] for r in regions}, "top_features": [], "base_value": None,
            "n_evals": 9, "latency_ms": 120.0, "counterfactual": None,
            "options": {"heatmap": [[0.0, 0.9], [-0.4, 0.0]], "target": "cat", "regions": regions,
                        "focus": {"threshold": 0.5, "share": 0.5, "area": 0.25, "centroid": [0.75, 0.25], "bbox": [0.5, 0, 1, 0.5], "position": "upper right"},
                        "evidence": {"label": "cat", "full": 0.97, "without_focus": 0.41, "focus_only": 0.95, "fill": "gray"},
                        "summary": "Half of the evidence for cat sits in the upper right."},
            "metadata": {}, "created_at": "2026-10-25T15:04:05Z",
        }
    )
    html = _card(record)
    assert html.count("data:image/png;base64,") == 3 and ">Photo<" in html and ">Evidence<" in html and ">Focus<" in html
    assert "Half of the evidence for cat" in html
    assert "P(cat) photo 97.0%" in html and "focus hidden 41.0%" in html and "focus only 95.0%" in html
    assert "1 · upper right" in html and "2 · lower left" in html and 'class="km-fill for"' in html and 'class="km-fill against"' in html
    assert "For cat" in html and "Against cat" in html and "8×8×3 image" in html
    assert "&quot;[8×8×3 values]&quot;" in html  # the JSON view elides the pixels
    assert "Attributions</div>" not in html  # regions replace the tabular bars
    table, columns = display.rows(record)
    assert [row["name"] for row in table] == ["upper right", "lower left"] and columns[:2] == ["id", "name"]


def test_html_takes_plain_dicts_and_unknown_types():
    html = _card({"object": "widget", "id": "w_1", "name": "Thing", "size": 3, "parts": [{"a": 1}, {"a": 2}]})
    assert "Widget" in html and "Thing" in html and ">3<" in html and "Parts" in html
    with pytest.raises(TypeError):
        display.html("not a record")


# --- what a model's log says: monitoring, population, comparison, similar cases, holdings, methods ---
#
# The records below are the API's answers for its demo models (energy-demand-v1 through its staged drift
# incident, credit-risk-v3), cut down to a few windows, fields and items.


def _window(day, predictions, mean=None, rows=0, status="insufficient_data", psi=None, top=None, chance=None, beyond=None, levels=None):
    return {
        "start": f"2026-11-{day:02d}T00:00:00Z", "end": f"2026-11-{day + 1:02d}T00:00:00Z", "predictions": predictions,
        "output": {"n": predictions, "mean": mean}, "rows": rows, "status": status, "overall_psi": psi, "top_feature": top,
        "chance": chance, "beyond_chance": beyond, "drift": {top: psi} if top else {}, "levels": levels or ({top: chance} if top else {}),
    }  # fmt: skip


MONITORING = {
    "object": "monitoring", "model_id": "mdl_a0fee1c154842f759c51", "window": "day", "days": 30,
    "since": "2026-10-28T00:22:47.593043Z", "until": "2026-11-27T00:22:47.593043Z", "predictions": 1002, "truncated": False,
    "trailing": 500, "reference_rows": 600, "thresholds": {"moderate": 0.1, "significant": 0.25}, "drift_unavailable": None,
    "features": [
        {"name": "temperature", "type": "numeric", "unit": "°C", "peak_psi": 1.263156, "latest_psi": 1.263156, "status": "significant"},
        {"name": "wind_speed", "type": "numeric", "unit": "m/s", "peak_psi": None, "latest_psi": 0.0720598, "status": "stable"},
        {"name": "humidity", "type": "numeric", "unit": "%", "peak_psi": None, "latest_psi": 0.11, "status": "moderate"},
    ],
    "series": [
        _window(12, 0),
        _window(13, 39, 3234.437047, 39, "significant", 0.775897, "wind_speed", 1.752973, False),
        _window(24, 78, 3267.276463, 500, "stable", 0.0365383, "hour", 0.0859323, False),
        _window(25, 135, 7166.270006, 500, "significant", 0.34933, "temperature", 0.0848125, True),
        _window(27, 13, 3168.835207, 500, "significant", 1.263156, "temperature", 0.0848125, True,
                levels={"temperature": 0.0848125, "wind_speed": 0.0859323, "humidity": 0.158829}),
    ],
    "tape": {"feature": {"name": "temperature", "type": "numeric", "unit": "°C"}, "points": [], "sampled": False,
             "output": {"kind": "value", "name": "demand_mw", "unit": "MW", "threshold": None}},
    "insights": [],
}  # fmt: skip

POPULATION = {
    "object": "population", "model_id": "mdl_f075cc4cf5745563ec60", "days": 30, "since": "2026-10-28T00:22:47.884125Z",
    "until": "2026-11-27T00:22:47.884125Z", "predictions": 192, "shown": 192, "truncated": False, "outcomes": 173,
    "output": {"kind": "probability", "name": "P(decline)", "unit": None, "threshold": 0.5, "classes": ["approve", "decline"], "positive": "decline"},
    "fields": [
        {"key": "input:credit_score", "name": "credit_score", "source": "input", "type": "integer", "unit": None, "separation": 0.3176883315420912,
         "groups": [
             {"label": "≤ 643", "rows": 49, "mean": 0.762885, "predicted": {"approve": 5, "decline": 44}, "actual": {"approve": 18, "decline": 26}},
             {"label": "643 to 690", "rows": 47, "mean": 0.507002, "predicted": {"approve": 26, "decline": 21}, "actual": {"approve": 23, "decline": 19}},
             {"label": "> 742", "rows": 48, "mean": 0.203124, "predicted": {"approve": 43, "decline": 5}, "actual": {"approve": 32, "decline": 12}},
         ]},
        {"key": "input:employment_years", "name": "employment_years", "source": "input", "type": "numeric", "unit": "years", "separation": 0.07001659338762245,
         "groups": [{"label": "≤ 3.01", "rows": 48, "mean": 0.477128, "predicted": {"approve": 25, "decline": 23}, "actual": {"approve": 28, "decline": 12}},
                    {"label": "> 3.01", "rows": 144, "mean": 0.483, "predicted": {"approve": 78, "decline": 66}, "actual": {"approve": 73, "decline": 60}}]},
        {"key": "metadata:region", "name": "region", "source": "metadata", "type": "categorical", "unit": None, "separation": 0.0,
         "groups": [{"label": "us-east-1", "rows": 94, "mean": 0.483913, "predicted": {"approve": 50, "decline": 44}, "actual": {}},
                    {"label": "missing", "rows": 98, "mean": 0.48, "predicted": {"approve": 53, "decline": 45}, "actual": {}}]},
        {"key": "prediction", "name": "predicted class", "source": "prediction", "type": "categorical", "unit": None, "separation": None,
         "groups": [{"label": "approve", "rows": 103, "mean": 0.227569, "predicted": {"approve": 103}, "actual": {"approve": 70, "decline": 22}},
                    {"label": "decline", "rows": 89, "mean": 0.777499, "predicted": {"decline": 89}, "actual": {"approve": 31, "decline": 50}}]},
    ],
    "rows": {"id": ["pred_1", "pred_2"], "time": ["2026-11-13T02:00:00Z", "2026-11-27T00:20:00Z"], "value": [0.283022, 0.716978],
             "predicted": [0, 1], "actual": [0, None], "groups": {"input:credit_score": [2, 0], "prediction": [0, 1]}},
}  # fmt: skip

HOLDINGS = {
    "object": "model_holdings", "model_id": "mdl_f075cc4cf5745563ec60", "as_of": "2026-11-27T00:22:47.984975Z",
    "runs": {"by": "kaimei", "sentence": "Kaimei runs this model itself: it is one of Kaimei's own reference models (`credit-risk`), so no model of yours is involved."},
    "items": [
        {"key": "structure", "label": "Weights and structure", "held": "yes", "count": None, "unit": None, "bytes": None,
         "detail": "This is one of Kaimei's own reference models (`credit-risk`): its weights are part of Kaimei, and nothing was uploaded for it."},
        {"key": "reference_rows", "label": "Reference rows", "held": "yes", "count": 600, "unit": "row", "bytes": 157690,
         "detail": "600 rows, each with its label. Each row holds one value per input."},
        {"key": "insights", "label": "Insights", "held": "yes", "count": 1, "unit": "insight", "bytes": 734, "detail": "1 insight. 2 analyses are on record."},
        {"key": "credential", "label": "Credential for your endpoint", "held": "no", "count": 0, "unit": "header", "bytes": None,
         "detail": "No credential is held: this model has no endpoint for Kaimei to call."},
    ],
    "crosses": [{"when": "predict", "what": "The row you send reaches Kaimei, which runs its own reference model on it."},
                {"when": "log", "what": "When your own system logs a prediction, the inputs and the answer are stored by Kaimei."}],
}  # fmt: skip

_EXPLAINED = [_method_explanation(method, {"input": APPLICANT}) for method in ("shap", "shapley", "lime")]
METHODS = {"object": "method_comparison", "model": "credit-risk-v3", "methods": ["shap", "shapley", "lime"], "explanations": _EXPLAINED, **summarize(_EXPLAINED)}

#: Each card's record, and the sentence it leads with.
LOG_CARDS = {
    "monitoring": (MONITORING, "Significant drift in the latest window: temperature scores PSI 1.26, where 500 rows drawn from the reference data itself reach 0.085."),
    "population": (POPULATION, "Of 192 predictions logged in the last 30 days, credit_score separates the answers most: 32% of their variation lies between its groups."),
    "comparison": (COMPARISON, "Loan forest and Loan scorecard decide 2 of 3 reference rows the same way (66.7%)."),
    "similar_cases": (SIMILAR, "Of the 2 reference rows most like this case, 1 was labelled approve and 1 decline."),
    "model_holdings": (HOLDINGS, "Kaimei runs this model itself: it is one of Kaimei's own reference models (credit-risk), so no model of yours is involved."),
    "method_comparison": (METHODS, "They part on what matters most: KernelSHAP and Permutation Shapley say income, LIME says credit_score."),
    "model_case_study": (REPORT, "A yes-or-no classifier that reads 9 inputs and decides between approve and decline."),
}  # fmt: skip
#: What each says of a record that holds nothing but its type.
EMPTY_LEADS = {
    "monitoring": "This history has no windows to read.",
    "population": "No predictions were logged.",
    "comparison": "The first model and the second model were compared.",
    "similar_cases": "No reference rows came back as like this case.",
    "model_holdings": "Nothing is listed for this model.",
    "method_comparison": "There is nothing to compare: fewer than two methods explained this input.",
    "model_case_study": None,
}
MARKUP = '"><script>alert(1)</script><img src=x onerror=alert(2)>'


def _said(html):
    """The sentence a card leads with, as the page shows it."""
    found = re.search(r'<div class="km-say">(.*?)</div>', _unstyled(html), re.S)
    return found.group(1) if found else None


def _body(record):
    """A card as GitHub's notebook viewer keeps it: no stylesheet, no <details> or <summary>, inline styles only."""
    return re.sub(r"</?(details|summary)[^>]*>", "", display.html(record).split("</style>", 1)[1])


def _hostile(value, keep=("object",)):
    """``value`` with markup for every string in it; the keys of the maps a caller names (inputs, classes) too."""
    named = {"input", "inputs", "attributions", "counts", "levels", "drift", "predicted", "actual", "gaps", "widths", "first", "ranks"}
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in keep:
                out[key] = item
            elif key in named and isinstance(item, dict):
                out[key] = {MARKUP + str(name): _hostile(inner, ()) for name, inner in item.items()}
            else:
                out[key] = _hostile(item, ())
        return out
    if isinstance(value, list):
        return [_hostile(item, ()) for item in value]
    return MARKUP if isinstance(value, str) else value


def _renamed(record, *names):
    """``record`` with markup in place of each of ``names`` (a model, an input, a class) wherever it is written."""
    text = json.dumps(record)
    for name in names:
        text = text.replace(json.dumps(name)[1:-1], json.dumps(MARKUP)[1:-1])
    return json.loads(text)


@pytest.mark.parametrize("kind", sorted(LOG_CARDS))
def test_each_log_card_leads_with_the_sentence_a_person_would_say(kind):
    record, lead = LOG_CARDS[kind]
    assert record["object"] == kind
    bundle, metadata = APIObject(record)._repr_mimebundle_()
    assert set(bundle) == {"text/html", "text/plain"} and metadata == {}
    assert _said(bundle["text/html"]) == escape(lead, quote=True)
    html = _unstyled(bundle["text/html"])
    blocks = ('class="km-badges"', 'class="km-tiles"', 'class="km-table"', 'class="km-sides"', 'class="km-method"', 'class="km-foot"')
    assert html.index('class="km-say"') < min(html.index(block) for block in blocks if block in html)  # the sentence comes first
    assert "Open in dashboard" in html and "<script" not in html and "`" not in html.split("</style>", 1)[1]
    # No inline style leans on the stylesheet for a colour: each var() carries the light theme's value.
    styles = re.findall(r' style="([^"]*)"', bundle["text/html"].split("</style>", 1)[1])
    assert styles and not any(re.search(r"var\(--[a-z0-9-]+\)", style) for style in styles)


@pytest.mark.parametrize("kind", sorted(LOG_CARDS))
def test_a_log_card_of_a_nearly_empty_record_still_says_something(kind):
    record, _lead = LOG_CARDS[kind]
    html = _card({"object": kind})
    assert 'class="km"' in html and "kai<i>mei</i>" in html and _said(html) == EMPTY_LEADS[kind]
    assert "None" not in re.sub(r"<[^>]+>", " ", html.split("</style>", 1)[1])  # nothing printed for what is not there
    # An older or partial record: every field null, then every list and map empty, then each top-level key missing.
    hollow = {key: (value if key == "object" else None) for key, value in record.items()}
    bare = {key: (value if key == "object" else type(value)() if isinstance(value, (list, dict)) else value) for key, value in record.items()}
    without = [{k: v for k, v in record.items() if k != key} for key in record if key != "object"]
    for partial in [hollow, bare] + without:
        bundle, _metadata = APIObject(partial)._repr_mimebundle_()
        assert 'class="km"' in _unstyled(bundle["text/html"])


@pytest.mark.parametrize("kind", sorted(LOG_CARDS))
def test_log_cards_escape_every_name_the_api_or_the_user_gave(kind):
    record, _lead = LOG_CARDS[kind]
    names = ("temperature", "wind_speed", "credit_score", "decline", "approve", "Loan forest", "loan-forest-v1", "income", "home",
             "Weights and structure", "credit-risk", "mdl_", "°C", "MW", "demand_mw", "P(decline)", "Credit <risk> scorer")  # fmt: skip
    for hostile in (_hostile(record), _renamed(record, *names)):
        body = _body(hostile)
        assert "<script" not in body and "<img" not in body and MARKUP not in body
        assert "&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in body  # shown as the text it is
        for link in re.findall(r'href="([^"]*)"', body):
            assert link.startswith("https://dashboard.kaimei.ai/") and "<" not in link


@pytest.mark.parametrize("kind", sorted(LOG_CARDS))
def test_raw_mode_turns_the_log_cards_off(kind, monkeypatch):
    record, _lead = LOG_CARDS[kind]
    with display.using("raw"):
        bundle, metadata = APIObject(record)._repr_mimebundle_()
    assert "text/html" not in bundle and bundle["application/json"] == record
    assert json.loads(bundle["text/plain"]) == json.loads(json.dumps(record)) and metadata["application/json"]["root"] == kind
    monkeypatch.setenv("KAIMEI_DISPLAY", "raw")
    assert set(APIObject(record)._repr_mimebundle_()[0]) == {"application/json", "text/plain"}


def test_a_card_that_cannot_be_drawn_falls_back_to_the_plain_representation(monkeypatch):
    def broken(data):
        raise KeyError("a shape no card expects")

    monkeypatch.setattr(cards, "_monitoring", broken)
    bundle, metadata = APIObject(MONITORING)._repr_mimebundle_()
    assert bundle == {"text/plain": "<Kaimei monitoring>"} and metadata == {}
    with pytest.raises(KeyError):
        display.html(MONITORING)  # asked for by name, the failure is not hidden


def test_monitoring_card_reads_the_latest_window_and_draws_every_window():
    html = _card(MONITORING)
    assert "Monitoring" in html and "mdl_a0fee1c154842f759c51" in html and "read Nov 27, 2026, 00:22 UTC" in html
    assert 'class="km-badge bad"><i class=km-dot></i>Significant drift<' in html and "600 reference vs 500 logged rows" in html
    assert ">1,002<" in html and "last 30 days" in html and ">1.26<" in html and "on temperature" in html
    assert ">0.085<" in html and "what 500 reference rows reach" in html and ">3,169 MW<" in html and "demand_mw · Nov 27" in html
    # One bar per window, oldest first, toned by what its check amounts to, and every tone written out with its count.
    bars = re.findall(r'<i class="km-win-bar km-tone-(\w+)" title="([^"]*)"', html)
    assert [tone for tone, _title in bars] == ["none", "chance", "stable", "significant", "significant"]
    assert bars[1][1] == "Nov 13: Within chance, PSI 0.776 on wind_speed, chance reaches 1.75, 39 predictions"
    assert bars[0][1] == "Nov 12: Not enough rows, 0 predictions" and bars[4][1].startswith("Nov 27: Significant drift, PSI 1.26 on temperature")
    for words in ("stable · 1", "within chance · 1", "significant drift · 2", "not enough rows · 1"):
        assert words in html
    assert "moderate drift ·" not in html  # no window read as that
    assert html.count('class="km-win-line"') == 2 and "dashed lines at 0.1 and 0.25 · tallest bar PSI 1.26" in html
    styled = display.html(MONITORING)
    heights = [float(h) for h in re.findall(r'class="km-win-bar km-tone-\w+" style="[^"]*height:([0-9.]+)%', styled)]
    assert heights == [61.4, 3.0, 27.7, 100.0] and "bottom:7.9%" in styled and "bottom:19.8%" in styled  # one scale, the lines on it
    assert re.search(r'class="km-win-bar km-tone-significant" style="[^"]*background:var\(--bad,#d14b7a\);height:100.0%"', styled)
    # The inputs by peak PSI, each with its latest reading beside its own chance level.
    assert html.index("<b>temperature</b>") < html.index("<b>humidity</b>") < html.index("<b>wind_speed</b>")  # then by latest PSI
    assert 'class="km-fill significant"' in html and "It never scored above its own chance level" in html
    assert html.count('class="km-badge ok">Stable<') == 1 and 'class="km-badge">Within chance<' in html  # humidity: 0.110 under 0.159
    assert "/models/mdl_a0fee1c154842f759c51/monitoring" in html and "nothing was run or stored" in html


def test_monitoring_card_says_each_reading_in_words():
    def latest(**changed):
        return dict(MONITORING, series=MONITORING["series"][:-1] + [dict(MONITORING["series"][-1], **changed)])

    chance = _card(latest(status="moderate", overall_psi=0.131, top_feature="loan_amount", chance=0.265, beyond_chance=False, rows=192))
    assert _said(chance) == (
        "Within chance in the latest window: loan_amount scores PSI 0.131, past a drift threshold, but 192 rows drawn from the "
        "reference data itself reach 0.265."
    )
    assert 'class="km-badge"><i class=km-dot></i>Within chance<' in chance  # a neutral badge: the words carry it
    stable = _card(latest(status="stable", overall_psi=0.0365383, top_feature="hour", chance=0.0859323, beyond_chance=False))
    assert _said(stable) == "Stable in the latest window: the highest PSI is 0.037, on hour, where 500 rows drawn from the reference data itself reach 0.086."
    moderate = _card(latest(status="moderate", overall_psi=0.18, chance=0.0848125, beyond_chance=True))
    assert _said(moderate).startswith("Moderate drift in the latest window: temperature scores PSI 0.180, where 500 rows")
    assert 'class="km-badge warn"><i class=km-dot></i>Moderate drift<' in moderate
    older = _card(latest(status="significant", overall_psi=0.4, chance=None, beyond_chance=None))  # an API before chance levels
    assert _said(older) == "Significant drift in the latest window: temperature scores PSI 0.400." and "Chance level" not in older
    thin = _card(dict(MONITORING, reference_rows=600, series=[_window(26, 7, 0.4, rows=7)]))
    assert _said(thin) == (
        "Not enough rows for a drift reading in the latest window: a check compares at least 20 logged rows with at least 20 "
        "reference rows, and this one had 7 and 600."
    )
    assert "Not enough rows<" in thin and "Predictions per day" in thin and 'class="km-win-bar km-tone-traffic"' in thin
    # A model whose inputs are not a table: the traffic is drawn, and the API's own sentence says why there is no drift.
    why = "Drift over time is worked out for models whose inputs are all single values (a table)."
    traffic = [_window(day, count, 0.5) for day, count in ((24, 10), (25, 0), (26, 30))]
    picture = _card(dict(MONITORING, drift_unavailable=why, features=[], series=traffic, predictions=40, reference_rows=0))
    assert _said(picture) == f"40 predictions logged in the last 30 days. {why}" and "No drift reading<" in picture
    assert "Predictions per day" in picture and "busiest day: 30 predictions" in picture and "Inputs by peak PSI" not in picture
    assert re.findall(r'km-win-bar km-tone-(\w+)"', picture) == ["traffic", "none", "traffic"]
    assert _said(_card(dict(MONITORING, predictions=0, series=[]))) == "No predictions were logged in the last 30 days."


def test_population_card_names_the_field_that_separates_the_answers_and_draws_its_groups():
    html = _card(POPULATION)
    assert "Population" in html and ">192<" in html and ">173<" in html and "of 192 shown" in html and ">0.5<" in html
    assert "By credit_score" in html and "input · 32% of the variation in the answers" in html
    # A classifier's groups: the classes it predicted as one split bar, the same colour for a class in every group.
    assert '<span><i class="km-mix-0"></i>approve</span><span><i class="km-mix-1"></i>decline</span>' in html
    assert html.count('class="km-split"') == 3 and 'title="approve: 5, decline: 44"' in html
    styled = display.html(POPULATION)
    assert re.search(r'<i class="km-mix-0" style="background:var\(--series-1,#4566e8\);[^"]*flex:5 1 0%"></i><i class="km-mix-1" style="[^"]*flex:44 1 0%"', styled)
    assert "<b>≤ 643</b>" in html and ">49<" in html and "90% decline" in html and ">0.763<" in html and "55% approve" in html
    assert "Mean P(decline)" in html and "Most predicted" in html
    # The next fields by how far they separate the answers; one that separates nothing is not listed.
    assert "employment_years 7.0%" in html and "region" not in html.split("Then</div>")[1]
    assert "/models/mdl_f075cc4cf5745563ec60/population" in html and "nothing was run" in html


def test_population_card_of_a_regression_a_cut_log_and_no_ranking():
    numbers = dict(
        POPULATION, predictions=12345, shown=2000, truncated=True,
        output={"kind": "value", "name": "demand_mw", "unit": "MW", "threshold": None, "classes": [], "positive": None},
        fields=[{"key": "input:temperature", "name": "temperature", "source": "input", "type": "numeric", "unit": "°C", "separation": 0.7248941251137369,
                 "groups": [{"label": "≤ 12.9", "rows": 250, "mean": 3448.66383}, {"label": "> 41.2", "rows": 251, "mean": 7954.020675}]}],
        rows={"id": [], "value": [3000.0, 5000.0]},
    )  # fmt: skip
    html = _card(numbers)
    assert _said(html) == (
        "Of the newest 2,000 of 12,345 predictions logged in the last 30 days, temperature separates the answers most: 72% of "
        "their variation lies between its groups."
    )
    assert 'class="km-split"' not in html and html.count('class="km-fill share"') == 2 and ">7,954 MW<" in html and ">4,000 MW<" in html
    assert "input, °C · 72% of the variation in the answers" in html and "Mean demand_mw" in html
    assert "width:43.36%" in display.html(numbers) and "width:100.00%" in display.html(numbers)  # bars from zero, on one scale
    below = dict(numbers, fields=[dict(numbers["fields"][0], groups=[{"label": "a", "rows": 3, "mean": -2.0}, {"label": "b", "rows": 4, "mean": 1.0}])])
    assert 'class="km-fill neg"' in _card(below) and 'class="km-fill pos"' in _card(below)  # either side of zero when some are below it
    flat = dict(POPULATION, fields=[dict(field, separation=0.0 if field["separation"] is not None else None) for field in POPULATION["fields"]])
    html = _card(flat)
    assert _said(html) == (
        "Of 192 predictions logged in the last 30 days, no field separates the answers: none of the 3 ranked for it accounts "
        "for any of their variation."
    )
    assert "By predicted class" in html and "the answers themselves" in html and "Then</div>" not in html
    one = dict(flat, fields=flat["fields"][2:])
    assert _said(_card(one)) == (
        "Of 192 predictions logged in the last 30 days, the one field ranked by how far it separates the answers (region) "
        "accounts for none of their variation."
    )


def test_comparison_card_says_how_far_two_models_agree_and_where_they_part():
    html = _card(COMPARISON)
    assert "Comparison" in html and "loan-forest-v1 vs loan-scorecard-v1" in html and "3 reference rows" in html
    assert ">66.7%<" in html and "2 of 3 rows" in html and ">0.91<" in html and ">0.1<" in html and ">0.38<" in html and "in P(decline)" in html
    # Each model's score with the rows it is measured on, and where it draws its line.
    assert "<b>Loan forest</b>" in html and ">73.8%<" in html and ">0.810<" in html and ">74.6%<" in html and ">0.830<" in html
    assert "Rows with truth" in html and html.count(">decline from 0.5<") == 2
    # The rows furthest apart, the furthest first, with both answers and what really happened.
    rows = re.findall(r'<tr><td class="mono">(\w+)</td>(.*?)</tr>', html.split("Where they differ most")[1])
    assert [row for row, _cells in rows] == ["204", "0", "1"]
    assert '0.22 <span class="km-muted">approve</span>' in rows[0][1] and '0.602 <span class="km-muted">decline</span>' in rows[0][1]
    assert ">0.381<" in rows[0][1] and rows[0][1].endswith('<td class="muted">—</td>') and rows[1][1].endswith('<td class="muted">decline</td>')
    assert 'title="loan-forest-v1: 0.22, loan-scorecard-v1: 0.602"' in html and "with a line at the threshold 0.5" in html
    styled = display.html(COMPARISON)
    assert re.search(r'<i class="km-pin km-mix-0" style="[^"]*left:22.0%"></i><i class="km-pin km-mix-1" style="[^"]*left:60.2%"></i>', styled)
    assert "/models/loan-forest-v1/compare?with=loan-scorecard-v1" in html and "both models were run and nothing was stored" in html
    numbers = dict(
        COMPARISON, source="predictions",
        output={"kind": "value", "name": "load", "unit": "MW", "classes": [], "positive": None},
        agreement={"rows": 3, "rate": None, "same": None, "different": None, "correlation": 0.91, "mean_gap": 0.1, "max_gap": 0.38},
        scores=[{"n": 2, "mae": 0.5, "r2": 0.9}, {"n": 2, "mae": 0.25, "r2": None}],
        rows=dict(COMPARISON["rows"], decisions=None, actual=[3.5, None, 2]),
    )  # fmt: skip
    html = _card(numbers)
    assert _said(html) == "The answers of Loan forest and Loan scorecard correlate at 0.91 over 3 logged predictions, 0.1 MW apart on average."
    assert "Agreement" not in html and "MAE" in html and "R²" in html and ">0.5 MW<" in html and ">0.900<" in html
    assert "drawn from 0.22 MW to 0.602 MW" in html and "threshold" not in html
    assert '<td class="muted">2 MW</td>' in html  # what really happened is a number here


def test_similar_cases_card_lists_each_case_with_what_is_known_and_where_it_differs():
    html = _card(SIMILAR)
    assert "Similar cases" in html and "2 nearest of 600 reference rows" in html
    assert "labelled approve · 1" in html and "labelled decline · 1" in html and "nearest 99.0% alike" in html
    assert 'Asked about <span class="km-mono">income 52,000 · home rent</span>.' in html
    assert html.index(">251<") < html.index(">377<") and ">99.0%<" in html and ">46.2%<" in html and "<b>approve</b>" in html
    assert '<b>home</b> own <span class="km-faint">vs rent</span> · <b>income</b> 44,024 <span class="km-faint">vs 52,000</span>' in html
    assert "width:99.0%" in display.html(SIMILAR) and "/models/mdl_1/explain" in html
    logged = dict(
        SIMILAR, source="predictions", prediction_id="pred_9", considered=191,
        cases=[{"id": "pred_7", "distance": 0.0, "similarity": 1.0, "inputs": {"income": 52000}, "differs": [], "time": "2026-11-25T10:00:00Z",
                "output": {"label": "decline"}, "predicted": "decline", "value": 0.81, "outcome": None},
               dict(SIMILAR["cases"][1], id="pred_8", predicted="approve", value=0.2, outcome="approve", time="2026-11-24T09:30:00Z")],
        summary={"cases": 2, "labels": None, "predicted": {"known": 2, "counts": {"decline": 1, "approve": 1}},
                 "outcomes": {"known": 1, "counts": {"approve": 1}},
                 "sentence": "Of the 2 logged predictions most like this case, the model predicted decline for 1 and approve for 1."},
    )  # fmt: skip
    html = _card(logged)
    assert _said(html).startswith("Of the 2 logged predictions most like this case") and "2 nearest of 191 logged predictions" in html
    assert "predicted decline · 1" in html and "outcome approve · 1" in html and 'Asked about prediction <span class="km-mono">pred_9</span>.' in html
    assert '<b>decline</b> <span class="km-faint">0.81</span><span class="km-note">outcome not recorded</span>' in html
    assert '<span class="km-note">outcome approve</span>' in html and "Nov 25, 10:00" in html and "the same on every input" in html
    apart = dict(logged, cases=[dict(logged["cases"][0], distance=0.2, similarity=0.8)])
    assert "the same on every input" not in _card(apart)  # said only of a case at no distance
    assert "/models/mdl_1/predictions" in html
    many = dict(SIMILAR, cases=[dict(SIMILAR["cases"][0], id=index) for index in range(15)])
    assert "3 more in similar.cases" in _card(many) and _card(many).count("<tr>") == 13


def test_holdings_card_sets_what_stays_with_you_beside_what_kaimei_holds():
    html = _card(HOLDINGS)
    assert "What Kaimei holds" in html and "read Nov 27, 2026, 00:22 UTC" in html and 'class="km-badge info">Run by Kaimei<' in html
    assert "3 of 4 items held" in html and "158 KB in 2 measured items" in html
    yours, theirs = html.split('<div class="km-side">')[1:]
    assert "Stays with you" in yours and "Held by Kaimei" in theirs
    assert "<b>Credential for your endpoint</b>" in yours and "No credential is held: this model has no endpoint for Kaimei to call." in yours
    assert "0 headers" not in yours  # a count of nothing is not a measure
    assert '<b>Reference rows</b><span class="km-measure">600 rows · 158 KB</span>' in theirs and "1 insight · 734 bytes" in theirs
    # A holding with no count and no size is read from its sentence; a measured one keeps it for the hover.
    assert "its weights are part of Kaimei, and nothing was uploaded for it." in theirs.split("<b>Reference rows</b>")[0]
    assert 'class="km-note">600 rows, each with its label' not in theirs and 'title="600 rows, each with its label.' in theirs
    assert "What travels between your network and Kaimei is in holdings.crosses, for predict, log." in html
    assert "/models/mdl_f075cc4cf5745563ec60/settings" in html
    summary = dict(HOLDINGS["items"][0], held="partly", count=12, unit="parameter", bytes=2048, detail="The structure was sent as a summary: shapes without values.")
    yours_run = dict(HOLDINGS, runs={"by": "your_agent", "sentence": "An agent in your network runs this model."}, items=[summary, HOLDINGS["items"][3]])
    html = _card(yours_run)
    assert "Run by your agent" in html and "12 parameters · 2.05 KB" in html and 'class="km-badge warn">held in part<' in html
    assert "The structure was sent as a summary: shapes without values." in html.split("Held by Kaimei")[1]
    none_held = _card(dict(HOLDINGS, items=[HOLDINGS["items"][3]]))
    assert "Nothing on this list: Kaimei holds none of these." in none_held and "0 of 1 item held" in none_held
    all_held = _card(dict(HOLDINGS, items=HOLDINGS["items"][:3]))
    assert "Nothing on this list: Kaimei holds at least part of every item." in all_held
    # An item that does not say whether it is held (an older record) is claimed for neither side.
    unsaid = _card(dict(HOLDINGS, items=[{"key": "structure", "label": "Weights and structure", "count": 3, "unit": "tensor"}, HOLDINGS["items"][1]]))
    assert "The record does not say whether Kaimei holds: Weights and structure." in unsaid and "1 of 2 items held" in unsaid
    assert "<b>Weights and structure</b>" not in unsaid


def test_method_comparison_card_draws_each_methods_top_three_on_one_scale():
    html = _card(METHODS)
    assert "Method comparison" in html and "credit-risk-v3" in html and "3 methods" in html
    assert f"rank agreement {METHODS['agreement']['rank_correlation']:.2f}" in html and "answer approve · 87.9%" in html and "6 inputs" in html
    assert "pushes toward decline" in html and "pushes away" in html and "pushed opposite ways" not in html
    blocks = html.split('<div class="km-method">')[1:]
    assert [re.match(r"<span>([^<]*)</span>", block).group(1) for block in blocks] == ["KernelSHAP", "Permutation Shapley", "LIME"]
    assert 'class="km-badge ok" title="all 510 coalitions of the inputs over 64 reference rows">exact<' in blocks[0]
    assert "32.8K model calls" in blocks[0] and "758 ms" in blocks[0] and ">approximate<" in blocks[1] and "514 model calls" in blocks[2]
    assert "a straight-line fit near this input, R² 0.77" in blocks[2]
    # Its own top three, by size: LIME puts credit_score first where the other two put income.
    assert re.findall(r"<b>(\w+)</b>", blocks[0]) == ["income", "credit_score", "loan_amount"]
    assert re.findall(r"<b>(\w+)</b>", blocks[2]) == ["credit_score", "income", "debt_to_income"]
    assert "· 162,739" in blocks[0] and "−0.278" in blocks[0] and "+0.103" in blocks[2]
    # One scale for every bar: only the largest attribution of all (−0.28, Permutation Shapley) fills its half.
    widths = [float(w) for w in re.findall(r'class="km-fill (?:pos|neg)" style="[^"]*;width:([0-9.]+)%', display.html(METHODS))]
    assert len(widths) == 9 and widths.count(50.0) == 1 and widths[3] == 50.0 and widths[6] == round(0.1683 / 0.28 * 50, 2)
    assert "3 explanations stored, one per method" in html and "/explanations/exp_shap" in html
    priced = [dict(item, output={"value": 3534.845, "name": "demand_mw", "unit": "MW"}) for item in _EXPLAINED]
    html = _card(dict(METHODS, explanations=priced))
    assert "answer 3,534.845 MW" in html and "pushes higher" in html and "pushes lower" in html  # a regression's answer and legend
    opposed = dict(METHODS, agreement=dict(METHODS["agreement"], opposed=["loan_amount"], rank_correlation=None))
    assert "pushed opposite ways: loan_amount" in _card(opposed) and "rank agreement not defined" in _card(opposed)
    # An older record without the explanations: the top three come from its rows.
    assert re.findall(r"<b>(\w+)</b>", _card(dict(METHODS, explanations=[])).split('<div class="km-method">')[3]) == ["credit_score", "income", "debt_to_income"]


def test_log_cards_keep_their_layout_where_style_blocks_are_dropped():
    """What GitHub's notebook viewer keeps of each new card still places its bars, columns and dots."""
    body = _body(MONITORING)
    assert "<style" not in body and re.search(r'<div style="[^"]*display:flex;align-items:flex-end;gap:2px;height:64px[^"]*" class="km-win">', body)
    assert re.search(r'<span class="km-win-line" style="position:absolute;[^"]*border-top:1px dashed var\(--faint,#7a8096\)[^"]*bottom:7.9%"', body)
    assert re.search(r'<div style="font-size:15px;[^"]*color:var\(--text,#12141c\)[^"]*" class="km-say">', body)
    body = _body(POPULATION)
    assert re.search(r'<div style="display:flex;gap:2px;height:10px[^"]*" class="km-split"', body)
    body = _body(HOLDINGS)
    assert body.count("flex:1 1 280px") == 2 and re.search(r'style="display:flex;gap:10px;flex-wrap:wrap[^"]*" class="km-sides"', body)
    body = _body(COMPARISON)
    assert re.search(r'class="km-pin km-mix-1" style="background:var\(--series-2,#0891b2\);position:absolute;[^"]*border-radius:50%;[^"]*left:', body)
