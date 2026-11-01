"""Notebook display: the themed cards, the raw mode and the DataFrame / JSON exports."""

from __future__ import annotations

import json
import re

import pytest

import kaimei
from kaimei import APIObject, Page, display
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
