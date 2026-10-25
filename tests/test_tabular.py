from __future__ import annotations

import subprocess
import sys

import pytest

from kaimei._objects import APIObject
from kaimei.tabular import (
    AWAY,
    TOWARD,
    compare_explanations,
    counterfactual_summary,
    explanation_summary,
    plot_counterfactual,
    plot_explanation,
    plot_insight,
    plot_insights,
)

MODEL = APIObject(
    {
        "object": "model",
        "slug": "titanic-forest",
        "task": "binary_classification",
        "features": [
            {"name": "Pclass", "type": "integer"},
            {"name": "Sex", "type": "categorical", "categories": ["female", "male"]},
            {"name": "Age", "type": "numeric", "unit": "years"},
            {"name": "Fare", "type": "numeric", "unit": "GBP"},
            {"name": "IsAlone", "type": "binary"},
        ],
        "outputs": {"labels": ["died", "survived"], "positive_label": "survived"},
    }
)
PASSENGER = {"Pclass": 3, "Sex": "male", "Age": 22.0, "Fare": 7.25, "IsAlone": False}


def _explanation(counterfactual=None, **overrides):
    record = {
        "object": "explanation",
        "id": "expl_1",
        "model_id": "mdl_forest",
        "method": "shap",
        "input": dict(PASSENGER),
        "output": {
            "label": "died",
            "probability": 0.926,
            "probabilities": {"died": 0.926, "survived": 0.074},
            "target_label": "survived",
            "value": 0.074,
        },
        "attributions": {"Pclass": -0.042, "Sex": -0.091, "Age": -0.008, "Fare": -0.037, "IsAlone": 0.0004},
        "base_value": 0.252,
        "options": {"exact": True, "background_rows": 58, "frozen_features": ["Sex", "Age"], "max_changes": 3},
        "counterfactual": counterfactual,
    }
    record.update(overrides)
    return APIObject(record)


SHORT = {
    "found": False,
    "changes": [
        {"feature": "Fare", "from": 7.25, "to": 73.5, "output_after": 0.229, "label_after": "died"},
        {"feature": "Pclass", "from": 3, "to": 1, "output_after": 0.359, "label_after": "died"},
    ],
    "output_before": 0.074,
    "output_after": 0.359,
    "target": {"kind": "flip_decision", "threshold": 0.5},
}
FLIPPED = {
    "found": True,
    "changes": [{"feature": "Pclass", "from": 3, "to": 1, "output_after": 0.541, "label_after": "survived"}],
    "output_before": 0.128,
    "output_after": 0.541,
    "target": {"kind": "flip_decision", "threshold": 0.5},
}


def _insight(kind, payload, summary="A `summary`."):
    return APIObject({"object": "insight", "id": f"ins_{kind}", "type": kind, "title": kind.title(),
                      "summary": summary, "payload": payload})


INSIGHTS = [
    _insight("drift", {"features": [{"feature": "Age", "psi": 0.03, "status": "stable"},
                                     {"feature": "Fare", "psi": 0.31, "status": "significant"}],
                       "overall_psi": 0.31, "status": "significant", "reference_rows": 891, "current_rows": 418,
                       "output_drift": {"psi": 0.01, "status": "stable"}}),
    _insight("global_importance", {"features": [{"feature": "Sex", "share": 0.27}, {"feature": "Fare", "share": 0.15}],
                                   "baseline_metric": {"name": "accuracy", "value": 0.85}}),
    _insight("behavior", {"histogram": {"counts": [5, 3, 2, 4], "edges": [0.0, 0.25, 0.5, 0.75, 1.0]},
                          "threshold": 0.5, "class_balance": {"survived": 0.4, "died": 0.6}}),
    _insight("partial_dependence", {"features": [
        {"feature": "Sex", "type": "categorical", "grid": ["female", "male"], "pd": [0.6, 0.3],
         "ice": [[0.8, 0.4], [0.5, 0.2]], "monotonicity": {"direction": "decreasing"}},
        {"feature": "Pclass", "type": "integer", "grid": [1, 2, 3], "pd": [0.5, 0.45, 0.3],
         "ice": [[0.6, 0.5, 0.4]], "monotonicity": {"direction": "decreasing"}},
        {"feature": "Age", "type": "numeric", "grid": [1.0, 20.0, 40.0], "pd": [0.5, 0.4, 0.38],
         "monotonicity": {"direction": "non_monotonic"}},
    ]}),
    _insight("interactions", {"pairs": [{"features": ["Sex", "Pclass"], "strength": 0.12}]}),
    _insight("robustness", {"epsilon": 0.05, "fragile_features": [{"feature": "Fare", "sensitivity": 0.03},
                                                                  {"feature": "Sex", "sensitivity": 0.0}]}),
    _insight("learned_rules", {"rules": [{"statement": "`Sex` = female pushes the probability of survived highest."}]}),
]


def test_summary_names_the_answer_the_average_and_the_inputs_that_matter():
    text = explanation_summary(_explanation(), model=MODEL)
    assert text.startswith("The model predicts died: P(survived) = 7.4%, against 25.2% on average")
    assert "Pulling P(survived) down: Sex = male (−9.1 pts), Pclass = 3 (−4.2 pts), Fare = 7.25 GBP" in text
    # IsAlone's +0.04 pts is under 5% of the largest attribution, so it is not worth a mention.
    assert "IsAlone" not in text and "Pushing" not in text


def test_summary_of_a_regression_keeps_the_output_units():
    record = _explanation(
        output={"value": 412.0, "name": "price", "unit": "k$"},
        attributions={"rooms": 30.0, "age": -12.0},
        input={"rooms": 4, "age": 31.5},
        base_value=394.0,
    )
    text = explanation_summary(record)
    assert text.startswith("The model predicts 412 k$, against 394 on average")
    assert "Pushing it up: rooms = 4 (+30)" in text
    assert "Pulling it down: age = 31.5 (−12)" in text


def test_counterfactual_summary_says_what_flips_the_decision_or_how_close_it_got():
    assert counterfactual_summary(_explanation(FLIPPED), model=MODEL) == (
        "Changing Pclass from 3 to 1 flips the decision to survived (Sex and Age held fixed): "
        "P(survived) 12.8% → 54.1%."
    )
    short = counterfactual_summary(_explanation(SHORT), model=MODEL)
    assert short.startswith("No change to at most 3 features flips the decision (Sex and Age held fixed).")
    assert "Fare from 7.25 GBP to 73.5 GBP and Pclass from 3 to 1" in short
    assert short.endswith("still short of the 50.0% threshold.")
    assert "counterfactual" in counterfactual_summary(_explanation())


def test_plot_explanation_walks_from_the_average_to_the_prediction():
    plt = pytest.importorskip("matplotlib.pyplot")
    from matplotlib.colors import to_hex

    fig = plot_explanation(_explanation(), model=MODEL, max_features=3)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["Average (reference data)", "Sex = male", "Pclass = 3", "Fare = 7.25 GBP",
                      "2 other features", "This prediction"]
    bars = [p for p in ax.patches if p.get_width() > 0]
    assert len(bars) == 4
    # Every row pulls P(survived) down, the folded Age + IsAlone row included (-0.8 + 0.04 pts).
    assert {to_hex(p.get_facecolor()) for p in bars} == {AWAY}
    texts = [t.get_text() for t in ax.texts]
    assert "25.2%" in texts and "7.4%" in texts and "threshold 50.0%" in texts
    # The bars add up: the last edge lands on the prediction.
    lefts = [p.get_x() for p in bars] + [p.get_x() + p.get_width() for p in bars]
    assert min(lefts) == pytest.approx(0.074, abs=1e-3)
    plt.close(fig)


def test_plot_counterfactual_marks_each_side_of_the_threshold():
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plot_counterfactual(_explanation(FLIPPED), model=MODEL)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Now", "Pclass\n3 → 1"]
    notes = [t.get_text() for t in ax.texts]
    assert "12.8%\ndied" in notes and "54.1%\nsurvived" in notes
    markers = [line.get_color() for line in ax.lines if line.get_marker() == "o"]
    assert markers == [AWAY, TOWARD]
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_counterfactual(_explanation())


def test_compare_explanations_draws_one_series_per_model():
    plt = pytest.importorskip("matplotlib.pyplot")
    other = _explanation(attributions={"Pclass": -0.126, "Sex": -0.209, "Age": 0.032, "Fare": -0.008, "IsAlone": 0.073})
    fig = compare_explanations([_explanation(), other], ["Forest", "Logistic"], model=MODEL)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Forest", "Logistic"]
    assert [t.get_text() for t in ax.get_yticklabels()][0] == "Sex = male"
    plt.close(fig)
    with pytest.raises(ValueError):
        compare_explanations([_explanation()] * 4)


@pytest.mark.parametrize("insight", INSIGHTS, ids=lambda insight: insight.type)
def test_every_insight_type_draws(insight):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plot_insight(insight, model=MODEL)
    assert fig.texts[0].get_text() == insight.title
    assert all("`" not in t.get_text() for t in fig.texts)
    # IPython silently skips a figure without axes, so even the text-only cards need one.
    assert fig.axes
    plt.close(fig)


def test_drift_bars_carry_their_status_in_words():
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plot_insight(INSIGHTS[0])
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert "0.310 · significant" in labels and "0.030 · stable" in labels
    plt.close(fig)


def test_plot_insights_reads_a_page_and_orders_the_report():
    plt = pytest.importorskip("matplotlib.pyplot")
    page = APIObject({"object": "list", "data": [i.to_dict() for i in INSIGHTS]})
    figures = plot_insights(page, model=MODEL)
    assert [f.texts[0].get_text() for f in figures] == [
        "Global_Importance", "Learned_Rules", "Partial_Dependence", "Interactions", "Behavior", "Robustness", "Drift",
    ]
    for fig in figures:
        plt.close(fig)
    with pytest.raises(ValueError):
        plot_insight(_insight("mystery", {}))


def test_importing_the_module_does_not_load_matplotlib():
    code = "import sys, kaimei.tabular; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_nested_objects_print_their_data():
    record = _explanation()
    assert repr(record) == "<Kaimei explanation expl_1>"
    assert repr(record.output).startswith("{'label': 'died'")


def test_regression_waterfall_labels_share_one_precision():
    plt = pytest.importorskip("matplotlib.pyplot")
    record = _explanation(
        output={"value": -3.3, "name": "log-odds of survived", "unit": "log-odds"},
        attributions={"Sex": -1.25, "Fare": -0.621, "Parch": 0.000554},
        input={"Sex": "male", "Fare": 7.25, "Parch": 0},
        base_value=-1.4286,
    )
    fig = plot_explanation(record)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert "−1.25" in labels and "−0.62" in labels and "+0.00" in labels
    plt.close(fig)


def test_drift_threshold_labels_sit_on_either_side_of_their_lines():
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plot_insight(INSIGHTS[0])
    labels = {t.get_text().strip(): t.get_horizontalalignment() for t in fig.axes[0].texts}
    assert labels["moderate"] == "right" and labels["significant"] == "left"
    plt.close(fig)
