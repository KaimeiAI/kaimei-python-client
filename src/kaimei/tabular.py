"""Tabular explanations and model insights, in words and in the dashboard's colours.

Everything here reads the objects the client returns (or plain dicts with the same fields)::

    from kaimei.tabular import (
        counterfactual_summary, explanation_summary, plot_counterfactual, plot_explanation, plot_insight,
    )

    model = client.models.retrieve("titanic-forest")
    explanation = client.explanations.create(
        model=model.slug, input=passenger, method="shap", options={"counterfactual": True}
    )
    print(explanation_summary(explanation, model=model))   # the answer and its main reasons, in words
    plot_explanation(explanation, model=model)             # waterfall from the average to this answer
    print(counterfactual_summary(explanation, model=model))
    plot_counterfactual(explanation, model=model)          # the smallest change that flips the decision
    for insight in client.insights.list(model=model.slug).data:
        plot_insight(insight, model=model)                 # importance, behaviour curves, drift, ...

``compare_explanations`` puts the same input explained by several models side by side.

Colours follow the dashboard: blue pushes the prediction toward the explained class, pink pushes
it away; model-level charts use the dashboard's series blue, and drift uses its status colours,
always next to a written status. Passing ``model`` (from ``client.models.retrieve``) adds the
units of its features (``Age = 22 years``) and names the output (``P(survived)``).

The text helpers need nothing beyond the client. The plots need matplotlib
(``pip install matplotlib``), which this module imports only when a plot is drawn.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "explanation_summary",
    "counterfactual_summary",
    "plot_explanation",
    "plot_counterfactual",
    "compare_explanations",
    "plot_insight",
    "plot_insights",
    "TOWARD",
    "AWAY",
    "SERIES",
    "STATUS",
]

#: Bars that push the prediction toward the explained class, and away from it (the dashboard's
#: blue and pink, one step darker so they keep 3:1 contrast on white).
TOWARD = "#4f8dff"
AWAY = "#e5609f"
#: Model-level series (importance, behaviour curves) and the models of ``compare_explanations``.
SERIES = ("#4566e8", "#0891b2", "#c2410c")
#: Drift status colours; every bar also carries the status in words.
STATUS = {"stable": "#0f8f5b", "moderate": "#9a6700", "significant": "#d14b7a"}

_INK, _MUTED, _FAINT, _GRID, _SURFACE = "#12141c", "#53566a", "#7a8096", "#e4e7f0", "#ffffff"
#: Individual (ICE) curves under a behaviour curve.
_ICE = "#c5cbe0"
_MINUS = "−"
#: Insight types in the order a report reads best.
_INSIGHT_ORDER = (
    "global_importance",
    "learned_rules",
    "partial_dependence",
    "interactions",
    "behavior",
    "robustness",
    "drift",
)

# --- words ---------------------------------------------------------------------------------------


def explanation_summary(explanation: Any, *, model: Any = None, top: int = 3) -> str:
    """The prediction and the inputs that moved it most, in a few sentences.

    Classification attributions are probabilities, so they read as percentage points of
    ``P(target)``; regression attributions keep the output's units.
    """
    output = _get(explanation, "output") or {}
    attributions = _attributions(explanation)
    inputs = _get(explanation, "input") or {}
    base = _get(explanation, "base_value")
    value = _output_value(explanation)
    classification = _is_classification(explanation)
    target = _target(explanation)
    if classification:
        label = output.get("label", target)
        sentence = f"The model predicts {label}: P({target}) = {_pct(value)}"
        if base is not None:
            sentence += f", against {_pct(float(base))} on average over the reference data"
    else:
        sentence = f"The model predicts {_num(value)}{_unit_suffix(output.get('unit'))}"
        if base is not None:
            sentence += f", against {_num(float(base))} on average over the reference data"
    sentence += "."
    # Name only contributions worth reading: at least 5% of the largest one.
    floor = 0.05 * max((abs(a) for a in attributions.values()), default=0.0)
    up = sorted(((n, a) for n, a in attributions.items() if a > floor), key=lambda item: -item[1])[:top]
    down = sorted(((n, a) for n, a in attributions.items() if a < -floor), key=lambda item: item[1])[:top]
    subject = f"P({target})" if classification else "it"

    def listing(items: Sequence[Tuple[str, float]]) -> str:
        return ", ".join(
            f"{_feature_text(name, inputs.get(name), model)} ({_delta(a, classification)})" for name, a in items
        )

    parts = []
    if up:
        parts.append(f"Pushing {subject} up: {listing(up)}.")
    if down:
        parts.append(f"Pulling {subject} down: {listing(down)}.")
    if base is not None and value is not None and float(value) < float(base):
        parts.reverse()
    return " ".join([sentence, *parts])


def counterfactual_summary(explanation: Any, *, model: Any = None) -> str:
    """What the counterfactual search found, in one or two sentences.

    Needs an explanation created with ``options={"counterfactual": True}``.
    """
    cf = _get(explanation, "counterfactual")
    if not cf:
        return "No counterfactual was requested; create the explanation with options={'counterfactual': True}."
    classification = _is_classification(explanation)
    target = _target(explanation)
    changes = list(cf.get("changes") or [])
    before = cf.get("output_before", _output_value(explanation))
    after = cf.get("output_after", before)
    goal = cf.get("target") or {}
    frozen = list((_get(explanation, "options") or {}).get("frozen_features") or [])
    held = f" ({_join(frozen)} held fixed)" if frozen else ""
    if not changes and cf.get("found"):
        return "The prediction already meets the target; nothing needs to change."
    edits = _join(_change_text(change, model) for change in changes)
    if classification:
        moved = f"P({target}) {_pct(before)} → {_pct(after)}"
        flipped = changes[-1].get("label_after") if changes else None
        if cf.get("found"):
            return f"Changing {edits} flips the decision to {flipped or 'the other class'}{held}: {moved}."
        threshold = goal.get("threshold", 0.5)
        limit = (_get(explanation, "options") or {}).get("max_changes", len(changes))
        if not changes:
            return f"No change to at most {limit} features moves the prediction toward the threshold{held}."
        return (
            f"No change to at most {limit} features flips the decision{held}. The closest: {edits}, "
            f"which takes {moved}, still short of the {_pct(threshold)} threshold."
        )
    moved = f"the prediction {_num(before)} → {_num(after)}"
    if cf.get("found"):
        return f"Changing {edits} reaches the target of {_num(goal.get('target_value'))}{held}: {moved}."
    return f"The closest the search got{held}: {edits}, which takes {moved}."


# --- explanation plots ---------------------------------------------------------------------------


def plot_explanation(
    explanation: Any,
    *,
    model: Any = None,
    max_features: int = 10,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
) -> Any:
    """A waterfall from the model's average prediction to this one, one row per input.

    The first row is the average over the reference data (the base value), each following row
    moves it by one feature's attribution (blue toward the explained class, pink away), the
    smallest ones fold into one "other features" row, and the last row is this prediction. For a
    binary classifier a hairline marks the decision threshold (``model``'s, or 0.5; pass
    ``threshold`` to override). Returns the matplotlib figure.
    """
    classification = _is_classification(explanation)
    target = _target(explanation)
    inputs = _get(explanation, "input") or {}
    attributions = _attributions(explanation)
    value = _output_value(explanation)
    base = _get(explanation, "base_value")
    if base is None:
        base = (value or 0.0) - sum(attributions.values())
    base = float(base)
    ordered = sorted(attributions.items(), key=lambda item: -abs(item[1]))
    shown = ordered[:max_features]
    rest = ordered[max_features:]
    rows: List[Tuple[str, float]] = [(_feature_text(name, inputs.get(name), model), a) for name, a in shown]
    if rest:
        rows.append((f"{len(rest)} other features", sum(a for _, a in rest)))
    end = base + sum(a for _, a in rows)
    cut = _threshold(explanation, model, threshold) if classification else None

    label_width = min(3.4, 0.35 + 0.075 * max(len(text) for text, _ in rows + [("This prediction", 0.0)]))
    n = len(rows) + 2
    fig, ax = _figure(
        plot_height=0.34 * n + 0.1,
        title=title or _explanation_title(explanation),
        subtitle=_explanation_subtitle(explanation, classification, target),
        label_width=label_width,
        right=1.0,
    )
    ys = list(range(n))
    running = base
    edges = [base]
    for index, (_, a) in enumerate(rows, start=1):
        left = min(running, running + a)
        ax.barh(index, abs(a), left=left, height=0.62, color=TOWARD if a >= 0 else AWAY, linewidth=0, zorder=3)
        # Connector from the previous bar (or the average's marker) to this bar.
        start = index - 1 + (0.31 if index > 1 else 0.0)
        ax.plot([running, running], [start, index - 0.31], color=_FAINT, linewidth=0.8, zorder=2)
        running += a
        edges.append(running)
    ax.plot([end, end], [n - 1.69, n - 1], color=_FAINT, linewidth=0.8, zorder=2)
    # End markers: the average and this prediction.
    for row, point in ((0, base), (n - 1, end)):
        ax.plot([point], [row], marker="o", markersize=8, color=_INK, markeredgecolor=_SURFACE, markeredgewidth=2,
                zorder=4)

    low, high = min(edges + [end]), max(edges + [end])
    if cut is not None:
        low, high = min(low, cut), max(high, cut)
    span = max(high - low, 1e-9)
    pad = 0.08 * span
    lo, hi = low - pad, high + pad
    if classification:
        lo, hi = max(0.0, lo), min(1.0, hi)
        if hi - lo < 0.05:
            lo, hi = max(0.0, lo - 0.025), min(1.0, hi + 0.025)
    ax.set_xlim(lo, hi)
    offset = 0.012 * (hi - lo)
    # Regression labels share one precision, so a tiny attribution reads +0.00, not +0.000554.
    decimals = None if classification else _shared_decimals(a for _, a in rows)
    running = base
    for index, (_, a) in enumerate(rows, start=1):
        right_end = max(running, running + a)
        ax.text(right_end + offset, index, _delta(a, classification, decimals), va="center", ha="left", fontsize=9,
                color=_MUTED)
        running += a
    ax.text(base + offset * 1.6, 0, _out(base, classification), va="center", ha="left", fontsize=9.5, color=_INK)
    ax.text(end + offset * 1.6, n - 1, _out(end, classification), va="center", ha="left", fontsize=9.5,
            color=_INK, fontweight="bold")
    if cut is not None:
        ax.axvline(cut, color=_MUTED, linewidth=1, zorder=1)
        ax.text(cut, -0.75, f"threshold {_pct(cut)}", ha="center", va="bottom", fontsize=8.5, color=_MUTED)
    labels = ["Average (reference data)"] + [text for text, _ in rows] + ["This prediction"]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(n - 0.4, -0.9)
    _style(ax, axis="x", percent=classification)
    ax.set_xlabel(f"P({target})" if classification else _output_name_of(explanation), color=_MUTED, fontsize=9)
    return fig


def plot_counterfactual(
    explanation: Any, *, model: Any = None, title: Optional[str] = None
) -> Any:
    """The counterfactual search step by step: the prediction after each change, and the target.

    Needs an explanation created with ``options={"counterfactual": True}``. Returns the figure.
    """
    cf = _get(explanation, "counterfactual")
    if not cf:
        raise ValueError("This explanation has no counterfactual; create it with options={'counterfactual': True}.")
    classification = _is_classification(explanation)
    target = _target(explanation)
    changes = list(cf.get("changes") or [])
    before = float(cf.get("output_before", _output_value(explanation) or 0.0))
    values = [before] + [float(change.get("output_after", before)) for change in changes]
    goal = cf.get("target") or {}
    line = goal.get("threshold") if classification else goal.get("target_value")
    labels = ["Now"] + [_change_label(change, model) for change in changes]
    fig, ax = _figure(
        plot_height=2.6,
        title=title or ("Smallest change that flips the decision" if cf.get("found") else "Closest the search got"),
        subtitle=counterfactual_summary(explanation, model=model),
        label_width=0.75,
        right=0.6,
        width=max(6.4, 1.9 * len(labels) + 1.4),
    )
    xs = list(range(len(values)))
    ax.plot(xs, values, color=_FAINT, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    other = _other_label(explanation)
    cut = float(line) if classification and line is not None else None
    for x, y in zip(xs, values):
        # Blue on the explained class's side of the threshold, pink on the other; the label says which.
        colour = SERIES[0] if cut is None else (TOWARD if y >= cut else AWAY)
        ax.plot([x], [y], marker="o", markersize=9, color=colour, markeredgecolor=_SURFACE, markeredgewidth=2,
                zorder=4)
        text = _out(y, classification)
        if cut is not None:
            text += f"\n{target if y >= cut else other}"
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 9), ha="center", va="bottom", fontsize=9,
                    color=_INK, linespacing=1.15)
    if line is not None:
        ax.axhline(float(line), color=_MUTED, linewidth=1, zorder=1)
        ax.annotate(
            ("threshold " if classification else "target ") + _out(float(line), classification),
            (-0.45, float(line)), textcoords="offset points", xytext=(2, 4), ha="left", fontsize=8.5,
            color=_MUTED,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.45, len(values) - 0.55)
    if classification:
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _style(ax, axis="y", percent=classification)
    ax.set_ylabel(f"P({target})" if classification else _output_name_of(explanation), color=_MUTED, fontsize=9)
    return fig


def compare_explanations(
    explanations: Sequence[Any],
    names: Optional[Sequence[str]] = None,
    *,
    model: Any = None,
    max_features: int = 10,
    title: Optional[str] = None,
) -> Any:
    """One input explained by several models (up to three), attribution by attribution.

    Features are ordered by their largest attribution in any model; bars right of the zero line
    push toward the explained class, bars to the left push away. Returns the figure.
    """
    explanations = list(explanations)
    if not 1 <= len(explanations) <= len(SERIES):
        raise ValueError(f"compare_explanations takes 1 to {len(SERIES)} explanations.")
    names = list(names) if names is not None else [str(_get(e, "model_id") or f"model {i + 1}")
                                                  for i, e in enumerate(explanations)]
    classification = _is_classification(explanations[0])
    target = _target(explanations[0])
    inputs = _get(explanations[0], "input") or {}
    tables = [_attributions(e) for e in explanations]
    features = sorted(
        {name for table in tables for name in table},
        key=lambda name: -max(abs(table.get(name, 0.0)) for table in tables),
    )[:max_features]
    rows = [_feature_text(name, inputs.get(name), model) for name in features]
    answers = " · ".join(
        f"{name}: {_out(_output_value(e), classification)}" for name, e in zip(names, explanations)
    )
    subtitle = (f"P({target}) for this input. " if classification else "Prediction for this input. ") + answers
    label_width = min(3.4, 0.35 + 0.075 * max(len(text) for text in rows))
    group = 0.78
    bar = group / len(explanations)
    fig, ax = _figure(
        plot_height=max(1.6, (0.22 + 0.2 * len(explanations)) * len(rows) + 0.2),
        title=title or "Same input, different models",
        subtitle=subtitle,
        label_width=label_width,
        right=0.4,
    )
    for k, (name, table) in enumerate(zip(names, tables)):
        offsets = [i - group / 2 + bar * (k + 0.5) for i in range(len(features))]
        ax.barh(offsets, [table.get(f, 0.0) for f in features], height=bar * 0.86, color=SERIES[k], label=name,
                linewidth=0, zorder=3)
    ax.axvline(0, color=_INK, linewidth=0.8, zorder=2)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(rows)
    ax.set_ylim(len(features) - 0.5, -0.5)
    _style(ax, axis="x", percent=False)
    if classification:
        ax.xaxis.set_major_formatter(_formatter(_points_tick))
        ax.set_xlabel(f"Attribution (percentage points of P({target}))", color=_MUTED, fontsize=9)
    else:
        ax.set_xlabel("Attribution", color=_MUTED, fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(names), frameon=False, fontsize=9,
              labelcolor=_INK, borderaxespad=0.2, handlelength=1.2)
    return fig


# --- insight plots -------------------------------------------------------------------------------


def plot_insight(insight: Any, *, model: Any = None, max_items: int = 10, ice: bool = True) -> Any:
    """Draw one insight from ``client.insights.list`` the way its dashboard card reads.

    ``global_importance``, ``interactions`` and ``robustness`` are ranked bars;
    ``partial_dependence`` draws one small panel per feature (with the individual curves when
    ``ice``); ``behavior`` is the histogram of predictions around the threshold; ``drift``
    colours each feature's PSI by status; ``learned_rules`` lists the statements. Returns the
    figure.
    """
    kind = str(_get(insight, "type") or "")
    drawers = {
        "global_importance": _plot_importance,
        "interactions": _plot_interactions,
        "robustness": _plot_robustness,
        "partial_dependence": _plot_partial_dependence,
        "behavior": _plot_behavior,
        "drift": _plot_drift,
        "learned_rules": _plot_rules,
    }
    if kind not in drawers:
        raise ValueError(f"No plot for insight type {kind!r}.")
    return drawers[kind](insight, model=model, max_items=max_items, ice=ice)


def plot_insights(insights: Iterable[Any], *, model: Any = None, types: Optional[Sequence[str]] = None) -> List[Any]:
    """``plot_insight`` for each insight, in report order (importance first, drift last)."""
    # A page from ``client.insights.list`` or any iterable of insights.
    items = list(insights.get("data") or []) if isinstance(insights, Mapping) else list(insights)
    wanted = list(types) if types is not None else list(_INSIGHT_ORDER)
    rank = {kind: i for i, kind in enumerate(wanted)}
    chosen = [item for item in items if str(_get(item, "type")) in rank]
    chosen.sort(key=lambda item: rank[str(_get(item, "type"))])
    return [plot_insight(item, model=model) for item in chosen]


def _plot_importance(insight: Any, *, model: Any, max_items: int, ice: bool) -> Any:
    payload = _get(insight, "payload") or {}
    features = list(payload.get("features") or [])[:max_items]
    shares = [float(f.get("share", f.get("importance", 0.0))) for f in features]
    metric = payload.get("baseline_metric") or {}
    note = ""
    if metric.get("name"):
        note = f"Model {metric['name']} on those rows: {float(metric.get('value', 0.0)):.3f}."
    return _ranked_bars(insight, [str(f.get("feature")) for f in features], shares,
                        [_pct(s, 0) for s in shares], note, percent=True)


def _plot_interactions(insight: Any, *, model: Any, max_items: int, ice: bool) -> Any:
    payload = _get(insight, "payload") or {}
    pairs = list(payload.get("pairs") or [])[:max_items]
    names = [" × ".join(str(f) for f in pair.get("features") or []) for pair in pairs]
    strengths = [float(pair.get("strength", 0.0)) for pair in pairs]
    return _ranked_bars(insight, names, strengths, [_pct(s, 0) for s in strengths],
                        "Share of each pair's joint effect that is not the sum of the two alone (Friedman's H).",
                        percent=True)


def _plot_robustness(insight: Any, *, model: Any, max_items: int, ice: bool) -> Any:
    payload = _get(insight, "payload") or {}
    rows = [f for f in payload.get("fragile_features") or [] if float(f.get("sensitivity", 0.0)) > 0][:max_items]
    values = [float(f.get("sensitivity", 0.0)) for f in rows]
    epsilon = float(payload.get("epsilon", 0.05))
    note = f"Mean change in the output when one input moves by {epsilon:.0%} of its spread."
    if not rows:
        return _message(insight, "No input moves the output under this much noise. " + note)
    return _ranked_bars(insight, [str(f.get("feature")) for f in rows], values, [f"{v:.3f}" for v in values], note,
                        percent=False)


def _plot_partial_dependence(insight: Any, *, model: Any, max_items: int, ice: bool) -> Any:
    plt = _pyplot()
    payload = _get(insight, "payload") or {}
    curves = list(payload.get("features") or [])[: min(max_items, 9)]
    if not curves:
        return _message(insight, "No behaviour curves in this insight.")
    output = _model_output_name(model)
    probability = output.startswith("P(")
    columns = min(3, len(curves))
    lines = math.ceil(len(curves) / columns)
    width = 3.0 * columns + 0.6
    fig = plt.figure(figsize=(width, 1.0))
    title, subtitle = _clean(_get(insight, "title")), _clean(_get(insight, "summary"))
    top = _header(fig, width, title, subtitle)
    panel_h, gap_h, bottom = 2.0, 0.75, 0.25
    height = top + lines * (panel_h + gap_h) + bottom
    fig.set_size_inches(width, height)
    _header(fig, width, title, subtitle)
    values = [float(v) for c in curves for v in (c.get("pd") or [])]
    if ice:
        values += [float(v) for c in curves for row in (c.get("ice") or []) for v in row]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    pad = 0.06 * max(high - low, 1e-9)
    ylim = (max(0.0, low - pad), min(1.0, high + pad)) if probability else (low - pad, high + pad)
    for index, curve in enumerate(curves):
        row, col = divmod(index, columns)
        left = 0.55 + col * 3.0
        y0 = height - top - (row + 1) * (panel_h + gap_h) + gap_h
        ax = fig.add_axes([left / width, y0 / height, 2.45 / width, panel_h / height])
        grid = list(curve.get("grid") or [])
        pd_values = [float(v) for v in curve.get("pd") or []]
        kind = str(curve.get("type") or "numeric")
        nominal = kind in ("categorical", "binary") or any(isinstance(g, str) for g in grid)
        xs: List[float] = list(range(len(grid))) if nominal else [float(g) for g in grid]
        if ice:
            for row_values in curve.get("ice") or []:
                ax.plot(xs, [float(v) for v in row_values], color=_ICE, linewidth=0.7, zorder=1,
                        linestyle="none" if nominal else "-", marker="o" if nominal else None, markersize=2.5)
        if nominal:
            ax.plot(xs, pd_values, linestyle="none", marker="o", markersize=8, color=SERIES[0],
                    markeredgecolor=_SURFACE, markeredgewidth=2, zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels([_short_value(g) for g in grid], fontsize=8.5, rotation=0)
            ax.set_xlim(-0.5, len(xs) - 0.5)
        else:
            ax.plot(xs, pd_values, color=SERIES[0], linewidth=2, solid_capstyle="round", zorder=3)
            if kind == "integer" and len(xs) <= 12:
                ax.set_xticks(xs)
                ax.set_xticklabels([_short_value(g) for g in grid], fontsize=8.5)
        ax.set_ylim(*ylim)
        _style(ax, axis="y", percent=probability, size=8.5)
        unit = _feature_unit(model, str(curve.get("feature")))
        direction = str((curve.get("monotonicity") or {}).get("direction") or "").replace("_", "-")
        ax.set_title(f"{curve.get('feature')}" + (f" ({unit})" if unit else ""), loc="left", fontsize=10,
                     color=_INK)
        if direction and not nominal:  # categories have no order to rise or fall along
            ax.text(1.0, 1.02, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    color=_MUTED)
        if col == 0:
            ax.set_ylabel(output, color=_MUTED, fontsize=9)
    return fig


def _plot_behavior(insight: Any, *, model: Any, max_items: int, ice: bool) -> Any:
    payload = _get(insight, "payload") or {}
    histogram = payload.get("histogram") or {}
    counts = [float(c) for c in histogram.get("counts") or []]
    edges = [float(e) for e in histogram.get("edges") or []]
    if not counts or len(edges) != len(counts) + 1:
        return _message(insight, "No histogram in this insight.")
    threshold = payload.get("threshold")
    output = _model_output_name(model)
    probability = threshold is not None
    fig, ax = _figure(plot_height=2.6, title=_clean(_get(insight, "title")),
                      subtitle=_clean(_get(insight, "summary")), label_width=0.7, right=0.35)
    balance = payload.get("class_balance") or {}
    names = list(balance.keys())
    for left, right, count in zip(edges, edges[1:], counts):
        middle = (left + right) / 2
        colour = SERIES[0] if threshold is None else (TOWARD if middle >= float(threshold) else AWAY)
        ax.bar(left, count, width=right - left, align="edge", color=colour, edgecolor=_SURFACE, linewidth=1.5,
               zorder=3)
    if threshold is not None:
        ax.axvline(float(threshold), color=_INK, linewidth=1, zorder=4)
        ax.text(float(threshold) + 0.008, max(counts) * 1.1, f"threshold {_pct(float(threshold))}", ha="left",
                va="top", fontsize=8.5, color=_MUTED)
        if len(names) == 2:
            from matplotlib.patches import Patch

            handles = [
                Patch(color=TOWARD, label=f"predicted {names[0]} ({_pct(float(balance[names[0]]), 0)})"),
                Patch(color=AWAY, label=f"predicted {names[1]} ({_pct(float(balance[names[1]]), 0)})"),
            ]
            ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, ncol=2,
                      bbox_to_anchor=(1.0, 1.0), labelcolor=_INK, borderaxespad=0.1)
    ax.set_ylim(0, max(counts) * 1.15)
    _style(ax, axis="y", percent=False)
    if probability:
        ax.xaxis.set_major_formatter(_formatter(lambda v: _pct(v, 0)))
    ax.tick_params(axis="x", colors=_MUTED, labelsize=9)
    ax.set_xlabel(output, color=_MUTED, fontsize=9)
    ax.set_ylabel("rows", color=_MUTED, fontsize=9)
    return fig


def _plot_drift(insight: Any, *, model: Any, max_items: int, ice: bool) -> Any:
    payload = _get(insight, "payload") or {}
    features = list(payload.get("features") or [])[:max_items]
    if payload.get("status") == "insufficient_data" or not features:
        return _message(insight, _clean(_get(insight, "summary")) or "Not enough data to measure drift.")
    names = [str(f.get("feature")) for f in features]
    psi = [float(f.get("psi", 0.0)) for f in features]
    statuses = [str(f.get("status") or "stable") for f in features]
    note = (
        f"Population stability index of each input: {payload.get('reference_rows', '?')} reference rows vs. "
        f"the latest {payload.get('current_rows', '?')} logged predictions. Below 0.1 is stable, above 0.25 significant."
    )
    output_drift = payload.get("output_drift") or {}
    if output_drift:
        note += f" The predictions themselves: PSI {float(output_drift.get('psi', 0.0)):.3f} ({output_drift.get('status')})."
    fig, ax = _bars_figure(insight, names, note)
    for index, (value, status) in enumerate(zip(psi, statuses)):
        ax.barh(index, value, height=0.62, color=STATUS.get(status, SERIES[0]), linewidth=0, zorder=3)
    high = max(psi + [0.27])
    # "moderate" ends at its line and "significant" starts at its own, so they never overlap.
    for line, word, align in ((0.1, "moderate ", "right"), (0.25, " significant", "left")):
        ax.axvline(line, color=_FAINT, linewidth=0.8, zorder=1)
        ax.text(line, -0.75, word, ha=align, va="bottom", fontsize=8, color=_MUTED)
    ax.set_xlim(0, high * 1.22)
    for index, (value, status) in enumerate(zip(psi, statuses)):
        ax.text(value + high * 0.015, index, f"{value:.3f} · {status}", va="center", fontsize=8.5,
                color=_MUTED)
    _style(ax, axis="x", percent=False)
    ax.set_xlabel("PSI", color=_MUTED, fontsize=9)
    return fig


def _plot_rules(insight: Any, *, model: Any, max_items: int, ice: bool) -> Any:
    payload = _get(insight, "payload") or {}
    rules = [_clean(rule.get("statement")) for rule in payload.get("rules") or []][:max_items]
    if not rules:
        return _message(insight, "No stable relationships were found.")
    plt = _pyplot()
    width = 8.0
    wrapped = [textwrap.wrap(rule, _chars(width - 0.9, 10)) for rule in rules]
    fig = plt.figure(figsize=(width, 1.0))
    title = _clean(_get(insight, "title"))
    top = _header(fig, width, title, "Read from the behaviour curves of the most important inputs.")
    line_h = 0.21
    height = top + sum(len(lines) * line_h + 0.12 for lines in wrapped) + 0.2
    fig.set_size_inches(width, height)
    _header(fig, width, title, "Read from the behaviour curves of the most important inputs.")
    _canvas(fig)
    y = height - top - 0.05
    for number, lines in enumerate(wrapped, start=1):
        fig.text(0.25 / width, y / height, f"{number}.", ha="left", va="top", fontsize=10, color=_MUTED)
        fig.text(0.55 / width, y / height, "\n".join(lines), ha="left", va="top", fontsize=10, color=_INK,
                 linespacing=1.35)
        y -= len(lines) * line_h + 0.12
    return fig


# --- layout --------------------------------------------------------------------------------------

_WIDTH = 8.0
_TITLE_SIZE, _SUBTITLE_SIZE = 12.5, 9.5


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("kaimei.tabular plots need matplotlib: pip install matplotlib") from exc
    return plt


def _chars(width_inches: float, fontsize: float) -> int:
    """Characters per line of text ``width_inches`` wide (DejaVu Sans averages ~0.55 em)."""
    return max(20, int(width_inches * 72 / (fontsize * 0.58)))


def _header(fig: Any, width: float, title: str, subtitle: str) -> float:
    """Draw the title and wrapped subtitle at the top left; returns the inches they take."""
    for text in list(fig.texts):
        if getattr(text, "_kaimei_header", False):
            text.remove()
    height = fig.get_size_inches()[1]
    lines = textwrap.wrap(subtitle, _chars(width - 0.5, _SUBTITLE_SIZE)) if subtitle else []
    used = 0.18
    heading = fig.text(0.22 / width, 1 - used / height, title, ha="left", va="top", fontsize=_TITLE_SIZE,
                       color=_INK, fontweight="bold")
    heading._kaimei_header = True
    used += 0.3
    if lines:
        body = fig.text(0.22 / width, 1 - used / height, "\n".join(lines), ha="left", va="top",
                        fontsize=_SUBTITLE_SIZE, color=_MUTED, linespacing=1.35)
        body._kaimei_header = True
        used += 0.2 * len(lines) + 0.04
    return used + 0.25


def _figure(
    *,
    plot_height: float,
    title: str,
    subtitle: str,
    label_width: float,
    right: float,
    width: float = _WIDTH,
    bottom: float = 0.62,
) -> Tuple[Any, Any]:
    """A figure with a header, one axes ``plot_height`` inches tall, room for labels and the x axis."""
    plt = _pyplot()
    fig = plt.figure(figsize=(width, 1.0), facecolor=_SURFACE)
    top = _header(fig, width, title, subtitle)
    height = top + plot_height + bottom
    fig.set_size_inches(width, height)
    _header(fig, width, title, subtitle)
    ax = fig.add_axes([label_width / width, bottom / height, (width - label_width - right) / width,
                       plot_height / height])
    ax.set_facecolor(_SURFACE)
    return fig, ax


def _bars_figure(insight: Any, names: Sequence[str], note: str) -> Tuple[Any, Any]:
    label_width = min(3.0, 0.35 + 0.075 * max(len(name) for name in names))
    title = _clean(_get(insight, "title"))
    summary = _clean(_get(insight, "summary"))
    fig, ax = _figure(plot_height=0.32 * len(names) + 0.15, title=title, subtitle=f"{summary} {note}".strip(),
                      label_width=label_width, right=1.1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(len(names) - 0.5, -0.9)
    return fig, ax


def _ranked_bars(
    insight: Any, names: Sequence[str], values: Sequence[float], texts: Sequence[str], note: str, *, percent: bool
) -> Any:
    if not names:
        return _message(insight, "Nothing to rank in this insight.")
    fig, ax = _bars_figure(insight, names, note)
    ax.barh(range(len(values)), values, height=0.62, color=SERIES[0], linewidth=0, zorder=3)
    high = max(max(values), 1e-9)
    ax.set_xlim(0, high * 1.15)
    for index, (value, text) in enumerate(zip(values, texts)):
        ax.text(value + high * 0.015, index, text, va="center", fontsize=9, color=_MUTED)
    _style(ax, axis="x", percent=percent)
    return fig


def _message(insight: Any, text: str) -> Any:
    plt = _pyplot()
    width = _WIDTH
    fig = plt.figure(figsize=(width, 1.0), facecolor=_SURFACE)
    top = _header(fig, width, _clean(_get(insight, "title")), text)
    fig.set_size_inches(width, top)
    _header(fig, width, _clean(_get(insight, "title")), text)
    _canvas(fig)
    return fig


def _canvas(fig: Any) -> None:
    """An empty, invisible axes: IPython does not display a figure without one, text or not."""
    fig.add_axes([0, 0, 1, 1]).set_axis_off()


def _style(ax: Any, *, axis: str, percent: bool, size: float = 9) -> None:
    """Hairline grid on the value axis only, no frame, muted ticks."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=axis, color=_GRID, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=_MUTED, labelsize=size)
    for label in ax.get_yticklabels() if axis == "x" else ax.get_xticklabels():
        label.set_color(_INK)
    if percent:
        formatter = _formatter(lambda v: _pct(v, 0))
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def _formatter(fn: Any) -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _pos: fn(value))


# --- reading the records -------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _attributions(explanation: Any) -> Dict[str, float]:
    raw = _get(explanation, "attributions") or {}
    if raw:
        return {str(k): float(v) for k, v in dict(raw).items()}
    return {str(f.get("feature")): float(f.get("attribution", 0.0)) for f in _get(explanation, "top_features") or []}


def _is_classification(explanation: Any) -> bool:
    output = _get(explanation, "output") or {}
    return bool(output.get("probabilities"))


def _target(explanation: Any) -> str:
    output = _get(explanation, "output") or {}
    return str(output.get("target_label") or output.get("label") or "the prediction")


def _other_label(explanation: Any) -> str:
    """The class on the other side of a binary threshold."""
    output = _get(explanation, "output") or {}
    target = _target(explanation)
    others = [str(label) for label in (output.get("probabilities") or {}) if str(label) != target]
    return others[0] if len(others) == 1 else f"not {target}"


def _output_value(explanation: Any) -> Optional[float]:
    output = _get(explanation, "output") or {}
    value = output.get("value")
    if value is None:
        probabilities = output.get("probabilities") or {}
        value = probabilities.get(_target(explanation)) if isinstance(probabilities, Mapping) else None
    return None if value is None else float(value)


def _output_name_of(explanation: Any) -> str:
    output = _get(explanation, "output") or {}
    return str(output.get("name") or "prediction") + _unit_suffix(output.get("unit"), brackets=True)


def _model_output_name(model: Any) -> str:
    outputs = _get(model, "outputs") or {}
    labels = list(outputs.get("labels") or [])
    if labels:
        positive = outputs.get("positive_label") or (labels[1] if len(labels) == 2 else labels[0])
        return f"P({positive})"
    if model is not None and str(_get(model, "task") or "").endswith("classification"):
        return "P(positive class)"
    name = outputs.get("name") or ("average prediction" if model is None else "prediction")
    return str(name) + _unit_suffix(outputs.get("unit"), brackets=True)


def _threshold(explanation: Any, model: Any, override: Optional[float]) -> Optional[float]:
    if override is not None:
        return float(override)
    output = _get(explanation, "output") or {}
    if len(output.get("probabilities") or {}) != 2:
        return None
    outputs = _get(model, "outputs") or {}
    return float(outputs.get("threshold", 0.5) if outputs.get("threshold") is not None else 0.5)


def _features(model: Any) -> Dict[str, Mapping[str, Any]]:
    return {str(f.get("name")): f for f in _get(model, "features") or []}


def _feature_unit(model: Any, name: str) -> Optional[str]:
    feature = _features(model).get(name) or {}
    unit = feature.get("unit")
    return str(unit) if unit else None


def _feature_text(name: str, value: Any, model: Any) -> str:
    return f"{name} = {_value_text(value, _feature_unit(model, name))}"


def _value_text(value: Any, unit: Optional[str] = None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = _short_value(value)
    return f"{text} {unit}" if unit and isinstance(value, (int, float)) else text


def _short_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"
    return str(value)


def _change_text(change: Mapping[str, Any], model: Any) -> str:
    name = str(change.get("feature") or change.get("name"))
    unit = _feature_unit(model, name)
    return f"{name} from {_value_text(change.get('from'), unit)} to {_value_text(change.get('to'), unit)}"


def _change_label(change: Mapping[str, Any], model: Any) -> str:
    name = str(change.get("feature") or change.get("name"))
    unit = _feature_unit(model, name)
    return f"{name}\n{_value_text(change.get('from'), unit)} → {_value_text(change.get('to'), unit)}"


def _explanation_title(explanation: Any) -> str:
    output = _get(explanation, "output") or {}
    if _is_classification(explanation):
        return f"Why the model predicts {output.get('label', _target(explanation))}"
    return "What drives this prediction"


def _explanation_subtitle(explanation: Any, classification: bool, target: str) -> str:
    method = str(_get(explanation, "method") or "")
    options = _get(explanation, "options") or {}
    how = {"shap": "KernelSHAP", "shapley": "Shapley sampling", "lime": "LIME"}.get(method, method or "attributions")
    if options.get("exact"):
        how += f", exact over {options.get('background_rows')} reference rows"
    if classification:
        return (f"Each bar moves P({target}) from the average toward this prediction: blue pushes toward {target}, "
                f"pink pushes away. Method: {how}.")
    return f"Each bar moves the prediction from the average to this one: blue pushes it up, pink down. Method: {how}."


def _clean(text: Any) -> str:
    return str(text or "").replace("`", "").strip()


def _join(items: Iterable[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: Any) -> str:
    if value is None:
        return "—"
    value = float(value)
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def _points_tick(value: float) -> str:
    """Axis tick in percentage points: ``−20``, ``0``, ``+5``."""
    points = round(value * 100, 6)
    if points == 0:
        return "0"
    text = f"{abs(points):g}"
    return ("+" if points > 0 else _MINUS) + text


def _out(value: Any, classification: bool) -> str:
    return _pct(value) if classification else _num(value)


def _delta(value: float, classification: bool, decimals: Optional[int] = None) -> str:
    sign = "+" if value >= 0 else _MINUS
    if classification:
        return f"{sign}{abs(value) * 100:.1f} pts"
    if decimals is not None:
        return f"{sign}{abs(value):.{decimals}f}"
    return f"{sign}{_num(abs(value))}"


def _shared_decimals(values: Iterable[float]) -> Optional[int]:
    """Decimals that give the largest value three significant figures, for a column of labels."""
    largest = max((abs(v) for v in values), default=0.0)
    if largest == 0:
        return None
    return max(0, 2 - int(math.floor(math.log10(largest))))


def _unit_suffix(unit: Any, brackets: bool = False) -> str:
    if not unit:
        return ""
    return f" ({unit})" if brackets else f" {unit}"
