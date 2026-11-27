"""How records look in a notebook, and how to get at their numbers.

A record the client returns (an explanation, an insight, a model, a page of predictions, ...) is
shown in a Jupyter cell the way its dashboard card reads: the Kaimei mark, the dashboard's colours,
the same numbers formatted the same way, with hover details and collapsible sections. Nothing in
the card needs a script, so it renders in JupyterLab, classic Jupyter, VS Code and Colab, and it
follows the host's dark theme. A monitoring history, a population, a comparison of two models,
similar cases, what Kaimei holds for a model and a comparison of explanation methods each lead with
the sentence that says what was found, then the numbers behind it.

Three ways to get the plain data instead::

    explanation.to_dict()      # the JSON the API returned, as a dict
    explanation.to_json()      # the same, as a string
    explanation.to_frame()     # its main table as a pandas DataFrame (features × attribution, share, ...)

and a switch for people who never want the cards::

    import kaimei
    kaimei.set_display("raw")      # cells show the JSON (a collapsible tree in JupyterLab)
    kaimei.set_display("rich")     # back to the cards (the default)

    with kaimei.display.using("raw"):
        explanation                # raw for this block only

``KAIMEI_DISPLAY=raw`` in the environment sets the default. ``kaimei.display.html(record)`` returns
a card's HTML for a report of your own.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = ["MODES", "set_mode", "mode", "using", "html", "frame", "rows", "flatten", "mimebundle"]

#: ``rich`` shows the themed card; ``raw`` shows the record's JSON.
MODES = ("rich", "raw")

_mode: Optional[str] = None


def set_mode(value: str) -> None:
    """Choose how records display in notebooks: ``"rich"`` (cards) or ``"raw"`` (JSON)."""
    if value not in MODES:
        raise ValueError(f"display mode must be one of {MODES}, not {value!r}.")
    global _mode
    _mode = value


def mode() -> str:
    """The current display mode (``KAIMEI_DISPLAY`` sets the default, else ``rich``)."""
    if _mode is not None:
        return _mode
    configured = os.environ.get("KAIMEI_DISPLAY", "").strip().lower()
    return configured if configured in MODES else "rich"


@contextmanager
def using(value: str) -> Iterator[None]:
    """``with kaimei.display.using("raw"):`` — a display mode for one block of code."""
    global _mode
    previous = _mode
    set_mode(value)
    try:
        yield
    finally:
        _mode = previous


def html(record: Any) -> str:
    """The notebook card of ``record`` as a self-contained HTML string (styles included)."""
    from ._html import render

    return render(_data(record))


def mimebundle(record: Any) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """What IPython shows for ``record`` in the current mode, as ``(data, metadata)``."""
    data = _data(record)
    if not data.get("object"):
        return None  # a nested value such as ``explanation.output``: its repr is the data itself
    if mode() == "raw":
        text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        root = str(data.get("object"))
        return {"application/json": data, "text/plain": text}, {"application/json": {"expanded": True, "root": root}}
    try:
        card = html(data)
    except Exception:
        # A record in a shape no card expects still shows: a cell never fails over how its result is drawn.
        return {"text/plain": repr(record)}, {}
    return {"text/html": card, "text/plain": repr(record)}, {}


# --- tables ---------------------------------------------------------------------------------------


def frame(record: Any) -> Any:
    """``rows(record)`` as a pandas DataFrame (``pip install pandas``)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("to_frame() needs pandas: pip install pandas") from exc
    table, columns = rows(record)
    if not table:
        return pd.DataFrame(columns=columns)
    data = pd.DataFrame(table)
    ordered = [c for c in columns if c in data.columns] + [c for c in data.columns if c not in columns]
    return data[ordered]


def rows(record: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The main table of a record, as a list of flat dicts and the column order to show first.

    - an explanation: one row per feature, ranked by |attribution| — the API's ``top_features``
      rows exactly, then the remaining ``attributions`` with ``share`` and ``rank`` computed the
      same way (image explanations list their regions instead);
    - a decision record: one row per input's effect, strongest first, with the sentence that states it
      (a record that is not explained yet lists its inputs as recorded);
    - a reason table: one row per input, with how it is treated and its approved wording;
    - a model's evidence: one row per requirement, with what is held and what is missing for it;
    - a model documentation report: one row per finding of its analyses;
    - an insight: the payload's table for its type (importance shares, drift PSI per feature,
      interaction pairs, fragile features, learned rules, one row per grid point of each behaviour
      curve, histogram bins);
    - a model: its features;
    - a performance report: one row per window;
    - a monitoring history: one row per window, with each input's PSI as a ``drift.<input>`` column;
    - a population: one row per logged prediction, with a column per field holding the row's group;
    - similar cases: one row per case, nearest first, with what is known about it and an ``input:<name>``
      column per input; what Kaimei holds for a model: one row per item;
    - a comparison: one row per row both models answered, with each one's value and decision, the gap
      between them and an ``input:<name>`` column per input;
    - a model report: one row per section, with its status and its summary;
    - a page: one row per item, nested fields flattened to dotted columns (``output.label``,
      ``attributions.Age``);
    - anything else: the record as one flattened row.
    """
    data = _data(record)
    kind = str(data.get("object") or "")
    if kind == "explanation":
        return _explanation_rows(data)
    if kind == "decision_record":
        factors = [dict(f) for f in data.get("factors") or [] if isinstance(f, Mapping)]
        if factors:
            columns = ["rank", "feature", "label", "display", "attribution", "share", "direction", "stderr", "sentence"]
            return factors, columns
        inputs = [dict(i) for i in data.get("inputs") or [] if isinstance(i, Mapping)]
        return inputs, ["name", "label", "value", "exact", "unit", "type"]
    if kind == "evidence":
        table = []
        for framework in data.get("frameworks") or []:
            for requirement in (framework or {}).get("requirements") or []:
                parts = [part for part in requirement.get("evidence") or [] if isinstance(part, Mapping)]
                table.append(
                    {
                        "rule": framework.get("name"),
                        "requirement": requirement.get("title"),
                        "reference": requirement.get("reference"),
                        "status": requirement.get("status"),
                        "held": [part.get("label") for part in parts if part.get("status") == "held"],
                        "missing": [part.get("label") for part in parts if part.get("status") in ("missing", "stale")],
                        "stays_with_you": requirement.get("yours"),
                    }
                )
        return table, ["rule", "requirement", "reference", "status", "held", "missing", "stays_with_you"]
    if kind == "model_report":
        findings = [dict(f) for f in data.get("findings") or [] if isinstance(f, Mapping)]
        return findings, ["name", "summary", "severity", "created_at", "status"]
    if kind == "reason_table":
        entries = data.get("entries") or {}
        never = data.get("never_state") or {}
        table = [
            {"input": str(name), "treatment": "reason", "reason": (entry or {}).get("reason"), "code": (entry or {}).get("code")}
            for name, entry in entries.items()
        ]
        table += [{"input": str(name), "treatment": "never state", "reason": why or None, "code": None} for name, why in never.items()]
        table += [
            {"input": str(name), "treatment": "undecided", "reason": None, "code": None}
            for name in (data.get("coverage") or {}).get("unmapped") or []
        ]
        return table, ["input", "treatment", "reason", "code"]
    if kind == "insight":
        return _insight_rows(data)
    if kind == "model":
        features = [dict(f) for f in data.get("features") or [] if isinstance(f, Mapping)]
        return features, ["name", "type", "unit", "min", "max", "categories", "baseline", "description"]
    if kind == "retention":
        table = []
        for item in data.get("models") or []:
            if not isinstance(item, Mapping):
                continue
            model, held, due = item.get("model") or {}, item.get("held") or {}, item.get("due") or {}
            table.append(
                {
                    "model": model.get("slug"),
                    "name": model.get("name"),
                    "keep_days": item.get("keep_days"),
                    "source": item.get("source"),
                    "decisions": item.get("decisions"),
                    "oldest_at": item.get("oldest_at"),
                    "due_decisions": due.get("decisions"),
                    "due_explanations": due.get("explanations"),
                    "held_model": held.get("model"),
                    "held_decisions": held.get("decisions"),
                }
            )
        return table, ["model", "name", "keep_days", "source", "decisions", "oldest_at", "due_decisions"]
    if kind == "list":
        listed = [item for item in data.get("data") or [] if isinstance(item, Mapping)]
        if listed and listed[0].get("object") == "audit_event":
            # One row per entry, with who and what first: the columns a reader of a log looks for.
            table = [
                {
                    "at": e.get("at"),
                    "who": (e.get("actor") or {}).get("name"),
                    "actor_type": (e.get("actor") or {}).get("type"),
                    "action": e.get("action"),
                    "summary": e.get("summary"),
                    "target_type": (e.get("target") or {}).get("type"),
                    "target_id": (e.get("target") or {}).get("id"),
                    "target": (e.get("target") or {}).get("name"),
                    "model_id": e.get("model_id"),
                    "detail": e.get("detail"),
                    "request_id": e.get("request_id"),
                    "id": e.get("id"),
                }
                for e in listed
            ]
            return table, ["at", "who", "action", "summary", "target_type", "target_id"]
        items = [flatten(item) for item in listed]
        return items, ["id", "object"]
    if kind == "performance":
        return [dict(w) for w in data.get("series") or [] if isinstance(w, Mapping)], list(_SERIES_COLUMNS)
    if kind == "monitoring":
        # One row per window; each input's PSI is a column of its own (``drift.temperature``).
        windows = [flatten(w) for w in data.get("series") or [] if isinstance(w, Mapping)]
        return windows, list(_MONITORING_COLUMNS)
    if kind == "population":
        return _population_rows(data), list(_POPULATION_COLUMNS)
    if kind in ("similar_cases", "model_holdings"):
        from ._cases import holdings_rows, similar_rows

        return similar_rows(data) if kind == "similar_cases" else holdings_rows(data)
    if kind == "comparison":
        return _comparison_rows(data), list(_COMPARISON_COLUMNS)
    if kind == "model_case_study":
        # One row per section, in the report's order: what it says, and whether it is on file.
        sections = [s for s in data.get("sections") or [] if isinstance(s, Mapping)]
        return [{column: s.get(column) for column in _REPORT_COLUMNS} for s in sections], list(_REPORT_COLUMNS)
    return [flatten(data)], []


#: A model report's columns: one row per section.
_REPORT_COLUMNS = ("key", "title", "status", "applies", "as_of", "summary")


#: A comparison's columns, the two models' answers first; one column per input (``input:<name>``) follows.
_COMPARISON_COLUMNS = ("id", "value", "other_value", "gap", "decision", "other_decision", "agree", "actual")


def _comparison_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per row both models answered: the first model's value and decision, the other's, and the inputs."""

    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    columns = mapping(data.get("rows"))
    classes = list(mapping(data.get("output")).get("classes") or [])
    inputs = mapping(columns.get("inputs"))

    def pair(name: str) -> Tuple[List[Any], List[Any]]:
        both = columns.get(name)
        if isinstance(both, list) and len(both) == 2 and all(isinstance(side, list) for side in both):
            return both[0], both[1]
        return [], []

    def at(column: Any, index: int) -> Any:
        return column[index] if isinstance(column, list) and index < len(column) else None

    def named(position: Any) -> Any:
        # A classifier's rows refer to its classes by position; a regression's truth is its number.
        if classes and isinstance(position, int) and not isinstance(position, bool):
            return classes[position] if 0 <= position < len(classes) else None
        return position

    def number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    values, decisions = pair("values"), pair("decisions")
    table: List[Dict[str, Any]] = []
    for index, identifier in enumerate(columns.get("id") or []):
        first, second = at(values[0], index), at(values[1], index)
        chose, other_chose = named(at(decisions[0], index)), named(at(decisions[1], index))
        row: Dict[str, Any] = {
            "id": identifier,
            "value": first,
            "other_value": second,
            "gap": abs(second - first) if number(first) and number(second) else None,
            "decision": chose,
            "other_decision": other_chose,
            # Whether the two decide the row the same way; a regression has no decision to agree on.
            "agree": chose == other_chose if chose is not None and other_chose is not None else None,
            "actual": named(at(columns.get("actual"), index)),
        }
        for name, column in inputs.items():
            row[f"input:{name}"] = at(column, index)
        table.append(row)
    return table


#: A population's columns, the prediction first; one column per field (by its key) follows.
_POPULATION_COLUMNS = ("id", "time", "value", "predicted", "actual")


def _population_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per logged prediction: classes by name, and under each field's key the group the row is in."""

    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    columns = mapping(data.get("rows"))
    classes = list(mapping(data.get("output")).get("classes") or [])
    placed = mapping(columns.get("groups"))
    fields = [
        (str(f.get("key")), [g.get("label") for g in f.get("groups") or [] if isinstance(g, Mapping)])
        for f in data.get("fields") or []
        if isinstance(f, Mapping) and f.get("key")
    ]

    def at(name: str, index: int) -> Any:
        column = columns.get(name)
        return column[index] if isinstance(column, list) and index < len(column) else None

    def named(position: Any) -> Any:
        # A classifier's rows refer to its classes by position; a regression's outcome is its number.
        if classes and isinstance(position, int) and not isinstance(position, bool):
            return classes[position] if 0 <= position < len(classes) else None
        return position

    table: List[Dict[str, Any]] = []
    for index, identifier in enumerate(columns.get("id") or []):
        row: Dict[str, Any] = {
            "id": identifier,
            "time": at("time", index),
            "value": at("value", index),
            "predicted": named(at("predicted", index)),
            "actual": named(at("actual", index)),
        }
        for key, labels in fields:
            group = placed.get(key)
            position = group[index] if isinstance(group, list) and index < len(group) else None
            row[key] = labels[position] if isinstance(position, int) and 0 <= position < len(labels) else None
        table.append(row)
    return table


#: A monitoring history's columns, the window first; the PSI of each input follows.
_MONITORING_COLUMNS = ("start", "end", "predictions", "output.mean", "rows", "status", "beyond_chance", "overall_psi", "chance", "top_feature")

#: A performance series' columns, the window first.
_SERIES_COLUMNS = ("start", "end", "n", "accuracy", "auc", "precision", "recall", "f1", "macro_f1", "mae", "rmse", "r2")


def flatten(record: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Nested dicts become dotted keys; tables (lists of dicts) and arrays (nested lists) are dropped."""
    out: Dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, name + "."))
        elif isinstance(value, (list, tuple)):
            if value and isinstance(value[0], (Mapping, list, tuple)):
                continue
            out[name] = list(value)
        else:
            out[name] = value
    return out


def _explanation_rows(data: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    options = data.get("options") or {}
    regions = options.get("regions") if isinstance(options, Mapping) else None
    if isinstance(regions, list) and regions:
        return [dict(r) for r in regions if isinstance(r, Mapping)], ["id", "name", "attribution", "area"]
    table: List[Dict[str, Any]] = [dict(f) for f in data.get("top_features") or [] if isinstance(f, Mapping)]
    seen = {str(f.get("feature")) for f in table}
    attributions = {str(k): float(v) for k, v in (data.get("attributions") or {}).items()}
    inputs = data.get("input") or {}
    total = sum(abs(v) for v in attributions.values()) or 1.0
    rest = sorted(((k, v) for k, v in attributions.items() if k not in seen), key=lambda kv: -abs(kv[1]))
    for rank, (name, value) in enumerate(rest, start=len(table) + 1):
        table.append(
            {
                "feature": name,
                "attribution": round(value, 6),
                "value": inputs.get(name) if isinstance(inputs, Mapping) else None,
                "share": round(abs(value) / total, 4),
                "direction": "positive" if value >= 0 else "negative",
                "rank": rank,
            }
        )
    return table, ["rank", "feature", "value", "attribution", "share", "direction", "stderr"]


def _insight_rows(data: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    kind = str(data.get("type") or "")
    payload = data.get("payload") or {}
    if not isinstance(payload, Mapping):
        return [], []
    if kind == "interactions":
        table = []
        for pair in _dicts(payload.get("pairs")):
            features = list(pair.get("features") or [])
            row = {"feature_a": features[0] if features else None, "feature_b": features[1] if len(features) > 1 else None}
            row.update({k: v for k, v in pair.items() if k != "features"})
            table.append(row)
        return table, ["feature_a", "feature_b", "strength"]
    if kind == "partial_dependence":
        table = []
        for curve in _dicts(payload.get("features")):
            direction = curve.get("monotonicity")
            if isinstance(direction, Mapping):
                direction = direction.get("direction")
            for grid, pd_value in zip(curve.get("grid") or [], curve.get("pd") or []):
                table.append({"feature": curve.get("feature"), "type": curve.get("type"), "grid": grid, "pd": pd_value,
                              "monotonicity": direction})
        return table, ["feature", "type", "grid", "pd", "monotonicity"]
    if kind in ("behavior", "behaviour"):
        histogram = payload.get("histogram") or {}
        counts = list(histogram.get("counts") or []) if isinstance(histogram, Mapping) else []
        edges = list(histogram.get("edges") or []) if isinstance(histogram, Mapping) else []
        table = [{"bin_start": lo, "bin_end": hi, "count": c} for lo, hi, c in zip(edges, edges[1:], counts)]
        return table, ["bin_start", "bin_end", "count"]
    if kind == "robustness":
        return _dicts(payload.get("fragile_features") or payload.get("features")), ["feature", "sensitivity"]
    if kind == "learned_rules":
        return _dicts(payload.get("rules")), ["statement"]
    if kind == "performance":
        return _dicts(payload.get("series")), list(_SERIES_COLUMNS)
    for key in ("features", "pairs", "rules", "items"):
        if isinstance(payload.get(key), list):
            return _dicts(payload[key]), ["feature", "share", "psi", "status"]
    return [flatten(payload)], []


def _dicts(value: Any) -> List[Dict[str, Any]]:
    return [dict(item) for item in (value or []) if isinstance(item, Mapping)] if isinstance(value, Sequence) else []


def _data(record: Any) -> Dict[str, Any]:
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"Expected a Kaimei record or a dict, not {type(record).__name__}.")
