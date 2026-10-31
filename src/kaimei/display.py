"""How records look in a notebook, and how to get at their numbers.

A record the client returns (an explanation, an insight, a model, a page of predictions, ...) is
shown in a Jupyter cell the way its dashboard card reads: the Kaimei mark, the dashboard's colours,
the same numbers formatted the same way, with hover details and collapsible sections. Nothing in
the card needs a script, so it renders in JupyterLab, classic Jupyter, VS Code and Colab, and it
follows the host's dark theme.

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
    return {"text/html": html(data), "text/plain": repr(record)}, {}


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
    - an insight: the payload's table for its type (importance shares, drift PSI per feature,
      interaction pairs, fragile features, learned rules, one row per grid point of each behaviour
      curve, histogram bins);
    - a model: its features;
    - a page: one row per item, nested fields flattened to dotted columns (``output.label``,
      ``attributions.Age``);
    - anything else: the record as one flattened row.
    """
    data = _data(record)
    kind = str(data.get("object") or "")
    if kind == "explanation":
        return _explanation_rows(data)
    if kind == "insight":
        return _insight_rows(data)
    if kind == "model":
        features = [dict(f) for f in data.get("features") or [] if isinstance(f, Mapping)]
        return features, ["name", "type", "unit", "min", "max", "categories", "baseline", "description"]
    if kind == "list":
        items = [flatten(item) for item in data.get("data") or [] if isinstance(item, Mapping)]
        return items, ["id", "object"]
    return [flatten(data)], []


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
