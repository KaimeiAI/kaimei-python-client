"""Tables for two answers about a model: its similar cases, and what Kaimei holds for it.

``kaimei.display.rows`` calls these; they take the answer as a plain mapping and give flat rows with the
columns to show first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

#: A similar-cases table's columns, the case first; one column per input (``input:<name>``) follows.
SIMILAR_COLUMNS = ("id", "similarity", "distance", "label", "predicted", "value", "outcome", "time", "differs")
#: The columns that say what is known about a case: a reference row has the first, a logged prediction the rest.
_KNOWN = ("label", "predicted", "value", "outcome", "time")
#: What Kaimei holds, one row per item.
HOLDINGS_COLUMNS = ("key", "label", "held", "count", "unit", "bytes", "detail")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def similar_rows(data: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """One row per case, nearest first: how near it is, what is known about it, and its inputs.

    A reference row has its ``label``; a logged prediction has what was ``predicted``, its ``value``, its
    recorded ``outcome`` and its ``time``. ``differs`` names the inputs the case is furthest from the asked
    one on, the furthest first. Columns that no case has a value for are left out.
    """
    reference = str(data.get("source") or "reference") == "reference"
    known = ("label",) if reference else ("predicted", "value", "outcome", "time")
    table: List[Dict[str, Any]] = []
    for case in data.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        row: Dict[str, Any] = {
            "id": case.get("id"),
            "similarity": case.get("similarity"),
            "distance": case.get("distance"),
        }
        for column in known:
            row[column] = case.get(column)
        row["differs"] = ", ".join(
            str(item.get("feature")) for item in case.get("differs") or [] if isinstance(item, Mapping)
        )
        for name, value in _mapping(case.get("inputs")).items():
            row[f"input:{name}"] = value
        table.append(row)
    return table, [column for column in SIMILAR_COLUMNS if column not in _KNOWN or column in known]


def holdings_rows(data: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """One row per thing Kaimei could hold for the model: whether it does, how much, and what exactly."""
    items = [item for item in data.get("items") or [] if isinstance(item, Mapping)]
    return [{column: item.get(column) for column in HOLDINGS_COLUMNS} for item in items], list(HOLDINGS_COLUMNS)
