"""Reading a precision-recall curve: the threshold that meets a target.

``client.models.pr_curve(model)`` gives the curve the API measured on a model's outcomes. These helpers only read
it; the model's own threshold changes only when you register a new one (``outputs.threshold``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["threshold_for"]


def threshold_for(
    curve: Optional[Mapping[str, Any]],
    *,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> Optional[Mapping[str, Any]]:
    """The point on ``curve`` that best meets one target, or ``None`` when no threshold does.

    With ``min_precision``, the point with the highest recall among those whose precision is at least that (the
    most positives found while that share of the positive answers is right). With ``min_recall``, the point with
    the highest precision among those that find at least that share of positives. Ties go to the higher
    threshold, which says positive less often. The point carries ``threshold``, the counts and the measures.

    >>> curve = client.models.pr_curve("fraud-v2")
    >>> threshold_for(curve, min_recall=0.9)["threshold"]
    0.31
    """
    if (min_precision is None) == (min_recall is None):
        raise ValueError("Give exactly one of min_precision and min_recall.")
    points = list((curve or {}).get("points") or [])
    if min_precision is not None:
        held, gain = "precision", "recall"
        target = float(min_precision)
    else:
        held, gain = "recall", "precision"
        target = float(min_recall)  # type: ignore[arg-type]
    best: Optional[Mapping[str, Any]] = None
    for point in points:
        kept, got = point.get(held), point.get(gain)
        if kept is None or got is None or kept < target:
            continue
        if best is None or got > best[gain] or (got == best[gain] and point["threshold"] > best["threshold"]):
            best = point
    return best
