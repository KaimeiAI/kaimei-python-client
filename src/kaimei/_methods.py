"""Several explanation methods on one input: which inputs each puts first, and how far they agree.

``client.explanations.compare_methods`` calls these. They are arithmetic on the explanations the API returned,
and say the same thing in the same words as the dashboard's "What the other methods say".
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

#: What each method is called.
METHOD_LABELS = {
    "shap": "KernelSHAP",
    "shapley": "Permutation Shapley",
    "lime": "LIME",
    "gradcam": "Grad-CAM",
    "occlusion": "Occlusion",
}

_ALL = {2: "Both", 3: "All three", 4: "All four", 5: "All five"}
_PLACES = {2: "two", 3: "three"}


def comparable(methods: Sequence[str]) -> List[str]:
    """The methods worth comparing, in the order given: ``auto`` is one of the others under another name."""
    kept: List[str] = []
    for method in methods:
        if method != "auto" and method not in kept:
            kept.append(method)
    return kept


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _list_of(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def exactness(explanation: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether an explanation's numbers are exact or approximate, from what the API stored with it.

    ``exact`` is ``True`` for a method that computed its numbers in full, ``False`` for one that sampled or fitted
    them, and ``None`` when the record does not say; ``detail`` says how in a few words.
    """
    method = explanation.get("method")
    options = explanation.get("options") or {}
    exact = options.get("exact") is True
    if method == "shap":
        coalitions = options.get("n_coalitions", options.get("coalitions"))
        rows = f" over {options['background_rows']:,} reference rows" if _number(options.get("background_rows")) else ""
        if exact:
            every = f"all {coalitions:,} coalitions" if _number(coalitions) else "every coalition"
            return {"exact": True, "detail": f"{every} of the inputs{rows}"}
        sampled = f"{options['n_samples']:,} sampled coalitions" if _number(options.get("n_samples")) else "sampled coalitions"
        return {"exact": False, "detail": f"{sampled}{rows}"}
    if method == "shapley":
        if exact:
            return {"exact": True, "detail": "every way of hiding some of the inputs"}
        orders = options.get("permutations", options.get("n_permutations"))
        detail = f"{orders:,} sampled orders of the inputs" if _number(orders) else "sampled orders of the inputs"
        return {"exact": False, "detail": detail}
    if method == "lime":
        fit = f", R² {options['local_r2']:.2f}" if _number(options.get("local_r2")) else ""
        return {"exact": False, "detail": f"a straight-line fit near this input{fit}"}
    if method == "occlusion":
        return {"exact": True, "detail": "each input hidden once"}
    return {"exact": None, "detail": "the record does not say"}


def _attributions(explanation: Mapping[str, Any]) -> Dict[str, float]:
    given = {name: float(value) for name, value in (explanation.get("attributions") or {}).items() if _number(value)}
    if given:
        return given
    listed = explanation.get("top_features") or []
    return {str(item["feature"]): float(item["attribution"]) for item in listed if _number(item.get("attribution"))}


def _ordered(attributions: Mapping[str, float]) -> List[str]:
    """Inputs in a method's order: largest absolute attribution first, the record's own order between equals."""
    names = list(attributions)
    return sorted(names, key=lambda name: (-abs(attributions[name]), names.index(name)))


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Ranks for a correlation: equal values share the mean of the places they take."""
    order = sorted(range(len(values)), key=lambda index: -values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for place in range(start, end + 1):
            ranks[order[place]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    top = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    spread_a = sum((x - mean_a) ** 2 for x in a)
    spread_b = sum((y - mean_b) ** 2 for y in b)
    if spread_a <= 0 or spread_b <= 0:
        return None
    return float(top / (spread_a * spread_b) ** 0.5)


def rank_agreement(maps: Sequence[Mapping[str, float]]) -> Optional[float]:
    """How far the methods agree on the order of the inputs.

    Spearman's rank correlation of the absolute attributions, over the inputs every method has, averaged over
    the pairs of methods: 1 is the same order, 0 no relation, -1 the reverse. ``None`` with fewer than two
    methods or three shared inputs.
    """
    if len(maps) < 2:
        return None
    shared = [name for name in maps[0] if all(name in other for other in maps)]
    if len(shared) < 3:
        return None
    ranks = [_average_ranks([abs(each[name]) for name in shared]) for each in maps]
    total = 0.0
    pairs = 0
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            r = _pearson(ranks[i], ranks[j])
            if r is None:
                return None
            total += r
            pairs += 1
    return total / pairs if pairs else None


def _says_what(labels: Sequence[str], named: Sequence[str], verb: str = "puts") -> List[str]:
    groups: Dict[str, List[str]] = {}
    for label, name in zip(labels, named):
        groups.setdefault(name, []).append(label)
    plural = "put" if verb == "puts" else "say"
    return [f"{_list_of(who)} {plural if len(who) > 1 else verb} {name}" for name, who in groups.items()]


def opposed_inputs(orders: Sequence[Sequence[str]], maps: Sequence[Mapping[str, float]]) -> List[str]:
    """The inputs two methods push opposite ways, where neither number is small beside the largest."""
    largest = max((abs(value) for each in maps for value in each.values()), default=0.0)
    found: List[str] = []
    for name in orders[0] if orders else []:
        values = [each[name] for each in maps if name in each and abs(each[name]) >= 0.05 * largest]
        if any(value > 0 for value in values) and any(value < 0 for value in values):
            found.append(name)
    return found


def agreement_sentence(labels: Sequence[str], orders: Sequence[Sequence[str]], maps: Sequence[Mapping[str, float]]) -> str:
    """What the methods say together, in a sentence that follows their numbers."""
    n = len(orders)
    if n < 2 or any(not order for order in orders):
        return ""
    firsts = [order[0] for order in orders]
    everyone = _ALL.get(n, f"All {n}")
    if all(name == firsts[0] for name in firsts):
        depth = min(len(order) for order in orders)
        same = 1
        while same < depth and all(order[same] == orders[0][same] for order in orders):
            same += 1
        if same >= depth and depth > 3:
            text = f"{everyone} put {firsts[0]} first and rank every input in the same order."
        elif same >= 2:
            text = f"{everyone} put {firsts[0]} first and agree on the top {_PLACES[min(same, 3)]}."
        elif depth >= 3 and all(sorted(order[:3]) == sorted(orders[0][:3]) for order in orders):
            text = f"{everyone} put {firsts[0]} first and name the same top three, in a different order."
        elif depth >= 2:
            text = f"{everyone} put {firsts[0]} first, then part: {_list_of(_says_what(labels, [order[1] for order in orders]))} second."
        else:
            text = f"{everyone} put {firsts[0]} first."
    else:
        text = f"They part on what matters most: {', '.join(_says_what(labels, firsts, 'says'))}."
    opposed = opposed_inputs(orders, maps)
    if opposed:
        text += f" They disagree on which way {_list_of(opposed[:3])} {'pushes' if len(opposed) == 1 else 'push'}."
    return text


def summarize(explanations: Sequence[Mapping[str, Any]], top: int = 5) -> Dict[str, Any]:
    """The comparison of several explanations of one input, one per method.

    ``agreement`` says how far they agree (a ``sentence``, the ``rank_correlation``, the input each puts
    ``first``, the inputs they push opposite ways); ``rows`` has one row for every input any method ranks in its
    top ``top``, ordered by its mean rank, with each method's attribution and rank; ``costs`` has each method's
    model calls, time and whether it is exact.
    """
    methods = [str(item.get("method")) for item in explanations]
    labels = [METHOD_LABELS.get(method, method) for method in methods]
    maps = [_attributions(item) for item in explanations]
    orders = [_ordered(each) for each in maps]
    places = [{name: index + 1 for index, name in enumerate(order)} for order in orders]
    everything: List[str] = []
    for order in orders:
        for name in order:
            if name not in everything:
                everything.append(name)
    shown: List[str] = []
    for order in orders:
        for name in order[:top]:
            if name not in shown:
                shown.append(name)
    worst = len(everything) + 1

    def mean_rank(name: str) -> float:
        return sum(place.get(name, worst) for place in places) / len(places)

    def mean_size(name: str) -> float:
        return sum(abs(each.get(name, 0.0)) for each in maps) / len(maps)

    first_input = explanations[0].get("input") or {} if explanations else {}
    rows = [
        {
            "feature": name,
            "value": first_input.get(name),
            "attributions": {method: maps[index].get(name) for index, method in enumerate(methods)},
            "ranks": {method: places[index].get(name) for index, method in enumerate(methods)},
        }
        for name in sorted(shown, key=lambda name: (mean_rank(name), -mean_size(name), name))
    ]
    costs = [
        {
            "method": method,
            "label": labels[index],
            "explanation_id": item.get("id"),
            "n_evals": item.get("n_evals"),
            "latency_ms": item.get("latency_ms"),
            **exactness(item),
        }
        for index, (method, item) in enumerate(zip(methods, explanations))
    ]
    return {
        "agreement": {
            "sentence": agreement_sentence(labels, orders, maps),
            "rank_correlation": rank_agreement(maps),
            "first": {method: (orders[index][0] if orders[index] else None) for index, method in enumerate(methods)},
            "same_first": len({order[0] for order in orders if order}) == 1 and all(orders),
            "opposed": opposed_inputs(orders, maps),
            "inputs": len(everything),
        },
        "rows": rows,
        "costs": costs,
    }
