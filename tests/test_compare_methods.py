import json

import httpx
import pytest

from kaimei import Kaimei
from kaimei._methods import agreement_sentence, comparable, exactness, rank_agreement, summarize


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


APPLICANT = {"income": 162738.78, "credit_score": 781, "utilization": 0.4157, "debt_to_income": 0.456, "loan_amount": 50400, "home_ownership": "rent"}

# Three explanations of one input of the demo credit model, as POST /v1/explanations returned them (cut to six inputs).
ANSWERS = {
    "shap": {
        "attributions": {"income": -0.2781, "credit_score": -0.2633, "utilization": -0.0664, "debt_to_income": 0.1062, "loan_amount": 0.1164, "home_ownership": 0.0261},
        "n_evals": 32771,
        "latency_ms": 758.06,
        "options": {"estimator": "kernel", "exact": True, "n_coalitions": 510, "n_samples": 510, "background_rows": 64},
    },
    "shapley": {
        "attributions": {"income": -0.28, "credit_score": -0.2563, "utilization": -0.0672, "debt_to_income": 0.1048, "loan_amount": 0.1197, "home_ownership": 0.026},
        "n_evals": 41026,
        "latency_ms": 906.23,
        "options": {"n_permutations": 4096, "background_rows": 64},
    },
    "lime": {
        "attributions": {"income": -0.1037, "credit_score": -0.1683, "utilization": -0.0456, "debt_to_income": 0.1034, "loan_amount": 0.0496, "home_ownership": 0.0274},
        "n_evals": 514,
        "latency_ms": 16.92,
        "options": {"n_samples": 512, "kernel_width": 2.25, "local_r2": 0.7742},
    },
}


def _explanation(method: str, body: dict) -> dict:
    answer = ANSWERS[method]
    return {
        "object": "explanation",
        "id": f"exp_{method}",
        "model_id": "mdl_1",
        "prediction_id": body.get("prediction_id"),
        "method": method,
        "input": body["input"],
        "output": {"label": "approve", "probability": 0.8788, "target_label": "decline", "value": 0.1212},
        "base_value": 0.4786,
        "top_features": [],
        "counterfactual": None,
        **answer,
        "options": {**answer["options"], **body.get("options", {})},
    }


def _api(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "GET":
            return httpx.Response(200, json={"object": "model", "slug": "credit-risk-v3", "features": [], "explain_methods": ["shap", "shapley", "lime", "auto"]})
        # What "auto" runs is stored under its own name.
        return httpx.Response(201, json=_explanation("shapley" if body["method"] == "auto" else body["method"], body))

    return handler


def test_compare_methods_explains_the_input_once_with_each_method_the_model_lists():
    seen: list = []
    comparison = _client(_api(seen)).explanations.compare_methods("credit-risk-v3", APPLICANT)

    assert [(method, path) for method, path, _ in seen] == [("GET", "/v1/models/credit-risk-v3")] + [("POST", "/v1/explanations")] * 3
    # One call per method, "auto" left out; the same input each time, and no prediction logged for any of them.
    assert [body["method"] for _, _, body in seen[1:]] == ["shap", "shapley", "lime"]
    assert all(body["input"] == APPLICANT and body["options"] == {"log_prediction": False} for _, _, body in seen[1:])
    assert all("prediction_id" not in body for _, _, body in seen[1:])

    assert comparison.object == "method_comparison" and comparison.model == "credit-risk-v3"
    assert comparison.methods == ["shap", "shapley", "lime"]
    assert [explanation.id for explanation in comparison.explanations] == ["exp_shap", "exp_shapley", "exp_lime"]
    assert comparison.explanations[2].attributions.income == -0.1037

    agreement = comparison.agreement
    assert agreement.sentence == "They part on what matters most: KernelSHAP and Permutation Shapley say income, LIME says credit_score."
    assert agreement.first.to_dict() == {"shap": "income", "shapley": "income", "lime": "credit_score"}
    assert agreement.same_first is False and agreement.opposed == [] and agreement.inputs == 6
    assert 0.7 < agreement.rank_correlation < 1

    # A row per input any method ranks in its top five, by mean rank, with each method's own number and rank.
    assert [row.feature for row in comparison.rows] == ["income", "credit_score", "loan_amount", "debt_to_income", "utilization"]
    income = comparison.rows[0]
    assert income.value == 162738.78
    assert income.attributions.to_dict() == {"shap": -0.2781, "shapley": -0.28, "lime": -0.1037}
    assert income.ranks.to_dict() == {"shap": 1, "shapley": 1, "lime": 2}

    assert [(cost.method, cost.label, cost.explanation_id, cost.n_evals, cost.latency_ms, cost.exact) for cost in comparison.costs] == [
        ("shap", "KernelSHAP", "exp_shap", 32771, 758.06, True),
        ("shapley", "Permutation Shapley", "exp_shapley", 41026, 906.23, False),
        ("lime", "LIME", "exp_lime", 514, 16.92, False),
    ]
    assert comparison.costs[0].detail == "all 510 coalitions of the inputs over 64 reference rows"
    assert comparison.costs[1].detail == "4,096 sampled orders of the inputs"
    assert comparison.costs[2].detail == "a straight-line fit near this input, R² 0.77"


def test_compare_methods_takes_the_methods_a_prediction_to_attach_to_and_options_for_every_call():
    seen: list = []
    comparison = _client(_api(seen)).explanations.compare_methods(
        "credit-risk-v3", APPLICANT, ["lime", "shap"], prediction_id="pred_1", options={"target_class": "approve", "seed": 7}, top=2
    )
    # The methods were given, so the model is not read.
    assert [(method, path) for method, path, _ in seen] == [("POST", "/v1/explanations")] * 2
    assert [body["method"] for _, _, body in seen] == ["lime", "shap"]
    assert all(body["prediction_id"] == "pred_1" for _, _, body in seen)
    assert all(body["options"] == {"log_prediction": False, "target_class": "approve", "seed": 7} for _, _, body in seen)
    assert comparison.methods == ["lime", "shap"]
    assert [row.feature for row in comparison.rows] == ["credit_score", "income"]
    assert comparison.agreement.sentence == "They part on what matters most: LIME says credit_score, KernelSHAP says income."


def test_compare_methods_refuses_fewer_than_two_methods():
    seen: list = []
    client = _client(_api(seen))
    with pytest.raises(ValueError, match="at least two"):
        client.explanations.compare_methods("credit-risk-v3", APPLICANT, ["shap"])
    assert seen == []

    def no_methods(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "model", "slug": "detector-v1", "features": [], "explain_methods": []})

    with pytest.raises(ValueError, match="lists no explanation method"):
        _client(no_methods).explanations.compare_methods("detector-v1", APPLICANT)


def test_auto_is_not_a_method_of_its_own():
    assert comparable(["shap", "shapley", "lime", "auto", "shap"]) == ["shap", "shapley", "lime"]


def test_exactness_reads_what_the_api_stored():
    assert exactness({"method": "shap", "options": {"n_samples": 2048}}) == {"exact": False, "detail": "2,048 sampled coalitions"}
    assert exactness({"method": "shapley", "options": {"exact": True, "coalitions": 4}})["exact"] is True
    assert exactness({"method": "occlusion", "options": {}}) == {"exact": True, "detail": "each input hidden once"}
    assert exactness({"method": "provided"})["exact"] is None


def test_the_sentence_follows_the_orders():
    def say(orders, labels=("A", "B", "C")):
        maps = [{name: 1 - index * 0.1 for index, name in enumerate(order)} for order in orders]
        return agreement_sentence(labels[: len(orders)], orders, maps)

    assert say([list("xyzw")] * 3) == "All three put x first and rank every input in the same order."
    assert say([list("xyzwv"), list("xyzvw"), list("xyzwv")]) == "All three put x first and agree on the top three."
    assert say([list("xyzw"), list("xywz"), list("xyzw")]) == "All three put x first and agree on the top two."
    assert say([list("xyzw"), list("xzyw")]) == "Both put x first and name the same top three, in a different order."
    assert say([list("xyzw"), list("xwyz"), list("xywz")]) == "All three put x first, then part: A and C put y and B puts w second."
    assert say([list("xy"), list("yx")]) == "They part on what matters most: A says x, B says y."
    opposed = agreement_sentence(["A", "B"], [["x", "y"], ["x", "y"]], [{"x": 0.5, "y": 0.2}, {"x": 0.4, "y": -0.2}])
    assert opposed == "Both put x first and agree on the top two. They disagree on which way y pushes."
    quiet = agreement_sentence(["A", "B"], [["x", "y"], ["x", "y"]], [{"x": 0.5, "y": 0.01}, {"x": 0.4, "y": -0.01}])
    assert quiet == "Both put x first and agree on the top two."


def test_rank_agreement_is_spearman_averaged_over_the_pairs():
    same = {"a": 3, "b": 2, "c": 1}
    assert rank_agreement([same, {"a": -30, "b": 20, "c": -1}]) == pytest.approx(1)
    assert rank_agreement([same, {"a": 1, "b": 2, "c": 3}]) == pytest.approx(-1)
    assert rank_agreement([same, same, {"a": 1, "b": 2, "c": 3}]) == pytest.approx(-1 / 3)
    assert rank_agreement([same]) is None
    assert rank_agreement([{"a": 1, "b": 2}, {"a": 1, "b": 2}]) is None
    assert rank_agreement([{"a": 1, "b": 1, "c": 1}, same]) is None


def test_summarize_leaves_a_method_without_a_number_for_an_input_blank():
    summary = summarize(
        [
            {"method": "shap", "id": "exp_1", "input": {"x": 1, "y": 2}, "attributions": {"x": 0.5, "y": -0.2}, "n_evals": 10, "latency_ms": 1.0, "options": {}},
            {"method": "lime", "id": "exp_2", "input": {"x": 1, "y": 2}, "attributions": {"x": 0.3}, "n_evals": 5, "latency_ms": 0.5, "options": {}},
        ]
    )
    assert summary["rows"][1] == {"feature": "y", "value": 2, "attributions": {"shap": -0.2, "lime": None}, "ranks": {"shap": 2, "lime": None}}
    assert summary["agreement"]["rank_correlation"] is None
    assert summary["agreement"]["same_first"] is True
