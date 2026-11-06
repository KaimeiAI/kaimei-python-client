from __future__ import annotations

import json
import warnings

import httpx
import numpy as np
import pytest

from kaimei import Kaimei, StructureError, estimator_to_spec
from kaimei.structure import (
    MAX_REFERENCE_ROWS,
    cap_reference_rows,
    detect_framework,
    spec_is_executable,
)


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/models" and request.method == "POST":
        body = json.loads(request.content)
        assert body["structure"]["layers"]
        return httpx.Response(
            201,
            json={
                "object": "model",
                "id": "mdl_up",
                "slug": body.get("slug") or "churn-net",
                "name": body["name"],
                "framework": body.get("framework") or "sklearn",
                "task": body.get("task") or "binary_classification",
                "features": body.get("features") or [],
            },
        )
    if path == "/v1/models/mdl_up/structure" and request.method == "PUT":
        raise AssertionError("upload() must not PUT structure after POST already included it")
    if path == "/v1/models/mdl_up/reference-data" and request.method == "PUT":
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"object": "reference_data", "model_id": "mdl_up", "rows": len(body["rows"])},
        )
    return httpx.Response(404, json={"error": {"type": "not_found_error", "message": path}})


@pytest.fixture
def client() -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(_handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_estimator_to_spec_reads_linear_weights():
    class Linear:
        coef_ = np.array([[0.25, -0.5]])
        intercept_ = np.array([0.1])
        classes_ = np.array(["no", "yes"])
        n_features_in_ = 2

    spec = estimator_to_spec(Linear(), feature_names=["income", "score"])
    assert spec["task"] == "binary_classification"
    assert spec["layers"][1]["weight"] == [[0.25, -0.5]]
    assert spec["layers"][1]["bias"] == [0.1]
    assert spec_is_executable(spec)


def test_estimator_to_spec_reads_keras_dense():
    class Dense:
        name = "hidden"
        units = 1
        activation = type("act", (), {"__name__": "sigmoid"})()

        def get_weights(self):
            return [np.array([[0.5], [-1.0]]), np.array([0.1])]

    class Sequential:
        layers = [Dense()]
        input_shape = (None, 2)

    spec = estimator_to_spec(Sequential(), feature_names=["a", "b"])
    assert spec["architecture"] == "mlp"
    assert spec["layers"][1]["weight"] == [[0.5, -1.0]]


def test_upload_sends_structure_and_reference_rows(client: Kaimei):
    class Linear:
        coef_ = np.array([[0.25, -0.5]])
        intercept_ = np.array([0.1])
        classes_ = np.array(["no", "yes"])
        feature_names_in_ = np.array(["income", "score"])

    model = client.models.upload(
        Linear(),
        name="Churn net",
        X=[[72000, 681], [41000, 540]],
        y=["no", "yes"],
    )
    assert model.id == "mdl_up"
    assert model.framework == "sklearn"


def test_unknown_estimator_raises():
    class Unknown:
        n_features_in_ = 2

    with pytest.raises(StructureError) as exc:
        estimator_to_spec(Unknown(), feature_names=["a", "b"])
    assert exc.value.code == "unsupported_estimator"


def _sklearn_problem():
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification

    return make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0, random_state=0)


NAMES = ["a", "b", "c", "d"]


def test_forest_over_the_limit_is_a_summary_not_its_first_tree():
    X, y = _sklearn_problem()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    forest = RandomForestClassifier(31, max_depth=2, random_state=0).fit(X, y)
    wrapped = [
        forest,
        GridSearchCV(RandomForestClassifier(31, max_depth=2, random_state=0), {"max_features": [2]}, cv=2).fit(X, y),
        make_pipeline(StandardScaler(), RandomForestClassifier(31, max_depth=2, random_state=0)).fit(X, y),
    ]
    for estimator in wrapped:
        spec = estimator_to_spec(estimator, feature_names=NAMES)
        assert spec["family"] == "ensemble_summary"
        assert spec["n_estimators"] == 31
        assert not spec_is_executable(spec)
    assert estimator_to_spec(RandomForestClassifier(30, max_depth=2, random_state=0).fit(X, y), feature_names=NAMES)["family"] == "ensemble"


def test_boosting_is_a_summary_not_an_average():
    X, y = _sklearn_problem()
    from sklearn.ensemble import AdaBoostClassifier

    for rounds in (10, 50):
        spec = estimator_to_spec(AdaBoostClassifier(n_estimators=rounds, random_state=0).fit(X, y), feature_names=NAMES)
        assert spec["family"] == "ensemble_summary"


def test_meta_estimators_that_change_the_prediction_are_not_read_as_their_parts():
    X, y = _sklearn_problem()
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import (
        BaggingClassifier,
        IsolationForest,
        StackingClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    logistic = LogisticRegression()
    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    for estimator in (
        VotingClassifier([("l", logistic), ("t", tree)], voting="soft").fit(X, y),
        VotingClassifier([("a", tree), ("b", DecisionTreeClassifier(max_depth=3))], voting="hard").fit(X, y),
        StackingClassifier([("t", tree), ("l", logistic)]).fit(X, y),
        BaggingClassifier(logistic, n_estimators=3, random_state=0).fit(X, y),
        BaggingClassifier(tree, n_estimators=3, max_features=0.5, random_state=0).fit(X, y),
        CalibratedClassifierCV(logistic, cv=2).fit(X, y),
        TransformedTargetRegressor(LinearRegression(), func=np.cbrt, inverse_func=np.power, check_inverse=False).fit(X, y.astype(float)),
        IsolationForest(n_estimators=3, random_state=0).fit(X),
    ):
        with pytest.raises(StructureError) as exc:
            estimator_to_spec(estimator, feature_names=NAMES)
        assert exc.value.code == "unsupported_estimator"


def test_plain_averages_of_trees_stay_exact():
    X, y = _sklearn_problem()
    from sklearn.ensemble import BaggingClassifier, VotingClassifier
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    for estimator in (
        VotingClassifier([("a", tree), ("b", DecisionTreeClassifier(max_depth=3, random_state=0))], voting="soft").fit(X, y),
        BaggingClassifier(tree, n_estimators=3, random_state=0).fit(X, y),
    ):
        assert estimator_to_spec(estimator, feature_names=NAMES)["family"] == "ensemble"


def test_large_averages_of_trees_upload_their_members_importances():
    """Regression: bagging and voting have no ``feature_importances_``, so the summary was empty and
    the API refused it with "importances must have one value per feature"."""
    X, y = _sklearn_problem()
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=31, random_state=0).fit(X, y)
    spec = estimator_to_spec(bagging, feature_names=NAMES)
    assert spec["family"] == "ensemble_summary"
    assert spec["n_estimators"] == 31
    assert len(spec["importances"]) == len(NAMES)
    assert sum(spec["importances"]) == pytest.approx(1.0)


def test_a_vendor_list_of_models_is_not_its_first_model():
    class Linear:
        coef_ = np.array([[1.0, -1.0]])
        intercept_ = np.array([0.0])
        n_features_in_ = 2

    class VendorBlend:
        def __init__(self):
            self.models_ = [Linear(), Linear()]

    with pytest.raises(StructureError) as exc:
        estimator_to_spec(VendorBlend(), feature_names=["a", "b"])
    assert exc.value.code == "unsupported_estimator"


def test_linear_coefficients_under_another_output_are_drawn_not_run():
    X, y = _sklearn_problem()
    from sklearn.linear_model import (
        LinearRegression,
        LogisticRegression,
        PoissonRegressor,
        Ridge,
        SGDClassifier,
    )
    from sklearn.svm import SVC

    positive = np.abs(X[:, 0]) + 1
    three = (X[:, 1] > 0).astype(int) + (X[:, 2] > 0).astype(int)
    for estimator in (
        PoissonRegressor().fit(X, positive),
        SVC(kernel="linear", probability=True, random_state=0).fit(X, y),
        SGDClassifier(loss="log_loss", random_state=0).fit(X, three),
    ):
        spec = estimator_to_spec(estimator, feature_names=NAMES)
        assert spec["approximate"] is True
        assert not spec_is_executable(spec)
    for estimator in (
        LogisticRegression().fit(X, y),
        LogisticRegression().fit(X, three),
        LinearRegression().fit(X, positive),
        Ridge().fit(X, positive),
    ):
        spec = estimator_to_spec(estimator, feature_names=NAMES)
        assert "approximate" not in spec
        assert spec_is_executable(spec)


def test_a_linear_layer_with_nothing_to_compare_stands():
    class Linear:
        coef_ = np.array([[1.0, -1.0]])
        intercept_ = np.array([0.0])
        n_features_in_ = 2

        def predict(self, rows):
            raise RuntimeError("needs a live session")

    assert "approximate" not in estimator_to_spec(Linear(), feature_names=["a", "b"])


def test_scalers_are_folded_as_sklearn_applies_them():
    X, y = _sklearn_problem()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    for scalers in (
        [StandardScaler()],
        [StandardScaler(with_mean=False)],
        [StandardScaler(with_std=False)],
        [StandardScaler(with_mean=False, with_std=False)],
        [StandardScaler(), MinMaxScaler()],
        [MinMaxScaler(), StandardScaler()],
    ):
        pipeline = make_pipeline(*scalers, LogisticRegression()).fit(X, y)
        layer = next(item for item in estimator_to_spec(pipeline, feature_names=NAMES)["layers"] if item["type"] == "standardize")
        folded = (X - np.asarray(layer["center"])) / np.asarray(layer["scale"])
        np.testing.assert_allclose(folded, pipeline[:-1].transform(X))


def test_a_scaler_in_front_of_a_forest_moves_every_tree():
    X, y = _sklearn_problem()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(3, max_depth=2, random_state=0)).fit(X, y)
    spec = estimator_to_spec(pipeline, feature_names=NAMES)
    assert spec["family"] == "ensemble" and len(spec["trees"]) == 3
    bare = estimator_to_spec(pipeline[-1], feature_names=NAMES)
    moved = [node for tree in spec["trees"] for node in tree["nodes"] if node.get("threshold") is not None]
    still = [node for tree in bare["trees"] for node in tree["nodes"] if node.get("threshold") is not None]
    assert moved and all(a["threshold"] != b["threshold"] for a, b in zip(moved, still))


def test_wrapper_around_linear_is_extracted():
    class Linear:
        coef_ = np.array([[0.25, -0.5]])
        intercept_ = np.array([0.1])
        classes_ = np.array(["no", "yes"])
        n_features_in_ = 2

    class VendorModel:
        def __init__(self, inner):
            self.model_ = inner

    spec = estimator_to_spec(VendorModel(Linear()), feature_names=["income", "score"])
    assert spec["layers"][1]["weight"] == [[0.25, -0.5]]
    assert spec_is_executable(spec)


def test_nested_search_wrapper_is_extracted():
    class Linear:
        coef_ = np.array([[1.0, -1.0]])
        intercept_ = np.array([0.0])
        n_features_in_ = 2

    class Tuned:
        def __init__(self, inner):
            self.best_estimator_ = inner

    spec = estimator_to_spec(Tuned(Linear()), feature_names=["a", "b"])
    assert spec["family"] == "network"


def test_detect_framework_walks_core_base_classes():
    class CoreBase:
        pass

    CoreBase.__module__ = "sklearn.base"

    class VendorEstimator(CoreBase):
        pass

    VendorEstimator.__module__ = "vendor.ensemble"
    assert detect_framework(VendorEstimator()) == "sklearn"


def test_cap_reference_rows_warns():
    rows = [{"x": i} for i in range(MAX_REFERENCE_ROWS + 10)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capped, labels = cap_reference_rows(rows, list(range(len(rows))))
    assert len(capped) == MAX_REFERENCE_ROWS
    assert labels is not None and len(labels) == MAX_REFERENCE_ROWS
    assert any("Reference data" in str(item.message) for item in caught)


def test_summary_spec_is_not_executable():
    assert not spec_is_executable({"family": "ensemble_summary", "importances": [0.5, 0.5]})
    assert not spec_is_executable({"family": "network", "approximate": True})


def test_sklearn_logistic_to_spec():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    del sklearn
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    spec = estimator_to_spec(clf, feature_names=["a", "b"])
    assert spec["family"] == "network"
    assert spec["layers"][1]["type"] == "dense"
    assert spec_is_executable(spec)
