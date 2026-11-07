from __future__ import annotations

import json
import warnings

import httpx
import numpy as np
import pytest

from kaimei import Kaimei, StructureError, estimator_to_spec
from kaimei.ensembles import _evaluate, verify
from kaimei.resources.models import _plainly_named
from kaimei.structure import (
    MAX_REFERENCE_ROWS,
    _names_it_was_fitted_on,
    cap_reference_rows,
    detect_framework,
    extract_structure,
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


def _matches_its_library(estimator, names=NAMES):
    """The spec runs, and running it reproduces the estimator on rows around its own splits."""
    spec = estimator_to_spec(estimator, feature_names=names)
    assert spec_is_executable(spec), spec.get("summary")
    assert verify(spec, estimator, names) is not False
    return spec


def test_a_forest_over_the_inline_limit_runs_every_tree_not_its_first():
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
        spec = _matches_its_library(estimator)
        assert spec["family"] == "ensemble"
        assert len(spec["forest"]["roots"]) == 31
    small = RandomForestClassifier(30, max_depth=2, random_state=0).fit(X, y)
    assert len(estimator_to_spec(small, feature_names=NAMES)["trees"]) == 30


def test_sklearns_default_forest_runs():
    X, y = _sklearn_problem()
    from sklearn.ensemble import RandomForestClassifier

    spec = _matches_its_library(RandomForestClassifier(random_state=0).fit(X, y))
    assert len(spec["forest"]["roots"]) == 100


def test_a_large_forest_travels_as_binary_in_full_precision():
    X, y = _sklearn_problem()
    from sklearn.ensemble import RandomForestClassifier

    from kaimei import extract_structure

    forest = RandomForestClassifier(120, random_state=0).fit(X, y)
    extraction = extract_structure(forest, feature_names=NAMES, externalize_above=100)
    assert extraction.externalized
    assert extraction.tensors["forest.threshold"].dtype == np.float64
    assert extraction.tensors["forest.value"].dtype == np.float64
    assert extraction.tensors["forest.left"].dtype == np.float32
    assert extraction.spec["forest"]["threshold"]["$tensor"] == "forest.threshold"
    summary = extraction.summary_spec()
    assert all("$shape" in summary["forest"][key] for key in ("feature", "threshold", "value"))


@pytest.mark.parametrize(
    "build",
    [
        lambda s: s["ensemble"].AdaBoostClassifier(n_estimators=10, random_state=0),
        lambda s: s["ensemble"].AdaBoostClassifier(n_estimators=50, random_state=0),
        lambda s: s["ensemble"].GradientBoostingClassifier(n_estimators=10, random_state=0),
        lambda s: s["ensemble"].HistGradientBoostingClassifier(max_iter=10, random_state=0),
        lambda s: s["ensemble"].BaggingClassifier(s["tree"].DecisionTreeClassifier(max_depth=2), n_estimators=40, random_state=0),
        lambda s: s["ensemble"].BaggingClassifier(s["linear"].LogisticRegression(), n_estimators=3, max_features=0.5, random_state=0),
        lambda s: s["ensemble"].VotingClassifier([("l", s["linear"].LogisticRegression()), ("t", s["tree"].DecisionTreeClassifier(max_depth=2))], voting="soft"),
        lambda s: s["ensemble"].VotingClassifier([("l", s["linear"].LogisticRegression()), ("t", s["tree"].DecisionTreeClassifier(max_depth=2))], voting="soft", weights=[2, 1]),
        lambda s: s["ensemble"].StackingClassifier([("t", s["tree"].DecisionTreeClassifier(max_depth=2)), ("l", s["linear"].LogisticRegression())], cv=2),
        lambda s: s["calibration"].CalibratedClassifierCV(s["linear"].LogisticRegression(), cv=2),
        lambda s: s["calibration"].CalibratedClassifierCV(s["linear"].LogisticRegression(), cv=2, method="isotonic"),
        lambda s: s["calibration"].CalibratedClassifierCV(s["ensemble"].RandomForestClassifier(5, max_depth=2, random_state=0), cv=2),
        lambda s: s["multiclass"].OneVsRestClassifier(s["linear"].LogisticRegression()),
    ],
)
def test_ensembles_and_meta_estimators_run_and_match(build):
    X, y = _sklearn_problem()
    from sklearn import calibration, ensemble, linear_model, multiclass, svm, tree

    modules = {"calibration": calibration, "ensemble": ensemble, "linear": linear_model, "multiclass": multiclass, "svm": svm, "tree": tree}
    estimator = build(modules).fit(X, y)
    spec = _matches_its_library(estimator)
    assert verify(spec, estimator, NAMES) is True


def test_a_hard_vote_runs_and_picks_the_same_labels():
    X, y = _sklearn_problem()
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    estimator = VotingClassifier([("l", LogisticRegression()), ("t", DecisionTreeClassifier(max_depth=2))], voting="hard").fit(X, y)
    spec = _matches_its_library(estimator)
    assert spec["combine"]["op"] == "vote"


def test_regressors_run_and_match():
    X, y = _sklearn_problem()
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
        StackingRegressor,
        VotingRegressor,
    )
    from sklearn.linear_model import LinearRegression, PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    target = X[:, 0] * 3 + X[:, 1]
    positive = np.abs(target) + 1
    tree = DecisionTreeRegressor(max_depth=2, random_state=0)
    for estimator, fitted_on in (
        (RandomForestRegressor(40, max_depth=2, random_state=0), target),
        (GradientBoostingRegressor(n_estimators=8, random_state=0), target),
        (AdaBoostRegressor(n_estimators=6, random_state=0), target),
        (VotingRegressor([("l", LinearRegression()), ("t", tree)], weights=[1, 2]), target),
        (StackingRegressor([("l", LinearRegression()), ("t", tree)], cv=2), target),
        (TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler()), target),
        (TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1), positive),
        (PoissonRegressor(), positive),
    ):
        estimator.fit(X, fitted_on)
        assert verify(_matches_its_library(estimator), estimator, NAMES) is True


@pytest.mark.parametrize("library", ["xgboost", "lightgbm", "catboost"])
def test_tree_libraries_run_and_match(library):
    module = pytest.importorskip(library)
    X, y = _sklearn_problem()
    three = (X[:, 1] > 0).astype(int) + (X[:, 2] > 0).astype(int)
    target = X[:, 0] * 3 + X[:, 1]
    options = {"verbose": False, "allow_writing_files": False} if library == "catboost" else {"verbose": -1} if library == "lightgbm" else {}
    size = {"iterations": 8} if library == "catboost" else {"n_estimators": 8}
    names = {"xgboost": ("XGBClassifier", "XGBRegressor"), "lightgbm": ("LGBMClassifier", "LGBMRegressor"), "catboost": ("CatBoostClassifier", "CatBoostRegressor")}[library]
    classifier = getattr(module, names[0])
    regressor = getattr(module, names[1])
    multi = {"loss_function": "MultiClass"} if library == "catboost" else {}
    for estimator in (classifier(**size, **options).fit(X, y), classifier(**size, **options, **multi).fit(X, three), regressor(**size, **options).fit(X, target)):
        spec = _matches_its_library(estimator)
        assert spec["family"] == "ensemble"
        assert verify(spec, estimator, NAMES) is True


def test_what_cannot_be_read_exactly_is_refused():
    X, y = _sklearn_problem()
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVC

    for estimator in (
        IsolationForest(n_estimators=3, random_state=0).fit(X),
        TransformedTargetRegressor(LinearRegression(), func=lambda v: v / 10, inverse_func=lambda v: v * 10).fit(X, y.astype(float)),
        SVC(kernel="rbf").fit(X, y),
    ):
        with pytest.raises(StructureError) as exc:
            estimator_to_spec(estimator, feature_names=NAMES)
        assert exc.value.code == "unsupported_estimator"


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


def test_output_that_does_not_match_the_reading_is_drawn_not_run():
    class Odd:
        coef_ = np.array([0.5, -0.25])
        intercept_ = 0.1
        n_features_in_ = 2

        def predict(self, rows):
            return np.tanh(rows @ self.coef_ + self.intercept_)

    spec = estimator_to_spec(Odd(), feature_names=["a", "b"], task="regression")
    assert spec["approximate"] is True
    assert not spec_is_executable(spec)


def test_linear_models_with_other_outputs_have_exact_forms():
    X, _ = _sklearn_problem()
    from sklearn.linear_model import PoissonRegressor, SGDClassifier

    positive = np.abs(X[:, 0]) + 1
    three = (X[:, 1] > 0).astype(int) + (X[:, 2] > 0).astype(int)
    for estimator in (PoissonRegressor().fit(X, positive), SGDClassifier(loss="log_loss", random_state=0).fit(X, three)):
        assert verify(_matches_its_library(estimator), estimator, NAMES) is True


def test_scores_that_are_not_probabilities_are_marked():
    X, y = _sklearn_problem()
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    assert estimator_to_spec(LinearSVC().fit(X, y), feature_names=NAMES)["uncalibrated"] is True
    assert "uncalibrated" not in estimator_to_spec(LogisticRegression().fit(X, y), feature_names=NAMES)


def test_a_linear_layer_with_no_way_of_being_asked_stands():
    class Linear:
        coef_ = np.array([[1.0, -1.0]])
        intercept_ = np.array([0.0])
        n_features_in_ = 2

    assert "approximate" not in estimator_to_spec(Linear(), feature_names=["a", "b"])


def test_a_model_that_refuses_to_answer_isnt_shown_to_match():
    class Linear:
        coef_ = np.array([[1.0, -1.0]])
        intercept_ = np.array([0.0])
        n_features_in_ = 2

        def predict(self, rows):
            raise RuntimeError("needs a live session")

    assert estimator_to_spec(Linear(), feature_names=["a", "b"]).get("approximate")


@pytest.mark.parametrize(
    "library,build,fit_on_a_table",
    [
        ("catboost", "CatBoostClassifier", True),
        ("catboost", "CatBoostClassifier", False),
        ("lightgbm", "LGBMClassifier", True),
        ("lightgbm", "LGBMClassifier", False),
        ("xgboost", "XGBClassifier", True),
        ("xgboost", "XGBClassifier", False),
    ],
)
def test_a_booster_keeps_its_own_names_or_takes_the_tables(library, build, fit_on_a_table):
    module = pytest.importorskip(library)
    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    values = rng.normal(size=(120, 3))
    labels = (values[:, 0] > 0).astype(int)
    table = pandas.DataFrame(values, columns=["age", "income", "tenure"])
    options = {
        "catboost": {"iterations": 3, "depth": 2, "verbose": False, "allow_writing_files": False},
        "lightgbm": {"n_estimators": 3, "verbose": -1, "min_child_samples": 5},
        "xgboost": {"n_estimators": 3, "max_depth": 2},
    }[library]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = getattr(module, build)(**options).fit(table if fit_on_a_table else values, labels)
    assert _names_it_was_fitted_on(fitted) == (["age", "income", "tenure"] if fit_on_a_table else None)
    assert _plainly_named(table, fitted) is not fit_on_a_table
    if fit_on_a_table:
        assert estimator_to_spec(fitted)["feature_names"] == ["age", "income", "tenure"]


def test_a_one_vs_rest_model_decides_by_its_margins_where_its_probabilities_tie():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(12)
    values = rng.normal(size=(300, 2))
    labels = np.digitize(values[:, 0], [-0.5, 0.5])
    names = ["a", "b"]
    for model in (
        SGDClassifier(loss="log_loss", random_state=0),
        make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", random_state=0)),
        OneVsRestClassifier(LogisticRegression(max_iter=500)),
        OneVsRestClassifier(SGDClassifier(loss="log_loss", random_state=0)),
    ):
        fitted = model.fit(values, labels)
        spec = estimator_to_spec(fitted, feature_names=names)
        assert spec_is_executable(spec), spec.get("summary")
        far = rng.normal(size=(200, 2)) * 1000.0
        ours = np.asarray(_evaluate(spec, {name: far[:, at] for at, name in enumerate(names)}, far.shape[0]))
        assert fitted.predict(far).tolist() == ours.argmax(axis=1).tolist()
        theirs = fitted.predict_proba(far)
        # OneVsRestClassifier has no probability for a row every class rejects.
        known = np.isfinite(theirs).all(axis=1)
        assert ours[known] == pytest.approx(theirs[known], abs=1e-12)


def test_lightgbm_keeps_the_names_of_a_table_with_spaces_and_bad_rows_leave_no_model_behind():
    lightgbm = pytest.importorskip("lightgbm")
    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    values = rng.normal(size=(120, 2))
    labels = (values[:, 0] > 0).astype(int)
    table = pandas.DataFrame(values, columns=["alpha one", "beta"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = lightgbm.LGBMClassifier(n_estimators=3, verbose=-1, min_child_samples=5).fit(table, labels)
    assert _plainly_named(table, fitted)
    sent = []

    def respond(request):
        sent.append((request.method, request.url.path))
        if request.url.path.endswith("/reference-data"):
            return httpx.Response(200, json={"object": "reference_data", "rejected": 0})
        return httpx.Response(201 if request.method == "POST" else 200, json={"object": "model", "id": "m"})

    client = Kaimei(api_key="k", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(respond))
    client.models.upload(fitted, name="spaces", X=table, y=labels)
    assert ("POST", "/v1/models") in sent
    sent.clear()
    with pytest.raises(StructureError, match="no column"):
        client.models.upload(fitted, name="wrong", feature_names=["alpha one", "beta"], X=table[["beta"]])
    assert sent == []


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

def test_a_missing_value_travels_as_null():
    from kaimei._http import encode_json

    sent = json.loads(encode_json({"rows": [{"a": float("nan"), "b": 1.0}], "values": np.array([1.0, np.nan])}))
    assert sent == {"rows": [{"a": None, "b": 1.0}], "values": [1.0, None]}
    with pytest.raises(ValueError):
        encode_json({"a": float("inf")})


@pytest.mark.parametrize("trees", [5, 40])
def test_a_model_that_routes_missing_values_says_so(trees):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from kaimei.structure import features_from_spec

    generator = np.random.default_rng(0)
    rows = generator.normal(size=(200, 3))
    labels = (rows[:, 0] > 0).astype(int)
    holes = rows.copy()
    holes[generator.random(rows.shape) < 0.2] = np.nan
    names = ["a", "b", "c"]
    spec = estimator_to_spec(RandomForestClassifier(trees, max_depth=4, random_state=0).fit(holes, labels), feature_names=names)
    assert spec["missing_values"] is True
    assert all(feature["allow_missing"] for feature in features_from_spec(spec))
    plain = estimator_to_spec(LogisticRegression().fit(rows, labels), feature_names=names)
    assert "missing_values" not in plain
    assert not any(feature.get("allow_missing") for feature in features_from_spec(plain))


def test_empty_classes_mean_no_classes():
    class Regressor:
        classes_: list = []

    from kaimei.ensembles import _labels

    assert _labels(Regressor()) is None


def test_output_in_another_shape_does_not_match():
    class TwoOutputs:
        coef_ = np.array([0.5, -0.25])
        intercept_ = 0.1
        n_features_in_ = 2

        def predict(self, rows):
            first = rows @ self.coef_ + self.intercept_
            return np.stack([first, first], axis=1)

    spec = estimator_to_spec(TwoOutputs(), feature_names=["a", "b"], task="regression")
    assert spec["approximate"] is True
    assert not spec_is_executable(spec)


def test_several_outputs_are_drawn_not_run():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    rows = np.random.default_rng(0).normal(size=(40, 2))
    target = np.stack([rows.sum(axis=1), rows[:, 0]], axis=1)
    spec = estimator_to_spec(LinearRegression().fit(rows, target), feature_names=["a", "b"])
    assert spec["approximate"] is True
    assert "several outputs" in spec["summary"]


@pytest.mark.parametrize("trees", [5, 40])
def test_models_fitted_on_missing_values_can_be_sent_as_json(trees):
    """sklearn gives them infinite thresholds. JSON has no infinity, and the check couldn't score rows near them."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    generator = np.random.default_rng(0)
    rows = generator.normal(size=(200, 3))
    labels = (rows[:, 0] > 0).astype(int)
    rows[generator.random(rows.shape) < 0.2] = np.nan
    estimator = RandomForestClassifier(trees, max_depth=4, random_state=0).fit(rows, labels)
    spec = estimator_to_spec(estimator, feature_names=["a", "b", "c"])
    assert not spec.get("approximate"), spec.get("summary")
    assert verify(spec, estimator, ["a", "b", "c"]) is True
    assert json.dumps(extract_structure(estimator, feature_names=["a", "b", "c"]).spec, allow_nan=False)
