"""What's claimed about an upload holds, and what's private stays private."""

from __future__ import annotations

import json
import warnings

import httpx
import numpy as np
import pytest

from kaimei import Kaimei, StructureError, estimator_to_spec
from kaimei._http import encode_json
from kaimei.ensembles import _evaluate, verify
from kaimei.structure import (
    extract_structure,
    features_from_spec,
    rows_from_inputs,
    spec_is_executable,
)

NAMES = ["a", "b", "c", "d"]


def _problem():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _with_gaps(X):
    gaps = X.copy()
    gaps[np.random.default_rng(1).random(gaps.shape) < 0.2] = np.nan
    return gaps


@pytest.mark.parametrize("seed", range(20))
def test_an_imputer_means_no_gaps_are_offered(seed):
    X, y = _problem()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    for final in (
        DecisionTreeClassifier(max_depth=4, random_state=seed),
        RandomForestClassifier(n_estimators=5, max_depth=3, random_state=seed),
    ):
        pipeline = make_pipeline(SimpleImputer(strategy="mean"), final).fit(_with_gaps(X), y)
        spec = estimator_to_spec(pipeline, feature_names=NAMES)
        assert not spec.get("approximate")
        assert spec.get("missing_values") is False
        fills = pipeline.steps[0][1].statistics_
        assert [spec["baselines"][name] for name in NAMES] == pytest.approx(list(fills))
        assert [item["baseline"] for item in features_from_spec(spec)] == pytest.approx(list(fills))
        # Complete rows, the only ones it will be sent, still match.
        rows = X[:64]
        ours = np.asarray(_evaluate(spec, {n: rows[:, i : i + 1] for i, n in enumerate(NAMES)}, 64))
        assert ours[:, 1] == pytest.approx(pipeline.predict_proba(rows)[:, 1])


def test_an_imputer_in_a_member_means_no_gaps_either():
    X, y = _problem()
    from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    vote = VotingClassifier(
        [
            ("imputed", make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=2, random_state=0))),
            ("gaps", HistGradientBoostingClassifier(max_iter=5, random_state=0)),
        ],
        voting="soft",
    ).fit(_with_gaps(X), y)
    assert not estimator_to_spec(vote, feature_names=NAMES).get("missing_values")


@pytest.mark.parametrize("marker", [-999.0, 0, 0.0])
def test_an_imputer_that_replaces_real_numbers_isnt_read_as_nothing(marker):
    X, y = _problem()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    rows = X.copy()
    rows[::5, 0] = marker
    pipeline = make_pipeline(SimpleImputer(missing_values=marker), Ridge()).fit(rows, X[:, 1])
    with pytest.raises(StructureError, match="replaces the value"):
        estimator_to_spec(pipeline, feature_names=NAMES)


def test_an_imputer_after_a_scaler_fills_in_the_inputs_units():
    X, y = _problem()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), SimpleImputer(strategy="median"), LogisticRegression()).fit(_with_gaps(X), y)
    spec = estimator_to_spec(pipeline, feature_names=NAMES)
    row = np.full((1, 4), np.nan)
    filled = np.array([[spec["baselines"][name] for name in NAMES]])
    assert pipeline.predict_proba(row) == pytest.approx(pipeline.predict_proba(filled))


def _table_only(model, change=lambda values: values):
    class TableOnly:
        model_ = model

        def predict(self, table):
            if not hasattr(table, "columns"):
                raise TypeError("A table, please.")
            return change(self.model_.predict(table.to_numpy()))

    return TableOnly()


def test_a_model_that_only_takes_a_table_is_compared_as_one():
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    X, y = _problem()
    forest = RandomForestRegressor(5, max_depth=3, random_state=0).fit(X, y.astype(float))
    assert spec_is_executable(estimator_to_spec(_table_only(forest), feature_names=NAMES))
    spec = estimator_to_spec(_table_only(forest, lambda values: values * 2 + 7), feature_names=NAMES)
    assert spec.get("approximate") and "drawn, not run" in spec["summary"]


def _decides_by_its_probabilities(model, names, rows):
    """Whether the library has a probability for every row and picks the class the first highest one is for.

    Before 1.8, scikit-learn gave no probabilities for some rows far from the data, and broke a one-vs-rest tie for
    the last class. A reading can't match either, so those versions draw these models.
    """
    from kaimei.ensembles import far_rows

    rows = np.vstack([rows, *far_rows(names)])
    probabilities = model.predict_proba(rows)
    if np.isnan(probabilities).any():
        return False
    picked = [str(label) for label in model.predict(rows)]
    return picked == [str(model.classes_[at]) for at in probabilities.argmax(axis=1)]


def test_a_vote_of_one_vs_rest_models_keeps_the_ties_its_library_sees():
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_wine
    from sklearn.ensemble import BaggingClassifier, VotingClassifier
    from sklearn.linear_model import SGDClassifier

    X, y = load_wine(return_X_y=True)
    names = [f"f{at}" for at in range(X.shape[1])]
    sgd = SGDClassifier(loss="log_loss", random_state=0)
    for model in (
        VotingClassifier([("a", sgd), ("b", SGDClassifier(loss="log_loss", random_state=1))], voting="soft"),
        BaggingClassifier(sgd, n_estimators=3, random_state=0),
    ):
        fitted = model.fit(X, y)
        spec = extract_structure(fitted, feature_names=names, check_rows=X, externalize_above=None).spec
        if not _decides_by_its_probabilities(fitted, names, X):
            assert spec.get("approximate")
            continue
        assert spec_is_executable(spec), spec.get("summary")
        ours = np.asarray(_evaluate(spec, {name: X[:, at] for at, name in enumerate(names)}, X.shape[0]))
        assert ours.argmax(axis=1).tolist() == fitted.predict(X).tolist()


def test_a_subclass_or_someones_own_class_inside_a_model_is_drawn():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import BaggingRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    class Capped(LinearRegression):
        def predict(self, X):
            return np.minimum(super().predict(X), 1.0)

    class Own:
        def __init__(self, model):
            self.model_ = model

        def fit(self, X, y):
            self.model_.fit(X, y)
            return self

        def predict(self, X):
            return np.minimum(self.model_.predict(X), 1.0)

        def get_params(self, deep=True):
            return {"model": self.model_}

        def set_params(self, **params):
            return self

    X, y = _problem()
    target = X @ np.array([100.0, 50.0, 0.0, 0.0]) + 500.0
    for model in (
        VotingRegressor([("capped", Capped()), ("ridge", Ridge())]),
        BaggingRegressor(Capped(), n_estimators=3, random_state=0),
    ):
        spec = estimator_to_spec(model.fit(X, target), feature_names=NAMES)
        assert spec.get("approximate") and "drawn, not run" in spec["summary"], type(model).__name__
    vote = VotingRegressor([("own", Own(LinearRegression())), ("ridge", Ridge())])
    vote.named_estimators_ = None
    vote.estimators_ = [Own(LinearRegression()).fit(X, target), Ridge().fit(X, target)]
    assert estimator_to_spec(vote, feature_names=NAMES).get("approximate")


def test_a_hard_vote_or_one_vs_rest_of_svcs_with_probabilities_is_drawn():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import VotingClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = _problem()
    three = np.digitize(X[:, 0], [-0.5, 0.5])
    member = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True, random_state=0))
    hard = VotingClassifier([("a", member), ("b", member)], voting="hard").fit(X, y)
    assert estimator_to_spec(hard, feature_names=NAMES).get("approximate")
    wrapped = OneVsRestClassifier(SVC(kernel="linear", probability=True, random_state=0)).fit(X, three)
    assert estimator_to_spec(wrapped, feature_names=NAMES).get("approximate")


def test_an_imputer_behind_a_scaler_fills_in_the_inputs_units():
    pytest.importorskip("sklearn")
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X, y = _problem()
    shifted = X * 10.0 + 100.0
    inner = make_pipeline(SimpleImputer(strategy="constant", fill_value=1.0), DecisionTreeClassifier(max_depth=3))
    model = make_pipeline(StandardScaler(), inner).fit(shifted, y)
    spec = estimator_to_spec(model, feature_names=NAMES)
    row = np.array([[np.nan, 100.0, 100.0, 100.0]])
    filled = row.copy()
    filled[0, 0] = spec["baselines"]["a"]
    assert model.predict_proba(row) == pytest.approx(model.predict_proba(filled))


@pytest.mark.parametrize("name", ["logistic", "sgd", "scaled logistic", "linear svc", "ridge", "perceptron", "lda", "svc"])
def test_one_vs_rest_over_anything_with_a_margin_runs_and_decides_as_it_does(name):
    pytest.importorskip("sklearn")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC, LinearSVC

    member = {
        "logistic": LogisticRegression(max_iter=500),
        "sgd": SGDClassifier(loss="log_loss", random_state=0),
        "scaled logistic": make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=500)),
        "linear svc": LinearSVC(random_state=0),
        "ridge": RidgeClassifier(),
        "perceptron": Perceptron(random_state=0),
        "lda": LinearDiscriminantAnalysis(),
        "svc": SVC(kernel="linear"),
    }[name]
    X, _ = _problem()
    y = np.digitize(X[:, 0] + X[:, 1], [-0.5, 0.5])
    model = OneVsRestClassifier(member).fit(X, y)
    spec = estimator_to_spec(model, feature_names=NAMES)
    assert spec_is_executable(spec), spec.get("summary")
    rows = np.vstack([X[:50], np.random.default_rng(0).normal(size=(200, 4)) * 1000.0])
    ours = np.asarray(_evaluate(spec, {column: rows[:, at] for at, column in enumerate(NAMES)}, rows.shape[0]))
    assert ours.argmax(axis=1).tolist() == model.predict(rows).tolist()
    if not hasattr(model, "predict_proba"):
        assert spec.get("uncalibrated")


def test_one_vs_rest_trees_decide_as_the_library_does_and_share_nothing_when_all_reject():
    pytest.importorskip("sklearn")
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, _ = _problem()
    y = np.digitize(X[:, 0] + X[:, 1], [-0.5, 0.5])
    model = OneVsRestClassifier(DecisionTreeClassifier(random_state=0)).fit(X, y)
    spec = estimator_to_spec(model, feature_names=NAMES)
    rows = np.random.default_rng(1).normal(size=(2000, 4))
    if not _decides_by_its_probabilities(model, NAMES, rows):
        assert spec.get("approximate")
        return
    theirs = model.predict_proba(rows)
    assert spec_is_executable(spec), spec.get("summary")
    ours = np.asarray(_evaluate(spec, {column: rows[:, at] for at, column in enumerate(NAMES)}, rows.shape[0]))
    assert ours.argmax(axis=1).tolist() == model.predict(rows).tolist()
    assert ours == pytest.approx(theirs, abs=1e-12)


def test_a_scaler_in_front_of_a_stack_reaches_what_it_passes_through():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    X, _ = _problem()
    shifted = X * np.array([1.0, 100.0, 0.01, 10.0]) + np.array([0.0, 1000.0, 0.0, -50.0])
    target = X[:, 0] * 3.0 + X[:, 1] ** 2
    for final in (LinearRegression(), DecisionTreeRegressor(max_depth=4, random_state=0)):
        stack = StackingRegressor(
            [("a", Ridge()), ("b", RandomForestRegressor(5, max_depth=3, random_state=0))],
            final_estimator=final,
            passthrough=True,
            cv=3,
        )
        model = make_pipeline(StandardScaler(), stack).fit(shifted, target)
        spec = estimator_to_spec(model, feature_names=NAMES)
        assert spec_is_executable(spec), spec.get("summary")
        ours = _evaluate(spec, {column: shifted[:, at] for at, column in enumerate(NAMES)}, shifted.shape[0])
        assert np.asarray(ours).ravel() == pytest.approx(model.predict(shifted), rel=1e-9, abs=1e-9)


def test_xgboosts_missing_marker_behind_a_scaler_is_drawn():
    xgboost = pytest.importorskip("xgboost")
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _problem()
    model = make_pipeline(StandardScaler(), xgboost.XGBClassifier(n_estimators=3, missing=-1.0)).fit(X, y)
    spec = estimator_to_spec(model, feature_names=NAMES)
    assert spec.get("approximate") and "marker" in spec["summary"]


def test_a_member_thats_only_drawn_makes_its_stack_only_drawn():
    xgboost = pytest.importorskip("xgboost")
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _problem()
    marked = make_pipeline(StandardScaler(), xgboost.XGBRegressor(n_estimators=3, missing=0.0))
    stack = StackingRegressor([("a", Ridge()), ("b", marked)], final_estimator=RidgeCV(), cv=3)
    assert estimator_to_spec(stack.fit(X, y.astype(float)), feature_names=NAMES).get("approximate")


def test_a_classifier_with_several_outputs_of_different_sizes_is_drawn():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    X, y = _problem()
    both = np.column_stack([y, np.digitize(X[:, 2], [-0.5, 0.5])])
    spec = estimator_to_spec(RandomForestClassifier(3, max_depth=2, random_state=0).fit(X, both), feature_names=NAMES)
    assert spec.get("approximate")


@pytest.mark.parametrize("alpha", [1.0, 0.0])
def test_a_log_link_overflows_as_its_library_does(alpha):
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import PoissonRegressor

    X, y = load_diabetes(return_X_y=True)
    names = [f"f{at}" for at in range(X.shape[1])]
    spec = estimator_to_spec(PoissonRegressor(alpha=alpha, max_iter=500).fit(X, y), feature_names=names)
    assert spec_is_executable(spec), spec.get("summary")


@pytest.mark.parametrize("build", ["bag", "hard", "soft"])
def test_parts_that_dont_fit_together_are_drawn_rather_than_raising(build):
    pytest.importorskip("sklearn")
    from sklearn.ensemble import BaggingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    X, _ = _problem()
    y = np.digitize(X[:, 0], [-1.0, 0.0, 1.0])
    svc = SVC(kernel="linear", probability=build == "soft", random_state=0)
    model = {
        "bag": lambda: BaggingClassifier(svc, n_estimators=3, random_state=0),
        "hard": lambda: VotingClassifier([("a", svc), ("b", LogisticRegression())], voting="hard"),
        "soft": lambda: VotingClassifier([("a", svc), ("b", LogisticRegression())], voting="soft"),
    }[build]()
    assert estimator_to_spec(model.fit(X, y), feature_names=NAMES).get("approximate")


def test_a_pipeline_of_steps_inside_a_pipeline_is_read():
    pytest.importorskip("sklearn")
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _problem()
    shifted = X * 10.0 + 100.0
    prep = make_pipeline(SimpleImputer(), StandardScaler())
    model = Pipeline([("prep", prep), ("model", LogisticRegression())]).fit(shifted, y)
    spec = estimator_to_spec(model, feature_names=NAMES)
    assert spec_is_executable(spec), spec.get("summary")
    ours = _evaluate(spec, {column: shifted[:, at] for at, column in enumerate(NAMES)}, shifted.shape[0])
    assert np.asarray(ours) == pytest.approx(model.predict_proba(shifted), abs=1e-9)


def test_a_scaler_in_front_of_a_scaled_pipeline_gets_a_name_of_its_own():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _problem()
    model = make_pipeline(StandardScaler(), make_pipeline(StandardScaler(), LogisticRegression())).fit(X[:, :1], y)
    layers = estimator_to_spec(model, feature_names=["a"])["layers"]
    ids = [layer["id"] for layer in layers]
    assert len(ids) == len(set(ids)) and {"standardize", "standardize_2"} <= set(ids)


@pytest.mark.parametrize("scaler", ["minmax", "robust", "standard"])
def test_a_target_scaled_by_any_of_the_scalers_runs(scaler):
    pytest.importorskip("sklearn")
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    X, _ = _problem()
    transformer = {"minmax": MinMaxScaler((-1, 1)), "robust": RobustScaler(), "standard": StandardScaler()}[scaler]
    model = TransformedTargetRegressor(LinearRegression(), transformer=transformer).fit(X, X[:, 0] * 30.0 + 7.0)
    spec = estimator_to_spec(model, feature_names=NAMES)
    assert spec_is_executable(spec), spec.get("summary")
    ours = _evaluate(spec, {column: X[:, at] for at, column in enumerate(NAMES)}, X.shape[0])
    assert np.asarray(ours).ravel() == pytest.approx(model.predict(X), rel=1e-9)


def test_feature_names_that_move_a_name_are_refused_even_if_others_are_new():
    pytest.importorskip("sklearn")
    pandas = pytest.importorskip("pandas")
    from sklearn.linear_model import LogisticRegression

    X, y = _problem()
    model = LogisticRegression().fit(pandas.DataFrame(X, columns=NAMES), y)
    with pytest.raises(StructureError, match="different order"):
        estimator_to_spec(model, feature_names=["d", "c", "b", "x"])
    assert estimator_to_spec(model, feature_names=["a", "b", "c", "x"])["feature_names"] == ["a", "b", "c", "x"]


@pytest.mark.parametrize("name", ["linear", "standardize", "softmax", "input", "output"])
def test_a_feature_can_share_its_name_with_a_step_of_the_reading(name):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _problem()
    names = [name, "b", "c", "d"]
    spec = estimator_to_spec(make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y), feature_names=names)
    assert spec_is_executable(spec), spec.get("summary")
    ids = {str(layer["id"]) for layer in spec["layers"]}
    assert not ids & set(names)


def test_a_torch_model_whose_layers_share_names_with_its_features_keeps_its_wiring():
    torch = pytest.importorskip("torch")

    class Named(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 1)

        def forward(self, x):
            return self.b(x + torch.relu(self.a(x)))

    spec = estimator_to_spec(Named().eval(), feature_names=NAMES)
    assert spec_is_executable(spec), spec.get("summary")
    ids = [str(layer["id"]) for layer in spec["layers"]]
    assert not set(ids) & set(NAMES)
    for layer in spec["layers"]:
        for source in layer.get("inputs") or []:
            assert str(source).split("__")[0] in ids


def test_layers_that_cant_be_replayed_are_drawn_and_can_be_refused_with_a_reason():
    torch = pytest.importorskip("torch")

    class Gated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(4, 1)
            self.last = torch.nn.Linear(4, 1)

        def forward(self, x):
            return self.last(x * torch.sigmoid(self.gate(x)))

    # The gate broadcasts one number over four: Kaimei cannot replay that, so the step is drawn and
    # the model is drawn, not run, with the inputs it really reads.
    spec = estimator_to_spec(Gated().eval(), feature_names=NAMES)
    assert spec["approximate"] is True
    (mul,) = [layer for layer in spec["layers"] if layer["type"] == "opaque"]
    assert mul["op"] == "mul" and mul["reason"] == "multiply with broadcasting" and mul["shape"] == [4]
    sigmoid = next(layer["id"] for layer in spec["layers"] if layer.get("activation") == "sigmoid")
    assert set(mul["inputs"]) == {"input", sigmoid}
    assert spec["layers"][-1]["id"] == "last" and spec["layers"][-1].get("inputs", [mul["id"]]) == [mul["id"]]
    with pytest.raises(StructureError, match="multiply with broadcasting"):
        estimator_to_spec(Gated().eval(), feature_names=NAMES, on_unsupported="error")


def _rows_on_every_split(spec, width):
    """Each threshold and the numbers either side of it, on rows otherwise from a plain grid."""
    forest = spec.get("forest")
    if forest is not None:
        pairs = zip(np.asarray(forest["feature"]).tolist(), np.asarray(forest["threshold"]).tolist())
    else:
        nodes = spec["nodes"] if "nodes" in spec else [node for tree in spec["trees"] for node in tree["nodes"]]
        pairs = [(node["feature"], node["threshold"]) for node in nodes if not node.get("leaf")]
    rows = []
    for feature, threshold in pairs:
        if feature < 0 or abs(threshold) > 1e300:
            continue
        for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
            row = np.zeros(width)
            row[int(feature)] = value
            rows.append(row)
    return np.array(rows)


def _scaled_boosters():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier

    built = {
        "tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
        "forest": lambda: RandomForestClassifier(9, max_depth=3, random_state=0),
        "boosting": lambda: GradientBoostingClassifier(n_estimators=5, random_state=0),
        "hist boosting": lambda: HistGradientBoostingClassifier(max_iter=5, min_samples_leaf=5, random_state=0),
    }
    for library, name, options in (
        ("xgboost", "XGBClassifier", {"n_estimators": 5, "max_depth": 3}),
        ("lightgbm", "LGBMClassifier", {"n_estimators": 5, "verbose": -1, "min_child_samples": 5}),
        ("catboost", "CatBoostClassifier", {"n_estimators": 5, "depth": 3, "verbose": 0, "allow_writing_files": False}),
    ):
        try:
            found = getattr(__import__(library), name)
        except ImportError:
            continue
        built[library] = lambda found=found, options=options: found(**options)
    return built


@pytest.mark.parametrize("name", ["tree", "forest", "boosting", "hist boosting", "xgboost", "lightgbm", "catboost"])
def test_a_scaler_folded_into_trees_is_exact_on_the_splits_themselves(name):
    build = _scaled_boosters().get(name)
    if build is None:
        pytest.skip(f"{name} isn't installed")
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    rng = np.random.default_rng(1)
    # Wide scales and far-off centres are where multiplying a threshold back out rounds the most.
    X = rng.normal(size=(300, 4)) * np.array([1.0, 1e4, 1e-3, 100.0]) + np.array([0.0, 1e6, 0.0, -50.0])
    y = (X[:, 0] + (X[:, 1] - 1e6) / 1e4 + rng.normal(size=300) * 0.3 > 0).astype(int)
    for scalers in ([StandardScaler()], [RobustScaler()], [MinMaxScaler()], [StandardScaler(), MinMaxScaler((-1, 1))]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = make_pipeline(*scalers, build()).fit(X, y)
        spec = estimator_to_spec(pipeline, feature_names=NAMES)
        assert spec_is_executable(spec), (name, spec.get("summary"))
        rows = _rows_on_every_split(spec, 4)
        ours = _evaluate(spec, {column: rows[:, at] for at, column in enumerate(NAMES)}, rows.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = pipeline.predict_proba(rows)
        wrong = np.abs(np.asarray(ours).reshape(theirs.shape[0], -1)[:, -1] - theirs[:, -1]) > 1e-5
        assert not wrong.any(), (name, [type(step).__name__ for step in scalers], rows[wrong][:3].tolist())


def test_a_threshold_past_any_number_a_scaler_can_reach_stays_finite():
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import StandardScaler

    from kaimei.ensembles import moved_thresholds, through_scalers

    step = StandardScaler().fit(np.array([[-1.0], [1.0]]))
    scaled = through_scalers([step], 1)
    for strict in (False, True):
        moved = moved_thresholds(
            [0, 0, 0], [1e308, -1e308, 0.0], step.mean_, step.scale_, scaled, single=True, strict=strict
        )
        assert np.isfinite(moved).all()


def test_a_multiclass_svc_isnt_read_as_a_linear_model():
    pytest.importorskip("sklearn")
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    for shift in (0.0, 100.0):
        spec = estimator_to_spec(SVC(kernel="linear").fit(X + shift, y), feature_names=NAMES)
        assert not spec_is_executable(spec)
        assert "drawn, not run" in spec["summary"]


def test_names_in_another_order_are_refused_and_new_names_are_not():
    pandas = pytest.importorskip("pandas")
    X, y = _problem()
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(pandas.DataFrame(X, columns=NAMES), y)
    with pytest.raises(StructureError, match="different order"):
        estimator_to_spec(model, feature_names=list(reversed(NAMES)))
    assert estimator_to_spec(model, feature_names=["p", "q", "r", "s"])["feature_names"] == ["p", "q", "r", "s"]
    assert estimator_to_spec(model, feature_names=NAMES)["feature_names"] == NAMES


def test_a_library_with_real_gap_support_still_says_so():
    X, y = _problem()
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=10, random_state=0).fit(_with_gaps(X), y)
    assert estimator_to_spec(model, feature_names=NAMES).get("missing_values") is True


class _Tiny:
    """Outputs around 1e-7."""

    n_features_in_ = 1

    def predict(self, rows):
        return np.where(rows[:, 0] <= 0.0, 1e-7, 3e-7)


def _tiny_tree(left, right):
    return {
        "family": "tree",
        "task": "regression",
        "feature_names": ["a"],
        "root": "r",
        "nodes": [
            {"id": "r", "feature": 0, "threshold": 0.0, "left": "x", "right": "y"},
            {"id": "x", "leaf": True, "output": [left]},
            {"id": "y", "leaf": True, "output": [right]},
        ],
    }


def test_the_check_scales_with_the_size_of_the_outputs():
    assert verify(_tiny_tree(1e-7, 3e-7), _Tiny(), ["a"]) is True
    assert verify(_tiny_tree(0.0, 0.0), _Tiny(), ["a"]) is False
    assert verify(_tiny_tree(1e-7, 3.1e-7), _Tiny(), ["a"]) is False


def _every_number(value):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in {"stats"}:
                yield from _every_number(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _every_number(item)


def _private_models():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        AdaBoostClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    return {
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
        "small forest": RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0),
        "large forest": RandomForestClassifier(n_estimators=60, max_depth=3, random_state=0),
        "boosted": GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0),
        "adaboost": AdaBoostClassifier(n_estimators=4, random_state=0),
        "calibrated": CalibratedClassifierCV(DecisionTreeClassifier(max_depth=2, random_state=0), method="isotonic", cv=2),
        "scaled linear": make_pipeline(StandardScaler(), LogisticRegression()),
    }


@pytest.mark.parametrize(
    "name", ["tree", "small forest", "large forest", "boosted", "adaboost", "calibrated", "scaled linear"]
)
def test_a_private_upload_sends_nothing_the_model_learned(name):
    X, y = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _private_models()[name].fit(X, y)
    full = extract_structure(model, feature_names=NAMES, externalize_above=None).spec
    summary = extract_structure(model, feature_names=NAMES, weights="summary").spec
    assert not spec_is_executable(summary)
    assert summary["weights"] == "summary"
    learned = {round(number, 9) for number in _every_number(_numbers_of(full)) if abs(number) not in (0.0, 1.0)}
    sent = {round(number, 9) for number in _every_number(_numbers_of(summary)) if abs(number) not in (0.0, 1.0)}
    assert not (learned & sent), sorted(learned & sent)[:5]
    text = json.dumps(summary)
    for key in ("nodes", "calibrators", '"threshold": [', '"coef"', '"base"'):
        assert key not in text


def test_statistics_of_a_small_model_dont_go_with_it():
    X, y = _problem()
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for model in (
        GradientBoostingClassifier(n_estimators=1, max_depth=1, random_state=0),
        make_pipeline(StandardScaler(), LogisticRegression()),
    ):
        summary = extract_structure(model.fit(X, y), feature_names=NAMES, weights="summary").spec
        assert '"stats"' not in json.dumps(summary)


def test_statistics_of_a_large_layer_still_go():
    X, y = _problem()
    from sklearn.neural_network import MLPClassifier

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPClassifier((128,), max_iter=5, random_state=0).fit(X, y)
    summary = extract_structure(model, feature_names=NAMES, weights="summary").spec
    assert '"stats"' in json.dumps(summary)


def _torch_models():
    torch = pytest.importorskip("torch")
    nn, functional = torch.nn, torch.nn.functional

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 8)
            self.act = nn.ReLU()
            self.last = nn.Linear(8, 1)

        def forward(self, x):
            return self.last(self.act(self.first(x)))

    class Functional(Plain):
        def forward(self, x):
            return self.last(functional.relu(self.first(x)))

    class Skip(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4)
            self.last = nn.Linear(4, 1)

        def forward(self, x):
            return self.last(x + self.first(x))

    return torch, nn, Plain, Functional, Skip


def test_a_network_is_only_called_exact_when_it_gives_its_own_answers():
    torch, nn, Plain, Functional, Skip = _torch_models()
    torch.manual_seed(0)
    for model in (
        Plain(),
        nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 1)),
    ):
        spec = estimator_to_spec(model.eval(), feature_names=NAMES)
        assert spec_is_executable(spec), spec.get("summary")
    # Read op by op, what forward() does between its layers is part of the model.
    functional = Functional().eval()
    spec = estimator_to_spec(functional, feature_names=NAMES)
    assert spec_is_executable(spec), spec.get("summary")
    rows = np.random.default_rng(3).normal(size=(32, 4))
    columns = {name: rows[:, index : index + 1] for index, name in enumerate(NAMES)}
    with torch.no_grad():
        truth = functional(torch.as_tensor(rows, dtype=torch.float32)).numpy()
    assert np.asarray(_evaluate(spec, columns, 32)) == pytest.approx(truth, abs=1e-5)
    # A skip from the raw inputs is a layer too. This client can't score a branch, and the API's check does.
    spec = estimator_to_spec(Skip().eval(), feature_names=NAMES)
    assert spec_is_executable(spec), spec.get("summary")
    assert [layer["inputs"] for layer in spec["layers"] if layer.get("type") == "add"][0][0] == "input"


def test_a_torch_classifier_that_ends_in_logits_gets_probabilities():
    torch, nn, *_ = _torch_models()
    torch.manual_seed(0)
    rows = np.random.default_rng(0).normal(size=(32, 4))
    names = {name: rows[:, index : index + 1] for index, name in enumerate(NAMES)}
    binary = nn.Sequential(nn.Linear(4, 1)).eval()
    spec = estimator_to_spec(binary, feature_names=NAMES, task="binary_classification")
    assert spec_is_executable(spec), spec.get("summary")
    with torch.no_grad():
        truth = torch.sigmoid(binary(torch.as_tensor(rows, dtype=torch.float32))).numpy().ravel()
    assert np.asarray(_evaluate(spec, names, 32))[:, 1] == pytest.approx(truth, abs=1e-5)
    several = nn.Sequential(nn.Linear(4, 3)).eval()
    spec = estimator_to_spec(several, feature_names=NAMES, task="multiclass_classification")
    assert spec_is_executable(spec), spec.get("summary")
    with torch.no_grad():
        truth = torch.softmax(several(torch.as_tensor(rows, dtype=torch.float32)), dim=1).numpy()
    assert np.asarray(_evaluate(spec, names, 32)) == pytest.approx(truth, abs=1e-5)
    already = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()).eval()
    spec = estimator_to_spec(already, feature_names=NAMES, task="binary_classification")
    assert spec_is_executable(spec), spec.get("summary")
    assert sum(item.get("activation") == "sigmoid" for item in spec["layers"]) == 1


def _numbers_of(spec):
    """Thresholds, leaves, weights, curves and scaler stats: everything a spec holds apart from shapes and indexes."""
    found = []

    def walk(value, key=None):
        if isinstance(value, dict):
            if "$shape" in value:
                return
            for inner, item in value.items():
                if inner in {"importances", "n_estimators", "reader", "$shape", "shape", "stats", "roots", "feature", "left", "right", "tree_class", "members", "id", "root"}:
                    continue
                walk(item, inner)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            found.append(float(value))

    walk(spec)
    return found


def _recording(answers):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"id": "m", "features": answers})
        return httpx.Response(200, json={"id": "out", "object": "x", "output": {"value": 1.0}})

    client = Kaimei(api_key="k", base_url="https://api.kaimei.ai", max_retries=0)
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(handler))
    return client, seen


@pytest.mark.parametrize("nan", [float("nan"), np.nan, np.float64("nan"), np.float32("nan")])
def test_nan_is_refused_where_the_model_cant_take_a_gap(nan):
    client, seen = _recording([{"name": "x", "type": "numeric", "baseline": 3.0}])
    for call in (
        lambda: client.models.predict("m", input={"x": nan}),
        lambda: client.explanations.create(model="m", input={"x": nan}),
        lambda: client.models.trace_structure("m", input={"x": nan}),
    ):
        with pytest.raises(ValueError, match="doesn't take missing values"):
            call()
    assert all(request.method == "GET" for request in seen)


def test_nan_goes_as_null_where_the_model_can_take_a_gap():
    client, seen = _recording([{"name": "x", "type": "numeric", "allow_missing": True}])
    client.models.predict("m", input={"x": float("nan")})
    assert json.loads(seen[-1].content)["input"] == {"x": None}


@pytest.mark.parametrize(
    "given",
    [
        lambda: zip(["x"], [1.0]),
        lambda: [("x", 1.0)],
        lambda: {"x": 1.0, "items": 2.0},
    ],
)
def test_whatever_dict_takes_still_goes_as_input(given):
    client, seen = _recording([{"name": "x", "type": "numeric"}])
    expected = dict(given())
    client.models.predict("m", input=given())
    assert json.loads(seen[-1].content)["input"] == expected
    client.explanations.create(model="m", input=given())
    assert json.loads(seen[-1].content)["input"] == expected
    client.models.trace_structure("m", input=given())
    assert json.loads(seen[-1].content)["input"] == expected


def test_no_extra_request_without_a_nan():
    client, seen = _recording([{"name": "x", "type": "numeric"}])
    client.models.predict("m", input={"x": 1.0})
    client.models.predict("m", input={"x": None})
    client.explanations.create(model="m", input={"x": float("nan")}, attributions={"x": 0.5})
    assert all(request.method == "POST" for request in seen)


def test_infinity_is_a_plain_error():
    with pytest.raises(ValueError, match="Infinity can't be sent"):
        encode_json({"input": {"x": float("inf")}})
    with pytest.raises(ValueError, match="Infinity can't be sent"):
        encode_json({"input": {"x": float("nan"), "y": float("-inf")}})
    assert json.loads(encode_json({"input": {"x": float("nan")}})) == {"input": {"x": None}}


def test_the_wrong_number_of_names_is_refused_for_every_kind():
    X, y = _problem()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    for model in (
        LogisticRegression(),
        DecisionTreeClassifier(max_depth=2),
        RandomForestClassifier(n_estimators=3, max_depth=2),
        RandomForestClassifier(n_estimators=60, max_depth=2),
        make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=2)),
    ):
        model.fit(X, y)
        for names in (NAMES[:3], NAMES + ["e"]):
            with pytest.raises(StructureError, match="feature"):
                estimator_to_spec(model, feature_names=names)
    assert estimator_to_spec(model, feature_names=NAMES)["feature_names"] == NAMES


def test_a_lightgbm_subclass_keeps_its_labels():
    lightgbm = pytest.importorskip("lightgbm")
    X, _ = _problem()
    y = np.where(X[:, 0] > 0, "yes", "no")

    class Mine(lightgbm.LGBMClassifier):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Mine(n_estimators=3, verbose=-1).fit(X, y)
    spec = estimator_to_spec(model, feature_names=NAMES)
    assert spec["classes"] == ["no", "yes"]
    assert spec_is_executable(spec)


def test_a_booster_with_the_wrong_number_of_names_is_refused():
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = lightgbm.LGBMClassifier(n_estimators=3, verbose=-1).fit(X, y).booster_
    with pytest.raises(StructureError, match="feature"):
        estimator_to_spec(booster, feature_names=NAMES[:2])


def test_a_subclass_with_its_own_predict_is_drawn_not_run():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    class Capped(LinearRegression):
        def predict(self, X):
            return np.minimum(super().predict(X), 1000.0)

    class Named(LinearRegression):
        """Changes nothing about how it predicts, so its numbers still describe it."""

    X = np.random.default_rng(0).uniform(0, 30, size=(100, 1))
    y = 100 * X[:, 0]
    assert extract_structure(Capped().fit(X, y), feature_names=["x"]).spec.get("approximate") is True
    assert extract_structure(Named().fit(X, y), feature_names=["x"]).spec.get("approximate") is None


def test_an_svc_whose_labels_can_disagree_with_its_probabilities_is_drawn_not_run():
    pytest.importorskip("sklearn")
    from sklearn.svm import SVC

    X, y = _problem()
    spec = extract_structure(SVC(kernel="linear", probability=True, random_state=0).fit(X, y), feature_names=NAMES).spec
    assert spec.get("approximate") is True
    assert not spec_is_executable(spec)
    spec = extract_structure(SVC(kernel="linear").fit(X, y), feature_names=NAMES).spec
    assert not spec.get("approximate")


def test_the_upload_check_uses_the_customers_rows():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    class Odd:
        """A wrapper of a model it predicts as, except on values only its real data reaches."""

        def __init__(self, model):
            self.model = model

        def predict(self, X):
            answer = self.model.predict(X)
            return np.where((X[:, 0] > 12345.0) & (X[:, 0] < 12346.0), 0.0, answer)

    own = np.random.default_rng(0).uniform(12345.0, 12346.0, size=(100, 1))
    wrapper = Odd(LinearRegression().fit(own, 100 * own[:, 0]))
    assert not extract_structure(wrapper, feature_names=["x"]).spec.get("approximate")
    assert extract_structure(wrapper, feature_names=["x"], check_rows=own).spec.get("approximate") is True


def test_a_cap_far_from_zero_is_seen_without_the_customers_rows():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    class Capped:
        def __init__(self, model):
            self.model = model

        def predict(self, X):
            return np.minimum(self.model.predict(X), 1000.0)

    X = np.random.default_rng(0).uniform(0, 30, size=(100, 1))
    assert extract_structure(Capped(LinearRegression().fit(X, 100 * X[:, 0])), feature_names=["x"]).spec.get("approximate")


def test_the_librarys_answers_go_with_the_model_and_stay_home_with_a_summary():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    X, y = _problem()
    model = RandomForestClassifier(5, max_depth=3, random_state=0).fit(X, y)
    check = extract_structure(model, feature_names=NAMES, check_rows=X).spec["check"]
    rows = np.asarray(check["rows"], dtype=float)
    assert rows.shape[1] == 4 and sum(check["groups"]) == rows.shape[0]
    assert np.asarray(check["outputs"]) == pytest.approx(model.predict_proba(rows))
    summary = extract_structure(model, feature_names=NAMES, check_rows=X, weights="summary").spec
    assert "check" not in summary
    assert not extract_structure(model, feature_names=NAMES).summary_spec().get("check")


def test_a_table_without_the_models_columns_is_refused():
    pandas = pytest.importorskip("pandas")
    table = pandas.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(StructureError, match="no column 'c'"):
        rows_from_inputs(table, ["a", "c"])
    assert rows_from_inputs(table, ["b", "a"])[0] == [{"b": 2.0, "a": 1.0}]


def test_a_table_names_a_model_fitted_on_an_array():
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    X, y = _problem()
    model = RandomForestClassifier(3, max_depth=2, random_state=0).fit(X, y)
    sent = []

    def respond(request):
        sent.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path.endswith("/reference-data"):
            return httpx.Response(200, json={"object": "reference_data", "rejected": 0})
        return httpx.Response(201 if request.method == "POST" else 200, json={"object": "model", "id": "mdl_1", "slug": "m"})

    client = Kaimei(api_key="k", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(respond))
    client.models.upload(model, name="m", X=pandas.DataFrame(X[:5], columns=NAMES))
    stored = next(body for _method, path, body in sent if path.endswith("/reference-data"))
    assert stored["rows"][0] == dict(zip(NAMES, X[0].tolist()))


def test_names_are_checked_against_wherever_a_library_keeps_them():
    catboost = pytest.importorskip("catboost")
    pandas = pytest.importorskip("pandas")
    X, _ = _problem()
    table = pandas.DataFrame(X, columns=NAMES)
    model = catboost.CatBoostRegressor(iterations=3, depth=2, verbose=False, allow_writing_files=False).fit(table, X[:, 0])
    with pytest.raises(StructureError, match="different order"):
        extract_structure(model, feature_names=list(reversed(NAMES)))


def test_a_custom_lightgbm_objective_isnt_read_as_probabilities():
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _problem()

    def logistic(target, score):
        probability = 1.0 / (1.0 + np.exp(-score))
        return probability - target, probability * (1.0 - probability)

    model = lightgbm.LGBMClassifier(n_estimators=3, objective=logistic, verbose=-1).fit(X, y)
    assert not spec_is_executable(extract_structure(model, feature_names=NAMES).spec)


@pytest.mark.parametrize("X", [None, np.zeros((4, 3)), "not a table", np.array([["a", "b"]])])
def test_rows_that_arent_a_table_of_numbers_arent_used(X):
    from kaimei.ensembles import rows_to_check

    assert rows_to_check(X, ["a", "b"]) is None


def test_rows_with_gaps_are_left_out_of_the_check():
    from kaimei.ensembles import rows_to_check

    rows = rows_to_check(np.array([[1.0, np.nan], [2.0, 3.0]]), ["a", "b"])
    assert rows.tolist() == [[2.0, 3.0]]


def test_a_catboost_regressor_read_as_a_summary_stays_a_regressor():
    catboost = pytest.importorskip("catboost")
    X, _ = _problem()
    model = catboost.CatBoostRegressor(grow_policy="Lossguide", iterations=3, depth=2, verbose=False, allow_writing_files=False)
    spec = extract_structure(model.fit(X, X[:, 0]), feature_names=NAMES).spec
    assert spec["task"] == "regression"


def test_torch_activations_are_compared_with_what_the_api_runs():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(0)
    names = ["a", "b", "c", "d"]
    runs = [nn.LeakyReLU(0.1), nn.ELU(), nn.ReLU6(), nn.Hardswish(), nn.SiLU(), nn.GELU(approximate="tanh")]
    for activation in runs:
        spec = extract_structure(nn.Sequential(nn.Linear(4, 8), activation, nn.Linear(8, 1)), feature_names=names).spec
        assert not spec.get("approximate"), type(activation).__name__
    # The API runs the tanh form of GELU, and torch's default is the exact one. Close isn't the same answer.
    spec = extract_structure(nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1)), feature_names=names).spec
    assert spec.get("approximate") is True


def test_a_torch_layer_called_twice_reads_as_two_steps():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.step, self.act, self.out = nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1)

        def forward(self, x):
            return self.out(self.act(self.step(self.act(self.step(x)))))

    torch.manual_seed(0)
    spec = extract_structure(Twice(), feature_names=NAMES).spec
    ids = [layer["id"] for layer in spec["layers"]]
    assert len(ids) == len(set(ids))
    assert not spec.get("approximate")


def test_a_refused_reference_row_is_reported():
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.method == "PUT" and request.url.path.endswith("/reference-data"):
            return httpx.Response(200, json={"object": "reference_data", "model_id": "m", "rows": 1, "has_labels": False, "rejected": 1})
        if request.method == "PUT":
            return httpx.Response(200, json={"object": "structure"})
        return httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "m"})

    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    client = Kaimei(api_key="k", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(respond))
    X = np.array([[1.0], [2.0], [3.0]])
    with pytest.warns(UserWarning, match="reference rows"):
        client.models.upload(LinearRegression().fit(X, X[:, 0]), name="m", feature_names=["x"], X=X)


def test_reference_rows_go_up_in_pieces(monkeypatch):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    from kaimei.resources import models as module

    monkeypatch.setattr(module, "REFERENCE_PIECE_BYTES", 100)
    pieces = []

    def respond(request):
        if request.method == "PUT" and request.url.path.endswith("/reference-data"):
            sent = json.loads(request.content)
            pieces.append((sent.get("mode", "replace"), len(sent["rows"]), sent.get("labels")))
            return httpx.Response(200, json={"object": "reference_data", "model_id": "m", "rows": 1, "rejected": 0})
        if request.method == "PUT":
            return httpx.Response(200, json={"object": "structure"})
        return httpx.Response(201, json={"object": "model", "id": "mdl_1", "slug": "m"})

    client = Kaimei(api_key="k", base_url="https://api.kaimei.ai")
    client._transport._client = httpx.Client(base_url="https://api.kaimei.ai", transport=httpx.MockTransport(respond))
    X = np.arange(40, dtype=float).reshape(-1, 1)
    client.models.upload(LinearRegression().fit(X, X[:, 0]), name="m", feature_names=["x"], X=X, y=X[:, 0])
    assert len(pieces) > 1
    assert pieces[0][0] == "replace" and all(piece[0] == "append" for piece in pieces[1:])
    assert sum(piece[1] for piece in pieces) == 40
    assert sum(len(piece[2]) for piece in pieces) == 40
