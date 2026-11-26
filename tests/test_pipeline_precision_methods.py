"""The precision a table pipeline is computed in is settled by every method its model has that gives numbers.

A pipeline's model was asked whether it tells doubles apart by the first method that answered. A naive Bayes
model has no ``decision_function``, and its probabilities far from its data are 0 and 1 in either precision, so
behind columns whose range nothing says (uploaded with no rows) it was kept in single precision, called exact
and verified, while its probabilities were 0.004 off on rows like its own. Its log-probabilities do differ, and
are asked now, as a bare model's always were.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
pd = pytest.importorskip("pandas")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from kaimei import tabular_pipelines  # noqa: E402
from kaimei.structure import extract_structure  # noqa: E402


def _rows(count: int, seed: int):
    random = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "plan": random.choice(["free", "pro", "team"], count),
            "ts": 1_700_000_000 + random.uniform(0, 86_400, count),
            "age": random.normal(40, 12, count),
        }
    )
    late = ((frame["ts"] - 1_700_000_000) / 8640.0 + random.normal(0, 1.0, count) > 5).astype(int)
    return frame, late


def _pipeline(model):
    X, y = _rows(400, 0)
    columns = ColumnTransformer([("c", OneHotEncoder(), ["plan"])], remainder="passthrough")
    return make_pipeline(columns, model).fit(X, y)


def test_a_model_whose_probabilities_say_nothing_is_asked_by_its_other_numbers():
    model = _pipeline(GaussianNB())[-1]
    width = int(model.n_features_in_)
    # Its probabilities at whole numbers past 2**24 are the same in either precision; its log-probabilities are not.
    assert tabular_pipelines._tells_doubles_apart(model, [], width, np)
    # A tree rounds its inputs to single precision itself: no method of it tells doubles apart.
    tree = _pipeline(DecisionTreeClassifier(max_depth=3, random_state=0))[-1]
    assert not tabular_pipelines._tells_doubles_apart(tree, [], int(tree.n_features_in_), np)


def test_naive_bayes_behind_columns_of_unknown_range_is_computed_in_doubles():
    pipeline = _pipeline(GaussianNB())
    spec = extract_structure(pipeline).spec  # no rows given: nothing says where "ts" lies
    assert spec["metadata"].get("model_precision") == "double", spec["metadata"]
    assert not spec.get("approximate"), spec.get("summary")
