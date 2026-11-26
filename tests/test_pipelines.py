"""The two scikit-learn pipelines real data most often comes in: words through a bag of words, and tables with
columns of categories. Both are read exactly, and held to the pipeline's own answers."""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnx")
pytest.importorskip("skl2onnx")
pd = pytest.importorskip("pandas")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import (  # noqa: E402
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import LinearSVC  # noqa: E402

from kaimei import StructureError  # noqa: E402
from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402
from kaimei.tokenizers import Tokenizer  # noqa: E402

GOOD = ["good movie", "great film loved it", "wonderful acting", "a great story", "loved the music", "good fun", "Café crème was lovely"]
BAD = ["bad movie", "terrible film hated it", "awful acting", "a boring story", "hated the music", "bad fun", "the café was awful"]
TEXTS = (GOOD + BAD) * 3
LABELS = np.array(([1] * len(GOOD) + [0] * len(BAD)) * 3)


def runs(spec):
    return "check" in spec and not spec.get("approximate") and spec_is_executable(spec)


@pytest.mark.parametrize(
    "pipeline, target",
    [
        (Pipeline([("v", TfidfVectorizer()), ("m", LogisticRegression())]), LABELS),
        (
            Pipeline([
                ("v", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", stop_words="english")),
                ("m", LogisticRegression()),
            ]),
            np.arange(len(TEXTS)) % 3,
        ),
        (Pipeline([("c", CountVectorizer(binary=True)), ("t", TfidfTransformer(norm="l1")), ("m", MultinomialNB())]), LABELS),
        (Pipeline([("v", TfidfVectorizer()), ("m", LinearSVC())]), LABELS),
        (Pipeline([("v", TfidfVectorizer()), ("m", Ridge())]), LABELS.astype(float)),
    ],
    ids=["tfidf-logistic", "bigrams-sublinear-accents-stopwords", "counts-tfidf-bayes", "tfidf-svc", "tfidf-ridge"],
)
def test_a_bag_of_words_pipeline_is_read_exactly(pipeline, target):
    pipeline.fit(TEXTS, target)
    spec = extract_structure(pipeline, check_rows=TEXTS, feature_names=["review"]).spec
    assert runs(spec) and spec["features"][0]["name"] == "review" and spec["features"][0]["type"] == "text"
    described = spec["tokenizers"]["review"]
    assert described["type"] == "word" and described["pattern"] == "sklearn"
    # The tokenizer the API runs splits a new text as scikit-learn does, word for word.
    analyzer = pipeline.steps[0][1].build_analyzer()
    splitter = Tokenizer(described)
    for text in ["Unseen words, GREAT film", "the café crème", "a b c good"]:
        words = [term for term in analyzer(text) if " " not in term]
        known = [word if word in splitter.ids else "<unk>" for word in words]
        assert [splitter.tokens[index] for index in splitter.encode(text)] == known


def test_a_custom_token_pattern_is_refused_with_why():
    pipeline = Pipeline([("v", TfidfVectorizer(token_pattern=r"\w+")), ("m", LogisticRegression())]).fit(TEXTS, LABELS)
    with pytest.raises(StructureError) as caught:
        extract_structure(pipeline, check_rows=TEXTS)
    assert "token_pattern" in caught.value.message


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(1)
    rows = 300
    frame = pd.DataFrame({
        "age": rng.integers(18, 80, rows).astype(float),
        "income": rng.normal(50, 15, rows),
        "city": rng.choice(["NY", "SF", "LA"], rows),
        "plan": rng.choice(["basic", "pro"], rows),
        "extra": rng.normal(size=rows),
    })
    answer = ((frame.age > 40) ^ (frame.city == "SF")).astype(int).to_numpy()
    return frame, answer


def test_a_table_with_columns_of_categories_is_read_exactly(table):
    frame, answer = table
    pipeline = Pipeline([
        ("pre", ColumnTransformer([("num", StandardScaler(), ["age", "income"]), ("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "plan"])])),
        ("clf", LogisticRegression()),
    ]).fit(frame, answer)
    spec = extract_structure(pipeline, check_rows=frame).spec
    assert spec["task"] == "binary_classification" and runs(spec)
    features = {item["name"]: item for item in spec["features"]}
    assert features["city"] == {"name": "city", "type": "categorical", "categories": ["LA", "NY", "SF"]}
    assert features["age"] == {"name": "age", "type": "numeric"}
    # The check holds a category as its position in the feature's categories, as the API encodes it.
    assert {row[0] for row in spec["check"]["inputs"]["city"]} <= {0.0, 1.0, 2.0}

    def value(name, held):
        listed = features[name].get("categories")
        return listed[int(held)] if listed else held

    rows = [
        {name: value(name, spec["check"]["inputs"][name][index][0]) for name in features}
        for index in range(len(spec["check"]["outputs"]))
    ]
    own = pipeline.predict_proba(pd.DataFrame(rows)[list(frame.columns[:4])])
    np.testing.assert_allclose(spec["check"]["outputs"], own, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("which", ["forest", "boosting"])
def test_dropped_columns_ordinal_codes_and_the_remainder_are_read_too(table, which):
    frame, answer = table
    if which == "forest":
        front = ColumnTransformer([("num", MinMaxScaler(), ["age"]), ("cat", OneHotEncoder(drop="first"), ["city", "plan"])], remainder="passthrough")
        pipeline = Pipeline([("pre", front), ("clf", RandomForestClassifier(20, max_depth=4, random_state=0))]).fit(frame, answer)
    else:
        front = ColumnTransformer([("cat", OrdinalEncoder(), ["city", "plan"]), ("num", "passthrough", ["age", "income"])])
        pipeline = Pipeline([("pre", front), ("reg", GradientBoostingRegressor(n_estimators=20, random_state=0))]).fit(frame, frame.income)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(pipeline, check_rows=frame).spec
    assert runs(spec)
    assert {item["name"] for item in spec["features"] if item["type"] == "categorical"} == {"city", "plan"}
