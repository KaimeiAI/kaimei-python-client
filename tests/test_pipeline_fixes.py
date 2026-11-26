"""Fixes to the scikit-learn pipelines Kaimei reads itself: tables with columns of categories (``tabular_pipelines``)
and bags of words (``text_pipelines``).

Each pipeline that is read is also run where the check did not look: the ONNX model the reading was taken from (the
graph the spec holds) is given rows and texts the check did not hold, as the API gives them (a category as its
position, a missing value as its baseline, a text split by the spec's own tokenizer), and must give the pipeline's
own answers."""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")
pd = pytest.importorskip("pandas")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import (  # noqa: E402
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeClassifier  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.multioutput import MultiOutputClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import LinearSVC  # noqa: E402

from kaimei import StructureError, text_pipelines  # noqa: E402
from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402
from kaimei.tokenizers import Tokenizer  # noqa: E402

FILM = ["good film great plot", "bad film poor plot", "great acting good story", "poor acting bad story", "fine film", "awful plot really bad", "not good at all", "really great"] * 6
FILM_LABELS = np.array([1, 0, 1, 0, 1, 0, 0, 1] * 6)
UNSEEN = ["good plot and a great film", "the film was not good at all", "a bad bad story", "zebra quokka", "fine, fine: a FINE film!", "really really bad acting"]
CURLY = ["I don’t like it", "don’t go", "great café", "I do like it", "bad film", "good film", "naïve plot", "it’s fine"] * 4
CURLY_LABELS = np.array([0, 0, 1, 1, 0, 1, 1, 1] * 4)


def runs(spec):
    return "check" in spec and not spec.get("approximate") and spec_is_executable(spec)


def read(pipeline, rows, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return extract_structure(pipeline, check_rows=rows, **options)


def shown(values, link):
    """Numbers as a head with this link shows them: a sigmoid of one margin, a softmax of several, or as they are."""
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    if link == "sigmoid":
        return 1.0 / (1.0 + np.exp(-values))
    if link == "softmax":
        shifted = np.exp(values - values.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)
    return values


def kept(extraction):
    """ONNX Runtime on the model the reading was taken from, and the output its prediction is read from."""
    model = onnx.load_from_string(extraction.onnx)
    floats = [item.name for item in model.graph.output if item.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE)]
    margins = [name for name in floats if name.endswith("margin")]
    session = ort.InferenceSession(extraction.onnx, providers=["CPUExecutionProvider"])
    return session, (margins or floats)[-1]


def agrees(extraction, pipeline, given, feeds):
    """The kept model on ``feeds`` (one per row) gives the pipeline's own answers on ``given``, as the head reads them."""
    session, output = kept(extraction)
    ours = np.stack([np.asarray(session.run([output], feed)[0], dtype=float).reshape(-1) for feed in feeds])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = np.asarray(getattr(pipeline, extraction.spec["metadata"]["library_method"])(given), dtype=float)
    link = extraction.spec["head"].get("link") or "none"
    np.testing.assert_allclose(shown(ours, link), shown(theirs.reshape(len(ours), -1), link), rtol=1e-4, atol=1e-5)


def table_feeds(extraction, frame):
    """A table's rows as the API gives them to the model: a category as its position, a gap as the baseline."""
    session, _output = kept(extraction)
    types = {item.name: item.type for item in session.get_inputs()}
    features = {item["name"]: item for item in extraction.spec["features"]}
    feeds = []
    for record in frame.to_dict(orient="records"):
        feed = {}
        for name, feature in features.items():
            value = record[name]
            if value is None or (isinstance(value, float) and value != value):
                value = feature["baseline"]  # a null takes the feature's baseline
            if feature["type"] == "categorical":
                text = value if isinstance(value, str) and value in feature["categories"] else str(value.item() if hasattr(value, "item") else value)
                feed[name] = np.array([[feature["categories"].index(text)]], dtype=np.int64)
            else:
                feed[name] = np.array([[float(value)]], dtype=np.float64 if types[name] == "tensor(double)" else np.float32)
        feeds.append(feed)
    return feeds


def text_feeds(extraction, texts):
    """Texts split by the spec's own tokenizer, padded to the length the model reads."""
    feature = extraction.spec["features"][0]
    splitter = Tokenizer(extraction.spec["tokenizers"][feature["name"]])
    length, pad = int(feature["max_length"]), splitter.ids["<pad>"]
    return [{"ids": np.array([(ids := splitter.encode(text)[:length]) + [pad] * (length - len(ids))], dtype=np.int64)} for text in texts]


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(3)
    rows = 320
    frame = pd.DataFrame(
        {
            "city": rng.choice(["Oslo", "Lima", "Pune", "Kyiv"], rows, p=[0.4, 0.3, 0.2, 0.1]),
            "plan": rng.choice(["free", "pro", "team"], rows, p=[0.6, 0.3, 0.1]),
            "age": rng.normal(40, 12, rows).round(1),
            "income": rng.normal(52, 9, rows).round(2),
            "floor": rng.integers(1, 5, rows),
            "member": rng.choice([True, False], rows),
        }
    )
    answer = np.where(((frame.plan != "free") & (frame.age > 35)) | (frame.city == "Oslo") ^ frame.member, "leaves", "stays")
    return frame.iloc[:240].reset_index(drop=True), answer[:240], frame.iloc[240:].reset_index(drop=True)


@pytest.fixture(scope="module")
def gappy(table):
    """The table with values missing in every kind of column, in the rows fitted on and in the fresh ones."""
    frame, answer, fresh = table
    rng = np.random.default_rng(4)
    holes = []
    for data in (frame, fresh):
        data = data.copy()
        data["city"] = data["city"].astype(object)
        for column in ("city", "age", "income"):
            data.loc[rng.random(len(data)) < 0.12, column] = np.nan
        holes.append(data)
    return holes[0], answer, holes[1]


# --- strip_accents="ascii", and splitting compared on real texts ----------------------------------------------------


def test_ascii_accent_stripping_is_refused_with_why():
    pipeline = make_pipeline(TfidfVectorizer(strip_accents="ascii"), LogisticRegression()).fit(CURLY, CURLY_LABELS)
    with pytest.raises(StructureError) as caught:
        read(pipeline, CURLY)
    assert "strip_accents='ascii'" in caught.value.message and "strip_accents='unicode'" in caught.value.message


def test_a_tokenizer_that_splits_a_text_differently_is_refused(monkeypatch):
    # Were "ascii" let through, Kaimei's words would differ from scikit-learn's on "don’t" ("don" against "dont"):
    # the comparison of how each splits the texts refuses it.
    monkeypatch.setattr(text_pipelines, "_unsupported", lambda vectorizer: None)
    pipeline = make_pipeline(TfidfVectorizer(strip_accents="ascii"), LogisticRegression()).fit(CURLY, CURLY_LABELS)
    with pytest.raises(StructureError) as caught:
        read(pipeline, CURLY)
    assert "would read" in caught.value.message and "don’t" in caught.value.message


def test_the_check_gives_scikit_learn_the_callers_texts_as_written(monkeypatch):
    # With both refusals off, the check's rows of the caller's texts are given to scikit-learn as they were written,
    # not as their ids spell them, so the difference in splitting shows: the reading is drawn, not run.
    monkeypatch.setattr(text_pipelines, "_unsupported", lambda vectorizer: None)
    monkeypatch.setattr(text_pipelines, "_split_differently", lambda vectorizer, splitter, texts: None)
    pipeline = make_pipeline(TfidfVectorizer(strip_accents="ascii"), LogisticRegression()).fit(CURLY, CURLY_LABELS)
    assert read(pipeline, CURLY).spec.get("approximate")


def test_unicode_accent_stripping_reads_curly_quotes_and_accents_exactly():
    pipeline = make_pipeline(TfidfVectorizer(strip_accents="unicode"), LogisticRegression()).fit(CURLY, CURLY_LABELS)
    extraction = read(pipeline, CURLY)
    assert runs(extraction.spec)
    unseen = ["don’t", "I don’t like it at all", "Straße café", "naïve and NAÏVE"]
    agrees(extraction, pipeline, unseen, text_feeds(extraction, unseen))


# --- encoders on columns of numbers and flags ----------------------------------------------------------------------


@pytest.mark.parametrize("encoder", [OneHotEncoder(handle_unknown="ignore"), OrdinalEncoder()], ids=["one-hot", "ordinal"])
def test_numbers_and_flags_encoded_as_categories_are_read_exactly(table, encoder):
    frame, answer, fresh = table
    front = ColumnTransformer([("cat", encoder, ["floor", "member", "city"]), ("num", StandardScaler(), ["age"])])
    pipeline = Pipeline([("pre", front), ("clf", LogisticRegression(max_iter=500))]).fit(frame, answer)
    extraction = read(pipeline, frame)
    assert runs(extraction.spec)
    features = {item["name"]: item for item in extraction.spec["features"]}
    assert features["floor"]["categories"] == ["1", "2", "3", "4"] and features["member"]["categories"] == ["False", "True"]
    columns = ["floor", "member", "city", "age"]
    agrees(extraction, pipeline, fresh[columns], table_feeds(extraction, fresh[columns]))


# --- classifiers without predict_proba -----------------------------------------------------------------------------


@pytest.mark.parametrize("classes", [2, 3])
def test_a_classifier_with_scores_and_no_probabilities_is_read_behind_the_columns(table, classes):
    frame, answer, fresh = table
    target = answer if classes == 2 else frame["plan"].to_numpy()
    front = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]), ("num", StandardScaler(), ["age", "income"])])
    pipeline = Pipeline([("pre", front), ("clf", LinearSVC())]).fit(frame, target)
    extraction = read(pipeline, frame)
    assert runs(extraction.spec) and extraction.spec["metadata"]["library_method"] == "decision_function"
    columns = ["city", "age", "income"]
    agrees(extraction, pipeline, fresh[columns], table_feeds(extraction, fresh[columns]))


# --- TfidfVectorizer(use_idf=False), and tf-idf twice --------------------------------------------------------------


@pytest.mark.parametrize(
    "steps",
    [
        [TfidfVectorizer(use_idf=False)],
        [TfidfVectorizer(norm="l1", use_idf=False)],
        [TfidfVectorizer(use_idf=False, sublinear_tf=True)],
        [TfidfVectorizer(), TfidfTransformer(sublinear_tf=True)],
    ],
    ids=["l2-norm", "l1-norm", "sublinear", "two-tf-idf-steps"],
)
def test_tf_idf_scales_and_normalises_without_idf_weights_too(steps):
    pipeline = make_pipeline(*steps, LogisticRegression()).fit(FILM, FILM_LABELS)
    extraction = read(pipeline, FILM)
    assert runs(extraction.spec)
    agrees(extraction, pipeline, UNSEEN, text_feeds(extraction, UNSEEN))


# --- texts as a Series, an array, a frame --------------------------------------------------------------------------


LONG = [" ".join(["good film great plot really"] * 20 + [word]) for word in ("one", "two", "three")] + [
    " ".join(["bad film poor plot awful"] * 20 + [word]) for word in ("one", "two", "three")
]


@pytest.mark.parametrize("form", ["list", "Series", "array", "column", "frame", "named"])
def test_texts_of_any_one_dimensional_kind_are_found_and_held_by_the_check(form):
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(LONG, [1, 1, 1, 0, 0, 0])
    given = {
        "list": list(LONG),
        "Series": pd.Series(LONG),
        "array": np.array(LONG),
        "column": np.array(LONG).reshape(-1, 1),
        "frame": pd.DataFrame({"review": LONG}),
        "named": pd.DataFrame({"id": range(len(LONG)), "review": LONG}),
    }[form]
    extraction = read(pipeline, given, feature_names=["review"])
    assert runs(extraction.spec)
    # Room for a long document whatever the texts given, and more only when they are longer than half of it: a
    # longer text is refused by the API, not cut (tests/test_round3_pipelines.py).
    assert extraction.spec["features"][0]["max_length"] == text_pipelines.DEFAULT_TEXT_TOKENS
    # Every one of the texts is in the check, as it was written.
    splitter = Tokenizer(extraction.spec["tokenizers"]["review"])
    held = {tuple(int(value) for value in row[:101]) for row in extraction.spec["check"]["inputs"]["review"]}
    assert all(tuple(splitter.encode(text)) in held for text in LONG)


# --- infrequent categories -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoder",
    [
        OneHotEncoder(min_frequency=40, handle_unknown="infrequent_if_exist"),
        OneHotEncoder(max_categories=3, handle_unknown="infrequent_if_exist"),
        OneHotEncoder(min_frequency=0.15, drop="first"),
        OrdinalEncoder(min_frequency=40),
    ],
    ids=["min-frequency", "max-categories", "drop-first", "ordinal"],
)
def test_infrequent_categories_sharing_a_column_are_read_exactly(table, encoder):
    frame, answer, fresh = table
    front = ColumnTransformer([("cat", encoder, ["city", "plan"]), ("num", StandardScaler(), ["age"])])
    pipeline = Pipeline([("pre", front), ("clf", LogisticRegression(max_iter=500))]).fit(frame, answer)
    extraction = read(pipeline, frame)
    assert runs(extraction.spec)
    # Every category can still be sent, the rare ones too: the graph gives each what the encoder gives it.
    features = {item["name"]: item for item in extraction.spec["features"]}
    assert features["city"]["categories"] == ["Kyiv", "Lima", "Oslo", "Pune"]
    columns = ["city", "plan", "age"]
    agrees(extraction, pipeline, fresh[columns], table_feeds(extraction, fresh[columns]))


# --- columns chosen by masks and slices ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "words, numbers",
    [
        (np.array([True, True, False, False]), np.array([False, False, True, True])),
        ([np.True_, np.True_, np.False_, np.False_], [np.False_, np.False_, np.True_, np.True_]),
        (slice("city", "plan"), slice("age", "income")),
        (np.array([0, 1]), slice(2, 4)),
    ],
    ids=["numpy-mask", "numpy-bools", "names-slice", "numpy-positions"],
)
def test_columns_chosen_by_a_mask_or_a_slice_are_read(table, words, numbers):
    frame, answer, fresh = table
    ordered = frame[["city", "plan", "age", "income"]]
    front = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), words), ("num", StandardScaler(), numbers)])
    pipeline = Pipeline([("pre", front), ("clf", LogisticRegression(max_iter=500))]).fit(ordered, answer)
    extraction = read(pipeline, ordered)
    assert runs(extraction.spec)
    assert {item["name"] for item in extraction.spec["features"]} == {"city", "plan", "age", "income"}
    agrees(extraction, pipeline, fresh[ordered.columns], table_feeds(extraction, fresh[ordered.columns]))


# --- the common pipelines that were refused or drawn ---------------------------------------------------------------


def test_the_usual_template_of_an_imputer_and_an_encoder_for_each_kind_of_column(gappy):
    frame, answer, fresh = gappy
    front = ColumnTransformer(
        [
            ("words", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("hot", OneHotEncoder(handle_unknown="ignore"))]), ["city", "plan"]),
            ("numbers", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age", "income"]),
        ]
    )
    pipeline = Pipeline([("columns", front), ("model", LogisticRegression(max_iter=500))]).fit(frame, answer)
    extraction = read(pipeline, frame)
    spec = extraction.spec
    assert runs(spec)
    # How a missing value is sent: as null, which the API gives the model as the feature's baseline, the imputer's fill.
    features = {item["name"]: item for item in spec["features"]}
    assert features["city"]["baseline"] == frame["city"].mode()[0]
    assert features["age"]["baseline"] == pytest.approx(frame["age"].median())
    assert "sent as null" in spec["summary"]
    columns = ["city", "plan", "age", "income"]
    assert fresh[columns].isna().any(axis=1).sum() > 10
    agrees(extraction, pipeline, fresh[columns], table_feeds(extraction, fresh[columns]))


def test_an_encoder_straight_into_a_model_is_read_exactly(gappy):
    frame, answer, fresh = gappy
    words = ["city", "plan"]
    for steps in ([OneHotEncoder(handle_unknown="ignore")], [SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")]):
        given = frame[words].dropna() if len(steps) == 1 else frame[words]
        pipeline = make_pipeline(*steps, LogisticRegression(max_iter=500)).fit(given, answer[given.index])
        extraction = read(pipeline, given)
        assert runs(extraction.spec)
        rows = fresh[words].dropna() if len(steps) == 1 else fresh[words]
        agrees(extraction, pipeline, rows, table_feeds(extraction, rows))


def test_a_column_transformer_of_numbers_chosen_by_name_is_read(table):
    frame, answer, fresh = table
    front = ColumnTransformer([("num", StandardScaler(), ["age", "income"]), ("raw", "passthrough", ["floor"])])
    pipeline = Pipeline([("pre", front), ("clf", LogisticRegression(max_iter=500))]).fit(frame, answer)
    extraction = read(pipeline, frame)
    assert runs(extraction.spec)
    columns = ["age", "income", "floor"]
    agrees(extraction, pipeline, fresh[columns], table_feeds(extraction, fresh[columns]))


def test_a_part_kaimei_does_not_write_down_is_refused_by_name(table):
    frame, answer, _fresh = table
    front = ColumnTransformer([("cat", OneHotEncoder(), ["city"]), ("bins", KBinsDiscretizer(n_bins=3, encode="ordinal"), ["age"])])
    pipeline = Pipeline([("pre", front), ("clf", LogisticRegression(max_iter=500))]).fit(frame, answer)
    with pytest.raises(StructureError) as caught:
        read(pipeline, frame)
    assert "'bins'" in caught.value.message and "KBinsDiscretizer" in caught.value.message and "agent" in caught.value.message


def test_parts_chosen_by_position_are_still_left_to_skl2onnx(table):
    frame, answer, fresh = table
    numbers = frame[["age", "income"]]
    front = ColumnTransformer([("num", StandardScaler(), [0]), ("bins", KBinsDiscretizer(n_bins=3, encode="onehot-dense"), [1])])
    pipeline = Pipeline([("pre", front), ("clf", LogisticRegression(max_iter=500))]).fit(numbers, answer)
    assert runs(read(pipeline, numbers).spec)


@pytest.mark.parametrize(
    "vectorizer, model",
    [
        (TfidfVectorizer(ngram_range=(1, 2), stop_words="english"), LogisticRegression()),
        (TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english"), LinearSVC()),
        (TfidfVectorizer(ngram_range=(2, 2)), LogisticRegression()),
        # Phrases one-word counting never forms, the last column among them: always 0, as in scikit-learn.
        (CountVectorizer(vocabulary=["good", "bad", "plot", "good film", "great acting"]), LogisticRegression()),
    ],
    ids=["bigrams-stop-words", "bigrams-stop-words-svc", "bigrams-only", "phrase-vocabulary"],
)
def test_n_grams_and_stop_words_are_read_exactly(vectorizer, model):
    pipeline = make_pipeline(vectorizer, model).fit(FILM, FILM_LABELS)
    extraction = read(pipeline, FILM)
    assert runs(extraction.spec)
    agrees(extraction, pipeline, UNSEEN, text_feeds(extraction, UNSEEN))


@pytest.mark.parametrize(
    "steps",
    [
        [TfidfVectorizer(max_features=6), RandomForestClassifier(8, max_depth=4, random_state=0)],
        [TfidfVectorizer(), TruncatedSVD(3, random_state=0), LogisticRegression()],
    ],
    ids=["forest", "svd-then-logistic"],
)
def test_any_model_after_the_counts_is_converted_and_joined_on(steps):
    pipeline = make_pipeline(*steps).fit(FILM, FILM_LABELS)
    extraction = read(pipeline, FILM)
    assert runs(extraction.spec) and "skl2onnx" in extraction.spec["summary"]
    agrees(extraction, pipeline, UNSEEN, text_feeds(extraction, UNSEEN))


@pytest.mark.parametrize("analyzer", ["char", "char_wb"])
def test_a_character_analyzer_is_refused_with_why(analyzer):
    pipeline = make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=(2, 3)), LogisticRegression()).fit(FILM, FILM_LABELS)
    with pytest.raises(StructureError) as caught:
        read(pipeline, FILM)
    assert f"analyzer={analyzer!r}" in caught.value.message and "analyzer='word'" in caught.value.message


def test_a_vocabulary_word_longer_than_a_token_is_refused_with_why():
    long_word = "a" + "b" * 300
    pipeline = make_pipeline(CountVectorizer(), LogisticRegression()).fit(FILM + [long_word] * 2, list(FILM_LABELS) + [1, 0])
    with pytest.raises(StructureError) as caught:
        read(pipeline, FILM)
    assert "301 characters" in caught.value.message


def test_a_pipeline_that_ends_without_a_model_is_refused_with_why():
    pipeline = make_pipeline(CountVectorizer(), TfidfTransformer()).fit(FILM)
    with pytest.raises(StructureError) as caught:
        read(pipeline, FILM)
    assert "TfidfTransformer, is not a model" in caught.value.message


# --- multi-label classifiers at the end of a pipeline --------------------------------------------------------------


def check_frame(spec):
    """The check's rows as a table of the pipeline's columns: a category's position back to its name."""
    features = {item["name"]: item for item in spec["features"]}
    columns = {}
    for name, rows in spec["check"]["inputs"].items():
        listed = features[name].get("categories")
        columns[name] = [listed[int(row[0])] if listed else float(row[0]) for row in rows]
    return pd.DataFrame(columns)


def tags_of(frame):
    """Three labels a row may have any of: a table of yes and no, as a multi-label classifier is fitted on."""
    return np.column_stack([frame["age"] > 40, frame["city"] == "Oslo", frame["plan"] != "free"]).astype(int)


@pytest.mark.parametrize(
    "model",
    [OneVsRestClassifier(LogisticRegression(max_iter=500)), MLPClassifier((8,), max_iter=300, random_state=0)],
    ids=["one-vs-rest", "mlp"],
)
def test_a_multi_label_classifier_after_the_columns_gives_each_labels_own_probability(table, model):
    frame, _answer, fresh = table
    front = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "plan"]), ("num", StandardScaler(), ["age", "income"])])
    pipeline = Pipeline([("pre", front), ("clf", model)]).fit(frame, tags_of(frame))
    extraction = read(pipeline, frame)
    spec = extraction.spec
    assert runs(spec) and spec["task"] == "multilabel_classification" and spec["head"]["link"] == "none"
    # The check holds the pipeline's own probabilities, one for each label, never put through a softmax.
    columns = ["city", "plan", "age", "income"]
    np.testing.assert_allclose(spec["check"]["outputs"], pipeline.predict_proba(check_frame(spec)[columns]), rtol=1e-4, atol=1e-5)
    agrees(extraction, pipeline, fresh[columns], table_feeds(extraction, fresh[columns]))


def test_a_multi_label_classifier_of_scores_squashes_each_label_on_its_own(table):
    frame, _answer, fresh = table
    front = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "plan"]), ("num", StandardScaler(), ["age", "income"])])
    pipeline = Pipeline([("pre", front), ("clf", RidgeClassifier())]).fit(frame, tags_of(frame))
    extraction = read(pipeline, frame)
    spec = extraction.spec
    assert runs(spec) and spec["task"] == "multilabel_classification" and spec["head"]["link"] == "sigmoid"
    assert "each label's share shown is its own score" in spec["summary"]
    columns = ["city", "plan", "age", "income"]
    agrees(extraction, pipeline, fresh[columns], table_feeds(extraction, fresh[columns]))


def test_a_multi_label_classifier_after_tf_idf_gives_each_labels_own_probability():
    tags = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0]] * 6)
    pipeline = make_pipeline(TfidfVectorizer(), OneVsRestClassifier(LogisticRegression())).fit(FILM, tags)
    extraction = read(pipeline, FILM)
    spec = extraction.spec
    assert runs(spec) and spec["task"] == "multilabel_classification" and spec["head"]["link"] == "none"
    agrees(extraction, pipeline, UNSEEN, text_feeds(extraction, UNSEEN))


def test_a_multi_label_classifier_read_as_another_task_is_refused_with_why(table):
    frame, _answer, _fresh = table
    front = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "plan"]), ("num", StandardScaler(), ["age"])])
    pipeline = Pipeline([("pre", front), ("clf", OneVsRestClassifier(LogisticRegression(max_iter=500)))]).fit(frame, tags_of(frame))
    with pytest.raises(StructureError) as caught:
        read(pipeline, frame, task="multiclass_classification")
    assert "task='multilabel_classification'" in caught.value.message


@pytest.mark.parametrize(
    "model", [MultiOutputClassifier(LogisticRegression(max_iter=500)), RandomForestClassifier(5, max_depth=3, random_state=0)], ids=["multi-output", "forest"]
)
def test_a_classifier_of_several_outputs_is_refused_with_why(table, model):
    frame, _answer, _fresh = table
    front = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "plan"]), ("num", StandardScaler(), ["age"])])
    pipeline = Pipeline([("pre", front), ("clf", model)]).fit(frame, tags_of(frame))
    with pytest.raises(StructureError) as caught:
        read(pipeline, frame)
    assert "of its 3 outputs on its own" in caught.value.message
