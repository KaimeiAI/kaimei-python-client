"""Fixes to the scikit-learn pipelines Kaimei reads itself, from the third review: texts longer than a model reads,
pipelines fitted on arrays, values far from zero, the class Kaimei reports, and what is refused or said plainly.

As in ``test_pipeline_fixes.py``, a pipeline that is read is also run where its check did not look: the ONNX model
the reading was taken from is given rows and texts the check did not hold, as the API gives them, and must give
the pipeline's own answers. (The same readings are run on the API's own engine in the review's notes.)"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")
pd = pytest.importorskip("pandas")

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.feature_extraction.text import (  # noqa: E402
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import (  # noqa: E402
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
)
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC, SVR  # noqa: E402

from kaimei import StructureError, graphs, text_pipelines  # noqa: E402
from kaimei.tokenizers import Tokenizer  # noqa: E402
from test_pipeline_fixes import (  # noqa: E402
    agrees,
    kept,
    read,
    runs,
    shown,
    table_feeds,
    text_feeds,
)

WORDS = "good great fine lovely bad awful poor worst film plot story acting music really very quite".split()


def reviews(count, seed, longest=30):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(int(rng.integers(4, longest)))) for _ in range(count)]


TEXTS = reviews(120, 0)
TWO = np.array([int(sum(word in text.split() for word in WORDS[:4]) > sum(word in text.split() for word in WORDS[4:8])) for text in TEXTS])
THREE = np.arange(len(TEXTS)) % 3
UNSEEN = reviews(30, 1) + ["zebra quokka", "Good GOOD good, bad!", "a superb, superb ending; dull start"]
DAY = 1_700_000_000


def frame(rows, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "plan": rng.choice(["free", "pro", "team"], rows, p=[0.5, 0.3, 0.2]),
            "ts": DAY + rng.uniform(0, 86_400, rows),  # a day of seconds, far from zero, with fractions
            "age": rng.normal(40, 12, rows).round(1),
        }
    )


TABLE, FRESH = frame(300, 0), frame(80, 1)
HOUR = ((TABLE.ts - DAY) / 3600.0).to_numpy()
LATE = (TABLE.ts > TABLE.ts.median()).astype(int).to_numpy()
THIRD = np.digitize(HOUR, [8, 16])


def columns(*parts, remainder="drop"):
    return ColumnTransformer(list(parts), remainder=remainder)


def reported(extraction, feeds):
    """What Kaimei reports for each row: its class's position, read off the kept model's output by the API's rule."""
    session, output = kept(extraction)
    link = extraction.spec["head"].get("link") or "none"
    scores = np.stack([np.asarray(session.run([output], feed)[0], dtype=float).reshape(-1) for feed in feeds])
    return [int(graphs.reported_class(extraction.spec, row)) for row in shown(scores, link)]


def same_class(extraction, pipeline, given, feeds):
    """Kaimei's class for every row is the class the pipeline's own predict() gives it."""
    classes = [str(item) for item in pipeline.classes_]
    ours = [classes[at] for at in reported(extraction, feeds)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = [str(item) for item in pipeline.predict(given)]
    assert ours == theirs


# --- a text longer than the model reads ----------------------------------------------------------------------------


def test_a_bag_of_words_reads_a_long_text_whole_and_says_a_longer_one_is_refused():
    """Read for 32 tokens and cut to them, this review of 100 words was called class 1 where scikit-learn says 0."""
    texts = ["good film great plot", "bad film poor plot", "great acting good story", "poor acting bad story", "fine film",
             "awful plot really bad", "not good at all", "really great", "worst film awful", "great great film"] * 5
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, [1, 0, 1, 0, 1, 0, 0, 1, 0, 1] * 5)
    extraction = read(pipeline, texts)
    feature = extraction.spec["features"][0]
    assert runs(extraction.spec)
    assert feature["max_length"] == text_pipelines.DEFAULT_TEXT_TOKENS == 4096
    # The API refuses a text of more tokens than that with an error, where it would cut one for another model.
    assert feature["longer"] == "refuse"
    assert "a text of more than 4,096 words it finds is refused, not cut" in extraction.spec["summary"]
    long_review = "great film " * 20 + "bad awful worst " * 20
    for unseen in ([long_review], reviews(6, 3, longest=900)):
        agrees(extraction, pipeline, unseen, text_feeds(extraction, unseen))
        same_class(extraction, pipeline, unseen, text_feeds(extraction, unseen))


def test_room_for_a_text_grows_with_the_callers_own_and_max_length_is_kept_as_given():
    longer = reviews(4, 5, longest=40) + [" ".join(np.random.default_rng(6).choice(WORDS, size=2500))]
    pipeline = make_pipeline(CountVectorizer(), LogisticRegression()).fit(TEXTS, TWO)
    grown = read(pipeline, longer).spec
    assert runs(grown) and grown["features"][0]["max_length"] == 5000 and grown["features"][0]["longer"] == "refuse"
    # A row of the check is as long as a text may be: the longer that is, the fewer rows, and the long text is one.
    rows = grown["check"]["inputs"]["text"]
    assert len(rows) * 5000 <= 2 * text_pipelines._CHECK_VALUES
    assert any(sum(value != rows[0][-1] for value in row) == 2500 for row in rows)
    asked = read(pipeline, longer, max_length=64).spec
    assert runs(asked) and asked["features"][0]["max_length"] == 64 and asked["features"][0]["longer"] == "refuse"
    assert "more than 64 words" in asked["summary"]


def test_the_check_holds_short_made_up_texts_however_long_a_text_may_be():
    """Four thousand words drawn at random say little of any one term: a few of the vocabulary's own terms say more."""
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression()).fit(TEXTS, TWO)
    spec = read(pipeline, None).spec
    assert runs(spec)
    pad = len(spec["tokenizers"]["text"]["vocab"]) - 1
    lengths = sorted(sum(value != pad for value in row) for row in spec["check"]["inputs"]["text"])
    assert sum(length <= 80 for length in lengths) >= 4 and lengths[-1] > 1000


# --- a pipeline fitted on an array ---------------------------------------------------------------------------------

ARRAY = TABLE[["ts", "age", "plan"]].to_numpy(dtype=object)
ARRAY_FRESH = FRESH[["ts", "age", "plan"]].to_numpy(dtype=object)


def test_a_pipeline_fitted_on_an_array_says_which_column_each_feature_is_read_from():
    """Its features are listed in the order its parts read columns, and the columns it drops are left out."""
    front = ColumnTransformer([("c", OneHotEncoder(), [2])], remainder=StandardScaler())
    spec = read(Pipeline([("pre", front), ("m", LogisticRegression(max_iter=2000))]).fit(ARRAY, LATE), ARRAY).spec
    assert runs(spec) and spec["feature_names"] == ["x2", "x0", "x1"]
    assert spec["metadata"]["table_columns"] == [2, 0, 1]
    some = Pipeline([("pre", ColumnTransformer([("n", StandardScaler(), [1, 0])])), ("m", LinearRegression())]).fit(ARRAY, HOUR)
    spec = read(some, ARRAY).spec
    assert runs(spec) and spec["feature_names"] == ["x1", "x0"] and spec["metadata"]["table_columns"] == [1, 0]
    # A pipeline fitted on a table reads its columns by name: there is no position to say.
    named = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"])), LinearRegression())
    assert "table_columns" not in read(named.fit(TABLE, HOUR), TABLE).spec["metadata"]


def test_feature_names_name_the_columns_of_a_pipeline_fitted_on_an_array():
    front = ColumnTransformer([("c", OneHotEncoder(), [2]), ("n", StandardScaler(), [0])])
    pipeline = Pipeline([("pre", front), ("m", LogisticRegression(max_iter=2000))]).fit(ARRAY, LATE)
    extraction = read(pipeline, ARRAY, feature_names=["when", "age", "plan"])
    spec = extraction.spec
    assert runs(spec) and spec["feature_names"] == ["plan", "when"] and spec["metadata"]["table_columns"] == [2, 0]
    assert {item["name"]: item["type"] for item in spec["features"]} == {"plan": "categorical", "when": "numeric"}
    named = pd.DataFrame(ARRAY_FRESH, columns=["when", "age", "plan"]).astype({"when": float})
    agrees(extraction, pipeline, ARRAY_FRESH, table_feeds(extraction, named))
    # A table that has those names is read by them, whatever order its columns are in.
    shuffled = pd.DataFrame(ARRAY, columns=["when", "age", "plan"]).astype({"when": float})[["plan", "age", "when"]]
    by_name = read(pipeline, shuffled, feature_names=["when", "age", "plan"]).spec
    assert runs(by_name) and len(by_name["check"]["outputs"]) > 40
    wrong_names = (
        (["when", "plan"], "feature_names has 2 names, and this pipeline was fitted on a table of 3 columns"),
        (["when", "when", "plan"], "feature_names gives two columns the same name"),
    )
    for wrong, said in wrong_names:
        with pytest.raises(StructureError) as caught:
            read(pipeline, ARRAY, feature_names=wrong)
        assert said in caught.value.message and "one name for each column of that table" in caught.value.message
    # A pipeline fitted on a table keeps the table's names.
    table = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"])), LinearRegression())
    assert read(table.fit(TABLE, HOUR), TABLE, feature_names=["a", "b", "c"]).spec["feature_names"] == ["plan", "ts"]


# --- pipelines skl2onnx converts whole -----------------------------------------------------------------------------


def test_a_front_this_module_does_not_write_down_is_converted_whole_when_its_columns_are_by_position():
    rng = np.random.default_rng(5)
    rows, fresh = rng.normal(size=(300, 3)), rng.normal(size=(60, 3))
    answer = (rows[:, 0] + rows[:, 1] > 0).astype(int)
    twice = ColumnTransformer([("a", StandardScaler(), [0, 1]), ("b", SimpleImputer(), [1, 2])])
    pipeline = Pipeline([("pre", twice), ("m", LogisticRegression())]).fit(rows, answer)
    extraction = read(pipeline, rows)
    assert runs(extraction.spec) and extraction.spec["feature_names"] == ["x0", "x1", "x2"]
    assert extraction.spec["summary"].startswith("scikit-learn LogisticRegression converted to ONNX")
    session, output = kept(extraction)
    ours = np.stack([session.run([output], {session.get_inputs()[0].name: row.reshape(1, -1).astype(np.float32)})[0].reshape(-1) for row in fresh])
    np.testing.assert_allclose(ours, pipeline.predict_proba(fresh), rtol=1e-4, atol=1e-5)
    # An imputer that adds columns saying what was missing is not written down either: no refusal, skl2onnx's reading.
    gappy = rows.copy()
    gappy[rng.random(rows.shape) < 0.1] = np.nan
    flagged = Pipeline([("pre", ColumnTransformer([("a", SimpleImputer(add_indicator=True), [0, 1, 2])])), ("m", LogisticRegression())])
    assert read(flagged.fit(gappy, answer), rows).spec["family"] == "graph"


def test_chosen_by_name_such_a_front_is_still_refused_with_this_modules_reason():
    twice = columns(("a", StandardScaler(), ["ts", "age"]), ("b", SimpleImputer(), ["age"]))
    with pytest.raises(StructureError) as caught:
        read(make_pipeline(twice, LinearRegression()).fit(TABLE, HOUR), TABLE)
    assert "read by two of its parts" in caught.value.message


# --- an SVC that scores pairs --------------------------------------------------------------------------------------


@pytest.mark.parametrize("classes", [3, 4])
def test_an_svc_that_scores_each_pair_of_classes_is_refused_behind_a_pipeline_too(classes):
    """Three pairs of three classes fitted a head of three and were read as the classes; four classes gave an error
    about ``labels`` nobody passed."""
    text = make_pipeline(TfidfVectorizer(), SVC(decision_function_shape="ovo")).fit(TEXTS, np.arange(len(TEXTS)) % classes)
    table = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"])), SVC(decision_function_shape="ovo"))
    table.fit(TABLE, np.digitize(HOUR, np.linspace(0, 24, classes + 1)[1:-1]))
    for pipeline, rows in ((text, TEXTS), (table, TABLE)):
        with pytest.raises(StructureError) as caught:
            read(pipeline, rows)
        assert "each pair of classes" in caught.value.message and "decision_function_shape='ovr'" in caught.value.message
        assert caught.value.param is None


def test_an_svc_that_counts_votes_says_so_behind_a_pipeline_too():
    text = make_pipeline(TfidfVectorizer(), SVC()).fit(TEXTS, THREE)
    table = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"])), SVC()).fit(TABLE, THIRD)
    for pipeline, rows in ((text, TEXTS), (table, TABLE)):
        with pytest.warns(UserWarning, match="one-vs-one"):
            from kaimei.structure import extract_structure

            spec = extract_structure(pipeline, check_rows=rows).spec
        if not spec.get("approximate"):  # where a check row is a tie of votes it is drawn, and says so instead
            assert spec["metadata"]["predicted_class"] == graphs.VOTES_NOTE
            assert "one-vs-one votes tie" in spec["summary"]
        else:
            assert graphs.NOT_ITS_CLASS in spec["summary"]


# --- values far from zero ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("given", ["all", "five", "none"])
def test_a_time_in_seconds_is_read_as_the_double_it_is(given):
    """Read in single precision, 1.7e9 is rounded to the nearest 128 before the scaler sees it: a regression on a time
    of day was 0.0115 off on answers of 0 to 10 and called exact, and a steep classifier changed class."""
    rows = {"all": TABLE, "five": TABLE.head(5), "none": None}[given]
    scaled = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"])), LinearRegression()).fit(TABLE, HOUR)
    steep = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"])), LogisticRegression(C=1000, max_iter=5000))
    steep.fit(TABLE, LATE)
    for pipeline in (scaled, steep):
        extraction = read(pipeline, rows)
        assert runs(extraction.spec)
        session, _output = kept(extraction)
        assert {item.name: item.type for item in session.get_inputs()}["ts"] == "tensor(double)"
        agrees(extraction, pipeline, FRESH, table_feeds(extraction, FRESH))
    same_class(extraction, steep, FRESH, table_feeds(extraction, FRESH))
    # The pipeline was asked about numbers single precision cannot hold, in the range of the data it was fitted on.
    held = [row[0] for row in extraction.spec["check"]["inputs"]["ts"]]
    assert all(DAY - 86_400 < value < DAY + 2 * 86_400 for value in held)
    assert any(value != float(np.float32(value)) for value in held)


def test_a_column_passed_through_to_a_linear_model_takes_double_precision_all_the_way():
    passed = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", "passthrough", ["ts"])), LinearRegression()).fit(TABLE, HOUR)
    extraction = read(passed, TABLE)
    assert runs(extraction.spec) and extraction.spec["metadata"]["model_precision"] == "double"
    assert "computed in double precision" in extraction.spec["summary"]
    session, output = kept(extraction)
    ours = np.array([session.run([output], feed)[0].reshape(-1)[0] for feed in table_feeds(extraction, FRESH)])
    np.testing.assert_allclose(ours, passed.predict(FRESH), rtol=0, atol=1e-6)
    # Scaled first, single precision gives the pipeline's numbers on rows from its data, and is what is kept.
    scaled = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", MinMaxScaler(), ["ts"])), LinearRegression()).fit(TABLE, HOUR)
    assert read(scaled, None).spec["metadata"]["model_precision"] == "single"


def test_with_no_rows_and_no_statistics_the_model_is_computed_as_it_computes_itself_and_the_summary_says_so():
    """Nothing says where a column passed through lies. A linear model tells doubles apart and is read in doubles; a
    tree does not (scikit-learn rounds its inputs to single precision itself) and is read in single."""
    front = columns(("c", OneHotEncoder(), ["plan"]), ("n", "passthrough", ["ts"]))
    line = make_pipeline(front, LinearRegression()).fit(TABLE, HOUR)
    trees = make_pipeline(front, GradientBoostingRegressor(n_estimators=20, random_state=0)).fit(TABLE, HOUR)
    for pipeline, precision in ((line, "double"), (trees, "single")):
        extraction = read(pipeline, None)
        spec = extraction.spec
        assert runs(spec) and spec["metadata"]["model_precision"] == precision
        assert "Nothing in the pipeline says where the values of ts lie" in spec["summary"]
        assert "give rows of the table" in spec["summary"]
        session, output = kept(extraction)
        ours = np.array([session.run([output], feed)[0].reshape(-1)[0] for feed in table_feeds(extraction, FRESH)])
        np.testing.assert_allclose(ours, pipeline.predict(FRESH), rtol=0, atol=1e-5)


# --- a linear SVC or SVR after TF-IDF ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, target",
    [(SVC(kernel="linear"), TWO), (SVC(kernel="linear"), THREE), (SVR(kernel="linear"), TWO.astype(float)), (LogisticRegression(), TWO)],
    ids=["svc", "svc-three-classes", "svr", "sparsified"],
)
def test_weights_kept_as_a_sparse_matrix_are_read(model, target):
    """Fitted on a sparse table, a linear SVC or SVR keeps ``coef_`` sparse: reading it raised, and nothing else was tried."""
    pipeline = make_pipeline(TfidfVectorizer(), model).fit(TEXTS, target)
    if isinstance(model, LogisticRegression):
        pipeline[-1].sparsify()
    extraction = read(pipeline, TEXTS)
    assert runs(extraction.spec)
    agrees(extraction, pipeline, UNSEEN, text_feeds(extraction, UNSEEN))


# --- the class Kaimei reports against predict() --------------------------------------------------------------------


def test_a_threshold_of_the_models_own_is_carried_through_a_pipeline():
    from sklearn.model_selection import TunedThresholdClassifierCV

    tuned = TunedThresholdClassifierCV(LogisticRegression(max_iter=2000), scoring="recall")
    table = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts", "age"])), tuned).fit(TABLE, LATE)
    text = make_pipeline(TfidfVectorizer(), TunedThresholdClassifierCV(LogisticRegression(), scoring="recall")).fit(TEXTS, TWO)
    for pipeline, rows, fresh, feeds in ((table, TABLE, FRESH, table_feeds), (text, TEXTS, UNSEEN, text_feeds)):
        extraction = read(pipeline, rows)
        assert runs(extraction.spec)
        assert extraction.spec["outputs"]["threshold"] == pytest.approx(pipeline[-1].best_threshold_)
        same_class(extraction, pipeline, fresh, feeds(extraction, fresh))


def test_probabilities_that_are_not_what_predict_goes_by_give_way_to_the_scores():
    """A modified-Huber SGD classifier predicts by its scores; the most probable class of its predict_proba is
    another on some rows. Its scores are read, so Kaimei's class is the model's own."""
    huber = SGDClassifier(loss="modified_huber", random_state=0)
    pipeline = make_pipeline(columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts", "age"])), huber).fit(TABLE, THIRD)
    extraction = read(pipeline, TABLE)
    assert runs(extraction.spec) and extraction.spec["metadata"]["library_method"] == "decision_function"
    same_class(extraction, pipeline, FRESH, table_feeds(extraction, FRESH))


@pytest.mark.parametrize("kind", ["text", "table"])
def test_a_reading_whose_class_is_not_predicts_is_drawn_and_says_on_how_many_rows(kind):
    """Its scores are the pipeline's, and still it is not the pipeline: predict() answers something else."""
    if kind == "text":
        pipeline, rows = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(TEXTS, TWO), TEXTS
    else:
        front = columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"]))
        pipeline, rows = make_pipeline(front, LogisticRegression(max_iter=2000)).fit(TABLE, LATE), TABLE
    assert runs(read(pipeline, rows).spec)
    pipeline[-1].predict = lambda X: np.zeros(X.shape[0], dtype=int)  # a rule of its own, as a subclass may have
    spec = read(pipeline, rows).spec
    assert spec.get("approximate") and "check" not in spec
    assert "differ from the model's own predict() on" in spec["summary"] and "rows it was checked on" in spec["summary"]
    assert spec["metadata"]["predicted_class"].startswith("Not the model's own on")


def test_class_names_given_as_an_array_name_a_pipelines_classes():
    """``labels=encoder.classes_`` is an array, which has no truth value: both readers asked ``if labels``."""
    from kaimei import tabular_pipelines

    front = columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["ts"]))
    table = make_pipeline(front, LogisticRegression(max_iter=2000)).fit(TABLE, LATE)
    text = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(TEXTS, TWO)
    for reader, pipeline, rows in ((tabular_pipelines, table, TABLE), (text_pipelines, text, TEXTS)):
        spec = reader.through_graph(pipeline, [], np, check_rows=rows, labels=np.array(["early", "late"]))
        assert spec is not None and spec["outputs"]["labels"] == ["early", "late"] and not spec.get("approximate")


# --- limits, fixed or said plainly ---------------------------------------------------------------------------------


def test_a_vocabulary_word_no_text_gives_counts_nothing_and_the_reading_stays_exact():
    """``Good`` under ``lowercase=True``, a stop word and a single letter are never found by the vectorizer. Made-up
    texts used to spell them, scikit-learn read other words, and the reading was called approximate."""
    own = ["good", "bad", "great film", "Good", "the", "a", "plot", "really bad"]
    vectorizer = CountVectorizer(vocabulary=own, ngram_range=(1, 2), stop_words=["the"])
    pipeline = make_pipeline(vectorizer, LogisticRegression()).fit(TEXTS, TWO)
    extraction = read(pipeline, TEXTS)
    assert runs(extraction.spec)
    assert set(extraction.spec["tokenizers"]["text"]["vocab"]) == {"good", "bad", "great", "film", "plot", "really", "<unk>", "<pad>"}
    unseen = UNSEEN + ["Good the a plot", "the great film was really bad"]
    agrees(extraction, pipeline, unseen, text_feeds(extraction, unseen))


def test_a_hashing_vectorizer_is_refused_by_name():
    pipeline = make_pipeline(HashingVectorizer(n_features=64), LogisticRegression()).fit(TEXTS, TWO)
    with pytest.raises(StructureError) as caught:
        read(pipeline, TEXTS)
    assert "HashingVectorizer" in caught.value.message and "keeps no vocabulary" in caught.value.message


def test_summaries_say_what_is_refused_where_scikit_learn_answers():
    text = read(make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(TEXTS, TWO), TEXTS).spec
    assert "A text in which it finds no word at all" in text["summary"] and "is refused too" in text["summary"]
    # The API refuses an empty text: what the tokenizer makes of "?" or "a b c" is nothing.
    assert Tokenizer(text["tokenizers"]["text"]).encode("? a b c") == []
    lenient = columns(("c", OneHotEncoder(handle_unknown="ignore"), ["plan"]), ("n", StandardScaler(), ["age"]))
    table = read(make_pipeline(lenient, LogisticRegression()).fit(TABLE, LATE), TABLE).spec
    assert "A value of plan that its encoder never saw is refused" in table["summary"]
    strict = columns(("c", OneHotEncoder(), ["plan"]), ("n", StandardScaler(), ["age"]))
    assert "never saw" not in read(make_pipeline(strict, LogisticRegression()).fit(TABLE, LATE), TABLE).spec["summary"]
