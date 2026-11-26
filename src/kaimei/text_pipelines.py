"""Text classifiers made of a bag of words and a model, read exactly.

scikit-learn's ``TfidfVectorizer`` (or a ``CountVectorizer``, with a ``TfidfTransformer`` after it) followed by a model
is the most common text classifier there is. ``skl2onnx`` writes the vectorizer with string operators that nothing here
can check, so Kaimei reads such a pipeline itself:

* the words are found as scikit-learn finds them, by a Kaimei ``word`` tokenizer with scikit-learn's token pattern, its
  lower-casing, its accent stripping (``strip_accents="unicode"``) and its stop words, so the API can split a text the
  same way;
* ONNX's ``TfIdfVectorizer`` counts them, n-grams included (scikit-learn forms n-grams from neighbouring words, and a
  word it does not know still stands between its neighbours, which is why unknown words get an id of their own rather
  than being dropped);
* the counts are made binary or sub-linear, weighted by the idf and normalised as the vectorizer (and a
  ``TfidfTransformer`` after it) does;
* a linear model or a naive Bayes scores them with its own coefficients (a ``predict_proba`` where the model has one,
  else its decision scores). Any other model, and any steps between the counts and the model (``TruncatedSVD``,
  ``SelectKBest``, a scaler), is converted by ``skl2onnx`` from the table of numbers and joined on.

Before that, Kaimei's tokenizer and the vectorizer both split the caller's texts and Kaimei's sample texts, and a
pipeline whose words they find differently is refused. The check holds the pipeline's own answers: on the caller's
texts as they are written, and on made-up rows of token ids, each given to scikit-learn as the text its ids spell.

A bag of words counts every word of a text, so a text cut to fit is another text, with another answer. The graph
reads a fixed number of token positions (``max_length``, or :data:`DEFAULT_TEXT_TOKENS` and more when the caller's
texts are longer), and the text feature says ``"longer": "refuse"``: the API refuses a text of more tokens with an
error that says how many it has and how many are read, where it would cut a transformer's.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

#: Token positions a text is read for when nothing says how many (``max_length``, or texts longer than half of it):
#: room for a document of several pages. A position costs the API next to nothing: what a reading costs to run is
#: set by its vocabulary.
DEFAULT_TEXT_TOKENS = 4096
#: The most token positions chosen here, for a caller whose own texts are that long: the most a text feature may
#: have on the API. A longer text is refused by the API rather than cut.
MAX_TEXT_TOKENS = 32_768
#: A word no vocabulary holds, spelling an unknown token for scikit-learn: it stands between its neighbours.
_UNKNOWN_WORD = "kaimeiunknownword"
#: The caller's texts the token length is chosen from and the splitting is compared on.
_MOST_TEXTS = 200
#: The caller's texts the check holds as they are written, at most.
_CHECK_TEXTS = 24
#: Token positions the rows of a check made here hold between them, about (a row is as long as a text may be).
_CHECK_VALUES = 64_000
#: Scores closer than this are level: the class between them is decided by rounding.
_LEVEL = 1e-12
#: scikit-learn's default token pattern, the one Kaimei's ``sklearn`` word pattern is.
_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
#: What a refusal suggests when Kaimei cannot split text as the pipeline does.
_ELSEWHERE = "or register the model with an agent (kaimei.agent) to run it where it is"


def _final(pipeline: Any) -> Any:
    steps = getattr(pipeline, "steps", None)
    return steps[-1][1] if isinstance(steps, list) and steps else pipeline


def text_parts(estimator: Any) -> Optional[Tuple[Any, Optional[Any], Any]]:
    """``(vectorizer, tf-idf transformer or None, the rest of the pipeline)`` when ``estimator`` is a bag-of-words
    pipeline: a ``CountVectorizer`` or ``TfidfVectorizer`` first, a model last, and any steps between.

    A pipeline that starts with a ``HashingVectorizer`` is one too. No route reads it (it has no vocabulary to make
    a tokenizer of), and it is taken here to be refused with that reason."""
    steps = getattr(estimator, "steps", None)
    if not isinstance(steps, list) or len(steps) < 2:
        return None
    try:
        from sklearn.feature_extraction.text import (
            CountVectorizer,
            HashingVectorizer,
            TfidfTransformer,
        )
        from sklearn.pipeline import Pipeline
    except ImportError:
        return None
    first = steps[0][1]
    if not isinstance(first, (CountVectorizer, HashingVectorizer)):  # TfidfVectorizer is a CountVectorizer
        return None
    after = [(name, step) for name, step in steps[1:] if step is not None and not isinstance(step, str)]
    transformer = None
    if len(after) > 1 and isinstance(after[0][1], TfidfTransformer):
        transformer, after = after[0][1], after[1:]
    if not after:
        return None
    return first, transformer, after[0][1] if len(after) == 1 else Pipeline(after)


def _refuse(why: List[str], reason: str) -> Optional[Dict[str, Any]]:
    """Say why the pipeline is not read here; ``None`` for the caller to return."""
    why.append(reason)
    return None


def _unsupported(vectorizer: Any) -> Optional[str]:
    """Why Kaimei's word tokenizer cannot find this vectorizer's words, or ``None`` when it can."""
    if not hasattr(vectorizer, "vocabulary_") and type(vectorizer).__name__ == "HashingVectorizer":
        return (
            "its HashingVectorizer hashes each word to a column and keeps no vocabulary, and a Kaimei tokenizer is made "
            f"of a vocabulary; use a CountVectorizer or a TfidfVectorizer, {_ELSEWHERE}"
        )
    analyzer = getattr(vectorizer, "analyzer", "word")
    if callable(analyzer):
        return f"its vectorizer finds words with an analyzer of its own, which Kaimei cannot run; use analyzer='word', {_ELSEWHERE}"
    if analyzer != "word":
        return (
            f"its vectorizer counts runs of characters (analyzer={analyzer!r}), and Kaimei's word tokenizer finds whole "
            f"words; use analyzer='word', {_ELSEWHERE}"
        )
    if getattr(vectorizer, "input", "content") != "content":
        return (
            f"its vectorizer reads {vectorizer.input}s (input={vectorizer.input!r}), and Kaimei gives it the text itself; "
            "use input='content'"
        )
    if callable(getattr(vectorizer, "tokenizer", None)) or getattr(vectorizer, "preprocessor", None) is not None:
        return f"its vectorizer splits text with a tokenizer or a preprocessor of its own, which Kaimei cannot run; {_ELSEWHERE}"
    if str(getattr(vectorizer, "token_pattern", "")) != _TOKEN_PATTERN:
        return "its vectorizer has a token_pattern of its own; Kaimei splits text with scikit-learn's default"
    accents = getattr(vectorizer, "strip_accents", None)
    if accents == "ascii":
        return (
            "its vectorizer strips accents with strip_accents='ascii', which also deletes every character that is not "
            "ASCII (\u201cdon\u2019t\u201d becomes \u201cdont\u201d), and Kaimei's word tokenizer keeps them; refit it with "
            f"strip_accents='unicode' (which strips the same accents and keeps other characters), {_ELSEWHERE}"
        )
    if accents not in (None, "unicode"):
        return "its vectorizer strips accents with a function of its own"
    return None


def _texts(rows: Any, feature_names: Optional[Sequence[str]]) -> List[str]:
    """The caller's texts, from any one-dimensional sequence of them: a list, a tuple, a pandas Series, a NumPy array,
    or a table of one column (or with the column ``feature_names`` names). What is not text (a gap) is left out."""
    if rows is None:
        return []
    if isinstance(rows, str):
        rows = [rows]
    columns = getattr(rows, "columns", None)
    if columns is not None and hasattr(rows, "iloc"):
        listed = [str(column) for column in columns]
        wanted = str(feature_names[0]) if feature_names else None
        if wanted is not None and wanted in listed:
            rows = rows.iloc[:, listed.index(wanted)]
        elif len(listed) == 1:
            rows = rows.iloc[:, 0]
        else:
            return []
    shape = getattr(rows, "shape", None)
    if shape is not None and len(shape) == 2 and shape[1] == 1:
        rows = (row[0] for row in rows)
    elif shape is not None and len(shape) != 1:
        return []
    try:
        items = list(itertools.islice(iter(rows), _MOST_TEXTS))
    except TypeError:
        return []
    return [str(item) for item in items if isinstance(item, str)]


def _shown(text: str, limit: int = 60) -> str:
    return repr(text if len(text) <= limit else text[: limit - 1] + "\u2026")


def _split_differently(vectorizer: Any, splitter: Any, texts: Sequence[str]) -> Optional[str]:
    """Why Kaimei's tokenizer would read some text differently from the vectorizer, or ``None`` when it would not.

    Both split the caller's texts and Kaimei's sample texts (accents, other scripts, curly quotes, long words): the
    words the vectorizer keeps, each the unknown token where the vocabulary lacks it, must be the tokens Kaimei finds.
    """
    from .text import SAMPLE_TEXTS
    from .tokenizers import MAX_TEXT_CHARS

    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    stops = vectorizer.get_stop_words()
    for text in [*texts, *SAMPLE_TEXTS]:
        if len(text) > MAX_TEXT_CHARS:
            continue  # a text that long is not split by Kaimei at all: the API refuses it for this reading
        words = [word for word in tokenize(preprocess(vectorizer.decode(text))) if not stops or word not in stops]
        theirs = [word if word in splitter.ids else "<unk>" for word in words]
        ours = [splitter.tokens[index] for index in splitter.encode(text)]
        if theirs != ours:
            return (
                f"Kaimei's word tokenizer would read {_shown(text)} as {_shown(' '.join(ours))}, where its vectorizer reads "
                f"{_shown(' '.join(theirs))}, so it would not read texts as this pipeline does; {_ELSEWHERE}"
            )
    return None


def _found_in_texts(vectorizer: Any) -> Callable[[str], bool]:
    """Whether the vectorizer can find ``word`` in a text at all.

    A vocabulary of one's own may hold a word no text gives: one with a capital under ``lowercase=True``, a stop
    word, a single letter, one with an accent that is stripped. The vectorizer changes or drops such a word when it
    meets it, so its count is always 0, and Kaimei's tokenizer must not know it either. A word is found exactly
    when the vectorizer, given the word alone, gives the word back: what it finds in any text it also finds so.
    """
    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    stops = vectorizer.get_stop_words()

    def found(word: str) -> bool:
        return [item for item in tokenize(preprocess(word)) if not stops or item not in stops] == [word]

    return found


def _made_up(vectorizer: Any, word_id: Dict[str, int], unknown: int, length: int, numpy: Any) -> List[List[int]]:
    """Short made-up texts as token ids: a few of the vocabulary's own terms (so its n-grams are met, and a term
    twice), with a word it does not know between some. The longer made-up rows of the check hold thousands of words
    drawn at random, in which one term weighs next to nothing."""
    generator = numpy.random.default_rng(0)
    terms = [
        [word_id[part] for part in str(term).split(" ")]
        for term in sorted(str(term) for term in vectorizer.vocabulary_)
        if all(part in word_id for part in str(term).split(" "))
    ]
    rows: List[List[int]] = []
    for size in (1, 2, 3, 5, 8, 13, 21, 34):
        ids: List[int] = []
        picked = [terms[int(at)] for at in generator.integers(0, len(terms), size=size)] if terms else []
        for at, term in enumerate(picked):
            ids += term
            if at % 4 == 2:
                ids += [unknown]  # a word it does not know stands between its neighbours
        if size >= 3 and picked:
            ids += picked[0]  # a term a second time: a count above one
        rows.append((ids or [unknown])[:length])
    return rows


def _even_split(estimator: Any, text: str, numpy: Any) -> bool:
    """Whether the pipeline's own scores for ``text`` leave its class to rounding: a margin of nothing, or its two
    best classes level, to within what double precision can tell apart. ``False`` for a pipeline with no scores."""
    for name in ("decision_function", "predict_proba"):
        try:
            method = getattr(estimator, name, None)
            scores = numpy.asarray(method([text]), dtype=float).reshape(-1) if callable(method) else None
        except Exception:  # noqa: BLE001 - a method that does not apply to this model: the next one is asked
            scores = None
        if scores is None or not scores.size or not numpy.isfinite(scores).all():
            continue
        if scores.size == 1:
            return bool(abs(float(scores[0])) <= _LEVEL)
        if name == "predict_proba" and bool(numpy.any(numpy.abs(scores - 0.5) <= _LEVEL)):
            return True  # a label of a multi-label model, or one of two classes, at an even chance
        best = numpy.sort(scores)[-2:]
        return bool(float(best[1] - best[0]) <= _LEVEL * max(1.0, float(numpy.abs(best).max())))
    return False


def _stages(vectorizer: Any, transformer: Any, numpy: Any) -> List[Dict[str, Any]]:
    """How counts become the pipeline's numbers: each tf-idf step in turn (the vectorizer's own, then a
    ``TfidfTransformer`` after it), with its sub-linear scaling, idf weights and norm."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    stages: List[Dict[str, Any]] = []
    for source in (vectorizer if isinstance(vectorizer, TfidfVectorizer) else None, transformer):
        if source is None:
            continue
        # use_idf=False leaves out the weights alone: the sub-linear scaling and the norm still apply.
        stages.append(
            {
                "sublinear": bool(getattr(source, "sublinear_tf", False)),
                "idf": numpy.asarray(source.idf_, dtype=float) if getattr(source, "use_idf", False) else None,
                "norm": getattr(source, "norm", None),
            }
        )
    return stages


def _dense(value: Any) -> Any:
    """``value`` as a plain array: a model fitted on a sparse table (an SVC, an SVR) or made sparse keeps its weights
    as a sparse matrix, which NumPy cannot read as numbers."""
    return value.toarray() if hasattr(value, "toarray") else value


def _scorer(final: Any, numpy: Any) -> Optional[Tuple[str, Any, Any, str]]:
    """``(kind, weights [features, outputs], bias [outputs], ending)`` of a linear model or naive Bayes, or ``None``
    for any other model: ``skl2onnx`` converts that one."""
    name = type(final).__name__
    if hasattr(final, "feature_log_prob_") and name in ("MultinomialNB", "ComplementNB"):
        weights = numpy.asarray(_dense(final.feature_log_prob_), dtype=float).T
        prior = numpy.asarray(final.class_log_prior_, dtype=float) if name == "MultinomialNB" else numpy.zeros(weights.shape[1])
        return "proba", weights, prior, "softmax"
    try:
        found = getattr(final, "coef_", None)  # an SVC with another kernel raises from the attribute itself
    except Exception:  # noqa: BLE001 - whatever it raises, it has no table of weights to read
        found = None
    if found is None:
        return None
    classes = getattr(final, "classes_", None)
    if hasattr(final, "dual_coef_") and classes is not None and len(classes) > 2:
        # An SVC of several classes keeps a row of weights for each pair of them, and its scores for each class are
        # made from the pairs' votes: not a linear model of the counts.
        return None
    try:
        coef = numpy.atleast_2d(numpy.asarray(_dense(found), dtype=float))
        bias = numpy.atleast_1d(numpy.asarray(_dense(getattr(final, "intercept_", 0.0)), dtype=float))
    except (TypeError, ValueError):
        return None  # not a table of numbers
    if classes is None:  # a regressor
        return "predict", coef.T, numpy.broadcast_to(bias, (coef.shape[0],)).copy(), "none"
    probabilistic = callable(getattr(final, "predict_proba", None)) and name in ("LogisticRegression", "LogisticRegressionCV")
    if probabilistic:
        return "proba", coef.T, bias, "sigmoid_pair" if coef.shape[0] == 1 else "softmax"
    return "decision", coef.T, bias, "none"


def _counting(
    vectorizer: Any, stages: List[Dict[str, Any]], words: List[str], precision: Any, numpy: Any
) -> Tuple[List[Any], List[Any], str, Callable[..., str]]:
    """The nodes that turn token ids into the vectorizer's numbers: ``(nodes, constants, output, constant)``."""
    from onnx import TensorProto, helper, numpy_helper

    word_id = {word: index for index, word in enumerate(words)}
    low, high = (int(item) for item in getattr(vectorizer, "ngram_range", (1, 1)))
    by_length: Dict[int, List[Tuple[List[int], int]]] = {size: [] for size in range(1, high + 1)}
    never = len(words) + 2  # ids no text is split into: the unknown token and the padding come before them
    for term, column in dict(vectorizer.vocabulary_).items():
        parts = str(term).split(" ")
        if low <= len(parts) <= high and all(part in word_id for part in parts):
            by_length[len(parts)].append(([word_id[part] for part in parts], int(column)))
        else:
            # A term of a vocabulary of one's own that the vectorizer never forms (it is of another length, or holds
            # a word the vectorizer finds in no text, see _found_in_texts): its count is always 0.
            by_length[low].append(([never] * low, int(column)))
            never += 1
    pool: List[int] = []
    counts: List[int] = []
    indexes: List[int] = []
    for size in range(1, high + 1):
        counts.append(len(pool))  # where the n-grams of this length start in the pool
        for ids, column in sorted(by_length[size], key=lambda item: item[1]):
            pool.extend(ids)
            indexes.append(column)
    nodes = [
        helper.make_node(
            "TfIdfVectorizer", ["ids"], ["counts"], name="vectorizer/count", mode="TF",
            min_gram_length=low, max_gram_length=high, max_skip_count=0,
            ngram_counts=counts, ngram_indexes=indexes, pool_int64s=pool,
        )
    ]
    constants: List[Any] = []

    def constant(name: str, value: Any, dtype: Any = precision) -> str:
        constants.append(numpy_helper.from_array(numpy.asarray(value, dtype=dtype), name))
        return name

    current = "counts"
    if numpy.dtype(precision) == numpy.float64:
        nodes.append(helper.make_node("Cast", [current], ["counts_double"], name="vectorizer/double", to=TensorProto.DOUBLE))
        current = "counts_double"
    if bool(getattr(vectorizer, "binary", False)):
        nodes.append(helper.make_node("Min", [current, constant("one", [1.0])], ["binary"], name="vectorizer/binary"))
        current = "binary"
    for index, stage in enumerate(stages):
        tag = "" if index == 0 else f"_{index}"
        path = "vectorizer" if index == 0 else "tfidf"
        if stage["sublinear"]:
            nodes += [
                helper.make_node("Log", [current], [f"log_tf{tag}"], name=f"{path}/log"),
                helper.make_node("Add", [f"log_tf{tag}", constant(f"one_s{tag}", [1.0])], [f"log_tf1{tag}"], name=f"{path}/plus_one"),
                helper.make_node("Greater", [current, constant(f"zero{tag}", [0.0])], [f"present{tag}"], name=f"{path}/present"),
                helper.make_node("Where", [f"present{tag}", f"log_tf1{tag}", constant(f"zero_w{tag}", [0.0])], [f"sublinear{tag}"], name=f"{path}/sublinear"),
            ]
            current = f"sublinear{tag}"
        if stage["idf"] is not None:
            nodes.append(helper.make_node("Mul", [current, constant(f"idf{tag}", stage["idf"])], [f"weighted{tag}"], name=f"{path}/idf"))
            current = f"weighted{tag}"
        if stage["norm"] in ("l2", "l1"):
            axis = constant(f"axis{tag}", [1], numpy.int64)
            if stage["norm"] == "l2":
                nodes += [
                    helper.make_node("Mul", [current, current], [f"squares{tag}"], name=f"{path}/square"),
                    helper.make_node("ReduceSum", [f"squares{tag}", axis], [f"total{tag}"], name=f"{path}/sum", keepdims=1),
                    helper.make_node("Sqrt", [f"total{tag}"], [f"size{tag}"], name=f"{path}/norm"),
                ]
            else:
                nodes += [
                    helper.make_node("Abs", [current], [f"sizes{tag}"], name=f"{path}/abs"),
                    helper.make_node("ReduceSum", [f"sizes{tag}", axis], [f"size{tag}"], name=f"{path}/norm", keepdims=1),
                ]
            nodes += [
                helper.make_node("Max", [f"size{tag}", constant(f"tiny{tag}", [1e-30])], [f"safe{tag}"], name=f"{path}/safe"),
                helper.make_node("Div", [current, f"safe{tag}"], [f"normalized{tag}"], name=f"{path}/normalize"),
            ]
            current = f"normalized{tag}"
    return nodes, constants, current, constant


def _model(nodes: List[Any], constants: List[Any], output: str, width: int, length: int, precision: Any, opset: int, numpy: Any) -> Any:
    import onnx
    from onnx import TensorProto, helper

    real = TensorProto.DOUBLE if numpy.dtype(precision) == numpy.float64 else TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        "text_pipeline",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, length])],
        [helper.make_tensor_value_info(output, real, [1, width])],
        constants,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", int(opset))], producer_name="kaimei")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _graph(
    vectorizer: Any, stages: List[Dict[str, Any]], scorer: Tuple[str, Any, Any, str], words: List[str], length: int, precision: Any, numpy: Any
) -> Any:
    """The pipeline as an ONNX model reading token ids ``[1, length]``: the counts, scored by a linear model."""
    from onnx import helper

    nodes, constants, current, constant = _counting(vectorizer, stages, words, precision, numpy)
    _kind, matrix, bias, ending = scorer
    nodes.append(helper.make_node("Gemm", [current, constant("coef", matrix), constant("bias", bias)], ["scores"], name="model/linear"))
    output = "scores"
    if ending == "softmax":
        nodes.append(helper.make_node("Softmax", ["scores"], ["probabilities"], name="model/softmax", axis=-1))
        output = "probabilities"
    elif ending == "sigmoid_pair":
        nodes += [
            helper.make_node("Sigmoid", ["scores"], ["positive"], name="model/sigmoid"),
            helper.make_node("Sub", [constant("one_p", [1.0]), "positive"], ["negative"], name="model/complement"),
            helper.make_node("Concat", ["negative", "positive"], ["probabilities"], name="model/pair", axis=-1),
        ]
        output = "probabilities"
    width = 2 if ending == "sigmoid_pair" else int(numpy.asarray(matrix).shape[1])
    return _model(nodes, constants, output, width, length, precision, 17, numpy)


def _converted(
    vectorizer: Any,
    stages: List[Dict[str, Any]],
    rest: Any,
    words: List[str],
    length: int,
    precision: Any,
    numpy: Any,
    *,
    classifier: bool,
    margin: Optional[Callable[[Dict[str, Any]], Any]],
    raw_scores: bool = False,
) -> Tuple[Any, Optional[str]]:
    """The counts joined to ``skl2onnx``'s conversion of the rest of the pipeline: ``(model, output to read)``.

    ``margin``, for a two-class classifier that gives one margin, runs the rest of the pipeline's
    ``decision_function`` on a row of counts: the conversion's column that is that margin is read. ``raw_scores``
    asks the conversion for a classifier's scores where it would give its probabilities.
    """
    from onnx import compose

    from . import graphs

    final = _final(rest)
    columns = len(vectorizer.vocabulary_)
    tools = graphs._importer()
    back = tools.load_model(graphs._convert_sklearn(rest, final, columns, precision, classifier, numpy, raw_scores=raw_scores))
    read_from: Optional[str] = None
    if margin is not None:
        # One margin, where the conversion gives two scores: the column that is the margin is the output.
        back, read_from = graphs._one_margin(tools, back, margin, columns, precision, numpy)
    opset = max([17, *[int(item.version) for item in back.opset_import if item.domain in ("", "ai.onnx")]])
    nodes, constants, current, _constant = _counting(vectorizer, stages, words, precision, numpy)
    front = _model(nodes, constants, current, columns, length, precision, opset, numpy)
    back_input = str(back.graph.input[0].name)
    back = compose.add_prefix(back, prefix="model/")
    front.ir_version = back.ir_version = max(int(front.ir_version), int(back.ir_version))
    joined = compose.merge_models(front, back, io_map=[(current, f"model/{back_input}")])
    return joined, f"model/{read_from}" if read_from else None


def through_graph(
    estimator: Any,
    why: List[str],
    numpy: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    check_rows: Any = None,
    labels: Optional[Sequence[Any]] = None,
    framework: Optional[str] = None,
    name: Optional[str] = None,
    max_length: Optional[int] = None,
    mask_layout: Optional[str] = None,
    threshold: Optional[float] = None,
    keep: Optional[Dict[str, Any]] = None,
    **_unused: Any,
) -> Optional[Dict[str, Any]]:
    """A text pipeline (see the module docstring) as a ``family: "graph"`` spec, or ``None`` with the reason in ``why``."""
    parts = text_parts(estimator)
    if parts is None:
        return _refuse(why, "it is not a bag-of-words text pipeline")
    vectorizer, transformer, rest = parts
    reason = _unsupported(vectorizer)
    if reason is not None:
        return _refuse(why, reason)
    final = _final(rest)
    if not callable(getattr(final, "predict", None)):
        return _refuse(why, f"its last step, {type(final).__name__}, is not a model: it has no predict to give the counts to")
    from .graphs import (
        _cluster_answer,
        _is_clusterer,
        _is_regressor,
        _multilabel,
        _NoConversion,
        _pairwise,
        _per_output,
        _read_onnx,
        _sklearn_answer,
    )
    from .structure import StructureError
    from .tabular_pipelines import (
        _PAIRS_NOT_CLASSES,
        _per_output_reason,
        _scores_note,
        _settle_class,
        _task_of,
    )
    from .tokenizers import MAX_TEXT_CHARS, MAX_TOKEN_CHARS, Tokenizer, TokenizerError

    # The words a text can give: one the vectorizer finds in no text (see _found_in_texts) is left out, and the
    # terms that hold it count nothing, as they do in scikit-learn.
    found_in_texts = _found_in_texts(vectorizer)
    words = sorted(word for word in {part for term in vectorizer.vocabulary_ for part in str(term).split(" ")} if found_in_texts(word))
    longest_word = max(words, key=len, default="")
    if len(longest_word) > MAX_TOKEN_CHARS:
        return _refuse(
            why,
            f"its vocabulary holds a word of {len(longest_word):,} characters ({_shown(longest_word, 40)}), and a Kaimei "
            f"tokenizer holds words of at most {MAX_TOKEN_CHARS}; fit it on a vocabulary without such words",
        )
    stops = sorted(str(word) for word in (vectorizer.get_stop_words() or []))
    tokens = [*words, "<unk>", "<pad>"]
    tokenizer = {
        "type": "word",
        "pattern": "sklearn",
        "lowercase": bool(getattr(vectorizer, "lowercase", True)),
        "strip_accents": getattr(vectorizer, "strip_accents", None) == "unicode",
        "stop_words": stops,
        "vocab": tokens,
        "unk": "<unk>",
        "pad": "<pad>",
    }
    try:
        splitter = Tokenizer(tokenizer)
    except TokenizerError as exc:
        return _refuse(why, f"its words cannot be a Kaimei tokenizer's: {exc}")
    texts = _texts(check_rows, feature_names)
    differ = _split_differently(vectorizer, splitter, texts)
    if differ is not None:
        return _refuse(why, differ)
    # A text of more characters than Kaimei splits is refused by the API: it is not one this reading is asked about.
    texts = [text for text in texts if len(text) <= MAX_TEXT_CHARS]
    encoded = [splitter.encode(text) for text in texts]
    longest = max((len(ids) for ids in encoded), default=0)
    # Room for a long document, and for twice the caller's longest text: a longer one is refused, never cut.
    length = int(max_length) if max_length is not None else min(MAX_TEXT_TOKENS, max(DEFAULT_TEXT_TOKENS, 2 * longest))
    pad = len(tokens) - 1
    # A row of the check is as many numbers as the graph reads positions, most of them padding: the longer a text
    # may be, the fewer rows the check holds, so that it stays a small part of what is uploaded.
    room = max(4, _CHECK_VALUES // length)
    made_up = max(2, min(8, room // 3))
    # The caller's texts the check holds as written: each whose tokens fit (an empty one is refused by the API),
    # the longest of them first, so that a long text read whole is among the few a long reading has room for.
    written: Dict[Tuple[int, ...], str] = {}
    fitting = [at for at, ids in enumerate(encoded) if ids and len(ids) <= length]
    for at in [*sorted(fitting, key=lambda at: -len(encoded[at]))[:1], *fitting]:
        if len(written) < max(2, min(_CHECK_TEXTS, room - made_up)):
            written.setdefault(tuple(encoded[at]), texts[at])
    spelled = [*words, _UNKNOWN_WORD]

    def text_of(feeds: Dict[str, Any]) -> str:
        ids = [int(item) for item in numpy.asarray(feeds["ids"]).reshape(-1)]
        while ids and ids[-1] == pad:
            ids.pop()  # the padding after the last token
        found = written.get(tuple(ids))
        if found is not None:
            return found  # one of the caller's texts, as it was written: scikit-learn splits it itself
        # A made-up row, spelled out. A padding id before the end stands between its neighbours as an unknown word
        # does, which is how the graph counts it.
        return " ".join(spelled[item] if 0 <= item < len(words) else _UNKNOWN_WORD for item in ids)

    def ids_of(ids: List[int]) -> Any:
        return numpy.asarray([ids + [pad] * (length - len(ids))], dtype=numpy.int64)

    # Short made-up texts first and last, the caller's own between them. One the pipeline itself cannot decide (its
    # two best classes an even split to within rounding, as words that cancel give a model fitted on as many texts
    # of each class) is left out: the class of such a text is rounding's, in scikit-learn and in Kaimei alike.
    made = [
        ids_of(ids)
        for ids in _made_up(vectorizer, {word: index for index, word in enumerate(words)}, len(words), length, numpy)
        if not _even_split(estimator, text_of({"ids": ids}), numpy)
    ][:made_up]
    samples = [*made[:4], *[ids_of(list(ids)) for ids in written], *made[4:]] or [ids_of([0])]
    found = getattr(final, "classes_", None)
    if found is not None and _per_output(found):
        return _refuse(why, _per_output_reason(found))
    classes = list(found) if found is not None else []
    classifier = len(classes) >= 2 and not _is_regressor(final)
    # A yes or a no for each label (fitted on a table of them): its probabilities are each label's own.
    multilabel = classifier and _multilabel(rest, final, len(vectorizer.vocabulary_), None, numpy)
    clusterer = not classifier and _is_clusterer(final)
    scorer = _scorer(final, numpy) if rest is final else None
    stages = _stages(vectorizer, transformer, numpy)
    # The caller's names may be an array (an encoder's classes_), which has no truth value of its own.
    given_labels = list(labels) if labels is not None else []
    named = given_labels or ([str(item.item() if hasattr(item, "item") else item) for item in classes] if classifier else None)
    task = _task_of(task, classes, classifier, multilabel, clusterer)
    feature = str(feature_names[0]) if feature_names else "text"

    def margin_of(feeds: Dict[str, Any]) -> Any:
        return rest.decision_function(numpy.asarray(next(iter(feeds.values())), dtype=float).reshape(1, -1))

    failures: List[str] = []

    def read(route: str, kind: str, method: Callable[[Any], Any], raw_scores: bool = False) -> Optional[Tuple[Dict[str, Any], Any, str]]:
        """``(spec, the ONNX model it was read from, route)`` of the pipeline read by one of its methods (``kind`` is
        its name): in single precision, or in double where single only comes close. ``None`` when it cannot be read
        this way, with the reason in ``failures``. ``raw_scores`` asks skl2onnx for a classifier's scores where it
        would give its probabilities."""

        def library(feeds: Dict[str, Any]) -> Any:
            return numpy.asarray(method([text_of(feeds)]), dtype=float).reshape(1, -1)

        kept: Optional[Tuple[Dict[str, Any], Any, str]] = None
        for precision in (numpy.float32, numpy.float64):
            if precision is numpy.float64 and kept is None:
                # Doubles mend a reading single precision came close to on this route, not one it could not make;
                # and skl2onnx never returns from converting in doubles a tree with an infinite threshold.
                break
            try:
                if route == "linear" and scorer is not None:
                    model, read_from = _graph(vectorizer, stages, scorer, words, length, precision, numpy), None
                else:
                    one = classifier and kind == "decision_function" and len(classes) == 2 and not multilabel
                    model, read_from = _converted(
                        vectorizer, stages, rest, words, length, precision, numpy,
                        classifier=classifier, margin=margin_of if one else None, raw_scores=raw_scores,
                    )
            except _NoConversion as exc:
                failures.append(f"skl2onnx could not convert {_named(rest)} ({exc}); replace that step, {_ELSEWHERE}")
                continue
            except Exception as exc:  # noqa: BLE001 - joining a conversion on can fail in any way ONNX can
                failures.append(f"{_named(rest)} could not be joined to the counts ({type(exc).__name__}: {str(exc)[:160]})")
                continue
            try:
                spec = _read_onnx(
                    model,
                    library=library,
                    feature_names=None,
                    task=task,
                    labels=named,
                    inputs={"ids": {"feature": feature, "part": "ids", "pad_id": pad, "vocab": len(tokens)}},
                    input_shapes=None,
                    samples={"ids": samples[0]},
                    output=read_from,
                    mask_layout=mask_layout,
                    threshold=threshold,
                    name=name,
                    tokenizer=tokenizer,
                    check_rows=None,
                    framework=framework or "sklearn",
                    more_samples=[{"ids": sample} for sample in samples[1:]],
                    # predict_proba's numbers are the probabilities: shown as they are, never through a softmax again.
                    probabilities=classifier and kind == "predict_proba",
                )
            except StructureError as exc:
                if exc.param in ("labels", "inputs", "output", "tokenizer", "threshold"):
                    raise  # the caller's own argument: another route would not make it right
                failures.append(exc.message)
                continue
            for item in spec.get("features") or []:
                if str(item.get("name")) == feature and str(item.get("type")) in ("text", "tokens"):
                    # Every word is counted: the API refuses a text of more tokens than are read, where it would
                    # cut one for a model that reads the start of a text.
                    item["longer"] = "refuse"
            if kept is None or not spec.get("approximate"):
                kept = (spec, model, route)
            if not spec.get("approximate"):
                break
        return kept

    def own_class(row: Mapping[str, Any]) -> Any:
        """What the pipeline's own ``predict`` gives one row of a check: its token ids, as the text they spell."""
        ids = [int(value) if value >= 0 else pad for value in numpy.asarray(row[feature], dtype=float).reshape(-1)]
        return estimator.predict([text_of({"ids": ids})])

    def read_again(kind: str) -> Optional[Tuple[Dict[str, Any], Any, str]]:
        """The pipeline read by another of its methods, as skl2onnx converts the model after the counts."""
        other = getattr(estimator, kind, None)
        return read("converted", kind, other, True) if callable(other) else None

    best: Optional[Tuple[Dict[str, Any], Any, str, str]] = None
    for route in (["linear"] if scorer is not None else []) + ["converted"]:
        if route == "linear" and scorer is not None:
            kind = {"proba": "predict_proba", "decision": "decision_function", "predict": "predict"}[scorer[0]]
            method: Callable[[Any], Any] = getattr(estimator, kind)
        else:
            found, kind = _cluster_answer(estimator) if clusterer else _sklearn_answer(estimator, classifier)
            if found is None:
                failures.append(f"its last step, {type(final).__name__}, has no predict_proba, decision_function or predict")
                continue
            if classifier and kind == "decision_function" and len(classes) > 2 and _pairwise(final):
                # Three classes have three pairs: the scores would fit a head of three classes, and be read as them.
                failures.append(_PAIRS_NOT_CLASSES)
                continue
            method = found
        got = read(route, kind, method)
        if got is None:
            continue
        if classifier:
            got, kind = _settle_class(got, kind, final, classes, own_class, read_again)
        if best is None or (best[0].get("approximate") and not got[0].get("approximate")):
            best = (got[0], got[1], got[2], kind)
        if not best[0].get("approximate"):
            break
    if best is None:
        return _refuse(why, "; ".join(dict.fromkeys(failures)) or "it could not be read")
    spec, model, route, kind = best
    if keep is not None:
        keep["onnx"] = bytes(model.SerializeToString())
    steps = len(spec["graph"].get("nodes") or [])
    grams = ", ".join(str(n) for n in range(int(vectorizer.ngram_range[0]), int(vectorizer.ngram_range[1]) + 1))
    if route == "linear":
        how = f"{type(final).__name__} read as a graph of {steps:,} operations: words found as scikit-learn finds them, counted ({grams}-grams), weighted and scored."
    else:
        how = (
            f"{_named(rest)} read as a graph of {steps:,} operations: words found as scikit-learn finds them, counted "
            f"({grams}-grams) and weighted by Kaimei, then given to {_named(rest)} as skl2onnx converts it."
        )
    spec["summary"] = f"scikit-learn {type(vectorizer).__name__} and {how} {spec.get('summary') or ''}".strip()
    spec.setdefault("metadata", {})["library_method"] = kind
    if kind == "decision_function" and classifier:
        spec["summary"] += _scores_note(multilabel)
    if not spec.get("approximate"):
        spec["summary"] += _limits_note(length)
    return spec


def _limits_note(length: int) -> str:
    """What the summary says of the texts this reading is not given: longer ones, and ones with no words."""
    return (
        f" It counts every word of a text, so a text of more than {length:,} words it finds is refused, not cut (pass "
        "max_length= to read longer ones). A text in which it finds no word at all (no run of two or more letters or "
        "digits that is not a stop word) is refused too; scikit-learn answers such a text from the model's intercept "
        "alone."
    )


def _named(rest: Any) -> str:
    """The rest of a pipeline by its steps' classes: ``TruncatedSVD and LogisticRegression``."""
    steps = getattr(rest, "steps", None)
    names = [type(step).__name__ for _name, step in steps] if isinstance(steps, list) else [type(rest).__name__]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


__all__ = ["DEFAULT_TEXT_TOKENS", "MAX_TEXT_TOKENS", "text_parts", "through_graph"]
