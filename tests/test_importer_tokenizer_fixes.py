"""The ONNX importer and the tokenizers, as the review and the audit found them wrong.

* A model that defines its own operators (model-local functions) was refused: ``import onnx`` does not load
  ``onnx.inliner``. Shown in a fresh interpreter, where nothing else has loaded it.
* A node in a branch, a loop's body or a function whose last inputs are listed under empty names stopped the
  process inside ONNX Runtime (an opset-20 ``DFT``). Shown in a fresh interpreter, so a crash fails the test.
* Byte-pair merges were made in stretches of 256 bytes, so a long run with no break (a Chinese sentence, a
  long number) split differently from Hugging Face's, and so did merge tables Hugging Face reads otherwise.
* A word was split into a token the owner added that the WordPiece vocabulary does not hold, and WordPiece
  words longer than 256 characters were one unknown token whatever the tokenizer said.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import pytest

import kaimei
from kaimei.text import SAMPLE_TEXTS, tokenizer_spec
from kaimei.tokenizers import Tokenizer

SOURCE = str(Path(kaimei.__file__).resolve().parents[1])


def _fresh(code: str, timeout: float = 240) -> str:
    """Run ``code`` in a new interpreter that imports this checkout's kaimei; its output, once it exits cleanly."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SOURCE + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUTF8", "1")
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=timeout, env=env
    )
    assert done.returncode == 0, f"exit code {done.returncode}\n{done.stdout[-2000:]}\n{done.stderr[-3000:]}"
    return done.stdout


# --- model-local functions ---------------------------------------------------------------------------------


_FUNCTION_MODEL = """
    import sys
    import warnings

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    warnings.simplefilter("ignore")
    twice = helper.make_function(
        "local", "Twice", ["a"], ["b"], [helper.make_node("Add", ["a", "a"], ["b"])], [helper.make_opsetid("", 17)]
    )
    weight = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0, "w")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["h"], name="lin"),
            helper.make_node("Twice", ["h"], ["y"], domain="local", name="twice"),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])],
        [weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("local", 1)], functions=[twice]
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
"""


def test_a_model_that_defines_its_own_operators_is_read_in_a_fresh_interpreter():
    pytest.importorskip("onnx")
    pytest.importorskip("numpy")
    out = _fresh(
        _FUNCTION_MODEL
        + """
    from kaimei import onnx_graph, onnx_reader

    print("inliner loaded before:", "onnx.inliner" in sys.modules)
    raw = onnx_reader.read_model(model)
    print("read:", [node["op"] for node in raw["nodes"]])
    spec = onnx_graph.import_onnx(model, feature_names=["a", "b", "c", "d"], task="regression")
    row = np.asarray([spec["check"]["inputs"][name][0][0] for name in "abcd"], dtype=np.float32)
    expected = 2 * (row @ (np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0))
    print("imported:", [node["op"] for node in spec["graph"]["nodes"]])
    print("check agrees:", bool(np.allclose(spec["check"]["outputs"][0], expected, atol=1e-5)))
    """
    )
    assert "inliner loaded before: False" in out  # what made the bug: nothing had loaded it
    assert "read: ['MatMul', 'Add']" in out
    assert "imported: ['Linear', 'Add']" in out and "check agrees: True" in out


def test_without_an_inliner_the_refusal_says_what_to_install(monkeypatch):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("numpy")
    from onnx import TensorProto, helper

    from kaimei import onnx_graph, onnx_reader

    function = helper.make_function(
        "local", "Same", ["a"], ["b"], [helper.make_node("Identity", ["a"], ["b"])], [helper.make_opsetid("", 17)]
    )
    graph = helper.make_graph(
        [helper.make_node("Same", ["x"], ["y"], domain="local")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("local", 1)], functions=[function]
    )
    # An onnx package from before onnx.inliner: the submodule cannot be imported.
    monkeypatch.delattr(onnx, "inliner", raising=False)
    monkeypatch.setitem(sys.modules, "onnx.inliner", None)
    with pytest.raises(onnx_reader.OnnxReadError, match="pip install -U onnx"):
        onnx_reader.read_model(model)
    with pytest.raises(onnx_graph.OnnxImportError, match="no onnx.inliner"):
        onnx_graph.import_onnx(model, feature_names=["a", "b"], task="regression")


# --- inputs left out at the end of a nested node -----------------------------------------------------------


_NESTED_DFT = """
    import sys
    import warnings

    import numpy as np
    from onnx import TensorProto, helper

    warnings.simplefilter("ignore")
    which = sys.argv[1]


    def dft(name, source="x"):
        return helper.make_node("DFT", [source, "", ""], [name], name=name + "_dft")  # dft_length and axis left out


    functions = []
    opsets = [helper.make_opsetid("", 20)]
    initializers = []
    if which == "if":
        def branch(name):
            out = helper.make_tensor_value_info(name + "_out", TensorProto.FLOAT, [1, 8, 2])
            return helper.make_graph([dft(name + "_out")], name, [], [out])

        nodes = [
            helper.make_node("ReduceSum", ["x"], ["total"], keepdims=0),
            helper.make_node("Greater", ["total", "zero"], ["positive"]),
            helper.make_node("If", ["positive"], ["y"], then_branch=branch("then"), else_branch=branch("other")),
        ]
        initializers = [helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])]
    elif which == "loop":
        body = helper.make_graph(
            [dft("spectrum"), helper.make_node("Identity", ["cond_in"], ["cond_out"])],
            "body",
            [
                helper.make_tensor_value_info("i", TensorProto.INT64, []),
                helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                helper.make_tensor_value_info("carried", TensorProto.FLOAT, [1, 8, 2]),
            ],
            [
                helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
                helper.make_tensor_value_info("spectrum", TensorProto.FLOAT, [1, 8, 2]),
            ],
        )
        nodes = [helper.make_node("Loop", ["trips", "", "start"], ["y"], body=body)]
        initializers = [
            helper.make_tensor("trips", TensorProto.INT64, [], [2]),
            helper.make_tensor("start", TensorProto.FLOAT, [1, 8, 2], [0.0] * 16),
        ]
    elif which == "scan":
        body = helper.make_graph(
            [
                helper.make_node("Unsqueeze", ["step", "axis0"], ["batched"]),
                dft("spectrum", "batched"),
                helper.make_node("Squeeze", ["spectrum", "axis0"], ["out_step"]),
            ],
            "body",
            [helper.make_tensor_value_info("step", TensorProto.FLOAT, [8, 1])],
            [helper.make_tensor_value_info("out_step", TensorProto.FLOAT, [8, 2])],
            [helper.make_tensor("axis0", TensorProto.INT64, [1], [0])],
        )
        nodes = [helper.make_node("Scan", ["x"], ["y"], body=body, num_scan_inputs=1)]
    else:  # a function of the file's own, run as it is written (not expanded first)
        spectrum = helper.make_function(
            "local", "Spectrum", ["a"], ["b"], [helper.make_node("DFT", ["a", "", ""], ["b"])], [helper.make_opsetid("", 20)]
        )
        functions = [spectrum]
        opsets.append(helper.make_opsetid("local", 1))
        nodes = [helper.make_node("Spectrum", ["x"], ["y"], domain="local")]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=opsets, functions=functions)
    model.ir_version = 9
    feeds = {"x": np.arange(8, dtype=np.float32).reshape(1, 8, 1)}
    spectrum = np.fft.fft(feeds["x"][0, :, 0])  # the DFT along its default axis, -2
    expected = np.stack([spectrum.real, spectrum.imag], axis=-1).reshape(1, 8, 2)

    from kaimei import onnx_graph

    found = onnx_graph.run_reference(model, feeds)["y"]
    print("same as the reference:", bool(np.allclose(found, expected, atol=1e-4)))
    if which != "function":
        spec = onnx_graph.import_onnx(model, task="generation")
        print("imported:", len(spec["check"]["outputs"]), "check rows")
"""


@pytest.mark.parametrize("where", ["if", "loop", "scan", "function"])
def test_a_nested_node_with_inputs_left_out_at_its_end_does_not_stop_onnx_runtime(where):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("numpy")
    code = "import sys\nsys.argv[1:] = [" + repr(where) + "]\n" + textwrap.dedent(_NESTED_DFT)
    out = _fresh(code)
    assert "same as the reference: True" in out
    if where != "function":
        assert "imported: 4 check rows" in out


def test_every_nested_list_of_inputs_loses_its_empty_end_and_the_model_given_is_left_as_it_is():
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from kaimei import onnx_graph

    def clip(name):
        return helper.make_node("Clip", ["x", "", ""], [name])

    deeper = helper.make_node(
        "If",
        ["c"],
        ["deeper"],
        then_branch=helper.make_graph([clip("d1")], "t", [], []),
        else_branch=helper.make_graph([clip("d2")], "e", [], []),
    )
    branch = helper.make_graph([clip("inner"), deeper], "branch", [], [])
    function = helper.make_function("local", "F", ["x"], ["f"], [clip("f")], [helper.make_opsetid("", 17)])
    graph = helper.make_graph(
        [clip("top"), helper.make_node("If", ["c"], ["y"], then_branch=branch, else_branch=branch)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, functions=[function])
    stripped = onnx_graph._without_trailing_empties(model)
    lists = [list(node.input) for body in onnx_graph._bodies(stripped) for node in body.node]
    # top and its If, the function's Clip, and in each of the two branches a Clip, an If and its two Clips
    assert len(lists) == 11 and lists.count(["x"]) == 8
    assert not any(names and names[-1] == "" for names in lists)
    assert stripped is not model and list(model.graph.node[0].input) == ["x", "", ""]  # the caller's is not touched
    assert onnx_graph._without_trailing_empties(stripped) is stripped  # nothing to take off: no copy is made


def test_a_model_that_draws_and_lists_inputs_left_out_is_recorded_with_the_evaluators_draws():
    """ONNX's evaluator could not run the node with its empty inputs, so ONNX Runtime's draws were recorded: the
    check then held the model to numbers no other engine draws."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    from onnx import TensorProto, helper
    from onnx.reference import ReferenceEvaluator

    from kaimei import onnx_graph

    nodes = [
        helper.make_node("RandomNormalLike", ["x"], ["noise"], seed=1.0, scale=0.1),
        helper.make_node("Add", ["x", "noise"], ["noisy"]),
        helper.make_node("DFT", ["noisy", "", ""], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    feeds = {"x": np.arange(8, dtype=np.float32).reshape(1, 8, 1)}
    written_out = onnx.ModelProto()
    written_out.CopyFrom(model)
    del written_out.graph.node[2].input[1:]
    expected = ReferenceEvaluator(written_out).run(None, feeds, intermediate=True)
    found = onnx_graph.run_reference(model, feeds)
    assert np.allclose(found["noise"], expected["noise"]) and np.allclose(found["y"], expected["y"], atol=1e-4)


def _branch_reads(model_kind):
    """A model whose If branches read a tensor that a rewrite at the top would otherwise take away."""
    np = pytest.importorskip("numpy")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    initializers = [numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "zero")]
    if model_kind == "identity":  # x -> Identity -> x2, which only the branches read
        reads, top = ["x2"], [helper.make_node("Identity", ["x"], ["x2"])]
    else:  # MatMul then Add, a Linear step, with the MatMul's own output read in the branches as well
        reads = ["h", "y1"]
        top = [helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Add", ["h", "b"], ["y1"])]
        initializers += [
            numpy_helper.from_array(rng.normal(size=(4, 4)).astype(np.float32), "w"),
            numpy_helper.from_array(rng.normal(size=4).astype(np.float32), "b"),
        ]

    def branch(name, op):
        out = helper.make_tensor_value_info(name + "_out", TensorProto.FLOAT, [1, 4])
        return helper.make_graph([helper.make_node(op, reads, [name + "_out"])], name, [], [out])

    nodes = [
        *top,
        helper.make_node("ReduceSum", ["x"], ["total"], keepdims=0),
        helper.make_node("Greater", ["total", "zero"], ["positive"]),
        helper.make_node(
            "If",
            ["positive"],
            ["y"],
            then_branch=branch("then", "Relu" if model_kind == "identity" else "Add"),
            else_branch=branch("other", "Neg" if model_kind == "identity" else "Sub"),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


@pytest.mark.parametrize("model_kind", ["identity", "linear"])
def test_a_tensor_only_a_branch_reads_is_not_taken_away_by_a_rewrite(model_kind):
    """An Identity whose output only a branch read was dropped, and a MatMul fused into a Linear step though a
    branch read its output: the branch then read a tensor nothing made, and the API refused the model."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from kaimei import onnx_graph

    graph = onnx_graph.import_onnx(_branch_reads(model_kind), task="generation")["graph"]
    made = {item["name"] for item in graph["inputs"]} | set(graph["constants"])
    made |= {name for node in graph["nodes"] for name in node["outputs"] if name}
    bodies = [value["graph"] for node in graph["nodes"] for value in node["attrs"].values() if isinstance(value, dict) and "graph" in value]
    assert len(bodies) == 2
    for body in bodies:
        local = {name for node in body["nodes"] for name in node["outputs"]} | {item["name"] for item in body["inputs"]}
        outside = {name for node in body["nodes"] for name in node["inputs"] if name and name not in local}
        assert outside and outside <= made, outside - made


# --- byte-pair encoding ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf():
    return pytest.importorskip("tokenizers")


def _byte_level_bpe(hf, tokens, merges, alphabet=True, **model):
    """A Hugging Face byte-level BPE tokenizer from its tokenizer.json, as GPT-2's is written: every byte (unless
    ``alphabet`` is false), then ``tokens``, and ``merges``."""
    from tokenizers import pre_tokenizers

    vocab = sorted(pre_tokenizers.ByteLevel.alphabet()) if alphabet else []
    vocab += [token for token in tokens if token not in vocab]
    document = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {token: index for index, token in enumerate(vocab)},
            "merges": [list(pair) for pair in merges],
            **model,
        },
    }
    return hf.Tokenizer.from_str(json.dumps(document))


def _exact(original, samples=()):
    """The translation, which must pass the client's check without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        described = tokenizer_spec(original, samples=list(samples))
    assert described is not None and described["type"] in ("bpe", "wordpiece")
    return described


def _same(original, described, texts):
    ours = Tokenizer(described)
    for text in texts:
        assert ours.encode(text) == original.encode(text).ids, (text[:60], original.encode(text).tokens[:20])


def test_a_run_longer_than_256_bytes_is_merged_whole(hf):
    """``ab`` doubled seven times over: 400 bytes of ``ab`` merge into 3 tokens, where stretches of 256 gave 4."""
    tokens = ["a", "b"] + ["ab" * 2**power for power in range(8)]
    merges = [("a", "b")] + [("ab" * 2**power, "ab" * 2**power) for power in range(7)]
    original = _byte_level_bpe(hf, tokens, merges)
    word = "ab" * 200
    assert original.encode(word).tokens == ["ab" * 128, "ab" * 64, "ab" * 8]
    described = _exact(original, samples=[word])
    _same(original, described, [word, "ab" * 129, "ab" * 1000, "b" + "ab" * 300, "ab" * 128 + " " + "ab" * 128])


def test_merges_are_made_one_at_a_time_lowest_rank_and_leftmost_first(hf):
    """Two merges make ``abc``; Hugging Face's queue joins ``abc`` and ``a`` before the second ``abc`` forms."""
    tokens = ["a", "b", "c", "bc", "ab", "abc", "abca"]
    merges = [("b", "c"), ("a", "b"), ("ab", "c"), ("abc", "a"), ("a", "bc")]
    original = _byte_level_bpe(hf, tokens, merges)
    assert original.encode("abcabc").tokens == ["abca", "bc"]  # merging every place at once gives abc, abc
    described = _exact(original, samples=["abcabc"])
    _same(original, described, ["abcabc", "abcabcabc", "aabcabcc", "cabcab"])


def test_a_merge_listed_twice_has_the_rank_of_its_last_listing(hf):
    """As a tokenizer.json lists it: Hugging Face's own copy writes each merge once, at the rank it took."""
    original = _byte_level_bpe(hf, ["a", "b", "c", "ab", "bc"], [("a", "b"), ("b", "c"), ("a", "b")])
    assert original.encode("abc").tokens == ["a", "bc"]
    document = json.loads(original.to_str())
    document["model"]["merges"] = [["a", "b"], ["b", "c"], ["a", "b"]]
    for given in (original, document):
        described = _exact(given)
        _same(original, described, ["abc", "abcabc", "ab", "bcab"])


def test_hugging_faces_bpe_options_are_followed(hf):
    """A character the vocabulary lacks is left out (its neighbours may merge), is the unknown token, one
    unknown token for a run of them, or its bytes; and a word the vocabulary holds whole can be taken whole."""
    tokens = ["a", "b", "c", "ab", "bc", "abc", "<unk>"] + [f"<0x{byte:02X}>" for byte in range(256)]
    merges = [("a", "b"), ("b", "c")]
    cases = {
        "left out": ({}, "axb", ["ab"]),
        "unknown token": ({"unk_token": "<unk>"}, "axyb", ["a", "<unk>", "<unk>", "b"]),
        "fused": ({"unk_token": "<unk>", "fuse_unk": True}, "axyb", ["a", "<unk>", "b"]),
        "bytes": ({"byte_fallback": True}, "a!b", ["a", "<0x21>", "b"]),
        "whole": ({"ignore_merges": True}, "abc", ["abc"]),
    }
    for label, (options, text, expected) in cases.items():
        original = _byte_level_bpe(hf, tokens, merges, alphabet=False, **options)
        assert original.encode(text).tokens == expected, label
        described = _exact(original, samples=[text])
        _same(original, described, [text, "abc abc", "xx ab xx", "a!!b", "c?", "abcabc", "q"])


def test_a_trained_tokenizer_missing_bytes_is_exact_with_each_option(hf):
    """As trained on a little English, without the whole byte alphabet: the sample texts write bytes it lacks."""
    from tokenizers import models, pre_tokenizers, trainers

    made = hf.Tokenizer(models.BPE())
    made.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["The quick brown fox jumps over the lazy dog.", "tokenization tokenizers 123 4567"] * 20
    made.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>"]))
    document = json.loads(made.to_str())
    texts = [*SAMPLE_TEXTS, "the" + "é" * 3 + "fox", "x你y", "token😀ization", "Ā ÿ \x01"]
    for options in ({}, {"unk_token": "<unk>"}, {"unk_token": "<unk>", "fuse_unk": True}):
        changed = json.loads(json.dumps(document))
        changed["model"].update(options)
        original = hf.Tokenizer.from_str(json.dumps(changed))
        _same(original, _exact(original), texts)


def test_a_long_unbroken_run_costs_little(hf):
    """A text as long as Kaimei tokenizes, with no break at all: one word of 200,000 characters."""
    tokens = ["a", "b"] + ["ab" * 2**power for power in range(8)]
    merges = [("a", "b")] + [("ab" * 2**power, "ab" * 2**power) for power in range(7)]
    original = _byte_level_bpe(hf, tokens, merges)
    ours = Tokenizer(_exact(original))
    text = "b" + "ab" * 99_999 + "a"
    started = time.perf_counter()
    ids = ours.encode(text)
    assert time.perf_counter() - started < 60  # merging every place the best pair is, over and over, takes hours
    assert ids == original.encode(text).ids


def test_the_sample_texts_hold_runs_longer_than_256_bytes():
    long_runs = [text for text in SAMPLE_TEXTS if len(text.encode("utf-8")) > 256 and not any(c.isspace() for c in text)]
    assert len(long_runs) >= 5
    assert any(text.isdigit() for text in long_runs) and any(text.isalpha() and text.isascii() for text in long_runs)
    assert any(ord(char) > 0x4E00 for char in "".join(long_runs))


# --- WordPiece ---------------------------------------------------------------------------------------------


#: A BERT vocabulary written out, so that which pieces there are is known: no ``ab``, no ``qz``, but ``the``.
PIECES = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "q", "t", "x", "z", "1", "2", "3", "the"]
PIECES += [f"##{piece}" for piece in ("a", "b", "c", "e", "h", "o", "q", "r", "x", "y", "z", "1", "2", "3", "12", "ory")]


def _bert_like(hf, max_chars=100):
    from tokenizers import models, normalizers, pre_tokenizers, processors

    vocab = {token: index for index, token in enumerate(PIECES)}
    made = hf.Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=max_chars))
    made.normalizer = normalizers.BertNormalizer(lowercase=True)
    made.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    made.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    made.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    return made


def test_a_whole_word_token_the_owner_added_is_never_a_piece_of_a_word(hf):
    """``abc123`` was ``ab ##c ##12 ##3`` with ``ab`` added as a whole word; Hugging Face gives ``a ##b ##c ...``."""
    from tokenizers import AddedToken

    original = _bert_like(hf)
    original.add_tokens([AddedToken("ab", single_word=True), AddedToken("qz", single_word=True, normalized=False)])
    original.add_tokens([AddedToken("the", single_word=True)])  # already one of the pieces, and still one
    texts = ["abc123", "ab", "ab abc", "xab", "qzx", "qz", "theory", "the", "thethe", "abab", "zab qz1"]
    assert original.encode("abc123").tokens == ["[CLS]", "a", "##b", "##c", "##12", "##3", "[SEP]"]
    assert original.encode("theory").tokens == ["[CLS]", "the", "##ory", "[SEP]"]
    described = _exact(original, samples=texts)
    found = {item["content"]: item for item in described["added_tokens"]}
    assert found["ab"]["piece"] is False and found["qz"]["piece"] is False
    assert "piece" not in found["the"] and "piece" not in found["[MASK]"]
    _same(original, described, texts)


def test_two_added_tokens_that_are_one_text_once_normalized_are_not_called_exact(hf):
    """``newtoken`` and ``NewToken`` lower-cased are one text: Hugging Face takes the first in some processes and the
    second in others, so no reading gives its ids every time, and the vocabulary alone is sent."""
    from tokenizers import AddedToken

    original = _bert_like(hf)
    original.add_tokens([AddedToken("newtoken", single_word=True), AddedToken("NewToken")])
    with pytest.warns(UserWarning, match="'newtoken' once normalized"):
        described = tokenizer_spec(original)
    assert described["type"] == "pieces"
    alone = _bert_like(hf)
    alone.add_tokens([AddedToken("NewToken"), AddedToken("OtherToken", single_word=True)])
    _same(alone, _exact(alone), ["NewToken newtoken", "xnewtokeny", "othertoken aothertoken"])


def test_a_wordpiece_tokenizer_with_longer_words_splits_them_as_hugging_face_does(hf):
    original = _bert_like(hf, max_chars=1000)
    word = "abcqx" * 112  # 560 letters
    assert len(original.encode(word).ids) > 3  # split, not one unknown token
    described = _exact(original, samples=[word])
    assert described["max_word_chars"] == 1000
    _same(original, described, [word, word[:257], word[:1000], word + "x" * 440, word + "x" * 441, "the" * 300])


def test_a_long_word_is_matched_without_trying_stretches_longer_than_any_piece():
    vocab = ["[UNK]", "a", "##a", "aa", "##aa"]
    ours = Tokenizer({"type": "wordpiece", "vocab": vocab, "unk": "[UNK]", "max_word_chars": 10**9})
    started = time.perf_counter()
    ids = ours.encode("a" * 150_001)
    assert time.perf_counter() - started < 30  # trying every end of the word first is quadratic in its length
    assert ids == [3] + [4] * 74_999 + [2]
