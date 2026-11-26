"""Three modules are byte-for-byte copies of the API's: the ONNX reader, the ONNX importer and the tokenizers.

The API runs what the importer writes and tokenizes with the same code the client checked a tokenizer against,
so the copies must not drift. This repository cannot see the API's files; what it can check is that the copies
are here, say where they come from, hold nothing of this package's own, and still offer what the rest of the
client calls.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import kaimei

PACKAGE = Path(kaimei.__file__).resolve().parent
MIRRORED = {
    "onnx_reader.py": "kaimei_api/engine/structure/nd/onnx_reader.py",
    "onnx_graph.py": "kaimei_api/engine/structure/nd/onnx_graph.py",
    "tokenizers.py": None,
}


@pytest.mark.parametrize("name", sorted(MIRRORED))
def test_the_mirrored_file_is_present_and_is_python(name):
    path = PACKAGE / name
    assert path.is_file(), f"{name} is missing: copy it from the API (see CHANGELOG, 0.11.0)"
    source = path.read_text(encoding="utf-8")
    assert len(source) > 2000
    ast.parse(source)


@pytest.mark.parametrize("name", ["onnx_reader.py", "onnx_graph.py"])
def test_the_onnx_copies_say_which_file_they_mirror(name):
    source = (PACKAGE / name).read_text(encoding="utf-8")
    assert MIRRORED[name] in source


@pytest.mark.parametrize("name", sorted(MIRRORED))
def test_a_mirrored_file_imports_nothing_else_of_this_package(name):
    """Client-only logic lives in kaimei.graphs and kaimei.text, so a copy from the API never loses any."""
    tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            assert node.module == "onnx_reader", f"{name} imports from .{node.module}"
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("kaimei") for alias in node.names)
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert not str(node.module or "").startswith("kaimei"), f"{name} imports {node.module}"


def test_the_tokenizers_copy_offers_what_the_client_calls():
    tokenizers = importlib.import_module("kaimei.tokenizers")
    assert set(tokenizers.KINDS) >= {"char", "word", "wordpiece", "bpe"}
    built = tokenizers.build_tokenizer({"type": "word", "vocab": ["<unk>", "a", "film"], "unk": "<unk>"})
    assert built.encode("A film, zzz") == [1, 2, 0, 0]
    assert built.size == 3 and built.token(2) == "film"
    with pytest.raises(tokenizers.TokenizerError):
        tokenizers.build_tokenizer({"type": "sentencepiece", "vocab": ["a"]})


def test_the_onnx_copies_offer_what_the_client_calls():
    pytest.importorskip("numpy")
    reader = importlib.import_module("kaimei.onnx_reader")
    graph = importlib.import_module("kaimei.onnx_graph")
    for name in ("load_model", "read_model", "read_graph", "OnnxReadError"):
        assert hasattr(reader, name), name
    for name in ("import_onnx", "run_reference", "load_model", "OnnxImportError", "_input_infos", "_onnx"):
        assert hasattr(graph, name), name
    import inspect

    accepted = set(inspect.signature(graph.import_onnx).parameters)
    assert accepted >= {
        "feature_names", "task", "labels", "inputs", "input_shapes", "samples", "output", "framework", "name",
        "check", "base_dir", "reference",
    }
