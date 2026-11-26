"""Models from TensorFlow files, JAX and GGUF files, read through ONNX and checked against their own framework.

Each test skips when its framework is not installed: TensorFlow (with tf2onnx) for SavedModels and TFLite
files, JAX (with jax2onnx) for JAX, and PyTorch with transformers, gguf and accelerate for GGUF files.
"""

from __future__ import annotations

import os
import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnx")

from kaimei import StructureError, converters  # noqa: E402
from kaimei.structure import extract_structure, spec_is_executable  # noqa: E402


def runs(spec):
    return "check" in spec and not spec.get("approximate") and spec_is_executable(spec)


def table_rows(spec):
    names = spec["feature_names"]
    return np.array([[spec["check"]["inputs"][name][row][0] for name in names] for row in range(len(spec["check"]["outputs"]))])


@pytest.fixture(scope="module")
def keras_model():
    keras = pytest.importorskip("keras")
    pytest.importorskip("tf2onnx")
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(32, 6)).astype("float32")
    model = keras.Sequential([keras.Input((6,)), keras.layers.Dense(5, activation="relu"), keras.layers.Dense(3, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(rows, (rows[:, 0] > 0).astype(int) + (rows[:, 1] > 0).astype(int), epochs=1, verbose=0)
    return model


def test_a_tensorflow_savedmodel_directory_is_read_and_checked_against_tensorflow(keras_model, tmp_path):
    folder = str(tmp_path / "saved")
    keras_model.export(folder)
    assert converters.model_file_kind(folder) == "saved_model"
    spec = extract_structure(folder, feature_names=[f"f{i}" for i in range(6)]).spec
    assert spec["family"] == "graph" and spec["framework"] == "tensorflow" and spec["task"] == "multiclass_classification"
    assert runs(spec) and spec["feature_names"] == [f"f{i}" for i in range(6)]
    np.testing.assert_allclose(spec["check"]["outputs"], keras_model.predict(table_rows(spec), verbose=0), rtol=1e-5, atol=1e-6)


def test_a_tensorflow_lite_file_is_read_and_checked_against_its_interpreter(keras_model, tmp_path):
    tf = pytest.importorskip("tensorflow")
    folder = str(tmp_path / "saved")
    keras_model.export(folder)
    path = tmp_path / "model.tflite"
    path.write_bytes(tf.lite.TFLiteConverter.from_saved_model(folder).convert())
    assert converters.model_file_kind(str(path)) == "tflite"
    spec = extract_structure(str(path)).spec
    assert spec["framework"] == "tflite" and spec["task"] == "multiclass_classification" and runs(spec)
    assert spec["feature_names"] == [f"x{i}" for i in range(6)]


def test_a_jax_function_is_read_and_checked_against_jax():
    jax = pytest.importorskip("jax")
    pytest.importorskip("jax2onnx")
    jnp = jax.numpy
    rng = np.random.default_rng(1)
    w1 = jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.float32)
    w2 = jnp.asarray(rng.normal(size=(8, 3)), dtype=jnp.float32)

    def classify(x):
        return jax.nn.softmax(jnp.tanh(x @ w1) @ w2)

    jitted = jax.jit(classify)
    assert converters.is_jax(jitted) and not converters.is_jax(classify) and converters.is_jax(classify, "jax")
    sample = rng.normal(size=(1, 4)).astype("float32")
    spec = extract_structure(jitted, sample=sample, labels=["a", "b", "c"]).spec
    assert spec["framework"] == "jax" and spec["task"] == "multiclass_classification" and runs(spec)
    np.testing.assert_allclose(spec["check"]["outputs"], np.asarray(classify(table_rows(spec))), rtol=1e-5, atol=1e-6)
    with pytest.raises(StructureError) as caught:
        extract_structure(jitted)
    assert caught.value.param == "sample"


def tiny_gguf(path):
    gguf = pytest.importorskip("gguf")
    vocab, hidden, inner, heads = 40, 16, 32, 2
    rng = np.random.default_rng(0)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_context_length(32)
    writer.add_embedding_length(hidden)
    writer.add_block_count(1)
    writer.add_feed_forward_length(inner)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(hidden // heads)
    writer.add_vocab_size(vocab)
    tokens = ["<unk>", "<s>", "</s>"] + [f"▁w{i}" for i in range(vocab - 3)]
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * 3 + [-float(i) for i in range(vocab - 3)])
    writer.add_token_types([2, 3, 3] + [1] * (vocab - 3))
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)

    def weight(name, *shape):
        writer.add_tensor(name, (rng.normal(size=shape) * 0.2).astype(np.float32))

    weight("token_embd.weight", vocab, hidden)
    writer.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, np.float32))
    for part in ("attn_q", "attn_k", "attn_v", "attn_output"):
        weight(f"blk.0.{part}.weight", hidden, hidden)
    writer.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, np.float32))
    weight("blk.0.ffn_gate.weight", inner, hidden)
    weight("blk.0.ffn_up.weight", inner, hidden)
    weight("blk.0.ffn_down.weight", hidden, inner)
    writer.add_tensor("output_norm.weight", np.ones(hidden, np.float32))
    weight("output.weight", vocab, hidden)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_a_gguf_file_is_opened_as_a_language_model(tmp_path):
    for package in ("torch", "transformers", "accelerate"):
        pytest.importorskip(package)
    path = tmp_path / "tiny-llama.gguf"
    tiny_gguf(path)
    assert converters.model_file_kind(str(path)) == "gguf"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = extract_structure(str(path)).spec
    assert spec["framework"] == "gguf" and spec["task"] == "text_generation" and runs(spec)
    assert spec["features"][0]["max_length"] == converters.GGUF_DEFAULT_TOKENS


def test_a_framework_that_is_not_installed_is_named(monkeypatch, tmp_path):
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "saved_model.pb").write_bytes(b"")
    monkeypatch.setattr(converters, "_has", lambda package: package not in ("tensorflow", "tf2onnx"))
    with pytest.raises(StructureError) as caught:
        extract_structure(str(folder))
    assert "pip install tensorflow tf2onnx" in caught.value.message
    assert converters.model_file_kind(str(tmp_path / "absent.gguf")) is None
    assert not os.path.exists(tmp_path / "absent.gguf")
