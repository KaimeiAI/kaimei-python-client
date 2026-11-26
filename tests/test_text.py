"""Tokenizers written down for Kaimei: translated, then shown to split text exactly as the original does."""

from __future__ import annotations

import json
import warnings

import pytest

from kaimei import StructureError
from kaimei.text import SAMPLE_TEXTS, TokenizerNotTranslated, tokenizer_spec, vocabulary_spec
from kaimei.tokenizers import Tokenizer, TokenizerError

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "A film can be great, awful, moving or merely fine; this one was wonderful!",
    "Tokenization splits words into pieces: tokenizers, tokenized, untokenizable.",
    "Numbers such as 1, 22, 333 and 4,444.5 appear, as do e-mails like someone@example.com.",
    "naïve café résumé façade jalapeño Zürich",
    "It's John's book, isn't it? They've said they'll go.",
    "import numpy as np; x = np.zeros((3, 4))",
    "你好，世界！ こんにちは世界 안녕하세요",
    "emoji 😀 🎉 and symbols ™ © ± ×",
] * 4


def test_there_are_enough_sample_texts_and_they_vary():
    assert len(SAMPLE_TEXTS) >= 40 and len(set(SAMPLE_TEXTS)) == len(SAMPLE_TEXTS)
    joined = "".join(SAMPLE_TEXTS)
    assert any(ord(char) > 0x4E00 for char in joined) and "\t" in joined and "\n" in joined and "é" in joined
    assert any(char.isupper() for char in joined) and any(char.isdigit() for char in joined)


def test_a_kaimei_description_is_checked_and_returned_as_it_is():
    described = {"type": "word", "vocab": ["<pad>", "<unk>", "good", "film"], "unk": "<unk>", "pad": "<pad>"}
    assert tokenizer_spec(described) == described and tokenizer_spec(described) is not described
    letters = {"type": "char", "vocab": list("abc ")}
    assert tokenizer_spec(letters, samples=["abc"]) == letters
    for wrong in (
        {"type": "sentencepiece", "vocab": ["a"]},
        {"type": "word"},
        {"type": "word", "vocab": ["a"], "unk": "<unk>"},
        {"type": "bpe", "vocab": ["a", "b"], "merges": ["a b c"]},
    ):
        with pytest.raises(StructureError) as caught:
            tokenizer_spec(wrong)
        assert caught.value.code == "invalid_tokenizer" and caught.value.param == "tokenizer"


def test_something_that_is_not_a_tokenizer_is_refused_with_a_warning():
    for thing in (object(), "bert-base-uncased", 7):
        with pytest.warns(UserWarning, match="token ids"):
            assert tokenizer_spec(thing) is None

    class Slow:
        is_fast = False

    with pytest.warns(UserWarning, match="use_fast=True"):
        assert tokenizer_spec(Slow()) is None


# --- HuggingFace tokenizers -------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf():
    return pytest.importorskip("tokenizers")


def bert_like(hf, lowercase=True):
    from tokenizers import models, normalizers, pre_tokenizers, processors, trainers

    made = hf.Tokenizer(models.WordPiece(unk_token="[UNK]"))
    made.normalizer = normalizers.BertNormalizer(lowercase=lowercase)
    made.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=400, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    made.train_from_iterator(CORPUS, trainer)
    made.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", made.token_to_id("[CLS]")), ("[SEP]", made.token_to_id("[SEP]"))],
    )
    return made


def gpt2_like(hf, add_prefix_space=False):
    from tokenizers import models, pre_tokenizers, processors, trainers

    made = hf.Tokenizer(models.BPE())
    made.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
    trainer = trainers.BpeTrainer(
        vocab_size=500, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    made.train_from_iterator(CORPUS, trainer)
    made.post_processor = processors.ByteLevel(trim_offsets=False)
    return made


MORE_TEXTS = [
    "A wonderful film, isn't it?",
    "Zürich's café was untokenizable in 2024!",
    "   spaces   everywhere   ",
    "你好 world 😀",
    "x = np.zeros((3, 4)); y=x+1",
]


def same_ids(described, original, texts=MORE_TEXTS):
    ours = Tokenizer(described)
    return all(ours.encode(text) == original.encode(text).ids for text in texts)


def test_a_bert_tokenizer_is_translated_with_its_special_tokens(hf):
    original = bert_like(hf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        described = tokenizer_spec(original)
    assert described["type"] == "wordpiece" and described["lowercase"] is True and described["strip_accents"] is True
    assert described["prefix"] == ["[CLS]"] and described["suffix"] == ["[SEP]"]
    assert described["unk"] == "[UNK]" and described["pad"] == "[PAD]" and described["mask"] == "[MASK]"
    assert described["continuation"] == "##" and described["vocab"]["[PAD]"] == 0
    assert same_ids(described, original)
    json.dumps(described)
    cased = tokenizer_spec(bert_like(hf, lowercase=False))
    assert cased["lowercase"] is False and cased["strip_accents"] is False
    assert Tokenizer(cased).encode("The Film") != Tokenizer(cased).encode("the film")


def test_a_gpt2_tokenizer_is_translated_byte_for_byte(hf):
    original = gpt2_like(hf)
    described = tokenizer_spec(original)
    assert described["type"] == "bpe" and described["byte_level"] is True and described["add_prefix_space"] is False
    assert described["eos"] == "<|endoftext|>" and "prefix" not in described and "unk" not in described
    assert len(described["merges"]) > 100
    assert same_ids(described, original)
    ours = Tokenizer(described)
    assert ours.decode(ours.encode("Zürich's café 😀")) == "Zürich's café 😀"
    spaced = tokenizer_spec(gpt2_like(hf, add_prefix_space=True))
    assert spaced["add_prefix_space"] is True


def test_a_roberta_style_tokenizer_keeps_what_it_adds_around_a_text(hf):
    from tokenizers import models, pre_tokenizers, processors, trainers

    original = hf.Tokenizer(models.BPE())
    original.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=500, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    original.train_from_iterator(CORPUS, trainer)
    original.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0))
    described = tokenizer_spec(original)
    assert described["prefix"] == ["<s>"] and described["suffix"] == ["</s>"]
    assert described["pad"] == "<pad>" and described["mask"] == "<mask>" and described["eos"] == "</s>"
    assert same_ids(described, original)
    assert Tokenizer(described).encode("a film")[0] == 0 and Tokenizer(described).encode("a film")[-1] == 2


def test_a_transformers_fast_tokenizer_is_read_through_its_backend(hf):
    class Fast:
        """What transformers wraps around a tokenizers.Tokenizer."""

        is_fast = True
        pad_token = "[PAD]"
        unk_token = "[UNK]"
        mask_token = "[MASK]"
        eos_token = None

        def __init__(self, backend):
            self.backend_tokenizer = backend

    original = bert_like(hf)
    # A tokenizer set up to pad and cut still has to be compared on what it does to the text itself.
    original.enable_padding(length=24, pad_id=0, pad_token="[PAD]")
    original.enable_truncation(max_length=24)
    described = tokenizer_spec(Fast(original), samples=["the film was wonderful", "naïve café"])
    assert described is not None and described["pad"] == "[PAD]"
    assert Tokenizer(described).encode("the film was wonderful") == [
        index for index in original.encode("the film was wonderful").ids if index != 0
    ]
    assert tokenizer_spec(original) == described


def test_a_tokenizer_kaimei_cannot_run_is_sent_as_its_vocabulary(hf):
    from tokenizers import models, normalizers, pre_tokenizers, trainers

    words = hf.Tokenizer(models.WordLevel(unk_token="[UNK]"))
    words.pre_tokenizer = pre_tokenizers.Whitespace()
    words.train_from_iterator(CORPUS, trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    with pytest.warns(UserWarning, match="WordLevel") as caught:
        pieces = tokenizer_spec(words)
    assert "its vocabulary alone is sent" in str(caught[0].message) and "tokenizer=" in str(caught[0].message)
    assert pieces["type"] == "pieces" and pieces["unk"] == "[UNK]" and "space" not in pieces and "byte_level" not in pieces
    assert pieces["vocab"] == [words.id_to_token(index) for index in range(words.get_vocab_size())]
    built = Tokenizer(pieces)
    assert built.can_encode is False and built.size == words.get_vocab_size()
    ids = words.encode("the film was wonderful").ids
    assert [built.token(index) for index in ids] == ["the", "film", "was", "wonderful"]
    with pytest.raises(TokenizerError, match="not one Kaimei can run"):
        built.encode("the film")
    # A byte-level vocabulary keeps its spelling, so its tokens still read as text.
    lowered = gpt2_like(hf)
    lowered.normalizer = normalizers.Lowercase()
    with pytest.warns(UserWarning, match="normalizes text first"):
        spelled = tokenizer_spec(lowered)
    assert spelled["type"] == "pieces" and spelled["byte_level"] is True and spelled["eos"] == "<|endoftext|>"
    assert Tokenizer(spelled).decode(lowered.encode("Zürich's café").ids) == "zürich's café"
    odd = bert_like(hf)
    odd.pre_tokenizer = pre_tokenizers.Whitespace()
    with pytest.warns(UserWarning, match="not BERT's"):
        assert tokenizer_spec(odd)["type"] == "pieces"
    kept = bert_like(hf)
    kept.normalizer = normalizers.BertNormalizer(handle_chinese_chars=False)
    with pytest.warns(UserWarning, match="Chinese"):
        named = tokenizer_spec(kept)
    assert named["type"] == "pieces" and (named["pad"], named["mask"], named["unk"]) == ("[PAD]", "[MASK]", "[UNK]")


def test_a_translation_that_splits_any_sample_differently_is_not_used(hf):
    original = bert_like(hf)

    class Differs:
        """Says it is BERT's, and is not on one text."""

        def __init__(self, on):
            self.on = on

        def to_str(self):
            return original.to_str()

        def encode(self, text):
            ids = original.encode(text).ids
            return list(reversed(ids)) if text == self.on else ids

    with pytest.warns(UserWarning, match="differently"):
        refused = tokenizer_spec(Differs(SAMPLE_TEXTS[5]))
    # Not the translation: its vocabulary, which names tokens and splits nothing.
    assert refused["type"] == "pieces" and Tokenizer(refused).can_encode is False
    # The caller's own texts are compared as well, and they are the ones that matter.
    own = "a text only this caller would send"
    assert tokenizer_spec(Differs(own))["type"] == "wordpiece"
    with pytest.warns(UserWarning, match="a text only this caller"):
        assert tokenizer_spec(Differs(own), samples=[own])["type"] == "pieces"

    class Raises(Differs):
        def encode(self, text):
            raise RuntimeError("no")

    with pytest.warns(UserWarning, match="could not encode"):
        assert tokenizer_spec(Raises(""))["type"] == "pieces"


def sentencepiece_like(hf):
    """A T5 or Llama-2 style tokenizer, trained here: a unigram model over pieces that mark the space before a word."""
    from tokenizers import decoders, models, pre_tokenizers, processors, trainers

    made = hf.Tokenizer(models.Unigram())
    made.pre_tokenizer = pre_tokenizers.Metaspace()
    made.decoder = decoders.Metaspace()
    trainer = trainers.UnigramTrainer(vocab_size=200, special_tokens=["<pad>", "</s>", "<unk>"], unk_token="<unk>")
    made.train_from_iterator(CORPUS, trainer)
    made.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", made.token_to_id("</s>"))])
    return made


def test_a_sentencepiece_tokenizer_is_written_down_as_its_pieces(hf):
    original = sentencepiece_like(hf)
    with pytest.warns(UserWarning, match="Unigram"):
        pieces = tokenizer_spec(original)
    assert pieces["type"] == "pieces" and pieces["space"] == "\u2581" and "byte_level" not in pieces
    assert (pieces["pad"], pieces["eos"], pieces["unk"]) == ("<pad>", "</s>", "<unk>")
    assert pieces["vocab"][:3] == ["<pad>", "</s>", "<unk>"] and len(pieces["vocab"]) == original.get_vocab_size()
    built = Tokenizer(pieces)
    ids = original.encode("The film was wonderful.").ids
    assert ids[-1] == 1  # the tokenizer's own end of text, which Kaimei could not have known to add
    assert built.decode(ids[:-1]) == "The film was wonderful."
    assert built.token(ids[0]).startswith(" ")  # the mark of a space reads as one
    json.dumps(pieces)
    # Whoever has the tokenizer splits the text: Kaimei's copy of the vocabulary checks what comes back.
    there = built.with_encoder(lambda text, most: original.encode(text).ids[:most])
    assert there.encode("The film was wonderful.", 4) == ids[:4]
    with pytest.raises(TokenizerError, match="vocabulary"):
        built.with_encoder(lambda text, most: [10**6]).encode("anything")


class SlowT5:
    """What transformers gives for a SentencePiece model with no fast tokenizer: named tokens, and no JSON."""

    is_fast = False
    unk_token, pad_token, eos_token, mask_token = "<unk>", "<pad>", "</s>", None
    pieces = ["<pad>", "</s>", "<unk>", "\u2581the", "\u2581film", "s", "\u2581was", "\u2581good", "."]

    def __len__(self):
        return len(self.pieces)

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[index] for index in ids]

    def __call__(self, text, truncation=False, max_length=None):
        ids = [self.pieces.index("\u2581" + word) if "\u2581" + word in self.pieces else 2 for word in text.lower().split()] + [1]
        return {"input_ids": ids[:max_length] if truncation else ids}


class Processor:
    """sentencepiece.SentencePieceProcessor, as far as this module asks it anything."""

    def get_piece_size(self):
        return len(SlowT5.pieces)

    def id_to_piece(self, index):
        return SlowT5.pieces[index]

    def unk_id(self):
        return 2

    def pad_id(self):
        return -1  # this model has no padding piece

    def eos_id(self):
        return 1

    def encode(self, text):
        return SlowT5()(text)["input_ids"]


class Encoding:
    """tiktoken.Encoding, as far as this module asks it anything: tokens are bytes, and not all of them are text."""

    n_vocab = 6
    eot_token = 5
    raw = [b" the", b" film", b"\xc3", b"\xa9", b"!", b"<|endoftext|>"]

    def decode_single_token_bytes(self, index):
        return self.raw[index]

    def encode(self, text):
        return [0, 1, 4]


def test_other_tokenizers_are_read_through_what_they_offer():
    with pytest.warns(UserWarning, match="use_fast=True"):
        slow = tokenizer_spec(SlowT5())
    assert slow == {"type": "pieces", "vocab": SlowT5.pieces, "space": "\u2581", "unk": "<unk>", "pad": "<pad>", "eos": "</s>"}
    with pytest.warns(UserWarning, match="its vocabulary alone is sent"):
        processor = tokenizer_spec(Processor())
    assert processor == {"type": "pieces", "vocab": SlowT5.pieces, "space": "\u2581", "unk": 2, "eos": 1}
    assert Tokenizer(processor).eos == 1 and Tokenizer(processor).pad is None
    with pytest.warns(UserWarning):
        encoding = tokenizer_spec(Encoding())
    assert encoding["type"] == "pieces" and encoding["byte_level"] is True and encoding["eos"] == 5
    built = Tokenizer(encoding)
    assert built.token(0) == " the" and built.decode([0, 1, 2, 3, 4]) == " the filmé!"  # two tokens make one letter
    assert vocabulary_spec(SlowT5()) == slow


def test_a_vocabulary_larger_than_kaimei_takes_is_not_sent():
    class Huge(SlowT5):
        def __len__(self):
            return 400_001

        def convert_ids_to_tokens(self, ids):
            return [f"t{index}" for index in ids]

    with pytest.warns(UserWarning, match="400,001 tokens") as caught:
        assert tokenizer_spec(Huge()) is None
    assert "token ids instead of text" in str(caught[0].message)

    class Wordy(SlowT5):
        pieces = ["<pad>", "</s>", "<unk>", "x" * 300]

    with pytest.warns(UserWarning, match="300 characters long"):
        assert tokenizer_spec(Wordy()) is None
    with pytest.raises(TokenizerNotTranslated):
        vocabulary_spec(object())


def test_the_contents_of_a_tokenizer_json_are_checked_like_the_tokenizer_itself(hf, monkeypatch):
    original = bert_like(hf)
    document = json.loads(original.to_str())
    assert tokenizer_spec(document) == tokenizer_spec(original)
    # Without the tokenizers package nothing can say what the file does to a text: the vocabulary is what is sent.
    import builtins

    real = builtins.__import__

    def without_tokenizers(name, *args, **kwargs):
        if name == "tokenizers":
            raise ImportError("No module named 'tokenizers'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tokenizers)
    with pytest.warns(UserWarning, match="pip install tokenizers"):
        unchecked = tokenizer_spec(document)
    assert unchecked["type"] == "pieces" and unchecked["vocab"][:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
