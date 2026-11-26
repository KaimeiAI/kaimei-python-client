"""Tokenizers labelled exact must split every text as Hugging Face does: special tokens written in a text,
underscores and numbers in GPT-2's pattern, and BERT's lower-casing one character at a time. One that still
differs is sent as its vocabulary alone, with a warning."""

from __future__ import annotations

import warnings

import pytest

from kaimei import text as kaimei_text
from kaimei.text import SAMPLE_TEXTS, tokenizer_spec
from kaimei.tokenizers import Tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Paris is the capital of France, and the film was wonderful!",
    "foo_(bar) and return self._cache.get(key) and user_'s profile",
    "snake_case_names_-_dash __init__ __dict__",
    "x² + y³ = z½, H₂O and CO₂ and chapter Ⅻ",
    # Both endings of a Greek sigma, so lower-casing a whole word (ς) and each letter alone (σ) differ.
    "ΑΘΗΝΑ ΚΑΙ ΘΕΣΣΑΛΟΝΙΚΗΣ, ΟΔΟΣ ΣΟΦΟΣ, θεσσαλονικης οδος",
    "İstanbul DİYARBAKIR naïve café",
] * 6

#: The cases the pre-deploy review found, beside the sample texts.
REVIEWED = [
    "Paris is the capital of [MASK].",
    "[CLS] a [SEP] b [SEP]",
    "the end<|endoftext|>next document",
    "<mask> fills here and a <mask> there",
    "foo_(bar)",
    "return self._cache.get(key)",
    "user_'s profile",
    "snake_case_names_-_dash",
    "ΘΕΣΣΑΛΟΝΙΚΗΣ",
    "the covid19vaccine works",
]


@pytest.fixture(scope="module")
def hf():
    return pytest.importorskip("tokenizers")


def bert_like(hf):
    from tokenizers import models, normalizers, pre_tokenizers, processors, trainers

    made = hf.Tokenizer(models.WordPiece(unk_token="[UNK]"))
    made.normalizer = normalizers.BertNormalizer(lowercase=True)
    made.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    made.train_from_iterator(CORPUS, trainer)
    made.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", made.token_to_id("[CLS]")), ("[SEP]", made.token_to_id("[SEP]"))],
    )
    made.add_tokens(["covid19vaccine"])
    return made


def gpt2_like(hf):
    from tokenizers import AddedToken, models, pre_tokenizers, processors, trainers

    made = hf.Tokenizer(models.BPE())
    made.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    made.train_from_iterator(CORPUS, trainer)
    made.post_processor = processors.ByteLevel(trim_offsets=False)
    made.add_special_tokens([AddedToken("<mask>", lstrip=True, normalized=False)])
    made.add_tokens(["covid19vaccine"])
    return made


def _exact(original):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return tokenizer_spec(original, samples=REVIEWED)


def test_the_reviewed_cases_are_among_the_sample_texts():
    joined = "\n".join(SAMPLE_TEXTS)
    for case in ("[MASK]", "[CLS]", "<mask>", "<|endoftext|>", "foo_(bar)", "user_'s", "_-_", "ΘΕΣΣΑΛΟΝΙΚΗΣ", "²", "Ⅻ"):
        assert case in joined


def test_a_bert_tokenizer_with_special_and_added_tokens_is_exact(hf):
    original = bert_like(hf)
    described = _exact(original)
    assert described["type"] == "wordpiece"
    found = {item["content"]: item for item in described["added_tokens"]}
    assert found["[MASK]"]["special"] is True and found["[MASK]"]["normalized"] is False
    assert found["covid19vaccine"]["special"] is False and found["covid19vaccine"]["normalized"] is True
    ours = Tokenizer(described)
    for text in [*SAMPLE_TEXTS, *REVIEWED]:
        assert ours.encode(text) == original.encode(text).ids, text
    assert original.token_to_id("[MASK]") in ours.encode("Paris is the capital of [MASK].")


def test_a_gpt2_tokenizer_with_special_and_added_tokens_is_exact(hf):
    original = gpt2_like(hf)
    described = _exact(original)
    assert described["type"] == "bpe"
    found = {item["content"]: item for item in described["added_tokens"]}
    assert found["<mask>"]["lstrip"] is True and found["<|endoftext|>"]["special"] is True
    ours = Tokenizer(described)
    for text in [*SAMPLE_TEXTS, *REVIEWED]:
        assert ours.encode(text) == original.encode(text).ids, text


class _NoAddedTokens(Tokenizer):
    """Kaimei's tokenizer as it was: special tokens written in a text were split like any other text."""

    def __init__(self, spec):
        super().__init__({key: value for key, value in spec.items() if key != "added_tokens"})


class _WholeWordLower(Tokenizer):
    """Kaimei's tokenizer as it was: ``str.lower`` over a whole text, which ends a word's Σ in ς."""

    def _normalize(self, text):
        return super()._normalize(text.lower() if self.lowercase else text)


@pytest.mark.parametrize("older", [_NoAddedTokens, _WholeWordLower])
def test_a_translation_that_still_differs_is_sent_as_pieces_with_a_warning(hf, monkeypatch, older):
    original = bert_like(hf)
    monkeypatch.setattr(kaimei_text, "Tokenizer", older)
    with pytest.warns(UserWarning, match="differently"):
        described = tokenizer_spec(original)
    assert described["type"] == "pieces"


def test_added_tokens_are_checked_on_texts_that_write_them(hf):
    texts = kaimei_text._added_samples({"added_tokens": [{"content": "<sep2>"}]})
    assert texts and all("<sep2>" in text for text in texts)
    assert kaimei_text._added_samples({}) == []
