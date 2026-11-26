"""Tokenizers: writing down how a model's text becomes token ids, so Kaimei can take text for it.

A language model reads token ids. To send it text, Kaimei needs the model's tokenizer, and it is never sent
a tokenizer file: :func:`tokenizer_spec` writes the tokenizer down as plain JSON (its kind, vocabulary,
merges and special tokens) in the form :mod:`kaimei.tokenizers` reads, and then checks, on a set of sample
texts, that Kaimei's tokenizer built from that description gives exactly the ids the original gives.

Written down exactly: a Kaimei description (``{"type": "word" | "char" | "wordpiece" | "bpe", "vocab":
[...], ...}``), and HuggingFace tokenizers (a ``tokenizers.Tokenizer`` or a ``transformers`` fast tokenizer)
of two kinds: WordPiece as BERT uses it, and byte-level BPE as GPT-2 and RoBERTa use it. Their added tokens
(the special ones, and any added with ``add_tokens``) are written down with how each is found in a text
(``added_tokens``), and are checked on texts that write them.

Any other tokenizer (SentencePiece's, tiktoken's, one of your own), and one whose translation does not pass
the check, is written down as its vocabulary alone: ``{"type": "pieces", "vocab": [...]}``. Kaimei can then
name every token and turn ids back into text, but it cannot split text. Text for such a model is split where
the tokenizer is, by the model's agent (:mod:`kaimei.agent`, started with ``tokenizer=``), or the model is
sent token ids.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import KaimeiError
from .structure import StructureError
from .tokenizers import MAX_TOKEN_CHARS, MAX_VOCABULARY, Tokenizer, TokenizerError, build_tokenizer

#: Texts every translated tokenizer is compared on: plain and odd spacing, punctuation, digits, accents, other
#: scripts, emoji, very long words, and the characters tokenizers clean away.
SAMPLE_TEXTS: Tuple[str, ...] = (
    "Hello world",
    "hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "THE QUICK BROWN FOX",
    "  leading and trailing spaces  ",
    "two  spaces and\ttabs\nand new lines",
    "It's John's book, isn't it? They've said they'll go; I'm not sure we'd agree.",
    "numbers 0 1 22 333 4444 1,000,000 3.14159 -42 1e-5 50% $9.99",
    "e-mail me at someone@example.com or visit https://example.com/path?q=1&r=2",
    "state-of-the-art self-attention, encoder–decoder and co-operate",
    "naïve café résumé façade coöperate jalapeño Zürich Ångström",
    "Übergrößenträger und Straße",
    "Привет, мир! Как дела?",
    "Γειά σου Κόσμε",
    "こんにちは世界、元気ですか？",
    "你好，世界！机器学习很有趣。",
    "안녕하세요 세계",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "emoji 😀 👍🏽 🎉 and symbols ™ © ® ± × ÷ √ ∞",
    "quotes “curly” ‘single’ «angle» and `backticks`",
    "brackets (round) [square] {curly} <angle>",
    "an ellipsis... and a dash — and another – and more!!! really??",
    "under_scores and camelCaseWords and snake_case_names",
    "supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis",
    "a" * 120,
    "x",
    "I",
    ".",
    "?!",
    "@#$%^&*()_+-=[]{}|;':\",./<>?",
    "tokenization tokenizers tokenized untokenizable retokenize",
    "running runs ran runner unrunnable",
    "unbelievably antidisestablishmentarianism internationalization",
    "the the the the",
    "A sentence. Another sentence! A third one? Yes.",
    "mixed 123abc abc123 a1b2c3 v2.0 3rd 1st",
    "tab\tseparated\tvalues",
    "line one\nline two\r\nline three",
    "non breaking thin　ideographic spaces",
    "control\x00chars\x07and�replacement",
    "soft­hyphen and zero​width and joiner‍",
    "ﬁligature ﬂow and ½ ¾ ² ³",
    "ｆｕｌｌｗｉｄｔｈ ｔｅｘｔ",
    "The film was really wonderful, but the ending was not good.",
    "import numpy as np; x = np.zeros((3, 4))",
    "C++ and C# and F# and .NET",
    # Special tokens written in a text are those tokens, not their letters.
    "Paris is the capital of [MASK].",
    "[CLS] a [SEP] b [SEP] [PAD] [UNK] [MASK][MASK]",
    "<s> a </s> and <mask> fills here, <pad> <unk>",
    "the end<|endoftext|>next document <|endoftext|>",
    # Underscores, which Python's \w counts as a letter and GPT-2's pattern does not.
    "foo_(bar) and return self._cache.get(key)",
    "user_'s profile, x = obj._x + y_ and see _README_.md",
    "snake_case_names_-_dash __init__ __dict__ ___ ____bar",
    # Numbers that are not digits, which Python's \w counts as letters.
    "x² + y³ = z½, H₂O and CO₂, km² and 1st2nd³",
    "chapter Ⅻ and ⅓ cup and ① ⑳ and ٣ ४ ໓",
    # Lower case one character at a time: a final capital sigma is σ, not ς.
    "ΑΘΗΝΑ ΚΑΙ ΘΕΣΣΑΛΟΝΙΚΗΣ, ΟΔΟΣ ΣΟΦΟΣ",
    "İstanbul DİYARBAKIR and ǅungla Ǉ ǋ ẞ",
    "é café and ñ (accents written apart)",
    "don'T WON'T I'M YOU'RE 'S 'll 'D",
    "line sep para\u0085nel\x0bvt\x0cff\x1cfs\x1fus",
    "\ttab first",
    "\nnew line first",
    "Ⓐ Ⓑ 🅐 circled and zero‌width‍joiners",
    "\U00020000\U00020001 and \U0002b820\U0002b821 and 豈 (ideographs past the first block)",
    # Runs of hundreds of bytes with no break, which are merged and matched whole however long they are:
    # Chinese and Japanese written without punctuation, a long word, a long number and a long run of symbols.
    "机器学习是人工智能的一个分支它使用统计方法让计算机系统能够从数据中学习并且在没有明确编程的情况下逐步提高完成特定任务的性能这一领域与计算统计学密切相关后者侧重于使用计算机进行预测",
    "これはにほんごのぶんしょうでくとうてんがひとつもなくてながくつづいていくれいですがほんとうにながいぶんしょうはこのようにどこまでもつづいていくことがありますそしてまだまだつづきます",
    "methionylthreonylthreonylglutaminylarginyltyrosylglutamylserylleucylphenylalanylalanylglutaminylleucyllysyl"
    "glutamylarginyllysylglutamylglycylalanylphenylalanylvalylprolylphenylalanylvalylthreonylleucylglycylaspartyl"
    "prolylglycylisoleucylglutamylglutaminylserylleucyllysylisoleucylaspartylthreonylleucylisoleucylaspartylalanyl",
    "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067"
    "9821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819"
    "6442881097566593344612847564823378678316527120190914564856692346034861045432664821339360726024914127",
    "/" + "-_./?&=%#" * 33,
)

#: What SentencePiece writes for the space before a word, and GPT-2's stand-in for the byte of a space.
SPACE_MARK = "\u2581"
BYTE_SPACE = "\u0120"
_PAD_NAMES = ("[PAD]", "<pad>", "<|pad|>", "<PAD>", "[pad]")
_MASK_NAMES = ("[MASK]", "<mask>", "<|mask|>", "<MASK>")
_EOS_NAMES = ("</s>", "<|endoftext|>", "<eos>", "[EOS]", "<|im_end|>", "<|eot_id|>", "<EOS>")


class TokenizerNotTranslated(KaimeiError):
    """A tokenizer this module cannot write down exactly, with the reason."""


def tokenizer_spec(tokenizer: Any, samples: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """The tokenizer as the JSON description Kaimei reads.

    ``tokenizer`` is a Kaimei description (checked, then returned as it is), a HuggingFace
    ``tokenizers.Tokenizer``, a ``transformers`` tokenizer, a SentencePiece processor or a tiktoken encoding.

    A HuggingFace WordPiece or byte-level BPE tokenizer is translated and then compared with the original on
    :data:`SAMPLE_TEXTS` and on ``samples`` (texts of your own, which are the ones that matter most): every
    text must give exactly the same ids. One that passes is returned in full, and Kaimei splits text with it.

    Any other tokenizer, and one for which any text differs, is returned as its vocabulary alone
    (``{"type": "pieces", ...}``) with a warning that says why: Kaimei then names tokens and turns ids back
    into text, and text is split by the model's agent or sent as ids. ``None``, with a warning, when not
    even the vocabulary can be written down (it cannot be read, or is larger than Kaimei takes): the model
    then takes token ids and nothing more.
    """
    if isinstance(tokenizer, Mapping) and "model" not in tokenizer:
        described = dict(tokenizer)
        try:
            build_tokenizer(described)
        except TokenizerError as exc:
            raise StructureError(
                f"This tokenizer description can't be used: {exc}", code="invalid_tokenizer", param="tokenizer"
            ) from exc
        return described
    try:
        described = _translate(tokenizer)
        texts = [*SAMPLE_TEXTS, *_added_samples(described), *[str(text) for text in samples or []]]
        _same_ids(tokenizer, described, texts)
        return described
    except TokenizerNotTranslated as exc:
        why = exc.message
    try:
        pieces = vocabulary_spec(tokenizer)
    except TokenizerNotTranslated as exc:
        warnings.warn(
            f"This tokenizer is not sent to Kaimei ({why}; and {exc.message}). The model will take token ids "
            "instead of text: encode the text yourself and send the ids.",
            UserWarning,
            stacklevel=2,
        )
        return None
    warnings.warn(
        f"Kaimei cannot split text with this tokenizer itself ({why}), so its vocabulary alone is sent: tokens "
        "are named and ids are turned back into text. Text for this model is split by its agent, started with "
        "the tokenizer (kaimei.agent.serve(..., tokenizer=...)); without one, send token ids.",
        UserWarning,
        stacklevel=2,
    )
    return pieces


def _byte_text() -> Dict[int, str]:
    """GPT-2's printable stand-in for every byte, which is how a byte-level vocabulary is written as text."""
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    mapped = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {byte: chr(code) for byte, code in zip(visible, mapped)}


def _tokens_of(tokenizer: Any) -> Tuple[List[str], bool]:
    """``(the text of every token, in id order; whether they are spelled as bytes)`` from whatever names its tokens."""
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(converter):  # transformers, fast or slow
        try:
            count = len(tokenizer)
        except TypeError:
            count = int(getattr(tokenizer, "vocab_size", 0) or 0)
        return ["" if token is None else str(token) for token in converter(list(range(count)))], False
    backend = getattr(tokenizer, "backend_tokenizer", None)
    source = backend if backend is not None else tokenizer
    namer, sizer = getattr(source, "id_to_token", None), getattr(source, "get_vocab_size", None)
    if callable(namer) and callable(sizer):  # tokenizers.Tokenizer
        try:
            count = int(sizer(with_added_tokens=True))
        except TypeError:
            count = int(sizer())
        return [str(namer(index) or "") for index in range(count)], False
    namer = getattr(tokenizer, "id_to_piece", None) or getattr(tokenizer, "IdToPiece", None)
    sizer = getattr(tokenizer, "get_piece_size", None) or getattr(tokenizer, "GetPieceSize", None)
    if callable(namer) and callable(sizer):  # sentencepiece.SentencePieceProcessor
        return [str(namer(index)) for index in range(int(sizer()))], False
    as_bytes, size = getattr(tokenizer, "decode_single_token_bytes", None), getattr(tokenizer, "n_vocab", None)
    if callable(as_bytes) and isinstance(size, int):  # tiktoken.Encoding: tokens are bytes, not always text
        spelled = _byte_text()
        tokens: List[str] = []
        for index in range(size):
            try:
                raw = bytes(as_bytes(index))
            except Exception:  # noqa: BLE001 - an id the encoding does not use
                raw = b""
            tokens.append("".join(spelled[byte] for byte in raw))
        return tokens, True
    try:
        document = _document(tokenizer)  # a tokenizer.json, or anything that writes one
    except TokenizerNotTranslated:
        raise TokenizerNotTranslated("it does not say what its tokens are") from None
    vocab = (document.get("model") or {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise TokenizerNotTranslated("it does not say what its tokens are")
    by_id = {int(index): str(token) for token, index in vocab.items()}
    for item in document.get("added_tokens") or []:
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            by_id.setdefault(int(item["id"]), str(item.get("content")))
    return [by_id.get(index, "") for index in range(max(by_id) + 1)], False


def vocabulary_spec(tokenizer: Any) -> Dict[str, Any]:
    """A tokenizer as its vocabulary alone: ``{"type": "pieces", "vocab": [...]}``, with the marks it writes
    spaces with and its special tokens.

    This is what is sent for a tokenizer Kaimei cannot run (see :func:`tokenizer_spec`, which calls it when
    it has to). Raises :class:`TokenizerNotTranslated` when the vocabulary cannot be read or is more than
    Kaimei takes.
    """
    tokens, spelled_as_bytes = _tokens_of(tokenizer)
    if not tokens:
        raise TokenizerNotTranslated("its vocabulary is empty")
    if len(tokens) > MAX_VOCABULARY:
        raise TokenizerNotTranslated(f"its vocabulary has {len(tokens):,} tokens, and Kaimei takes up to {MAX_VOCABULARY:,}")
    longest = max(len(token) for token in tokens)
    if longest > MAX_TOKEN_CHARS:
        raise TokenizerNotTranslated(f"one of its tokens is {longest:,} characters long, and Kaimei takes up to {MAX_TOKEN_CHARS}")
    described: Dict[str, Any] = {"type": "pieces", "vocab": tokens}
    marked = sum(1 for token in tokens if token.startswith(SPACE_MARK))
    spelled = sum(1 for token in tokens if token.startswith(BYTE_SPACE))
    if marked and marked >= spelled and not spelled_as_bytes:
        described["space"] = SPACE_MARK
    elif spelled_as_bytes or spelled * 50 >= len(tokens):
        described["byte_level"] = True
    known = set(tokens)
    for name, attribute, method in (("unk", "unk_token", "unk_id"), ("pad", "pad_token", "pad_id"), ("mask", "mask_token", ""), ("eos", "eos_token", "eos_id")):
        token = getattr(tokenizer, attribute, None)
        if isinstance(token, str) and token in known:
            described[name] = token
            continue
        finder = getattr(tokenizer, method, None) if method else None
        if callable(finder):  # SentencePiece says where its special tokens are, and -1 where it has none
            try:
                index = int(finder())
            except Exception:  # noqa: BLE001 - a tokenizer without that token
                continue
            if 0 <= index < len(tokens):
                described[name] = index
    end = getattr(tokenizer, "eot_token", None)  # tiktoken's end of text
    if "eos" not in described and isinstance(end, int) and not isinstance(end, bool) and 0 <= end < len(tokens):
        described["eos"] = end
    if not any(key in described for key in ("unk", "pad", "mask", "eos")):
        _specials_from_document(tokenizer, described, tokens)
    try:
        build_tokenizer(described)
    except TokenizerError as exc:
        raise TokenizerNotTranslated(str(exc)) from exc
    return described


def _specials_from_document(tokenizer: Any, described: Dict[str, Any], tokens: Sequence[str]) -> None:
    """The special tokens a ``tokenizers.Tokenizer`` names in its own JSON, when nothing else said."""
    try:
        document = _document(tokenizer)
    except TokenizerNotTranslated:
        return
    known = set(tokens)
    model = document.get("model") or {}
    unknown = model.get("unk_token")
    where = model.get("unk_id")  # a unigram model says where its unknown piece is, not what it is called
    if isinstance(unknown, str) and unknown in known:
        described["unk"] = unknown
    elif isinstance(where, int) and not isinstance(where, bool) and 0 <= where < len(tokens):
        described["unk"] = tokens[where]
    special = {str(item.get("content")) for item in document.get("added_tokens") or [] if isinstance(item, dict) and item.get("special")}
    for name, names in (("pad", _PAD_NAMES), ("mask", _MASK_NAMES), ("eos", _EOS_NAMES)):
        token = next((item for item in names if item in special and item in known), None)
        if token is not None:
            described[name] = token


def _document(tokenizer: Any) -> Dict[str, Any]:
    """The tokenizer's own JSON (what ``tokenizer.json`` holds)."""
    if isinstance(tokenizer, Mapping):
        return dict(tokenizer)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    source = backend if backend is not None else tokenizer
    writer = getattr(source, "to_str", None)
    if not callable(writer):
        if getattr(tokenizer, "is_fast", None) is False:
            raise TokenizerNotTranslated(
                "it is a slow (pure Python) tokenizer; load the fast one with use_fast=True"
            )
        raise TokenizerNotTranslated(f"{type(tokenizer).__name__} is not a HuggingFace tokenizer or a description")
    try:
        document = json.loads(writer())
    except (TypeError, ValueError) as exc:
        raise TokenizerNotTranslated("its JSON could not be read") from exc
    if not isinstance(document, dict) or not isinstance(document.get("model"), dict):
        raise TokenizerNotTranslated("its JSON has no model section")
    return document


def _translate(tokenizer: Any) -> Dict[str, Any]:
    document = _document(tokenizer)
    model = document.get("model") or {}
    kind = str(model.get("type") or "")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise TokenizerNotTranslated(f"its {kind or 'model'} has no vocabulary of tokens and ids")
    tokens: Dict[str, int] = {str(token): int(index) for token, index in vocab.items()}
    added = [item for item in document.get("added_tokens") or [] if isinstance(item, dict)]
    taken = set(tokens.values())
    for item in added:
        content, index = str(item.get("content")), item.get("id")
        if content not in tokens and isinstance(index, int) and index not in taken:
            tokens[content] = index
            taken.add(index)
    if sorted(taken) != list(range(len(taken))):
        raise TokenizerNotTranslated("its token ids are not the numbers from 0 up without gaps")
    normalizer = document.get("normalizer")
    pre = document.get("pre_tokenizer") or {}
    if kind == "WordPiece":
        lowercase, strip_accents = _bert_normalizer(normalizer)
        if str(pre.get("type") or "") != "BertPreTokenizer":
            raise TokenizerNotTranslated(f"WordPiece with a {pre.get('type') or 'missing'} pre-tokenizer is not BERT's")
        described: Dict[str, Any] = {
            "type": "wordpiece",
            "vocab": tokens,
            "lowercase": lowercase,
            "strip_accents": strip_accents,
            "continuation": str(model.get("continuing_subword_prefix") or "##"),
            "max_word_chars": int(model.get("max_input_chars_per_word") or 100),
        }
    elif kind == "BPE":
        if not _is_nothing(normalizer):
            raise TokenizerNotTranslated("its byte-pair encoding normalizes text first, which Kaimei's does not")
        byte_level = _byte_level(pre)
        if byte_level is None:
            raise TokenizerNotTranslated("its byte-pair encoding is not byte-level (GPT-2's is)")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix") or model.get("dropout"):
            raise TokenizerNotTranslated("its byte-pair encoding marks word ends or drops merges")
        merges = model.get("merges")
        if not isinstance(merges, list):
            raise TokenizerNotTranslated("its byte-pair encoding has no list of merges")
        described = {
            "type": "bpe",
            "vocab": tokens,
            "merges": [list(merge) if isinstance(merge, (list, tuple)) else str(merge) for merge in merges],
            "byte_level": True,
            "add_prefix_space": bool(byte_level.get("add_prefix_space", False)),
        }
        for option in ("fuse_unk", "byte_fallback", "ignore_merges"):
            if model.get(option):
                described[option] = True  # what it does with a character it lacks, and with a word it holds whole
    else:
        raise TokenizerNotTranslated(
            f"a {kind or 'nameless'} tokenizer is not one of the kinds Kaimei reads (WordPiece and byte-level BPE)"
        )
    unknown = model.get("unk_token") or getattr(tokenizer, "unk_token", None)
    if isinstance(unknown, str) and unknown in tokens:
        described["unk"] = unknown
    prefix, suffix = _added_around(document.get("post_processor"))
    for name, found in (("prefix", prefix), ("suffix", suffix)):
        if any(token not in tokens for token in found):
            raise TokenizerNotTranslated("it adds a special token that is not in its vocabulary")
        if found:
            described[name] = found
    written = _added_tokens(added, tokens, vocab)
    if written:
        # Found whole in a text before it is split, as the original finds them.
        described["added_tokens"] = written
    special = {str(item.get("content")) for item in added if item.get("special")}
    padding: Dict[str, Any] = {}
    if isinstance(document.get("padding"), dict):
        padding = document["padding"]
    for name, attribute, known in (
        ("pad", "pad_token", _PAD_NAMES),
        ("mask", "mask_token", _MASK_NAMES),
        ("eos", "eos_token", _EOS_NAMES),
    ):
        token = getattr(tokenizer, attribute, None)
        if not isinstance(token, str) and name == "pad":
            token = padding.get("pad_token")
        if not isinstance(token, str):
            token = next((item for item in known if item in special), None)
        if isinstance(token, str) and token in tokens:
            described[name] = token
    try:
        built = build_tokenizer(described)
    except TokenizerError as exc:
        raise TokenizerNotTranslated(str(exc)) from exc
    _one_way_to_find_each(built)
    return described


def _one_way_to_find_each(built: Tokenizer) -> None:
    """Refuse a tokenizer with two added tokens that are the same text once normalized but are taken differently.

    Hugging Face looks for both with one pattern, and which of the two it takes is not the same from one process
    to the next (tokenizers 0.23: in some runs the first, in others the second), so no reading of the tokenizer
    gives its ids every time.
    """
    taken: Dict[str, Tuple[Any, ...]] = {}
    for item in built.added:
        if not item["normalized"]:
            continue
        text = built._normalize(item["content"])
        how = (item["id"], item["single_word"], item["lstrip"], item["rstrip"])
        if taken.setdefault(text, how) != how:
            raise TokenizerNotTranslated(
                f"two of its added tokens are {text!r} once normalized, and which of them Hugging Face takes is "
                "not the same from one run to the next"
            )


def _added_tokens(
    added: Sequence[Mapping[str, Any]], tokens: Mapping[str, int], own: Optional[Mapping[str, Any]] = None
) -> List[Dict[str, Any]]:
    """A tokenizer.json's ``added_tokens`` as Kaimei reads them: each with its id and how it is found.

    ``own`` is the vocabulary of the tokenizer's model (its WordPiece or BPE). A token added that it does not
    hold, as ``add_tokens`` adds one, is ``piece: false``: Hugging Face finds it whole in a text and never
    splits a word into it.
    """
    written: List[Dict[str, Any]] = []
    for item in added:
        content, index = item.get("content"), item.get("id")
        if not isinstance(content, str) or not content:
            continue
        if not isinstance(index, int) or isinstance(index, bool):
            index = tokens.get(content)
            if index is None:
                continue
        special = bool(item.get("special", False))
        entry: Dict[str, Any] = {
            "content": content,
            "id": int(index),
            "special": special,
            "single_word": bool(item.get("single_word", False)),
            "lstrip": bool(item.get("lstrip", False)),
            "rstrip": bool(item.get("rstrip", False)),
            # Hugging Face's own default: a special token is matched in the text as written.
            "normalized": bool(item.get("normalized", not special)),
        }
        if own is not None and content not in own:
            entry["piece"] = False
        written.append(entry)
    return written


def _added_samples(described: Mapping[str, Any], limit: int = 64) -> List[str]:
    """Texts that write the tokenizer's own added tokens beside words, spaces, digits and each other.

    The sample texts know the usual special tokens; these check every token a tokenizer adds (its own
    special tokens, and those added with ``add_tokens``), and how each is found: whole words only, the
    spaces around it taken into it, matched before or after the text is normalised.
    """
    texts: List[str] = []
    for item in list(described.get("added_tokens") or [])[:limit]:
        content = str(item.get("content") or "")
        if content:
            texts.extend(
                [
                    f"a {content} b",
                    f"x{content}y and 1{content}2",
                    f"{content}{content} {content.upper()} {content.lower()}",
                    f"  {content}  ends\t{content}\n",
                    f"word_{content}_word, ({content}).",
                ]
            )
    return texts


def _is_nothing(section: Any) -> bool:
    if section is None:
        return True
    if isinstance(section, dict) and section.get("type") == "Sequence":
        return all(_is_nothing(item) for item in section.get("normalizers") or section.get("pretokenizers") or [])
    return False


def _bert_normalizer(normalizer: Any) -> Tuple[bool, bool]:
    """``(lowercase, strip accents)`` of a BERT normalizer, which is the only one Kaimei's WordPiece follows."""
    if not isinstance(normalizer, dict) or normalizer.get("type") != "BertNormalizer":
        kind = normalizer.get("type") if isinstance(normalizer, dict) else "missing"
        raise TokenizerNotTranslated(f"WordPiece with a {kind} normalizer is not BERT's")
    if normalizer.get("clean_text") is False or normalizer.get("handle_chinese_chars") is False:
        raise TokenizerNotTranslated("its normalizer keeps control characters or does not split Chinese characters")
    lowercase = bool(normalizer.get("lowercase", True))
    strip = normalizer.get("strip_accents")
    return lowercase, lowercase if strip is None else bool(strip)


def _byte_level(pre: Any) -> Optional[Dict[str, Any]]:
    """The byte-level step of a pre-tokenizer, when that is all it does."""
    if not isinstance(pre, dict):
        return None
    if pre.get("type") == "ByteLevel":
        if pre.get("use_regex") is False:
            return None
        return pre
    if pre.get("type") == "Sequence":
        steps = [item for item in pre.get("pretokenizers") or [] if isinstance(item, dict)]
        if len(steps) == 1:
            return _byte_level(steps[0])
    return None


def _added_around(post: Any) -> Tuple[List[str], List[str]]:
    """The special tokens a post-processor puts before and after one text."""
    if not isinstance(post, dict):
        return [], []
    kind = post.get("type")
    if kind == "ByteLevel":
        return [], []
    if kind in ("BertProcessing", "RobertaProcessing"):
        first, last = post.get("cls"), post.get("sep")
        if not (isinstance(first, list) and first and isinstance(last, list) and last):
            raise TokenizerNotTranslated("its post-processor does not name its special tokens")
        return [str(first[0])], [str(last[0])]
    if kind == "TemplateProcessing":
        before: List[str] = []
        after: List[str] = []
        seen = False
        known: Dict[str, Any] = {}
        if isinstance(post.get("special_tokens"), dict):
            known = post["special_tokens"]
        for piece in post.get("single") or []:
            if not isinstance(piece, dict):
                continue
            if "Sequence" in piece:
                if seen:
                    raise TokenizerNotTranslated("its template uses the text twice")
                seen = True
            elif "SpecialToken" in piece:
                name = str((piece["SpecialToken"] or {}).get("id"))
                written = (known.get(name) or {}).get("tokens") if isinstance(known.get(name), dict) else None
                (after if seen else before).extend([str(token) for token in written] if written else [name])
        if not seen:
            raise TokenizerNotTranslated("its template has no place for the text")
        return before, after
    if kind == "Sequence":
        before, after = [], []
        for step in post.get("processors") or []:
            more_before, more_after = _added_around(step)
            before, after = [*more_before, *before], [*after, *more_after]
        return before, after
    raise TokenizerNotTranslated(f"its {kind} post-processor is not one Kaimei follows")


def _original(tokenizer: Any) -> Any:
    """The tokenizer to compare with: the HuggingFace one, never padding or cutting what it is given."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    source = backend if backend is not None else tokenizer
    rebuild = getattr(type(source), "from_str", None)
    writer = getattr(source, "to_str", None)
    if callable(rebuild) and callable(writer):
        try:
            copy = rebuild(writer())
            copy.no_padding()
            copy.no_truncation()
            return copy
        except Exception:  # noqa: BLE001 - a tokenizer that cannot be copied is compared as it is
            return source
    return source


def _same_ids(tokenizer: Any, described: Dict[str, Any], texts: Sequence[str]) -> None:
    """Raise unless Kaimei's tokenizer and the original give the same ids for every text."""
    ours = Tokenizer(described)
    if isinstance(tokenizer, Mapping):
        # The contents of a tokenizer.json: only the tokenizers package can say what it does to a text.
        try:
            from tokenizers import Tokenizer as Original
        except ImportError:
            raise TokenizerNotTranslated(
                "checking a tokenizer.json needs the tokenizers package (pip install tokenizers)"
            ) from None
        try:
            original = Original.from_str(json.dumps(dict(tokenizer)))
            original.no_padding()
            original.no_truncation()
        except Exception as exc:  # noqa: BLE001 - not a tokenizer.json after all
            raise TokenizerNotTranslated(f"it is not a tokenizer.json the tokenizers package reads: {str(exc)[:120]}") from exc
    else:
        original = _original(tokenizer)
    encode = getattr(original, "encode", None)
    if not callable(encode):
        raise TokenizerNotTranslated("it cannot be asked to encode a text, so the translation cannot be checked")
    for text in texts:
        try:
            encoded = encode(text)
        except Exception as exc:  # noqa: BLE001 - whatever it raises, the translation was not shown to match
            raise TokenizerNotTranslated(f"it could not encode a sample text: {str(exc)[:120]}") from exc
        theirs = [int(index) for index in getattr(encoded, "ids", encoded)]
        if ours.encode(text) != theirs:
            shown = text if len(text) <= 40 else text[:37] + "..."
            raise TokenizerNotTranslated(f"Kaimei's reading of it splits {shown!r} differently")


__all__ = ["SAMPLE_TEXTS", "TokenizerNotTranslated", "tokenizer_spec", "vocabulary_spec"]
