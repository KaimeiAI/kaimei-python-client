"""Tokenizers: how a text feature becomes the token ids a language model reads.

A tokenizer is part of a model. Kaimei is not sent a tokenizer file; the uploader's client writes the
tokenizer down as plain JSON beside the model's structure (its kind, its vocabulary, its merges, the
special tokens it adds) and checks that this module, given that description, splits a set of sample
texts exactly as the original does. A tokenizer that does not pass is not stored, and the model then
takes token ids instead of text.

Supported kinds, which cover the tokenizers of most published language models:

* ``char``: one token per character.
* ``word``: lower-cased words and punctuation marks, looked up in a vocabulary.
* ``wordpiece``: BERT's: basic cleaning and splitting, then greedy longest-match pieces (``##ing``).
* ``bpe``: GPT-2's byte-level byte-pair encoding, and plain (character-level) BPE with end-of-word marks.
* ``pieces``: any other tokenizer (SentencePiece's, a hand-written one). Its vocabulary is written down, so
  every token can be named and ids turned back into text, but its way of splitting text is not: text is
  split where the tokenizer is, by the model's agent, or the model is given token ids.

Every kind but ``pieces`` may also have ``added_tokens``, as a Hugging Face tokenizer does: its special
tokens (``[MASK]``, ``<|endoftext|>``) and any added with ``add_tokens``. Each is found in the text and
kept whole before the text is cleaned and split, as Hugging Face's added vocabulary finds them, so a
``[MASK]`` written in a text is the mask token and not ``[``, ``mask``, ``]``. One that the tokenizer's own
vocabulary of pieces does not hold (``"piece": false``) is only ever found whole: no word is split into it.

The steps follow Hugging Face's ``tokenizers`` exactly, character classes included: GPT-2's pattern reads
Unicode's letters and numbers (not Python's ``\\w`` and ``\\d``), and BERT's normalizer lower-cases one
character at a time and reads its control, mark and punctuation characters from tables of an older
Unicode than Python's (the differences are written out below). A word of any length is split whole, as
Hugging Face splits it: byte-pair merges are queued by rank, so the work grows with a word's length times
its logarithm, and WordPiece never tries a piece longer than the longest in its vocabulary.

``decode`` turns ids back into text for display. It is exact for ``char`` and byte-level ``bpe`` and
close (spacing around pieces is rebuilt) for the others.
"""

from __future__ import annotations

import copy
import heapq
import re
import unicodedata
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Pattern, Sequence, Set, Tuple

#: Tokens a vocabulary may hold, and characters one token may have.
MAX_VOCABULARY = 400_000
MAX_TOKEN_CHARS = 256
#: Merges a byte-pair encoding may hold.
MAX_MERGES = 400_000
#: Tokens a tokenizer may add to its vocabulary's (its special tokens, and any added with ``add_tokens``).
MAX_ADDED_TOKENS = 10_000
#: Characters of one text that are tokenized; the rest is cut off before any work is done on it.
MAX_TEXT_CHARS = 200_000
#: Words up to this many characters keep their pieces for the next time they are met.
CACHED_WORD_CHARS = 256

KINDS = ("char", "word", "wordpiece", "bpe", "pieces")
#: Splits a text into token ids somewhere else: ``(text, most tokens) -> ids``, special tokens included.
Encoder = Callable[[str, Optional[int]], Any]


class TokenizerError(ValueError):
    """A tokenizer description that cannot be used, or a text it cannot encode."""


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's printable stand-in for every byte, so a vocabulary can be written as text."""
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    mapped = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {byte: chr(code) for byte, code in zip(visible, mapped)}


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {char: byte for byte, char in _BYTE_ENCODER.items()}
_WORD_SPLIT = re.compile(r"\w+|[^\w\s]", re.UNICODE)
#: How a ``word`` tokenizer finds its words, by name. Only these: a pattern sent by a caller could be written to
#: run without end. ``words`` is a word or a punctuation mark; ``sklearn`` is scikit-learn's default token pattern
#: (two or more word characters), what its CountVectorizer and TfidfVectorizer split text with.
_WORD_PATTERNS = {"words": _WORD_SPLIT, "sklearn": re.compile(r"(?u)\b\w\w+\b")}
#: The most stop words a tokenizer may drop.
MAX_STOP_WORDS = 10_000
_SURROGATES = re.compile("[\ud800-\udfff]")


def _code_points(ranges: str) -> FrozenSet[int]:
    """``"61d 2e43-2e4f"`` as the set of code points it names."""
    found: Set[int] = set()
    for item in ranges.split():
        low, _, high = item.partition("-")
        found.update(range(int(low, 16), int(high or low, 16) + 1))
    return frozenset(found)


# Hugging Face's BERT normalizer and pre-tokenizer read a character's category from tables of Unicode 8.0,
# and Python's are newer. These are the characters on which they disagree, found by asking tokenizers 0.23
# about every code point: whatever Python's Unicode, Kaimei's reading then agrees with Hugging Face's.
#: Format characters Hugging Face does not count as control characters, so does not clean away.
_HF_FORMAT_KEPT = _code_points("890-891 8e2 110cd 13430-1343f")
#: Non-spacing marks it does not know as such, so does not strip with the accents.
_HF_MARKS_KEPT = _code_points(
    "7fd 898-89f 8ca-8e1 9fe afa-aff b55 c04 c3c d00 d3b-d3c d81 eba ece 180f 1885-1886 1abf-1ace 1df6-1dfb a82c "
    "a8c5 a8ff a9bd 10d24-10d27 10eab-10eac 10efd-10eff 10f46-10f50 10f82-10f85 11070 11073-11074 110c2 111c9 "
    "111cf 1123e 11241 1133b 11438-1143f 11442-11444 11446 1145e 1182f-11837 11839-1183a 1193b-1193c 1193e 11943 "
    "119d4-119d7 119da-119db 119e0 11a01-11a0a 11a33-11a38 11a3b-11a3e 11a47 11a51-11a56 11a59-11a5b 11a8a-11a96 "
    "11a98-11a99 11c30-11c36 11c38-11c3d 11c3f 11c92-11ca7 11caa-11cb0 11cb2-11cb3 11cb5-11cb6 11d31-11d36 11d3a "
    "11d3c-11d3d 11d3f-11d45 11d47 11d90-11d91 11d95 11d97 11ef3-11ef4 11f00-11f01 11f36-11f3a 11f40 11f42 13440 "
    "13447-13455 16f4f 16fe4 1cf00-1cf2d 1cf30-1cf46 1e000-1e006 1e008-1e018 1e01b-1e021 1e023-1e024 1e026-1e02a "
    "1e08f 1e130-1e136 1e2ae 1e2ec-1e2ef 1e4ec-1e4ef 1e944-1e94a"
)
#: A character it strips as a non-spacing mark that Unicode now calls a spacing one.
_HF_MARKS_STRIPPED = _code_points("1734")
#: Punctuation it does not know as such, so does not split off.
_HF_NOT_PUNCTUATION = _code_points(
    "61d 9fd a76 c77 c84 1b7d-1b7e 2e43-2e4f 2e52-2e5d 10ead 10f55-10f59 10f86-10f89 1144b-1144f 1145a-1145b 1145d "
    "11660-1166c 116b9 1183b 11944-11946 119e2 11a3f-11a46 11a9a-11a9c 11a9e-11aa2 11b00-11b09 11c41-11c45 "
    "11c70-11c71 11ef7-11ef8 11f43-11f4f 11fff 12ff1-12ff2 16e97-16e9a 16fe2 1e95e-1e95f"
)
#: Characters it still splits off as punctuation that Unicode no longer calls punctuation.
_HF_PUNCTUATION = _code_points("166d 111c9")
#: Unicode's White_Space, which is what Hugging Face strips beside an added token.
_WHITE_SPACE = _code_points("9-d 20 85 a0 1680 2000-200a 2028-2029 202f 205f 3000")
#: Oniguruma's ``\s``, which GPT-2's pattern is run with: White_Space less the separator controls it adds.
_ONIG_SPACES = "\t\n\x0b\x0c\r\x85 \xa0  -     　"
#: Letters by Unicode's Other_Alphabetic that are symbols by category (circled and squared letters).
_ALPHABETIC_SYMBOLS = _code_points("24b6-24e9 1f130-1f149 1f150-1f169 1f170-1f189")

_GPT2_SPLIT: Optional[Pattern[str]] = None


def _gpt2_split() -> Pattern[str]:
    """GPT-2's pre-tokenization: ``'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+``.

    Python's ``re`` has no ``\\p{L}`` or ``\\p{N}``, and its ``\\w`` and ``\\d`` are other classes (``\\w``
    holds ``_``, marks and numbers such as ``²``), so letters and numbers are spelled out from
    :mod:`unicodedata`, once, the first time a byte-level text is split.
    """
    global _GPT2_SPLIT
    if _GPT2_SPLIT is None:
        runs: Dict[str, List[Tuple[int, int]]] = {"L": [], "N": []}
        previous, start = "", 0
        for code in range(0x110001):
            major = unicodedata.category(chr(code))[0] if code < 0x110000 else ""
            if major != previous:
                if previous in runs:
                    runs[previous].append((start, code - 1))
                previous, start = major, code
        letters, numbers = (
            "".join(
                re.escape(chr(low)) + ("" if low == high else "-" + re.escape(chr(high))) for low, high in runs[key]
            )
            for key in ("L", "N")
        )
        spaces = _ONIG_SPACES
        _GPT2_SPLIT = re.compile(
            f"'s|'t|'re|'ve|'m|'ll|'d| ?[{letters}]+| ?[{numbers}]+| ?[^{spaces}{letters}{numbers}]+"
            f"|[{spaces}]+(?![^{spaces}])|[{spaces}]+"
        )
    return _GPT2_SPLIT


def _lower(text: str) -> str:
    """Lower case one character at a time, as Hugging Face does (``str.lower`` ends a word's ``Σ`` in ``ς``)."""
    if text.isascii():
        return text.lower()
    return "".join(char.lower() for char in text)


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    if code < 128:
        return False
    if code in _HF_PUNCTUATION:
        return True
    return unicodedata.category(char).startswith("P") and code not in _HF_NOT_PUNCTUATION


def _is_mark(char: str) -> bool:
    """A non-spacing mark, as BERT's normalizer strips them with the accents."""
    code = ord(char)
    if code in _HF_MARKS_STRIPPED:
        return True
    return unicodedata.category(char) == "Mn" and code not in _HF_MARKS_KEPT


def _is_word(char: str) -> bool:
    """A character of a word, as a ``single_word`` added token may not have one beside it (regex ``\\w``)."""
    code = ord(char)
    category = unicodedata.category(char)
    return (
        category[0] in "LM" or category in ("Nd", "Nl", "Pc") or code in (0x200C, 0x200D) or code in _ALPHABETIC_SYMBOLS
    )


def _is_cjk(code: int) -> bool:
    # Hugging Face's ranges, which start the fifth extension at U+2B920, not BERT's U+2B820.
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B920 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


class _AddedTokens:
    """Tokens found whole in a text before it is split, as Hugging Face's added vocabulary finds them.

    The longest token that starts leftmost is taken and the search goes on after it. A ``single_word``
    token is only taken with no word character either side of it; ``lstrip`` and ``rstrip`` take the
    whitespace before and after it into the token.
    """

    def __init__(self, tokens: Sequence[Tuple[str, int, bool, bool, bool]]) -> None:
        self.found: Dict[str, Tuple[int, bool, bool, bool]] = {}
        for text, index, single_word, lstrip, rstrip in tokens:
            if text:
                self.found.setdefault(text, (index, single_word, lstrip, rstrip))
        # Longest first: Python's alternation takes the first that matches, so this is the longest.
        ordered = sorted(self.found, key=len, reverse=True)
        self.pattern = re.compile("|".join(re.escape(text) for text in ordered)) if ordered else None

    def split(self, text: str) -> List[Tuple[str, Optional[int]]]:
        """``text`` as ``(piece, None)`` stretches between ``(token text, id)`` tokens, in order."""
        if self.pattern is None or not text:
            return [(text, None)]
        pieces: List[Tuple[str, Optional[int]]] = []
        offset = 0
        for match in self.pattern.finditer(text):
            start, stop = match.span()
            if start < offset:
                continue  # in the whitespace the token before took
            index, single_word, lstrip, rstrip = self.found[match.group()]
            if single_word and (
                (start > 0 and _is_word(text[start - 1])) or (stop < len(text) and _is_word(text[stop]))
            ):
                continue
            if lstrip:
                while start > offset and ord(text[start - 1]) in _WHITE_SPACE:
                    start -= 1
            if rstrip:
                while stop < len(text) and ord(text[stop]) in _WHITE_SPACE:
                    stop += 1
            if offset < start:
                pieces.append((text[offset:start], None))
            pieces.append((text[start:stop], index))
            offset = stop
        if offset < len(text):
            pieces.append((text[offset:], None))
        return pieces


class Tokenizer:
    """A tokenizer built from its JSON description. Build one with :func:`build_tokenizer`."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        kind = str(spec.get("type") or "").lower()
        if kind not in KINDS:
            raise TokenizerError(f"Tokenizer type '{spec.get('type')}' is not one of {', '.join(KINDS)}.")
        self.kind = kind
        vocab = spec.get("vocab")
        if isinstance(vocab, dict):
            try:
                pairs = sorted(((int(index), str(token)) for token, index in vocab.items()))
            except (TypeError, ValueError) as exc:
                raise TokenizerError("A tokenizer's vocab must map tokens to whole-number ids.") from exc
            size = (pairs[-1][0] + 1) if pairs else 0
            if size > MAX_VOCABULARY or (pairs and pairs[0][0] < 0):
                raise TokenizerError("A tokenizer's vocab has ids out of range.")
            tokens = [""] * size
            for index, token in pairs:
                tokens[index] = token
        elif isinstance(vocab, list):
            tokens = [str(token) for token in vocab]
        else:
            raise TokenizerError("A tokenizer needs a vocab: a list of tokens, or tokens mapped to ids.")
        if not tokens or len(tokens) > MAX_VOCABULARY:
            raise TokenizerError(f"A tokenizer's vocab needs 1 to {MAX_VOCABULARY:,} tokens.")
        if any(len(token) > MAX_TOKEN_CHARS for token in tokens):
            raise TokenizerError(f"A token is longer than {MAX_TOKEN_CHARS} characters.")
        self.tokens: List[str] = tokens
        self.ids: Dict[str, int] = {}
        for index, token in enumerate(tokens):
            self.ids.setdefault(token, index)
        self.lowercase = bool(spec.get("lowercase", kind in ("word", "wordpiece")))
        self.strip_accents = bool(spec.get("strip_accents", self.lowercase and kind == "wordpiece"))
        pattern = str(spec.get("pattern") or "words")
        if kind == "word" and pattern not in _WORD_PATTERNS:
            raise TokenizerError(f"A word tokenizer's pattern is one of {', '.join(_WORD_PATTERNS)}.")
        self.pattern = pattern
        stops = spec.get("stop_words") or []
        if not isinstance(stops, list) or len(stops) > MAX_STOP_WORDS:
            raise TokenizerError(f"stop_words is a list of at most {MAX_STOP_WORDS:,} words.")
        #: Words a ``word`` tokenizer leaves out before anything else sees them.
        self.stop_words = frozenset(str(word) for word in stops)
        self.unk = self._special(spec, "unk")
        self.pad = self._special(spec, "pad")
        self.mask = self._special(spec, "mask")
        #: The token that ends a text, when the model has one: generation stops at it.
        self.eos = self._special(spec, "eos")
        self.prefix = [self._id_of(token) for token in spec.get("prefix") or []]
        self.suffix = [self._id_of(token) for token in spec.get("suffix") or []]
        self.continuation = str(spec.get("continuation", "##"))
        self.byte_level = bool(spec.get("byte_level", kind == "bpe"))
        #: What a vocabulary of pieces writes for the space before a word (SentencePiece's is "▁").
        self.space = str(spec.get("space") or "")[:8]
        #: For ``pieces``: whoever can split text with the tokenizer itself (see :meth:`with_encoder`).
        self.encoder: Optional[Encoder] = None
        self.add_prefix_space = bool(spec.get("add_prefix_space", False))
        self.end_of_word = str(spec.get("end_of_word") or "")
        # A longer word is one unknown token, as Hugging Face's WordPiece makes it (max_input_chars_per_word).
        self.max_word_chars = max(1, int(spec.get("max_word_chars", 100)))
        #: Hugging Face's BPE options: a run of unknown characters as one unknown token, an unknown character
        #: written as its bytes (``<0x41>``), and a word the vocabulary holds whole taken as it is.
        self.fuse_unk = bool(spec.get("fuse_unk", False))
        self.byte_fallback = bool(spec.get("byte_fallback", False))
        self.ignore_merges = bool(spec.get("ignore_merges", False))
        self.ranks: Dict[Tuple[str, str], int] = {}
        if kind == "bpe":
            merges = spec.get("merges") or []
            if not isinstance(merges, list) or len(merges) > MAX_MERGES:
                raise TokenizerError(f"A byte-pair encoding needs a list of at most {MAX_MERGES:,} merges.")
            for rank, merge in enumerate(merges):
                pair = merge.split(" ") if isinstance(merge, str) else list(merge)
                if len(pair) != 2:
                    raise TokenizerError("Each merge is a pair of tokens.")
                # A pair listed twice has the rank of its last listing, as Hugging Face reads it.
                self.ranks[(str(pair[0]), str(pair[1]))] = rank
        self._cache: Dict[str, List[str]] = {}
        self.added = self._added_tokens(spec.get("added_tokens"))
        #: Added tokens the vocabulary of pieces does not hold (Hugging Face's model does not know them): each is
        #: found whole in a text, and no word is split into it.
        self._whole_only = frozenset(item["content"] for item in self.added if not item["piece"])
        if kind == "bpe":
            self._merges_fit()
        #: The most characters a WordPiece piece has: no longer stretch of a word is tried.
        self._longest_piece = max((len(token) for token in self.ids), default=1) if kind == "wordpiece" else 0
        # Found in the text as written, and then in the text once it is normalised (cleaned, lower-cased).
        self._raw_added = _AddedTokens(
            [
                (item["content"], item["id"], item["single_word"], item["lstrip"], item["rstrip"])
                for item in self.added
                if not item["normalized"]
            ]
        )
        self._normalized_added = _AddedTokens(
            [
                (self._normalize(item["content"]), item["id"], item["single_word"], item["lstrip"], item["rstrip"])
                for item in self.added
                if item["normalized"]
            ]
        )

    def _merges_fit(self) -> None:
        """Refuse a merge that joins, or makes, a piece the vocabulary does not hold, as Hugging Face refuses one.

        A word is split into the characters the vocabulary holds before any merge is tried, and each piece is
        looked up once the merges are done. A merge that makes a piece the vocabulary lacks leaves that piece
        wherever no later merge joins it to something the vocabulary holds: such a word is read as unknown, where
        the tokenizer the merges came from would have split it into pieces it knows, and nothing would say so.
        """
        for first, second in self.ranks:
            for piece in (first, second, first + second):
                if self._piece(piece) is None:
                    raise TokenizerError(
                        f"The merge of {first!r} and {second!r} needs {piece!r}, which is not a token of the "
                        "vocabulary. Each piece a merge joins, and the piece it makes, has to be in the vocabulary."
                    )

    def _special(self, spec: Dict[str, Any], name: str) -> Optional[int]:
        token = spec.get(name)
        if token is None:
            return None
        return self._id_of(token)

    def _id_of(self, token: Any) -> int:
        if isinstance(token, int) and not isinstance(token, bool):
            if not 0 <= token < len(self.tokens):
                raise TokenizerError(f"Token id {token} is outside the vocabulary.")
            return token
        index = self.ids.get(str(token))
        if index is None:
            raise TokenizerError(f"Special token {token!r} is not in the vocabulary.")
        return index

    def _added_tokens(self, raw: Any) -> List[Dict[str, Any]]:
        """``added_tokens``: ``[{"content", "id", "special", "single_word", "lstrip", "rstrip", "normalized"}]``.

        ``id`` defaults to the content's place in the vocabulary, and ``normalized`` to true for a token
        that is not special, as in Hugging Face's ``AddedToken``. ``"piece": false`` marks a token the
        tokenizer's own vocabulary of pieces (its WordPiece or BPE) does not hold, one added with
        ``add_tokens``: Hugging Face finds it whole in a text and never splits a word into it.
        """
        if raw is None:
            return []
        if not isinstance(raw, list) or len(raw) > MAX_ADDED_TOKENS:
            raise TokenizerError(f"added_tokens must be a list of at most {MAX_ADDED_TOKENS:,} tokens.")
        found: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise TokenizerError("Each added token is an object with its content.")
            content = item.get("content")
            if not isinstance(content, str) or not content or len(content) > MAX_TOKEN_CHARS:
                raise TokenizerError(f"An added token's content is text of 1 to {MAX_TOKEN_CHARS} characters.")
            special = bool(item.get("special", False))
            found.append(
                {
                    "content": content,
                    "id": self._id_of(item["id"] if item.get("id") is not None else content),
                    "special": special,
                    "single_word": bool(item.get("single_word", False)),
                    "lstrip": bool(item.get("lstrip", False)),
                    "rstrip": bool(item.get("rstrip", False)),
                    "normalized": bool(item.get("normalized", not special)),
                    "piece": bool(item.get("piece", True)),
                }
            )
        return found

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def can_encode(self) -> bool:
        """Whether text can be split here. A vocabulary of pieces alone names tokens and cannot make them."""
        return self.kind != "pieces" or self.encoder is not None

    def with_encoder(self, encoder: Encoder) -> Tokenizer:
        """This tokenizer, with its text split by ``encoder``. The vocabulary is shared, not copied."""
        other = copy.copy(self)
        other.encoder = encoder
        return other

    # --- encoding ---

    def _elsewhere(self, text: str, max_length: Optional[int]) -> List[int]:
        if self.encoder is None:
            raise TokenizerError(
                "This model's tokenizer is not one Kaimei can run, so text cannot be split into tokens here. "
                "Give the token ids, or run the model's agent with its tokenizer, which splits text itself."
            )
        found = self.encoder(text, max_length)
        if not isinstance(found, (list, tuple)) or len(found) > MAX_TEXT_CHARS:
            raise TokenizerError("The model's agent did not return a list of token ids for this text.")
        ids: List[int] = []
        for item in found:
            if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < len(self.tokens):
                raise TokenizerError("The model's agent returned a token id that is not in the model's vocabulary.")
            ids.append(item)
        return ids if max_length is None else ids[: max(int(max_length), 0)]

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Token ids for ``text``, with the model's special tokens added, cut to ``max_length`` tokens.

        As Hugging Face does it: the added tokens written in the text are found first, then what is left
        between them is normalised, found again for the added tokens that are matched normalised, split
        into words and turned into tokens; the special tokens that open and close a text go around it.
        """
        text = str(text)[:MAX_TEXT_CHARS]
        if _SURROGATES.search(text):
            text = _SURROGATES.sub("�", text)  # half a character, which no tokenizer is given
        if self.kind == "pieces":
            # Whoever runs the tokenizer itself adds its special tokens too.
            return self._elsewhere(text, max_length)
        ids: List[int] = []
        for piece, index in self._raw_added.split(text):
            if index is not None:
                ids.append(index)
                continue
            for part, found in self._normalized_added.split(self._normalize(piece)):
                if found is not None:
                    ids.append(found)
                elif part:
                    ids.extend(token for token in self._model(part) if token is not None)
        if max_length is not None:
            room = max(int(max_length) - len(self.prefix) - len(self.suffix), 0)
            ids = ids[:room]
        return [*self.prefix, *ids, *self.suffix]

    def _normalize(self, text: str) -> str:
        """What the tokenizer does to a text before splitting it: BERT's cleaning, or lower-casing."""
        if self.kind == "wordpiece":
            return self._bert_normalize(text)
        if self.kind == "bpe" and self.byte_level:
            return text
        if self.kind == "word" and self.pattern == "sklearn":
            # scikit-learn's own steps: str.lower (which ends a word's capital sigma in a final one), then
            # strip_accents="unicode": NFKD, with the combining marks left out.
            text = text.lower() if self.lowercase else text
        else:
            text = _lower(text) if self.lowercase else text
        if self.kind == "word" and self.strip_accents and not text.isascii():
            text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
        return text

    def _model(self, text: str) -> List[Optional[int]]:
        """A normalised stretch of text with no added token in it, split into words and then tokens."""
        if self.kind == "char":
            return [self._lookup(char) for char in text]
        if self.kind == "word":
            words = _WORD_PATTERNS[self.pattern].findall(text)
            return [self._lookup(word) for word in words if word not in self.stop_words]
        if self.kind == "wordpiece":
            return self._wordpiece(text)
        return self._bpe(text)

    def _lookup(self, token: str) -> Optional[int]:
        index = self.ids.get(token)
        if index is None:
            return self.unk  # None drops a token the vocabulary has no stand-in for
        return index

    def _piece(self, token: str) -> Optional[int]:
        """The id of ``token`` as a piece a word can be split into: not an added token the pieces do not hold."""
        if token in self._whole_only:
            return None
        return self.ids.get(token)

    def _bert_normalize(self, text: str) -> str:
        """BERT's normalizer, in Hugging Face's order: clean, space out Chinese characters, strip accents, lower."""
        cleaned: List[str] = []
        for char in text:
            code = ord(char)
            if code < 128:
                if char in "\t\n\r":
                    cleaned.append(" ")
                elif code >= 32 and code != 127:
                    cleaned.append(char)
                continue  # the other controls are cleaned away
            category = unicodedata.category(char)
            if code == 0xFFFD or (category in ("Cc", "Cf", "Co") and code not in _HF_FORMAT_KEPT):
                continue
            if category in ("Zs", "Zl", "Zp"):
                cleaned.append(" ")
            elif _is_cjk(code):
                cleaned.append(f" {char} ")
            else:
                cleaned.append(char)
        text = "".join(cleaned)
        if self.strip_accents and not text.isascii():
            text = "".join(char for char in unicodedata.normalize("NFD", text) if not _is_mark(char))
        return _lower(text) if self.lowercase else text

    def _basic_words(self, text: str) -> List[str]:
        """BERT's pre-tokenizer on a normalised text: split on whitespace, and every punctuation mark alone."""
        words: List[str] = []
        for word in text.split():
            piece: List[str] = []
            for char in word:
                if _is_punctuation(char):
                    if piece:
                        words.append("".join(piece))
                        piece = []
                    words.append(char)
                else:
                    piece.append(char)
            if piece:
                words.append("".join(piece))
        return words

    def _wordpiece(self, text: str) -> List[Optional[int]]:
        ids: List[Optional[int]] = []
        for word in self._basic_words(text):
            if len(word) > self.max_word_chars:
                ids.append(self.unk)
                continue
            start = 0
            pieces: List[int] = []
            while start < len(word):
                # The longest piece that starts here, as Hugging Face looks for it from the word's end: a stretch
                # longer than the longest piece there is cannot be one, so the search starts at that length.
                end = min(len(word), start + self._longest_piece)
                found = None
                while start < end:
                    candidate = word[start:end]
                    if start > 0:
                        candidate = self.continuation + candidate
                    found = self._piece(candidate)
                    if found is not None:
                        break
                    end -= 1
                if found is None:
                    pieces = []
                    break
                pieces.append(found)
                start = end
            if pieces:
                ids.extend(pieces)
            else:
                ids.append(self.unk)
        return ids

    def _merge(self, symbols: List[str]) -> List[str]:
        """Apply the learned merges to one word, as Hugging Face's BPE model does, however long the word is.

        The queued pair of lowest rank is merged first, the leftmost of equal ones, one merge at a time, and the
        pairs a merge makes with its neighbours join the queue. (Merging every place the best pair occurs at
        once is not always the same: two merges can make one token.) A merge takes a symbol away, so a word of
        ``n`` symbols is done in at most ``n`` merges and ``3n`` steps of the queue.
        """
        count = len(symbols)
        if count < 2:
            return symbols
        ranks = self.ranks
        pieces = list(symbols)
        live = [True] * count  # False once merged into the piece before it
        after = list(range(1, count + 1))  # where the next live piece is; count after the last
        before = list(range(-1, count - 1))  # where the live piece before is; -1 before the first
        queue: List[Tuple[int, int]] = []  # (rank, where the pair's first piece is)
        for index in range(count - 1):
            rank = ranks.get((pieces[index], pieces[index + 1]))
            if rank is not None:
                queue.append((rank, index))
        heapq.heapify(queue)
        while queue:
            rank, index = heapq.heappop(queue)
            following = after[index]
            if not live[index] or following >= count or ranks.get((pieces[index], pieces[following])) != rank:
                continue  # a neighbour was merged first, so the pair queued is no longer there
            merged = pieces[index] + pieces[following]
            pieces[index] = merged
            live[following] = False
            beyond = after[following]
            after[index] = beyond
            if beyond < count:
                before[beyond] = index
                rank = ranks.get((merged, pieces[beyond]))
                if rank is not None:
                    heapq.heappush(queue, (rank, index))
            previous = before[index]
            if previous >= 0:
                rank = ranks.get((pieces[previous], merged))
                if rank is not None:
                    heapq.heappush(queue, (rank, previous))
        return [piece for piece, kept in zip(pieces, live) if kept]

    def _bpe_word(self, word: str) -> List[str]:
        """One word's pieces, as Hugging Face's BPE model makes them.

        Each character (each byte, for a byte-level vocabulary) is a symbol, and the merges join them. A
        character the vocabulary of pieces does not hold is written as its bytes when the tokenizer falls back
        to them, or else is the unknown token (one for a run of them when the tokenizer fuses them), or with
        none is left out, and the symbols either side of it may then merge.
        """
        chars = [_BYTE_ENCODER[byte] for byte in word.encode("utf-8")] if self.byte_level else list(word)
        if self.ignore_merges and self._piece("".join(chars)) is not None:
            return ["".join(chars)]
        if self.end_of_word and chars and not self.byte_level:
            chars[-1] = chars[-1] + self.end_of_word
        unknown = self.tokens[self.unk] if self.unk is not None else None
        symbols: List[str] = []
        waiting: Optional[str] = None  # an unknown token, added before the next character the vocabulary holds
        for char in chars:
            if self._piece(char) is not None:
                if waiting is not None:
                    symbols.append(waiting)
                    waiting = None
                symbols.append(char)
                continue
            if self.byte_fallback:
                spelled = [f"<0x{byte:02X}>" for byte in char.encode("utf-8")]
                if all(self._piece(piece) is not None for piece in spelled):
                    symbols.extend(spelled)  # ahead of an unknown token still waiting, where Hugging Face puts them
                    continue
            if unknown is not None:
                if waiting is not None and not self.fuse_unk:
                    symbols.append(waiting)
                waiting = unknown
        if waiting is not None:
            symbols.append(waiting)
        return self._merge(symbols)

    def _bpe(self, text: str) -> List[Optional[int]]:
        ids: List[Optional[int]] = []
        if self.byte_level:
            # Hugging Face adds the space to each stretch between added tokens that does not start with one.
            if self.add_prefix_space and text and not text.startswith(" "):
                text = " " + text
            words = _gpt2_split().findall(text)
        else:
            words = text.split()
        for word in words:
            pieces = self._cache.get(word)
            if pieces is None:
                pieces = self._bpe_word(word)
                if len(self._cache) < 50_000 and len(word) <= CACHED_WORD_CHARS:
                    self._cache[word] = pieces
            ids.extend(self._lookup(piece) for piece in pieces)
        return ids

    # --- decoding ---

    def token(self, index: int) -> str:
        """The text of one token as a person reads it (a leading space shown as a space)."""
        if not 0 <= int(index) < len(self.tokens):
            return f"#{int(index)}"
        token = self.tokens[int(index)]
        if self.kind in ("bpe", "pieces") and self.byte_level:
            try:
                return bytes(_BYTE_DECODER[char] for char in token).decode("utf-8", "replace")
            except KeyError:
                return token
        if self.kind == "bpe" and self.end_of_word and token.endswith(self.end_of_word):
            return token[: -len(self.end_of_word)] + " "
        if self.kind == "pieces" and self.space:
            return token.replace(self.space, " ")
        return token

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        special = {index for index in (self.pad, self.mask, *self.prefix, *self.suffix) if index is not None}
        kept = [int(index) for index in ids if int(index) >= 0 and not (skip_special and int(index) in special)]
        if self.kind == "char":
            return "".join(self.tokens[i] if i < len(self.tokens) else "" for i in kept)
        if self.kind in ("bpe", "pieces") and self.byte_level:
            data = bytearray()
            for index in kept:
                if index < len(self.tokens):
                    data.extend(_BYTE_DECODER.get(char, ord("?")) for char in self.tokens[index])
            return data.decode("utf-8", "replace")
        if self.kind == "bpe":
            return "".join(self.token(index) for index in kept).strip()
        if self.kind == "pieces":
            # Pieces carry their own spacing when the vocabulary marks it; otherwise they are set side by side.
            joined = ("" if self.space else " ").join(self.token(index) for index in kept)
            return joined.strip()
        words: List[str] = []
        for index in kept:
            token = self.tokens[index] if index < len(self.tokens) else ""
            if self.kind == "wordpiece" and token.startswith(self.continuation) and words:
                words[-1] += token[len(self.continuation) :]
            else:
                words.append(token)
        text = " ".join(words)
        return re.sub(r" ([.,!?;:%)\]}])", r"\1", text)

    def describe(self) -> Dict[str, Any]:
        """What a client needs to label tokens, without the vocabulary itself."""
        return {
            "type": self.kind,
            "vocab_size": self.size,
            # False when text has to be split somewhere else and nothing is there to do it: token ids only.
            "encodes": self.can_encode,
            "lowercase": self.lowercase,
            "pad_id": self.pad,
            "unk_id": self.unk,
            "mask_id": self.mask,
            "prefix": list(self.prefix),
            "suffix": list(self.suffix),
            **({"pattern": self.pattern} if self.kind == "word" and self.pattern != "words" else {}),
            **({"stop_words": len(self.stop_words)} if self.stop_words else {}),
        }


def build_tokenizer(spec: Any) -> Tokenizer:
    """A :class:`Tokenizer` from its JSON description, or a :class:`TokenizerError` saying what is wrong with it."""
    if not isinstance(spec, dict):
        raise TokenizerError("A tokenizer must be an object.")
    try:
        return Tokenizer(spec)
    except TokenizerError:
        raise
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError, re.error) as exc:
        raise TokenizerError(f"This tokenizer is malformed: {str(exc)[:200]}") from exc
