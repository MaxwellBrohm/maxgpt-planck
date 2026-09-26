"""The v0 pre-tokenizer: single digits, punctuation split from letters, contractions, whitespace."""
from __future__ import annotations

import re
import unicodedata

import pytest
from tokenizers import Regex, pre_tokenizers

import health
import spec

CASES = [
    ("Hello, world! I'm Pearl's owner; it's 2026-09-25 at 10:30.",
     ["Hello", ",", " world", "!", " I", "'m", " Pearl", "'s", " owner", ";", " it", "'s", " ", "2", "0", "2", "6",
      "-", "0", "9", "-", "2", "5", " at", " ", "1", "0", ":", "3", "0", "."]),
    ("don't 'data' ...\n\n  x", ["don", "'t", " '", "data", "'", " .", ".", ".", "\n\n", " ", " x"]),
    ("We’re here, they'LL go", ["We", "’re", " here", ",", " they", "'LL", " go"]),
    ("e\u0301t\xe9 caf\xe9 \u6771\u4eac \U0001F600!", ["e\u0301t\xe9", " caf\xe9", " \u6771\u4eac", " \U0001F600", "!"]),
    ("x=3.14159;y", ["x", "=", "3", ".", "1", "4", "1", "5", "9", ";", "y"]),
    ("a\t b  \r\n c", ["a", "\t", " b", "  \r\n", " c"]),
    ("<note>k: v</note>", ["<", "note", ">", "k", ":", " v", "<", "/", "note", ">"]),
    ("$1,000 (approx.)", ["$", "1", ",", "0", "0", "0", " (", "approx", ".", ")"]),
]


def _split(s):
    return [p for p, _ in pre_tokenizers.Split(Regex(spec.PRETOKENIZE_REGEX), "isolated").pre_tokenize_str(s)]


@pytest.mark.parametrize("text,pieces", CASES)
def test_pieces(text, pieces):
    assert _split(text) == pieces
    assert "".join(pieces) == text


@pytest.mark.parametrize("text,pieces", CASES)
def test_sequence_is_split_then_bytelevel(text, pieces):
    dec = health.byte_decoder()
    got = [bytes(dec[c] for c in p).decode("utf-8") for p, _ in spec.pre_tokenizer().pre_tokenize_str(text)]
    assert got == pieces


# a merge-made token may hold an apostrophe and letters only as (a prefix of) a contraction piece: 's, 'r, 're
CONTRACTION = re.compile(r"^['’](?:s|t|re?|ve?|m|ll?|d)$", re.I)


def test_trained_tokens_respect_the_split(family):
    """No merge-made token holds a digit, and none mixes letters with punctuation except a contraction."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(family["paths"][max(family["paths"])])
    dec = health.byte_decoder()
    seen_contraction = False
    for i in range(spec.N_BASE, tok.get_vocab_size()):
        s = health.token_bytes(tok, i, dec).decode("utf-8", "replace")
        assert not any(c.isdigit() for c in s), repr(s)
        core = s.lstrip(" ")
        cats = [unicodedata.category(c) for c in core if c != "\ufffd"]
        has_alpha, has_punct = any(k[0] in "LM" for k in cats), any(k[0] in "PS" for k in cats)
        if has_alpha and has_punct:
            assert CONTRACTION.match(core), repr(s)
            seen_contraction = True
    assert seen_contraction, "the synthetic corpus has contractions; expected at least one 's-style token"


def test_numbers_encode_digit_by_digit(family):
    from tokenizers import Tokenizer
    for p in family["paths"].values():
        tok = Tokenizer.from_file(p)
        e = tok.encode("call 5550142 now", add_special_tokens=False)
        digits = [tok.decode([i]) for i in e.ids if tok.decode([i]).strip().isdigit()]
        assert digits == list("5550142")
