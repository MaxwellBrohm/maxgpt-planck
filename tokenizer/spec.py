"""Tokenizer family v0: the fixed parts (special tokens, pre-tokenizer, decoder, sizes).

Special tokens take the lowest ids, in this order, in every size of the family:
  0 <|pad|>  1 <|endoftext|>  2 <|system|>  3 <|user|>  4 <|assistant|>  5 <|end|>  6 <|tool|>
  7 <note>  8 </note>  9 <think>  10 </think>  11 <lookup>  12 </lookup>  13 <result>  14 </result>
Ids 0-6 are the harness's control tokens: the role tokens and END_TOKEN of harness/chat_template.py,
<|pad|> (from_pipeline.py looks it up for pad_id) and <|endoftext|> (rc12/planck_responder.py EOS_STRINGS,
the text-document EOT). They keep the ids of harness/toy_tokenizer.py and make_fake_data.SPECIAL, so
configs written against the toy tokenizer (pad_id 0, eot_id 1) stay valid. They are added as special
tokens (special=True): decode(..., skip_special_tokens=True) drops them.
Ids 7-14 are the reserved markup tags of PLAN Phase 1, spelled as pipeline/SPEC.txt and pipeline/events_c.py
already write lookups ("<lookup>q</lookup>", "<result>k: v</result>"). They are added tokens with
special=False: atomic in text like the control tokens, but kept by a default decode, because they are
content the model writes and graders read. The `special` flag is therefore the line between "control
token, must never come from text" and "markup tag, may appear in text". The tokenizer JSON alone does not
enforce the first half (a control-token string in text encodes to the control id), so every harness load
site sets encode_special_tokens=True (harness/data.py load_tokenizer, harness/from_pipeline.py _tokenizer;
the flag is not saved in the JSON).
Then 256 byte-level characters (ids 15-270, sorted by code point), then merges in merge order.

Pre-tokenizer: Split(PRETOKENIZE_REGEX, "isolated") then ByteLevel(add_prefix_space=False, use_regex=False).
The regex alternatives, first match wins:
  ['’](?i:s|t|re|ve|m|ll|d)(?![\\p{L}\\p{M}])   English contractions ('s 't 're 've 'm 'll 'd, straight or
                                               curly apostrophe) as their own piece: "Pearl's" -> Pearl + 's
   ?[\\p{L}\\p{M}]+                             a word (letters and combining marks) with at most one
                                               leading space (GPT-2 convention)
  \\p{N}                                       ONE digit per piece, no leading space (numbers are copied
                                               digit by digit; the space before a number is its own piece)
   ?[^\\s\\p{L}\\p{M}\\p{N}]                     ONE punctuation or symbol character, optional leading space
                                               (so "...", "?!" and "'s" never fuse with letters)
  \\s*[\\r\\n]+                                  a newline run with the whitespace before it
  \\s+(?!\\S)  and  \\s+                         other whitespace (the last space joins the next word)
Normalizer: none. decode(encode(s), skip_special_tokens=False) == s for every string (the library's default
decode drops control-token text). NFC is applied upstream by corpus/hygiene.py normalize(), which the corpus
readers run before corpus/make_tok_sample.py writes the sample, not inside the tokenizer, so text a user
types is never rewritten.
Byte fallback: every byte is in the base alphabet (byte-level BPE), so nothing is ever unknown;
the BPE model's own byte_fallback flag (sentencepiece-style <0xNN> tokens) stays off.
"""
from __future__ import annotations

CONTROL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
                  "<|tool|>"]
TAG_TOKENS = ["<note>", "</note>", "<think>", "</think>", "<lookup>", "</lookup>", "<result>", "</result>"]
SPECIALS = CONTROL_TOKENS + TAG_TOKENS
N_SPECIAL = len(SPECIALS)          # 15
N_BYTES = 256
N_BASE = N_SPECIAL + N_BYTES       # 271: ids below this are not created by merges

PRETOKENIZE_REGEX = (
    r"['’](?i:s|t|re|ve|m|ll|d)(?![\p{L}\p{M}])"
    r"| ?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

TOP_VOCAB = 32768
NESTED_SIZES = [2048, 4096, 8192, 16384, 32768]
FAMILY = "tok_v0"


def size_label(v: int) -> str:
    """2048 -> '2k'; sizes that are not a whole number of 1024 keep the number."""
    return f"{v // 1024}k" if v % 1024 == 0 else str(v)


def file_name(v: int, prefix: str = FAMILY) -> str:
    return f"{prefix}_{size_label(v)}.json"


def check_harness_spelling() -> None:
    """The role tokens here must be exactly harness/chat_template.py's (fails loudly if either drifts)."""
    import os
    import sys
    harness = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness")
    if harness not in sys.path:
        sys.path.append(harness)
    from chat_template import END_TOKEN, ROLE_TOKENS
    want = {ROLE_TOKENS[r] for r in ("system", "user", "assistant", "tool")} | {END_TOKEN}
    assert want <= set(CONTROL_TOKENS), f"harness role tokens {want} not all in {CONTROL_TOKENS}"
    assert set(ROLE_TOKENS.values()) | {END_TOKEN} == set(CONTROL_TOKENS[2:]), "extra or missing role token"


def added_tokens():
    from tokenizers import AddedToken
    return ([AddedToken(s, special=True, normalized=False) for s in CONTROL_TOKENS]
            + [AddedToken(s, special=False, normalized=False) for s in TAG_TOKENS])


def pre_tokenizer():
    from tokenizers import Regex, pre_tokenizers
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])


def empty_tokenizer():
    """An untrained BPE with the family's pre-tokenizer and decoder (no normalizer, no post-processor)."""
    from tokenizers import Tokenizer, decoders, models
    tok = Tokenizer(models.BPE(byte_fallback=False, ignore_merges=False))
    tok.pre_tokenizer = pre_tokenizer()
    tok.decoder = decoders.ByteLevel()
    return tok


def byte_alphabet() -> list[str]:
    from tokenizers import pre_tokenizers
    a = sorted(pre_tokenizers.ByteLevel.alphabet())
    assert len(a) == N_BYTES
    return a
