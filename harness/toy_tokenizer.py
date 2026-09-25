"""A toy byte-level tokenizer for plumbing tests (built in memory: no download, no training).

Special ids match make_fake_data.SPECIAL, so a model trained on the fake shards and this
tokenizer agree on the role tokens: 0 <|pad|>, 1 <|endoftext|> (the text EOT), 2 <|system|>,
3 <|user|>, 4 <|assistant|>, 5 <|end|>, 6 <|tool|>, 7 <|reserved_7|>. Ids 8..263 are the 256
GPT-2 byte-level characters with no merges, so every string encodes (about one token per
byte) and decode(encode(s)) == s. VOCAB = 264. It is NOT a Planck tokenizer (none exists yet).

  python toy_tokenizer.py OUT/tokenizer.json      writes it for CLI smoke runs
"""
from __future__ import annotations

import sys

SPECIALS = ["<|pad|>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            "<|tool|>", "<|reserved_7|>"]
VOCAB = len(SPECIALS) + 256


def build():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    assert len(alphabet) == 256
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    for ch in alphabet:
        vocab[ch] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    assert tok.get_vocab_size() == VOCAB
    assert all(tok.token_to_id(s) == i for i, s in enumerate(SPECIALS))
    return tok


def write(path: str) -> str:
    build().save(path)
    return path


if __name__ == "__main__":
    print(write(sys.argv[1]))
