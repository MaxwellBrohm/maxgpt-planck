"""Tiny deterministic synthetic text for the tokenizer tests (no real data)."""
from __future__ import annotations

import random

# the tiny family: trained once at TOP, truncated to SIZES; SEP is trained separately for the P-098 check.
# (Kept here, not in conftest.py: corpus/tests has its own conftest module.)
TOP = 1024
SIZES = [300, 400, 512, 768]
SEP = 512

SYL = ["ka", "lo", "mi", "ter", "an", "sho", "ru", "vel", "po", "dan", "ist", "or", "ne", "qua", "zi", "bel",
       "tor", "en", "ma", "rik", "sa", "the", "ing", "ly"]
PUNCT = [",", ".", "!", "?", ";", ":", "...", " -", " (", ")", '"']
EXTRA = ["caf\xe9", "na\xefve", "Zo\xeb", "\u6771\u4eac", "\U0001f600", "e\u0301", "don't", "Pearl's", "we're", "I'll", "\t", "  ", "\n\n"]
TAGGED = ["<lookup>weather</lookup>", "<|end|>", "<|endoftext|>", "<result>ok</result>", "<note>x</note>"]


def lexicon(rng: random.Random, n: int = 600) -> list[str]:
    words = set()
    while len(words) < n:
        words.add("".join(rng.choice(SYL) for _ in range(rng.randint(1, 4))))
    return sorted(words)


def sentence(rng: random.Random, lex: list[str], weights: list[float]) -> str:
    ws = rng.choices(lex, weights, k=rng.randint(4, 14))
    ws[0] = ws[0].capitalize()
    out = []
    for w in ws:
        out.append(w)
        r = rng.random()
        if r < 0.12:
            out[-1] += rng.choice(PUNCT)
        elif r < 0.17:
            out.append(str(rng.randint(0, 99999)))
        elif r < 0.21:
            out.append(rng.choice(EXTRA))
    return " ".join(out) + rng.choice([".", "?", "!", "."])


def corpus(seed: int = 0, n_docs: int = 3000) -> list[str]:
    rng = random.Random(seed)
    lex = lexicon(random.Random(99))                     # one shared lexicon for train and held-out
    weights = [1.0 / (k + 1) for k in range(len(lex))]   # Zipf
    docs = []
    for _ in range(n_docs):
        doc = " ".join(sentence(rng, lex, weights) for _ in range(rng.randint(1, 4)))
        if rng.random() < 0.15:
            doc = doc + " " + rng.choice(TAGGED) + " " + sentence(rng, lex, weights)
        docs.append(doc)
    return docs


def chat_lines(seed: int = 2, n: int = 200) -> list[str]:
    rng = random.Random(seed)
    lex = lexicon(random.Random(99))
    weights = [1.0 / (k + 1) for k in range(len(lex))]
    return [f"Hi! My name is {rng.choice(lex).capitalize()} and I'm {rng.randint(10, 90)}. "
            + sentence(rng, lex, weights) for _ in range(n)]

