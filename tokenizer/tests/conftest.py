"""pytest setup for the tokenizer tests (CPU only, tiny synthetic corpus, no downloads)."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(TOK_DIR)
for p in (HERE, TOK_DIR, os.path.join(ROOT, "harness"), os.path.join(ROOT, "rc12")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

import synth  # noqa: E402
from synth import SEP, SIZES, TOP  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    return synth.corpus(seed=0, n_docs=3000)


@pytest.fixture(scope="session")
def heldout():
    return {"synth": synth.corpus(seed=1, n_docs=300), "chat": synth.chat_lines(seed=2, n=200)}


@pytest.fixture(scope="session")
def family(corpus, tmp_path_factory):
    import train_bpe
    out = tmp_path_factory.mktemp("family")
    paths = train_bpe.build_nested(iter(corpus), str(out), top=TOP, sizes=SIZES, prefix="tok_t")
    sep = train_bpe.build_separate(iter(corpus), str(out), SEP, prefix="tok_t_sep")
    return {"paths": paths, "sep": sep, "dir": str(out)}
