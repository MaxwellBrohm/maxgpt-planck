"""pytest setup for data_prep (CPU; the fixture corpus is hand-made, the tokenizers are the real v0 files).
Session fixtures build the shards and eval sets once; tests read them and never write into them."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.dirname(HERE)
for p in (DP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402

import prep_common as C  # noqa: E402  (puts harness/, corpus/, tokenizer/ on sys.path)
import fixtures_dp as FX  # noqa: E402

TOK8 = C.tokenizer_path(8192)
TOK2 = C.tokenizer_path(2048)
PRETOK_ARGS = ["--workers", "2", "--chunk-mb", "0.004", "--shard-mb", "0.02"]


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    return FX.build(str(tmp_path_factory.mktemp("corpus")))


@pytest.fixture(scope="session")
def shards(corpus, tmp_path_factory):
    import pretokenize
    out = str(tmp_path_factory.mktemp("shards") / "out")
    rc = pretokenize.main([corpus["input"], out, "--heldout", corpus["heldout"], *PRETOK_ARGS])
    assert rc == 0
    return out


@pytest.fixture(scope="session")
def evalsets(corpus, shards, tmp_path_factory):
    import eval_sets
    out = str(tmp_path_factory.mktemp("evalsets") / "v0")
    assert eval_sets.main([corpus["heldout"], out, "--train-shards", shards]) == 0
    return out


@pytest.fixture(scope="session")
def tok8():
    return C.load_tokenizer(TOK8)


@pytest.fixture(scope="session")
def tok2():
    return C.load_tokenizer(TOK2)
