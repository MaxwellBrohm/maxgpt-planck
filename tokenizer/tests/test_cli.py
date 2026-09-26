"""End to end through the CLIs, on a sample directory in make_tok_sample.py's layout."""
from __future__ import annotations

import json
import os

import pytest
from tokenizers import Tokenizer

import health
import sample_io
import synth
import train_bpe


def _write(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, t in enumerate(texts):
            f.write(json.dumps({"id": f"d{k}", "text": t, "meta": {}}, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("sample"))
    docs = synth.corpus(seed=3, n_docs=1500)
    _write(os.path.join(d, "train", "web.jsonl"), docs[:1000])
    _write(os.path.join(d, "train", "chat.jsonl"), synth.chat_lines(seed=4, n=400))
    _write(os.path.join(d, "heldout", "web.jsonl"), docs[1000:])
    _write(os.path.join(d, "heldout", "chat.jsonl"), synth.chat_lines(seed=5, n=100))
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump({"weights": {"chat": 3}}, f)
    return d


def test_iter_texts_weights(sample_dir):
    assert len(list(sample_io.iter_texts(sample_dir))) == 1000 + 3 * 400          # manifest weights
    assert len(list(sample_io.iter_texts(sample_dir, weights={"web": 2, "chat": 0}))) == 2000
    assert len(list(sample_io.iter_texts(sample_dir, "heldout", {}))) == 600
    with pytest.raises(AssertionError):
        list(sample_io.iter_texts(sample_dir, weights={"nope": 1}))


def test_weights_change_the_merges(sample_dir, tmp_path):
    a = train_bpe.train_json(sample_io.iter_texts(sample_dir, weights={"web": 1, "chat": 1}), 500)
    b = train_bpe.train_json(sample_io.iter_texts(sample_dir, weights={"web": 1, "chat": 20}), 500)
    assert a["model"]["merges"] != b["model"]["merges"]


def test_nested_separate_and_health_cli(sample_dir, tmp_path):
    out = str(tmp_path / "out")
    assert train_bpe.main(["nested", "--sample", sample_dir, "--out", out, "--top", "700",
                           "--sizes", "300,512", "--prefix", "tok_c"]) == 0
    assert train_bpe.main(["separate", "--sample", sample_dir, "--out", out, "--size", "512",
                           "--prefix", "tok_c_sep"]) == 0
    names = sorted(os.listdir(out))
    assert names == ["tok_c_300.json", "tok_c_512.json", "tok_c_700.json", "tok_c_manifest.json",
                     "tok_c_sep_512.json", "tok_c_sep_manifest.json"]
    man = json.load(open(os.path.join(out, "tok_c_manifest.json")))
    assert man["weights"] == {"chat": 3} and man["normalizer"] is None
    assert tmp_path.name not in json.dumps(man) and sample_dir not in json.dumps(man), "no local paths in the manifest"
    assert set(man["files"]) == {"tok_c_300.json", "tok_c_512.json", "tok_c_700.json"}
    assert man["files"]["tok_c_512.json"]["sha256"] == sample_io.sha256_file(os.path.join(out, "tok_c_512.json"))
    a = json.load(open(os.path.join(out, "tok_c_512.json")))
    b = json.load(open(os.path.join(out, "tok_c_sep_512.json")))
    assert a["model"] == b["model"], "truncated 512 differs from a separately trained 512 on the same sample"
    for n in ("tok_c_300.json", "tok_c_700.json"):
        assert Tokenizer.from_file(os.path.join(out, n)).get_vocab_size() == int(n[6:9])
    rep = str(tmp_path / "health.json")
    toks = [os.path.join(out, n) for n in names if not n.endswith("manifest.json")]
    assert health.main(toks + ["--heldout", os.path.join(sample_dir, "heldout"), "--json", rep]) == 0
    reps = json.load(open(rep))
    assert all(r["ok"] for r in reps) and all(set(r["bytes_per_token"]) == {"web", "chat"} for r in reps)


def test_weight_flag_overrides_manifest(sample_dir, tmp_path):
    out = str(tmp_path / "o")
    assert train_bpe.main(["separate", "--sample", sample_dir, "--out", out, "--size", "400",
                           "--weight", "chat=1", "--weight", "web=2"]) == 0
    man = json.load(open(os.path.join(out, "tok_v0_sep_manifest.json")))
    assert man["weights"] == {"chat": 1, "web": 2}
