"""Health checks on every member of the tiny family, and proof that each check can fail."""
from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer

import health
import sample_io
import spec
from synth import TOP

ODD = ["e\u0301 (decomposed) vs \xe9", "\x00\x01\x7f", "a\r\nb\rc\n", " " * 37 + "x", "\t\t\n\n\n ",
       "\ufeffBOM", "\U0001F468\u200d\U0001F469\u200d\U0001F467", "\u202eRTL", "12,345.67 and 0x1F", "\xff\xfe"]


def test_every_member_is_healthy(family, heldout):
    prev = None
    for v in sorted(family["paths"]):
        rep = health.check_tokenizer(family["paths"][v], heldout)
        assert rep["ok"], (v, rep["ids_errors"], rep["roundtrip_errors"], rep["special_errors"])
        assert rep["vocab"] == v
        bpt = rep["bytes_per_token"]
        assert set(bpt) == {"synth", "chat"} and all(x > 1.0 for x in bpt.values())
        if prev:
            assert all(bpt[k] >= prev[k] for k in bpt), "a bigger member compresses worse"
        prev = bpt
        assert 0 <= rep["unseen"]["n"] <= rep["unseen"]["of"] == v - spec.N_BASE
        lens = [x["bytes"] for x in rep["longest"]]
        assert lens == sorted(lens, reverse=True) and lens[0] > 3


def test_roundtrip_any_string(family):
    texts = ODD + health.random_strings(5, 400)
    for p in family["paths"].values():
        tok = Tokenizer.from_file(p)
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False).ids
            assert tok.decode(ids, skip_special_tokens=False).encode("utf-8") == t.encode("utf-8"), repr(t)


def test_specials_are_atomic(family, heldout):
    plain = [t for t in heldout["synth"] if not sample_io.SPECIAL_RE.search(t)]
    assert len(plain) > 100
    for v, p in family["paths"].items():
        tok = Tokenizer.from_file(p)
        for i, s in enumerate(spec.SPECIALS):
            assert tok.encode(s, add_special_tokens=False).ids == [i]
            ids = tok.encode("abc" + s + "def", add_special_tokens=False).ids
            assert ids.count(i) == 1 and not (set(ids) - {i}) & set(range(spec.N_SPECIAL))
        for t in plain:
            assert not set(tok.encode(t, add_special_tokens=False).ids) & set(range(spec.N_SPECIAL))
        for s in health.NEAR_MISSES:
            assert not set(tok.encode(s, add_special_tokens=False).ids) & set(range(spec.N_SPECIAL)), s
        both = tok.encode("<|user|>hi<|end|><|assistant|><lookup>q</lookup><|end|>", add_special_tokens=False).ids
        assert tok.decode(both) == "hi<lookup>q</lookup>", "default decode drops control tokens, keeps tags"


def _mutant(family, tmp_path, edit):
    d = json.load(open(family["paths"][TOP], encoding="utf-8"))
    edit(d)
    p = tmp_path / "mutant.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _lowercase(d):
    d["normalizer"] = {"type": "Lowercase"}


def _drop_tags(d):
    d["added_tokens"] = [a for a in d["added_tokens"] if a["content"] not in spec.TAG_TOKENS]


def _control_not_special(d):
    for a in d["added_tokens"]:
        a["special"] = False


def _tag_special(d):
    for a in d["added_tokens"]:
        a["special"] = True


def _gap(d):
    d["added_tokens"] = [a for a in d["added_tokens"] if a["content"] != "<|pad|>"]
    del d["model"]["vocab"]["<|pad|>"]


def _gap_last(d):
    """the last merge-made token moves to id V+5: a gap at V-1, specials untouched"""
    v = d["model"]["vocab"]
    last = max(v, key=v.get)
    v[last] += 6


def _swap_end_tool(d):
    """<|end|> and <|tool|> trade ids: still contiguous, but not the harness layout"""
    for a in d["added_tokens"]:
        a["id"] = {5: 6, 6: 5}.get(a["id"], a["id"])
    v = d["model"]["vocab"]
    v["<|end|>"], v["<|tool|>"] = v["<|tool|>"], v["<|end|>"]


@pytest.mark.parametrize("edit,field", [(_lowercase, "roundtrip_errors"), (_drop_tags, "special_errors"),
                                        (_control_not_special, "special_errors"), (_tag_special, "special_errors"),
                                        (_gap, "ids_errors"), (_gap_last, "ids_errors"),
                                        (_swap_end_tool, "ids_errors")])
def test_health_catches_broken_tokenizers(family, heldout, tmp_path, edit, field):
    rep = health.check_tokenizer(_mutant(family, tmp_path, edit), heldout)
    assert not rep["ok"] and rep[field], field


class _Stub:
    """A real tokenizer whose encode() lies for chosen strings, to prove check_specials looks at them."""
    def __init__(self, tok, lies):
        self.tok, self.lies = tok, lies

    def encode(self, s, add_special_tokens=False):
        e = self.tok.encode(s, add_special_tokens=add_special_tokens)
        return type("E", (), {"ids": self.lies.get(s, e.ids)})()

    def encode_batch(self, texts, add_special_tokens=False):
        return [self.encode(t, add_special_tokens) for t in texts]

    def decode(self, ids, skip_special_tokens=True):
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


@pytest.mark.parametrize("lie,want", [({"<NOTE>": [7]}, "near miss"), ({"<|end": [5]}, "near miss"),
                                      ({"plain words here": [3]}, "plain text")])
def test_check_specials_sees_near_misses_and_plain_text(family, lie, want):
    tok = Tokenizer.from_file(family["paths"][TOP])
    assert health.check_specials(_Stub(tok, {}), ["plain words here"]) == []
    errs = health.check_specials(_Stub(tok, lie), ["plain words here"])
    assert errs and all(want in e for e in errs), errs


def test_bytes_per_token_is_utf8_bytes_over_tokens(family):
    texts = ["café 東京 costs 12 euros.", "plain words only here"]
    tok = Tokenizer.from_file(family["paths"][TOP])
    n_tok = sum(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    assert n_bytes > sum(len(t) for t in texts)
    rep = health.check_tokenizer(family["paths"][TOP], {"x": texts})
    assert rep["bytes_per_token"]["x"] == round(n_bytes / n_tok, 4)
    used = set(tok.encode(texts[0], add_special_tokens=False).ids) | set(tok.encode(texts[1], add_special_tokens=False).ids)
    assert rep["unseen"]["n"] == len(set(range(spec.N_BASE, TOP)) - used)


def test_unseen_counts_tokens_the_heldout_never_uses(family):
    tok = Tokenizer.from_file(family["paths"][TOP])
    rep = health.check_tokenizer(family["paths"][TOP], {"one": ["the"]})
    assert rep["unseen"]["n"] >= tok.get_vocab_size() - spec.N_BASE - 3


def test_cli(family, heldout, tmp_path):
    held = tmp_path / "sample" / "heldout"
    held.mkdir(parents=True)
    for name, docs in heldout.items():
        w = sample_io.JsonlWriter(str(held / f"{name}.jsonl"))
        for t in docs:
            w.write(t)
        w.close()
    out = tmp_path / "h.json"
    paths = [family["paths"][v] for v in sorted(family["paths"])]
    assert health.main(paths + ["--heldout", str(held), "--json", str(out)]) == 0
    reps = json.loads(out.read_text())
    assert [r["vocab"] for r in reps] == sorted(family["paths"]) and all(r["ok"] for r in reps)
    bad = _mutant(family, tmp_path, _lowercase)
    assert health.main([bad, "--heldout", str(held)]) == 1
