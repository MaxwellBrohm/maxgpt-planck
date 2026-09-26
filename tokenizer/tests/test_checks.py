"""checks.py: bytes/token table, P-098, P-097 and the CLI, on the tiny synthetic family."""
from __future__ import annotations

import json
import os

import pytest
from tokenizers import Tokenizer

import checks
import checks_bpt as B
import checks_names as N
import synth
import train_bpe
from checks_bpt import POOLED
from synth import SEP


@pytest.fixture(scope="module")
def toks(family):
    return {v: Tokenizer.from_file(p) for v, p in family["paths"].items()}


def test_bpt_table(toks, heldout):
    t = B.bpt_table(toks, heldout)
    assert sorted(t) == sorted(toks)
    for v, rows in t.items():
        assert set(rows) == set(heldout) | {POOLED}
        for name, texts in heldout.items():
            want = sum(len(toks[v].encode(x, add_special_tokens=False).ids) for x in texts)
            assert rows[name]["tokens"] == want and rows[name]["docs"] == len(texts)
            assert rows[name]["bytes"] == sum(len(x.encode("utf-8")) for x in texts)
        tb = sum(rows[n]["bytes"] for n in heldout)
        tt = sum(rows[n]["tokens"] for n in heldout)
        assert rows[POOLED]["bytes"] == tb and rows[POOLED]["tokens"] == tt
        assert rows[POOLED]["bpt"] == round(tb / tt, 4)
    sizes = sorted(t)
    assert all(t[a][POOLED]["bpt"] < t[b][POOLED]["bpt"] for a, b in zip(sizes, sizes[1:]))


def test_count_tokens_chunks(toks, heldout):
    texts = heldout["synth"]
    assert B.count_tokens(toks[SEP], texts, chunk=7) == B.count_tokens(toks[SEP], texts, chunk=10_000)


def test_rel_diff_sign_and_value():
    assert B.rel_diff_pct(1000, 200, 1000, 250) == 25.0          # truncated: fewer tokens, better
    assert B.rel_diff_pct(1000, 250, 1000, 200) == -20.0
    assert B.rel_diff_pct(1000, 0, 1000, 200) is None


def test_p098_identical_family_member(family, toks, heldout):
    stats_t = B.source_stats(toks[SEP], heldout)
    stats_s = B.source_stats(Tokenizer.from_file(family["sep"]), heldout)
    r = B.p098(stats_t, stats_s, B.same_model(family["paths"][SEP], family["sep"]))
    assert r["identity"]["vocab_identical"] and r["identity"]["merges_identical"]
    assert r["pooled_rel_diff_pct"] == 0.0 and r["default_holds"]
    assert set(r["per_source"]) == set(heldout) | {POOLED}


def test_p098_detects_a_different_tokenizer(family, toks, heldout, tmp_path):
    other = train_bpe.build_separate(iter(synth.chat_lines(seed=9, n=2000)), str(tmp_path), SEP, prefix="o")
    stats_t = B.source_stats(toks[SEP], heldout)
    stats_o = B.source_stats(Tokenizer.from_file(other), heldout)
    idn = B.same_model(family["paths"][SEP], other)
    assert not idn["vocab_identical"] and not idn["merges_identical"]
    r = B.p098(stats_t, stats_o, idn)
    diff = r["pooled_rel_diff_pct"]
    want = ((stats_t[POOLED]["bytes"] / stats_t[POOLED]["tokens"])
            / (stats_o[POOLED]["bytes"] / stats_o[POOLED]["tokens"]) - 1) * 100
    assert diff == pytest.approx(want, abs=1e-3) and abs(diff) > 0.5
    assert not r["default_holds"]
    assert B.p098(stats_t, stats_o, idn, tol=abs(diff) + 0.01)["default_holds"]


def test_p098_identity_needs_vocab_and_merges():
    a = {"x": {"bytes": 100, "tokens": 10, "bpt": 10.0}, POOLED: {"bytes": 100, "tokens": 10, "bpt": 10.0}}
    b = {"x": {"bytes": 100, "tokens": 20, "bpt": 5.0}, POOLED: {"bytes": 100, "tokens": 20, "bpt": 5.0}}
    for vi, mi in ((True, False), (False, True)):
        r = B.p098(a, b, {"vocab_identical": vi, "merges_identical": mi, "n_merges": [1, 1]})
        assert r["pooled_rel_diff_pct"] == 100.0 and not r["default_holds"] and "outside" in r["verdict"]
    assert B.p098(a, b, {"vocab_identical": True, "merges_identical": True, "n_merges": [1, 1]})["default_holds"]
    assert list(B.p098(a, a, {"vocab_identical": True, "merges_identical": True, "n_merges": [1, 1]})
                ["per_source"]) == ["x", POOLED], "pooled row last"


def test_p098_refuses_different_text(toks, heldout):
    a = B.source_stats(toks[SEP], heldout)
    b = B.source_stats(toks[SEP], {"synth": heldout["synth"][:-1], "chat": heldout["chat"]})
    with pytest.raises(AssertionError):
        B.p098(a, b, {"vocab_identical": True, "merges_identical": True, "n_merges": [1, 1]})


def test_name_lists():
    assert len(N.FIRST_NAMES) == 100 and len(N.SURNAMES) == 100
    assert len(set(N.FIRST_NAMES) | set(N.SURNAMES)) == 200
    assert all(x.isalpha() and x[0].isupper() and x[1:].islower() for x in N.FIRST_NAMES + N.SURNAMES)


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class FakeTok:
    """Greedy longest-match over a fixed token list, enough to pin down the P-097 definitions."""

    def __init__(self, vocab):
        self.vocab = sorted(set(vocab) | set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.by = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, s, add_special_tokens=False):
        ids, i = [], 0
        while i < len(s):
            t = max((t for t in self.vocab if s.startswith(t, i)), key=len)
            ids.append(self.by[t])
            i += len(t)
        return _Enc(ids)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)


def test_name_row_definitions():
    tok = FakeTok([" Pearl", "Pe", "arl", "pe", "PE", "ARL", " Bo", "Bo", " bo", "BO", "b", "B"])
    p = N.name_row(tok, "Pearl")
    assert p["n_tokens"] == {"space_cap": 1, "cap": 2, "space_lower": 3, "upper": 2}
    assert p["count_differs"] and p["first_id_differs"] and p["first_piece_differs"]
    assert not p["same_split_space"]
    b = N.name_row(tok, "Bob")
    assert b["n_tokens"] == {"space_cap": 2, "cap": 2, "space_lower": 2, "upper": 2}
    assert not b["count_differs"] and b["first_id_differs"] and not b["first_piece_differs"]
    assert b["same_split_space"]
    s = N.summarize([p, b])
    assert s["one_token_frac"] == {"space_cap": 0.5, "cap": 0.0, "space_lower": 0.0, "upper": 0.0}
    assert s["mean_tokens"]["space_lower"] == 2.5 and s["count_differs_frac"] == 0.5
    assert s["first_piece_differs_frac"] == 0.5 and s["same_split_space_frac"] == 0.5
    r = N.p097({7: tok}, first=["Pearl"], last=["Bob"])
    assert r[7]["first"]["n"] == 1 and r[7]["all"] == s
    ex = {e["name"]: e for e in r[7]["examples"]}
    assert ex["Pearl"]["forms"]["cap"] == ["Pe", "arl"] and ex["Bob"]["n_tokens"]["upper"] == 2


def test_p097_on_family(toks):
    r = N.p097(toks)
    for v, x in r.items():
        assert x["all"]["n"] == 200 and x["first"]["n"] == 100 and x["last"]["n"] == 100
        mean = x["all"]["mean_tokens"]["space_cap"]
        want = sum(len(toks[v].encode(" " + n).ids) for n in N.FIRST_NAMES + N.SURNAMES) / 200
        assert mean == round(want, 4)
    sizes = sorted(r)
    assert r[sizes[0]]["all"]["mean_tokens"]["cap"] >= r[sizes[-1]]["all"]["mean_tokens"]["cap"]
