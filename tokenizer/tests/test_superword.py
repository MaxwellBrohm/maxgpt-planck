"""P-101 superword measurement (checks_superword.py) on tiny synthetic text."""
from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer

import checks_bpt
import checks_superword as S
import synth
from checks_bpt import POOLED
from synth import TOP


def test_units_and_runs():
    u = S.Units(["Hi, I'm Zo\xeb. How are you?\n\nI don't know how are you"])
    surf, run = u.docs[0]
    assert surf == ["Hi", ",", " I'm", " Zo\xeb", ".", " How", " are", " you", "?", "\n\n", "I", " don't",
                    " know", " how", " are", " you"]
    assert run == [1, 0, 2, 1, 0, 3, 2, 1, 0, 0, 6, 5, 4, 3, 2, 1]
    assert u.parts[" don't"] == (" don", "'t") and u.parts[" I'm"] == (" I", "'m")
    assert S.Units(["the 2 dogs"]).docs[0][1] == [1, 0, 0, 1], "a digit breaks a phrase"


def test_mine_counts_overlapping_2_to_4_units():
    c = S.mine(S.Units(["a b c d e", "x a b c"]))
    assert c["a b"] == 1 and c[" a b"] == 1 and c[" b c"] == 2 and c[" a b c"] == 1
    assert c["a b c d"] == 1 and "a b c d e" not in c, "no 5-unit phrases"
    assert c[" c d e"] == 1 and c[" b c d e"] == 1
    assert sum(c.values()) == (4 + 3 + 2) + (3 + 2 + 1)


@pytest.fixture(scope="module")
def base(family):
    with open(family["paths"][TOP], encoding="utf-8") as f:
        return json.load(f)


def test_piecewise_count_equals_encode(family, heldout):
    for v, p in family["paths"].items():
        tok = Tokenizer.from_file(p)
        for name, texts in heldout.items():
            texts = [S.sample_io.strip_specials(t) for t in texts]
            got, hits = S.greedy_tokens(S.Units(texts), S.PieceCost(tok), set())
            assert hits == 0 and got == checks_bpt.count_tokens(tok, texts), (v, name)


def test_greedy_prefers_the_longest_phrase(family):
    tok = Tokenizer.from_file(family["paths"][TOP])
    cost = S.PieceCost(tok)
    u = S.Units(["x how are you now"])

    def n(ph):
        return S.greedy_tokens(u, cost, set(ph))[0]
    parts = {s: sum(cost(p) for p in u.parts[s]) for s in u.docs[0][0]}
    assert n([]) == sum(parts.values())
    assert n([" how are you"]) == parts["x"] + 1 + parts[" now"]
    assert n([" how are", " how are you"]) == parts["x"] + 1 + parts[" now"], "longest match first"
    assert n([" are you now", " how are"]) == parts["x"] + 1 + parts[" you"] + parts[" now"], \
        "leftmost phrase wins even when a later one is longer"
    assert n(["how are you"]) == n([]), "no leading space: a different string, no match"


def test_p101_arms(base, family):
    chat_train, chat_held = synth.chat_lines(seed=7, n=800), synth.chat_lines(seed=2, n=200)
    r = S.p101(base, chat_train, {"chat": chat_held}, ns=(56, 256, 512 + 212))
    assert r["base_vocab"] == TOP and len(r["arms"]) == 6
    tok768 = Tokenizer.from_file(family["paths"][768])
    tok300 = Tokenizer.from_file(family["paths"][300])
    chat_held = [S.sample_io.strip_specials(t) for t in chat_held]
    assert r["base_tokens"][POOLED] == checks_bpt.count_tokens(Tokenizer.from_file(family["paths"][TOP]), chat_held)
    for a in r["arms"]:
        assert a["truncated_to"] == TOP - a["n"] and a["n_phrases"] == a["n"]
        assert a["gain_pct"][POOLED] == pytest.approx((r["base_tokens"][POOLED] / a["tokens"][POOLED] - 1) * 100, abs=1e-3)
        assert a["drop_merges_gain_pct"][POOLED] < 0, "dropping merges must cost compression"
        assert a["tokens"][POOLED] < a["tokens_without_phrases"][POOLED]
        want = {256: tok768, 724: tok300}.get(a["n"])
        if want is not None:
            assert a["tokens_without_phrases"][POOLED] == checks_bpt.count_tokens(want, chat_held), \
                "the V-N arm must be the family member of size V-N"
    freq = [a for a in r["arms"] if a["ranking"] == "freq"]
    counts = [e["count"] for e in freq[0]["examples"]]
    assert counts == sorted(counts, reverse=True)
    assert freq[0]["gain_pct"][POOLED] > 5 and r["passes_bar"], "templated synthetic chat should clear 5%"
    assert r["best_gain_pct_freq"] == max(a["gain_pct"][POOLED] for a in freq)
    saved = [a for a in r["arms"] if a["ranking"] == "saved"]
    assert [e["phrase"] for e in saved[0]["examples"]] != [e["phrase"] for e in freq[0]["examples"]]


def test_p101_mines_train_not_heldout(base):
    held = synth.chat_lines(seed=2, n=200)
    r = S.p101(base, synth.corpus(seed=5, n_docs=600), {"chat": held}, ns=(56,), rankings=("freq",))
    names = [e["phrase"] for e in r["arms"][0]["examples"]]
    assert " My name is" not in names and " My name" not in names
    good = S.p101(base, synth.chat_lines(seed=7, n=800), {"chat": held}, ns=(56,), rankings=("freq",))
    assert r["arms"][0]["gain_pct"][POOLED] < good["arms"][0]["gain_pct"][POOLED]


def test_p101_bar_uses_frequency_ranking(base):
    # web-like training text mined, chat scored: small gain, so the bar fails on both rankings
    r = S.p101(base, synth.corpus(seed=5, n_docs=600), {"chat": synth.chat_lines(seed=2, n=200)}, ns=(56,))
    assert r["best_gain_pct_freq"] < S.BAR_PCT and not r["passes_bar"]


def test_p101_strips_special_strings(base):
    held = synth.chat_lines(seed=2, n=100) + [d for d in synth.corpus(seed=1, n_docs=300) if "<" in d]
    assert any("<|end|>" in d for d in held)
    r = S.p101(base, synth.chat_lines(seed=7, n=300), {"chat": held}, ns=(56,), rankings=("freq",))
    tok = Tokenizer.from_str(json.dumps(base))
    assert r["base_tokens"][POOLED] == checks_bpt.count_tokens(tok, [S.sample_io.strip_specials(t) for t in held])
    assert r["base_tokens"][POOLED] != checks_bpt.count_tokens(tok, held)


def test_rank_drops_singletons_and_breaks_ties_by_string():
    import collections
    c = collections.Counter({" b c": 3, " a b": 3, " z y": 1, " q r s": 2})
    assert [p for p, _, _ in S.rank(c, "freq")] == [" a b", " b c", " q r s"]


def test_real_added_tokens_fire_inside_words(base):
    from tokenizers import AddedToken
    from truncate import truncate_json
    small = truncate_json(base, TOP - 1)
    held = {"x": ["all of these", "one of the dogs"]}
    real = S.real_tokens(small, {" of the"}, held)
    tok = Tokenizer.from_str(json.dumps(small))
    tok.add_tokens([AddedToken(" of the", normalized=False)])
    enc = [tok.encode(t, add_special_tokens=False).tokens for t in held["x"]]
    assert all(" of the" in e for e in enc), "an added token also fires inside ' of these'"
    assert real["x"] == real[POOLED] == sum(map(len, enc))
    greedy, hits = S.greedy_tokens(S.Units(held["x"]), S.PieceCost(Tokenizer.from_str(json.dumps(small))),
                                   {" of the"})
    assert hits == 1 and greedy != real["x"], "greedy matches only whole word units"
    assert S.real_tokens(small, set(), held)[POOLED] == checks_bpt.count_tokens(
        Tokenizer.from_str(json.dumps(small)), held["x"])


def test_p101_reports_the_real_added_token_gain(base):
    held = synth.chat_lines(seed=2, n=200)
    r = S.p101(base, synth.chat_lines(seed=7, n=800), {"chat": held}, ns=(56, 256), rankings=("freq",))
    for a in r["arms"]:
        assert a["real_gain_pct"][POOLED] == pytest.approx(
            (r["base_tokens"][POOLED] / a["real_tokens"][POOLED] - 1) * 100, abs=1e-3)
    assert r["best_real_gain_pct_freq"] == max(a["real_gain_pct"][POOLED] for a in r["arms"])
    assert r["passes_bar_real"] == (r["best_real_gain_pct_freq"] >= S.BAR_PCT)
