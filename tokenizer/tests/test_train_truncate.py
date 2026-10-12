"""Training and nested truncation (P-098) on the tiny synthetic corpus."""
from __future__ import annotations

import copy
import json

import pytest
from tokenizers import Tokenizer

import spec
import train_bpe
from synth import SEP, SIZES, TOP
from truncate import check_merge_order, creation_index, is_prefix, merge_pair, truncate_json


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_family_files_and_exact_sizes(family):
    assert sorted(family["paths"]) == SIZES + [TOP]
    for v, p in family["paths"].items():
        tok = Tokenizer.from_file(p)
        assert tok.get_vocab_size() == v
        vocab = tok.get_vocab(with_added_tokens=True)
        assert sorted(vocab.values()) == list(range(v)), "ids not contiguous"
        d = _load(p)
        assert len(d["model"]["vocab"]) == v
        assert len(d["model"]["merges"]) >= v - spec.N_BASE
        check_merge_order(d["model"])


def test_special_and_byte_layout(family):
    spec.check_harness_spelling()
    for v, p in family["paths"].items():
        d = _load(p)
        added = sorted(d["added_tokens"], key=lambda a: a["id"])
        assert [a["content"] for a in added] == spec.SPECIALS and [a["id"] for a in added] == list(range(15))
        assert [a["special"] for a in added] == [True] * 7 + [False] * 8, "control special, tags not"
        assert all(a["normalized"] is False for a in added)
        by_id = sorted(d["model"]["vocab"], key=d["model"]["vocab"].get)
        assert by_id[:15] == spec.SPECIALS
        assert by_id[15:271] == spec.byte_alphabet()
        assert d["normalizer"] is None and d["post_processor"] is None
        assert d["model"]["byte_fallback"] is False and d["model"]["ignore_merges"] is False


def test_sizes_are_prefixes_of_each_other(family):
    sizes = sorted(family["paths"])
    ds = {v: _load(family["paths"][v]) for v in sizes}
    for i, a in enumerate(sizes):
        for b in sizes[i + 1:]:
            assert is_prefix(ds[a], ds[b]), (a, b)
            sv, bv = ds[a]["model"]["vocab"], ds[b]["model"]["vocab"]
            assert all(bv[t] == i for t, i in sv.items())
            assert {t for t, i in bv.items() if i < a} == set(sv)


def test_small_member_only_emits_its_ids(family, heldout):
    for v, p in family["paths"].items():
        tok = Tokenizer.from_file(p)
        for t in heldout["synth"][:100]:
            assert max(tok.encode(t, add_special_tokens=False).ids) < v


def test_truncation_equals_separate_training(family):
    """BPE training is greedy, so the first K merges of the TOP run are what training at that size gives."""
    trunc = _load(family["paths"][SEP])
    sep = _load(family["sep"])
    assert sep["model"]["vocab"] == trunc["model"]["vocab"]
    assert [merge_pair(m) for m in sep["model"]["merges"]] == [merge_pair(m) for m in trunc["model"]["merges"]]


def test_truncate_to_full_size_is_identity(family):
    full = _load(family["paths"][TOP])
    assert truncate_json(full, TOP) == full


def test_truncate_keeps_merges_up_to_last_kept_id(family):
    full = _load(family["paths"][TOP])
    created = creation_index(full["model"])
    for v in SIZES:
        t = truncate_json(full, v)
        assert len(t["model"]["merges"]) == created[v - 1] + 1
        made = {full["model"]["vocab"]["".join(merge_pair(m))] for m in t["model"]["merges"]}
        assert made == set(range(spec.N_BASE, v)), "kept merges must create exactly ids N_BASE..V-1"


@pytest.mark.parametrize("v", [0, spec.N_BASE, TOP + 1])
def test_truncate_rejects_bad_sizes(family, v):
    with pytest.raises(ValueError):
        truncate_json(_load(family["paths"][TOP]), v)


def test_merge_order_violation_is_caught(family):
    full = _load(family["paths"][TOP])
    model = copy.deepcopy(full["model"])
    vocab = model["vocab"]
    # find a merge whose part was made by an earlier merge, and move it before that merge
    created = creation_index(model)
    for j, m in enumerate(model["merges"]):
        a, _ = merge_pair(m)
        if vocab[a] >= spec.N_BASE:
            k = created[vocab[a]]
            model["merges"].insert(k, model["merges"].pop(j))
            break
    with pytest.raises(AssertionError):
        check_merge_order(model)
    gap = copy.deepcopy(full["model"])
    del gap["vocab"][next(t for t, i in gap["vocab"].items() if i == 500)]
    with pytest.raises(AssertionError):
        check_merge_order(gap)


def _base_model(family):
    full = _load(family["paths"][TOP])["model"]
    m = copy.deepcopy(full)
    m["vocab"] = {t: i for t, i in full["vocab"].items() if i < spec.N_BASE}
    m["merges"] = []
    return m


def test_ids_out_of_merge_order_are_caught(family):
    """merge 0 makes id N_BASE+1 and merge 1 makes N_BASE: parts are fine, only the id order is wrong"""
    m = _base_model(family)
    b = spec.N_BASE
    m["vocab"].update({"ab": b + 1, "cd": b})
    m["merges"] = [["a", "b"], ["c", "d"]]
    with pytest.raises(AssertionError, match="not in merge order"):
        check_merge_order(m)
    m["vocab"].update({"ab": b, "cd": b + 1})
    check_merge_order(m)


def test_part_made_by_a_later_merge_is_caught(family):
    """ids follow merge order, but merge 0 uses "ab", which only merge 1 creates"""
    m = _base_model(family)
    b = spec.N_BASE
    m["vocab"].update({"abc": b, "ab": b + 1})
    m["merges"] = [["ab", "c"], ["a", "b"]]
    with pytest.raises(AssertionError, match="created later"):
        check_merge_order(m)
    m["vocab"].update({"ab": b, "abc": b + 1})
    m["merges"] = [["a", "b"], ["ab", "c"]]
    check_merge_order(m)


def test_is_prefix_rejects_non_prefixes(family, corpus):
    small, big = _load(family["paths"][400]), _load(family["paths"][TOP])
    assert is_prefix(small, big) and not is_prefix(big, small)
    swapped = copy.deepcopy(small)
    swapped["model"]["merges"][0], swapped["model"]["merges"][1] = swapped["model"]["merges"][1], swapped["model"]["merges"][0]
    assert not is_prefix(swapped, big)
    moved = copy.deepcopy(small)
    moved["model"]["vocab"][next(t for t, i in moved["model"]["vocab"].items() if i == 300)] = 10 ** 6
    assert not is_prefix(moved, big)
    other = train_bpe.train_json(iter(corpus[::-1][:800]), 400)
    assert not is_prefix(other, big), "a tokenizer trained on other text is not a member of the family"


def test_training_is_deterministic(corpus):
    a = train_bpe.train_json(iter(corpus), 400)
    b = train_bpe.train_json(iter(corpus), 400)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_special_strings_never_reach_training(family):
    """The synthetic corpus plants <lookup>, <|endoftext|> etc. in 15% of documents; the words inside them
    appear nowhere else, so they could only become tokens if the special strings reached the trainer."""
    vocab = _load(family["paths"][TOP])["model"]["vocab"]
    strings = {s for s, i in vocab.items() if i >= spec.N_BASE}
    assert any("weather" in s for s in strings), "the planted words around the specials were trained on"
    for w in ("lookup", "endoftext"):
        assert not any(w in s for s in strings), w


def test_too_small_sample_raises():
    with pytest.raises(RuntimeError, match="training stopped"):
        train_bpe.train_json(["tiny text only"] * 5, 2000)


def test_cli_truncate(family, tmp_path):
    out = tmp_path / "t.json"
    assert train_bpe.main(["truncate", family["paths"][TOP], "--size", "400", "--out", str(out)]) == 0
    assert _load(str(out)) == _load(family["paths"][400])
