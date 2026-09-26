"""eval_sets.py + evalwin.py: windows tile the scored bytes, context limits, seams, disjointness from the
training shards, OOD-H refusal, determinism. No torch."""
from __future__ import annotations

import json
import os

import pytest

import eval_sets
import evalwin as EW
import prep_common as C
import pretokenize
from conftest import PRETOK_ARGS

W, CX, CC = 2048, 1024, 3072


def load(ev, name):
    man = C.read_json(os.path.join(ev, "manifest.json"))
    docs = [json.loads(x) for x in open(os.path.join(ev, f"{name}.docs.jsonl"), encoding="utf-8")]
    wins = [json.loads(x) for x in open(os.path.join(ev, f"{name}.windows.jsonl"), encoding="utf-8")]
    return man, docs, wins


def is_bnd(b: bytes, p: int) -> bool:
    ws = b" \t\n\r\x0b\x0c"
    return 1 <= p < len(b) and b[p] in ws and b[p - 1] not in ws


def char_start(b: bytes, p: int) -> bool:
    return p == len(b) or (b[p] & 0xC0) != 0x80


@pytest.mark.parametrize("name", ["web", "books"])
def test_text_windows_tile_scored_region(evalsets, corpus, name):
    man, docs, wins = load(evalsets, name)
    held = {r["id"]: r for r in corpus["held"][name]}
    assert {d["id"] for d in docs} == {i for i, r in held.items() if any(
        is_bnd(C.clean_text(r["text"])[0].encode(), p) for p in range(len(r["text"].encode())))}
    total = 0
    for d in docs:
        b = d["text"].encode("utf-8")
        assert d["text"] == C.clean_text(held[d["id"]]["text"])[0]
        ws = [w for w in wins if w["d"] == d["i"]]
        assert [w["k"] for w in ws] == list(range(len(ws)))
        h = next(p for p in range(1, len(b)) if is_bnd(b, p))          # first word is context only
        assert ws[0]["s"] == h and ws[-1]["e"] == len(b)
        for w, nxt in zip(ws, ws[1:] + [None]):
            assert nxt is None or nxt["s"] == w["e"]
            assert 0 < w["b"] == w["e"] - w["s"] <= W
            assert all(char_start(b, p) for p in (w["c"], w["s"], w["e"]))
            assert 0 <= w["c"] < w["s"]
            if w["s"] - CX <= 0:
                assert w["c"] == 0
            else:
                assert w["c"] >= w["s"] - CX
                firsts = [p for p in range(w["s"] - CX, w["s"]) if is_bnd(b, p)]
                assert not firsts or w["c"] == firsts[0]
            if w["e"] < len(b) and any(is_bnd(b, p) for p in range(w["s"] + W - 256, w["s"] + W + 1)):
                assert is_bnd(b, w["e"]) and w["e"] == max(p for p in range(w["s"] + 1, w["s"] + W + 1)
                                                           if is_bnd(b, p))
        total += sum(w["b"] for w in ws)
    assert total == man["sets"][name]["scored_bytes"] and man["sets"][name]["windows"] == len(wins)
    assert len(wins) > len(docs)                                      # long docs really have several windows


@pytest.mark.parametrize("name", ["oasst2", "dolly"])
def test_chat_windows_tile_turns_and_context(evalsets, name):
    man, docs, wins = load(evalsets, name)
    per_role = {}
    for d in docs:
        tb = [t["text"].encode("utf-8") for t in d["turns"]]
        ws = [w for w in wins if w["d"] == d["i"]]
        for t, b in enumerate(tb):
            tw = [w for w in ws if w["t"] == t]
            if d["turns"][t]["role"] not in ("user", "assistant") or not b:
                assert not tw
                continue
            assert tw[0]["s"] == 0 and tw[-1]["e"] == len(b)
            assert all(x["e"] == y["s"] for x, y in zip(tw, tw[1:]))
            for w in tw:
                assert w["role"] == d["turns"][t]["role"] and w["b"] == w["e"] - w["s"] <= W
                ct, cc = w["ct"], w["cc"]
                assert ct <= t and char_start(tb[ct], cc)
                ctx = (w["s"] - (cc if ct == t else 0)) + sum(len(tb[j]) for j in range(ct, t)) - (
                    cc if ct < t else 0)
                assert 0 <= ctx <= CC
                if ct > 0 and cc == 0 and ct < t:
                    assert ctx + len(tb[ct - 1]) > CC                  # the next older turn did not fit
                if ct == t and cc == 0 and t > 0 and w["s"] < CC:
                    assert len(tb[t - 1]) > CC - w["s"]
                per_role[w["role"]] = per_role.get(w["role"], 0) + w["b"]
    st = man["sets"][name]
    assert sum(per_role.values()) == st["scored_bytes"]
    assert all(st[f"scored_bytes_{r}"] == v for r, v in per_role.items())
    assert "scored_bytes_system" not in st


def test_seams_tokenize_like_the_whole(evalsets, tok8):
    """At a boundary cut, tok(context) + tok(target) == tok(context + target): the windows cost no
    tokenization artifact, for every text window whose ends are boundaries (or the doc start/end)."""
    enc = C.encoder(tok8)
    n = 0
    for name in ("web", "books"):
        _, docs, wins = load(evalsets, name)
        for w in wins:
            b = docs[w["d"]]["text"].encode("utf-8")
            if not all(p in (0, len(b)) or is_bnd(b, p) for p in (w["c"], w["s"], w["e"])):
                continue
            ctx, tgt = b[w["c"]:w["s"]].decode(), b[w["s"]:w["e"]].decode()
            assert enc(ctx) + enc(tgt) == enc(ctx + tgt)
            n += 1
    assert n > 20


def test_disjoint_from_training_shards(evalsets, shards):
    man = C.read_json(os.path.join(evalsets, "manifest.json"))
    assert all(v == {"id": 0, "split_key": 0, "text_sha1": 0} for v in man["checks"]["train_overlap"].values())
    ids, keys, hashes, n = eval_sets.train_index([shards])
    assert n == C.read_json(os.path.join(shards, "manifest.json"))["totals"]["docs"] == man["train_docs_checked"]
    for name in man["sets"]:
        _, docs, _ = load(evalsets, name)
        for d in docs:
            assert d["id"] not in ids and d.get("tree_id", "-") not in keys
    assert man["checks"]["oodh_trees_checked"] == 4


def test_overlap_with_training_refused(corpus, tmp_path):
    leaky = str(tmp_path / "leaky")
    assert pretokenize.main([corpus["input"], leaky, "--no-heldout", "--sources", "dolly", *PRETOK_ARGS]) == 0
    out = str(tmp_path / "ev")
    assert eval_sets.main([corpus["heldout"], out, "--sources", "dolly", "--train-shards", leaky]) == 3
    assert not os.path.exists(out) and not os.path.exists(out + ".partial")


@pytest.mark.parametrize("what", ["split_key", "text_sha1"])
def test_overlap_by_split_key_or_text_alone_refused(corpus, tmp_path, what):
    """A training doc with another id that shares only a Dolly context, or only the exact text."""
    r = corpus["held"]["dolly"][0]
    line = (f"dolly:other\t{r['meta']['split_key']}\t0000000000000000" if what == "split_key"
            else f"dolly:other\tctx:none\t{C.text_hash(C.clean_text(r['text'])[0])}")
    fake = tmp_path / "fake"
    (fake / "dolly").mkdir(parents=True)
    (fake / "dolly" / "dolly-00000.ids").write_text(line + "\n")
    out = str(tmp_path / "ev")
    assert eval_sets.main([corpus["heldout"], out, "--sources", "dolly", "--train-shards", str(fake)]) == 3
    assert not os.path.exists(out)


def test_oodh_reserved_tree_refused(tmp_path):
    import fixtures_dp as FX
    c = FX.build(str(tmp_path / "c"), reserved_tree=True)
    held = str(tmp_path / "held")
    FX.write_jsonl(os.path.join(held, "oasst2.jsonl"), [c["records"]["oasst2"][7]])
    out = str(tmp_path / "ev")
    assert eval_sets.main([held, out]) == 3 and not os.path.exists(out)


def test_deterministic_rebuild(evalsets, corpus, shards, tmp_path):
    out = str(tmp_path / "again")
    assert eval_sets.main([corpus["heldout"], out, "--train-shards", shards]) == 0
    a, b = C.read_json(os.path.join(evalsets, "manifest.json")), C.read_json(os.path.join(out, "manifest.json"))
    assert a["evalset_sha256"] == b["evalset_sha256"] and a["sets"] == b["sets"]


def test_cut_falls_back_to_char_starts():
    text = "a " + "é" * 3000 + " end"              # no boundary for 6000 bytes: cuts at character starts
    ws = EW.text_windows(text, 1000, 300)
    b = text.encode()
    assert ws[0]["s"] == 1 and ws[-1]["e"] == len(b)
    assert all(char_start(b, w["e"]) and w["b"] <= 1000 for w in ws)
    assert all(x["e"] == y["s"] for x, y in zip(ws, ws[1:]))
    assert EW.text_windows("nospace", 100, 10) == [] and EW.text_windows("", 100, 10) == []


def test_fit_drops_context_from_the_left():
    ids, n, cut = EW.fit(list(range(10)), 6, 7)
    assert ids == list(range(3, 10)) and n == 3 and cut
    assert EW.fit(list(range(5)), 2, 7) == (list(range(5)), 2, False)
    with pytest.raises(ValueError):
        EW.fit(list(range(10)), 3, 7)
