"""ngram_overlap.py on hand-made eval sets and shards (numpy + tokenizers; no torch, no model)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import ngram_overlap as NG
import prep_common as C

DUP = ("The quick brown fox jumps over the lazy dog while the band plays on and the rain keeps falling on "
       "the old tin roof of the barn near the river. Later the farmer counted his sheep twice, found one "
       "missing, and walked up the hill with a lantern to look for it before the storm arrived.")
FRESH = "Zyxwv qutsr ponml kjihg fedcb azbyc xdwev ufgth sijrk qlpmo nopqr stuvw xyzab cdefg hijkl mnopq rstuv"
Q = "Could you explain how photosynthesis turns light into chemical energy inside the chloroplasts of a leaf?"
A = "Plants capture photons with chlorophyll, split water, release oxygen and build sugar from carbon dioxide."
BOIL = "Copyright all rights reserved by the site owners and the terms of use apply to every single page here."
H1, H2 = "the cat sat on the mat with a", "red hat and the dog ran far away"   # each under 13 tokens


@pytest.fixture(scope="module")
def result(tmp_path_factory, tok8):
    enc = lambda s: tok8.encode(s, add_special_tokens=False).ids     # noqa: E731
    tmp = tmp_path_factory.mktemp("ng")
    ev, sh = tmp / "ev", tmp / "sh"
    for d in (ev, sh / "web", sh / "chat"):
        d.mkdir(parents=True)
    sets = {"t": [{"i": 0, "id": "dup", "text": DUP}, {"i": 1, "id": "fresh", "text": FRESH}],
            "s": [{"i": 0, "id": "seam", "text": H1 + " " + H2}],
            "b": [{"i": 0, "id": "b1", "text": BOIL + " " + FRESH}, {"i": 1, "id": "b2", "text": BOIL + " " + A}],
            "c": [{"i": 0, "id": "c0", "turns": [{"role": "system", "text": FRESH}, {"role": "user", "text": Q},
                                                 {"role": "assistant", "text": A}]}],
            "cs": [{"i": 0, "id": "cs0", "turns": [{"role": "user", "text": H1 + " " + H2}]}]}
    for name, docs in sets.items():
        (ev / f"{name}.docs.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (ev / "manifest.json").write_text(json.dumps({"evalset_sha256": "test", "sets": {
        n: {"files": {"docs": {"path": f"{n}.docs.jsonl"}}} for n in sets}}))
    assert len(enc(H1)) < 13 and len(enc(" " + H2)) < 13 and enc(H1 + " " + H2) == enc(H1) + enc(" " + H2)
    assert len(enc(H1 + " " + H2)) >= 14
    web = enc("Some other words first. ") + enc(DUP)[1:] + [1] + enc(H1) + [1] + enc(" " + H2) + [1] + enc(BOIL) + [1]
    np.array(web, dtype="<u2").tofile(sh / "web" / "web-00000.bin")
    recs = [{"id": "x", "turns": [{"role": "user", "ids": enc(Q)}, {"role": "assistant", "ids": enc("Sure.")}]},
            {"id": "y", "turns": [{"role": "user", "ids": enc(H1)}, {"role": "assistant", "ids": enc(" " + H2)}]}]
    (sh / "chat" / "chat-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    for s, kind, ext in (("web", "tokens", "bin"), ("chat", "chat", "jsonl")):
        (sh / s / "manifest.json").write_text(json.dumps({"kind": kind, "shards": [{"path": f"{s}/{s}-00000.{ext}"}]}))
    info = C.tokenizer_info(C.tokenizer_path(8192))
    (sh / "manifest.json").write_text(json.dumps({"sources": {"web": {}, "chat": {}}, "tokenizer": info}))
    out = tmp / "out.json"
    assert NG.main([str(ev), str(sh), "--tokenizer", C.tokenizer_path(8192), "--workers", "2",
                    "--out", str(out)]) == 0
    return json.loads(out.read_text())["sets"], enc


def test_a_copied_doc_is_found_and_a_fresh_one_is_not(result):
    r = result[0]["t"]
    assert r["docs_hit_ge_90pct"] == 1 and r["docs_hit_ge_50pct"] == 1 and r["examples_ge_90pct"] == ["dup"]
    assert r["hit_share_by_train_source"]["web"] > 0.3 and r["hit_share_by_train_source"]["chat"] == 0


def test_no_match_across_a_doc_or_turn_seam(result):
    assert result[0]["s"]["hit_share"] == 0 and result[0]["s"]["sampled"] > 0
    assert result[0]["cs"]["hit_share"] == 0 and result[0]["cs"]["sampled"] > 0


def test_chat_turns_by_role_and_source(result):
    r = result[0]["c"]
    assert r["user_turns_hit_ge_50pct"] == 1 and r["user_hit_share"] == 1.0
    assert r["assistant_turns_hit_ge_50pct"] == 0 and r["hit_share_by_train_source"]["web"] == 0
    assert "system_hit_share" not in r                                   # system turns are never sampled


def test_stride_and_sample_count(result):
    sets, enc = result
    want = sum(math.ceil((len(enc(t)) - 12) / 2) for t in (Q, A))
    assert sets["c"]["sampled"] == want


def test_repeated_boilerplate_is_left_out_of_the_once_seen_share(result):
    r = result[0]["b"]
    assert r["hit_share"] > 0.1 and r["hit_share_of_once_seen"] == 0


def test_hash_sees_every_token_of_the_ngram():
    a = np.arange(2, 40)
    b = a.copy()
    b[12] = 999
    assert NG.hashes(a, 13)[0] != NG.hashes(b, 13)[0] and NG.hashes(a, 13)[1] != NG.hashes(b, 13)[1]
    assert NG.hashes(a, 13)[13] == NG.hashes(b, 13)[13] and len(NG.hashes(a[:12], 13)) == 0
