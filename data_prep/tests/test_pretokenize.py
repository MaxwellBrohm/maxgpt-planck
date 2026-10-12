"""pretokenize.py: round trip, chat masks through the harness template, held-out exclusion, manifest,
determinism, OOD-H refusal, vocab flag. No torch except the two harness-integration tests."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

import prep_common as C
import pretokenize
import spec
from chat_template import ChatTemplate
from conftest import PRETOK_ARGS
from sources import ChatJsonlSource, TokenShardSource

TEXT, CHAT = ("web", "books"), ("oasst2", "dolly")


def shard_docs(out: str, source: str):
    """-> [(id, ids without EOT)] in shard order, read with plain numpy (no harness code)."""
    man = C.read_json(os.path.join(out, source, "manifest.json"))
    docs = []
    for sh in man["shards"]:
        a = np.fromfile(os.path.join(out, sh["path"]), dtype="<u2").astype(np.int64)
        ends = np.flatnonzero(a == 1)
        assert len(ends) and ends[-1] == len(a) - 1, "shard must end with EOT"
        ids = [line.split("\t")[0] for line in open(os.path.join(out, sh["ids"]), encoding="utf-8")]
        starts = np.concatenate([[0], ends[:-1] + 1])
        assert len(ids) == len(ends) == sh["docs"]
        docs += [(i, a[s:e]) for i, s, e in zip(ids, starts, ends)]
    return docs


def chat_records(out: str, source: str):
    man = C.read_json(os.path.join(out, source, "manifest.json"))
    return [json.loads(line) for sh in man["shards"] for line in open(os.path.join(out, sh["path"]))]


def expected(corpus, source):
    ex = corpus["excluded_ids"][source]
    return {r["id"]: r for r in corpus["records"][source] if r["id"] not in ex and r.get("text", "x") != ""}


@pytest.mark.parametrize("source", TEXT)
def test_text_round_trip(shards, corpus, tok8, source):
    docs = shard_docs(shards, source)
    want = expected(corpus, source)
    assert sorted(i for i, _ in docs) == sorted(want)
    for i, ids in docs:
        assert ids.min() >= spec.N_SPECIAL, f"{i}: control or tag id inside text"
        assert tok8.decode(ids.tolist(), skip_special_tokens=False) == C.clean_text(want[i]["text"])[0]
        assert tok8.encode(C.clean_text(want[i]["text"])[0], add_special_tokens=False).ids == ids.tolist()


def test_special_strings_replaced_and_counted(shards, corpus):
    for s in TEXT:
        man = C.read_json(os.path.join(shards, s, "manifest.json"))
        want = sum(C.clean_text(r["text"])[1] for r in expected(corpus, s).values())
        assert want > 0 and man["counts"]["special_replaced_docs"] == want


@pytest.mark.parametrize("source", TEXT)
def test_harness_token_source_reads_every_doc(shards, source):
    docs = shard_docs(shards, source)
    man = C.read_json(os.path.join(shards, source, "manifest.json"))
    src = TokenShardSource([os.path.join(shards, s["path"]) for s in man["shards"]], 1, 10**7, shuffle=False)
    for _, ids in docs:
        got, flags = src.next_item()
        assert got.tolist() == ids.tolist() + [1] and not flags[0] and flags[1:].all()


def reference_flags(rec_turns):
    """Supervised = assistant content + its <|end|>; never a role token, user, system or tool turn."""
    out = []
    for t in rec_turns:
        sup = t["role"] == "assistant" and t.get("loss", True)
        out += [False] + [sup] * len(t["ids"]) + [sup]
    out[0] = False
    return out


@pytest.mark.parametrize("source", CHAT)
def test_chat_matches_text_mode_and_masks(shards, corpus, tok8, source):
    tmpl = ChatTemplate.from_tokenizer(tok8, "assistant")
    enc = C.encoder(tok8)
    want = expected(corpus, source)
    recs = chat_records(shards, source)
    assert sorted(r["id"] for r in recs) == sorted(want)
    n_sup = 0
    for r in recs:
        orig = want[r["id"]]
        clean = {"turns": [{"role": t["role"], "text": C.clean_text(t["text"])[0]} for t in orig["turns"]]}
        if orig.get("system"):
            clean["system"] = C.clean_text(orig["system"])[0]
            assert r["turns"][0]["role"] == "system" and r["turns"][0]["loss"] is False
        ids_t, fl_t = tmpl.render(clean, enc)              # the harness text path, tokenizing on the fly
        ids_p, fl_p = tmpl.render(r)                       # the pretokenized record
        assert ids_p.tolist() == ids_t.tolist() and fl_p.tolist() == fl_t.tolist()
        assert fl_p.tolist() == reference_flags(r["turns"])
        assert all(min(t["ids"], default=spec.N_SPECIAL) >= spec.N_SPECIAL for t in r["turns"])
        n_sup += int(fl_p.sum())
    man = C.read_json(os.path.join(shards, source, "manifest.json"))
    assert man["counts"]["sup_tokens"] == n_sup and man["kind"] == "chat"


@pytest.mark.parametrize("source", CHAT)
def test_harness_chat_source_reads_every_record(shards, tok8, source):
    man = C.read_json(os.path.join(shards, source, "manifest.json"))
    paths = [os.path.join(shards, s["path"]) for s in man["shards"]]
    src = ChatJsonlSource(paths, ChatTemplate.from_tokenizer(tok8), 10**6, None, shuffle=False)
    lens = [len(src.next_item()[0]) for _ in range(man["counts"]["docs"])]
    assert sum(lens) == man["counts"]["tokens"] and src.dropped_long == 0


def test_heldout_excluded_by_id_split_key_and_tree(shards, corpus):
    for s in TEXT + CHAT:
        man = C.read_json(os.path.join(shards, s, "manifest.json"))
        keys = {k for r in corpus["held"][s] for k in C.record_keys(r)}
        for sh in man["shards"]:
            for line in open(os.path.join(shards, sh["ids"]), encoding="utf-8"):
                i, k, _ = line.rstrip("\n").split("\t")
                assert i not in keys and not ({x for x in k.split("|") if x} & keys), (s, i)
        assert man["counts"]["excluded_docs"] == len(corpus["excluded_ids"][s])
    assert "dolly:5" in corpus["excluded_ids"]["dolly"]


def test_manifest_hashes_and_counts(shards):
    top = C.read_json(os.path.join(shards, "manifest.json"))
    for s, v in top["sources"].items():
        man = C.read_json(os.path.join(shards, s, "manifest.json"))
        assert C.sha256_file(os.path.join(shards, s, "manifest.json")) == v["manifest_sha256"]
        tok = 0
        for sh in man["shards"]:
            p = os.path.join(shards, sh["path"])
            assert C.sha256_file(p) == sh["sha256"] and os.path.getsize(p) == sh["bytes"]
            if man["kind"] == "tokens":
                tok += os.path.getsize(p) // 2
            else:
                tok += sum(sum(len(t["ids"]) + 2 for t in json.loads(x)["turns"]) for x in open(p))
        assert tok == v["tokens"] == man["counts"]["tokens"] == sum(sh["tokens"] for sh in man["shards"])
        assert len(man["shards"]) > 1 or s in CHAT
        got = sorted(glob.glob(os.path.join(shards, top["harness_data"]["sources"][list(top["sources"]).index(s)]
                                           ["paths"][0])))
        assert got == sorted(os.path.join(shards, sh["path"]) for sh in man["shards"])
    assert top["totals"]["tokens"] == sum(v["tokens"] for v in top["sources"].values())
    for hs in top["harness_data"]["sources"]:
        want = {"name": hs["name"], "kind": top["sources"][hs["name"]]["kind"], "paths": hs["paths"]}
        if want["kind"] == "tokens":
            want["eot_id"] = 1
        assert hs == want and hs["kind"] == ("chat" if hs["name"] in CHAT else "tokens")
    assert top["harness_data"]["chat"] == {"loss": "assistant", "end_id": 5,
                                           "role_ids": {"system": 2, "user": 3, "assistant": 4, "tool": 6}}
    assert top["tokenizer"]["file"] == "tok_v0_8k.json" and top["harness_data"]["pad_id"] == 0


def test_shard_is_key_mod_n_and_sorted_by_key(shards):
    for s in TEXT + CHAT:
        man = C.read_json(os.path.join(shards, s, "manifest.json"))
        seen = 0
        for sh in man["shards"]:
            b = int(sh["path"].rsplit("-", 1)[1].split(".")[0])
            keys = [C.doc_key(0, s, line.split("\t")[0]) for line in open(os.path.join(shards, sh["ids"]))]
            assert keys == sorted(keys) and all(k % man["n_shards"] == b for k in keys)
            seen += len(keys)
        assert seen == man["counts"]["docs"]


def test_deterministic_across_workers_and_chunks(shards, corpus, tmp_path):
    out = str(tmp_path / "again")
    assert pretokenize.main([corpus["input"], out, "--heldout", corpus["heldout"], "--workers", "1",
                             "--chunk-mb", "64", "--shard-mb", PRETOK_ARGS[-1]]) == 0
    for s in TEXT + CHAT:
        a = C.read_json(os.path.join(shards, s, "manifest.json"))
        b = C.read_json(os.path.join(out, s, "manifest.json"))
        assert a == b
    assert C.read_json(os.path.join(shards, "manifest.json")) == C.read_json(os.path.join(out, "manifest.json"))


def test_seed_reorders_but_keeps_docs(shards, corpus, tmp_path):
    out = str(tmp_path / "seed1")
    assert pretokenize.main([corpus["input"], out, "--heldout", corpus["heldout"], "--sources", "web",
                             "--seed", "1", *PRETOK_ARGS]) == 0
    a, b = shard_docs(shards, "web"), shard_docs(out, "web")
    assert [i for i, _ in a] != [i for i, _ in b]
    assert sorted((i, x.tolist()) for i, x in a) == sorted((i, x.tolist()) for i, x in b)


def test_oodh_reserved_tree_refused(tmp_path):
    import fixtures_dp as FX
    c = FX.build(str(tmp_path / "c"), reserved_tree=True)
    out = str(tmp_path / "out")
    assert pretokenize.main([c["input"], out, "--no-heldout", "--sources", "oasst2", *PRETOK_ARGS]) == 3
    assert not os.path.exists(os.path.join(out, "oasst2")) and not os.path.exists(os.path.join(out, "oasst2.partial"))


def test_vocab_flag_uses_the_nested_member(shards, corpus, tmp_path, tok2):
    out = str(tmp_path / "v2k")
    assert pretokenize.main([corpus["input"], out, "--heldout", corpus["heldout"], "--sources", "web",
                             "--vocab", "2048", *PRETOK_ARGS]) == 0
    man = C.read_json(os.path.join(out, "web", "manifest.json"))
    assert man["tokenizer"]["file"] == "tok_v0_2k.json" and man["tokenizer"]["vocab"] == 2048
    want = expected(corpus, "web")
    for i, ids in shard_docs(out, "web"):
        assert ids.max() < 2048 and tok2.decode(ids.tolist(), skip_special_tokens=False) == C.clean_text(
            want[i]["text"])[0]
    assert man["counts"]["tokens"] > C.read_json(os.path.join(shards, "web", "manifest.json"))["counts"]["tokens"]


def test_load_tokenizer_matches_harness(tok8):
    pytest.importorskip("torch")
    from data import load_tokenizer
    h = load_tokenizer(C.tokenizer_path(8192))
    s = "a <|end|> b <|endoftext|> <think> c"
    assert h.encode(s, add_special_tokens=False).ids == tok8.encode(s, add_special_tokens=False).ids
    assert 5 not in tok8.encode(s, add_special_tokens=False).ids


def test_harness_block_builds_a_loader(shards):
    pytest.importorskip("torch")
    from data import build_loader
    top = C.read_json(os.path.join(shards, "manifest.json"))
    d = dict(top["harness_data"], mode="pack", window_tokens=4096)
    d["sources"] = [dict(s, share=1.0) for s in d["sources"]]
    ld = build_loader(d, 512, 4, shards, seed=0)
    for _ in range(3):
        b = ld.next_batch()
        assert tuple(b["idx"].shape) == (4, 512) and int(b["idx"].max()) < 8192
    assert all(v > 0 for k, v in ld.stats().items() if k.startswith("drawn_"))
