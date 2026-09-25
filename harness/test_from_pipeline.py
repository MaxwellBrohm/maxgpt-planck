"""from_pipeline.py: FAKE pipeline shards -> adapter -> harness Loader round trip.

The pipeline shards come from the real pipeline driver against its FAKE stub (pipeline_fixture.py:
make_pipeline_fake.py, 200 skeletons, about 6 s; PLANCK_PIPELINE_FAKE=<run dir> reuses a run).
The reference is built from the raw pipeline lines with no adapter or template code: per conversation, the system text
(if any) then every turn, each as (role, text, supervised) with supervised = assistant and mask 0.
Items are recovered from the Loader's packed rows by document id, split at role tokens, decoded
with the toy tokenizer, and their loss flags read back from the targets.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

import from_pipeline as FP
import pipeline_fixture as PF
import toy_tokenizer
from chat_template import ChatTemplate
from data import build_loader

ROLE = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>", "tool": "<|tool|>"}
SEQ = 2048


@pytest.fixture(scope="module")
def fake_run():
    return PF.fake_run_dir()


@pytest.fixture(scope="module")
def raw(fake_run):
    out = PF.read_raw(fake_run)
    assert len(out) >= 150
    return out


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    return toy_tokenizer.write(str(tmp_path_factory.mktemp("tok") / "tok.json"))


@pytest.fixture(scope="module")
def converted(fake_run, tok_path, tmp_path_factory):
    d = tmp_path_factory.mktemp("conv")
    runs = {"tokens": FP.run([fake_run], str(d / "tokens"), allow_nontrainable=True, tokenizer=tok_path),
            "text": FP.run([fake_run], str(d / "text"), allow_nontrainable=True)}
    return {k: (str(d / k), m) for k, m in runs.items()}


def reference(rec):
    segs = [("system", rec["system"], False)] if rec["system"] else []
    return segs + [(t["role"], t["text"], t["role"] == "assistant" and t["mask"] == 0) for t in rec["turns"]]


def rendered(segs):
    return "".join(ROLE[r] + text + "<|end|>" for r, text, _ in segs)


def loader_items(out_dir, manifest, tok_path, mode):
    dcfg = {"mode": "pack", "pad_id": 0, "max_item_len": SEQ, "window_tokens": 8 * SEQ,
            "sources": [{"name": "pipeline", "kind": "chat", "shuffle": False,
                         "paths": [os.path.join(out_dir, "chat-*.jsonl")]}]}
    if mode == "tokens":        # token shards: role ids from the manifest, no tokenizer at load time
        dcfg["chat"] = dict(manifest["harness_data"]["chat"])
    else:
        dcfg["chat"], dcfg["tokenizer"] = {"loss": "assistant"}, tok_path
    ld = build_loader(dcfg, SEQ, 4)
    while ld.sources[0].epoch == 0 or ld._u < len(ld._units):     # drain the window that wrapped
        b = {k: v.numpy() for k, v in ld.next_batch().items()}
        for r in range(b["idx"].shape[0]):
            for d in np.unique(b["doc"][r]):
                sel = b["doc"][r] == d
                ids, tgt = b["idx"][r][sel], b["tgt"][r][sel]
                if ids[0] == 0 and (ids == 0).all() and (tgt == -100).all():
                    continue                                   # the row's padding document
                assert tgt[-1] == -100
                flags = np.zeros(len(ids), dtype=bool)
                flags[1:] = tgt[:-1] != -100
                assert (tgt[:-1][flags[1:]] == ids[1:][flags[1:]]).all(), "a target is not the next token"
                yield ids, flags
    assert ld.stats()["dropped_long_pipeline"] == 0


def segments(ids, flags, tok):
    inv = {tok.token_to_id(s): r for r, s in ROLE.items()}
    end, out, i = tok.token_to_id("<|end|>"), [], 0
    while i < len(ids):
        role = inv[int(ids[i])]
        j = i + 1 + list(ids[i + 1:]).index(end)
        sup = set(flags[i + 1:j + 1].tolist())
        assert not flags[i] and len(sup) == 1, "role token supervised, or a turn half supervised"
        out.append((role, tok.decode([int(x) for x in ids[i + 1:j]], skip_special_tokens=False), sup.pop()))
        i = j + 1
    return out


@pytest.mark.parametrize("mode", ["tokens", "text"])
def test_round_trip_through_loader(raw, converted, tok_path, mode):
    tok = toy_tokenizer.build()
    ref = {rendered(reference(rec)): reference(rec) for _, _, _, rec in raw}
    assert len(ref) == len(raw), "two pipeline records render the same"
    out_dir, manifest = converted[mode]
    seen, masked, tool, system = set(), 0, 0, 0
    for ids, flags in loader_items(out_dir, manifest, tok_path, mode):
        segs = segments(ids, flags, tok)
        key = rendered(segs)
        assert tok.decode([int(x) for x in ids], skip_special_tokens=False) == key
        assert key in ref, "the Loader produced a conversation the pipeline never wrote"
        assert segs == ref[key], "turn text or loss mask differs from the pipeline record"
        if key not in seen:
            masked += sum(r == "assistant" and not s for r, _, s in segs)
            tool += sum(r == "tool" for r, _, _ in segs)
            system += segs[0][0] == "system"
        seen.add(key)
    assert seen == set(ref), f"{len(set(ref) - seen)} conversations never came out of the Loader"
    # the fixture must exercise every rule: masked assistant turns, tool turns, system text
    want_masked = sum(t["mask"] for _, _, _, rec in raw for t in rec["turns"])
    assert masked == want_masked > 0 and tool > 0 and system > 0


def test_token_shards_equal_text_rendering(converted, tok_path):
    tok = toy_tokenizer.build()
    tmpl = ChatTemplate.from_tokenizer(tok)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
    recs = {k: [json.loads(x) for p in m["shards"] for x in open(p)] for k, (_, m) in converted.items()}
    assert len(recs["tokens"]) == len(recs["text"])
    for a, b in zip(recs["tokens"], recs["text"]):
        ia, fa = tmpl.render(a)
        ib, fb = tmpl.render(b, enc)
        assert ia.tolist() == ib.tolist() and fa.tolist() == fb.tolist()
        assert a["adapter"]["n_tokens"] == len(ia)


def test_loss_all_mode_never_trains_tool_or_masked_turns(converted):
    tok = toy_tokenizer.build()
    tmpl = ChatTemplate.from_tokenizer(tok, loss="all")
    _, m = converted["tokens"]
    n_tool = n_masked = 0
    for line in open(m["shards"][0]):
        rec = json.loads(line)
        ids, flags = tmpl.render(rec)
        at = 0
        for t in rec["turns"]:
            n = len(t["ids"]) + 2
            want = t["role"] == "user" or (t["role"] == "assistant" and t.get("mask") == 0)
            got = flags[at:at + n].tolist()
            assert got == [want and at > 0] + [want] * (n - 1), (t["role"], t.get("mask"))
            n_tool += t["role"] == "tool"
            n_masked += t["role"] == "assistant" and t.get("mask") == 1
            at += n
        assert at == len(ids)
    assert n_tool > 0 and n_masked > 0


def test_provenance_preserved(raw, converted, tok_path):
    _, m = converted["tokens"]
    outs = [json.loads(x) for p in m["shards"] for x in open(p)]
    assert len(outs) == len(raw) == m["counts"]["admitted"] == m["counts"]["admitted_nontrainable"]
    tok_sha = hashlib.sha256(open(tok_path, "rb").read()).hexdigest()
    for (path, n, line, rec), o in zip(raw, outs):
        assert o["id"] == rec["conv_id"]
        assert o["pipeline"] == {k: v for k, v in rec.items() if k != "turns"}
        assert o["pipeline"]["trainable"] is False and o["pipeline"]["blocked"]
        assert o["source"] == {"file": path, "line": n, "sha256": hashlib.sha256(line.rstrip(b"\n")).hexdigest()}
        assert o["adapter"]["admitted_nontrainable"] is True and o["adapter"]["tokenizer_sha256"] == tok_sha
        body = [t for t in o["turns"] if t["src_i"] is not None]
        assert [t["src_i"] for t in body] == list(range(len(rec["turns"])))
        for t, s in zip(body, rec["turns"]):
            assert (t["role"], t["text"], t["author"], t["mask"], t["spans"]) == \
                (s["role"], s["text"], s["author"], s["mask"], s["spans"])
        assert (o["turns"][0]["role"] == "system") == bool(rec["system"])
    assert [i["sha256"] for i in m["inputs"]] == [FP.sha256_file(p) for p in sorted({r[0] for r in raw})]
