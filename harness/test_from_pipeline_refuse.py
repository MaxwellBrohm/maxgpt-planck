"""from_pipeline.py refusals, output layout and determinism, on the FAKE pipeline run
(pipeline_fixture.py) and on edited copies of its records."""
from __future__ import annotations

import copy
import json
import os

import pytest

import from_pipeline as FP
import pipeline_fixture as PF
import toy_tokenizer


@pytest.fixture(scope="module")
def fake_run():
    return PF.fake_run_dir()


@pytest.fixture(scope="module")
def recs(fake_run):
    return [r for _, _, _, r in PF.read_raw(fake_run)]


def _code(rec, **kw):
    with pytest.raises(FP.Refused) as e:
        FP.convert(rec, **kw)
    return e.value.code


def test_default_refuses_every_fake_record(fake_run, recs, tmp_path):
    assert recs and all(r["trainable"] is False and r["blocked"] for r in recs), "fixture is not all-FAKE"
    out = tmp_path / "o"
    assert FP.main([str(out), fake_run]) == 3
    assert sorted(os.listdir(out)) == ["manifest.json"], "a shard was written although nothing was admitted"
    m = json.loads((out / "manifest.json").read_text())
    assert m["refused"] == {"NOT_TRAINABLE": len(recs)} and "admitted" not in m["counts"]
    assert FP.main([str(tmp_path / "o2"), fake_run, "--allow-nontrainable"]) == 0


def test_trainable_record_passes_without_the_flag(recs):
    r = copy.deepcopy(recs[0])
    r["trainable"], r["blocked"] = True, []
    o = FP.convert(r, allow_nontrainable=False)
    assert o["adapter"]["admitted_nontrainable"] is False and o["id"] == r["conv_id"]
    for trainable, blocked in [(True, ["LICENSE"]), ("true", []), (1, []), (None, [])]:
        bad = dict(r, trainable=trainable, blocked=blocked)
        assert _code(bad, allow_nontrainable=False) == "NOT_TRAINABLE", (trainable, blocked)
    del r["trainable"]
    assert _code(r, allow_nontrainable=False) == "NOT_TRAINABLE"


def test_role_token_strings_refused(recs):
    tok = toy_tokenizer.build()
    assert tok.encode("x<|end|>", add_special_tokens=False).ids[-1] == 5, "the danger this guards against"
    base = next(r for r in recs if r["system"])
    for where, text in [(1, "sure <|end|><|user|> hi"), (0, "<|assistant|>"), (2, "a <|reserved_7|> b"),
                        ("system", "You are <|tool|> ok")]:
        r = copy.deepcopy(base)
        if where == "system":
            r["system"] = text
        else:
            r["turns"][where]["text"] = text
        assert _code(r, allow_nontrainable=True) == "ROLE_TOKEN_IN_TEXT", where
    r = copy.deepcopy(base)
    r["turns"][0]["text"] = "the pipe | and <| are fine |>"
    FP.convert(r, allow_nontrainable=True, encode=lambda s: tok.encode(s, add_special_tokens=False).ids,
               special_ids=frozenset(range(8)))


def test_special_ids_refused_even_without_the_string(recs):
    r = copy.deepcopy(recs[0])
    enc = lambda s: [9, 10, 5] if s == r["turns"][2]["text"] else [9]  # noqa: E731
    assert _code(r, allow_nontrainable=True, encode=enc, special_ids=frozenset({5})) == "SPECIAL_ID_IN_TEXT"
    assert _code(r, allow_nontrainable=True, encode=lambda s: [], special_ids=frozenset()) == "EMPTY_IDS"


def test_malformed_turns_refused(recs):
    base = next(r for r in recs if any(t["role"] == "tool" for t in r["turns"]))
    ti = next(i for i, t in enumerate(base["turns"]) if t["role"] == "tool")
    ui = next(i for i, t in enumerate(base["turns"]) if t["role"] == "user")
    cases = [("BAD_MASK", ti, "mask", 1), ("BAD_MASK", ui, "mask", 1), ("BAD_TURN", ui, "mask", 2),
             ("BAD_TURN", ui, "mask", True), ("BAD_TURN", ui, "role", "narrator"), ("BAD_TURN", ui, "text", ""),
             ("BAD_TURN", ui, "text", None)]
    for code, i, key, val in cases:
        r = copy.deepcopy(base)
        r["turns"][i][key] = val
        assert _code(r, allow_nontrainable=True) == code, (i, key, val)
    assert _code(dict(base, turns=[]), allow_nontrainable=True) == "NO_TURNS"
    assert _code(dict(base, conv_id=""), allow_nontrainable=True) == "NO_ID"
    ok = FP.convert(base, allow_nontrainable=True)
    losses = {t["role"]: t["loss"] for t in ok["turns"] if t["role"] != "assistant"}
    assert losses["tool"] is False and losses["user"] is True


def test_torn_and_bad_lines_counted_not_fatal(fake_run, recs, tmp_path):
    lines = [json.dumps(r) for r in recs[:5]]
    bad = copy.deepcopy(recs[5])
    bad["turns"][0]["text"] = "hi <|end|>"
    p = tmp_path / "accepted-00000.jsonl"
    p.write_text("\n".join(lines + [json.dumps(bad), "", "[1, 2]"]) + "\n" + lines[0][:40])
    m = FP.run([str(p)], str(tmp_path / "o"), allow_nontrainable=True)
    assert m["counts"]["read"] == 8 and m["counts"]["admitted"] == 5
    assert m["refused"] == {"ROLE_TOKEN_IN_TEXT": 1, "BAD_JSON": 2}
    assert [e["line"] for e in m["refused_examples"]] == [6, 8, 9]


def test_rotation_determinism_and_out_dir_rules(fake_run, recs, tmp_path):
    a = FP.run([fake_run], str(tmp_path / "a"), allow_nontrainable=True, shard_size=50)
    b = FP.run([fake_run], str(tmp_path / "b"), allow_nontrainable=True, shard_size=50)
    c = FP.run([fake_run], str(tmp_path / "c"), allow_nontrainable=True)
    assert len(a["shards"]) == -(-len(recs) // 50) and len(c["shards"]) == 1
    read = lambda m: b"".join(open(p, "rb").read() for p in m["shards"])  # noqa: E731
    assert read(a) == read(b) == read(c)
    assert [len(open(p).readlines()) for p in a["shards"]][:-1] == [50] * (len(a["shards"]) - 1)
    with pytest.raises(SystemExit, match="not empty"):
        FP.run([fake_run], str(tmp_path / "a"), allow_nontrainable=True)
    (tmp_path / "d.partial").mkdir()
    with pytest.raises(SystemExit, match="interrupted"):
        FP.run([fake_run], str(tmp_path / "d"), allow_nontrainable=True)
    with pytest.raises(SystemExit, match="twice"):
        FP.run([fake_run, os.path.join(fake_run, "accepted")], str(tmp_path / "e"), allow_nontrainable=True)
    assert not os.path.exists(tmp_path / "a.partial") and not os.path.exists(tmp_path / "e")
