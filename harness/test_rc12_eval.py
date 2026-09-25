"""RC-12 responder for Planck checkpoints (rc12/planck_responder.py) driven by the real rc12 runner.

A random (scrambled) ~0.5M model and the toy byte-level tokenizer, CPU only. Scores are near zero by design;
what is checked is the plumbing: the prompt is the training render, greedy is argmax of the real forward,
sampling is seeded per reply and leaves the global RNG alone, the history is the model's own, the context limit
and the truncation rule hold, and the transcripts have the fake responders' schema and score through score.py.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

import rc12_eval as E
import toy_tokenizer as TT
from config import PlanckConfig
from count_params import count_module
from model import build_model
from testutil import scramble

E.use_rc12()
import fakes_family as FF        # noqa: E402  (rc12)
import hf_responder as HR        # noqa: E402
import planck_responder as PR    # noqa: E402
import runner as RN              # noqa: E402

MODEL = dict(vocab_size=TT.VOCAB, d_model=96, n_layers=4, n_heads=3, n_kv_heads=1, head_dim=32,
             mlp_hidden=320, seq_len=512)
NEEDED = ("S", "C", "I", "O", "Q", "T")          # SPEC s1 kinds whose loss fails a later probe
STOPS = {"eot", "eos", "role", "cap"}


def random_model(seed=3, **kw):
    torch.manual_seed(seed)
    return scramble(build_model(PlanckConfig(**{**MODEL, **kw})), seed)


def spy(model):
    """counts embedding calls (one per model or decoder forward) and records the grad mode they ran in."""
    seen = {"calls": 0, "grad": set()}

    def hook(mod, inp, out):
        seen["calls"] += 1
        seen["grad"].add(torch.is_grad_enabled())
    model.tok_emb.register_forward_hook(hook)
    return seen


@pytest.fixture(scope="module")
def tok():
    return TT.build()


def responder(model, tok, **kw):
    r = PR.PlanckResponder(model, tok, PR.template_for({}, tok), MODEL["seq_len"], PR.eos_for({}, tok), **kw)
    r.keep_history = True
    return r


def test_model_size_and_role_ids(tok):
    assert 0.45e6 < count_module(random_model())["total"] < 0.55e6
    t = PR.template_for({}, tok)
    assert (t.role_ids, t.end_id, PR.eos_for({}, tok)) == ({"system": 2, "user": 3, "assistant": 4, "tool": 6}, 5, [1])
    with pytest.raises(ValueError):          # a config id that contradicts the tokenizer is refused
        PR.template_for({"chat": {"role_ids": {"user": 4, "assistant": 3}, "end_id": 5}}, tok)


def test_prompt_is_the_training_render(tok):
    rec = RN.load()[0]
    u = [t["text"] for t in rec["turns"]]
    msgs = [{"role": "user", "content": u[0]}, {"role": "assistant", "content": "Sure, noted."},
            {"role": "user", "content": u[1]}]
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids   # noqa: E731
    want = [3] + enc(u[0]) + [5] + [4] + enc("Sure, noted.") + [5] + [3] + enc(u[1]) + [5] + [4]
    r = responder(random_model(), tok)
    assert r.encode(msgs) == want and r.count(msgs) == len(want)
    plain = responder(random_model(), tok, render="plain")
    assert plain.encode(msgs) == enc(f"User: {u[0]}\nAssistant: Sure, noted.\nUser: {u[1]}\nAssistant:")


def test_greedy_is_argmax_of_the_full_forward(tok):
    m = random_model()
    r, ref = responder(m, tok), responder(m, tok, cache=False)
    kinds = set()
    for rec in RN.load()[:3]:
        r.start(rec, None, "template")
        ref.start(rec, None, "template")
        for t in rec["turns"][:4]:
            msgs = [{"role": "user", "content": t["text"]}]
            text, stop = r.reply(msgs, t["i"])
            assert (text, stop) == ref.reply(msgs, t["i"]), "KV-cached and full-forward replies differ"
            ids = r.encode(msgs)
            out, n = r.log[-1]["new_ids"], r.log[-1]["n_new"]
            assert len(out) == n and text == tok.decode(out, skip_special_tokens=False)
            with torch.no_grad():
                arg = m(torch.tensor([ids + out]))[0][0].argmax(-1).tolist()
            assert arg[len(ids) - 1:len(ids) - 1 + n] == out, "a greedy token is not the argmax"
            assert 5 not in out and 1 not in out and not set(out[:-1]) & {2, 3, 4, 6}, "a stop token was passed"
            if stop == "eot":
                assert arg[len(ids) - 1 + n] == 5
            elif stop == "eos":
                assert arg[len(ids) - 1 + n] == 1
            elif stop == "role":
                assert out[-1] in (2, 3, 4, 6)
            else:
                assert stop == "cap" and n == min(256, 512 - len(ids))
            kinds.add(stop)
    assert len(kinds) >= 2, kinds


def test_sampling_is_seeded_per_reply_and_leaves_global_state(tok):
    m = random_model()
    seen = spy(m)
    m.train()
    r, ref = responder(m, tok), responder(m, tok, cache=False)
    rec = RN.load()[0]
    msgs = [{"role": "user", "content": rec["turns"][0]["text"]}]
    rng = torch.get_rng_state()
    outs = {}
    for seed in (1, 1, 2):
        r.start(rec, seed, "template")
        outs.setdefault(seed, []).append([r.reply(msgs, i)[0] for i in (1, 2, 3)])
    ref.start(rec, 1, "template")
    assert [ref.reply(msgs, i)[0] for i in (1, 2, 3)] == outs[1][0]
    assert torch.equal(rng, torch.get_rng_state()), "sampling touched the global torch RNG"
    ids = r.encode(msgs)
    for x in r.log[:3]:                  # replay: softmax(logits / 0.6) drawn from md5(seed, conversation, turn)
        g = torch.Generator().manual_seed(HR.conv_seed(1, rec["id"], x["turn"]))
        seq = ids + x["new_ids"]
        with torch.no_grad():
            logits = m(torch.tensor([seq]))[0][0]
        draws = [int(torch.multinomial(torch.softmax(logits[len(ids) - 1 + k] / 0.6, -1), 1, generator=g))
                 for k in range(len(x["new_ids"]) + (x["stop"] in ("eot", "eos")))]
        assert draws[:x["n_new"]] == x["new_ids"] and len(draws) > 0
        if x["stop"] in ("eot", "eos"):
            assert draws[-1] == (5 if x["stop"] == "eot" else 1)
    assert outs[1][0] == outs[1][1], "same seed, conversation and turn must give the same reply"
    assert outs[1][0] != outs[2][0] and len(set(outs[1][0])) > 1
    assert m.training, "train/eval mode not restored"
    assert seen["grad"] == {False} and seen["calls"] > 3, "generation ran with grad enabled"


def test_context_room_caps_generation(tok):
    m = random_model()
    seen = spy(m)
    r = responder(m, tok)
    r.start(RN.load()[0], None, "template")
    long = {"role": "user", "content": "word " * 100}          # 500 bytes + 3 specials
    r.max_new = 256
    text, stop = r.reply([long], 1)
    room = r.log[-1]["room"]
    assert 0 < room < 256 and r.log[-1]["n_new"] <= room and r.short == 1
    calls = seen["calls"]
    text, stop = r.reply([{"role": "user", "content": "word " * 110}], 2)
    assert (text, stop, seen["calls"]) == ("", "cap", calls), "a full prompt must not run the model"


@pytest.fixture(scope="module")
def e2e(tok, tmp_path_factory):
    recs = E.subset(RN.load(), None, 1)                         # one per family, BIND as a pair: 13
    r = responder(random_model(), tok)
    out = tmp_path_factory.mktemp("e2e")
    rows = RN.run(recs, r, "template", [None], r.ctx, str(out / "planck"), "planck-random")
    fake = RN.run(recs, FF.make("IDEAL"), "template", [None], None, str(out / "ideal"), "IDEAL")
    return dict(recs={x["id"]: x for x in recs}, rows=rows, fake=fake, r=r, out=out)


def test_e2e_writes_the_fake_schema(e2e):
    back = [json.loads(line) for line in open(e2e["out"] / "planck" / "transcripts.jsonl")]
    assert back == json.loads(json.dumps(e2e["rows"])) and len(back) == 13
    assert (e2e["out"] / "planck" / "scores.jsonl").stat().st_size > 0
    for row, ideal in zip(back, e2e["fake"]):
        assert row["id"] == ideal["id"] and set(row) == set(ideal)
        assert [set(t) for t in row["turns"]] == [set(t) for t in ideal["turns"]]
        for p, q in zip(row["probes"], ideal["probes"]):
            assert set(p) - {"lost"} == set(q), (row["id"], p, q)
        for t in row["turns"]:
            assert t["stop"] in STOPS and isinstance(t["reply"], str) and isinstance(t["dropped"], int)


def test_e2e_history_is_the_models_own(e2e):
    by_turn = {(x["rid"], x["turn"]): x for x in e2e["r"].log}
    for row in e2e["rows"]:
        replies = {t["i"]: t["reply"] for t in row["turns"]}
        users = {t["i"]: t["user"] for t in row["turns"]}
        for t in row["turns"]:
            h = by_turn[(row["id"], t["i"])]["history"]
            first = t["dropped"] + 1
            assert len(h) == 2 * (t["i"] - t["dropped"]) - 1
            for k, m in enumerate(h):
                q = first + k // 2
                assert m == ({"role": "user", "content": users[q]} if k % 2 == 0
                             else {"role": "assistant", "content": replies[q]})


def test_e2e_stop_reasons(e2e):
    for x in e2e["r"].log:
        ids = x["new_ids"]
        assert 5 not in ids and 1 not in ids and not set(ids[:-1]) & {2, 3, 4, 6}, "a stop token was passed"
        assert (x["stop"] == "role") == bool(ids and ids[-1] in (2, 3, 4, 6))
        assert (x["stop"] == "cap") == (x["n_new"] == min(256, x["room"]) and x["stop"] != "role")
    assert {x["stop"] for x in e2e["r"].log} == STOPS          # every stop reason occurs in this run


def test_e2e_context_limit_and_truncation_rule(e2e):
    log = e2e["r"].log
    assert all(x["prompt_len"] + x["n_new"] <= 512 for x in log)
    assert all(x["prompt_len"] <= 512 - 256 for x in log if len(x["history"]) > 1)
    assert sum(t["dropped"] > 0 for row in e2e["rows"] for t in row["turns"]) > 20
    n_trunc = 0
    for row in e2e["rows"]:
        rec = e2e["recs"][row["id"]]
        for p, res in zip(rec["probes"], row["probes"]):
            need = {t["i"] for t in rec["turns"] if t["i"] < p["turn"] and t["kind"] in NEEDED} | set(p["src"] or [])
            lost = bool(need) and min(need) <= row["turns"][p["turn"] - 1]["dropped"]
            assert ("trunc" in res["fails"]) == lost and (not lost or not res["ok"]), (row["id"], p["turn"])
            n_trunc += lost
    assert n_trunc > 0


def test_e2e_scores_through_score_py(e2e):
    path = e2e["out"] / "planck" / "transcripts.jsonl"
    rc = subprocess.run([sys.executable, "-B", "score.py", str(path)], cwd=E.RC12, capture_output=True, text=True,
                        timeout=60)
    assert rc.returncode == 0, rc.stderr[-500:]
    s = json.loads(rc.stdout)
    assert s["R"] is not None and 0 <= s["R"] <= 15, s["R"]
    assert set(s["families"]) == {"RECALL", "CORR", "BIND", "TWOHOP", "PERSIST", "OWN", "TOPIC", "ROLE", "LOOKUP",
                                  "LOOP"}
