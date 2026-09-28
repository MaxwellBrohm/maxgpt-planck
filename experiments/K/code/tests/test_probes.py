"""SPEC 9 scorer fixtures: a uniform model scores chance exactly, a hand-built copy-the-latest model scores IDEAL,
padding never moves a score, checkpoints load with a weights hash."""
import functools
import math

import numpy as np
import torch

import evalsets as E
import kcommon as K
import oracles as O
import probes as PR
import world as Wd
from kfix import lex, pools, tiny_model

VOCAB = 8192
ROLES = {K.SYS: "system", K.USER: "user", K.ASST: "assistant", K.TOOL: "tool"}


@functools.lru_cache(maxsize=1)
def items():
    return E.skill_set(pools(), lex(), "eval", K.SEEDS["eval"], 24, tag="K-EVAL-ID")


@functools.lru_cache(maxsize=1)
def facts():
    w = Wd.make_world(pools())
    return w, E.bio_prompts(w, pools(), list(range(40)), 1, "FB-PROBE", csize=8)


def split_turns(ids):
    turns, cur = [], None
    for t in ids:
        if t in ROLES and cur is None:
            cur = {"role": ROLES[t], "ids": []}
        elif t == K.END and cur is not None:
            turns.append(cur)
            cur = None
        else:
            cur["ids"].append(t)
    return turns, cur


def ideal_fn(prompts):
    """Copy-the-latest model built from the world rule on the prompt's own tokens (and the world for fact prompts)."""
    w, _ = facts()
    names = {tuple(n): i for i, n in enumerate(w.names[:1000].tolist())}
    out = torch.zeros(len(prompts), VOCAB)
    for i, p in enumerate(prompts):
        if p[0] not in ROLES:                                    # bio prompt: F M L (a v)* a
            ans = int(w.values[names[tuple(p[:3])], list(w.attrs).index(p[-1])])
        else:
            turns, open_ = split_turns(p)
            if open_["ids"] and (open_["ids"][-1] in lex().val or open_["ids"][-1] == K.NONE):
                ans = K.END
            elif not open_["ids"] and turns[0]["role"] == "system":
                ans = turns[0]["ids"][0]
            elif turns[-1]["role"] == "tool" or (len(turns) == 1 and turns[0]["ids"][0] in lex().pos[0]):
                ans = int(w.values[names[tuple(turns[-1]["ids"][:3])], list(w.attrs).index(turns[-1]["ids"][3])])
            else:
                rec = {"turns": turns + [{"role": "assistant", "ids": [0]}]}
                st, qs, _ = O.parse(rec, lex())
                ans = O.answers(st, qs[-1], [], lex())["IDEAL"]
        out[i, ans] = 50.0
    return out


def test_uniform_model_scores_chance():
    res = PR.score(lambda ps: torch.zeros(len(ps), VOCAB), items(), pools()["V"])
    lik = np.mean([o["lik"] for o in res])
    chance = np.mean([1 / o["n_c"] for o in res])
    assert abs(lik - chance) < 1e-12 and chance < 0.5
    assert all(abs(o["ce_c"] - math.log(o["n_c"])) < 1e-4 for o in res)
    assert not any(o["gen"] for o in res)


def test_ideal_model_scores_one():
    res = PR.score(ideal_fn, items(), pools()["V"], batch=7)
    assert all(o["lik"] == 1.0 and o["gen"] for o in res)
    ps = [o for o in res if o["fam"] == "P"]
    assert ps and all(o["marker"] for o in ps)
    _, fp = facts()
    fr = PR.score(ideal_fn, fp, pools()["V"])
    assert all(o["lik"] == 1.0 and o["greedy"] == o["gold"] for o in fr)
    assert abs(PR.stored_bits(fr, 40, "ce_v") - 40 * 6 * 11) < 1.0


def test_uniform_over_v_stores_no_latent_bits():
    _, fp = facts()
    res = PR.score(lambda ps: torch.zeros(len(ps), VOCAB), fp, pools()["V"])
    assert abs(PR.stored_bits(res, 40, "ce_v")) < 1e-3
    assert PR.stored_bits(res, 40, "ce_full") < 0


def test_padding_and_batching_do_not_move_scores():
    m = tiny_model(vocab=VOCAB, d_model=32, n_layers=2, mlp_hidden=24, seq_len=512)
    fn = PR.model_fn(m)
    a = PR.score(fn, items()[:40], pools()["V"], batch=1)
    b = PR.score(fn, items()[:40], pools()["V"], batch=16)
    for x, y in zip(a, b):
        assert abs(x["ce_c"] - y["ce_c"]) < 1e-4 and x["greedy"] == y["greedy"]


def test_checkpoint_load_and_weights_hash(tmp_path):
    m = tiny_model(vocab=VOCAB, seq_len=512)
    p = str(tmp_path / "ck.pt")
    torch.save({"model": m.state_dict(), "model_cfg": m.cfg.to_dict()}, p)
    m2, sha = PR.load(p)
    idx = torch.tensor([[3, 100, 200, 5, 4]])
    assert torch.equal(m(idx)[0], m2(idx)[0])
    assert sha == PR.weights_sha(m)
    with torch.no_grad():
        m2.blocks[0].mlp.up_proj.weight[0, 0] += 1e-3
    assert PR.weights_sha(m2) != sha


def test_prompts_end_where_gold_is_predicted():
    for rec in items():
        p = PR.prompts_of(rec)
        q = rec["qs"][-1]
        ans = rec["turns"][q["turn"]]["ids"]
        assert p["main"][-1] == (ans[0] if len(ans) == 2 else K.ASST) and p["gold"] == ans[-1]
        tail = p["main"][-4:-1] if len(ans) == 2 else p["main"][-3:]
        assert tail == [K.Q, K.END, K.ASST]


def test_gen_is_strict_about_end():
    """A model that says the gold answer and then keeps talking (no <|end|>) gets LIK but not GEN."""
    def babble(prompts):
        out = ideal_fn(prompts)
        for i, p in enumerate(prompts):
            if out[i].argmax() == K.END:
                out[i] = 0.0
                out[i, int(pools()["V"][0])] = 50.0
        return out
    res = PR.score(babble, items(), pools()["V"])
    assert all(o["lik"] == 1.0 for o in res) and not any(o["gen"] for o in res)
