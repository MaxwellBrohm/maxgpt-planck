"""SPEC 9 scorer fixtures: a uniform model scores chance exactly, a hand-built copy-the-latest model scores IDEAL,
padding never moves a score, checkpoints load with a weights hash."""
import functools
import math

import numpy as np
import pytest
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
    # K3 round 16 (2026-11-21, REVIEW 17b): P's persistence marker is read, not assumed: the uniform model's argmax
    # is token 0, never the marker y, so no P item reads a right marker (mutant r17c_marker_always: every one did)
    ps = [o for o in res if "marker" in o]
    assert ps and not any(o["marker"] for o in ps), len(ps)


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


def test_skill_candidates_are_spec_9c():
    """K3 round 15 (REVIEW 16b T-3): SPEC 9 C's candidate set, over which every LIK, CE_C and chance reading is taken,
    read from written skill records (evalsets.skill_set, K-EVAL-ID's draw): the gold, every V value the record holds
    (statements, earlier answers), exactly 3 V values it does not hold, and " none" unless " none" is the gold; no
    candidate twice. Nothing read a record's cands before (mutant r16_cands_two_extra: 2 outside values)."""
    lx, n_none = lex(), 0
    for rec in items():
        gold, c = rec["qs"][-1]["gold"], rec["cands"]
        inside = {x for t in rec["turns"] for x in t["ids"] if x in lx.val} - {gold}
        out = [x for x in c if x != gold and x not in inside and x != K.NONE]
        assert len(c) == len(set(c)) and gold in c and inside <= set(c), (rec["id"], c)
        assert len(out) == 3 and all(x in lx.v_only for x in out), (rec["id"], out)
        assert c.count(K.NONE) == 1, (rec["id"], c)           # the gold or the appended one
        assert len(c) == 1 + len(inside) + 3 + (gold != K.NONE), (rec["id"], len(c), len(inside))
        n_none += gold == K.NONE
    assert 0 < n_none < len(items()), n_none          # both kinds of record were read (A's gold is " none")


def test_conflict_candidates_hold_the_stated_and_the_memorized_value():
    """K3 round 16 (2026-11-21, REVIEW 17b): SPEC 9's CONFLICT reading splits picks into context-follow (the stated
    value, the gold), memory (the FB value) and other, so each CONFLICT record's candidates must hold both, with
    " none" once and no repeats; its CONSISTENT twin states the FB value, so there gold and memory are one value.
    Nothing read a CONFLICT record's cands before (mutant r17_conflict_cands_no_memory: the FB value left out, so
    every memory rate would read 0)."""
    w, _ = facts()
    lx = lex()
    recs = E.conflict(w, pools(), list(range(40)), 1, 64)
    kinds = {"conflict": 0, "consistent": 0}
    for rec in recs:
        c, gold, mem = rec["cands"], rec["qs"][-1]["gold"], rec["mem"]
        assert mem == int(w.values[rec["ent"], rec["attr"]]), rec["id"]
        assert len(c) == len(set(c)) and c.count(K.NONE) == 1 and gold in c and mem in c, (rec["id"], c)
        assert (gold != mem) == (rec["fam"] == "conflict"), rec["id"]
        assert all(x in lx.v_only for x in c if x not in (gold, mem, K.NONE)), (rec["id"], c)
        kinds[rec["fam"]] += 1
    assert kinds == {"conflict": 64, "consistent": 64}


def test_k5_hit_sets_are_spec_8s():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutants r17_cf_hit_not_counterfactual and r17_unseen_hit_world_name
    survived): SPEC 8's K5 hit sets read the tool against memory, so each must hold what its name says. FB-HIT: the
    tool states the entity's FB values, gold = memory. CF-HIT: the tool states a V value for the asked attribute that
    is none of the entity's FB values, gold = that stated value, memory = the FB value. UNSEEN-HIT: a name drawn from
    the T pools (no world entity's name), no memory. Every tool line states gold for the asked attribute."""
    w, _ = facts()
    pl = pools()
    recs = E.k5_sets(w, pl, list(range(40)))
    names = {tuple(int(x) for x in n) for n in w.names}
    T = [set(int(x) for x in p) for p in pl.triple("T")]
    seen = {"FB-HIT": 0, "CF-HIT": 0, "UNSEEN-HIT": 0}
    for r in recs:
        if r["set"] not in seen:
            continue
        seen[r["set"]] += 1
        tool, user = r["turns"][0]["ids"], r["turns"][1]["ids"]
        nm, a = user[:3], user[3]
        assert tool[:5] == [K.LOOKUP, *nm, K.LOOKUP_E] and tool[5] == K.RESULT and tool[-1] == K.RESULT_E, r["id"]
        body = dict(zip(tool[6:-1:2], tool[7:-1:2]))
        assert int(a) == int(w.attrs[r["attr"]]) and body[a] == r["gold"], r["id"]
        if r["set"] == "UNSEEN-HIT":
            assert r["ent"] == -1 and r["mem"] == -1 and tuple(nm) not in names
            assert all(x in t for x, t in zip(nm, T)), r["id"]
            continue
        fb = [int(v) for v in w.values[r["ent"]]]
        assert nm == [int(x) for x in w.names[r["ent"]]] and r["mem"] == fb[r["attr"]], r["id"]
        if r["set"] == "FB-HIT":
            assert r["gold"] == r["mem"] and sorted(body.values()) == sorted(fb), r["id"]
        else:
            assert r["gold"] != r["mem"] and r["gold"] not in fb and r["gold"] in set(pl["V"].tolist()), r["id"]
    assert seen == {"FB-HIT": 512, "CF-HIT": 2048, "UNSEEN-HIT": 512}


def test_summary_chance_and_stored_bits_by_hand():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutants r17c_chance_off and r17c_bits_one_order survived): SPEC 9's chance
    is the mean of 1/|C| (here |C| 4 and 8: 0.1875), and stored bits average CE over a key's BIO prompts, both orders,
    before the mean over keys: S = N x 6 x (log2 |V| - mean CE / ln 2), by hand on two keys of two orders each."""
    rows = [{"set": "X", "fam": "R", "lik": 1.0, "n_c": 4, "ce_c": 0.5, "gen": True},
            {"set": "X", "fam": "R", "lik": 0.0, "n_c": 8, "ce_c": 1.5, "gen": False}]
    assert PR.summarize(rows)["X/R"] == {"n": 2, "lik": 0.5, "chance": 0.1875, "ce_c": 1.0, "gen": 0.5}
    res = [{"ent": 0, "attr": 0, "ce_full": 1.0}, {"ent": 0, "attr": 0, "ce_full": 3.0},
           {"ent": 1, "attr": 2, "ce_full": 2.0}, {"ent": 1, "attr": 2, "ce_full": 4.0}]
    ce = (2.0 + 3.0) / 2                                    # each key's two orders averaged, then the keys
    assert PR.stored_bits(res, 10) == pytest.approx(10 * K.N_ATTR * (11.0 - ce / math.log(2)))
