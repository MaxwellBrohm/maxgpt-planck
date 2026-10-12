"""SPEC 5: every purity, balance and shortcut check has a planted violation that must fail it."""
import copy
import functools

import numpy as np

import items as I
import kcommon as K
import purity as U
import skill as S
from kfix import lex, pools


@functools.lru_cache(maxsize=None)
def clean(split="train", n=64):
    g = S.Gen(pools(), split)
    out = []
    for fam in "RUBFAP":
        out += [I.render(it, f"{fam}.b0.i{i}") for i, it in enumerate(S.block(g, fam, n, (5, ord(fam), 0)))]
    return tuple(out)


def run(recs, split="train", tok=None):
    return U.accept(list(recs), pools(), split, lex(), tok, block=64)


def first(recs, fam):
    return next(i for i, r in enumerate(recs) if r["fam"] == fam)


def test_clean_blocks_pass():
    assert run(clean())["fails"] == []
    assert run(clean("eval"), "eval")["fails"] == []


def _plant(split, fn):
    recs = copy.deepcopy(list(clean(split)))
    fn(recs)
    return run(recs, split)


def test_fb_name_in_skill_item_fails():
    def f(recs):
        recs[0]["turns"][0]["ids"][0] = int(pools()["F1"][0])
    assert any("fb_or_filler_name" in x for x in _plant("train", f)["fails"])


def test_skill_train_token_in_eval_fails():
    def f(recs):
        recs[0]["turns"][0]["ids"][-1] = int(pools()["S2"][0])
    assert any("skill_train_token" in x for x in _plant("eval", f)["fails"])


def test_kevs_triple_in_training_fails():
    def f(recs):
        t = [int(x) for x in pools().kevs[0]]
        recs[0]["turns"][0]["ids"] = t + recs[0]["turns"][0]["ids"][3:] if recs[0]["turns"][0]["ids"][0] in \
            lex().pos[0] else recs[0]["turns"][0]["ids"]
        recs[1]["turns"].insert(0, {"role": "user", "ids": t + [int(pools()["A"][0]), int(pools()["V"][0])]})
        recs[1]["turns"].insert(1, {"role": "assistant", "ids": [K.OK]})
        for q in recs[1]["qs"]:
            q["turn"] += 2
    assert any("kevs_triple" in x for x in _plant("train", f)["fails"])


def test_heldout_value_in_training_fails():
    def f(recs):
        i = first(recs, "R")
        t = recs[i]["turns"][0]["ids"]
        t[-1] = int(pools()["VH"][0])
    assert any("heldout_token" in x for x in _plant("train", f)["fails"])


def test_ack_with_value_fails():
    def f(recs):
        recs[0]["turns"][1]["ids"] = [K.OK, int(pools()["V"][5])]
    assert any("ack" in x for x in _plant("train", f)["fails"])


def test_wrong_gold_fails():
    def f(recs):
        i = first(recs, "U")
        t = recs[i]["qs"][-1]["turn"]
        recs[i]["turns"][t]["ids"][-1] = int(pools()["V"][7])
    assert any("world rule" in x for x in _plant("train", f)["fails"])


def test_form_balance_break_fails():
    def f(recs):
        i = first(recs, "B")
        t = recs[i]["qs"][-1]["turn"] - 1
        ids = recs[i]["turns"][t]["ids"]
        recs[i]["turns"][t]["ids"] = [ids[3], *ids[:3], K.Q] if ids[0] in lex().pos[0] else [*ids[1:4], ids[0], K.Q]
    assert any("form" in x for x in _plant("train", f)["fails"])


def test_recency_shortcut_fails():
    """Move the asked key's latest statement to the end of the last statement turn in every R item: O2 and the
    last-statement rate must both flag it."""
    def f(recs):
        for r in recs:
            if r["fam"] != "R":
                continue
            q = r["qs"][-1]
            key = q["ask"]
            turns = [t for t in r["turns"][:q["turn"]] if t["role"] == "user" and t["ids"][-1] != K.Q]
            for t in turns:
                ids = t["ids"]
                for j in range(0, len(ids) - 4):
                    if ids[j:j + 4] == key or (ids[j] == key[3] and ids[j + 1:j + 4] == key[:3]):
                        stmt = ids[j:j + 5]
                        del ids[j:j + 5]
                        turns[-1]["ids"] += stmt
                        break
            r["turns"] = [t for t in r["turns"] if t["role"] != "user" or t["ids"]]
    fails = _plant("train", f)["fails"]
    assert any("R: O2" in x for x in fails) or any("R: last-statement" in x for x in fails), fails


def test_round_trip_break_fails():
    tok = K.load_tokenizer()
    v = tok.get_vocab()
    pair = None
    for s, i in sorted(v.items(), key=lambda x: x[1])[200:3000]:
        if s.startswith("\u0120") and len(s) > 3 and s[:2] in v and s[2:] in v:
            pair = [v[s[:2]], v[s[2:]]]
            break
    assert pair is not None
    recs = copy.deepcopy(list(clean()))
    recs[0]["turns"][1]["ids"] = pair
    rep = U.purity(recs, pools(), "train", tok)
    assert rep[0]["round_trip"] >= 1
    assert U.purity(list(clean()), pools(), "train", tok)[0]["round_trip"] == 0


def test_thresholds_catch_a_raised_rate():
    import oracles as O
    tab = {"U": {**{o: 0.0 for o in O.NAMES}, "IDEAL": 1.0, "n": 512}}
    assert U.threshold_fails(tab, {}) == []
    tab["U"]["O1"] = 0.60
    assert any("O1" in x for x in U.threshold_fails(tab, {}))
    assert U.threshold_fails({"B": {**tab["U"], "O1": 0.0, "O3_12": 0.6}}, {}) == []
    assert U.threshold_fails({}, {"O3_12": 0.3, "n": 96}) and not U.threshold_fails({}, {"O3_12": 0.2, "n": 96})
    assert U.threshold_fails({}, {"B-ADAPT": 0.3, "n": 96})          # K3 round 3: B cube groups (limit 0.226)
    assert np.isclose(0.5 + 3 * np.sqrt(0.25 / 512), 0.5663, atol=1e-4)
