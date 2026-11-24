"""SPEC 5: every purity, balance and shortcut check has a planted violation that must fail it."""
import copy
import functools
import math
from collections import Counter

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


def _frame_plant(fams, blocks, seed, plant=True):
    """REVIEW 14b's frame_r_40.py plant, on every item: the key's latest statement, when named (full or reord), takes
    the question's form (full for "F M L a ?", reord for "a F M L ?"), an item-level rewrite before rendering (the
    world rule, the gold and every rule blind to statement form unchanged). plant False: the same draw untouched."""
    out = []
    for f in fams:
        g = S.Gen(pools(), "train")
        for b in range(blocks):
            for i, it in enumerate(S.block(g, f, 512, (seed, ord(f), b))):
                q = it.ex[-1][1]
                xi, j = [(xi, j) for xi, (k, body) in enumerate(it.ex) if k == "S" for j, s in enumerate(body)
                         if (s.ent, s.attr) == (q.ent, q.attr)][-1]
                s = it.ex[xi][1][j]
                if plant and s.kind in ("full", "reord"):
                    body = list(it.ex[xi][1])
                    body[j] = I.St("full" if q.form == 0 else "reord", s.ent, s.attr, s.val, s.alias)
                    it = I.Item(it.fam, it.names, it.ex[:xi] + [("S", body)] + it.ex[xi + 1:], it.marker, it.meta)
                out.append(I.render(it, f"{f}.b{b}.i{i}"))
    return out


def test_o5_bars_fail_a_frame_copy_cue_on_r_and_p():
    """K3 round 13 (REVIEW 14b T-2): O5n's bars (SPEC 5: <= 0.65 on every family) are the only check that fails a
    frame-copy cue on R and P (the gate's models read it at 0.63-0.66). The frame plant on 8 blocks of 512 R and P
    (4,096 each, the tolerance 0.65 + 3 sqrt(0.65 x 0.35 / 4096) = 0.672) reads O5_2 and O5_3 at 0.69-0.71 and fails
    exactly those four cells; the same draw unplanted passes. Each family's O5_1-3 cell fails just over its tolerance
    and passes just under it (mutant m14_o5_bars_loose: every O5 bar 0.95)."""
    import math
    import oracles as O
    rep = U.accept(_frame_plant("RP", 8, 1331), pools(), "train", lex(), None, block=512)
    assert sorted(" ".join(x.split()[:2]) for x in rep["fails"]) == ["P: O5_2", "P: O5_3", "R: O5_2", "R: O5_3"], (
        rep["fails"])
    clean_rep = U.accept(_frame_plant("RP", 8, 1331, plant=False), pools(), "train", lex(), None, block=512)
    assert clean_rep["fails"] == [] and max(clean_rep["oracles"][f]["O5_2"] for f in "RP") < 0.5, clean_rep["fails"]
    lim = 0.65 + 3 * math.sqrt(0.65 * 0.35 / 512)
    for fam in "RUBFAP":
        for o in ("O5_1", "O5_2", "O5_3"):
            tab = {fam: {**{x: 0.0 for x in [*O.NAMES, *O.TH]}, "IDEAL": 1.0, "n": 512, o: lim + 0.002}}
            assert U.threshold_fails(tab, {}) == [f"{fam}: {o} {lim + 0.002:.3f} > 0.65"], (fam, o)
            tab[fam][o] = lim - 0.002
            assert U.threshold_fails(tab, {}) == [], (fam, o)


def _cut(recs, fams):
    """recs with one record per family cut: its first statement turn loses its last token (it no longer parses)."""
    recs = copy.deepcopy(list(recs))
    for fam in fams:
        t = next(t for t in recs[first(recs, fam)]["turns"] if t["role"] == "user" and t["ids"][-1] != K.Q)
        t["ids"] = t["ids"][:-1]
    return recs


def test_unparseable_records_fail_acceptance():
    """K3 round 13 (REVIEW 14b T-3): a record that does not parse fails acceptance; it never leaves the tables
    silently. One cut record per family in the train blocks (rules on) and per family of an OOD-DFAR set read as
    gen.py check reads the OOD sets (rules off, where nothing else would notice) (mutant m14_parse_fail_dropped)."""
    fails = run(_cut(clean(), "RUBFAP"))["fails"]
    assert [x for x in fails if "do not parse" in x] == [x for x in fails if x.startswith("6 items do not parse, ")
                                                          ] and len([x for x in fails if "do not parse" in x]) == 1, fails
    go = S.Gen(pools(), "eval", ood="DFAR")
    ood = [I.render(it, f"{f}.b0.i{i}") for f in "RU" for i, it in enumerate(S.block(go, f, 64, (14302, ord(f), 0)))]
    assert U.accept(ood, pools(), "eval", lex(), None, block=512, rules=False)["fails"] == []
    fails = U.accept(_cut(ood, "RU"), pools(), "eval", lex(), None, block=512, rules=False)["fails"]
    assert len(fails) == 1 and fails[0].startswith("2 items do not parse, first: R.b0.i0: "), fails


def test_filler_name_in_skill_item_fails():
    """K3 round 13 (REVIEW 14b T-4): SPEC 5's "0 FB or filler name tokens in skill items" covers the filler pools
    Z1-Z3, which test_fb_name_in_skill_item_fails (an F1 token) never planted. In the first item of each family one
    name triple is replaced everywhere by a Z1, Z2, Z3 triple (a consistent relabeling: parsing, the world rule and
    every oracle unchanged); purity fails it on train and eval, and nothing else does (mutant
    m14_purity_filler_names_off)."""
    import oracles as O
    p = pools()
    z = [int(p[f"Z{i}"][0]) for i in (1, 2, 3)]

    def f(recs):
        done = set()
        for r in recs:
            if r["fam"] in done:
                continue
            nm = list(O.parse(r, lex())[2][0])
            for t in r["turns"]:
                for j in range(len(t["ids"]) - 2):
                    if t["ids"][j:j + 3] == nm:
                        t["ids"][j:j + 3] = z
            done.add(r["fam"])
    for split in ("train", "eval"):
        rep = _plant(split, f)
        assert rep["purity"]["fb_or_filler_name"] == 6 and rep["fails"] == ["purity fb_or_filler_name: 6"], (
            split, rep["fails"])


def test_full_block_shares_are_exact():
    """K3 round 14 (REVIEW 15b T-8): SPEC 5's per-block shares hold exactly in every full block (purity.balance's
    `exact`; the k and rank cells excepted, largest remainder). Where m divides the block, as in every block gen.py
    writes (512) and these tests use (64), one item off already moves sum x m by m, so "within m items" is exact
    there; the clause binds on a full block m does not divide, where no count is 1/m. clean()'s 64-item F block read
    whole passes; with two items dropped that are neither key_named nor follow_alias it is a full block of 62 whose
    key_named and follow_alias counts (16 of 62) cannot be 1/4 and fail (mutant r15_block_exact_off: within m
    items, 16 x 4 - 62 = 2, so they pass). The drop also moves the loose k and rank cells; only these two are read.
    This test found that key_named (and key_alias) were exempt as k cells by their leading "k" (purity.py fixed in
    round 14; mutant r14_exact_k_prefix restores the old test: key_named passes)."""
    facts = [U.item_facts(r, lex()) for r in clean() if r["fam"] == "F"]
    assert len(facts) == 64 and U.balance(facts, 64)[1] == []
    drop = [i for i, f in enumerate(facts) if not f["key_named"] and not f["follow_alias"]][-2:]
    kept = [f for i, f in enumerate(facts) if i not in drop]
    assert sum(f["key_named"] for f in kept) == sum(f["follow_alias"] for f in kept) == 16
    fails = [x for x in U.balance(kept, 62)[1] if x.split(": ", 1)[1].startswith(("key_named ", "follow_alias "))]
    assert sorted(x.split(": ", 1)[1] for x in fails) == ["follow_alias 16 of 62, not exactly 1/4",
                                                          "key_named 16 of 62, not exactly 1/4"], fails


@functools.lru_cache(maxsize=None)
def big(fam):
    """Two 512-item training blocks of one family, as item facts (seeds 0 and 1 of clean()'s draw)."""
    g = S.Gen(pools(), "train")
    return tuple(U.item_facts(I.render(it, f"{fam}.b{sd}.i{i}"), lex()) for sd in (0, 1)
                 for i, it in enumerate(S.block(g, fam, 512, (5, ord(fam), sd))))


def test_last_rate_floor_and_u_mirror_tolerance():
    """K3 round 15 (REVIEW 16b T-4): two balance tolerances no test read. BD-2's last-statement rule: |rate - chance|
    within max(0.05, 3 SE); on 1,024 R items 3 SE is 0.033, so the 0.05 floor binds: an excess of 0.07 fails and one
    of 0.04 passes (mutant r16_last_rate_tol: floor 0.10, the 0.07 excess passes). U's mirror cells: (key, foil,
    follow) as often as (foil, key, no follow) within 3 SD of their sum; a 4.5 SD imbalance planted on 1,024 U items
    fails (mutant r16_u_mirror_tol: 6 SD, it passes). Planted on the item facts balance() reads."""
    F = [dict(f) for f in big("R")]
    assert U.balance(F, 512)[1] == []
    n, got = len(F), sum(f["is_last"] for f in F)
    exp = sum(f["n_key"] / f["n_st"] for f in F) / n
    assert all(f["n_key"] for f in F) and 3 * math.sqrt(exp * (1 - exp) / n) < 0.04, exp
    for excess, fails in ((0.07, True), (0.04, False)):
        G = [dict(f) for f in F]
        flip = [f for f in G if not f["is_last"]][:round((exp + excess) * n) - got]
        for f in flip:
            f["is_last"] = True
        rate = sum(f["is_last"] for f in G) / n
        assert abs(rate - exp - excess) < 1e-3, (rate, exp)
        last = [x for x in U.balance(G, 512)[1] if x.startswith("R: last-statement rate")]
        assert bool(last) == fails, (excess, rate, exp, last)
    F = [dict(f) for f in big("U")]
    assert U.balance(F, 512)[1] == []
    cnt = Counter(f["u_cell"] for f in F)
    a, b, w = cell = max(cnt, key=lambda c: (cnt[c] + cnt[(c[1], c[0], not c[2])], c))
    mir = (b, a, not w)
    sd = math.sqrt(cnt[cell] + cnt[mir])
    d = math.ceil((4.5 * sd - (cnt[cell] - cnt[mir])) / 2)
    moved = [f for f in F if f["u_cell"] == mir][:d]
    assert len(moved) == d, (cell, cnt[cell], cnt[mir], d)
    for f in moved:
        f["u_cell"] = cell
    z = (cnt[cell] - cnt[mir] + 2 * d) / sd
    assert 4.5 <= z < 4.6, z
    line = f"U: (key {a}, foil {b}, follow {w}) in {cnt[cell] + d} items, its mirror in {cnt[mir] - d}"
    assert line in U.balance(F, 512)[1], (line, z)


def test_an_answer_that_is_not_a_value_token_fails():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17_cand_token_off survived): every non-ack assistant turn ends in
    a value token (V, or VH on OOD) or " none" (SPEC 9: every candidate is one token). An R item's final answer
    replaced by an attribute token fails purity's candidate rule (and the world rule, which the mutant leaves)."""
    def f(recs):
        i = first(recs, "R")
        recs[i]["turns"][recs[i]["qs"][-1]["turn"]]["ids"][-1] = int(pools()["A"][0])
    rep = _plant("train", f)
    assert rep["purity"]["candidate_not_one_token"] == 1 and "purity candidate_not_one_token: 1" in rep["fails"]


def _block_facts(fam):
    return [U.item_facts(r, lex()) for r in clean() if r["fam"] == fam]


def test_a_both_lures_and_u_no_update_shares_are_held_per_block():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutants r17_both_lures_off and r17_no_update_off survived): SPEC 4's A
    both-lures half and U's no-update quarter hold per block. On clean()'s 64-item blocks, read as item facts, both
    pass; four A items moved into the both-lures cell, and four U items moved out of the no-update cell, each fail
    its own rule."""
    A = [dict(f) for f in _block_facts("A")]
    assert len(A) == 64 and U.balance(A, 64)[1] == []
    assert sum(f["has_attr"] and f["has_name"] for f in A) == 32
    for f in [f for f in A if not (f["has_attr"] and f["has_name"])][:4]:
        f["has_attr"] = f["has_name"] = True
    assert any(": both_lures 36 of 64, not exactly 1/2" in x for x in U.balance(A, 64)[1]), U.balance(A, 64)[1]
    Uf = [dict(f) for f in _block_facts("U")]
    assert len(Uf) == 64 and U.balance(Uf, 64)[1] == []
    assert sum(f["u_cell"][:2] == (1, 2) for f in Uf) == 16
    for f in [f for f in Uf if f["u_cell"][:2] == (1, 2)][:4]:
        f["u_cell"] = (3, 2, f["u_cell"][2])
    assert any(": no_update 12 of 64, not exactly 1/4" in x for x in U.balance(Uf, 64)[1]), U.balance(Uf, 64)[1]


def test_ideal_below_one_fails_with_rules_on():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17_ideal_fail_off survived): a family whose IDEAL (the world
    rule read from the item alone) is not 1.0 fails threshold_fails, as it does with rules off (the OOD read)."""
    import oracles as O
    tab = {"U": {**{o: 0.0 for o in O.NAMES}, "IDEAL": 1.0, "n": 512}}
    assert U.threshold_fails(tab, {}) == []
    tab["U"]["IDEAL"] = 0.998
    assert U.threshold_fails(tab, {}) == ["U: IDEAL 0.998"]


def test_pooled_intro_rank_tolerance_is_three_se():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17_rank_tol_double survived): over a family's items of one k, each
    intro rank is asked at 1/k within 3 SE (SE = sqrt(1/k (1 - 1/k) / n)). On big("R")'s 1,024 items, items of the
    largest k moved from rank 0 to rank 1 until rank 1 sits 1.5 tolerances over 1/k: the pooled line fails (within
    6 SE, the mutant, it passes)."""
    F = [dict(f) for f in big("R")]
    assert [x for x in U.balance(F, 512)[1] if "intro rank" in x] == []
    k = max(f["k"] for f in F if f["rank"] >= 0)
    Fk = [f for f in F if f["k"] == k and f["rank"] >= 0]
    n, tol = len(Fk), 3 * math.sqrt((1 / k) * (1 - 1 / k) / len(Fk))
    have = sum(f["rank"] == 1 for f in Fk)
    move = math.ceil((1 / k + 1.5 * tol) * n) - have
    src = [f for f in Fk if f["rank"] == 0][:move]
    assert len(src) == move > 0, (k, n, have, move)
    for f in src:
        f["rank"] = 1
    p = (have + move) / n
    assert 1.5 * tol <= p - 1 / k < 2 * tol, (p, tol)
    assert any(x.startswith(f"R: k {k} intro rank 1 asked at") for x in U.balance(F, 512)[1])


def test_a_cell_of_a_block_that_is_not_full_may_miss_by_less_than_one_item():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17_block_tol_double survived): a block that is not full (an eval
    set's 256 items read against 512) holds each 1/m cell within m on sum x m (less than one item off). clean()'s
    U block read as part of a 512 block passes; one item moved out of the no-update cell (15 of 64: sum x 4 is 4
    off) fails; within 2 m, the mutant, it passes."""
    Uf = [dict(f) for f in _block_facts("U")]
    assert len(Uf) == 64 and [x for x in U.balance(Uf, 512)[1] if "no_update" in x] == []
    f = next(f for f in Uf if f["u_cell"][:2] == (1, 2))
    f["u_cell"] = (3, 2, f["u_cell"][2])
    assert any(": no_update 15 of 64, not exactly 1/4" in x for x in U.balance(Uf, 512)[1]), U.balance(Uf, 512)[1]


def test_an_ood_read_reports_ideal_below_one():
    """K3 round 16 (2026-11-22, REVIEW 17b; mutant r17_ood_ideal_off survived): with rules off (gen.py check's OOD
    read) a family whose IDEAL is not 1.0 is named in the fails, beside the world-rule line (which alone would still
    fail the set: the mutant changes the report, not the verdict)."""
    recs = copy.deepcopy(list(clean("eval")))
    i = first(recs, "U")
    recs[i]["turns"][recs[i]["qs"][-1]["turn"]]["ids"][-1] = int(pools()["V"][7])
    rep = U.accept(recs, pools(), "eval", lex(), None, block=64, rules=False)
    ideal = [x for x in rep["fails"] if x.startswith("U: IDEAL ")]
    assert ideal == [f"U: IDEAL {rep['oracles']['U']['IDEAL']}"] and rep["oracles"]["U"]["IDEAL"] < 1.0, rep["fails"]
    assert "1 items: a question's gold is not the world rule" in rep["fails"]
    assert U.accept(list(clean("eval")), pools(), "eval", lex(), None, block=64, rules=False)["fails"] == []
