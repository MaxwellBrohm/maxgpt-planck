"""SPEC 4 and 5: skill families, the world rule, the token-level IDEAL and the acceptance checks on real blocks."""
import copy
import functools

import numpy as np
import pytest

import items as I
import kcommon as K
import oracles as O
import purity as U
import skill as S
from kfix import lex, old_questions, pools


_FLAGS = S.block_flags          # the unpatched flags (the planting helpers below wrap this one, never a patched one)


@functools.lru_cache(maxsize=None)
def blocks(split: str, n: int = 512, ood=None, groups=False):
    g = S.Gen(pools(), split, ood)
    out = []
    for fam in (S.OOD_FAMS[ood] if ood else "RUBFAP"):
        its = S.block(g, fam, n, (K.TAGS["skill_train"], ord(fam), 0), groups=groups and fam == "B")
        out += [(it, I.render(it, f"{fam}.b0.i{i}")) for i, it in enumerate(its)]
    return out


def test_train_blocks_pass_acceptance():
    recs = [r for _, r in blocks("train")]
    rep = U.accept(recs, pools(), "train", lex(), K.load_tokenizer())
    assert rep["fails"] == [], rep["fails"]
    assert rep["oracles"]["A"]["O9|10"] <= 0.5 + 1e-9 and rep["oracles"]["R"]["O4"] == 1.0


def _twin(it):
    """The build's reversed B twin (statements in reverse order, same question), for the comparison below."""
    turns = [list(reversed(b)) for k, b in reversed(it.ex) if k == "S"]
    return I.Item(it.fam, it.names, [("S", t) for t in turns] + [it.ex[-1]], it.marker, dict(it.meta))


def test_eval_groups_defeat_partial_keys():
    """K3 round 3 (REVIEW 4a S-1): a partial key with a tie-break a reversal keeps (B-ADAPT: the smallest value id)
    got both twins of a pair as often as one item, over the pair bar 0.30. In a cube group (the item and 3 copies
    asking the asked name's two-token neighbours) every partial-key and name-blind rule gives two questions the same
    answer, so it gets no group; IDEAL gets every one."""
    its = blocks("eval", 256, groups=True)
    rep = U.accept([r for _, r in its], pools(), "eval", lex(), block=256)
    assert rep["fails"] == [], rep["fails"]
    assert rep["groups"]["n"] == 48 and rep["groups"]["IDEAL"] == 1.0
    assert max(rep["groups"][o] for o in [*O.O3, *O.BCUE, *O.BPART]) == 0.0, rep["groups"]
    B = [(it, r) for it, r in its if it.fam == "B"]
    assert [it.meta.get("grp", -1) for it, _ in B].count(0) == 48 and sum(it.meta["k"] == 8 for it, _ in B) == 192
    for it, r in B:
        if it.meta.get("grp", -1) >= 0:         # every copy asks a two-token neighbour, one question per item
            st, qs, _ = O.parse(r, lex())
            assert len(qs) == 1 and sum(x == y for x, y in zip(qs[0]["name"], it.names[0])) == (2, 3)[
                it.meta["grp"] == 0]
    pairs = [(U.item_facts(r, lex()), U.item_facts(I.render(_twin(it), r["id"]), lex())) for it, r in B
             if it.meta.get("grp") == 0]
    hit = np.mean([a["ans"]["B-ADAPT"] == a["gold"] and b["ans"]["B-ADAPT"] == b["gold"] for a, b in pairs])
    assert hit > 0.3, hit


@pytest.mark.parametrize("split,ood", [("train", None), ("eval", None), ("evals", None), ("eval", "DFAR"),
                                       ("eval", "KMANY"), ("eval", "ENTS"), ("eval", "VALH"), ("eval", "ATTH"),
                                       ("eval", "PLONG")])
def test_generator_gold_equals_token_ideal(split, ood):
    """The generator's world rule and oracles.parse's IDEAL agree on every question of every item."""
    for it, rec in blocks(split, 64, ood):
        st, qs, _ = O.parse(rec, lex())
        golds = [q["gold"] for q in rec["qs"]]
        ideal = [O.answers(st, q, O.prompt_of(rec, q["turn"]), lex())["IDEAL"] for q in qs]
        assert ideal == golds, rec
        assert I.n_tokens(rec) <= S.MAX_TOKENS
        d = rec["qs"][-1]["d"]
        if rec["fam"] != "A":
            assert (16 <= d <= 24) if ood == "DFAR" else (0 <= d <= S.D_MAX)


def test_splits_use_their_name_pools():
    p = pools()
    kevs = p.kevs_set()
    for split, prefix in (("train", "S"), ("eval", "T"), ("evals", "S")):
        names = {n for it, _ in blocks(split, 64) for n in it.names}
        for i, pos in enumerate(p.triple(prefix)):
            assert {n[i] for n in names} <= set(pos.tolist())
        inter = names & kevs
        assert (not inter) if split == "train" else (split == "eval" or inter == names)


def test_block_flags_are_exact():
    r = np.random.default_rng(0)
    for fam in "RUBFAP":
        fl = S.block_flags(fam, 512, r)
        assert sum(f["qform"] for f in fl) == 256
        if fam in "RUF":
            assert sum(f["follow"] for f in fl) == 256
    u = S.block_flags("U", 512, r)
    assert sum(f["var"] == "noupd" for f in u) == 128 and all(f["follow"] for f in u if f["var"] == "noupd")
    assert sum(f["sub"] == "one" for f in u) == 128 and not any(f["follow"] for f in u if f["sub"] == "one")
    for k in (2, 3, 4):                     # K3: U's (k, intro rank) exact per block; round 4: the foil's rank too,
        ks = [f for f in u if f.get("k") == k]          # every ordered (key, foil) rank pair within one of its share
        assert abs(len(ks) - 512 / 3) < 1 and all(abs(sum(f[x] == i for f in ks) - len(ks) / k) < 1
                                                  for i in range(k) for x in ("rank", "frank"))
        assert all(f["frank"] != f["rank"] for f in ks) and all(
            abs(sum((f["rank"], f["frank"]) == (i, j) for f in ks) - len(ks) / (k * (k - 1))) < 1
            for i in range(k) for j in range(k) if i != j)
    for fam in "RPF":                       # K3 round 4: R's, P's and F's (k, intro rank) exact per block
        fl = S.block_flags(fam, 512, r)
        for k in {f["k"] for f in fl}:
            ks = [f for f in fl if f["k"] == k]
            assert all(abs(sum(f["rank"] == i for f in ks) - len(ks) / k) < 1 for i in range(k)), (fam, k)
        assert {f["k"] for f in fl} == {"R": {1, 2, 3, 4}, "P": {3, 4, 5}, "F": {2, 3, 4}}[fam], fam
    b = S.block_flags("B", 512, r)          # K3 round 2 (BD-22): B's cube in 3/4, (k, intro rank) exact
    assert sum(f["struct"] for f in b) == 384 and all((f["k"] == 8) == f["struct"] for f in b)
    for k in (2, 3, 4, 8):
        ks = [f for f in b if f["k"] == k]
        assert abs(len(ks) - (384 if k == 8 else 128 / 3)) < 1 and all(
            abs(sum(f["rank"] == i for f in ks) - len(ks) / k) < 1 for i in range(k))


def test_reference_family_rules():
    n = 0
    for it, rec in blocks("train", 128):
        if it.fam != "F":
            continue
        q = it.ex[-1][1]
        key = [s for k, b in it.ex if k == "S" for s in b if (s.ent, s.attr) == (q.ent, q.attr) and s.kind != "def"]
        case = it.meta["case"]
        assert key[-1].kind in {"alias": ("alias",), "ell": ("ell",), "full_after": ("full", "reord")}[case]
        if case == "full_after":
            assert any(s.kind in ("alias", "ell") for s in key[:-1])
        n += 1
    assert n == 128


def test_items_deterministic():
    g = S.Gen(pools(), "train")
    a = [I.render(x, "a") for x in S.block(g, "F", 32, (1, 2, 3))]
    b = [I.render(x, "a") for x in S.block(g, "F", 32, (1, 2, 3))]
    assert a == b


def test_no_earlier_question_asks_the_final_attribute():
    """K3 R-4: an earlier answer of the final attribute named the gold (a copy) or excluded a foil's value (EXCL read
    0.686 on F). Only U's lure may ask it, the final key itself with a later statement of that key after it."""
    lures = 0
    for it, rec in blocks("train", 128):
        f = U.item_facts(rec, lex())
        assert not f["early_attr"], rec["id"]
        st, qs, _ = O.parse(rec, lex())
        lures += any(p["attr"] == qs[-1]["attr"] for p in qs[:-1])
        assert rec["fam"] == "U" or not any(p["attr"] == qs[-1]["attr"] for p in qs[:-1])
    assert lures > 10


def test_planted_early_question_fails_acceptance():
    """A copy of the final question with its answer, planted just before it (R-4's copy cue): the check fails."""
    recs = [r for _, r in blocks("train", 128)]
    rec = next(r for r in recs if r["fam"] == "R")
    st, qs, _ = O.parse(rec, lex())
    at = qs[-1]["turn"] - 1
    bad = dict(rec, turns=rec["turns"][:at] + copy.deepcopy(rec["turns"][at:at + 2]) + rec["turns"][at:])
    assert U.item_facts(bad, lex())["early_attr"] and U.item_facts(bad, lex())["all_ideal"]
    fails = U.accept([bad if r is rec else r for r in recs], pools(), "train", lex(), block=128)["fails"]
    assert any("an earlier question asks the final attribute" in x for x in fails), fails


def _key_foil(rec):
    st, qs, intro = O.parse(rec, lex())
    q = qs[-1]
    S_ = st[:q["n"]]
    foil = next(s["name"] for s in S_ if s["attr"] == q["attr"] and s["name"] != q["name"])
    n = [sum(s["name"] == e and s["attr"] == q["attr"] for s in S_) for e in (q["name"], foil)]
    size = [sum(s["name"] == e for s in S_) for e in (q["name"], foil)]
    return n, size, U.item_facts(rec, lex())["follow"]


def test_u_key_and_foil_are_exchangeable():
    """K3: the fitted cue model read 0.87 on U when the key was the more repeated key and its entity the bigger
    one. Now (key count, foil count, follow) and (foil count, key count, not follow) are equally likely: noupd's
    (1, 2, follow) and upd-one's (2, 1, no follow) exactly per block, the rest within 3 SE, entity sizes too."""
    from collections import Counter
    g = S.Gen(pools(), "train")
    rows = [_key_foil(I.render(it, "U.b0.i0")) for b in range(2) for it in S.block(g, "U", 512, (93, 85, b))]
    c = Counter((a, b, f) for (a, b), _, f in rows)
    assert c[(1, 2, True)] == c[(2, 1, False)] == 256 and c[(1, 2, False)] == c[(2, 1, True)] == 0
    for (a, b, f), m in c.items():
        mm = c[(b, a, not f)]
        assert abs(m - mm) <= 3 * np.sqrt(m + mm), ((a, b, f), m, mm)
    d = np.array([s[0] - s[1] for _, s, _ in rows])
    assert abs(d.mean()) < 3 * d.std() / np.sqrt(len(d)), d.mean()


def test_a_both_items_are_r_or_u_contexts():
    """K3: a both-item is an R or U item's context whose question names an entity that does not state the asked
    attribute (the asked name and attribute both occur, the key never)."""
    from collections import Counter
    src = Counter()
    for it, rec in blocks("train", 512):
        if it.fam != "A" or it.meta["lure"] != "both":
            continue
        f = U.item_facts(rec, lex())
        assert f["has_name"] and f["has_attr"] and f["n_key"] == 0 and rec["qs"][-1]["gold"] == K.NONE
        src[it.meta["rctx"][0] == "U"] += 1
    assert src[True] == src[False] == 128, src                     # the context kind a block flag (K3 round 4)


def _old_fam_b(g, fl, r):
    """The build design of B (4 names, the asked one the unique centre), kept here to show the checks catch it."""
    import fams as Fm
    from items import St, intro_rank

    def variant(n0, keep, existing):
        for _ in range(50):
            t = tuple(n0[i] if i in keep else int(g.npos[i][int(r.integers(0, len(g.npos[i])))]) for i in range(3))
            if all(t[i] != n0[i] for i in range(3) if i not in keep) and t not in existing and t not in g.kevs:
                return t
        return None
    fl.pop("k", None)               # the build drew k and the intro rank per item
    fl.pop("rank", None)
    n0 = g.name(r)
    names = [n0]
    if fl["struct"]:
        for keep in ((0, 1), (0, 2), (1, 2)):
            names.append(variant(n0, keep, names))
    else:
        for _ in range(fl.setdefault("k", int(r.integers(2, 5))) - 1):
            names.append(variant(n0, tuple(sorted(r.choice(3, int(r.integers(1, 3)), replace=False).tolist())),
                                 names))
    if any(x is None for x in names):
        return None, None, None
    k = len(names)
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    keys = [(e, astar) for e in range(k)] + [(e, Fm._attrs(g, r, 1, exclude=astar)[0]) for e in range(k)
                                             if r.random() < 0.5]
    units = [[St(Fm._form(r), e, a, v)] for (e, a), v in zip(keys, g.values(r, len(keys)))]
    return Fm.arrange(units, lambda flat: intro_rank(flat, 0) == fl["rank"], r), (0, astar), {"k": k, "names": names}


def test_b_names_are_a_cube_and_the_asked_name_is_any_of_them():
    """K3 round 2 (REVIEW 3a B-CENTER): 3/4 of B items name 8 entities {a, a'} x {b, b'} x {c, c'}, the rest 2-4 of
    one such cube; the asked name is uniform among them, so the rules that read the names' token structure without
    the question (B-CENTER, B-TOKFREQ, B-MAJOR) sit near 1/k (the build design read 0.92)."""
    from collections import Counter
    ks, pos = Counter(), Counter()
    for it, rec in blocks("train"):
        if it.fam != "B":
            continue
        st, qs, intro = O.parse(rec, lex())
        assert all(len({u[i] for u in intro}) <= 2 for i in range(3)), rec["id"]
        ks[len(intro)] += 1
        if len(intro) == 8:
            assert len(set(intro)) == 8 and all(len({u[i] for u in intro}) == 2 for i in range(3))
            pos[sorted(intro).index(qs[-1]["name"])] += 1
    assert ks[8] == 384 and set(ks) <= {2, 3, 4, 8}
    assert all(abs(c - 48) < 3 * np.sqrt(48) for c in pos.values()) and len(pos) == 8, pos
    rep = U.accept([r for _, r in blocks("train")], pools(), "train", lex())
    assert max(rep["oracles"]["B"][o] for o in O.BCUE) < 0.3, rep["oracles"]["B"]
    small = [U.item_facts(r, lex()) for it, r in blocks("train") if it.fam == "B" and it.meta["k"] < 8]
    for o in O.BCUE:                # the 2-4 name quarter too: near E[1/k] (a star around the asked name: 0.8)
        assert np.mean([f["ans"][o] == f["gold"] for f in small]) < 0.5, o


def test_old_b_design_fails_the_name_structure_bars(monkeypatch):
    import fams as Fm
    monkeypatch.setattr(Fm, "fam_B", _old_fam_b)
    g = S.Gen(pools(), "eval")
    recs = [I.render(it, f"B.b0.i{i}") for i, it in enumerate(S.block(g, "B", 256, (93, 66, 0)))]
    rep = U.accept(recs, pools(), "eval", lex(), block=256)
    assert rep["oracles"]["B"]["B-CENTER"] > 0.85, rep["oracles"]["B"]
    for o in O.BCUE:
        assert any(f"B: {o} " in x for x in rep["fails"]), o


def test_p_items_hold_two_to_four_questions():
    """SPEC 4: P holds 2-4 questions (PLONG 5-7) though earlier questions skip the final attribute (BD-15's redraw;
    REVIEW 3b T-1: without it 6% of P draws kept only the final question and no check counted them)."""
    for split, ood, lo, hi in (("train", None, 2, 4), ("eval", None, 2, 4), ("eval", "PLONG", 5, 7)):
        n = [len(rec["qs"]) for it, rec in blocks(split, 128 if ood is None else 64, ood) if it.fam == "P"]
        assert n and min(n) >= lo and max(n) <= hi, (split, ood, min(n), max(n))
    recs = [r for _, r in blocks("train", 128)]
    rec = next(r for r in recs if r["fam"] == "P")
    st, qs, _ = O.parse(rec, lex())
    keep = {q["turn"] - 1 for q in qs[:-1]} | {q["turn"] for q in qs[:-1]}
    one = dict(rec, turns=[t for i, t in enumerate(rec["turns"]) if i not in keep])
    assert U.item_facts(one, lex())["n_q"] == 1 and U.item_facts(one, lex())["all_ideal"]
    fails = U.accept([one if r is rec else r for r in recs], pools(), "train", lex(), block=128)["fails"]
    assert any("P: 1 items with a question count" in x for x in fails), fails


def _u_block(monkeypatch, change, fam="U"):
    def flags(f_, n, r, *a):
        return [change(dict(f)) if f_ == fam else f for f in _FLAGS(f_, n, r, *a)]
    monkeypatch.setattr(S, "block_flags", flags)
    g = S.Gen(pools(), "train")
    return [I.render(it, f"{fam}.b0.i{i}") for i, it in enumerate(S.block(g, fam, 512, (93, ord(fam), 7)))]


def test_u_balance_rules_catch_planted_breaks(monkeypatch):
    """REVIEW 3b D-1: SPEC 4's U rules are checked on every set: the no-update cell (key 1, foil 2, follow) and its
    mirror (2, 1, no follow) exactly 1/4 per block, every (key, foil, follow) cell as often as its mirror (3 SE), and
    U's k and intro rank exact per block. Planted: upd-one items with the foil after the key; (k, rank) per item."""
    assert U.accept(_u_block(monkeypatch, lambda f: f), pools(), "train", lex())["fails"] == []
    fails = U.accept(_u_block(monkeypatch, lambda f: {**f, "follow": True} if f["sub"] == "one" else f), pools(),
                     "train", lex())["fails"]
    assert any(": mirror 0 of 512" in x for x in fails) and any("its mirror in 0" in x for x in fails), fails
    fails = U.accept(_u_block(monkeypatch, lambda f: {k: v for k, v in f.items() if k not in ("k", "rank")}),
                     pools(), "train", lex())["fails"]
    assert any(": k2 " in x or ": k3 " in x or ": k4 " in x for x in fails), fails


def _b_plant(monkeypatch, share, small_ks):
    """One training B block (tag 4472, REVIEW 4b's b_small_plant.py) whose flags are built as skill.block_flags
    builds them, but with the cube share and the small quarter's k values given."""
    def flags(fam, n, r, *a):
        fl = _FLAGS(fam, n, r, *a)
        if fam != "B":
            return fl
        sk = S.nested(n, [(True, share), (False, 1 - share)],
                      lambda c: [(8, 1.)] if c else [(k, 1 / len(small_ks)) for k in small_ks], r)
        rk = [0] * n
        for k in (2, 3, 4, 8):
            idx = [i for i, (_, kk) in enumerate(sk) if kk == k]
            for i, x in zip(idx, S.balanced(len(idx), [(j, 1 / k) for j in range(k)], r)):
                rk[i] = x
        return [{**f, "struct": c, "k": k, "rank": x} for f, (c, k), x in zip(fl, sk, rk)]
    monkeypatch.setattr(S, "block_flags", flags)
    g = S.Gen(pools(), "train")
    return [I.render(it, f"B.b0.i{i}") for i, it in enumerate(S.block(g, "B", 512, (4472, 66, 0)))]


def test_b_k_and_rank_rules_catch_per_item_draws(monkeypatch):
    """BD-22: B's cube share (3/4), k (2-4 in thirds in the small quarter) and intro rank are exact per block, and
    checked; drawn per item they fail the per-block rules. K3 round 3 (REVIEW 4b T-1): the cube in 1/2 of a block
    fails only b_small, the small quarter all k 4 only its k shares (mutants b_small_rule_off, b_small_k_rules_off)."""
    assert U.accept(_u_block(monkeypatch, lambda f: f, "B"), pools(), "train", lex())["fails"] == []
    fails = U.accept(_u_block(monkeypatch, lambda f: {k: v for k, v in f.items() if k not in ("k", "rank")}, "B"),
                     pools(), "train", lex())["fails"]
    assert any(": k8_rank" in x for x in fails), fails
    assert U.accept(_b_plant(monkeypatch, 0.75, (2, 3, 4)), pools(), "train", lex())["fails"] == []
    fails = U.accept(_b_plant(monkeypatch, 0.5, (2, 3, 4)), pools(), "train", lex())["fails"]
    assert any("B.b0: b_small 256 of 512" in x for x in fails), fails
    fails = U.accept(_b_plant(monkeypatch, 0.75, (4,)), pools(), "train", lex())["fails"]
    assert any("B.b0: k2 0 of 128" in x for x in fails) and any("B.b0: k4 128 of 128" in x for x in fails), fails


def test_reported_a_rules_lose_in_the_mix():
    """REVIEW 4a A-mix (reported, A1H's class): A-MANYFREE, A-QNEW and A-QREPEAT beat O9|10's 0.50 on A items, but
    they say " none" on R, U and P items too, so at SPEC 4's shares each loses against O9|10 (the same rule without
    the A cue). They are rows of every check.json, not bars (SPEC 9 splits A's LIK by context). K3 round 4 (REVIEW 5a
    S-1): earlier questions ask bystanders only, so A-QREPEAT fires only on U's lures and reads O9|10 elsewhere."""
    tab = U.accept([r for _, r in blocks("train")], pools(), "train", lex())["oracles"]
    for o in ("A1H", "A-MANYFREE", "A-QNEW", "A-QREPEAT"):
        net = sum(S.FAM_SHARE[f] * (tab[f][o] - tab[f]["O9|10"]) for f in "RUBFAP")
        assert (tab["A"][o] > 0.52 or o == "A-QREPEAT") and min(tab[f][o] for f in "RUP") < 1.0 and net < 0, (
            o, net, tab["A"][o])
        assert o not in O.TH
    assert all(tab[f]["A-QREPEAT"] == tab[f]["O9|10"] for f in "RBFAP") and tab["U"]["A-QREPEAT"] < 1.0


def test_group_bars_catch_order_free_partial_keys():
    """REVIEW 4a S-1 in the group check: were a group's copies the item's reversed twin (the build's pairs, as 4),
    B-ADAPT's smallest-value-id tie-break would get the whole group as often as one item while the latest-match O3_S
    get none. The group bars must name B-ADAPT, and not O3_S (mutant b_group_bars_o3_only)."""
    recs = []
    for it, _ in blocks("eval", 256, groups=True):
        if it.fam == "B" and it.meta.get("grp") == 0:
            for j, x in enumerate((it, _twin(it), it, _twin(it))):
                recs.append(I.render(I.Item(x.fam, x.names, x.ex, x.marker, {**x.meta, "grp": j}),
                                     f"B.b0.i{len(recs)}"))
    rep = U.accept(recs, pools(), "eval", lex(), block=256)
    assert rep["groups"]["n"] == 48 and rep["groups"]["B-ADAPT"] > 0.3 and max(rep["groups"][o] for o in O.O3) == 0
    assert any("B groups: B-ADAPT " in x for x in rep["fails"]), rep["fails"]
    assert not any("B groups: O3_" in x for x in rep["fails"]), rep["fails"]


_old_questions = old_questions
_NEW_QUESTIONS = S._questions


def _no_reserve(turns, final, n_extra, lure_key, r, skip=()):
    """Bystander questions without R's and U's reserved bystander (the first round-4 draw: the gate read A 0.536)."""
    return _NEW_QUESTIONS(turns, final, n_extra, lure_key, r)


def test_earlier_questions_ask_bystanders_only(monkeypatch):
    """K3 round 4 (REVIEW 5a S-1): earlier questions name neither the final entity nor any holder of the final
    attribute (U's stale lure aside), so whether the final name was asked earlier says nothing about its size; R and
    U keep one bystander unasked, as an A both-item's asked entity is; P has a bystander entity of its own and a
    foil in every item (3-5 entities). From tokens on every family, the OOD columns included; the pre-round-4 draw,
    a draw asking holders and one asking every bystander each fail the check."""
    lures, n_prev, pk = 0, {f: 0 for f in "RUBFAP"}, set()
    for split, ood in (("train", None), ("eval", None), ("eval", "DFAR"), ("eval", "PLONG"), ("eval", "ENTS")):
        for it, rec in blocks(split, 128 if ood is None else 64, ood):
            f = U.item_facts(rec, lex())
            assert not f["early_name"] and not f["all_free_asked"], (split, ood, rec["id"])
            st, qs, intro = O.parse(rec, lex())
            q = qs[-1]
            asked = [p for p in qs[:-1] if p["name"] == q["name"]]
            assert all(p["attr"] == q["attr"] and rec["fam"] == "U" for p in asked), rec["id"]
            lures += len(asked)
            n_prev[rec["fam"]] += (len(qs) - 1) * (split == "train" and ood is None)
            if rec["fam"] == "P":
                pk.add(f["k"])
                assert any(not any(s["name"] == u and s["attr"] == q["attr"] for s in st) for u in intro), rec["id"]
    assert lures > 5 and n_prev["R"] > 10 and n_prev["P"] >= 128 and n_prev["B"] == 0, (lures, n_prev)
    assert pk == {3, 4, 5}, pk
    for draw, fams, want in ((_old_questions, "RP", "names the final name or a holder"),
                             (_foil_questions, "R", "names the final name or a holder"),
                             (_no_reserve, "RU", "R or U items: every bystander asked")):
        monkeypatch.setattr(S, "_questions", draw)       # (_foil_questions: P's nofoil items have no foil to ask)
        g = S.Gen(pools(), "train")
        recs = [I.render(it, f"{f}.b0.i{i}") for f in fams for i, it in enumerate(S.block(g, f, 128, (93, ord(f), 3)))]
        fails = U.accept(recs, pools(), "train", lex(), block=128)["fails"]
        assert any(want in x for x in fails), (draw, fails)


def _foil_questions(turns, final, n_extra, lure_key, r, skip=()):
    """A planted draw that asks only holders of the final attribute other than the final entity (their other keys),
    so the check must read holders, not only the final name."""
    hold = {s.ent for t in turns for s in t if s.attr == final.attr and s.kind != "def"} - {final.ent}
    ex, stated = [], []
    for t in turns:
        ex.append(("S", t))
        stated += [(s.ent, s.attr) for s in t if s.kind != "def" and (s.ent, s.attr) not in stated]
    pool = [k for k in stated if k[0] in hold and k[1] != final.attr]
    if pool and n_extra:
        key = pool[int(r.integers(0, len(pool)))]
        ex.append(("Q", I.Qn(key[0], key[1], int(r.integers(0, 2)))))
    return ex + [("Q", final)]


def test_u_foil_rank_is_a_block_flag(monkeypatch):
    """K3 round 4 (REVIEW 5a S-2): U's foil kept a big entity's early intro while the key's rank was exact, so the
    holder introduced last was the key in 0.541 of non-lure items. The (key, foil) rank pair is now a block flag:
    from tokens the foil's rank per k is exact per block, and the holder introduced last is the key half the time.
    With the foil's rank left free (the round-3 draw) the per-block foil-rank rule fails."""
    g = S.Gen(pools(), "train")
    recs = [I.render(it, f"U.b{b}.i{i}") for b in range(2) for i, it in enumerate(S.block(g, "U", 512, (93, 85, b)))]
    assert U.accept(recs, pools(), "train", lex())["fails"] == []
    late = []
    for rec in recs:
        st, qs, intro = O.parse(rec, lex())
        q = qs[-1]
        hold = [u for u in intro if any(s["name"] == u and s["attr"] == q["attr"] for s in st[:q["n"]])]
        late.append(hold[-1] == q["name"])
    assert len(late) == 1024 and abs(np.mean(late) - 0.5) < 3 * 0.5 / np.sqrt(1024), np.mean(late)
    fails = U.accept(_u_block(monkeypatch, lambda f: {k: v for k, v in f.items() if k != "frank"}), pools(),
                     "train", lex())["fails"]
    assert any("_frank" in x for x in fails) and not any("_rank" in x for x in fails), fails


def test_b_groups_are_counted():
    """K3 round 4 (REVIEW 5b O-1): an eval B set whose records leave skill.group's order scores no group; the check
    now fails it (every cube item must open a scored group) instead of skipping every group bar."""
    recs = [r for _, r in blocks("eval", 256, groups=True) if r["fam"] == "B"]
    assert U.accept(recs, pools(), "eval", lex(), block=256)["fails"] == []
    fails = U.accept(recs[::-1], pools(), "eval", lex(), block=256)["fails"]
    assert "B groups: 0 scored of 48 cube items" in fails, fails


def test_r_f_p_rank_rules_catch_per_item_draws(monkeypatch):
    """K3 round 4: R's, F's and P's intro rank is exact per block within each k (drawn per item, one K-EVAL-sized R
    block read k 3 rank 1 at 0.506, 3.3 SE, and failed SPEC 5's set-level rank rule). Drawn per item again, a block
    fails the per-block rule."""
    for fam in "RF":
        assert U.accept(_u_block(monkeypatch, lambda f: f, fam), pools(), "train", lex())["fails"] == [], fam
        fails = U.accept(_u_block(monkeypatch, lambda f: {k: v for k, v in f.items() if k not in ("k", "rank")},
                                  fam), pools(), "train", lex())["fails"]
        assert any(f"{fam}.b0: k" in x and "_rank" in x for x in fails), (fam, fails)


def test_p_holders_keep_r_design():
    """K3 round 4 (BD-30): P's bystander is an entity of its own (fams.fam_R by), so its other entities keep R's
    design: besides the key and the foil each states the asked attribute with probability 1/3 (1-3 of 6 attributes).
    Without it (mutant p_bystander_off: R's rule for every other entity, the item redrawn until one is a bystander)
    the k 5 items with 3 or more holders read 0.741 against 5/9 (k 4 here: 0.357 against 1/3)."""
    import math
    for k, share in ((4, 1 / 3), (5, 5 / 9)):          # k 5: 2 others, at least one holder: 1 - (2/3)^2
        hs = []
        for it, rec in blocks("train"):
            if rec["fam"] != "P":
                continue
            st, qs, intro = O.parse(rec, lex())
            if len(intro) == k:
                hs.append(len({s["name"] for s in st[:qs[-1]["n"]] if s["attr"] == qs[-1]["attr"]}) > 2)
        p = sum(hs) / len(hs)
        assert len(hs) > 150 and abs(p - share) < 3 * math.sqrt(share * (1 - share) / len(hs)), (k, p, len(hs))
