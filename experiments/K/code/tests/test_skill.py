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
        if fam == "F":
            assert sum(f["follow"] for f in fl) == 256
        assert fam == "F" or not any("follow" in f or "foil" in f for f in fl), fam   # K3 round 6: natural
    u = S.block_flags("U", 512, r)
    assert sum(f["var"] == "noupd" for f in u) == 128 and sum(f["sub"] == "one" for f in u) == 128
    a = [f["ctx"] for f in S.block_flags("A", 512, r) if f["lure"] == "both"]   # K3 round 7: R 3/8, U 5/8
    assert a.count("R") == 96 and a.count("U") == 160
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
        assert {f["k"] for f in fl} == {"R": {2, 3, 4}, "P": {3, 4, 5}, "F": {2, 3, 4}}[fam], fam
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
    (1, 2) and upd-one's (2, 1) exactly per block, every cell within 3 SE of its mirror, entity sizes too. K3 round 6:
    recency is natural (a (1, 2) pair's single statement is the later one in about 1/3 of items)."""
    from collections import Counter
    g = S.Gen(pools(), "train")
    rows = [_key_foil(I.render(it, "U.b0.i0")) for b in range(2) for it in S.block(g, "U", 512, (93, 85, b))]
    c = Counter((a, b, f) for (a, b), _, f in rows)
    assert c[(1, 2, True)] + c[(1, 2, False)] == c[(2, 1, False)] + c[(2, 1, True)] == 256, c
    assert 0 < c[(1, 2, False)] < c[(1, 2, True)], c
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
    assert src[True] == 160 and src[False] == 96, src      # the context kind a block flag (K3 round 4; round 7: 5/8 U)


def test_a_both_asked_entity_is_uniform_among_names_lacking_an_attribute(monkeypatch):
    """K3 round 7 (REVIEW 8a S-1): fams.fam_A draws a U context's both-item's asked entity uniformly among the context's
    k entities and redraws the context, context and all, while that entity lacks nothing (the context law itself:
    test_a_u_context_redraw_law); within a kept context it is so uniform among the entities that lack an attribute
    the item states (an R context's keeps round 6's draw, its asked attribute the context's; there every such entity
    has 2 keys), and on 8 A blocks (1,024 both-items) the asked one's place among them, by
    key count then intro rank, is uniform in every cell of their key counts (purity A.exch_side). Round 6's draw in U
    contexts too (a non-holder of the context's asked attribute: mostly a name with few keys; A-ONEKEY3 read A
    0.56-0.57 with a mix gain) and the size law (the context's own asked entity, an attribute it lacks: a tie with a
    pick among 3 keys in the attribute-blind model) fail it. Every both-item's asked entity states something and
    lacks the asked attribute, which another entity states; the context's attribute is a key of its entity."""
    import fams as Fm
    import oldfams
    rep = U.accept(_fam_blocks("A", 8, 5307), pools(), "train", lex())
    assert rep["fails"] == [] and rep["balance"]["A.exch_side"][1] >= 10, (rep["fails"], rep["balance"])
    g = S.Gen(pools(), "train")
    for it in S.block(g, "A", 512, (5307, ord("A"), 0)):
        if it.meta["lure"] == "both":
            q = it.ex[-1][1]
            sts = [s for k, b in it.ex if k == "S" for s in b]
            assert any(s.ent == q.ent for s in sts) and not any((s.ent, s.attr) == (q.ent, q.attr) for s in sts)
            assert any(s.attr == q.attr for s in sts) and any((s.ent, s.attr) == (0, it.meta["cattr"]) for s in sts)
            assert it.meta["rctx"] != "R" or q.attr == it.meta["cattr"], it.meta      # R: round 6's draw
    for name, fa in (("round 6", oldfams.fam_A_r6), ("size law", oldfams.fam_A_size)):
        monkeypatch.setattr(Fm, "fam_A", fa)
        fails = U.accept(_fam_blocks("A", 8, 5307), pools(), "train", lex())["fails"]
        assert any(x.startswith("A.exch_side") for x in fails), (name, fails)


def test_a_u_context_redraw_law(monkeypatch):
    """K3 round 8 (REVIEW 9b T-3): in a U context fams.fam_A draws the asked entity uniformly among the context's k
    entities and returns None when it states every attribute the item states (skill.block then redraws the item,
    context and all), so a context with m names lacking an attribute is kept with probability m / k: every both-item
    is weighted by its context's share of lacking names, as a U item's asked entity is drawn by its keys. Per cell
    (k, m) of 6,000 draws (fam_U spied), the kept count is n m / k within 4 SD (exactly 0 or n at m 0 or k). The
    mutant a_u_lacking_only (uniform among the lacking names, the context kept: every context with m >= 1 kept)
    moved the U-context both-items' k 3 share 0.467 -> 0.510 and A-ONEKEY3's one-key cell 0.112 -> 0.133, its mix
    change -0.0033 -> -0.0020 (REVIEW 9b), and passed every other test."""
    import math
    from collections import Counter
    import fams as Fm
    seen, orig = [], Fm.fam_U

    def spy(g, fl, r, kmany=False):
        seen.append(orig(g, fl, r, kmany))
        return seen[-1]
    monkeypatch.setattr(Fm, "fam_U", spy)
    g, r = S.Gen(pools(), "train"), np.random.default_rng([K.W, 5309])
    draws, kept = Counter(), Counter()
    for _ in range(6000):
        seen.clear()
        order, key, meta = Fm.fam_A(g, {"lure": "both", "ctx": "U"}, r)
        (ctx, _, cm), = seen
        ats = {s.attr for u in ctx for s in u if s.kind != "def"}
        m = sum(bool(ats - {s.attr for u in ctx for s in u if s.ent == e}) for e in range(cm["k"]))
        draws[(cm["k"], m)] += 1
        kept[(cm["k"], m)] += order is not None
        assert order is None or (order is ctx and key[1] in ats - {s.attr for u in ctx for s in u if s.ent == key[0]})
    assert min(draws[(3, 2)], draws[(4, 3)]) > 300 and set(draws) <= {(2, 0), (3, 2), (3, 3), (4, 3), (4, 4)}, draws
    for (k, m), n in draws.items():
        p = m / k
        assert abs(kept[(k, m)] - n * p) <= 4 * math.sqrt(n * p * (1 - p)), ((k, m), n, kept[(k, m)])


def test_a_cntjoin_is_a_two_sided_reader():
    """K3 round 8 (REVIEW 9a O-1, ruled skill in SPEC 5): A-CNTJOIN says " none" when O9|10's cues fire or no key of the
    asked name (T's join) has the asked attribute's statement count in the item, else O4. Hand-built: the asked name
    states a1 (2 statements) and a2 (1); the asked attribute a0 stated 2 times by others: no fire (O4: no value); 3
    times: " none". On blocks it never fires on R, U, B or P (the asked key is a key of the asked name), reads A over
    0.6 and gains in the mix, so it is reported (SPEC 9 splits A's LIK by its cell), not barred (mutant
    a_cntjoin_any_key)."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    n = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(3)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = [int(x) for x in p["V"][:8]]
    St, Qn = I.St, I.Qn
    for extra, want in ((False, -1), (True, K.NONE)):
        ex = [("S", [St("full", 0, a1, v[0]), St("full", 1, a1, v[1])]), ("S", [St("reord", 0, a2, v[2])]),
              ("S", [St("full", 1, a0, v[3]), St("full", 2, a0, v[4])])] + (
            [("S", [St("reord", 2, a0, v[5])])] if extra else []) + [("Q", Qn(0, a0, 0))]
        rec = I.render(I.Item("A", n, ex), "A.b0.i0")
        st, qs, _ = O.parse(rec, lex())
        assert O.answers(st, qs[-1], O.prompt_of(rec, qs[-1]["turn"]), lex())["A-CNTJOIN"] == want, extra
    tab = U.accept([r for _, r in blocks("train")], pools(), "train", lex())["oracles"]
    assert all(tab[f]["A-CNTJOIN"] == tab[f]["O9|10"] for f in "RUBP") and tab["A"]["A-CNTJOIN"] > 0.6, tab["A"]
    net = sum(S.FAM_SHARE[f] * (tab[f]["A-CNTJOIN"] - tab[f]["O9|10"]) for f in "RUBFAP")
    assert net > 0 and "A-CNTJOIN" in O.REPORTED and "A-CNTJOIN" not in O.TH, net


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
    """REVIEW 3b D-1: SPEC 4's U rules are checked on every set: the no-update cell (key 1, foil 2) and its mirror
    (2, 1) exactly 1/4 per block, every (key, foil, follow) cell as often as its mirror (3 SE), and U's k and intro
    rank exact per block. Planted: upd-one items drawn as "sym" (no (2, 1) cell); (k, rank) per item."""
    assert U.accept(_u_block(monkeypatch, lambda f: f), pools(), "train", lex())["fails"] == []
    fails = U.accept(_u_block(monkeypatch, lambda f: {**f, "sub": "sym"} if f["sub"] == "one" else f), pools(),
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
    for o in ("A1H", "A-MANYFREE", "A-QNEW", "A-QREPEAT"):     # K3 round 6: A's both-items (R and U contexts) ask
        net = sum(S.FAM_SHARE[f] * (tab[f][o] - tab[f]["O9|10"]) for f in "RUBFAP")    # no earlier question, so
        assert tab["A"][o] >= 0.5 and min(tab[f][o] for f in "RUP") < 1.0 and net < 0, (    # A-QNEW reads 0.50 on A
            o, net, tab["A"][o])
        assert o not in O.TH
    assert tab["A"]["A-MANYFREE"] > 0.6, tab["A"]
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


def _no_reserve(turns, final, n_extra, lure_key, r, skip=(), comp=False, place=False):
    """Bystander questions without R's and U's reserved bystander (the first round-4 draw: the gate read A 0.536)."""
    return _NEW_QUESTIONS(turns, final, n_extra, lure_key, r)


def _no_comp(turns, final, n_extra, lure_key, r, skip=(), comp=False, place=False):
    """Round 5's bystander questions (no component rule): a bystander's key may be one of the asked entity's
    attributes (REVIEW 7a's "+eq": P 0.83-0.87 for an attribute-blind fit)."""
    return _NEW_QUESTIONS(turns, final, n_extra, lure_key, r, skip)


def test_earlier_questions_ask_bystanders_only(monkeypatch):
    """K3 round 4 (REVIEW 5a S-1): earlier questions name neither the final entity nor any holder of the final
    attribute (U's stale lure aside), so whether the final name was asked earlier says nothing about its size; R and
    U keep one bystander unasked, as an A both-item's asked entity is; P has a bystander entity of its own and a
    foil in every item (3-5 entities). From tokens on every family, the OOD columns included; the pre-round-4 draw,
    a draw asking holders and one asking every bystander each fail the check. K3 round 6 (REVIEW 7a S-1, "+eq"): in
    R, U, P and A's both-items no earlier question names an entity of the final key's component (R's items are one
    component, so they hold one question; P's bystander is a component of its own), and round 5's rule fails."""
    lures, n_prev, pk = 0, {f: 0 for f in "RUBFAP"}, set()
    for split, ood in (("train", None), ("eval", None), ("eval", "DFAR"), ("eval", "PLONG"), ("eval", "ENTS")):
        for it, rec in blocks(split, 128 if ood is None else 64, ood):
            f = U.item_facts(rec, lex())
            assert not f["early_name"] and not f["all_free_asked"] and not f["early_comp"], (split, ood, rec["id"])
            st, qs, intro = O.parse(rec, lex())
            q = qs[-1]
            asked = [p for p in qs[:-1] if p["name"] == q["name"]]
            assert all(p["attr"] == q["attr"] and rec["fam"] == "U" for p in asked), rec["id"]
            lures += len(asked)
            n_prev[rec["fam"]] += (len(qs) - 1) * (split == "train" and ood is None)
            if rec["fam"] == "P":
                pk.add(f["k"])
                assert any(not any(s["name"] == u and s["attr"] == q["attr"] for s in st) for u in intro), rec["id"]
    assert lures > 5 and n_prev["R"] == 0 and n_prev["P"] >= 128 and n_prev["B"] == 0, (lures, n_prev)
    assert pk == {3, 4, 5}, pk
    for draw, fams, want in ((_old_questions, "RP", "names the final name or a holder"),
                             (_foil_questions, "R", "names the final name or a holder"),
                             (_no_reserve, "RU", "R or U items: every bystander asked"),
                             (_no_comp, "RUP", "an entity of the final key's component")):
        monkeypatch.setattr(S, "_questions", draw)       # (_foil_questions: P's nofoil items have no foil to ask)
        g = S.Gen(pools(), "train")
        recs = [I.render(it, f"{f}.b0.i{i}") for f in fams for i, it in enumerate(S.block(g, f, 128, (93, ord(f), 3)))]
        fails = U.accept(recs, pools(), "train", lex(), block=128)["fails"]
        assert any(want in x for x in fails), (draw, fails)
        for fam in fams if draw in (_no_reserve, _no_comp) else "":    # K3 rounds 11, 12 (REVIEW 12b T-7, 13b T-5):
            fails = U.accept([x for x in recs if x["fam"] == fam], pools(), "train", lex(), block=128)["fails"]
            assert any(want in x for x in fails), (fam, fails)      # per family; pooled, R and U failed the component
            #  rule without P, R the reserve rule without U (mutants m12_early_comp_not_p, m13c_all_free_r_only)


def _foil_questions(turns, final, n_extra, lure_key, r, skip=(), comp=False, place=False):
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
    fails the per-block rule. K3 round 5 (re-run of round 4's tests lens): P was named but not drawn here, so P's
    per-block rule was never watched failing (mutant p_rank_rule_dropped survived the round-4 tests)."""
    for fam in "RFP":
        assert U.accept(_u_block(monkeypatch, lambda f: f, fam), pools(), "train", lex())["fails"] == [], fam
        fails = U.accept(_u_block(monkeypatch, lambda f: {k: v for k, v in f.items() if k not in ("k", "rank")},
                                  fam), pools(), "train", lex())["fails"]
        assert any(f"{fam}.b0: k" in x and "_rank" in x for x in fails), (fam, fails)


def _components(rec):
    """From tokens: the item's components (names joined by a shared attribute), each name's attributes, questions."""
    st, qs, intro = O.parse(rec, lex())
    keys = {u: {s["attr"] for s in st[:qs[-1]["n"]] if s["name"] == u} for u in intro}
    comps = []
    for u in intro:
        hit = [c for c in comps if any(keys[u] & keys[v] for v in c)]
        comps = [c for c in comps if c not in hit] + [set().union({u}, *hit)]
    return comps, keys, st, qs


def test_dfar_pads_add_no_holder():
    """K3 round 4 (skill._pad): a DFAR pad states the asked attribute only for an entity that already does (a bystander
    made a holder could be one an earlier question asked). K3 round 6: R items are one component with no earlier
    question and U's are mostly lures, so the sampled DFAR items never showed a breach and the mutant
    dfar_pad_makes_holders survived the full run; this pins the rule on _pad itself (200 draws of 13 pads each).
    G-2: N3, a bystander no question asks, states a3 first; since G-2's component redraw catches a pad (N2, a0) (N2
    is asked), only the holder rule keeps N3 from holding a0, and the G-2 full mutant run showed the fixture without
    N3 let dfar_pad_makes_holders survive."""
    p = pools()
    a0, a1, a2, a3 = (int(x) for x in p["A"][:4])
    v = [int(x) for x in p["V"][:5]]
    St, Qn = I.St, I.Qn
    ex = [("S", [St("full", 3, a3, v[4])]), ("S", [St("full", 0, a0, v[0])]), ("S", [St("full", 1, a0, v[1])]),
          ("S", [St("full", 2, a1, v[2])]), ("Q", Qn(2, a1, 0)), ("S", [St("full", 0, a2, v[3])])]
    orig = {id(s) for k, b in ex if k == "S" for s in b}
    g, n, bad = S.Gen(p, "eval", ood="DFAR"), 0, 0
    for seed in range(200):
        out = S._pad(ex, (0, a0), 4, g, np.random.default_rng([K.W, 9100, seed]), 16)
        pads = [s for k, b in out if k == "S" for s in b if id(s) not in orig]
        n += len(pads)
        bad += sum(s.attr == a0 and s.ent not in (0, 1) for s in pads)
    assert n == 200 * 13 and bad == 0, (n, bad)



def test_dfar_pads_keep_the_component_rule(monkeypatch):
    """G-2 (2026-10-11, REVIEW 12b T-4): a DFAR pad never joins an earlier question's entity to the final key's
    component (SPEC 4 [K3r6], every item). By hand: N0 and N1 hold a0 (the final key N0 a0), N0 states a2, N2 states
    a1 and an earlier question asks N2 a1; a pad (N2, a2), (N0, a1) or (N1, a1) joins N2. 200 draws of 12 pads, each
    rendered and read by purity.item_facts: no early_comp. The pre-G-2 pad (oldfams.pad_r13) joins on the same draws
    (the fixture tries), and skill._pad redraws a joining pad (its redraw draws one more integer)."""
    import oldfams
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(3)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = [int(x) for x in p["V"][:4]]
    St, Qn = I.St, I.Qn
    ex = [("S", [St("full", 0, a0, v[0])]), ("S", [St("full", 1, a0, v[1])]), ("S", [St("full", 2, a1, v[2])]),
          ("Q", Qn(2, a1, 0)), ("S", [St("full", 0, a2, v[3])]), ("Q", Qn(0, a0, 0))]
    g, got = S.Gen(p, "eval", ood="DFAR"), {}
    for name, pad in (("new", S._pad), ("old", oldfams.pad_r13)):
        bad = 0
        for seed in range(200):
            out = pad(ex, (0, a0), 3, g, np.random.default_rng([K.W, 9101, seed]), 16)
            assert sum(k == "S" for k, _ in out) == 4 + 12, name
            bad += U.item_facts(I.render(I.Item("U", nm, out), f"U.b0.i{seed}"), lex())["early_comp"]
        got[name] = bad
    assert got["new"] == 0 and got["old"] > 100, got


def test_dfar_pad_redraw_is_uniform_over_the_pads_that_join_nothing():
    """K3 round 14 (REVIEW 15b T-8): G-2's redraw of a joining pad is uniform among the pads that join nothing
    (SPEC 16 G-2, skill._pad's docstring). test_dfar_pads_keep_the_component_rule's item (names_k 3, 6 attributes,
    the final key N0 a0), one pad to add, an rng whose first draw is always (N2, a2), a joining pad: the redraw must
    draw one integer in [0, 13), 13 the pads by hand (18 less the key, N2 a0, which would make N2 a holder, and the
    three joining pads N2 a2, N0 a1, N1 a1), and its 13 values give the 13 pads, each once (mutant
    r15_pad_redraw_first: the first such pad every time, no draw)."""
    p = pools()
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = [int(x) for x in p["V"][:4]]
    St, Qn = I.St, I.Qn
    ex = [("S", [St("full", 0, a0, v[0])]), ("S", [St("full", 1, a0, v[1])]), ("S", [St("full", 2, a1, v[2])]),
          ("Q", Qn(2, a1, 0)), ("S", [St("full", 0, a2, v[3])]), ("Q", Qn(0, a0, 0))]
    g = S.Gen(p, "eval", ood="DFAR")
    attrs = [int(x) for x in g.attrs]
    assert len(attrs) == 6 and {a0, a1, a2} <= set(attrs)

    class Rng:          # the first pad draw forced to (N2, a2); the redraw's index given; everything else real
        def __init__(self, idx):
            self.real, self.idx, self.redraws = np.random.default_rng([K.W, 9102]), idx, []

        def integers(self, lo, hi=None):
            if hi == 3:
                return 2
            self.redraws.append((lo, hi))
            return self.idx

        def choice(self, x, *a, **kw):
            return a2 if x is g.attrs else self.real.choice(x, *a, **kw)

        def random(self, *a):
            return self.real.random(*a)
    want = {(e, a) for e in range(3) for a in attrs} - {(0, a0), (2, a0), (2, a2), (0, a1), (1, a1)}
    got = []
    for idx in range(13):
        r = Rng(idx)
        out = S._pad(ex, (0, a0), 3, g, r, 5)
        new = [s for k, b in out if k == "S" for s in b if s not in [x for k2, b2 in ex if k2 == "S" for x in b2]]
        assert r.redraws == [(0, 13)] and len(new) == 1, (idx, r.redraws, new)
        got.append((new[0].ent, new[0].attr))
    assert len(want) == 13 and sorted(got) == sorted(want), (got, want)


def test_dfar_ood_column_keeps_the_component_rule(monkeypatch):
    """G-2 (REVIEW 12b T-4): gen.py's own OOD-DFAR U column (write_evals' call, 128 items) holds no item whose earlier
    question names an entity of the final key's component, and accept with rules off (gen.py check's OOD read) checks
    SPEC 4's per-item rules there. With the pre-G-2 pad the same call gives the records of K3 rounds 8 to 13, whose
    U.b0.i87 breaks the rule and fails the OOD read (mutants g2_pad_redraw_off, g2_ood_item_rules_off)."""
    import evalsets as E
    import oldfams

    def column():
        return E.skill_set(pools(), lex(), "eval", K.SEEDS["ood"], 256 // len(S.OOD_FAMS["DFAR"]), fams="U",
                           ood="DFAR", groups=False, tag="OOD-DFAR")
    new = column()
    assert len(new) == 128 and [r["id"] for r in new if U.item_facts(r, lex())["early_comp"]] == []
    assert U.accept(new, pools(), "eval", lex(), rules=False)["fails"] == []
    monkeypatch.setattr(S, "_pad", oldfams.pad_r13)
    old = column()
    assert [r["id"] for r in old if U.item_facts(r, lex())["early_comp"]] == ["U.b0.i87"]
    assert U.accept(old, pools(), "eval", lex(), rules=False)["fails"] == [
        "1 items: an earlier question names an entity of the final key's component"]

def test_p_bystander_intro_rank_is_uniform():
    """K3 round 6 final: P's earlier questions ask only its bystander, at turn ends where one of its keys is stated
    (skill._questions place), so no P item is redrawn for its question count. Drawn at every turn end, a late
    bystander left fewer than 2 questions and the redraw kept early bystanders (P.exch_intro 290.5 on 27 df on 40,960
    fresh items, limit 63.2; at k 4 the bystander last in 0.184-0.215 against 0.25; mutant p_place_off). From tokens
    on 8 P blocks: the bystander's intro rank is uniform per k (chi-square under purity.wh), 2-4 questions each."""
    from collections import Counter, defaultdict
    cells = defaultdict(Counter)
    for rec in _fam_blocks("P", 8, 5307):
        st, qs, intro = O.parse(rec, lex())
        by = {p["name"] for p in qs[:-1]}
        assert len(by) == 1 and 2 <= len(qs) <= 4, rec["id"]
        cells[len(intro)][intro.index(by.pop())] += 1
    stat = sum((c[j] - sum(c.values()) / k) ** 2 / (sum(c.values()) / k) for k, c in cells.items() for j in range(k))
    df = sum(k - 1 for k in cells)
    assert set(cells) == {3, 4, 5} and stat < U.wh(df), (stat, df, {k: sorted(c.items()) for k, c in cells.items()})


def test_r_and_p_keep_one_law():
    """K3 round 6 (REVIEW 7a S-1; BD-38): fams.fam_R builds every key by one law: R's entities (P's main component)
    each state 2-3 of 3 attributes, each key once, so an R item is one component; P's bystander states 1-3 attributes
    no other entity states and is the only name its earlier questions ask. From tokens on a training block each."""
    from collections import Counter
    n = Counter()
    for it, rec in blocks("train"):
        if rec["fam"] not in "RP":
            continue
        comps, keys, st, qs = _components(rec)
        main = next(c for c in comps if qs[-1]["name"] in c)
        assert len(set().union(*(keys[u] for u in main))) <= 3 and all(2 <= len(keys[u]) <= 3 for u in main), rec["id"]
        assert set(Counter((s["name"], s["attr"]) for s in st).values()) == {1}, rec["id"]
        if rec["fam"] == "R":
            assert comps == [main], rec["id"]
        else:
            by = [c for c in comps if c is not main]
            assert len(by) == 1 and len(by[0]) == 1 and 1 <= len(keys[next(iter(by[0]))]) <= 3, rec["id"]
            assert {p["name"] for p in qs[:-1]} == by[0], rec["id"]
        n[rec["fam"]] += 1
    assert n == {"R": 512, "P": 512}, n


def _r4_fam_r(g, fl, r, k=None, by=False):
    """fams.fam_R before K3 round 5 (REVIEW 6a S-1): a uniform order kept only if the key has the block's intro rank
    and follow status (frac(key) conditioned jointly), the foil one slot, other entities holders by R's random draw."""
    import fams as Fm
    if k is not None and "k_ood" not in fl:
        fl.update(k_ood=k + by, k=k + by, rank=int(r.integers(0, k + by)))
    k = fl.setdefault("k", int(r.integers(2 if fl["foil"] else 1, 5)) + by)
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    keys = [(0, a) for a in Fm._attrs(g, r, int(r.integers(1, 4)), must=astar)]
    foil_slot = int(r.integers(1, k - by)) if fl["foil"] else -1
    for e in range(1, k):
        keys += [(e, a) for a in Fm._attrs(g, r, int(r.integers(1, 4)), must=astar if e == foil_slot else None,
                                           exclude=None if fl["foil"] and not (by and e == k - 1) else astar)]
    units = [[I.St(Fm._form(r), e, a, v)] for (e, a), v in zip(keys, g.values(r, len(keys)))]
    return Fm.arrange(units, Fm._rank_follow(fl, astar), r), (0, astar), {"k": k}


def _r4_fam_b(g, fl, r):
    """fams.fam_B before K3 round 5 (REVIEW 6a S-2): the cube, the order drawn until the asked name (slot 0) has the
    block's intro rank, whatever its statement count."""
    import fams as Fm
    cube = g.cube(r)
    k = fl.setdefault("k", 8 if fl["struct"] else int(r.integers(2, 5)))
    names = [cube[int(i)] for i in r.permutation(8)[:k]]
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    keys = [(e, astar) for e in range(k)] + [(e, Fm._attrs(g, r, 1, exclude=astar)[0]) for e in range(k)
                                             if r.random() < 0.5]
    units = [[I.St(Fm._form(r), e, a, v)] for (e, a), v in zip(keys, g.values(r, len(keys)))]
    return Fm.arrange(units, lambda flat: I.intro_rank(flat, 0) == fl["rank"], r), (0, astar), {"k": k, "names": names}


def _fam_blocks(fam, nb, tag, split="train"):
    g = S.Gen(pools(), split)
    return [I.render(it, f"{fam}.b{b}.i{i}") for b in range(nb) for i, it in enumerate(S.block(g, fam, 512, (tag, ord(fam),
                                                                                                          b)))]


def _size_cue(orig):
    """REVIEW 7b demo_size.py's planted cue: an item kept only if the key has the most statements among the holders
    of the asked attribute, else at 1/2 (else None: skill.block redraws)."""
    from collections import Counter

    def planted(g, fl, r, k=None, by=False):
        order, key, meta = orig(g, fl, r, k, by)
        size = Counter(s.ent for u in order for s in u)
        hold = {s.ent for u in order for s in u if s.attr == key[1]}
        if size[0] < max(size[e] for e in hold) and r.random() >= 0.5:
            return None, None, None
        return order, key, meta
    return planted


def _attr2_cue(orig):
    """REVIEW 8b T-3 (its demo_mut.py attr2): a cue confined to asked entities with 2 keys. When the asked entity
    states 2 attributes held by different numbers of entities and the asked key is the less-held one, it moves to the
    more-held one with probability 1/2, so the more-held key is asked in 3/4 of those items (the asked entity and the
    order as fam_R draws them; the order does not depend on which of its keys is asked)."""
    def planted(g, fl, r, k=None, by=False):
        order, key, meta = orig(g, fl, r, k, by)
        if order is None:
            return order, key, meta
        own = sorted({s.attr for u in order for s in u if s.ent == 0})
        h = {a: len({s.ent for u in order for s in u if s.attr == a}) for a in own}
        if len(own) == 2 and h[own[0]] != h[own[1]]:
            more = max(own, key=h.get)
            if key[1] != more and r.random() < 0.5:
                key = (0, more)
        return order, key, meta
    return planted


def _uniform_order(monkeypatch):
    """fams.sym_order replaced by the round-4/5 way: a uniform order of the units, drawn until the fixed entities
    (sigma_with's {entity: rank}) have their ranks, so bigger entities come first and the asked one's rank is
    least expected for its size more often than chance (mutant r_order_conditioned)."""
    import fams as Fm

    class Sig(list):
        fixed = {}

    def sig(k, fixed, r, _orig=Fm.sigma_with):
        out = Sig(_orig(k, fixed, r))
        out.fixed = dict(fixed)
        return out

    def order(units_of, sigma, r, _orig=Fm.sym_order):
        flat = [u for us in units_of.values() for u in us]
        got = Fm.arrange(flat, lambda f: all(I.intro_rank(f, e) == x for e, x in sigma.fixed.items()), r, tries=2000)
        return got if got is not None else _orig(units_of, sigma, r)
    monkeypatch.setattr(Fm, "sigma_with", sig)
    monkeypatch.setattr(Fm, "sym_order", order)


def test_r_and_p_keys_are_exchangeable(monkeypatch):
    """K3 round 6 (REVIEW 7a S-1): fams.fam_R draws the asked key uniformly over the item's keys. From tokens on 8 R and
    8 P blocks: no fail; the key is any of the m holders alike (its place among them uniform for each m, so R-POST
    and O2 read the mean of 1/m), in every cell of the holders' intro ranks and sizes and of the asked entity's keys
    (purity's exch rules, 30 df or more each). REVIEW 7b T-1: a planted size cue (kept only if the key has the most
    statements among the holders, else at 1/2) fails R.exch_size and P.exch_size, and not exch_rank (mutant
    exch_size_name_order); its None reaches fam_P, which returns it (REVIEW 7b O-1). Every entity's intro rank is
    uniform whatever its size (exch_intro); a uniform order kept for the asked entity's rank fails it (a cross-fitted
    rank-and-size cell rule 0.403 against E[1/m] 0.388 on 16 R blocks with it, 0.384 against 0.387 without). The
    round-4 R order (frac(key) conditioned) fails R.exch_rank and P.exch_rank. K3 round 7 (REVIEW 8b T-3): the asked
    entity has 2 keys in about 0.42 of R and P items; a cue confined to them (the more-held key asked in 3/4 of the
    2-key items whose keys differ in holders, _attr2_cue) fails R.exch_attr and P.exch_attr (mutant
    exch_attr_three_keys_only reads 3-key entities only and passes it: 62.1 on 48 df in REVIEW 8b)."""
    import fams as Fm
    import math
    import oldfams
    recs = _fam_blocks("R", 8, 5301) + _fam_blocks("P", 8, 5302)
    rep = U.accept(recs, pools(), "train", lex())
    assert rep["fails"] == [], rep["fails"]
    facts = [U.item_facts(r, lex()) for r in recs]
    for fam in "RP":
        F = [f for f in facts if f["fam"] == fam]
        want = float(np.mean([1 / len(f["hranks"]) for f in F]))
        for o in ("R-POST", "O2"):
            got = rep["oracles"][fam][o]
            assert abs(got - want) < 3 * math.sqrt(want * (1 - want) / len(F)), (fam, o, got, want)
        assert all(rep["balance"][f"{fam}.exch_{x}"][1] >= 30 for x in ("rank", "size", "attr")), rep["balance"]
        assert rep["balance"][f"{fam}.exch_intro"][1] >= 10, rep["balance"]
        for m in (2, 3):
            last = [f["hpos"] == m - 1 for f in F if len(f["hranks"]) == m]
            assert abs(np.mean(last) - 1 / m) < 3 * math.sqrt((1 / m) * (1 - 1 / m) / len(last)), (fam, m)
    real = Fm.fam_R
    monkeypatch.setattr(Fm, "fam_R", _size_cue(real))
    fails = U.accept(_fam_blocks("R", 8, 5301) + _fam_blocks("P", 8, 5302), pools(), "train", lex())["fails"]
    assert any(x.startswith("R.exch_size") for x in fails) and any(x.startswith("P.exch_size") for x in fails), fails
    assert not any("exch_rank" in x for x in fails), fails
    monkeypatch.setattr(Fm, "fam_R", _attr2_cue(real))     # K3 round 7 (REVIEW 8b T-3): the 2-key part of exch_attr
    fails = U.accept(_fam_blocks("R", 8, 5301) + _fam_blocks("P", 8, 5302), pools(), "train", lex())["fails"]
    assert any(x.startswith("R.exch_attr") for x in fails) and any(x.startswith("P.exch_attr") for x in fails), fails
    monkeypatch.setattr(Fm, "fam_R", real)
    _uniform_order(monkeypatch)
    fails = U.accept(_fam_blocks("R", 8, 5301) + _fam_blocks("P", 8, 5302), pools(), "train", lex())["fails"]
    assert any(x.startswith("R.exch_intro") for x in fails) and any(x.startswith("P.exch_intro") for x in fails), fails
    oldfams.use(monkeypatch, "RP")
    monkeypatch.setattr(Fm, "fam_R", _r4_fam_r)
    fails = U.accept(_fam_blocks("R", 8, 5301) + _fam_blocks("P", 8, 5302), pools(), "train", lex())["fails"]
    assert any(x.startswith("R.exch_rank") for x in fails) and any(x.startswith("P.exch_rank") for x in fails), fails


def test_round5_designs_fail_the_attribute_blind_bars(monkeypatch):
    """REVIEW 7a S-1: round 5 built the asked key unlike the asked entity's other keys (R and P: its attribute held by
    2 or more entities in every foil item; U: restated 2-4 times with a foil), so an attribute-blind tracker found it.
    On 8 blocks each the round-5 designs fail T's bar with TA-SHARE (R, P) and TA-CNT (U) and fail the exch_attr rules;
    round 6's read every TA rule near 1/n (under 0.50) and pass."""
    import oldfams
    tags = (("R", 5304), ("U", 5305), ("P", 5306))
    new = U.accept([r for f, t in tags for r in _fam_blocks(f, 8, t)], pools(), "train", lex())
    assert new["fails"] == [] and max(new["oracles"][f][o] for f in "RUP" for o in O.TA) < 0.5, new["oracles"]
    oldfams.use(monkeypatch)
    old = U.accept([r for f, t in tags for r in _fam_blocks(f, 8, t)], pools(), "train", lex())
    for fam, o in (("R", "TA-SHARE"), ("U", "TA-CNT"), ("P", "TA-SHARE")):
        assert any(x.startswith(f"{fam}: {o} ") for x in old["fails"]), (fam, o, old["oracles"][fam][o])
        assert any(x.startswith(f"{fam}.exch_attr") for x in old["fails"]), (fam, old["balance"])


def test_bars_cover_the_attribute_blind_class():
    """K3 round 6 (REVIEW 7a S-1, S-2): T and the TA rules carry T's bar (0.65) on R, U, F and P (P's T read
    0.660-0.676 in round 5 with no bar), never on B (its asked attribute is every entity's: TA reads 1.00 there by
    design) or A; R-POST on R and P. A table at 0.70 for each fails exactly those."""
    row = {o: 0.0 for o in [*O.NAMES, *O.VIEWS]} | {"IDEAL": 1.0, "n": 4096}
    tab = {f: dict(row) | {o: 0.7 for o in ("T", *O.TA, "R-POST")} for f in "RUBFAP"}
    fails = U.threshold_fails(tab, {})
    got = {(x.split(":")[0], x.split(" ")[1]) for x in fails}
    want = {(f, o) for f in "RUFP" for o in ("T", *O.TA)} | {("R", "R-POST"), ("P", "R-POST")}
    assert got == want, sorted(got ^ want)


def test_ta_rules_on_handmade_items():
    """K3 round 6: TA-SHARE (the asked entity's key whose attribute the most holders state), TA-REP (the key it
    states most, then holders), TA-CNT (most statements in the item), TA-SHREP (holders, then its count); ties to
    the key stated latest; the latest value of that key. None of them reads the question's attribute."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(3)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = sorted(int(x) for x in p["V"][:10])
    St, Qn = I.St, I.Qn
    sts = [St("full", 0, a0, v[0]), St("full", 1, a0, v[1]), St("full", 2, a0, v[2]),   # a0: 3 holders, 3 in all
           St("full", 0, a1, v[3]), St("reord", 0, a1, v[4]), St("full", 0, a1, v[5]),  # a1: entity 0 three times
           St("full", 0, a2, v[6]), St("full", 1, a2, v[7]), St("full", 1, a2, v[8]),   # a2: 2 holders, 4 in all
           St("reord", 1, a2, v[9])]
    for q in (Qn(0, a0, 0), Qn(0, a2, 1)):          # the question's attribute changes no answer
        rec = I.render(I.Item("R", nm, [("S", [x]) for x in sts] + [("Q", q)]), "R.b0.i0")
        st, qs, _ = O.parse(rec, lex())
        out = O.answers(st, qs[-1], O.prompt_of(rec, qs[-1]["turn"]), lex())
        assert (out["TA-SHARE"], out["TA-REP"], out["TA-CNT"], out["TA-SHREP"], out["T"]) == (
            v[0], v[5], v[6], v[0], v[6]), out
    tie = sts[:6] + [St("full", 2, a1, v[6]), St("full", 1, a1, v[7])]          # a0 and a1 both 3 holders: the key
    rec = I.render(I.Item("R", nm, [("S", [x]) for x in tie] + [("Q", Qn(0, a0, 0))]), "R.b0.i0")     # stated latest
    st, qs, _ = O.parse(rec, lex())
    assert O.answers(st, qs[-1], O.prompt_of(rec, qs[-1]["turn"]), lex())["TA-SHARE"] == v[5]


def _ell_item(fam="F"):
    """K3 round 7 (REVIEW 8b T-1): an F-shaped item whose asked key is an ellipsis ("a0 v1" right after N0's statement),
    a0 also stated by N1 and N2, a2 by N0 and N2: the asked entity's keys by T's join are a1, a0 (the ellipsis) and
    a2; by name alone only a1 and a2. Final question N0 a0, gold v1."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(3)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = sorted(int(x) for x in p["V"][:6])
    St = I.St
    sts = [St("full", 0, a1, v[0]), St("ell", 0, a0, v[1]), St("full", 1, a0, v[2]), St("full", 2, a0, v[5]),
           St("full", 0, a2, v[3]), St("full", 2, a2, v[4])]
    return I.render(I.Item(fam, nm, [("S", [x]) for x in sts] + [("Q", I.Qn(0, a0, 0))]), f"{fam}.b0.i0"), v


def test_ta_rules_join_an_ellipsis():
    """K3 round 7 (REVIEW 8b T-1): the TA rules take the asked entity's keys as T does, by name or by ellipsis adjacency
    (oracles: tent). On _ell_item the ellipsis key a0 has 3 holders and 3 statements, the others 1 or 2, so TA-SHARE,
    TA-CNT, TA-SHREP and TA-REP (counts tied at 1, then holders) all answer the ellipsis's value, the gold; with the
    keys taken by name only (probe ta_oracle_ellipsis_dropped, which read F's TA rules 0.195-0.238 against 0.440-0.567
    on 8 blocks) a0 is no key and they answer v3. T reads N0's latest statement, v3."""
    rec, v = _ell_item()
    st, qs, _ = O.parse(rec, lex())
    assert st[1]["kind"] == "ell" and st[1]["tent"] == qs[-1]["name"] and qs[-1]["gold"] == v[1]
    out = O.answers(st, qs[-1], O.prompt_of(rec, qs[-1]["turn"]), lex())
    assert out["IDEAL"] == v[1] and out["T"] == v[3], out
    assert all(out[o] == v[1] for o in O.TA), {o: out[o] for o in O.TA}


def test_u_tightspan_and_f_turnk_on_handmade_items():
    """K3 round 7: U-TIGHTSPAN (REVIEW 8a G-1; a ceiling row, check_k.ceilings): an earlier question identical to the
    final one is copied (its answer's statement's name, that name's latest value of the attribute); else the holder
    whose statements of the asked attribute span the fewest user turns, ties sharing credit. F-TURNK (REVIEW 8a N-1,
    reported): O4's quarter, else of the two reference statements the one sharing its turn with a named statement of
    the asked name (both or neither: a tie)."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(3)]
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = sorted(int(x) for x in p["V"][:8])
    St, Qn = I.St, I.Qn

    def ans(fam, ex, o):
        rec = I.render(I.Item(fam, nm, ex), f"{fam}.b0.i0")
        st, qs, _ = O.parse(rec, lex())
        return O.answers(st, qs[-1], O.prompt_of(rec, qs[-1]["turn"]), lex())[o]
    wide = [("S", [St("full", 0, a0, v[0])]), ("S", [St("full", 1, a0, v[1])]), ("S", [St("reord", 1, a0, v[2])]),
            ("S", [St("full", 0, a0, v[3])])]                        # N0's 3 exchanges apart, N1's 1
    assert ans("U", wide + [("Q", Qn(0, a0, 0))], "U-TIGHTSPAN") == v[2]
    lure = wide[:1] + [("Q", Qn(0, a0, 1))] + wide[1:] + [("Q", Qn(0, a0, 0))]     # the lure: copy N0's latest
    assert ans("U", lure, "U-TIGHTSPAN") == v[3]
    tie = [("S", [St("full", 1, a0, v[1]), St("reord", 1, a0, v[2])]), ("S", [St("full", 0, a0, v[4]),
                                                                           St("full", 0, a0, v[5])])]
    got = ans("U", tie + [("Q", Qn(1, a0, 0))], "U-TIGHTSPAN")
    assert sorted(got) == sorted([v[2], v[5]]) and O.credit(got, v[2]) == 0.5, got
    ex = [("S", [St("full", 1, a2, v[0]), St("ell", 1, a0, v[1])]),             # the foil's ellipsis, N1's turn
          ("S", [St("full", 0, a1, v[2]), St("ell", 0, a0, v[3])])]             # the key's, beside N0's statement
    assert ans("F", ex + [("Q", Qn(0, a0, 0))], "F-TURNK") == v[3]
    late = ex + [("S", [St("full", 0, a0, v[4])])]                              # O4's quarter
    assert ans("F", late + [("Q", Qn(0, a0, 0))], "F-TURNK") == v[4]
    both = [("S", [St("full", 0, a2, v[0]), St("ell", 0, a0, v[1])])] + ex[1:] + [("S", [St("full", 2, a1, v[5])])]
    assert sorted(ans("F", both + [("Q", Qn(0, a0, 0))], "F-TURNK")) == [v[1], v[3]]
    assert O.TH["U-TIGHTSPAN"] == ("RUP", .65) and "F-TURNK" in O.REPORTED and "F-TURNK" not in O.TH


def test_b_asked_name_is_exchangeable(monkeypatch):
    """K3 round 5 (REVIEW 6a S-2): the build drew B's order until the asked name had the block's intro rank whatever its
    statement count, while a name with 2 statements is usually introduced early, so B-SURPRISE (B-ADAPT's partial key,
    the tie broken toward the name whose rank is least expected for its count) read 0.68 on training items, over O3_S's
    bar. Now the asked name is the one at the flagged rank of a uniform order: on 16 training blocks no fail, its
    statement count at its rank is every name's there (B.exch), B-SURPRISE at B-ADAPT's level. The build's order fails
    B.exch and B-SURPRISE's bar."""
    import fams as Fm
    rep = U.accept(_fam_blocks("B", 16, 5303), pools(), "train", lex())
    assert rep["fails"] == [] and rep["balance"]["B.exch"][1] >= 15, (rep["fails"], rep["balance"])
    o = rep["oracles"]["B"]
    assert o["B-SURPRISE"] < 0.645 and abs(o["B-SURPRISE"] - o["B-ADAPT"]) < 0.03, o
    monkeypatch.setattr(Fm, "fam_B", _r4_fam_b)
    old = U.accept(_fam_blocks("B", 16, 5303), pools(), "train", lex())
    assert any(x.startswith("B.exch") for x in old["fails"]) and any("B: B-SURPRISE " in x for x in old["fails"]), (
        old["fails"], old["oracles"]["B"]["B-SURPRISE"])


def test_r_post_and_b_surprise_on_handmade_items():
    """K3 round 5: R-POST (holders in the order they state the asked attribute: 1 -> it, 2 -> the first, 3 or more ->
    the last) and B-SURPRISE (B-ADAPT's subset; its tie to the name whose intro rank is least expected for its
    statement count: rank / (k - 1) with 2 statements, 1 - rank / (k - 1) with 1; then the smallest value id)."""
    p = pools()
    s1, s2, s3 = p.triple("S")
    nm = [(int(s1[i]), int(s2[i]), int(s3[i])) for i in range(4)]
    a0, a1 = (int(x) for x in p["A"][:2])
    v = sorted(int(x) for x in p["V"][:8])

    def ans(fam, names, sts, q):
        rec = I.render(I.Item(fam, names, [("S", [s]) for s in sts] + [("Q", q)]), f"{fam}.b0.i0")
        st, qs, _ = O.parse(rec, lex())
        return O.answers(st, qs[-1], O.prompt_of(rec, qs[-1]["turn"]), lex())
    St, Qn = I.St, I.Qn
    two = [St("full", 1, a1, v[0]), St("full", 0, a0, v[1]), St("reord", 1, a0, v[2])]
    assert ans("R", nm, two, Qn(1, a0, 0))["R-POST"] == v[1]            # 2 holders: the first to state a0
    three = two + [St("full", 2, a0, v[3])]
    assert ans("R", nm, three, Qn(0, a0, 0))["R-POST"] == v[3]          # 3 holders: the last
    assert ans("R", nm, [St("full", 3, a0, v[4]), St("full", 0, a1, v[5])], Qn(0, a0, 0))["R-POST"] == v[4]
    x, y, z = ([int(t[0]), int(t[1])] for t in (s1, s2, s3))
    c = [(x[0], y[0], z[0]), (x[0], y[0], z[1]), (x[0], y[1], z[0]), (x[1], y[0], z[0])]
    # asked c[0]; each proper subset of its positions holds 2 or 3 names, the first with 2 is {1, 2}: c[0] and c[1].
    # c[0] is introduced first with 1 statement (surprise 1 - 0 / 3), c[1] second with 2 (1 / 3), so B-SURPRISE takes
    # c[0]'s value while B-ADAPT takes the smaller value id, c[1]'s
    b = [St("full", 0, a0, v[7]), St("full", 1, a1, v[0]), St("reord", 1, a0, v[2]), St("full", 2, a0, v[3]),
         St("full", 3, a0, v[4])]
    out = ans("B", c, b, Qn(0, a0, 0))
    assert out["B-ADAPT"] == v[2] and out["B-SURPRISE"] == v[7], out


def _one_hop(turns, ents):
    """skill.component with mutation_k's component_one_hop applied: it hops on only from `ents` (the final entity and
    the holders), so a bystander two hops out may be asked (REVIEW 11b comp_probe.py's draw)."""
    by_attr, by_ent = {}, {}
    for t in turns:
        for s in t:
            if s.kind != "def":
                by_attr.setdefault(s.attr, set()).add(s.ent)
                by_ent.setdefault(s.ent, set()).add(s.attr)
    out, todo = set(), list(ents)
    while todo:
        e = todo.pop()
        if e in out:
            continue
        out.add(e)
        todo += [x for a in by_ent.get(e, ()) for x in by_attr[a] if x not in out] if e in ents else []
    return out


def test_component_rule_reads_past_one_hop(monkeypatch):
    """K3 round 10 (REVIEW 11b T-2): purity's early_comp rule closes the component of the final name and the holders
    over shared attributes, past one hop; its only plant (round 5's question draw, _no_comp) breaks it one hop out.
    A U block drawn with skill.component cut to one hop asks an entity two or more hops out on some items (REVIEW 11b:
    68 of 2,048). From tokens (names joined by a named shared attribute before the final question), each such item
    is flagged and the block fails with the component line (mutant m11_purity_comp_one_hop: purity's component stops
    one hop out and passes them)."""
    monkeypatch.setattr(S, "component", _one_hop)
    recs, far = _fam_blocks("U", 1, 11811), 0
    for rec in recs:
        comps, keys, st, qs = _components(rec)
        q = qs[-1]
        seed = {q["name"]} | {s["name"] for s in st[:q["n"]] if s["attr"] == q["attr"] and s["name"] is not None}
        comp = set().union(seed, *(c for c in comps if c & seed))
        near = seed | {u for u in keys if any(keys[u] & keys.get(w, set()) for w in seed)}
        if any(p["name"] in comp - near for p in qs[:-1]):
            far += 1
            assert U.item_facts(rec, lex())["early_comp"], rec["id"]
    fails = U.accept(recs, pools(), "train", lex(), block=512)["fails"]
    assert far >= 5 and f"{far} items: an earlier question names an entity of the final key's component" in fails, (
        far, fails)
