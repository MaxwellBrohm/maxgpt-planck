"""Family F after review R-1 (2026-09-27): the reference shortcut oracles, SURF, and F's token-level design.

R-1: when only the asked entity used ellipsis, "the latest statement of the asked attribute that names no other
entity" (O12) scored 0.908 on training F. Each oracle below is pinned on a hand-built item; the design tests read the
generated items from tokens alone (oracles.parse), never from the generator's structures.
"""
import copy
import functools
import math
from collections import Counter

import items as I
import oracles as O
import purity as U
import skill as S
from items import Qn, St
from kfix import lex, pools


@functools.lru_cache(maxsize=None)
def fblock(split="train", n=512, seed=0):
    g = S.Gen(pools(), split)
    return tuple(I.render(it, f"F.b{seed}.i{i}") for i, it in enumerate(S.block(g, "F", n, (91, ord("F"), seed))))


def _tok():
    p = pools()
    s1, s2, s3 = p.triple("S")
    a = [int(x) for x in p["A"][:2]]
    v = [int(x) for x in p["V"][:8]]
    return (int(s1[0]), int(s2[0]), int(s3[0])), (int(s1[0]), int(s2[1]), int(s3[1])), a, v, int(p["AL"][0])


def _answers(it):
    rec = I.render(it, "F.b0.i0")
    st, qs, _ = O.parse(rec, lex())
    q = qs[-1]
    return O.answers(st, q, O.prompt_of(rec, q["turn"]), lex()), O.form_pattern(st, q), q["gold"]


def test_reference_oracles_on_handmade_items():
    n0, n1, (a0, a1), v, x = _tok()
    # key: stale named v1, ellipsis v4 (its latest); foil: alias v5 after the key's latest
    it = I.Item("F", [n0, n1], [("S", [St("full", 0, a1, v[0]), St("full", 0, a0, v[1])]),
                                ("S", [St("def", 1, alias=x), St("full", 1, a1, v[2])]),
                                ("S", [St("full", 0, a1, v[3]), St("ell", 0, a0, v[4])]),
                                ("S", [St("alias", 1, a0, v[5], x)]), ("Q", Qn(0, a0, 0))])
    ans, pat, gold = _answers(it)
    assert gold == v[4] and ans["IDEAL"] == v[4] and ans["O2"] == v[5] and ans["O4"] == v[1]
    assert [ans[o] for o in ("O11", "O11|2", "O12", "O12p")] == [v[5]] * 4
    assert (ans["O13e"], ans["O13a"], ans["O14e"], ans["O14a"]) == (v[5], v[4], v[4], v[1])
    assert pat == (("N", "E", "A"), 1)
    st, qs, _ = O.parse(I.render(it, "F.b0.i0"), lex())
    assert O.form_pattern(st, qs[-1], "ell")[0] == ("N", "e", "A")        # SURFe: the key's ellipsis resolved
    assert O.form_pattern(st, qs[-1], "alias")[0] == ("N", "E", "A")      # SURFa: the alias is the foil's
    # key: ellipsis v1; foil (sharing the asked name's first token) restates the attribute by name after it
    it = I.Item("F", [n0, n1], [("S", [St("def", 1, alias=x), St("full", 0, a1, v[0])]),
                                ("S", [St("ell", 0, a0, v[1])]), ("S", [St("reord", 1, a0, v[2])]),
                                ("Q", Qn(0, a0, 1))])
    ans, pat, gold = _answers(it)
    assert gold == v[1] and ans["O2"] == v[2] and ans["O11"] == v[1] and ans["O12"] == v[1] and ans["O12p"] == v[2]
    assert (ans["O13e"], ans["O13a"], ans["O14e"], ans["O14a"]) == (v[1], v[1], v[1], -1)
    assert pat == (("E", "O"), 0)


def test_surf_is_cross_fitted():
    alt = [{"pat": ("E", "A"), "gidx": i % 2} for i in range(8)]
    assert U.surf(alt) == 0.0                      # an in-sample fit would score 0.5 here
    same = [{"pat": ("E", "A", "N"), "gidx": 2}] * 6
    assert U.surf(same) == 1.0
    assert U.surf([{"pat": ("A", "E"), "gidx": 1}]) == 1.0         # unseen pattern: the latest statement
    fit = U.surf_counts(same + [{"pat": ("E", "A"), "gidx": 0}] * 3)
    assert fit == {"EAN": {"2": 6}, "EA": {"0": 3}}
    assert U.surf(alt, fit) == 0.5


def test_reference_thresholds():
    for o in [*O.REFS, "SURF", *O.FINTRO]:
        assert O.TH[o][0] == "F"
    row = {**{o: 0.0 for o in O.NAMES}, "IDEAL": 1.0, "SURF": 0.6, "O13e": .75, "O13a": .75, "SURFe": .8125,
           "SURFa": .8125, "n": 512}
    assert U.threshold_fails({"F": row}, {}) == []
    for o, bad in (("O12", 0.60), ("O11", 0.60), ("O13a", 0.85), ("O14e", 0.75), ("SURF", 0.75), ("SURFe", 0.91),
                   ("SURFa", 0.91), ("F-INTRO", 0.75), ("F-DEFX", 0.75)):
        assert any(f"F: {o} " in x for x in U.threshold_fails({"F": {**row, o: bad}}, {})), o


def test_f_blocks_meet_the_bars():
    for split in ("train", "eval"):
        rep = U.accept(list(fblock(split)), pools(), split, lex(), block=512)
        assert rep["fails"] == [], rep["fails"]
        row = rep["oracles"]["F"]
        assert row["IDEAL"] == 1.0 and row["O12"] == 0.5 and row["O13e"] == 0.75 and abs(row["O14a"] - .625) < .03
        assert max(row[o] for o in O.REFS[:4]) <= 0.5 and row["SURF"] <= 0.65 + 3 * math.sqrt(.65 * .35 / 512)
        for v in ("SURFe", "SURFa"):        # one kind resolved: the late quarter, that kind's half, a coin (0.8125)
            assert 0.75 < row[v] <= 0.85 + 3 * math.sqrt(.85 * .15 / 512), (v, row[v])


NB = 8          # F blocks pooled by the last-statement rate test (4 until K3 round 2)
NB_REF = 32     # and by the key-versus-foil ending test (REVIEW 3b T-3: a foil kept at 0.7 against the key's 0.5 passed
                # 4 blocks by 2 counts; at LAST_KEEP 0.35, a foil kept at 1.4 x LAST_KEEP read 607 vs 673 on 8 blocks,
                # limit 107, and 1,195 vs 1,365 on 16, limit 152; K3 round 3, after BD-25 changed F's draws: 1,192 vs
                # 1,323 on 16, limit 150, passed; 2,375 vs 2,638 on 32, limit 212; unmutated 2,264 vs 2,346)


def test_f_last_statement_rate_at_chance():
    """SPEC 4: the key's latest is the item's last statement at chance. Without fams.LAST_KEEP's thinning F read
    0.27 vs 0.20 (2 blocks, 2026-09-27); with BD-21 and LAST_KEEP 0.5 it read 0.228 vs 0.192 (fresh blocks, K3
    round 2), at 0.35 0.196 vs 0.192. Pooled over NB blocks the rate must sit within 0.025 (3.8 SE) of chance."""
    facts = [U.item_facts(r, lex()) for seed in range(NB) for r in fblock(seed=seed)]
    got = sum(f["is_last"] for f in facts) / len(facts)
    exp = sum(f["n_key"] / f["n_st"] for f in facts) / len(facts)
    assert abs(got - exp) < 0.025, (got, exp)


def test_f_key_and_foil_references_end_items_equally():
    """K3 R-3: fams.LAST_KEEP thinned only orders ending with the key's latest, so the foil's reference statement
    ended 0.272 of the reference items and the key's 0.192, and LASTSKIP (O4's quarter, else the reference statement
    that does not end the item) read 0.681. Both are thinned now: over NB_REF blocks the two counts agree within
    3 SE, and purity.balance applies the same rule to every checked set."""
    facts = [U.item_facts(r, lex()) for seed in range(NB_REF) for r in fblock(seed=seed)]
    ref = [f for f in facts if f["key_ref_latest"]]
    a, b = sum(f["ref_last"] is True for f in ref), sum(f["ref_last"] is False for f in ref)
    assert len(ref) == 384 * NB_REF and a + b > 1600 and abs(a - b) <= 3 * math.sqrt(a + b), (a, b)
    rep, _ = U.balance(facts, 512)
    assert rep["F.ref_last"] == (round(a / len(ref), 4), round(b / len(ref), 4))


def test_f_ref_last_rule_catches_a_planted_imbalance():
    facts = [dict(U.item_facts(r, lex())) for seed in range(4) for r in fblock(seed=seed)]
    free = [f for f in facts if f["key_ref_latest"] and f["ref_last"] is None][:200]
    for f in free:
        f["ref_last"] = False               # 200 more items that end with the foil's reference statement
    _, fails = U.balance(facts, 512)
    assert any("the item ends with the key's reference statement" in x for x in fails), fails


def _stmts(rec):
    st, qs, intro = O.parse(rec, lex())
    return st[:qs[-1]["n"]], qs[-1], intro


def test_f_design_from_tokens():
    """Every entity states exactly one attribute in a reference form; the foil states the asked attribute that way
    and never after it; stale named statements of the asked attribute come before both reference statements."""
    ell = Counter()
    for rec in fblock():
        S_, q, intro = _stmts(rec)
        refs = Counter(s["ent"] for s in S_ if s["kind"] in O.REF_KINDS)
        assert refs == Counter({e: 1 for e in intro}), rec["id"]
        ell.update(s["ent"] != q["name"] and s["attr"] != q["attr"] for s in S_ if s["kind"] == "ell")
        a_st = [(i, s) for i, s in enumerate(S_) if s["attr"] == q["attr"]]
        foils = {s["ent"] for _, s in a_st if s["ent"] != q["name"]}
        assert len(foils) == 1
        ref_at = {s["ent"]: i for i, s in a_st if s["kind"] in O.REF_KINDS}
        assert len(ref_at) == 2 and [s["kind"] in O.REF_KINDS for _, s in a_st if s["ent"] in foils][-1]
        kinds = {S_[i]["ent"] == q["name"]: S_[i]["kind"] for i in ref_at.values()}
        assert kinds[True] == kinds[False], rec["id"]           # BD-14: one reference kind per item
        ell["key_alias"] += kinds[True] == "alias"
        for i, s in a_st:           # a named statement after a reference one can only be the key's latest
            if s["kind"] in ("full", "reord") and i > min(ref_at.values()):
                assert s["ent"] == q["name"] and i > ref_at[q["name"]] and s["val"] == q["gold"], rec["id"]
    assert ell["key_alias"] == 256
    n_other = sum(len(_stmts(r)[2]) - 2 for r in fblock())
    assert abs(ell[True] / n_other - 0.5) < 3 * math.sqrt(0.25 / n_other), (ell[True], n_other)


def _to_named(rec, pred):
    """Rewrite every reference statement matching pred as a full named statement of its entity (tokens only)."""
    S_, q, _ = _stmts(rec)
    for s in S_:
        if s["kind"] in O.REF_KINDS and pred(s, q):
            ids = rec["turns"][s["turn"]]["ids"]
            j = next(j for j in range(len(ids)) if ids[j:j + len(s["toks"])] == s["toks"])
            ids[j:j + len(s["toks"])] = [*s["ent"], s["attr"], s["val"]]
    return rec


def test_named_foils_reproduce_r1_and_fail():
    recs = [_to_named(copy.deepcopy(r), lambda s, q: s["ent"] != q["name"]) for r in fblock(n=64, seed=1)]
    rep = U.accept(recs, pools(), "train", lex(), block=64)
    assert rep["oracles"]["F"]["O12"] == 1.0 and rep["oracles"]["F"]["IDEAL"] == 1.0
    assert any("F: O12 " in x for x in rep["fails"]) and any("follow_ref" in x for x in rep["fails"])


def test_f_balance_rules_catch_a_planted_break():
    """One follow item's trailing foil alias turned into a named statement: the follow_alias and follow_ref rules
    must fail (the oracles barely move)."""
    recs = [copy.deepcopy(r) for r in fblock(n=64, seed=2)]
    assert U.accept(recs, pools(), "train", lex(), block=64)["fails"] == []
    r = next(r for r in recs if U.item_facts(r, lex())["follow_alias"])
    _to_named(r, lambda s, q: s["ent"] != q["name"] and s["attr"] == q["attr"] and s["kind"] == "alias")
    assert U.item_facts(r, lex())["follow"] and not U.item_facts(r, lex())["follow_ref"]
    fails = U.accept(recs, pools(), "train", lex(), block=64)["fails"]
    assert any("follow_alias" in x for x in fails) and any("follow_ref" in x for x in fails), fails


def test_f_flags_nested_exact():
    import numpy as np
    for n in (512, 154, 64, 2):
        fl = S.block_flags("F", n, np.random.default_rng(n))
        assert all(f["kref"] == f["fref"] for f in fl)
        assert abs(sum(f["follow"] for f in fl) * 2 - n) < 2 and abs(sum(f["klate"] for f in fl) * 4 - n) < 4
        assert abs(sum(f["kref"] == "alias" for f in fl) * 2 - n) < 2
        assert abs(sum(f["follow"] and f["fref"] == "alias" for f in fl) * 4 - n) < 4
        assert abs(sum(not f["klate"] and f["kstale"] for f in fl) * 8 - 3 * n) < 8
        assert abs(sum(f["fstale"] for f in fl) * 2 - n) < 2


def test_mixed_kinds_reproduce_elimination_and_fail(monkeypatch):
    """BD-14: with the foil's reference kind the other one, resolving one kind names the key by elimination (SURFe
    and SURFa near 1.0; the independent kinds of the first redesign read 0.90); the one-kind bars and the same_kind
    rule must fail."""
    orig = S.block_flags

    def mixed(fam, n, r, *a):
        return [{**f, "fref": "ell" if f["kref"] == "alias" else "alias"} for f in orig(fam, n, r, *a)]
    monkeypatch.setattr(S, "block_flags", mixed)
    recs = [I.render(it, f"F.b0.i{i}") for i, it in
            enumerate(S.block(S.Gen(pools(), "train"), "F", 512, (91, ord("F"), 9)))]
    rep = U.accept(recs, pools(), "train", lex(), block=512)
    row = rep["oracles"]["F"]
    assert row["IDEAL"] == 1.0 and row["SURFe"] > 0.95 and row["SURFa"] > 0.95 and row["SURF"] < 0.7, row
    for x in ("F: SURFe ", "F: SURFa ", "same_kind"):
        assert any(x in f for f in rep["fails"]), (x, rep["fails"])


def test_intro_order_oracles_on_handmade_items():
    """F-INTRO and F-DEFX (REVIEW 3a): a reference before the asked name's first explicit mention, or an alias before
    its definition, cannot be the asked entity's; of the rest the earliest."""
    n0, n1, (a0, a1), v, _ = _tok()
    x0, x1 = (int(t) for t in pools()["AL"][:2])
    # foil's alias before the asked name's definition and before any mention of it: both rules drop it
    it = I.Item("F", [n0, n1], [("S", [St("def", 1, alias=x1), St("full", 1, a1, v[5])]),
                                ("S", [St("alias", 1, a0, v[0], x1)]), ("S", [St("def", 0, alias=x0)]),
                                ("S", [St("full", 0, a1, v[6]), St("alias", 0, a0, v[1], x0)]), ("Q", Qn(0, a0, 0))])
    ans, _, gold = _answers(it)
    assert gold == v[1] and ans["F-INTRO"] == v[1] and ans["F-DEFX"] == v[1]
    # the asked name is mentioned first, then the foil's alias, then the asked name's definition: only F-DEFX drops it
    it = I.Item("F", [n0, n1], [("S", [St("full", 0, a1, v[5]), St("def", 1, alias=x1)]),
                                ("S", [St("alias", 1, a0, v[0], x1)]), ("S", [St("def", 0, alias=x0)]),
                                ("S", [St("alias", 0, a0, v[1], x0)]), ("Q", Qn(0, a0, 1))])
    ans, _, gold = _answers(it)
    assert gold == v[1] and ans["F-INTRO"] == v[0] and ans["F-DEFX"] == v[1]
    # nothing dropped (ellipses after both first mentions): the earliest reference; a named key after it: O4
    it = I.Item("F", [n0, n1], [("S", [St("full", 0, a1, v[5]), St("full", 1, a1, v[6])]),
                                ("S", [St("full", 1, a1, v[7]), St("ell", 1, a0, v[0])]),
                                ("S", [St("full", 0, a1, v[2]), St("ell", 0, a0, v[1])]), ("Q", Qn(0, a0, 0))])
    ans, _, gold = _answers(it)
    assert gold == v[1] and ans["F-INTRO"] == v[0] and ans["F-DEFX"] == v[0]
    it.ex.insert(3, ("S", [St("reord", 0, a0, v[3])]))
    it.ex[-1] = ("Q", Qn(0, a0, 0))
    ans, _, gold = _answers(it)
    assert gold == v[3] and ans["F-INTRO"] == v[3] == ans["F-DEFX"] == ans["O4"]


def test_f_reference_units_follow_both_first_mentions():
    """BD-21 (K3 round 2, REVIEW 3a F-INTRO-DEFX), from tokens: the key's and the foil's reference statements come
    after both entities' first explicit mention and definition, and an ellipsis's antecedent is never its entity's
    first mention (with only the first rule, the order of the references followed the asked name's intro rank on
    ellipsis items, 0.54). So F-INTRO and F-DEFX drop nothing and read the design's 0.625."""
    facts = []
    for seed in range(4):
        for rec in fblock(seed=seed):
            st, qs, intro = O.parse(rec, lex())
            q = qs[-1]
            S_ = st[:q["n"]]
            refs = [i for i, s in enumerate(S_) if s["attr"] == q["attr"] and s["kind"] in O.REF_KINDS]
            ents = {S_[i]["ent"] for i in refs}
            first = {e: min([i for i, s in enumerate(S_) if s["name"] == e] +
                            [n - 0.5 for n, u, _ in q["defs"] if u == e]) for e in ents}
            defs = [n - 0.5 for n, u, _ in q["defs"] if u in ents]
            start = min(i - (S_[i]["kind"] == "ell") for i in refs)
            assert len(ents) == 2 and start > max(list(first.values()) + defs), rec["id"]
            facts.append(U.item_facts(rec, lex()))
    tab = U.oracle_table(facts)["F"]
    assert abs(tab["F-INTRO"] - 0.625) < 0.02 and abs(tab["F-DEFX"] - 0.625) < 0.02, tab


def test_f_aliases_sit_in_both_definition_windows():
    """K3 round 3 (REVIEW 4a S-2, BD-25): skill._in_range put each alias 1-10 exchanges after its own definition, so
    the key's alias never shared the asked name's definition turn while the foil's could (F-WINK: right on every one
    of the 4.7-5.4% of alias items where it fired; F-XNEAR lifted the alias half to 0.66). From tokens: both aliases of
    the asked attribute sit 1-10 exchanges after both entities' definitions, and over NB_REF blocks the alias nearer
    after the asked name's definition is the key's as often as the foil's (reference items, 3 SE)."""
    n_alias, facts = 0, []
    for seed in range(NB_REF):
        for rec in fblock(seed=seed):
            S_, q, _ = _stmts(rec)
            als = [s for s in S_ if s["attr"] == q["attr"] and s["kind"] == "alias"]
            dx = {u: e for e, u in q["dex"]}
            for s in als:
                assert all(1 <= s["ex"] - dx[x["ent"]] <= 10 for x in als), rec["id"]
            n_alias += bool(als)
            facts.append(U.item_facts(rec, lex()))
    for rec in fblock("eval", 256):
        assert U.item_facts(rec, lex())["alias_win"], rec["id"]
    assert n_alias == 256 * NB_REF
    x = [f["xnear"] for f in facts if f["xnear"] is not None]
    a, b = sum(x), len(x) - sum(x)
    assert a + b > 1800 and abs(a - b) <= 3 * math.sqrt(a + b), (a, b)
    assert U.balance(facts, 512)[0]["F.xnear"] == (a, b)


def test_alias_window_rules_catch_planted_breaks():
    """purity: one F item with an alias outside a definition's window fails the per-item rule; 200 more reference
    items whose nearer alias is the foil's fail the F.xnear balance rule."""
    facts = [dict(U.item_facts(r, lex())) for seed in range(4) for r in fblock(seed=seed)]
    assert U.balance(facts, 512)[1] == []
    bad = [dict(f) for f in facts]
    bad[3]["alias_win"] = False
    assert any("1 F items: an alias outside a definition's window" in x for x in U.balance(bad, 512)[1])
    bad = [dict(f) for f in facts]
    for f in [f for f in bad if f["xnear"] is None][:200]:
        f["xnear"] = False
    assert any("the alias nearer after the asked definition" in x for x in U.balance(bad, 512)[1])


def test_alias_window_reads_both_definitions_from_tokens():
    """K3 round 4 (REVIEW 5b T-a): purity's per-item rule (BD-25) from tokens, not a flag set by hand. A hand-built F
    item whose aliases sit 5 and 4 exchanges after their own definitions, but the foil's 12 after the asked name's
    (and the key's before the foil's definition): alias_win is False and the balance check fails it, while an item
    with both definitions first passes. A rule reading each alias against its own definition only (the pre-round-3
    window, mutant alias_win_own_def) passes the first item."""
    p = pools()
    n0, n1, _, _, _ = _tok()
    a0, a1, a2 = (int(x) for x in p["A"][:3])
    v = [int(x) for x in p["V"][:12]]
    x0, x1 = (int(t) for t in p["AL"][:2])
    d0, d1 = St("def", 0, alias=x0), St("def", 1, alias=x1)
    k_al, f_al = St("alias", 0, a0, v[0], x0), St("alias", 1, a0, v[1], x1)
    fill = [St("full", 0, a1, v[2]), St("reord", 1, a1, v[3]), St("full", 0, a2, v[4]), St("reord", 1, a2, v[5]),
            St("full", 0, a1, v[6]), St("reord", 1, a1, v[7]), St("full", 0, a2, v[8]), St("reord", 1, a2, v[9]),
            St("full", 0, a1, v[10])]
    bad = [d0, *fill[:4], k_al, *fill[4:6], d1, *fill[6:9], f_al]           # exchanges 0-12, the question at 13
    good = [d0, d1, *fill[:2], k_al, *fill[2:4], f_al]
    recs = [I.render(I.Item("F", [n0, n1], [("S", [x]) for x in b] + [("Q", Qn(0, a0, 0))]), "F.b9.i0")
            for b in (bad, good)]
    st, qs, _ = O.parse(recs[0], lex())
    dx = {u: e for e, u in qs[-1]["dex"]}
    al = {s["ent"]: s["ex"] for s in st if s["kind"] == "alias"}
    assert [al[u] - dx[u] for u in (n0, n1)] == [5, 4] and al[n1] - dx[n0] == 12 and al[n0] < dx[n1], (al, dx)
    fb, fg = (U.item_facts(r, lex()) for r in recs)
    assert fb["all_ideal"] and fb["gold"] == v[0] and not fb["alias_win"] and fg["alias_win"], (fb, fg)
    facts = [U.item_facts(r, lex()) for r in fblock(seed=0)]
    assert U.balance(facts + [fg], 512)[1] == []
    assert "1 F items: an alias outside a definition's window" in U.balance(facts + [fb], 512)[1]


def test_f_ante_reported_oracle_on_a_handmade_item():
    """REVIEW 4a S-3 (reported, not barred): F-ANTE drops an ellipsis whose antecedent's attribute the asked name
    states in another named statement (each entity states each other attribute once), so it reads the ellipsis's
    antecedent and a question-to-statement name match; here it finds the key's ellipsis that O11 misses."""
    n0, n1, _, v, _ = _tok()
    a0, a1, a2 = (int(x) for x in pools()["A"][:3])
    it = I.Item("F", [n0, n1], [("S", [St("full", 0, a1, v[0])]),
                                ("S", [St("full", 0, a2, v[4]), St("ell", 0, a0, v[5])]),
                                ("S", [St("full", 1, a1, v[2]), St("ell", 1, a0, v[3])]), ("Q", Qn(0, a0, 0))])
    ans, _, gold = _answers(it)
    assert gold == v[5] and ans["O11"] == v[3] and ans["F-ANTE"] == v[5] and "F-ANTE" not in O.TH
