"""No-model test of the E005 analyzer (analyze_e005*.py, evidence_e005.py, rules_e005.py). System python, CPU only.
Nothing is written anywhere except ../logs/test_analyze_e005.txt (analyze() is called; main() is not).
  A  E004's real records, read-only, must reproduce the independent audit (E004 AUDIT.md) through this code:
     H1/H2 LIK by reference form per seed; the alias adjacency counts 9/14/19; the pooled wrong picks on H1/H2
     alias items (147 wrong: 139 the previous statement, 5 older statements of the asked object, 3 the other
     object; 57 lures) and on H5 (78: 49 + 3 other object, 26 the asked object's previous); H5 by another object's
     indirect correction after (means .67 / .84); E004's own analyzer label and cells; the cross/chat continuity
     comparison by hash (184 matched, 0 flips, 200 only in E004); E004's fixed sub-reading strings are gone
  B  the rules on synthetic inputs: alias HIGH/LOW boundaries, the majority over PLANNED seeds, the AL note, the
     stopping yardstick (|chat - plain| <= 0.10 on every pass family), flips_by_hash, role and placement
     classification, the tie pick, the first-line variant, the join's consistency check
  C  analyze() on an empty out dir reads NOT_SCORED without crashing; on E004 vs itself every E005 - E004 is 0
usage: python3 -B test_analyze_e005.py   (exit 0 only if ALL PASS)"""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_e004 as A
import analyze_e005 as AN
import analyze_e005_b as B5
import evidence_e005 as V
import rules_e005 as R5

LOG = os.path.join(os.path.dirname(HERE), "logs", "test_analyze_e005.txt")
E4 = AN.E004_OUT
AUDIT = {"H1": {"alias": [.19, .33, .33, .19, .10], "ell": [.81, .76, .76, .81, .86], "pron": [1.0] * 5},
         "H2": {"alias": [.24, .52, .38, .33, .38], "ell": [.90, .90, .67, .67, .90], "pron": [.95] * 5}}
_RAW = {}


def raw(tag, name):
    if (tag, name) not in _RAW:
        _RAW[(tag, name)] = AN.recs(E4, tag, name)
    return _RAW[(tag, name)]


def part_audit(ck):
    evs = [V.evidence(V.join(raw(t, "e004__plain"), raw(t, "gen_e004__plain"))) for t in AN.SEEDS]
    for f, forms in AUDIT.items():
        for form, want in forms.items():
            got = [round(e["form"][f][form]["LIK"]["acc"], 2) for e in evs]
            ck(got == want, f"E004 {f} {form} LIK per seed {got} = audit {want}")
    n = {p: evs[0]["alias"]["place"][p]["LIK"]["n"] for p in V.PLACES}
    ck(n == {"adjacent": 9, "filler": 14, "B_sep": 19} and evs[0]["alias"]["all"]["LIK"]["n"] == 42,
       f"alias items 42, placement counts {n} = step 1's 9/14/19")
    tot, lure, nw = R5.pooled_roles(evs, "H1/H2 alias", "LIK")
    ck(nw == 147 and tot.get("A_prev") == 139 and tot.get("A_earlier") == 5 and
       tot.get("other_after", 0) + tot.get("other_before", 0) == 3 and lure == 57,
       f"H1/H2 alias wrong LIK picks {nw} {tot} lures {lure} = audit (81 + 66; 76 + 63 previous; 3 + 2 older; 3 B)")
    tot, lure, nw = R5.pooled_roles(evs, "H5", "LIK")
    ck(nw == 78 and tot.get("other_after") == 49 and tot.get("other_before") == 3 and tot.get("A_prev") == 26,
       f"H5 wrong LIK picks {nw} {tot} = audit (52 other object, 49 after; 26 asked object's own)")
    h5 = [sum(e["h5"]["other_ind_after"][k]["LIK"]["acc"] for e in evs) / 5 for k in ("True", "False")]
    ck(round(h5[0], 2) == .67 and round(h5[1], 2) == .84, f"H5 with / without another indirect after {h5} = .67/.84")
    blk = A.model_block(AN.MODEL, 5, E4, os.path.join(os.path.dirname(E4), "logs"))
    rd = AN.readings(AN.side(E4))
    same = all(AN.run_block(E4, t)["cells"]["LIK"] == blk["seeds"][t]["LIK"] for t in AN.SEEDS)
    ck(rd["rule"]["label"] == blk["reading"]["label"] == "FAIL" and same, "same label and cells as E004's analyzer")
    old = " ".join(blk["reading"]["sub"])
    ck("shortcut signature" in old and not any("shortcut" in s or "does not transfer" in s for s in rd["rule"]["sub"]),
       "E004's fixed sub-reading strings are replaced (present in E004's analyzer, absent here)")
    ck(any("by placement" in s for s in rd["rule"]["sub"]) and any("wrong LIK picks" in s for s in rd["rule"]["sub"]),
       "the FAIL sub-readings carry the placement split and the wrong picks")
    ck(rd["alias"]["label"] == "REAL LIMIT" and rd["alias"]["classes"] == ["LOW", "LOW", "MID", "LOW", "LOW"],
       f"alias reading on E004 (no alias training) {rd['alias']['label']} {rd['alias']['classes']}")
    fl = B5.repro_flips(E4, AN.MODEL, B5.E002_OUT)
    cc = fl.get("cross/chat", {})
    ck(cc.get("matched") == 184 and cc.get("flips") == 0 and cc.get("only_a") == 200 and cc.get("only_b") == 0,
       f"cross/chat compared by hash: {cc}")
    ck(all(v["flips"] == 0 and v["matched"] > 0 for v in fl.values()) and len(fl) == 8, f"8 continuity sets, 0 flips")
    ch = V.chat_cells(raw("s1", "gen_e004__chat"), raw("s1", "gen_e004__plain"))
    ck(ch["H1"]["capped"] == 1.0 and ch["H1"]["strict"] == 0.0 and ch["H1"]["first_line"] > 0.5,
       f"E004 s1 chat H1: capped {ch['H1']['capped']}, strict {ch['H1']['strict']}, first line {ch['H1']['first_line']}")


def item(stmts, asked=0, gold=None, turns=12):
    return dict(stmts=[dict(turn=t, obj=o, value=v, role=r, ref=ref, lure=l) for t, o, v, r, ref, l in stmts],
                asked=asked, gold=gold or [s[2] for s in stmts if s[1] == asked][-1], values=["Mon", "Tue", "Wed",
                "Thu", "Fri", "Sat", "Sun"])


def part_rules(ck):
    C = R5.alias_class
    ck([C(.8, .8), C(.79, .9), C(.5, .5), C(.51, .5), C(None, .9)] == ["HIGH", "MID", "LOW", "MID", None],
       "alias HIGH needs both >= 0.8, LOW both <= 0.5")
    AR = R5.alias_reading
    ck(AR([(.9, .9)] * 3 + [(.1, .1)] * 2, 5)["label"] == "DATA GAP", "3 HIGH of 5 -> DATA GAP")
    ck(AR([(.9, .9)] * 2 + [None] * 3, 5)["label"] == "INCONCLUSIVE", "2 HIGH + 3 missing of 5 planned -> INCONCLUSIVE")
    ck(AR([(.2, .3)] * 3 + [(.9, .9)] * 2, 5)["label"] == "REAL LIMIT", "3 LOW -> REAL LIMIT")
    ck(AR([(.6, .6)] * 5, 5)["label"] == "INCONCLUSIVE", "all MID -> INCONCLUSIVE")
    ck(AR([(.9, .9)] * 5, 5, [.3] * 3 + [.9] * 2)["note"] and not AR([(.9, .9)] * 5, 5, [.6] * 5)["note"],
       "AL note only when AL LIK < 0.5 on a majority")
    base = {f: .9 for f in R5.PASS}
    ck(R5.seed_stops({f: .8 for f in R5.PASS}, base) and not R5.seed_stops(dict(base, H3=.79), base)
       and not R5.seed_stops({f: .9 for f in R5.PASS[:-1]}, base) and not R5.seed_stops(dict(base, H1=.9), {f: .69 for f in R5.PASS}),
       "a seed stops iff |chat - plain| <= 0.10 on every pass family (literal 'within')")
    ok, bad = (base, base), ({f: .2 for f in R5.PASS}, base)
    ck(R5.stopping([ok] * 3 + [bad] * 2, 5)["learned"] and not R5.stopping([ok] * 2 + [None] * 3, 5)["learned"],
       "stopping learned on a majority of planned seeds")
    a = [{"h": "x", "scores": {"gold": -1, "c1": -2}}, {"h": "y", "scores": {"gold": -1, "c1": -2}},
         {"h": "x", "scores": {"gold": -1, "c1": -2}}]
    b = [{"h": "y", "scores": {"gold": -3, "c1": -2}}, {"h": "z", "scores": {"gold": -1, "c1": -2}}]
    r = B5.flips_by_hash(a, b)
    ck((r["matched"], r["flips"], r["only_a"], r["only_b"], r["n_a"]) == (1, 1, 1, 1, 3), f"flips_by_hash {r}")
    try:
        B5.flips_by_hash(a + [{"h": "x", "scores": {"gold": -5, "c1": -2}}], b)
        ck(False, "a hash with two different scores must raise")
    except ValueError:
        ck(True, "a hash with two different scores raises")
    c5 = {"s1": {"LIK": {f: .9 for f in R5.PASS}, "GEN": {f: .9 for f in R5.PASS}, "ID_LIK": .9, "ID_GEN": .9}}
    c4 = {"s1": {"LIK": {f: .7 for f in R5.PASS}, "GEN": {f: .8 for f in R5.PASS}, "ID_LIK": .5, "ID_GEN": .9},
          "s2": None}
    fd = B5.family_diff(c5, c4)
    ck(fd["LIK"]["H1"][2] == 0.2 and fd["GEN"]["H3"][2] == 0.1 and fd["LIK"]["ID"][2] == 0.4 and fd["GEN"]["ID"][2] == 0,
       "family_diff is E005 minus E004 (a missing seed is skipped)")
    it = item([(1, 0, "Mon", "orig", "full", True), (2, 1, "Tue", "orig", "full", False),
               (3, 0, "Wed", "corr", "full", False), (4, 0, "Thu", "corr", "head", False),
               (8, 0, "Sat", "corr", "alias", False), (9, 1, "Fri", "corr", "pron", False)])
    got = [V.role_of(it, v)[0] for v in ("Sat", "Thu", "Wed", "Mon", "Tue", "Fri", "Sun", None, V.TIE)]
    ck(got == ["gold", "A_prev", "A_earlier", "A_earlier", "other_before", "other_after", "not_in_context", "none", "tie"],
       f"role classification {got}")
    ck(V.role_of(it, "Mon")[1] and not V.role_of(it, "Wed")[1], "lure flag follows the picked statement")
    fx = [V.facts(item([(1, 0, "Mon", "orig", "full", False), (t, 0, "Tue", "corr", "alias", False)] + extra))["place"]
          for t, extra in ((2, []), (4, []), (4, [(3, 1, "Wed", "orig", "full", False)]))]
    ck(fx == ["adjacent", "filler", "B_sep"], f"placement classification {fx}")
    f2 = V.facts(it)
    ck(f2["other_after"] and f2["other_ind_after"] and f2["form"] == "alias" and f2["place"] == "filler",
       f"facts of the synthetic item {f2}")
    f1 = V.facts(item([(1, 0, "Mon", "orig", "full", False), (2, 1, "Tue", "orig", "full", False),
                       (3, 0, "Wed", "corr", "pron", False)]))
    ck(f1["other_after"] is False and f1["place"] is None and f1["form"] == "pron", f"nothing after the latest {f1}")
    f3 = V.facts(item([(1, 0, "Mon", "orig", "full", False), (2, 0, "Tue", "corr", "ell", False),
                       (3, 1, "Wed", "orig", "full", False), (4, 1, "Thu", "corr", "pron", False)]))
    ck(f3["other_after"] and f3["other_ind_after"] and f3["form"] == "ell", "H5 split flags")
    rec = {"scores": {"gold": -1.0, "c1": -1.0, "c2": -3.0}, "cand_vals": {"gold": "Mon", "c1": "Tue", "c2": "Wed"}}
    ck(V.lik_pick(rec) is V.TIE and V.lik_pick(dict(rec, scores={"gold": -2.0, "c1": -1.0, "c2": -3.0})) == "Tue",
       "LIK tie -> tie; a foil on top -> that foil")
    ev_it = V.eval_items()[("H1", 0)]
    g = ev_it["gold"]
    good = {"reply": f"{g}, based on what you told me.\n\n{g} again.\n\n{g} again.", "stop": "cap"}
    other = next(v for v in ev_it["candidates"] if v != g)
    ck(V.first_line_strict(good, ev_it) and not V.first_line_strict({"reply": f"{other}.\n{g}."}, ev_it),
       "first-line variant: the first line is graded as a stopped reply")
    r0 = dict(raw("s1", "e004__plain")[0])
    r0["cand_vals"] = dict(r0["cand_vals"], gold="Nonsense")
    try:
        V.join([r0], [])
        ck(False, "a record that does not match its item must raise")
    except ValueError:
        ck(True, "the join refuses a record that does not match its rebuilt item")


def part_analyze(ck):
    with tempfile.TemporaryDirectory() as d:
        res, lines = AN.analyze(d, E4)
        ck(res["e005"]["reading"]["rule"]["label"] == "NOT_SCORED" and res["e005"]["reading"]["alias"]["label"] ==
           "INCONCLUSIVE", "empty E005 out: NOT_SCORED, alias INCONCLUSIVE, no crash")
    res, lines = AN.analyze(E4, E4)
    z = all(v[2] == 0 for part in res["family_diff"].values() for v in part.values())
    ck(z and len(res["family_diff"]["LIK"]) == 10, "E004 vs itself: every family difference is 0")
    ck(any("0.797" in ln for ln in lines) and any("McNemar p=3.09e-08" in ln for ln in lines),
       "0.797 printed as 0.797 and a small McNemar p in scientific notation")


def run_checks():
    res = []

    def ck(ok, msg):
        res.append((bool(ok), msg))
    for part in (part_audit, part_rules, part_analyze):
        try:
            part(ck)
        except Exception as e:  # a crash is a failure, never a pass
            res.append((False, f"CRASH {part.__name__}: {e!r}"))
    return res


def main():
    res = run_checks()
    lines = [("PASS " if ok else "FAIL ") + m for ok, m in res]
    n_fail = sum(not ok for ok, _ in res)
    lines.append(f"{n_fail} failures; {'ALL PASS' if not n_fail else 'FAIL'}")
    open(LOG, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
