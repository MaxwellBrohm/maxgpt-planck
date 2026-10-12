"""E005 step 3: mutation test of the analyzer (rules_e005, evidence_e005, analyze_e005_b) against test_analyze_e005.
Each mutant breaks one rule or classification on purpose; test_analyze_e005.run_checks() must report at least one
FAIL that is not a crash (a crash is NOT a kill). The unmutated baseline must be all PASS first.
The render and loss-mask mutants (7 label layouts, 6 encoders, a prompt-scoring HF loss) live in test_render_e005.py.
usage: python3 -B mutation_e005_c.py > ../logs/mutation_e005_c.txt   (exit code 0 = every mutant killed)"""
import sys
sys.dont_write_bytecode = True

import gen_grade as G
import metrics_ft as MF
import rules_e004 as RU
import rules_e005 as R5
import evidence_e005 as V
import analyze_e005_b as B5
import test_analyze_e005 as T


def m_high_strict():
    def alias_class(lik, gen):
        if lik is None or gen is None:
            return None
        return "HIGH" if lik > .8 and gen > .8 else ("LOW" if lik <= .5 and gen <= .5 else "MID")
    return [(R5, "alias_class", alias_class)]


def m_majority_of_results():
    orig = R5.alias_reading
    return [(R5, "alias_reading", lambda per, n, al=None: orig(per, sum(x is not None for x in per), al))]


def m_stop_tol():
    return [(R5, "STOP_TOL", 0.2)]


def m_stop_one_sided():
    def seed_stops(c, p):
        return bool(c and p) and all(c.get(f) is not None and p.get(f) is not None and c[f] >= p[f] - .1 - 1e-9
                                     for f in R5.PASS)
    return [(R5, "seed_stops", seed_stops)]


def facts_mut(kind):
    orig = V.facts

    def facts(it):
        f = orig(it)
        if f["place"] and kind == "adjacent_off_by_one":
            A = [s for s in it["stmts"] if s["obj"] == it["asked"]]
            f["place"] = "B_sep" if f["place"] == "B_sep" else ("adjacent" if A[-1]["turn"] - A[-2]["turn"] == 2 else "filler")
        if f["place"] == "B_sep" and kind == "no_b_sep":
            f["place"] = "filler"
        if kind == "pron_only":
            la = [s for s in it["stmts"] if s["obj"] == it["asked"]][-1]
            f["other_ind_after"] = any(s["obj"] != it["asked"] and s["turn"] > la["turn"] and s["role"] != "orig"
                                       and s["ref"] == "pron" for s in it["stmts"])
        return f
    return [(V, "facts", facts)]


def role_mut(kind):
    orig = V.role_of

    def role_of(it, v):
        r, lure = orig(it, v)
        if kind == "no_prev" and r == "A_prev":
            r = "A_earlier"
        if kind == "swap_other":
            r = {"other_after": "other_before", "other_before": "other_after"}.get(r, r)
        return r, (False if kind == "no_lure" else lure)
    return [(V, "role_of", role_of)]


def m_tie_is_gold():
    def lik_pick(rec):
        sc = rec["scores"]
        best = max(sc.values())
        return rec["cand_vals"]["gold"] if sc["gold"] == best else rec["cand_vals"][max(sc, key=sc.get)]
    return [(V, "lik_pick", lik_pick)]


def m_first_line_whole():
    def first_line_strict(rec, it):
        gold, pool, obj = G.spec_e004(it)
        return G.grade((rec.get("reply") or "").strip(), rec.get("stop", "eos"), gold, pool, obj)["strict"]
    return [(V, "first_line_strict", first_line_strict)]


def m_positional_flips():
    def flips_by_hash(a, b):
        n = min(len(a), len(b))
        fl = sum(MF.right(x["scores"]) != MF.right(y["scores"]) for x, y in zip(a[:n], b[:n]))
        return {"n_a": len(a), "n_b": len(b), "matched": n, "flips": fl, "only_a": len(a) - n, "only_b": len(b) - n,
                "max_abs_score_diff": 0.0}
    return [(B5, "flips_by_hash", flips_by_hash)]


def m_join_unchecked():
    orig_items = V.eval_items

    def join(lik_recs, gen_recs):
        items, rows = orig_items(), {}
        for part, recs in (("LIK", lik_recs), ("GEN", gen_recs)):
            for r in recs or []:
                it = items[(r["family"], r["idx"])]
                row = rows.setdefault((r["family"], r["idx"]), dict(family=r["family"], idx=r["idx"], **V.facts(it)))
                ok = MF.right(r["scores"]) if part == "LIK" else bool(r["strict"])
                pick = V.lik_pick(r) if part == "LIK" else V.gen_pick(r["reply"], it)
                row[part], (row[part + "_role"], row[part + "_lure"]) = ok, V.role_of(it, pick)
        return list(rows.values())
    return [(V, "join", join)]


def m_old_subreadings():
    return [(R5, "reading", lambda cells, n, ids=(), mid="", evs=(): RU.reading(cells, n, ids, mid))]


def m_alias_h1_only():
    orig = V.evidence

    def evidence(rows):
        ev = orig(rows)
        ev["alias"]["all"] = ev["alias"]["H1"]
        return ev
    return [(V, "evidence", evidence)]


def m_diff_sign():
    orig = B5.family_diff

    def family_diff(c5, c4):
        return orig(c4, c5)
    return [(B5, "family_diff", family_diff)]


def m_p_rounded():
    return [(B5, "p_str", lambda p: "-" if p is None else f"{round(p, 5)}")]


def m_chat_strict_first_line():
    orig = V.chat_cells

    def chat_cells(chat, plain):
        out = orig(chat, plain)
        for f in out.values():
            if "first_line" in f:
                f["strict"] = f["first_line"]
        return out
    return [(V, "chat_cells", chat_cells)]


MUTANTS = {"alias HIGH uses >": m_high_strict, "alias majority over results": m_majority_of_results,
           "stopping tolerance 0.2": m_stop_tol, "stopping one-sided": m_stop_one_sided,
           "adjacency off by one": lambda: facts_mut("adjacent_off_by_one"), "no B-separated": lambda: facts_mut("no_b_sep"),
           "H5 split pronoun only": lambda: facts_mut("pron_only"), "no A_prev class": lambda: role_mut("no_prev"),
           "other before/after swapped": lambda: role_mut("swap_other"), "lure flag dropped": lambda: role_mut("no_lure"),
           "LIK tie counted as gold": m_tie_is_gold, "first line = whole reply": m_first_line_whole,
           "continuity flips by position": m_positional_flips, "join unchecked": m_join_unchecked,
           "E004 fixed sub-readings": m_old_subreadings, "alias items H1 only": m_alias_h1_only,
           "E005 - E004 sign flipped": m_diff_sign, "McNemar p rounded": m_p_rounded,
           "chat strict = first line": m_chat_strict_first_line}


def run(patches):
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    try:
        for mod, name, val in patches:
            setattr(mod, name, val)
        return T.run_checks()
    finally:
        for mod, name, val in saved:
            setattr(mod, name, val)


def main():
    base = run([])
    bad = [m for ok, m in base if not ok]
    print(f"baseline: {len(base)} checks, {len(bad)} failing {bad[:3]}")
    if bad:
        sys.exit(2)
    alive = []
    for name, mk in MUTANTS.items():
        res = run(mk())
        kills = [m for ok, m in res if not ok and not m.startswith("CRASH")]
        crashes = [m for ok, m in res if not ok and m.startswith("CRASH")]
        print(f"{'KILLED' if kills else 'ALIVE '} {name}: {len(kills)} failing checks"
              + (f", first: {kills[0][:110]}" if kills else "") + (f" (crashes {len(crashes)})" if crashes else ""))
        if not kills:
            alive.append(name)
    print(f"{len(MUTANTS) - len(alive)}/{len(MUTANTS)} mutants killed" + (f"; ALIVE: {alive}" if alive else ""))
    sys.exit(1 if alive else 0)


if __name__ == "__main__":
    main()
