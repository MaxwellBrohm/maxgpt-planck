"""RC-12 end-to-end gates with fake responders (SPEC s5 G1-G4), through the REAL runner: every fake plays every dev
conversation in the 12-turn loop on its own replies, is graded by the real graders, written to
runs/fakes/<NAME>/transcripts.jsonl + scores.jsonl, READ BACK from disk and scored by score.py. No model.

  G1 IDEAL and IDEAL_ALT: every family, cell, T0, K and COMPOSE key = 1.00; 0 degenerate replies; 0 role-leak
     events; Level A met (comparator = IDEAL itself).
  G2 every cheater, on the families it targets: family <= 0.40 and every cell <= 0.50; USERVOICE, ECHO, DEFLECT,
     SHOTGUN (plus ABSTAIN, ROLECAP) score 0.00 on every value-graded unit. Generic cheaters target all 10
     gated families (FAMS: the 9 composite families and LOOKUP, which left R on 2026-10-02 but keeps its s8 gates
     because its cells are still reported; Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md
     (item 1)); P_NEVER / P_ONCE / P_FIRST5 target PERSIST; P_OBEYALL targets PERSIST:override (it is
     IDEAL on hold by construction); L_* target LOOP.
  G3 no generic cheater meets any Level A content criterion (CORR:U, CORR:C_noupd, CORR:C_twoslot, BIND), judged
     both by score.level_a's verdict and against BAR level_a here (0.60 since the s10 re-anchor; Max, 2026-10-02: took
     all recommendations in rc12/DECISIONS_FOR_MAX.md (item D); the fixed "Okay, noted." ack already fails the loop
     criterion, so Level A as a whole must not be the only thing checked); every generic and LOOP cheater misses
     Level A as a whole (comparator = IDEAL); LOOP cheaters miss the loop bar. LOOKUP claim (Max, 2026-10-04: its
     own headline claim, prereg draft s10b): no LOOKUP cheater (the generic cheaters, the audit value rules and
     ORDER_ABS, which OD3 allows 0.50 at the cell bar) meets it, neither by score.lookup_claim nor against BAR lookup
     here; the family fakes are IDEAL on LOOKUP by construction. G1 also asks IDEAL and IDEAL_ALT to meet it.
     Level R (Max, 2026-10-04: took all recommendations in rc12/DECISIONS_LEVEL_R_FOR_MAX.md (decision 1, D)): no
     generic cheater or audit value rule reaches BAR s7 (40, Level R's S7 bar) on S7_ungated (the own-history OWN, an
     upper bound of the gated S7); and the shortcut ceiling (shortcut_ceiling: per STATE7 family the best of the
     SHORTCUT fakes, OWN gated by each fake's own --own-cf OWN run) stays below BAR s7. The ceiling is reported beside
     Level R (dev here; the sealed one is the same function on the sealed records), never a gate on a model.
  G4 per gated-family unit (FAMS), the number of the generic cheaters that pass it (unit score 1.0) is <= 3; histogram.
     OVERLAP_A and OVERLAP_B are the two variants of ONE cheater (SPEC s5), so 11 cheaters are counted (D14).
  Step 5 audit cheaters (fakes_audit.py): the value rules get G2 and G3 exactly like the generic cheaters;
     ORDER_ABS is gated on the cells it targets (RECALL:abstain, LOOKUP:*) at the cell bar; MYONLY, MARKER_OBJ
     (the skill a cell measures), CONSIST (OWN, see notes) and VARIED (LOOP, see notes) are reported, not gated.
     The G4 histogram is also written for the 11 generic + the audit value rules together (diagnostic).
  The table prints the loop rate and, beside it, the OD6 (iii) ack-repeat rate over every reply (reported only).
  No --own-cf runs here: OWN is the own-history score (keys), R is None (OD1 b) and the table prints R_ungated;
     the OD1 b gate itself is checked in validate_machinery.c_owngate.
Writes logs/e2e_fakes.txt; exit 1 if G1-G3 fail, 2 if only G4 fails (items to rebuild), 0 if all pass.
Run: python3 -B validate_e2e.py"""
import os
import sys
from collections import Counter, defaultdict

import fakes as F
import fakes_audit as FA
import fakes_family as FF
import runner as RN
import score as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "fakes")
VALUE_FAMS = ["RECALL", "CORR", "BIND", "TWOHOP", "OWN", "TOPIC", "ROLE", "LOOKUP"]
ZERO_ON_VALUE = ["USERVOICE", "ECHO", "DEFLECT", "SHOTGUN", "ABSTAIN", "ROLECAP"]
A_CONTENT = ["CORR:U", "CORR:C_noupd", "CORR:C_twoslot", "BIND"]
BAR = dict(family=0.40, cell=0.50, item=3, level_a=0.60, s7=40.0, lookup=0.60)
FAMS = S.COMPOSITE + S.REPORTED                     # the families the s8 gates cover (LOOKUP kept, item 1)
NAMES = ["IDEAL", "IDEAL_ALT"] + F.GENERIC + list(FF.FAMILY) + FA.AUDIT
GATED = F.GENERIC + FA.VALUE_RULES                 # G2 on every composite family and cell, and G3
SHORTCUT = GATED + FA.REFERENCE + ["ORDER_ABS", "CONSIST"]   # fixed-rule fakes not IDEAL on any family by design


def play(recs, name, render="plain", write=True):
    out_dir = os.path.join(OUT, name) if write else None
    rows = RN.run(recs, FF.make(name), render, [None], None, out_dir, name)
    return S.load_rows(os.path.join(out_dir, "transcripts.jsonl")) if write else rows


def cells_units(rows):
    us = S.units(rows)
    cells = defaultdict(list)
    for u in us:
        if (u["family"], u["cell"]) not in S.DIAG:
            cells[(u["family"], u["cell"])].append(u["score"])
    return us, {k: sum(v) / len(v) for k, v in cells.items()}


def g1(name, s):
    bad = [f"{k}={v:.2f}" for k, v in sorted(s["keys"].items()) if v != 1.0]
    if any(s["degenerate_rates"].values()):
        bad.append(f"degenerate {s['degenerate_rates']}")
    if s["capture_rate"]:
        bad.append(f"role-leak events in {s['capture_rate']:.3f} of conversations")
    if not s["level_a_met"]:
        bad.append("Level A not met")
    if not s["lookup_claim"]["met"]:
        bad.append("LOOKUP claim bar not met")
    return [f"G1 {name}: {b}" for b in bad]


def targets(name):
    if name in GATED:
        return FAMS, None
    return [FF.FAMILY[name][0]], FF.FAMILY[name][1]


def g2_order(name, cells):
    """ORDER_ABS: every RECALL:abstain and LOOKUP cell <= the cell bar (its family is reported, not gated)."""
    return [f"G2 {name}: cell {f}:{c}={v:.2f}" for (f, c), v in sorted(cells.items())
            if (f == "LOOKUP" or (f, c) == ("RECALL", "abstain")) and v > BAR["cell"]]


def g2(name, s, us, cells):
    if name == "ORDER_ABS":
        return g2_order(name, cells)
    if name in FA.REFERENCE + ["CONSIST", "VARIED"]:
        return []
    fams, only = targets(name)
    bad = []
    if only is None:
        bad += [f"family {f}={s['keys'][f]:.2f}" for f in fams if s["keys"][f] > BAR["family"]]
    for (f, c), v in sorted(cells.items()):
        if f in fams and (only is None or c == only) and v > BAR["cell"]:
            bad.append(f"cell {f}:{c}={v:.2f}")
    if name in ZERO_ON_VALUE:
        nz = [u["uid"] for u in us if u["family"] in VALUE_FAMS and (u["family"], u["cell"]) not in S.DIAG
              and u["score"] > 0]
        if nz:
            bad.append(f"{len(nz)} value-graded units above 0 (e.g. {nz[:3]})")
    return [f"G2 {name}: {b}" for b in bad]


def g3(name, s):
    bad = []
    if name in GATED:
        bad += [f"{k}={s['keys'][k]:.2f} meets its Level A criterion" for k in A_CONTENT if s["level_a"][k]["met"]]
        bad += [f"{k}={s['keys'][k]:.2f} >= {BAR['level_a']}" for k in A_CONTENT if s["keys"][k] >= BAR["level_a"]]
    if name.startswith("L_") and s["level_a"]["LOOP"]["met"]:
        bad.append(f"loop rate {s['loop_rate']:.3f} meets the loop bar")
    if (name in GATED or name.startswith("L_")) and s["level_a_met"]:
        bad.append("meets Level A")
    lk = s["keys"].get("LOOKUP")
    if (name in GATED or name == "ORDER_ABS") and (s["lookup_claim"]["met"] or lk is None or lk >= BAR["lookup"]):
        bad.append(f"LOOKUP={lk} meets the LOOKUP claim bar (s10b)")
    if name in GATED and (s["S7_ungated"] is None or s["S7_ungated"] >= BAR["s7"]):
        bad.append(f"S7_ungated={s['S7_ungated']} reaches the Level R S7 bar")
    return [f"G3 {name}: {b}" for b in bad]


def shortcut_ceiling(recs, rows):
    """the Level R shortcut ceiling: per STATE7 family the best SHORTCUT fake (OWN: its OD1 b gated score, from an
    --own-cf OWN run of the same fake on recs), S7 of that mix, and beside it the same with OWN at the best
    own-history OWN of the fakes not built to repeat their own pick (CONSIST is: 1.00 on its own history, 0 gated)."""
    own = [r for r in recs if r["family"] == "OWN"]
    best, slip = {}, (0.0, None)
    for n in SHORTCUT:
        s = S.summarize(rows[n] + RN.run(own, FF.make(n), "plain", [None], None, None, n, 0, True))
        for f in S.STATE7:
            if s["families"][f] is not None and s["families"][f] > best.get(f, (-1.0, None))[0]:
                best[f] = (s["families"][f], n)
        if n != "CONSIST" and s["keys"]["OWN"] > slip[0]:
            slip = (s["keys"]["OWN"], n)
    s7 = 100 * sum(best[f][0] for f in S.STATE7) / len(S.STATE7)
    s7_slip = s7 + 100 * (slip[0] - best["OWN"][0]) / len(S.STATE7)
    return dict(best=best, S7=s7, own_slipped=slip, S7_own_slipped=s7_slip)


def table(summ):
    keys = FAMS + ["CORR:U", "CORR:C_noupd", "CORR:C_twoslot", "T0", "K:followup"]
    lines = [f"{'fake':11s}" + "".join(f"{k.replace('CORR:', '')[:8]:>9s}" for k in keys) + "   loop    ack  R_ug S7_ug"]
    for n, s in summ.items():
        vals = "".join(f"{s['keys'].get(k, float('nan')):9.2f}" for k in keys)
        ack = "  None" if s["ack_repeat"] is None else f"{s['ack_repeat']:.3f}"
        lines.append(f"{n:11s}{vals}  {s['loop_rate']:.3f}  {ack} {s['R_ungated']:5.1f} {s['S7_ungated']:5.1f}")
    return lines


def g4(rows, names=F.GENERIC):
    """per composite unit: which generic cheaters pass it; OVERLAP_A / OVERLAP_B count as ONE cheater (notes D14)."""
    passes = defaultdict(set)
    for n in names:
        for u in S.units(rows[n]):
            if u["family"] in FAMS and (u["family"], u["cell"]) not in S.DIAG:
                key = (u["family"], u["cell"], u["uid"])
                passes[key] |= {n.split("_")[0] if n.startswith("OVERLAP") else n} if u["score"] == 1.0 else set()
    hist = Counter(len(v) for v in passes.values())
    over = sorted((k, sorted(v)) for k, v in passes.items() if len(v) > BAR["item"])
    return hist, [f"G4 {k[0]}:{k[1]} {k[2]} passed by {len(v)}: {','.join(v)}" for k, v in over]


def gates(write=True):
    recs = RN.load()
    rows = {n: play(recs, n, write=write) for n in NAMES}
    ideal = S.summarize(rows["IDEAL"])
    summ = {n: S.summarize(rows[n], comparator=ideal) for n in NAMES}
    core = []
    for n in NAMES:
        us, cells = cells_units(rows[n])
        core += g1(n, summ[n]) if n.startswith("IDEAL") else g2(n, summ[n], us, cells) + g3(n, summ[n])
    ceil = shortcut_ceiling(recs, rows)
    if ceil["S7"] >= BAR["s7"]:
        core.append(f"G3 shortcut ceiling S7 {ceil['S7']:.2f} reaches the Level R S7 bar {BAR['s7']}")
    hist, items = g4(rows)
    hist_x, items_x = g4(rows, F.GENERIC + FA.VALUE_RULES)
    return dict(recs=recs, rows=rows, summ=summ, core=core, items=items, hist=hist, hist_x=hist_x, items_x=items_x,
                ceiling=ceil)


def main():
    g = gates(write=True)
    lines = ["RC-12 end-to-end fake gates (validate_e2e.py). Plain render, greedy, no context limit, no model.",
             f"{len(g['recs'])} records x {len(NAMES)} fakes through runner.play -> runs/fakes/<NAME>/"
             "transcripts.jsonl, read back and scored by score.py.", ""] + table(g["summ"])
    hist = g["hist"]
    lines += ["", "G4 histogram (composite units passed by k of the 11 generic cheaters, OVERLAP = A or B): "
              + ", ".join(f"{k}:{hist[k]}" for k in sorted(hist)),
              "G4 diagnostic, 11 generic + " + str(len(FA.VALUE_RULES)) + " audit value rules: "
              + ", ".join(f"{k}:{g['hist_x'][k]}" for k in sorted(g["hist_x"])),
              "reported, not gated: " + ", ".join(f"{n} {k}={g['summ'][n]['keys'][k]:.2f}" for n, k in (
                  ("MYONLY", "RECALL"), ("MYONLY", "ROLE"), ("MARKER_OBJ", "CORR:U-same"), ("CONSIST", "OWN"),
                  ("VARIED", "LOOP"), ("ORDER_ABS", "LOOKUP"), ("ORDER_ABS", "RECALL"))),
              "OBEYALL on PERSIST:hold (IDEAL by construction, not gated): "
              f"{cells_units(g['rows']['P_OBEYALL'])[1][('PERSIST', 'hold')]:.2f}",
              "Level R shortcut ceiling (best SHORTCUT fake per state family, OWN gated by its --own-cf run): S7 "
              f"{g['ceiling']['S7']:.2f} ("
              + ", ".join(f"{f} {v:.3f} {n}" for f, (v, n) in g["ceiling"]["best"].items()) + "); with OWN at the "
              f"best own-history OWN ({g['ceiling']['own_slipped'][1]} {g['ceiling']['own_slipped'][0]:.3f}): "
              f"{g['ceiling']['S7_own_slipped']:.2f}; bar {BAR['s7']}", ""]
    lines += [f"FAIL {f}" for f in g["core"]] or ["G1-G3 PASS"]
    lines += [f"FAIL {f}" for f in g["items"]] or ["G4 PASS"]
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "e2e_fakes.txt"), "w") as f:
        f.write(text)
    print(text)
    return 1 if g["core"] else 2 if g["items"] else 0


if __name__ == "__main__":
    sys.exit(main())
