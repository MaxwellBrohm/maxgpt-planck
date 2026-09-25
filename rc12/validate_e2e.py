"""RC-12 end-to-end gates with fake responders (SPEC s5 G1-G4), through the REAL runner: every fake plays every dev
conversation in the 12-turn loop on its own replies, is graded by the real graders, written to
runs/fakes/<NAME>/transcripts.jsonl + scores.jsonl, READ BACK from disk and scored by score.py. No model.

  G1 IDEAL and IDEAL_ALT: every family, cell, T0, K and COMPOSE key = 1.00; 0 degenerate replies; 0 role-leak
     events; Level A met (comparator = IDEAL itself).
  G2 every cheater, on the families it targets: family <= 0.40 and every cell <= 0.50; USERVOICE, ECHO, DEFLECT,
     SHOTGUN (plus ABSTAIN, ROLECAP) score 0.00 on every value-graded unit. Generic cheaters target all 10
     composite families; P_NEVER / P_ONCE / P_FIRST5 target PERSIST; P_OBEYALL targets PERSIST:override (it is
     IDEAL on hold by construction); L_* target LOOP.
  G3 no generic cheater meets any Level A content criterion (CORR:U, CORR:C_noupd, CORR:C_twoslot, BIND), judged
     both by score.level_a's verdict and against 0.80 here (the fixed "Okay, noted." ack already fails the loop
     criterion, so Level A as a whole must not be the only thing checked); every generic and LOOP cheater misses
     Level A as a whole (comparator = IDEAL); LOOP cheaters miss the loop bar.
  G4 per composite unit, the number of the generic cheaters that pass it (unit score 1.0) is <= 3; histogram.
     OVERLAP_A and OVERLAP_B are the two variants of ONE cheater (SPEC s5), so 11 cheaters are counted (D14).
  Step 5 audit cheaters (fakes_audit.py): the value rules get G2 and G3 exactly like the generic cheaters;
     ORDER_ABS is gated on the cells it targets (RECALL:abstain, LOOKUP:*) at the cell bar; MYONLY, MARKER_OBJ
     (the skill a cell measures), CONSIST (OWN, see notes) and VARIED (LOOP, see notes) are reported, not gated.
     The G4 histogram is also written for the 11 generic + the audit value rules together (diagnostic).
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
BAR = dict(family=0.40, cell=0.50, item=3, level_a=0.80)
NAMES = ["IDEAL", "IDEAL_ALT"] + F.GENERIC + list(FF.FAMILY) + FA.AUDIT
GATED = F.GENERIC + FA.VALUE_RULES                 # G2 on every composite family and cell, and G3


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
    return [f"G1 {name}: {b}" for b in bad]


def targets(name):
    if name in GATED:
        return S.COMPOSITE, None
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
        bad += [f"family {f}={s['families'][f]:.2f}" for f in fams if s["families"][f] > BAR["family"]]
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
    return [f"G3 {name}: {b}" for b in bad]


def table(summ):
    keys = S.COMPOSITE + ["CORR:U", "CORR:C_noupd", "CORR:C_twoslot", "T0", "K:followup"]
    lines = [f"{'fake':11s}" + "".join(f"{k.replace('CORR:', '')[:8]:>9s}" for k in keys) + "   loop   R"]
    for n, s in summ.items():
        vals = "".join(f"{s['keys'].get(k, float('nan')):9.2f}" for k in keys)
        lines.append(f"{n:11s}{vals}  {s['loop_rate']:.3f} {s['R']:5.1f}")
    return lines


def g4(rows, names=F.GENERIC):
    """per composite unit: which generic cheaters pass it; OVERLAP_A / OVERLAP_B count as ONE cheater (notes D14)."""
    passes = defaultdict(set)
    for n in names:
        for u in S.units(rows[n]):
            if u["family"] in S.COMPOSITE and (u["family"], u["cell"]) not in S.DIAG:
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
    hist, items = g4(rows)
    hist_x, items_x = g4(rows, F.GENERIC + FA.VALUE_RULES)
    return dict(recs=recs, rows=rows, summ=summ, core=core, items=items, hist=hist, hist_x=hist_x, items_x=items_x)


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
              f"{cells_units(g['rows']['P_OBEYALL'])[1][('PERSIST', 'hold')]:.2f}", ""]
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
