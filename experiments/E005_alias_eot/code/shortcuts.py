"""E004 shortcut table (step 3): every fake scorer (O1-O8 with the O5 variants, X1-X10), the diagnostic topic
tracker T, IDEAL and chance, on the REAL held-out items of one draw (default: eval seed 4004), per family.

Gates (exit 1 if any fails):
  G1 IDEAL = 1.00 on every family.
  G2 (step 3) every fake scorer < 0.80 on every pass-rule family, except the NAMING rules on C_noupd (O5 variants,
     O6, X3, X4, X7, X9), which are correct by construction there (A has exactly one statement and it names A;
     see notes.txt step 3); those are reported, and must be <= 0.50 on every U family instead.
  G3 (notes (b) item-set acceptance) every O1-O8 row: <= 0.60 on at least two pass-rule families and < 0.80 on
     at least one (fails the seed pass rule); both O5 content variants <= 0.25 on H1 and H2; O2 <= 0.60 on every
     family.
Target (reported, not gated): every fake scorer <= 0.50 on every pass-rule family.
Usage: python3 -B shortcuts.py [eval|dev|probe] [--cells]"""
import sys

import items_e004 as I
from oracles_e004 import View, ideal
from shortcuts_x import FAKE, DIAGNOSTIC, score, chance

NAMING = {"O5 content words vs question", "O5 content words vs q+prefix", "O5 all words vs question",
          "O5 3-word runs vs q+prefix", "O6 latest naming asked", "X3 latest naming asked, full phrase",
          "X4 value at last naming", "X7 first naming asked", "X9 latest unmarked naming asked"}
U_FAMS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]


def table(D):
    fams = list(D)
    rows = [(name, {f: score(D[f], fn) for f in fams}) for name, fn in FAKE + DIAGNOSTIC + [("IDEAL", ideal)]]
    rows.append(("chance", {f: chance(D[f]) for f in fams}))
    return fams, rows


def fmt(fams, rows):
    head = f"{'scorer':38s}" + "".join(f"{f:>10s}" for f in fams)
    out = [head, "-" * len(head)]
    for name, r in rows:
        out.append(f"{name:38s}" + "".join(f"{r[f]:10.2f}" for f in fams))
    return out


def family_gate(f, scores):
    """G1 and G2 for one family; scores = {scorer name: score}. Returns (failures, notes)."""
    fail, notes = [], []
    if scores["IDEAL"] != 1.0:
        fail.append(f"G1 IDEAL {scores['IDEAL']:.2f} on {f}")
    if f not in I.PASS_FAMILIES:
        return fail, notes
    for name, _ in FAKE:
        v = scores[name]
        exempt = f == "C_noupd" and name in NAMING
        if v >= 0.80 and not exempt:
            fail.append(f"G2 {name} {v:.2f} on {f}")
        elif v > 0.50:
            notes.append(f"above target 0.50: {name} {v:.2f} on {f}" + (" (naming rule, exempt)" if exempt else ""))
        if name in NAMING and f in U_FAMS and v > 0.50:
            fail.append(f"G2 naming rule {name} {v:.2f} > 0.50 on U family {f}")
    return fail, notes


def gates(fams, rows):
    R = dict(rows)
    fail, notes = [], []
    P = [f for f in I.PASS_FAMILIES if f in fams]
    for f in fams:
        a, b = family_gate(f, {name: R[name][f] for name in R})
        fail, notes = fail + a, notes + b
    for name, _ in FAKE:
        if not name.startswith("O"):
            continue
        low = [f for f in P if R[name][f] <= 0.60]
        if len(low) < 2 or all(R[name][f] >= 0.80 for f in P):
            fail.append(f"G3 {name}: <= 0.60 on {len(low)} families")
        if name.startswith("O5 content"):
            fail += [f"G3 {name} {R[name][f]:.2f} > 0.25 on {f}" for f in ("H1", "H2") if f in P and R[name][f] > 0.25]
    fail += [f"G3 O2 {R['O2 last mention'][f]:.2f} > 0.60 on {f}" for f in fams if R["O2 last mention"][f] > 0.60
             and f in P]
    return fail, notes


def by_cell(D, fam, key):
    """per-cell (controls) or per-reference-form (H1/H2) scores of every scorer on one family."""
    groups = {}
    for it in D[fam]:
        g = it["cell"] if key == "cell" else it["meta"].get(key)
        groups.setdefault(g, []).append(it)
    names = [n for n, _ in FAKE + DIAGNOSTIC]
    fns = dict(FAKE + DIAGNOSTIC)
    out = [f"{fam} by {key}: " + ", ".join(f"{g} (n={len(v)})" for g, v in groups.items())]
    for n in names:
        out.append(f"  {n:38s}" + "".join(f"{score(v, fns[n]):7.2f}" for v in groups.values()))
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "eval"
    D = I.draw(name)
    fams, rows = table(D)
    lines = [f"E004 shortcut table, draw {name} (seed {I.DRAWS[name][0]}, {I.DRAWS[name][1]} per family)", ""]
    lines += fmt(fams, rows)
    fail, notes = gates(fams, rows)
    lines += [""] + notes + [""] + [f"FAIL {x}" for x in fail]
    lines.append("RESULT: " + ("ALL GATES PASS" if not fail else f"{len(fail)} GATE FAILURES"))
    if "--cells" in sys.argv:
        lines += [""] + by_cell(D, "C_noupd", "cell") + by_cell(D, "C_twoslot", "cell")
        lines += by_cell(D, "H1", "latest_ref") + by_cell(D, "H2", "latest_ref")
    print("\n".join(lines))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
