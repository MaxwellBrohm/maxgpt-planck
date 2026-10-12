"""E004 decision rules (notes.txt (c) and (d)), pure functions: no files, no model, no torch.
Used by analyze_e004.py and pick_lr_e004.py; tested with mutants by test_rules_e004.py.

  seed pass      for EVERY family f in PASS: LIK_f >= 0.8 AND GEN_f >= 0.8 (a missing score fails)
  model pass     a strict majority of the PLANNED seeds pass (3 of 5, 2 of 3); a missing seed counts as a fail
  reading        checked in the order PASS, PARTIAL-G, PARTIAL-X, FAIL; first match wins
    PARTIAL-G    a majority of seeds pass every LIK cell (named with the failing GEN families)
    PARTIAL-X    a majority of seeds pass every cell (LIK and GEN) of every family outside X, for some X made of
                 one or two of H3 H4 H5 H6 H7 (the smallest such X is named; ties to more passing seeds, then order)
    FAIL         anything else; sub-readings from LIK, each by a majority of seeds
  LR, SmolLM2    5e-5 stays unless 1.5e-4's dev min (over the 9 families, LIK) is higher by >= 0.10 with a mean
                 no lower; the 5e-5 run must be eligible, else NONE
  LR, tiny       E003's rule: best = max (min, mean, -lr); if best is the grid's largest LR and its min < 0.8 and
                 this is not the final pick: EXTEND to the next half-decade (once); no eligible run: NONE
"""
from itertools import combinations

PASS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "C_noupd", "C_twoslot"]
U_FAMS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
CONTROLS = ["C_noupd", "C_twoslot"]
X_AXES = ["H3", "H4", "H5", "H6", "H7"]
THRESH = 0.8
LR_135M_BASE, LR_135M_ALT = "5e-05", "1.5e-04"
LR_135M_MIN_GAIN = 0.10
NEXT_LR = {"1e-03": "3e-03", "3e-03": "1e-02"}


def ok(x):
    return isinstance(x, (int, float)) and x == x and x >= THRESH


def failing(cells, fams=PASS):
    """cells: {"LIK": {fam: acc}, "GEN": {fam: acc}} -> families with a failing (or missing) LIK or GEN cell."""
    return [f for f in fams if not (ok(cells.get("LIK", {}).get(f)) and ok(cells.get("GEN", {}).get(f)))]


def seed_pass(cells):
    return cells is not None and not failing(cells)


def lik_all(cells):
    return cells is not None and all(ok(cells.get("LIK", {}).get(f)) for f in PASS)


def majority(flags, n_planned):
    """strict majority of the planned seeds; flags holds one bool per seed that produced a result."""
    return sum(bool(x) for x in flags) * 2 > n_planned


def model_pass(seed_cells, n_planned):
    return majority([seed_pass(c) for c in seed_cells], n_planned)


def partial_x(seed_cells, n_planned):
    """-> (X, n passing seeds) for the smallest X of one or two axes, or None."""
    for size in (1, 2):
        best = None
        for X in combinations(X_AXES, size):
            n = sum(1 for c in seed_cells if c is not None and set(failing(c)) <= set(X))
            if n * 2 > n_planned and (best is None or n > best[1]):
                best = (list(X), n)
        if best:
            return best
    return None


def lik_majority(seed_cells, n_planned, fams, pred):
    return majority([c is not None and pred([(c.get("LIK") or {}).get(f) for f in fams]) for c in seed_cells],
                    n_planned)


def sub_readings(seed_cells, n_planned, id_accs):
    """FAIL sub-readings (notes (c)); id_accs: the ID family's LIK per seed (diagnostic)."""
    out = []
    if lik_majority(seed_cells, n_planned, ["H1", "H2"] + CONTROLS, lambda xs: not all(ok(x) for x in xs)):
        out.append("LIK failure on H1, H2 or a control: the E002 shortcut signature (never partial)")
    id_hi = majority([ok(x) for x in id_accs], n_planned)
    h12_lo = lik_majority(seed_cells, n_planned, ["H1", "H2"], lambda xs: not all(ok(x) for x in xs))
    if id_hi and h12_lo:
        out.append("ID >= 0.8 but H1/H2 < 0.8: learned the training task by a rule that does not transfer")
    if majority([not ok(x) for x in id_accs], n_planned):
        out.append("ID < 0.8: did not learn even in distribution (recipe or capacity)")
    u_ok = lik_majority(seed_cells, n_planned, U_FAMS, lambda xs: all(ok(x) for x in xs))
    c_bad = lik_majority(seed_cells, n_planned, CONTROLS, lambda xs: not all(ok(x) for x in xs))
    if u_ok and c_bad:
        out.append("U families pass but controls fail: a recency rule")
    return out


def reading(seed_cells, n_planned, id_accs=(), model_id=""):
    """-> dict(label, detail, sub). seed_cells: one cells dict (or None for a missing seed) per seed run."""
    seed_cells = list(seed_cells)
    n_pass = sum(seed_pass(c) for c in seed_cells)
    base = {"n_planned": n_planned, "n_results": sum(c is not None for c in seed_cells), "n_seed_pass": n_pass}
    if majority([seed_pass(c) for c in seed_cells], n_planned):
        return dict(base, label="PASS", detail="a strict majority of seeds pass every cell", sub=[])
    lik_seeds = [c for c in seed_cells if lik_all(c)]
    if majority([True] * len(lik_seeds), n_planned):
        gen_fail = sorted({f for c in lik_seeds for f in PASS if not ok(c.get("GEN", {}).get(f))},
                          key=PASS.index)
        return dict(base, label="PARTIAL-G", detail="ranks the latest value but does not reliably say it; "
                    f"failing GEN families: {', '.join(gen_fail)}", gen_fail=gen_fail, sub=[])
    px = partial_x(seed_cells, n_planned)
    if px:
        X, n = px
        det = f"updating with a named limit: {', '.join(X)} ({n} seeds pass everything else)"
        if "TinyStories" in model_id and "H4" in X:
            det += "; TinyStories positional confound named (positions 512+ had no pretraining gradient, 768+ " \
                   "no fine-tuning gradient): not a size finding"
        return dict(base, label="PARTIAL-X", detail=det, limit=X, sub=[])
    return dict(base, label="FAIL", detail="no PASS, PARTIAL-G or PARTIAL-X condition holds",
                sub=sub_readings(seed_cells, n_planned, list(id_accs)))


def untouched_pass(cells):
    """an untouched model is one run: it passes iff that run passes every cell (then it needs no fine-tune)."""
    return seed_pass(cells)


def below_chance(acc, chance):
    return acc is not None and chance is not None and acc < chance


# ---------------- learning-rate picks ----------------
def dev_score(fam_accs):
    """fam_accs: {family: LIK acc on the dev draw} -> {min, mean} over the 9 pass families (missing = 0)."""
    xs = [fam_accs.get(f) if isinstance(fam_accs.get(f), (int, float)) else 0.0 for f in PASS]
    return {"min": round(min(xs), 4), "mean": round(sum(xs) / len(xs), 4)}


def pick_135m(rows):
    """rows: {lr string: {min, mean} or None (ineligible)} for 5e-05 and 1.5e-04 -> (decision, lr or reason)."""
    b, alt = rows.get(LR_135M_BASE), rows.get(LR_135M_ALT)
    if b is None:
        return "NONE", f"the {LR_135M_BASE} run is not eligible"
    if alt is not None and alt["min"] >= b["min"] + LR_135M_MIN_GAIN - 1e-9 and alt["mean"] >= b["mean"]:
        return "CHOSEN", LR_135M_ALT
    return "CHOSEN", LR_135M_BASE


def pick_grid(rows, grid, final=False):
    """rows: {lr string: {min, mean} or None}; grid: the searched LR strings (without an extension).
    -> (decision, lr): CHOSEN lr, EXTEND next lr, or NONE reason."""
    good = [(lr, r) for lr, r in rows.items() if r is not None]
    if not good:
        return "NONE", "no eligible LR-search run"
    lr, r = max(good, key=lambda t: (t[1]["min"], t[1]["mean"], -float(t[0])))
    top = max(grid, key=float)
    if not final and lr == top and r["min"] < THRESH:
        nxt = NEXT_LR.get(top)
        if nxt is None:
            nxt = f"{3 * float(top):.0e}"
        return "EXTEND", nxt
    return "CHOSEN", lr
