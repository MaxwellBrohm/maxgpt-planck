"""T001 re-derivation: an independent reading of the raw run JSONs, written from PREREG.txt's text (with amendment
A1), importing nothing from this module. It indexes runs by the fields inside each JSON (not by file name), recomputes
every rate from the raw counts (ok / q, ov_ok / ov_q, ...), redoes C0, the LR picks and R1-R3, and prints one JSON.

  python rederive.py OUT_DIR
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

TOL = 1e-9
G0 = [8, 16, 24, 32, 48, 64, 96]
PICK = ["base", "gdn", "canon", "fg", "sb", "sa", "alibi"]
DIAG = ["smear", "gdn_nc", "base1", "smear1", "canon1"]
SEEDS = [1, 2, 3, 4, 5]


def rate(ok, q):
    return ok / q if q else None


def index(out):
    runs = {}
    for p in glob.glob(os.path.join(out, "*.json")):
        with open(p) as f:
            j = json.load(f)
        k = (j["arm"], j["family"], j["seed"], round(j["lr"], 12), j["steps"], j["n_hi"])
        assert k not in runs, f"two runs for {k}"
        runs[k] = j
    return runs


def dev_rates(j):
    """{n: (acc, acc_ov)} from raw dev counts; {} for a diverged run."""
    if j["diverged"] or not j["eval"]:
        return {}
    return {int(n): (rate(r["ok"], r["q"]), rate(r["ov_ok"], r["ov_q"])) for n, r in j["eval"]["dev"]["mqrar"].items()}


def c0(runs):
    steps, nhi, grid, fa, fb = 6000, 96, list(G0), False, False
    while True:
        d = dev_rates(runs[("base", "mqrar", 0, 1e-3, steps, nhi)])
        if not fa and (d.get(8, (0, 0))[0] or 0) < 0.5:
            steps, fa = 12000, True
            continue
        if not fb and all((d.get(n, (0, 0))[0] or 0) >= 0.9 and (d.get(n, (0, 0))[1] or 0) >= 0.9 for n in G0):
            nhi, grid, fb = 160, list(G0) + [128, 160], True
            continue
        return steps, nhi, grid, fa, fb


def lr_pick(runs, arm, steps, nhi, grid):
    def score(lr):
        d = dev_rates(runs[(arm, "mqrar", 0, lr, steps, nhi)])
        return sum((d.get(n, (0, 0))[0] or 0) + (d.get(n, (0, 0))[1] or 0) for n in grid) / (2 * len(grid))

    def best(lrs):
        top = max(score(lr) for lr in lrs)
        return min(lr for lr in lrs if score(lr) == top)
    first = best([3e-4, 1e-3, 3e-3])
    ext = {3e-4: 1e-4, 3e-3: 1e-2}.get(first)
    return best([3e-4, 1e-3, 3e-3] + ([ext] if ext else [])), {lr: score(lr) for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
                                                                 if (arm, "mqrar", 0, lr, steps, nhi) in runs}


def metrics(j, grid):
    if j["diverged"] or not j["eval"]:
        return dict(cap=4, lwo=0.0, ar32=0.0, cap_all=4, lw=0.0, censored=False, div=True)
    t = j["eval"]["test"]
    acc = {n: rate(t["mqrar"][str(n)]["ok"], t["mqrar"][str(n)]["q"]) for n in grid}
    ov = {n: rate(t["mqrar"][str(n)]["ov_ok"], t["mqrar"][str(n)]["ov_q"]) for n in grid}

    def cap(need_ov):
        c = 4
        for n in grid:
            if acc[n] is None or acc[n] < 0.9 or (need_ov and (ov[n] is None or ov[n] < 0.9)):
                break
            c = n
        return c
    lwo = sum(ov[n] or 0.0 for n in (8, 16, 24, 32)) / 4
    lw = sum(rate(t["mqrar"][str(n)]["upd_ok"], t["mqrar"][str(n)]["upd_q"]) or 0.0 for n in (8, 16, 24, 32)) / 4
    ar32 = rate(t["ar"]["32"]["ok"], t["ar"]["32"]["q"])
    c = cap(True)
    return dict(cap=c, lwo=lwo, ar32=ar32, cap_all=cap(False), lw=lw, censored=c == grid[-1], div=False)


def beats(t, x, y, key, thr):
    d = [t[x][s][key] - t[y][s][key] for s in SEEDS]
    m = sum(d) / 5
    if m >= thr - TOL and sum(v > 0 for v in d) >= 4:
        return 1
    if m <= -thr + TOL and sum(v < 0 for v in d) >= 4:
        return -1
    return 0


def cap_beats(t, x, y, thr):
    r = [t[x][s]["cap"] / t[y][s]["cap"] for s in SEEDS]
    g = math.exp(sum(math.log(v) for v in r) / 5)
    if g >= thr - TOL and sum(v > 1 for v in r) >= 4:
        return 1
    if g <= 1 / thr + TOL and sum(v < 1 for v in r) >= 4:
        return -1
    return 0


def guard(t, x):
    return sum(t[x][s]["ar32"] - t["base"][s]["ar32"] for s in SEEDS) / 5 >= -0.05 - TOL


def helps(t, x):
    return (beats(t, x, "base", "lwo", 0.05) == 1 or cap_beats(t, x, "base", 1.25) == 1) and guard(t, x)


def passes(t, x):
    hi = sum(t[x][s]["lwo"] >= 0.8 - TOL for s in SEEDS) >= 4
    return hi and (cap_beats(t, x, "base", 1.5) == 1 or beats(t, x, "base", "lwo", 0.10) == 1) and guard(t, x)


def choose(t, cands):
    order = ["fg", "alibi", "sa", "sb"]
    m = {x: sum(t[x][s]["lwo"] for s in SEEDS) / 5 for x in cands}
    top = [x for x in cands if max(m.values()) - m[x] <= 0.02 + TOL]
    if len(top) > 1:
        g = {x: math.exp(sum(math.log(t[x][s]["cap"] / t["base"][s]["cap"]) for s in SEEDS) / 5) for x in top}
        top = [x for x in top if g[x] * 1.05 >= max(g.values()) - TOL]
    return sorted(top, key=order.index)[0]


def main(out):
    runs = index(out)
    steps, nhi, grid, fa, fb = c0(runs)
    lrs, scores = {}, {}
    for a in PICK + DIAG:
        lrs[a], scores[a] = lr_pick(runs, a, steps, nhi, grid)
    t = {a: {s: metrics(runs[(a, "mqrar", s, lrs[a], steps, nhi)], grid) for s in SEEDS} for a in PICK + DIAG}
    hg, hc = helps(t, "gdn"), helps(t, "canon")
    if hg != hc:
        r1 = "P-022" if hg else "P-148"
    elif not hg:
        r1 = "P-148"
    else:
        w = beats(t, "gdn", "canon", "lwo", 0.05) or cap_beats(t, "gdn", "canon", 1.25)
        r1 = "P-148" if w < 0 else "P-022"
    c = ["fg", "sb", "sa", "alibi"]
    p = [x for x in c if passes(t, x)]
    h = [x for x in c if helps(t, x)]
    r2 = choose(t, p) if p else choose(t, h) if h else "fg"
    gap = [t["smear1"][s]["ar32"] - t["base1"][s]["ar32"] for s in SEEDS]
    m1 = sum(t["smear1"][s]["ar32"] for s in SEEDS) / 5
    m2 = sum(t["base"][s]["ar32"] for s in SEEDS) / 5
    if m1 >= m2 - 0.05 - TOL and sum(g >= 0.2 - TOL for g in gap) >= 4:
        r3 = "YES"
    elif sum(gap) / 5 <= 0.05 + TOL:
        r3 = "NO"
    else:
        r3 = "PARTIAL"
    trk = {}
    for a in ("base", "gdn", "canon"):
        for s in (1, 2, 3):
            j = runs.get((a, "track", s, lrs[a], steps, nhi))
            trk[f"{a}_s{s}"] = None if j is None or not j["eval"] else {
                n: rate(j["eval"]["test"]["track"][str(n)]["trk_ok"], j["eval"]["test"]["track"][str(n)]["trk_q"])
                for n in grid}
    return dict(c0=dict(steps=steps, n_hi=nhi, grid=grid, a=fa, b=fb), lr=lrs, lr_scores=scores, R1=r1, R2=r2, R3=r3,
                helps={x: helps(t, x) for x in PICK[1:] + ["smear", "gdn_nc"]}, passes={x: passes(t, x) for x in c},
                table=t, track=trk, runs=len(runs))


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1, sort_keys=True, default=str))
