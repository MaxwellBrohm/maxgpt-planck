"""T001 results: reads the run JSONs of a finished queue and writes RESULTS.txt with every number the pre-registration
names (PREREG.txt METRICS, PICK RULES, amendment A1). The rules themselves are metrics.py's; this file only tabulates.

  python results.py --out DIR [--write FILE]      (prints the tables; --write also saves them to FILE)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
import runplan as R  # noqa: E402
from model import DIAG_ARMS, PICK_ARMS  # noqa: E402

ALL = PICK_ARMS + DIAG_ARMS


def f3(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def gm(xs) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def runs(out):
    st, lrs, tab = R.table(out)
    js = {(a, s): R.load(out, R.name(a, "mqrar", s, lrs[a], st)) for a in ALL for s in M.SEEDS}
    return st, lrs, tab, js


def lr_lines(out, st, lrs):
    rows = ["LEARNING RATE (seed 0, dev; score = mean over the grid of (acc + acc_ov) / 2; A1)"]
    for a in ALL:
        cells = []
        for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            js = R.load(out, R.name(a, "mqrar", 0, lr, st))
            if js is None:
                continue
            d = R.dev_acc(js)
            sc = sum(R._get(d, n, "acc") + R._get(d, n, "acc_ov") for n in st["grid"]) / (2 * len(st["grid"]))
            cells.append(f"{lr:g} {sc:.3f}{' D' if js.get('diverged') else ''}")
        rows.append(f"  {a:7s} pick {lrs[a]:g}   " + "; ".join(cells))
    return rows


def per_seed(tab, a, key, fmt="{:.3f}"):
    return " ".join(fmt.format(tab[a][s][key]) for s in M.SEEDS)


def arm_lines(tab):
    rows = ["PER ARM, verdict seeds 1-5 at the picked LR (test sets). cap and LWO are A1's rule metrics; cap_all and LW",
            "the registered ones, reported only. * = censored cap (a lower bound); D = diverged",
            "  arm      cap per seed          cap_all per seed      LWO mean | per seed             LW mean  AR32 mean"]
    for a in ALL:
        caps = " ".join(f"{tab[a][s]['cap']}{'*' if tab[a][s]['censored'] else ''}" for s in M.SEEDS)
        capa = " ".join(str(tab[a][s]["cap_all"]) for s in M.SEEDS)
        div = " D" if any(tab[a][s]["diverged"] for s in M.SEEDS) else ""
        rows.append(f"  {a:7s}  {caps:20s}  {capa:20s}  {mean([tab[a][s]['lwo'] for s in M.SEEDS]):.3f} | "
                    f"{per_seed(tab, a, 'lwo')}  {mean([tab[a][s]['lw'] for s in M.SEEDS]):.3f}    "
                    f"{mean([tab[a][s]['ar32'] for s in M.SEEDS]):.3f}{div}")
    return rows


def by_n_lines(js, grid):
    def cell(a, n, fam, key):
        return mean([j["eval"]["test"][fam][str(n)][key] for (b, s), j in js.items() if b == a and j["eval"]])
    rows = ["PER n, means over seeds 1-5 (test). Block 1: acc / acc_ov (outvoted). Block 2: LW (updated) / acc_tie.",
            "Block 3: plain recall (family ar) acc. A diverged run is left out of these means (it counts worst in rules)."]
    head = "  arm     " + "".join(f"n={n:<11d}" for n in grid)
    for pair in ((("mqrar", "acc"), ("mqrar", "acc_ov")), (("mqrar", "lw"), ("mqrar", "acc_tie")), (("ar", "acc"),)):
        rows.append(head)
        for a in ALL:
            cells = ["/".join(f3(cell(a, n, fam, key)) for fam, key in pair) for n in grid]
            rows.append(f"  {a:7s} " + "".join(f"{c:13s}" for c in cells))
    return rows


def vs_base(tab, x, y="base"):
    """Two lines: the paired LWO differences and capacity ratios, then the AR32 guard, against y."""
    d = M.diffs(tab, x, y, M.LW_KEY)
    r = M.ratios(tab, x, y)
    g = M.diffs(tab, x, y, "ar32")
    return (f"\n      LWO - {y}: mean {mean(d):+.3f} [{' '.join(f'{v:+.3f}' for v in d)}]; cap / {y}: gm {gm(r):.3f} "
            f"[{' '.join(f'{v:.2f}' for v in r)}]\n      AR32 - {y}: mean {mean(g):+.3f} "
            f"[{' '.join(f'{v:+.3f}' for v in g)}]")


def rule_lines(tab):
    rows = ["R1 (P-022 vs P-148)"]
    for x in ("gdn", "canon"):
        rows.append(f"  {x:6s} helps={M.helps(tab, x)}  {vs_base(tab, x)}")
    rows.append(f"  head to head, gdn vs canon:{vs_base(tab, 'gdn', 'canon')}")
    rows.append(f"  R1 PICK: {M.r1(tab)['pick']} ({M.r1(tab)['why']})")
    rows.append("R2 (the P-020 variant; candidates fg, sb, sa, alibi)")
    for x in M.R2_ORDER:
        lw_abs = sum(1 for s in M.SEEDS if tab[x][s][M.LW_KEY] >= M.PASS_LW_ABS - M.EPS)
        rows.append(f"  {x:6s} pass={M.passes(tab, x)} helps={M.helps(tab, x)} LWO>=0.8 on {lw_abs}/5  "
                    f"{vs_base(tab, x)}")
    r2 = M.r2(tab)
    rows.append(f"  R2 PICK: {r2['pick']} ({r2['why']})")
    rows.append("R3 (P-024, 1 vs 2 layers; AR32 per seed)")
    for a in ("base", "base1", "smear1", "canon1"):
        rows.append(f"  {a:7s} {per_seed(tab, a, 'ar32')}  mean {mean([tab[a][s]['ar32'] for s in M.SEEDS]):.3f}")
    gap = M.diffs(tab, "smear1", "base1", "ar32")
    rows.append(f"  smear1 - base1 per seed [{' '.join(f'{v:+.3f}' for v in gap)}], mean {mean(gap):+.3f}; "
                f"seeds with gap >= 0.2: {sum(1 for v in gap if v >= M.R3_GAP - M.EPS)}")
    rows.append(f"  R3 READING: {M.r3(tab)['reading']}")
    rows.append("DIAGNOSTIC ARMS (reported only; no rule)")
    for x in ("smear", "gdn_nc"):
        rows.append(f"  {x:6s} helps={M.helps(tab, x)}  {vs_base(tab, x)}")
    rows.append(f"  canon1 vs base1:{vs_base(tab, 'canon1', 'base1')}")
    return rows


def track_lines(out, st, lrs):
    rows = ["TRACKING (family track, seeds 1-3, test; enters no rule): track acc overall / on tracked queries, by n"]
    for a in R.TRACK_ARMS:
        accs, trk = {}, {}
        for s in R.TRACK_SEEDS:
            js = R.load(out, R.name(a, "track", s, lrs[a], st))
            if js is None or not js["eval"]:
                continue
            for n in st["grid"]:
                r = js["eval"]["test"]["track"][str(n)]
                accs.setdefault(n, []).append(r["acc"])
                trk.setdefault(n, []).append(r["trk_acc"])
        rows.append(f"  {a:6s} " + "  ".join(f"{n}:{f3(mean(accs.get(n, [])))}/{f3(mean(trk.get(n, [])))}"
                                             for n in st["grid"]))
    return rows


def last_loss(j):
    xs = [v for _, v in j["loss_curve"] if v is not None]
    return xs[-1] if xs else None


def pooled(j, num, den, ns=M.LW_NS):
    """A count over another, pooled over the n of LW (test, mqrar)."""
    t = j["eval"]["test"]["mqrar"]
    d = sum(t[str(n)][den] for n in ns)
    return sum(t[str(n)][num] for n in ns) / d if d else None


def report_lines(js):
    rows = ["REPORTED ONLY (PREREG METRICS; no rule). Means over seeds 1-5; a diverged run is left out of these means.",
            "  STALE / FIRST: wrong answers that name an earlier value of the key / its moved first value, as a share of",
            "  updated queries pooled over n 8-32 (test). loss: the last logged train loss. wall: s per run (train+eval).",
            "  arm      params   STALE   FIRST   loss    wall_s"]
    for a in ALL:
        runs = [j for (b, s), j in sorted(js.items()) if b == a]
        ok = [j for j in runs if j["eval"]]
        rows.append(f"  {a:7s} {runs[0]['params']:7d}   {f3(mean([pooled(j, 'stale', 'upd_q') for j in ok]))}   "
                    f"{f3(mean([pooled(j, 'first', 'upd_q') for j in ok]))}   {f3(mean([last_loss(j) for j in ok]))}   "
                    f"{mean([j['wall_s'] for j in runs]):.0f}")
    rows.append("  Accuracy by distance (registered as reported only): NOT RECORDED. train.py's score() keeps no")
    rows.append("  per-distance counts and no run saved its weights, so it cannot be recomputed from these runs.")
    return rows


def provenance(out, st):
    files = sorted(f for f in os.listdir(out) if f.endswith(".json"))
    def rd(f):
        with open(os.path.join(out, f)) as fh:
            return json.load(fh)
    js = [rd(f) for f in files]
    tot = sum(j["wall_s"] for j in js)
    div = [f[:-5] for f, j in zip(files, js) if j.get("diverged")]
    return [f"RUNS: {len(files)} JSON files; total train+eval wall {tot / 3600:.2f} h; diverged: "
            f"{', '.join(div) if div else 'none'}",
            f"C0: steps {st['steps']}, n_hi {st['n_hi']}, grid {st['grid']}; rule (a) fired: {st['a']}; "
            f"rule (b) fired: {st['b']}"]


def build(out):
    st, lrs, tab, js = runs(out)
    lines = provenance(out, st) + [""] + lr_lines(out, st, lrs) + [""] + arm_lines(tab) + [""]
    lines += by_n_lines(js, st["grid"]) + [""] + report_lines(js) + [""] + rule_lines(tab) + [""]
    lines += track_lines(out, st, lrs)
    picks = dict(R1=M.r1(tab)["pick"], R2=M.r2(tab)["pick"], R3=M.r3(tab)["reading"])
    return lines, picks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", default=None)
    a = ap.parse_args()
    lines, picks = build(a.out)
    text = "\n".join(lines + ["PICKS " + json.dumps(picks)]) + "\n"
    print(text, end="")
    if a.write:
        with open(a.write, "w") as f:
            f.write(text)
