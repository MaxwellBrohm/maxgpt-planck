"""Monte Carlo check of SCREENS2's SEQUENTIAL RULE (screens2_seq.py) under C4's noise models, F(CHAT). Mac CPU, numpy,
no model, no batch 2 value (none exists). Writes screens2_seqsim.json beside this file and prints its tables.

  python experiments/screens/screens2_seqsim.py [--reps N] [--seed S] [--out PATH]

Model, per contrast and replicate (every number relative to BASE's mean F(CHAT), in %; delta = 1%):
  d_s ~ N(mu, sigma) for 10 seeds (101-110); the reference SD is a random estimate, SD_ref = m x r x sqrt(chi2_df/df),
  drawn once and used at both looks (C7 reads E3's one committed number at every look); the new-init reference that
  B4 (3) re-reads at is a second, independent draw (sqrt(2) sigma_init, df 4), except in the new-init classes, where it
  is the class reference itself (analyze2_lib.read_arm).
  classes (C4, B4 (2)): same init r = SD_d, m 1, df 7; SIA r = SD_d, m = u (1.353), df 7, true sigma SD_d (C4's
  loosening premise) and, as "SIA at u", true sigma u x SD_d; new init r = sqrt(2) sigma_init, m 1, df 4;
  "x1.5" rows: true sigma 1.5 x the class's reference (a mis-specified class; B4 (3) is what answers it).
rules: "bare" = the boundaries alone (continue unless T1 > c1; no sign clause, no EQUAL stop, no B4 (3)): the alpha
  spending itself, which must give 0.025 for BETTER at mu 0 and for EQUAL's upper side at mu = +delta; "C7 k5" =
  the draft's single look (t(0.975, df), INCONCLUSIVE final); "SEQ" =
  the registered rule (sign clauses, EQUAL, B4 (3)'s own check and re-read at each look; PROSE and the pooled SIA check
  are not simulated: both can only remove a CHAT crossing or raise an SD).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_lib as A  # noqa: E402
import screens2_seq as Q  # noqa: E402

E3 = json.load(open(os.path.join(A.E3D, "results.json")))
P3 = json.load(open(os.path.join(A.E3D, "results_part3.json")))
SDD = 100 * E3["noise_5M_250M"]["F(CHAT)"]["SD_d_rel"]
INIT = 100 * math.sqrt(2) * P3["families"]["F(CHAT)"]["INIT"]["rel"]
U = E3["quantiles"]["ub80_df7"]
CLASSES = {   # name: (true sigma, reference population sigma, multiplier, df)
    "same init": (SDD, SDD, 1.0, 7), "SIA": (SDD, SDD, U, 7), "SIA at u": (U * SDD, SDD, U, 7),
    "new init": (INIT, INIT, 1.0, 4),
    "same init x1.5": (1.5 * SDD, SDD, 1.0, 7), "SIA x1.5": (1.5 * U * SDD, SDD, U, 7),
    "new init x1.5": (1.5 * INIT, INIT, 1.0, 4)}
MUS = [0.0, -0.25, -0.4, -0.5, -0.6, -0.75, -1.0, -1.25, -1.5]
DESIGNS = {"Pocock-type (registered)": Q.spend(0.5), "O'Brien-Fleming-type": None, "linear": Q.ALPHA * 0.5,
           "fixed 90/10": 0.9 * Q.ALPHA}


def obf_a1() -> float:
    z = Q.NormalDist()
    return 2 * (1 - z.cdf(z.inv_cdf(1 - Q.ALPHA / 2) / math.sqrt(0.5)))


@lru_cache(maxsize=None)
def bounds_a1(a1: float, df: int) -> tuple[float, float]:
    """(c1, c2) for an arbitrary look-1 spend (the design table); the registered one is Q.bounds."""
    c1, lo, hi = Q._q(1 - a1, df), 0.0, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if Q.cont_cross(c1, mid, math.sqrt(0.5), df) > Q.ALPHA - a1 else (lo, mid)
    return c1, (lo + hi) / 2


def draw(rng, cls: str, mu: float, n: int) -> dict:
    s, r, m, df = CLASSES[cls]
    sd = m * r * np.sqrt(rng.chisquare(df, n) / df)
    init = INIT * np.sqrt(rng.chisquare(4, n) / 4)
    return {"d": rng.normal(mu, s, (n, 10)), "sd": sd, "init": sd if r == INIT else init, "df": df}


def look(x: dict, k: int, cfun, need: int, checks: bool) -> tuple:
    """One look on the first k seeds: labels B/W/E/U per replicate, with B4 (3) when checks."""
    d = x["d"][:, :k]
    sd, df = x["sd"].copy(), np.full(len(d), x["df"])
    if checks:
        obs = d.std(axis=1, ddof=1)
        fail = obs > x["sd"] * A.sd_bound(k - 1)
        use_obs = fail & (obs > x["init"])
        sd = np.where(fail, np.where(use_obs, obs, x["init"]), sd)
        df = np.where(fail, np.where(use_obs, k - 1, 4), df)
    c = np.empty(len(d))
    for v in np.unique(df):
        c[df == v] = cfun(int(v))
    m, thr = d.mean(axis=1), c * sd / math.sqrt(k)
    b = (-m > thr) & ((d < 0).sum(axis=1) >= need)
    w = (m > thr) & ((d > 0).sum(axis=1) >= need)
    e = (np.abs(m) + thr < 1.0) & ~b & ~w
    return b, w, e, ~(b | w | e)


def rule(x: dict, kind: str, a1: float | None = None) -> dict:
    if kind == "C7 k5":
        b, w, e, u = look(x, 5, lambda df: A.t_ppf(0.975, int(df)), 5, True)
        return {"B": b, "W": w, "E": e, "I": u, "ext": np.zeros_like(u)}
    bf = (lambda df: Q.bounds(int(df), 5, 10)) if a1 is None else (lambda df: dict(zip(("c1", "c2"), bounds_a1(a1, int(df)))))
    if kind == "bare":
        d, sd = x["d"], x["sd"]
        c1, c2 = bf(x["df"])["c1"], bf(x["df"])["c2"]
        t1, t2 = -d[:, :5].mean(1) / (sd / math.sqrt(5)), -d.mean(1) / (sd / math.sqrt(10))
        e1, e2 = (1.0 - d[:, :5].mean(1)) / (sd / math.sqrt(5)) > c1, (1.0 - d.mean(1)) / (sd / math.sqrt(10)) > c2
        return {"B": (t1 > c1) | ((t1 <= c1) & (t2 > c2)), "B1": t1 > c1, "E": e1 | (~e1 & e2), "E1": e1}
    b1, w1, e1, u1 = look(x, 5, lambda df: bf(df)["c1"], 5, True)
    b2, w2, e2, u2 = look(x, 10, lambda df: bf(df)["c2"], Q.sign_min(10), True)
    return {"B": b1 | (u1 & b2), "W": w1 | (u1 & w2), "E": e1 | (u1 & e2), "I": u1 & u2, "ext": u1, "B1": b1, "E1": e1}


def rates(rng, cls: str, mu: float, kind: str, reps: int, a1: float | None = None) -> dict:
    tot: dict = {}
    for i in range(0, reps, 200_000):
        r = rule(draw(rng, cls, mu, min(200_000, reps - i)), kind, a1)
        for k, v in r.items():
            tot[k] = tot.get(k, 0) + int(v.sum())
    return {k: v / reps for k, v in tot.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", default=os.path.join(HERE, "screens2_seqsim.json"))
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = {"reps": a.reps, "seed": a.seed, "sd_pct": {"SD_d": SDD, "u": U, "new_init": INIT},
           "bounds": {str(df): Q.bounds(df) for df in (4, 7, 9)}, "sign_min_10": Q.sign_min(10), "bare": {}, "grid": {},
           "designs": {}}
    print(f"SD_d {SDD:.4f}%  u {U:.4f}  new init {INIT:.4f}%  reps {a.reps}  seed {a.seed}")
    for df in (4, 7, 9):
        b = Q.bounds(df)
        print(f"df {df}: a1 {b['a1']:.5f} c1 {b['c1']:.4f} c2 {b['c2']:.4f} (single look t(0.975) {A.t_ppf(0.975, df):.4f})")
    for cls in ("same init", "SIA at u", "new init"):
        r, e = rates(rng, cls, 0.0, "bare", a.reps), rates(rng, cls, 1.0, "bare", a.reps)
        out["bare"][cls] = {"mu 0": r, "mu +delta": e}
        print(f"bare  {cls:15s} P(BETTER | mu 0) {r['B']:.5f} (look 1 {r['B1']:.5f}); P(EQUAL | mu +1%) {e['E']:.5f} "
              f"(look 1 {e['E1']:.5f}); targets 0.025 (look 1 0.0155)")
    for cls in CLASSES:
        out["grid"][cls] = {}
        for mu in MUS:
            row = {k: rates(rng, cls, mu, k, a.reps) for k in ("C7 k5", "SEQ")}
            out["grid"][cls][str(mu)] = row
            s, c = row["SEQ"], row["C7 k5"]
            print(f"{cls:15s} mu {mu:+.2f}  C7 k5 B {c['B']:.4f} W {c['W']:.4f} E {c['E']:.4f} I {c['I']:.4f} | "
                  f"SEQ B {s['B']:.4f} W {s['W']:.4f} E {s['E']:.4f} I {s['I']:.4f} ext {s['ext']:.4f}")
    for name, a1 in DESIGNS.items():
        a1 = obf_a1() if a1 is None else a1
        out["designs"][name] = {"a1": a1}
        for cls in ("SIA", "new init", "new init x1.5"):
            c1, c2 = bounds_a1(a1, CLASSES[cls][3])
            rows = {str(mu): rates(rng, cls, mu, "SEQ", a.reps // 4, a1) for mu in (0.0, -0.25, -0.5, -0.75, -1.0)}
            out["designs"][name][cls] = {"c1": c1, "c2": c2, "rows": rows}
            print(f"design {name:26s} {cls:14s} a1 {a1:.5f} c1 {c1:.3f} c2 {c2:.3f} " + " ".join(
                f"mu{mu}:B{v['B']:.3f}/E{v['E']:.3f}/I{v['I']:.4f}/x{v['ext']:.3f}" for mu, v in rows.items()))
    with open(a.out, "w") as f:
        json.dump(out, f, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
