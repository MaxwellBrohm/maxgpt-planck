"""T001 metrics and pick rules R1-R3, exactly as PREREG.txt METRICS and PICK RULES fix them, as amendment A1 changed
them: the rules read LWO (latest-wins on outvoted queries) and the vote-proof capacity; the registered LW and the
accuracy-only capacity are reported beside them. Pure functions on the per-run JSON written by train.py; no torch."""
from __future__ import annotations

import math

THR_ACC = 0.9                       # capacity threshold
CAP_FLOOR = 4                       # cap when acc(8) < 0.9
LW_NS = (8, 16, 24, 32)             # LW pooled over these n, equal weight
SEEDS = (1, 2, 3, 4, 5)
MIN_SIGN = 4                        # "on >= 4/5"
HELP_LW, HELP_CAP, GUARD = 0.05, 1.25, -0.05
PASS_CAP, PASS_LW_ABS, PASS_LW_GAIN = 1.5, 0.8, 0.10
TIE_LW, TIE_CAP = 0.02, 1.05
R2_ORDER = ("fg", "alibi", "sa", "sb")
R3_GAP, R3_NO, R3_MARGIN = 0.2, 0.05, 0.05
EPS = 1e-9                          # float noise only: a mean of exactly 0.05 is 0.05
LW_KEY = "lwo"                      # A1: the rules' latest-wins is the outvoted-query rate


def capacity(acc_by_n: dict, grid, acc_ov_by_n: dict | None = None) -> int:
    """Largest n in grid with acc >= THR_ACC at every grid n up to it (CAP_FLOOR if none). With acc_ov_by_n (A1), an
    n also needs its outvoted-query accuracy >= THR_ACC (None, no outvoted query, fails)."""
    cap = CAP_FLOOR
    for n in sorted(grid):
        a = acc_by_n.get(n)
        if a is None or a < THR_ACC:
            break
        if acc_ov_by_n is not None:
            o = acc_ov_by_n.get(n)
            if o is None or o < THR_ACC:
                break
        cap = n
    return cap


def run_metrics(js: dict) -> dict:
    """From one verdict run's JSON: the rule metrics cap (vote-proof), lwo, ar32 and cap's censoring; the registered
    cap_all (accuracy only) and lw (updated queries), reported only. A diverged run takes the worst values."""
    grid = js["grid"]
    if js.get("diverged") or not js.get("eval"):
        return dict(cap=CAP_FLOOR, lwo=0.0, ar32=0.0, censored=False, cap_all=CAP_FLOOR, lw=0.0, diverged=True)
    t = js["eval"]["test"]
    acc = {int(n): r["acc"] for n, r in t["mqrar"].items()}
    acc_ov = {int(n): r["acc_ov"] for n, r in t["mqrar"].items()}
    lw = sum(t["mqrar"][str(n)]["lw"] or 0.0 for n in LW_NS) / len(LW_NS)      # an n without such queries counts 0
    lwo = sum(t["mqrar"][str(n)]["acc_ov"] or 0.0 for n in LW_NS) / len(LW_NS)
    ar32 = t["ar"]["32"]["acc"] if "ar" in t else None
    cap = capacity(acc, grid, acc_ov)
    return dict(cap=cap, lwo=lwo, ar32=ar32, censored=cap == max(grid), cap_all=capacity(acc, grid), lw=lw,
                diverged=False)


def _count(xs, positive: bool) -> int:
    return sum(1 for x in xs if (x > 0 if positive else x < 0))


def _gm(rs) -> float:
    return math.exp(sum(math.log(r) for r in rs) / len(rs))


def diffs(tab: dict, a: str, b: str, key: str):
    return [tab[a][s][key] - tab[b][s][key] for s in SEEDS]


def ratios(tab: dict, a: str, b: str):
    return [tab[a][s]["cap"] / tab[b][s]["cap"] for s in SEEDS]


def guard(tab: dict, x: str) -> bool:
    d = diffs(tab, x, "base", "ar32")
    return sum(d) / len(d) >= GUARD - EPS


def lw_gain(tab: dict, x: str, y: str, thr: float) -> int:
    """+1 if x beats y on LW (mean >= thr, >= 4/5 positive), -1 if y beats x the same way, else 0."""
    d = diffs(tab, x, y, LW_KEY)
    m = sum(d) / len(d)
    if m >= thr - EPS and _count(d, True) >= MIN_SIGN:
        return 1
    if m <= -thr + EPS and _count(d, False) >= MIN_SIGN:
        return -1
    return 0


def cap_gain(tab: dict, x: str, y: str, thr: float) -> int:
    r = ratios(tab, x, y)
    g = _gm(r)
    if g >= thr - EPS and _count([v - 1 for v in r], True) >= MIN_SIGN:
        return 1
    if g <= 1 / thr + EPS and _count([v - 1 for v in r], False) >= MIN_SIGN:
        return -1
    return 0


def helps(tab: dict, x: str) -> bool:
    return (lw_gain(tab, x, "base", HELP_LW) == 1 or cap_gain(tab, x, "base", HELP_CAP) == 1) and guard(tab, x)


def r1(tab: dict) -> dict:
    hg, hc = helps(tab, "gdn"), helps(tab, "canon")
    if hg and not hc:
        return dict(pick="P-022", why="gdn helps, canon does not")
    if hc and not hg:
        return dict(pick="P-148", why="canon helps, gdn does not")
    if not hg and not hc:
        return dict(pick="P-148", why="neither helps: the toy measured P-022's own claim and found nothing")
    w = lw_gain(tab, "gdn", "canon", HELP_LW)
    if w:
        return dict(pick="P-022" if w > 0 else "P-148", why="both help; head to head on LW")
    w = cap_gain(tab, "gdn", "canon", HELP_CAP)
    if w:
        return dict(pick="P-022" if w > 0 else "P-148", why="both help; LW tied; head to head on capacity")
    return dict(pick="P-022", why="both help and tie: the bet untested at this size first (P-148 has its 5M screen)")


def passes(tab: dict, x: str) -> bool:
    """A1: LW_X >= 0.8 on >= 4/5 seeds is required on both paths (LEDGER P-020 lists capacity AND latest-wins)."""
    lw_abs = sum(1 for s in SEEDS if tab[x][s][LW_KEY] >= PASS_LW_ABS - EPS) >= MIN_SIGN
    by_cap = lw_abs and cap_gain(tab, x, "base", PASS_CAP) == 1
    by_lw = lw_abs and lw_gain(tab, x, "base", PASS_LW_GAIN) == 1
    return (by_cap or by_lw) and guard(tab, x)


def _best(tab: dict, cands) -> str:
    def mlw(x):
        return sum(tab[x][s][LW_KEY] for s in SEEDS) / len(SEEDS)
    order = sorted(cands, key=lambda x: (-mlw(x), R2_ORDER.index(x) if x in R2_ORDER else 99))
    top = [x for x in order if mlw(order[0]) - mlw(x) <= TIE_LW + EPS]
    if len(top) == 1:
        return top[0]
    gms = {x: _gm(ratios(tab, x, "base")) for x in top}
    best = max(gms.values())
    top = [x for x in top if gms[x] * TIE_CAP >= best - EPS]
    return min(top, key=lambda x: R2_ORDER.index(x) if x in R2_ORDER else 99)


def r2(tab: dict, cands=R2_ORDER) -> dict:
    cands = [c for c in cands if c in tab]
    p = [c for c in cands if passes(tab, c)]
    if p:
        return dict(pick=_best(tab, p), why="passing variants: " + ",".join(p))
    h = [c for c in cands if helps(tab, c)]
    if h:
        return dict(pick=_best(tab, h), why="none passes; helping variants: " + ",".join(h))
    return dict(pick="fg", why="no variant helps: S005's variant by default")


def r3(tab: dict) -> dict:
    gap = diffs(tab, "smear1", "base1", "ar32")
    m_s1 = sum(tab["smear1"][s]["ar32"] for s in SEEDS) / len(SEEDS)
    m_b2 = sum(tab["base"][s]["ar32"] for s in SEEDS) / len(SEEDS)
    if m_s1 >= m_b2 - R3_MARGIN - EPS and sum(1 for g in gap if g >= R3_GAP - EPS) >= MIN_SIGN:
        return dict(reading="YES")
    if sum(gap) / len(gap) <= R3_NO + EPS:
        return dict(reading="NO")
    return dict(reading="PARTIAL")


def lr_pick(dev_by_lr: dict, grid) -> float:
    """Seed-0 LR pick (A1): the highest mean over the grid of (dev acc + dev outvoted acc) / 2; ties to the smaller
    LR. dev_by_lr maps an LR to {n: {"acc": a, "acc_ov": o}} (None or a missing n counts 0; a diverged run maps to
    {})."""
    def score(lr):
        d = dev_by_lr[lr]
        return sum(((d.get(n) or {}).get("acc") or 0.0) + ((d.get(n) or {}).get("acc_ov") or 0.0)
                   for n in grid) / (2 * len(grid))
    return min(dev_by_lr, key=lambda lr: (-score(lr), lr))
