"""T001 pick-rule tests: every threshold in PREREG.txt PICK RULES on synthetic per-seed tables."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M  # noqa: E402

G = [8, 16, 24, 32, 48, 64, 96]
ARMS = ("base", "gdn", "canon", "fg", "sb", "sa", "alibi", "smear", "gdn_nc", "base1", "smear1", "canon1")


def tab(**over):
    """Every arm equal to base (cap 32, LWO 0.5, AR32 0.9) unless overridden: over[arm] = dict of per-seed lists."""
    out = {a: {s: dict(cap=32, lwo=0.5, ar32=0.9) for s in M.SEEDS} for a in ARMS}
    for arm, cols in over.items():
        for key, vals in cols.items():
            for s, v in zip(M.SEEDS, vals):
                out[arm][s][key] = v
    return out


def test_capacity_contiguous_from_the_bottom():
    assert M.capacity({8: .95, 16: .91, 24: .89, 32: .99}, G) == 16
    assert M.capacity({8: .9, 16: .95}, G) == 16                   # 0.9 itself passes
    assert M.capacity({8: .89, 16: .99}, G) == M.CAP_FLOOR == 4
    assert M.capacity({n: 1.0 for n in G}, G) == 96
    assert M.capacity({8: 1.0, 16: None, 24: 1.0}, G) == 8


def test_run_metrics_and_divergence():
    js = dict(grid=G, diverged=False, eval={"test": {
        "mqrar": {str(n): dict(acc=1.0 if n <= 24 else 0.5, acc_ov=1.0 if n <= 16 else 0.5,
                               lw=0.25 * (n in (8, 16, 24, 32))) for n in G},
        "ar": {"32": dict(acc=0.75)}}})
    r = M.run_metrics(js)
    assert r["cap"] == 16 and r["cap_all"] == 24 and not r["censored"]          # A1: cap needs acc_ov >= 0.9 too
    assert r["lw"] == 0.25 and r["lwo"] == 0.75 and r["ar32"] == 0.75
    js["eval"]["test"]["mqrar"]["8"]["lw"] = None                   # an n without updated queries counts 0
    js["eval"]["test"]["mqrar"]["24"]["acc_ov"] = None              # an n without outvoted queries counts 0
    r = M.run_metrics(js)
    assert r["lw"] == pytest.approx(0.1875) and r["lwo"] == pytest.approx(0.625) and r["cap"] == 16
    assert M.run_metrics(dict(grid=G, diverged=True, eval=None)) == dict(cap=4, lwo=0.0, ar32=0.0, censored=False,
                                                                          cap_all=4, lw=0.0, diverged=True)


def test_censoring_label():
    js = dict(grid=G, diverged=False, eval={"test": {"mqrar": {str(n): dict(acc=1.0, acc_ov=1.0, lw=1.0) for n in G},
                                                     "ar": {"32": dict(acc=1.0)}}})
    r = M.run_metrics(js)
    assert r["cap"] == 96 and r["censored"]
    js["eval"]["test"]["mqrar"]["96"]["acc_ov"] = 0.89
    r = M.run_metrics(js)
    assert r["cap"] == 64 and not r["censored"] and r["cap_all"] == 96


UP5 = [0.55] * 5                 # +0.05 on every seed
UP4 = [0.56, 0.56, 0.56, 0.56, 0.45]


def test_helps_thresholds():
    assert M.helps(tab(gdn=dict(lwo=UP5)), "gdn")
    assert not M.helps(tab(gdn=dict(lwo=[0.549] * 5)), "gdn")                       # mean below 0.05
    assert M.helps(tab(gdn=dict(lwo=UP4)), "gdn") is (sum(UP4) / 5 - 0.5 >= 0.05)
    assert not M.helps(tab(gdn=dict(lwo=[0.6, 0.6, 0.6, 0.49, 0.49])), "gdn")        # only 3 of 5 positive
    assert M.helps(tab(gdn=dict(cap=[40] * 5)), "gdn")                             # ratio 1.25 exactly
    assert not M.helps(tab(gdn=dict(cap=[39] * 5)), "gdn")
    assert not M.helps(tab(gdn=dict(cap=[64, 64, 64, 32, 32])), "gdn")             # 3 of 5 above
    assert not M.helps(tab(gdn=dict(lwo=UP5, ar32=[0.84] * 5)), "gdn")              # guard: AR32 -0.06
    assert M.helps(tab(gdn=dict(lwo=UP5, ar32=[0.85] * 5)), "gdn")                  # guard edge -0.05


def test_r1_branches():
    assert M.r1(tab(gdn=dict(lwo=UP5)))["pick"] == "P-022"
    assert M.r1(tab(canon=dict(lwo=UP5)))["pick"] == "P-148"
    assert M.r1(tab())["pick"] == "P-148"                                          # neither helps
    both = dict(gdn=dict(lwo=[0.62] * 5), canon=dict(lwo=[0.56] * 5))
    assert M.r1(tab(**both))["pick"] == "P-022"                                    # head to head +0.06
    both = dict(gdn=dict(lwo=[0.56] * 5), canon=dict(lwo=[0.62] * 5))
    assert M.r1(tab(**both))["pick"] == "P-148"
    lwtie = dict(gdn=dict(lwo=[0.56] * 5, cap=[64] * 5), canon=dict(lwo=[0.57] * 5, cap=[48] * 5))
    assert M.r1(tab(**lwtie))["pick"] == "P-022"                                   # capacity 64/48 = 1.33
    lwtie = dict(gdn=dict(lwo=[0.56] * 5, cap=[48] * 5), canon=dict(lwo=[0.57] * 5, cap=[64] * 5))
    assert M.r1(tab(**lwtie))["pick"] == "P-148"
    alltie = dict(gdn=dict(lwo=[0.56] * 5), canon=dict(lwo=[0.57] * 5))
    assert M.r1(tab(**alltie))["pick"] == "P-022"                                  # untested-first tie default


def test_passes_thresholds():
    hi = dict(base=dict(lwo=[0.8] * 5))
    assert M.passes(tab(sb=dict(cap=[48] * 5, lwo=[0.8] * 5), **hi), "sb")      # ratio 1.5 exactly, LW 0.8, no gain
    assert not M.passes(tab(sb=dict(cap=[47] * 5, lwo=[0.8] * 5), **hi), "sb")
    assert not M.passes(tab(sb=dict(cap=[64] * 5, lwo=[0.79] * 5), base=dict(lwo=[0.79] * 5)), "sb")  # A1: LW < 0.8
    assert not M.passes(tab(sb=dict(cap=[64] * 5)), "sb")                          # A1: capacity alone is not enough
    assert M.passes(tab(fg=dict(lwo=[0.8] * 5), base=dict(lwo=[0.7] * 5)), "fg")     # LW 0.8 and +0.10
    assert not M.passes(tab(fg=dict(lwo=[0.79] * 5), base=dict(lwo=[0.6] * 5)), "fg")   # below 0.8
    assert not M.passes(tab(fg=dict(lwo=[0.8] * 5), base=dict(lwo=[0.71] * 5)), "fg")   # gain 0.09
    assert not M.passes(tab(fg=dict(lwo=[0.9, 0.9, 0.9, 0.79, 0.79]), base=dict(lwo=[0.5] * 5)), "fg")  # 3/5 >= 0.8
    assert not M.passes(tab(sb=dict(cap=[48] * 5, lwo=[0.8] * 5, ar32=[0.8] * 5), **hi), "sb")          # guard


def test_r2_choice():
    assert M.r2(tab(sb=dict(cap=[64] * 5, lwo=[0.7] * 5), fg=dict(cap=[48] * 5, lwo=[0.6] * 5)))["pick"] == "sb"
    t2 = tab(sb=dict(cap=[64] * 5, lwo=[0.61] * 5), sa=dict(cap=[48] * 5, lwo=[0.6] * 5))
    assert M.r2(t2)["pick"] == "sb"                                                # LW within 0.02: capacity
    t3 = tab(sb=dict(cap=[48] * 5, lwo=[0.61] * 5), sa=dict(cap=[48] * 5, lwo=[0.6] * 5))
    assert M.r2(t3)["pick"] == "sa"                                                # then the order fg, alibi, sa, sb
    t4 = tab(alibi=dict(lwo=[0.56] * 5), sb=dict(lwo=[0.59] * 5))                    # none passes, both help
    assert M.r2(t4)["pick"] == "sb" and M.r2(t4)["why"].startswith("none passes")
    t5 = tab(alibi=dict(lwo=[0.57] * 5), sb=dict(lwo=[0.58] * 5))
    assert M.r2(t5)["pick"] == "alibi"
    assert M.r2(tab())["pick"] == "fg"
    assert M.r2(tab(alibi=dict(cap=[64] * 5)), cands=("fg", "sb", "sa", "alibi"))["pick"] == "alibi"


def test_r3_readings():
    base = dict(base=dict(ar32=[0.9] * 5), base1=dict(ar32=[0.1] * 5))
    assert M.r3(tab(smear1=dict(ar32=[0.85] * 5), **base))["reading"] == "YES"
    assert M.r3(tab(smear1=dict(ar32=[0.84] * 5), **base))["reading"] == "PARTIAL"   # below base - 0.05
    assert M.r3(tab(smear1=dict(ar32=[0.15] * 5), **base))["reading"] == "NO"
    assert M.r3(tab(smear1=dict(ar32=[0.9, 0.9, 0.9, 0.29, 0.29]), **base))["reading"] == "PARTIAL"  # 3/5 gaps


def dv(acc, ov):
    return dict(acc=acc, acc_ov=ov)


def test_lr_pick():
    assert M.lr_pick({3e-4: {8: dv(.5, .5)}, 1e-3: {8: dv(.7, .7)}, 3e-3: {8: dv(.6, .6)}}, [8]) == 1e-3
    assert M.lr_pick({3e-4: {8: dv(.7, .7)}, 1e-3: {8: dv(.7, .7)}, 3e-3: {8: dv(.6, .6)}}, [8]) == 3e-4  # tie: smaller
    assert M.lr_pick({3e-4: {8: dv(.4, .4), 16: None}, 1e-3: {}, 3e-3: {8: dv(.2, .2), 16: dv(.5, None)}},
                     [8, 16]) == 3e-3
    # A1: (acc + acc_ov) / 2, so a voting LR (acc .92, outvoted 0) loses to one that tracks updates
    assert M.lr_pick({3e-4: {8: dv(.92, 0.0)}, 1e-3: {8: dv(.8, .5)}}, [8]) == 1e-3
    assert M.lr_pick({3e-4: {8: dv(.9, .4)}, 1e-3: {8: dv(.8, .5)}}, [8]) == 3e-4         # 0.65 each: the smaller LR


def test_r3_gap_threshold():
    t = tab(base=dict(ar32=[0.9] * 5), base1=dict(ar32=[0.7] * 5), smear1=dict(ar32=[0.95, 0.95, 0.95, 0.85, 0.85]))
    assert M.r3(t)["reading"] == "PARTIAL"                      # gaps 0.25 x3, 0.15 x2: 3 of 5 reach 0.2
    t = tab(base=dict(ar32=[0.9] * 5), base1=dict(ar32=[0.7] * 5), smear1=dict(ar32=[0.95, 0.95, 0.95, 0.9, 0.85]))
    assert M.r3(t)["reading"] == "YES"                          # 4 of 5 reach 0.2


def test_r2_lw_tie_window_is_0p02():
    """fg and sb pass on LW; sb has the higher capacity ratio. Within 0.02 of the top LW, capacity decides."""
    t2 = tab(base=dict(lwo=[0.5] * 5), fg=dict(lwo=[0.9] * 5), sb=dict(lwo=[0.88] * 5, cap=[48] * 5))
    assert M.r2(t2)["pick"] == "sb"                  # fg leads by 0.02: tie, capacity picks sb
    t3 = tab(base=dict(lwo=[0.5] * 5), fg=dict(lwo=[0.9] * 5), sb=dict(lwo=[0.875] * 5, cap=[48] * 5))
    assert M.r2(t3)["pick"] == "fg"                  # fg leads by 0.025: no tie, fg on LW
    t4 = tab(base=dict(lwo=[0.5] * 5), fg=dict(lwo=[0.9] * 5), sb=dict(lwo=[0.885] * 5, cap=[48] * 5))
    assert M.r2(t4)["pick"] == "sb"                  # fg leads by 0.015: tie


def test_r2_cap_tie_window_is_5_percent():
    """fg and sb pass with equal LW; base cap 20. Capacity ratios within 5% fall to the order (fg first)."""
    def t(cap_sb, cap_fg=20.0):
        return tab(base=dict(lwo=[0.5] * 5, cap=[20.0] * 5), fg=dict(lwo=[0.9] * 5, cap=[cap_fg] * 5),
                   sb=dict(lwo=[0.9] * 5, cap=[cap_sb] * 5))
    assert M.r2(t(21.0))["pick"] == "fg"             # 1.05 exactly: tie, order
    assert M.r2(t(21.0, 20.2))["pick"] == "fg"       # 1.0396: tie
    assert M.r2(t(20.8))["pick"] == "fg"             # 1.04: tie
    assert M.r2(t(21.2))["pick"] == "sb"             # 1.06: no tie, sb's capacity
    assert M.r2(t(22.0))["pick"] == "sb"             # 1.10
