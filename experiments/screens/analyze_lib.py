"""SCREENS decision rules as pure functions (no file is read here; analyze.py does the I/O). Stdlib only.
experiments/SCREENS.txt: C4 (reference SDs per class, the noise checks), C5 (CCCC-ONLY), C7 (metric readings,
arm verdicts, DIVERGENCE, the Holm reading), C6 (IND onsets) and ORDER (the 25 h cap and its cut order).
Quantiles: E3's own t and chi-square functions (e3analyze.py; scipy is not in Planck's venv).

Run values come as {seed: {"CHAT": F, "PROSE": F, "cccc": F, "gutenberg": F, "wikimedia": F}} or
{seed: {"diverged": True}} (a non-finite loss: never resumed, never dropped; C7 DIVERGENCE).
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys

E3D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "E3_seed_noise")
if E3D not in sys.path:
    sys.path.insert(0, E3D)
from e3analyze import chi2_ppf, t_cdf, t_ppf  # noqa: E402

INF = float("inf")
METRICS = {"CHAT": "F(CHAT)", "PROSE": "F(PROSE)", "cccc": "PROSE cccc", "gutenberg": "PROSE gutenberg",
           "wikimedia": "PROSE wikimedia"}             # key in run values -> E3 noise metric
DELTA = 0.01                                           # E3's default delta_bet: 1% of BASE's mean
CAP, CUT_ORDER, NEVER_CUT = 25.0, ["S006", "S003", "S007", "S002"], ["S001", "S004", "S005", "BASE"]


def sd_ref(cls: str, metric: str, base_mean: float, noise: dict, u: float) -> tuple[float, int]:
    """C4: (SD_ref in bpb, df_ref). E3 carries SDs as a fraction of A's mean; times BASE's mean here (C7)."""
    n = noise[metric]
    rel = {"same init": n["SD_d_rel"], "SIA": u * n["SD_d_rel"],
           "new init": math.sqrt(2) * n["sigma_seed_rel"]}[cls]          # E3 Part 3 not run: sqrt(2) sigma_seed
    return rel * base_mean, int(n["df"])


def metric_reading(d: list[float], sd: float, df: int, base_mean: float) -> dict:
    """C7 for one metric: d = arm - BASE per seed (bpb; +inf for a diverged arm seed)."""
    k = len(d)
    dbar = INF if any(x == INF for x in d) else sum(d) / k
    se = sd / math.sqrt(k)
    thr, delta = t_ppf(0.975, df) * se, DELTA * base_mean
    better = -dbar > thr and all(x < 0 for x in d)
    worse = dbar > thr and all(x > 0 for x in d)
    equal = abs(dbar) + thr < delta
    label = "BETTER" if better else "WORSE" if worse else "EQUAL" if equal else "UNRESOLVED"
    p = 0.0 if not math.isfinite(dbar) else 2 * (1 - t_cdf(abs(dbar) / se, df))
    return {"label": label, "d": d, "k": k, "dbar": dbar, "thr": thr, "delta": delta, "sd_ref": sd, "df": df,
            "p_two_sided": p}


def arm_verdict(r: dict) -> tuple[str, list[str]]:
    """C7 + C5: r = {metric key: reading}. CCCC-ONLY (WORSE_PROSE with gutenberg and wikimedia both not WORSE)
    does not block a nomination (C5); a TIE keeps the plain guard (C7: TIE needs not WORSE_PROSE)."""
    worse_p = r["PROSE"]["label"] == "WORSE"
    cccc_only = worse_p and r["gutenberg"]["label"] != "WORSE" and r["wikimedia"]["label"] != "WORSE"
    c = r["CHAT"]["label"]
    labels = ["CCCC-ONLY"] if cccc_only else []
    if c == "BETTER":
        return ("NOMINATE" if not worse_p or cccc_only else "GUARD-FAIL"), labels
    if c == "WORSE":
        return "REJECT", labels
    if c == "EQUAL":
        return ("TIE" if not worse_p else "GUARD-FAIL"), labels
    return "INCONCLUSIVE", labels


def contrast(arm: dict, base: dict, cls: str, noise: dict, u: float) -> dict:
    """One arm against its BASE over the shared seeds (C7). A diverged BASE seed is dropped from the contrast;
    a diverged arm seed scores +inf on every metric (UNSTABLE). Fewer than 2 seeds left: INCONCLUSIVE."""
    seeds = sorted(s for s in arm if s in base)
    used = [s for s in seeds if not base[s].get("diverged")]
    out = {"class": cls, "seeds": used, "dropped_base_diverged": [s for s in seeds if s not in used], "labels": []}
    if len(used) < 2:
        return {**out, "verdict": "INCONCLUSIVE", "labels": ["fewer than 2 seeds"], "readings": {}}
    unstable = any(arm[s].get("diverged") for s in used)
    rd = {}
    for key, m in METRICS.items():
        bm = st.fmean(base[s][key] for s in used)
        d = [INF if arm[s].get("diverged") else arm[s][key] - base[s][key] for s in used]
        sd, df = sd_ref(cls, m, bm, noise, u)
        rd[key] = {**metric_reading(d, sd, df, bm), "base_mean": bm}
    verdict, labels = arm_verdict(rd)
    return {**out, "verdict": verdict, "labels": (["UNSTABLE"] if unstable else []) + labels, "readings": rd}


def sd_bound(df: int) -> float:
    """C4: an observed SD may exceed its reference by at most sqrt(chi2(0.95, df) / df)."""
    return math.sqrt(chi2_ppf(0.95, df) / df)


def residuals(c: dict, key: str) -> list[float]:
    """Relative residuals d_s - dbar (fraction of BASE's mean) of a read contrast; empty if any seed diverged."""
    d = c["readings"].get(key, {}).get("d", [])
    if not d or any(not math.isfinite(x) for x in d):
        return []
    m = sum(d) / len(d)
    return [(x - m) / c["readings"][key]["base_mean"] for x in d]


def same_init_check(c: dict, noise: dict) -> dict:
    """C4 same init (S003): the arm's own SD of d_s against SD_d (relative), per decided metric."""
    out = {}
    for key in ("CHAT", "PROSE"):
        r = residuals(c, key)
        if len(r) < 2:
            continue
        df = len(r) - 1
        sd = math.sqrt(sum(x * x for x in r) / df)
        lim = noise[METRICS[key]]["SD_d_rel"] * sd_bound(df)
        out[key] = {"sd_rel": sd, "limit_rel": lim, "df": df, "ok": sd <= lim}
    return out


def pooled_sia_check(contrasts: list[dict], noise: dict, u: float) -> dict:
    """C4 SIA: residuals of every SIA arm finished so far, pooled (df = sum of k - 1), against u x SD_d."""
    out = {}
    for key in ("CHAT", "PROSE"):
        rs = [residuals(c, key) for c in contrasts]
        rs = [r for r in rs if len(r) >= 2]
        df = sum(len(r) - 1 for r in rs)
        if not df:
            continue
        sd = math.sqrt(sum(x * x for r in rs for x in r) / df)
        lim = u * noise[METRICS[key]]["SD_d_rel"] * sd_bound(df)
        out[key] = {"sd_rel": sd, "limit_rel": lim, "df": df, "arms": len(rs), "ok": sd <= lim}
    return out


def reread_new_init(c: dict, arm: dict, base: dict, noise: dict, u: float, why: str) -> dict:
    """A failed noise check: the arm is re-read with the new-init SD and labelled underpowered (C4)."""
    r = contrast(arm, base, "new init", noise, u)
    r["class_registered"] = c["class"]
    r["labels"] = r["labels"] + [why, "underpowered"]
    return r


def holm(ps: dict, alpha: float = 0.05) -> dict:
    """Holm step-down over every contrast that ran: {name: True = "holds"}."""
    out, ok, items = {}, True, sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))
    for i, (name, p) in enumerate(items):
        ok = ok and p <= alpha / (len(items) - i)
        out[name] = ok
    return out


def cap_cut(remaining: dict, measured: float = 0.0, cap: float = CAP, current: dict | None = None,
            toy_favours_p022: bool = False) -> dict:
    """ORDER's cap: measured hours plus the queued estimate per screen ("BASE" for the BASE runs). Whole screens
    are cut in CUT_ORDER until the total fits (S004 first if the PLAN 3d toy favours P-022); a screen in its seed
    sets keeps its current set's hours (current). S001, S004 (while P-148 is the pick), S005 and BASE are never
    cut by the rule."""
    order = (["S004"] if toy_favours_p022 else []) + CUT_ORDER
    total, cut = measured + sum(remaining.values()), []
    for sid in order:
        if total <= cap:
            break
        if sid in remaining:
            total -= remaining[sid] - (current or {}).get(sid, 0.0)
            cut.append(sid)
    return {"total_before": measured + sum(remaining.values()), "cut": cut, "total_after": total,
            "fits": total <= cap, "cap": cap}


def onsets(curve: list[tuple[int, float]], base_final: float | None) -> dict:
    """C6 IND onsets from logged (step, IND): onset_half = first step reaching half the run's own final IND;
    onset_abs = first step reaching L_s = half of BASE's final IND at the same seed ("not reached" if never)."""
    if not curve:
        return {"onset_half": None, "onset_abs": None}
    final = curve[-1][1]
    half = next((s for s, v in curve if v >= final / 2), None) if final > 0 else None
    if base_final is None:
        return {"onset_half": half, "onset_abs": None, "final": final}
    L = base_final / 2
    ab = next((s for s, v in curve if v >= L), "not reached")
    return {"onset_half": half, "onset_abs": ab, "level_abs": L, "final": final}
