"""SCREENS.txt AMENDMENT C6-DIAG, READING RULE and NEAR ITS INIT, as pure functions (c6read.py does the I/O).
Written 2026-10-11 at W1's stage 2 verdict step, before any C6 value of any screen or E3 run was read.

  final IND, HD_asst, HD_user: d_s = m(arm, s) - m(BASE, s), dbar over k seeds; SD_ref by the arm's C4 class from
  E3 PART 4d's Part 1 numbers in absolute units (same init SD_d; SIA u x SD_d, u = sqrt(df / chi2(0.20, df)); new
  init sqrt(2) x sigma_seed), df = PART 4d's df; thr = t(0.975, df) x SD_ref / sqrt(k).
    HIGHER  dbar > thr and every d_s > 0     LOWER  -dbar > thr and every d_s < 0     else UNRESOLVED (no EQUAL)
  Holm per metric over the contrasts it read by rule (two-sided p of dbar / (SD_ref / sqrt(k)) on t(df)). C4's
  noise checks per metric with these SDs (same init: the arm's own SD of d_s; SIA: pooled residuals), a failed check
  re-reads at the new-init SD, labelled as C4 says, plus "underpowered". A metric PART 4d gives no SD, or a contrast
  with a value missing, stays descriptive (the paired mean and the seeds per sign).
Values come as {seed: float or None}. Units: nats (IND), bpb (HD).
CHANGED 2026-10-11 10:45 EDT, after the audit and AFTER every C6 value was read (no reading moves): reread() attached
"underpowered" to every re-read, but C4 gives it only to "an arm re-read with a larger SD" than the one it was read
at; it is now attached only then, and each re-read records the SD it replaced (sd_ref_registered). read_metric() adds
a "sensitivity" block (reported, no rule) wherever a check re-read contrasts: the same contrasts at the SD they were
registered at and, after a failed pooled SIA check, at the observed pooled SD (its df), each with Holm over the whole
family, so a re-read at a SMALLER SD than the one whose check failed is visible. The re-read readings stand.
"""
from __future__ import annotations

import math

from analyze_lib import chi2_ppf, sd_bound, t_cdf, t_ppf

RULED = ("IND", "HD_asst", "HD_user")
FAILED_SIA, FAILED_SAME = "noise check failed (pooled SIA)", "noise check failed"
NEAR = {"S004": 0.01, "S007": 0.05, "S005": 1023.0}      # NEAR ITS INIT thresholds (C6-DIAG), fixed before any value


def u_of(df: int) -> float:
    """C4's u at PART 4d's df: E3's 80% upper bound for an SD (1.3533 at df 7)."""
    return math.sqrt(df / chi2_ppf(0.20, df))


def sd_ref(cls: str, noise: dict) -> float:
    """SD_ref in absolute units from PART 4d's Part 1 noise (results_part4d.json metrics[m].part1_5M.noise)."""
    return {"same init": noise["SD_d"], "SIA": u_of(noise["df"]) * noise["SD_d"],
            "new init": math.sqrt(2) * noise["sigma_seed"]}[cls]


def reading(d: list[float], sd: float, df: int) -> dict:
    k = len(d)
    dbar = sum(d) / k
    se = sd / math.sqrt(k)
    thr = t_ppf(0.975, df) * se
    higher = dbar > thr and all(x > 0 for x in d)
    lower = -dbar > thr and all(x < 0 for x in d)
    label = "HIGHER" if higher else "LOWER" if lower else "UNRESOLVED"
    p = 2 * (1 - t_cdf(abs(dbar) / se, df)) if se > 0 else (0.0 if dbar else 1.0)
    return {"label": label, "d": d, "k": k, "dbar": dbar, "thr": thr, "sd_ref": sd, "df": df, "p_two_sided": p}


def descriptive(d: list[float]) -> dict:
    return {"d": d, "dbar": sum(d) / len(d) if d else None, "higher": sum(1 for x in d if x > 0),
            "lower": sum(1 for x in d if x < 0), "same": sum(1 for x in d if x == 0)}


def paired(arm: dict, base: dict) -> tuple[list[int], list[float], list[int]]:
    """-> (seeds with both values, d over them, seeds with a value missing)."""
    seeds = sorted(set(arm) | set(base))
    ok = [s for s in seeds if arm.get(s) is not None and base.get(s) is not None
          and math.isfinite(arm[s]) and math.isfinite(base[s])]
    return ok, [arm[s] - base[s] for s in ok], [s for s in seeds if s not in ok]


def contrast(arm: dict, base: dict, cls: str, noise: dict | None) -> dict:
    """One arm, one metric. noise None (PART 4d gave no SD) or a missing value: descriptive."""
    seeds, d, miss = paired(arm, base)
    out = {"class": cls, "seeds": seeds, "arm": arm, "base": base, "labels": []}
    why = ([f"value missing at seeds {miss}"] if miss else []) + (["PART 4d gives no SD"] if noise is None else [])
    if why or len(d) < 2:
        return {**out, "reading": "descriptive", "why": why or ["fewer than 2 seeds"], **descriptive(d)}
    return {**out, "reading": "rule", **reading(d, sd_ref(cls, noise), noise["df"])}


def residuals(c: dict) -> list[float]:
    d = c["d"]
    m = sum(d) / len(d)
    return [x - m for x in d]


def same_init_check(c: dict, noise: dict) -> dict:
    r = residuals(c)
    df = len(r) - 1
    sd = math.sqrt(sum(x * x for x in r) / df)
    lim = noise["SD_d"] * sd_bound(df)
    return {"sd": sd, "limit": lim, "df": df, "ok": sd <= lim}


def pooled_sia_check(cs: list[dict], noise: dict) -> dict | None:
    rs = [residuals(c) for c in cs if c["reading"] == "rule" and len(c["d"]) >= 2]
    df = sum(len(r) - 1 for r in rs)
    if not df:
        return None
    sd = math.sqrt(sum(x * x for r in rs for x in r) / df)
    lim = u_of(noise["df"]) * noise["SD_d"] * sd_bound(df)
    return {"sd": sd, "limit": lim, "df": df, "arms": len(rs), "ok": sd <= lim}


def reread(c: dict, noise: dict, why: str) -> dict:
    """C4's re-read at the new-init SD; "underpowered" only if that SD is larger than the one c was read at."""
    r = contrast(c["arm"], c["base"], "new init", noise)
    larger = r["sd_ref"] > c["sd_ref"]
    return {**r, "class_registered": c["class"], "sd_ref_registered": c["sd_ref"],
            "labels": c["labels"] + [why] + (["underpowered"] if larger else [])}


def holm(ps: dict, alpha: float = 0.05) -> dict:
    out, ok = {}, True
    items = sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))
    for i, (name, p) in enumerate(items):
        ok = ok and p <= alpha / (len(items) - i)
        out[name] = ok
    return out


def alt_family(res: dict, swap: dict) -> dict:
    """Holm over the whole family with the contrasts in swap read as given there; every other one as read."""
    ps = {k: (swap.get(k) or c)["p_two_sided"] for k, c in res.items() if c["reading"] == "rule"}
    hm = {k: "holds" if v else "does not hold" for k, v in holm(ps).items()}
    return {"readings": {k: {**{f: r[f] for f in ("label", "d", "dbar", "thr", "sd_ref", "df", "p_two_sided")},
                             "holm": hm[k]} for k, r in sorted(swap.items())}, "holm": dict(sorted(hm.items()))}


def sensitivity(first: dict, res: dict, pooled: dict | None) -> dict | None:
    """REPORTED, NO RULE (added 2026-10-11 after the audit, after the values were read). For the contrasts a failed
    check re-read: their reading at the SD they were registered at (first, before the re-read) and, for a failed
    pooled SIA check, at the observed pooled SD on its df; Holm over the whole family in each case."""
    re_ = sorted(k for k, c in res.items() if "class_registered" in c)
    if not re_:
        return None
    out = {"reread": re_, "smaller_than_registered": [k for k in re_ if res[k]["sd_ref"] < res[k]["sd_ref_registered"]],
           "at_registered_sd": alt_family(res, {k: first[k] for k in re_})}
    pool = [k for k in re_ if FAILED_SIA in res[k]["labels"]]
    if pool and pooled is not None:
        out["at_pooled_sd"] = alt_family(res, {k: reading(res[k]["d"], pooled["sd"], pooled["df"]) for k in pool})
    return out


def read_metric(values: dict, classes: dict, noise: dict | None) -> dict:
    """values: {contrast key: (arm {seed: v}, base {seed: v})}; classes: {key: C4 class}. The whole rule for one
    metric: readings, the noise checks, the re-reads, Holm; and the re-reads' sensitivity block (no rule)."""
    res = {k: contrast(a, b, classes[k], noise) for k, (a, b) in values.items()}
    first = dict(res)                  # each contrast as read at its registered class, before any re-read
    checks: dict = {}
    if noise is not None:
        for k, c in list(res.items()):
            if c["class"] == "same init" and c["reading"] == "rule":
                chk = checks[k] = same_init_check(c, noise)
                if not chk["ok"]:
                    res[k] = reread(c, noise, FAILED_SAME)
        sia = [k for k, c in res.items() if c["class"] == "SIA" and c["reading"] == "rule"]
        pooled = checks["pooled_sia"] = pooled_sia_check([res[k] for k in sia], noise)
        if pooled is not None and not pooled["ok"]:
            for k in sia:
                res[k] = reread(res[k], noise, FAILED_SIA)
    sens = sensitivity(first, res, checks.get("pooled_sia"))
    ps = {k: c["p_two_sided"] for k, c in res.items() if c["reading"] == "rule"}
    hm = holm(ps)
    for k, c in res.items():
        c["holm"] = None if k not in hm else ("holds" if hm[k] else "does not hold")
    return {"contrasts": res, "noise_checks": checks, "holm_family": sorted(ps)} | ({"sensitivity": sens} if sens else {})


def s005_flag(chat_label: str, hd_asst: dict | None) -> bool:
    """S005 DIAGNOSTICS: an arm BETTER on CHAT with lower HD (LOWER on HD_asst by the rule) is flagged."""
    return chat_label == "BETTER" and bool(hd_asst) and hd_asst.get("label") == "LOWER"


def near_init(sid: str, mods: dict) -> dict:
    """NEAR ITS INIT (C6-DIAG), per screen, from {seed: modules or halflife result}. True only if the condition holds
    at every seed given (both screen seeds). S004: a site whose mean |w| over every layer and tap < 0.01; S007: every
    layer's |alpha| < 0.05 for every KV head; S005: every layer and head's mean half-life > 1,023 tokens."""
    thr, per = NEAR[sid], {}
    for s, m in mods.items():
        if sid == "S004":
            site = {}
            for x in m["canon"]:
                site.setdefault(x["site"], []).append(x["mean_abs"])
            per[s] = {k: sum(v) / len(v) for k, v in site.items()}
        elif sid == "S007":
            per[s] = max(abs(a) for x in m["smear_alpha"] for a in x["alpha"])
        else:
            per[s] = min(h for x in m["by_layer"] if x for h in x["half_life"])
    if sid == "S004":
        sites = sorted(set.intersection(*(set(v) for v in per.values())))
        near = [k for k in sites if all(per[s][k] < thr for s in per)]
        return {"per_seed": per, "threshold": thr, "sites_near_init": near, "fires": bool(near)}
    fires = all(v < thr for v in per.values()) if sid == "S007" else all(v > thr for v in per.values())
    return {"per_seed": per, "threshold": thr, "fires": fires}
