"""SCREENS.txt AMENDMENT BASE-REPLAY's READINGS and LABEL ON THE STAGE 2 PAIRS, and the BASE LEVEL reading over BASE
seeds 101 to 105 (AMENDMENT BASE-DIAG) against E3 Part 1 arm A. Report-only: no verdict moves.
Written 2026-10-11 at W1's stage 2 verdict step, before any value of base_s101_rep1, base_s101_hookoff,
base_s101_rep2 or base_s103..105 was read.

  SD_rep(r) = sqrt((2 x E3^2 + pair^2) / 3) x base_s101(r), df 3; E3 = E3 results.json replay[r].sigma_rep_rel,
              pair = |rep1 - rep2| / sqrt(2) / mean(rep1, rep2)
  D(r) = mean(rep1, rep2) - base_s101; H(r) = hookoff - mean(rep1, rep2); t = x / (SD_rep x sqrt(3/2)); beyond replay
         noise iff |t| > t(0.975, 3)
  label: each S004 / S006 / S007 C7 verdict re-read with every d_s moved by -D(r) at both seeds (its registered
         SD_ref, thr, delta): "BASE replay: stable" or "BASE replay: sensitive (V at the replay level)"; each onset
         reading re-read with BASE's onset_half moved by D_onset (sigma unchanged). S005 and stage 1: not applicable.
  python basereplay.py --runs DIR --verdicts FILE [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

import analyze as A
import analyze_lib as AL
import screens_lib as L

REP1, HOOK, REP2, B101 = "base_s101_rep1", "base_s101_hookoff", "base_s101_rep2", "base_s101"
KEYS = list(AL.METRICS)                         # CHAT, PROSE, cccc, gutenberg, wikimedia
PAIRED = ("S004.canonac", "S006.mtp", "S007.smear")    # pair with stage 1's shared BASE runs
DF_REP = 3
E3_SEEDS = range(1, 9)


def sd_rep(e3_rel: float, rep1: float, rep2: float, base: float) -> float:
    pair = abs(rep1 - rep2) / math.sqrt(2) / ((rep1 + rep2) / 2)
    return math.sqrt((2 * e3_rel ** 2 + pair ** 2) / 3) * base


def shift(x: float, sd: float) -> dict:
    t = x / (sd * math.sqrt(1.5))
    lim = AL.t_ppf(0.975, DF_REP)
    return {"value": x, "t": t, "limit": lim, "p_two_sided": 2 * (1 - AL.t_cdf(abs(t), DF_REP)),
            "beyond": abs(t) > lim, "reads": "beyond replay noise" if abs(t) > lim else "not separated from replay noise"}


def readings(v: dict, e3_replay: dict) -> dict:
    """v: {run: {key: F} or None} for base_s101 and the three runs. -> {key: {sd_rep, D, H}}; None parts "not available"."""
    out = {}
    for k in KEYS:
        b, r1, r2, h = (None if v.get(n) is None or v[n].get("diverged") else v[n][k] for n in (B101, REP1, REP2, HOOK))
        if b is None or r1 is None or r2 is None:
            out[k] = {"D": "not available", "H": "not available"}
            continue
        sd = sd_rep(e3_replay[AL.METRICS[k]]["sigma_rep_rel"], r1, r2, b)
        m = (r1 + r2) / 2
        out[k] = {"sd_rep": sd, "df": DF_REP, "rep1_minus_rep2": r1 - r2, "D": shift(m - b, sd),
                  "H": "not available" if h is None else shift(h - m, sd)}
    return out


def relabel(c: dict, dmap: dict) -> str:
    """A C7 contrast re-read with d_s - D(r) for every reading r, at its registered SD_ref, df and BASE mean."""
    if not c.get("readings") or any(not isinstance(dmap.get(k), (int, float)) for k in KEYS):
        return "BASE replay: not available"
    rd = {}
    for k in KEYS:
        r = c["readings"][k]
        d = [x - dmap[k] for x in r["d"]]
        rd[k] = AL.metric_reading(d, r["sd_ref"], r["df"], r["base_mean"])
    v, _ = AL.arm_verdict(rd)
    return "BASE replay: stable" if v == c["verdict"] else f"BASE replay: sensitive ({v} at the replay level)"


def relabel_onset(o: dict, d_onset) -> str:
    if o.get("reading") != "rule":
        return "not applicable (onset descriptive)"
    if d_onset is None:
        return "not available"
    r = AL.onset_reading([x - d_onset for x in o["d"]], o["sigma_seed"], o["df"])
    return "BASE replay: stable" if r["label"] == o["label"] else f"BASE replay: sensitive ({r['label']} at the replay level)"


def base_level(base: dict, e3a: list[float], sigma_seed: float, df: int, seeds) -> dict:
    """stage 1's base_level_vs_e3_part1 statistic for the given BASE seeds."""
    xs = [base[s] for s in seeds]
    gap = st.fmean(xs) - st.fmean(e3a)
    t = gap / (sigma_seed * math.sqrt(1 / len(xs) + 1 / len(e3a)))
    return {"seeds": list(seeds), "base": xs, "gap": gap, "gap_rel": gap / st.fmean(e3a), "t": t,
            "p_two_sided": 2 * (1 - AL.t_cdf(abs(t), df)), "above_all_8": [x > max(e3a) for x in xs]}


def level(runs: str, e3_out: str, e3_noise: dict) -> dict:
    v = {s: A.final(runs, f"base_s{s}") for s in AL.ONSET_SEEDS}
    if any(x is None or x.get("diverged") for x in v.values()):
        return {"reading": "not available", "missing": [s for s, x in v.items() if not x or x.get("diverged")]}
    out = {}
    for k in ("CHAT", "PROSE"):
        e3a = [A.final(e3_out, f"e3_5m_A_s{s}")[k] for s in E3_SEEDS]
        n = e3_noise[AL.METRICS[k]]
        b = {s: v[s][k] for s in v}
        out[k] = {"e3_arm_a": e3a, "n5": base_level(b, e3a, n["sigma_seed"], n["df"], AL.ONSET_SEEDS),
                  "n3": base_level(b, e3a, n["sigma_seed"], n["df"], (103, 104, 105))}
    fresh = all(out["CHAT"]["n3"]["above_all_8"])
    out["seed_draw"] = ("the gap is not the seed draw" if fresh else "the seed draw is not excluded")
    return out


def analyze(runs: str, verdicts: dict, e3_out: str, e3: dict) -> dict:
    v = {n: A.final(runs, n) for n in (B101, REP1, REP2, HOOK)}
    rd = readings(v, e3["replay"])
    dmap = {k: rd[k]["D"]["value"] if isinstance(rd[k]["D"], dict) else None for k in KEYS}
    labels = {}
    for key, c in verdicts["contrasts"].items():
        labels[key] = relabel(c, dmap) if key in PAIRED else "not applicable"
    h = {n: A.onset_half(runs, n)[1] for n in (B101, REP1, REP2)}
    d_on = None if None in h.values() else (h[REP1] + h[REP2]) / 2 - h[B101]
    on = {k: relabel_onset(o, d_on) for k, o in verdicts["onsets"]["contrasts"].items()}
    return {"rule": "SCREENS.txt AMENDMENT BASE-REPLAY", "runs_present": {n: x is not None for n, x in v.items()},
            "readings": rd, "labels": labels, "onset_half": h, "D_onset": d_on, "onset_labels": on,
            "base_level_101_105": level(runs, e3_out, e3["noise_5M_250M"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--e3-out", default=os.path.join(L.E3D, "out"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = analyze(a.runs, json.load(open(a.verdicts)), a.e3_out, json.load(open(os.path.join(L.E3D, "results.json"))))
    txt = json.dumps(res, indent=1, sort_keys=True, default=str)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
