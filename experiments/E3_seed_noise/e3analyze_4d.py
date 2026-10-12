"""E3 Part 4d statistics (notes.txt PART 4d ADDENDUM, 2026-10-10): final IND and HD on E3's kept final checkpoints.
  python e3analyze_4d.py [--out DIR] [--dest DIR]   reads out/<run>/c6/{ind,hd}[.replay].json (the C6 runner's JSON,
                         copied from the PC); writes results_part4d.json and tables_part4d.txt beside this file (--dest)
Stdlib only. e3analyze.py's formulas and quantiles (noise() less its k_needed and relative columns, which assume a mean
far from 0); every SD in absolute units (nats for IND, bpb for HD and HD_none). A pair with a missing or non-finite
value is dropped from that metric's SDs (df reduced, counted); under MIN_PAIRS Part 1 pairs a metric gets no SD.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3analyze as E  # noqa: E402

ROLE = lambda role, key: lambda r: r["sets"]["oasst2"]["hd"]["by_role"][role][key]   # noqa: E731
METRICS = {"IND": ("ind", lambda r: r["ind"]), "HD_asst": ("hd", ROLE("assistant", "HD")),
           "HD_user": ("hd", ROLE("user", "HD")), "HDn_asst": ("hd", ROLE("assistant", "HD_none")),
           "HDn_user": ("hd", ROLE("user", "HD_none"))}
RULED = ("IND", "HD_asst", "HD_user")          # SCREENS AMENDMENT C6-DIAG reads these; HDn only beside HD
P1 = [f"e3_5m_{{}}_s{s}" for s in range(1, 9)]
P2 = [f"e3_20m_{{}}_s{s}" for s in (1, 2, 3)]
REPS = ("e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3")
MIN_PAIRS = 6


def value(out: str, run: str, metric: str, suffix: str = "") -> float | None:
    job, get = METRICS[metric]
    p = os.path.join(out, run, "c6", f"{job}{suffix}.json")
    if not os.path.exists(p):
        return None
    try:
        v = float(get(json.load(open(p))["result"]))
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def noise_abs(a: list[float], b: list[float]) -> dict:
    n = len(a)
    df, d = n - 1, [y - x for x, y in zip(a, b)]
    s_seed, s_d, f = math.sqrt((st.variance(a) + st.variance(b)) / 2), st.stdev(d), E.factor(n - 1)
    md, se, tq = st.fmean(d), s_d / math.sqrt(n), E.t_ppf(0.975, df)
    t = md / se if se else float("nan")
    return {"n": n, "df": df, "factor": f, "mean_a": st.fmean(a), "mean_b": st.fmean(b), "sigma_seed": s_seed,
            "SD_d": s_d, "rho": st.correlation(a, b) if s_seed else None, "mean_d": md, "t": t,
            "p_two_sided": 2 * (1 - E.t_cdf(abs(t), df)) if se else None, "ci95_d": [md - tq * se, md + tq * se],
            "crn_ratio": math.sqrt(2) * s_seed / s_d if s_d else None,
            "MDE": {str(k): {"paired": f * s_d / math.sqrt(k), "unpaired": f * s_seed * math.sqrt(2 / k),
                             "paired_ub80": f * s_d / math.sqrt(k) * E.ub(df),
                             "unpaired_ub80": f * s_seed * math.sqrt(2 / k) * E.ub(df)} for k in E.KS}}


def part(out: str, names: list[str], metric: str, min_pairs: int) -> dict:
    vals = [(s + 1, value(out, n.format("A"), metric), value(out, n.format("B"), metric)) for s, n in enumerate(names)]
    kept = [(s, a, b) for s, a, b in vals if a is not None and b is not None]
    res = {"values": {f"s{s}": [a, b] for s, a, b in vals}, "pairs": len(kept),
           "dropped": [f"s{s}" for s, a, b in vals if a is None or b is None]}
    if len(kept) < min_pairs:
        res["noise"] = None
        res["why"] = f"{len(kept)} pairs < {min_pairs}: no SD (SCREENS C6 stays descriptive)"
        return res
    res["noise"] = noise_abs([a for _, a, _ in kept], [b for _, _, b in kept])
    rest = [(a, b) for s, a, b in kept if s != 1]
    res["noise_seeds2to8"] = noise_abs([a for a, _ in rest], [b for _, b in rest]) if len(rest) >= 3 else None
    return res


def analyze(out: str) -> dict:
    res = {"metrics": {}, "ruled": list(RULED), "units": "absolute: nats (IND), bpb (HD, HD_none)",
           "min_pairs_part1": MIN_PAIRS}
    for m in METRICS:
        p1 = part(out, P1, m, MIN_PAIRS)
        rep = [value(out, r, m) for r in REPS]
        if p1["noise"] and all(x is not None for x in rep):
            s_rep = st.stdev(rep)
            p1["replay"] = {"values": rep, "sigma_rep": s_rep, "df": 2,
                            "share_of_paired_var": 2 * s_rep ** 2 / p1["noise"]["SD_d"] ** 2 if p1["noise"]["SD_d"] else None}
        else:
            p1["replay"] = "not computed"
        once, again = value(out, REPS[0], m), value(out, REPS[0], m, ".replay")
        p1["scorer_replay"] = (abs(again - once) if once is not None and again is not None else "not computed")
        res["metrics"][m] = {"part1_5M": p1, "part2_20M": part(out, P2, m, 3)}
    return res


def tables(r: dict) -> str:
    L = ["E3 PART 4d (notes.txt PART 4d ADDENDUM): final IND (nats) and HD / HD_none by role (bpb); absolute SDs",
         f"{'metric':10s} {'part':5s} {'n':>2s} {'mean A':>9s} {'mean d':>9s} {'sig_seed':>9s} {'SD_d':>9s} "
         f"{'MDEp k2':>9s} {'sig_rep':>9s} {'scorer':>9s}"]
    for m, v in r["metrics"].items():
        for part_name, p in (("5M", v["part1_5M"]), ("20M", v["part2_20M"])):
            z = p["noise"]
            if not z:
                L.append(f"{m:10s} {part_name:5s} {p['pairs']:2d} {p['why']}")
                continue
            rep = p.get("replay")
            sr = f"{rep['sigma_rep']:9.5f}" if isinstance(rep, dict) else f"{'-':>9s}"
            sc = p.get("scorer_replay")
            sc = f"{sc:9.2e}" if isinstance(sc, float) else f"{'-':>9s}"
            L.append(f"{m:10s} {part_name:5s} {z['n']:2d} {z['mean_a']:9.5f} {z['mean_d']:+9.5f} {z['sigma_seed']:9.5f} "
                     f"{z['SD_d']:9.5f} {z['MDE']['2']['paired']:9.5f} {sr} {sc}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "out")
    dest = argv[argv.index("--dest") + 1] if "--dest" in argv else HERE
    res = analyze(out)
    json.dump(res, open(os.path.join(dest, "results_part4d.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(dest, "tables_part4d.txt"), "w").write(tables(res))
    print(tables(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
