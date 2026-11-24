"""Track K's analyzer core (SPEC 9, 10, 12; SPEC 14 analyze_k.py): rule KR, the CE-DISAGREE and GENERAL-LANGUAGE COST
labels, DIVERGENCE, the Holm reading, the time course (peak, final, TRANSIENT), the skill summaries computed from the
scorer's item rows (probes.py), and the append-only readings file. Pure functions except append_reading and main.
Quantiles: E3's stdlib t functions (experiments/E3_seed_noise/e3analyze.py; scipy is not in Planck's venv).

Values per arm come as {seed: value}; a diverged arm seed is None (SPEC 10 DIVERGENCE, as SCREENS C7: it scores the
worst value, never dropped; a diverged reference seed drops that seed from the contrasts that use it).
  paired(arm, ref, higher_better)    -> {seed: d_s}, d_s = arm - ref, -inf (higher better) or +inf for a diverged arm
  kr(contrasts, delta)               SPEC 10 over one experiment's listed contrasts {name: {seed: d_s}}: SD_ref pooled
                                     over the finite contrasts' residuals d_s - dbar (df = sum of k - 1), thr = t(0.975,
                                     df) x SD_ref / sqrt(k); HIGHER dbar > thr, every d_s > 0, dbar >= delta; LOWER the
                                     mirror; EQUAL |dbar| + thr < delta; else UNRESOLVED; a diverged arm UNSTABLE; p
                                     = the two-sided p of dbar / (SD_ref / sqrt(k)) on t(df) for the Holm reading
  holm(ps)                           Holm step-down at 0.05 over every K contrast that ran: {name: holds}
  pick_g(table)                      K0-g's pick (K1 notes; K4's shape g-checks alike) at two selection seeds [G-2]:
                                     the least mean K-DEV objective over the seeds, ties to the lower LR, one
                                     factor-2 extension beyond an edge pick
  skill_label(lik, ce)               a skill HIGHER / LOWER needs CE_C's dbar to move the same way (down / up), else
                                     UNRESOLVED labelled CE-DISAGREE
  bpb_cost(d, sd_ref, df)            WORSE_bpb as SCREENS C7 (dbar > thr and every d_s > 0): GENERAL-LANGUAGE COST
  probe_curve(path)                  a run's k_probe.jsonl as [(step, K-PROBE LIK)] (or BIO_acc)
  course(curve), transient(curves)   SPEC 9 [C7]: peak = max of the 5-checkpoint moving mean, final = mean of the
                                     last 5; TRANSIENT if peak - final >= 3 points on 2 of 3 seeds (K1 [K1-d])
  summary(rows, items)               S_ID (K-EVAL-ID LIK), S_ID-S, per family LIK / CE_C / GEN / chance, F per the key's
                                     reference kind, B cube vs 2-4 names and B group accuracy, every OOD column
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
import time

import kcommon as K  # noqa: E402

E3D = os.path.join(K.ROOT, "experiments", "E3_seed_noise")
if E3D not in sys.path:
    sys.path.insert(0, E3D)
from e3analyze import t_cdf, t_ppf  # noqa: E402

INF = float("inf")
DELTA = {"S_ID": 0.03, "GEN": 0.05, "RATE": 0.10}      # SPEC 10; stored bits: 10% of the reference mean (bits_delta)


def bits_delta(ref: dict) -> float:
    return 0.10 * st.mean(v for v in ref.values() if v is not None)


def paired(arm: dict, ref: dict, higher_better: bool = True) -> dict:
    out = {}
    for s in sorted(set(arm) & set(ref)):
        if ref[s] is None:
            continue
        out[s] = (-INF if higher_better else INF) if arm[s] is None else float(arm[s]) - float(ref[s])
    return out


def kr(contrasts: dict, delta) -> dict:
    dl = delta if isinstance(delta, dict) else {n: float(delta) for n in contrasts}
    fin = {n: list(d.values()) for n, d in contrasts.items() if len(d) >= 2 and all(math.isfinite(x) for x in d.values())}
    df = sum(len(v) - 1 for v in fin.values())
    ss = sum((x - st.mean(v)) ** 2 for v in fin.values() for x in v)
    sd = math.sqrt(ss / df) if df else float("nan")
    out = {}
    for name, d in contrasts.items():
        xs, k = list(d.values()), len(d)
        r = {"k": k, "seeds": sorted(d), "d": dict(d), "sd_ref": sd, "df": df, "delta": dl[name], "labels": []}
        if k < 2 or not df:
            out[name] = {**r, "dbar": st.mean(xs) if xs else None, "thr": None, "p": None, "verdict": "UNRESOLVED",
                         "labels": ["fewer than 2 seeds" if k < 2 else "no finite residuals"]}
            continue
        dbar = sum(xs) / k
        thr = t_ppf(0.975, df) * sd / math.sqrt(k)
        unstable = not all(math.isfinite(x) for x in xs)
        if dbar > thr and all(x > 0 for x in xs) and dbar >= dl[name]:
            v = "HIGHER"
        elif -dbar > thr and all(x < 0 for x in xs) and -dbar >= dl[name]:
            v = "LOWER"
        elif abs(dbar) + thr < dl[name]:
            v = "EQUAL"
        else:
            v = "UNRESOLVED"
        p = None if unstable or sd == 0 else 2.0 * (1.0 - t_cdf(abs(dbar) / (sd / math.sqrt(k)), df))
        out[name] = {**r, "dbar": dbar, "thr": thr, "p": p, "verdict": v, "labels": ["UNSTABLE"] if unstable else []}
    return out


def holm(ps: dict, alpha: float = 0.05) -> dict:
    """{name: p or None} -> {name: True (holds) | False | None (no p)}; Holm step-down over the names with a p."""
    items = sorted(((n, p) for n, p in ps.items() if p is not None), key=lambda kv: (kv[1], kv[0]))
    out, ok = {n: None for n in ps}, True
    for i, (name, p) in enumerate(items):
        ok = ok and p <= alpha / (len(items) - i)
        out[name] = ok
    return out


G_GRID, SEL_SEEDS = (0.5, 1.0, 2.0), (199, 200)     # K1 notes K0-g; SPEC 11 [G-2]: every g-check at seeds 199 and 200


def pick_g(table: dict, seeds=SEL_SEEDS, grid=G_GRID) -> dict:
    """K0-g's rule at two seeds [G-2] (K4's shape g-checks alike): table {g: {seed: K-DEV objective (the mean NLL of
    K1 notes K0-g, lower is better) or None (diverged)}}, every g at exactly these seeds (else ValueError). The pick is
    the g with the least mean over the seeds; a g with a diverged seed has the mean +inf and is never picked (SPEC 10
    DIVERGENCE: never dropped); ties at 4 decimals of the mean go to the lower g, the lower LR. On the grid alone a
    pick at an edge is provisional ("final" False) and "extend" names the one factor-2 point beyond it, run at both
    seeds; with that point in the table the pick is final whatever it is (at most one extension). Reported: each
    seed's own argmin, labelled SEED-SPLIT when they differ (the rule is the mean's either way).
    K3 round 14 (2026-10-11, REVIEW 15b T-2, T-6): every grid point must be present, a point off the grid is taken
    only if it is the extension the grid's own pick asks (else ValueError), and a non-finite objective (NaN, inf) is
    a diverged seed, in the pick and in each seed's argmin alike."""
    ok = lambda v: v is not None and math.isfinite(float(v))       # noqa: E731
    gs = sorted(float(g) for g in table)
    for g in gs:
        if sorted(table[g]) != sorted(seeds):
            raise ValueError(f"g {g}: seeds {sorted(table[g])}, not {sorted(seeds)}")
    extra = [g for g in gs if g not in grid]
    if any(g not in gs for g in grid) or len(extra) > 1:
        raise ValueError(f"g values {gs}: the grid {list(grid)} and at most one point beyond it")
    mean = {g: INF if any(not ok(table[g].get(s)) for s in seeds)
            else sum(float(table[g][s]) for s in seeds) / len(seeds) for g in gs}
    per = {s: min((g for g in gs if ok(table[g].get(s))), key=lambda g: (round(float(table[g][s]), 4), g),
                  default=None) for s in seeds}
    labels = ["SEED-SPLIT"] if len(set(per.values())) > 1 else []

    def best(cands):
        fin = [g for g in cands if math.isfinite(mean[g])]
        return min(fin, key=lambda g: (round(mean[g], 4), g)) if fin else None
    gpick = best([g for g in gs if g in grid])
    ask = None if gpick not in (min(grid), max(grid)) else (gpick / 2 if gpick == min(grid) else gpick * 2)
    if extra and extra[0] != ask:
        raise ValueError(f"g {extra[0]}: the grid's pick {gpick} asks {'no extension' if ask is None else ask}")
    pick = best(gs)
    if pick is None:
        return {"g": None, "final": True, "extend": None, "mean": mean, "per_seed": per,
                "labels": labels + ["every g diverged"]}
    ext = None if extra else ask
    return {"g": pick, "final": ext is None, "extend": ext, "mean": mean, "per_seed": per, "labels": labels}


def skill_label(lik: dict, ce: dict) -> dict:
    """A KR result on a skill metric (LIK, S_ID) and the same contrast's KR result on CE_C (nats, lower is better)."""
    v, dce = lik["verdict"], ce.get("dbar")
    if v in ("HIGHER", "LOWER") and (dce is None or not ((v == "HIGHER" and dce < 0) or (v == "LOWER" and dce > 0))):
        return {**lik, "verdict": "UNRESOLVED", "labels": lik["labels"] + ["CE-DISAGREE"], "raw_verdict": v}
    return lik


def bpb_cost(d: dict, sd_ref: float, df: int) -> dict:
    """d: {seed: arm bpb - ref bpb} (positive = arm worse; a diverged arm seed +inf). sd_ref in bpb, E3's by class."""
    xs = list(d.values())
    if len(xs) < 2:
        return {"worse": False, "labels": ["fewer than 2 seeds"]}
    dbar, thr = sum(xs) / len(xs), t_ppf(0.975, df) * sd_ref / math.sqrt(len(xs))
    worse = dbar > thr and all(x > 0 for x in xs)
    return {"worse": worse, "dbar": dbar, "thr": thr, "labels": ["GENERAL-LANGUAGE COST"] if worse else []}


def course(curve: list, w: int = 5) -> dict:
    """curve: [(step, value)] in step order -> peak (max of the w-checkpoint moving mean), its step (the window's last),
    final (mean of the last w)."""
    pts = sorted(curve)
    if len(pts) < w:
        raise ValueError(f"time course needs at least {w} checkpoints, got {len(pts)}")
    vals = [v for _, v in pts]
    mm = [(sum(vals[i - w + 1:i + 1]) / w, pts[i][0]) for i in range(w - 1, len(vals))]
    peak, at = max(mm, key=lambda t: (t[0], -t[1]))
    return {"peak": peak, "peak_step": at, "final": sum(vals[-w:]) / w}


def probe_curve(path: str, key: str = "skill_lik", sset: str = "K-PROBE") -> list:
    """A run's out_dir/k_probe.jsonl (kprobe.py) -> [(step, value)]: key skill_lik (K-PROBE LIK) or fact_acc
    (BIO_acc); a reading that logged an error is refused, never skipped."""
    out = []
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            raise ValueError(f"{path}: step {r['step']} has an error reading: {r['error']}")
        out.append((int(r["step"]), float(r[key][sset])))
    return out


def transient(curves: dict, drop: float = 0.03, need: int = 2) -> dict:
    """curves: {seed: [(step, LIK)]} -> TRANSIENT if peak - final >= drop on `need` seeds (K1 [K1-d]: 2 of 3)."""
    per = {s: course(c) for s, c in curves.items()}
    n = sum(r["peak"] - r["final"] >= drop for r in per.values())
    return {"transient": n >= need, "seeds_dropping": n, "per_seed": per}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def summary(rows: list, items: list) -> dict:
    """rows: probes.score output; items: the scored records (joined by set and id)."""
    meta = {(r.get("set"), r["id"]): r for r in items}
    by = {}
    for o in rows:
        by.setdefault(o["set"], []).append(o)

    def fam_stats(rs):
        return {"n": len(rs), "lik": _mean([o["lik"] for o in rs]), "ce_c": _mean([o["ce_c"] for o in rs]),
                "gen": _mean([o["gen"] for o in rs]), "chance": _mean([1.0 / o["n_c"] for o in rs])}
    out = {"sets": {}}
    for s, rs in sorted(by.items()):
        fams = {}
        for o in rs:
            fams.setdefault(o["fam"], []).append(o)
        e = {"all": fam_stats(rs), "family": {f: fam_stats(v) for f, v in sorted(fams.items())}}
        F = [o for o in fams.get("F", []) if (s, o["id"]) in meta]
        if F:
            e["F_by_kref"] = {k: fam_stats([o for o in F if meta[(s, o["id"])]["meta"].get("kref") == k])
                              for k in sorted({meta[(s, o["id"])]["meta"].get("kref") for o in F})}
        B = [(o, meta[(s, o["id"])]["meta"]) for o in fams.get("B", []) if (s, o["id"]) in meta]
        if B:
            e["B_cube"] = fam_stats([o for o, m in B if m.get("struct")])
            e["B_small"] = fam_stats([o for o, m in B if not m.get("struct")])
            groups, cur = [], None
            for o, m in B:
                if m.get("grp") is None:
                    continue
                if m["grp"] == 0:
                    cur = []
                    groups.append(cur)
                if cur is None:
                    raise ValueError(f"{s}: B group member {o['id']} before its group's first item")
                cur.append(o["lik"] == 1.0)
            if groups:
                if any(len(g) != 4 for g in groups):
                    raise ValueError(f"{s}: a B group without exactly 4 questions")
                e["B_groups"] = {"n": len(groups), "acc": _mean([all(g) for g in groups])}
        out["sets"][s] = e
    pick = lambda s: out["sets"].get(s, {}).get("all", {}).get("lik")   # noqa: E731
    out["S_ID"], out["S_ID_S"] = pick("K-EVAL-ID"), pick("K-EVAL-S")
    out["OOD"] = {s: e["all"]["lik"] for s, e in out["sets"].items() if s.startswith("OOD-")}
    return out


def append_reading(path: str, rec: dict) -> dict:
    """SPEC 12: one line appended (time added), never overwritten; an existing file must be whole JSON lines."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()
        if raw and not raw.endswith(b"\n"):
            raise ValueError(f"{path}: the last line is incomplete; refusing to append")
        for i, line in enumerate(raw.splitlines()):
            json.loads(line)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **rec}
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return row
