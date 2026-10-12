"""E2 decision rules (notes.txt RULES) on recorded bpb.jsonl results. Stdlib only; Mac or PC; reads JSON, runs nothing.

  python e2pick.py stage A --runs DIR                          stage A: eta grid at r 1 (5M, 250M branch)
  python e2pick.py stage B --runs DIR --eta-a ETA              stage B: r grid at eta_A
  python e2pick.py stage C --runs DIR --eta-a ETA --r-b R      stage C: g grid (all LRs x g) at (eta_A, r_B)
  python e2pick.py q2 --runs DIR --eta5 ETA --r5 R             transfer: vertices, s, HOLDS / MOVES / UNRESOLVED
DIR holds one directory per run (~/planck/runs/E2 or its copy): <run>/bpb.jsonl, DIVERGED, GAP. Values read:
F(CHAT) and F(PROSE) = the final checkpoint's bpb (bpb_lines.py sets CHAT and PROSE, split all).
Rules implemented (fixed in notes.txt; this file adds no rule):
  pick     argmin F(CHAT) at 250M on the stage's axis, compared at 4 decimals, exact ties to the lower LR; if the
           pick's F(PROSE) is more than 1% above the axis's best F(PROSE), the pick moves one grid step toward
           the PROSE best (reported). A diverged run scores +inf; a GAP is left out, and a GAP next to the
           argmin is flagged "stop for a person".
  extend   a pick at a grid edge asks for one more factor-2 point beyond that edge, at most 2 per stage; stage C
           also while its 125M argmin is at an edge (Q2 needs it bracketed); the 20M grid (q2) while its 500M
           argmin is at an edge, at most 2.
  runner   stage C's runner-up (E3's arm B) = the pick's neighbour on the stage C axis with the lower 250M
           F(CHAT) (4 decimals, ties to the lower LR; a GAP neighbour does not count). q2 gives A20 (the 20M
           argmin at 500M) and B20 (its grid neighbour with the lower 500M F(CHAT)) the same way.
  vertex   parabola in log2(eta) through the F(CHAT) argmin and its two neighbours; an argmin at an edge gives
           only a bound [edge, +inf) or (-inf, edge].
  q2       s = v20(500M) - v5(125M) as an interval: HOLDS if inside (-1, 1), MOVES if |s| >= 1 throughout,
           else UNRESOLVED. Secondary s at equal tokens: v20(250M) - v5(250M). Q3 vertices per length.
Output: one JSON object on stdout; the orchestrator records it in results.json before the next stage is queued.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

INF = float("inf")
GRID = {"A": [0.75e-3, 1.5e-3, 3e-3, 6e-3, 12e-3], "B": [0.5, 1.0, 2.0, 4.0, 8.0], "C": [0.5, 1.0, 2.0],
        "20m": [0.5, 1.0, 2.0]}
MAX_EXT = 2


def num(x: float) -> str:
    return f"{x:.6g}"


def run_name(size: str, eta: float, r: float, branch: str) -> str:
    return f"{size}_e{num(eta * 1e3)}_r{num(r)}_{branch}"


def read_run(runs: str, name: str):
    """-> {"chat", "prose"} (inf if diverged), "gap", or None (no result yet)."""
    d = os.path.join(runs, name)
    if os.path.exists(os.path.join(d, "DIVERGED")):
        return {"chat": INF, "prose": INF, "diverged": True}
    if os.path.exists(os.path.join(d, "GAP")):
        return "gap"
    p = os.path.join(d, "bpb.jsonl")
    if not os.path.exists(p):
        return None
    got = {}
    for ln in open(p, encoding="utf-8"):
        r = json.loads(ln)
        if r["ckpt"].startswith("final_") and r["split"] == "all" and r["set"] in ("CHAT", "PROSE"):
            got[r["set"].lower()] = float(r["bpb"])
    return got if len(got) == 2 else None


def axis_points(base: list[float], present: set[float]) -> list[float]:
    """The base grid plus existing factor-2 extensions beyond either edge, at most MAX_EXT in all."""
    pts, lo, hi = list(base), min(base), max(base)
    for _ in range(MAX_EXT):
        nxt = [x for x in (lo / 2, hi * 2) if any(math.isclose(x, p, rel_tol=1e-9) for p in present)]
        if not nxt:
            break
        x = nxt[0]
        pts.append(x)
        lo, hi = min(lo, x), max(hi, x)
    return sorted(pts)


def pick(points: dict) -> dict:
    """points: {x: result or "gap" or None}, x ascending = LR ascending. Refuses if any point has no result."""
    xs = sorted(points)
    missing = [x for x in xs if points[x] is None]
    if missing:
        return {"ready": False, "missing": missing}
    ok = [x for x in xs if points[x] != "gap"]
    chat = {x: round(points[x]["chat"], 4) if points[x]["chat"] < INF else INF for x in ok}
    arg = min(ok, key=lambda x: (chat[x], x))
    best_prose = min(points[x]["prose"] for x in ok)
    out = {"ready": True, "argmin": arg, "pick": arg, "guard_moved": False,
           "gap_next_to_argmin": any(points[x] == "gap" for x in xs
                                     if abs(xs.index(x) - xs.index(arg)) == 1)}
    if points[arg]["prose"] > 1.01 * best_prose:
        target = min(ok, key=lambda x: (points[x]["prose"], x))
        i = ok.index(arg) + (1 if ok.index(target) > ok.index(arg) else -1)
        out.update(pick=ok[i], guard_moved=True)
    out["at_edge"] = "low" if out["pick"] == xs[0] else "high" if out["pick"] == xs[-1] else None
    return out


def vertex(points: dict) -> dict:
    """points: {eta: F(CHAT)}; -> {"v": log2 eta or None, "lo", "hi"} (an edge argmin gives a bound)."""
    xs = sorted(x for x in points if points[x] is not None and points[x] != "gap")
    ys = [points[x]["chat"] if isinstance(points[x], dict) else points[x] for x in xs]
    i = min(range(len(xs)), key=lambda k: (ys[k], xs[k]))
    lx = [math.log2(x) for x in xs]
    if i == 0:
        return {"v": None, "lo": -INF, "hi": lx[0], "argmin_eta": xs[0], "bound": "at or below the low edge"}
    if i == len(xs) - 1:
        return {"v": None, "lo": lx[-1], "hi": INF, "argmin_eta": xs[-1], "bound": "at or above the high edge"}
    (x0, x1, x2), (y0, y1, y2) = lx[i - 1:i + 2], ys[i - 1:i + 2]
    if INF in (y0, y2):     # a diverged neighbour: no parabola; any curve with this minimum has it in (x0, x2)
        return {"v": None, "lo": x0, "hi": x2, "argmin_eta": xs[i], "bound": "a neighbour diverged: bracketed"}
    den =(x1 - x0) * (y2 - y1) - (x2 - x1) * (y1 - y0)
    num_ = (x1 - x0) ** 2 * (y2 - y1) + (x2 - x1) ** 2 * (y1 - y0)
    v = x1 - 0.5 * num_ / den if den else x1       # a flat triple gives the middle point
    return {"v": v, "lo": v, "hi": v, "argmin_eta": xs[i], "bound": None}


def runner_up(points: dict, at: float):
    """The neighbour of `at` on the axis with the lower F(CHAT); None when neither neighbour has a result."""
    xs = sorted(points)
    i = xs.index(at)
    nb = [xs[j] for j in (i - 1, i + 1) if 0 <= j < len(xs) and isinstance(points[xs[j]], dict)]
    key = lambda x: (round(float(points[x]["chat"]), 4) if points[x]["chat"] < INF else INF, x)   # noqa: E731
    return min(nb, key=key) if nb else None


def transfer(v5: dict, v20: dict) -> dict:
    s_lo, s_hi = v20["lo"] - v5["hi"], v20["hi"] - v5["lo"]
    if -1 < s_lo and s_hi < 1:
        verdict = "HOLDS"
    elif s_lo >= 1 or s_hi <= -1:
        verdict = "MOVES"
    else:
        verdict = "UNRESOLVED"
    return {"s": s_lo if s_lo == s_hi else None, "s_interval": [s_lo, s_hi], "verdict": verdict}


def stage(runs: str, which: str, eta_a: float | None = None, r_b: float | None = None) -> dict:
    present = {d for d in os.listdir(runs)} if os.path.isdir(runs) else set()

    def coord_run(x, branch):
        if which == "A":
            return run_name("5m", x, 1.0, branch)
        if which == "B":
            return run_name("5m", eta_a, x, branch)
        return run_name("5m", x * eta_a, r_b, branch)             # C: x = g
    have = set()
    for d in present:
        for x in [b * f for b in GRID[which] for f in (0.25, 0.5, 2, 4)]:
            if d == coord_run(x, "b250M"):
                have.add(x)
    xs = axis_points(GRID[which], have)
    res = pick({x: read_run(runs, coord_run(x, "b250M")) for x in xs})
    res.update(stage=which, axis=xs, runs={num(x): coord_run(x, "b250M") for x in xs})
    n_ext = len(xs) - len(GRID[which])
    if res.get("ready"):
        want = []
        if res["at_edge"]:
            want.append(xs[0] / 2 if res["at_edge"] == "low" else xs[-1] * 2)
        if which == "C":
            p125 = pick({x: read_run(runs, coord_run(x, "b125M")) for x in xs})
            res["argmin_125M"] = p125.get("argmin")
            if p125.get("ready") and p125["argmin"] in (xs[0], xs[-1]):
                want.append(xs[0] / 2 if p125["argmin"] == xs[0] else xs[-1] * 2)
            ru = runner_up({x: read_run(runs, coord_run(x, "b250M")) for x in xs}, res["pick"])
            res["runner_up"] = ru
            res["lrs"] = {"pick": [res["pick"] * eta_a, r_b], "runner_up": None if ru is None else [ru * eta_a, r_b]}
        want = sorted(set(want))[:max(0, MAX_EXT - n_ext)]
        res["extend_with"] = want
        res["extensions_used"] = n_ext
        res["decided"] = not want
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("which", choices=["A", "B", "C"])
    s.add_argument("--runs", required=True)
    s.add_argument("--eta-a", type=float)
    s.add_argument("--r-b", type=float)
    q = sub.add_parser("q2")
    q.add_argument("--runs", required=True)
    q.add_argument("--eta5", type=float, required=True)
    q.add_argument("--r5", type=float, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        out = stage(a.runs, a.which, a.eta_a, a.r_b)
    else:
        def curve(size, br, grid):
            return {g * a.eta5: read_run(a.runs, run_name(size, g * a.eta5, a.r5, br)) for g in grid}
        present = {g for g in [0.125, 0.25, 0.5, 1, 2, 4, 8]
                   if os.path.isdir(os.path.join(a.runs, run_name("20m", g * a.eta5, a.r5, "b500M")))}
        g20 = axis_points(GRID["20m"], present)
        g5 = [g for g in [0.125, 0.25, 0.5, 1, 2, 4, 8]
              if os.path.isdir(os.path.join(a.runs, run_name("5m", g * a.eta5, a.r5, "b250M")))]
        c = {f"5m_{br}": curve("5m", br, g5) for br in ("b62M", "b125M", "b250M")}
        c.update({f"20m_{br}": curve("20m", br, g20) for br in ("b250M", "b500M")})
        v = {k: vertex(pts) for k, pts in c.items() if all(p is not None for p in pts.values())}
        out = {"vertices": v}
        if "5m_b125M" in v and "20m_b500M" in v:
            out["primary"] = transfer(v["5m_b125M"], v["20m_b500M"])
        if "5m_b250M" in v and "20m_b250M" in v:
            out["secondary"] = transfer(v["5m_b250M"], v["20m_b250M"])
        p20 = c["20m_b500M"]
        if p20 and all(p is not None for p in p20.values()):
            etas, arg = sorted(p20), pick(p20)["argmin"]
            edge = etas[0] / 2 if arg == etas[0] else etas[-1] * 2 if arg == etas[-1] else None
            ru = runner_up(p20, arg)
            out["grid_20m"] = {"argmin_g": arg / a.eta5, "runner_up_g": None if ru is None else ru / a.eta5,
                               "extensions_used": len(g20) - len(GRID["20m"]),
                               "extend_with_g": [edge / a.eta5] if edge is not None and len(g20) - len(
                                   GRID["20m"]) < MAX_EXT else []}
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
