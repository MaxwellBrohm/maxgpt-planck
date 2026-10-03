"""SCREENS analyzer: C2's run-directory RC-12 refusal, C3 picks, C7 verdicts with C4's noise checks and the Holm
reading (experiments/SCREENS.txt). Reads run directories only (~/planck/runs/SCREENS or a copy); runs nothing.

  python analyze.py picks --runs DIR                       C3: every arm's g pick at seed 1, S003's stages A-C
  python analyze.py verdicts --runs DIR [--out FILE]       C7 for every arm that has seed runs
Before anything else the whole runs directory is refused if any run directory in it holds rc12_eval.jsonl (RC-12
scored before the lock), and every run read is checked again. Values: the final checkpoint's bpb from
<run>/bpb.jsonl (E2's bpb_lines.py: sets CHAT and PROSE, and the PROSE sources cccc, gutenberg, wikimedia; split
all); a DIVERGED file = diverged (+inf, never dropped); a GAP or a missing run = not there.
C3 is E2's rule through E2's own e2pick.py (pick, PROSE guard, ties to the lower LR; one factor-2 point beyond an
edge pick, at most 2). Seeds and k: screens_lib.params() (E3 results.json, pinned by sha256).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager

import analyze_lib as AL
import screens_lib as L

import e2pick  # noqa: E402  (screens_lib puts E2's directory on sys.path)

GRID = [0.5, 1.0, 2.0]
SOURCES = {"CHAT": "CHAT", "PROSE": "PROSE", "cccc": "cccc", "gutenberg": "gutenberg", "wikimedia": "wikimedia"}


class Refused(RuntimeError):
    pass


def rc12_guard(runs: str) -> None:
    """C2 RC-12 GUARD, the run-directory half: refuse if any run directory holds rc12_eval.jsonl."""
    bad = sorted(d for d in os.listdir(runs) if os.path.isfile(os.path.join(runs, d, "rc12_eval.jsonl")))
    if bad:
        raise Refused(f"RC-12 GUARD: rc12_eval.jsonl in {bad} (RC-12 scored before the lock); nothing is read")


def final(runs: str, name: str) -> dict | None:
    d = os.path.join(runs, name)
    if os.path.isfile(os.path.join(d, "rc12_eval.jsonl")):
        raise Refused(f"RC-12 GUARD: {name} holds rc12_eval.jsonl")
    if os.path.exists(os.path.join(d, "DIVERGED")):
        return {"diverged": True}
    p = os.path.join(d, "bpb.jsonl")
    if os.path.exists(os.path.join(d, "GAP")) or not os.path.exists(p):
        return None
    got = {}
    for ln in open(p, encoding="utf-8"):
        r = json.loads(ln)
        if r["ckpt"].startswith("final_") and r["split"] == "all" and r["set"] in SOURCES.values():
            got[r["set"]] = float(r["bpb"])
    return {k: got[v] for k, v in SOURCES.items()} if all(v in got for v in SOURCES.values()) else None


def arm_name(sid: str, arm: str, g: float, seed: int) -> str:
    return f"{sid.lower()}_{arm}_g{L.num(g)}_s{seed}"


def c3_pick(runs: str, sid: str, arm: str) -> dict:
    """E2's pick on the g axis {0.5, 1, 2} plus the extensions present (at most 2), seed 1."""
    present = {g for g in L.G_ALLOWED if os.path.isdir(os.path.join(runs, arm_name(sid, arm, g, 1)))}
    xs = e2pick.axis_points(GRID, present)
    for g in xs:
        final(runs, arm_name(sid, arm, g, 1))                  # the RC-12 check on every run read
    res = e2pick.pick({g: e2pick.read_run(runs, arm_name(sid, arm, g, 1)) for g in xs})
    res.update(axis=xs, extensions_used=len(xs) - len(GRID))
    if res.get("ready"):
        want = [] if not res["at_edge"] else [xs[0] / 2 if res["at_edge"] == "low" else xs[-1] * 2]
        res["extend_with"] = want if res["extensions_used"] < e2pick.MAX_EXT else []
        res["decided"] = not res["extend_with"]
    return res


@contextmanager
def s003_names():
    """e2pick's 5M run names -> this batch's S003 names (screens.py s003_arm: s003_adamw + E2's name minus "5m")."""
    real = e2pick.run_name
    e2pick.run_name = lambda size, eta, r, branch: "s003_adamw" + real(size, eta, r, branch)[2:]
    try:
        yield
    finally:
        e2pick.run_name = real


def s003_pick(runs: str) -> dict:
    """S003's E2 search (stage A at r 1, B at eta_A, C at (eta_A, r_B)), each stage once the previous decided."""
    out = {}
    with s003_names():
        a = out["A"] = e2pick.stage(runs, "A", None, None)
        if a.get("decided"):
            b = out["B"] = e2pick.stage(runs, "B", a["pick"], None)
            if b.get("decided"):
                out["C"] = e2pick.stage(runs, "C", a["pick"], b["pick"])
    return out


def picks(runs: str) -> dict:
    rc12_guard(runs)
    out = {f"{sid}.{a}": c3_pick(runs, sid, a) for sid, s in L.SCREENS.items() if sid != "S003" for a in s["arms"]}
    out["S003"] = s003_pick(runs)
    return out


def verdicts(runs: str, p: dict, noise: dict, g_pick: dict, s003_lr: tuple | None) -> dict:
    """C7 for every arm with seed runs at its pick, then C4's noise checks, then the Holm reading."""
    rc12_guard(runs)
    res, vals = {}, {}
    for sid, s in L.SCREENS.items():
        seeds = p["seeds"][s["cls"]]
        bname = (lambda x: f"{sid.lower()}_base_s{x}") if L.engine(sid) else (lambda x: f"base_s{x}")
        base = {x: v for x in seeds if (v := final(runs, bname(x))) is not None}
        for a in s["arms"]:
            if sid == "S003":
                if s003_lr is None:
                    continue
                name = lambda x: f"s003_adamw_e{L.num(s003_lr[0] * 1e3)}_r{L.num(s003_lr[1])}_s{x}"  # noqa: E731
            elif f"{sid}.{a}" in g_pick:
                name = lambda x, g=g_pick[f"{sid}.{a}"]: arm_name(sid, a, g, x)  # noqa: E731
            else:
                continue
            arm = {x: v for x in seeds if (v := final(runs, name(x))) is not None}
            if not arm:
                continue
            key = f"{sid}.{a}"
            res[key] = AL.contrast(arm, base, s["cls"], noise, p["u"])
            vals[key] = (arm, base)
    for key, c in list(res.items()):                      # C4 same init: per arm
        if c["class"] == "same init" and c["readings"]:
            chk = AL.same_init_check(c, noise)
            if not all(x["ok"] for x in chk.values()):
                res[key] = AL.reread_new_init(c, *vals[key], noise, p["u"], "noise check failed")
            res[key]["noise_check"] = chk
    sia = [k for k, c in res.items() if c["class"] == "SIA" and c["readings"]]
    pooled = AL.pooled_sia_check([res[k] for k in sia], noise, p["u"])
    if not all(x["ok"] for x in pooled.values()):
        for k in sia:
            res[k] = AL.reread_new_init(res[k], *vals[k], noise, p["u"], "noise check failed (pooled SIA)")
    ps = {k: c["readings"]["CHAT"]["p_two_sided"] for k, c in res.items() if c["readings"]}
    hm = AL.holm(ps)
    for k, c in res.items():
        c["holm"] = None if k not in hm else ("holds" if hm[k] else "does not hold")
    return {"contrasts": res, "pooled_sia_check": pooled, "holm_family": sorted(ps)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("picks", "verdicts"):
        x = sub.add_parser(c)
        x.add_argument("--runs", required=True)
        x.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        out = picks(a.runs)
        if a.cmd == "verdicts":
            p, e3 = L.params(), json.load(open(os.path.join(L.E3D, "results.json")))
            g = {k: v["pick"] for k, v in out.items() if k != "S003" and v.get("decided")}
            c = out["S003"].get("C", {})
            s3 = tuple(c["lrs"]["pick"]) if c.get("decided") else None
            out = {"picks": out, **verdicts(a.runs, p, e3["noise_5M_250M"], g, s3)}
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    txt = json.dumps(out, indent=1, sort_keys=True, default=str)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
