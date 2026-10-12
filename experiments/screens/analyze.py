"""SCREENS analyzer: C2's run-directory RC-12 refusal, C3 picks, C7 verdicts with C4's noise checks and the Holm
reading (experiments/SCREENS.txt). Reads run directories only (~/planck/runs/SCREENS or a copy); runs nothing.

  python analyze.py picks --runs DIR                       C3: every arm's g pick at seed 1, S003's stages A-C
  python analyze.py verdicts --runs DIR [--out FILE] [--earlier JSON ...] [--no-stage1-earlier]
                                                           C7 for every arm that has seed runs, and the onsets
--earlier: a results.json (or an earlier verdicts --out file) whose "contrasts" were read in an earlier verdict step.
Stage 1's three results.json files (S001, S002, S003; read 2026-10-06) are always among them unless
--no-stage1-earlier is given (since 2026-10-10, SCREENS.txt ONSET RULE CODE; before, they had to be passed by hand).
C4: if the pooled SIA check fails, every SIA arm is re-read at the new-init SD; a verdict named there is
re-labelled "noise check failed later" (C4's text), an arm read for the first time "noise check failed (pooled SIA)".
"onsets" (AMENDMENT BASE-DIAG's ONSET RULE): onset_half of S004 canonac and S007 smear at g 1 and of S006 mtp's g 1
IND-only runs against the shared BASE at seeds 101 and 102, read with C7 at sigma_seed from BASE seeds 101-105,
with a Holm reading over those contrasts; descriptive (C6) when a needed run or onset is missing. No verdict moves.
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
import math
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


LATER, POOLED = "noise check failed later", "noise check failed (pooled SIA)"     # C4's label; this step's own
EXP = os.path.dirname(L.HERE)
STAGE1_RESULTS = [os.path.join(EXP, L.SCREENS[s]["dir"], "results.json") for s in ("S001", "S002", "S003")]


def earlier_keys(paths: list[str]) -> set[str]:
    """The contrasts that earlier verdict steps read: the "contrasts" keys of each results.json given."""
    for p in paths:
        if not os.path.isfile(p):
            raise Refused(f"--earlier: no file {p} (stage 1's are read by default; --no-stage1-earlier skips them)")
    return {k for p in paths for k in json.load(open(p))["contrasts"]}


ONSET_ARMS = {"S004.canonac": "pick", "S006.mtp": "ind", "S007.smear": "ind"}   # AMENDMENT BASE-DIAG's three
IND_EVERY = 153                                     # configs/screens_base.yaml eval.induction.every (tested equal)
IND_GRID = frozenset(range(IND_EVERY, L.STEPS, IND_EVERY)) | {L.STEPS}   # the hook's steps: 50 at 5M


def diag_problems(path: str) -> tuple[list, list[str]]:
    """([(step, IND)] from each step's last reading, problems) of a diag.jsonl (ONSET RULE CODE (a), (b) and its
    CORRECTION 1). NUL bytes are dropped (a power cut's tail; the hook's json never holds one). A problem: a line that
    is not a JSON object with an integer step, a step off IND_GRID, a grid step with no reading, or a step whose last
    reading is an error line (induction.py logs {"step", "error"} when a reading raises) or has no finite number ind."""
    by, probs = {}, []
    with open(path, "rb") as f:
        raw = f.read().replace(b"\0", b"").split(b"\n")
    for i, ln in enumerate(raw, 1):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
            ok = isinstance(r, dict) and type(r.get("step")) is int
        except ValueError:
            ok = False
        if not ok:
            probs.append(f"line {i} is not a reading")
            continue
        by[r["step"]] = r
    if sorted(set(by) - IND_GRID):
        probs.append(f"steps off the grid {sorted(set(by) - IND_GRID)}")
    if sorted(IND_GRID - set(by)):
        probs.append(f"no reading at steps {sorted(IND_GRID - set(by))}")
    curve = []
    for s, r in sorted(by.items()):
        x = r.get("ind")
        if "error" in r or type(x) not in (int, float) or not math.isfinite(x):
            probs.append(f"step {s}: " + ("error line" if "error" in r else "no finite ind"))
        else:
            curve.append((s, float(x)))
    return curve, probs


def ind_curve(runs: str, name: str) -> tuple[str, list]:
    """(state, [(step, IND)]) of a run: "missing" (no directory, GAP, no final bpb), "diverged", "diag incomplete"
    (no diag.jsonl, or any of diag_problems: so every grid step through the last, 7,630, has a finite reading) or "ok".
    A resumed run's diag.jsonl can repeat steps: the last reading of a step is the one on the trained path."""
    v = final(runs, name)
    if v is None:
        return "missing", []
    if v.get("diverged"):
        return "diverged", []
    p = os.path.join(runs, name, "diag.jsonl")
    if not os.path.isfile(p):
        return "diag incomplete", []
    curve, probs = diag_problems(p)
    return ("diag incomplete" if probs else "ok"), curve


def diag_why(runs: str, name: str) -> list[str]:
    p = os.path.join(runs, name, "diag.jsonl")
    return diag_problems(p)[1] if os.path.isfile(p) else ["no diag.jsonl"]


def onset_half(runs: str, name: str) -> tuple[str, int | None]:
    state, curve = ind_curve(runs, name)
    if state != "ok":
        return state, None
    h = AL.onsets(curve, None)["onset_half"]
    return ("ok", h) if h is not None else ("final IND <= 0", None)


def onset_readings(runs: str, p: dict, g_pick: dict) -> dict:
    """AMENDMENT BASE-DIAG's ONSET RULE (SCREENS.txt). sigma_seed from base_s101..105; each of ONSET_ARMS against
    base_s101, base_s102 at g 1 (S004 only when its pick is g 1: else LR-confounded, C6). A needed run missing,
    diverged, unscored or without a full IND curve, or an undefined onset_half (final IND <= 0): descriptive."""
    base = {s: onset_half(runs, f"base_s{s}") for s in AL.ONSET_SEEDS}
    sig = AL.onset_sigma({s: h for s, (st_, h) in base.items() if st_ == "ok"})
    out, ps = {}, {}
    for key, how in ONSET_ARMS.items():
        sid, arm = key.split(".")
        seeds = p["seeds"][L.SCREENS[sid]["cls"]]
        if how == "pick" and g_pick.get(key) != 1.0:
            out[key] = {"reading": "LR-confounded, not read", "why": f"pick {g_pick.get(key)} is not g 1 (C6)"}
            continue
        a = {s: onset_half(runs, arm_name(sid, arm, 1.0, s)) for s in seeds}
        d = [a[s][1] - base[s][1] for s in seeds if a[s][1] is not None and base[s][1] is not None]
        why = [f"{arm_name(sid, arm, 1.0, s)} {a[s][0]}" for s in seeds if a[s][1] is None]
        why += [f"base_s{s} {base[s][0]}" for s in seeds if base[s][1] is None]
        if sig["sigma_seed"] is None:
            why.append(sig["why"])
        rec = {"arm_onset_half": {s: a[s][1] for s in seeds}, "base_onset_half": {s: base[s][1] for s in seeds},
               "runs": [arm_name(sid, arm, 1.0, s) for s in seeds]}
        if why:
            out[key] = {**rec, "reading": "descriptive", "why": why, **AL.onset_descriptive(d)}
            continue
        out[key] = {**rec, "reading": "rule", **AL.onset_reading(d, sig["sigma_seed"], sig["df"])}
        ps[key] = out[key]["p_two_sided"]
    hm = AL.holm(ps)
    for k, c in out.items():
        c["holm"] = None if k not in hm else ("holds" if hm[k] else "does not hold")
    read = [f"base_s{s}" for s in base] + [n for c in out.values() for n in c.get("runs", [])]
    probs = {n: diag_why(runs, n) for n in read if ind_curve(runs, n)[0] == "diag incomplete"}
    return {"rule": "SCREENS.txt AMENDMENT BASE-DIAG, ONSET RULE", "sigma": sig,
            "base_states": {s: st_ for s, (st_, _) in base.items()}, "contrasts": out, "holm_family": sorted(ps),
            "diag_problems": probs}


def verdicts(runs: str, p: dict, noise: dict, g_pick: dict, s003_lr: tuple | None, earlier=frozenset()) -> dict:
    """C7 for every arm with seed runs at its pick, then C4's noise checks, then the Holm reading. earlier: the
    contrast keys read in an earlier verdict step (C4 re-labels those "noise check failed later")."""
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
            res[k] = AL.reread_new_init(res[k], *vals[k], noise, p["u"], LATER if k in earlier else POOLED)
    ps = {k: c["readings"]["CHAT"]["p_two_sided"] for k, c in res.items() if c["readings"]}
    hm = AL.holm(ps)
    for k, c in res.items():
        c["holm"] = None if k not in hm else ("holds" if hm[k] else "does not hold")
    return {"contrasts": res, "pooled_sia_check": pooled, "holm_family": sorted(ps), "read_earlier": sorted(earlier),
            "onsets": onset_readings(runs, p, g_pick)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("picks", "verdicts"):
        x = sub.add_parser(c)
        x.add_argument("--runs", required=True)
        x.add_argument("--out", default=None)
        x.add_argument("--earlier", action="append", default=[])
        x.add_argument("--no-stage1-earlier", action="store_true")
    a = ap.parse_args(argv)
    try:
        out = picks(a.runs)
        if a.cmd == "verdicts":
            p, e3 = L.params(), json.load(open(os.path.join(L.E3D, "results.json")))
            g = {k: v["pick"] for k, v in out.items() if k != "S003" and v.get("decided")}
            c = out["S003"].get("C", {})
            s3 = tuple(c["lrs"]["pick"]) if c.get("decided") else None
            ear = ([] if a.no_stage1_earlier else STAGE1_RESULTS) + a.earlier
            out = {"picks": out, **verdicts(a.runs, p, e3["noise_5M_250M"], g, s3, earlier_keys(ear))}
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
