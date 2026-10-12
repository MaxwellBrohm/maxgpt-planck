"""SCREENS.txt AMENDMENT C6-DIAG at the stage 2 verdict step: its CHECKS (1)-(4), its READING RULE (c6read_lib.py) on
final IND, HD_asst and HD_user for every contrast of the batch (stage 1's five and stage 2's four), and what it reports
with no rule (HD_none, HD per turn, SINK, S001's paired SINK, GATE, the per-module readings, NEAR ITS INIT).
Written 2026-10-11, before any C6 value of any screen or E3 run was read (09:47:53 EDT), EXCEPT base_by_seed and the
CHECKS on base_s103 to 105 (BASE5), added at 09:51 after values were read (reported, no rule). Reads
<runs>/<run>/c6/*.json (the runner's output), the run's bpb.jsonl final CHAT (check 1) and diag.jsonl's last IND
(check 3); runs nothing.

  python c6read.py --runs DIR --verdicts FILE --part4d FILE --record FILE [--out FILE]
--verdicts: analyze.py verdicts --out; --part4d: E3 results_part4d.json; --record: {"ckpt_sha256": {run: sha},
"code_sha256": {path: sha}} (the final checkpoints hashed on the PC; the C6 queue's commit's code files).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import analyze as A
import c6read_lib as C
import screens_lib as L

sys.path.insert(0, L.E3D)
import e3analyze_4d as P4  # noqa: E402  (the PART 4d value getters, so both read the same field)

BASE5 = [f"base_s{s}" for s in range(101, 106)]   # AMENDMENT BASE-DIAG's five shared BASE seeds
REL_TOL = 1e-5                       # CHECK (1): hd.json's oasst2 bpb against the scored F(CHAT), relative


def load(runs: str, run: str, job: str) -> dict | None:
    p = os.path.join(runs, run, "c6", f"{job}.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def value(js: dict | None, metric: str) -> float | None:
    if js is None:
        return None
    try:
        v = float(P4.METRICS[metric][1](js["result"]))
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def scored_chat(runs: str, run: str) -> float | None:
    v = A.final(runs, run)
    return None if not v or v.get("diverged") else v["CHAT"]


def hook_final(runs: str, run: str) -> float | None:
    p = os.path.join(runs, run, "diag.jsonl")
    if not os.path.isfile(p):
        return None
    curve, _ = A.diag_problems(p)
    return next((v for s, v in curve if s == L.STEPS), None)


def checks(runs: str, run: str, rec: dict) -> dict:
    """C6-DIAG CHECKS AT THE VERDICT STEP for one run. (1) and (2) gate the run's readings; (3) is reported."""
    js = {j: load(runs, run, j) for j in ("ind", "hd", "attn", "modules", "halflife")}
    js = {j: v for j, v in js.items() if v is not None}
    why = []
    hd, chat = js.get("hd"), scored_chat(runs, run)
    c1 = None
    if hd is None or chat is None:
        why.append("check 1: no hd.json or no scored F(CHAT)")
    else:
        c1 = abs(hd["result"]["sets"]["oasst2"]["bpb"] - chat) / chat
        if not c1 <= REL_TOL:
            why.append(f"check 1: hd.json oasst2 bpb off the scored F(CHAT) by {c1:.2e} relative")
    want = rec["ckpt_sha256"].get(run)
    for j, v in sorted(js.items()):
        if want is None or v.get("checkpoint_sha256") != want:
            why.append(f"check 2: {j}.json checkpoint_sha256 is not the run's final checkpoint")
        if v.get("code_sha256") != rec["code_sha256"]:
            why.append(f"check 2: {j}.json code_sha256 is not the launched commit's")
    ind, hook = value(js.get("ind"), "IND"), hook_final(runs, run)
    c3 = abs(ind - hook) if ind is not None and hook is not None else None
    return {"ok": not why, "why": why, "check1_rel": c1, "check3_ind_minus_hook_abs": c3, "jobs": sorted(js)}


def contrast_runs(picks: dict) -> dict:
    """{key: (sid, class, arm run name(seed), BASE run name(seed))} for every contrast with a pick (C3)."""
    out = {}
    for sid, s in L.SCREENS.items():
        base = (lambda x, p=sid.lower(): f"{p}_base_s{x}") if L.engine(sid) else (lambda x: f"base_s{x}")
        for arm in s["arms"]:
            if sid == "S003":
                c = picks["S003"].get("C", {})
                if not c.get("decided"):
                    continue
                e, r = c["lrs"]["pick"]
                name = lambda x, e=e, r=r: f"s003_adamw_e{L.num(e * 1e3)}_r{L.num(r)}_s{x}"  # noqa: E731
            else:
                pk = picks.get(f"{sid}.{arm}", {})
                if not pk.get("decided"):
                    continue
                name = lambda x, sid=sid, arm=arm, g=pk["pick"]: A.arm_name(sid, arm, g, x)  # noqa: E731
            out[f"{sid}.{arm}"] = (sid, s["cls"], name, base)
    return out


def mean_heads(rows: list | None) -> list | None:
    return None if rows is None else [None if r is None else sum(r) / len(r) for r in rows]


def readings(runs: str, verdicts: dict, part4d: dict, rec: dict) -> dict:
    seeds = [101, 102]
    cr = {k: v for k, v in contrast_runs(verdicts["picks"]).items() if k in verdicts["contrasts"]}
    names = sorted({f(s) for _, _, a, b in cr.values() for f in (a, b) for s in seeds} | set(BASE5))
    chk = {n: checks(runs, n, rec) for n in names}
    ok = lambda n: chk[n]["ok"]  # noqa: E731
    out: dict = {"rule": "SCREENS.txt AMENDMENT C6-DIAG, READING RULE", "checks": chk, "metrics": {}}
    classes = {k: v[1] for k, v in cr.items()}
    for m in C.RULED + ("HDn_asst", "HDn_user"):
        job = P4.METRICS[m][0]
        vals = {k: tuple({s: value(load(runs, f(s), job), m) if ok(f(s)) else None for s in seeds} for f in (a, b))
                for k, (_, _, a, b) in cr.items()}
        noise = part4d["metrics"][m]["part1_5M"]["noise"]
        if m in C.RULED:
            out["metrics"][m] = {**C.read_metric(vals, classes, noise), "part4d_noise": noise}
        else:                        # HD_none: reported beside HD, never read by the rule (C6)
            out["metrics"][m] = {"contrasts": {k: {**C.descriptive(C.paired(a, b)[1]), "arm": a, "base": b}
                                               for k, (a, b) in vals.items()}, "part4d_noise": noise}
    chat = verdicts["contrasts"].get("S005.forget", {}).get("readings", {}).get("CHAT", {}).get("label")
    hd5 = out["metrics"]["HD_asst"]["contrasts"].get("S005.forget")
    out["s005_flag"] = {"chat_label": chat, "hd_asst_label": (hd5 or {}).get("label"),
                        "flagged": C.s005_flag(chat or "", hd5)}
    out["scorer_replay"] = {m: part4d["metrics"][m]["part1_5M"].get("scorer_replay") for m in P4.METRICS}
    out["reported"] = reported(runs, cr, seeds, ok)
    out["base_by_seed"] = {m: {"base": {n: value(load(runs, n, P4.METRICS[m][0]), m) if ok(n) else None for n in BASE5},
                               "e3_arm_a": [v[0] for v in part4d["metrics"][m]["part1_5M"]["values"].values()]}
                           for m in C.RULED}       # reported, no rule: the shared BASE's five seeds beside E3 arm A
    return out


def reported(runs: str, cr: dict, seeds: list, ok) -> dict:
    """REPORTED, NO RULE: per run SINK (head means per layer), GATE (S001's BASE side), HD per turn and role, the own,
    foreign and none bpb; per screen the per-module readings and NEAR ITS INIT; S001's paired SINK per layer."""
    rep: dict = {"runs": {}, "modules": {}, "near_init": {}}
    for k, (sid, _, a, b) in cr.items():
        for f in (a, b):
            for s in seeds:
                n = f(s)
                if n in rep["runs"] or not ok(n):
                    continue
                at, hd = load(runs, n, "attn"), load(runs, n, "hd")
                r = {}
                if at:
                    x = at["result"]
                    r.update(sink=x.get("sink"), sink_layer_mean=mean_heads(x.get("sink")),
                             gate_mean=x.get("gate_mean"), gate_p10=x.get("gate_p10"), gate_p90=x.get("gate_p90"))
                if hd:
                    h = hd["result"]["sets"]["oasst2"]["hd"]
                    r.update(hd_by_role=h["by_role"], hd_by_turn=h["by_turn"],
                             hd_counts={c: h[c] for c in ("windows", "kept", "first_turn", "no_history", "no_foreign")
                                        if c in h})
                rep["runs"][n] = r
        if sid in C.NEAR:
            job = "halflife" if sid == "S005" else "modules"
            mods = {s: load(runs, a(s), job) for s in seeds}
            if all(v is not None and ok(a(s)) for s, v in mods.items()):
                rep["modules"][k] = {s: v["result"] for s, v in mods.items()}
                rep["near_init"][k] = C.near_init(sid, rep["modules"][k])
            else:
                rep["near_init"][k] = {"fires": None, "why": "a seed's JSON missing or its run failed a check"}
    if "S001.nogate" in cr:
        _, _, a, b = cr["S001.nogate"]
        per = {}
        for s in seeds:
            on, off = (rep["runs"].get(b(s), {}).get("sink_layer_mean"), rep["runs"].get(a(s), {}).get("sink_layer_mean"))
            per[s] = None if on is None or off is None else [x - y for x, y in zip(on, off)]
        nl = max((len(v) for v in per.values() if v), default=0)
        rep["s001_sink_paired"] = {"gate_on_minus_off_per_seed": per, "seeds_negative_per_layer":
                                   [[s for s in seeds if per[s] and per[s][i] < 0] for i in range(nl)]}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--runs", "--verdicts", "--part4d", "--record"):
        ap.add_argument(k, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = readings(a.runs, json.load(open(a.verdicts)), json.load(open(a.part4d)), json.load(open(a.record)))
    txt = json.dumps(res, indent=1, sort_keys=True, default=str)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
