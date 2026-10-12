"""SCREENS.txt STAGE 2 VERDICTS (2026-10-11): writes each stage 2 screen's results.json and out/<run>/ copies, and the
batch-wide files beside this script, from the outputs of analyze.py verdicts, c6read.py and basereplay.py on a copy of
the PC run directories. Reads JSON and run files only; no model.

  python stage2_report.py --runs DIR --verdicts F --c6 F --replay F --record F [--dest EXPERIMENTS_DIR]
Writes: experiments/S004_canon, S005_forget_gate, S006_mtp_aux, S007_smeared_key/{results.json, out/<run>/};
experiments/screens/{stage2_verdicts.json, c6_readings.json, base_replay.json, c6_record.json, c6_out/<run>/*.json,
base_out/<run>/}. stage2_tables.py writes the tables from these files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import analyze as A
import analyze_lib as AL
import report_lib as R
import screens_lib as L

STAGE2 = {"S004": "canonac", "S005": "forget", "S006": "mtp", "S007": "smear"}
ENTRY = "SCREENS.txt STAGE 2 VERDICTS (2026-10-11)"
READING = ("a 5M loss reading only (C8): F = bits per byte of the final checkpoint at 5M, 250.0M slots, starter mix, "
           "tok_v0_8k; no RC-12 split and no Tier 0 margin was scored (C8 waits for the lock)")
BASE_EXTRA = ["base_s103", "base_s104", "base_s105", "base_s101_rep1", "base_s101_hookoff", "base_s101_rep2"]
FILES = ("log.jsonl", "bpb.jsonl", "diag.jsonl")
HOLM_NOTE = ("family so far: the 9 contrasts of the batch that ran (stage 1's 5, stage 2's 4); C8's Tier 0 tests join "
             "it after the lock")


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def runs_of(sid: str, pick: float) -> tuple[list[str], list[str], list[str]]:
    """(arm runs, BASE runs, IND-only runs) of a stage 2 screen at seeds 101 and 102."""
    arm = STAGE2[sid]
    a = [f"{sid.lower()}_{arm}_g{L.num(pick)}_s{s}" for s in (101, 102)]
    b = [f"s005_base_s{s}" if sid == "S005" else f"base_s{s}" for s in (101, 102)]
    ind = [f"s006_mtp_g1_s{s}" for s in (101, 102)] if sid == "S006" else []
    return a, b, ind


def registered(sid: str, verdict: str) -> str:
    """The verdict's line(s) under METRIC AND DECISION RULE in the screen's notes.txt, verbatim (spaces joined)."""
    txt = open(os.path.join(os.path.dirname(L.HERE), L.SCREENS[sid]["dir"], "notes.txt")).read()
    sec = txt.split("METRIC AND DECISION RULE", 1)[1].split("\nDIAGNOSTICS", 1)[0]
    m = re.search(rf"^    {re.escape(verdict)} .*?(?=^    [A-Z]|\Z)", sec, re.S | re.M)
    return " ".join(m.group(0).split()) if m else f"{verdict}: as C7"


def at_e3_mean(c: dict, e3: dict) -> str:
    """The verdict with SD_ref, thr and delta scaled to E3 arm A's mean (STAGE 1 VERDICTS' BASE LEVEL check)."""
    rd = {}
    for k, m in AL.METRICS.items():
        r, mean = c["readings"][k], e3[m]["mean_a"]
        rd[k] = AL.metric_reading(r["d"], r["sd_ref"] * mean / r["base_mean"], r["df"], mean)
    return AL.arm_verdict(rd)[0]


def copy_run(src_runs: str, run: str, dst: str) -> dict:
    os.makedirs(dst, exist_ok=True)
    got = {}
    for f in FILES:
        p = os.path.join(src_runs, run, f)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(dst, f))
            got[f"{run}/{f}"] = sha(p)
    return got


def sens_part(v: dict, key: str) -> dict:
    """The metric's sensitivity block (c6read_lib.sensitivity, no rule) for one re-read contrast; {} if not re-read."""
    s = v.get("sensitivity") or {}
    return {f: s[f]["readings"][key] for f in ("at_registered_sd", "at_pooled_sd") if key in s.get(f, {}).get("readings", {})}


def c6_part(c6: dict, key: str, runs: list[str]) -> dict:
    out = {m: {**v["contrasts"][key], "noise_checks": v.get("noise_checks"), "holm_family": v.get("holm_family")}
           | ({"sensitivity": sp} if (sp := sens_part(v, key)) else {})
           for m, v in c6["metrics"].items() if key in v["contrasts"]}
    rep = c6["reported"]
    return {"rule": c6["rule"], "readings": out, "checks": {n: c6["checks"][n] for n in runs if n in c6["checks"]},
            "near_init": rep["near_init"].get(key), "modules": rep["modules"].get(key),
            "runs": {n: rep["runs"][n] for n in runs if n in rep["runs"]}}


def screen(sid: str, a: argparse.Namespace, V: dict, C6: dict, BR: dict, e3: dict, src: dict) -> dict:
    key = f"{sid}.{STAGE2[sid]}"
    c = V["contrasts"][key]
    pick = V["picks"][key]["pick"]
    arm, base, ind = runs_of(sid, pick)
    d = os.path.join(a.dest, L.SCREENS[sid]["dir"])
    files = {}
    for n in arm + base + ind:
        files.update(copy_run(a.runs, n, os.path.join(d, "out", n)))
    c5 = R.c5(a.runs, arm, base)
    res = {"screen": sid, "dir": L.SCREENS[sid]["dir"], "entry": ENTRY, "reading": READING,
           "source": {**src, "files_sha256": files}, "class": c["class"], "k": len(c["seeds"]), "seeds": c["seeds"],
           "pick": f"g {L.num(pick)} (C3 g-check, seed 1)", "contrasts": {key: c},
           "holm": {"family": V["holm_family"], "note": HOLM_NOTE,
                    "p_chat": {k: x["readings"]["CHAT"]["p_two_sided"] for k, x in V["contrasts"].items()}},
           "pooled_sia_check": V["pooled_sia_check"], "c5_reported": c5,
           "clip": {n: R.clip(os.path.join(a.runs, n)) for n in arm + base + ind},
           "tok_s": {n: R.tok_s(os.path.join(a.runs, n)) for n in arm + base + ind},
           "onset": V["onsets"]["contrasts"].get(key, "not read: S005 pairs with its own BASE (AMENDMENT BASE-DIAG)"),
           "onset_sigma": V["onsets"]["sigma"], "c6": c6_part(C6, key, arm + base),
           "base_replay": {"label": BR["labels"][key], "onset_label": BR["onset_labels"].get(key, "not applicable"),
                           "D": {k: BR["readings"][k]["D"] for k in ("CHAT", "PROSE")}},
           "verdict_at_e3_arm_a_mean": at_e3_mean(c, e3["noise_5M_250M"]),
           "verdicts": {key: c["verdict"]}, "consequence_registered": registered(sid, c["verdict"]),
           "out": {"files": sorted(f"out/{f}" for f in files),
                   "note": "C9 outputs: copied from the PC run directories on 2026-10-11 (read only), each sha256 equal "
                           "on both machines and to source.files_sha256; C6 JSON in experiments/screens/c6_out"}}
    if sid == "S006":
        res["mtp_parts"] = {n: R.mtp_parts(os.path.join(a.runs, n)) for n in arm + ind}
        res["ind_only_runs"] = ind
        res["ind_only_final_ind"] = {n: dict(A.ind_curve(a.runs, n)[1]).get(L.STEPS) for n in ind}   # the hook, 7,630
    if sid == "S005":
        res["s005_flag"] = C6["s005_flag"]
        h = {n: A.onset_half(a.runs, n)[1] for n in arm + base}       # C6: S005's onset stays descriptive
        res["onset_descriptive"] = {"arm_onset_half": [h[n] for n in arm], "base_onset_half": [h[n] for n in base],
                                    **AL.onset_descriptive([h[x] - h[y] for x, y in zip(arm, base)])}
    json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(d, "results.json"), "a").write("\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--runs", "--verdicts", "--c6", "--replay", "--record", "--source", "--pcsha"):
        ap.add_argument(k, required=True)
    ap.add_argument("--dest", default=os.path.dirname(L.HERE))
    a = ap.parse_args(argv)
    V, C6, BR = (json.load(open(p)) for p in (a.verdicts, a.c6, a.replay))
    e3, src = json.load(open(os.path.join(L.E3D, "results.json"))), json.load(open(a.source))
    sx = os.path.join(a.dest, "screens")
    for name, obj in (("stage2_verdicts.json", V), ("c6_readings.json", C6), ("base_replay.json", BR),
                      ("c6_record.json", json.load(open(a.record)))):
        with open(os.path.join(sx, name), "w") as f:
            f.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    for n in sorted(C6["checks"]):
        for j in C6["checks"][n]["jobs"]:
            os.makedirs(os.path.join(sx, "c6_out", n), exist_ok=True)
            shutil.copyfile(os.path.join(a.runs, n, "c6", f"{j}.json"), os.path.join(sx, "c6_out", n, f"{j}.json"))
    for n in BASE_EXTRA:
        copy_run(a.runs, n, os.path.join(sx, "base_out", n))
    for sid in STAGE2:
        screen(sid, a, V, C6, BR, e3, src)
    pc = {}
    for ln in open(a.pcsha):                       # the PC's sha256sum of every copied file, SCREENS/<run>/<file>
        h, p = ln.split()
        pc[p.removeprefix("SCREENS/")] = h
    want = sorted(r for r in pc if r.split("/")[0] in set(C6["checks"]) | set(BASE_EXTRA) | {"s006_mtp_g1_s101",
                                                                                             "s006_mtp_g1_s102"})
    with open(os.path.join(sx, "stage2_source.json"), "w") as f:
        f.write(json.dumps({**src, "pc_files_sha256": {r: pc[r] for r in want if r.split("/")[-1] in FILES
                                                       or "/c6/" in r}}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
