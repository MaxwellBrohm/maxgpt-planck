"""K0-g's reading (K1 notes STAGE 0, SPEC 11 [G-2]): the K-DEV objective of every K0-g run and the g pick at two
seeds. Built 2026-11-21 (K3 round 16), before any K run. Reads only K-DEV rows; nothing of any K-EVAL set.

  python k0read.py RUNS_DIR [--readings FILE]     RUNS_DIR holds k0g_g<g>_s<seed>/ (kconfig.py names)

Objective of one run (K1 notes K0-g: "the mean NLL on K-DEV, skill answer tokens and FB value tokens weighted
equally"): from the queue's scorer rows (probes.py on k_dev.jsonl, out_dir/k_eval.jsonl, set K-DEV), the mean
full-vocabulary NLL (ce_full, nats) of the scored answer token over the skill items (families R, U, B, F, A, P; one
row per item, its final question's gold) and the same mean over the FB value prompts (kind bio), averaged with
weight 1/2 each. A run with DIVERGED in its out_dir is None (SPEC 10: never picked, never dropped); a run neither
scored nor diverged is refused, as are rows of two different K-DEV files or a K-DEV count other than SPEC 8's 1,024
skill items plus 1,024 FB prompts. The pick is analyze_k.pick_g; the reading is appended to the readings file
(SPEC 12, append-only) with every objective, the table, the pick and the item file's sha256.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import analyze_k as A
import kconfig as KC

RUN_RE = re.compile(r"^k0g_g(?P<g>[0-9p]+)_s(?P<seed>\d+)$")
SKILL_FAMS = tuple("RUBFAP")
N_SKILL, N_FB = 1024, 1024                 # SPEC 8 K-DEV: 1,024 skill items + 1,024 FB prompts


def objective(out_dir: str) -> dict:
    """-> {"objective": float or None, "skill_nll", "fb_nll", "n_skill", "n_fb", "item_sha256"}."""
    if os.path.exists(os.path.join(out_dir, "DIVERGED")):
        return {"objective": None, "diverged": True}
    p = os.path.join(out_dir, "k_eval.jsonl")
    if not os.path.exists(p):
        raise ValueError(f"{out_dir}: neither scored (k_eval.jsonl) nor DIVERGED")
    rows = [r for r in (json.loads(x) for x in open(p)) if r.get("set") == "K-DEV"]
    shas = {r.get("item_sha256") for r in rows}
    if len(shas) != 1 or None in shas:
        raise ValueError(f"{out_dir}: K-DEV rows from {len(shas)} item files")
    sk = [r["ce_full"] for r in rows if r["fam"] in SKILL_FAMS]
    fb = [r["ce_full"] for r in rows if r["fam"] == "bio"]
    if (len(sk), len(fb), len(rows)) != (N_SKILL, N_FB, N_SKILL + N_FB):
        raise ValueError(f"{out_dir}: K-DEV rows {len(sk)} skill, {len(fb)} FB of {len(rows)}; want {N_SKILL} + {N_FB}")
    s, f = sum(sk) / len(sk), sum(fb) / len(fb)
    obj = 0.5 * s + 0.5 * f
    return {"objective": obj if math.isfinite(obj) else None, "skill_nll": s, "fb_nll": f, "n_skill": len(sk),
            "n_fb": len(fb), "item_sha256": shas.pop()}


def table(runs_dir: str) -> tuple[dict, dict]:
    """-> ({g: {seed: objective}}, {run: objective record}) over every K0-g run directory."""
    tab, per = {}, {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "k0g_g*_s*"))):
        m = RUN_RE.match(os.path.basename(d))
        if not m:
            continue
        rec = objective(d)
        per[os.path.basename(d)] = rec
        tab.setdefault(KC.g_of(m["g"]), {})[int(m["seed"])] = rec["objective"]
    items = {r["item_sha256"] for r in per.values() if r.get("item_sha256")}
    if len(items) > 1:
        raise ValueError(f"K0-g runs were scored on {len(items)} different K-DEV files")
    return tab, per


def read(runs_dir: str, readings: str = KC.READINGS) -> dict:
    tab, per = table(runs_dir)
    r = A.pick_g(tab)
    rec = {"reading": "K0-g", "g": r["g"], "final": r["final"], "extend": r["extend"], "labels": r["labels"],
           "mean": {f"{g:g}": v if math.isfinite(v) else "inf" for g, v in r["mean"].items()}, "per_seed": {str(s): g for s, g in r["per_seed"].items()},
           "table": {f"{g:g}": {str(s): v for s, v in d.items()} for g, d in sorted(tab.items())}, "runs": per}
    return A.append_reading(readings, rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--readings", default=KC.READINGS)
    a = ap.parse_args(argv)
    row = read(a.runs_dir, a.readings)
    print(json.dumps({k: row[k] for k in ("g", "final", "extend", "labels", "mean", "per_seed")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
