"""Track K's preflight (K3 round 16, 2026-11-21): E2's preflight.py --strict on a K run config, with its one rule
that cannot hold for K replaced by K's own, plus the K data root's gate. Runs on the PC before each K run (queue_k.sh).

  python kpreflight.py CONFIG --code CODE_ROOT --data DATA_ROOT [--out report.json]

E2's check_one runs unchanged (prereg committed, provenance hashes, every source glob matching files, the parameter
count, the engine keys, engine_fixed, the scorer invocation, the schedule). Its rule "data sources differ from the
shard manifest" compares the whole source list with the starter corpus's; a K config adds the K sources, so that one
refusal is dropped and K's rules stand in for it:
  lang       every config source the shard manifest names has the manifest's kind, eot id and path (its last two
             parts), in the manifest's order, and every one of the manifest's sources is there (CAP aside: no LANG)
  check      DATA_ROOT/check.json exists, passed (gen.py check exit 0), and its code_sha256 equals this code's
             (the gate that passed the data is the one in this checkout)
  probe      the config's K-PROBE sha256 equals DATA_ROOT/evals/manifest.json's (kprobe.py refuses at start too)
Exit 0 ok, 3 any refusal (E2's convention).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)

DROPPED = "data sources differ from the shard manifest"


def e2_preflight():
    p = os.path.join(K.ROOT, "experiments", "E2_lr_transfer")
    if p not in sys.path:
        sys.path.insert(0, p)
    import preflight
    return preflight


def k_rules(cfg: dict, code: str, data: str) -> list[str]:
    import check_k
    bad, pv = [], cfg.get("provenance") or {}
    srcs = cfg.get("data", {}).get("sources", [])
    mp = os.path.join(code, "planck_root", pv.get("shard_manifest", "?"))
    if os.path.isfile(mp):
        want = [(s["name"], s["kind"], s.get("eot_id"), s["paths"][0]) for s in json.load(open(mp))["harness_data"]
                ["sources"]]
        names = {w[0] for w in want}
        have = [(s.get("name"), s.get("kind"), s.get("eot_id"), "/".join(s["paths"][0].split("/")[-2:])) for s in srcs
                if s.get("name") in names]
        if have and have != want:
            bad.append(f"lang: the LANG sources {have} are not the shard manifest's {want}")
        if not have and any(not str(s.get("name", "")).startswith("fact") for s in srcs):
            bad.append("lang: no LANG source in a config that is not FACT-only (CAP)")
    else:
        bad.append(f"lang: no shard manifest at {mp}")
    cp = os.path.join(data, "check.json")
    if not os.path.isfile(cp):
        bad.append(f"check: no {cp} (run gen.py check first)")
    else:
        c = json.load(open(cp))
        mine = {f: K.sha256_file(os.path.join(K.HERE, f)) for f in check_k.CHECK_CODE}
        if not c.get("pass"):
            bad.append(f"check: {cp} did not pass ({len(c.get('fails', []))} fails)")
        if c.get("code_sha256") != mine:
            bad.append("check: check.json was written by other gate code than this checkout's (code_sha256)")
    pr = (cfg.get("k") or {}).get("probe")
    if pr:
        man = json.load(open(os.path.join(data, "evals", "manifest.json")))
        if [f["sha256"] for f in pr["files"]] != [man["sets"]["k_probe"]["sha256"]]:
            bad.append("probe: the config's K-PROBE sha256 is not the data root's evals manifest's")
    return bad


def run(cfg_path: str, code: str, data: str) -> dict:
    import runio
    pf = e2_preflight()
    rep = pf.check_one(cfg_path, code, True, pf.harness_reads(os.path.join(code, "harness")))
    rep["refusals"] = [r for r in rep["refusals"] if not r.startswith(DROPPED)]
    rep["refusals"] += k_rules(runio.load_yaml(cfg_path), code, data)
    rep["ok"] = not rep["refusals"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--code", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = run(os.path.abspath(a.config), os.path.abspath(a.code), os.path.abspath(a.data))
    if a.out:
        K.write_json(a.out, {"ok": rep["ok"], "runs": [rep]})
    print("\n".join(rep["refusals"]) or f"ok: {rep.get('name')}")
    return 0 if rep["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
