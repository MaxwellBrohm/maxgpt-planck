"""K0-smoke's reading (K1 notes STAGE 0: "labelled smoke, never read" for science; it checks the machinery). Built
2026-11-21 (K3 round 16), before any K run. Reads the two smoke runs' files and the data root; no model is run.

  python k0smoke.py RUNS_DIR DATA_ROOT [--readings FILE]   RUNS_DIR holds k0_smoke_s200/, k0_smoke_off_s200/ and
                                                           k0_smoke_rep_s200/ (kconfig.py k0)

Checks, each a named line of the reading (pass = every one holds):
  finished    the three runs wrote final_*.pt
  hook_off    the K-PROBE hook leaves training unchanged on this data. K1 notes ask bitwise equality; CUDA eager is
              not bitwise repeatable (SCREENS: two identical runs drift without deterministic algorithms), so
              (K3 round 16 DEVIATION K-D1, 2026-11-21, before any K run): the data and schedule fields of every
              log record (D1_FIELDS: step, tokens, sup_tokens, lr_factor, phase; and every drawn_<source> and
              dropped_<...>; each record must carry the named five and a drawn_ field) equal across the three runs,
              and either the probe-on and probe-off runs are bitwise equal (every log record but the CLOCK fields,
              and the final weights' hash), or the two probe-off runs are not (the run is not repeatable here) and the
              largest loss gap H between probe on and off is at most twice the largest gap R between the two
              probe-off runs (replay noise); repeatable replays demand a bitwise pair; H, R are reported.
              [2026-11-23, K review finding 1: the check compared every field but loss, gnorm and tok_per_s, so the
              harness's wall-clock "time" (runio.append_jsonl stamps every record) differed across the three runs
              and no real smoke could pass; it now reads the named fields, and "time" is never compared.]
  probe       k_probe.jsonl holds a reading at every 153 steps and the last, none with an error
  shares      the loader's drawn tokens per source (the last log record's drawn_<name>) sit within one item
              (data.max_item_len) of share x total, shares normalized as the loader does
  fact_draw   per FACT source, the draw projected to SPEC 7's 7,630 steps (drawn / step x 7,630) reaches the end of
              the FB region (95% of the written tokens, count_k's count) and stays inside the written stream: every
              FB exposure is drawn once, none twice
  memory      the queue's gpu_mem_max_mib.txt, when present, is at most 11,000 MiB (the runner's guard: 11,600)
On a pass (and only then) main() writes RUNS_DIR/marks/K0_smoke_passed, the mark plans/k0.txt's K0-g grid waits for
(SPEC 13: "the K smoke run passed" before any K run; 2026-11-22, REVIEW 17b).
Reported: tok/s (median of the log records after the first), the projected hours of a 7,630-step run with its
50 probe readings (49 at the multiples of 153 and step 7,630; corrected from 51 on 2026-11-22, port review note 7; the
code always used 50) (SPEC 13's T, estimates only [G-1]).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)

ON, OFF, REP = "k0_smoke_s200", "k0_smoke_off_s200", "k0_smoke_rep_s200"
D1_FIELDS = ("step", "tokens", "sup_tokens", "lr_factor", "phase")    # K-D1's data and schedule fields, with
D1_PREFIXES = ("drawn_", "dropped_")                                  # the loader's per-source counts
CLOCK = ("time", "tok_per_s")         # wall clock (runio stamps every record) and a rate: never compared (2026-11-23)
STEPS, EVERY, MEM_MIB = 7_630, 153, 11_000


def _log(d: str) -> list[dict]:
    return [json.loads(x) for x in open(os.path.join(d, "log.jsonl"))]


def _final(d: str) -> str | None:
    f = sorted(glob.glob(os.path.join(d, "final_*.pt")))
    return f[-1] if f else None


def weights_sha(path: str) -> str:
    import runio
    import torch
    import hashlib
    ck = runio.load_checkpoint(path)
    h = hashlib.sha256()
    for name, p in sorted(ck["model"].items()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes() if isinstance(p, torch.Tensor) else repr(p).encode())
    return h.hexdigest()


def read(runs_dir: str, data_root: str, cfg: dict, written: dict) -> dict:
    """cfg: the probe-on run's resolved config; written: {fact source: written tokens} (count_k)."""
    on, off, rep = (os.path.join(runs_dir, n) for n in (ON, OFF, REP))
    out = {"reading": "K0-smoke", "checks": {}}
    ck = out["checks"]
    fin = [_final(d) for d in (on, off, rep)]
    ck["finished"] = all(f is not None for f in fin)
    la, lb, lc = _log(on), _log(off), _log(rep)
    keep = lambda L: [{k: v for k, v in r.items() if k in D1_FIELDS or k.startswith(D1_PREFIXES)}  # noqa: E731
                      for r in L]
    named = lambda L: all(set(D1_FIELDS) <= r.keys() and any(k.startswith("drawn_") for k in r)  # noqa: E731
                          for r in L)
    gap = lambda X, Y: max(abs(x["loss"] - y["loss"]) for x, y in zip(X, Y))      # noqa: E731
    same_data = bool(la) and named(la) and keep(la) == keep(lb) == keep(lc)
    strip = lambda L: [{k: v for k, v in r.items() if k not in CLOCK} for r in L]     # noqa: E731
    sha = [weights_sha(f) for f in fin] if ck["finished"] else [None, 1, 2]
    bitwise = strip(la) == strip(lb) and sha[0] == sha[1]          # probe on against probe off
    det = strip(lb) == strip(lc) and sha[1] == sha[2]              # the two probe-off runs: repeatable here?
    H, R = (gap(la, lb), gap(lb, lc)) if same_data else (None, None)
    out["hook_off"] = {"same_data": same_data, "bitwise": bitwise, "repeatable": det, "H": H, "R": R}
    ck["hook_off"] = same_data and (bitwise or (not det and H <= 2 * R))
    total = int(cfg["train"]["total_steps"])
    pr = [json.loads(x) for x in open(os.path.join(on, "k_probe.jsonl"))] if os.path.exists(
        os.path.join(on, "k_probe.jsonl")) else []
    want = sorted(set(range(EVERY, total + 1, EVERY)) | {total})
    ck["probe"] = [r["step"] for r in pr] == want and not any("error" in r for r in pr)
    last = la[-1]
    srcs = cfg["data"]["sources"]
    tot_share = sum(float(s["share"]) for s in srcs)
    drawn = {s["name"]: int(last.get(f"drawn_{s['name']}", -1)) for s in srcs}
    tot = sum(drawn.values())
    lim = int(cfg["data"].get("max_item_len", 2048))
    out["shares"] = {n: {"drawn": d, "target": round(float(s["share"]) / tot_share * tot, 1)}
                     for (n, d), s in zip(drawn.items(), srcs)}
    ck["shares"] = all(d >= 0 and abs(d - v["target"]) <= lim for d, v in zip(drawn.values(), out["shares"].values()))
    step = int(last["step"])
    proj = {n: drawn[n] / step * STEPS for n in written}
    out["fact_draw"] = {n: {"projected": round(proj[n]), "fb_region_end": round(K.FB_REGION * written[n]),
                            "written": written[n]} for n in written}
    ck["fact_draw"] = bool(written) and all(K.FB_REGION * written[n] <= proj[n] <= written[n] for n in written)
    mp = os.path.join(on, "gpu_mem_max_mib.txt")
    out["gpu_mem_max_mib"] = int(open(mp).read().strip()) if os.path.exists(mp) else None
    ck["memory"] = out["gpu_mem_max_mib"] is None or out["gpu_mem_max_mib"] <= MEM_MIB
    tps = statistics.median(r["tok_per_s"] for r in la[1:]) if len(la) > 1 else None
    probe_s = statistics.mean(r["seconds"] for r in pr) if pr else 0.0
    out["tok_per_s"] = tps
    out["hours_7630"] = (last["tokens"] / step * STEPS / tps + probe_s * (STEPS // EVERY + 1)) / 3600 if tps else None
    out["pass"] = all(ck.values())
    return out


def main(argv=None) -> int:
    import runio
    import count_k
    import analyze_k as A
    import kconfig as KC
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("data_root")
    ap.add_argument("--readings", default=KC.READINGS)
    a = ap.parse_args(argv)
    cfg = runio.load_yaml(os.path.join(KC.CONF, ON + ".yaml"))
    cnt = count_k.count_arm(a.data_root, "FH")
    written = {n: v["total"]["tokens"] for n, v in cnt["sources"].items() if n.startswith("fact_")}
    rec = A.append_reading(a.readings, read(a.runs_dir, a.data_root, cfg, written))
    print(json.dumps({k: rec[k] for k in ("pass", "checks", "tok_per_s", "hours_7630")}))
    if rec["pass"]:
        mark(a.runs_dir)
    return 0 if rec["pass"] else 1


def mark(runs_dir: str) -> str:
    """2026-11-22 (REVIEW 17b): the pass mark plans/k0.txt's grid waits for (queue_k.sh wait_mark K, its slug: every
    character outside A-Z a-z 0-9 becomes _), written only after a passing reading is appended."""
    import kconfig as KC
    p = os.path.join(runs_dir, "marks", re.sub(r"[^A-Za-z0-9]", "_", KC.SMOKE_MARK))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()
    return p


if __name__ == "__main__":
    sys.exit(main())
