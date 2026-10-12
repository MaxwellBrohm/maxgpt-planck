"""C1-a gate part (1): a compiled smoke of one screen arm at its own 5M screen config. PC, CUDA, under gpu.lock.

  python gate_smoke.py CONFIG --scratch SCRATCH_ROOT --steps 300 --out smoke.json

Runs harness train.main on the arm's run config as written (train.compile default, its engine, the starter data)
for --steps steps through --max-steps (the schedule stays the full 7,630-step WSD; the run pauses there). The
config's planck_root must resolve inside SCRATCH_ROOT (a lean copy whose planck_root is a scratch directory with
the data linked in), so no checkpoint or runs.jsonl line lands in a real run directory; the script refuses
otherwise, and refuses an out_dir that already holds anything. Logged: Dynamo unique graphs, graph breaks and
their reasons, every torch._dynamo recompile / cache-limit log line, the start record, median tok/s, the first
log step's wall time, finite losses, peak reserved GPU memory. A smoke: never read as a result.
Exit 0 = the steps finished, every logged loss finite, no limit line; else 1. Checkpoints are deleted after.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flag_parity import Grab  # noqa: E402  (screens_lib puts harness/ on sys.path)

import runio  # noqa: E402


def paths(cfg_path: str, scratch: str) -> tuple[str, str]:
    """-> (out_dir, runs_jsonl), both refused unless they resolve inside scratch."""
    cfg, base = runio.load_yaml(cfg_path), os.path.dirname(os.path.abspath(cfg_path))
    out = os.path.realpath(os.path.join(base, cfg["out_dir"]))
    rj = os.path.realpath(os.path.join(base, cfg["runs_jsonl"]))
    root = os.path.realpath(scratch) + os.sep
    for p in (out, rj):
        if not p.startswith(root):
            raise SystemExit(f"refusing: {p} is outside the scratch root {root}")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"refusing: {out} is not empty (a smoke starts fresh)")
    return out, rj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out_dir, rj = paths(a.config, a.scratch)
    import torch
    from torch._dynamo.utils import counters
    import train
    grab = Grab()
    logging.getLogger("torch._dynamo").addHandler(grab)
    counters.clear()
    t0 = time.time()
    rc = train.main([a.config, "--device", "cuda", "--max-steps", str(a.steps), "--no-resume"])
    secs = time.time() - t0
    log = [json.loads(x) for x in open(os.path.join(out_dir, "log.jsonl"))]
    start = [json.loads(x) for x in open(rj) if '"event": "start"' in x][-1]
    recs = [r for r in log if "loss" in r]
    res = {"config": os.path.relpath(os.path.abspath(a.config), os.path.realpath(a.scratch)), "steps": a.steps,
           "rc": rc, "seconds": round(secs, 1), "last_step": recs[-1]["step"] if recs else 0,
           "start": {k: start.get(k) for k in ("run", "n_params", "doc_attn", "compile", "compile_dynamic",
                                              "optim_batched", "mtp", "batch_tokens", "precision", "schedule")},
           "losses_finite": bool(recs) and all(math.isfinite(r["loss"]) for r in recs) and
           not any("error" in r for r in log),
           "loss_first_last": [recs[0]["loss"], recs[-1]["loss"]] if recs else None,
           "tok_per_s_median": st.median(r["tok_per_s"] for r in recs[2:]) if len(recs) > 3 else None,
           "first_log_tok_per_s": recs[0]["tok_per_s"] if recs else None,
           "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
           "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2),
           "dynamo": {"unique_graphs": counters["stats"].get("unique_graphs", 0),
                      "graph_breaks": sum(counters["graph_break"].values()),
                      "break_reasons": {k.splitlines()[0][:160]: v for k, v in counters["graph_break"].items()},
                      "limit_msgs": list(grab.msgs)}}
    res["pass"] = (rc == 0 and res["last_step"] == a.steps and res["losses_finite"] and not grab.msgs
                   and res["start"]["compile"] == "default")
    for p in glob.glob(os.path.join(out_dir, "*.pt")):
        os.remove(p)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[smoke] {json.dumps(res)}", flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
