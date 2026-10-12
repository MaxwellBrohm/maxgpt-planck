"""Read the SMOKE run's outputs and check the plumbing (label smoke; no number here enters any E2/E3 table).
Runs on the PC after smoke.sh (loads checkpoints on the CPU, builds no model).

  python smoke_check.py SMOKE_DIR CODE_ROOT      -> JSON on stdout; exit 0 only if every check passes

Checks: loss falls (trunk's first three log records vs the last branch's last three); bits per byte falls on
CHAT and PROSE from the first checkpoint (trunk step 100) to the last (bend final); throughput (median
tok_per_s after the first 30 steps of each process) against the eager 5M bench; the pause at 250 and the
resume from its checkpoint; branches start from the trunk's stable checkpoints; drawn tokens per source
against the pinned shares; every checkpoint's config records the tokenizer and shard manifest hashes (equal
to the files) and the engine flags (equal to engine.yaml); config_sha256 in runs.jsonl, in the checkpoint
and recomputed from the config file agree.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import statistics as stt
import sys

# pc/bench_micro 5M, B16, T 2048, bf16, eager (harness/notes.txt baseline; speed workflow A/B of 2026-09-26)
BENCH = {"5M causal, reference path": 216220, "5M docmask, reference path (mask, per-matrix optim)": 162415,
         "5M docmask, varlen + batched optim (the engine.yaml values)": 268594}
RUNS = ("smoke_trunk", "smoke_bmid", "smoke_bend")


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if os.path.exists(p) else []


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    R, code = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    sys.path.insert(0, os.path.join(code, "harness"))
    import runio
    planck = os.path.join(code, "planck_root")
    E = os.path.join(code, "experiments", "E2_lr_transfer")
    checks, rep = {}, {}
    events = jl(os.path.join(R, "runs.jsonl"))
    starts = [e for e in events if e["event"] == "start"]
    rep["events"] = [{k: e.get(k) for k in ("event", "run", "step", "resumed_from", "doc_attn", "optim_batched",
                                             "lazy_metrics", "n_params", "batch_tokens", "precision")}
                     for e in events]
    for e in rep["events"]:
        if e.get("resumed_from"):
            e["resumed_from"] = os.path.basename(e["resumed_from"])
    logs = {r: [x for x in jl(os.path.join(R, r, "log.jsonl")) if "loss" in x] for r in RUNS}

    # loss
    tr, be = logs["smoke_trunk"], logs["smoke_bend"]
    first3, last3 = [x["loss"] for x in tr[:3]], [x["loss"] for x in be[-3:]]
    rep["loss"] = {r: {"first": v[0]["loss"], "last": v[-1]["loss"], "min": min(x["loss"] for x in v),
                       "steps": [v[0]["step"], v[-1]["step"]]} for r, v in logs.items() if v}
    rep["loss"]["trunk_first3_mean"], rep["loss"]["bend_last3_mean"] = stt.mean(first3), stt.mean(last3)
    checks["loss_falls"] = stt.mean(last3) < stt.mean(first3) - 1.0

    # throughput: drop the first 30 steps after each process start (CUDA warmup, first kernels)
    tps, fill = [], {}
    for r in RUNS:
        s0 = sorted({e["step"] for e in starts if e["run"] == r})
        for x in logs[r]:
            last_start = max([s for s in s0 if s < x["step"]], default=0)
            if x["step"] - last_start > 30:
                tps.append(x["tok_per_s"])
        if logs[r]:
            st = [e for e in starts if e["run"] == r]
            steps = logs[r][-1]["step"] - min(e["step"] for e in st)
            tok0 = 0 if min(e["step"] for e in st) == 0 else None
            if tok0 is not None:
                fill[r] = round(logs[r][-1]["tokens"] / (steps * st[0]["batch_tokens"]), 4)
    rep["tok_per_s"] = {"median": stt.median(tps), "min": min(tps), "max": max(tps), "n": len(tps),
                        "bench_5M_B16": BENCH, "row_fill_trunk": fill.get("smoke_trunk")}
    checks["throughput_at_least_eager_bench_x0.8"] = stt.median(tps) >= 0.8 * BENCH[
        "5M docmask, reference path (mask, per-matrix optim)"]

    # pause/resume and branch starts
    ts = [e for e in starts if e["run"] == "smoke_trunk"]
    checks["trunk_paused_and_resumed"] = (len(ts) == 2 and ts[1]["step"] == 250 and
                                          os.path.basename(ts[1].get("resumed_from") or "") == "ckpt_00000250.pt")
    bs = {e["run"]: os.path.basename(e.get("resumed_from") or "") for e in starts if e["run"] != "smoke_trunk"}
    checks["branches_from_stable"] = bs == {"smoke_bmid": "stable_00000400.pt", "smoke_bend": "stable_00001200.pt"}
    ends = {e["run"]: e for e in events if e["event"] == "end"}
    checks["all_runs_ended"] = set(ends) == set(RUNS)

    # drawn tokens vs the pinned shares (last log record of each run)
    cfg0 = runio.load_yaml(os.path.join(E, "smoke", "trunk.yaml"))
    sh = {s["name"]: float(s["share"]) for s in cfg0["data"]["sources"]}
    tot_sh = sum(sh.values())
    rep["drawn"] = {}
    for r in RUNS:
        if not logs[r]:
            continue
        last = logs[r][-1]
        drawn = {n: last.get(f"drawn_{n}", 0) for n in sh}
        tot = sum(drawn.values())
        dev = {n: drawn[n] - sh[n] / tot_sh * tot for n in sh}
        rep["drawn"][r] = {"total": tot, "share": {n: round(drawn[n] / tot, 5) for n in sh},
                           "dev_tokens": {n: round(v, 1) for n, v in dev.items()},
                           "max_abs_dev_items_2048": round(max(abs(v) for v in dev.values()) / 2048, 3),
                           "dropped_long": {k: v for k, v in last.items() if k.startswith("dropped_long")}}
    checks["drawn_within_one_item"] = all(v["max_abs_dev_items_2048"] <= 1.0 for v in rep["drawn"].values())

    # bits per byte: first checkpoint vs last
    def bpb(run, ck):
        p = os.path.join(R, run, "bpb", ck + ".json")
        if not os.path.exists(p):
            return None
        s = json.load(open(p))["sets"]
        pro = ("cccc", "gutenberg", "wikimedia")
        return {"CHAT": s["oasst2"]["bpb"],
                "PROSE": sum(s[k]["bits"] for k in pro) / sum(s[k]["bytes"] for k in pro),
                **{k: round(v["bpb"], 5) for k, v in s.items()},
                "truncated": {k: v["truncated"] for k, v in s.items() if v["truncated"]}}
    first = bpb("smoke_trunk", "ckpt_00000100")
    fin = {r: sorted(glob.glob(os.path.join(R, r, "final_*.pt"))) for r in RUNS}
    last = bpb("smoke_bend", os.path.basename(fin["smoke_bend"][-1])[:-3]) if fin["smoke_bend"] else None
    mid = bpb("smoke_bmid", os.path.basename(fin["smoke_bmid"][-1])[:-3]) if fin["smoke_bmid"] else None
    rep["bpb"] = {"trunk_step100": first, "bmid_final": mid, "bend_final": last}
    checks["bpb_falls"] = bool(first and last and last["CHAT"] < first["CHAT"] and last["PROSE"] < first["PROSE"])
    rep["bpb_files"] = sorted(os.path.relpath(p, R) for p in glob.glob(os.path.join(R, "*", "bpb", "*.json")))

    # provenance and engine flags recorded in every checkpoint's config
    eng = runio.load_yaml(os.path.join(E, "configs", "engine.yaml"))
    pv_ok, eng_ok, sha_ok, n = True, True, True, 0
    real = {"tokenizer_sha256": sha(os.path.join(code, "tokenizer/v0/tok_v0_8k.json")),
            "shard_manifest_sha256": sha(os.path.join(planck, "data/shards/starter_v0_8k/manifest.json"))}
    cfg_files = {"smoke_trunk": "trunk.yaml", "smoke_bmid": "bmid.yaml", "smoke_bend": "bend.yaml"}
    for r in RUNS:
        want_sha = hashlib.sha256(json.dumps(runio.load_yaml(os.path.join(E, "smoke", cfg_files[r])),
                                             sort_keys=True, default=str).encode()).hexdigest()
        run_sha = {e["config_sha256"] for e in events if e["event"] == "start" and e["run"] == r}
        for ck in sorted(glob.glob(os.path.join(R, r, "*.pt"))):
            c = runio.load_checkpoint(ck)
            n += 1
            pv = c["config"].get("provenance", {})
            pv_ok &= all(pv.get(k) == v for k, v in real.items())
            for sec in ("train", "optim", "data"):
                for k, v in (eng.get(sec) or {}).items():
                    eng_ok &= c["config"][sec].get(k) == v
            eng_ok &= c["config"].get("engine_fixed", "missing") == eng.get("engine_fixed")
            sha_ok &= c["config_sha256"] == want_sha and run_sha == {want_sha}
        rep.setdefault("config_sha256", {})[r] = want_sha
    rep["checkpoints_read"] = n
    rep["recorded"] = {**real, "engine": {s: eng.get(s) for s in ("train", "optim", "data")}}
    checks["provenance_hashes_recorded_and_equal_files"] = pv_ok and n > 0
    checks["engine_flags_recorded"] = eng_ok and n > 0
    checks["config_sha256_consistent"] = sha_ok and n > 0
    checks["n_params_5010133"] = {e.get("n_params") for e in starts} == {5010133}
    rep["checks"] = checks
    rep["pass"] = all(checks.values())
    print(json.dumps(rep, indent=1, sort_keys=True, default=str))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
