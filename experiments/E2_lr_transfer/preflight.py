"""E2/E3 preflight: check run configs against the files and the harness they will run with. Runs on the PC
(it imports harness/runio.py, which imports torch; no model is built). Used by smoke.sh and queue_lib.sh.

  python preflight.py CONFIG [CONFIG ...] --code CODE_ROOT [--strict] [--no-init-check] [--out report.json]

Per config (resolved with harness runio.load_yaml, `extends:` included, exactly as train.py loads it):
  prereg      runio.check_prereg (with --strict also committed unchanged in git)
  provenance  sha256 of the tokenizer and its manifest (under CODE_ROOT), the shard manifest and the eval
              manifest (under CODE_ROOT/planck_root) equal the recorded values; the eval manifest's own
              evalset_sha256 equals the recorded one
  data        source names, kinds, eot ids and path globs match the shard manifest's harness_data block in
              its order; pad, role and end ids match its tokenizer; every glob matches a file; shares > 0
  model       vocab_size equals the tokenizer's; parameter count from harness/budget.py (closed form)
  engine      ENGINE keys vs the keys the harness source reads (.get("key") in non-test modules):
              read but unset -> refused; set to a non-off value but not read -> refused; with --strict,
              engine_fixed null or any "auto" -> refused. data_seed set but not read -> refused.
  schedule    trunk: every branch point in [warmup, total_steps - 1] (trainer asserts a stable phase);
              branch: init_from exists, has this model shape, LRs and seed, and total_steps equals its step
              plus decay_steps
  scorer      eval_bpb equals the invocation queue_lib.sh runs (all sets, fp32, batch 16384 tokens)
Exit 0 all ok, 3 any refusal. The report lists the resolved engine values and the config_sha256 train.py
will record (sha256 of json.dumps(cfg, sort_keys=True, default=str)).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

ENGINE = {"train": ["precision", "doc_attn", "lazy_metrics", "compile", "compile_backend", "compile_dynamic",
                    "ce_chunk_rows", "cuda_mem_cap_gib", "cuda_mem_margin_gib", "auto_micro_on_oom"],
          "optim": ["batched"], "data": ["mode", "window_tokens", "max_item_len"]}
ANY = object()   # the key does nothing while its switch is off, so an unread value is harmless
OFF = {"lazy_metrics": False, "compile": False, "compile_backend": ANY, "compile_dynamic": ANY,
       "ce_chunk_rows": 0, "cuda_mem_cap_gib": None, "cuda_mem_margin_gib": ANY, "auto_micro_on_oom": False}
SCORER = {"sets": "all", "precision": "fp32", "batch_tokens": 16384}   # queue_lib.sh score_run, E2 METRIC
READ_RE = re.compile(r"""\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def harness_reads(hdir: str) -> set[str]:
    keys: set[str] = set()
    for p in glob.glob(os.path.join(hdir, "*.py")):
        b = os.path.basename(p)
        if b.startswith(("test_", "mutant", "mutation_")) or b == "conftest.py":
            continue
        with open(p, encoding="utf-8") as f:
            keys |= set(READ_RE.findall(f.read()))
    return keys


def check_one(cfg_path: str, code: str, strict: bool, reads: set[str], init_check: bool = True) -> dict:
    import runio
    from budget import count_analytic
    from config import PlanckConfig
    from sources import expand
    bad, rep = [], {"config": os.path.relpath(cfg_path, code)}
    base = os.path.dirname(os.path.abspath(cfg_path))
    planck = os.path.join(code, "planck_root")
    cfg = runio.load_yaml(cfg_path)
    rep["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    rep["file_sha256"] = sha256_file(cfg_path)
    rep["name"] = cfg.get("name")
    try:
        pr = runio.check_prereg(cfg_path, strict)
        rep["prereg"] = {"id": pr["id"], "sha256": pr["sha256"], "committed": pr["committed"]}
    except runio.PreregError as e:
        bad.append(f"prereg: {e}")

    pv = cfg.get("provenance") or {}
    for key, root in (("tokenizer", code), ("tokenizer_manifest", code), ("shard_manifest", planck)):
        p = os.path.join(root, pv.get(key, "?"))
        got = sha256_file(p) if os.path.isfile(p) else None
        if got != pv.get(key + "_sha256"):
            bad.append(f"provenance {key}: file {pv.get(key)} sha256 {got} != recorded {pv.get(key + '_sha256')}")
    evm = os.path.join(planck, pv.get("evalset_dir", "?"), "manifest.json")
    if not os.path.isfile(evm) or sha256_file(evm) != pv.get("evalset_manifest_sha256"):
        bad.append(f"provenance: eval manifest {evm} missing or its sha256 differs")
    elif json.load(open(evm))["evalset_sha256"] != pv.get("evalset_sha256"):
        bad.append("provenance: evalset_sha256 in the eval manifest differs from the recorded one")

    d, srcs = cfg.get("data", {}), cfg.get("data", {}).get("sources", [])
    man_p = os.path.join(planck, pv.get("shard_manifest", "?"))
    if os.path.isfile(man_p):
        man = json.load(open(man_p))
        hd, tk = man["harness_data"], man["tokenizer"]
        want = [(s["name"], s["kind"], s.get("eot_id"), s["paths"][0]) for s in hd["sources"]]
        have = [(s.get("name"), s.get("kind"), s.get("eot_id"), "/".join(s["paths"][0].split("/")[-2:]))
                for s in srcs]
        if want != have:
            bad.append(f"data sources differ from the shard manifest: {have} vs {want}")
        chat = d.get("chat", {})
        if (d.get("pad_id"), chat.get("end_id"), chat.get("role_ids")) != (tk["pad_id"], tk["end_id"],
                                                                        tk["role_ids"]):
            bad.append("data pad/end/role ids differ from the shard manifest's tokenizer")
        if cfg["model"]["vocab_size"] != tk["vocab"]:
            bad.append(f"model vocab {cfg['model']['vocab_size']} != tokenizer vocab {tk['vocab']}")
    tot = sum(float(s.get("share", 0)) for s in srcs)
    rep["shares"] = {s["name"]: round(float(s.get("share", 0)) / tot, 6) for s in srcs} if tot else {}
    for s in srcs:
        if not float(s.get("share", 0)) > 0:
            bad.append(f"source {s.get('name')}: share missing or not positive")
        n = len(expand(s["paths"], base))
        if n == 0:
            bad.append(f"source {s.get('name')}: no file matches {s['paths']}")
        rep.setdefault("n_files", {})[s.get("name")] = n
    mcfg = PlanckConfig.from_dict(cfg["model"])
    rep["n_params"] = count_analytic(mcfg)["total"]

    eng = {}
    for sec, keys in ENGINE.items():
        for k in keys:
            has = k in (cfg.get(sec) or {})
            v = (cfg.get(sec) or {}).get(k)
            eng[f"{sec}.{k}"] = v
            if k in reads and not has:
                bad.append(f"engine: harness reads {sec}.{k} but the config does not set it")
            if has and k not in reads and OFF.get(k, "<none>") is not ANY and v != OFF.get(k, "<none>"):
                bad.append(f"engine: {sec}.{k} = {v!r} but this harness does not read it")
            if strict and v == "auto":
                bad.append(f"engine: {sec}.{k} is auto (strict runs write the resolved value)")
    rep["engine"] = eng
    rep["engine_not_read_by_harness"] = sorted(k for k in eng if k.split(".")[1] not in reads)
    rep["engine_fixed"] = cfg.get("engine_fixed")
    if strict and not cfg.get("engine_fixed"):
        bad.append("engine: engine_fixed is null (the FIXED step has not run)")
    if "data_seed" in cfg and "data_seed" not in reads:
        bad.append("data_seed is set but this harness does not read it (E3 Part 3 needs the patch)")
    ev = cfg.get("eval_bpb") or {}
    if {k: ev.get(k) for k in SCORER} != SCORER:
        bad.append(f"eval_bpb {ev} is not the scorer invocation queue_lib.sh runs {SCORER}")

    sc, tc = cfg.get("schedule", {}), cfg.get("train", {})
    mode, total = sc.get("mode", "full"), tc.get("total_steps")
    rep["schedule"] = {"mode": mode, "total_steps": total, "warmup_steps": sc.get("warmup_steps")}
    if sc.get("warmup_steps") is None:
        bad.append("schedule.warmup_steps not written")
    if mode == "trunk":
        pts = sorted(int(p) for p in sc.get("branch_points", []))
        if not pts or pts[0] < int(sc.get("warmup_steps") or 0) or pts[-1] > int(total) - 1:
            bad.append(f"trunk branch points {pts} not inside [warmup, total_steps - 1 = {int(total) - 1}]")
    if mode == "branch":
        ip = os.path.normpath(os.path.join(base, sc.get("init_from", "?")))
        rep["init_from"] = os.path.relpath(ip, code)
        if not os.path.isfile(ip):
            if init_check:
                bad.append(f"branch init_from {rep['init_from']} does not exist")
        else:
            ck = runio.load_checkpoint(ip)
            if ck["model_cfg"] != mcfg.to_dict():
                bad.append("branch init_from has another model shape")
            if int(ck["step"]) + int(sc.get("decay_steps", -1)) != int(total):
                bad.append(f"branch total_steps {total} != init step {ck['step']} + decay_steps")
            oc, po = cfg.get("optim", {}), ck["config"].get("optim", {})
            for k in ("lr", "embed_lr", "scalar_lr"):
                if float(oc.get(k, -1)) != float(po.get(k, -2)):
                    bad.append(f"branch optim.{k} {oc.get(k)} != trunk's {po.get(k)} (the trunk's is used)")
            if ck["config"].get("seed") != cfg.get("seed"):
                bad.append("branch seed differs from its trunk's")
    rep["ok"], rep["refusals"] = not bad, bad
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--code", required=True, help="code root (holds harness/, tokenizer/, planck_root)")
    ap.add_argument("--strict", action="store_true", help="scored runs: committed prereg, engine fixed")
    ap.add_argument("--no-init-check", action="store_true", help="plan check before the trunks exist")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    code = os.path.abspath(a.code)
    sys.path.insert(0, os.path.join(code, "harness"))
    reads = harness_reads(os.path.join(code, "harness"))
    reps = [check_one(os.path.abspath(c), code, a.strict, reads, not a.no_init_check) for c in a.configs]
    out = {"ok": all(r["ok"] for r in reps), "strict": a.strict, "runs": reps}
    txt = json.dumps(out, indent=1, sort_keys=True, default=str)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt if not out["ok"] else json.dumps({"ok": True, "n": len(reps),
                                                 "names": [r["name"] for r in reps]}), flush=True)
    return 0 if out["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
