"""RC-12 dev-baseline batch for one model and one render (notes STEP 9; draft s12, s19 item 3). Loads the engine
ONCE and plays, in lockstep, every requested seed on the dev split plus each seed's --own-cf twin on the OWN records
(OD1 b: without the twin R is None). Layout: <root>/<model>/<render>/<seed>/ and <root>/<model>/<render>/<seed>_owncf/
(<model> = the id after its last "/", <seed> = greedy or the integer), each with transcripts.jsonl, scores.jsonl and
meta.json. A run is written to <dir>.partial and renamed only when complete, then marked DONE; a finished run is
never overwritten (it is skipped on restart); a stale .partial is deleted and redone.
Parity gate (draft s4: a greedy HF-vs-vLLM parity check before any scored run): a vllm: responder runs only if
<root>/<model>/parity/parity.json says IDENTICAL or NEAR_TIE, or engines.json names vllm for that model (the engine
decision after reading the parity, notes STEP 9 PARITY), or with --no-parity-gate (recorded in meta.json). A model
engines.json lists runs only on the engine it names (engine gate).
engines.json may also name a dtype per model ({"engine": "vllm", "dtype": "float32"}); it overrides --dtype. meta.json
records the run config (verifier 2026-09-26): dtype, max_model_len, batch_invariant, decode (hf_responder.DECODE) and
audit (engine_audit: the stop list, thinking flag, ctx, the sampling kwargs, and for vLLM the stop ids it adds from
generation_config.json).
hf: responders need --hf-untested-ok (runner.py rule) and go through lockstep.Serial (no batching); hfb: (hf_batched.py,
batched, same flag) is gated on its parity.json like vllm: (parity_hf_vllm.py --engine hfb).
Run on the PC under the GPU lock (pc_jobs.py writes the job):
  python -B dev_batch.py --responder vllm:Qwen/Qwen2.5-0.5B-Instruct --render template --seeds greedy,1,2,3 \\
      --root ~/planck/runs/rc12_dev"""
import argparse
import json
import os
import shutil
import sys
import time

import hf_responder as HR
import runner as R
import score as S

PASSING = ("IDENTICAL", "NEAR_TIE")
ENGINES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engines.json")


def slug(model_id):
    return model_id.rstrip("/").split("/")[-1]


def seed_name(seed):
    return "greedy" if seed is None else str(seed)


def chosen_engine(model_id, path=ENGINES):
    """the engine engines.json names for model_id (the per-model engine decision after parity, notes STEP 9), or
    None when the file or the model is not listed."""
    if not os.path.exists(path):
        return None
    return json.load(open(path))["models"].get(model_id, {}).get("engine")


def chosen_dtype(model_id, path=ENGINES):
    """the dtype engines.json names for model_id (e.g. "float32" when only the fp32 parity passed), or None."""
    if not os.path.exists(path):
        return None
    return json.load(open(path))["models"].get(model_id, {}).get("dtype")


def engine_audit(eng):
    """the engine's run config for meta.json: VLLMResponder.audit(), else the HF fields (stop list, thinking, ctx)."""
    if hasattr(eng, "audit"):
        return eng.audit()
    out = {k: getattr(eng, k, None) for k in ("stop_ids", "eos", "eot", "qwen3", "ctx", "has_template")}
    gc = getattr(getattr(eng, "model", None), "generation_config", None)
    try:                                               # what generate's explicit kwargs override
        if gc is not None:
            out["hf_generation_config"] = {k: v for k, v in gc.to_diff_dict().items() if k != "transformers_version"}
    except Exception as e:  # noqa: BLE001  (recorded, never fatal)
        out["hf_generation_config"] = f"unreadable: {type(e).__name__}"
    return out


def parity_verdict(root, model_id):
    p = os.path.join(root, slug(model_id), "parity", "parity.json")
    return json.load(open(p))["verdict"] if os.path.exists(p) else None


def plan(root, model_id, render, seeds):
    """[(seed, own_cf, final dir)] still to run; raises if a final dir exists without its DONE mark."""
    todo = []
    for seed in seeds:
        for own_cf in (False, True):
            d = os.path.join(root, slug(model_id), render, seed_name(seed) + ("_owncf" if own_cf else ""))
            if os.path.exists(os.path.join(d, "DONE")):
                continue
            if os.path.exists(d):
                raise SystemExit(f"{d} exists without DONE: not touching it")
            todo.append((seed, own_cf, d))
    return todo


def versions():
    out = {"python": sys.version.split()[0]}
    for m in ("vllm", "transformers", "torch"):
        mod = sys.modules.get(m)
        out[m] = getattr(mod, "__version__", None) if mod else None
    return out


def run_one(eng, name, data, render, seed, own_cf, final, ctx, extra):
    tmp = final + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    before = len(getattr(eng, "batches", []))
    t0 = time.time()
    rows = R.run(data, eng, render, [seed], ctx, tmp, name, extra["train_seed"], own_cf, lockstep=True)
    summ = S.summarize(rows)
    meta = dict(extra, model=name, render=render, seed=seed, own_cf=own_cf, conversations=len(rows), ctx=ctx,
                seconds=round(time.time() - t0, 1), batches=getattr(eng, "batches", [])[before:], versions=versions(),
                R_ungated=summ.get("R_ungated"), loop_rate=summ.get("loop_rate"),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1)
    open(os.path.join(tmp, "DONE"), "w").close()
    os.replace(tmp, final)
    print(f"{final}: {len(rows)} conversations, {meta['seconds']} s, R_ungated {meta['R_ungated']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responder", required=True, help="vllm:<id>, hfb:<id> or hf:<id>")
    ap.add_argument("--render", choices=["template", "plain"], required=True)
    ap.add_argument("--seeds", default="greedy,1,2,3")
    ap.add_argument("--root", required=True)
    ap.add_argument("--data", default=R.DEV)
    ap.add_argument("--limit", type=int, default=None, help="smoke runs only (a limited run is still marked DONE)")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--no-parity-gate", action="store_true")
    ap.add_argument("--engines", default=ENGINES, help="the per-model engine decisions (engines.json)")
    ap.add_argument("--hf-untested-ok", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--batch-invariant", action="store_true")
    args = ap.parse_args()
    kind, _, model_id = args.responder.partition(":")
    root = os.path.expanduser(args.root)
    gated = kind in ("vllm", "hfb")               # hfb: its parity is batched HF vs serial HF (--engine hfb)
    verdict = parity_verdict(root, model_id) if gated else None
    chosen = chosen_engine(model_id, args.engines)
    dtype = chosen_dtype(model_id, args.engines) or args.dtype   # engines.json's dtype wins (verifier 2026-09-26)
    if chosen is not None and chosen != kind and not args.no_parity_gate:
        sys.exit(f"engine gate: engines.json names {chosen} for {model_id}, not {kind}")
    if gated and verdict not in PASSING and chosen != kind and not args.no_parity_gate:
        sys.exit(f"parity gate: {model_id} has parity verdict {verdict}; run parity_hf_vllm.py first")
    todo = plan(root, model_id, args.render, R.parse_seeds(args.seeds))
    if not todo:
        print("nothing to run: every requested run is DONE")
        return 0
    ns = argparse.Namespace(render=args.render, dtype=dtype, device=args.device, gpu_mem=args.gpu_mem,
                            max_model_len=args.max_model_len, batch_invariant=args.batch_invariant, lockstep=True,
                            hf_untested_ok=args.hf_untested_ok, vllm_untested_ok=kind == "vllm")
    eng, name = R.make_responder(args.responder, ns)
    ctx = getattr(eng, "ctx", None)
    recs, own = R.load(args.data, None, args.limit), R.load(args.data, ["OWN"], args.limit)
    extra = dict(engine=kind, parity=verdict, engine_decision=chosen,
                 parity_gate_skipped=args.no_parity_gate and verdict not in PASSING and chosen != kind,
                 limit=args.limit, train_seed=args.train_seed, gpu_mem=args.gpu_mem, data=os.path.basename(args.data),
                 dtype=dtype, max_model_len=args.max_model_len, batch_invariant=args.batch_invariant,
                 decode=HR.DECODE, audit=engine_audit(eng))
    for seed, own_cf, final in todo:
        run_one(eng, name, own if own_cf else recs, args.render, seed, own_cf, final, ctx, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
