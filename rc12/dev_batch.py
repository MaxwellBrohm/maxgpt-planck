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
engines.json may also set, for an hf / hfb model only, "trust_remote_code": true (the checkpoint's own modeling code)
and "attn_implementation" (e.g. "eager"), for hfb only "max_batch" (rows per generate call, default
hf_batched.MAX_BATCH), and for any model a "python" (the venv the queue runs it with). Doge uses all four (notes
STEP 9c). meta.json records them (max_batch None = the default) and sys.executable; its audit records the attention
the model loaded with. Item 14 extras (notes STEP 11): "token_budget" (hfb only, rows x width^2 per chunk, default
hf_batched.TOKEN_BUDGET) and "chat_template" (any engine: a Jinja chat template set on the tokenizer before any
prompt is built, for a checkpoint that ships none; Loom-Spark-3.2's card format); meta.json records both.
hf: responders need --hf-untested-ok (runner.py rule) and go through lockstep.Serial (no batching); hfb: (hf_batched.py,
batched, same flag) is gated on its parity.json like vllm: (parity_hf_vllm.py --engine hfb).
Run on the PC under the GPU lock (pc_jobs.py writes the job):
  python -B dev_batch.py --responder vllm:Qwen/Qwen2.5-0.5B-Instruct --render template --seeds greedy,1,2,3 \\
      --root ~/planck/runs/rc12_dev"""
import argparse
import errno
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


def hf_options(model_id, path=ENGINES):
    """engines.json's HF options for model_id: trust_remote_code (True only where the checkpoint's own code must run;
    a non-boolean value counts as False), attn_implementation (None = the library default), max_batch and
    token_budget (hfb; None = hf_batched's defaults), chat_template (any engine; None = the tokenizer's own; notes
    STEP 11, item 14), nan_guard (hf / hfb: hf_responder.guard_nans's module class; None = no hook) and use_cache
    (hf / hfb: hf_responder.set_use_cache; None = the checkpoint's config) (item 14)."""
    e = json.load(open(path))["models"].get(model_id, {}) if os.path.exists(path) else {}
    return dict(trust_remote_code=e.get("trust_remote_code", False) is True,
                attn_implementation=e.get("attn_implementation"), max_batch=e.get("max_batch"),
                token_budget=e.get("token_budget"), chat_template=e.get("chat_template"),
                nan_guard=e.get("nan_guard"), use_cache=e.get("use_cache"))


DECODE_KEYS = ("top_p", "repetition_penalty")       # what a reported decoding row may change (item 14 card row)


def decode_override(model_id, path=ENGINES):
    """engines.json "decode" for model_id: {} when absent (the s4 decoding, hf_responder.DECODE). A separate engines
    file (engines_card.json) carries it for a REPORTED decoding row run to its own root, e.g. a model card's
    recommended sampling (NOVELTY_2026-10 s6: Vertex-0.6-15M, top_p 0.9, repetition penalty 1.3; notes STEP 11,
    ITEM 14). Only DECODE_KEYS may change: top_p in (0, 1], repetition_penalty >= 1. Raises ValueError otherwise."""
    e = json.load(open(path))["models"].get(model_id, {}) if os.path.exists(path) else {}
    d = e.get("decode", {})
    if not isinstance(d, dict) or set(d) - set(DECODE_KEYS):
        raise ValueError(f"engines.json decode for {model_id}: a dict of {DECODE_KEYS} only, not {d!r}")
    for k, v in d.items():
        if type(v) not in (int, float) or not (0 < v <= 1 if k == "top_p" else v >= 1):
            raise ValueError(f"engines.json decode {k}={v!r} for {model_id} is out of range")
    return dict(d)


def apply_decode(model_id, path, kind):
    """set engines.json's decode (decode_override) into hf_responder.DECODE, which every engine reads per request;
    vllm only (hf_batched's row sampler draws at the temperature alone, without top_p). Returns the override."""
    d = decode_override(model_id, path)
    if d and kind != "vllm":
        raise ValueError(f"engines.json decode for {model_id}: vllm only, not {kind}")
    HR.DECODE.update(d)
    return d


def engine_audit(eng):
    """the engine's run config for meta.json: VLLMResponder.audit(), else the HF fields (stop list, thinking, ctx)."""
    if hasattr(eng, "audit"):
        return eng.audit()
    out = {k: getattr(eng, k, None) for k in ("stop_ids", "eos", "eot", "qwen3", "ctx", "has_template",
                                              "trust_remote_code", "attn")}
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


_DIR_WARNED = set()


def fsync_path(path, is_dir=False):
    """fsync a file, or a directory (so a rename in it is on disk). The rule of harness/durable.py; rc12 is synced to
    the PC alone, so its few lines live here. A file fsync error raises. A directory fsync is best effort on every
    platform: the files are fsynced first, so a refusal (a drvfs mount such as /mnt/d, FAT, a network fs) can lose
    only the rename, never publish empty files; it is warned once per error kind on stderr and the run goes on."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as e:
        if not is_dir:
            raise
        code = errno.errorcode.get(e.errno, str(e.errno))
        if code not in _DIR_WARNED:
            _DIR_WARNED.add(code)
            print(f"dev_batch: directory fsync refused on {path} ({code}): a rename there may not survive a power cut",
                  file=sys.stderr, flush=True)


def publish_run(tmp, final):
    """every file of the finished run (transcripts, scores, meta.json, DONE) and the run dir fsynced, then the rename,
    then the parent fsynced: a power cut cannot leave a published run dir holding DONE and empty files (incident
    2026-09-29, experiments/E2_lr_transfer/notes.txt CRASH RESUME)."""
    for d, _, files in os.walk(tmp, topdown=False):
        for n in sorted(files):
            fsync_path(os.path.join(d, n))
        fsync_path(d, is_dir=True)
    os.replace(tmp, final)
    fsync_path(os.path.dirname(os.path.abspath(final)), is_dir=True)


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
    publish_run(tmp, final)
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
    hfo = hf_options(model_id, args.engines)
    hf_only = ("attn_implementation", "nan_guard", "use_cache")
    if (hfo["trust_remote_code"] or any(hfo[k] is not None for k in hf_only)) and kind not in ("hf", "hfb"):
        sys.exit(f"engines.json sets HF load options for {model_id}: hf / hfb only, not {kind}")
    for k in ("max_batch", "token_budget"):
        if hfo[k] is not None and (kind != "hfb" or type(hfo[k]) is not int or hfo[k] < 1):
            sys.exit(f"engines.json {k} {hfo[k]!r} for {model_id}: a positive int, hfb only (not {kind})")
    for k in ("chat_template", "nan_guard"):
        if hfo[k] is not None and (type(hfo[k]) is not str or not hfo[k]):
            sys.exit(f"engines.json {k} for {model_id}: a non-empty string")
    if hfo["use_cache"] is not None and type(hfo["use_cache"]) is not bool:
        sys.exit(f"engines.json use_cache for {model_id}: true or false")
    if chosen is not None and chosen != kind and not args.no_parity_gate:
        sys.exit(f"engine gate: engines.json names {chosen} for {model_id}, not {kind}")
    try:
        apply_decode(model_id, args.engines, kind)          # meta.json "decode" records HR.DECODE as run
    except ValueError as e:
        sys.exit(str(e))
    if gated and verdict not in PASSING and chosen != kind and not args.no_parity_gate:
        sys.exit(f"parity gate: {model_id} has parity verdict {verdict}; run parity_hf_vllm.py first")
    todo = plan(root, model_id, args.render, R.parse_seeds(args.seeds))
    if not todo:
        print("nothing to run: every requested run is DONE")
        return 0
    ns = argparse.Namespace(render=args.render, dtype=dtype, device=args.device, gpu_mem=args.gpu_mem,
                            max_model_len=args.max_model_len, batch_invariant=args.batch_invariant, lockstep=True,
                            hf_untested_ok=args.hf_untested_ok, vllm_untested_ok=kind == "vllm",
                            **hfo)
    eng, name = R.make_responder(args.responder, ns)
    ctx = getattr(eng, "ctx", None)
    recs, own = R.load(args.data, None, args.limit), R.load(args.data, ["OWN"], args.limit)
    extra = dict(engine=kind, parity=verdict, engine_decision=chosen,
                 parity_gate_skipped=args.no_parity_gate and verdict not in PASSING and chosen != kind,
                 limit=args.limit, train_seed=args.train_seed, gpu_mem=args.gpu_mem, data=os.path.basename(args.data),
                 dtype=dtype, max_model_len=args.max_model_len, batch_invariant=args.batch_invariant,
                 decode=HR.DECODE, audit=engine_audit(eng), executable=sys.executable, **hfo)
    for seed, own_cf, final in todo:
        run_one(eng, name, own if own_cf else recs, args.render, seed, own_cf, final, ctx, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
