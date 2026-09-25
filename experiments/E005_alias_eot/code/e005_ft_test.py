"""E005 alias / end-of-turn: train one model on the E005 stream and score it. ONE model, ONE run, ONE process.
Run only through guard.py (one model process at a time). The name contains "ft_test.py" on purpose: every
E001-E005 guard treats a python command line containing it as a model job and refuses to overlap it.

E004's entry point (e004_ft_test.py, not edited; its parser checks, plan, load checks, scorer self-test and scoring
loop are reused) with the E005 training data (notes.txt THE THREE CHANGES):
  data     train_e005.stream(seed) through e005_sets: every example in its drawn render, plain (E004's target
           " answer\\n") or chat (SmolLM2's template, target "answer<|im_end|>"); > --max-len tokens redrawn
  checks   loss self-check vs the HF loss on the first batch, a plain batch and a chat batch (e005_train);
           --dry: 5 steps padded to --max-len, wrong-loss mutants on all three batches, both renders trained,
           2 items per set, scorer self-test
  sets     unchanged from E004 (eval LIK + GEN, plain and with --chat the chat render; cont; know; probe)
  --lr     defaults to 1.5e-4 (E004's dev pick, pre-registered for E005; no LR search in E005)
usage: e005_ft_test.py <hf_model_id> --seed S [--steps 400] [--lr 1.5e-4] [--bs 4] [--accum 4] [--max-len 768]
                       [--grad-ckpt] [--sets eval,cont,know] [--chat] [--tag T] [--save 0|1] [--dry] [--plan]
Writes ../out/<slug>__<tag>__<set>__<render>.jsonl (LIK), ../out/<slug>__<tag>__gen_<set>__<render>.jsonl (GEN),
../out/<slug>__<tag>__run.json (the same names as E004, in E005's out/)."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import json, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e004_ft_test as M4
import e005_sets as S
import gen_run as R
import train_e005 as T5

LR_E005 = 1.5e-4
TRAIN_STREAM = ("train_e005.stream(seed): R_mix = Random(1000 * seed + 46) draws block (E004 .70 = train_e004.stream"
                "(seed) in order, ALIAS .20, IND .10; R_new = Random(1000 * seed + 45)) then render (chat .50)")


def build_parser():
    ap = M4.build_parser()
    ap.prog = "e005_ft_test.py"
    ap.set_defaults(lr=LR_E005)
    return ap


def resolve(argv=None):
    """E004's resolve (same checks and tags) on the E005 parser."""
    ap = build_parser()
    a = ap.parse_args(argv)
    a.set_names = {s.strip() for s in a.sets.split(",") if s.strip()}
    if not a.set_names or not a.set_names <= set(S.SET_NAMES):
        ap.error(f"unknown --sets {a.sets!r} (allowed: {', '.join(S.SET_NAMES)})")
    if "dev" in a.set_names and a.set_names & {"eval"}:
        ap.error("--sets dev must not be combined with the eval draw")
    if a.steps < 0 or a.bs < 1 or a.accum < 1 or a.max_len < 64 or a.probe_every < 1 or a.lr <= 0:
        ap.error("steps >= 0, bs >= 1, accum >= 1, max-len >= 64, probe-every >= 1 and lr > 0 are required")
    if a.max_new != R.MAX_NEW:
        ap.error(f"--max-new is pre-registered at {R.MAX_NEW}")
    if a.dry:
        a.steps = 5
    a.tag = a.tag or ("dry" if a.dry else ("base" if a.steps == 0 else f"s{a.seed}"))
    return a


def plan(a):
    p = M4.plan(a)
    if a.steps > 0:
        p.update(train_stream=TRAIN_STREAM, blocks=T5.BLOCKS, p_chat=T5.P_CHAT,
                 chat_target="sentence + <|im_end|> (labels on both only)", plain_target="' ' + sentence + '\\n'")
    return p


def main(argv=None):
    a = resolve(argv)
    if a.plan:
        print(json.dumps(plan(a), indent=1))
        return
    M4.check_env(a)
    import torch
    import e004_core as C
    import e005_train as TR
    os.makedirs(M4.OUT, exist_ok=True)
    stem = M4.stem_of(a)
    t0 = time.time()
    torch.manual_seed(a.seed)
    model, tok, info = M4.load(a)
    if a.chat and not getattr(tok, "chat_template", None):
        sys.exit(f"refusing: --chat, but {a.model} has no chat template")
    if a.steps > 0:
        try:
            S.eot_id(tok)
        except ValueError as e:
            sys.exit(f"refusing to train: the E005 stream has chat-rendered examples and {e}")
    mem = C.Mem(a.device)
    checks, fatal = C.load_checks(model, info, a.model, a.device, tok)
    print(f"loaded {a.model} params={checks['params_model']:,} sanity_nll={checks['sanity_nll']} "
          f"{time.time()-t0:.1f}s {mem.s()}", flush=True)
    if fatal:
        json.dump({"model": a.model, "tag": a.tag, "fatal": fatal, "checks": checks}, open(f"{stem}__run.json", "w"), indent=1)
        sys.exit(f"FATAL load check: {fatal}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    meta = dict(model=a.model, seed=a.seed, steps=a.steps, lr=a.lr, bs=a.bs, accum=a.accum, eff_batch=a.bs * a.accum,
                max_len=a.max_len, grad_ckpt=a.grad_ckpt, dtype="fp32", device=a.device, dry=a.dry, tag=a.tag,
                sets=sorted(a.set_names), chat=a.chat, max_new=a.max_new, params=checks["params_model"],
                checks=checks, trained_ctx=C.trained_ctx(a.model), torch=torch.__version__,
                train_stream=TRAIN_STREAM if a.steps > 0 else None, p_chat=T5.P_CHAT,
                watermarks=[os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                            os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")])
    fails, losses, traj, stats = [], [], [], S.new_stats()
    if a.steps > 0:
        losses, traj, stats = TR.train(model, tok, a, meta, mem, pad, S.probe_items(a.dry), fails)
    meta.update(train_stats=TR.stream_summary(stats) if a.steps > 0 else None, losses=losses, probe=traj,
                lock_in_step=C.lock_in(traj), loss_finite=all(x == x and abs(x) != float("inf") for x in losses))
    model.eval()
    t2, counts, summ = time.time(), {}, {}
    for set_name, render, its in S.lik_sets(a.set_names, a.chat, a.dry):
        with open(f"{stem}__{set_name}__{render}.jsonl", "w") as f:
            recs, dt = C.score_set(model, tok, a.model, set_name, render, its, a.device, f, mem)
        counts[f"{set_name}/{render}"] = len(recs)
        if set_name in ("e004", "dev"):
            summ[f"LIK {set_name}/{render}"] = C.family_acc(recs, S.S4.E.FAMILIES)
        print(f"scored {set_name}/{render}: {len(recs)} items in {dt:.0f}s {mem.s()}", flush=True)
        if a.dry and not C.finite_scores(recs):
            fails.append(f"non-finite score in {set_name}/{render}")
    empty = torch.mps.empty_cache if a.device == "mps" else None
    for set_name, render, its in S.gen_sets(a.set_names, a.chat, a.dry):
        t3 = time.time()
        recs = R.run_gen(model, tok, its, render, a.device, f"{stem}__gen_{set_name}__{render}.jsonl", R.MAX_NEW, empty)
        counts[f"gen_{set_name}/{render}"] = len(recs)
        n_s = sum(r["strict"] for r in recs)
        summ[f"GEN {set_name}/{render}"] = {"strict": n_s, "lenient": sum(r["lenient"] for r in recs), "n": len(recs),
                                           "capped": sum(r["capped"] for r in recs), "bare": sum(r["bare"] for r in recs)}
        print(f"generated {set_name}/{render}: strict {n_s}/{len(recs)} in {time.time()-t3:.0f}s {mem.s()}", flush=True)
    meta.update(eval_s=round(time.time() - t2, 1), eval_counts=counts, summary=summ)
    if a.dry:
        M4.scorer_selftest(model, tok, a, fails)
        meta["selftest_fail"] = fails
    meta.update(peak_mps_driver_gb=round(mem.peak_driver / 2**30, 3), peak_mps_alloc_gb=round(mem.peak_alloc / 2**30, 3),
                total_s=round(time.time() - t0, 1))
    if a.save and a.steps > 0 and not a.dry:
        try:  # for the chat probe and AL re-scoring; experiments/**/weights/ is gitignored
            wdir = os.path.join(os.path.dirname(HERE), "weights", f"{a.model.replace('/', '__')}__{a.tag}")
            model.save_pretrained(wdir, safe_serialization=True)
            tok.save_pretrained(wdir)
            meta["weights"] = wdir
        except Exception as e:  # never lose the eval over a failed save
            meta["weights_error"] = repr(e)
    json.dump(meta, open(f"{stem}__run.json", "w"), indent=1)
    print(f"DONE {a.model} {a.tag} total {meta['total_s']:.0f}s peak_driver={meta['peak_mps_driver_gb']}G "
          f"lock_in_step={meta['lock_in_step']} loss_finite={meta['loss_finite']}", flush=True)
    for m in fails:
        print("SELFTEST FAIL", m, flush=True)
    if fails:
        sys.exit(5)


if __name__ == "__main__":
    main()
