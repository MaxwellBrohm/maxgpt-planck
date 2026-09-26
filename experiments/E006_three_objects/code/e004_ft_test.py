"""E004 general updating: train one model on the E004 generator and score it. ONE model, ONE run, ONE process.
Run only through guard.py (one model process at a time). The file name contains "ft_test.py" on purpose: every
E001-E004 guard treats a python command line containing it as a model job and refuses to overlap it.

Adapted from E002 ft_test.py and E003 e003_ft_test.py (same loss, schedule, checks and scorer):
  data     train_e004.stream(seed) = Random(1000 * seed + 44); target = the answer sentence after "Assistant:"
           (loss on its tokens only, "\\n" included); > --max-len tokens redrawn, never truncated (e004_sets.py)
  sets     eval  E004 held-out eval draw (4004): LIK (forced prefix, lik.py) + free GEN (hardened grader)
           cont  E002 continuity sets re-scored unchanged + continuity free generation
           know  closed-book knowledge (khard, kbig)
           dev   E004 dev draw (4104), LIK only: the learning-rate choice; LR runs pass --sets dev and never
                 touch the eval draw
  probe    E004 probe draw (4204, 16 per family), LIK every --probe-every steps; lock-in = first probe step from
           which every one of the 9 pass-rule families stays >= 0.8
  chat     --chat adds SmolLM2's own chat-template render (reported, not ruled; base models have no template)
  checks   E003's load checks, loss self-check vs the HF loss (every run), --dry: 5 steps padded to --max-len,
           2 items per set, loss mutants, scorer self-test (tie and gold-only must not count as right)
  --steps 0 scores the untouched model (the paired baseline).
  --plan   prints the resolved configuration, the sets and the output paths, then exits: loads no model, no
           tokenizer, touches no device (the static self-check uses it).
usage: e004_ft_test.py <hf_model_id> --seed S [--steps 400] [--lr 5e-5] [--bs 4] [--accum 4] [--max-len 768]
                       [--grad-ckpt] [--sets eval,cont,know | dev] [--chat] [--tag T] [--save 0|1] [--dry] [--plan]
Writes ../out/<slug>__<tag>__<set>__<render>.jsonl (LIK), ../out/<slug>__<tag>__gen_<set>__<render>.jsonl (GEN),
../out/<slug>__<tag>__run.json."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, HERE)
import e004_sets as S
import gen_run as R

WATERMARKS = ("0.7", "0.6")


def build_parser():
    ap = argparse.ArgumentParser(prog="e004_ft_test.py")
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4, help="micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = bs * accum)")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--sets", default="eval,cont,know", help="comma list of: eval, cont, know, dev")
    ap.add_argument("--chat", action="store_true", help="also score the model's chat-template render")
    ap.add_argument("--max-new", type=int, default=R.MAX_NEW)
    ap.add_argument("--dry", action="store_true", help="5 steps at worst-case length, 2 items per set, self-tests")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit; loads nothing")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", type=int, default=0, help="save fine-tuned weights (scratch, for post-hoc re-scoring)")
    return ap


def resolve(argv=None):
    """parse and validate; -> args with set_names and tag. Raises SystemExit on a bad command line."""
    ap = build_parser()
    a = ap.parse_args(argv)
    a.set_names = {s.strip() for s in a.sets.split(",") if s.strip()}
    if not a.set_names or not a.set_names <= set(S.SET_NAMES):
        ap.error(f"unknown --sets {a.sets!r} (allowed: {', '.join(S.SET_NAMES)})")
    if "dev" in a.set_names and a.set_names & {"eval"}:
        ap.error("--sets dev (LR choice) must not be combined with the eval draw")
    if a.steps < 0 or a.bs < 1 or a.accum < 1 or a.max_len < 64 or a.probe_every < 1 or a.lr <= 0:
        ap.error("steps >= 0, bs >= 1, accum >= 1, max-len >= 64, probe-every >= 1 and lr > 0 are required")
    if a.max_new != R.MAX_NEW:
        ap.error(f"--max-new is pre-registered at {R.MAX_NEW}")
    if a.dry:
        a.steps = 5
    a.tag = a.tag or ("dry" if a.dry else ("base" if a.steps == 0 else f"s{a.seed}"))
    return a


def stem_of(a):
    return os.path.join(OUT, f"{a.model.replace('/', '__')}__{a.tag}")


def plan(a):
    """-> dict: the sets, item counts and output paths this run would produce (no model, no tokenizer)."""
    lik = [(s, r, len(its)) for s, r, its in S.lik_sets(a.set_names, a.chat, a.dry)]
    gen = [(s, r, len(its)) for s, r, its in S.gen_sets(a.set_names, a.chat, a.dry)]
    stem = stem_of(a)
    outs = [f"{stem}__{s}__{r}.jsonl" for s, r, _ in lik] + [f"{stem}__gen_{s}__{r}.jsonl" for s, r, _ in gen]
    return dict(model=a.model, seed=a.seed, steps=a.steps, lr=a.lr, eff_batch=a.bs * a.accum, max_len=a.max_len,
                sets=sorted(a.set_names), chat=a.chat, tag=a.tag, dry=a.dry, lik=lik, gen=gen,
                probe=len(S.probe_items(a.dry)) if a.steps > 0 else 0, outputs=outs + [f"{stem}__run.json"])


def check_env(a):
    if a.device == "mps":
        hw, lw = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")
        if (hw, lw) != WATERMARKS:
            sys.exit(f"refusing: MPS watermarks must be 0.7/0.6 (got {hw}/{lw}); run through guard.py")


def load(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model, info = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, low_cpu_mem_usage=True,
                                                           output_loading_info=True)
    except TypeError:
        model, info = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, low_cpu_mem_usage=True), None
    return model.to(a.device).eval(), tok, info


def scorer_selftest(model, tok, a, fails):
    """on this model: identical candidates (a tie) and a gold-only set must not count as right."""
    import lik
    import metrics_ft as MF
    it = S.e004_lik("dev", 1, ["H1"])[0]
    prompt, add_sp = lik.render(it, "plain", tok, a.model)
    g = it["cands"]["gold"]
    sc, _ = lik.score_item(model, tok, prompt, {"gold": g, "c1": g}, a.device, add_sp)
    if MF.right({k: v[0] for k, v in sc.items()}):
        fails.append(f"identical candidates scored as right: {sc}")
    sc, _ = lik.score_item(model, tok, prompt, {"gold": g}, a.device, add_sp)
    if MF.right({k: v[0] for k, v in sc.items()}):
        fails.append("gold-only candidate set scored as right")


def main(argv=None):
    a = resolve(argv)
    if a.plan:
        print(json.dumps(plan(a), indent=1))
        return
    check_env(a)
    import torch
    import e004_core as C
    import e004_train as TR
    os.makedirs(OUT, exist_ok=True)
    stem = stem_of(a)
    t0 = time.time()
    torch.manual_seed(a.seed)
    model, tok, info = load(a)
    if a.chat and not getattr(tok, "chat_template", None):
        sys.exit(f"refusing: --chat, but {a.model} has no chat template (base models are scored plain only)")
    mem = C.Mem(a.device)
    checks, fatal = C.load_checks(model, info, a.model, a.device, tok)
    print(f"loaded {a.model} params={checks['params_model']:,} (config {checks['params_config']['total']:,}, "
          f"body {checks['params_config']['body']:,}) sanity_nll={checks['sanity_nll']} {time.time()-t0:.1f}s {mem.s()}",
          flush=True)
    if fatal:
        json.dump({"model": a.model, "tag": a.tag, "fatal": fatal, "checks": checks}, open(f"{stem}__run.json", "w"), indent=1)
        sys.exit(f"FATAL load check: {fatal}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    meta = dict(model=a.model, seed=a.seed, steps=a.steps, lr=a.lr, bs=a.bs, accum=a.accum, eff_batch=a.bs * a.accum,
                max_len=a.max_len, grad_ckpt=a.grad_ckpt, dtype="fp32", device=a.device, dry=a.dry, tag=a.tag,
                sets=sorted(a.set_names), chat=a.chat, max_new=a.max_new, params=checks["params_model"],
                checks=checks, trained_ctx=C.trained_ctx(a.model), torch=torch.__version__,
                train_stream="train_e004.stream(seed) = Random(1000 * seed + 44)",
                watermarks=[os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                            os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")])
    fails, losses, traj, stats = [], [], [], S.new_stats()
    if a.steps > 0:
        losses, traj, stats = TR.train(model, tok, a, meta, mem, pad, S.probe_items(a.dry), fails)
    meta.update(train_stats=TR.stream_summary(stats), losses=losses, probe=traj, lock_in_step=C.lock_in(traj),
                loss_finite=all(x == x and abs(x) != float("inf") for x in losses))
    model.eval()
    t2, counts, summ = time.time(), {}, {}
    for set_name, render, its in S.lik_sets(a.set_names, a.chat, a.dry):
        with open(f"{stem}__{set_name}__{render}.jsonl", "w") as f:
            recs, dt = C.score_set(model, tok, a.model, set_name, render, its, a.device, f, mem)
        counts[f"{set_name}/{render}"] = len(recs)
        if set_name in ("e004", "dev"):
            summ[f"LIK {set_name}/{render}"] = C.family_acc(recs, S.E.FAMILIES)
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
        scorer_selftest(model, tok, a, fails)
        meta["selftest_fail"] = fails
    meta.update(peak_mps_driver_gb=round(mem.peak_driver / 2**30, 3), peak_mps_alloc_gb=round(mem.peak_alloc / 2**30, 3),
                total_s=round(time.time() - t0, 1))
    if a.save and a.steps > 0 and not a.dry:
        try:  # scratch copy for post-hoc re-scoring; experiments/**/weights/ is gitignored
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
