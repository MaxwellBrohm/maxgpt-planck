"""E005 AL scoring (notes.txt ADDITION and STEP 5): score the 64 AL items on ONE model in ONE process, no training.
Run only through guard.py. The name contains "ft_test.py" on purpose: every E001-E005 guard counts a python command
line containing it as a model job and refuses to overlap it.
Items: ../al/al_items.jsonl, refused unless its sha256 equals ../al/al_items.sha256 (recorded in notes.txt STEP 5).
Code paths are E005's scoring calls (e005_ft_test.py), fed the AL items instead of the E004 draw:
  load     e004_ft_test.load (fp32, the model's own tokenizer), e004_core.load_checks (a weights directory reads its
           own config.json; an HF id reads the cached snapshot, as E004/E005)
  LIK      e004_sets.lik_item -> e004_core.score_set (lik.py: forced prefix, gold vs every other in-context value,
           right only if the gold beats each of them), plain render and SmolLM2's chat render
  GEN      e004_sets.gen_item -> gen_run.run_gen (greedy, 48 new tokens, hardened strict grader gen_grade.grade;
           plain stops at a newline or "User:", chat at end of turn), plain and chat
  --dry    4 items per render, the E004 scorer self-test (a tie and a gold-only set must not count as right)
  --plan   prints the plan and exits; loads no model, no tokenizer, no torch
usage: e005_al_ft_test.py <hf_model_id | weights dir> --tag T [--device mps] [--dry] [--plan]
Writes ../al/out/<tag>__al__<render>.jsonl (LIK), ../al/out/<tag>__gen_al__<render>.jsonl (GEN),
../al/out/<tag>__run.json (checks, item sha256, weights sha256, per-cell summary)."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, hashlib, json, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items_al as A
import e004_sets as S4
import gen_run as R

OUT = os.path.join(A.AL_DIR, "out")
RENDERS = ("plain", "chat")


def build_parser():
    ap = argparse.ArgumentParser(prog="e005_al_ft_test.py")
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--plan", action="store_true")
    return ap


def sets(dry):
    items = A.load()
    if len(items) != 64:
        sys.exit(f"refusing: {len(items)} AL items, expected 64")
    if dry:
        items = [it for c in A.CELLS for it in items if it["cell"] == c][::16]
    lik = [S4.lik_item(it) for it in items]
    gen = [S4.gen_item(it, "al") for it in items]
    return items, lik, gen


def stem_of(tag):
    return os.path.join(OUT, tag)


def plan(a):
    items, lik, gen = sets(a.dry)
    s = stem_of(a.tag)
    outs = [f"{s}__al__{r}.jsonl" for r in RENDERS] + [f"{s}__gen_al__{r}.jsonl" for r in RENDERS] + [f"{s}__run.json"]
    return dict(model=a.model, tag=a.tag, dry=a.dry, n_items=len(items), items_sha256=items_sha(),
                lik=[("al", r, len(lik)) for r in RENDERS], gen=[("al", r, len(gen)) for r in RENDERS], outputs=outs)


def items_sha():
    return hashlib.sha256(open(A.ITEMS_PATH, "rb").read()).hexdigest()


def weights_file(model):
    if os.path.isdir(model):
        return os.path.join(model, "model.safetensors")
    import params as PR
    return os.path.join(PR.snapshot_dir(model), "model.safetensors")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def patch_local_config():
    """e004_core.load_checks reads config.json from the HF cache; a weights directory has its own."""
    import params as PR
    orig = PR.load_config
    PR.load_config = lambda mid: (json.load(open(os.path.join(mid, "config.json"))) if os.path.isdir(mid)
                                  else orig(mid))


def by_cell(recs, key):
    out = {}
    for c in ("all",) + A.CELLS:
        xs = [bool(r[key]) for r in recs if c == "all" or r.get("cell") == c]
        out[c] = [sum(xs), len(xs)]
    return out


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.plan:
        print(json.dumps(plan(a), indent=1))
        return
    import e004_ft_test as M4
    M4.check_env(a)
    items, lik_its, gen_its = sets(a.dry)
    import torch
    import e004_core as C
    import metrics_ft as MF
    patch_local_config()
    os.makedirs(OUT, exist_ok=True)
    stem, t0 = stem_of(a.tag), time.time()
    wf = weights_file(a.model)
    meta = dict(model=a.model, tag=a.tag, dry=a.dry, device=a.device, items=A.ITEMS_PATH, items_sha256=items_sha(),
                weights_file=wf, weights_sha256=sha_file(wf), torch=torch.__version__, max_new=R.MAX_NEW,
                watermarks=[os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                            os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")])
    torch.manual_seed(0)
    model, tok, info = M4.load(a)
    if not getattr(tok, "chat_template", None):
        sys.exit(f"refusing: {a.model} has no chat template (AL scores plain and chat)")
    mem = C.Mem(a.device)
    checks, fatal = C.load_checks(model, info, a.model, a.device, tok)
    meta["checks"] = checks
    print(f"loaded {a.model} params={checks['params_model']:,} sanity_nll={checks['sanity_nll']} "
          f"{time.time() - t0:.1f}s {mem.s()}", flush=True)
    if fatal:
        meta["fatal"] = fatal
        json.dump(meta, open(f"{stem}__run.json", "w"), indent=1)
        sys.exit(f"FATAL load check: {fatal}")
    fails, summ = [], {}
    for render in RENDERS:
        with open(f"{stem}__al__{render}.jsonl", "w") as f:
            recs, dt = C.score_set(model, tok, a.model, "al", render, lik_its, a.device, f, mem)
        summ[f"LIK al/{render}"] = by_cell(recs, "right")
        print(f"scored al/{render}: {summ[f'LIK al/{render}']['all']} in {dt:.0f}s {mem.s()}", flush=True)
        if not C.finite_scores(recs):
            fails.append(f"non-finite score in al/{render}")
    empty = torch.mps.empty_cache if a.device == "mps" else None
    for render in RENDERS:
        t3 = time.time()
        recs = R.run_gen(model, tok, gen_its, render, a.device, f"{stem}__gen_al__{render}.jsonl", R.MAX_NEW, empty)
        summ[f"GEN al/{render}"] = dict(by_cell(recs, "strict"), capped=sum(r["capped"] for r in recs),
                                        lenient=sum(r["lenient"] for r in recs))
        print(f"generated al/{render}: strict {summ[f'GEN al/{render}']['all']} in {time.time() - t3:.0f}s", flush=True)
    if a.dry:
        M4.scorer_selftest(model, tok, argparse.Namespace(model=a.model, device=a.device), fails)
    meta.update(summary=summ, selftest_fail=fails, peak_mps_driver_gb=round(mem.peak_driver / 2**30, 3),
                total_s=round(time.time() - t0, 1), right_rule="metrics_ft.right: gold > every other candidate")
    assert MF.right({"gold": 1.0, "c1": 0.0}) and not MF.right({"gold": 1.0, "c1": 1.0})
    json.dump(meta, open(f"{stem}__run.json", "w"), indent=1)
    print(f"DONE {a.model} {a.tag} total {meta['total_s']:.0f}s peak_driver={meta['peak_mps_driver_gb']}G", flush=True)
    for m in fails:
        print("SELFTEST FAIL", m, flush=True)
    if fails:
        sys.exit(5)


if __name__ == "__main__":
    main()
