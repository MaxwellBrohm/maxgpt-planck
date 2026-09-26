"""E006 entry point: train ONE model (arm C, P or G) or load one (--weights, --steps 0), save and hash it, then score
it from the same process. ONE model, ONE process. Run on the PC only, through guard_e006_pc.py under gpu.lock.
The name contains "ft_test.py" on purpose (every E001-E006 guard counts such a process as a model job).
E005's entry point (e005_ft_test.py, e004_ft_test.py: parser, load, load checks, scorer self-test, scoring calls,
not edited) with, per notes.txt:
  device   --device cuda (default); numerics_e006.apply() before the model loads (fp32, no TF32 unless --tf32,
           deterministic algorithms, SDPA math kernel only); versions, CUDA, driver and GPU into run.json
  arms     C = E005's loop and stream; P = E005's loop, stream train_e006p.stream_p; G = e006_train.g_train with the
           replay pool --replay-pool (sha256 checked against its .sha256 file)
  weights  trained runs save to ../weights/<slug>__<tag> BEFORE scoring and hash model.safetensors; --weights DIR
           scores saved weights (the E004/E005 references); the untouched model hashes the snapshot's file
  records  every LIK and GEN record gets weights_sha256, tag and arm (a writer wrapper; e004_core/gen_run unedited);
           no output file is ever overwritten (a name that exists stops the job)
  sets     eval (draw 4004), big (BIG), h5l (H5L), al (AL), cont, know; plain and, with --chat, chat
usage: e006_ft_test.py HuggingFaceTB/SmolLM2-135M-Instruct --arm C|P|G --seed S [--weights DIR --steps 0] [--tf32]
       [--sets eval,big,h5l,al,cont,know] [--chat] [--tag T] [--save 1] [--dry] [--plan] [--replay-pool F]"""
import os
import hashlib
import json
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import e004_ft_test as M4
import e005_ft_test as M5
import e005_sets as S
import gen_run as R

SET_NAMES = ("eval", "big", "h5l", "al", "cont", "know")
SANITY_E005, SANITY_TOL = 1.9598, 1e-3
POOL = os.path.join(EXP, "replay", "replay_pool.jsonl")
STREAMS = {"C": M5.TRAIN_STREAM,
           "P": M5.TRAIN_STREAM + "; change (a)/(b) from R_pos = Random(1000 * seed + 47): train_e006p.stream_p",
           "G": M5.TRAIN_STREAM + "; plus one replay micro-batch per step (e006_train.g_train)"}


def build_parser():
    ap = M5.build_parser()
    ap.prog = "e006_ft_test.py"
    ap.set_defaults(device="cuda", sets="eval,big,h5l,al,cont,know")
    ap.add_argument("--arm", choices=("C", "P", "G"))
    ap.add_argument("--weights", default="")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--replay-pool", default=POOL)
    return ap


def resolve(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    a.set_names = {s.strip() for s in a.sets.split(",") if s.strip()}
    if not a.set_names or not a.set_names <= set(SET_NAMES):
        ap.error(f"unknown --sets {a.sets!r} (allowed: {', '.join(SET_NAMES)})")
    if a.dry:
        a.steps = 5
    if a.weights and a.steps:
        ap.error("--weights scores saved weights: use --steps 0")
    if a.steps > 0 and not a.arm:
        ap.error("training needs --arm C, P or G")
    if a.max_new != R.MAX_NEW:
        ap.error(f"--max-new is pre-registered at {R.MAX_NEW}")
    a.tag = a.tag or ("dry" if a.dry else "base" if a.steps == 0 and not a.weights else f"{a.arm}{a.seed}")
    return a


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pool_rows(path):
    raw = open(path, "rb").read()
    want = open(path.replace(".jsonl", ".sha256")).read().split()[0]
    if hashlib.sha256(raw).hexdigest() != want:
        sys.exit(f"refusing: {path} does not match its .sha256")
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()], want


def extra_sets(names, chat, dry):
    """-> (lik sets, gen sets) for big, h5l, al: [(set name, render, items)]."""
    import e004_sets as S4
    import items_big_e006 as BG
    lik, gen = [], []
    renders = ("plain", "chat") if chat else ("plain",)
    if names & {"big", "h5l"}:
        D = BG.load()
        for name, fams in (("big", BG.BIG_FAMS), ("h5l", ("H5L",))):
            if name in names:
                its = [it for f in fams for it in D[f]]
                lik += [(name, r, [S4.lik_item(it) for it in its]) for r in renders]
                gen += [(name, r, [S4.gen_item(it, name) for it in its]) for r in renders]
    if "al" in names:
        import e005_al_ft_test as AL
        _, al_lik, al_gen = AL.sets(False)
        lik += [("al", r, al_lik) for r in ("plain", "chat")]
        gen += [("al", r, al_gen) for r in ("plain", "chat")]
    if dry:
        lik, gen = [(s, r, x[:2]) for s, r, x in lik], [(s, r, x[:2]) for s, r, x in gen]
    return lik, gen


class RecWriter:
    """a write-once jsonl file whose every record gets the run's weights hash, tag and arm."""

    def __init__(self, path, extra):
        self.f, self.extra = open(path, "x"), extra

    def write(self, line):
        for l in line.splitlines():
            if l.strip():
                self.f.write(json.dumps(dict(json.loads(l), **self.extra)) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()


def plan(a):
    lik, gen = extra_sets(a.set_names, a.chat, a.dry)
    base = [(s, r, len(x)) for s, r, x in S.lik_sets(a.set_names & {"eval", "cont", "know"}, a.chat, a.dry)]
    return dict(model=a.model, weights=a.weights or None, arm=a.arm, seed=a.seed, steps=a.steps, lr=a.lr, tag=a.tag,
                device=a.device, tf32=a.tf32, sets=sorted(a.set_names), chat=a.chat, dry=a.dry,
                lik=base + [(s, r, len(x)) for s, r, x in lik], gen=[(s, r, len(x)) for s, r, x in gen])


def main(argv=None):
    a = resolve(argv)
    if a.plan:
        print(json.dumps(plan(a), indent=1))
        return
    import numerics_e006 as NU
    numerics = NU.apply(a.tf32)
    import torch
    import e004_core as C
    import e006_train as TR
    import e006_score as SC
    out_dir = M4.OUT
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{a.model.replace('/', '__')}__{a.tag}")
    if os.path.exists(f"{stem}__run.json"):
        sys.exit(f"refusing: {stem}__run.json exists (outputs are never overwritten; rerun as <tag>_r2)")
    t0 = time.time()
    torch.manual_seed(a.seed)
    model, tok, info, wfile = SC.load(a)
    checks, fatal = C.load_checks(model, info, a.weights or a.model, a.device, tok)
    if not a.weights and abs(checks["sanity_nll"] - SANITY_E005) > SANITY_TOL:
        fatal = fatal or f"sanity NLL {checks['sanity_nll']} differs from E005's {SANITY_E005} by > {SANITY_TOL}"
    pool, pool_sha = (pool_rows(a.replay_pool) if a.arm == "G" and a.steps > 0 else (None, None))
    meta = dict(model=a.model, weights_in=a.weights or None, arm=a.arm, seed=a.seed, steps=a.steps, lr=a.lr, bs=a.bs,
                accum=a.accum, max_len=a.max_len, grad_ckpt=a.grad_ckpt, dtype="fp32", device=a.device, dry=a.dry,
                tag=a.tag, sets=sorted(a.set_names), chat=a.chat, max_new=a.max_new, checks=checks,
                numerics=numerics, tf32=a.tf32, replay_pool_sha256=pool_sha, params=checks["params_model"],
                train_stream=STREAMS.get(a.arm) if a.steps > 0 else None)
    if fatal:
        SC.write_json(f"{stem}__run.json", dict(meta, fatal=fatal))
        sys.exit(f"FATAL load check: {fatal}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    mem = C.Mem(a.device)
    fails, losses, traj, stats = [], [], [], S.new_stats()
    if a.steps > 0:
        S.eot_id(tok)
        if a.arm == "G":
            losses, traj, stats = TR.g_train(model, tok, a, meta, mem, pad, S.probe_items(a.dry), fails, pool)
        else:
            losses, traj, stats = TR.train_cp(model, tok, a, meta, mem, pad, S.probe_items(a.dry), fails)
    meta.update(train_stats=S.stream_summary(stats) if a.steps > 0 else None, losses=losses, probe=traj,
                lock_in_step=C.lock_in(traj), loss_finite=all(x == x and abs(x) != float("inf") for x in losses))
    wsha, wrel = SC.save_and_hash(model, tok, a, wfile, EXP)
    meta.update(weights_sha256=wsha, weights=wrel)
    extra = {"weights_sha256": wsha, "tag": a.tag, "arm": a.arm}
    model.eval()
    lik2, gen2 = extra_sets(a.set_names, a.chat, a.dry)
    meta.update(SC.score_all(model, tok, a, stem, extra, lik2, gen2, fails, RecWriter))
    if a.dry:
        M4.scorer_selftest(model, tok, a, fails)
    meta.update(selftest_fail=fails, total_s=round(time.time() - t0, 1),
                peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None)
    SC.write_json(f"{stem}__run.json", meta)
    print(f"DONE {a.model} {a.tag} total {meta['total_s']:.0f}s weights_sha256={wsha[:16]} "
          f"lock_in_step={meta['lock_in_step']} loss_finite={meta['loss_finite']}", flush=True)
    for m in fails:
        print("SELFTEST FAIL", m, flush=True)
    if fails:
        sys.exit(5)


if __name__ == "__main__":
    main()
