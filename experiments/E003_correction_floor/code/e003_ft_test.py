"""E003 correction floor: E002's ft_test on much smaller BASE language models. ONE model, ONE run, ONE process.
Run only through guard.py (one model process at a time, memory and wall-clock kills).
(The file name contains "ft_test.py" on purpose: E001/E002's guard.py treat any python command line containing
that string as a model job, so an E002 or E001 guard will also refuse to overlap an E003 job.)

Question: at what BODY size does "400 short fine-tuning steps teach latest-value-after-a-correction" stop working?

Changes relative to E002 code/ft_test.py (everything else is the same code path):
  models   base LMs (Pythia GPT-NeoX, TinyStories GPT-Neo). The answer-only loss uses model.base_model and
           model.get_output_embeddings() instead of the Llama-only model.model / model.lm_head. The loss
           self-check against the HF loss (same as E002) runs on every training run and exits on a mismatch.
  renders  plain "User:/Assistant:" only. Base models have no chat template, so E002's chat-render sets are
           not scored. The pass rule was always plain-only.
  sets     --sets eval  = E002's plain sets: new (E001 items), extra (noupd_incid), old battery, uprobe, khard,
                          kbig (closed-book knowledge), cross (E002's post-hoc crossed control)
           --sets dev   = a SEPARATE draw of the pass-rule items (items_new seed 3003, d10, same_k1/2/3, twoslot,
                          noupd; 320 items) used ONLY to choose the learning rate. The LR-search runs score dev only
                          and never touch the eval items.
  checks   every run: parameter count vs params.py (config), loading info (no parameter left uninitialised),
           untied Pythia head is not the input embedding, and a sanity NLL on one plain sentence (< 8 nats/token;
           a random head is ~10.8). --dry adds: 5 steps padded to --max-len, 2 items per set, and a loss-mutant
           self-test (two wrong answer-only losses must NOT match the HF loss, else the self-check is vacuous).
  seeds    --seed sets both the torch seed and the training-data stream (as E002: Random(1000 * seed + 17)).

usage: e003_ft_test.py <hf_model_id> --seed S [--steps 400] [--lr 5e-5] [--bs 4] [--accum 4] [--max-len 768]
                       [--grad-ckpt] [--sets eval,dev] [--tag T] [--save 0|1] [--dry]
Writes ../out/<slug>__<tag>__<set>__plain.jsonl and ../out/<slug>__<tag>__run.json.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, hashlib, json, math, random, sys, time
import torch
import torch.nn.functional as Fnn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, HERE)
import items as I
import items_new as N
import eval_extra as X
import uprobe_items as UP
import khard_items as KH
import kbig_items as KB
import lik
import train_data as TD
import metrics_ft as MF
import params as PR

SANITY_TEXT = ("Once upon a time, there was a little girl named Lily. She liked to play with her red ball in the park. "
               "One day, she saw a big dog near the tree.")
SANITY_MAX_NLL = 8.0
DEV_SEED = 3003
DEV_VARS = ("same_k1", "same_k2", "same_k3", "twoslot", "noupd")
TRAINED_CTX = {"roneneldan/TinyStories": 512, "EleutherAI/pythia": 2048}   # TinyStories-33M card: context_length=512


def slug(m):
    return m.replace("/", "__")


def phash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def trained_ctx(model_id):
    for k, v in TRAINED_CTX.items():
        if model_id.startswith(k):
            return v
    return None


def dev_items():
    """The LR-selection draw: E001 pass-rule variants at d10, 32 scenarios x 2 families, seed 3003."""
    return [x for x in N.build(n_scen=32, distances=(10,), seed=DEV_SEED) if x["var"] in DEV_VARS]


# ---------------- memory telemetry ----------------
class Mem:
    def __init__(self, device):
        self.device, self.peak_driver, self.peak_alloc = device, 0, 0

    def sample(self):
        if self.device != "mps":
            return 0, 0
        a, d = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
        self.peak_alloc, self.peak_driver = max(self.peak_alloc, a), max(self.peak_driver, d)
        return a, d

    def s(self):
        a, d = self.sample()
        return f"mps_alloc={a/2**30:.2f}G driver={d/2**30:.2f}G peak_driver={self.peak_driver/2**30:.2f}G"


# ---------------- training data (as E002) ----------------
def encode(tok, ex):
    prompt = TD.transcript(ex["turns"], ex["question"], ex["prefix"])
    pre, ans, how = lik.cand_ids(tok, prompt, ex["answer"], True)
    ids = list(pre) + list(ans)
    labels = [-100] * len(pre) + list(ans)
    return ids, labels, how


def example_stream(tok, rng, max_len, stats):
    while True:
        ex = TD.gen(rng)
        ids, labels, how = encode(tok, ex)
        if len(ids) > max_len:
            stats["rejected_long"] += 1
            continue
        stats["kinds"][ex["kind"]] = stats["kinds"].get(ex["kind"], 0) + 1
        stats["n"] += 1
        stats["len_sum"] += len(ids)
        stats["len_max"] = max(stats["len_max"], len(ids))
        stats["split"] += how == "split"
        yield ids, labels


def make_batch(exs, pad_id, pad_to=None):
    L = pad_to or max(len(x) for x, _ in exs)
    B = len(exs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, (x, y) in enumerate(exs):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids, lab, att


def answer_loss(model, ids, att, lab):
    """Mean next-token cross-entropy over labelled (answer) positions only; architecture-generic.
    Same quantity as the HF loss, but the vocabulary projection runs only where a label exists."""
    h = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    m = tgt != -100
    logits = model.get_output_embeddings()(h[:, :-1][m])
    return Fnn.cross_entropy(logits.float(), tgt[m])


def close(a, b):
    return abs(a - b) <= 1e-3 * max(1.0, abs(a))


def loss_mutants(model, ids, att, lab):
    """Two WRONG answer-only losses; the self-check comparator must reject both (dry-run self-test)."""
    with torch.no_grad():
        h = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
        head = model.get_output_embeddings()
        m = lab != -100
        no_shift = Fnn.cross_entropy(head(h[m]).float(), lab[m]).item()        # predicts the token it reads
        lab2 = ids.clone()
        lab2[att == 0] = -100
        with_prompt = answer_loss(model, ids, att, lab2).item()                  # prompt tokens labelled
    return {"no_shift": no_shift, "prompt_labelled": with_prompt}


# ---------------- eval ----------------
def eval_sets(names, dry):
    sets = []
    if "eval" in names:
        sets += [("new", "plain", N.build()), ("extra", "plain", X.build()), ("old", "plain", I.build()),
                 ("uprobe", "plain", UP.build()), ("khard", "plain", KH.build()), ("kbig", "plain", KB.build()),
                 ("cross", "plain", X.build_crossed())]
    if "dev" in names:
        sets += [("dev", "plain", dev_items())]
    if dry:
        sets = [(s, r, its[:2]) for s, r, its in sets]
    return sets


def probe_items(dry):
    its = [x for x in N.build(n_scen=16, distances=(10,), seed=777) if x["var"] in ("same_k1", "twoslot", "noupd")]
    return its[:6] if dry else its


@torch.no_grad()
def score_set(model, tok, model_id, set_name, render, its, device, f=None, mem=None):
    recs = []
    t0 = time.time()
    for i, it in enumerate(its):
        if "turns" in it:
            prompt, add_sp = lik.render(it, render, tok, model_id)
        else:
            prompt, add_sp = it["prompt"], True
        sc, L = lik.score_item(model, tok, prompt, it["cands"], device, add_sp)
        rec = {"set": set_name, "render": render, "id": i, "h": phash(prompt), "seq_len": L}
        for kk in ("task", "cond", "d", "fam", "var", "sid", "k", "cat"):
            if kk in it:
                rec[kk] = it[kk]
        rec["scores"] = {lab: v[0] for lab, v in sc.items()}
        rec["ntok"] = {lab: v[1] for lab, v in sc.items()}
        recs.append(rec)
        if f is not None:
            f.write(json.dumps(rec) + "\n")
        if device == "mps" and i % 200 == 0:
            torch.mps.empty_cache()
            if mem is not None:
                mem.sample()
    return recs, time.time() - t0


def probe(model, tok, model_id, its, device):
    model.eval()
    recs, _ = score_set(model, tok, model_id, "probe", "plain", its, device)
    model.train()
    out = {}
    for name, var in (("LW", "same_k1"), ("TS", "twoslot"), ("NU", "noupd")):
        xs = [MF.right(r["scores"]) for r in recs if r["var"] == var]
        out[name] = round(sum(xs) / len(xs), 4) if xs else None
    return out


@torch.no_grad()
def sanity_nll(model, tok, device):
    ids = torch.tensor([tok(SANITY_TEXT).input_ids], device=device)
    return float(model(input_ids=ids, labels=ids).loss.item())


def load_checks(model, info, model_id, device, tok):
    """-> (dict, fatal message or None)."""
    cfg_count = PR.count(PR.load_config(model_id))
    n_params = sum(p.numel() for p in model.parameters())
    names = dict(model.named_parameters())
    missing = [k for k in (info or {}).get("missing_keys", []) if k in names]
    tied_head = cfg_count["out_head"] == 0
    head_w, emb_w = model.get_output_embeddings().weight, model.get_input_embeddings().weight
    shares = head_w.data_ptr() == emb_w.data_ptr()
    if tied_head:
        missing = [k for k in missing if not k.startswith("lm_head")]
    nll = sanity_nll(model, tok, device)
    out = {"params_config": cfg_count, "params_model": n_params, "params_match": n_params == cfg_count["total"],
           "missing_param_keys": missing, "unexpected_keys": sorted(map(str, (info or {}).get("unexpected_keys", [])))[:10],
           "mismatched_keys": [str(x) for x in (info or {}).get("mismatched_keys", [])][:10],
           "head_tied_to_input_embedding": shares, "config_says_tied": tied_head, "sanity_nll": round(nll, 4)}
    fatal = None
    if missing:
        fatal = f"parameters not in the checkpoint (would be random): {missing[:5]}"
    elif out["mismatched_keys"]:
        fatal = f"shape-mismatched keys: {out['mismatched_keys']}"
    elif shares != tied_head:
        fatal = f"output head tie state {shares} differs from config ({tied_head})"
    elif not (nll < SANITY_MAX_NLL):
        fatal = f"sanity NLL {nll:.2f} >= {SANITY_MAX_NLL} (random or broken head?)"
    return out, fatal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4, help="micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = bs * accum)")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--sets", default="eval", help="comma list of: eval, dev")
    ap.add_argument("--dry", action="store_true", help="5 steps at worst-case length, 2 items per set, self-tests")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", type=int, default=0, help="save fine-tuned weights (scratch, for post-hoc re-scoring)")
    a = ap.parse_args()
    set_names = {s.strip() for s in a.sets.split(",") if s.strip()}
    if not set_names <= {"eval", "dev"}:
        sys.exit(f"unknown --sets {a.sets}")
    if a.dry:
        a.steps = 5
    if a.device == "mps":
        hw, lw = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")
        if hw != "0.7" or lw != "0.6":
            sys.exit(f"refusing: MPS watermarks must be 0.7/0.6 (got {hw}/{lw}); run through guard.py")
    tag = a.tag or (("dry" if a.dry else ("base" if a.steps == 0 else f"s{a.seed}")))
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"{slug(a.model)}__{tag}")
    t0 = time.time()
    torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model, info = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, low_cpu_mem_usage=True,
                                                           output_loading_info=True)
    except TypeError:
        model, info = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, low_cpu_mem_usage=True), None
    model = model.to(a.device).eval()
    mem = Mem(a.device)
    checks, fatal = load_checks(model, info, a.model, a.device, tok)
    print(f"loaded {a.model} params={checks['params_model']:,} (config {checks['params_config']['total']:,}, "
          f"body {checks['params_config']['body']:,}) sanity_nll={checks['sanity_nll']} in {time.time()-t0:.1f}s {mem.s()}", flush=True)
    if not checks["params_match"]:
        print(f"PARAM MISMATCH model {checks['params_model']} vs config {checks['params_config']['total']}", flush=True)
    if fatal:
        json.dump({"model": a.model, "tag": tag, "fatal": fatal, "checks": checks}, open(f"{stem}__run.json", "w"), indent=1)
        sys.exit(f"FATAL load check: {fatal}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    meta = dict(model=a.model, seed=a.seed, steps=a.steps, lr=a.lr, bs=a.bs, accum=a.accum, eff_batch=a.bs * a.accum,
                max_len=a.max_len, grad_ckpt=a.grad_ckpt, dtype="fp32", device=a.device, dry=a.dry, tag=tag,
                sets=sorted(set_names), params=checks["params_model"], checks=checks, trained_ctx=trained_ctx(a.model),
                torch=torch.__version__,
                watermarks=[os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")])
    if a.dry and hasattr(model.base_model, "wpe"):
        w = model.base_model.wpe.weight.detach().float().norm(dim=1).cpu()
        meta["wpe_row_norm_mean"] = {"0-511": round(w[:512].mean().item(), 4), "512-767": round(w[512:768].mean().item(), 4),
                                     "768-2047": round(w[768:].mean().item(), 4)}
        print(f"wpe row norms {meta['wpe_row_norm_mean']}", flush=True)
    traj, losses = [], []
    stats = {"n": 0, "rejected_long": 0, "kinds": {}, "len_sum": 0, "len_max": 0, "split": 0}
    p_items = probe_items(a.dry)
    selftest_fail = []
    if a.steps > 0:
        if a.grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
        rng = random.Random(1000 * a.seed + 17)
        stream = example_stream(tok, rng, a.max_len, stats)
        # self-check: the answer-only loss equals the HF loss on one real batch
        ids, lab, att = make_batch([next(stream) for _ in range(a.bs)], pad)
        ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=att, labels=lab).loss.item()
            mine = answer_loss(model, ids, att, lab).item()
        meta["loss_selfcheck"] = {"hf": ref, "answer_only": mine}
        print(f"loss self-check hf={ref:.5f} answer_only={mine:.5f}", flush=True)
        if not close(ref, mine) or not math.isfinite(mine):
            sys.exit("loss self-check failed")
        if a.dry:
            mut = loss_mutants(model, ids, att, lab)
            meta["loss_mutants"] = mut
            for k, v in mut.items():
                if close(ref, v):
                    selftest_fail.append(f"loss mutant {k} ({v:.5f}) is not distinguished from the HF loss ({ref:.5f})")
            print(f"loss mutants {mut} (must all differ from {ref:.5f})", flush=True)
        traj.append({"step": 0, **probe(model, tok, a.model, p_items, a.device)})
        print(f"probe step 0 {traj[-1]} {mem.s()}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        t1 = time.time()
        for step in range(a.steps):
            tot = 0.0
            for _ in range(a.accum):
                exs = [next(stream) for _ in range(a.bs)]
                ids, lab, att = make_batch(exs, pad, pad_to=a.max_len if a.dry else None)
                loss = answer_loss(model, ids.to(a.device), att.to(a.device), lab.to(a.device)) / a.accum
                loss.backward()
                tot += loss.item()
                mem.sample()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(round(tot, 5))
            if a.dry or step % 10 == 0 or step == a.steps - 1:
                print(f"step {step} loss {tot:.4f} gnorm {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{time.time()-t1:.0f}s {mem.s()}", flush=True)
            if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
                traj.append({"step": step + 1, **probe(model, tok, a.model, p_items, a.device)})
                print(f"probe step {step+1} {traj[-1]}", flush=True)
        meta["train_s"] = round(time.time() - t1, 1)
        meta["s_per_step"] = round(meta["train_s"] / a.steps, 3)
        del opt
        if a.device == "mps":
            torch.mps.empty_cache()
    meta["train_stats"] = dict(stats, mean_len=round(stats["len_sum"] / stats["n"], 1) if stats["n"] else None)
    meta["losses"] = losses
    meta["loss_finite"] = all(math.isfinite(x) for x in losses)
    meta["probe"] = traj
    meta["lock_in_step"] = MF.lock_in_step(traj) if traj else None
    model.eval()
    t2 = time.time()
    summ = {}
    for set_name, render, its in eval_sets(set_names, a.dry):
        path = f"{stem}__{set_name}__{render}.jsonl"
        with open(path, "w") as f:
            recs, dt = score_set(model, tok, a.model, set_name, render, its, a.device, f, mem)
        print(f"scored {set_name}/{render}: {len(recs)} items in {dt:.0f}s {mem.s()}", flush=True)
        summ[f"{set_name}/{render}"] = len(recs)
        if a.dry:
            for r in recs:
                if not all(isinstance(v, float) and math.isfinite(v) for v in r["scores"].values()):
                    selftest_fail.append(f"non-finite score in {set_name}: {r['scores']}")
    meta["eval_s"] = round(time.time() - t2, 1)
    meta["eval_counts"] = summ
    if a.dry:
        # scorer self-test on this model: an 'empty answer' (gold text == foil text) is a tie and must not count
        it = N.build(n_scen=1, distances=(0,), seed=1)[0]
        prompt, add_sp = lik.render(it, "plain", tok, a.model)
        g = it["cands"]["gold"]
        sc, _ = lik.score_item(model, tok, prompt, {"gold": g, "orig": g}, a.device, add_sp)
        tie = {k: v[0] for k, v in sc.items()}
        if MF.right(tie):
            selftest_fail.append(f"identical candidates scored as right: {tie}")
        sc, _ = lik.score_item(model, tok, prompt, {"gold": g}, a.device, add_sp)
        if MF.right({k: v[0] for k, v in sc.items()}):
            selftest_fail.append("gold-only candidate set scored as right")
        meta["selftest_fail"] = selftest_fail
    meta["peak_mps_driver_gb"] = round(mem.peak_driver / 2**30, 3)
    meta["peak_mps_alloc_gb"] = round(mem.peak_alloc / 2**30, 3)
    meta["total_s"] = round(time.time() - t0, 1)
    if a.save and a.steps > 0 and not a.dry:
        # scratch copy for post-hoc re-scoring; experiments/**/weights/ is gitignored
        try:
            wdir = os.path.join(os.path.dirname(HERE), "weights", f"{slug(a.model)}__{tag}")
            model.save_pretrained(wdir, safe_serialization=True)
            tok.save_pretrained(wdir)
            meta["weights"] = wdir
        except Exception as e:  # never lose the eval over a failed save
            meta["weights_error"] = repr(e)
    json.dump(meta, open(f"{stem}__run.json", "w"), indent=1)
    print(f"DONE {a.model} {tag} total {meta['total_s']:.0f}s peak_driver={meta['peak_mps_driver_gb']}G "
          f"lock_in_step={meta['lock_in_step']} loss_finite={meta['loss_finite']}", flush=True)
    if selftest_fail:
        for m in selftest_fail:
            print("SELFTEST FAIL", m, flush=True)
        sys.exit(5)


if __name__ == "__main__":
    main()
