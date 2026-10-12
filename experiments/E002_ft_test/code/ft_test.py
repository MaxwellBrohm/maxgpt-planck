"""Repaired ft_test (E002; REPORT 7.3 step 2c, 7.4; LEDGER P-001). ONE model, ONE seed, ONE process.
Run only through guard.py (one model process at a time, memory and wall-clock kills).

Question: can a few hundred fine-tuning steps install "latest statement wins" updating (and
binding) in a small chat model, WITHOUT the "most recently mentioned weekday" shortcut?

Fixes relative to research/capacity_probe/ft_test.py:
  memory   micro-batch 4 (135M) / 2 (360M) with gradient accumulation to an effective 16; gradient
           checkpointing with --max-len 768 (REPORT 7.4 allows "512 or gradient checkpointing"; at 512 the
           length rejection removed most long two-slot items, see notes.txt); longer examples are redrawn,
           never truncated; loss computed only at answer
           positions (no full-vocab logits for the whole sequence); optional gradient checkpointing;
           refuses to run on MPS unless PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO are 0.7/0.6;
           MPS driver memory logged every micro-step; --dry = 5 steps at worst-case length.
  design   training mixes updates with two-slot, no-update, incidental-mention, middle and revert
           items (train_data.py), in two value types, with wording held out from every eval set;
           eval = E001 control items (plain and own chat template) + an incidental no-update variant
           + the old battery + uprobe + khard + an enlarged closed-book set (kbig) for the paired
           knowledge CI; a small in-training probe every --probe-every steps for the lock-in step.
  seeds    --seed sets both the torch seed and the training-data stream (the old script fixed 0).
  --steps 0 scores the untouched model (the paired baseline).

usage: ft_test.py <hf_model_id> --seed S [--steps 400] [--lr 5e-5] [--bs 4] [--accum 4]
                  [--max-len 768] [--grad-ckpt] [--probe-every 50] [--dry] [--device mps]
Writes ../out/<slug>__<tag>__<set>__<render>.jsonl and ../out/<slug>__<tag>__run.json.
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


def slug(m):
    return m.replace("/", "__")


def phash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


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


# ---------------- training data ----------------
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
    """Mean next-token cross-entropy over labelled (answer) positions only. Same quantity as the HF
    loss, but the vocabulary projection runs only where a label exists."""
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    m = tgt != -100
    logits = model.lm_head(h[:, :-1][m])
    return Fnn.cross_entropy(logits.float(), tgt[m])


# ---------------- eval ----------------
def eval_sets(dry):
    """(set name, render, items) in a fixed order; items carry prompt-free structure or 'prompt'."""
    sets = [("new", "plain", N.build()), ("new", "chat", N.build()),
            ("extra", "plain", X.build()), ("extra", "chat", X.build()),
            ("old", "plain", I.build()), ("uprobe", "plain", UP.build()),
            ("khard", "plain", KH.build()), ("kbig", "plain", KB.build()),
            ("cross", "plain", X.build_crossed()), ("cross", "chat", X.build_crossed())]  # cross: post hoc, from seed 1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4, help="micro-batch")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = bs * accum)")
    ap.add_argument("--max-len", type=int, default=768,
                    help="768 with --grad-ckpt keeps every training example (longest ~733); 512 would reject 55%% of d=10 "
                         "examples and bias the long-distance mix toward no-update items")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--dry", action="store_true", help="5 steps at worst-case length, 2 eval items per set")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", type=int, default=1, help="save fine-tuned weights (scratch, for post-hoc re-scoring)")
    a = ap.parse_args()
    if a.dry:
        a.steps = 5
    if a.device == "mps":
        hw, lw = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")
        if hw != "0.7" or lw != "0.6":
            sys.exit(f"refusing: MPS watermarks must be 0.7/0.6 (got {hw}/{lw}); run through guard.py")
    tag = a.tag or (("dry_" if a.dry else "") + ("base" if a.steps == 0 else f"s{a.seed}"))
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"{slug(a.model)}__{tag}")
    t0 = time.time()
    torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, low_cpu_mem_usage=True).to(a.device)
    mem = Mem(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {a.model} params={n_params/1e6:.1f}M in {time.time()-t0:.1f}s {mem.s()}", flush=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    meta = dict(model=a.model, seed=a.seed, steps=a.steps, lr=a.lr, bs=a.bs, accum=a.accum, eff_batch=a.bs * a.accum,
                max_len=a.max_len, grad_ckpt=a.grad_ckpt, dtype="fp32", device=a.device, dry=a.dry, tag=tag,
                params=n_params, torch=torch.__version__,
                watermarks=[os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"), os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")])
    traj, losses = [], []
    stats = {"n": 0, "rejected_long": 0, "kinds": {}, "len_sum": 0, "len_max": 0, "split": 0}
    p_items = probe_items(a.dry)
    if a.steps > 0:
        if a.grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
        rng = random.Random(1000 * a.seed + 17)
        stream = example_stream(tok, rng, a.max_len, stats)
        # self-check: the answer-only loss equals the HF loss on one real batch
        ids, lab, att = make_batch([next(stream) for _ in range(a.bs)], pad)
        with torch.no_grad():
            ref = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), labels=lab.to(a.device)).loss.item()
            mine = answer_loss(model, ids.to(a.device), att.to(a.device), lab.to(a.device)).item()
        meta["loss_selfcheck"] = {"hf": ref, "answer_only": mine}
        print(f"loss self-check hf={ref:.5f} answer_only={mine:.5f}", flush=True)
        if abs(ref - mine) > 1e-3 * max(1.0, abs(ref)):
            sys.exit("loss self-check failed")
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
        del opt
        if a.device == "mps":
            torch.mps.empty_cache()
    meta["train_stats"] = dict(stats, mean_len=round(stats["len_sum"] / stats["n"], 1) if stats["n"] else None)
    meta["losses"] = losses
    meta["probe"] = traj
    meta["lock_in_step"] = MF.lock_in_step(traj) if traj else None
    model.eval()
    t2 = time.time()
    summ = {}
    for set_name, render, its in eval_sets(a.dry):
        path = f"{stem}__{set_name}__{render}.jsonl"
        with open(path, "w") as f:
            recs, dt = score_set(model, tok, a.model, set_name, render, its, a.device, f, mem)
        print(f"scored {set_name}/{render}: {len(recs)} items in {dt:.0f}s {mem.s()}", flush=True)
        summ[f"{set_name}/{render}"] = len(recs)
    meta["eval_s"] = round(time.time() - t2, 1)
    meta["eval_counts"] = summ
    meta["peak_mps_driver_gb"] = round(mem.peak_driver / 2**30, 3)
    meta["peak_mps_alloc_gb"] = round(mem.peak_alloc / 2**30, 3)
    meta["total_s"] = round(time.time() - t0, 1)
    if a.save and a.steps > 0 and not a.dry:
        # scratch copy for post-hoc re-scoring; experiments/**/weights/ is gitignored; deleted when E002 is done
        try:
            wdir = os.path.join(os.path.dirname(HERE), "weights", f"{slug(a.model)}__{tag}")
            model.save_pretrained(wdir, safe_serialization=True)
            tok.save_pretrained(wdir)
            meta["weights"] = wdir
        except Exception as e:  # never lose the eval over a failed save
            meta["weights_error"] = repr(e)
    json.dump(meta, open(f"{stem}__run.json", "w"), indent=1)
    print(f"DONE {a.model} {tag} total {meta['total_s']:.0f}s peak_driver={meta['peak_mps_driver_gb']}G "
          f"lock_in_step={meta['lock_in_step']}", flush=True)


if __name__ == "__main__":
    main()
