"""E004 training loop (called by e004_ft_test.py; same schedule as E002/E003): AdamW, weight decay 0, 20 warmup
steps then cosine to 0, grad clip 1.0, fp32, micro-batch x accumulation, answer-only loss (self-checked against the
HF loss on the first real batch; exit on a mismatch), optional gradient checkpointing, probe every --probe-every
steps on the E004 probe draw. --dry: 5 steps padded to --max-len, plus the two wrong-loss mutants."""
import math, sys, time

import torch

import lik
import e004_core as C
import e004_sets as S


def train(model, tok, a, meta, mem, pad, p_items, selftest_fail):
    """-> (losses, probe trajectory, stream stats). a: parsed args (steps, lr, bs, accum, max_len, grad_ckpt,
    probe_every, dry, device, seed)."""
    stats = S.new_stats()
    traj, losses = [], []
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    stream = S.example_stream(tok, a.seed, a.max_len, stats, lik.cand_ids)
    ids, lab, att = C.make_batch([next(stream) for _ in range(a.bs)], pad)
    ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=att, labels=lab).loss.item()
        mine = C.answer_loss(model, ids, att, lab).item()
    meta["loss_selfcheck"] = {"hf": ref, "answer_only": mine}
    print(f"loss self-check hf={ref:.5f} answer_only={mine:.5f}", flush=True)
    if not C.close(ref, mine) or not math.isfinite(mine):
        sys.exit("loss self-check failed")
    if a.dry:
        mut = C.loss_mutants(model, ids, att, lab)
        meta["loss_mutants"] = mut
        for k, v in mut.items():
            if C.close(ref, v):
                selftest_fail.append(f"loss mutant {k} ({v:.5f}) is not distinguished from the HF loss ({ref:.5f})")
        print(f"loss mutants {mut} (must all differ from {ref:.5f})", flush=True)
    traj.append({"step": 0, **C.probe(model, tok, a.model, p_items, a.device)})
    print(f"probe step 0 {traj[-1]} {mem.s()}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    t1 = time.time()
    for step in range(a.steps):
        tot = 0.0
        for _ in range(a.accum):
            exs = [next(stream) for _ in range(a.bs)]
            ids, lab, att = C.make_batch(exs, pad, pad_to=a.max_len if a.dry else None)
            loss = C.answer_loss(model, ids.to(a.device), att.to(a.device), lab.to(a.device)) / a.accum
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
            traj.append({"step": step + 1, **C.probe(model, tok, a.model, p_items, a.device)})
            print(f"probe step {step+1} {traj[-1]}", flush=True)
    meta["train_s"] = round(time.time() - t1, 1)
    meta["s_per_step"] = round(meta["train_s"] / a.steps, 3)
    del opt
    if a.device == "mps":
        torch.mps.empty_cache()
    model.config.use_cache = True
    return losses, traj, stats


def stream_summary(stats):
    """kind shares drawn vs kept after the length rejection (the notes' <= 2 point check reads this)."""
    nd, nk = sum(stats["drawn"].values()), sum(stats["kept"].values())
    shares = {k: {"drawn": round(stats["drawn"][k] / nd, 4) if nd else None,
                  "kept": round(stats["kept"].get(k, 0) / nk, 4) if nk else None} for k in sorted(stats["drawn"])}
    worst = max((abs(v["drawn"] - v["kept"]) for v in shares.values() if v["kept"] is not None), default=None)
    return dict(stats, mean_len=round(stats["len_sum"] / stats["n"], 1) if stats["n"] else None, shares=shares,
                max_share_shift=round(worst, 4) if worst is not None else None)
