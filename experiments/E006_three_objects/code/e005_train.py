"""E005 training loop (called by e005_ft_test.py). E004's loop (e004_train.py, not edited) with the E005 stream in
both renders; schedule, optimizer, loss, clip, probe and dry-run padding are E004's line for line:
AdamW, weight decay 0, 20 warmup steps then cosine to 0, grad clip 1.0, fp32, micro-batch x accumulation,
answer-only loss (e004_core.answer_loss), optional gradient checkpointing, probe every --probe-every steps.
Loss self-check (notes (c): "runs on a plain AND a chat batch"): the answer-only loss must equal the HF loss on
  first  the first a.bs kept examples of the training stream (consumed and not trained, as E004's check batch)
  plain  the first a.bs plain-rendered kept examples   } drawn from a SEPARATE iterator of the same seed's stream,
  chat   the first a.bs chat-rendered kept examples    } so the training iterator is not moved by them
  a mismatch exits. --dry: 5 steps padded to --max-len, both wrong-loss mutants (e004_core.loss_mutants) on every
  check batch must differ from the HF loss, and the 5 dry steps must have trained both renders."""
import math, sys, time

import torch

import lik
import e004_core as C
import e005_sets as S


def check_batches(tok, a, stream):
    """-> {"first": ..., "plain": ..., "chat": ...}, each a list of a.bs (ids, labels); stream: the training
    iterator (its first a.bs examples are consumed here)."""
    out = {"first": [(x, y) for x, y, _ in (next(stream) for _ in range(a.bs))], "plain": [], "chat": []}
    side = S.example_stream(tok, a.seed, a.max_len, S.new_stats(), lik.cand_ids)
    while len(out["plain"]) < a.bs or len(out["chat"]) < a.bs:
        x, y, r = next(side)
        if len(out[r]) < a.bs:
            out[r].append((x, y))
    return out


def loss_selfcheck(model, batches, pad, device, dry, meta, selftest_fail):
    """HF loss vs answer-only loss on every check batch; -> list of failed batch names (the caller exits).
    With dry, the two wrong-loss mutants must differ from the HF loss on every batch (else a selftest failure)."""
    bad, meta["loss_selfcheck"], meta["loss_mutants"] = [], {}, {}
    for name, batch in batches.items():
        ids, lab, att = (t.to(device) for t in C.make_batch(batch, pad))
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=att, labels=lab).loss.item()
            mine = C.answer_loss(model, ids, att, lab).item()
        meta["loss_selfcheck"][name] = {"hf": ref, "answer_only": mine, "n_labelled": int((lab != -100).sum())}
        print(f"loss self-check [{name}] hf={ref:.5f} answer_only={mine:.5f}", flush=True)
        if not C.close(ref, mine) or not math.isfinite(mine):
            bad.append(name)
        if dry:
            mut = C.loss_mutants(model, ids, att, lab)
            meta["loss_mutants"][name] = mut
            for k, v in mut.items():
                if C.close(ref, v):
                    selftest_fail.append(f"loss mutant {k} on the {name} batch ({v:.5f}) is not distinguished "
                                         f"from the HF loss ({ref:.5f})")
            print(f"loss mutants [{name}] {mut} (must all differ from {ref:.5f})", flush=True)
    return bad


def train(model, tok, a, meta, mem, pad, p_items, selftest_fail):
    """-> (losses, probe trajectory, stream stats). a: parsed args (steps, lr, bs, accum, max_len, grad_ckpt,
    probe_every, dry, device, seed)."""
    stats = S.new_stats()
    traj, losses, trained = [], [], {r: 0 for r in S.RENDERS}
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    stream = S.example_stream(tok, a.seed, a.max_len, stats, lik.cand_ids)
    bad = loss_selfcheck(model, check_batches(tok, a, stream), pad, a.device, a.dry, meta, selftest_fail)
    if bad:
        sys.exit(f"loss self-check failed on: {', '.join(bad)}")
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
            for _, _, r in exs:
                trained[r] += 1
            ids, lab, att = C.make_batch([(x, y) for x, y, _ in exs], pad, pad_to=a.max_len if a.dry else None)
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
    meta["renders_trained"] = trained
    if a.dry and not all(trained.values()):
        selftest_fail.append(f"the dry run did not train both renders: {trained}")
    del opt
    if a.device == "mps":
        torch.mps.empty_cache()
    model.config.use_cache = True
    return losses, traj, stats


stream_summary = S.stream_summary
