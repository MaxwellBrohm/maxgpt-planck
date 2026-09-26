"""E006 training loops (called by e006_ft_test.py; notes.txt ARMS). No model is loaded here.
  C, P   e005_train.train UNCHANGED (E005's loop line for line: AdamW wd 0, 20 warmup steps then cosine to 0, clip
         1.0, answer-only token-mean loss per micro-batch / accum, probe, loss self-check on the first / plain /
         chat batches, dry-run padding and mutants). patch_stream() routes the e005_sets.example_stream calls it
         makes to the arm's source (C: None, i.e. train_e005.stream itself; P: train_e006p.stream_p) through a
         read-only digest tap: the first stream opened is the training stream, and its sha256 over the
         (ids, labels) it yielded goes into run.json (validate_e006 compares it with the no-model stream).
  G      g_train: the same loop with one addition per optimizer step: after C's accum (4) update micro-batches,
         in C's order and with C's /accum scaling, ONE replay micro-batch of 4 threads, also scaled /accum; then
         clip and step. The update part is C's byte for byte (6,400 trained + the 4-example check batch); the
         replay batch adds 1/5 of the micro-batches (critique MUST 2, option b: additive, not a replacement).
         The loss self-check adds a "replay" batch (the first 4 threads of a separate replay iterator; the
         training iterator does not move). Logged per step: total, update-only and replay-only loss, the update
         gradient's norm before the replay backward, and the clipped total norm."""
import hashlib
import json
import math
import sys
import time

import e005_sets as S5
import e006_sets as S6

_ORIG = S5.example_stream


def patch_stream(arm, registry):
    """-> restore(): route e005_sets.example_stream to the arm's source, with a digest tap per opened stream."""
    def tapped(tok, seed, max_len, stats, cand_ids, source=None):
        src = S6.source(arm, seed) if source is None else source
        tap = {"h": hashlib.sha256(), "n": 0}
        registry.append(tap)
        for ids, labels, r in _ORIG(tok, seed, max_len, stats, cand_ids, source=src):
            tap["h"].update(json.dumps([list(ids), list(labels)]).encode())
            tap["n"] += 1
            yield ids, labels, r
    S5.example_stream = tapped

    def restore():
        S5.example_stream = _ORIG
    return restore


def digest_of(tap):
    return {"sha256": tap["h"].hexdigest(), "n": tap["n"]}


def train_cp(model, tok, a, meta, mem, pad, p_items, fails):
    import e005_train as TR5
    registry = []
    restore = patch_stream(a.arm, registry)
    try:
        out = TR5.train(model, tok, a, meta, mem, pad, p_items, fails)
    finally:
        restore()
    meta["update_digest"] = digest_of(registry[0])
    return out


def g_schedule(update_stream, replay_iter, bs, accum):
    """one optimizer step: accum update micro-batches of bs, then one replay micro-batch of bs."""
    for _ in range(accum):
        yield "u", [next(update_stream) for _ in range(bs)]
    yield "r", [next(replay_iter) for _ in range(bs)]


def _grad_norm(model):
    import torch
    gs = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    return float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs))) if gs else 0.0


def g_train(model, tok, a, meta, mem, pad, p_items, fails, pool):
    import torch
    import lik
    import e004_core as C
    import e005_train as TR5
    registry = []
    restore = patch_stream("G", registry)
    try:
        stats, rstats = S5.new_stats(), S6.new_replay_stats()
        traj, losses, trained = [], [], {r: 0 for r in S5.RENDERS}
        lu, lr_, gnu, gna, rids, rh = [], [], [], [], [], hashlib.sha256()
        if a.grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
        stream = S5.example_stream(tok, a.seed, a.max_len, stats, lik.cand_ids)
        batches = TR5.check_batches(tok, a, stream)
        side = S6.replay_stream(tok, pool, a.seed, a.max_len, S6.new_replay_stats())
        batches["replay"] = [(x, y) for x, y, _ in (next(side) for _ in range(a.bs))]
        bad = TR5.loss_selfcheck(model, batches, pad, a.device, a.dry, meta, fails)
        if bad:
            sys.exit(f"loss self-check failed on: {', '.join(bad)}")
        traj.append({"step": 0, **C.probe(model, tok, a.model, p_items, a.device)})
        print(f"probe step 0 {traj[-1]} {mem.s()}", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        replay = S6.replay_stream(tok, pool, a.seed, a.max_len, rstats)
        t1 = time.time()
        for step in range(a.steps):
            tot = {"u": 0.0, "r": 0.0}
            for kind, exs in g_schedule(stream, replay, a.bs, a.accum):
                if kind == "u":
                    for _, _, r in exs:
                        trained[r] += 1
                else:
                    gnu.append(round(_grad_norm(model), 5))
                    rids += [t for _, _, t in exs]
                    rh.update(json.dumps([[list(x), list(y)] for x, y, _ in exs]).encode())
                ids, lab, att = C.make_batch([(x, y) for x, y, _ in exs], pad, pad_to=a.max_len if a.dry else None)
                loss = C.answer_loss(model, ids.to(a.device), att.to(a.device), lab.to(a.device)) / a.accum
                loss.backward()
                tot[kind] += loss.item()
                mem.sample()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(round(tot["u"] + tot["r"], 5))
            lu.append(round(tot["u"], 5))
            lr_.append(round(tot["r"], 5))
            gna.append(round(gn, 5))
            if a.dry or step % 10 == 0 or step == a.steps - 1:
                print(f"step {step} loss {losses[-1]:.4f} (update {tot['u']:.4f} replay {tot['r']:.4f}) gnorm {gn:.2f} "
                      f"(update part {gnu[-1]:.2f}) lr {sched.get_last_lr()[0]:.2e} {time.time()-t1:.0f}s {mem.s()}",
                      flush=True)
            if (step + 1) % a.probe_every == 0 or step == a.steps - 1:
                traj.append({"step": step + 1, **C.probe(model, tok, a.model, p_items, a.device)})
                print(f"probe step {step+1} {traj[-1]}", flush=True)
        meta.update(train_s=round(time.time() - t1, 1), renders_trained=trained, losses_update=lu, losses_replay=lr_,
                    gnorm_update_part=gnu, gnorm_total=gna, replay_stats=rstats, replay_threads=rids,
                    replay_digest={"sha256": rh.hexdigest(), "n": len(rids)}, update_digest=digest_of(registry[0]))
        meta["s_per_step"] = round(meta["train_s"] / a.steps, 3)
        if a.dry and not (all(trained.values()) and rids):
            fails.append(f"the dry run did not train both renders and replay: {trained}, {len(rids)} threads")
        if len(set(rids)) != len(rids):
            fails.append("a replay thread was trained twice")
        del opt
        model.config.use_cache = True
        return losses, traj, stats
    finally:
        restore()
