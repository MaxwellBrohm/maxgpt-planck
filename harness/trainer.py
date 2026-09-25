"""The optimizer-step loop: gradient accumulation, token-normalized loss, clipping,
schedule, logging, checkpoints. Setup (config, prereg, model, data, schedule) is train.py.

Loss per step = sum of token losses over all micro-batches / supervised tokens in the step,
so packed rows with different fill and bucket batches of different shape weigh each
supervised token equally. Adapted in spirit from Max's MaxGPT-Ultra train/trainer.py
(train_step, clip, save/resume); written fresh for Planck's smaller scope.
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext

import torch

import runio
from device import sync
from optim import set_lr


class Trainer:
    def __init__(self, *, model, optimizer, loader, sched, device, amp, cfg: dict, out_dir: str,
                 grad_accum: int, grad_clip: float, log_every: int, ckpt_every: int,
                 keep_last: int, stable_points: set[int], meta: dict, hooks=None):
        self.model, self.opt, self.loader, self.sched = model, optimizer, loader, sched
        self.device, self.amp, self.cfg, self.out_dir = device, amp, cfg, out_dir
        self.grad_accum, self.grad_clip = grad_accum, grad_clip
        self.log_every, self.ckpt_every, self.keep_last = log_every, ckpt_every, keep_last
        self.stable_points = set(stable_points)
        self.meta = meta                     # n_params, prereg, config sha: stored in ckpts
        self.step = 0
        self.tokens = 0                      # real (non-pad) tokens fed
        self.sup_tokens = 0                  # supervised target tokens
        self.stop_requested = False
        self.log_path = os.path.join(out_dir, "log.jsonl")
        self.hooks = list(hooks or [])      # called as hook(trainer) after every step (rc12_eval.RC12Eval)

    # ------------------------------------------------------------------ #
    def _to(self, t):
        return None if t is None else t.to(self.device, non_blocking=True)

    def train_step(self) -> dict:
        set_lr(self.opt, self.sched.factor(self.step))
        g0 = self.opt.param_groups[0]
        applied = g0["lr"] / g0["base_lr"] if g0["base_lr"] else 0.0   # logged: what the update used
        batches = [self.loader.next_batch() for _ in range(self.grad_accum)]
        n_sup = sum(int((b["tgt"] != -100).sum()) for b in batches)
        n_real = 0
        loss_sum = 0.0
        for b in batches:
            n_real += int((b["idx"] != self.loader.pad).sum())   # pad_id is reserved
            ctx = self.amp() if self.amp else nullcontext()
            with ctx:
                _, ls = self.model(self._to(b["idx"]), self._to(b["tgt"]), self._to(b["doc"]),
                                   self._to(b["pos"]), reduction="sum")
            (ls / max(1, n_sup)).backward()
            loss_sum += float(ls.detach())
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens += n_real
        self.sup_tokens += n_sup
        return {"loss": loss_sum / max(1, n_sup), "gnorm": gnorm, "n_sup": n_sup, "lr_factor": applied}

    # ------------------------------------------------------------------ #
    def payload(self) -> dict:
        p = {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
             "step": self.step, "tokens": self.tokens, "sup_tokens": self.sup_tokens,
             "data_state": self.loader.state_dict(), "schedule": self.sched.to_dict(),
             "phase": self.sched.phase(self.step), "config": self.cfg,
             "rng_torch": torch.get_rng_state(), **self.meta}
        if self.device == "cuda":
            p["rng_cuda"] = torch.cuda.get_rng_state_all()
        return p

    def save(self, tag: str | None = None) -> str:
        return runio.save_checkpoint(self.out_dir, self.payload(), self.step, self.keep_last, tag)

    def load_state(self, ck: dict, with_schedule_step: bool = True) -> None:
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.loader.load_state_dict(ck["data_state"])
        torch.set_rng_state(ck["rng_torch"])
        if with_schedule_step:
            self.step = int(ck["step"])
        self.tokens, self.sup_tokens = int(ck.get("tokens", 0)), int(ck.get("sup_tokens", 0))

    # ------------------------------------------------------------------ #
    def run(self, max_steps: int | None = None) -> dict:
        """Train to the schedule's end (or max_steps more steps). Returns the last record."""
        stop_at = self.sched.total_steps if max_steps is None else min(
            self.sched.total_steps, self.step + max_steps)
        stop_file = os.path.join(self.out_dir, "STOP")
        rec: dict = {"step": self.step}
        t0, tok0, last_loss = time.time(), self.tokens, float("nan")
        while self.step < stop_at:
            out = self.train_step()
            last_loss = out["loss"]
            if not math.isfinite(last_loss):
                runio.append_jsonl(self.log_path, {"step": self.step, "error": "non-finite loss"})
                raise FloatingPointError(f"non-finite loss at step {self.step}")
            if self.step in self.stable_points:
                assert self.sched.phase(self.step) == "stable", "branch point outside stable phase"
                self.save(tag="stable")
            if self.step % self.log_every == 0 or self.step == stop_at:
                sync(self.device)
                dt = max(1e-9, time.time() - t0)
                rec = {"step": self.step, "loss": round(last_loss, 5),
                       "lr_factor": round(out["lr_factor"], 6),
                       "gnorm": round(out["gnorm"], 4), "tokens": self.tokens,
                       "sup_tokens": self.sup_tokens,
                       "tok_per_s": round((self.tokens - tok0) / dt, 1),
                       "phase": self.sched.phase(self.step - 1), **self.loader.stats()}
                runio.append_jsonl(self.log_path, rec)
                t0, tok0 = time.time(), self.tokens
            if self.ckpt_every and self.step % self.ckpt_every == 0 and self.step < self.sched.total_steps:
                self.save()
            for hook in self.hooks:
                hook(self)
            if os.path.exists(stop_file) or self.stop_requested:
                self.save()
                return {**rec, "stopped": True, "loss": last_loss}
        if self.step >= self.sched.total_steps:
            self.save(tag="final")
        elif self.step == stop_at:
            self.save()
        return {**rec, "loss": last_loss, "step": self.step}
