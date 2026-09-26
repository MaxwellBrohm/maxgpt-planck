"""The optimizer-step loop: gradient accumulation, token-normalized loss, clipping,
schedule, logging, checkpoints. Setup (config, prereg, model, data, schedule) is train.py.

Loss per step = sum of token losses over all micro-batches / supervised tokens in the step,
so packed rows with different fill and bucket batches of different shape weigh each
supervised token equally. Adapted in spirit from Max's MaxGPT-Ultra train/trainer.py
(train_step, clip, save/resume); written fresh for Planck's smaller scope.

lazy_metrics (train key lazy_metrics, default off): the per-micro-batch loss sums and the
grad norm stay on the device and are read only on log steps, so no step blocks the host on
the GPU (float(loss) after backward stopped the optimizer's launches from overlapping it).
The logged numbers are the same floats. Non-finite losses are caught by a device-side flag
that is read on log steps and before every checkpoint save (so no NaN state is saved), not
after every step.

mtp (S006, train.mtp; mtp.py): a training-only t+2 aux head. Loss per step = (sum of next-token
losses + w_t * sum of aux losses) / n_sup, n_sup the step's NEXT-TOKEN supervised count, w_t =
mtp_weight * the schedule factor of the step. "loss" stays the next-token loss alone; log records
add mtp_loss (aux CE per aux target), mtp_w (w_t) and mtp_n (aux targets). The grad norm and its clip
cover the head too; checkpoints hold its state under "mtp". Off (mtp=None): the pre-flag step.
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext

import torch

import runio
from device import sync
from mtp import mtp_targets
from optim import set_lr


class Trainer:
    def __init__(self, *, model, optimizer, loader, sched, device, amp, cfg: dict, out_dir: str,
                 grad_accum: int, grad_clip: float, log_every: int, ckpt_every: int,
                 keep_last: int, stable_points: set[int], meta: dict, hooks=None,
                 lazy_metrics: bool = False, mtp=None, mtp_weight: float = 1.0):
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
        self.lazy_metrics = bool(lazy_metrics)
        self._finite = None                  # lazy_metrics: device bool, all losses finite since the last check
        self._checked_at = 0
        self.mtp, self.mtp_weight = mtp, float(mtp_weight)   # S006: the aux head (mtp.MTPHead) or None
        self._hid = {} if mtp is None else {"return_hidden": True}

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
        loss_sum, parts = 0.0, []
        aux = None if self.mtp is None else {"w": self.mtp_weight * self.sched.factor(self.step),
                                             "n": 0, "parts": []}
        for b in batches:
            n_real += int((b["idx"] != self.loader.pad).sum())   # pad_id is reserved
            ctx = self.amp() if self.amp else nullcontext()
            with ctx:
                out = self.model(self._to(b["idx"]), self._to(b["tgt"]), self._to(b["doc"]),
                                 self._to(b["pos"]), reduction="sum", **self._hid)
                ls = obj = out[1]
                if aux is not None:
                    obj = ls + aux["w"] * self._mtp_sum(out[2], b, aux)
            (obj / max(1, n_sup)).backward()
            if self.lazy_metrics:
                parts.append(ls.detach())
            else:
                loss_sum += float(ls.detach())
        params = self.model.parameters() if self.mtp is None else [*self.model.parameters(),
                                                                   *self.mtp.parameters()]
        gnorm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        if not self.lazy_metrics:
            gnorm = float(gnorm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens += n_real
        self.sup_tokens += n_sup
        if self.lazy_metrics:
            parts = torch.stack(parts)
            ok = torch.isfinite(parts).all()
            if aux is not None:
                aux["parts"] = torch.stack(aux["parts"])
                ok = ok & torch.isfinite(aux["parts"]).all()
            self._finite = ok if self._finite is None else self._finite.logical_and_(ok)
            return {"loss_parts": parts, "gnorm": gnorm, "n_sup": n_sup, "lr_factor": applied,
                    **self._mtp_out(aux)}
        return {"loss": loss_sum / max(1, n_sup), "gnorm": gnorm, "n_sup": n_sup, "lr_factor": applied,
                **self._mtp_out(aux)}

    def _mtp_sum(self, z, b: dict, aux: dict):
        """S006: one micro-batch's summed aux loss (t+2 targets from the loader's CPU tensors)."""
        t2 = mtp_targets(b["idx"], b["tgt"], b["doc"])
        n = int((t2 != -100).sum())
        aux["n"] += n
        s = self.mtp.loss_sum(z, self.model.lm_head, self._to(t2), n)
        aux["parts"].append(s.detach())
        return s

    def _mtp_out(self, aux) -> dict:
        if aux is None:
            return {}
        if self.lazy_metrics:
            return {"mtp_parts": aux["parts"], "mtp_w": aux["w"], "mtp_n": aux["n"]}
        s = sum((float(p) for p in aux["parts"]), 0.0)
        return {"mtp_loss": s / max(1, aux["n"]), "mtp_w": aux["w"], "mtp_n": aux["n"]}

    def read_metrics(self, out: dict) -> dict:
        """lazy_metrics: a train_step result's device values -> the floats the eager path
        returns (the same sum in the same order: one sync)."""
        if "loss_parts" not in out:
            return out
        loss_sum = sum(out["loss_parts"].tolist(), 0.0)
        rest = {k: v for k, v in out.items() if k not in ("loss_parts", "mtp_parts")}
        if "mtp_parts" in out:
            rest["mtp_loss"] = sum(out["mtp_parts"].tolist(), 0.0) / max(1, out["mtp_n"])
        return {**rest, "loss": loss_sum / max(1, out["n_sup"]), "gnorm": float(out["gnorm"])}

    def check_finite(self) -> None:
        """lazy_metrics: raise if any loss since the last check was non-finite (one sync)."""
        if self._finite is not None and not bool(self._finite):
            runio.append_jsonl(self.log_path, {"step": self.step, "error": "non-finite loss",
                                               "after_step": self._checked_at})
            raise FloatingPointError(f"non-finite loss in steps {self._checked_at + 1}..{self.step}")
        self._finite, self._checked_at = None, self.step

    # ------------------------------------------------------------------ #
    def payload(self) -> dict:
        p = {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
             "step": self.step, "tokens": self.tokens, "sup_tokens": self.sup_tokens,
             "data_state": self.loader.state_dict(), "schedule": self.sched.to_dict(),
             "phase": self.sched.phase(self.step), "config": self.cfg,
             "rng_torch": torch.get_rng_state(), **self.meta}
        if self.device == "cuda":
            p["rng_cuda"] = torch.cuda.get_rng_state_all()
        if self.mtp is not None:
            p["mtp"] = self.mtp.state_dict()
        return p

    def save(self, tag: str | None = None) -> str:
        if self.lazy_metrics:
            self.check_finite()
        return runio.save_checkpoint(self.out_dir, self.payload(), self.step, self.keep_last, tag)

    def load_state(self, ck: dict, with_schedule_step: bool = True) -> None:
        if ("mtp" in ck) != (self.mtp is not None):
            raise ValueError("checkpoint and run disagree on train.mtp (the S006 aux head)")
        if self.mtp is not None:
            self.mtp.load_state_dict(ck["mtp"])
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
            if self.lazy_metrics and (self.step % self.log_every == 0 or self.step == stop_at):
                out = self.read_metrics(out)
                self.check_finite()
            if "loss" in out:                # every step; only log steps under lazy_metrics
                last_loss = out["loss"]
                if not (math.isfinite(last_loss) and math.isfinite(out.get("mtp_loss", 0.0))):
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
                if "mtp_w" in out:            # S006
                    rec.update(mtp_loss=round(out["mtp_loss"], 5), mtp_w=round(out["mtp_w"], 6),
                               mtp_n=out["mtp_n"])
                runio.append_jsonl(self.log_path, rec)
                t0, tok0 = time.time(), self.tokens
            if self.ckpt_every and self.step % self.ckpt_every == 0 and self.step < self.sched.total_steps:
                self.save()
            for hook in self.hooks:
                hook(self)
            if os.path.exists(stop_file) or self.stop_requested:
                self.save()
                if "loss_parts" in out:
                    last_loss = self.read_metrics(out)["loss"]
                return {**rec, "stopped": True, "loss": last_loss}
        if self.step >= self.sched.total_steps:
            self.save(tag="final")
        elif self.step == stop_at:
            self.save()
        return {**rec, "loss": last_loss, "step": self.step}
