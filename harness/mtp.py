"""S006: a t+2 auxiliary prediction head, training only (train.mtp; experiments/S006_mtp_aux/notes.txt).

  z_t        the model's final hidden state after its final norm (PlanckLM.forward(..., return_hidden=True))
  aux logits lm_head(RMSNorm_mtp(W_mtp z_t)), lm_head the model's own (tied) output head
  init       W_mtp (d, d) = identity, the norm gain = ones: nothing drawn from the torch RNG, so every BASE
             parameter keeps its draw and the aux head starts as a copy of the next-token head
  target     of position t: the token at t + 2, only where t, t + 1 and t + 2 are in one document (runs of
             equal doc ids, the mask's rule) and the t + 1 target is supervised (not -100); else -100
  loss       per step (trainer.py) = (sum of next-token losses + w_t * sum of aux losses) / n_sup, n_sup the
             step's NEXT-TOKEN supervised count; w_t = mtp_weight * the schedule factor at step t
The head is not a module of the model: the model's state_dict, model_cfg and n_params stay BASE's, so bpb.py
(which builds PlanckLM from model_cfg) and every eval read the next-token logits. Checkpoints keep the head's
state under "mtp" (resume); its parameters are reported as training-only (budget.count_mtp). W_mtp goes to
the NorMuon matrix group, the gain to 'scalar' (optim.split_params, extra=). The head reads z from whichever
attention engine ran (mask or varlen); the targets use doc ids only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import RMSNorm


class MTPHead(nn.Module):
    """One extra head (t + 2). W_mtp is a bare Parameter, never an nn.Linear (nn.Linear draws at build)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d_model))      # W_mtp: identity, no RNG draw
        self.norm = RMSNorm(d_model, eps)                   # gain ones

    def logits(self, z: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        return lm_head(self.norm(F.linear(z, self.weight)))

    def loss_sum(self, z: torch.Tensor, lm_head: nn.Module, tgt2: torch.Tensor, n: int) -> torch.Tensor:
        """Summed CE of the aux logits against tgt2 (-100 = no target); n = tgt2's target count."""
        logits = self.logits(z, lm_head)
        flat = logits.view(-1, logits.size(-1)).float()
        if n:
            return F.cross_entropy(flat, tgt2.reshape(-1), ignore_index=-100, reduction="sum")
        return flat.sum() * 0.0


def mtp_targets(idx: torch.Tensor, tgt: torch.Tensor, doc: torch.Tensor | None) -> torch.Tensor:
    """(B, T) tokens, next-token targets and doc ids (None = one document per row) -> (B, T) t+2 targets:
    out[t] = idx[t + 2] where tgt[t + 1] != -100 and doc[t] == doc[t + 1] == doc[t + 2]; else -100.
    The last two positions have no t + 2 in the row (the loader's rows end at an item end or padding)."""
    out = torch.full_like(tgt, -100)
    ok = tgt[:, 1:-1] != -100                               # the t + 1 target is supervised
    if doc is not None:
        ok = ok & (doc[:, :-2] == doc[:, 1:-1]) & (doc[:, 1:-1] == doc[:, 2:])   # one document
    out[:, :-2] = torch.where(ok, idx[:, 2:], out[:, :-2])
    return out


def mtp_settings(tc: dict) -> tuple[int, float]:
    """train keys -> (mtp, mtp_weight). mtp: 0 (off, the default) or 1 (one t+2 head; more heads are not
    built); mtp_weight: finite number >= 0, default 1.0, checked always, used only when mtp is 1."""
    mtp = tc.get("mtp", 0)
    if isinstance(mtp, bool) or not isinstance(mtp, int) or mtp not in (0, 1):
        raise ValueError(f"train.mtp must be 0 or 1, got {mtp!r}")
    w = tc.get("mtp_weight", 1.0)
    if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
        raise ValueError(f"train.mtp_weight must be a finite number >= 0, got {w!r}")
    return int(mtp), float(w)
