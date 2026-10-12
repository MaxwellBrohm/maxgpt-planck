"""Startup self-tests: does any information leak where attention must not reach?

PLAN.md (Environment, review 23) requires a causal-leak self-test at startup, run twice:
with gradients enabled and under torch.no_grad, because on MPS the leaking fused kernel
was taken only in no-grad passes (so evaluation is the exposed path). Planck adds the
document test for packed rows: a packed document must not see any other document, and
its outputs must equal the same document run alone (positions restart at 0).

Each check returns a leak ratio: the largest logit change at positions that must NOT
move, divided by the largest change at positions that should move. Zero means no leak.

Step 3 additions: the causal check also runs through the document-mask path (one
document covering the row, which is what packed training uses), and every check also
runs on a SCRAMBLED copy of the model (every parameter random). A fresh model hides two
paths: the attention gate is exactly 1 (zero-init projection) and the value residual's
second weight is 0, so a leak routed through either cannot show at init.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import torch


def _logits(model, idx, doc=None, amp=None, grad=True):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, (amp() if amp else nullcontext()):
        logits, _ = model(idx, doc=doc)
    return logits.detach().float()


@torch.no_grad()
def scrambled_copy(model, seed: int = 0):
    """A copy with every parameter random: matrices N(0, 1/fan_in), vectors and scalars
    1 + N(0, 0.25). Same device and dtype as the model."""
    m2 = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    for p in m2.parameters():
        r = torch.randn(p.shape, generator=g)
        r = r / math.sqrt(p.shape[1]) if p.dim() >= 2 else 1.0 + 0.5 * r
        p.copy_(r.to(p.device, p.dtype))
    return m2


def causal_leak(model, T: int, device, amp=None, grad=True, seed: int = 0,
                masked: bool = False) -> float:
    """Change every token from position j on; logits before j must not move. masked=True
    passes one document id for the whole row, so the document-mask attention path runs."""
    g = torch.Generator().manual_seed(seed)
    V = model.cfg.vocab_size
    idx = torch.randint(0, V, (2, T), generator=g)
    j = T // 2
    idx2 = idx.clone()
    idx2[:, j:] = torch.randint(0, V, (2, T - j), generator=g)
    doc = torch.zeros(2, T, dtype=torch.long).to(device) if masked else None
    a = _logits(model, idx.to(device), doc, amp=amp, grad=grad)
    b = _logits(model, idx2.to(device), doc, amp=amp, grad=grad)
    moved = (a[:, j:] - b[:, j:]).abs().max().item()
    leak = (a[:, :j] - b[:, :j]).abs().max().item()
    return leak / max(moved, 1e-12)


def document_leak(model, T: int, device, amp=None, grad=True, seed: int = 1) -> float:
    """Pack doc A (first half) and doc B (second half) in one row. Changing A must not move
    B's logits, and B's logits must equal B run alone from position 0."""
    g = torch.Generator().manual_seed(seed)
    V = model.cfg.vocab_size
    a_len = T // 2
    idx = torch.randint(0, V, (1, T), generator=g)
    doc = torch.zeros(1, T, dtype=torch.long)
    doc[:, a_len:] = 1
    idx2 = idx.clone()
    idx2[:, :a_len] = torch.randint(0, V, (1, a_len), generator=g)
    p1 = _logits(model, idx.to(device), doc.to(device), amp, grad)
    p2 = _logits(model, idx2.to(device), doc.to(device), amp, grad)
    alone = _logits(model, idx[:, a_len:].to(device), amp=amp, grad=grad)
    moved = (p1[:, :a_len] - p2[:, :a_len]).abs().max().item()
    leak_cross = (p1[:, a_len:] - p2[:, a_len:]).abs().max().item()
    leak_alone = (p1[:, a_len:] - alone).abs().max().item()
    return max(leak_cross, leak_alone) / max(moved, 1e-12)


def run_selftests(model, device, amp=None, T: int | None = None, packed: bool = True,
                  tol: float | None = None, scramble: bool = True) -> dict:
    """Run every check with and without gradients, on the model and (scramble=True) on a
    scrambled copy of it. Raises AssertionError on any leak. Keys: [scr_]check_grad|nograd.
    tol defaults to 1e-3 in fp32 and 3e-2 under autocast (bf16 rounding differs between a
    packed row and the same document alone, but a real leak moves logits by O(1) ratio)."""
    T = T or min(64, model.cfg.seq_len)
    tol = tol if tol is not None else (3e-2 if amp else 1e-3)
    was_training = model.training
    model.eval()
    out = {}
    try:
        variants = [("", model)] + ([("scr_", scrambled_copy(model))] if scramble else [])
        for prefix, m in variants:
            for grad in (True, False):
                tag = "grad" if grad else "nograd"
                out[f"{prefix}causal_{tag}"] = causal_leak(m, T, device, amp, grad)
                if packed:
                    out[f"{prefix}causal_masked_{tag}"] = causal_leak(m, T, device, amp, grad,
                                                                     masked=True)
                    out[f"{prefix}document_{tag}"] = document_leak(m, T, device, amp, grad)
        del variants
    finally:
        model.train(was_training)
    bad = {k: v for k, v in out.items() if not v < tol}
    assert not bad, f"attention leak self-test FAILED (ratio >= {tol}): {bad}"
    return out
