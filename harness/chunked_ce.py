"""Chunked lm_head + cross-entropy: the loss path without the full logits (train.ce_chunk_rows).

Off by default. The reference loss path (model.py) holds the whole (B*T, V) logits in the
autocast dtype, an fp32 copy, and the fp32 log-probs that autograd keeps for backward; the
backward then makes two more (B*T, V) fp32 tensors. At T=2048, V=8192 that is about 0.5 GiB
of bf16 logits plus 1 GiB per fp32 tensor for every 8 rows, alive at the start of the
backward, where the step's memory peaks. Here the B*T rows go through lm_head and the loss
`chunk` rows at a time, and when a gradient is wanted each chunk's gradient w.r.t. the
hidden states and the weight is formed right there (softmax - onehot, two GEMMs), so no
(rows, V) tensor outlives its chunk and nothing is recomputed. The backward only multiplies
the stored gradients by the incoming one (the loss is the graph's end: trainer and bench
call backward on loss / n or loss directly).

The math is the reference's: logits from the lm_head GEMM in the autocast dtype (bf16 on
CUDA/MPS, whatever autocast says on CPU, plain fp32 without autocast), upcast to fp32, then
F.cross_entropy's own two steps (fp32 log-softmax, minus the target's log-prob); the logits
gradient (softmax - onehot, over the supervised count for "mean") is rounded to the GEMM
dtype before the two gradient GEMMs, and the weight gradient, accumulated over the chunks in
fp32, is rounded through the GEMM dtype once at the end, as the reference's single GEMM
output is. One rounding point moves: the incoming gradient (the trainer's 1 / n_sup) is
applied after that rounding instead of before it. Targets equal to -100 contribute nothing, exactly
like F.cross_entropy(ignore_index=-100); a batch with no supervised token gives a zero loss
and zero gradients (the reference's `flat.sum() * 0.0` branch) without the host sync that
the reference's `.any()` test costs.
"""
from __future__ import annotations

import torch

IGNORE = -100


def _acc_mm(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """acc + a @ b in fp32. On CUDA a low-precision a @ b accumulates in fp32 inside the GEMM
    (out_dtype), as the reference's single weight-gradient GEMM does; elsewhere the product
    is formed in the input dtype and upcast."""
    if a.dtype != torch.float32 and a.is_cuda:
        if torch.compiler.is_compiling():   # inductor (2.13) lowers mm.dtype but not addmm.dtype
            return acc + torch.mm(a, b, out_dtype=torch.float32)
        return torch.addmm(acc, a, b, out_dtype=torch.float32)
    return acc + (a @ b).float()


def _chunk_rows(lg: torch.Tensor, col: torch.Tensor, row_w: torch.Tensor, cdt: torch.dtype):
    """One chunk of logits lg (c, V) in cdt: the targets' fp32 log-probs (c,) and the logits
    gradient (softmax - onehot) * row_w (c, V) rounded to cdt. Written without in-place ops
    so that torch.compile fuses it into two kernels (row reductions, then one pass)."""
    lp = torch.log_softmax(lg.float(), dim=-1)                  # the reference's fp32 log-probs
    tlp = lp.gather(1, col).squeeze(1)
    onehot = torch.arange(lp.shape[1], device=lp.device) == col
    g = ((lp.exp() - onehot.to(lp.dtype)) * row_w[:, None]).to(cdt)
    return tlp, g


FUSE_CUDA = True        # eager CUDA runs _chunk_rows compiled (tests switch it off to compare)
_fused = None


def _chunk_core(lg, col, row_w, cdt):
    global _fused
    if FUSE_CUDA and lg.is_cuda and not torch.compiler.is_compiling():
        if _fused is None:
            _fused = torch.compile(_chunk_rows)
        return _fused(lg, col, row_w, cdt)
    return _chunk_rows(lg, col, row_w, cdt)   # CPU/MPS, or traced inside a compiled model


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, tgt, chunk: int, mean: bool, cdt: torch.dtype, need_h: bool,
                need_w: bool):
        # h (N, d) any float dtype, w (V, d) the master weight, tgt (N,) int64.
        f32 = torch.float32
        with torch.autocast(device_type=h.device.type, enabled=False):
            hc, wc = h.to(cdt), w.to(cdt)
            valid = tgt != IGNORE
            safe = torch.where(valid, tgt, torch.zeros_like(tgt))
            denom = valid.sum().clamp(min=1).to(f32)
            row_w = valid.to(f32) / denom if mean else valid.to(f32)   # d loss / d row loss
            N = hc.shape[0]
            tlp = torch.empty(N, dtype=f32, device=h.device)       # log-prob of each target
            gh = torch.empty(h.shape, dtype=h.dtype, device=h.device) if need_h else None
            gw = torch.zeros(w.shape, dtype=f32, device=w.device) if need_w else None
            for s in range(0, N, chunk):
                e = min(N, s + chunk)
                col = safe[s:e, None]
                lg = hc[s:e] @ wc.t()                               # (c, V) logits in cdt
                if need_h or need_w:
                    tlp[s:e], g = _chunk_core(lg, col, row_w[s:e], cdt)
                    if need_h:
                        gh[s:e] = g @ wc
                    if need_w:
                        gw = _acc_mm(gw, g.t(), hc[s:e])
                else:
                    tlp[s:e] = torch.log_softmax(lg.float(), dim=-1).gather(1, col).squeeze(1)
            # F.cross_entropy's sum: ignored rows are skipped (never multiplied by 0)
            total = torch.where(valid, -tlp, torch.zeros_like(tlp)).sum()
            if mean:
                total = total / denom
            if need_w:
                gw = gw.to(cdt).to(w.dtype)                         # the reference's one rounding
        ctx.save_for_backward(gh, gw)
        return total

    @staticmethod
    def backward(ctx, g):
        gh, gw = ctx.saved_tensors
        g = g.to(torch.float32)
        return (None if gh is None else gh * g.to(gh.dtype),
                None if gw is None else gw * g.to(gw.dtype), None, None, None, None, None, None)


def chunked_lm_loss(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int,
                    reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy of lm_head(h) against targets, `chunk` rows at a time.

    h: (..., d) final-norm output; weight: (V, d) lm_head weight (tied or not);
    targets: (...) int64 with -100 = no loss. reduction "sum" or "mean" (over supervised
    tokens), as model.forward's. Returns a 0-dim fp32 loss. The gradients are computed in
    this call when autograd will want them (grad mode on and h or weight requires grad)."""
    assert reduction in ("sum", "mean"), reduction
    assert chunk >= 1, chunk
    dev = h.device.type
    cdt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else h.dtype
    grad = torch.is_grad_enabled()
    return _ChunkedLinearCE.apply(h.reshape(-1, h.size(-1)), weight, targets.reshape(-1),
                                  int(chunk), reduction == "mean", cdt,
                                  grad and h.requires_grad, grad and weight.requires_grad)
