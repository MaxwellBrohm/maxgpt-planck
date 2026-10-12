"""Batched NorMuon and AdamW steps: the same math as optim.Muon._muon_group and _adamw_group,
with far fewer kernel launches (optim key batched: true; default off).

Why: on the RTX 5070 the eager training step is launch bound, and the per-matrix NorMuon loop
issued about 4,300 kernel launches per step at 5M (7,500 at 20M) for 6-18 ms of kernel time
(the GPU profile of 2d9c65e). Measured with torch.profiler, clip + opt.step issue 4,785 ->
261 kernels at 5M, 8,516 -> 265 at 20M, 9,168 -> 267 at 30M. Here:
  NorMuon  momentum and Nesterov with torch._foreach_*; ONE Newton-Schulz per oriented shape
           on a stacked (k, r, c) tensor (r <= c; tall matrices go in transposed, exactly as
           the per-matrix code transposes them), with each matrix normalized by its own norm;
           row variance, the RMS-0.2 scale, cautious weight decay and the update on the stack
           or with foreach ops.
  AdamW    foreach ops over each group (params with equal step counts together).
Optimizer state keeps the per-parameter layout of optim.Muon (momentum, row_v, step,
exp_avg, exp_avg_sq), so checkpoints move between the two implementations in both
directions. Parity with the per-matrix code: test_optim_batched.py.
"""
from __future__ import annotations

import math

import torch

from optim import _NS_COEFFS


def zeropower_via_newtonschulz5_batched(X: torch.Tensor, steps: int = 5, dtype=None) -> torch.Tensor:
    """optim.zeropower_via_newtonschulz5 for a (k, r, c) stack with r <= c, each matrix
    normalized by its own Frobenius norm. The same elementwise ops in the same order as the
    per-matrix code (not baddbmm): on CPU fp32 the result is bit-equal for most shapes, and
    fusing b*A + c*(A@A) into baddbmm moved it by 1-2e-6 relative, about what the per-matrix
    code itself moves under a 1e-7 relative input perturbation (measured; saves ~5 launches
    per NS step and group, which is not worth losing that)."""
    assert X.ndim == 3 and X.size(1) <= X.size(2), tuple(X.shape)
    a, b, c = _NS_COEFFS
    X = X.to(dtype) if dtype is not None else X
    X = X / (torch.linalg.vector_norm(X, dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def _ns_plan(opt, ps: list, ms: list) -> list:
    """[(wide_idx, tall_idx)] per (short side, long side, dtype, device), cached per param set.
    wide_idx: r <= c (NS as is); tall_idx: r > c (NS on the transpose)."""
    key = tuple(id(p) for p in ps)
    cache = opt.__dict__.setdefault("_batched_plans", {})
    if key not in cache:
        plan: dict = {}
        for i, m in enumerate(ms):
            r, c = m.shape
            k = (min(r, c), max(r, c), m.dtype, m.device)
            plan.setdefault(k, ([], []))[1 if r > c else 0].append(i)
        cache[key] = list(plan.values())
    return cache[key]


def _cautious_factors(U: list, P: list, lr: float, wd: float) -> list:
    """[1 - lr*wd*(u*p > 0)] per tensor, bit-equal to the per-tensor expression:
    clamp_min(sign(u*p), 0) is 1 exactly where u*p > 0, else 0; 1 + (-lr*wd)*m == 1 - lr*wd*m."""
    m = torch._foreach_mul(U, P)
    torch._foreach_sign_(m)
    torch._foreach_clamp_min_(m, 0.0)
    torch._foreach_mul_(m, -lr * wd)
    torch._foreach_add_(m, 1.0)
    return m


def _decay_and_update(P: list, U: list, lr: float, wd: float, cautious: bool) -> None:
    if wd:
        if cautious:
            torch._foreach_mul_(P, _cautious_factors(U, P, lr, wd))
        else:
            torch._foreach_mul_(P, 1 - lr * wd)
    torch._foreach_add_(P, U, alpha=-lr)


@torch.no_grad()
def muon_group_batched(opt, group) -> None:
    lr, wd, mom, eps = group["lr"], group["weight_decay"], group["momentum"], group["eps"]
    ps = [p for p in group["params"] if p.grad is not None]
    if not ps:
        return
    gs = [p.grad if p.grad.ndim == 2 else p.grad.reshape(p.grad.size(0), -1) for p in ps]
    for p, g in zip(ps, gs):
        st = opt.state[p]
        if "momentum" not in st:
            st["momentum"] = torch.zeros_like(g)
            if group["normalize"]:
                st["row_v"] = torch.zeros(g.size(0), device=g.device, dtype=torch.float32)
    bufs = [opt.state[p]["momentum"] for p in ps]
    torch._foreach_mul_(bufs, mom)
    torch._foreach_add_(bufs, gs)
    for wide, tall in _ns_plan(opt, ps, gs):
        idx = wide + tall
        if group["nesterov"]:
            ms = list(torch._foreach_add([gs[i] for i in idx], [bufs[i] for i in idx], alpha=mom))
        else:
            ms = [bufs[i] for i in idx]
        X = torch.stack(ms[:len(wide)] + [m.mT for m in ms[len(wide):]])
        del ms
        O = zeropower_via_newtonschulz5_batched(X, group["ns_steps"], opt.ns_dtype).float()
        del X
        parts = []
        if wide:
            parts.append((wide, O[:len(wide)]))
        if tall:
            parts.append((tall, O[len(wide):].mT.contiguous()))   # back to (rows, cols)
        del O
        for sub, S in parts:
            if group["normalize"]:                         # NorMuon: per-row second moment
                vs = [opt.state[ps[i]]["row_v"] for i in sub]
                torch._foreach_mul_(vs, group["beta2"])
                torch._foreach_add_(vs, list(S.pow(2).mean(dim=2).unbind(0)), alpha=1 - group["beta2"])
                S = S / (torch.stack(vs).sqrt().unsqueeze(2) + eps)
            n = S.size(1) * S.size(2)
            S = S * (0.2 * math.sqrt(n) / (torch.linalg.vector_norm(S, dim=(1, 2), keepdim=True) + eps))
            P = [ps[i] for i in sub]
            U = [u.view(p.shape).to(p.dtype) for u, p in zip(S.unbind(0), P)]
            _decay_and_update(P, U, lr, wd, group["cautious"])


@torch.no_grad()
def adamw_group_batched(opt, group) -> None:
    lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
    by_step: dict[int, list] = {}
    for p in group["params"]:
        if p.grad is None:
            continue
        st = opt.state[p]
        if "step" not in st:
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)
        st["step"] += 1
        by_step.setdefault(st["step"], []).append(p)
    for step, P in by_step.items():
        G = [p.grad for p in P]
        M = [opt.state[p]["exp_avg"] for p in P]
        V = [opt.state[p]["exp_avg_sq"] for p in P]
        torch._foreach_mul_(M, b1)
        torch._foreach_add_(M, G, alpha=1 - b1)
        torch._foreach_mul_(V, b2)
        torch._foreach_addcmul_(V, G, G, value=1 - b2)
        bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
        num = torch._foreach_div(M, bc1)
        den = torch._foreach_div(V, bc2)
        torch._foreach_sqrt_(den)
        torch._foreach_add_(den, eps)
        U = torch._foreach_div(num, den)
        del num, den
        _decay_and_update(P, U, lr, wd, group["cautious"])
