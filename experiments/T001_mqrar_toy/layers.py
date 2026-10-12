"""T001 mixers (PREREG.txt MODELS AND ARMS): softmax attention and its recency variants, the gated delta rule, the
depthwise causal conv. Shapes: x (B, L, d); per-head tensors (B, H, L, D)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_KINDS = ("base", "fg", "sb", "sa", "alibi", "smear")


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    L, D = x.shape[-2], x.shape[-1]
    half = D // 2
    inv = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(L, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def prev_token(x: torch.Tensor) -> torch.Tensor:
    """x[..., t - 1, :] along the sequence axis (-2), zero at t = 0."""
    return F.pad(x, (0, 0, 1, 0))[..., :-1, :]


class CausalConv(nn.Module):
    """Depthwise causal conv: out[t] = sum_{j < k} w[:, j] * x[t - j] (zero before the row start)."""

    def __init__(self, d: int, k: int = 4, zero: bool = True):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.zeros(d, k))
        self.zero = zero                   # zero init draws nothing from the RNG (model.init_weights skips it)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        xp = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return F.conv1d(xp, self.weight.flip(-1).unsqueeze(1), groups=d).transpose(1, 2)


def stick_breaking(z: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
    """A[i, j] = sigmoid(z[i, j]) * prod_{j < k <= i} (1 - sigmoid(z[i, k])) for j <= i, else 0."""
    lnb = F.logsigmoid(-z).masked_fill(~causal, 0.0)
    after = torch.flip(torch.cumsum(torch.flip(lnb, [-1]), -1), [-1]) - lnb      # sum over k > j (k <= i by the mask)
    return torch.exp(F.logsigmoid(z) + after).masked_fill(~causal, 0.0)


def selective_mask(s0: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
    """Selective Attention: S = ReLU(head-0 logits) on the causal part, BOS column and diagonal zeroed, shifted one
    query down, cumulative over queries. s0 (B, L, L) -> F (B, L, L), to subtract from every head's logits."""
    L = s0.shape[-1]
    S = torch.relu(s0.masked_fill(~causal, 0.0))
    keep = torch.ones(L, L, dtype=torch.bool, device=s0.device)
    keep[:, 0] = False
    keep.fill_diagonal_(False)
    S = S * keep
    S = F.pad(S, (0, 0, 1, 0))[..., :-1, :]
    return torch.cumsum(S, dim=-2)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, kind: str = "base", fg_len: int = 865):
        super().__init__()
        if kind not in ATTN_KINDS:
            raise ValueError(f"unknown attention kind {kind!r}")
        self.kind, self.h, self.dh = kind, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if kind == "fg":
            f = 2.0 ** (-1.0 / fg_len)
            self.fw = nn.Parameter(torch.zeros(n_heads, d))
            self.fb = nn.Parameter(torch.full((n_heads,), math.log(f / (1.0 - f))))
        if kind == "alibi":
            m = torch.tensor([2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])
            self.ap = nn.Parameter(torch.log(torch.expm1(m)))            # softplus(ap) = m
        if kind == "smear":
            self.alpha = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.kind == "smear":
            k = k + self.alpha.view(1, -1, 1, 1) * prev_token(k)
        if self.kind != "sb":
            q, k = rope(q), rope(k)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        if self.kind == "fg":
            c = F.logsigmoid(x @ self.fw.t() + self.fb).cumsum(1).transpose(1, 2)        # (B, H, L)
            s = s + (c[..., :, None] - c[..., None, :])
        elif self.kind == "alibi":
            pos = torch.arange(L, device=x.device)
            dist = (pos[:, None] - pos[None, :]).clamp(min=0).to(s.dtype)
            s = s - F.softplus(self.ap).view(1, -1, 1, 1) * dist
        elif self.kind == "sa":
            s = s - selective_mask(s[:, 0], causal)[:, None]
        if self.kind == "sb":
            y = stick_breaking(s, causal) @ v
        else:
            y = torch.softmax(s.masked_fill(~causal, float("-inf")), dim=-1) @ v
        return self.o(y.transpose(1, 2).reshape(B, L, d))


def delta_naive(q, k, v, log_alpha, beta):
    """Reference gated delta rule, one step at a time. q, k (B, H, L, Dk); v (B, H, L, Dv); log_alpha, beta (B, H, L).
    M_t = alpha_t (I - beta_t k_t k_t^T) M_{t-1} + beta_t k_t v_t^T (M: Dk x Dv), o_t = M_t^T q_t."""
    B, H, L, Dk = k.shape
    M = k.new_zeros(B, H, Dk, v.shape[-1])
    out = []
    for t in range(L):
        kt, vt = k[:, :, t], v[:, :, t]
        a = log_alpha[:, :, t].exp()[..., None, None]
        b = beta[:, :, t][..., None, None]
        kM = torch.einsum("bhd,bhde->bhe", kt, M)
        M = a * (M - b * kt[..., :, None] * kM[..., None, :]) + b * kt[..., :, None] * vt[..., None, :]
        out.append(torch.einsum("bhd,bhde->bhe", q[:, :, t], M))
    return torch.stack(out, dim=2)


def delta_chunked(q, k, v, log_alpha, beta, chunk: int = 64):
    """The same recurrence, chunkwise. Within a chunk with start state M0 and g_r = prod_{i<=r} alpha_i:
    u_r = beta_r (v_r - g_r M0^T k_r - sum_{i<r} (g_r/g_i)(k_r.k_i) u_i), solved as a unit lower-triangular system;
    o_r = g_r M0^T q_r + sum_{i<=r} (g_r/g_i)(q_r.k_i) u_i; M_end = g_C M0 + sum_i (g_C/g_i) k_i u_i^T."""
    B, H, L, Dk = k.shape
    M = k.new_zeros(B, H, Dk, v.shape[-1])
    out = []
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        c = e - s
        Q, Kc, Vc, b = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], beta[:, :, s:e]
        G = log_alpha[:, :, s:e].cumsum(-1)
        diff = G[..., :, None] - G[..., None, :]
        strict = torch.ones(c, c, dtype=torch.bool, device=k.device).tril(-1)
        incl = torch.ones(c, c, dtype=torch.bool, device=k.device).tril()
        gam_s = diff.masked_fill(~strict, float("-inf")).exp()
        gam = diff.masked_fill(~incl, float("-inf")).exp()
        g = G.exp()[..., None]
        A = b[..., :, None] * gam_s * (Kc @ Kc.transpose(-1, -2))
        rhs = b[..., None] * (Vc - g * (Kc @ M))
        eye = torch.eye(c, dtype=k.dtype, device=k.device)
        U = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
        out.append(g * (Q @ M) + ((Q @ Kc.transpose(-1, -2)) * gam) @ U)
        M = G[..., -1].exp()[..., None, None] * M + Kc.transpose(-1, -2) @ ((G[..., -1:] - G).exp()[..., None] * U)
    return torch.cat(out, dim=2)


class GatedDelta(nn.Module):
    """P-022's layer (PREREG gdn): projections, optional kernel-4 causal conv + SiLU, L2-normed q and k, gates."""

    def __init__(self, d: int, n_heads: int, conv: bool = True, chunk: int = 64):
        super().__init__()
        self.h, self.dh, self.chunk = n_heads, d // n_heads, chunk
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.gates = nn.Linear(d, 2 * n_heads, bias=True)
        self.conv = CausalConv(3 * d, 4, zero=False) if conv else None
        self.onorm = nn.RMSNorm(self.dh)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        u = self.qkv(x)
        u = F.silu(self.conv(u) if self.conv is not None else u)
        q, k, v = u.view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        a, b = self.gates(x).transpose(1, 2).chunk(2, dim=1)              # (B, H, L) each
        y = delta_chunked(q, k, v, F.logsigmoid(a), torch.sigmoid(b), self.chunk)
        return self.o(self.onorm(y).transpose(1, 2).reshape(B, L, d))
