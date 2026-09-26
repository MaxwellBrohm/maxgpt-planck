"""Transformer building blocks for MaxGPT-Planck.

Adapted from Max's MaxGPT-Ultra model/model.py (Max's AI Model/maxgpt-ultra, the block
validated in his 124M A/B): RMSNorm, RoPE, GQA attention with QK-norm, per-head gated
attention, normalized value residual, SwiGLU, pre-norm block with 1/sqrt(depth) norm
scaling. Changes for Planck, all marked "Planck:" below:
  - head_dim is explicit, so attention width (n_heads * head_dim) can differ from d_model
    (MLP-light arms move parameters from the MLP into attention).
  - W_q/W_k can be shared across layers and K can be tied to V (P-110).
  - norm scaling uses the EFFECTIVE depth index, passed at call time, because a looped
    block runs at several depths (P-111, P-150).
  - mlp_hidden = 0 builds an attention-only block.
  - the KV-cache decode path is left out for now (training harness first).
  - document masking for packed rows: an optional boolean attention mask (True = attend)
    replaces the plain causal flag, and RoPE tables may be per-row (positions restart at
    every document), see model.py.
  - S004 Canon layers (config canon, default off): CanonConv, residual causal depthwise convs
    on the normed inputs of attention and the MLP, stopped at document starts.
  - S005 forget gate (config forget_gate, default off): Attention.forget_bias, an additive logit
    bias sum over l in (j, i] of log f[l], in fp32, on the mask path and the doc=None path.
  - S007 smeared keys (config smear_key, default off): Attention.smear, raw key k_t + alpha_h * k_(t-1)
    (same document) before QK-norm and RoPE, on every path (doc=None, mask, varlen); values unsmeared.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PlanckConfig


class RMSNorm(nn.Module):
    """From Ultra: RMS rescale in fp32, then a learned per-feature gain. No bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def precompute_rope(head_dim: int, seq_len: int, theta: float):
    """From Ultra: (seq_len, head_dim) cos/sin tables for rotary positions."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, heads, T, hd); cos/sin already broadcastable: (1, 1, T, hd) or (B, 1, T, hd).
    # Planck: per-row tables let positions restart at each packed document.
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """From Ultra: (B, n_kv, T, hd) -> (B, n_kv*n_rep, T, hd) for GQA."""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return x[:, :, None].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)


FORGET_B0 = 7.99   # S005 gate bias init: f = sigmoid(7.99) = 2^(-1/2048) to 3 digits (half-life 2,048 tokens)


class Attention(nn.Module):
    def __init__(self, cfg: PlanckConfig, has_vr: bool,
                 shared_qk: tuple[nn.Linear, nn.Linear] | None = None):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        d, hq, hkv = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
        # Planck: a share-group follower reuses its owner's W_q/W_k modules (P-110). The
        # module objects are the same, so parameters() counts them once.
        if shared_qk is None:
            self.q_proj = nn.Linear(d, hq, bias=False)
            self.k_proj = nn.Linear(d, hkv, bias=False)
        else:
            self.q_proj, self.k_proj = shared_qk
        self.kv_tie = cfg.kv_tie  # Planck: K = V drops the value projection
        if not cfg.kv_tie:
            self.v_proj = nn.Linear(d, hkv, bias=False)
        self.o_proj = nn.Linear(hq, d, bias=False)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        # From Ultra: per-head gate 2*sigmoid(W x), W zero-initialized so it starts as identity.
        self.attn_gate = cfg.attn_gate
        if cfg.attn_gate:
            self.attn_gate_proj = nn.Linear(d, cfg.n_heads, bias=False)
        # From Ultra: V = s * (a1*V + a2*V_1) / sqrt(a1^2 + a2^2), starting at (1, 1, 0).
        self.has_vr = has_vr
        if has_vr:
            self.vr_scale = nn.Parameter(torch.ones(()))
            self.vr_alpha = nn.Parameter(torch.tensor([1.0, 0.0]))
        # Planck (S005, P-020): forget gate f[t, h] = sigmoid(forget_w[h] . x_t + forget_b[h]) on the
        # attention input x. Constants, no RNG draw (every other parameter keeps its draw): w = 0, b = 7.99.
        self.forget_gate = cfg.forget_gate
        if cfg.forget_gate:
            self.forget_w = nn.Parameter(torch.zeros(cfg.n_heads, d))
            self.forget_b = nn.Parameter(torch.full((cfg.n_heads,), FORGET_B0))
        # Planck (S007, P-024): smeared keys, alpha[h] per KV head. A constant 0 (no RNG draw), so the arm starts
        # as the exact flag-off function and every other parameter keeps its draw. 1-D: AdamW 'scalar' group.
        self.smear_key = cfg.smear_key
        if cfg.smear_key:
            self.smear_alpha = nn.Parameter(torch.zeros(cfg.n_kv_heads))

    def smear(self, k, starts=None):
        """S007: raw keys k (B, T, Hkv, hd) -> k_t + alpha_h * k_(t-1). k_(t-1) reads zero at t = 0 and, with
        starts (B, T) bool (True where a document starts; model.forward, from doc), at every document start."""
        prev = F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))       # prev[:, t] = k[:, t - 1], 0 at t = 0
        if starts is not None:
            prev = prev.masked_fill(starts[:, :, None, None], 0.0)
        return k + self.smear_alpha.to(k.dtype).view(-1, 1) * prev

    def forget_logf(self, x):
        """S005: log f, fp32 (B, H, T), from the attention input x (B, T, d); autocast off (at init
        log f = -3.4e-4, below bf16's resolution next to the sums it joins)."""
        with torch.autocast(x.device.type, enabled=False):
            z = F.linear(x.float(), self.forget_w.float(), self.forget_b.float())
            return F.logsigmoid(z).transpose(1, 2)

    def forget_bias(self, x, fg):
        """S005: fp32 (B, H, T, T) logit bias, -inf where attention is not allowed. fg = (allowed bool
        (B|1, 1, T, T), seg = allowed as fp32 (B|1, T, T)), built once per forward (model.forget_segments).
        c = seg @ log f is the IN-DOCUMENT cumulative sum (c[t] = sum of log f[l], l <= t in t's document)
        with no arithmetic path from another document, and c[i] - c[j] = sum over l in (j, i] of log f[l]."""
        allowed, seg = fg
        with torch.autocast(x.device.type, enabled=False):
            logf = self.forget_logf(x)
            c = torch.matmul(seg, logf.transpose(1, 2)).transpose(1, 2)          # (B, H, T)
            bias = c.unsqueeze(-1) - c.unsqueeze(-2)
            return bias.masked_fill(~allowed, float("-inf"))

    def forward(self, x, cos, sin, v1=None, mask=None, fg=None, smear=None):
        """Returns (out, v_local). v_local is this layer's own pre-mix values.
        mask: None (plain causal), bool (B, 1, T, T), True = may attend (Planck), or a
        docattn.VarlenDocs (packed rows without a mask, doc_attn "varlen").
        fg: S005 (forget_gate only), model.forget_segments of the same mask (doc=None: plain causal).
        smear: S007 (smear_key only), document starts (B, T) bool, or None (the row is one sequence)."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k_raw = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = k_raw if self.kv_tie else self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        k = k_raw
        if self.smear_key:                    # S007: every path (doc=None, mask, varlen); V keeps k_raw
            k = self.smear(k_raw, smear)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        v_local = v
        if self.has_vr and v1 is not None:
            a1, a2 = self.vr_alpha[0], self.vr_alpha[1]
            v = self.vr_scale * (a1 * v + a2 * v1) * torch.rsqrt(a1 * a1 + a2 * a2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        bias = None
        if self.forget_gate:                  # S005: every path, doc=None included (model refuses varlen)
            bias = self.forget_bias(x, fg)
        if mask is not None and not isinstance(mask, torch.Tensor):
            out = mask.attend(q, k, v)        # docattn.VarlenDocs (GQA inside the kernel)
        else:
            k_, v_ = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
            if bias is not None:              # S005: allowed-or--inf plus the forget bias, cast here only
                out = F.scaled_dot_product_attention(q, k_, v_, attn_mask=bias.to(q.dtype))
            elif mask is None:
                out = F.scaled_dot_product_attention(q, k_, v_, is_causal=True)
            else:  # Planck: causal AND same-document, built by the model
                out = F.scaled_dot_product_attention(q, k_, v_, attn_mask=mask)
        if self.attn_gate:
            gate = 2.0 * torch.sigmoid(self.attn_gate_proj(x))           # (B, T, H)
            out = out * gate.transpose(1, 2).unsqueeze(-1).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), v_local


class SwiGLU(nn.Module):
    """From Ultra: down(silu(gate(x)) * up(x))."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CanonConv(nn.Module):
    """Planck (S004, P-148; experiments/S004_canon/notes.txt): a Canon layer, a residual causal
    depthwise 1-D convolution over time, one K-tap kernel per channel, no bias:
        out[t] = h[t] + sum_{j=0..K-1} weight[:, j] * h[t - j]
    A tap that would read before the start of t's document (in-document position of t < j)
    reads zero: before the row start by zero padding, at packed document starts through
    skip[j - 1] (model.canon_skip_masks). Zero init from torch.zeros, no RNG draw, so the arm
    starts as the exact flag-off function and every other parameter keeps its draw."""

    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d, kernel))

    def forward(self, h, skip=None):
        """h (B, T, d). skip: None (the row is one sequence: the doc=None path) or K-1 bool
        (B, T, 1) masks, skip[j - 1] True where tap j must read zero."""
        K, T = self.weight.size(1), h.size(1)
        w = self.weight.to(h.dtype)
        hp = F.pad(h, (0, 0, K - 1, 0))                 # hp[:, K - 1 + t] = h[:, t]
        out = torch.addcmul(h, h, w[:, 0])              # h + tap 0
        for j in range(1, K):
            s = hp[:, K - 1 - j:K - 1 - j + T]          # s[:, t] = h[:, t - j], 0 before t = 0
            if skip is not None:
                s = s.masked_fill(skip[j - 1], 0.0)
            out = torch.addcmul(out, s, w[:, j])
        return out


class Block(nn.Module):
    """From Ultra: pre-norm attention + SwiGLU with residuals. Planck: the norm scale is
    passed per call (effective depth), and mlp_hidden = 0 drops the MLP sublayer."""

    def __init__(self, cfg: PlanckConfig, has_vr: bool, shared_qk=None):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg, has_vr, shared_qk)
        self.has_mlp = cfg.mlp_hidden > 0
        if self.has_mlp:
            self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
            self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)
        # Planck (S004): Canon convs on the normed input of attention (A) and of the MLP (C)
        sites = cfg.canon_sites()
        self.canon_a = CanonConv(cfg.d_model, cfg.canon_kernel) if "A" in sites else None
        self.canon_c = CanonConv(cfg.d_model, cfg.canon_kernel) if "C" in sites else None

    def forward(self, x, cos, sin, v1=None, norm_scale: float = 1.0, mask=None, canon_skip=None, fg=None,
                smear=None):
        h = self.attn_norm(x)
        if norm_scale != 1.0:
            h = h * norm_scale
        if self.canon_a is not None:        # every path (doc=None, mask, varlen); skip from doc
            h = self.canon_a(h, canon_skip)
        a, v_local = self.attn(h, cos, sin, v1, mask, fg, smear)
        x = x + a
        if self.has_mlp:
            h = self.mlp_norm(x)
            if norm_scale != 1.0:
                h = h * norm_scale
            if self.canon_c is not None:
                h = self.canon_c(h, canon_skip)
            x = x + self.mlp(h)
        return x, v_local


def norm_scale_for(cfg: PlanckConfig, pos: int) -> float:
    """From Ultra's LayerNorm scaling, keyed on effective depth index pos (0-based)."""
    return 1.0 / math.sqrt(pos + 1) if cfg.norm_scaling else 1.0
