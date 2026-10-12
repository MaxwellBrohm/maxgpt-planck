"""KV-cached incremental decoding for PlanckLM, as an adapter (model.py and blocks.py are unchanged).

It runs the model's OWN modules (embedding, norms, projections, gates, MLPs, head) in the order of
PlanckLM.forward with doc=None (one plain causal sequence), keeping post-RoPE keys and post-mix values per
EFFECTIVE layer (a looped block runs at several depths and sees different inputs at each, so it gets one cache
per application). Value residual: the mix is per position, so caching the mixed values is exact; v1 is effective
layer 0's own values of the new positions. Only two shapes are supported: prefill from an empty cache (plain
causal SDPA) and then one token at a time (the single query attends every cached key).

S004 Canon convs (cfg.canon): each (effective layer, site) keeps its last K - 1 conv INPUTS (the normed, scaled
h), and the conv runs on those plus the new positions (exact: a tap never reaches further back than K - 1).
S005 forget gate (cfg.forget_gate): each effective layer keeps the running in-document sums c of log f (fp32, one
per cached key; the cache is one document), and the new queries get c_i - c_j as their logit bias.
S007 smeared keys (cfg.smear_key): each effective layer keeps its last RAW key (after k_proj, before the smear),
and the block's own Attention.smear runs on it plus the new positions (the cache holds the smeared keys).
test_decode.py checks prefill + steps against the full forward at every position on every testutil.ARMS arm
(scrambled weights). Any change to Attention.forward, Block.forward or PlanckLM.forward must keep that test green.
Used by rc12/planck_responder.py (cache=True, the default there).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from blocks import apply_rope, norm_scale_for, repeat_kv


class KVDecoder:
    def __init__(self, model):
        self.m, self.cfg = model, model.cfg
        self.k: list = [None] * len(model.sched)
        self.v: list = [None] * len(model.sched)
        self.n = 0                                   # tokens already in the cache
        self.hist: dict = {}                         # S004: (layer, site) -> last K - 1 Canon inputs
        self.c: list = [None] * len(model.sched)     # S005: per effective layer, fp32 (B, H, n) sums of log f
        self.kraw: list = [None] * len(model.sched)  # S007: per effective layer, the last raw key (B, 1, Hkv, hd)

    def _canon(self, conv, h, key):
        """The block's own CanonConv on the cached last K - 1 inputs plus the new positions."""
        if conv is None:
            return h
        prev = self.hist.get(key)
        full = h if prev is None else torch.cat([prev, h], dim=1)
        self.hist[key] = full[:, -(conv.weight.size(1) - 1):]
        return conv(full)[:, -h.size(1):]

    def _attn(self, att, x, cos, sin, v1, i):
        B, T, _ = x.shape
        q = att.q_proj(x).view(B, T, att.n_heads, att.head_dim)
        k_raw = att.k_proj(x).view(B, T, att.n_kv_heads, att.head_dim)
        v = k_raw if att.kv_tie else att.v_proj(x).view(B, T, att.n_kv_heads, att.head_dim)
        k = k_raw
        if att.smear_key:
            k = self._smear(att, k_raw, i)
        if att.qk_norm:
            q, k = att.q_norm(q), att.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        v_local = v
        if att.has_vr and v1 is not None:
            a1, a2 = att.vr_alpha[0], att.vr_alpha[1]
            v = att.vr_scale * (a1 * v + a2 * v1) * torch.rsqrt(a1 * a1 + a2 * a2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.k[i] is not None:
            k, v = torch.cat([self.k[i], k], dim=2), torch.cat([self.v[i], v], dim=2)
        self.k[i], self.v[i] = k, v
        k_, v_ = repeat_kv(k, att.n_rep), repeat_kv(v, att.n_rep)
        if att.forget_gate:
            out = F.scaled_dot_product_attention(q, k_, v_, attn_mask=self._forget_bias(att, x, i).to(q.dtype))
        elif self.n == 0:
            out = F.scaled_dot_product_attention(q, k_, v_, is_causal=True)
        else:
            assert T == 1, "after the prefill, decode one token at a time"
            out = F.scaled_dot_product_attention(q, k_, v_)
        if att.attn_gate:
            gate = 2.0 * torch.sigmoid(att.attn_gate_proj(x))
            out = out * gate.transpose(1, 2).unsqueeze(-1).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, T, att.n_heads * att.head_dim)
        return att.o_proj(out), v_local

    def _smear(self, att, k_raw, i):
        """S007: the block's own smear on the cached last raw key plus the new positions (one document)."""
        prev = self.kraw[i]
        full = k_raw if prev is None else torch.cat([prev, k_raw], dim=1)
        self.kraw[i] = full[:, -1:]
        return att.smear(full)[:, -k_raw.size(1):]

    def _forget_bias(self, att, x, i):
        """S005: fp32 (B, H, T, n + T) bias for the new queries; c extends the cached sums (autocast off)."""
        with torch.autocast(x.device.type, enabled=False):
            c = torch.cumsum(att.forget_logf(x), dim=-1)                       # (B, H, T)
            if self.c[i] is not None:
                c = c + self.c[i][..., -1:]
                c_all = torch.cat([self.c[i], c], dim=-1)
            else:
                c_all = c
            self.c[i] = c_all
            bias = c.unsqueeze(-1) - c_all.unsqueeze(-2)
            if self.n == 0:
                T = x.size(1)
                bias = bias.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
            else:
                assert x.size(1) == 1, "after the prefill, decode one token at a time"
            return bias

    @torch.no_grad()
    def forward(self, ids: list[int], device) -> torch.Tensor:
        """feed ids (the whole prompt first, then one token per call); -> logits (T, V) of the fed positions."""
        m, cfg, P, T = self.m, self.cfg, self.n, len(ids)
        assert T >= 1 and P + T <= cfg.seq_len, f"{P}+{T} tokens exceed seq_len {cfg.seq_len}"
        x = m.tok_emb(torch.tensor([ids], dtype=torch.long, device=device))
        cos = m.rope_cos[P:P + T].to(x.dtype)[None, None]
        sin = m.rope_sin[P:P + T].to(x.dtype)[None, None]
        v1 = None
        for i, u in enumerate(m.sched):
            blk, scale = m.blocks[u], norm_scale_for(cfg, i)
            h = blk.attn_norm(x)
            if scale != 1.0:
                h = h * scale
            h = self._canon(blk.canon_a, h, (i, "A"))
            a, v_local = self._attn(blk.attn, h, cos, sin, v1, i)
            x = x + a
            if blk.has_mlp:
                h = blk.mlp_norm(x)
                if scale != 1.0:
                    h = h * scale
                h = self._canon(blk.canon_c, h, (i, "C"))
                x = x + blk.mlp(h)
            if i == 0 and cfg.value_residual:
                v1 = v_local
        self.n = P + T
        return m.lm_head(m.norm(x))[0]
