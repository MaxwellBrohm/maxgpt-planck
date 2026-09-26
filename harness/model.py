"""PlanckLM: the decoder-only model every Planck arm is built from.

Structure and init follow Max's MaxGPT-Ultra model/model.py (MaxGPTUltra class, Max's AI
Model/maxgpt-ultra). Planck adds a layer schedule: unique blocks (prelude, core, coda) and
an effective forward order in which the core may loop (config.schedule()). Shared blocks
are the same module objects, so parameters() and the parameter count see them once.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import docattn
from blocks import Block, RMSNorm, norm_scale_for, precompute_rope
from config import PlanckConfig


class PlanckLM(nn.Module):
    def __init__(self, cfg: PlanckConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        vr = cfg.vr_blocks()
        blocks: list[Block] = []
        for u in range(cfg.n_unique):
            owner = cfg.qk_owner(u)
            shared = None
            if owner != u:  # Planck (P-110): reuse the group owner's W_q and W_k modules
                a = blocks[owner].attn
                shared = (a.q_proj, a.k_proj)
            blocks.append(Block(cfg, has_vr=(u in vr), shared_qk=shared))
        self.blocks = nn.ModuleList(blocks)
        self.sched = cfg.schedule()
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.doc_attn = "mask"   # packed-row attention kernel, docattn.set_doc_attn (not in cfg)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """From Ultra: normal(0, init_std) for linears and the embedding, output projections
        scaled by 1/sqrt(2 * depth), attention gate zero. Planck uses EFFECTIVE depth here,
        since that is how many residual additions the stream receives."""
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and m.weight.device.type != "meta":
                nn.init.normal_(m.weight, mean=0.0, std=std)
        for name, p in self.named_parameters():
            if p.device.type == "meta":
                continue
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * self.cfg.depth))
            elif name.endswith("attn_gate_proj.weight"):
                nn.init.zeros_(p)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                doc: torch.Tensor | None = None, pos: torch.Tensor | None = None,
                reduction: str = "mean", return_hidden: bool = False):
        """idx, targets: (B, T) int64. targets uses -100 for positions without loss
        (assistant-only masking is applied by the data side).
        Planck (packing): doc (B, T) gives each token a document id; attention is then causal
        AND same-document, and RoPE positions restart at every document (pos, derived from
        doc when not given). doc=None is plain causal attention over the whole row.
        reduction "mean" (default) or "sum" (the trainer divides by the supervised-token
        count of the whole step, so micro-batches of different fill weigh correctly).
        Returns (logits, loss); return_hidden (S006, train.mtp): (logits, loss, z), z the final
        hidden state after the final norm, which the trainer's t+2 aux head (mtp.py) reads."""
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"T={T} exceeds seq_len {self.cfg.seq_len}"
        if self.cfg.forget_gate and self.doc_attn != "mask":   # S005: never run varlen without the bias
            raise RuntimeError(f"forget_gate needs doc_attn mask, got {self.doc_attn!r} (varlen takes no bias)")
        x = self.tok_emb(idx)
        mask = None
        if doc is None and pos is None:
            cos = self.rope_cos[:T].to(x.dtype)[None, None]
            sin = self.rope_sin[:T].to(x.dtype)[None, None]
        else:
            if pos is None:
                pos = positions_from_doc(doc)
            cos = self.rope_cos[pos].to(x.dtype)[:, None]          # (B, 1, T, hd)
            sin = self.rope_sin[pos].to(x.dtype)[:, None]
            if doc is not None and self.doc_attn == "varlen":
                mask = docattn.VarlenDocs(doc)     # no mask: one flash sequence per document
            elif doc is not None:
                mask = document_causal_mask(doc)
        canon_skip = None   # S004: Canon taps stop at document starts (doc=None: row start only)
        if self.cfg.canon and doc is not None:
            canon_skip = canon_skip_masks(doc, self.cfg.canon_kernel)
        fg = None           # S005: one (allowed, seg) pair for every layer; doc=None: plain causal
        if self.cfg.forget_gate:
            if mask is None:
                mask_fg = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()[None, None]
            else:
                mask_fg = mask
            fg = forget_segments(mask_fg)
        smear = None        # S007: keys smear up to document starts, from doc (doc=None: the row start only)
        if self.cfg.smear_key and doc is not None:
            smear = docattn.doc_starts(doc)
        v1 = None
        for i, u in enumerate(self.sched):
            x, v_loc = self.blocks[u](x, cos, sin, v1, norm_scale_for(self.cfg, i), mask,
                                      canon_skip=canon_skip, fg=fg, smear=smear)
            if i == 0 and self.cfg.value_residual:
                v1 = v_loc
        z = self.norm(x)
        logits = self.lm_head(z)
        loss = None
        if targets is not None:
            flat = logits.view(-1, logits.size(-1)).float()
            tgt = targets.reshape(-1)
            if (tgt != -100).any():
                loss = F.cross_entropy(flat, tgt, ignore_index=-100, reduction=reduction)
            else:
                loss = flat.sum() * 0.0
        if return_hidden:
            return logits, loss, z
        return logits, loss


def positions_from_doc(doc: torch.Tensor) -> torch.Tensor:
    """(B, T) document ids -> position of each token inside its document (0 at a doc start).
    A document is a maximal run of equal ids, so ids only need to differ between neighbours."""
    B, T = doc.shape
    ar = torch.arange(T, device=doc.device).expand(B, T)
    start = torch.ones_like(doc, dtype=torch.bool)
    start[:, 1:] = doc[:, 1:] != doc[:, :-1]
    first = torch.where(start, ar, torch.zeros_like(ar))
    return ar - torch.cummax(first, dim=1).values


def canon_skip_masks(doc: torch.Tensor, kernel: int) -> list[torch.Tensor]:
    """S004: (B, T) document ids -> kernel - 1 bool (B, T, 1) masks; [j - 1] is True where
    Canon tap j would read before the token's document start (in-document position < j).
    Built from doc (runs of equal ids, the mask's and varlen's rule), never from a given pos."""
    p = positions_from_doc(doc).unsqueeze(-1)
    return [p < j for j in range(1, kernel)]


def forget_segments(allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """S005: allowed bool (B|1, 1, T, T) -> (allowed, seg), seg = allowed as fp32 (B|1, T, T): row t is 1 on
    t's own document up to t, so seg @ log f is the in-document cumulative sum (blocks.Attention.forget_bias).
    Built once per forward, so every layer's matmul saves the same tensor for backward."""
    return allowed, allowed[:, 0].to(torch.float32)


def document_causal_mask(doc: torch.Tensor) -> torch.Tensor:
    """(B, T) document ids -> bool (B, 1, T, T): query t may attend key s iff s <= t and
    both are in the same document. The diagonal is always True, so no row is empty.
    Documents are maximal runs of equal ids (relabelled here), so a reused id further
    along the row still starts a new document."""
    T = doc.size(1)
    start = torch.ones_like(doc, dtype=torch.bool)
    start[:, 1:] = doc[:, 1:] != doc[:, :-1]
    run = torch.cumsum(start.long(), dim=1)
    causal = torch.ones(T, T, dtype=torch.bool, device=doc.device).tril()
    same = run[:, :, None] == run[:, None, :]
    return (same & causal)[:, None]


def build_model(cfg: PlanckConfig, device: str = "cpu") -> PlanckLM:
    """Build on a device. 'meta' allocates nothing (exact shapes, no values), which is
    how large sizes are counted without spending RAM."""
    with torch.device(device):
        return PlanckLM(cfg)
