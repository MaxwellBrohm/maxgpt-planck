"""Packed-row attention without a dense mask: the doc_attn switch.

  "mask"    (default, the reference) model.document_causal_mask builds a bool (B, 1, T, T)
            mask and SDPA runs with attn_mask. No flash kernel takes a mask, so SDPA falls
            back to the mem-efficient kernel with a dense bias per layer.
  "varlen"  flash varlen attention (torch.nn.attention.varlen.varlen_attn, present in the
            PC's torch 2.13; CUDA, bf16/fp16). The (B, T) rows are flattened into one token
            stream and every document is its own causal sequence, so no mask exists at all
            and a token can only ever read keys of its own document.

Document = maximal run of equal ids in a row, and a row start always starts one (exactly
model.document_causal_mask's rule: padding is its own document, a reused id further along
starts a new one). RoPE positions are unchanged (model.positions_from_doc).
Chosen per model instance, not in PlanckConfig (a kernel choice, not an architecture, so
checkpoints and their model_cfg stay the same): set_doc_attn(model, "varlen"). Wired from
train.py (train.doc_attn) and pc/bench_micro.py (--doc-attn).
  "auto"    (the train.py / bench default since 2026-09-26, resolve_doc_attn) = varlen on cuda
            with bf16 when this torch has varlen_attn, else mask. The model attribute itself
            still defaults to "mask", so decode, RC-12 and tests that build a model directly
            keep the reference path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMPLS = ("mask", "varlen")
CHOICES = ("auto",) + IMPLS


def varlen_available() -> bool:
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_doc_attn(impl: str, device: str, precision: str) -> str:
    """"auto" -> "varlen" on cuda with bf16 (and a torch that has it), else "mask". An explicit
    "mask" or "varlen" is returned unchanged (set_doc_attn validates it)."""
    if impl != "auto":
        return impl
    ok = device == "cuda" and precision == "bf16" and varlen_available()
    return "varlen" if ok else "mask"


def set_doc_attn(model, impl: str, device: str | None = None) -> None:
    """Select the packed-row attention for a PlanckLM. varlen needs CUDA; say so up front."""
    if impl not in IMPLS:
        raise ValueError(f"doc_attn must be one of {IMPLS}, got {impl!r}")
    if impl == "varlen" and device is not None and device != "cuda":
        raise ValueError(f"doc_attn varlen needs device cuda (flash varlen), got {device!r}")
    model.doc_attn = impl


def doc_starts(doc: torch.Tensor) -> torch.Tensor:
    """(B, T) ids -> bool (B, T), True where a document starts (every row start included)."""
    start = torch.ones_like(doc, dtype=torch.bool)
    start[:, 1:] = doc[:, 1:] != doc[:, :-1]
    return start


def cu_seqlens(doc: torch.Tensor) -> torch.Tensor:
    """(B, T) ids -> int32 (n_docs + 1,) offsets of every document in the flattened B*T
    stream, ending with B*T. nonzero() makes one host sync per forward (the output size)."""
    s = doc_starts(doc).reshape(-1).nonzero().squeeze(1).to(torch.int32)
    return F.pad(s, (0, 1), value=doc.numel())


def _flash_varlen(q, k, v, cu, max_len: int, gqa: bool):
    """q (N, Hq, hd), k/v (N, Hkv, hd) -> (N, Hq, hd); causal inside each cu segment."""
    if q.device.type != "cuda":
        raise RuntimeError("doc_attn varlen runs on CUDA only (flash varlen)")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"doc_attn varlen needs bf16/fp16 activations, got {q.dtype} "
                           "(train.precision bf16)")
    from torch.nn.attention.varlen import varlen_attn
    return varlen_attn(q, k, v, cu, cu, max_len, max_len, window_size=(-1, 0), enable_gqa=gqa)


class VarlenDocs:
    """What PlanckLM.forward passes to every attention layer in place of the dense mask."""

    def __init__(self, doc: torch.Tensor):
        self.B, self.T = doc.shape
        self.cu = cu_seqlens(doc)

    def attend(self, q, k, v):
        """q (B, Hq, T, hd), k/v (B, Hkv, T, hd), post-RoPE, K/V NOT repeated for GQA.
        -> (B, Hq, T, hd), like F.scaled_dot_product_attention's output."""
        B, H, T, hd = q.shape
        if torch.is_autocast_enabled(q.device.type):    # a custom op: autocast does not cast
            dt = torch.get_autocast_dtype(q.device.type)  # it, SDPA's inputs would be
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
        flat = lambda x: x.transpose(1, 2).reshape(B * T, x.size(1), hd)  # noqa: E731
        # max_len T (an upper bound; the true longest document would need a second sync)
        o = _flash_varlen(flat(q), flat(k), flat(v), self.cu, T, k.size(1) != H)
        return o.view(B, T, H, hd).transpose(1, 2)
