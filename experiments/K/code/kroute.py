"""K3-R gradient routing (K3 notes K3-R; SPEC 14 train.grad_route), installed by ktrain.py with no harness edit.

  k:
    grad_route:
      sources: [fact_bio, fact_qa]   # data: source names whose items are FACT items (bio docs, fact QA)
      reserved: 122                  # the last 122 SwiGLU units of every layer are the reserved block

Gradient routing (Cloud et al., https://arxiv.org/abs/2410.04332): fact[t] says the item at position t of a packed row
came from a listed source. In every SwiGLU, unit j passes gradient at position t only when fact[t] == (j is
reserved): the gradient into the unit's activation and into its W_down column from that position are zero
otherwise. So the reserved block's W_gate and W_up rows and W_down columns learn from FACT positions only, every
other unit from the other positions, and attention and embeddings from all of them (through the allowed units and
the residual). The forward value is the plain SwiGLU's bit for bit (the same F.linear on the same activation), so a
routed step's loss equals the unrouted step's; only gradients change. Nothing draws from an RNG.

RoutedLoader is the harness Loader plus "src" in every batch: per position the index of the source its item came
from, -1 on padding. Each window is packed a second time as a shadow (every token replaced by its index in the
window's draw) with the window's own rng seed; pack_rows depends only on item lengths and that rng. Every real row is
checked against the shadow token by token (ids, doc, pos) before a source is read off it. Pack mode only.
"""
from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F

from blocks import SwiGLU
from data import Loader
from packing import pack_rows


class RoutedLoader(Loader):
    """Use as_routed(loader): the harness loader, unchanged in what it yields, plus b["src"]."""

    def _k_init(self) -> None:
        assert self.mode == "pack", "k.grad_route needs data.mode pack"
        self._k_rec, self._k_src, self._k_pending = [], [], []
        self.k_queue = collections.deque()
        for i, s in enumerate(self.sources):
            nxt = s.next_item

            def rec(nxt=nxt, i=i):
                ids, flags = nxt()
                self._k_rec.append((i, ids))
                return ids, flags
            s.next_item = rec

    def _build_window(self) -> None:
        self._k_rec = []
        super()._build_window()
        lens = [len(ids) for _, ids in self._k_rec]
        offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        flat = np.concatenate([np.asarray(ids, dtype=np.int64) for _, ids in self._k_rec])
        tok_src = np.repeat(np.asarray([i for i, _ in self._k_rec], dtype=np.int64), lens)
        shadow = [(np.arange(offs[j], offs[j + 1]), np.ones(lens[j], dtype=bool)) for j in range(len(lens))]
        rows = pack_rows(shadow, self.L, -1, np.random.default_rng([self.seed, self._w, 11]))
        if len(rows) != len(self._units):
            raise RuntimeError("k.grad_route: the shadow packing does not align with the real rows (row count)")
        self._k_src = []
        for r, u in zip(rows, self._units):
            g = r["idx"]
            m = g >= 0
            if not (np.array_equal(u["idx"][m], flat[g[m]]) and np.array_equal(r["doc"], u["doc"])
                    and np.array_equal(r["pos"], u["pos"])):
                raise RuntimeError("k.grad_route: the shadow packing does not align with the real rows")
            self._k_src.append(np.where(m, tok_src[np.maximum(g, 0)], -1))

    def _next_unit(self):
        unit = super()._next_unit()
        self._k_pending.append(self._k_src[self._u - 1])
        return unit

    def next_batch(self) -> dict:
        self._k_pending = []
        b = super().next_batch()
        b["src"] = torch.from_numpy(np.stack(self._k_pending))
        self.k_queue.append(b["src"])
        return b


def as_routed(loader: Loader) -> RoutedLoader:
    loader.__class__ = RoutedLoader
    loader._k_init()
    return loader


class _RoutedDown(torch.autograd.Function):
    """y = F.linear(h, w); backward with allowed[b, t, j] = fact[b, t] == reserved[j]:
    dh = (dy @ w) * allowed, dw = dy^T (h * allowed)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, w, fact, reserved):
        ctx.save_for_backward(h, w, fact, reserved)
        return F.linear(h, w)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gy):
        h, w, fact, reserved = ctx.saved_tensors
        allowed = fact[..., None] == reserved
        gh = torch.matmul(gy, w.to(gy.dtype)) * allowed
        hm = (h * allowed).to(gy.dtype)
        gw = torch.matmul(gy.reshape(-1, gy.shape[-1]).t(), hm.reshape(-1, hm.shape[-1]))
        return gh.to(h.dtype), gw.to(w.dtype), None, None


class RouteCtx:
    """What the routed MLPs read: mask = fact (B, T) bool on the model's device during a training forward, else
    None (every other call, the self-test, eval hooks and scorers run the plain SwiGLU)."""

    def __init__(self):
        self.mask = None


class RoutedSwiGLU(SwiGLU):
    def forward(self, x):
        ctx = getattr(self, "_k_route", None)
        if ctx is None or ctx.mask is None:
            return super().forward(x)
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        res = self._k_reserved
        if res.device != h.device:
            res = self._k_reserved = res.to(h.device)
        return _RoutedDown.apply(h, self.down_proj.weight, ctx.mask, res)


def reserved_units(hidden: int, n: int) -> list[int]:
    """The reserved block: the last n units (K3-R: 122 of 488)."""
    if not 0 < int(n) < int(hidden):
        raise ValueError(f"k.grad_route.reserved {n} must be between 1 and mlp_hidden - 1 ({hidden - 1})")
    return list(range(hidden - int(n), hidden))


def install(model, n_reserved: int, ctx: RouteCtx) -> dict:
    """Turn every block's SwiGLU into a RoutedSwiGLU (same parameters, names and state_dict). -> {layer: units}."""
    assert len(set(model.sched)) == len(model.sched), "k.grad_route assumes no looped or shared blocks"
    out = {}
    for li, blk in enumerate(model.blocks):
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            continue
        assert type(mlp) is SwiGLU, f"block {li}: expected a plain SwiGLU, got {type(mlp).__name__}"
        H = mlp.gate_proj.out_features
        units = reserved_units(H, n_reserved)
        res = torch.zeros(H, dtype=torch.bool)
        res[units] = True
        mlp.__class__ = RoutedSwiGLU
        mlp._k_route, mlp._k_reserved = ctx, res
        out[li] = units
    if not out:
        raise ValueError("k.grad_route: the model has no MLP")
    return out


class RoutedForward:
    """Wraps the trainer's forward: before each training micro-batch forward, the next src from the loader's queue
    (the trainer draws its micro-batches, then runs them in that order) becomes the fact mask."""

    def __init__(self, fwd, loader: RoutedLoader, ctx: RouteCtx, fact_sources: list[int]):
        self.fwd, self.loader, self.ctx = fwd, loader, ctx
        self.fact = torch.tensor(sorted(fact_sources), dtype=torch.long)

    def __call__(self, idx, tgt=None, doc=None, pos=None, **kw):
        src = self.loader.k_queue.popleft()
        if tuple(src.shape) != tuple(idx.shape):
            raise RuntimeError("k.grad_route: src and the batch disagree in shape")
        self.ctx.mask = torch.isin(src, self.fact).to(idx.device)
        try:
            return self.fwd(idx, tgt, doc, pos, **kw)
        finally:
            self.ctx.mask = None
