"""K2 and K3 tools on a harness PlanckLM (SPEC 14 k2_ablate.py and prune.py), by forward hooks: no harness edit.

Ablation(model, mlp={layer: [units]}, heads={layer: [heads]}, sublayers=[("attn"|"mlp", layer)], mode="zero"|"mean")
  is a context manager. An empty spec registers no hook, so the forward is the unablated one bit for bit. Units:
  N1 SwiGLU unit j of layer l = row j of W_gate and W_up, column j of W_down (zeroed as the down_proj input);
  N2 head h = its slice of the o_proj input; N3 a whole sublayer's output (the residual passes unchanged; the
  attention sublayer still hands its values to the value residual, which a skipped block in a real model would not).
prune_mlp removes units (smaller SwiGLU modules; the model stays trainable). reset_units is K3's reset: rows of
W_gate and W_up redrawn at the harness init std, column of W_down zeroed, so step 0 equals zero ablation.
trace is M4 causal tracing: restore one (layer, position) residual of a clean run inside a corrupted run.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)
from blocks import SwiGLU                                          # noqa: E402


def _check_plain(model):
    assert len(set(model.sched)) == len(model.sched), "K tools assume no looped or shared blocks"


class Ablation:
    def __init__(self, model, mlp=None, heads=None, sublayers=None, mode="zero", means=None):
        assert mode in ("zero", "mean")
        _check_plain(model)
        self.model, self.mode, self.means = model, mode, means or {}
        self.mlp = {int(k): sorted(set(int(u) for u in v)) for k, v in (mlp or {}).items() if len(v)}
        self.heads = {int(k): sorted(set(int(h) for h in v)) for k, v in (heads or {}).items() if len(v)}
        self.sub = list(sublayers or [])
        self.handles = []

    def _mlp_hook(self, layer):
        units = torch.tensor(self.mlp[layer])

        def pre(mod, args):
            h = args[0].clone()
            u = units.to(h.device)
            h[..., u] = 0.0 if self.mode == "zero" else self.means["mlp"][layer].to(h)[u]
            return (h,)
        return pre

    def _head_hook(self, layer, hd):
        cols = torch.tensor([h * hd + i for h in self.heads[layer] for i in range(hd)])

        def pre(mod, args):
            x = args[0].clone()
            c = cols.to(x.device)
            x[..., c] = 0.0 if self.mode == "zero" else self.means["heads"][layer].to(x)[c]
            return (x,)
        return pre

    def __enter__(self):
        blocks = self.model.blocks
        for layer in self.mlp:
            self.handles.append(blocks[layer].mlp.down_proj.register_forward_pre_hook(self._mlp_hook(layer)))
        for layer in self.heads:
            hd = blocks[layer].attn.head_dim
            self.handles.append(blocks[layer].attn.o_proj.register_forward_pre_hook(self._head_hook(layer, hd)))
        for kind, layer in self.sub:
            if kind == "mlp":
                h = blocks[layer].mlp.register_forward_hook(lambda m, a, out: torch.zeros_like(out))
            else:
                h = blocks[layer].attn.register_forward_hook(lambda m, a, out: (torch.zeros_like(out[0]), out[1]))
            self.handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


@torch.no_grad()
def collect_means(model, prompts: list[list[int]], batch: int = 32) -> dict:
    """Mean down_proj input (per MLP unit) and o_proj input (per head column) over the real tokens of prompts."""
    _check_plain(model)
    sums, n = {"mlp": {}, "heads": {}}, [0]
    mask_box = {}

    def grab(kind, layer):
        def pre(mod, args):
            x = args[0][mask_box["m"]].double().sum(0)
            sums[kind][layer] = sums[kind].get(layer, 0) + x
        return pre
    hs = []
    for i, b in enumerate(model.blocks):
        hs.append(b.mlp.down_proj.register_forward_pre_hook(grab("mlp", i)))
        hs.append(b.attn.o_proj.register_forward_pre_hook(grab("heads", i)))
    try:
        for lo in range(0, len(prompts), batch):
            ps = prompts[lo:lo + batch]
            T = max(len(p) for p in ps)
            idx = torch.zeros(len(ps), T, dtype=torch.long)
            m = torch.zeros(len(ps), T, dtype=torch.bool)
            for i, p in enumerate(ps):
                idx[i, :len(p)], m[i, :len(p)] = torch.tensor(p), True
            mask_box["m"] = m
            model(idx)
            n[0] += int(m.sum())
    finally:
        for h in hs:
            h.remove()
    return {k: {layer: (v / n[0]).float() for layer, v in d.items()} for k, d in sums.items()}


def prune_mlp(model, units: dict) -> dict:
    """Remove SwiGLU units in place: {layer: [units]} -> new smaller modules (trainable). Returns kept indices."""
    _check_plain(model)
    kept = {}
    for layer, drop in units.items():
        old = model.blocks[layer].mlp
        keep = [j for j in range(old.gate_proj.out_features) if j not in set(drop)]
        new = SwiGLU(old.gate_proj.in_features, len(keep)).to(old.gate_proj.weight.device,
                                                              old.gate_proj.weight.dtype)
        with torch.no_grad():
            new.gate_proj.weight.copy_(old.gate_proj.weight[keep])
            new.up_proj.weight.copy_(old.up_proj.weight[keep])
            new.down_proj.weight.copy_(old.down_proj.weight[:, keep])
        model.blocks[layer].mlp = new
        kept[layer] = keep
    return kept


@torch.no_grad()
def reset_units(model, units: dict, seed: int) -> None:
    """K3 reset (K3 notes RESET AND REGROW): for each layer ascending and unit ascending, draw the W_gate row then
    the W_up row from N(0, init_std) with torch.Generator seeded `seed` (7700 + s); zero the W_down column."""
    _check_plain(model)
    g = torch.Generator().manual_seed(int(seed))
    std = model.cfg.init_std
    for layer in sorted(units):
        mlp = model.blocks[layer].mlp
        for j in sorted(set(int(u) for u in units[layer])):
            for w in (mlp.gate_proj.weight, mlp.up_proj.weight):
                w[j] = (torch.randn(w.shape[1], generator=g) * std).to(w)
            mlp.down_proj.weight[:, j] = 0.0


def _run_capture(model, idx):
    outs = []
    hs = [model.tok_emb.register_forward_hook(lambda m, a, o: outs.append(o.detach().clone()))]
    hs += [b.register_forward_hook(lambda m, a, o: outs.append(o[0].detach().clone())) for b in model.blocks]
    try:
        logits, _ = model(idx)
    finally:
        for h in hs:
            h.remove()
    return logits, outs


@torch.no_grad()
def trace(model, clean: list[int], corrupt: list[int], v_true: int, v_other: int) -> np.ndarray:
    """(n_layers + 1, T) restoration scores; row 0 is the embedding output, row i the output of block i - 1.
    score = (LD_restored - LD_corrupt) / (LD_clean - LD_corrupt), LD = logit(v_true) - logit(v_other) at the last
    position (symmetric token replacement and logit difference, Zhang and Nanda 2309.16042)."""
    _check_plain(model)
    assert len(clean) == len(corrupt)
    ci, xi = torch.tensor([clean]), torch.tensor([corrupt])
    lc, clean_acts = _run_capture(model, ci)
    lx, _ = model(xi)
    ld = lambda lg: float(lg[0, -1, v_true] - lg[0, -1, v_other])       # noqa: E731
    ld_c, ld_x = ld(lc), ld(lx)
    mods = [model.tok_emb] + list(model.blocks)
    out = np.zeros((len(mods), len(clean)))
    for li, mod in enumerate(mods):
        for p in range(len(clean)):
            def hook(m, a, o, li=li, p=p):
                if li == 0:
                    o = o.clone()
                    o[:, p] = clean_acts[0][:, p]
                    return o
                x = o[0].clone()
                x[:, p] = clean_acts[li][:, p]
                return (x, o[1])
            h = mod.register_forward_hook(hook)
            try:
                out[li, p] = (ld(model(xi)[0]) - ld_x) / (ld_c - ld_x) if ld_c != ld_x else float("nan")
            finally:
                h.remove()
    return out


def unit_effects(model, metric, layers=None, mode="zero", means=None) -> np.ndarray:
    """M3: metric(model) -> 1-D array of scores (e.g. [F, S]); returns (n_units_total, len) of the change when each
    MLP unit alone is ablated, rows ordered layer-major (row = layer * hidden + unit)."""
    base = np.asarray(metric(model), dtype=float)
    rows = []
    for layer in (layers if layers is not None else range(len(model.blocks))):
        for j in range(model.blocks[layer].mlp.gate_proj.out_features):
            with Ablation(model, mlp={layer: [j]}, mode=mode, means=means):
                rows.append(np.asarray(metric(model), dtype=float) - base)
    return np.stack(rows)


def grad_x_act(model, loss_fn) -> dict:
    """Registered M3 fallback: |sum over tokens of dL/dh * h| per MLP unit (h = the down_proj input)."""
    acts, hs = {}, []

    def keep(layer):
        def pre(mod, args):
            args[0].retain_grad()
            acts[layer] = args[0]
        return pre
    for i, b in enumerate(model.blocks):
        hs.append(b.mlp.down_proj.register_forward_pre_hook(keep(i)))
    try:
        model.zero_grad(set_to_none=True)
        loss_fn(model).backward()
    finally:
        for h in hs:
            h.remove()
    return {i: (a.grad * a).detach().sum(dim=tuple(range(a.dim() - 1))).abs() for i, a in acts.items()}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
