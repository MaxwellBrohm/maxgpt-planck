"""NorMuon with cautious weight decay for matrices, AdamW for embeddings, norms and gains.

Copied from Max's MaxGPT-Ultra train/muon.py (Max's AI Model/maxgpt-ultra; the optimizer
validated in his 124M A/B, configs/ab/normuon_arch.yaml) with its references:
  Muon (Jordan et al., 2024): orthogonalized momentum via 5 Newton-Schulz steps.
  Moonlight (Liu et al., 2025): decoupled weight decay and an update RMS of 0.2, so AdamW
    learning rates carry over.
  NorMuon (Li et al., 2025): per-row second-moment normalization after orthogonalizing.
  Cautious weight decay (ICLR 2026): decay only coordinates whose update shares the
    weight's sign.
Planck changes, marked "Planck:": the grouping lives here (make_optimizer), every group
carries its own base_lr so the embedding LR is tuned per vocab (PLAN.md: "LR tuned at 5M
for each arm, with the embedding LR tuned for each vocab"), and the schedule multiplies
base_lr by one factor (set_lr).
"""
from __future__ import annotations

import math

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)   # quintic Newton-Schulz polynomial (Muon paper)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, dtype=None) -> torch.Tensor:
    """From Ultra: G's singular vectors with all singular values ~1."""
    assert G.ndim == 2
    a, b, c = _NS_COEFFS
    X = G.to(dtype) if dtype is not None else G
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """From Ultra. Groups with use_muon=False run AdamW (embeddings, 1D params)."""

    def __init__(self, params, lr=3e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5,
                 normalize=True, cautious=True, beta2=0.95, betas=(0.9, 0.95), eps=1e-8, ns_dtype=None,
                 batched=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, normalize=normalize, cautious=cautious, beta2=beta2,
                        betas=betas, eps=eps, use_muon=True)
        super().__init__(params, defaults)
        self.ns_dtype = ns_dtype
        # Planck: batched=True runs optim_batched (same math, far fewer kernel launches). An
        # attribute, not a group key, so state_dicts and checkpoints are the same either way.
        self.batched = bool(batched)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.batched:
            from optim_batched import adamw_group_batched, muon_group_batched
        for group in self.param_groups:
            if self.batched:
                (muon_group_batched if group["use_muon"] else adamw_group_batched)(self, group)
            elif group["use_muon"]:
                self._muon_group(group)
            else:
                self._adamw_group(group)
        return loss

    def _muon_group(self, group) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim != 2:
                g = g.reshape(g.size(0), -1)
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros_like(g)
                if group["normalize"]:
                    st["row_v"] = torch.zeros(g.size(0), device=g.device, dtype=torch.float32)
            buf = st["momentum"]
            buf.mul_(group["momentum"]).add_(g)
            m = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
            O = zeropower_via_newtonschulz5(m, group["ns_steps"], self.ns_dtype).float()
            if group["normalize"]:                            # NorMuon: per-row second moment
                v = st["row_v"]
                v.mul_(group["beta2"]).add_(O.pow(2).mean(dim=1), alpha=1 - group["beta2"])
                O = O / (v.sqrt().unsqueeze(1) + group["eps"])
            O = O * (0.2 * math.sqrt(O.numel()) / (O.norm() + group["eps"]))  # Moonlight RMS 0.2
            O = O.reshape(p.shape).to(p.dtype)
            if wd:
                if group["cautious"]:
                    p.mul_(1 - lr * wd * (O * p > 0).to(p.dtype))
                else:
                    p.mul_(1 - lr * wd)
            p.add_(O, alpha=-lr)

    def _adamw_group(self, group) -> None:
        lr, wd, (b1, b2), eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
            bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
            u = (st["exp_avg"] / bc1) / ((st["exp_avg_sq"] / bc2).sqrt() + eps)
            if wd:
                if group["cautious"]:
                    p.mul_(1 - lr * wd * (u * p > 0).to(p.dtype))
                else:
                    p.mul_(1 - lr * wd)
            p.add_(u, alpha=-lr)


def split_params(model) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Planck: 'matrix' = 2D block weights (NorMuon); 'embed' = token embedding and an untied
    head (AdamW); 'scalar' = norms, gains, value-residual scalars (AdamW, no decay).
    named_parameters() dedups, so tied and shared weights appear once."""
    out = {"matrix": [], "embed": [], "scalar": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(("tok_emb.", "lm_head.")):
            out["embed"].append((name, p))
        elif p.dim() >= 2:
            out["matrix"].append((name, p))
        else:
            out["scalar"].append((name, p))
    return out


def resolve_batched(value, device_type: str) -> bool:
    """optim key batched: true | false | auto (the default since 2026-09-26). auto = on for
    cuda, where it won its A/B on the 5070 (+11-32% tok/s) and passed parity; off on cpu and
    mps (never measured there, so they keep the per-matrix reference)."""
    if value is None or value == "auto":
        return device_type == "cuda"
    if isinstance(value, bool):
        return value
    raise ValueError(f"optim.batched must be true, false or auto, got {value!r}")


def make_optimizer(model, ocfg: dict, device_type: str = "cpu") -> Muon:
    """ocfg keys (defaults): kind 'normuon' | 'muon' | 'adamw'; lr 3e-3; embed_lr (=lr);
    scalar_lr (=lr); weight_decay 0.1; embed_wd (=weight_decay); cautious_wd True;
    momentum 0.95; betas [0.9, 0.95]; eps 1e-8; batched auto. kind 'adamw' runs every group
    as AdamW (the comparison arm). batched True: optim_batched's foreach/stacked steps, the
    same update to float rounding (test_optim_batched.py) with far fewer kernel launches;
    auto (the default, see resolve_batched) = True on cuda, False on cpu/mps."""
    kind = ocfg.get("kind", "normuon")
    assert kind in ("normuon", "muon", "adamw"), kind
    lr = float(ocfg.get("lr", 3e-3))
    wd = float(ocfg.get("weight_decay", 0.1))
    parts = split_params(model)
    groups = [
        {"name": "matrix", "params": [p for _, p in parts["matrix"]], "base_lr": lr,
         "weight_decay": wd, "use_muon": kind != "adamw"},
        {"name": "embed", "params": [p for _, p in parts["embed"]],
         "base_lr": float(ocfg.get("embed_lr", lr)),
         "weight_decay": float(ocfg.get("embed_wd", wd)), "use_muon": False},
        {"name": "scalar", "params": [p for _, p in parts["scalar"]],
         "base_lr": float(ocfg.get("scalar_lr", lr)), "weight_decay": 0.0, "use_muon": False},
    ]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["lr"] = g["base_lr"]
    ns_dtype = torch.bfloat16 if device_type == "cuda" else None
    return Muon(groups, lr=lr, weight_decay=wd, momentum=float(ocfg.get("momentum", 0.95)),
                normalize=(kind == "normuon"), cautious=bool(ocfg.get("cautious_wd", True)),
                betas=tuple(ocfg.get("betas", (0.9, 0.95))), eps=float(ocfg.get("eps", 1e-8)),
                ns_dtype=ns_dtype, batched=resolve_batched(ocfg.get("batched", "auto"), device_type))


def set_lr(optimizer, factor: float) -> None:
    """Planck: every group's lr = its base_lr times the schedule factor."""
    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * factor
