"""K4b per-group weight decay (K4 notes K4b; SPEC 14), installed by ktrain.py with no harness edit.

  k:
    wd_mult: {mlp: 4.0}     # the MLP matrices' weight decay x 4 (0.1 -> 0.4, cautious as the rest)

split_mlp_wd moves every SwiGLU matrix (blocks.<i>.mlp.gate_proj / up_proj / down_proj .weight) out of the
optimizer's "matrix" group into a "matrix_mlp" group placed right after it: the same keys (base_lr, lr, momentum,
use_muon, ...) with weight_decay times the multiplier. Attention matrices, embeddings and scalars keep their groups
and values. Groups keep their order across a save and a resume (the split is a function of the model alone). It
runs before the first step (the optimizer holds no state yet) and draws nothing from any RNG.
"""
from __future__ import annotations

KEYS = ("mlp",)


def check(wcfg: dict) -> dict:
    bad = sorted(set(wcfg) - set(KEYS))
    if bad:
        raise ValueError(f"k.wd_mult: unknown keys {bad} (known: {list(KEYS)})")
    out = {}
    for k, v in wcfg.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0:
            raise ValueError(f"k.wd_mult.{k} must be a number >= 0, got {v!r}")
        out[k] = float(v)
    return out


def mlp_param_names(model) -> list[str]:
    return [n for n, p in model.named_parameters() if ".mlp." in n and p.dim() == 2]


def split_mlp_wd(opt, model, mult: float) -> dict:
    """-> {"group": "matrix_mlp", "params": [names], "weight_decay": wd}."""
    if opt.state:
        raise RuntimeError("k.wd_mult must split the groups before the first step")
    names = {id(p): n for n, p in model.named_parameters()}
    mlp = set(mlp_param_names(model))
    gi = [i for i, g in enumerate(opt.param_groups) if g.get("name") == "matrix"]
    if len(gi) != 1:
        raise RuntimeError("k.wd_mult: expected exactly one 'matrix' optimizer group")
    g = opt.param_groups[gi[0]]
    moved = [p for p in g["params"] if names.get(id(p)) in mlp]
    if len(moved) != len(mlp) or not moved:
        raise RuntimeError(f"k.wd_mult: {len(moved)} of {len(mlp)} MLP matrices are in the matrix group")
    keep = [p for p in g["params"] if names.get(id(p)) not in mlp]
    new = {**g, "name": "matrix_mlp", "params": moved, "weight_decay": g["weight_decay"] * float(mult)}
    g["params"] = keep
    groups = list(opt.param_groups)
    groups.insert(gi[0] + 1, new)
    if not keep:
        groups.remove(g)
    opt.param_groups[:] = groups
    return {"group": "matrix_mlp", "params": sorted(names[id(p)] for p in moved),
            "weight_decay": new["weight_decay"]}
