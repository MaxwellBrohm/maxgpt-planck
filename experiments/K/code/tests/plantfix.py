"""Planted-set fixture for K2's masks (K2 notes DEPENDENCIES and K2-e; review R-2). A scrambled tiny harness model
(every unit does something to every prompt) in which chosen SwiGLU units are rewritten to serve one prompt each: the
unit reads the MLP input of that prompt's last position, projected off the MLP inputs of every other position of
every prompt (so it is silent everywhere else at the unablated point), and writes that prompt's gold token. Facts
F1-F4 have one unit each, skills S1-S2 one each, and X is one shared unit that serves fact F5 and skill S3 (their
gold is one token). Skill prompts S4-S7 are served only by the scrambled model (gold = its own argmax, read after
planting). OUT sets each plant's output against the residual: F1, F3 and S1 are saturated, the rest graded.
CASES are the K2 objectives run on it (tests/test_masks.py, tests/mask_sweep.py). Built once per process (cached).
"""
from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F

from kfix import tiny_model

V = 97
CFG = dict(seed=3, vocab=V, n_layers=3, d_model=64, head_dim=32, mlp_hidden=32)
FACTS = {"F1": [10, 11, 12], "F2": [13, 14, 15], "F3": [16, 17, 18], "F4": [19, 20, 21], "F5": [22, 23, 24]}
SKILLS = {f"S{i + 1}": [30 + 3 * i, 31 + 3 * i, 32 + 3 * i] for i in range(7)}
GOLD = {"F1": 50, "F2": 51, "F3": 52, "F4": 53, "F5": 55, "S1": 60, "S2": 61, "S3": 55}
UNIT = {"F1": (2, 5), "F2": (1, 11), "F3": (0, 20), "F4": (2, 27), "S1": (1, 3), "S2": (0, 9), "X": (1, 17)}
SERVES = {"F1": ["F1"], "F2": ["F2"], "F3": ["F3"], "F4": ["F4"], "S1": ["S1"], "S2": ["S2"], "X": ["F5", "S3"]}
FOUR = ("F1", "F2", "F3", "F4")
ALLF = FOUR + ("F5",)
# K2 objectives on the plant: name -> (objective, fact prompts, budget, the plant names it must find)
CASES = {
    "SF1": ("SF", ("F1",), 1, {"F1"}), "SF2": ("SF", ("F1", "F2"), 2, {"F1", "F2"}),
    "SF4": ("SF", FOUR, 4, set(FOUR)),
    "KF1": ("KF", ("F1",), 1, {"F1"}), "KF2": ("KF", ("F1", "F2"), 2, {"F1", "F2"}),
    "KF4": ("KF", FOUR, 4, set(FOUR)),
    "MX-SF": ("SF", ALLF, 4, set(FOUR)), "MX-KF": ("KF", ALLF, 5, set(FOUR) | {"X"}),
    "MX-SS": ("SS", ALLF, 2, {"S1", "S2"}), "MX-KS": ("KS", ALLF, 3, {"S1", "S2", "X"}),
}
ACT = 3.0                    # planted pre-activation on its prompt(s)
# output norm / the residual norm there: 2 keeps CE graded in the gate, 20 saturates it (CE stays near its base
# until the gate is under about 0.1, so the gate's gradient is near zero at z = 1: a confidently known fact)
OUT_DEFAULT = {"F1": 20.0, "F2": 2.0, "F3": 20.0, "F4": 5.0, "S1": 20.0, "S2": 2.0, "X": 5.0}
OUT = dict(OUT_DEFAULT)


def _last_inputs(m, layer, prompts):
    """MLP input (after the block's norm) at every position of every prompt: {name: (T, d)}."""
    out = {}
    for name, p in prompts.items():
        box = {}
        h = m.blocks[layer].mlp.register_forward_pre_hook(lambda mod, a: box.update(x=a[0][0].clone()))
        with torch.no_grad():
            m(torch.tensor([p]))
        h.remove()
        out[name] = box["x"]
    return out


def _resid_norm(m, layer, prompt):
    box = {}
    h = m.blocks[layer].register_forward_hook(lambda mod, a, o: box.update(x=o[0][0, -1].clone()))
    with torch.no_grad():
        m(torch.tensor([prompt]))
    h.remove()
    return float(box["x"].norm())


@functools.lru_cache(maxsize=1)
def build():
    """-> (model, facts, skills, gold, units): units maps a plant name to (layer, unit)."""
    m = tiny_model(**{k: v for k, v in CFG.items() if k not in ("seed", "vocab")}, seed=CFG["seed"], vocab=V)
    prompts = {**FACTS, **SKILLS}
    E = m.lm_head.weight.detach()
    for key in sorted(UNIT, key=lambda k: UNIT[k][0]):           # ascending layers: inputs are read after the
        layer, j = UNIT[key]                                     # plants below them
        xs = _last_inputs(m, layer, prompts)
        own = SERVES[key]
        others = torch.cat([xs[n] if n not in own else xs[n][:-1] for n in prompts])
        Q, _ = torch.linalg.qr(others.T.double())
        perp = [(xs[n][-1].double() - Q @ (Q.T @ xs[n][-1].double())) for n in own]
        # d in the span of the perp parts with d . x_n = 1 for each served prompt n, then scaled to ACT
        P = torch.stack(perp)
        d = P.T @ torch.linalg.solve(P @ P.T, torch.ones(len(own), dtype=torch.float64))
        d = (d * ACT).float()
        w = E[GOLD[own[0]]] / E[GOLD[own[0]]].norm()
        act = F.silu(torch.tensor(ACT)) * ACT
        scale = OUT[key] * max(_resid_norm(m, layer, prompts[n]) for n in own) / float(act)
        with torch.no_grad():
            mlp = m.blocks[layer].mlp
            mlp.gate_proj.weight[j] = d
            mlp.up_proj.weight[j] = d
            mlp.down_proj.weight[:, j] = scale * w
    gold = dict(GOLD)
    with torch.no_grad():
        for n in SKILLS:
            if n not in gold:
                gold[n] = int(m(torch.tensor([SKILLS[n]]))[0][0, -1].argmax())
    for p in m.parameters():
        p.requires_grad_(False)
    return m, FACTS, SKILLS, gold, dict(UNIT)


def batch_ce(names, prompts, gold):
    """-> model -> mean CE over the named prompts at their last position (one batched forward)."""
    ps = [prompts[n] for n in names]
    T = max(len(p) for p in ps)
    idx = torch.tensor([p + [0] * (T - len(p)) for p in ps])
    last = torch.tensor([len(p) - 1 for p in ps])
    tgt = torch.tensor([gold[n] for n in names])

    def ce(model):
        return F.cross_entropy(model(idx)[0][torch.arange(len(ps)), last], tgt)
    return ce


def objectives(facts=("F1", "F2", "F3", "F4", "F5"), skills=tuple(SKILLS)):
    """-> (fact_ce, skill_ce, floors) with K2's normalization: the fact floor is the fact CE with every planted fact
    unit ablated (the no-facts model, F0's role), the skill floor is chance over the vocabulary (log V)."""
    import ablate as A
    m, fa, sk, gold, units = build()
    fc, sc = batch_ce(list(facts), fa, gold), batch_ce(list(skills), sk, gold)
    fact_units = [units[k] for k in units if any(n in facts for n in SERVES[k])]
    spec: dict = {}
    for layer, j in fact_units:
        spec.setdefault(layer, []).append(j)
    with torch.no_grad(), A.Ablation(m, mlp=spec):
        ff = float(fc(m))
    return fc, sc, (ff, math.log(V))


def as_set(units: dict) -> set:
    return {(layer, j) for layer, js in units.items() for j in js}
