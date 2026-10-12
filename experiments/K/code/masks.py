"""K2 M1: HardConcrete gates on MLP units, trained with the model's weights frozen (Chang et al. 2311.09060's best
family; Louizos et al. L0 gates, https://arxiv.org/abs/1712.01312).

A gate z in [0, 1] per SwiGLU unit multiplies the down_proj input (z = 1 keeps the unit). fit() minimizes
    -objective(g_fact, g_skill) + lam * relu(E[dropped] - b) / b
with g = (CE - CE_unablated) / (CE_floor - CE_unablated) per side (K2 notes, range-normalized), E[dropped] the
expected number of closed gates, then hardens to the b units with the lowest gate score; of `restarts` such fits
(gate noise seeds seed, seed + 7919, ...) it keeps the one whose hardened set scores best with its units ablated
(the exact objective, K3 round 3). Objectives: KF g_fact,
SF g_fact - g_skill, KS g_skill, SS g_skill - g_fact. fact_ce and skill_ce are callables model -> scalar CE tensor
(differentiable), built by the caller on SPLIT A items only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

OBJECTIVES = {"KF": (1.0, 0.0), "SF": (1.0, -1.0), "KS": (0.0, 1.0), "SS": (-1.0, 1.0)}


class HardConcrete(nn.Module):
    def __init__(self, n: int, init: float = 3.0, beta: float = 2 / 3, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((n,), float(init)))
        self.beta, self.gamma, self.zeta = beta, gamma, zeta

    def forward(self, sample: bool = True) -> torch.Tensor:
        if sample:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def p_open(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta))


class GatedModel:
    """Attaches one HardConcrete per block to the down_proj input; `gates` sets sampling on or off."""

    def __init__(self, model, layers=None, init: float = 3.0):
        self.model = model
        self.layers = list(layers if layers is not None else range(len(model.blocks)))
        self.gates = {l: HardConcrete(model.blocks[l].mlp.gate_proj.out_features, init) for l in self.layers}
        self.sample, self.handles = True, []

    def __enter__(self):
        for l in self.layers:
            g = self.gates[l]

            def pre(mod, args, g=g):
                return (args[0] * g(self.sample).to(args[0]),)
            self.handles.append(self.model.blocks[l].mlp.down_proj.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def params(self):
        return [g.log_alpha for g in self.gates.values()]

    def expected_dropped(self) -> torch.Tensor:
        return sum((1 - g.p_open()).sum() for g in self.gates.values())

    def ranking(self) -> list[tuple[int, int]]:
        """(layer, unit) from most to least dropped (lowest log_alpha first)."""
        allu = [(float(g.log_alpha[j].detach()), l, j) for l, g in self.gates.items()
                for j in range(len(g.log_alpha))]
        return [(l, j) for _, l, j in sorted(allu)]


def penalty(expected_dropped, budget: int, lam: float, per_budget: bool = True):
    """The budget term: lam x relu(E[dropped] - b), divided by b when per_budget (K2 notes K2-f): one unit over
    budget then costs lam / b, the same share of the budget at the fixture's b (1-5) and K2's (78-781)."""
    return lam * (torch.relu(expected_dropped - budget) / (budget if per_budget else 1))    # as swept: lam x (pen)


def fit(model, fact_ce, skill_ce, objective: str, budget: int, steps: int = 2000, lr: float = 0.05,
        lam: float = 6.0, floors=(None, None), seed: int = 0, layers=None, eps: float = 1e-6,
        init: float = 0.0, per_budget: bool = True, restarts: int = 2) -> dict:
    """-> {"units": {layer: [units]} (the hardened top `budget`), "history": [...], "ce0": (f0, s0), "p_drop":
    {layer: [P(gate closed) per unit]} (the soft mask at the end), "hard": the objective with the hardened set
    ablated, "runs": per restart (hard, units), "restart": the index kept}. Weights stay frozen.
    eps: Adam's epsilon, kept well above float32 round-off gradients (about 1e-9 here), which Adam's default
    1e-8 would otherwise scale up to full-size steps on units that do nothing.
    per_budget: the penalty is lam x relu(E[dropped] - b) / b, so lam prices a unit over budget as a share of the
    budget, the same meaning at the fixture's b (1-5) as at K2's (78-781).
    restarts (K2 notes K2-h, K3 round 3, REVIEW 4b M-1): independent fits (gate noise seeds seed, seed + 7919, ...),
    keeping the one whose hardened set scores best with its units ablated. One run can leave a saturated unit stuck
    open: its gate sees no gain until nearly shut while the budget penalty pushes every gate open (MX-KF seed 98, the
    default floors: F1 at P(closed) 0.001, the unplanted (0, 1) hardened; 4,000 steps or lam 10 do not free it), and
    the stuck set scores lower (3.83 against the planted set's 4.95). A linear penalty warmup was tried and dropped:
    it made MX-KF's single runs stick in 13 of 104 against 2 of 200.
    Defaults (K2 notes K2-f, K3 round 1; restarts K2-h, K3 round 3): 2,000 steps, lam 6 per budget unit, gate init
    0.0 (P(closed) 0.17 at step 0), lr 0.05, 2 restarts. On tests/plantfix.py (10 cases; F1, F3 and S1 saturated) one
    run found every planted set on seeds 0-11 under both floor conventions (240 of 240, the seeds used to choose; max
    other P(closed) 0.2805, min planted 0.566) and 1,199 of 1,200 on seeds 12-31, 60-79 and 80-99, never used to
    choose (planted units 0.438 or more, the saturated S1 on MX-SS seed 74 and under 0.04 at 1,500 steps there, K2
    notes K2-g; the miss MX-KF seed 98, REVIEW 4b); with 2 restarts, 400 of 400 on seeds 160-179, never used (K2-h).
    R-2's values (1,000 steps, lam 1 per unit) missed the saturated skill unit S1 on 4 of 120 fits on seeds 6-11
    (REVIEW 2 R-5); 1,000 steps at lam 6 still missed one (MX-SS seed 6, default floors). At these values init 1.0
    also found 120 of 120 on seeds 0-5 at 2,000 steps (4 misses at 1,000), so init 0.0 is kept from R-2 (K2-e), not
    re-chosen; init 3.0 (gates clamped open) misses MX-SS seed 6. K2's lam is still set on seed 201's SPLIT A, from
    values that pass this fixture."""
    if restarts > 1:
        runs = [fit(model, fact_ce, skill_ce, objective, budget, steps, lr, lam, floors, seed + 7919 * i, layers,
                    eps, init, per_budget, 1) for i in range(restarts)]
        best = max(range(restarts), key=lambda i: runs[i]["hard"])
        return {**runs[best], "runs": [(r["hard"], r["units"]) for r in runs], "restart": best}
    wf, ws = OBJECTIVES[objective]
    torch.manual_seed(seed)
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            f0, s0 = float(fact_ce(model)), float(skill_ce(model))
        ff = floors[0] if floors[0] is not None else f0 + 1.0
        sf = floors[1] if floors[1] is not None else s0 + 1.0
        gm = GatedModel(model, layers, init)
        opt = torch.optim.Adam(gm.params(), lr=lr, eps=eps)
        hist = []
        with gm:
            for step in range(steps):
                gf = (fact_ce(model) - f0) / (ff - f0)
                gs = (skill_ce(model) - s0) / (sf - s0)
                loss = -(wf * gf + ws * gs) + penalty(gm.expected_dropped(), budget, lam, per_budget)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 50 == 0 or step == steps - 1:
                    hist.append({"step": step, "g_fact": float(gf.detach()), "g_skill": float(gs.detach()),
                                 "drop": float(gm.expected_dropped().detach())})
        top = gm.ranking()[:budget]
        units: dict = {}
        for l, j in top:
            units.setdefault(l, []).append(j)
        p_drop = {l: [float(x) for x in 1 - g.p_open().detach()] for l, g in gm.gates.items()}
        units = {l: sorted(v) for l, v in units.items()}
        hard = _hard(model, units, fact_ce, skill_ce, (f0, s0), (ff, sf), (wf, ws))
        return {"units": units, "history": hist, "ce0": (f0, s0), "p_drop": p_drop, "hard": hard,
                "runs": [(hard, units)], "restart": 0}
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)


@torch.no_grad()
def _hard(model, units, fact_ce, skill_ce, ce0, floors, w) -> float:
    """The fit's objective with the hardened set's units zeroed (z 0 on them, 1 elsewhere), no penalty."""
    hs = []
    for layer, js in units.items():
        def pre(mod, a, js=js):
            h = a[0].clone()
            h[..., js] = 0
            return (h,)
        hs.append(model.blocks[layer].mlp.down_proj.register_forward_pre_hook(pre))
    try:
        gf = (float(fact_ce(model)) - ce0[0]) / (floors[0] - ce0[0])
        gs = (float(skill_ce(model)) - ce0[1]) / (floors[1] - ce0[1])
    finally:
        for h in hs:
            h.remove()
    return w[0] * gf + w[1] * gs


def random_matched(units: dict, hidden: int, seed: int, effect=None, bins: int = 5) -> dict:
    """K2 M2 / K3 P-R: a random unit set with the same count per layer as `units`, disjoint from it. With `effect`
    ({layer: per-unit skill effect array}) each chosen unit is matched inside its layer's effect quintile [K2-d]."""
    g = torch.Generator().manual_seed(int(seed))
    out = {}
    for layer in sorted(units):
        chosen = set(int(u) for u in units[layer])
        if effect is None:
            strata = {0: [j for j in range(hidden) if j not in chosen]}
            need = {0: len(chosen)}
        else:
            e = torch.as_tensor(effect[layer], dtype=torch.float64)
            q = torch.quantile(e, torch.linspace(0, 1, bins + 1, dtype=torch.float64)[1:-1])
            b = torch.bucketize(e, q)
            strata = {k: [j for j in range(hidden) if int(b[j]) == k and j not in chosen] for k in range(bins)}
            need = {k: sum(int(b[u]) == k for u in chosen) for k in range(bins)}
        pick = []
        for k, n in need.items():
            pool = strata[k]
            assert n <= len(pool), f"layer {layer} stratum {k}: need {n}, have {len(pool)}"
            pick += [pool[int(i)] for i in torch.randperm(len(pool), generator=g)[:n]]
        out[layer] = sorted(pick)
    return out
