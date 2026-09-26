"""Parameter-budget solver for MaxGPT-Planck.

count_analytic(cfg) counts a PlanckConfig in closed form; test_budget.py proves it equals
the built PyTorch module (count_params.py) on every switch. solve(target, Constraints)
returns the shape whose exact TOTAL (embeddings included, looped or shared weights once)
is within tol of the target, preferring the constraint's depth-over-width aspect.

Aspect = d_model / unique layers (prelude + core + coda). A looped arm is solved on its
unique layers, so "30M looped" gets the same unique shape as "30M dense" and only its
effective depth differs (P-150). Smaller aspect = deeper and thinner.

  python budget.py --target 5e6 --vocab 8192
  python budget.py --target 10e6 --vocab 16384 --mlp-ratio 1.333 --fit-mlp 0.1
Tables for the plan: python budget_report.py
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

from config import PlanckConfig


def count_analytic(cfg: PlanckConfig) -> dict:
    """Exact parameter count of PlanckLM(cfg), mirroring blocks.py and model.py."""
    d, hd = cfg.d_model, cfg.head_dim
    hq, hkv = cfg.n_heads * hd, cfg.n_kv_heads * hd
    vr = cfg.vr_blocks()
    blocks = 0
    for u in range(cfg.n_unique):
        n = d                                   # attn_norm
        if cfg.qk_owner(u) == u:
            n += d * hq + d * hkv               # W_q, W_k (followers reuse the owner's)
        if not cfg.kv_tie:
            n += d * hkv                        # W_v
        n += hq * d                             # W_o
        if cfg.qk_norm:
            n += 2 * hd
        if cfg.attn_gate:
            n += d * cfg.n_heads
        if u in vr:
            n += 3                              # vr_scale + vr_alpha[2]
        if cfg.mlp_hidden > 0:
            n += d + 3 * d * cfg.mlp_hidden     # mlp_norm + SwiGLU
        n += len(cfg.canon_sites()) * d * cfg.canon_kernel   # S004 Canon kernels (0 when off)
        if cfg.forget_gate:
            n += cfg.n_heads * (d + 1)          # S005 forget gate w (H, d) + b (H,)
        if cfg.smear_key:
            n += cfg.n_kv_heads                 # S007 smeared-key alpha, one per KV head
        blocks += n
    emb = cfg.vocab_size * d * (1 if cfg.tie_embeddings else 2)
    body = blocks + d                           # + final norm
    return {"total": emb + body, "embedding": emb, "body": body}


def count_mtp(cfg: PlanckConfig, mtp: int) -> int:
    """S006 (train.mtp; mtp.MTPHead): TRAINING-ONLY parameters of the t+2 aux head, W_mtp (d x d) plus its
    norm gain (d) per head. Never in count_analytic: the deployed model (BASE's total) is unchanged, and the
    run record reports the two apart (runs.jsonl start: n_params and mtp.training_only_params)."""
    return mtp * (cfg.d_model * cfg.d_model + cfg.d_model)


@dataclass
class Constraints:
    vocab_size: int = 8192
    head_dim: int = 64
    mlp_ratio: float = 8 / 3     # SwiGLU hidden / d_model; MLP-light arms use less (P-152)
    mlp_multiple: int = 32       # nominal hidden width is rounded to this
    kv_group: int = 1            # query heads per KV head (1 = full multi-head)
    attn_mult: float = 1.0       # attention width (heads * head_dim) / d_model
    aspect: float = 24.0         # preferred d_model per unique layer (PLAN table: 22-28)
    d_multiple: int = 0          # d_model grid step; 0 = head_dim
    fixed_d: int = 0             # solve layers only (e.g. the vocab grid at a fixed width)
    fixed_layers: int = 0        # solve width only (e.g. a depth sweep, P-151)
    fit_mlp: float = 0.1         # MLP width may move this fraction to land on target (0 = off)
    fit_multiple: int = 8        # hidden width step when fitting
    mlp_weight: float = 1.0      # ranking cost per unit |log(mlp / (mlp_ratio * d))|
    err_weight: float = 5.0      # ranking cost per unit |log(total / target)|
    tol: float = 0.02            # |total / target - 1| allowed (hard limit)
    min_layers: int = 1
    base: dict = field(default_factory=dict)  # other PlanckConfig fields (loops, flags, ...)


@dataclass
class Solution:
    cfg: PlanckConfig
    total: int
    embedding: int
    body: int
    target: float

    @property
    def err(self) -> float:
        return self.total / self.target - 1.0

    @property
    def aspect(self) -> float:
        return self.cfg.d_model / self.cfg.n_unique

    def row(self) -> dict:
        c = self.cfg
        return {"d": c.d_model, "layers": c.n_layers, "unique": c.n_unique, "depth": c.depth,
                "heads": c.n_heads, "kv": c.n_kv_heads, "hd": c.head_dim,
                "mlp": c.mlp_hidden, "mlp_ratio": round(c.mlp_hidden / c.d_model, 3),
                "vocab": c.vocab_size, "embedding": self.embedding, "body": self.body,
                "total": self.total, "err": round(self.err, 4)}


def nominal_mlp(d: int, k: Constraints) -> int:
    if k.mlp_ratio <= 0:
        return 0
    return max(k.mlp_multiple, round(k.mlp_ratio * d / k.mlp_multiple) * k.mlp_multiple)


def make_cfg(d: int, layers: int, mlp: int, k: Constraints) -> PlanckConfig | None:
    heads_f = k.attn_mult * d / k.head_dim
    heads = round(heads_f)
    if heads < 1 or abs(heads - heads_f) > 1e-9 or heads % k.kv_group:
        return None
    fields = dict(k.base)
    fields.update(vocab_size=k.vocab_size, d_model=d, n_layers=layers, n_heads=heads,
                  n_kv_heads=heads // k.kv_group, head_dim=k.head_dim, mlp_hidden=mlp)
    return PlanckConfig.from_dict(fields)


def _fit(cfg: PlanckConfig, target: float, k: Constraints) -> PlanckConfig:
    """Move the MLP width (within fit_mlp of nominal) to bring the total closest to target."""
    f0 = cfg.mlp_hidden
    if k.fit_mlp <= 0 or f0 == 0:
        return cfg
    per_unit = 3 * cfg.d_model * cfg.n_unique           # params per +1 hidden width
    want = f0 + (target - count_analytic(cfg)["total"]) / per_unit
    lo, hi = f0 * (1 - k.fit_mlp), f0 * (1 + k.fit_mlp)
    m = k.fit_multiple
    best = cfg
    for f in {math.floor(want / m) * m, math.ceil(want / m) * m}:
        f = int(min(max(f, math.ceil(lo / m) * m), math.floor(hi / m) * m))
        c = cfg.replace(mlp_hidden=f)
        if abs(count_analytic(c)["total"] - target) < abs(count_analytic(best)["total"] - target):
            best = c
    return best


def candidates(target: float, k: Constraints) -> list[Solution]:
    """Every (d, layers) near the target, fitted, with exact totals (unfiltered)."""
    step = k.d_multiple or k.head_dim
    out: list[Solution] = []
    d_values = [k.fixed_d] if k.fixed_d else range(step, 8192 + 1, step)
    for d in d_values:
        mlp = nominal_mlp(d, k)
        probe = make_cfg(d, 1, mlp, k)
        if probe is None:
            continue
        c1 = count_analytic(probe)["total"]
        if c1 > target * (1 + k.tol) and not k.fixed_d:
            break                                          # widths only grow from here
        per = count_analytic(probe.replace(n_layers=2))["total"] - c1
        if k.fixed_layers:
            layer_values = [k.fixed_layers]
        else:
            est = 1 + round((target - c1) / per)
            layer_values = range(max(k.min_layers, est - 2), max(k.min_layers, est + 2) + 1)
        for L in layer_values:
            cfg = _fit(probe.replace(n_layers=L), target, k)
            n = count_analytic(cfg)
            out.append(Solution(cfg, n["total"], n["embedding"], n["body"], target))
    return out


def _cost(s: Solution, k: Constraints) -> float:
    cost = abs(math.log(s.aspect / k.aspect)) + k.err_weight * abs(math.log(s.total / s.target))
    if k.mlp_ratio > 0 and s.cfg.mlp_hidden > 0:
        cost += k.mlp_weight * abs(math.log(s.cfg.mlp_hidden / (k.mlp_ratio * s.cfg.d_model)))
    return cost


def solve(target: float, k: Constraints | None = None, top: int = 1):
    """Best in-tolerance shapes. Ranking cost = |log(aspect / preferred aspect)| +
    mlp_weight * |log(mlp / (mlp_ratio * d))| + err_weight * |log(total / target)|, so the
    MLP fit only fine-tunes the total and cannot be spent to buy a better aspect, and a
    tight budget match is preferred inside the hard tolerance. Raises
    ValueError (listing the nearest misses) when nothing fits."""
    k = k or Constraints()
    cands = candidates(target, k)
    ok = [s for s in cands if abs(s.err) <= k.tol]
    if not ok:
        near = sorted(cands, key=lambda s: abs(s.err))[:3]
        msg = "; ".join(f"d={s.cfg.d_model} L={s.cfg.n_layers} err={s.err:+.2%}" for s in near)
        raise ValueError(f"no shape within {k.tol:.1%} of {target:,.0f}; nearest: {msg}")
    ok.sort(key=lambda s: (_cost(s, k), abs(s.err)))
    return ok[0] if top == 1 else ok[:top]


def fmt_m(n: float) -> str:
    return f"{n / 1e6:.3f}M"


def print_solutions(sols: list[Solution]) -> None:
    hdr = f"{'d':>5} {'L':>3} {'uniq':>4} {'depth':>5} {'H':>3} {'kv':>3} {'hd':>3} " \
          f"{'mlp':>5} {'ratio':>5} {'vocab':>6} {'embed':>9} {'body':>9} {'total':>9} {'err':>7}"
    print(hdr)
    for s in sols:
        r = s.row()
        print(f"{r['d']:>5} {r['layers']:>3} {r['unique']:>4} {r['depth']:>5} {r['heads']:>3} "
              f"{r['kv']:>3} {r['hd']:>3} {r['mlp']:>5} {r['mlp_ratio']:>5.2f} {r['vocab']:>6} "
              f"{fmt_m(r['embedding']):>9} {fmt_m(r['body']):>9} {fmt_m(r['total']):>9} "
              f"{r['err']:>+7.2%}")


def constraints_from_args(a) -> Constraints:
    base = {}
    for key, val in (("n_loops", a.loops), ("n_prelude", a.prelude), ("n_coda", a.coda),
                     ("loop_order", a.loop_order), ("qk_share", a.qk_share)):
        if val is not None:
            base[key] = val
    if a.kv_tie:
        base["kv_tie"] = True
    if a.untied:
        base["tie_embeddings"] = False
    return Constraints(vocab_size=a.vocab, head_dim=a.head_dim, mlp_ratio=a.mlp_ratio,
                       mlp_multiple=a.mlp_multiple, kv_group=a.kv_group, attn_mult=a.attn_mult,
                       aspect=a.aspect, d_multiple=a.d_multiple, fixed_d=a.fixed_d,
                       fixed_layers=a.fixed_layers, fit_mlp=a.fit_mlp,
                       mlp_weight=a.mlp_weight, tol=a.tol, base=base)


def add_constraint_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--mlp-ratio", type=float, default=8 / 3)
    ap.add_argument("--mlp-multiple", type=int, default=32)
    ap.add_argument("--kv-group", type=int, default=1)
    ap.add_argument("--attn-mult", type=float, default=1.0)
    ap.add_argument("--aspect", type=float, default=24.0)
    ap.add_argument("--d-multiple", type=int, default=0)
    ap.add_argument("--fixed-d", type=int, default=0)
    ap.add_argument("--fixed-layers", type=int, default=0)
    ap.add_argument("--fit-mlp", type=float, default=0.1)
    ap.add_argument("--mlp-weight", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--loops", type=int)
    ap.add_argument("--loop-order", choices=["cyclic", "immediate"])
    ap.add_argument("--prelude", type=int)
    ap.add_argument("--coda", type=int)
    ap.add_argument("--qk-share", type=int)
    ap.add_argument("--kv-tie", action="store_true")
    ap.add_argument("--untied", action="store_true")


def main() -> None:
    ap = argparse.ArgumentParser(description="Planck parameter-budget solver")
    ap.add_argument("--target", type=float, required=True, help="total parameters, e.g. 5e6")
    ap.add_argument("--top", type=int, default=5, help="also list this many alternatives")
    add_constraint_args(ap)
    a = ap.parse_args()
    k = constraints_from_args(a)
    try:
        sols = solve(a.target, k, top=max(1, a.top))
    except ValueError as e:
        raise SystemExit(str(e))
    sols = sols if isinstance(sols, list) else [sols]
    print(f"target {fmt_m(a.target)}, vocab {k.vocab_size}, aspect {k.aspect}, tol {k.tol:.1%}"
          f" (first row = chosen)")
    print_solutions(sols)


if __name__ == "__main__":
    main()
