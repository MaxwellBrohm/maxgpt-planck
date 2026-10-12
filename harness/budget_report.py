"""Print the budget tables the plan needs, each total checked against the built module.

  python budget_report.py            # PLAN shapes table + vocab grid (head_dim 64)
  python budget_report.py --head-dim 32 --device cpu

Every row's total is recounted by building PlanckLM (count_params.build_and_count), on
'meta' by default so 150M costs no RAM; 'cpu' builds real tensors.
"""
from __future__ import annotations

import argparse

import torch

from budget import Constraints, count_analytic, fmt_m, make_cfg, print_solutions, solve
from count_params import build_and_count

# PLAN.md section 5 shapes table (d, layers, heads, embedding, body, total as printed there).
PLAN_ROWS = [
    ("5M", 5e6, 192, 8, 3, 1.57e6, 3.54e6, 5.11e6),
    ("10M", 10e6, 256, 10, 4, 2.10e6, 7.86e6, 9.96e6),
    ("20M", 20e6, 320, 14, 5, 2.62e6, 17.2e6, 19.8e6),
    ("30M", 30e6, 384, 15, 6, 3.15e6, 26.5e6, 29.7e6),
    ("60M", 60e6, 512, 18, 8, 4.19e6, 56.6e6, 60.8e6),
    ("150M", 150e6, 640, 29, 10, 5.24e6, 142.5e6, 147.8e6),
]
GRID_TARGETS = [3e6, 5e6, 10e6, 20e6, 30e6]
GRID_VOCABS = [4096, 8192, 16384, 32768]


def verified(cfg, device: str) -> int:
    a = count_analytic(cfg)
    m = build_and_count(cfg, device)
    assert a == m, f"analytic {a} != module {m} for {cfg}"
    return m["total"]


def plan_table(k: Constraints, device: str) -> None:
    print("== PLAN.md shapes, counted exactly (nominal SwiGLU 8/3 d rounded to 32) ==")
    print(f"{'size':>5} {'d':>4} {'L':>3} {'mlp':>5} {'plan total':>11} {'exact total':>12} "
          f"{'exact embed':>12} {'exact body':>11} {'vs target':>9}")
    for name, target, d, L, H, _, _, plan_total in PLAN_ROWS:
        cfg = make_cfg(d, L, round(8 / 3 * d / 32) * 32, Constraints(vocab_size=8192))
        assert cfg.n_heads == H
        n = count_analytic(cfg)
        verified(cfg, device)
        print(f"{name:>5} {d:>4} {L:>3} {cfg.mlp_hidden:>5} {fmt_m(plan_total):>11} "
              f"{fmt_m(n['total']):>12} {fmt_m(n['embedding']):>12} {fmt_m(n['body']):>11} "
              f"{n['total'] / target - 1:>+9.2%}")
    print()
    print(f"== Solver at the PLAN targets, vocab 8192 (aspect {k.aspect}, fit_mlp {k.fit_mlp},"
          f" tol {k.tol:.0%}) ==")
    sols = []
    for name, target, d, L, *_ in PLAN_ROWS:
        s = solve(target, Constraints(**{**k.__dict__, "vocab_size": 8192}))
        verified(s.cfg, device)
        same = "same d, L as PLAN" if (s.cfg.d_model, s.cfg.n_layers) == (d, L) else \
            f"PLAN had d={d} L={L}"
        sols.append((s, same))
    print_solutions([s for s, _ in sols])
    for (s, same), row in zip(sols, PLAN_ROWS):
        print(f"  {row[0]:>5}: {same}")


def vocab_grid(k: Constraints, device: str) -> None:
    print()
    print(f"== Vocab grid (head_dim {k.head_dim}, aspect {k.aspect}, fit_mlp {k.fit_mlp},"
          f" tol {k.tol:.0%}); embed share = embedding / total ==")
    for target in GRID_TARGETS:
        rows = []
        for v in GRID_VOCABS:
            kk = Constraints(**{**k.__dict__, "vocab_size": v})
            try:
                s = solve(target, kk)
            except ValueError as e:
                print(f"  {fmt_m(target)} vocab {v}: {e}")
                continue
            verified(s.cfg, device)
            rows.append(s)
        print(f"-- target {fmt_m(target)}")
        print_solutions(rows)
        print("   embed share: " + ", ".join(
            f"{s.cfg.vocab_size}: {s.embedding / s.total:.0%}" for s in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="Planck budget tables")
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--aspect", type=float, default=24.0)
    ap.add_argument("--fit-mlp", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--device", default="meta", choices=["meta", "cpu"])
    a = ap.parse_args()
    torch.set_num_threads(2)
    k = Constraints(head_dim=a.head_dim, aspect=a.aspect, fit_mlp=a.fit_mlp, tol=a.tol)
    plan_table(k, a.device)
    vocab_grid(k, a.device)


if __name__ == "__main__":
    main()
