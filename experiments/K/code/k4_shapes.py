"""K4 shapes (K4 notes, THE ONE MANIPULATION): (n_heads, SwiGLU width) at the BASE total, d 192 and depth 8 fixed,
solved by harness/budget.py (attn_mult = heads x 64 / d, fixed_d, fixed_layers, fit_mlp). No harness change: the
solver already supports an MLP ratio at a fixed total through these keys.
  python k4_shapes.py [--write]   prints the table; --write puts model blocks in ../K4_allocation/configs/
"""
from __future__ import annotations

import os
import sys

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)
import budget as B                                                   # noqa: E402

BASE_TOTAL = 5_010_133
REGISTERED = {"BASE": (3, 488, 5_010_133), "H4": (4, 400, 4_999_381), "H6": (6, 232, 5_014_741),
              "H7": (7, 144, 5_003_989)}                              # K4 notes, count_analytic 2026-09-27


def solve(heads: int):
    k = B.Constraints(vocab_size=8192, head_dim=64, mlp_ratio=488 / 192, attn_mult=heads * 64 / 192, fixed_d=192,
                      fixed_layers=8, fit_mlp=0.9, fit_multiple=8, tol=0.02)
    return B.solve(BASE_TOTAL, k)


def flops_per_token(cfg, ctx: int) -> int:
    """Forward FLOPs per token (critique suggestion, K4): 2 x matmul weights of the blocks and the tied head, plus
    attention scores and mixing, 4 x layers x heads x head_dim x ctx (ctx = keys attended)."""
    n = B.count_analytic(cfg)
    return int(2 * (n["body"] + cfg.vocab_size * cfg.d_model) + 4 * cfg.n_layers * cfg.n_heads * cfg.head_dim * ctx)


def table() -> list[dict]:
    rows = []
    for name, (h, m, total) in REGISTERED.items():
        s = solve(h)
        c = s.cfg
        attn = c.n_layers * (4 * c.d_model * c.n_heads * c.head_dim)
        mlp = c.n_layers * 3 * c.d_model * c.mlp_hidden
        rows.append({"shape": name, "heads": c.n_heads, "mlp_hidden": c.mlp_hidden, "total": s.total,
                     "registered": total, "err": round(s.total / BASE_TOTAL - 1, 5),
                     "m": round(c.mlp_hidden / 192, 3),
                     "r": round(mlp / attn, 3), "flops_ctx128": flops_per_token(c, 128),
                     "flops_ctx2048": flops_per_token(c, 2048), "cfg": c})
    return rows


def write(dirpath: str) -> list[str]:
    os.makedirs(dirpath, exist_ok=True)
    out = []
    for r in table():
        if r["shape"] == "BASE":
            continue
        c = r["cfg"]
        p = os.path.join(dirpath, f"model_{r['shape']}.yaml")
        with open(p, "w") as f:
            f.write(f"# K4 {r['shape']} (k4_shapes.py; harness/budget.py count_analytic {r['total']:,}, "
                    f"{r['err']:+.2%} vs BASE).\n# Merge into a K1 FH run config's model: block.\n")
            f.write("model:\n")
            for key in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "mlp_hidden",
                        "seq_len", "tie_embeddings"):
                v = getattr(c, key)
                f.write(f"  {key}: {str(v).lower() if isinstance(v, bool) else v}\n")
        out.append(p)
    return out


if __name__ == "__main__":
    for r in table():
        print({k: v for k, v in r.items() if k != "cfg"})
    if "--write" in sys.argv:
        print(write(os.path.join(K.HERE, "..", "K4_allocation", "configs")))
