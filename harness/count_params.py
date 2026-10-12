"""Count parameters by BUILDING the harness module, the ground truth budget.py must match.

Total counts every stored parameter once (tied embedding, shared or looped blocks and
shared W_q/W_k are single tensors, so nn.Module.parameters() already dedups them).
  embedding = token embedding, plus the output head when it is not tied
  body      = everything else (blocks and the final norm)

Usage (CPU by default; 'meta' builds the exact shapes without allocating memory):
  python count_params.py --d 192 --layers 8 --heads 3 --vocab 8192 --mlp 512
"""
from __future__ import annotations

import argparse
import json

import torch

from config import PlanckConfig
from model import build_model


def count_module(model) -> dict:
    # parameters() dedups by tensor identity, so tied, shared and looped weights count once
    # (mutation-tested: remove_duplicate=False makes test_budget.py fail).
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok_emb.weight.numel()
    if model.lm_head.weight is not model.tok_emb.weight:
        emb += model.lm_head.weight.numel()
    return {"total": total, "embedding": emb, "body": total - emb}


def build_and_count(cfg: PlanckConfig, device: str = "cpu") -> dict:
    assert device in ("cpu", "meta"), "counting never touches the GPU"
    model = build_model(cfg, device)
    out = count_module(model)
    del model
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="JSON file with PlanckConfig fields")
    ap.add_argument("--d", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--mlp", type=int)
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--loops", type=int)
    ap.add_argument("--device", default="cpu", choices=["cpu", "meta"])
    a = ap.parse_args()
    torch.set_num_threads(2)
    fields = json.load(open(a.config)) if a.config else {}
    for k, v in (("d_model", a.d), ("n_layers", a.layers), ("n_heads", a.heads),
                 ("n_kv_heads", a.kv_heads), ("head_dim", a.head_dim),
                 ("mlp_hidden", a.mlp), ("vocab_size", a.vocab), ("n_loops", a.loops)):
        if v is not None:
            fields[k] = v
    if "n_heads" in fields and "n_kv_heads" not in fields:
        fields["n_kv_heads"] = fields["n_heads"]
    cfg = PlanckConfig.from_dict(fields)
    c = build_and_count(cfg, a.device)
    print(json.dumps({**c, "config": cfg.to_dict()}, indent=1))


if __name__ == "__main__":
    main()
