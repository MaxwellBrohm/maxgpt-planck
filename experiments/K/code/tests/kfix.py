"""Shared fixtures for the K tests: pools, a tiny harness model, the scratch data dir."""
from __future__ import annotations

import functools
import math

import torch

import pools as P


@functools.lru_cache(maxsize=1)
def pools():
    return P.get()


@functools.lru_cache(maxsize=1)
def lex():
    return P.Lex(pools())


def tiny_model(seed: int = 0, vocab: int = 97, scramble: bool = True, **kw):
    from config import PlanckConfig
    from model import build_model
    cfg = PlanckConfig(**{**dict(vocab_size=vocab, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, head_dim=16,
                                 mlp_hidden=24, seq_len=64), **kw})
    torch.manual_seed(seed)
    m = build_model(cfg)
    if scramble:                       # every parameter non-trivial (harness testutil.scramble's rule)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in m.parameters():
                r = torch.randn(p.shape, generator=g)
                p.copy_(r / math.sqrt(p.shape[1]) if p.dim() >= 2 else 1.0 + 0.5 * r)
    return m.eval()


def old_questions(turns, final, n_extra, lure_key, r, skip=()):
    """skill._questions before K3 round 4: every stated key of another attribute, holders and the final entity
    included (REVIEW 5a S-1's QFIT: R 0.683, P 0.707). For planted-break tests (test_skill, test_cues)."""
    import items as I
    ok = [j for j in range(len(turns)) if j + 1 >= len(turns) or turns[j + 1][0].kind != "ell"]
    at = sorted(r.choice(ok, min(n_extra, len(ok)), replace=False).tolist()) if n_extra else []
    ex, stated = [], []
    for j, t in enumerate(turns):
        ex.append(("S", t))
        stated += [(s.ent, s.attr) for s in t if s.kind != "def" and (s.ent, s.attr) not in stated]
        pool = [k for k in stated if k[1] != final.attr]
        lure = lure_key in stated and any((s.ent, s.attr) == lure_key for u in turns[j + 1:] for s in u)
        for _ in range(at.count(j)):
            if lure and r.random() < 0.5:
                key = lure_key
            elif pool:
                key = pool[int(r.integers(0, len(pool)))]
            else:
                continue
            ex.append(("Q", I.Qn(key[0], key[1], int(r.integers(0, 2)))))
    return ex + [("Q", final)]
