"""Tests for the budget solver and the exact counter. CPU only; built models <= ~1M params.

Run: python -m pytest -q test_budget.py   (from the harness directory)
"""
from __future__ import annotations

import math

import pytest
import torch

from budget import Constraints, candidates, count_analytic, make_cfg, solve
from config import PlanckConfig
from count_params import build_and_count, count_module
from model import build_model

torch.set_num_threads(2)

TINY = dict(vocab_size=256, d_model=64, n_layers=3, n_heads=2, n_kv_heads=2, head_dim=32,
            mlp_hidden=160, seq_len=64)

# Every switch that changes the parameter count, one at a time and in combination.
VARIANTS = [
    {},
    {"tie_embeddings": False},
    {"n_heads": 4, "n_kv_heads": 2},                       # GQA
    {"n_heads": 4, "n_kv_heads": 1, "head_dim": 16},      # MQA, attention narrower than d
    {"n_heads": 6, "head_dim": 32},                        # attention wider than d (MLP-light)
    {"qk_norm": False}, {"attn_gate": False}, {"value_residual": False},
    {"norm_scaling": False},
    {"mlp_hidden": 0},                                     # attention-only blocks
    {"mlp_hidden": 40},                                    # MLP-light
    {"n_loops": 2}, {"n_loops": 3, "loop_order": "immediate"},
    {"n_loops": 2, "n_prelude": 1, "n_coda": 1},
    {"n_loops": 2, "n_layers": 1},                         # single core block, looped
    {"qk_share": 2}, {"qk_share": 3, "n_layers": 5},
    {"kv_tie": True}, {"kv_tie": True, "qk_share": 2, "n_heads": 4, "n_kv_heads": 2},
    {"n_loops": 2, "n_prelude": 2, "qk_share": 2, "value_residual": False},
    {"smear_key": True}, {"smear_key": True, "n_heads": 4, "n_kv_heads": 2, "n_loops": 2, "qk_share": 2},  # S007
]


def tiny(**kw) -> PlanckConfig:
    return PlanckConfig(**{**TINY, **kw})


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "base")
def test_analytic_matches_cpu_module(kw):
    cfg = tiny(**kw)
    got = build_and_count(cfg, "cpu")
    assert got["total"] < 1_100_000
    assert count_analytic(cfg) == got


@pytest.mark.parametrize("kw", VARIANTS[:6])
def test_meta_equals_cpu(kw):
    cfg = tiny(**kw)
    assert build_and_count(cfg, "meta") == build_and_count(cfg, "cpu")


def test_count_is_what_numel_says():
    """count_module must equal a plain dedup-by-identity sum over the state dict tensors."""
    cfg = tiny(n_loops=2, qk_share=3, tie_embeddings=True)
    m = build_model(cfg)
    by_ptr = {t.data_ptr(): t.numel() for t in m.state_dict().values()}
    assert count_module(m)["total"] == sum(by_ptr.values())


def test_tied_embedding_counted_once_untied_twice():
    t = count_analytic(tiny())
    u = count_analytic(tiny(tie_embeddings=False))
    assert u["total"] - t["total"] == 256 * 64
    assert t["embedding"] == 256 * 64 and u["embedding"] == 2 * 256 * 64


def test_loops_count_unique_parameters_once():
    """A 3x loop of the core costs only the value-residual scalars that block 0 now needs."""
    dense = build_and_count(tiny(), "cpu")["total"]
    looped = build_and_count(tiny(n_loops=3), "cpu")["total"]
    assert looped - dense == 3                              # block 0 now runs at depth > 0
    cfg = tiny(n_loops=3)
    assert cfg.depth == 9 and cfg.n_unique == 3
    m = build_model(cfg)
    assert len(m.blocks) == 3 and len(m.sched) == 9


def test_schedule_orders():
    c = tiny(n_layers=2, n_loops=2, n_prelude=1, n_coda=1)
    assert c.schedule() == [0, 1, 2, 1, 2, 3]
    c = tiny(n_layers=2, n_loops=2, n_prelude=1, n_coda=1, loop_order="immediate")
    assert c.schedule() == [0, 1, 1, 2, 2, 3]


def test_qk_share_reuses_modules():
    m = build_model(tiny(n_layers=4, qk_share=2))
    a = [b.attn for b in m.blocks]
    assert a[1].q_proj is a[0].q_proj and a[1].k_proj is a[0].k_proj
    assert a[2].q_proj is not a[0].q_proj and a[3].k_proj is a[2].k_proj
    assert a[1].o_proj is not a[0].o_proj


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()) or "base")
def test_forward_backward_runs(kw):
    torch.manual_seed(0)
    cfg = tiny(**kw)
    m = build_model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))       # independent of idx: loss ~ log V
    logits, loss = m(idx, tgt)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert math.isfinite(loss.item()) and abs(loss.item() - math.log(cfg.vocab_size)) < 1.0
    loss.backward()
    assert all(p.grad is not None for n, p in m.named_parameters()
               if "attn_gate_proj" not in n and "vr_" not in n)


# --------------------------------------------------------------------------- solver
PLAN = [(5e6, 192, 8), (10e6, 256, 10), (20e6, 320, 14), (30e6, 384, 15),
        (60e6, 512, 18), (150e6, 640, 29)]


@pytest.mark.parametrize("target,d,L", PLAN)
def test_solver_reproduces_plan_shapes(target, d, L):
    s = solve(target, Constraints(vocab_size=8192))
    assert (s.cfg.d_model, s.cfg.n_layers, s.cfg.n_heads) == (d, L, d // 64)
    assert abs(s.err) <= 0.005                              # MLP fit lands well inside 2%
    assert build_and_count(s.cfg, "meta")["total"] == s.total


@pytest.mark.parametrize("target", [3e6, 5e6, 10e6, 20e6, 30e6])
@pytest.mark.parametrize("vocab", [4096, 8192, 16384, 32768])
def test_vocab_grid_within_tolerance(target, vocab):
    k = Constraints(vocab_size=vocab)
    s = solve(target, k)
    assert abs(s.total / target - 1) <= k.tol
    assert s.cfg.vocab_size == vocab and s.cfg.tie_embeddings
    assert build_and_count(s.cfg, "meta") == count_analytic(s.cfg)
    assert s.embedding == vocab * s.cfg.d_model


def test_no_fit_keeps_nominal_mlp_and_honours_tolerance():
    k = Constraints(fit_mlp=0.0, tol=0.02)
    for target in (10e6, 20e6, 30e6):
        s = solve(target, k)
        assert s.cfg.mlp_hidden == round(8 / 3 * s.cfg.d_model / 32) * 32
        assert abs(s.err) <= 0.02
    with pytest.raises(ValueError, match="no shape within"):
        solve(5.05e6, Constraints(fit_mlp=0.0, tol=0.001, fixed_d=192))


def test_fixed_width_and_fixed_depth():
    s = solve(10e6, Constraints(vocab_size=16384, fixed_d=256))
    assert s.cfg.d_model == 256 and abs(s.err) <= 0.02
    s = solve(20e6, Constraints(fixed_layers=24))
    assert s.cfg.n_layers == 24 and abs(s.err) <= 0.02


def test_mlp_light_arm_goes_deeper_at_equal_total():
    base = solve(5e6, Constraints(fixed_d=192))
    light = solve(5e6, Constraints(fixed_d=192, mlp_ratio=4 / 3))
    assert light.cfg.mlp_hidden < base.cfg.mlp_hidden
    assert light.cfg.n_layers > base.cfg.n_layers
    assert abs(light.err) <= 0.02 and abs(base.err) <= 0.02


def test_looped_arm_matches_dense_unique_shape():
    dense = solve(30e6, Constraints())
    looped = solve(30e6, Constraints(base={"n_loops": 2}))
    assert (looped.cfg.d_model, looped.cfg.n_layers) == (dense.cfg.d_model, dense.cfg.n_layers)
    assert looped.cfg.depth == 2 * dense.cfg.depth
    assert abs(looped.total - dense.total) <= 3 + 3 * looped.cfg.d_model * looped.cfg.n_unique * 8


def test_gqa_and_attention_width_constraints():
    s = solve(20e6, Constraints(kv_group=2))
    assert s.cfg.n_heads % 2 == 0 and s.cfg.n_kv_heads == s.cfg.n_heads // 2
    s = solve(10e6, Constraints(attn_mult=1.5, mlp_ratio=4 / 3))
    assert s.cfg.n_heads * s.cfg.head_dim == int(1.5 * s.cfg.d_model)
    assert build_and_count(s.cfg, "meta")["total"] == s.total


def test_every_candidate_count_is_exact():
    """The solver's internal counts, not just its winners, agree with the module."""
    for c in candidates(5e6, Constraints(vocab_size=4096))[:12]:
        assert build_and_count(c.cfg, "meta")["total"] == c.total


def test_make_cfg_rejects_fractional_heads():
    assert make_cfg(96, 2, 256, Constraints(head_dim=64)) is None
    assert make_cfg(192, 2, 512, Constraints(kv_group=2)) is None   # 3 heads, groups of 2
