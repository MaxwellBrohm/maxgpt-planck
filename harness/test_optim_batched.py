"""optim_batched (optim key batched: true) == the per-matrix optimizer in optim.Muon.

CPU fp32: the batched Newton-Schulz uses the per-matrix code's elementwise ops in the same
order, so it is bit-equal for most shapes; bmm vs mm blocking can still move the last bits
(measured up to 6e-7 relative on a 3x192 stack). The per-matrix code itself moves by 1-2e-6
relative when its input is perturbed by 1e-7 relative, so REL = 1e-5 is a tolerance at float
noise, not a loose one. Compared: the cumulative UPDATE of every parameter (p - p0), not p
(p0 would hide an update error), over several steps with random gradients, every group kind
and switch, and state_dicts moved between the two implementations.
CUDA parts (bf16 Newton-Schulz, deterministic algorithms): test_optim_batched_cuda.py.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import budget
from model import build_model
from optim import make_optimizer, set_lr, zeropower_via_newtonschulz5
from optim_batched import zeropower_via_newtonschulz5_batched
from testutil import tiny

REL = 1e-5
SHAPES = [(16, 24), (24, 16), (32, 32), (3, 32), (32, 3), (1, 8), (8, 1), (5, 5), (24, 16)]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize("shape", sorted(set(SHAPES)) + [(192, 512), (512, 192), (3, 192)])
def test_newton_schulz_batched_matches_per_matrix(shape):
    g = torch.Generator().manual_seed(0)
    Ms = [torch.randn(shape, generator=g) for _ in range(3)]
    Ms[1] = Ms[1] * 1e3                              # per-matrix norms, not one shared norm
    ref = [zeropower_via_newtonschulz5(m) for m in Ms]
    tall = shape[0] > shape[1]
    out = zeropower_via_newtonschulz5_batched(torch.stack([m.mT if tall else m for m in Ms]))
    for o, r in zip(out, ref):
        assert rel(o.mT if tall else o, r) < REL


class Bag(nn.Module):
    """Every matrix orientation (wide, tall, square, rows of 1), repeated shapes, a 3D
    matrix-group tensor, an embedding and 0-dim/1-dim scalars, named like the model."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.tok_emb = nn.Embedding(11, 8)
        self.mats = nn.ParameterList([nn.Parameter(torch.randn(s, generator=g) / s[1] ** 0.5)
                                      for s in SHAPES + [(4, 6, 5)]])
        self.gains = nn.ParameterList([nn.Parameter(1 + 0.1 * torch.randn(s, generator=g))
                                       for s in [(8,), (), (2,), (24,)]])


def set_grads(models, step: int, drop: frozenset = frozenset()):
    """Same random gradient on each model's parameter i; None for indices in drop."""
    g = torch.Generator().manual_seed(1000 + step)
    for i, ps in enumerate(zip(*[list(m.parameters()) for m in models])):
        gr = torch.randn(ps[0].shape, generator=g) * (10.0 if i % 3 == 0 else 0.1)
        for p in ps:
            p.grad = None if i in drop else gr.clone()


def run_pair(m_ref, ocfg, steps, drop_at=None, tweak=None):
    m_bat = copy.deepcopy(m_ref)
    p0 = [p.detach().clone() for p in m_ref.parameters()]
    o_ref = make_optimizer(m_ref, {**ocfg, "batched": False})
    o_bat = make_optimizer(m_bat, {**ocfg, "batched": True})
    assert not o_ref.batched and o_bat.batched
    for o in (o_ref, o_bat):
        if tweak:
            tweak(o)
    for s in range(steps):
        set_grads([m_ref, m_bat], s, (drop_at or {}).get(s, frozenset()))
        for o in (o_ref, o_bat):
            set_lr(o, 1.0 - 0.1 * s)
            o.step()
    return m_ref, m_bat, p0, o_ref, o_bat


def assert_same_updates(m_ref, m_bat, p0, tol=REL, max_flips=0):
    """max_flips: coordinates allowed to differ by more than 1e-4 of the update's max, in
    total. Cautious decay is a step function of sign(O * p): where O * p is within float
    noise of 0, the per-matrix code flips too when its input moves by one ulp. Measured on
    the 5M shapes: 3 of 5.0M coordinates, and none with weight_decay 0 or plain decay. The
    rest must still agree to tol."""
    flips = 0
    for (name, a), b, z in zip(m_ref.named_parameters(), m_bat.parameters(), p0):
        d_ref, d_bat = a.detach() - z, b.detach() - z
        if d_ref.norm() == 0:
            assert torch.equal(d_bat, d_ref), name
            continue
        off = (d_bat - d_ref).abs() > 1e-4 * d_ref.abs().max()
        flips += int(off.sum())
        keep = ~off
        assert rel(d_bat[keep], d_ref[keep]) < tol, (name, rel(d_bat[keep], d_ref[keep]))
    assert flips <= max_flips, flips


def assert_same_state(o_ref, o_bat, tol=REL):
    for a, b in zip(o_ref.param_groups, o_bat.param_groups):
        for pa, pb in zip(a["params"], b["params"]):
            sa, sb = o_ref.state[pa], o_bat.state[pb]
            assert sorted(sa) == sorted(sb)
            for k in sa:
                if torch.is_tensor(sa[k]):
                    assert sa[k].shape == sb[k].shape and rel(sb[k], sa[k]) < tol, k
                else:
                    assert sa[k] == sb[k], k


OCFGS = [
    {"kind": "normuon", "lr": 3e-2, "weight_decay": 0.1, "embed_wd": 0.05},
    {"kind": "normuon", "lr": 3e-2, "weight_decay": 0.0},
    {"kind": "normuon", "lr": 3e-2, "weight_decay": 0.3, "cautious_wd": False},
    {"kind": "muon", "lr": 3e-2, "weight_decay": 0.1},
    {"kind": "adamw", "lr": 3e-2, "weight_decay": 0.1},
]


@pytest.mark.parametrize("ocfg", OCFGS)
def test_bag_updates_match(ocfg):
    m_ref, m_bat, p0, o_ref, o_bat = run_pair(Bag(), ocfg, steps=6)
    assert_same_updates(m_ref, m_bat, p0)
    assert_same_state(o_ref, o_bat)


def test_bag_without_nesterov_and_with_missing_grads():
    """nesterov off; some parameters without a gradient on some steps (skipped, and AdamW
    step counts then differ between parameters of one group)."""
    def no_nesterov(o):
        for g in o.param_groups:
            g["nesterov"] = False
    drop = {1: frozenset({0, 2, 11}), 3: frozenset({4, 13, 14})}
    m_ref, m_bat, p0, o_ref, o_bat = run_pair(Bag(1), OCFGS[0], steps=5, drop_at=drop,
                                              tweak=no_nesterov)
    assert_same_updates(m_ref, m_bat, p0)
    assert_same_state(o_ref, o_bat)
    steps = {o_ref.state[p]["step"] for p in o_ref.param_groups[2]["params"]}
    assert len(steps) > 1, "the fixture should give AdamW params different step counts"


@pytest.mark.parametrize("kw", [{}, {"tie_embeddings": False, "n_heads": 3, "n_kv_heads": 3},
                                {"n_loops": 2, "qk_share": 2, "kv_tie": True},
                                {"mlp_hidden": 0, "n_prelude": 1, "n_coda": 1}])
def test_tiny_model_updates_match(kw):
    m_ref, m_bat, p0, o_ref, o_bat = run_pair(build_model(tiny(**kw)), OCFGS[0], steps=4)
    assert_same_updates(m_ref, m_bat, p0)


@pytest.mark.parametrize("target", [5e6, 30e6])
@pytest.mark.parametrize("oi", [0, 1])
def test_curve_shapes_update_match(target, oi):
    """The real shapes of the 5M and 30M curve points (budget.solve, as bench_micro), with
    cautious decay (a few sign flips allowed, see assert_same_updates) and without decay."""
    import argparse
    ap = argparse.ArgumentParser()
    budget.add_constraint_args(ap)
    cfg = budget.solve(target, budget.constraints_from_args(ap.parse_args([]))).cfg
    torch.manual_seed(0)
    m_ref, m_bat, p0, _, _ = run_pair(build_model(cfg.replace(seq_len=64)), OCFGS[oi], steps=2)
    n = sum(p.numel() for p in m_ref.parameters())
    assert_same_updates(m_ref, m_bat, p0, max_flips=int(2e-6 * n) if oi == 0 else 0)


@pytest.mark.parametrize("first", [False, True])
def test_state_dict_moves_between_implementations(first):
    """Train 3 steps with one implementation, load its state_dict into the other, go on 3
    steps with both: the same updates (checkpoints resume across the flag either way)."""
    m_a = Bag(2)
    o_a = make_optimizer(m_a, {**OCFGS[0], "batched": first})
    for s in range(3):
        set_grads([m_a], s)
        o_a.step()
    m_b = copy.deepcopy(m_a)
    o_b = make_optimizer(m_b, {**OCFGS[0], "batched": not first})
    o_b.load_state_dict(copy.deepcopy(o_a.state_dict()))
    p0 = [p.detach().clone() for p in m_a.parameters()]
    for s in range(3, 6):
        set_grads([m_a, m_b], s)
        o_a.step()
        o_b.step()
    assert_same_updates(m_a, m_b, p0)
    assert_same_state(o_a, o_b)


def test_tiny_lm_loss_curve_matches():
    """40 real training steps (forward, backward, clip, step) on CPU fp32: same losses."""
    curves = []
    for batched in (False, True):
        torch.manual_seed(0)
        m = build_model(tiny())
        opt = make_optimizer(m, {"lr": 1e-2, "batched": batched})
        g = torch.Generator().manual_seed(3)
        data = [torch.randint(0, 97, (4, 33), generator=g) for _ in range(3)]
        losses = []
        for i in range(40):
            x = data[i % 3]
            _, loss = m(x[:, :-1], x[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        curves.append(losses)
    ref, bat = curves
    assert ref[-1] < 0.7 * ref[0]
    assert max(abs(a - b) / a for a, b in zip(ref, bat)) < 1e-5, (ref[-3:], bat[-3:])
