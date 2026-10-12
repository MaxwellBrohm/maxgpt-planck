"""Optimizer grouping, NorMuon behaviour, cautious decay, and the WSD schedule. CPU only."""
from __future__ import annotations

import math

import pytest
import torch

import schedule as S
from config import PlanckConfig
from model import build_model
from optim import Muon, make_optimizer, set_lr, split_params


def tiny(**kw):
    base = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16,
                mlp_hidden=64, seq_len=32)
    base.update(kw)
    return PlanckConfig(**base)


@pytest.mark.parametrize("kw", [{}, {"tie_embeddings": False}, {"qk_share": 2, "n_loops": 2}])
def test_groups_cover_every_param_once(kw):
    m = build_model(tiny(**kw))
    parts = split_params(m)
    ids = [id(p) for g in parts.values() for _, p in g]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    assert all(p.dim() == 2 for _, p in parts["matrix"])
    assert all(n.startswith(("tok_emb", "lm_head")) for n, _ in parts["embed"])
    assert len(parts["embed"]) == (1 if m.cfg.tie_embeddings else 2)
    assert all(p.dim() <= 1 for _, p in parts["scalar"])


def test_group_lrs_and_decay():
    m = build_model(tiny())
    opt = make_optimizer(m, {"lr": 1e-2, "embed_lr": 5e-3, "scalar_lr": 2e-3, "weight_decay": 0.1})
    g = {x["name"]: x for x in opt.param_groups}
    assert g["matrix"]["use_muon"] and not g["embed"]["use_muon"] and not g["scalar"]["use_muon"]
    assert g["scalar"]["weight_decay"] == 0.0 and g["embed"]["weight_decay"] == 0.1
    set_lr(opt, 0.5)
    assert (g["matrix"]["lr"], g["embed"]["lr"], g["scalar"]["lr"]) == (5e-3, 2.5e-3, 1e-3)
    assert not any(x["use_muon"] for x in make_optimizer(m, {"kind": "adamw"}).param_groups)


def test_muon_update_is_orthogonal_and_rms_matched():
    """Plain Muon (no NorMuon, no decay): the first update of a square matrix is
    orthogonal up to scale, with RMS 0.2 * lr."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(16, 16))
    opt = Muon([{"params": [p]}], lr=1.0, weight_decay=0.0, normalize=False, cautious=False,
               nesterov=False, momentum=0.0)
    g = torch.randn(16, 16)
    p.grad = g.clone()
    opt.step()
    u = -p.detach()
    assert abs(u.pow(2).mean().sqrt().item() - 0.2) < 1e-3
    s, sg = torch.linalg.svdvals(u), torch.linalg.svdvals(g)
    # NS5's quintic pushes singular values into roughly [0.6, 1.2], not exactly to 1
    assert (s.max() / s.min()).item() < 2.0 < 10 < (sg.max() / sg.min()).item()


def test_cautious_decay_only_where_signs_agree():
    p = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
    opt = Muon([{"params": [p], "use_muon": False}], lr=0.1, weight_decay=0.5, cautious=True)
    p.grad = torch.tensor([[1.0, -1.0], [1.0, -1.0]])   # Adam step ~ sign(grad)
    before = p.detach().clone()
    opt.step()
    u = torch.sign(p.grad)
    step = -0.1 * u
    decayed = (u * before > 0)
    exp = before * (1 - 0.1 * 0.5 * decayed.float()) + step
    assert torch.allclose(p.detach(), exp, atol=1e-4)
    assert decayed.tolist() == [[True, False], [False, True]]


def test_normuon_trains_tiny_model():
    torch.manual_seed(0)
    m = build_model(tiny())
    opt = make_optimizer(m, {"lr": 3e-3})
    x = torch.randint(0, 64, (4, 32))
    losses = []
    for _ in range(30):
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < 0.7 * losses[0]


# ------------------------------ schedule ------------------------------

def test_full_wsd_shape():
    s = S.full(1000, warmup_frac=0.01, decay_frac=0.2)
    assert (s.warmup_steps, s.decay_start, s.decay_steps) == (10, 800, 200)
    assert s.factor(0) == pytest.approx(0.1) and s.factor(9) == 1.0
    assert s.factor(10) == 1.0 and s.factor(799) == 1.0
    assert s.factor(800) == 1.0 and s.factor(900) == pytest.approx(0.5)
    assert s.factor(999) == pytest.approx(1 / 200) and s.factor(1000) == 0.0
    fs = [s.factor(i) for i in range(800, 1000)]
    assert all(a > b for a, b in zip(fs, fs[1:]))             # strictly decreasing, linear
    assert all(abs((a - b) - 1 / 200) < 1e-12 for a, b in zip(fs, fs[1:]))
    assert [s.phase(i) for i in (0, 10, 800)] == ["warmup", "stable", "decay"]


def test_trunk_never_decays_and_branch_shape():
    t = S.trunk(10_000, warmup_steps=100)
    assert all(t.factor(i) == 1.0 for i in (100, 5000, 9999))
    b = S.branch(t, 4000, decay_frac=0.2)
    assert (b.decay_start, b.decay_steps, b.total_steps, b.warmup_steps) == (4000, 1000, 5000, 100)
    # a branch is the same curve as a full run of s / (1 - f) steps with the trunk's warmup
    f = S.full(5000, decay_frac=0.2, warmup_steps=100)
    assert all(math.isclose(b.factor(i), f.factor(i)) for i in range(0, 5000, 7))
    assert S.branch(t, 4000, decay_steps=321).total_steps == 4321
    assert S.branch(t, 10_000, decay_frac=0.5).decay_steps == 10_000   # trunk end is stable


def test_branch_refuses_non_stable_points():
    t = S.trunk(1000, warmup_steps=50)
    with pytest.raises(AssertionError, match="warmup"):
        S.branch(t, 20)
    f = S.full(1000)
    with pytest.raises(AssertionError, match="stable"):
        S.branch(f, 900)
    b = S.branch(t, 500)
    with pytest.raises(AssertionError, match="branch"):
        S.branch(b, 500)


def test_tpp_branch_points():
    pts = S.branch_steps_from_tpp([500, 2000], n_params=5_000_000, batch_tokens=32_768)
    assert pts == [math.ceil(500 * 5e6 / 32768), math.ceil(2000 * 5e6 / 32768)]
    assert S.WSD.from_dict(S.full(100).to_dict()) == S.full(100)


def test_normuon_evens_row_norms():
    """NorMuon divides each row by its running RMS: after one step from a gradient whose
    rows differ 100x in scale, the update's row norms are nearly equal (plain Muon's are not)."""
    torch.manual_seed(0)
    g = torch.randn(8, 32) * torch.logspace(0, 2, 8)[:, None]
    rows = {}
    for norm in (False, True):
        p = torch.nn.Parameter(torch.zeros(8, 32))
        opt = Muon([{"params": [p]}], lr=1.0, weight_decay=0.0, normalize=norm, momentum=0.0,
                   nesterov=False)
        p.grad = g.clone()
        opt.step()
        r = p.detach().norm(dim=1)
        rows[norm] = (r.max() / r.min()).item()
    assert rows[True] < 1.01 < 1.5 < rows[False]
