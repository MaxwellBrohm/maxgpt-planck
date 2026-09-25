"""(e) NorMuon decreases loss on toy problems, and every parameter lands in the right group.

Group reference (written from parameter NAMES, not from tensor dims like split_params):
  embed   tok_emb.weight, and lm_head.weight when untied          -> AdamW, embed_lr, embed_wd
  matrix  every *_proj.weight inside blocks                         -> NorMuon, lr, weight_decay
  scalar  every norm gain, vr_scale, vr_alpha, the final norm       -> AdamW, scalar_lr, no decay
"""
from __future__ import annotations

import pytest
import torch

from model import build_model
from optim import make_optimizer, set_lr
from testutil import scramble, tiny

ARMS = [{}, {"tie_embeddings": False}, {"n_loops": 2, "qk_share": 2, "kv_tie": True},
        {"n_layers": 1, "n_loops": 3, "n_prelude": 1, "n_coda": 1, "mlp_hidden": 0}]


def expected_group(name: str) -> str:
    if name in ("tok_emb.weight", "lm_head.weight"):
        return "embed"
    if name.endswith("_proj.weight"):
        return "matrix"
    if name.endswith(("norm.weight", "vr_scale", "vr_alpha")):
        return "scalar"
    raise AssertionError(f"unclassified parameter {name}")


@pytest.mark.parametrize("kw", ARMS)
def test_every_parameter_in_its_group_once(kw):
    m = build_model(tiny(**kw))
    opt = make_optimizer(m, {"lr": 1e-2, "embed_lr": 5e-3, "scalar_lr": 2e-3,
                             "weight_decay": 0.1, "embed_wd": 0.01})
    by_id = {id(p): n for n, p in m.named_parameters()}
    got = {}
    for g in opt.param_groups:
        for p in g["params"]:
            assert id(p) not in got, f"{by_id[id(p)]} is in two groups"
            got[id(p)] = g["name"]
    assert set(got) == set(by_id), "some parameter is in no group"
    for pid, name in by_id.items():
        assert got[pid] == expected_group(name), name
    g = {x["name"]: x for x in opt.param_groups}
    assert (g["matrix"]["base_lr"], g["embed"]["base_lr"], g["scalar"]["base_lr"]) == (1e-2, 5e-3, 2e-3)
    assert (g["matrix"]["weight_decay"], g["embed"]["weight_decay"], g["scalar"]["weight_decay"]) == \
        (0.1, 0.01, 0.0)
    assert g["matrix"]["use_muon"] and not g["embed"]["use_muon"] and not g["scalar"]["use_muon"]
    assert opt.defaults["normalize"] is True                         # NorMuon, not plain Muon
    n_embed = 1 if m.cfg.tie_embeddings else 2
    assert len(g["embed"]["params"]) == n_embed
    if kw.get("qk_share", 1) > 1:                                    # shared W_q/W_k appear once
        names = [by_id[id(p)] for p in g["matrix"]["params"]]
        assert sum(n.endswith("q_proj.weight") for n in names) < m.cfg.n_unique


def batch(seed=0, V=97):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (4, 33), generator=g)
    return x[:, :-1], x[:, 1:]


@pytest.mark.parametrize("kind", ["normuon", "muon", "adamw"])
def test_one_step_decreases_loss_and_moves_every_parameter(kind):
    torch.manual_seed(0)
    m = scramble(build_model(tiny(n_loops=2)), 5).train()
    opt = make_optimizer(m, {"kind": kind, "lr": 3e-3})
    x, y = batch()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    _, l0 = m(x, y)
    l0.backward()
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        _, l1 = m(x, y)
    assert l1 < l0, (kind, float(l0), float(l1))
    stuck = [n for n, p in m.named_parameters() if torch.equal(p, before[n])]
    assert not stuck, f"{kind}: parameters not updated: {stuck}"


def test_normuon_fits_a_toy_regression():
    """Pure matrix problem: min ||W A - B||^2 with W 16x24 in the NorMuon group only."""
    g = torch.Generator().manual_seed(0)
    A, W_true = torch.randn(24, 64, generator=g), torch.randn(16, 24, generator=g) / 5
    B = W_true @ A

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.Linear(24, 16, bias=False)       # a 2D non-embed param
            torch.nn.init.zeros_(self.blocks.weight)

        def named_parameters(self, *a, **k):
            return iter([("blocks.0.attn.q_proj.weight", self.blocks.weight)])

    toy = Toy()
    opt = make_optimizer(toy, {"lr": 0.05, "weight_decay": 0.0})
    assert [x["name"] for x in opt.param_groups] == ["matrix"]
    losses = []
    for step in range(150):
        set_lr(opt, 1.0 - step / 150)
        loss = ((toy.blocks.weight @ A - B) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    assert losses[1] < losses[0] and losses[-1] < 0.02 * losses[0], (losses[0], losses[-1])


def test_normuon_trains_a_tiny_lm_on_a_fixed_batch():
    torch.manual_seed(0)
    m = build_model(tiny())
    opt = make_optimizer(m, {"lr": 1e-2})
    x, y = batch(1)
    losses = []
    for _ in range(60):
        _, loss = m(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.3 * losses[0], (losses[0], losses[-1])


@pytest.mark.parametrize("kind", ["normuon", "muon"])
def test_matrix_group_alone_decreases_loss(kind):
    """embed_lr = scalar_lr = 0: the loss drop must come from the (Nor)Muon matrices alone
    (with them on, AdamW's larger steps on the embedding can hide a broken matrix update)."""
    torch.manual_seed(0)
    m = scramble(build_model(tiny(n_loops=2)), 6).train()
    opt = make_optimizer(m, {"kind": kind, "lr": 3e-3, "embed_lr": 0.0, "scalar_lr": 0.0})
    x, y = batch(2)
    emb = m.tok_emb.weight.detach().clone()
    _, l0 = m(x, y)
    l0.backward()
    opt.step()
    with torch.no_grad():
        _, l1 = m(x, y)
    assert torch.equal(m.tok_emb.weight, emb)                         # AdamW groups frozen
    assert l1 < l0, (kind, float(l0), float(l1))
