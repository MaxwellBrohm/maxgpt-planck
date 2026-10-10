"""T001 mixer tests, part 2: whole-layer references for gdn and sb, gdn's gate init, RoPE's relative form."""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers import Attention, GatedDelta, delta_naive, rope  # noqa: E402
from model import FG_LEN, ToyLM  # noqa: E402
from task import seq_len  # noqa: E402

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("conv", [True, False])
def test_gated_delta_layer_matches_reference(conv):
    torch.manual_seed(1)
    m = GatedDelta(8, 2, conv=conv, chunk=5)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.5)
    x = torch.randn(2, 13, 8)
    u = x @ m.qkv.weight.t()
    if conv:                                       # depthwise causal kernel 4: sum_j w[:, j] * u[t - j]
        w = m.conv.weight
        u = torch.stack([sum(w[:, j] * u[:, s - j] for j in range(4) if s - j >= 0) for s in range(13)], 1)
    u = u * torch.sigmoid(u)
    q, k, v = (u[..., i * 8:(i + 1) * 8].view(2, 13, 2, 4).transpose(1, 2) for i in range(3))
    q, k = q / q.norm(dim=-1, keepdim=True), k / k.norm(dim=-1, keepdim=True)
    g = x @ m.gates.weight.t() + m.gates.bias
    la = torch.log(torch.sigmoid(g[..., :2])).transpose(1, 2)
    be = torch.sigmoid(g[..., 2:]).transpose(1, 2)
    y = delta_naive(q, k, v, la, be)
    y = y / torch.sqrt((y * y).mean(-1, keepdim=True) + torch.finfo(y.dtype).eps) * m.onorm.weight
    want = y.transpose(1, 2).reshape(2, 13, 8) @ m.o.weight.t()
    assert torch.allclose(m(x), want, atol=1e-9)


def test_gdn_gate_init():
    torch.manual_seed(0)
    m = ToyLM("gdn")
    gd = m.blocks[1].mix
    f = 2.0 ** (-1.0 / FG_LEN)
    assert torch.allclose(torch.sigmoid(gd.gates.bias[:2]), torch.full((2,), f))     # alpha at the fg half-life
    assert torch.equal(gd.gates.bias[2:], torch.zeros(2))                            # beta 0.5
    assert isinstance(m.blocks[0].mix, Attention) and m.blocks[0].mix.kind == "base"
    assert ToyLM("gdn_nc").blocks[1].mix.conv is None and gd.conv is not None


def test_rope_is_relative():
    torch.manual_seed(2)
    q, k = torch.randn(1, 1, 20, 8), torch.randn(1, 1, 20, 8)
    a = rope(q) @ rope(k).transpose(-1, -2)
    q2, k2 = torch.roll(q, 3, 2), torch.roll(k, 3, 2)          # same vectors three positions later
    b = rope(q2) @ rope(k2).transpose(-1, -2)
    assert torch.allclose(a[..., :17, :17], b[..., 3:, 3:], atol=1e-5)          # angles are fp32


def test_base_attention_uses_rope_on_q_and_k():
    torch.manual_seed(3)
    a = Attention(8, 2, "base")
    x = torch.randn(1, 6, 8)
    q, k, v = (x @ a.qkv.weight.t()).view(1, 6, 3, 2, 4).permute(2, 0, 3, 1, 4)
    s = rope(q) @ rope(k).transpose(-1, -2) / 2.0
    s = s.masked_fill(~torch.ones(6, 6, dtype=torch.bool).tril(), float("-inf"))
    want = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(1, 6, 8) @ a.o.weight.t()
    assert torch.allclose(a(x), want, atol=1e-10)


def test_sb_attention_reference_no_rope():
    torch.manual_seed(4)
    a = Attention(8, 2, "sb")
    x = torch.randn(1, 5, 8)
    q, k, v = (x @ a.qkv.weight.t()).view(1, 5, 3, 2, 4).permute(2, 0, 3, 1, 4)
    z = q @ k.transpose(-1, -2) / 2.0
    A = torch.zeros(1, 2, 5, 5)
    for i in range(5):
        for j in range(i + 1):
            A[..., i, j] = torch.sigmoid(z[..., i, j]) * math.prod(
                [1 - torch.sigmoid(z[..., i, kk]) for kk in range(j + 1, i + 1)] or [torch.ones(1, 2)])
    want = (A @ v).transpose(1, 2).reshape(1, 5, 8) @ a.o.weight.t()
    assert torch.allclose(a(x), want, atol=1e-10)


def test_fg_init_half_life():
    assert FG_LEN == seq_len(96)                               # the longest training row
    a = Attention(8, 2, "fg", fg_len=FG_LEN)
    assert torch.equal(a.fw, torch.zeros(2, 8))
    f = torch.sigmoid(a.fb)
    assert torch.allclose(f ** FG_LEN, torch.full((2,), 0.5))
