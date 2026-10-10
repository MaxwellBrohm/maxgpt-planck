"""T001 mixer tests: hand-computed cases, chunked == naive, causality for every arm, zero-init arms equal base."""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layers  # noqa: E402
from layers import Attention, CausalConv, delta_chunked, delta_naive, selective_mask, stick_breaking  # noqa: E402
from model import ARMS, ToyLM  # noqa: E402

torch.set_default_dtype(torch.float64)


def t(x):
    return torch.tensor(x, dtype=torch.float64)


def test_delta_rule_hand_case():
    """Dk = Dv = 2, one head. Worked by hand (M is Dk x Dv; o_t = M_t^T q_t):
    t1 k=[1,0] v=[2,3] beta=1 alpha=1        -> M1 = [[2,3],[0,0]],             o1 (q=[1,0]) = [2,3]
    t2 k=[1,0] v=[5,7] beta=0.5 alpha=0.8    -> M2 = 0.8*(M1 - 0.5*[[2,3],[0,0]]) + 0.5*[[5,7],[0,0]]
                                                   = [[0.8,1.2],[0,0]] + [[2.5,3.5],[0,0]] = [[3.3,4.7],[0,0]]
    t3 k=[0,1] v=[1,1] beta=1 alpha=1        -> orthogonal key: M3 = [[3.3,4.7],[1,1]], o3 (q=[1,0]) = [3.3,4.7]
    t4 k=[.6,.8] v=[1,1] beta=1 alpha=1      -> k^T M3 = [2.78,3.62]; M4 = M3 - k (x) [2.78,3.62] + k (x) [1,1]
                                                   = [[2.232,3.128],[-0.424,-1.096]], o4 (q=k) = [1,1] (exact overwrite)"""
    k = t([[1, 0], [1, 0], [0, 1], [0.6, 0.8]])
    v = t([[2, 3], [5, 7], [1, 1], [1, 1]])
    q = t([[1, 0], [1, 0], [1, 0], [0.6, 0.8]])
    la = torch.log(t([1.0, 0.8, 1.0, 1.0]))
    be = t([1.0, 0.5, 1.0, 1.0])
    want = t([[2, 3], [3.3, 4.7], [3.3, 4.7], [1, 1]])
    for fn in (delta_naive, lambda *a: delta_chunked(*a, chunk=3)):
        o = fn(q[None, None], k[None, None], v[None, None], la[None, None], be[None, None])[0, 0]
        assert torch.allclose(o, want, atol=1e-12), o
    q2 = t([[0, 0], [0, 0], [0, 0], [1, 0]])          # read row 0 of M4 directly
    o = delta_naive(q2[None, None], k[None, None], v[None, None], la[None, None], be[None, None])[0, 0, 3]
    assert torch.allclose(o, t([2.232, 3.128]), atol=1e-12)


def rand_delta(seed, L=37, B=2, H=3, Dk=8, Dv=5):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, H, L, Dk, generator=g), dim=-1).requires_grad_()
    k = F.normalize(torch.randn(B, H, L, Dk, generator=g), dim=-1).requires_grad_()
    v = torch.randn(B, H, L, Dv, generator=g).requires_grad_()
    la = F.logsigmoid(torch.randn(B, H, L, generator=g) + 1.5).requires_grad_()
    be = torch.sigmoid(torch.randn(B, H, L, generator=g)).requires_grad_()
    return q, k, v, la, be


@pytest.mark.parametrize("chunk", [1, 4, 16, 36, 37, 64])
def test_chunked_equals_naive_values_and_grads(chunk):
    ins = rand_delta(chunk)
    w = torch.randn(2, 3, 37, 5, generator=torch.Generator().manual_seed(99))
    gn = torch.autograd.grad((delta_naive(*ins) * w).sum(), ins)
    gc = torch.autograd.grad((delta_chunked(*ins, chunk=chunk) * w).sum(), ins)
    assert torch.allclose(delta_chunked(*ins, chunk=chunk), delta_naive(*ins), atol=1e-10)
    for a, b in zip(gn, gc):
        assert torch.allclose(a, b, atol=1e-9)


def test_stick_breaking_hand_case():
    """One query row i = 2 with logits z = [0, ln 3, ln(1/3)] -> sigmoids b = [1/2, 3/4, 1/4].
    A[2,2] = 1/4; A[2,1] = 3/4 * (1 - 1/4) = 9/16; A[2,0] = 1/2 * (3/4) * (1/4) = 3/32."""
    z = t([[0.0, 9.0, 9.0], [0.0, 0.0, 9.0], [0.0, math.log(3), math.log(1 / 3)]])
    causal = torch.ones(3, 3, dtype=torch.bool).tril()
    A = stick_breaking(z, causal)
    assert torch.allclose(A[2], t([3 / 32, 9 / 16, 1 / 4]), atol=1e-12)
    assert torch.allclose(A[1], t([0.5 * 0.5, 0.5, 0.0]), atol=1e-12)       # row 1: b = [1/2, 1/2], future zero
    assert A[0, 0] == 0.5 and A[0, 1] == 0 and A[0, 2] == 0


def test_selective_mask_hand_case():
    """s0 (one batch) causal logits; S = ReLU, BOS column and diagonal zeroed, shifted one query down, summed.
    rows: q0 [5], q1 [1, 2], q2 [3, 4, -1], q3 [0, 6, -2, 7]. S (after zeroing col 0 and the diagonal):
    q1 [0,0], q2 [0,4,0], q3 [0,6,0,0]. F[i] = sum of S rows < i: F1 = 0, F2 = S1 = 0, F3 = S1 + S2 = [0,4,0,0]."""
    s0 = t([[5, 0, 0, 0], [1, 2, 0, 0], [3, 4, -1, 0], [0, 6, -2, 7]])[None]
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    Fm = selective_mask(s0, causal)[0]
    assert torch.allclose(Fm, t([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 0, 0]]))


def scrambled(arm, seed=0):
    torch.manual_seed(seed)
    m = ToyLM(arm)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if any(s in n for s in ("canon_", "alpha", ".fw", ".ap", ".fb", "gates")):
                p.add_(0.5 * torch.randn(p.shape, generator=g))    # make every zero-init path active
    return m


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_causal_every_arm(arm):
    m = scrambled(arm)
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 228, (2, 40), generator=g)
    base = m(x)
    for pos in (1, 17, 39):
        x2 = x.clone()
        x2[:, pos] = (x2[:, pos] + 7) % 228
        out = m(x2)
        assert torch.allclose(out[:, :pos], base[:, :pos], atol=1e-10), f"{arm}: change at {pos} reached earlier"
        assert not torch.allclose(out[:, pos:], base[:, pos:])


def test_forget_gate_bias_matches_loop():
    torch.manual_seed(0)
    a = Attention(16, 2, "fg", fg_len=10)
    with torch.no_grad():
        a.fw.normal_(0, 1.0)
        a.qkv.weight.zero_()                     # uniform logits: the weights are softmax(bias) over j <= i
    x = torch.randn(1, 6, 16)
    logf = F.logsigmoid(x[0] @ a.fw.t() + a.fb).detach()     # (L, H)
    with torch.no_grad():
        a.o.weight.copy_(torch.eye(16))
        a.qkv.weight[32:].copy_(torch.eye(16))   # v = x, o = identity
    y = a(x)[0]
    for h in range(2):
        for i in range(6):
            bias = t([sum(logf[l, h] for l in range(j + 1, i + 1)) for j in range(i + 1)])
            w = torch.softmax(bias, 0)
            assert torch.allclose(y[i, 8 * h:8 * h + 8], (w[:, None] * x[0, :i + 1, 8 * h:8 * h + 8]).sum(0), atol=1e-10)


def test_alibi_slopes_and_sign():
    a = Attention(16, 2, "alibi")
    assert torch.allclose(F.softplus(a.ap), t([2 ** -4, 2 ** -8]))
    with torch.no_grad():
        a.qkv.weight.zero_()
        a.qkv.weight[32:].copy_(torch.eye(16))
        a.o.weight.copy_(torch.eye(16))
    x = torch.randn(1, 5, 16)
    y = a(x)[0]
    w = torch.softmax(-(2 ** -4) * t([4, 3, 2, 1, 0]), 0)      # head 0, query 4: nearer keys weigh more
    assert torch.allclose(y[4, :8], (w[:, None] * x[0, :, :8]).sum(0), atol=1e-10)


def test_conv_matches_loop_and_smear_form():
    c = CausalConv(3, 4)
    with torch.no_grad():
        c.weight.copy_(torch.arange(12.0).view(3, 4))
    x = torch.randn(2, 9, 3)
    want = torch.stack([sum(c.weight[:, j] * x[:, s - j] for j in range(4) if s - j >= 0) for s in range(9)], 1)
    assert torch.allclose(c(x), want, atol=1e-12)
    k = torch.randn(1, 2, 5, 4)
    assert torch.allclose(layers.prev_token(k)[:, :, 1:], k[:, :, :-1]) and (layers.prev_token(k)[:, :, 0] == 0).all()


@pytest.mark.parametrize("arm", ["canon", "smear", "canon1", "smear1"])
def test_zero_init_arms_equal_base_at_init(arm):
    ref = "base1" if arm.endswith("1") else "base"
    torch.manual_seed(3)
    mb = ToyLM(ref)
    torch.manual_seed(3)
    ma = ToyLM(arm)
    pb = dict(mb.named_parameters())
    for n, p in ma.named_parameters():
        if n in pb:
            assert torch.equal(p, pb[n]), n
    x = torch.randint(0, 228, (2, 30))
    assert torch.equal(ma(x), mb(x))
