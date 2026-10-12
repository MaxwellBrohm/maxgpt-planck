"""Chunked lm_head + loss (chunked_ce.py, train.ce_chunk_rows) against the reference loss path.

CPU fp32: the loss and every parameter gradient equal the reference path's to float
precision, for several arms, both reductions, chunk sizes that split rows mid-sequence
(1, 7, 64 = one row, larger than the batch) and -100 masks; a batch with no supervised
token gives loss 0 and all-zero gradients, like the reference. CPU bf16 autocast: the
chunked path is as close to the fp32 answer as the reference bf16 path is, and differs from
the reference bf16 path by under 0.2x the reference's own bf16 error (measured 0.04x). Trainer,
grad_accum 2, packed chat rows (assistant-only targets) with unequal supervised counts per
micro-batch: same n_sup, loss, gnorm, gradients and updated weights. Leak: the loss of one
supervised token has an exactly-zero gradient at every embedding position it may not see
(testutil.allowed_matrix) and a nonzero one everywhere it may, with chunks straddling rows.
The CUDA versions are in test_chunked_ce_cuda.py; with varlen and compile: test_speed3_cuda.py.
The arms include the screen flags (canon_ac, forget_gate, smear_key): the chunked loss replaces only
what follows the final norm.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch

import schedule as S
from model import build_model
from optim import make_optimizer
from testutil import ARMS, allowed_matrix, scramble, tiny
from trainer import Trainer

BF16 = lambda: torch.autocast("cpu", dtype=torch.bfloat16)  # noqa: E731


def batch(B=3, T=64, V=97, seed=0, masked=0.4):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, V, (B, T), generator=g)
    tgt = torch.randint(0, V, (B, T), generator=g)
    tgt[torch.rand(B, T, generator=g) < masked] = -100
    starts = torch.rand(B, T, generator=g) < 0.1
    starts[:, 0] = True
    return idx, tgt, torch.cumsum(starts.long(), 1)


def run(m, idx, tgt, doc, chunk, reduction="sum", amp=None):
    """-> (logits, loss, {name: grad}) for one forward + backward."""
    m.zero_grad(set_to_none=True)
    kw = {"ce_chunk": chunk} if chunk else {}
    with (amp() if amp else nullcontext()):
        lg, loss = m(idx, tgt, doc, reduction=reduction, **kw)
    loss.backward()
    grads = {n: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
             for n, p in m.named_parameters()}
    return lg, loss.detach(), grads


def close(a, b, rel=2e-4) -> bool:
    """fp32-rounding agreement, scaled by the tensor's norm: a summed gradient (vr_alpha,
    norm gains) differs by up to ~1e-4 of its norm between the two paths, about 2x what
    merely permuting the batch rows does to the reference; a real error is O(1)."""
    return bool((a - b).norm() <= rel * b.norm() + 1e-12)


def model(arm="base_gqa", seed=1):
    torch.manual_seed(seed)
    return scramble(build_model(tiny(**ARMS[arm])), seed).train()


@pytest.mark.parametrize("arm", ["base_gqa", "untied", "loop2_cyclic", "prelude_loop_coda", "kv_tie",
                                 "canon_ac", "forget_gate", "smear_key"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_fp32_loss_and_grads_equal_reference(arm, reduction, chunk):
    m = model(arm)
    idx, tgt, doc = batch()
    lg_r, l_r, g_r = run(m, idx, tgt, doc, 0, reduction)
    lg_c, l_c, g_c = run(m, idx, tgt, doc, chunk, reduction)
    assert lg_r is not None and lg_c is None
    assert l_c.dtype == torch.float32 and float(l_c) == pytest.approx(float(l_r), rel=2e-6)
    for n in g_r:
        assert close(g_c[n], g_r[n]), n
    assert g_r["tok_emb.weight"].abs().sum() > 0


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_no_grad_loss_equals_reference(reduction):
    """Under torch.no_grad (eval) the chunked path computes no gradients and returns the
    reference's loss, in fp32 and in bf16 autocast."""
    m = model("prelude_loop_coda")
    idx, tgt, doc = batch()
    for amp in (None, BF16):
        with torch.no_grad(), (amp() if amp else nullcontext()):
            _, l_r = m(idx, tgt, doc, reduction=reduction)
            lg, l_c = m(idx, tgt, doc, reduction=reduction, ce_chunk=7)
        assert lg is None and not l_c.requires_grad
        assert float(l_c) == pytest.approx(float(l_r), rel=1e-5)     # measured 0 on CPU


def test_no_supervised_token_gives_zero_loss_and_zero_grads():
    m = model()
    idx, _, doc = batch()
    tgt = torch.full_like(idx, -100)
    for reduction in ("sum", "mean"):
        _, l_r, g_r = run(m, idx, tgt, doc, 0, reduction)
        _, l_c, g_c = run(m, idx, tgt, doc, 5, reduction)
        assert float(l_r) == 0.0 and float(l_c) == 0.0
        assert all(torch.equal(g_c[n], g_r[n]) and not g_c[n].any() for n in g_r)


def test_masked_targets_carry_no_loss():
    """Changing the vocabulary id under a -100 target never matters, and turning one of
    them on changes the loss by exactly that token's cross-entropy."""
    m = model()
    idx, tgt, doc = batch()
    _, l0, _ = run(m, idx, tgt, doc, 7)
    off = (tgt == -100).nonzero()[0].tolist()
    tgt2 = tgt.clone()
    tgt2[off[0], off[1]] = 5
    _, l1, _ = run(m, idx, tgt2, doc, 7)
    with torch.no_grad():
        lg, _ = m(idx, doc=doc)
        ce = -torch.log_softmax(lg[off[0], off[1]].float(), -1)[5]
    assert float(l1 - l0) == pytest.approx(float(ce), rel=1e-4)


@pytest.mark.parametrize("chunk", [7, 64, 100])
def test_bf16_autocast_as_close_to_fp32_as_the_reference(chunk):
    m = model("loop2_cyclic")
    idx, tgt, doc = batch(B=4)
    _, l32, g32 = run(m, idx, tgt, doc, 0)
    _, l_r, g_r = run(m, idx, tgt, doc, 0, amp=BF16)
    _, l_c, g_c = run(m, idx, tgt, doc, chunk, amp=BF16)
    assert abs(float(l_c - l32)) <= 1.5 * abs(float(l_r - l32)) + 1e-4 * float(l32)
    for n in g32:
        e_r, e_c = (g_r[n] - g32[n]).norm(), (g_c[n] - g32[n]).norm()
        assert e_c <= 1.5 * e_r + 1e-6 * g32[n].norm(), (n, float(e_c), float(e_r))
        assert (g_c[n] - g_r[n]).norm() <= 0.2 * e_r + 1e-6 * g32[n].norm(), n  # measured <= 0.04


def test_trainer_accum_unequal_supervised_counts_matches_reference(device="cpu"):
    """grad_accum 2 through Trainer.train_step on packed chat rows (assistant-only targets):
    the two micro-batches have different supervised counts; the chunked path gives the same
    n_sup, loss, gnorm, gradients at the optimizer step, and weights after it."""
    from packing import collate_rows, pack_rows
    from test_s3_loss_mask import ListLoader, records
    from testutil import template
    tm = template()
    rows = pack_rows([tm.render(r) for r in records(3, n=16)], 64, 0, np.random.default_rng(0))
    b1, b2 = collate_rows(rows[:len(rows) // 2 - 1]), collate_rows(rows[len(rows) // 2 - 1:])
    n1, n2 = int((b1["tgt"] != -100).sum()), int((b2["tgt"] != -100).sum())
    assert n1 != n2 and n1 > 0 and n2 > 0 and b1["idx"].shape[0] != b2["idx"].shape[0]
    import chunked_ce
    real_loss, calls, outs = chunked_ce.chunked_lm_loss, [], []
    for chunk in (None, 13):               # None: the reference arm is Trainer's default
        mp = pytest.MonkeyPatch()          # count chunked calls: the chunked arm must use them
        mp.setattr(chunked_ce, "chunked_lm_loss", lambda *a, **k: (calls.append(chunk), real_loss(*a, **k))[1])
        m = model("loop2_cyclic", seed=3)
        opt = make_optimizer(m, {"lr": 1e-3})
        seen = {}
        real_step = opt.step

        def step(*a, _m=m, _seen=seen, _real=real_step, **k):
            _seen.update({n: p.grad.clone() for n, p in _m.named_parameters() if p.grad is not None})
            return _real(*a, **k)
        opt.step = step
        tr = Trainer(model=m, optimizer=opt, loader=ListLoader([b1, b2]), sched=S.full(10),
                     device=device, amp=None, cfg={}, out_dir=".", grad_accum=2, grad_clip=1.0,
                     log_every=1, ckpt_every=0, keep_last=1, stable_points=set(), meta={},
                     **({} if chunk is None else {"ce_chunk": chunk}))
        out = tr.train_step()
        mp.undo()
        outs.append((out, seen, {n: p.detach().clone() for n, p in m.named_parameters()}))
    (o_r, g_r, w_r), (o_c, g_c, w_c) = outs
    assert calls == [13, 13]                 # both micro-batches of the chunked arm, none of the other
    assert o_r["n_sup"] == o_c["n_sup"] == n1 + n2
    assert o_c["loss"] == pytest.approx(o_r["loss"], rel=2e-6)
    assert o_c["gnorm"] == pytest.approx(o_r["gnorm"], rel=1e-5)
    assert g_r.keys() == g_c.keys() and len(g_r) > 10
    for n in g_r:
        assert close(g_c[n], g_r[n]), n
        assert close(w_c[n], w_r[n], 1e-5), n


T = 20
DOC = [0] * 4 + [1] + [2] * 6 + [0] * 3 + [3] * 3 + [9] * 3   # test_s3_leak.DOC


@pytest.mark.parametrize("arm", ["base_gqa", "loop2_cyclic", "prelude_loop_coda", "kv_tie"])
@pytest.mark.parametrize("packed", [False, True])
def test_one_token_loss_sees_only_allowed_positions(arm, packed):
    m = model(arm, seed=5)
    V = m.cfg.vocab_size
    g = torch.Generator().manual_seed(4)
    idx = torch.randint(0, V, (2, T), generator=g)
    doc = torch.tensor([DOC, DOC]) if packed else None
    A = allowed_matrix(DOC if packed else None, T)
    x0 = {}
    h = m.tok_emb.register_forward_hook(lambda mod, i, out: (out.retain_grad(), x0.update(x=out))[1])
    try:
        for b, t in [(0, 0), (0, 9), (1, 4), (1, 13), (1, T - 1)]:
            tgt = torch.full((2, T), -100)
            tgt[b, t] = int(idx[b, t]) % V
            run(m, idx, tgt, doc, 3)
            gx = x0["x"].grad.abs().sum(-1)                       # (2, T)
            assert not gx[1 - b].any(), (b, t, "other row")
            for s in range(T):
                if A[t, s]:
                    assert gx[b, s] > 0, (b, t, s, "allowed position has no gradient")
                else:
                    assert gx[b, s] == 0, (b, t, s, "leak")
    finally:
        h.remove()
