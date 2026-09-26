"""doc_attn "varlen" (docattn.py, flash varlen for packed rows) against the reference mask path.

CPU: the cu_seqlens segmentation against testutil.allowed_matrix (plain loops), and the whole
plumbing (flatten, GQA, unflatten, the autocast cast) with the flash kernel swapped for an
exact per-document SDPA stand-in, so test_s3_leak's bitwise probes and an fp32 parity check
run on the Mac. CUDA (skipped without it): the real kernel under bf16 autocast: parity with
the mask path judged against an fp32 reference (varlen may not be further from it than the
mask path is), accumulated gradients over 2 micro-batches of unequal supervised counts, the
bitwise leak probes, and the startup self-test (which must also catch a broken segmentation).
"""
from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

import docattn
import selftest
from model import build_model
from test_s3_leak import DOC, T, jacobian_check, perturb_check
from testutil import ARMS, allowed_matrix, scramble, tiny

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="needs CUDA (flash varlen)")
# row boundaries: DOC ends in id 9 and the next row starts with 9 (must still split)
ROWS = [DOC, [9] * 5 + [4] * 15, [3] * T, list(range(T))]


def ref_varlen(q, k, v, cu, max_len, gqa):
    """Exact stand-in for the flash kernel: SDPA is_causal on each [cu[i], cu[i+1]) alone."""
    rep = q.size(1) // k.size(1)
    assert gqa == (rep > 1)
    bounds = cu.tolist()
    assert bounds[0] == 0 and bounds[-1] == q.size(0) and max_len >= max(
        b - a for a, b in zip(bounds[:-1], bounds[1:]))
    out = torch.empty_like(q)
    for a, b in zip(bounds[:-1], bounds[1:]):
        qs, ks, vs = (x[a:b].transpose(0, 1)[None] for x in (q, k, v))       # (1, H, n, hd)
        ks, vs = ks.repeat_interleave(rep, 1), vs.repeat_interleave(rep, 1)
        out[a:b] = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)[0].transpose(0, 1)
    return out


def segments(cu: torch.Tensor, B: int, n: int) -> torch.Tensor:
    seg = torch.full((B * n,), -1, dtype=torch.long)
    for i, (a, b) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        seg[a:b] = i
    return seg.view(B, n)


def test_cu_seqlens_matches_reference_runs():
    doc = torch.tensor(ROWS)
    cu = docattn.cu_seqlens(doc)
    assert cu.dtype == torch.int32 and cu[0] == 0 and cu[-1] == doc.numel()
    assert (cu[1:] > cu[:-1]).all()
    seg = segments(cu, *doc.shape)
    for r, row in enumerate(ROWS):
        A = allowed_matrix(row, T)
        for t in range(T):
            for s in range(T):
                assert (s <= t and seg[r, t] == seg[r, s]) == A[t, s], (r, t, s)
    for r in range(1, len(ROWS)):
        assert seg[r].min() > seg[r - 1].max()                # no document spans two rows


def test_set_doc_attn_validates():
    m = build_model(tiny())
    assert m.doc_attn == "mask"
    with pytest.raises(ValueError):
        docattn.set_doc_attn(m, "flex")
    with pytest.raises(ValueError, match="cuda"):
        docattn.set_doc_attn(m, "varlen", "cpu")
    docattn.set_doc_attn(m, "varlen", "cuda")
    assert m.doc_attn == "varlen"


def test_real_kernel_refuses_cpu_and_doc_none_is_untouched():
    m = build_model(tiny()).eval()
    x = torch.randint(0, 97, (2, 12))
    ref, _ = m(x)
    m.doc_attn = "varlen"
    assert torch.equal(m(x)[0], ref)                          # doc=None: plain causal SDPA
    with pytest.raises(RuntimeError, match="CUDA only"):
        m(x, doc=torch.zeros(2, 12, dtype=torch.long))


def fwd_bwd(m, impl, batches, amp=None):
    """Trainer-style accumulation (sum reduction / supervised tokens of the whole step).
    -> (logits of batch 0, summed loss, {name: grad})."""
    m.doc_attn = impl
    m.zero_grad(set_to_none=True)
    n_sup = sum(int((t != -100).sum()) for _, t, _ in batches)
    first, total = None, 0.0
    for idx, tgt, doc in batches:
        with (amp() if amp else nullcontext()):
            logits, ls = m(idx, tgt, doc, reduction="sum")
        (ls / n_sup).backward()
        total += float(ls.detach()) / n_sup
        first = logits.detach().float() if first is None else first
    return first, total, {n: p.grad.detach().float().clone() for n, p in m.named_parameters()
                          if p.grad is not None}


def make_batches(rows, V, device, seed=0, sup_frac=(0.8, 0.3)):
    """One micro-batch per sup_frac: same docs, fresh tokens, unequal supervised counts."""
    g = torch.Generator().manual_seed(seed)
    doc = torch.tensor(rows) if isinstance(rows, list) else rows
    out = []
    for f in sup_frac:
        idx = torch.randint(0, V, doc.shape, generator=g)
        tgt = torch.randint(0, V, doc.shape, generator=g)
        tgt[torch.rand(doc.shape, generator=g) > f] = -100
        out.append(tuple(x.to(device) for x in (idx, tgt, doc)))
    return out


@pytest.mark.parametrize("arm", list(ARMS))
def test_varlen_equals_mask_cpu_reference_kernel(arm, monkeypatch):
    calls = []
    monkeypatch.setattr(docattn, "_flash_varlen", lambda *a: calls.append(1) or ref_varlen(*a))
    m = scramble(build_model(tiny(**ARMS[arm])), 0)
    batches = make_batches(ROWS[:2], 97, "cpu")
    lm, sm, gm = fwd_bwd(m, "mask", batches)
    assert not calls                                          # the default never takes varlen
    lv, sv, gv = fwd_bwd(m, "varlen", batches)
    assert len(calls) == m.cfg.depth * len(batches)           # every layer, every micro-batch
    assert torch.allclose(lv, lm, atol=1e-5, rtol=1e-5) and abs(sv - sm) < 1e-5 * abs(sm)
    assert gm.keys() == gv.keys()
    for n in gm:
        assert torch.allclose(gv[n], gm[n], atol=1e-6, rtol=1e-4), (arm, n)


@pytest.mark.parametrize("arm", ["base_gqa", "mqa_4h", "loop_share_tie", "prelude_loop_coda"])
def test_varlen_leak_probes_cpu_reference_kernel(arm, monkeypatch):
    monkeypatch.setattr(docattn, "_flash_varlen", ref_varlen)
    m = scramble(build_model(tiny(**ARMS[arm])), 0)
    m.doc_attn = "varlen"
    for grad in (True, False):
        perturb_check(m, DOC, grad)
    jacobian_check(m, DOC)


# ---------------------------------------------------------------- CUDA, the real kernel
BF16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
BIG = dict(vocab_size=8192, d_model=192, n_layers=4, n_heads=3, n_kv_heads=3, head_dim=64,
           mlp_hidden=488, seq_len=2048)                    # the 5M shape, 4 layers


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@cuda_only
@pytest.mark.parametrize("case", ["base_gqa", "mqa_4h", "loop_share_tie", "big"])
def test_cuda_varlen_parity_against_fp32_reference(case):
    if case == "big":
        g = torch.Generator().manual_seed(1)
        starts = torch.rand(2, 2048, generator=g) < 8 / 2048
        starts[:, 0] = True
        rows, kw = torch.cumsum(starts.long(), 1), BIG
    else:
        rows, kw = ROWS, ARMS[case]
    torch.manual_seed(0)
    m = scramble(build_model(tiny(**kw)), 0).cuda()
    batches = make_batches(rows, m.cfg.vocab_size, "cuda")
    lr, sr, gr = fwd_bwd(m, "mask", batches)                   # fp32: the reference
    lm, sm, gm = fwd_bwd(m, "mask", batches, BF16)
    lv, sv, gv = fwd_bwd(m, "varlen", batches, BF16)
    em, ev = rel(lm, lr), rel(lv, lr)
    print(f"\n{case}: logits rel err mask {em:.3g} varlen {ev:.3g}; loss {sr:.5f} {sm:.5f} {sv:.5f}")
    assert ev <= 1.5 * em + 1e-4
    assert abs(sv - sr) <= 1.5 * abs(sm - sr) + 1e-4 * abs(sr)
    flat = lambda d: torch.cat([d[n].flatten() for n in sorted(d)])  # noqa: E731
    gem, gev = rel(flat(gm), flat(gr)), rel(flat(gv), flat(gr))
    ex = {n: rel(gv[n], gr[n]) - 1.5 * rel(gm[n], gr[n]) for n in gr}   # per parameter
    w = max(ex, key=ex.get)
    print(f"{case}: grad rel err mask {gem:.3g} varlen {gev:.3g}; worst per-param excess "
          f"{ex[w]:.3g} ({w}: mask {rel(gm[w], gr[w]):.3g} varlen {rel(gv[w], gr[w]):.3g})")
    assert gev <= 1.5 * gem + 1e-4 and ex[w] <= 1e-2           # 1e-2: bf16 eps is 7.8e-3


def cuda_leak_probes(m, doc_list):
    """test_s3_leak's perturb and jacobian probes on CUDA under bf16 autocast, bitwise."""
    V, n = m.cfg.vocab_size, len(doc_list)
    idx = torch.randint(0, V, (1, n), generator=torch.Generator().manual_seed(3)).cuda()
    doc = torch.tensor([doc_list]).cuda()
    A = allowed_matrix(doc_list, n)
    with torch.no_grad(), BF16():
        base = m(idx, doc=doc)[0][0]
        for j in range(n):
            idx2 = idx.clone()
            idx2[0, j] = (idx[0, j] + 1) % V
            out = m(idx2, doc=doc)[0][0]
            for t in range(n):
                assert A[t, j] or torch.equal(out[t], base[t]), (t, j)
            assert not torch.equal(out[j], base[j]), j
    store = []
    h = m.tok_emb.register_forward_hook(lambda mod, i, o: store.append(o))
    try:
        with BF16():
            out = m(idx, doc=doc)[0][0].float()
    finally:
        h.remove()
    r = torch.randn(out.size(-1), generator=torch.Generator().manual_seed(5)).cuda()
    for t in range(n):
        (gr,) = torch.autograd.grad((out[t] * r).sum(), store[0], retain_graph=True)
        mag = gr[0].abs().amax(-1)
        for s in range(n):
            assert (mag[s] > 0) if A[t, s] else (mag[s] == 0), (t, s, float(mag[s]))


@cuda_only
@pytest.mark.parametrize("impl", ["mask", "varlen"])
@pytest.mark.parametrize("arm", ["base_gqa", "mqa_4h", "loop_share_tie", "prelude_loop_coda"])
def test_cuda_leak_probes(arm, impl):
    m = scramble(build_model(tiny(**ARMS[arm])), 0).cuda()
    m.doc_attn = impl
    cuda_leak_probes(m, DOC)


@cuda_only
def test_cuda_selftest_passes_and_catches_broken_segments(monkeypatch):
    torch.manual_seed(0)
    m = build_model(tiny()).cuda()
    docattn.set_doc_attn(m, "varlen", "cuda")
    out = selftest.run_selftests(m, "cuda", BF16)
    assert len(out) == 12 and all(v < 3e-2 for v in out.values()), out
    one_seq = lambda doc: torch.tensor([0, doc.numel()], dtype=torch.int32,  # noqa: E731
                                       device=doc.device)
    monkeypatch.setattr(docattn, "cu_seqlens", one_seq)
    with pytest.raises(AssertionError, match="leak"):
        selftest.run_selftests(m, "cuda", BF16)
