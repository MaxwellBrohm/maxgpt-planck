"""CUDA checks for chunked_ce.py (skipped without CUDA; they run on the PC).

Parity in bf16 autocast with torch.use_deterministic_algorithms(True) and
CUBLAS_WORKSPACE_CONFIG=:4096:8, on a V=8192 model on packed rows: the reference path run
twice is bitwise equal (so the comparison is not inside run-to-run noise), the chunked path
run twice is bitwise equal, the losses are equal to 0.2x the reference's own bf16 error, and
every chunked gradient is as close to the fp32 run's as the reference bf16 gradient is
(error ratio <= 1.15; measured max 1.04). Unlike on CPU, the two bf16 paths are not near
copies of each other here (their difference is up to 0.9x the reference's bf16 error):
cuBLAS picks other kernels for 300-row chunks than for 1024 rows, so logits round
differently and the flips propagate; with allow_bf16_reduced_precision_reduction (the
default) the reference's split-K weight gradient is also the less accurate of the two
(norm.weight: chunked error 0.39x the reference's). grad_accum 2 with unequal supervised counts through Trainer on CUDA fp32
(same n_sup, loss, gradients, weights). Memory: on a 16k-row, V=8192 head, the chunked loss
path's peak allocation is under a quarter of the reference's and under the size of one
full bf16 logits tensor. torch.compile(model): fp32 equal to the compiled reference, bf16
repeatable and inside the reference paths' error envelope. The eagerly compiled per-chunk
core (FUSE_CUDA) matches the op-by-op one.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before any cuBLAS handle

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from chunked_ce import chunked_lm_loss  # noqa: E402
from config import PlanckConfig  # noqa: E402
from model import build_model  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
BF16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731


@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def grads_of(m, idx, tgt, doc, chunk, amp):
    m.zero_grad(set_to_none=True)
    kw = {"ce_chunk": chunk} if chunk else {}
    with (amp() if amp else torch.autocast("cuda", enabled=False)):
        _, loss = m(idx, tgt, doc, reduction="sum", **kw)
    loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach()), {n: p.grad.detach().clone() for n, p in m.named_parameters()}


def test_bf16_parity_deterministic(deterministic):
    torch.manual_seed(0)
    cfg = PlanckConfig(vocab_size=8192, d_model=128, n_layers=2, n_heads=2, n_kv_heads=1,
                       head_dim=64, mlp_hidden=256, seq_len=256, n_loops=2)
    m = build_model(cfg, "cpu").to("cuda").train()
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 8192, (4, 256), generator=g)
    tgt = torch.randint(0, 8192, (4, 256), generator=g)
    tgt[torch.rand(4, 256, generator=g) < 0.5] = -100
    doc = torch.cumsum(torch.rand(4, 256, generator=g) < 0.02, 1)
    idx, tgt, doc = idx.cuda(), tgt.cuda(), doc.cuda()
    l32, g32 = grads_of(m, idx, tgt, doc, 0, None)
    lr, gr = grads_of(m, idx, tgt, doc, 0, BF16)
    lr2, gr2 = grads_of(m, idx, tgt, doc, 0, BF16)
    lc, gc = grads_of(m, idx, tgt, doc, 300, BF16)
    lc2, gc2 = grads_of(m, idx, tgt, doc, 300, BF16)
    rows = {n: (float((gc[n] - g32[n]).norm()), float((gr[n] - g32[n]).norm()),
                float((gc[n] - gr[n]).norm()), float(g32[n].norm())) for n in g32}
    ok = [r for r in rows.values() if r[1] > 0]
    print(f"loss fp32 {l32:.6f} ref {lr:.6f} chunked {lc:.6f}; worst chunked error / reference "
          f"error {max(r[0] / r[1] for r in ok):.3f}; worst |chunked - reference| / reference "
          f"error {max(r[2] / r[1] for r in ok):.3f}")
    assert lr == lr2 and all(torch.equal(gr[n], gr2[n]) for n in gr), "reference not deterministic"
    assert lc == lc2 and all(torch.equal(gc[n], gc2[n]) for n in gc), "chunked not deterministic"
    assert abs(lc - lr) <= 0.2 * abs(lr - l32) + 1e-6 * abs(l32)
    for n, (e_c, e_r, d, nrm) in rows.items():
        assert e_c <= 1.15 * e_r + 1e-7 * nrm, (n, e_c, e_r)
        assert d <= 1.5 * e_r + 1e-7 * nrm, n


def test_trainer_accum_unequal_counts_cuda():
    import test_chunked_ce as cpu
    mp = pytest.MonkeyPatch()
    real = cpu.model
    mp.setattr(cpu, "model", lambda arm="base_gqa", seed=1: real(arm, seed).to("cuda"))
    try:
        cpu.test_trainer_accum_unequal_supervised_counts_matches_reference(device="cuda")
    finally:
        mp.undo()


def head_peak(chunk: int):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device="cuda").manual_seed(0)
    h = torch.randn(16384, 256, device="cuda", generator=g).requires_grad_()  # fp32 norm output
    w = (0.05 * torch.randn(8192, 256, device="cuda", generator=g)).requires_grad_()
    tgt = torch.randint(0, 8192, (16384,), device="cuda", generator=g)
    with BF16():
        if chunk:
            logits, loss = None, chunked_lm_loss(h, w, tgt, chunk, "sum")
        else:
            logits = F.linear(h, w)                     # held through backward, as trainer's `_`
            loss = F.cross_entropy(logits.view(-1, 8192).float(), tgt, reduction="sum")
    (loss / 16384).backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return peak, float(loss), h.grad.detach().clone(), w.grad.detach().clone()


def test_loss_path_peak_memory_drops():
    p_r, l_r, gh_r, gw_r = head_peak(0)
    p_c, l_c, gh_c, gw_c = head_peak(2048)
    print(f"head peak: reference {p_r / 2**20:.0f} MiB, chunked {p_c / 2**20:.0f} MiB")
    assert p_c < 0.25 * p_r
    assert p_c < 16384 * 8192 * 2       # below one full bf16 logits tensor: none was ever made
    assert abs(l_c - l_r) <= 1e-4 * abs(l_r)
    assert (gh_c - gh_r).norm() <= 2e-2 * gh_r.norm() and (gw_c - gw_r).norm() <= 2e-2 * gw_r.norm()


def test_torch_compile_composes(deterministic):
    """torch.compile(model) with ce_chunk runs and returns no logits, and computes the same
    loss and gradients as the compiled reference. fp32 (no autocast): equal to fp32 rounding,
    the CPU tests' tolerance (test_chunked_ce.close). bf16 autocast, deterministic: both
    compiled paths are bitwise repeatable, and the chunked loss and every gradient are no
    further from the eager fp32 answer than 1.5x the worse of the two bf16 reference paths
    (eager, compiled), globally no further than 1.2x. The distance between the two bf16
    paths is not a criterion: chunks run other GEMM shapes than the full batch, so logits
    round differently; that moves some gradients by several times what eager -> compiled
    moves the reference, whose big GEMMs are the same (this batch on the 5070: norm.weight
    4.2x, blocks.1.attn.vr_alpha 2.9x; up to 33x on CPU with the aot_eager backend), while
    the error against fp32 stays inside the envelope (worst 0.95x of the worse reference)."""
    from test_chunked_ce import close
    torch.manual_seed(0)
    cfg = PlanckConfig(vocab_size=8192, d_model=128, n_layers=2, n_heads=2, n_kv_heads=1,
                       head_dim=64, mlp_hidden=256, seq_len=256, n_loops=2)
    m = build_model(cfg, "cpu").to("cuda").train()
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(0, 8192, (4, 256), generator=g)
    tgt = torch.randint(0, 8192, (4, 256), generator=g)
    tgt[torch.rand(4, 256, generator=g) < 0.5] = -100
    doc = torch.cumsum(torch.rand(4, 256, generator=g) < 0.02, 1)
    idx, tgt, doc = idx.cuda(), tgt.cuda(), doc.cuda()
    l32, g32 = grads_of(m, idx, tgt, doc, 0, None)
    le, ge = grads_of(m, idx, tgt, doc, 0, BF16)
    torch._dynamo.reset()
    mc = torch.compile(m)

    def run(chunk, amp):
        m.zero_grad(set_to_none=True)
        kw = {"ce_chunk": chunk} if chunk else {}
        with (amp() if amp else torch.autocast("cuda", enabled=False)):
            lg, loss = mc(idx, tgt, doc, reduction="sum", **kw)
        loss.backward()
        torch.cuda.synchronize()
        return lg, float(loss.detach()), {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    try:
        _, fr, fgr = run(0, None)
        lg_f, fc, fgc = run(300, None)
        lg_r, lr, gr = run(0, BF16)
        _, lr2, gr2 = run(0, BF16)
        lg_c, lc, gc = run(300, BF16)
        _, lc2, gc2 = run(300, BF16)
    finally:
        torch._dynamo.reset()
    assert lg_r is not None and lg_c is None and lg_f is None
    assert fc == pytest.approx(fr, rel=2e-6)
    for n in fgr:
        assert close(fgc[n], fgr[n]), n
    assert lr == lr2 and all(torch.equal(gr[n], gr2[n]) for n in gr), "compiled ref not deterministic"
    assert lc == lc2 and all(torch.equal(gc[n], gc2[n]) for n in gc), "compiled chunked not deterministic"
    err = lambda gg: {n: float((gg[n] - g32[n]).norm()) for n in g32}  # noqa: E731
    e_c, e_r, e_e = err(gc), err(gr), err(ge)
    tot = lambda e: sum(v * v for v in e.values()) ** 0.5  # noqa: E731
    worst = max(g32, key=lambda n: e_c[n] / max(e_r[n], e_e[n], 1e-30))
    print(f"compiled: fp32 loss {fr:.6f} chunked {fc:.6f}; bf16 loss fp32 {l32:.6f} eager {le:.6f} "
          f"ref {lr:.6f} chunked {lc:.6f}; worst chunked error / worse ref error {worst} "
          f"{e_c[worst] / max(e_r[worst], e_e[worst]):.3f}; global chunked {tot(e_c):.4e} "
          f"ref {tot(e_r):.4e} eager ref {tot(e_e):.4e}")
    assert abs(lc - l32) <= 1.5 * max(abs(lr - l32), abs(le - l32)) + 1e-6 * abs(l32)
    for n in g32:
        assert e_c[n] <= 1.5 * max(e_r[n], e_e[n]) + 1e-6 * float(g32[n].norm()), (n, e_c[n], e_r[n], e_e[n])
    assert tot(e_c) <= 1.2 * max(tot(e_r), tot(e_e))


def test_fused_core_matches_the_eager_core(deterministic):
    """Eager CUDA runs the per-chunk core compiled (chunked_ce.FUSE_CUDA); the same code run
    op by op gives the same loss to fp32 rounding and gradients equal to within bf16 rounding
    of the logits gradient (inductor's exp and log round the last bits differently)."""
    import chunked_ce
    g = torch.Generator(device="cuda").manual_seed(3)
    h = torch.randn(9000, 192, device="cuda", generator=g).requires_grad_()   # 9000 = 4 x 2048 + 808
    w = (0.05 * torch.randn(8192, 192, device="cuda", generator=g)).requires_grad_()
    tgt = torch.randint(0, 8192, (9000,), device="cuda", generator=g)
    tgt[::3] = -100
    out = {}
    try:
        for fuse in (False, True, True):
            chunked_ce.FUSE_CUDA = fuse
            h.grad = w.grad = None
            with BF16():
                loss = chunked_lm_loss(h, w, tgt, 2048, "mean")
            loss.backward()
            out.setdefault(fuse, []).append((float(loss), h.grad.clone(), w.grad.clone()))
    finally:
        chunked_ce.FUSE_CUDA = True
    (l_e, gh_e, gw_e), = out[False]
    (l_f, gh_f, gw_f), (l_f2, gh_f2, gw_f2) = out[True]
    assert l_f == l_f2 and torch.equal(gh_f, gh_f2) and torch.equal(gw_f, gw_f2)   # deterministic
    assert l_f == pytest.approx(l_e, rel=1e-6)
    assert (gh_f - gh_e).norm() <= 4e-3 * gh_e.norm() and (gw_f - gw_e).norm() <= 4e-3 * gw_e.norm()
