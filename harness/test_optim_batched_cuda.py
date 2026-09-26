"""optim_batched on CUDA (skipped without CUDA): bf16 Newton-Schulz, deterministic algorithms.

On CUDA the Newton-Schulz runs in bf16 (make_optimizer, device_type cuda), and batched
GEMMs need not round like single ones, so bit equality is not the bar. The bar is the
reference's own precision noise, measured in the same test:
  update  rel(batched, reference) < rel(reference with fp32 NS, reference): batching moves
          the update less than the reference's own bf16 choice does.
  curve   200 steps of a 5M-shape model (bf16 autocast, clip, constant lr) under
          torch.use_deterministic_algorithms. The reference run twice is bit-equal here, in
          deterministic AND default mode (measured on the 5070), so "baseline twice" gives
          a floor of 0 and cannot be the bar. The floor is the reference against itself with
          a tiny change: weights times (1 + 1e-7 N(0,1)), and fp32 instead of bf16 NS. Any
          such change grows (the dynamics are chaotic) to max |dloss| ~0.12 by step ~50 and a
          mean |dloss| ~0.013 over steps 100-199. Batched must stay within 1.5x the larger
          floor on: max |dloss| over steps 0-9 (before chaos), over all steps, and the late mean.
          This catches a systematic divergence; float-level parity is the CPU test's job.
Run this file in its own process (CUBLAS_WORKSPACE_CONFIG must be set before cuBLAS starts):
  python -m pytest -q -s test_optim_batched_cuda.py
"""
from __future__ import annotations

import argparse
import copy
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402

import budget  # noqa: E402
from device import amp_factory  # noqa: E402
from model import build_model  # noqa: E402
from optim import make_optimizer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def shape_5m(T: int):
    ap = argparse.ArgumentParser()
    budget.add_constraint_args(ap)
    return budget.solve(5e6, budget.constraints_from_args(ap.parse_args([]))).cfg.replace(seq_len=T)


@pytest.fixture
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def rel(a, b) -> float:
    return float((a - b).norm() / b.norm())


def updates(model, batched: bool, fp32_ns: bool = False, steps: int = 3) -> list:
    m = copy.deepcopy(model)
    p0 = [p.detach().clone() for p in m.parameters()]
    opt = make_optimizer(m, {"lr": 3e-3, "batched": batched}, device_type="cuda")
    if fp32_ns:
        opt.ns_dtype = None
    for s in range(steps):
        g = torch.Generator(device="cuda").manual_seed(50 + s)
        for p in m.parameters():
            p.grad = torch.randn(p.shape, generator=g, device="cuda") * 1e-2
        opt.step()
    return [p.detach() - z for p, z in zip(m.parameters(), p0)]


def test_cuda_update_within_reference_precision_noise(deterministic):
    torch.manual_seed(0)
    model = build_model(shape_5m(64)).cuda()
    ref, bat, ref32 = updates(model, False), updates(model, True), updates(model, False, True)
    d_bat = max(rel(b, r) for b, r in zip(bat, ref))
    d_prec = max(rel(x, r) for x, r in zip(ref32, ref))
    print(f"\nupdate rel: batched vs ref {d_bat:.3e}; ref fp32-NS vs ref bf16-NS {d_prec:.3e}")
    assert d_bat < d_prec


def curve(model, batched: bool, fp32_ns: bool = False, steps: int = 200, B: int = 8) -> list:
    m = copy.deepcopy(model)
    opt = make_optimizer(m, {"lr": 3e-3, "batched": batched}, device_type="cuda")
    if fp32_ns:
        opt.ns_dtype = None
    amp = amp_factory("bf16", "cuda")
    T, V = m.cfg.seq_len, m.cfg.vocab_size
    g = torch.Generator().manual_seed(9)
    data = [torch.randint(0, V // 8, (B, T + 1), generator=g).cuda() for _ in range(8)]
    out = []
    for s in range(steps):
        x = data[s % len(data)]
        tgt = x[:, 1:].clone()
        tgt[:, : T // 4] = -100                      # unequal supervised counts are the trainer's job;
        with amp():                                  # here: some masked targets on the loss path
            _, loss = m(x[:, :-1], tgt, reduction="sum")
        (loss / (B * (T - T // 4))).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        out.append(loss.detach() / (B * (T - T // 4)))
    return torch.stack(out).tolist()


def perturbed(model, eps: float):
    m = copy.deepcopy(model)
    g = torch.Generator(device="cuda").manual_seed(123)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(1 + eps * torch.randn(p.shape, generator=g, device="cuda"))
    return m


def test_cuda_200_step_curve_matches(deterministic):
    torch.manual_seed(0)
    model = build_model(shape_5m(256)).cuda()
    ref = curve(model, False)
    runs = {"batched": curve(model, True), "ref_w1e-7": curve(perturbed(model, 1e-7), False),
            "ref_fp32NS": curve(model, False, fp32_ns=True)}

    def stats(c):
        d = [abs(x - y) for x, y in zip(c, ref)]
        return max(d[:10]), max(d), sum(d[100:]) / len(d[100:])
    st = {k: stats(c) for k, c in runs.items()}
    print(f"\nref {ref[0]:.4f} -> {ref[-1]:.5f}; (max|d| 0-9, max|d|, mean|d| 100-199), final:")
    for k, v in st.items():
        print(f"  {k:11s} {v[0]:.2e} {v[1]:.2e} {v[2]:.2e}  {runs[k][-1]:.5f}")
    assert ref[-1] < 0.8 * ref[0]
    assert curve(model, False) == ref, "deterministic mode: the reference is bit-reproducible"
    floor = [max(st["ref_w1e-7"][i], st["ref_fp32NS"][i]) for i in range(3)]
    assert all(b <= 1.5 * f for b, f in zip(st["batched"], floor)), (st, floor)
