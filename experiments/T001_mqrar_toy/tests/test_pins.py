"""T001 pins for registered forms and constants the first suite left loose (critique D4): Selective Attention reads
head 0, Canon-A/C read the normed input, the init scale, AdamW's settings, the clip, the training n range, the eval
sizes, divergence handling and TF32 off."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402
from layers import Attention, CausalConv, rope, selective_mask  # noqa: E402
from model import Block, ToyLM  # noqa: E402
from task import SPLIT_SIZE  # noqa: E402


def test_selective_attention_reads_head_0():
    torch.manual_seed(5)
    a = Attention(8, 2, "sa").double()
    with torch.no_grad():
        a.qkv.weight.mul_(4.0)                         # large logits, so ReLU(S) is non-zero and the heads differ
    x = torch.randn(1, 9, 8, dtype=torch.float64)
    q, k, v = (x @ a.qkv.weight.t()).view(1, 9, 3, 2, 4).permute(2, 0, 3, 1, 4)
    s = rope(q) @ rope(k).transpose(-1, -2) / 2.0
    causal = torch.ones(9, 9, dtype=torch.bool).tril()

    def out(head):
        z = (s - selective_mask(s[:, head], causal)[:, None]).masked_fill(~causal, float("-inf"))
        return (torch.softmax(z, -1) @ v).transpose(1, 2).reshape(1, 9, 8) @ a.o.weight.t()
    assert not torch.allclose(out(0), out(1), atol=1e-6)                # the fixture tells the heads apart
    assert torch.allclose(a(x), out(0), atol=1e-10)


def test_canon_convs_read_the_normed_input():
    torch.manual_seed(6)
    b = Block("base", True, 2).double()
    with torch.no_grad():
        b.canon_a.weight.normal_(0, 0.5)
        b.canon_c.weight.normal_(0, 0.5)
        b.n1.weight.uniform_(0.5, 2.0)
        b.n2.weight.uniform_(0.5, 2.0)
    x = 3.0 * torch.randn(2, 7, 128, dtype=torch.float64)
    h = b.n1(x)
    x2 = x + b.mix(h + b.canon_a(h))
    h = b.n2(x2)
    want = x2 + b.down(F.gelu(b.up(h + b.canon_c(h))))
    assert torch.allclose(b(x), want, atol=1e-10)


@pytest.mark.parametrize("arm,layers", [("base", 2), ("base1", 1), ("gdn", 2)])
def test_init_scales(arm, layers):
    torch.manual_seed(0)
    m = ToyLM(arm)
    out_std = 0.02 / math.sqrt(2 * layers)
    for name, p in m.named_parameters():
        if p.dim() < 2 or isinstance(dict(m.named_modules())[name.rsplit(".", 1)[0]], CausalConv):
            continue
        last = name.endswith(".o.weight") or name.endswith(".down.weight")
        assert p.std().item() == pytest.approx(out_std if last else 0.02, rel=0.06), name


def test_optimizer_settings():
    m = ToyLM("fg")
    opt = train.make_opt(m, 1e-3)
    g0, g1 = opt.param_groups
    assert g0["weight_decay"] == 0.1 and g1["weight_decay"] == 0.0
    assert {id(p) for p in g0["params"]} == {id(p) for p in m.parameters() if p.dim() >= 2}
    assert {id(p) for p in g1["params"]} == {id(p) for p in m.parameters() if p.dim() < 2} and len(g1["params"]) > 0
    for g in (g0, g1):
        assert g["betas"] == (0.9, 0.95) and g["eps"] == 1e-8 and g["lr"] == 1e-3
    assert isinstance(opt, torch.optim.AdamW)


def test_update_clips_the_gradient_norm_to_1():
    torch.manual_seed(1)
    m = ToyLM("base1")
    x = torch.randint(0, 228, (2, 10))
    loss = 1000.0 * m(x).logsumexp(-1).mean()
    raw = torch.autograd.grad(loss, list(m.parameters()), retain_graph=True)
    assert torch.sqrt(sum((g * g).sum() for g in raw)).item() > 10.0     # the fixture's raw norm is far above 1
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    train.update(m, opt, loss)
    norm = torch.sqrt(sum((p.grad * p.grad).sum() for p in m.parameters())).item()
    assert norm == pytest.approx(1.0, rel=1e-4)


def test_training_n_covers_both_ends():
    rng = np.random.default_rng(0)
    ns = [train.draw_n(rng, 8, 96) for _ in range(20000)]
    assert min(ns) == 8 and max(ns) == 96 and len(set(ns)) == 89
    a = train.parse(["--arm", "base", "--seed", "1", "--lr", "1e-3", "--out", "x"])
    assert (a.n_lo, a.n_hi, a.steps, a.batch, a.family) == (8, 96, 6000, 32, "mqrar")
    assert a.grid == "8,16,24,32,48,64,96"


def test_eval_sizes():
    assert SPLIT_SIZE == {"test": 256, "dev": 128}
    a = train.parse(["--arm", "base", "--seed", "1", "--lr", "1e-3", "--out", "x"])
    assert (a.eval_test, a.eval_dev) == (256, 128)


def tiny_args(tmp_path):
    return train.parse(["--arm", "base", "--seed", "1", "--lr", "1e-3", "--out", str(tmp_path / "x.json"),
                        "--steps", "3", "--n-hi", "16", "--grid", "8,16", "--eval-test", "2", "--eval-dev", "2",
                        "--log-every", "1", "--device", "cpu", "--smoke"])


def test_divergence_stops_and_marks_the_run(tmp_path, monkeypatch):
    real = F.cross_entropy
    calls = []

    def ce(*a, **kw):
        calls.append(1)
        out = real(*a, **kw)
        return out * float("nan") if len(calls) == 2 else out
    monkeypatch.setattr(train.F, "cross_entropy", ce)
    r = train.run(tiny_args(tmp_path))
    assert r["diverged"] and r["eval"] is None and r["steps_done"] == 1 and r["loss_curve"][-1] == [1, None]


def test_tf32_is_off_after_setup(tmp_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train.run(tiny_args(tmp_path))
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
