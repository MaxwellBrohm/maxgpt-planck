"""train.compile (torch.compile of the training forward), CPU checks on the model and Trainer.

The compiled paths here use backend "aot_eager": Dynamo capture plus AOTAutograd, no code
generation, so they run on any CPU without a C++ toolchain and test what compile changes in
the harness (graph capture of forward + loss, the skipped all-ignored guard, which module
owns the parameters). fullgraph=True turns any graph break into an error. train.main runs
(off by default, resume, switching across a resume): test_compile_train.py. Inductor on the
GPU: test_compile_cuda.py, test_speed3_cuda.py, compile_parity.py. Varlen with a CPU stand-in
kernel: test_compile_varlen.py.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model import build_model, compile_forward, compile_mode
from test_s3_leak import DOC, T, perturb_check
from testutil import ARMS, scramble, tiny
from trainer import Trainer


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()      # every test compiles from scratch (no cache-limit fallbacks)
    torch._dynamo.utils.counters.clear()
    yield
    torch._dynamo.reset()


def fullgraph(model):
    return torch.compile(model, backend="aot_eager", fullgraph=True)


def test_compile_mode_values():
    for off in (None, False, "off", "none", "false", "eager"):
        assert compile_mode(off) is None
    assert compile_mode(True) == "default"
    for m in ("default", "max-autotune-no-cudagraphs"):
        assert compile_mode(m) == m
    for bad in ("fast", "reduce-overhead", "max-autotune"):      # CUDA graphs: untested, refused
        with pytest.raises(ValueError):
            compile_mode(bad)
    m = build_model(tiny())
    assert compile_forward(m, None) is m
    c = compile_forward(m, "default", backend="aot_eager")
    assert c is not m and c._orig_mod is m
    assert list(c.state_dict()) != list(m.state_dict())          # the wrapper adds _orig_mod.
    assert not any(k.startswith("_orig_mod.") for k in m.state_dict())


def test_sum_over_zero_supervised_tokens_is_exactly_zero():
    """Why the compiled path may drop the (tgt != -100).any() guard for reduction "sum":
    the sum over an empty set is 0 and its gradient is exactly 0. "mean" is 0/0 = nan,
    which is what the guard exists for, so "mean" keeps it."""
    logits = torch.randn(12, 50, requires_grad=True)
    tgt = torch.full((12,), -100)
    ls = F.cross_entropy(logits, tgt, ignore_index=-100, reduction="sum")
    assert ls.item() == 0.0
    ls.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert torch.isnan(F.cross_entropy(logits, tgt, ignore_index=-100, reduction="mean"))


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_all_ignored_batch_compiled_in_one_graph(packed):
    """A micro-batch with no supervised token (all -100): the compiled forward + loss is one
    graph (fullgraph: the guard would break it), the loss is exactly 0, every grad exactly 0,
    and eager (which keeps the guard) gives the same. Reduction "mean" keeps the guard when
    compiled (a graph break, not a nan)."""
    torch.manual_seed(0)
    m = scramble(build_model(tiny()), 1).train()
    idx = torch.randint(0, 97, (2, T))
    tgt = torch.full((2, T), -100)
    doc = torch.tensor([DOC] * 2) if packed else None
    for fwd in (m, fullgraph(m)):
        m.zero_grad(set_to_none=False)
        _, ls = fwd(idx, tgt, doc, reduction="sum")
        assert ls.item() == 0.0
        ls.backward()
        assert all(torch.equal(p.grad, torch.zeros_like(p)) for p in m.parameters())
    assert torch.compile(m, backend="aot_eager")(idx, tgt, doc)[1].item() == 0.0


ARM_SUBSET = ["base_gqa", "attn_only", "kv_tie", "loop3_immediate", "prelude_loop_coda",
              "loop_share_tie", "untied", "plain_block", "canon_ac", "forget_gate", "smear_key"]


@pytest.mark.parametrize("arm", ARM_SUBSET)
@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_compiled_loss_and_grads_match_eager(arm, packed):
    torch.manual_seed(0)
    m = scramble(build_model(tiny(**ARMS[arm])), 2).train()
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, 97, (3, T), generator=g)
    tgt = torch.randint(0, 97, (3, T), generator=g)
    tgt[torch.rand(3, T, generator=g) < 0.4] = -100
    doc = torch.tensor([DOC] * 3) if packed else None
    out = []
    for fwd in (m, fullgraph(m)):
        m.zero_grad(set_to_none=True)
        _, ls = fwd(idx, tgt, doc, reduction="sum")
        ls.backward()
        out.append((ls.detach(), [p.grad.clone() for p in m.parameters()]))
    (le, ge), (lc, gc) = out
    assert torch.allclose(le, lc, rtol=1e-6, atol=1e-5)
    for a, b in zip(ge, gc):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("arm", ["base_gqa", "loop_share_tie", "prelude_loop_coda"])
@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_compiled_forward_has_no_leak(arm, packed):
    """test_s3_leak's one-token perturbation probe (bitwise, scrambled weights) through the
    compiled forward, with and without gradients."""
    torch.manual_seed(0)
    m = scramble(build_model(tiny(**ARMS[arm])), 0)
    c = torch.compile(m, backend="aot_eager")
    for grad in (True, False):
        perturb_check(c, DOC if packed else None, grad)


class FixedLoader:
    """next_batch() hands out the given micro-batches in order (Trainer only needs this)."""
    pad = 0

    def __init__(self, batches):
        self.batches, self.i = batches, 0

    def next_batch(self):
        b = self.batches[self.i % len(self.batches)]
        self.i += 1
        return b


class FlatSched:
    def factor(self, step):
        return 1.0


class GradSpy:
    """Wraps the optimizer: records every parameter's gradient when step() is called."""

    def __init__(self, opt):
        self.opt, self.grads = opt, None

    def __getattr__(self, k):
        return getattr(self.opt, k)

    def step(self):
        self.grads = [p.grad.clone() for g in self.opt.param_groups for p in g["params"]]
        self.opt.step()


def accum_batches(packed: bool):
    """grad_accum 3 with unequal supervised counts: about 90 %, about 10 % and 0 tokens."""
    g = torch.Generator().manual_seed(11)
    out = []
    for frac in (0.9, 0.1, 0.0):
        idx = torch.randint(1, 97, (2, T), generator=g)
        tgt = torch.randint(0, 97, (2, T), generator=g)
        tgt[torch.rand(2, T, generator=g) >= frac] = -100
        doc = torch.tensor([DOC] * 2) if packed else None
        out.append({"idx": idx, "tgt": tgt, "doc": doc, "pos": None})
    return out


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_grad_accum_unequal_supervised_counts_compiled(packed, tmp_path):
    """The trainer's step through the compiled forward: loss and gradients equal ONE eager
    forward over all rows at once, divided by the step's supervised-token count (the
    reference does not go through the trainer's accumulation at all)."""
    from optim import make_optimizer
    batches = accum_batches(packed)
    n = [int((b["tgt"] != -100).sum()) for b in batches]
    assert n[0] > 3 * n[1] > 0 and n[2] == 0
    torch.manual_seed(0)
    ref = scramble(build_model(tiny()), 4).train()
    cat = {k: None if batches[0][k] is None else torch.cat([b[k] for b in batches])
           for k in ("idx", "tgt", "doc")}
    _, ls = ref(cat["idx"], cat["tgt"], cat["doc"], reduction="sum")
    (ls / sum(n)).backward()
    ref_loss, ref_grads = ls.item() / sum(n), [p.grad.clone() for p in ref.parameters()]
    for compiled in (False, True):
        torch.manual_seed(0)
        m = scramble(build_model(tiny()), 4).train()
        opt = GradSpy(make_optimizer(m, {}, "cpu"))
        tr = Trainer(model=m, optimizer=opt, loader=FixedLoader(batches), sched=FlatSched(),
                     device="cpu", amp=None, cfg={}, out_dir=str(tmp_path), grad_accum=3,
                     grad_clip=1e9, log_every=1, ckpt_every=0, keep_last=1, stable_points=set(),
                     meta={}, forward=fullgraph(m) if compiled else None)
        out = tr.train_step()
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == (1 if compiled else 0)
        assert out["n_sup"] == sum(n)
        assert out["loss"] == pytest.approx(ref_loss, rel=1e-6)
        got = dict(zip([id(p) for g in opt.param_groups for p in g["params"]], opt.grads))
        assert len(got) == len(ref_grads)
        for p, r in zip(m.parameters(), ref_grads):
            assert r.abs().max() > 0
            assert torch.allclose(got[id(p)], r, rtol=1e-4, atol=1e-7), compiled


def test_trainer_refuses_a_compiled_wrapper_as_model(tmp_path):
    """The optimizer, clipping and state_dict must see the plain module (no _orig_mod.)."""
    m = build_model(tiny())
    with pytest.raises(ValueError, match="plain model"):
        Trainer(model=torch.compile(m, backend="aot_eager"), optimizer=None, loader=None,
                sched=None, device="cpu", amp=None, cfg={}, out_dir=str(tmp_path), grad_accum=1,
                grad_clip=1.0, log_every=1, ckpt_every=0, keep_last=1, stable_points=set(), meta={})
