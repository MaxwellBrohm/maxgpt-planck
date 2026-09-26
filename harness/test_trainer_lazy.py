"""Trainer lazy_metrics (train key lazy_metrics: true): loss and grad norm stay on the device
and are read on log steps only. The logged floats, the gradients and the weights must be
exactly the eager path's; a non-finite loss must still stop the run before any checkpoint
holds it. Also: checkpoints resume across optim batched on/off. CPU only.
"""
from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.nn.functional as F

import runio
import schedule as S
import train
from make_fake_data import make
from model import build_model
from optim import make_optimizer
from test_s3_loss_mask import ListLoader
from test_train import final_weights, read_jsonl, write_run
from testutil import scramble, tiny
from trainer import Trainer


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_lazy")
    make(str(d), docs=200, chats=150)
    return str(d)


def fixed_batches():
    """Two micro-batches with very different supervised counts (61 vs 5 tokens)."""
    g = torch.Generator().manual_seed(7)
    out = []
    for n_sup in (61, 5):
        idx = torch.randint(1, 97, (2, 40), generator=g)
        tgt = torch.randint(1, 97, (2, 40), generator=g)
        keep = torch.zeros(80, dtype=torch.bool)
        keep[torch.randperm(80, generator=g)[:n_sup]] = True
        tgt = torch.where(keep.view(2, 40), tgt, torch.full_like(tgt, -100))
        out.append({"idx": idx, "tgt": tgt, "doc": None, "pos": None})
    return out


def make_trainer(m, lazy, batches, tmp):
    opt = make_optimizer(m, {"lr": 3e-3})
    grads = []
    step = opt.step

    def spy():                                     # the gradient the optimizer consumes
        grads.append([p.grad.clone() for p in m.parameters()])
        step()
    opt.step = spy
    tr = Trainer(model=m, optimizer=opt, loader=ListLoader(batches), sched=S.full(10),
                 device="cpu", amp=None, cfg={}, out_dir=str(tmp), grad_accum=2, grad_clip=1.0,
                 log_every=1, ckpt_every=0, keep_last=1, stable_points=set(), meta={},
                 lazy_metrics=lazy)
    return tr, grads


def test_fixed_batch_accum2_unequal_sup_same_loss_and_grads(tmp_path):
    batches = fixed_batches()
    torch.manual_seed(0)
    m_eager = scramble(build_model(tiny()), 4).train()
    m_lazy = copy.deepcopy(m_eager)
    with torch.no_grad():                          # independent reference for step 1's loss
        ref = sum(float(F.cross_entropy(m_eager(b["idx"])[0].flatten(0, 1), b["tgt"].flatten(),
                                        ignore_index=-100, reduction="sum")) for b in batches) / 66
    t_e, g_e = make_trainer(m_eager, False, batches, tmp_path)
    t_l, g_l = make_trainer(m_lazy, True, batches, tmp_path)
    for i in range(3):
        oe, ol = t_e.train_step(), t_l.train_step()
        assert "loss" not in ol and torch.is_tensor(ol["gnorm"])     # nothing read yet
        ol = t_l.read_metrics(ol)
        assert oe == ol, (oe, ol)                   # the same floats, bit for bit
        assert oe["n_sup"] == 66
        if i == 0:
            assert oe["loss"] == pytest.approx(ref, rel=1e-6)
    for a, b in zip(g_e, g_l):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert all(torch.equal(a, b) for a, b in zip(m_eager.parameters(), m_lazy.parameters()))


@pytest.mark.parametrize("log_every", [1, 5])
def test_lazy_run_logs_and_weights_equal_eager(tmp_path, fake, log_every):
    tc = {"total_steps": 12, "log_every": log_every, "grad_accum": 2}
    a = write_run(str(tmp_path / "a"), fake, train=tc)
    b = write_run(str(tmp_path / "b"), fake, train={**tc, "lazy_metrics": True})
    assert train.main([a]) == 0 and train.main([b]) == 0
    la, lb = (read_jsonl(tmp_path / r / "out" / "log.jsonl") for r in "ab")
    drop = lambda r: {k: v for k, v in r.items() if k not in ("tok_per_s", "time")}  # noqa: E731
    assert [r["step"] for r in lb] == ([*range(1, 13)] if log_every == 1 else [5, 10, 12])
    assert [drop(r) for r in la] == [drop(r) for r in lb]
    wa, wb = final_weights(str(tmp_path / "a")), final_weights(str(tmp_path / "b"))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


@pytest.mark.parametrize("lazy", [False, True])
def test_nonfinite_loss_stops_before_a_checkpoint_holds_it(tmp_path, lazy):
    """A hook poisons the final norm after step 2 and restores the weights after step 3, so
    only step 3's loss is NaN (step 4's is finite: the flag must remember step 3). Eager
    raises at step 3; lazy (log_every 5) must raise at the step-4 checkpoint, unwritten."""
    torch.manual_seed(0)
    m = build_model(tiny()).train()
    tr, _ = make_trainer(m, lazy, fixed_batches(), tmp_path)
    tr.log_every, tr.ckpt_every = 5, 4

    saved = {}

    def poison(t):
        if t.step == 2:
            saved.update(copy.deepcopy(t.model.state_dict()))
            with torch.no_grad():
                t.model.norm.weight.fill_(float("nan"))
        if t.step == 3:
            t.model.load_state_dict(saved)
    tr.hooks.append(poison)
    with pytest.raises(FloatingPointError):
        tr.run()
    assert tr.step == (4 if lazy else 3)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".pt")]
    err = read_jsonl(tmp_path / "log.jsonl")[-1]
    assert err["error"] == "non-finite loss" and err["step"] == tr.step


def test_resume_across_batched_and_lazy(tmp_path, fake):
    """5 steps with the reference optimizer, resumed to 12 with batched + lazy_metrics:
    the same weights as 12 reference steps (optimizer state layout is shared)."""
    straight = write_run(str(tmp_path / "a"), fake)
    assert train.main([straight]) == 0
    p = write_run(str(tmp_path / "b"), fake)
    assert train.main([p, "--max-steps", "5"]) == 0
    p = write_run(str(tmp_path / "b"), fake, optim={"lr": 3e-3, "batched": True},
                  train={"lazy_metrics": True})
    assert train.main([p]) == 0
    ck = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "b" / "out")))
    assert ck["step"] == 12
    wa, wb = final_weights(str(tmp_path / "a")), final_weights(str(tmp_path / "b"))
    worst = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    assert worst < 1e-5, worst
