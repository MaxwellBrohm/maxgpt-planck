"""S006 t+2 aux head (train.mtp; mtp.py), step level (split from test_screen_mtp_aux.py for the file size). CPU.
  gradient   the head and the trunk get aux gradient; W_mtp in NorMuon 'matrix' and the gain in 'scalar' (after
             the model's own parameters, which keep their groups and order); one step moves both.
  loss path  Trainer, grad_accum 2, unequal supervised counts, eager and lazy_metrics: loss, mtp_loss, mtp_n,
             mtp_w (w x the warmup factor 1/4), gnorm (model + head) and every gradient against an eager
             (sum NTP + w_t sum aux) / n_sup reference written out by hand (plain-loop targets, RMSNorm, tied head).
  weight 0   mtp_weight 0: the next-token loss and every shared gradient == the mtp-off step's, bitwise.
  non-finite a NaN aux loss stops the run. Eager: in the NaN step. lazy_metrics: the device flag is read only on
             log steps and before saves, so the run stops at the next log step or checkpoint save (the steps in
             between run) and no NaN checkpoint is written. Tested with step 1 a checkpoint step, and with step 1
             neither a log nor a checkpoint step (stop step == the next log step, or the next save).
"""
from __future__ import annotations

import copy
import json
import os
import re

import pytest
import torch
import torch.nn.functional as F

import schedule as S
from model import build_model
from mtp import MTPHead, mtp_targets
from optim import make_optimizer
from test_s3_docattn import ROWS
from test_s3_leak import T
from test_s3_loss_mask import ListLoader
from test_screen_mtp_aux import arm, gen, loader_tgt, ref_t2
from testutil import tiny
from trainer import Trainer


def packed_batches(sup_ps=(0.9, 0.3), seed=6):
    g, out = gen(seed), []
    for p in sup_ps:
        idx = torch.randint(0, 97, (len(ROWS), T), generator=g)
        tgt = torch.cat([loader_tgt(idx[i:i + 1], r, (torch.rand(T, generator=g) < p).tolist())
                         for i, r in enumerate(ROWS)])
        out.append({"idx": idx, "tgt": tgt, "doc": torch.tensor(ROWS), "pos": None})
    return out


def test_gradient_reaches_head_and_trunk_and_groups():
    torch.manual_seed(0)
    m, b = build_model(tiny()), packed_batches()[0]
    head = MTPHead(m.cfg.d_model, m.cfg.rms_eps)
    t2 = mtp_targets(b["idx"], b["tgt"], b["doc"])
    z = m(b["idx"], doc=b["doc"], return_hidden=True)[2]
    head.loss_sum(z, m.lm_head, t2, int((t2 != -100).sum())).backward()
    assert (head.weight.grad != 0).any() and (head.norm.weight.grad != 0).any()
    assert (m.blocks[0].attn.q_proj.weight.grad != 0).any() and (m.tok_emb.weight.grad != 0).any()
    opt = make_optimizer(m, {"lr": 1e-2, "embed_lr": 5e-3, "scalar_lr": 2e-3}, extra=head)
    g = {x["name"]: x for x in opt.param_groups}
    ids = {k: [id(p) for p in x["params"]] for k, x in g.items()}
    plain = {x["name"]: [id(p) for p in x["params"]] for x in make_optimizer(m, {"lr": 1e-2}).param_groups}
    assert ids["matrix"] == plain["matrix"] + [id(head.weight)] and g["matrix"]["use_muon"]
    assert ids["scalar"] == plain["scalar"] + [id(head.norm.weight)] and ids["embed"] == plain["embed"]
    assert not g["scalar"]["use_muon"] and g["scalar"]["base_lr"] == 2e-3 and g["scalar"]["weight_decay"] == 0
    before = [p.detach().clone() for p in head.parameters()]
    opt.step()
    assert all(not torch.equal(p, q) for p, q in zip(head.parameters(), before))


def trainer(m, head, batches, tmp, w, lazy=False, log_every=1, ckpt_every=0):
    opt = make_optimizer(m, {"lr": 3e-3}, extra=head)
    grads, step = [], opt.step

    def spy():                                                # the gradients the optimizer consumes
        grads.append([p.grad.clone() for p in opt_params])
        step()
    opt_params = [*m.parameters(), *(head.parameters() if head is not None else [])]
    opt.step = spy
    tr = Trainer(model=m, optimizer=opt, loader=ListLoader(batches), sched=S.full(10, warmup_steps=4),
                 device="cpu", amp=None, cfg={}, out_dir=str(tmp), grad_accum=2, grad_clip=1e9, log_every=log_every,
                 ckpt_every=ckpt_every, keep_last=1, stable_points=set(), meta={}, lazy_metrics=lazy, mtp=head, mtp_weight=w)
    return tr, grads


@pytest.mark.parametrize("lazy", [False, True])
def test_loss_path_accum2_against_eager_reference(tmp_path, lazy):
    m, head = arm("base_gqa", 1)
    batches, w = packed_batches(), 0.7
    rm, rh = copy.deepcopy(m), copy.deepcopy(head)
    tr, grads = trainer(m, head, batches, tmp_path, w, lazy)
    out = tr.read_metrics(tr.train_step())
    n_sup = sum(int((b["tgt"] != -100).sum()) for b in batches)
    ntp = aux = 0.0
    total, n_aux, wt = 0.0, 0, w * 0.25                       # factor(0) = 1/4 (warmup 4 steps)
    for b in batches:
        logits, _, z = rm(b["idx"], doc=b["doc"], return_hidden=True)
        t2 = torch.tensor([ref_t2(i, t, r) for i, t, r in zip(b["idx"].tolist(), b["tgt"].tolist(), ROWS)])
        h = z @ rh.weight.T
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + rm.cfg.rms_eps) * rh.norm.weight
        l1 = F.cross_entropy(logits.reshape(-1, 97), b["tgt"].reshape(-1), ignore_index=-100, reduction="sum")
        l2 = F.cross_entropy((h @ rm.tok_emb.weight.T).reshape(-1, 97), t2.reshape(-1), ignore_index=-100,
                             reduction="sum")
        total = total + l1 + wt * l2
        ntp, aux, n_aux = ntp + float(l1.detach()), aux + float(l2.detach()), n_aux + int((t2 != -100).sum())
    (total / n_sup).backward()
    ref = [p.grad for p in [*rm.parameters(), *rh.parameters()]]
    assert n_sup > 3 * int((batches[1]["tgt"] != -100).sum()) > 0 and out["mtp_n"] == n_aux
    assert out["mtp_w"] == wt and abs(out["loss"] / (ntp / n_sup) - 1) < 1e-6
    assert abs(out["mtp_loss"] / (aux / n_aux) - 1) < 1e-6
    assert abs(out["gnorm"] / float(torch.stack([g.norm() for g in ref]).norm()) - 1) < 1e-5
    for g, r in zip(grads[0], ref):
        torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-7)


def test_weight_zero_equals_off(tmp_path):
    m, head = arm("base_gqa", 2)
    batches = packed_batches()
    tr1, g1 = trainer(copy.deepcopy(m), head, batches, tmp_path / "a", 0.0)
    tr0, g0 = trainer(copy.deepcopy(m), None, batches, tmp_path / "b", 1.0)
    o1, o0 = tr1.train_step(), tr0.train_step()
    assert o1["loss"] == o0["loss"] and o1["mtp_w"] == 0.0 and "mtp_w" not in o0
    assert len(g1[0]) == len(g0[0]) + 2 and all(torch.equal(a, b) for a, b in zip(g1[0], g0[0]))


@pytest.mark.parametrize("lazy", [False, True])
def test_nonfinite_aux_stops_the_run(tmp_path, lazy):
    m, head = arm("base_gqa", 3)
    with torch.no_grad():
        head.weight.fill_(float("nan"))                       # the next-token loss of step 1 stays finite
    tr, _ = trainer(m, head, packed_batches(), tmp_path, 1.0, lazy, log_every=10, ckpt_every=1)
    with pytest.raises(FloatingPointError):                   # step 1: not a log step, but a checkpoint step
        tr.run(max_steps=2)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".pt")]


@pytest.mark.parametrize("lazy,log_every,ckpt_every,stop", [(False, 3, 0, 1), (True, 3, 0, 3), (True, 10, 4, 4)])
def test_nonfinite_aux_stop_step(tmp_path, lazy, log_every, ckpt_every, stop):
    """Only step 1's aux loss is NaN, and step 1 is neither a log nor a checkpoint step. The NaN is in the value
    only (its gradient is the finite one), so the weights and every later loss stay finite and only the lazy
    flag can remember step 1. Eager stops at step 1; lazy_metrics at the next log step (3) or save (4), unsaved."""
    m, head = arm("base_gqa", 3)
    orig, calls = head.loss_sum, []

    def nan_value_in_step1(*a):
        s = orig(*a)
        calls.append(1)
        return s + (float("nan") - s.detach()) if len(calls) <= 2 else s   # grad_accum 2: step 1's micro-batches
    head.loss_sum = nan_value_in_step1
    tr, _ = trainer(m, head, packed_batches(), tmp_path, 1.0, lazy, log_every=log_every, ckpt_every=ckpt_every)
    msg = f"non-finite loss in steps 1..{stop}" if lazy else "non-finite loss at step 1"
    with pytest.raises(FloatingPointError, match=re.escape(msg) + "$"):
        tr.run(max_steps=6)
    rec = [json.loads(line) for line in open(tr.log_path)][-1]
    assert len(calls) == 2 * stop and tr.step == stop and rec["step"] == stop and rec["error"] == "non-finite loss"
    assert rec.get("after_step") == (0 if lazy else None)
    assert all(torch.isfinite(p).all() for p in [*m.parameters(), *head.parameters()])
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".pt")]
