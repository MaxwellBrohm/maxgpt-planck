"""S006 t+2 aux head, run level (train.py; the model and step tests are test_screen_mtp_aux*.py). CPU.
  settings   train.mtp 0 or 1 only, mtp_weight a finite number >= 0; train.py refuses a bad value.
  off        mtp absent == mtp 0 (an mtp_weight given is ignored): 20 CPU steps, bitwise log records, weights,
             optimizer, loader, model_cfg, no "mtp" in the checkpoint, log or start record; and == the pre-flag
             commit's harness (git show; skipped without git or the commit, e.g. in a mutation_check copy).
  weight     logged mtp_w == mtp_weight x the schedule factor (and x the logged lr_factor), warmup to decay.
  resume     20 + resume + 20 == 40 bitwise with mtp on (the head's state too), per-matrix and batched optimizer;
             a checkpoint and a run that disagree on train.mtp are refused. lazy_metrics on == off with mtp on.
  bf16       the autocast path runs on CPU with mtp on.
  record     runs.jsonl start and checkpoint: n_params == count_analytic (deployed, unchanged) and the head's
             training_only_params == count_mtp, apart; the checkpoint's model loads strictly into a plain PlanckLM
             (bpb.py's load) and the trained head sits apart under "mtp"; data_prep/bpb.py on an mtp checkpoint
             (its head scrambled, far from the next-token head) scores the next-token logits (skipped when
             data_prep is not beside the harness, e.g. a mutation_check scratch copy).
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

import runio
import schedule as S
import train
from budget import count_analytic, count_mtp
from config import PlanckConfig
from make_fake_data import make
from model import build_model
from mtp import MTPHead, mtp_settings
from test_s3_resume import state_equal
from test_screen_canon_run import PREFLAG, assert_same_run, git_tree, outcome, run_in
from testutil import read_jsonl, scramble, write_run

HERE = os.path.dirname(os.path.abspath(__file__))
ON = {"mtp": 1, "mtp_weight": 0.5}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("mtp_data")
    make(str(d), docs=150, chats=120)
    return str(d)


@pytest.fixture(scope="module")
def mtp_run(tmp_path_factory, data):
    d = tmp_path_factory.mktemp("mtp_run")
    cfg = write_run(str(d / "on"), data, train={"mtp": 1, "total_steps": 20})
    assert train.main([cfg]) == 0
    return cfg


def starts(d) -> dict:
    return {r["run"]: r for r in read_jsonl(os.path.join(str(d), "runs.jsonl")) if r.get("event") == "start"}


def test_settings_values(tmp_path, data):
    assert mtp_settings({}) == (0, 1.0) and mtp_settings({"mtp": 1, "mtp_weight": 0}) == (1, 0.0)
    for bad in ({"mtp": 2}, {"mtp": -1}, {"mtp": True}, {"mtp": 1.0}, {"mtp": "1"}, {"mtp_weight": -0.1},
                {"mtp_weight": float("nan")}, {"mtp_weight": float("inf")}, {"mtp_weight": True}, {"mtp_weight": "1"}):
        with pytest.raises(ValueError):
            mtp_settings(bad)
    cfg = write_run(str(tmp_path / "bad"), data, train={"mtp": 2, "total_steps": 2})
    with pytest.raises(ValueError, match="train.mtp"):
        train.main([cfg])


def test_off_equals_absent_20_steps(tmp_path, data):
    tr = {"total_steps": 20}
    a = write_run(str(tmp_path / "absent"), data, train=tr)
    b = write_run(str(tmp_path / "off"), data, train={**tr, "mtp": 0, "mtp_weight": 0.3})
    c = write_run(str(tmp_path / "on"), data, train={**tr, **ON})
    assert train.main([a]) == 0 and train.main([b]) == 0 and train.main([c]) == 0
    ra, rb, rc = outcome(a), outcome(b), outcome(c)
    assert_same_run(ra, rb, 20)
    assert not {"mtp", "mtp_params"} & (set(ra[0]) | set(rb[0])) and {"mtp", "mtp_params"} <= set(rc[0])
    st = starts(tmp_path)
    assert "mtp" not in st["absent"] and "mtp" not in st["off"] and st["on"]["mtp"]["heads"] == 1
    assert all("mtp_w" not in r for r in ra[1] + rb[1]) and all("mtp_w" in r for r in rc[1])
    assert [r["loss"] for r in rc[1]] != [r["loss"] for r in ra[1]]    # the probe can see a change


def test_off_equals_preflag_commit_20_steps(tmp_path, data):
    old = tmp_path / "preflag_harness"
    old.mkdir()
    if git_tree(PREFLAG, str(old)) is None:
        pytest.skip(f"git or commit {PREFLAG} unavailable")
    assert "mtp" not in (old / "trainer.py").read_text() + (old / "train.py").read_text()
    a = write_run(str(tmp_path / "preflag"), data, train={"total_steps": 20})
    b = write_run(str(tmp_path / "now_off"), data, train={"total_steps": 20, "mtp": 0})
    run_in(str(old), a)
    run_in(HERE, b)
    assert_same_run(outcome(a), outcome(b), 20)


def test_weight_follows_schedule_factor(tmp_path, data):
    cfg = write_run(str(tmp_path / "w"), data, train=ON)       # 40 steps: warmup 4, decay from step 32
    assert train.main([cfg]) == 0
    log, sched = outcome(cfg)[1], S.full(40, 0.1, 0.2)
    for r in log:
        assert r["mtp_w"] == round(0.5 * sched.factor(r["step"] - 1), 6)
        assert abs(r["mtp_w"] - 0.5 * r["lr_factor"]) <= 1e-6 and r["mtp_n"] > 0 and math.isfinite(r["mtp_loss"])
    assert len(log) == 40 and {r["phase"] for r in log} == {"warmup", "stable", "decay"}
    assert log[0]["mtp_w"] == 0.125 and log[10]["mtp_w"] == 0.5 and log[-1]["mtp_w"] == 0.0625


@pytest.mark.parametrize("optim", [{"lr": 3e-3}, {"lr": 3e-3, "batched": True}], ids=["ref", "batched"])
def test_resume_20_plus_20_equals_40_with_mtp(tmp_path, data, optim):
    over = dict(train=ON, optim=optim)
    straight = write_run(str(tmp_path / "straight"), data, **over)
    split = write_run(str(tmp_path / "split"), data, **over)
    assert train.main([straight]) == 0 and train.main([split, "--max-steps", "20"]) == 0
    assert train.main([split]) == 0
    a, b = outcome(straight), outcome(split)
    assert a[0]["step"] == 40
    assert_same_run(a, b, 40)
    assert state_equal(a[0]["mtp"], b[0]["mtp"]) == []
    assert not torch.equal(a[0]["mtp"]["weight"], torch.eye(32))   # W_mtp left the identity: it trained


def test_resume_refuses_mtp_mismatch(tmp_path, data):
    for first, second in ((ON, {}), ({}, ON)):
        d = str(tmp_path / f"r{len(first)}")
        assert train.main([write_run(d, data, train={**first, "total_steps": 20}), "--max-steps", "5"]) == 0
        with pytest.raises(ValueError, match="train.mtp"):
            train.main([write_run(d, data, train={**second, "total_steps": 20})])


def test_lazy_metrics_equals_eager_with_mtp(tmp_path, data):
    a = write_run(str(tmp_path / "eager"), data, train={**ON, "total_steps": 12})
    b = write_run(str(tmp_path / "lazy"), data, train={**ON, "total_steps": 12, "lazy_metrics": True})
    assert train.main([a]) == 0 and train.main([b]) == 0
    ra, rb = outcome(a), outcome(b)
    assert_same_run(ra, rb, 12)
    assert state_equal(ra[0]["mtp"], rb[0]["mtp"]) == []


def test_bf16_cpu_runs_with_mtp(tmp_path, data):
    cfg = write_run(str(tmp_path / "bf16"), data, train={**ON, "precision": "bf16", "total_steps": 8})
    assert train.main([cfg]) == 0
    log = outcome(cfg)[1]
    assert len(log) == 8 and all(math.isfinite(r["loss"]) and math.isfinite(r["mtp_loss"]) for r in log)


def test_record_counts_and_plain_model_load(mtp_run):
    ck = outcome(mtp_run)[0]
    mc = PlanckConfig.from_dict(ck["model_cfg"])
    st = starts(os.path.dirname(os.path.dirname(mtp_run)))["on"]
    assert st["n_params"] == ck["n_params"] == count_analytic(mc)["total"]
    assert st["mtp"] == {"heads": 1, "weight": 1.0, "training_only_params": count_mtp(mc, 1)}
    assert ck["mtp_params"] == count_mtp(mc, 1) == 32 * 32 + 32
    plain = build_model(mc)
    plain.load_state_dict(ck["model"])                           # strict, as data_prep/bpb.py load_model
    head = MTPHead(mc.d_model, mc.rms_eps)
    head.load_state_dict(ck["mtp"])
    idx = torch.randint(0, mc.vocab_size, (2, 32), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits, _, z = plain(idx, return_hidden=True)
        assert torch.equal(plain(idx)[0], logits)
    assert not torch.equal(head.weight, torch.eye(32))          # the head trained; the model state holds none of it


def test_bpb_scores_next_token_logits(mtp_run, tmp_path):
    dp = os.path.normpath(os.path.join(HERE, "..", "data_prep"))
    if not os.path.isfile(os.path.join(dp, "bpb.py")):
        pytest.skip("data_prep/bpb.py is not beside the harness (a mutation_check scratch copy)")
    if dp not in sys.path:
        sys.path.append(dp)                                       # appended: harness module names win
    import bpb
    ck = runio.load_checkpoint(runio.latest_checkpoint(os.path.join(os.path.dirname(mtp_run), "out")))
    head = scramble(MTPHead(32), 3)                               # an aux head far from the next-token head
    ck["mtp"] = head.state_dict()
    torch.save(ck, tmp_path / "ck.pt")
    model, mc, ck = bpb.load_model(str(tmp_path / "ck.pt"), "cpu")
    g = torch.Generator().manual_seed(2)
    items = [(torch.randint(8, mc.vocab_size, (n,), generator=g).tolist(), 4, n, None) for n in (30, 17, 9)]
    nll, ntok = bpb.nll_items(model, items, "cpu", 0, 4096)
    for (ids, n_ctx, _, _), got in zip(items, nll):
        x, tgt = torch.tensor([ids]), torch.tensor(ids[n_ctx:])
        with torch.no_grad():
            logits, _, z = model(x, return_hidden=True)
            ref = F.cross_entropy(logits[0, n_ctx - 1:-1].double(), tgt, reduction="sum")
            aux = F.cross_entropy(head.logits(z, model.lm_head)[0, n_ctx - 1:-1].double(), tgt, reduction="sum")
        assert abs(got - float(ref)) <= 1e-6 * float(ref) and abs(float(aux) - float(ref)) > 1e-3 * float(ref)
