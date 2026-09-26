"""S005 forget gate, run level and plumbing (model.forget_gate; semantics, gradient, precision, leak and doc=None
tests are in test_screen_forget_gate.py). CPU.
  config     only true/false accepted; to_dict drops the off flag (an off run's model_cfg is the pre-flag dict).
  off        forget_gate absent == false: 20 CPU steps, bitwise log records, weights, optimizer, loader,
             model_cfg; and == the pre-flag commit's harness (git show; skipped without git or the commit, e.g. in
             a mutation_check scratch copy unless GIT_DIR is set).
  optimizer  w and b get non-zero gradients at init (w = 0); both in the AdamW 'scalar' group (scalar_lr, no
             decay), not NorMuon; one step moves them.
  resume     20 + resume + 20 == 40 bitwise with the gate on, per-matrix and batched optimizer.
  engine     doc_attn varlen refused with the gate on (set_doc_attn, the model's forward, train.py startup).
  also       the startup self-test, budget.py counts (5M: 5,010,133 -> 5,014,765), KV decode == full forward.
"""
from __future__ import annotations

import os

import pytest
import torch

import docattn
import selftest
import train
from budget import count_analytic
from config import PlanckConfig
from count_params import build_and_count
from decode import KVDecoder
from make_fake_data import make
from model import build_model
from optim import make_optimizer
from test_s3_docattn import ROWS
from test_s3_leak import DOC, T
from test_screen_canon_run import PREFLAG, assert_same_run, git_tree, outcome, run_in
from test_screen_forget_gate import ARMS, arm_model, gen
from testutil import tiny, write_run

HERE = os.path.dirname(os.path.abspath(__file__))
B5 = dict(vocab_size=8192, d_model=192, n_layers=8, n_heads=3, n_kv_heads=3, head_dim=64,
          mlp_hidden=488, seq_len=2048)                       # SCREENS C1 BASE shape


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("fg_data")
    make(str(d), docs=150, chats=120)
    return str(d)


def test_config_values_and_dict():
    for bad in (1, 0, "true", None):
        with pytest.raises(AssertionError):
            tiny(forget_gate=bad)
    off, on = tiny(), tiny(forget_gate=True)
    assert set(off.to_dict()) == set(on.to_dict()) - {"forget_gate"}
    assert on.to_dict()["forget_gate"] is True and PlanckConfig.from_dict(on.to_dict()) == on
    assert PlanckConfig.from_dict(off.to_dict()) == tiny(forget_gate=False) == off


def test_off_equals_absent_20_steps(tmp_path, data):
    tr = {"total_steps": 20}
    a = write_run(str(tmp_path / "absent"), data, train=tr)
    b = write_run(str(tmp_path / "off"), data, model={"forget_gate": False}, train=tr)
    c = write_run(str(tmp_path / "on"), data, model={"forget_gate": True}, train=tr)
    assert train.main([a]) == 0 and train.main([b]) == 0 and train.main([c]) == 0
    ra, rb, rc = outcome(a), outcome(b), outcome(c)
    assert_same_run(ra, rb, 20)
    assert "forget_gate" not in ra[0]["model_cfg"] and rc[0]["model_cfg"]["forget_gate"] is True
    assert set(rc[0]["model"]) - set(ra[0]["model"]) and not set(ra[0]["model"]) - set(rc[0]["model"])
    assert [r["loss"] for r in rc[1]] != [r["loss"] for r in ra[1]]    # the probe can see a change


def test_off_equals_preflag_commit_20_steps(tmp_path, data):
    old = tmp_path / "preflag_harness"
    old.mkdir()
    if git_tree(PREFLAG, str(old)) is None:
        pytest.skip(f"git or commit {PREFLAG} unavailable")
    assert "forget_gate" not in (old / "config.py").read_text()
    a = write_run(str(tmp_path / "preflag"), data, train={"total_steps": 20})
    b = write_run(str(tmp_path / "now_off"), data, model={"forget_gate": False}, train={"total_steps": 20})
    run_in(str(old), a)
    run_in(HERE, b)
    assert_same_run(outcome(a), outcome(b), 20)


def test_gradient_reaches_gate_and_scalar_group():
    torch.manual_seed(0)
    m = build_model(tiny(forget_gate=True, n_loops=2))
    idx, tgt = (torch.randint(0, 97, (len(ROWS), T), generator=torch.Generator().manual_seed(s)) for s in (3, 4))
    m(idx, tgt, torch.tensor(ROWS))[1].backward()
    gs = {n: p for n, p in m.named_parameters() if ".forget_" in n}
    assert len(gs) == 2 * m.cfg.n_unique
    assert all(p.grad is not None and (p.grad != 0).all() for p in gs.values())
    opt = make_optimizer(m, {"lr": 1e-2, "embed_lr": 5e-3, "scalar_lr": 2e-3, "weight_decay": 0.1})
    g = {x["name"]: x for x in opt.param_groups}
    ids = {name: {id(p) for p in x["params"]} for name, x in g.items()}
    assert all(id(p) in ids["scalar"] and id(p) not in ids["matrix"] for p in gs.values())
    assert g["scalar"]["weight_decay"] == 0.0 and g["scalar"]["base_lr"] == 2e-3
    assert not g["scalar"]["use_muon"]
    before = {n: p.detach().clone() for n, p in gs.items()}
    opt.step()
    assert all(not torch.equal(p, before[n]) for n, p in gs.items())


@pytest.mark.parametrize("optim", [{"lr": 3e-3}, {"lr": 3e-3, "batched": True}], ids=["ref", "batched"])
def test_resume_20_plus_20_equals_40_with_forget_gate(tmp_path, data, optim):
    over = dict(model={"forget_gate": True}, optim=optim)
    straight = write_run(str(tmp_path / "straight"), data, **over)
    split = write_run(str(tmp_path / "split"), data, **over)
    assert train.main([straight]) == 0 and train.main([split, "--max-steps", "20"]) == 0
    assert train.main([split]) == 0
    a, b = outcome(straight), outcome(split)
    assert a[0]["step"] == 40 and a[0]["model_cfg"]["forget_gate"] is True
    assert_same_run(a, b, 40)
    w = [v for n, v in a[0]["model"].items() if n.endswith(".forget_w")]
    assert w and all(v.any() for v in w)                     # w left its zero init: the gate trained


def test_train_refuses_varlen(tmp_path, data):
    cfg = write_run(str(tmp_path / "varlen"), data, model={"forget_gate": True},
                    train={"doc_attn": "varlen", "total_steps": 2})
    with pytest.raises(ValueError, match="forget_gate"):
        train.main([cfg])


def test_varlen_refused():
    m, off = arm_model("fg"), build_model(tiny())
    docattn.set_doc_attn(off, "varlen", "cuda")               # flag off: allowed as before
    with pytest.raises(ValueError, match="forget_gate"):
        docattn.set_doc_attn(m, "varlen", "cuda")
    assert m.doc_attn == "mask"
    m.doc_attn = "varlen"                                     # set directly: the forward refuses too
    idx = torch.randint(0, 97, (1, T), generator=gen(9))
    for doc in (None, torch.tensor([DOC])):
        with pytest.raises(RuntimeError, match="forget_gate"):
            m(idx, doc=doc)


def test_startup_selftest_passes():
    torch.manual_seed(0)
    out = selftest.run_selftests(build_model(tiny(forget_gate=True)), "cpu")
    assert len(out) == 12 and all(v < 1e-3 for v in out.values()), out


@pytest.mark.parametrize("arm", list(ARMS))
def test_budget_counts_forget_gate(arm):
    cfg = tiny(**ARMS[arm])
    got = count_analytic(cfg)
    assert got == build_and_count(cfg, "cpu") == build_and_count(cfg, "meta")
    extra = cfg.n_heads * (cfg.d_model + 1) * cfg.n_unique
    assert got["total"] - count_analytic(cfg.replace(forget_gate=False))["total"] == extra


def test_budget_5m_base_and_arm():
    assert count_analytic(PlanckConfig(**B5))["total"] == 5_010_133
    on = PlanckConfig(**B5, forget_gate=True)
    assert count_analytic(on)["total"] == build_and_count(on, "meta")["total"] == 5_014_765
    assert 5_014_765 - 5_010_133 == 8 * 3 * (192 + 1) and 5_014_765 / 5_010_133 - 1 < 0.02


@pytest.mark.parametrize("prefill", [1, 2, 5])
@pytest.mark.parametrize("arm", list(ARMS))
def test_decode_matches_full_forward(arm, prefill):
    m = arm_model(arm, 11)
    ids = torch.randint(0, 97, (20,), generator=gen(5)).tolist()
    with torch.no_grad():
        full = m(torch.tensor([ids]))[0][0]
    dec = KVDecoder(m)
    got = [dec.forward(ids[:prefill], "cpu")] + [dec.forward([t], "cpu") for t in ids[prefill:]]
    torch.testing.assert_close(torch.cat(got), full, atol=2e-5, rtol=1e-4)
