"""S007 smeared keys, run level (model.smear_key; semantics, isolation, leak, doc=None, budget and decode tests are
in test_screen_smear_key.py). CPU.
  config     only true/false accepted; to_dict drops the off flag (an off run's model_cfg is the pre-flag dict).
  off        smear_key absent == false: 20 CPU steps, bitwise log records, weights, optimizer, loader, model_cfg;
             and == the pre-flag commit's harness (git show; skipped without git or the commit, e.g. in a
             mutation_check scratch copy unless GIT_DIR is set).
  optimizer  every alpha gets a non-zero gradient at its zero init; alphas are in the AdamW 'scalar' group
             (scalar_lr, no decay), not NorMuon; one step moves them.
  resume     20 + resume + 20 == 40 bitwise with the flag on, per-matrix and batched optimizer.
  engine     varlen is allowed (the smear happens before attention): set_doc_attn accepts it.
"""
from __future__ import annotations

import os

import pytest
import torch

import docattn
import train
from config import PlanckConfig
from make_fake_data import make
from model import build_model
from optim import make_optimizer, split_params
from test_s3_docattn import ROWS
from test_s3_leak import T
from test_screen_canon_run import PREFLAG, assert_same_run, git_tree, outcome, run_in
from testutil import tiny, write_run

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("smear_data")
    make(str(d), docs=150, chats=120)
    return str(d)


def test_config_values_and_dict():
    for bad in (1, 0, "true", None):
        with pytest.raises(AssertionError):
            tiny(smear_key=bad)
    off, on = tiny(), tiny(smear_key=True)
    assert set(off.to_dict()) == set(on.to_dict()) - {"smear_key"}
    assert on.to_dict()["smear_key"] is True and PlanckConfig.from_dict(on.to_dict()) == on
    assert PlanckConfig.from_dict(off.to_dict()) == tiny(smear_key=False) == off


def test_off_equals_absent_20_steps(tmp_path, data):
    tr = {"total_steps": 20}
    a = write_run(str(tmp_path / "absent"), data, train=tr)
    b = write_run(str(tmp_path / "off"), data, model={"smear_key": False}, train=tr)
    c = write_run(str(tmp_path / "on"), data, model={"smear_key": True}, train=tr)
    assert train.main([a]) == 0 and train.main([b]) == 0 and train.main([c]) == 0
    ra, rb, rc = outcome(a), outcome(b), outcome(c)
    assert_same_run(ra, rb, 20)
    assert "smear_key" not in ra[0]["model_cfg"] and rc[0]["model_cfg"]["smear_key"] is True
    extra = set(rc[0]["model"]) - set(ra[0]["model"])
    assert extra and all(n.endswith(".smear_alpha") for n in extra) and set(ra[0]["model"]) <= set(rc[0]["model"])
    assert [r["loss"] for r in rc[1]] != [r["loss"] for r in ra[1]]    # the probe can see a change


def test_off_equals_preflag_commit_20_steps(tmp_path, data):
    old = tmp_path / "preflag_harness"
    old.mkdir()
    if git_tree(PREFLAG, str(old)) is None:
        pytest.skip(f"git or commit {PREFLAG} unavailable")
    assert "smear_key" not in (old / "config.py").read_text()
    a = write_run(str(tmp_path / "preflag"), data, train={"total_steps": 20})
    b = write_run(str(tmp_path / "now_off"), data, model={"smear_key": False}, train={"total_steps": 20})
    run_in(str(old), a)
    run_in(HERE, b)
    assert_same_run(outcome(a), outcome(b), 20)


def test_gradient_reaches_alpha_and_scalar_group():
    torch.manual_seed(0)
    m = build_model(tiny(smear_key=True, n_kv_heads=2, n_loops=2))
    idx, tgt = (torch.randint(0, 97, (len(ROWS), T), generator=torch.Generator().manual_seed(s)) for s in (3, 4))
    m(idx, tgt, torch.tensor(ROWS))[1].backward()
    al = {n: p for n, p in m.named_parameters() if ".smear_" in n}
    assert len(al) == m.cfg.n_unique and all(p.numel() == 2 for p in al.values())
    assert all(p.grad is not None and (p.grad != 0).all() for p in al.values())   # every KV head, at alpha = 0
    assert {n for n, _ in split_params(m)["scalar"]} >= set(al)
    opt = make_optimizer(m, {"lr": 1e-2, "embed_lr": 5e-3, "scalar_lr": 2e-3, "weight_decay": 0.1})
    g = {x["name"]: x for x in opt.param_groups}
    ids = {name: {id(p) for p in x["params"]} for name, x in g.items()}
    assert all(id(p) in ids["scalar"] and id(p) not in ids["matrix"] for p in al.values())
    assert g["scalar"]["weight_decay"] == 0.0 and g["scalar"]["base_lr"] == 2e-3
    assert not g["scalar"]["use_muon"]
    opt.step()
    assert all(p.any() for p in al.values())


@pytest.mark.parametrize("optim", [{"lr": 3e-3}, {"lr": 3e-3, "batched": True}], ids=["ref", "batched"])
def test_resume_20_plus_20_equals_40_with_smear(tmp_path, data, optim):
    over = dict(model={"smear_key": True}, optim=optim)
    straight = write_run(str(tmp_path / "straight"), data, **over)
    split = write_run(str(tmp_path / "split"), data, **over)
    assert train.main([straight]) == 0 and train.main([split, "--max-steps", "20"]) == 0
    assert train.main([split]) == 0
    a, b = outcome(straight), outcome(split)
    assert a[0]["step"] == 40 and a[0]["model_cfg"]["smear_key"] is True
    assert_same_run(a, b, 40)
    al = [v for n, v in a[0]["model"].items() if n.endswith(".smear_alpha")]
    assert len(al) == 2 and all(v.any() for v in al)          # alpha left its zero init: the smear trained


def test_varlen_allowed():
    m = build_model(tiny(smear_key=True))
    docattn.set_doc_attn(m, "varlen", "cuda")                 # not refused (unlike S005's bias)
    assert m.doc_attn == "varlen"
