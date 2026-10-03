"""flag_parity.py and gate_smoke.py on CPU without a training run (the gate itself needs CUDA): the patched
compile_parity builds every arm with the target's flag and engine keys, targets name one arm or a screen's own
BASE, the dynamo log grab sees a real recompile-limit line, an arm that did not carry its flag fails the gate, the
smoke refuses a real run directory, and the patches leave nothing behind for other screens.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_flag_parity.py
"""
import copy
import json
import logging
import os
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import flag_parity as FP  # noqa: E402
import gate_smoke as GS  # noqa: E402
import screens_lib as L  # noqa: E402

WANT = {"S004": ({"canon": "AC", "canon_kernel": 4}, {}), "S005": ({"forget_gate": True}, {"doc_attn": "mask"}),
        "S006": ({}, {"mtp": 1, "mtp_weight": 1.0}), "S007": ({"smear_key": True}, {}),
        "S001": ({"attn_gate": False}, {}), "S002:novres": ({"value_residual": False}, {}),
        "S002:noqknorm": ({"qk_norm": False}, {}), "S002:nonormscale": ({"norm_scaling": False}, {}),
        "S005:base": ({}, {"doc_attn": "mask"}), "S005:forget": ({"forget_gate": True}, {"doc_attn": "mask"})}


@pytest.fixture
def cp(monkeypatch):
    import compile_parity as CP
    for attr in ("write_run", "run_arm"):
        monkeypatch.setattr(CP, attr, getattr(CP, attr))
    monkeypatch.setattr(CP.budget, "solve", CP.budget.solve)
    yield CP
    for h in list(logging.getLogger("torch._dynamo").handlers):
        if isinstance(h, FP.Grab):
            logging.getLogger("torch._dynamo").removeHandler(h)


@pytest.mark.parametrize("target", sorted(WANT))
def test_every_parity_arm_carries_the_flag(target, cp, tmp_path):
    model, train = WANT[target]
    CP, _ = FP.install(target)
    sol = CP.budget.solve(5e6, CP.budget.Constraints())
    cfg = sol.cfg.to_dict()
    assert all(cfg[k] == v for k, v in model.items()) and sol.cfg.d_model == 192 and sol.cfg.n_layers == 8
    if target == "S005:base":
        assert not cfg.get("forget_gate")
    path = CP.write_run(str(tmp_path / "arm0"), str(tmp_path), model=cfg, data={},
                        train={"device": "cuda", "micro_batch": 8, "grad_accum": 2, "compile": "default"})
    c = yaml.safe_load(open(path))
    assert all(c["model"][k] == v for k, v in model.items())
    assert all(c["train"][k] == v for k, v in train.items()) and c["train"]["compile"] == "default"
    assert FP.flag_keys(target)[2] == ({"micro_batch": 8, "grad_accum": 2} if target.startswith("S005") else {})


@pytest.mark.parametrize("target", ["S003", "S002", "S004:base", "S002:nosuch"])
def test_targets_without_one_gateable_arm_are_refused(target):
    with pytest.raises(AssertionError):
        FP.flag_keys(target)       # S003: optim only; S002: three arms, name one; S004: no own BASE


def test_grab_sees_recompile_limit_lines(cp):
    _, grab = FP.install("S007")
    logging.getLogger("torch._dynamo.convert_frame").warning("torch._dynamo hit config.recompile_limit (8)")
    logging.getLogger("torch._dynamo.output_graph").warning("an unrelated line")
    assert len(grab.msgs) == 1 and "recompile_limit" in grab.msgs[0]


def test_grab_sees_a_real_recompile_limit(cp):
    import torch
    _, grab = FP.install("S007")
    torch._dynamo.reset()
    try:
        f = torch.compile(lambda x: x * 2 + 1, backend="eager", dynamic=False)
        for n in range(1, 13):           # 12 static shapes > Dynamo's recompile limit (8)
            f(torch.ones(n))
    finally:
        torch._dynamo.reset()
    assert any("recompile_limit" in m for m in grab.msgs), grab.msgs


def rec(arm, flags, **start):
    return {"arm": arm, "ran": {"model_flags": flags, **start}}


def test_an_arm_without_its_flag_or_mode_fails_the_gate():
    m, t = {"forget_gate": True}, {"doc_attn": "mask"}
    assert FP.carried_ok(rec("eager:det", m), m, t) and FP.carried_ok(rec("default", m, compile="default"), m, t)
    assert not FP.carried_ok(rec("eager", {"forget_gate": None}), m, t)
    assert not FP.carried_ok(rec("eager", m, doc_attn="varlen"), m, t)
    assert not FP.carried_ok(rec("default:det", m), m, t)                    # compiled arm ran eager
    assert not FP.carried_ok(rec("eager", m, compile="default"), m, t)       # eager arm ran compiled
    mt = {"mtp": 1, "mtp_weight": 1.0}
    assert FP.carried_ok(rec("eager", {}, doc_attn="varlen", mtp={"heads": 1}), {}, mt)
    assert not FP.carried_ok(rec("eager", {}, doc_attn="varlen"), {}, mt)


def lean(tmp_path, link_outside: bool):
    s = tmp_path / "lean"
    d = s / "experiments" / "S004_canon" / "configs"
    d.mkdir(parents=True)
    if link_outside:
        (tmp_path / "home").mkdir()
        os.symlink(tmp_path / "home", s / "planck_root")
    else:
        (s / "planck_root").mkdir()
    p = d / "s004_canonac_g1_s1.yaml"
    p.write_text("name: s004_canonac_g1_s1\nout_dir: ../../../planck_root/runs/SCREENS/s004_canonac_g1_s1\n"
                 "runs_jsonl: ../../../planck_root/runs/runs.jsonl\n")
    return str(s), str(p)


def test_smoke_stays_in_its_scratch_root(tmp_path):
    s, p = lean(tmp_path, link_outside=False)
    out, rj = GS.paths(p, s)
    assert out.startswith(os.path.realpath(s)) and rj.startswith(os.path.realpath(s))
    os.makedirs(out)
    open(os.path.join(out, "ckpt_00000153.pt"), "w").close()
    with pytest.raises(SystemExit):
        GS.paths(p, s)                   # an out_dir that holds anything


def test_smoke_refuses_a_planck_root_outside_scratch(tmp_path):
    s, p = lean(tmp_path, link_outside=True)    # the queue's link_root: planck_root -> the real planck home
    with pytest.raises(SystemExit):
        GS.paths(p, s)


def test_gate_verdict_needs_parity_no_limit_line_and_the_flag(tmp_path, monkeypatch):
    out, m = str(tmp_path / "o.json"), {"smear_key": True}

    def fake(runs, passed=True):
        class CP:
            @staticmethod
            def main(argv):
                json.dump({"pass": passed, "runs": runs}, open(argv[argv.index("--out") + 1], "w"))
                return 0 if passed else 1
        monkeypatch.setattr(FP, "install", lambda target: (CP, None))
        return FP.main(["S007", "--out", out])

    dyn = {"unique_graphs": 6, "graph_breaks": 3, "limit_msgs": []}
    good = [{"arm": "eager:det", "dynamo": dict(dyn, unique_graphs=0), "ran": {"model_flags": m, "doc_attn": "varlen"}},
            {"arm": "default:det", "dynamo": dyn, "ran": {"model_flags": m, "doc_attn": "varlen", "compile": "default"}}]
    assert fake(good) == 0 and json.load(open(out))["c1a_gate"]["compiled_graphs"] == [6]
    bad = copy.deepcopy(good)
    bad[1]["dynamo"]["limit_msgs"] = ["torch._dynamo hit config.recompile_limit (8)"]
    assert fake(bad) == 1
    bad = copy.deepcopy(good)
    bad[0]["ran"]["model_flags"] = {"smear_key": None}
    assert fake(bad) == 1 and json.load(open(out))["c1a_gate"]["flag_carried"] is False
    assert fake(good, passed=False) == 1


def test_patches_are_undone_between_tests(cp):
    sol = cp.budget.solve(5e6, cp.budget.Constraints())
    assert not sol.cfg.forget_gate and not sol.cfg.canon and not sol.cfg.smear_key and sol.cfg.attn_gate
