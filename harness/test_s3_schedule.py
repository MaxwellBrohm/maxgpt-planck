"""(d) WSD shapes and decay branches (P-186).

Reference shapes are written out independently (ref_full) and compared step by step.
The end-to-end test is the claim P-186 rests on: a decay branch taken from a trunk's
stable checkpoint trains EXACTLY like a run that planned that length from the start
(same data, same optimizer state, same lr at every step), so the final weights are equal
bit for bit.
"""
from __future__ import annotations

import os

import pytest
import torch

import runio
import schedule as S
import train
from make_fake_data import make
from testutil import read_jsonl, write_run


def ref_full(total: int, w: int, D: int) -> list[float]:
    out = []
    for s in range(total):
        if s < w:
            out.append((s + 1) / w)
        elif s < total - D:
            out.append(1.0)
        else:
            out.append((total - s) / D)
    return out


@pytest.mark.parametrize("total,wf,df", [(1000, 0.01, 0.2), (37, 0.1, 0.3), (5000, 0.01, 0.2),
                                         (10, 0.01, 0.2)])
def test_full_shape_matches_reference(total, wf, df):
    sc = S.full(total, wf, df)
    w, D = max(1, round(total * wf)), round(total * df)
    assert (sc.warmup_steps, sc.decay_steps) == (w, D)
    f = [sc.factor(s) for s in range(total)]
    assert f == pytest.approx(ref_full(total, w, D), abs=1e-12)
    assert max(f) == 1.0 and sc.factor(total) == 0.0 and f[-1] == pytest.approx(1 / D)
    dec = f[total - D:]
    steps = [a - b for a, b in zip(dec, dec[1:])]
    assert all(x == pytest.approx(1 / D) for x in steps)              # linear
    assert [sc.phase(s) for s in (0, w, total - D - 1, total - D)] == \
        ["warmup", "stable", "stable", "decay"]


def test_trunk_never_decays():
    tr = S.trunk(3000, 0.01)
    assert all(tr.factor(s) == 1.0 for s in range(tr.warmup_steps, 3000))
    assert all(tr.phase(s) == "stable" for s in range(tr.warmup_steps, 3000))


@pytest.mark.parametrize("at", [100, 400, 800, 1600])
def test_branch_equals_full_run_of_the_implied_length(at):
    """Branch at s with f = 0.2 has the shape of a full run of length s / (1 - f)."""
    trunk = S.trunk(4000, warmup_steps=40)
    br = S.branch(trunk, at, decay_frac=0.2)
    total = round(at / 0.8)
    assert br.total_steps == total and br.decay_start == at
    ref = ref_full(total, 40, total - at)
    assert [br.factor(s) for s in range(total)] == pytest.approx(ref, abs=1e-12)
    assert all(br.factor(s) == trunk.factor(s) for s in range(at))    # shares the trunk's past
    assert br.factor(at) == 1.0 and br.factor(total - 1) == pytest.approx(1 / (total - at))


def test_branch_explicit_steps_and_refusals():
    trunk = S.trunk(1000, warmup_steps=10)
    assert S.branch(trunk, 500, decay_steps=37).total_steps == 537
    with pytest.raises(AssertionError, match="warmup"):
        S.branch(trunk, 9)
    full = S.full(1000, warmup_steps=10)                            # decay starts at 800
    S.branch(full, 799)                                             # last stable step: fine
    with pytest.raises(AssertionError, match="stable"):
        S.branch(full, 800)
    with pytest.raises(AssertionError, match="branch"):
        S.branch(S.branch(trunk, 500), 520)


def test_tpp_points():
    assert S.branch_steps_from_tpp([500, 2000], 1_000_000, 65_536) == [7630, 30518]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("sched_data")
    make(str(d), docs=120, chats=100)
    return str(d)


def weights(run_dir, name=None):
    out = os.path.join(run_dir, "out")
    return runio.load_checkpoint(os.path.join(out, name) if name else runio.latest_checkpoint(out))


def test_trainer_applies_the_schedule_every_step(tmp_path, data):
    cfg = write_run(str(tmp_path / "r"), data, schedule={"warmup_steps": 4, "decay_frac": 0.25},
                    train={"total_steps": 24})
    assert train.main([cfg]) == 0
    log = read_jsonl(tmp_path / "r" / "out" / "log.jsonl")
    assert [r["lr_factor"] for r in log] == pytest.approx(ref_full(24, 4, 6), abs=1e-6)


def test_decay_branch_is_bitwise_a_planned_run(tmp_path, data):
    """trunk (warmup 4) -> stable ckpt at 30 -> branch decaying 10 steps
    == a full run of 40 steps with warmup 4 and decay 10, bit for bit."""
    trunk = write_run(str(tmp_path / "trunk"), data,
                      schedule={"mode": "trunk", "warmup_steps": 4, "branch_points": [20, 30]},
                      train={"total_steps": 60})
    assert train.main([trunk, "--max-steps", "32"]) == 0
    stable = str(tmp_path / "trunk" / "out" / "stable_00000030.pt")
    assert runio.load_checkpoint(stable)["phase"] == "stable"
    br = write_run(str(tmp_path / "br"), data,
                   schedule={"mode": "branch", "init_from": stable, "decay_steps": 10})
    assert train.main([br]) == 0
    planned = write_run(str(tmp_path / "planned"), data,
                        schedule={"warmup_steps": 4, "decay_frac": 0.25}, train={"total_steps": 40})
    assert train.main([planned]) == 0
    a, b = weights(str(tmp_path / "br")), weights(str(tmp_path / "planned"))
    assert a["step"] == b["step"] == 40
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    la = read_jsonl(tmp_path / "br" / "out" / "log.jsonl")
    lb = read_jsonl(tmp_path / "planned" / "out" / "log.jsonl")[30:]
    assert [(r["step"], r["loss"], r["lr_factor"]) for r in la] == \
        [(r["step"], r["loss"], r["lr_factor"]) for r in lb]
    # the second branch point exists too, and it is still in the stable phase
    assert runio.load_checkpoint(str(tmp_path / "trunk" / "out" / "stable_00000020.pt"))["step"] == 20
