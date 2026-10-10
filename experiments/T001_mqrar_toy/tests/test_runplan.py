"""T001 run-order tests: C0's two rules fire once each, LR picks and their one extension, seed-major verdict order,
then the track runs, then DONE. Fake run files stand in for train.py's JSON."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runplan as R  # noqa: E402
from model import DIAG_ARMS, PICK_ARMS  # noqa: E402

ALL = PICK_ARMS + DIAG_ARMS


def fake(out, sp, dev):
    """dev: acc for every grid n (a float), a dict n -> acc, or a pair (acc, outvoted acc) of those; the outvoted
    accuracy equals acc unless given."""
    acc, ov = dev if isinstance(dev, tuple) else (dev, dev)
    acc = acc if isinstance(acc, dict) else {n: acc for n in sp["grid"]}
    ov = ov if isinstance(ov, dict) else {n: ov for n in sp["grid"]}
    js = dict(arm=sp["arm"], seed=sp["seed"], lr=sp["lr"], grid=sp["grid"], diverged=False,
              eval={"dev": {"mqrar": {str(n): dict(acc=acc.get(n, 0.0), acc_ov=ov.get(n, 0.0))
                                      for n in sp["grid"]}}})
    with open(os.path.join(out, sp["name"] + ".json"), "w") as f:
        json.dump(js, f)


def drive(out, dev_fn, limit=400):
    """Runs the plan to the end, answering every run with dev_fn(spec); returns the specs in order."""
    seen = []
    for _ in range(limit):
        sp, phase = R.plan(out)
        if sp is None:
            return seen
        seen.append((phase, sp))
        fake(out, sp, dev_fn(sp))
    raise AssertionError("plan did not finish")


def test_c0_no_rule(tmp_path):
    seen = drive(str(tmp_path), lambda sp: 0.7 if sp["lr"] == 1e-3 else 0.5)
    assert seen[0][0] == "C0" and seen[0][1]["name"] == "base_mqrar_s0_lr0.001_st6000_n96"
    lr_runs = [s for p, s in seen if p == "LR"]
    assert [s["arm"] for s in lr_runs][:2] == ["base", "base"]           # base's 1e-3 run is the C0 run
    assert len(lr_runs) == 3 * len(ALL) - 1
    ver = [(s["seed"], s["arm"]) for p, s in seen if p == "VERDICT"]
    assert ver == [(s, a) for s in (1, 2, 3, 4, 5) for a in ALL]          # seed-major, arm order
    assert all(s["lr"] == 1e-3 for p, s in seen if p == "VERDICT")
    trk = [(s["seed"], s["arm"], s["family"]) for p, s in seen if p == "TRACK"]
    assert trk == [(s, a, "track") for s in (1, 2, 3) for a in ("base", "gdn", "canon")]
    assert R.plan(str(tmp_path)) == (None, "DONE")


def test_c0_rule_a_fires_once(tmp_path):
    seen = drive(str(tmp_path), lambda sp: {8: 0.4} if sp["arm"] == "base" else 0.5)
    c0 = [s for p, s in seen if p == "C0"]
    assert [s["steps"] for s in c0] == [6000, 12000]                     # rerun at 12,000, never a third time
    assert all(s["steps"] == 12000 for p, s in seen if p != "C0")


def test_c0_rule_b_extends_grid_once(tmp_path):
    seen = drive(str(tmp_path), lambda sp: 0.95)
    c0 = [s for p, s in seen if p == "C0"]
    assert [(s["n_hi"], tuple(s["grid"])) for s in c0] == [(96, tuple(R.GRID0)), (160, tuple(R.GRID0) + (128, 160))]
    assert all(s["n_hi"] == 160 for p, s in seen if p != "C0")


def test_c0_both_rules(tmp_path):
    def dev(sp):
        return {8: 0.4} if sp["steps"] == 6000 else 0.95
    seen = drive(str(tmp_path), dev)
    c0 = [(s["steps"], s["n_hi"]) for p, s in seen if p == "C0"]
    assert c0 == [(6000, 96), (12000, 96), (12000, 160)]


@pytest.mark.parametrize("best,ext,final", [(3e-4, 1e-4, 1e-4), (3e-3, 1e-2, 3e-3), (1e-3, None, 1e-3)])
def test_lr_extension(tmp_path, best, ext, final):
    def dev(sp):
        if sp["arm"] != "gdn":
            return 0.5
        score = {best: 0.8, ext: 0.9 if final == ext else 0.1}
        return score.get(sp["lr"], 0.3)
    seen = drive(str(tmp_path), dev)
    lrs = [s["lr"] for p, s in seen if p == "LR" and s["arm"] == "gdn"]
    assert lrs == [3e-4, 1e-3, 3e-3] + ([ext] if ext else [])
    assert {s["lr"] for p, s in seen if p == "VERDICT" and s["arm"] == "gdn"} == {final}


def test_train_args_round_trip(tmp_path):
    sp, _ = R.plan(str(tmp_path))
    a = R.train_args(sp, str(tmp_path)).split()
    assert a[a.index("--arm") + 1] == "base" and a[a.index("--lr") + 1] == "0.001"
    assert a[a.index("--grid") + 1] == "8,16,24,32,48,64,96" and a[a.index("--steps") + 1] == "6000"
    assert a[a.index("--out") + 1].endswith("base_mqrar_s0_lr0.001_st6000_n96.json")


def test_c0_rule_b_needs_outvoted_accuracy_too(tmp_path):
    """A1: base at acc 0.95 everywhere but 0.85 on outvoted queries at n 96 (a vote-counting model) does not widen."""
    seen = drive(str(tmp_path), lambda sp: (0.95, {n: 0.95 if n < 96 else 0.85 for n in sp["grid"]}))
    c0 = [s for p, s in seen if p == "C0"]
    assert [s["n_hi"] for s in c0] == [96] and all(s["n_hi"] == 96 for p, s in seen)


def test_lr_pick_reads_outvoted_accuracy(tmp_path):
    """A1: canon's 3e-3 votes (acc 0.95, outvoted 0.1); its 1e-3 tracks updates (0.8, 0.8) and is picked."""
    def dev(sp):
        if sp["arm"] != "canon":
            return 0.5
        return {3e-4: (0.6, 0.6), 1e-3: (0.8, 0.8), 3e-3: (0.95, 0.1)}[sp["lr"]]
    seen = drive(str(tmp_path), dev)
    assert {s["lr"] for p, s in seen if p == "VERDICT" and s["arm"] == "canon"} == {1e-3}
