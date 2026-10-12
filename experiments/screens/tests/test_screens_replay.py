"""SCREENS AMENDMENT BASE-REPLAY (2026-10-10; cuts audit S7, row 29): the run-name rule (screens_lib NAME_RE and
BASE_REPLAYS; check's C2 and C6 for _rep / _hookoff names) and its refusals; screens_hours.slot; the 3 configs and the
plan as screens.py writes them; plans/CURRENT; the acceptance (replay_lib.replay_configs) on scratch states; the
hook-off run trains bitwise like its replay on a tiny CPU model. The entry: test_screens_replay_entry.py; the
same-code guard: test_screens_replay_code.py. No verdict, no model above unit-test size, no stage 2 result read.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_replay.py
"""
import os, sys  # noqa: E401

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import screens_hours as H, screens_lib as L, replay_lib as R  # noqa: E401,E402
from basediag_lib import RUNS as DIAG_RUNS  # noqa: E402
from closed_lib import CLOSED_PLANS, closed_state  # noqa: E402  (BATCH 1 CLOSED, later)
from robust_lib import ROBUST_PLANS, robust_configs  # noqa: E402  (AMENDMENT SEEDS-103-105, later)
from test_screens_configs import P  # noqa: E402
from test_screens_refusals import refusals, scratch  # noqa: E402,F401  (scratch: the scratch config tree fixture)
from test_screens_smoke import fake, run_tiny  # noqa: E402,F401  (fake: the tiny data fixture)

HEADER = "SCREENS AMENDMENT BASE-REPLAY (SCREENS.txt): base_s101 replayed twice with its hook-off run between them, after BASE-DIAG"
REG = "a _rep / _hookoff name must be a registered shared BASE replay"
NULL = "must set eval.induction null"
HOOK_ON = "does not carry BASE's eval.induction hook unchanged"


def test_name_rule_and_registry():
    assert L.BASE_REPLAYS == R.KIND and list(R.KIND) == R.RUNS
    for n in R.RUNS:
        m = L.NAME_RE.match(n)
        assert (m["sid"], m["arm"], m["seed"], m["diag"]) == (None, "base", "101", n.split("_")[-1])
        assert H.slot(n) == ("BASE", None)           # on top of the BASE slots, never base_s101's
    for n in ("base_s101_rep0", "base_s101_rep10", "base_s101_hook", "base_rep1", "s003_adamw_e3_r1_trunk_rep1",
              "base_s101_hookoff_rep1"):
        assert not L.NAME_RE.match(n), n
    for p in L.all_configs():                       # every older name parses as before: no suffix
        n = os.path.basename(p)[:-5]
        assert (L.NAME_RE.match(n)["diag"] is not None) == (n in R.KIND), n


def test_a_replay_takes_no_base_slot():
    runs = {k: v for k, v in H.load()[0].items() if k != "base_s101"}      # base_s101's slot left open
    t0, t = H.tally(P, runs), H.tally(P, {**runs, "base_s101_rep1": 0.3, "base_s101_hookoff": 0.3})
    assert {"base_s101_rep1", "base_s101_hookoff"} <= set(t["extra"]) and ("BASE", "base", 101) not in t["seen"]
    assert t["queued"] == t0["queued"] and t["n_queued"]["BASE"] == 1
    assert t["measured"]["BASE"] == pytest.approx(t0["measured"]["BASE"] + 0.6)


CASES = [   # (repo config, edit of its text, what the refusal says)
    ("base_s101", lambda t: t.replace("base_s101", "base_s102_rep1").replace("seed: 101", "seed: 102"), REG),
    ("base_s101", lambda t: t.replace("base_s101", "base_s101_rep3"), REG),
    ("s004_canonac_g1_s101", lambda t: t.replace("s004_canonac_g1_s101", "s004_canonac_g1_s101_rep1"), REG),
    ("s005_base_s101", lambda t: t.replace("s005_base_s101", "s005_base_s101_rep1"), REG),
    ("base_s101_rep1", lambda t: t + "eval: {induction: null}\n", HOOK_ON),
    ("base_s101_hookoff", lambda t: t.replace("eval: {induction: null}\n", ""), NULL),
    ("base_s101_hookoff", lambda t: t.replace("{induction: null}", "{induction: {every: 0}}"), NULL),
    ("base_s101_hookoff", lambda t: t.replace("{induction: null}", "{induction: {}}"), NULL),
    ("base_s101_hookoff", lambda t: t.replace("{induction: null}", "{induction: false}"), NULL),
    ("base_s101", lambda t: t + "eval: {induction: null}\n", HOOK_ON),
    ("s007_smear_g1_s101", lambda t: t + "eval: {induction: null}\n", HOOK_ON),
    ("base_s101_rep1", lambda t: t.replace("optim: {lr: 0.003", "optim: {lr: 0.006"), "differs from its BASE in optim.lr"),
    ("base_s101_hookoff", lambda t: t + "train: {log_every: 5}\n", "differs from its BASE in train.log_every"),
    ("base_s101_rep2", lambda t: t.replace("seed: 101", "seed: 102"), "seed key 102 does not match the name"),
]


@pytest.mark.parametrize("name,edit,want", CASES, ids=[f"{c[0]}:{i}" for i, c in enumerate(CASES)])
def test_check_refuses(tmp_path, name, edit, want):
    assert want in refusals(tmp_path, name, edit)


@pytest.mark.parametrize("extra,name,old,new", [
    ({"base_s101_hookoff": "replay"}, "base_s101_hookoff", "", ""),                  # suffix and kind disagree
    ({"base_s101_rep1": "hook-off"}, "base_s101_rep1", "", ""),
    ({"s004_canonac_g1_s101_rep1": "replay"}, "s004_canonac_g1_s101", "s004_canonac_g1_s101", "s004_canonac_g1_s101_rep1"),
    ({"s005_base_s101_rep1": "replay"}, "s005_base_s101", "s005_base_s101", "s005_base_s101_rep1"),
])
def test_registry_alone_does_not_open_the_rule(tmp_path, monkeypatch, extra, name, old, new):
    monkeypatch.setattr(L, "BASE_REPLAYS", {**L.BASE_REPLAYS, **extra})
    assert REG in refusals(tmp_path, name, lambda t: t.replace(old, new) if old else t)


def test_registered_names_pass_and_the_registry_is_the_gate(tmp_path, monkeypatch):
    for n in R.RUNS:
        assert refusals(tmp_path, n, lambda t: t) == ""
    edit = lambda t: t.replace("base_s101", "base_s102_rep1").replace("seed: 101", "seed: 102")  # noqa: E731
    monkeypatch.setattr(L, "BASE_REPLAYS", {**L.BASE_REPLAYS, "base_s102_rep1": "replay"})
    assert refusals(tmp_path, "base_s101", edit) == ""


def test_configs_and_plan_are_the_generators(scratch):
    repo = {r: open(os.path.join(L.HERE, "configs", r + ".yaml")).read() for r in R.RUNS + ["base_s101"]}
    plan_txt = open(R.REPLAY_PLAN).read()
    tmp_path, screens = scratch
    (tmp_path / "screens" / "configs" / "base_s101.yaml").write_text(repo["base_s101"])
    assert [screens.replay(r, P) for r in R.RUNS] == R.RUNS
    for r in R.RUNS:
        assert open(tmp_path / "screens" / "configs" / f"{r}.yaml").read() == repo[r], r
        assert repo[r] == repo["base_s101"].replace("SCREENS BASE (C1: E3 arm A, C1-b eager)", "SCREENS BASE " + (
            "hook-off of base_s101 (AMENDMENT BASE-REPLAY; eval.induction off)" if R.KIND[r] == "hook-off" else
            "replay of base_s101 (AMENDMENT BASE-REPLAY)")).replace("base_s101\n", f"{r}\n") + (
            "eval: {induction: null}\n" if R.KIND[r] == "hook-off" else "")
    screens.plan("stage2_replay", R.RUNS, HEADER, "SCREENS BASE REPLAY DONE", "SCREENS SCREENS BASE DIAG DONE")
    assert open(tmp_path / "plans" / "stage2_replay.txt").read() == plan_txt
    for bad in ("base_s102_rep1", "base_s101", "s004_canonac_g1_s101_rep1"):
        with pytest.raises(SystemExit, match="is not a registered BASE replay"):
            screens.replay(bad, P)


def test_plan_current_and_acceptance():
    lines = open(R.REPLAY_PLAN).read().splitlines()
    assert lines == [f"# {HEADER} (screens.py). train = screens check, preflight, train, score",
                     "wait_mark SCREENS SCREENS BASE DIAG DONE"] + [f"train {r}" for r in R.RUNS] + [R.END]
    diag = open(os.path.join(L.HERE, "plans", "stage2_base_diag.txt")).read().splitlines()
    assert diag[-1] == "mark SCREENS BASE DIAG DONE"
    cur = [ln.strip() for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")]
    later = CLOSED_PLANS if closed_state() else []          # BATCH 1 CLOSED moved CURRENT on, later
    later = (ROBUST_PLANS if robust_configs() else []) or later      # and AMENDMENT SEEDS-103-105 after it
    assert cur == (later or R.REPLAY_PLANS) and R.REPLAY_PLANS == ["stage2_replay"] and R.replay_configs() == set(R.RUNS)


def variant(tmp_path, name, edit):
    p = tmp_path / f"{name}.txt"
    p.write_text(edit(open(R.REPLAY_PLAN).read()))
    return str(p)


def copy_run(tmp_path, run, old, new, mp=None):
    """Scratch copies of the 3 configs, `run` with one edit, for a find() stub (check stubbed by the caller). With mp the
    scratch dir also stands for experiments/screens/configs, so only the content check can refuse the state."""
    paths = {"base_s101": L.find("base_s101")[0]}
    for r in R.RUNS:
        txt = open(L.find(r)[0]).read()
        assert r != run or txt.count(old) == 1
        txt = txt.replace(old, new) if r == run else txt
        (tmp_path / f"{r}.yaml").write_text(txt.replace("extends: screens_base.yaml", f"extends: {L.BASE_YAML}"))
        paths[r] = str(tmp_path / f"{r}.yaml")
    if mp:
        orig = L.config_dir
        mp.setattr(L, "config_dir", lambda sid: str(tmp_path) if sid is None else orig(sid))
    return lambda r: [paths[r]] if r in paths else L.find(r)


def stub(run, old, new, inside=True):   # the BASE-DIAG state is passed in: base_diag_configs reads the real dirs
    return lambda tmp, mp: {"diag": set(DIAG_RUNS), "find": copy_run(tmp, run, old, new, mp if inside else None),
                            "check": lambda p: []}


BAD = {   # name: (edit of the plan text, keyword arguments or a function of (tmp_path, monkeypatch) giving them)
    "runs_out_of_order": (lambda t: t.replace("train base_s101_rep1\ntrain base_s101_hookoff\n",
                                              "train base_s101_hookoff\ntrain base_s101_rep1\n"), {}),
    "a_replay_dropped": (lambda t: t.replace("train base_s101_rep2\n", ""), {}),
    "a_base_run_added": (lambda t: t.replace(R.END, "train base_s103\n" + R.END), {}),
    "waits_on_the_s006_mark": (lambda t: t.replace("BASE DIAG DONE\n", "STAGE 2 S006 SEEDS DONE\n", 1), {}),
    "runs_past_its_mark": (lambda t: t + "train base_s101_rep1\n", {}),
    "ends_at_the_diag_mark": (lambda t: t.replace(R.END, "mark SCREENS BASE DIAG DONE"), {}),
    "not_recorded": (lambda t: t, {"recorded": []}),
    "recorded_two_runs": (lambda t: t, {"recorded": R.RUNS[:2]}),
    "registry_widened": (lambda t: t, {"registry": {**R.KIND, "base_s102_rep1": "replay"}}),
    "a_config_fails_check": (lambda t: t, {"check": lambda p: ["refused"] if "rep2" in p else []}),
    "no_diag_plan": (lambda t: t, lambda tmp, mp: {"diag_plan": str(tmp / "absent.txt")}),
    "a_replay_at_another_lr": (lambda t: t, stub("base_s101_rep2", "{lr: 0.003,", "{lr: 0.006,")),
    "a_replay_with_the_hook_off": (lambda t: t, stub("base_s101_rep1", "scalar_lr: 0.003}\n",
                                                      "scalar_lr: 0.003}\neval: {induction: null}\n")),
    "the_hook_off_run_with_the_hook": (lambda t: t, stub("base_s101_hookoff", "eval: {induction: null}\n", "")),
    "the_hook_off_run_set_false": (lambda t: t, stub("base_s101_hookoff", "{induction: null}", "{induction: false}")),
    "a_config_outside_the_shared_dir": (lambda t: t, stub("base_s101_rep1", "seed: 101", "seed: 101", inside=False)),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_acceptance_refuses(tmp_path, monkeypatch, name):
    edit, kw = BAD[name]
    kw = kw(tmp_path, monkeypatch) if callable(kw) else kw
    with pytest.raises(AssertionError):
        R.replay_configs(variant(tmp_path, name, edit), **kw)


def test_acceptance_takes_the_repo_plan(tmp_path, monkeypatch):
    assert R.replay_configs(variant(tmp_path, "same", lambda t: t)) == set(R.RUNS) and len(BAD) == 16
    find = copy_run(tmp_path, "base_s101_rep1", "seed: 101", "seed: 101", monkeypatch)    # unedited copies: taken
    assert R.replay_configs(variant(tmp_path, "same", lambda t: t), find=find, check=lambda p: [],
                            diag=set(DIAG_RUNS)) == set(R.RUNS)
    assert R.replay_configs(str(tmp_path / "absent.txt")) == set()
    line = "  BASE RUNS ADDED BY AMENDMENT BASE-REPLAY: base_s101_rep1, base_s101_hookoff, base_s101_rep2 (3 runs)\n"
    assert R.recorded_replay(line) == R.RUNS and R.recorded_replay(" " + line) == []
    assert R.recorded_replay(line.replace(" BY", ",")) == []


def test_hook_off_trains_bitwise_like_its_replay(fake, tmp_path):
    rep = run_tiny(L.find("base_s101_rep1")[0], fake, str(tmp_path / "rep"))
    off = run_tiny(L.find("base_s101_hookoff")[0], fake, str(tmp_path / "off"))
    assert rep["diag"] and off["diag"] is None and "induction" in rep["start"] and "induction" not in off["start"]
    assert [x["loss"] for x in rep["log"]] == [x["loss"] for x in off["log"]]
    a, b = rep["ck"]["model"], off["ck"]["model"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
