"""The stage2_seeds state (plans/stage2_seeds.txt, written by the stage 2 autopilot after the stage 2 picks; its record
is experiments/screens/AUTOPILOT.txt): the tests that pin the config set, plans/CURRENT's sequence and S005's configs
accept that state only through test_screens_configs.stage2_seed_configs(). These tests run that function on scratch
plans and configs written by screens.py seeds --stage 2 (a monkeypatched config tree, as test_screens_refusals.py
does), in every state of the repo. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage2_seed_state.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
from test_screens_configs import stage2_seed_configs  # noqa: E402
from test_screens_refusals import scratch  # noqa: E402,F401  (the scratch config tree fixture)

WAIT, MARK = "SCREENS SCREENS STAGE 2 SELECTION DONE", "SCREENS STAGE 2 SEEDS DONE"
ARMS = ("s005_forget", "s004_canonac", "s007_smear", "s006_mtp")


def build(tmp_path, screens, picks: dict, rewrite: bool = True) -> str:
    """screens.py seeds --stage 2 into the scratch tree; then (as the autopilot does) screens.plan rewrites the plan
    with the same runs and mark and a wait line on the selection mark."""
    screens.main(["seeds", "--stage", "2"] + [f"--pick={k}={v:g}" for k, v in picks.items()])
    plan = str(tmp_path / "plans" / "stage2_seeds.txt")
    runs = [ln.split()[1] for ln in open(plan) if ln.startswith("train ")]
    if rewrite:
        screens.plan("stage2_seeds", runs, "stage 2 seed sets (test)", MARK, WAIT)
    return plan


def picks(**g) -> dict:
    out = {"S005.forget": 1.0, "S004.canonac": 1.0, "S007.smear": 1.0, "S006.mtp": 1.0}
    out.update({k.replace("_", "."): v for k, v in g.items()})
    return out


def test_no_plan_adds_nothing(tmp_path):
    assert stage2_seed_configs(str(tmp_path / "stage2_seeds.txt")) == set()


def test_the_generators_plan_is_accepted(scratch):
    tmp_path, screens = scratch
    got = stage2_seed_configs(build(tmp_path, screens, picks()))
    assert got == {f"{a}_g1_s{s}" for a in ARMS for s in (101, 102)}


def test_matched_lr_ind_runs_are_accepted_for_s006_and_s007_only(scratch):
    tmp_path, screens = scratch
    got = stage2_seed_configs(build(tmp_path, screens, picks(S007_smear=2.0, S006_mtp=0.5)))
    assert {"s007_smear_g2_s101", "s007_smear_g1_s101", "s006_mtp_g0.5_s102", "s006_mtp_g1_s102"} <= got
    assert len(got) == 12


def test_a_config_that_fails_check_is_refused(scratch):
    tmp_path, screens = scratch
    plan = build(tmp_path, screens, picks())
    p = tmp_path / "S004_canon" / "configs" / "s004_canonac_g1_s102.yaml"
    p.write_text(p.read_text().replace("canon_kernel: 4", "canon_kernel: 3"))
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan)


def test_a_plan_without_the_wait_line_is_refused(scratch):
    tmp_path, screens = scratch
    with pytest.raises(AssertionError):
        stage2_seed_configs(build(tmp_path, screens, picks(), rewrite=False))


def test_a_run_outside_the_seed_sets_is_refused(scratch):
    tmp_path, screens = scratch
    plan = build(tmp_path, screens, picks())
    txt = open(plan).read()
    for extra in ("s004_canonac_g1_s1", "s001_nogate_g1_s101"):
        open(plan, "w").write(txt.replace("train s006_mtp_g1_s102\n", f"train s006_mtp_g1_s102\ntrain {extra}\n"))
        with pytest.raises(AssertionError):
            stage2_seed_configs(plan)


def test_two_picks_for_one_arm_are_refused_unless_the_ind_pair(scratch):
    tmp_path, screens = scratch
    build(tmp_path, screens, picks(S004_canonac=0.5))
    plan = build(tmp_path, screens, picks())
    txt = open(plan).read()
    open(plan, "w").write(txt.replace("train s004_canonac_g1_s101\n", "train s004_canonac_g1_s101\ntrain "
                                      "s004_canonac_g0.5_s101\n").replace("train s004_canonac_g1_s102\n",
                                      "train s004_canonac_g1_s102\ntrain s004_canonac_g0.5_s102\n"))
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan)
    open(plan, "w").write(txt.replace("train s004_canonac_g1_s102\n", ""))      # one seed missing
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan)
