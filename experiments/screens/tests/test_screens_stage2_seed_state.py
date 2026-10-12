"""The stage2_seeds state (plans/stage2_seeds.txt, screens.py seeds --stage 2 after the stage 2 picks; the stage 2
autopilot's design, AUTOPILOT.txt, written by SCREENS.txt S006 EXTENSION RESULT / STAGE 2 SEEDS LAUNCH CHECK): the
tests that pin the config set, plans/CURRENT's sequence and S005's configs accept that state only through
test_screens_configs.stage2_seed_configs(). These tests run that function on scratch plans and configs written by
screens.py seeds --stage 2 (a monkeypatched config tree, as test_screens_refusals.py does), in every state of the repo:
each passes its cuts explicitly, so the repo's own record (recorded_cuts) does not enter. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage2_seed_state.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
from test_screens_configs import recorded_cuts, seeds2_wait, stage2_seed_configs  # noqa: E402
from test_screens_refusals import scratch  # noqa: E402,F401  (the scratch config tree fixture)

WAIT, MARK = seeds2_wait()[len("wait_mark "):], "SCREENS STAGE 2 SEEDS DONE"
ARMS = ("s005_forget", "s004_canonac", "s007_smear", "s006_mtp")
CUT_LINE = "  CUT BY ORDER'S CAP RULE BEFORE THE STAGE 2 SEED SETS: S006 (its seed runs and IND runs; bet not run)\n"


def build(tmp_path, screens, picks: dict, wait: bool = True, cut=()) -> str:
    """screens.py seeds --stage 2 into the scratch tree, with the wait line and any cut screens."""
    screens.main(["seeds", "--stage", "2"] + [f"--pick={k}={v:g}" for k, v in picks.items()] +
                 (["--wait", WAIT] if wait else []) + [f"--cut={c}" for c in cut])
    return str(tmp_path / "plans" / "stage2_seeds.txt")


def picks(*cut, **g) -> dict:
    out = {"S005.forget": 1.0, "S004.canonac": 1.0, "S007.smear": 1.0, "S006.mtp": 1.0}
    out.update({k.replace("_", "."): v for k, v in g.items()})
    return {k: v for k, v in out.items() if k.split(".")[0] not in cut}


def test_no_plan_adds_nothing(tmp_path):
    assert stage2_seed_configs(str(tmp_path / "stage2_seeds.txt"), cuts=set()) == set()


def test_the_wait_line_is_the_end_mark_of_the_plan_before():
    assert seeds2_wait() == "wait_mark SCREENS SCREENS S006 EXTENSION DONE"


def test_the_generators_plan_is_accepted(scratch):
    tmp_path, screens = scratch
    got = stage2_seed_configs(build(tmp_path, screens, picks()), cuts=set())
    assert got == {f"{a}_g1_s{s}" for a in ARMS for s in (101, 102)}


def test_matched_lr_ind_runs_are_accepted_for_s006_and_s007_only(scratch):
    tmp_path, screens = scratch
    got = stage2_seed_configs(build(tmp_path, screens, picks(S007_smear=2.0, S006_mtp=0.5)), cuts=set())
    assert {"s007_smear_g2_s101", "s007_smear_g1_s101", "s006_mtp_g0.5_s102", "s006_mtp_g1_s102"} <= got
    assert len(got) == 12


def test_a_config_that_fails_check_is_refused(scratch):
    tmp_path, screens = scratch
    plan = build(tmp_path, screens, picks())
    p = tmp_path / "S004_canon" / "configs" / "s004_canonac_g1_s102.yaml"
    p.write_text(p.read_text().replace("canon_kernel: 4", "canon_kernel: 3"))
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan, cuts=set())


def test_a_plan_without_the_wait_line_or_on_the_old_mark_is_refused(scratch):
    tmp_path, screens = scratch
    with pytest.raises(AssertionError):
        stage2_seed_configs(build(tmp_path, screens, picks(), wait=False), cuts=set())
    plan = build(tmp_path, screens, picks())
    txt = open(plan).read()
    open(plan, "w").write(txt.replace(f"wait_mark {WAIT}\n", "wait_mark SCREENS SCREENS STAGE 2 SELECTION DONE\n"))
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan, cuts=set())


def test_a_run_outside_the_seed_sets_is_refused(scratch):
    tmp_path, screens = scratch
    plan = build(tmp_path, screens, picks())
    txt = open(plan).read()
    for extra in ("s004_canonac_g1_s1", "s001_nogate_g1_s101"):
        open(plan, "w").write(txt.replace("train s006_mtp_g1_s102\n", f"train s006_mtp_g1_s102\ntrain {extra}\n"))
        with pytest.raises(AssertionError):
            stage2_seed_configs(plan, cuts=set())


def test_two_picks_for_one_arm_are_refused_unless_the_ind_pair(scratch):
    tmp_path, screens = scratch
    build(tmp_path, screens, picks(S004_canonac=0.5))
    plan = build(tmp_path, screens, picks())
    txt = open(plan).read()
    open(plan, "w").write(txt.replace("train s004_canonac_g1_s101\n", "train s004_canonac_g1_s101\ntrain "
                                      "s004_canonac_g0.5_s101\n").replace("train s004_canonac_g1_s102\n",
                                      "train s004_canonac_g1_s102\ntrain s004_canonac_g0.5_s102\n"))
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan, cuts=set())
    open(plan, "w").write(txt.replace("train s004_canonac_g1_s102\n", ""))      # one seed missing
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan, cuts=set())


def test_a_recorded_cut_leaves_its_screen_out(scratch):
    """The generator with --cut S006 writes no S006 config and no S006 line; the plan is accepted with the cut
    recorded, refused without it (a screen silently left out), and the header names the cut."""
    tmp_path, screens = scratch
    plan = build(tmp_path, screens, picks("S006"), cut=("S006",))
    got = stage2_seed_configs(plan, cuts={"S006"})
    assert got == {f"{a}_g1_s{s}" for a in ARMS[:3] for s in (101, 102)}
    assert not os.listdir(tmp_path / "S006_mtp_aux" / "configs") and "s006" not in open(plan).read()
    assert open(plan).readline().startswith("# SCREENS stage 2 seed sets, seed-major (ORDER 2); cut by ORDER's cap "
                                            "rule (SCREENS.txt), no run: S006 (screens.py)")
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan, cuts=set())


def test_a_recorded_cut_whose_runs_stay_is_refused(scratch):
    tmp_path, screens = scratch
    plan = build(tmp_path, screens, picks(S006_mtp=2.0))
    with pytest.raises(AssertionError):
        stage2_seed_configs(plan, cuts={"S006"})


def test_a_cut_the_rule_may_not_make_is_refused(scratch):
    """S005 and S004 are never cut by the rule (S004 only while P-148 is the must-run pick): neither the generator
    nor the acceptance takes such a cut, recorded or not; a cut of a stage 1 screen is refused too."""
    tmp_path, screens = scratch
    for c in ("S005", "S004", "S003", "BASE"):
        with pytest.raises(SystemExit):
            build(tmp_path, screens, picks(c), cut=(c,))
    with pytest.raises(SystemExit):
        build(tmp_path, screens, picks("S006"), cut=("S006", "S006"))
    with pytest.raises(SystemExit):                       # a pick for the cut screen is refused too
        build(tmp_path, screens, picks(), cut=("S006",))
    plan = build(tmp_path, screens, picks())
    txt = open(plan).read()
    open(plan, "w").write("".join(ln for ln in txt.splitlines(True) if not ln.startswith("train s005_")))
    for cuts in ({"S005"}, {"S005", "S006"}, set()):
        with pytest.raises(AssertionError):
            stage2_seed_configs(plan, cuts=cuts)


def test_the_record_is_the_entry_line_only():
    assert recorded_cuts(CUT_LINE) == {"S006"} and recorded_cuts(CUT_LINE.replace("S006", "S006, S007")) == {"S006", "S007"}
    for bad in (CUT_LINE.lstrip(), "  " + CUT_LINE, CUT_LINE.replace("STAGE 2", "STAGE 1"), CUT_LINE.lower(),
                "  CUT RULE: nothing is cut now.\n"):
        assert recorded_cuts(bad) == set(), bad
