"""SCREENS.txt BATCH 1 CLOSED (2026-10-11; E3 Part 3, cuts audit S6): plans/CURRENT names batch1_closed, whose plan
trains nothing; the acceptance (closed_lib.closed_state) on scratch states; the same-code guard of stage2_replay no
longer reads the code; the entry's record line, its statements and its numbers against the files. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_closed.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import closed_lib as C  # noqa: E402
import robust_lib as RB  # noqa: E402  (AMENDMENT SEEDS-103-105 moved CURRENT on, later)
import privacy  # noqa: E402
import replay_lib as R  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_replay_code import current  # noqa: E402


def flat(s: str) -> str:
    return " ".join(s.split())


def test_current_is_the_closed_plan_and_it_trains_nothing():
    later = RB.ROBUST_PLANS if RB.robust_configs() else []      # AMENDMENT SEEDS-103-105 moved CURRENT on, later
    assert [current()] == (later or C.CLOSED_PLANS) and C.CLOSED_PLANS == ["batch1_closed"] and C.closed_state()
    lines = [ln.strip() for ln in open(C.CLOSED_PLAN) if ln.strip() and not ln.startswith("#")]
    assert lines == ["wait_mark SCREENS SCREENS BASE REPLAY DONE", "stop"]
    assert not [ln for ln in lines if ln.split()[0] in ("train", "mark")]
    assert open(R.REPLAY_PLAN).read().splitlines()[-1] == R.END == "mark SCREENS BASE REPLAY DONE"


def test_the_replay_code_guard_reads_no_code_now():
    assert current() not in R.REPLAY_PLANS          # test_screens_replay_code.py's first test skips by design


def write(tmp_path, body: str) -> str:
    p = tmp_path / "batch1_closed.txt"
    p.write_text(body)
    return str(p)


GOOD = "# fixture\nwait_mark SCREENS SCREENS BASE REPLAY DONE\nstop\n"


@pytest.mark.parametrize("body,why", [
    ("# fixture\nwait_mark SCREENS SCREENS BASE REPLAY DONE\ntrain base_s101\nstop\n", "a train line"),
    ("# fixture\nwait_mark SCREENS SCREENS BASE REPLAY DONE\nmark SCREENS X\n", "a mark in place of stop"),
    ("# fixture\nwait_mark SCREENS SCREENS BASE DIAG DONE\nstop\n", "a wait on another mark"),
    ("# fixture\nstop\n", "no wait"),
    ("wait_mark SCREENS SCREENS BASE REPLAY DONE\nstop\n", "no comment line"),
])
def test_acceptance_refuses_other_plans(tmp_path, body, why):
    with pytest.raises(AssertionError):
        C.closed_state(write(tmp_path, body), replay={"x"})


def test_acceptance_needs_the_replay_state_and_the_entry(tmp_path):
    p = write(tmp_path, GOOD)
    assert C.closed_state(p, replay={"x"}) and not C.closed_state(str(tmp_path / "absent.txt"))
    with pytest.raises(AssertionError, match="no BASE-REPLAY state"):
        C.closed_state(p, replay=set())
    with pytest.raises(AssertionError, match="no BATCH 1 CLOSED entry"):
        C.closed_state(p, replay={"x"}, text=C.entry_text().replace(C.RECORD, "  CLOSED: something else"))
    with pytest.raises(StopIteration):
        C.closed_state(p, replay={"x"}, text=C.entry_text().replace("\n" + C.HEAD, "\nBATCH ONE CLOSED ("))


def test_the_entry():
    sec = C.section()
    f = flat(sec)
    assert sec.startswith(C.HEAD + "2026-10-11") and C.RECORD in sec
    assert "2026-10-10 22:28:09" in f and "2026-10-10 23:51:19" in f        # the C6 queue's DONE, the replay's end
    assert "SCREENS_BASE_REPLAY_DONE" in f and "ede95ce" in f
    assert "never to stage 1 or stage 2" in f and "sqrt(2) x sigma_seed" in f
    assert "harness/test_data_seed.py" in f and "93bc10b" in f and "plan says stop" in f
    assert "test_screens_replay_code.py" in f and "boot.d/40_screens.sh" in f
    assert not privacy.hits(sec) and "\u2014" not in sec and "\u2013" not in sec
    txt = C.entry_text()
    assert txt.index("\nSTAGE 2 VERDICTS (2026-10-11") < txt.index("\n" + C.HEAD)        # appended after the verdicts
    assert open(os.path.join(L.HERE, "plans", "CURRENT")).read().count("BATCH 1 CLOSED") == 1
