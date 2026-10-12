"""The CLOSED state (SCREENS.txt BATCH 1 CLOSED, 2026-10-11; E3 Part 3, cuts audit S6): plans/CURRENT names
batch1_closed, a plan that waits on stage2_replay's end mark and stops, training nothing, so CURRENT no longer names
stage2_replay and the E3 Part 3 data_seed change may land in harness/ (tests/test_screens_replay_code.py checks the code
only while CURRENT names stage2_replay). The tests that pin plans/CURRENT's sequence accept that state only through
closed_state(), which is False while plans/batch1_closed.txt does not exist. When it exists: the BASE-REPLAY state holds
(replay_lib.replay_configs is not empty), the plan is a comment, the wait on stage2_replay's end mark and "stop", with no
train or mark line, and SCREENS.txt holds the entry with its record line. Data and functions; no test here.
"""
import os

import screens_lib as L

CLOSED_PLANS = ["batch1_closed"]          # after REPLAY_PLANS (replay_lib)
CLOSED_PLAN = os.path.join(L.HERE, "plans", "batch1_closed.txt")
HEAD = "BATCH 1 CLOSED ("
RECORD = "  CLOSED: plans/CURRENT = batch1_closed (stop), after stage2_replay's mark SCREENS BASE REPLAY DONE"


def entry_text() -> str:
    from test_screens_configs import SCREENS_TXT
    return open(SCREENS_TXT).read()


def section(text: str | None = None) -> str:
    lines = (entry_text() if text is None else text).splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith(HEAD))
    j = next((j for j in range(i, len(lines)) if not lines[j].strip()), len(lines))
    return "\n".join(lines[i:j])


def closed_state(plan: str = CLOSED_PLAN, replay=None, text=None) -> bool:
    """True when plans/batch1_closed.txt exists and the state above holds (asserted); False while it does not exist."""
    if not os.path.exists(plan):
        return False
    from replay_lib import END, replay_configs
    rep = replay_configs() if replay is None else replay
    assert rep, "the batch closes after AMENDMENT BASE-REPLAY's plan: no BASE-REPLAY state"
    lines = open(plan).read().splitlines()
    assert len(lines) == 3 and lines[0].startswith("# "), lines
    assert lines[1:] == ["wait_mark SCREENS " + END[len("mark "):], "stop"], lines
    assert RECORD in section(text), "SCREENS.txt has no BATCH 1 CLOSED entry with its record line"
    return True
