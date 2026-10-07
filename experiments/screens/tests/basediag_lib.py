"""The BASE-DIAG state (SCREENS.txt AMENDMENT BASE-DIAG, 2026-10-06): plans/stage2_base_diag.txt runs the shared BASE at
seeds 103, 104 and 105 after S006's seed sets, from its own bundle. The tests that pin the config set and plans/CURRENT's
sequence accept that state only through base_diag_configs(), which is empty while the plan does not exist. When it
exists: the S006 reinstated state holds (reinstate_lib.s006_seed_configs is not empty); SCREENS.txt records exactly
these three runs as added by the amendment; the plan waits on stage2_s006_seeds' end mark and ends at its own mark; it
trains base_s103, base_s104 and base_s105 in that order and nothing else; each config sits in experiments/screens/
configs, passes screens.py check and differs from base_s101 only in name, out_dir and seed. Data and functions; no test.
"""
import os
import re

import screens_lib as L


DIAG_PLANS = ["stage2_base_diag"]        # after REINSTATED_PLANS (reinstate_lib)
DIAG_PLAN = os.path.join(L.HERE, "plans", "stage2_base_diag.txt")
END = "mark SCREENS BASE DIAG DONE"
RUNS = ["base_s103", "base_s104", "base_s105"]
REC_RE = re.compile(r"^  BASE RUNS ADDED BY AMENDMENT BASE-DIAG: (base_s\d+(?:, base_s\d+)*) \(", re.M)


def recorded_diag(text: str | None = None) -> list:
    """The BASE runs SCREENS.txt records as added by AMENDMENT BASE-DIAG (the line, at the entry's 2-space indent, that
    the amendment writes), in the recorded order."""
    from test_screens_configs import SCREENS_TXT
    text = open(SCREENS_TXT).read() if text is None else text
    return [r for m in REC_RE.finditer(text) for r in m[1].split(", ")]


def base_diag_configs(plan: str = DIAG_PLAN, find=None, check=None, recorded=None, s006_plan=None) -> set:
    """The configs plans/stage2_base_diag.txt adds (empty while that plan does not exist); asserts the state above."""
    if not os.path.exists(plan):
        return set()
    from reinstate_lib import S006_PLAN, s006_seed_configs
    from test_screens_configs import P
    find, check = find or L.find, check or (lambda p: L.check(p, P))
    s006_plan = s006_plan or S006_PLAN
    assert s006_seed_configs(s006_plan), "BASE-DIAG runs after S006's reinstated seed sets: no S006 plan"
    rec = recorded_diag() if recorded is None else list(recorded)
    assert rec == RUNS, rec
    wait = "wait_mark SCREENS " + open(s006_plan).read().splitlines()[-1][len("mark "):]
    lines = open(plan).read().splitlines()
    assert lines[0].startswith("# ") and lines[1] == wait and lines[-1] == END, lines
    runs = [ln.split()[1] for ln in lines[2:-1] if ln.startswith("train ")]
    assert len(runs) == len(lines) - 3 and runs == RUNS, lines
    ref = L.flat(L.resolve(find("base_s101")[0]))
    for r in runs:
        assert len(find(r)) == 1 and check(find(r)[0]) == [], r
        f = L.flat(L.resolve(find(r)[0]))
        assert L.diff(f, ref) == {"name", "out_dir", "seed"} and f["seed"] == int(r[len("base_s"):]), r
        assert os.path.dirname(find(r)[0]) == L.config_dir(None), find(r)
    return set(runs)
