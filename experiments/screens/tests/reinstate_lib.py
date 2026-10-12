"""The S006 reinstatement state (SCREENS.txt AMENDMENT S006-REINSTATE, 2026-10-06): plans/stage2_s006_seeds.txt runs
S006's seed sets after stage2_seeds, from its own bundle. The tests that pin the config set and plans/CURRENT's
sequence accept that state only through s006_seed_configs(), which is empty while the plan does not exist. When it
exists: S006 is recorded both as cut by ORDER's cap rule and as reinstated; stage2_seeds still leaves S006 out and its
end mark is this plan's wait line; the plan ends at its own mark; it runs S006's one arm at one g pick on seeds 101 and
102, seed 101 first, each seed's C6 IND-only run at g 1 right after that seed's run when the pick is not g 1; every
config passes screens.py check. Data and functions; no test here.
"""
import os
import re

import screens_lib as L

REINSTATED_PLANS = ["stage2_s006_seeds"]       # after LATER_PLANS (test_screens_configs.py)
S006_PLAN = os.path.join(L.HERE, "plans", "stage2_s006_seeds.txt")
END = "mark SCREENS STAGE 2 S006 SEEDS DONE"
REIN_RE = re.compile(r"^  REINSTATED BY AMENDMENT S006-REINSTATE: (S00\d(?:, S00\d)*) \(", re.M)
RUN_RE = re.compile(r"^s006_mtp_g([0-9.]+)_s(10[12])$")


def recorded_reinstated(text: str | None = None) -> set:
    """The screens SCREENS.txt records as reinstated (the line, at the entry's 2-space indent, that the amendment
    writes)."""
    from test_screens_configs import SCREENS_TXT
    text = open(SCREENS_TXT).read() if text is None else text
    return {s for m in REIN_RE.finditer(text) for s in m[1].split(", ")}


def s006_seed_configs(plan: str = S006_PLAN, find=None, check=None, cuts=None, reinstated=None, seeds_plan=None) -> set:
    """The configs plans/stage2_s006_seeds.txt adds (empty while that plan does not exist); asserts the state above."""
    if not os.path.exists(plan):
        return set()
    from test_screens_configs import P, STAGE2_SEEDS, recorded_cuts, stage2_seed_configs
    find, check = find or L.find, check or (lambda p: L.check(p, P))
    cuts = recorded_cuts() if cuts is None else set(cuts)
    rein = recorded_reinstated() if reinstated is None else set(reinstated)
    assert "S006" in cuts and rein == {"S006"}, (cuts, rein)
    seeds_plan = seeds_plan or STAGE2_SEEDS
    s2 = stage2_seed_configs(seeds_plan, find=find, check=check, cuts=cuts)
    assert not [r for r in s2 if r.startswith("s006_")], s2
    wait = "wait_mark SCREENS " + open(seeds_plan).read().splitlines()[-1][len("mark "):]
    lines = open(plan).read().splitlines()
    assert lines[0].startswith("# ") and lines[1] == wait and lines[-1] == END, lines
    runs = [ln.split()[1] for ln in lines[2:-1] if ln.startswith("train ")]
    assert len(runs) == len(lines) - 3 and runs and RUN_RE.match(runs[0]), lines
    g = RUN_RE.match(runs[0])[1]
    want = [r for s in (101, 102) for r in [f"s006_mtp_g{g}_s{s}"] + ([f"s006_mtp_g1_s{s}"] if float(g) != 1 else [])]
    assert runs == want, (runs, want)
    for r in runs:
        assert len(find(r)) == 1 and check(find(r)[0]) == [], r
    return set(runs)
