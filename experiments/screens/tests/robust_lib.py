"""The ROBUSTNESS state (SCREENS.txt AMENDMENT SEEDS-103-105, 2026-10-11): plans/robustness_k5.txt runs every screen arm
at its recorded pick on seeds 103, 104, 105 after batch 1 closed, from its own bundle. The tests that pin the config set
and plans/CURRENT's sequence accept that state only through robust_configs(), which is empty while the plan does not
exist. When it exists: the CLOSED state holds (closed_lib.closed_state); SCREENS.txt records the plan line below; the
plan waits on stage2_replay's end mark, trains exactly SEQUENCE (seed-major; in each seed set the shared BASE line, then
ORDER's screens in order, S005 after its own BASE, S006 at its g 2 pick then its g 1 IND-only run) and ends at its own
mark; each new config passes screens.py check, sits in its screen's configs, and differs from the same run at seed 101
only in name, out_dir and seed. The BASE lines name BASE-DIAG's configs. Also the same-code check (test_screens_robust_
code.py): harness/, data_prep/, tokenizer/, corpus/ and the 5 E2 files a SCREENS run reads must equal 77ab43f's, the tree
E3 notes PART 3 CUDA PARITY tested on CUDA, while plans/CURRENT names robustness_k5. Data and functions; no test here.
"""
import os
import re

import screens_lib as L

ROBUST_PLANS = ["robustness_k5"]          # after CLOSED_PLANS (closed_lib)
ROBUST_PLAN = os.path.join(L.HERE, "plans", "robustness_k5.txt")
WAIT = "wait_mark SCREENS SCREENS BASE REPLAY DONE"
END = "mark SCREENS ROBUSTNESS K5 DONE"
SEEDS = (103, 104, 105)
HEAD = "AMENDMENT SEEDS-103-105 ("
RECORD = "  ROBUSTNESS PLAN: plans/robustness_k5.txt, 36 train lines, 3 seed sets, 33 new runs (BASE lines skipped)"
# per seed set, in ORDER's screen order; {s} is the seed (the amendment's list)
SET = ["base_s{s}", "s001_nogate_g1_s{s}", "s002_novres_g1_s{s}", "s002_noqknorm_g1_s{s}", "s002_nonormscale_g1_s{s}",
       "s003_adamw_e3_r2_s{s}", "s005_base_s{s}", "s005_forget_g1_s{s}", "s004_canonac_g1_s{s}", "s007_smear_g1_s{s}",
       "s006_mtp_g2_s{s}", "s006_mtp_g1_s{s}"]
SEQUENCE = [x.format(s=s) for s in SEEDS for x in SET]
NEW = [r for r in SEQUENCE if not r.startswith("base_")]
SEED_RE = re.compile(r"_s(10[345])$")
CODE_REF = "77ab43f3ca836793d0cde94146c6dc18d1192fc5"     # E3 notes PART 3 CUDA PARITY's "new" tree
CODE_N, CODE_DIGEST = 256, "71433bfb97b0de5907fafd5462576ca21ab01da58842d9708186baaf9419e260"  # git ls-tree -r 77ab43f
# over replay_lib.CODE_DIRS and CODE_E2 (the same paths and recipe as AMENDMENT BASE-REPLAY's guard)


def entry_text() -> str:
    from test_screens_configs import SCREENS_TXT
    return open(SCREENS_TXT).read()


def section(text: str | None = None) -> str:
    lines = (entry_text() if text is None else text).splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith(HEAD))
    j = next((j for j in range(i, len(lines)) if not lines[j].strip()), len(lines))
    return "\n".join(lines[i:j])


def robust_configs(plan: str = ROBUST_PLAN, find=None, check=None, closed=None, text=None) -> set:
    """The configs plans/robustness_k5.txt adds (empty while that plan does not exist); asserts the state above."""
    if not os.path.exists(plan):
        return set()
    from closed_lib import closed_state
    from test_screens_configs import P
    find, check = find or L.find, check or (lambda p: L.check(p, P))
    assert (closed_state() if closed is None else closed), "the robustness stage runs after BATCH 1 CLOSED"
    assert RECORD in section(text), "SCREENS.txt has no AMENDMENT SEEDS-103-105 entry with its plan line"
    lines = open(plan).read().splitlines()
    assert lines[0].startswith("# ") and lines[1] == WAIT and lines[-1] == END, (lines[:2], lines[-1])
    runs = [ln.split()[1] for ln in lines[2:-1] if ln.startswith("train ")]
    assert len(runs) == len(lines) - 3 and runs == SEQUENCE, runs
    for r in SEQUENCE:
        assert len(find(r)) == 1, r
    for r in NEW:
        p = find(r)[0]
        assert check(p) == [], r
        sid = r.split("_")[0].upper()
        assert os.path.dirname(p) == L.config_dir(sid), p
        ref = find(SEED_RE.sub("_s101", r))
        assert len(ref) == 1, r
        f, g = L.flat(L.resolve(p)), L.flat(L.resolve(ref[0]))
        assert L.diff(f, g) == {"name", "out_dir", "seed"} and f["seed"] == int(r[-3:]), (r, L.diff(f, g))
    return set(NEW)
