"""chain_waiter.sh (SCREENS.txt AMENDMENT BASE-DIAG): its arguments, and its decision on fixture queue logs, marks and stop
files for the chain stage2_seeds (earlier) -> stage2_s006_seeds (after) -> stage2_base_diag (next): an earlier plan still
last, a clean end, the mark with the queue still running, a kill, a 2 h gpu.lock wait, a run failure, STOP, another plan
or commit, a double start. The loop and the launch: test_chain_waiter_run.py. No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_chain_waiter.py
"""
import os, subprocess, sys  # noqa: E401

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import screens_lib as L  # noqa: E402


W = os.path.join(L.HERE, "chain_waiter.sh")
REF6 = "3227c6011be2f76e4401551f8f8da20fcfbe44d0"          # the commit S006's queue runs from (planck_s006.bundle)
REF2 = "d21df3de007146c2d619ba7c3f46f264286d318c"          # the commit stage2_seeds runs from (planck.bundle)
MARK6 = "SCREENS STAGE 2 S006 SEEDS DONE"
ARGS = ["base_diag", "stage2_s006_seeds", REF6, MARK6, "stage2_base_diag", "planck_diag.bundle", f"stage2_seeds@{REF2}"]
T = "2026-10-07 02:20:10 "
S2 = [f"2026-10-06 20:38:01 queue start: exp SCREENS, plan stage2_seeds.txt (sha256 ed36242e5b8b45c2), code commit {REF2}",
      "2026-10-06 20:38:03 gpu.lock held; GPU memory in use 600 MiB", "2026-10-06 20:38:04 train s005_base_s101.yaml"]
S2END = ["2026-10-07 00:40:00 MARK SCREENS STAGE 2 SEEDS DONE", "2026-10-07 00:40:00 plan finished: stage2_seeds.txt"]
START = f"2026-10-07 00:42:30 queue start: exp SCREENS, plan stage2_s006_seeds.txt (sha256 57ac8f3e5911720b), code commit {REF6}"
RUN = ["2026-10-07 00:42:30 seen SCREENS: SCREENS STAGE 2 SEEDS DONE", "2026-10-07 00:42:32 waiting for gpu.lock",
       "2026-10-07 00:42:32 gpu.lock held; GPU memory in use 600 MiB", "2026-10-07 00:42:33 train s006_mtp_g2_s101.yaml"]
LAST = ["2026-10-07 02:19:00 done s006_mtp_g1_s102 rc 0", "2026-10-07 02:19:00 score s006_mtp_g1_s102/final_00007630"]
CLEAN = [T + "MARK " + MARK6, T + "plan finished: stage2_s006_seeds.txt"]
LOCKWAIT = ["2026-10-07 03:00:00 waiting for gpu.lock", "2026-10-07 05:00:00 gpu.lock not free after 2 h",
            "2026-10-07 05:00:00 queue stops at: train s006_mtp_g1_s102"]
FAIL = [T + "crash s006_mtp_g2_s102 rc 1: resuming once from its last checkpoint",
        T + "scoring failed for final_00007630 (see score.err)", T + "queue stops at: train s006_mtp_g2_s102"]
STOPPED = [T + "STOP present: queue stops", T + "queue stops at: train s006_mtp_g1_s102"]
GAP = [T + "GAP s006_mtp_g2_s102: second crash (rc 1)"]
DSTART = T + "queue start: exp SCREENS, plan stage2_base_diag.txt (sha256 0123456789abcdef), code commit " + "b" * 40
MID = S2 + S2END + [START] + RUN
CASES = {   # name: (log lines, mark file, files under ~/planck, running, the decision's first word, a phrase in it)
    "earlier_plan_running": (S2, False, (), "1", "WAIT", "queue or launcher runs"),
    "earlier_plan_ended_s006_not_started": (S2 + S2END, False, (), "0", "WAIT", "stage2_seeds at d21df3de0071, earlier"),
    "earlier_plan_failed": (S2 + FAIL, False, (), "0", "WAIT", "earlier in the chain than stage2_s006_seeds"),
    "earlier_plan_at_another_commit": ([S2[0].replace(REF2, "c" * 40)] + S2[1:] + S2END, False, (), "0", "GIVEUP",
                                       "not stage2_s006_seeds at 3227c6011be2"),
    "clean_end": (MID + LAST + CLEAN, True, (), "0", "LAUNCH", "stage2_s006_seeds ended cleanly"),
    "clean_end_with_a_gap_run": (MID + GAP + LAST + CLEAN, True, (), "0", "LAUNCH", "ended cleanly"),
    "clean_end_queue_still_running": (MID + LAST + CLEAN, True, (), "1", "WAIT", "queue or launcher runs"),
    "killed_no_end_line": (MID, False, (), "0", "HOLD", "no end line"),
    "mark_present_but_killed_before_its_end": (MID + LAST + CLEAN[:1], True, (), "0", "HOLD", "no end line"),
    "lock_wait_end": (MID + LOCKWAIT, False, (), "0", "HOLD", "2 h gpu.lock wait"),
    "crash_then_scoring_stop": (MID + FAIL, False, (), "0", "GIVEUP", "ended another way"),
    "stop_logged": (MID + STOPPED, False, (), "0", "GIVEUP", "ended another way"),
    "stop_file_while_running": (MID, False, ("STOP",), "1", "GIVEUP", "STOP exists"),
    "relaunch_refused": (MID + LOCKWAIT + [T + "refusing: uncommitted changes in the code checkout:"], False, (), "0",
                         "GIVEUP", "refusing"),
    "clean_end_no_mark_file": (MID + LAST + CLEAN, False, (), "0", "GIVEUP", "no marks/SCREENS_STAGE_2_S006_SEEDS_DONE"),
    "finished_without_its_mark_line": (MID + LAST + CLEAN[1:], True, (), "0", "GIVEUP", "no MARK line"),
    "earlier_plans_mark_line_only": (MID + LAST + [S2END[0], CLEAN[1]], True, (), "0", "GIVEUP", "no MARK line"),
    "a_line_after_the_end": (MID + LAST + CLEAN + [T + "train base_s103.yaml"], True, (), "0", "GIVEUP", "lines after"),
    "last_start_another_plan": (S2[:1] + S2END + [START.replace("stage2_s006_seeds.txt", "stage2_s006x.txt")] + RUN + LAST
                                + CLEAN, True, (), "0", "GIVEUP", "not stage2_s006_seeds"),
    "last_start_another_commit": (S2 + S2END + [START.replace(REF6, REF2)] + RUN + LAST + CLEAN, True, (), "0", "GIVEUP",
                                  "not stage2_s006_seeds at 3227c6011be2"),
    "double_start_stamp": (MID + LAST + CLEAN, True, ("logs/base_diag_waiter.launched",), "0", "DONE", "already launched"),
    "double_start_log_line": (MID + LAST + CLEAN + [DSTART], True, (), "1", "DONE", "stage2_base_diag start line"),
    "cancel_file": (MID + LAST + CLEAN, True, ("logs/base_diag_waiter.CANCEL",), "0", "GIVEUP", "cancel file"),
    "pgrep_failed": (MID + LAST + CLEAN, True, (), "E", "GIVEUP", "pgrep failed"),
    "s006_waiters_stamp_is_not_this_one": (MID + LAST + CLEAN, True, ("logs/s006_waiter.launched",), "0", "LAUNCH",
                                           "ended cleanly"),
}


def tree(tmp, lines, mark=True, files=(), nul=False):
    p = tmp / "planck"
    out = p / "runs" / "SCREENS"
    os.makedirs(out / "marks", exist_ok=True)
    os.makedirs(p / "logs", exist_ok=True)
    text = "\n".join(lines) + "\n"
    if nul:                                     # a power cut's NUL tail glued to the next line (40_screens notes)
        text = text.replace(START, "\0" * 40 + START)
    (out / "queue_screens.txt").write_bytes(text.encode())
    if mark:
        (out / "marks" / "SCREENS_STAGE_2_S006_SEEDS_DONE").touch()
    for f in files:
        (p / f).touch()
    return p


def sh(p, body, args=ARGS):
    q = " ".join("'" + a + "'" for a in args)
    r = subprocess.run(["/bin/bash", "-c", f'. "{W}"; args {q} || {{ echo REFUSED; exit 0; }}; {body}'],
                       env={**os.environ, "CW_HOME": str(p)}, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stderr == "", r.stderr
    return r.stdout.strip()


def decide(p, running="0"):
    return sh(p, f"decide {running}")


@pytest.mark.parametrize("name", sorted(CASES))
def test_decision(tmp_path, name):
    lines, mark, files, running, word, phrase = CASES[name]
    got = decide(tree(tmp_path, lines, mark, files), running)
    assert got.split()[0] == word and phrase in got, got


def test_nul_bytes_in_the_log_do_not_hide_the_clean_end(tmp_path):
    assert decide(tree(tmp_path, MID + LAST + CLEAN, nul=True)).startswith("LAUNCH")


def test_no_queue_log_and_no_start_line(tmp_path):
    p = tree(tmp_path, ["2026-10-07 00:00:00 waiting for gpu.lock"])
    assert decide(p).startswith("GIVEUP no queue start line")
    os.unlink(p / "runs" / "SCREENS" / "queue_screens.txt")
    assert decide(p).startswith("GIVEUP no queue log")


def test_without_an_earlier_plan_the_earlier_start_is_a_giveup(tmp_path):
    p = tree(tmp_path, S2 + S2END, mark=False)
    assert sh(p, "decide 0", ARGS[:6]).startswith("GIVEUP the last start line is not stage2_s006_seeds")


def test_arguments_set_the_names_and_the_mark_slug(tmp_path):
    got = sh(tmp_path, 'echo "$NAME|$AFTER_PLAN|$AFTER_REF|$AFTER_MARK|$NEXT_PLAN|$BUNDLE|$EARLIER|$MARKF|$STAMP|$CANCEL"')
    logs = f"{tmp_path}/logs/base_diag_waiter"
    assert got == "|".join(ARGS[:6] + [ARGS[6], "SCREENS_STAGE_2_S006_SEEDS_DONE", logs + ".launched", logs + ".CANCEL"])
    slug = subprocess.run(["/bin/bash", "-c", "echo \"$1\" | tr -c 'A-Za-z0-9\\n' '_'", "-", MARK6], capture_output=True,
                          text=True).stdout.strip()     # queue_screens.sh's slug() of the same text
    assert slug == "SCREENS_STAGE_2_S006_SEEDS_DONE" and sh(tmp_path, "echo $POLL $GRACE", ARGS[:6]) == "300 2700"


BAD_ARGS = {
    "five_arguments": ARGS[:5], "name_with_a_slash": ["base/diag"] + ARGS[1:], "short_ref": ARGS[:2] + [REF6[:12]] + ARGS[3:],
    "mark_lowercase": ARGS[:3] + [MARK6.lower()] + ARGS[4:], "mark_double_space": ARGS[:3] + ["SCREENS  X"] + ARGS[4:],
    "next_is_after": ARGS[:4] + ["stage2_s006_seeds"] + ARGS[5:], "bundle_path": ARGS[:5] + ["../planck_diag.bundle"],
    "earlier_without_ref": ARGS[:6] + ["stage2_seeds"], "earlier_is_after": ARGS[:6] + [f"stage2_s006_seeds@{REF6}"],
    "earlier_is_next": ARGS[:6] + [f"stage2_base_diag@{REF2}"], "plan_with_glob": ARGS[:1] + ["stage2_*"] + ARGS[2:],
}


@pytest.mark.parametrize("name", sorted(BAD_ARGS))
def test_bad_arguments_are_refused(tmp_path, name):
    assert sh(tmp_path, "echo SET", BAD_ARGS[name]) == "REFUSED"
