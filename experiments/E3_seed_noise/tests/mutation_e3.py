"""Mutation check for E3 Part 3's tests (test_part3_plan.py, test_e3p3_waiter.py), on scratch copies only.

  ~/.venvs/planck/bin/python experiments/E3_seed_noise/tests/mutation_e3.py [--only NAME,...] [--workers 4]

Copies experiments/E3_seed_noise (without out/, audit/ and logs/), E2's preflight.py and harness/*.py into a fresh
scratch tree per mutant, applies ONE edit (the old text must occur exactly once), runs the named test file there with
E3_TEST_ROOT set to the tree, and wants it to FAIL. The unmutated copy must pass every named file first. Exit 0 = every
mutant killed.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
X = "experiments/E3_seed_noise/"
W, PL, CUR, GEN, NOTES, CF = X + "e3p3_waiter.sh", X + "plans/part3.txt", X + "plans/CURRENT", X + "e3plan.py", \
    X + "notes.txt", X + "configs/"
TP, TW = "test_part3_plan.py", "test_e3p3_waiter.py"
_T = open(os.path.join(ROOT, "harness", "train.py")).read()
_A, _B = '    ds = cfg.get("data_seed")', '    was = ck.get("data_seed", ck.get("seed"))'
HSPAN = _T[_T.index(_A):_T.index(_B) + len(_B)]     # train.py from its first data_seed read to its second
MUTANTS = [
    # the plan, CURRENT, the configs (test_part3_plan.py)
    ("plan_pair_swapped", PL, "train e3_5m_A_init2\ntrain e3_5m_A_data2\n", "train e3_5m_A_data2\ntrain e3_5m_A_init2\n", TP),
    ("plan_shared_member_first", PL, "train e3_5m_A_init2\n", "train e3_5m_A_s1_P3\ntrain e3_5m_A_init2\n", TP),
    ("plan_shared_member_dropped", PL, "train e3_5m_A_s1_P3\n", "", TP),
    ("plan_waits_on_part1", PL, "wait_mark E3 E3 PART 2 DONE", "wait_mark E3 E3 PART 1 DONE", TP),
    ("plan_no_end_mark", PL, "mark E3 PART 3 DONE\n", "", TP),
    ("current_left_at_part2", CUR, "\npart3\n", "\npart2\n", TP),
    ("cfg_init3_wrong_data_seed", CF + "e3_5m_A_init3.yaml", "data_seed: 1\n", "data_seed: 3\n", TP),
    ("cfg_data4_wrong_seed", CF + "e3_5m_A_data4.yaml", "seed: 1\ndata_seed", "seed: 4\ndata_seed", TP),
    ("cfg_rerun_without_data_seed", CF + "e3_5m_A_s1_P3.yaml", "data_seed: 1\n", "", TP),
    ("cfg_rerun_into_a1s_dir", CF + "e3_5m_A_s1_P3.yaml", "runs/E3/e3_5m_A_s1_P3\n", "runs/E3/e3_5m_A_s1\n", TP),
    ("cfg_init5_other_lr", CF + "e3_5m_A_init5.yaml", "{lr: 0.003,", "{lr: 0.0015,", TP),
    ("cfg_part2_edited", CF + "e3_20m_B_s2.yaml", "seed: 2\n", "seed: 3\n", TP),
    ("gen_rerun_moved", GEN, "            if k == 4:", "            if k == 2:", TP),
    ("gen_data_seed_not_written", GEN, '        txt += f"data_seed: {data_seed}\\n"\n', "        pass\n", TP),
    ("preflight_guard_gone", "experiments/E2_lr_transfer/preflight.py",
     'if "data_seed" in cfg and "data_seed" not in reads:', "if False:", TP),     # (a scratch copy of E2's file)
    ("harness_reads_no_data_seed", "harness/train.py", HSPAN,
     HSPAN.replace('cfg.get("data_seed")', 'cfg["data_seed"]').replace('ck.get("data_seed", ck.get("seed"))',
                                                                      'ck["data_seed"] if "data_seed" in ck else ck.get("seed")'),
     TP),     # one edit over both reads: preflight's harness_reads no longer sees data_seed
    ("notes_hash_typo", NOTES, "b364101f884e8dc3", "b364101f884e8dc4", TP),
    ("notes_scope_widened", NOTES, "never to stage 1 or stage 2 of SCREENS", "also to stage 1 or stage 2 of SCREENS", TP),
    # the waiter (test_e3p3_waiter.py)
    ("w_relaunch_any_end", W, '    echo "GIVEUP the queue ended another way', '    echo "RELAUNCH the queue ended another way', TW),
    ("w_no_lockwait_relaunch", W, '            echo "RELAUNCH part3 stopped after a 2 h gpu.lock wait: [$last]"; return\n',
     "", TW),
    ("w_no_memwait_relaunch", W,
     '''            echo "RELAUNCH part3 stopped after 30 min of another process's GPU memory under the lock: [$last]"; return\n''',
     "", TW),
    ("w_uppercase_run_names_missed", W, "train e3_5m_A_[A-Za-z0-9_]+$", "train e3_5m_A_[a-z0-9_]+$", TW),
    ("w_done_without_mark", W, '        if [ -e "$OUT/marks/$MARKF" ]; then echo "DONE', '        if true; then echo "DONE', TW),
    ("w_maxr_off_by_one", W, '[ "$n" -le "$MAXR" ]', '[ "$n" -lt "$MAXR" ]', TW),
    ("w_plan_sha_unchecked", W, '[ "$(sha256sum "$Q/$XREL/plans/part3.txt" 2>/dev/null | cut -c1-64)" = "$PLAN_SHA" ] ||',
     "true ||", TW),
    ("w_current_unchecked", W, '[ "$PLAN" = part3 ] ||', "true ||", TW),
    ("w_cancel_unchecked", W, '[ -e "$CANCEL" ] && { say', '[ -e "$CANCEL.x" ] && { say', TW),
    ("w_stop_unchecked", W, '[ -e "$P/STOP" ] && { say', '[ -e "$P/STOP.x" ] && { say', TW),
    ("w_done_mark_unchecked_at_start", W, '[ -e "$OUT/marks/$MARKF" ] && { say "DONE', '[ -e "$OUT/marks/x" ] && { say "DONE', TW),
    ("w_flock_unchecked", W, "    flock -n 7 || {", "    true || {", TW),
    ("w_runs_beside_a_queue", W, '        if [ "$r" = 1 ]; then', "        if false; then", TW),
    ("w_start_line_unchecked", W, '    if ! printf \'%s\\n\' "$txt" | grep -aqE "^$TS queue start:', '    if false && printf \'%s\\n\' "$txt" | grep -aqE "^$TS queue start:', TW),
    ("w_reads_the_whole_log", W, 'new=$(tail -c +$((before + 1)) "$QLOG" 2>/dev/null)', 'new=$(cat "$QLOG" 2>/dev/null)', TW),
    ("w_pin_typo", W, "PLAN_SHA=07018f0f", "PLAN_SHA=07018f0e", TW),
    # the 2026-10-11 verify's waiter survivors (F4) that a test can pin
    ("w_done_any_rc", W, '    if [ "$rc" = 0 ] && printf', '    if true && printf', TW),
    ("w_relaunch_any_rc", W, '    if [ "$rc" = 1 ] && printf', '    if true && printf', TW),
    ("w_pgrep_error_is_none", W, "case $? in 0) echo 1 ;; 1) echo 0 ;; *) echo E ;; esac",
     "case $? in 0) echo 1 ;; *) echo 0 ;; esac", TW),
    # the fixture's own guard (2026-10-11: an inherited GIT_DIR let it commit onto the real repository's main)
    ("t_fixture_env_keeps_git_dir", X + "tests/" + TW, "return {**{k: v for k, v in os.environ.items() if k not in GIT_ENV}, **over}",
     "return {**os.environ, **over}", TW),
]


def copy_tree(dst: str) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "out", "audit", "logs")
    shutil.copytree(os.path.join(ROOT, X), os.path.join(dst, X), ignore=ign)
    os.makedirs(os.path.join(dst, "harness"))
    for p in glob.glob(os.path.join(ROOT, "harness", "*.py")):
        shutil.copy(p, os.path.join(dst, "harness"))
    os.makedirs(os.path.join(dst, "experiments", "E2_lr_transfer"))
    shutil.copy(os.path.join(ROOT, "experiments", "E2_lr_transfer", "preflight.py"),
                os.path.join(dst, "experiments", "E2_lr_transfer"))


def run_tests(dst: str, spec: str) -> int:
    env = {**os.environ, "E3_TEST_ROOT": dst, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", os.path.join(dst, X, "tests", spec)]
    return subprocess.run(cmd, cwd=dst, env=env, capture_output=True, text=True, timeout=900).returncode


def one(work: str, m) -> tuple[str, str, int]:
    name, rel, old, new, spec = m
    dst = os.path.join(work, name)
    shutil.rmtree(dst, ignore_errors=True)
    copy_tree(dst)
    p = os.path.join(dst, rel)
    txt = open(p).read()
    if txt.count(old) != 1:
        shutil.rmtree(dst, ignore_errors=True)
        return name, f"INVALID (the old text occurs {txt.count(old)} times in {rel})", -1
    open(p, "w").write(txt.replace(old, new))
    rc = run_tests(dst, spec)
    shutil.rmtree(dst, ignore_errors=True)
    return name, ("killed" if rc not in (0, -1) else "SURVIVED"), rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    names = [m[0] for m in MUTANTS]
    assert len(set(names)) == len(names), "duplicate mutant names"
    todo = [m for m in MUTANTS if not a.only or m[0] in a.only.split(",")]
    work = tempfile.mkdtemp(prefix="e3_mut_")
    clean = os.path.join(work, "clean")
    copy_tree(clean)
    for spec in sorted({m[4] for m in todo}):
        rc = run_tests(clean, spec)
        print(f"unmutated {spec}: {'pass' if rc == 0 else 'FAIL rc %d' % rc}", flush=True)
        if rc != 0:
            return 2
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda m: one(work, m), todo))
    bad = [n for n, s, _ in res if s != "killed"]
    for n, s, rc in res:
        print(f"{s:9s} {n} (rc {rc})", flush=True)
    print(f"{len(res) - len(bad)} of {len(res)} killed" + (f"; not killed: {bad}" if bad else ""))
    shutil.rmtree(work, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
