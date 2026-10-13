"""Mutation check for the addendum's tests (test_add1_plan.py, test_e3p1b_waiter.py; notes.txt ADDENDUM 2026-10-12), on
scratch copies only.
  ~/.venvs/planck/bin/python experiments/E3_seed_noise/tests/mutation_add1.py [--only NAME,...] [--workers 4]
Copies experiments/E3_seed_noise (without out/ and audit/) into a fresh scratch tree per mutant, applies ONE edit (the old
text must occur exactly once), runs the named test file there with E3_TEST_ROOT set to the tree (and E3A1_GIT_DIR naming
this repository's git dir, used by the tree-pin test as a --git-dir argument only, read only), and wants it to FAIL. The
unmutated copy must pass every named file first. Exit 0 = every mutant killed. The analyzers' own mutants are in
e3analyze_add1_mutants.py."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
X = "experiments/E3_seed_noise/"
W, PL, CUR, GEN, NOTES, CF, LIB = (X + "e3p1b_waiter.sh", X + "plans/part1b_p3at20m.txt", X + "plans/CURRENT", X + "e3plan.py",
                                   X + "notes.txt", X + "configs/", X + "e3add1_lib.py")
LOG = X + "logs/parity_p1b/e3p1b_parity_cuda.log"
TP, TW = "test_add1_plan.py", "test_e3p1b_waiter.py"
MUTANTS = [
    ("plan_1b_mark_late", PL, "train e3_5m_A_s21\nmark E3 PART 1B DONE\n", "mark E3 PART 1B DONE\ntrain e3_5m_A_s21\n", TP),
    ("plan_20m_pair_swapped", PL, "train e3_20m_A_init2\ntrain e3_20m_A_data2\n", "train e3_20m_A_data2\ntrain e3_20m_A_init2\n", TP),
    ("plan_shared_dropped", PL, "train e3_20m_A_s1_P3\n", "", TP),
    ("plan_no_screens_wait", PL, "wait_mark SCREENS SCREENS ROBUSTNESS K5 DONE\n", "", TP),
    ("plan_no_end_mark", PL, "mark E3 PART 3 AT 20M DONE\n", "", TP),
    ("current_left_at_part3", CUR, "\npart1b_p3at20m\n", "\npart3\n", TP),
    ("cfg_s14_wrong_seed", CF + "e3_5m_A_s14.yaml", "seed: 14\n", "seed: 4\n", TP),
    ("cfg_s15_data_seed", CF + "e3_5m_A_s15.yaml", "seed: 15\n", "seed: 15\ndata_seed: 15\n", TP),
    ("cfg_init3_wrong_data_seed", CF + "e3_20m_A_init3.yaml", "data_seed: 1\n", "data_seed: 3\n", TP),
    ("cfg_data4_wrong_seed", CF + "e3_20m_A_data4.yaml", "seed: 1\ndata_seed", "seed: 4\ndata_seed", TP),
    ("cfg_s1p3_into_part2_dir", CF + "e3_20m_A_s1_P3.yaml", "runs/E3/e3_20m_A_s1_P3\n", "runs/E3/e3_20m_A_s1\n", TP),
    ("cfg_20m_lr", CF + "e3_20m_A_init5.yaml", "{lr: 0.003,", "{lr: 0.0015,", TP),
    ("cfg_earlier_edited", CF + "e3_5m_A_s8.yaml", "seed: 8\n", "seed: 9\n", TP),
    ("gen_part1b_from_11", GEN, "for s in range(12, 22):", "for s in range(11, 21):", TP),
    ("gen_20m_as_5m", GEN, 'files[n] = config(n, "20m", "A", *a20, seed, dseed', 'files[n] = config(n, "5m", "A", *a20, seed, dseed', TP),
    ("notes_cfg_hash_typo", NOTES, "0e26246756996c60", "0e26246756996c61", TP),
    ("notes_scope_widened", NOTES, "RULE 4 AT 20M applies only to verdicts read after", "RULE 4 AT 20M applies to verdicts read after", TP),
    ("notes_parity_log_sha", NOTES, "sha256 d85dd6ac", "sha256 d85dd6ad", TP),
    ("parity_log_more_imports", LOG, "] new_imports: ['durable']\n[default_parity] modules only the reference imported: []\n[default_parity] FAIL\n"
     "2026-10-12 21:15:00 run A", "] new_imports: ['durable', 'induction']\n[default_parity] modules only the reference imported: []\n"
     "[default_parity] FAIL\n2026-10-12 21:15:00 run A", TP),
    ("lib_config_sha", LIB, '"e3_20m_A_data5": "4e65e3fc', '"e3_20m_A_data5": "4e65e3fd', TP),
    ("lib_plan_sha", LIB, 'PLAN_SHA = "part1b_p3at20m.txt", "17c939a5', 'PLAN_SHA = "part1b_p3at20m.txt", "17c939a4', TP),
    ("w_relaunch_any_end", W, '    echo "GIVEUP the queue ended another way', '    echo "RELAUNCH the queue ended another way', TW),
    ("w_no_lockwait_relaunch", W, '            echo "RELAUNCH $PLAN stopped after a 2 h gpu.lock wait: [$last]"; return\n', "", TW),
    ("w_no_memwait_relaunch", W, '''            echo "RELAUNCH $PLAN stopped after 30 min of another process's GPU memory under the lock: [$last]"; return\n''', "", TW),
    ("w_20m_runs_missed", W, "train e3_(5m|20m)_A_[A-Za-z0-9_]+$", "train e3_5m_A_[A-Za-z0-9_]+$", TW),
    ("w_done_without_mark", W, '        if [ -e "$OUT/marks/$MARKF" ]; then echo "DONE', '        if true; then echo "DONE', TW),
    ("w_maxr_off_by_one", W, '[ "$n" -le "$MAXR" ]', '[ "$n" -lt "$MAXR" ]', TW),
    ("w_plan_sha_unchecked", W, '[ "$(sha256sum "$Q/$XREL/plans/$PLAN.txt" 2>/dev/null | cut -c1-64)" = "$PLAN_SHA" ] ||', "true ||", TW),
    ("w_current_unchecked", W, '[ "$CUR" = "$PLAN" ] ||', "true ||", TW),
    ("w_trees_unchecked", W, '[ "$got" = "${t##*:}" ] ||', "true ||", TW),
    ("w_tree_pin_typo", W, "harness:67282e87", "harness:67282e88", TW),
    ("w_cancel_unchecked", W, '[ -e "$CANCEL" ] && { echo "GIVEUP', '[ -e "$CANCEL.x" ] && { echo "GIVEUP', TW),
    ("w_stop_unchecked", W, '[ -e "$P/STOP" ] && { echo "GIVEUP', '[ -e "$P/STOP.x" ] && { echo "GIVEUP', TW),
    ("w_done_mark_unchecked", W, '[ -e "$OUT/marks/$MARKF" ] && { echo "DONE', '[ -e "$OUT/marks/x" ] && { echo "DONE', TW),
    ("w_flock_unchecked", W, "    flock -n 7 || {", "    true || {", TW),
    ("w_runs_beside_e3", W, '[ "$1" = 1 ] && { echo "WAIT an E3', '[ "$1" = 9 ] && { echo "WAIT an E3', TW),
    ("w_runs_beside_screens", W, '[ "$2" = 1 ] && { echo "WAIT a SCREENS', '[ "$2" = 9 ] && { echo "WAIT a SCREENS', TW),
    ("w_no_robust_mark_wait", W, '[ -e "$SOUT/marks/$RMARK" ] || { echo "WAIT', '[ -e "$SOUT" ] || { echo "WAIT', TW),
    ("w_no_c6_line_wait", W, 'grep -aqE "^$TS DONE: robustness_k5', 'true || grep -aqE "^$TS DONE: robustness_k5', TW),
    ("w_no_disk_wait", W, '[ "${f:-0}" -ge "$MINFREE" ] ||', "true ||", TW),
    ("w_relaunch_waits_on_screens", W, '    [ "${3:-}" = relaunch ] && { echo "LAUNCH relaunch after a lock wait"; return; }\n', "", TW),
    ("w_screens_log_unchecked", W, 'printf \'%s\\n\' "$m" | grep -aqE "^$TS plan finished: robustness_k5\\.txt$" || {',
     'true || {', TW),
    ("w_e3_other_plan_ok", W, '    echo "GIVEUP E3\'s last queue ($plan) did not end cleanly', '    echo "OK E3\'s last queue ($plan) did not end cleanly', TW),
    ("w_start_line_unchecked", W, "    if ! printf '%s\\n' \"$txt\" | grep -aqE \"^$TS queue start:", "    if false && printf '%s\\n' \"$txt\" | grep -aqE \"^$TS queue start:", TW),
    ("w_reads_the_whole_log", W, 'new=$(tail -c +$((before + 1)) "$QLOG" 2>/dev/null)', 'new=$(cat "$QLOG" 2>/dev/null)', TW),
    ("w_pin_typo", W, "PLAN_SHA=17c939a5", "PLAN_SHA=17c939a4", TW),
    ("w_no_qmin", W, "QUEUE_MIN_FREE_GB=$MINFREE bash", "bash", TW),
    ("w_done_any_rc", W, '    if [ "$rc" = 0 ] && printf', '    if true && printf', TW),
    ("w_relaunch_any_rc", W, '    if [ "$rc" = 1 ] && printf', '    if true && printf', TW),
    ("w_pgrep_error_is_none", W, "case $? in 0) echo 1 ;; 1) echo 0 ;; *) echo E ;; esac", "case $? in 0) echo 1 ;; *) echo 0 ;; esac", TW),
    ("w_decide_launches", W, '        decide "$(running "$PAT")" "$(running "$SPAT")"; exit 0\n', '        decide "$(running "$PAT")" "$(running "$SPAT")"\n', TW),
    ("t_fixture_env_keeps_git_dir", X + "tests/w1bfixture.py", "return {**{k: v for k, v in os.environ.items() if k not in GIT_ENV}, **over}",
     "return {**os.environ, **over}", TW),
]


def copy_tree(dst: str) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "out", "audit")
    shutil.copytree(os.path.join(ROOT, X), os.path.join(dst, X), ignore=ign)


def run_tests(dst: str, spec: str) -> int:
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, "E3_TEST_ROOT": dst, "PYTHONDONTWRITEBYTECODE": "1",
           "E3A1_GIT_DIR": os.path.join(ROOT, ".git")}
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
    work = tempfile.mkdtemp(prefix="e3a1_tmut_")
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
