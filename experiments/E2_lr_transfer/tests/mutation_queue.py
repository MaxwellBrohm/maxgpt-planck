"""Mutation check for tests/test_queue.py: each mutant is one deliberate bug applied to a scratch copy of the
E2 directory; killed = the test file fails (pytest exit 1). A pattern that is not found exactly once is
INVALID, never killed. PC only (the tests need bash >= 4 and flock).
  ~/planck/venv/bin/python tests/mutation_queue.py [--part i/n | --only a,b] [--list]
(each part must finish inside a 2-minute command: --part i/4)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
Q, L = "queue_e2.sh", "queue_lib.sh"
MUTANTS = [
    ("no_require_committed", Q, 'train_run "$cfg" "$out" --require-committed; rc=$?', 'train_run "$cfg" "$out"; rc=$?'),
    ("preflight_not_strict", Q, '--code "$CODE" --strict --out', '--code "$CODE" --out'),
    ("no_inheritance", Q, '            if inh=$(inherited "$OUT/preflight/$name.json"); then',
     '            if false; then'),
    ("inherit_gap_as_diverged", Q, '[ -e "$CODE/$(dirname "$init")/DIVERGED" ] && { echo DIVERGED; return 0; }',
     '[ -e "$CODE/$(dirname "$init")/DIVERGED" ] && { echo GAP; return 0; }'),
    ("divergence_treated_as_crash", L, 'if grep -qs "non-finite loss" "$out/log.jsonl" "$out/train.out"; then',
     'if false; then'),
    ("gap_after_three_crashes", L, '-ge 2 ]; then', '-ge 3 ]; then'),
    ("score_one_rolling", L, "sort | tail -2)", "sort | tail -1)"),
    ("score_trunks_not_branches", Q, 'if [ "$mode" = trunk ]; then touch', 'if [ "$mode" != trunk ]; then touch'),
    ("no_skip_marker", Q, '[ -e "$out/SCORED" ] || [ -e "$out/TRUNK_DONE" ]; then', 'false; then'),
    ("no_gpu_lock", L, "    flock -w 7200 9 ||", "    true ||"),
    ("no_stop_watch", L, '[ -e "$P/STOP" ] && touch "$out/STOP"', 'true'),
    ("no_heat_stop", L, '[ $hot -ge 3 ] && {', '[ $hot -ge 300 ] && {'),
    ("no_start_temp_guard", L, '[ -z "$t" ] || [ "$t" -gt 75 ]; do', 'false; do'),
    ("no_gpu_mem_guard", L, '[ "$used" -le 3000 ] &&', '[ "$used" -le 300000 ] &&'),
    ("no_dirty_check", Q, '[ -z "$DIRTY" ] ||', 'true ||'),
    ("no_disk_guard", L, 'if [ "${f:-0}" -lt "$MINFREE" ]; then', 'if false; then'),
    ("stop_file_ignored", L, 'if [ -e "$P/STOP" ]; then qlog', 'if false; then qlog'),
    ("wait_mark_ignores_stop", Q, '[ -e "$P/STOP" ] && { qlog "STOP while waiting', '[ -e "$P/NOPE" ] && { qlog "STOP while waiting'),
    ("stop_rc_goes_on", Q, '        *)  gpu_unlock; return 1 ;;          # 12 STOP, 13 prereg refused',
     '        *)  ;;'),
    ("heat_not_retried", Q, '            continue\n', '            return 1\n'),
    ("plan_stop_ignored", Q, '        stop) qlog "plan says stop"; exit 0 ;;', '        stop) qlog "plan says stop" ;;'),
    ("bad_prose_pool", "bpb_lines.py", 'CHAT, PROSE = "oasst2", ("cccc", "gutenberg", "wikimedia")',
     'CHAT, PROSE = "oasst2", ("cccc", "gutenberg")'),
]


def run_one(name, fname, old, new) -> str:
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "E2_lr_transfer")
        shutil.copytree(E2, dst, ignore=shutil.ignore_patterns("__pycache__"))
        p = os.path.join(dst, fname)
        src = open(p).read()
        if src.count(old) != 1:
            return f"INVALID (pattern found {src.count(old)} times)"
        open(p, "w").write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            os.path.join(dst, "tests", "test_queue.py")], capture_output=True, text=True, timeout=300)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        return {1: "killed", 0: "SURVIVED"}.get(r.returncode, f"INVALID rc {r.returncode}") + f"  [{last}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of mutant names")
    a = ap.parse_args()
    i, n = (int(x) for x in a.part.split("/"))
    todo = [m for m in MUTANTS if m[0] in a.only.split(",")] if a.only else MUTANTS[i - 1::n]
    if a.list:
        print("\n".join(m[0] for m in todo))
        return 0
    bad = 0
    for m in todo:
        res = run_one(*m)
        bad += not res.startswith("killed")
        print(f"{m[0]:32s} {res}", flush=True)
    print(f"{len(todo) - bad}/{len(todo)} killed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
