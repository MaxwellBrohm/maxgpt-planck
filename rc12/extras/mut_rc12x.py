"""Mutants of rc12x_extras.sh (notes STEP 14X): each replaces one text in a scratch copy; test_rc12x_extras.py must fail
on each (KILLED = exit 1). The unmutated copy must pass; MALFORMED = the text is not found exactly once.
  python mut_rc12x.py [-j N]   (exit 1 unless the baseline passes and every mutant is killed)"""
import concurrent.futures
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "rc12x_extras.sh")).read()
MUTS = [
    ("base_not_plain", "B) echo plain;;", "B) echo \"\";;"),
    ("cria_cap_10800", 'HOLD="Q_SESSION_S=1500 Q_SESSION_TAIL=600 Q_TIMEOUT=1020"',
     'HOLD="Q_SESSION_S=1500 Q_SESSION_TAIL=600"'),
    ("session_1800", 'HOLD="Q_SESSION_S=1500', 'HOLD="Q_SESSION_S=1800'),
    ("tail_300", 'Q_SESSION_TAIL=600 Q_TIMEOUT', 'Q_SESSION_TAIL=300 Q_TIMEOUT'),
    ("hold_dropped_with_renders", 'env $HOLD Q_CODE="$CODE" Q_MODELS="$models" Q_RENDERS',
     'env Q_CODE="$CODE" Q_MODELS="$models" Q_RENDERS'),
    ("cria_not_last", 'PHASES=${*:-A B D C}', 'PHASES=${*:-C A B D}'),
    ("minimind_missing", 'PHASES=${*:-A B D C}', 'PHASES=${*:-A B C}'),
    ("one_quiet_poll", '[ $ok -ge 2 ]', '[ $ok -ge 1 ]'),
    ("no_mark_check", 'if [ "$mark" = present ] && quiet', 'if quiet'),
    ("any_mark_absent_ignored", '|| mark="absent ($f)"', '|| true'),
    ("first_mark_only", 'for f in $MARKS; do', 'for f in ${MARKS%% *}; do'),
    ("marks_old_diag_only", "RX_MARKS:-runs/SCREENS/marks/SCREENS_BASE_REPLAY_DONE c6diag/DONE}",
     "RX_MARKS:-runs/SCREENS/marks/SCREENS_BASE_DIAG_DONE}"),
    ("no_c6_done", " c6diag/DONE}", "}"),
    ("no_replay_waiter", "|replay_waiter\\.sh|", "|"),
    ("no_diag_start", "|diag_start\\.sh|", "|"),
    ("no_diag_runner", "|diag_runner\\.py|", "|"),
    ("no_c1a_gate", "|c1a_gate\\.sh'", "'"),
    ("log_whole", 't=$(tail -n +$((n + 1)) "$QL"', 't=$(cat "$QL"'),
    ("no_chain_waiter", "|chain_waiter\\.sh|", "|"),
    ("no_base_diag_waiter", "|base_diag_waiter\\.sh|", "|"),
    ("no_diag_queue", "|diag_queue\\.sh|", "|"),
    ("no_s006_waiter", "|s006_waiter\\.sh|", "|"),
    ("no_rc12_pattern", "RPAT='queue_dev_baselines\\.sh|dev_batch\\.py'", "RPAT='queue_dev_baselines\\.sh'"),
    ("no_rc12_wait", '    wait_for rc12\n', '\n'),
    ("chain_wait_every_phase", '[ "$name" = C ] && [ "$CHAIN" = 1 ] && wait_for chain', '[ "$CHAIN" = 1 ] && wait_for chain'),
    ("no_chain_wait_ignored", '{ CHAIN=0; shift; }', '{ shift; }'),
    ("chain_wait_off", '    CHAIN=1\n', '    CHAIN=0\n'),
    ("pgrep_error_quiet", '[ $rc -eq 1 ] || return 1', '[ $rc -eq 0 ] && return 1'),
    ("no_disk_check", '[ "$gb" -ge "$MIN_GB" ]', '[ "$gb" -ge 0 ]'),
    ("no_clean_end_check", '| tail -1 | grep -aq "queue: end;"', '| grep -aq "queue:"'),
    ("no_start_line_check", 'grep -aqF "queue: start, code $CODE,"', 'true'),
    ("fd7_to_queue", 'queue_dev_baselines.sh" 7>&-\n    fi', 'queue_dev_baselines.sh"\n    fi'),
    ("no_stopfile", '"$L/rc12_dev_queue.STOP" "$L/rc12x_extras.CANCEL"', '"$L/rc12x_extras.CANCEL"'),
    ("no_phase_preflight", '    preflight $models', '    true $models'),
    ("no_first_preflight", '    preflight $all', '    true $all'),
    ("no_split_check", '[ "${x:-1}" -ge 2 ] 2>/dev/null ||', 'true ||'),
    ("no_run_count_check", '| grep -aqE "^complete [0-9]+ of $want runs$"', '| grep -aq complete'),
    ("b_count_16", 'A) echo 48;; B) echo 8;;', 'A) echo 48;; B) echo 16;;'),
    ("d_count_8", 'C|D) echo 16;;', 'C) echo 16;; D) echo 8;;'),
    ("no_flock", 'flock -n 7 ||', 'true ||'),
    ("q_logs_set", 'env $HOLD Q_CODE="$CODE" Q_MODELS="$models" Q_RENDERS', 'env $HOLD Q_LOGS=$L/x Q_CODE="$CODE" Q_MODELS="$models" Q_RENDERS'),
    ("bad_phase_accepted", '[A|B|C|D ...]"; exit 2;; esac', '[A|B|C|D ...]";; esac'),
    ("minimind_wrong_model", 'MINI=jingyaogong/minimind-3', 'MINI=jingyaogong/minimind-2'),
]


def one(m):
    name, a, b = m
    if SRC.count(a) != 1:
        return name, "MALFORMED", ""
    d = tempfile.mkdtemp(prefix="mutx_")
    p = os.path.join(d, "rc12x_extras.sh")
    open(p, "w").write(SRC.replace(a, b, 1))
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_rc12x_extras.py"), p], capture_output=True, text=True)
    first = next((x for x in r.stdout.splitlines() if x.startswith("FAIL")), (r.stdout or r.stderr)[-140:])
    return name, "KILLED" if r.returncode == 1 else "SURVIVED", first


def main():
    jobs = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 1
    base = subprocess.run([sys.executable, os.path.join(HERE, "test_rc12x_extras.py")], capture_output=True, text=True)
    print("baseline:", "pass" if base.returncode == 0 else "FAIL", flush=True)
    killed = 0
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        for name, verdict, first in ex.map(one, MUTS):
            killed += verdict == "KILLED"
            print(f"{name:28s} {verdict:9s} {first[:140]}", flush=True)
    print(f"{killed} / {len(MUTS)} killed")
    return 0 if base.returncode == 0 and killed == len(MUTS) else 1


if __name__ == "__main__":
    sys.exit(main())
