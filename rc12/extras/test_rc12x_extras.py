"""Fakes-only test of rc12x_extras.sh (notes STEP 14X): shims for flock (real fcntl locks), pgrep (a file of running
command lines, read again at every call) and df (a fake free size); a fake queue_dev_baselines.sh that logs its
environment and plays a clean end, a STOP end, "another queue holds" and other bad ends; the repo's engines.json in
the fake code dir. No model, no GPU; runs the same on the Mac and the PC.
  python test_rc12x_extras.py [SCRIPT] [ENGINES]   exit 1 on any failure"""
import fcntl
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rc12x_extras.sh")
ENGINES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "engines.json")
SIBS = ["tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum", "tiiuae/Falcon-H1-Tiny-90M-Instruct-pre-DPO",
        "tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO"]
BASE, CRIA, MINI = "tiiuae/Falcon-H1-Tiny-90M-Base", "sz14/cRia-LM-75M-Instruct", "jingyaogong/minimind-3"
QUEUE = r"""#!/bin/bash
L=$RX_HOME/logs
echo "Q_CODE=${Q_CODE-UNSET}|Q_MODELS=${Q_MODELS-UNSET}|Q_RENDERS=${Q_RENDERS-UNSET}|Q_TIMEOUT=${Q_TIMEOUT-UNSET}|Q_SESSION_S=${Q_SESSION_S-UNSET}|Q_SESSION_TAIL=${Q_SESSION_TAIL-UNSET}|Q_LOGS=${Q_LOGS-UNSET}|fd7=$([ -e /dev/fd/7 ] && echo open || echo closed)" >> "$RX_HOME/calls.txt"
mode=$(sed -n "$(wc -l < "$RX_HOME/calls.txt" | tr -d ' ')p" "$RX_HOME/modes.txt" 2>/dev/null); mode=${mode:-ok}
[ "$mode" = held ] && { echo "t queue: another queue holds the lock, exiting" >> "$L/rc12_dev_queue.log"; exit 0; }
[ "$mode" = silent ] && exit 0
rm -f "$L/rc12_dev_queue.DONE"
echo "t queue: start, code $Q_CODE, root r, pid $$, gpu.lock sessions of 1 s (tail 1 s), run x.$$" >> "$L/rc12_dev_queue.log"
[ "$mode" = stop ] && { echo "t queue: STOP file, exiting" >> "$L/rc12_dev_queue.log"; exit 0; }
n=$(( $(echo $Q_MODELS | wc -w) * $(echo ${Q_RENDERS:-template plain} | wc -w) * 8 )); x=$n
[ "$mode" = default208 ] && { n=208; x=208; }
[ "$mode" = fewer ] && x=$((n - 8))
[ "$mode" = othercode ] && sed -i.bak '$ s/code [^,]*,/code \/elsewhere,/' "$L/rc12_dev_queue.log"
[ "$mode" = dropbase ] && rm -rf "$RX_HOME/hf/hub/models--tiiuae--Falcon-H1-Tiny-90M-Base"
[ "$mode" = busy ] && echo "python -B dev_batch.py --responder x" >> "$FAKE_PS"
echo "complete $x of $n runs" > "$L/rc12_dev_queue.DONE"
[ "$mode" = killed ] && exit 137
echo "t queue: end; complete $x of $n runs" >> "$L/rc12_dev_queue.log"
"""
FLOCK = f"""#!{sys.executable}
import fcntl, sys
a = sys.argv[1:]
try:
    fcntl.flock(int([x for x in a if x.isdigit()][-1]), fcntl.LOCK_EX | (fcntl.LOCK_NB if "-n" in a else 0))
except OSError:
    sys.exit(1)
"""
PGREP = '#!/bin/bash\n[ -n "${FAKE_PS_ERR:-}" ] && exit 2\npat="${@: -1}"\n' \
        '[ -f "$FAKE_PS" ] && grep -Eq -- "$pat" "$FAKE_PS" && exit 0\nexit 1\n'
DF = '#!/bin/bash\necho " Avail"; echo " ${FAKE_GB:-100}G"\n'
HOLD = "Q_TIMEOUT=1020|Q_SESSION_S=1500|Q_SESSION_TAIL=600|Q_LOGS=UNSET|fd7=closed"
fails = []


def put(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    if mode:
        os.chmod(path, mode)


MARKS = ["runs/SCREENS/marks/SCREENS_BASE_REPLAY_DONE", "c6diag/DONE"]   # W1's chain ends (review D2, 2026-10-10)


def box(mark=True, running=(), gb=100, modes=(), drop_engine=None, drop_cache=None, stops=(), unsplit=False):
    t = tempfile.mkdtemp(prefix="rc12x_")
    home, code, shims = f"{t}/home", f"{t}/code", f"{t}/shims"
    put(f"{shims}/flock", FLOCK, 0o755)
    put(f"{shims}/pgrep", PGREP, 0o755)
    put(f"{shims}/df", DF, 0o755)
    put(f"{code}/queue_dev_baselines.sh", QUEUE)
    eng = json.load(open(ENGINES))
    if drop_engine:
        del eng["models"][drop_engine]
    if unsplit:
        eng["models"][CRIA].pop("shards", None)
    put(f"{code}/engines.json", json.dumps(eng))
    for m in SIBS + [BASE, CRIA, MINI]:
        if m != drop_cache:
            put(f"{home}/hf/hub/models--{m.replace('/', '--')}/refs/main", "abc")
    put(f"{home}/runs/SCREENS/marks/SCREENS_BASE_DIAG_DONE", "")   # on the PC since 10-07: never enough alone
    for m in (MARKS if mark is True else mark or []):
        put(f"{home}/{m}", "")
    put(f"{home}/logs/rc12_dev_queue.DONE", "complete 208 of 208 runs\n")   # the PC's DONE before any phase
    put(f"{t}/ps.txt", "".join(f"bash /x/{r} --plan p\n" for r in running))
    put(f"{home}/modes.txt", "".join(f"{m}\n" for m in modes))
    for s in stops:
        put(f"{home}/{s}", "")
    env = dict(os.environ, RX_HOME=home, RX_CODE=code, RX_POLL="0", RX_MAXPOLLS="5", RX_PY=sys.executable,
               PATH=f"{shims}:{os.environ['PATH']}", FAKE_PS=f"{t}/ps.txt", FAKE_GB=str(gb))
    return t, home, code, env


def run(home, env, args=()):
    rc = subprocess.run(["bash", SCRIPT, *args], env=env, timeout=60, capture_output=True).returncode
    log = open(f"{home}/logs/rc12x_extras.log").read() if os.path.exists(f"{home}/logs/rc12x_extras.log") else ""
    calls = open(f"{home}/calls.txt").read().splitlines() if os.path.exists(f"{home}/calls.txt") else []
    return rc, log, calls


def check(name, cond, info=""):
    if not cond:
        fails.append(f"{name} {info}")
        print("FAIL", name, info[:300])


def case(name, want_rc, want_calls, args=(), **kw):
    t, home, code, env = box(**{k: v for k, v in kw.items() if k != "env"})
    env.update(kw.get("env", {}))
    rc, log, calls = run(home, env, args)
    check(name, rc == want_rc and len(calls) == want_calls, f"rc {rc} calls {len(calls)} log [{log[-300:]}]")
    return home, code, log, calls


def end_of(home):
    p = f"{home}/logs/rc12x_extras.DONE"
    return open(p).read().splitlines() if os.path.exists(p) else []


def want_call(code, models, renders="UNSET"):
    return f"Q_CODE={code}|Q_MODELS={' '.join(models)}|Q_RENDERS={renders}|{HOLD}"


# full run: A, B, D, then C (last), each with exactly its environment, then the end marker in that order
home, code, log, calls = case("full_run", 0, 4)
check("full_run_env", calls == [want_call(code, SIBS), want_call(code, [BASE], "plain"), want_call(code, [MINI]),
                                 want_call(code, [CRIA])], "\n".join(calls))
check("full_run_end", end_of(home) == ["complete 48 of 48 runs", "complete 8 of 8 runs", "complete 16 of 16 runs",
                                       "complete 16 of 16 runs"], str(end_of(home)))
check("full_run_copies", all(os.path.exists(f"{home}/logs/rc12x_{p}.DONE") for p in "ABCD"))
check("full_run_start_line", f"START (chain): marks [{' '.join(MARKS)}] present" in log, log[-400:])
home, code, log, calls = case("phases_ABC", 0, 3, args=("A", "B", "C"), drop_cache=MINI)
check("phases_ABC_env", [c.split("|")[1] for c in calls] == [f"Q_MODELS={' '.join(SIBS)}", f"Q_MODELS={BASE}",
                                                               f"Q_MODELS={CRIA}"], "\n".join(calls))
case("bad_phase", 2, 0, args=("A", "E"))
# the SCREENS chain gates C only: A, B and D run, C waits (GIVEUP at the test bound)
case("no_mark", 3, 3, mark=False)
case("replay_mark_only", 3, 3, mark=MARKS[:1])
case("c6_done_only", 3, 3, mark=MARKS[1:])
for r in ["queue_screens.sh", "launch_screens.sh", "s006_waiter.sh", "chain_waiter.sh", "base_diag_waiter.sh",
          "replay_waiter.sh", "diag_queue.sh", "diag_start.sh", "diag_runner.py", "c1a_gate.sh"]:
    case(f"busy_{r}", 3, 3, running=[r])
    case(f"busy_{r}_no_chain_wait", 0, 4, args=("--no-chain-wait",), running=[r])
case("other_mark_name", 3, 3, env={"RX_MARKS": "runs/SCREENS/marks/SCREENS_STAGE_2_S006_SEEDS_DONE"})
# another RC-12 queue (a boot resume) holds every phase; one quiet poll is not enough; pgrep failing is not quiet
for r in ["queue_dev_baselines.sh", "dev_batch.py"]:
    case(f"busy_rc12_{r}", 3, 0, running=[r])
case("one_quiet_poll", 3, 0, env={"RX_MAXPOLLS": "1"})
case("pgrep_error", 3, 0, env={"FAKE_PS_ERR": "1"})
case("rc12_busy_after_A", 3, 1, modes=["busy"])
# refusals before any queue call
case("low_disk", 1, 0, gb=44)
for s in ["STOP", "PAUSE", "logs/rc12_dev_queue.STOP", "logs/rc12x_extras.CANCEL"]:
    case(f"stopfile_{s}", 1, 0, stops=[s])
case("engine_missing", 1, 0, drop_engine=BASE)
case("engine_missing_minimind", 1, 0, drop_engine=MINI)
case("cache_missing", 1, 0, drop_cache=CRIA)
case("cria_unsplit", 1, 0, unsplit=True)
case("cria_unsplit_not_chosen", 0, 2, args=("A", "B"), unsplit=True)
case("engine_missing_before_wait", 1, 0, mark=False, drop_engine=CRIA)
# a phase that does not end cleanly stops the script, with no end marker
for m in ["stop", "held"]:
    home, code, log, calls = case(f"phase_{m}", 1, 1, modes=[m])
    check(f"phase_{m}_no_end", not end_of(home) and not os.path.exists(f"{home}/logs/rc12x_A.DONE"))
case("phase_B_stop", 1, 2, modes=["ok", "stop"])
for m in ["killed", "othercode", "default208"]:
    home, code, log, calls = case(f"phase_{m}", 1, 1, modes=[m])
    check(f"phase_{m}_no_copy", not os.path.exists(f"{home}/logs/rc12x_A.DONE"))
case("cache_dropped_during_A", 1, 1, modes=["dropbase"])
# a queue that writes nothing (e.g. rc 127): D's start, end and DONE lines are older than C's call, never C's end
home, code, log, calls = case("phase_C_silent", 1, 4, modes=["ok", "ok", "ok", "silent"])
check("phase_C_silent_no_end", not end_of(home) and not os.path.exists(f"{home}/logs/rc12x_C.DONE"))
home, code, log, calls = case("phase_A_fewer_runs", 0, 4, modes=["fewer"])
check("phase_A_fewer_end", end_of(home)[:1] == ["complete 40 of 48 runs"], str(end_of(home)))
# single instance: another holder of locks/rc12x_extras.lock
t, home, code, env = box()
os.makedirs(f"{home}/locks", exist_ok=True)
fd = os.open(f"{home}/locks/rc12x_extras.lock", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
fcntl.flock(fd, fcntl.LOCK_EX)
rc, log, calls = run(home, env)
check("single_instance", rc == 1 and not calls and "another rc12x_extras holds" in log, f"rc {rc} [{log[-200:]}]")
os.close(fd)
print("test_rc12x_extras:", "PASS" if not fails else f"{len(fails)} failures")
sys.exit(1 if fails else 0)
