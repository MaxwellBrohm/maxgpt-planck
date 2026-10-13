"""Fixture for tests/test_e3p1b_waiter.py: a planck home (E3 and SCREENS queue logs, marks, robust_waiter.log) and a
planck-kit whose planck_e3p1b.bundle is a fixture repo (CURRENT, the real plan, the five code trees as small stand-ins,
and a stub queue_e3.sh that appends the queue log lines of one outcome per call). macOS lacks flock, sha256sum and
GNU df: stubs stand in. Every git call runs without git's location variables (an inherited GIT_DIR once let a fixture
commit onto the real repository's main, notes.txt PART 3 RESULT, INCIDENT)."""
import os
import subprocess

E3 = os.environ.get("E3_TEST_ROOT") and os.path.join(os.environ["E3_TEST_ROOT"], "experiments", "E3_seed_noise") or \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(E3, "e3p1b_waiter.sh")
PLAN = os.path.join(E3, "plans", "part1b_p3at20m.txt")
TREES = ("harness", "data_prep", "tokenizer", "corpus", "experiments/E2_lr_transfer")
GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR")
E3_PRIOR = ["2026-10-11 13:12:28 queue start: exp E3, plan part3.txt (sha256 07018f0fec572994), code commit " + "4" * 40,
            "2026-10-11 22:00:56 MARK E3 PART 3 DONE", "2026-10-11 22:00:56 plan finished: part3.txt"]
S_LOG = ["2026-10-12 00:57:33 queue start: exp SCREENS, plan robustness_k5.txt (sha256 54c68c8b1bd44a5f), code commit " + "1" * 40,
         "2026-10-12 17:32:46 MARK SCREENS ROBUSTNESS K5 DONE", "2026-10-12 17:32:46 plan finished: robustness_k5.txt"]
RW_DONE = "2026-10-12 17:38:46 DONE: robustness_k5 ended cleanly at its mark and the C6 robust group is done"
STUB = r"""#!/bin/bash
echo "$* QMIN=$QUEUE_MIN_FREE_GB" >> "$E3W_STUBLOG"
Q=$(dirname "$(dirname "$(dirname "$0")")")
LOG=$E3W_HOME/runs/E3/queue_e3.txt
o=$(head -1 "$E3W_OUTCOMES"); sed -i.bak 1d "$E3W_OUTCOMES"
t() { echo "$(date '+%F %T') $*" >> "$LOG"; }
[ "$o" = refuse ] && { t "refusing: uncommitted changes in the code checkout:"; exit 1; }
sha=$(shasum -a 256 "$Q/experiments/E3_seed_noise/plans/part1b_p3at20m.txt" | cut -c1-16)
[ "$o" = othersha ] && sha=0123456789abcdef
t "queue start: exp E3, plan part1b_p3at20m.txt (sha256 $sha), code commit $(git -C "$Q" rev-parse HEAD)"
t "seen E3: E3 PART 3 DONE"; t "seen SCREENS: SCREENS ROBUSTNESS K5 DONE"
case $o in
  clean|othersha) t "MARK E3 PART 3 AT 20M DONE"; touch "$E3W_HOME/runs/E3/marks/E3_PART_3_AT_20M_DONE"; t "plan finished: part1b_p3at20m.txt"; exit 0 ;;
  nomark) t "MARK E3 PART 3 AT 20M DONE"; t "plan finished: part1b_p3at20m.txt"; exit 0 ;;
  lockwait) t "waiting for gpu.lock"; t "gpu.lock not free after 2 h"; t "queue stops at: train e3_5m_A_s12"; exit 1 ;;
  lockwait20) t "MARK E3 PART 1B DONE"; t "gpu.lock not free after 2 h"; t "queue stops at: train e3_20m_A_s1_P3"; exit 1 ;;
  lockwait_screens) nohup /bin/sleep 3.14159 > /dev/null 2>&1 &
                    t "gpu.lock not free after 2 h"; t "queue stops at: train e3_5m_A_s13"; exit 1 ;;
  memwait) t "GPU memory in use [8328] MiB > 3000 with the lock held: waiting"; t "queue stops at: train e3_20m_A_init4"; exit 1 ;;
  fail) t "scoring failed for final_00007630 (see score.err)"; t "queue stops at: train e3_5m_A_s12"; exit 1 ;;
  disk) t "disk: 40 GB free (< 41): queue stops"; t "queue stops at: train e3_5m_A_s14"; exit 1 ;;
  kill) exit 137 ;;
  clean_rc1) t "MARK E3 PART 3 AT 20M DONE"; touch "$E3W_HOME/runs/E3/marks/E3_PART_3_AT_20M_DONE"; t "plan finished: part1b_p3at20m.txt"; exit 1 ;;
  lockwait_rc0) t "gpu.lock not free after 2 h"; t "queue stops at: train e3_5m_A_s12"; exit 0 ;;
esac
"""


def clean_env(**over):
    return {**{k: v for k, v in os.environ.items() if k not in GIT_ENV}, **over}


def git(cwd, *a):
    env = clean_env(DEVELOPER_DIR="/Library/Developer/CommandLineTools", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=fixture", "-c", "init.defaultBranch=main", *a],
                          cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def home(tmp, done=False, robust=True, rw_done=True, s_log=None, e3_log=None):
    p = tmp / "planck"
    for d in ("runs/E3/marks", "runs/SCREENS/marks", "logs"):
        os.makedirs(p / d)
    (p / "runs" / "E3" / "queue_e3.txt").write_text("\n".join(E3_PRIOR if e3_log is None else e3_log) + "\n")
    for m in ("E3_PART_1_DONE", "E3_PART_2_DONE", "E3_PART_3_DONE") + (("E3_PART_3_AT_20M_DONE",) if done else ()):
        (p / "runs" / "E3" / "marks" / m).touch()
    if robust:
        (p / "runs" / "SCREENS" / "marks" / "SCREENS_ROBUSTNESS_K5_DONE").touch()
    (p / "runs" / "SCREENS" / "queue_screens.txt").write_text("\n".join(S_LOG if s_log is None else s_log) + "\n")
    (p / "logs" / "robust_waiter.log").write_text("2026-10-12 17:32:46 step 3\n" + (RW_DONE + "\n" if rw_done else ""))
    return p


def kit(tmp, current="part1b_p3at20m", plan=None, bundle=True):
    """planck-kit: the waiter and a fixture repo as planck_e3p1b.bundle; returns (kit, commit, TREES for E3W_TREES)."""
    k, src = tmp / "kit", tmp / "src"
    x = src / "experiments" / "E3_seed_noise"
    os.makedirs(x / "plans")
    os.makedirs(k)
    for t in TREES:
        os.makedirs(src / t, exist_ok=True)
        (src / t / "stand_in.txt").write_text(t + "\n")
    (x / "plans" / "CURRENT").write_text(f"# fixture\n{current}\n")
    (x / "plans" / "part1b_p3at20m.txt").write_bytes(open(PLAN, "rb").read() if plan is None else plan)
    (x / "queue_e3.sh").write_text(STUB)
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "fixture")
    if bundle:
        git(src, "bundle", "create", "-q", str(k / "planck_e3p1b.bundle"), "HEAD", "main")
    (k / "e3p1b_waiter.sh").write_text(open(W).read())
    return k, git(src, "rev-parse", "HEAD"), " ".join(f"{t}:{git(src, 'rev-parse', 'HEAD:' + t)}" for t in TREES)


def run(tmp, p, k, trees, outcomes=(), maxr="24", flock_rc="0", pat="e3w_no_e3_process_[0-9]{12}",
        spat="e3w_no_screens_process_[0-9]{12}", poll="0", avail="167", args=(), timeout=60):
    os.makedirs(tmp / "bin", exist_ok=True)
    (tmp / "bin" / "flock").write_text(f"#!/bin/sh\nexit {flock_rc}\n")
    (tmp / "bin" / "sha256sum").write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
    (tmp / "bin" / "df").write_text('#!/bin/sh\nprintf "Avail\\n %sG\\n" "$(cat "$E3W_TEST_AVAIL")"\n')
    for f in ("flock", "sha256sum", "df"):
        os.chmod(tmp / "bin" / f, 0o755)
    if not (tmp / "avail").exists():
        (tmp / "avail").write_text(avail)
    (tmp / "outcomes").write_text("".join(o + "\n" for o in outcomes))
    env = clean_env(E3W_HOME=str(p), E3W_KIT=str(k), E3W_POLL=poll, E3W_MAXR=maxr, E3W_PAT=pat, E3W_SPAT=spat,
                    E3W_TREES=trees, E3W_STUBLOG=str(tmp / "stub.log"), E3W_OUTCOMES=str(tmp / "outcomes"),
                    E3W_TEST_AVAIL=str(tmp / "avail"), PATH=f"{tmp / 'bin'}:{os.environ['PATH']}",
                    DEVELOPER_DIR="/Library/Developer/CommandLineTools")
    r = subprocess.run(["/bin/bash", str(k / "e3p1b_waiter.sh"), *args], env=env, capture_output=True, text=True, timeout=timeout)
    lp = p / "logs" / "e3p1b_waiter.log"
    log = lp.read_text() if lp.exists() else ""
    stub = (tmp / "stub.log").read_text().splitlines() if (tmp / "stub.log").exists() else []
    return r.returncode, log + r.stdout, stub


def call(p, ref):
    return f"--plan {p / 'qcode' / ref[:12]}/experiments/E3_seed_noise/plans/part1b_p3at20m.txt QMIN=41"
