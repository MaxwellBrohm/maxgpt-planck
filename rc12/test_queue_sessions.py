"""Fakes for test_queue.py and its gpu.lock SESSION cases (notes STEP 9 QUEUE, STEP 9g). No model, no GPU.
As a program it is one of two fakes that setup() installs in <base>/shim (flock on the queue's PATH, fakepy as Q_PY):
  flock   util-linux flock with REAL locks (fcntl.flock): fd mode (flock [-n] [-E c] [-w s] FD) locks the caller's
          inherited fd, file mode (flock [...] FILE CMD...) holds FILE while CMD runs (CMD inherits it, as util-linux
          does). Every call goes to $SHIM_LOG (args, locked path); the first $FLOCK_FAIL gpu.lock calls exit with the -E
          code instead (a wait timeout); $FLOCK_CLOCK_ADD s are added to the fake clock per gpu.lock call (a wait).
  fakepy  logs every dev_batch.py process to $PYLOG (start and end records: the python it ran as, its args, gpu.lock
          held at its start and end, the queue's lock fds 8 / 9 it inherited) and every queue_status.py call (gpu.lock
          held or not). With $FAKE_DUR ("MODEL=s ...") a process is fake: it adds its seconds to the clock file $Q_CLOCK
          touches the STOP file if it is process number $FAKE_STOP_AT, leaves a grandchild (sleep $FAKE_CHILD, every
          inherited fd kept), sleeps $FAKE_SLEEP real seconds; without it the real dev_batch.py runs. Else: real python.
  python3 -B test_queue.py     (runs these cases too)"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
fails = []


def check(cond, what):
    if not cond:
        fails.append(what)


def fd_path(fd):
    if hasattr(fcntl, "F_GETPATH"):
        return fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0")[0].decode()
    return os.readlink(f"/proc/self/fd/{fd}")


def logrec(var, rec):
    if os.environ.get(var):
        with open(os.environ[var], "a") as f:
            f.write(json.dumps(rec) + "\n")


def read(p):
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def held(path):
    """True when some process holds an flock on path (tried on a fresh open file description)."""
    fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def flock_main(argv):
    code, wait, nb, i = 1, None, False, 0
    while argv[i].startswith("-"):
        if argv[i] == "-n":
            nb, i = True, i + 1
        elif argv[i] in ("-E", "-w"):
            code, wait = (int(argv[i + 1]), wait) if argv[i] == "-E" else (code, float(argv[i + 1]))
            i += 2
        else:
            return 64
    target, cmd = argv[i], argv[i + 1:]
    fd = int(target) if target.isdigit() else os.open(target, os.O_RDONLY | os.O_CREAT, 0o666)
    path = fd_path(fd)
    gpu = os.path.basename(path) == "gpu.lock"
    n = 1 + sum(os.path.basename(x["path"]) == "gpu.lock" for x in read(os.environ.get("SHIM_LOG", "/nonexistent")))
    logrec("SHIM_LOG", {"args": argv, "path": path})
    if gpu and os.environ.get("FLOCK_CLOCK_ADD"):
        t = int(open(os.environ["Q_CLOCK"]).read())
        open(os.environ["Q_CLOCK"], "w").write(str(t + int(os.environ["FLOCK_CLOCK_ADD"])))
    if gpu and n <= int(os.environ.get("FLOCK_FAIL") or 0):
        return code
    end = time.time() + (wait if wait is not None else 1e9)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if nb or time.time() >= end:
                return code
            time.sleep(0.02)
    if cmd:
        os.set_inheritable(fd, True)
        return subprocess.call(cmd, close_fds=False)
    return 0


def ok(f, *a):   # f(*a) raised no OSError (an open fd: os.fstat; a live pid: os.kill with 0)
    try:
        f(*a)
        return True
    except OSError:
        return False


def until(cond, tries):
    for _ in range(tries):
        if cond():
            return
        time.sleep(0.1)


def process(me, args):
    fds = [fd for fd in (8, 9) if ok(os.fstat, fd)]
    gl = os.path.join(os.environ["Q_LOCKS"], "gpu.lock")
    model = args[args.index("--responder") + 1].split(":", 1)[1]
    rec = {"py": me, "args": args, "model": model, "held0": held(gl), "fds": fds}
    fake = "FAKE_DUR" in os.environ
    if fake:
        c = os.environ["Q_CLOCK"]
        rec["t0"] = int(open(c).read())
        dur = dict(kv.split("=") for kv in os.environ["FAKE_DUR"].split())
        open(c, "w").write(str(rec["t0"] + int(dur.get(model, 0))))
        n = 1 + sum(x.get("ev") == "start" for x in read(os.environ["PYLOG"]))
        if str(n) == os.environ.get("FAKE_STOP_AT"):
            open(os.path.join(os.environ["Q_LOGS"], "rc12_dev_queue.STOP"), "w").close()
        if os.environ.get("FAKE_CHILD"):
            rec["child"] = subprocess.Popen(["sleep", os.environ["FAKE_CHILD"]], close_fds=False,
                                            start_new_session=True).pid
    logrec("PYLOG", dict(rec, ev="start"))
    if fake:
        time.sleep(float(os.environ.get("FAKE_SLEEP") or 0))
        rc = 0
    else:
        rc = subprocess.call([PY] + args, close_fds=False)
    logrec("PYLOG", dict(rec, ev="end", held1=held(gl), rc=rc))
    return rc


def py_main(argv):
    me, args = argv[0], argv[1:]
    if "dev_batch.py" in args:
        return process(me, args)
    if "queue_status.py" in args:
        logrec("PYLOG", {"ev": "status", "progress": "--progress" in args,
                         "held": held(os.path.join(os.environ["Q_LOCKS"], "gpu.lock"))})
    os.execv(PY, [PY] + args)


def setup(base):
    os.makedirs(f"{base}/shim")
    for name, role in (("flock", "flock"), ("fakepy", "py")):
        open(f"{base}/shim/{name}", "w").write(f'#!/bin/sh\nexec "{PY}" -B "{os.path.abspath(__file__)}" {role} '
                                               f'"$0" "$@"\n')
        os.chmod(f"{base}/shim/{name}", 0o755)
    json.dump({"models": {m: {"engine": "fake"} for m in ("IDEAL", "CAPPER", "SAMPLER", "NOPE")}},
              open(f"{base}/engines.json", "w"))
    open(f"{base}/clock", "w").write("1000000000")


def queue(base, models, seeds="greedy 1", renders="template plain", wait=True, **env):
    e = dict(os.environ, Q_CODE=HERE, Q_ROOT=f"{base}/root", Q_LOGS=f"{base}/logs", Q_LOCKS=f"{base}/locks",
             Q_PY=f"{base}/shim/fakepy", Q_ENGINES=f"{base}/engines.json", Q_MODELS=models, Q_SEEDS=seeds,
             Q_RENDERS=renders, Q_LIMIT="12", PATH=f"{base}/shim:" + os.environ["PATH"], SHIM_LOG=f"{base}/flock.log",
             PYLOG=f"{base}/py.log")
    e.update(env)
    for k in ("SHIM_LOG", "PYLOG"):
        if os.path.exists(e[k]):
            os.remove(e[k])
    cmd = ["bash", os.path.join(HERE, "queue_dev_baselines.sh")]
    return subprocess.run(cmd, env=e, timeout=300).returncode if wait else subprocess.Popen(cmd, env=e)


def status(base):
    p = f"{base}/root/queue_status.txt"
    return [dict([("key", " ".join(line.split()[:5]))] + [kv.split("=", 1) for kv in line.split()[5:]])
            for line in open(p)] if os.path.exists(p) else []


def lock_calls(base):
    gl = os.path.realpath(f"{base}/locks/gpu.lock")
    return [x["args"] for x in read(f"{base}/flock.log") if os.path.realpath(x["path"]) == gl]


def procs(base, ev="end"):
    return [x for x in read(f"{base}/py.log") if x["ev"] == ev]


def sizes(base):
    """session sizes in run order from the dev status lines' notes; None unless each counts processes 1, 2, ..."""
    out, last = [], None
    for x in status(base):
        if x["key"].endswith(" dev") and x.get("note", "").startswith("session_"):
            _, sid, _, k = x["note"].split("_")
            if sid != last:
                out.append(0)
                last = sid
            out[-1] += 1
            if int(k) != out[-1]:
                return None
    return out


def fake(tag, models, seeds, dur, renders="template", wait=True, **env):
    b = tempfile.mkdtemp(prefix=f"rc12_qs{tag}_")
    setup(b)
    return b, queue(b, models, seeds=seeds, renders=renders, wait=wait, Q_CLOCK=f"{b}/clock", FAKE_DUR=dur, **env)


def sessions():
    b, rc = fake("A", "IDEAL CAPPER", "greedy 1 2 3", "IDEAL=150 CAPPER=150", renders="template plain",
                 FLOCK_CLOCK_ADD="40", FAKE_CHILD="4")
    gl, p, lc = f"{b}/locks/gpu.lock", procs(b), lock_calls(b)
    kids = [x["child"] for x in p if ok(os.kill, x["child"], 0)]
    check(rc == 0 and len(p) == 16 and sizes(b) == [11, 5], f"A: 150 s processes, sessions {sizes(b)} (want 11, 5)")
    check(len(lc) == 2 and all(c == ["-E", "75", "-w", "7200", "9"] for c in lc), f"A: lock calls {lc}")
    check(all(x["held0"] and x["held1"] and x["fds"] == [] for x in p), "A: a process without the lock or with fd 8/9")
    check([x["lock_secs"] for x in status(b)] == ["150"] * 32, "A: t0 / t1 are not the process's own start and end")
    st = [x["held"] for x in read(f"{b}/py.log") if x["ev"] == "status"]
    check(st == [True] * 10 + [False] + [True] * 5 + [False], f"A: lock at status / DONE calls {st}")
    check(kids and not held(gl) and os.path.exists(f"{b}/logs/rc12_dev_queue.DONE"), f"A: exit, lock / kids {kids}")
    b, rc = fake("B", "CAPPER", "greedy 1 2", "CAPPER=1000")
    check(sizes(b) == [1, 1, 1] and len(lock_calls(b)) == 3, f"B: 1000 s processes, sessions {sizes(b)}")
    b, rc = fake("C", "IDEAL CAPPER", "greedy 1", "IDEAL=0 CAPPER=5000", Q_SESSION_S="0", Q_SESSION_TAIL="0")
    check(sizes(b) == [1, 1, 1, 1] and len(lock_calls(b)) == 4, f"C: SESSION_S 0, sessions {sizes(b)}")
    b, rc = fake("D", "IDEAL", "greedy 1 2 3", "IDEAL=10", FAKE_STOP_AT="2")
    log = open(f"{b}/logs/rc12_dev_queue.log").read()
    check(len(procs(b)) == 2 and len(lock_calls(b)) == 1 and sizes(b) == [2] and not held(f"{b}/locks/gpu.lock")
          and not os.path.exists(f"{b}/logs/rc12_dev_queue.DONE") and "(STOP file)" in log, "D: STOP in a session")
    b, q = fake("E", "IDEAL", "greedy", "IDEAL=10", wait=False, FAKE_SLEEP="2", FAKE_CHILD="8")
    until(lambda: procs(b, "start"), 300)
    q.send_signal(signal.SIGKILL)
    q.wait()
    time.sleep(0.3)
    gl, ql = f"{b}/locks/gpu.lock", f"{b}/locks/rc12_dev_queue.lock"
    during, qheld = not procs(b) and held(gl), held(ql)   # the process still runs, and gpu.lock is held
    until(lambda: procs(b), 100)
    time.sleep(0.5)
    s = procs(b, "start")
    kid = s[0]["child"] if s else None
    check(during and not qheld, f"E: queue killed mid-process: gpu.lock held {during}, queue lock held {qheld}")
    check(procs(b) and not held(gl) and kid and ok(os.kill, kid, 0) and s[0]["fds"] == [], "E: after the process")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit(flock_main(sys.argv[3:]) if sys.argv[1] == "flock" else py_main(sys.argv[2:]))
    sessions()
    print("\n".join(f"FAIL {f}" for f in fails) + f"\nsessions: {'PASS' if not fails else f'{len(fails)} FAILURES'}")
    sys.exit(1 if fails else 0)
