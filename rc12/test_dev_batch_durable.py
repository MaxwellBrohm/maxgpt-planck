"""dev_batch.py publishes a finished run durably (incident 2026-09-29, experiments/E2_lr_transfer/notes.txt CRASH
RESUME: a power cut left renamed files at 0 bytes). In-process with fake:IDEAL_ALT, no model. os.fsync and os.replace
are wrapped to record, in call order, the inode each fsync touched and each rename's destination. For every run dir:
  files    transcripts.jsonl, scores.jsonl, meta.json and DONE are each fsynced BEFORE the <dir>.partial rename
  dir      the run dir itself is fsynced before the rename and its parent (<root>/<model>/<render>) after it
  rules    fsync_path raises for a file; a directory fsync (or open) the filesystem refuses never raises on any
           platform (a drvfs /mnt/d mount must not crash a run) and is warned once per error kind on stderr
Run: python3 -B test_dev_batch_durable.py   (exit 1 on any failure)"""
import errno
import io
import os
import sys
import tempfile

import dev_batch as DB

RUN_FILES = ("transcripts.jsonl", "scores.jsonl", "meta.json", "DONE")


def key(p):
    st = os.stat(p)
    return st.st_dev, st.st_ino


def recorded(fn):
    """run fn() with os.fsync / os.replace recording; -> the event list."""
    ev, real_fsync, real_replace = [], os.fsync, os.replace

    def fsync(fd):
        st = os.fstat(fd)
        ev.append(("fsync", (st.st_dev, st.st_ino)))
        return real_fsync(fd)

    def replace(src, dst, *a, **k):
        real_replace(src, dst, *a, **k)
        ev.append(("replace", os.path.abspath(dst)))

    os.fsync, os.replace = fsync, replace
    try:
        fn()
    finally:
        os.fsync, os.replace = real_fsync, real_replace
    return ev


def check_run(ev, d):
    fails = []
    idx = [i for i, e in enumerate(ev) if e == ("replace", os.path.abspath(d))]
    if not idx:
        return [f"{d}: never renamed into place"]
    i = idx[-1]
    for n in RUN_FILES:
        p = os.path.join(d, n)
        if not os.path.exists(p):
            fails.append(f"files: {d}/{n} missing")
        elif ("fsync", key(p)) not in ev[:i]:
            fails.append(f"files: {os.path.basename(d)}/{n} not fsynced before the .partial rename")
    if ("fsync", key(d)) not in ev[:i]:
        fails.append(f"dir: {os.path.basename(d)} not fsynced before its rename")
    if ("fsync", key(os.path.dirname(d))) not in ev[i + 1:]:
        fails.append(f"dir: parent of {os.path.basename(d)} not fsynced after the rename")
    return fails


def rules(root):
    fails = []
    real_fsync, real_platform, real_stderr = os.fsync, sys.platform, sys.stderr

    def raising(code):
        def f(fd):
            raise OSError(code, os.strerror(code))
        return f
    p = os.path.join(root, "f")
    open(p, "w").close()
    try:
        DB.fsync_path(root, is_dir=True)                     # the real call on this machine
        os.fsync = raising(errno.EIO)
        try:
            DB.fsync_path(p)
            fails.append("rules: a failing file fsync was swallowed")
        except OSError:
            pass
        DB._DIR_WARNED.clear()
        sys.stderr = err = io.StringIO()
        for plat in ("darwin", "linux"):
            sys.platform = plat
            for code in (errno.EINVAL, errno.EACCES, errno.EPERM, errno.EIO):
                os.fsync = raising(code)
                DB.fsync_path(root, is_dir=True)
            os.fsync = real_fsync
            DB.fsync_path(os.path.join(root, "no_such_dir"), is_dir=True)   # the open itself refused
        warned = err.getvalue()
        for code in ("EINVAL", "EACCES", "EPERM", "EIO", "ENOENT"):
            if warned.count(f"({code})") != 1:
                fails.append(f"rules: directory {code} warned {warned.count(f'({code})')} times, not once")
    except OSError as e:
        fails.append(f"rules: a directory fsync raised ({e}); it must only warn (drvfs /mnt/d)")
    finally:
        os.fsync, sys.platform, sys.stderr = real_fsync, real_platform, real_stderr
    return fails


def checks():
    fails = []
    with tempfile.TemporaryDirectory() as root:
        argv = ["dev_batch.py", "--responder", "fake:IDEAL_ALT", "--render", "plain", "--root", root, "--limit", "6",
                "--seeds", "greedy,1"]
        old = sys.argv
        sys.argv = argv
        try:
            ev = recorded(DB.main)
        except SystemExit as e:
            return [f"run: dev_batch exited {e.code}"]
        finally:
            sys.argv = old
        base = os.path.join(root, "IDEAL_ALT", "plain")
        runs = [os.path.join(base, n) for n in ("greedy", "greedy_owncf", "1", "1_owncf")]
        for d in runs:
            fails += check_run(ev, d)
        fails += rules(root)
    return fails


def main():
    try:
        fails = checks()
    except Exception as e:  # noqa: BLE001  (a raising check is a failure line, not a crash)
        fails = [f"checks raised {type(e).__name__}: {str(e)[:200]}"]
    print("\n".join(f"FAIL {f}" for f in fails) or "ALL DEV_BATCH DURABLE CHECKS PASS (fake:IDEAL_ALT, no model)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
