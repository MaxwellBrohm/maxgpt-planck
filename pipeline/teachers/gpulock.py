"""Is ~/planck/locks/gpu.lock held by this process or one of its ancestors? (Linux /proc/locks; the PC only.)

serve.load() refuses to load a teacher unless it is, so a teacher is never on the shared 5070 outside the lock.
The PC's callers take the lock with flock(1):

    flock -w 7200 ~/planck/locks/gpu.lock timeout -k 60 1500 python capacity.py ...

flock(1) takes the lock in its own process and runs the command as its child, so the holder /proc/locks names is
an ancestor of the python process (a bash `exec 9>lock; flock 9` holder is the python's parent shell). A blocked
waiter ("->" lines) does not count. Where /proc/locks does not exist (the Mac) nothing holds it, so nothing loads.
When /proc/locks names no ancestor, an ancestor's /proc/<pid>/fdinfo lock lines are checked (fdinfo_holds)."""
import os

DEFAULT = os.path.expanduser(os.environ.get("PLANCK_GPU_LOCK", "~/planck/locks/gpu.lock"))


class GpuLockError(RuntimeError):
    pass


def holders(path, locks_text):
    """pids holding an exclusive flock on path, from the text of /proc/locks
    ('1: FLOCK  ADVISORY  WRITE 356173 08:30:277664 0 EOF'; device major:minor in hex, inode in decimal)."""
    st = os.stat(path)
    key = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    out = []
    for ln in locks_text.splitlines():
        f = ln.split()
        if "->" in f or len(f) < 6 or f[1] != "FLOCK" or f[3] != "WRITE":
            continue
        try:
            maj, mnr, ino = f[5].split(":")
            if (int(maj, 16), int(mnr, 16), int(ino)) == key:
                out.append(int(f[4]))
        except ValueError:
            continue
    return out


def ancestors(pid=None, proc="/proc"):
    """[pid, parent, grandparent, ...] from /proc/<pid>/stat (the comm field may hold spaces and parentheses)."""
    pid, chain = pid or os.getpid(), []
    while pid > 0 and pid not in chain:
        chain.append(pid)
        try:
            with open(os.path.join(proc, str(pid), "stat")) as f:
                s = f.read()
        except OSError:
            break
        pid = int(s[s.rindex(")") + 2:].split()[1])
    return chain


def fdinfo_holds(path, pid, proc="/proc"):
    """does one of pid's open files hold an exclusive flock on path? (/proc/<pid>/fdinfo/<fd> 'lock:' lines, which
    list granted locks only, never a blocked waiter's). On the PC at 22:29 on 2026-09-26 /proc/locks left out a live
    holder of gpu.lock (RC-12's flock) whose fdinfo showed the lock, so check() falls back to this."""
    d = os.path.join(proc, str(pid), "fdinfo")
    try:
        names = os.listdir(d)
    except OSError:
        return False
    for n in names:
        try:
            with open(os.path.join(d, n)) as f:
                lines = [ln[len("lock:"):].strip() for ln in f if ln.startswith("lock:")]
        except OSError:
            continue
        if lines and holders(path, "\n".join(lines)):
            return True
    return False


def check(path=DEFAULT, locks_path="/proc/locks", proc="/proc", pid=None):
    """-> the holder's pid when it is this process or an ancestor; GpuLockError otherwise."""
    try:
        with open(locks_path) as f:
            text = f.read()
    except OSError as e:
        raise GpuLockError(f"cannot read {locks_path} ({e}): no teacher loads on this machine") from e
    if not os.path.exists(path):
        raise GpuLockError(f"{path} does not exist")
    held = holders(path, text)
    mine = ancestors(pid, proc)
    for p in held:
        if p in mine:
            return p
    for p in mine:
        if fdinfo_holds(path, p, proc):
            return p
    raise GpuLockError(f"{path} is not held by this process or an ancestor (holders: {held or 'none'}); "
                       "run under flock -w 7200 ~/planck/locks/gpu.lock")
