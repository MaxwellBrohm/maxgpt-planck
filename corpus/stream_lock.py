"""The shared heavy-CPU lock (~/planck/locks/pc_heavy.lock) for stream_core.py, the
MemAvailable worker cap, and the pool initializer that ends a worker whose parent died. GPU speed
benchmarks share the PC and are sensitive to CPU load, so heavy work holds this flock only while it
runs (per chunk), like `flock -w TIMEOUT`.
"""
import fcntl
import os
import signal
import threading
import time

MAX_WORKERS = 20


class LockTimeout(Exception):
    pass


class HeavyLock:
    """flock on the shared heavy-CPU lock file, like `flock -w timeout` (blocking wait with an
    alarm in the main thread, polling elsewhere). .waited = seconds spent waiting."""

    def __init__(self, path, timeout):
        self.path, self.timeout = os.path.expanduser(path), timeout

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd, t0 = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644), time.time()
        try:
            if threading.current_thread() is threading.main_thread():
                def on_alarm(*_):
                    raise LockTimeout(self.path)
                old = signal.signal(signal.SIGALRM, on_alarm)
                signal.setitimer(signal.ITIMER_REAL, max(0.01, self.timeout))
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)
            else:
                while True:
                    try:
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.time() - t0 > self.timeout:
                            raise LockTimeout(self.path)
                        time.sleep(1)
        except BaseException:
            os.close(self.fd)                 # also drops the lock if the alarm came late
            raise
        self.waited = round(time.time() - t0, 1)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def lock_free_now(path) -> bool:
    """True when nobody holds the lock at this instant (a hint; it can change right after)."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return True
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def mem_cap(cfg):
    try:
        with open("/proc/meminfo") as f:
            kb = next(int(x.split()[1]) for x in f if x.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return MAX_WORKERS
    return max(1, int((kb / 2**20 - cfg["mem_reserve_gb"]) / cfg["mem_per_worker_gb"]))


def exit_with_parent(ppid):
    """Pool initializer (initargs=(os.getpid(),)): a thread in the worker ends it (os._exit) within
    a second of the parent's death, so a SIGKILLed runner leaves no orphaned workers holding memory,
    or CPU without the heavy lock. The parent's death shows as a changed getppid() (reparenting)."""
    def watch():
        while os.getppid() == ppid:
            time.sleep(1)
        os._exit(1)
    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()
