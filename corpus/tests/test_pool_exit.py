"""stream_lock.exit_with_parent: the pool workers of pass A (stream_loop._executor), of
core_finalize.run_chunked and of near-dedup's band pool end within seconds when their parent is
SIGKILLed, instead of living on as orphans. Each case runs a parent script that starts a pool
task (it records its pid, then sleeps a minute) and kills itself once a task is running."""
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.dirname(HERE)

SCRIPT = r'''
import os, signal, sys, threading, time
sys.path[:0] = [sys.argv[3]]
import numpy as np


def nap(*_):
    open(os.path.join(sys.argv[2], str(os.getpid())), "w").close()
    time.sleep(60)


def killer():
    while not os.listdir(sys.argv[2]):
        time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    mode, root = sys.argv[1], os.path.dirname(sys.argv[2])
    threading.Thread(target=killer, daemon=True).start()
    if mode == "loop":
        import stream_loop as L
        L._executor({}, 1).submit(nap).result()
    elif mode == "finalize":
        import core_finalize as CF
        cfg = dict(stage1=root, lock_file=os.path.join(root, "heavy.lock"), lock_timeout=30,
                   workers=2, mem_reserve_gb=0, mem_per_worker_gb=0.001)
        CF.run_chunked(cfg, [1, 2], nap, print, "test")
    else:
        import neardedup as ND
        import neardedup_lsh as NL
        spec = []
        for k in range(2):
            stem = os.path.join(root, f"s{k}")
            ND.save_atomic(stem + ".mh.npy", ND.sketch([f"text number {k} of the test"]))
            spec.append({"stem": stem, "tier": 0})
        NL._bands_job = nap
        NL.find_near_dups(spec, os.path.join(root, "work"), workers=2)
'''


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("mode", ["loop", "finalize", "nd"])
def test_pool_workers_exit_when_the_parent_is_killed(tmp_path, mode):
    script, pids = tmp_path / "parent.py", tmp_path / "pids"
    script.write_text(SCRIPT)
    pids.mkdir()
    with open(tmp_path / "parent.log", "w") as log:     # a file, not a pipe: an orphan holding
        rc = subprocess.run([sys.executable, str(script), mode, str(pids), CORPUS],   # a pipe
                            stdout=log, stderr=log, timeout=60).returncode   # would stall run()
    assert rc == -9, (tmp_path / "parent.log").read_text()[-2000:]
    time.sleep(0.5)                                   # a second task may be starting
    left, t0 = [int(p) for p in os.listdir(pids)], time.time()
    while any(map(alive, left)) and time.time() - t0 < 10:
        time.sleep(0.1)
    stuck = [p for p in left if alive(p)]
    for p in stuck:
        os.kill(p, 9)                                 # do not leave them behind either way
    assert left and not stuck, f"workers {stuck} outlived their killed parent by 10 s"
