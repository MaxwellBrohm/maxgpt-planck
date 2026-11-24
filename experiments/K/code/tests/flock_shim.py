#!/usr/bin/env python3
"""util-linux flock's fd forms for queue tests on a machine without it (the Mac): flock [-n | -w SECS | -u] FD. The
lock belongs to the open file description the shell holds on FD, so it stays held after this process exits, as with
util-linux flock."""
import fcntl
import sys
import time

a = sys.argv[1:]
fd = int(a[-1])
if a[0] == "-u":
    fcntl.flock(fd, fcntl.LOCK_UN)
    sys.exit(0)
wait = 0.0 if a[0] == "-n" else float(a[1]) if a[0] == "-w" else 1e9
t0 = time.time()
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(0)
    except OSError:
        if time.time() - t0 >= wait:
            sys.exit(1)
        time.sleep(0.05)
