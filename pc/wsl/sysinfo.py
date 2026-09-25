"""Host checks for the PC runner and heartbeat: free disk and WSL memory (stdlib only)."""
from __future__ import annotations

import os
import shutil
import socket
import time


def free_gb(path: str) -> float:
    """Free space in GB (1e9 bytes) on the filesystem holding path."""
    p = os.path.expanduser(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free / 1e9


def mem_available_mib(meminfo: str = "/proc/meminfo") -> float | None:
    """MemAvailable in MiB, or None where /proc/meminfo does not exist (the Mac)."""
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def host() -> str:
    return socket.gethostname()
