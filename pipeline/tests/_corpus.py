"""Shared fixture for the skeleton tests: the pipeline on sys.path and one cached corpus per (n, register), so the
5,000-skeleton sample is built once per test process. PLANCK_SKEL_N overrides the size for quick runs."""
import os
import sys

sys.dont_write_bytecode = True
PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)

import skeleton as S  # noqa: E402

N_BIG = int(os.environ.get("PLANCK_SKEL_N", "5000"))
_CACHE = {}


def corpus(n=N_BIG, register="RM"):
    key = (n, register)
    if key not in _CACHE:
        _CACHE[key] = [S.build(s, register) for s in range(n)]
    return _CACHE[key]
