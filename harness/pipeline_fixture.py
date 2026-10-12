"""Shared test fixture: one FAKE pipeline run per test process (make_pipeline_fake.py, 200 skeletons,
about 6 s), or the run directory named by PLANCK_PIPELINE_FAKE (the mutation runner makes it once)."""
from __future__ import annotations

import atexit
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE: dict[str, str] = {}


def fake_run_dir(n: int = 200) -> str:
    if os.environ.get("PLANCK_PIPELINE_FAKE"):
        return os.environ["PLANCK_PIPELINE_FAKE"]
    if "run" not in _CACHE:
        root = tempfile.mkdtemp(prefix="planck_pfake_")
        atexit.register(shutil.rmtree, root, True)
        subprocess.run([sys.executable, "-B", os.path.join(HERE, "make_pipeline_fake.py"),
                        os.path.join(root, "p"), "--n", str(n)], check=True, capture_output=True, timeout=110)
        _CACHE["run"] = os.path.join(root, "p", "run")
    return _CACHE["run"]


def read_raw(run: str) -> list[tuple[str, int, bytes, dict]]:
    """[(file, line number, line bytes, record)] straight from the pipeline's accepted shards."""
    out = []
    for p in sorted(glob.glob(os.path.join(run, "accepted", "accepted-*.jsonl"))):
        with open(p, "rb") as f:
            out += [(p, n, ln, json.loads(ln)) for n, ln in enumerate(f, 1) if ln.strip()]
    return out
