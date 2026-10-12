"""RESULTS.txt is reproducible from the committed raw outputs: runs.tgz holds exactly the files runs.sha256 lists,
results.py on them prints RESULTS.txt's TABLES section verbatim, and the independent rederive.py gives the same picks
as the PICKS line."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
T001 = os.path.dirname(HERE)
sys.path.insert(0, T001)
import rederive  # noqa: E402
import results  # noqa: E402

MARK = "TABLES (results.py --out <runs.tgz extracted>, verbatim)\n"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("t001_runs")
    with tarfile.open(os.path.join(T001, "runs.tgz")) as tf:
        tf.extractall(out, filter="data")
    return str(out)


def test_archive_matches_sha_list(runs):
    with open(os.path.join(T001, "runs.sha256")) as f:
        want = dict(reversed(line.split()) for line in f if line.strip())
    have = {}
    for name in os.listdir(runs):
        with open(os.path.join(runs, name), "rb") as f:
            have[name] = hashlib.sha256(f.read()).hexdigest()
    assert have == want and len(have) == 115


def test_results_tables_verbatim(runs):
    with open(os.path.join(T001, "RESULTS.txt")) as f:
        text = f.read()
    assert text.count(MARK) == 1
    lines, picks = results.build(runs)
    body = "\n".join(lines + ["PICKS " + json.dumps(picks)]) + "\n"
    assert text.split(MARK)[1] == body


def test_rederive_picks_equal_the_picks_line(runs):
    with open(os.path.join(T001, "RESULTS.txt")) as f:
        picks = json.loads([ln for ln in f if ln.startswith("PICKS {")][0][6:])
    ind = rederive.main(runs)
    assert (ind["R1"], ind["R2"], ind["R3"]) == (picks["R1"], picks["R2"], picks["R3"])
    assert ind["runs"] == 115 and ind["c0"]["a"] and not ind["c0"]["b"]
