"""jobq.py publishes job and status files durably (incident 2026-09-29: a power cut left renamed files at 0 bytes;
an empty job file in queue/running/ is unreadable, so the runner moves it to failed/ and never retries it).
os.fsync and os.replace are wrapped to record, in call order, the inode each fsync touched and each rename's
destination: save() and write_json() fsync the data BEFORE the rename and the directory AFTER; move() fsyncs both
directories after the rename (the file it moves was fsynced when it was saved). CPU only, stdlib only."""
import os

import pytest

import durable
import jobq

PC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Rec:
    def __init__(self, mp):
        self.ev = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            st = os.fstat(fd)
            self.ev.append(("fsync", (st.st_dev, st.st_ino)))
            return real_fsync(fd)

        def replace(src, dst, *a, **k):
            real_replace(src, dst, *a, **k)
            self.ev.append(("replace", os.path.abspath(dst)))

        mp.setattr(os, "fsync", fsync)
        mp.setattr(os, "replace", replace)

    def fsynced(self, path, lo=0, hi=None):
        st = os.stat(path)
        return ("fsync", (st.st_dev, st.st_ino)) in self.ev[lo:hi]

    def renamed(self, path):
        idx = [i for i, e in enumerate(self.ev) if e == ("replace", os.path.abspath(path))]
        assert idx, f"{path}: never renamed into place"
        return idx[-1]

    def check_file(self, path):
        i = self.renamed(path)
        assert self.fsynced(path, 0, i), f"{os.path.basename(path)}: data not fsynced before its rename"
        assert self.fsynced(os.path.dirname(path), i + 1), f"{os.path.basename(path)}: directory not fsynced after"
        return i


@pytest.fixture
def home(tmp_path):
    jobq.ensure_layout(str(tmp_path))
    return str(tmp_path)


def test_save_order(monkeypatch, home):
    rec = Rec(monkeypatch)
    p = os.path.join(jobq.qdir(home, "pending"), "010_x.json")
    jobq.save({"name": "x", "cmd": ["true"]}, p)
    rec.check_file(p)
    assert jobq.load(p)["cmd"] == ["true"]


def test_write_json_order(monkeypatch, home):
    rec = Rec(monkeypatch)
    p = os.path.join(home, "status", "runner.json")
    jobq.write_json(p, {"state": "idle"})
    rec.check_file(p)


def test_move_fsyncs_both_directories(monkeypatch, home):
    p = os.path.join(jobq.qdir(home, "pending"), "010_x.json")
    jobq.save({"name": "x", "cmd": ["true"]}, p)
    rec = Rec(monkeypatch)
    dst = jobq.move(p, home, "running")
    i = rec.renamed(dst)
    assert rec.fsynced(jobq.qdir(home, "running"), i + 1), "running/ not fsynced after the move"
    assert rec.fsynced(jobq.qdir(home, "pending"), i + 1), "pending/ not fsynced after the move"


def test_copy_matches_harness():
    with open(os.path.join(PC, "wsl", "durable.py"), "rb") as f:
        here = f.read()
    canon = os.path.join(os.path.dirname(PC), "harness", "durable.py")
    if os.path.exists(canon):                          # a pc/ kit copied alone has no harness/ next to it
        with open(canon, "rb") as f:
            assert here == f.read(), "pc/wsl/durable.py drifted from harness/durable.py"
    assert callable(durable.atomic_write)
