"""The driver's state files are published durably (incident 2026-09-29: a power cut left renamed files at 0 bytes;
an empty run.json or decode.json stops every resume of that run dir). os.fsync and os.replace are wrapped to record,
in call order, the inode each fsync touched and each rename's destination: run.json (driver_state.pin, first session
and a resume) and decode.json (driver.pin_flags) must be fsynced BEFORE the rename and their directory AFTER; the
copy of durable.py here must equal harness/durable.py. No model, no server."""
import json
import os
import shutil
import tempfile
import unittest

import driver_fixtures as DF  # noqa: F401  (puts the pipeline on sys.path)
import driver
import driver_state
import durable

PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path, mode="r"):
    with open(path, mode) as f:
        return json.load(f) if mode == "r" else f.read()


class Rec:
    def __init__(self, test):
        self.ev = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            st = os.fstat(fd)
            self.ev.append(("fsync", (st.st_dev, st.st_ino)))
            return real_fsync(fd)

        def replace(src, dst, *a, **k):
            real_replace(src, dst, *a, **k)
            self.ev.append(("replace", os.path.abspath(dst)))

        os.fsync, os.replace = fsync, replace
        test.addCleanup(setattr, os, "fsync", real_fsync)
        test.addCleanup(setattr, os, "replace", real_replace)

    def fsynced(self, path, lo=0, hi=None):
        st = os.stat(path)
        return ("fsync", (st.st_dev, st.st_ino)) in self.ev[lo:hi]

    def check_file(self, test, path):
        idx = [i for i, e in enumerate(self.ev) if e == ("replace", os.path.abspath(path))]
        test.assertTrue(idx, f"{path}: never renamed into place")
        i = idx[-1]
        test.assertTrue(self.fsynced(path, 0, i), f"{os.path.basename(path)}: data not fsynced before its rename")
        test.assertTrue(self.fsynced(os.path.dirname(os.path.abspath(path)), i + 1),
                        f"{os.path.basename(path)}: directory not fsynced after the rename")


class TestDurableState(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="planck_durable_")
        self.addCleanup(shutil.rmtree, self.out, True)

    def cfg(self, n=10):
        return {k: f"v-{k}" for k in driver_state.PINNED} | {"n": n, "created": "2026-10-02"}

    def test_run_json_first_session_and_resume(self):
        rec = Rec(self)
        driver_state.pin(self.out, self.cfg())
        path = os.path.join(self.out, "run.json")
        rec.check_file(self, path)
        rec.ev.clear()
        have = driver_state.pin(self.out, self.cfg(n=20))
        rec.check_file(self, path)
        self.assertEqual((have["sessions"], have["n"]), (2, 20))
        self.assertEqual(read(path)["sessions"], 2)

    def test_decode_json(self):
        rec = Rec(self)
        flags = dict(driver.FLAGS_OFF, structured="labels_exact")
        driver.pin_flags(self.out, flags, fresh=True)
        path = os.path.join(self.out, "decode.json")
        rec.check_file(self, path)
        self.assertEqual(read(path), flags)

    def test_copy_matches_harness(self):
        here = read(os.path.join(PIPE, "durable.py"), "rb")
        canon = os.path.join(os.path.dirname(PIPE), "harness", "durable.py")
        if os.path.exists(canon):                      # a pipeline synced alone has no harness/ next to it
            self.assertEqual(here, read(canon, "rb"), "pipeline/durable.py drifted from harness/durable.py")
        self.assertTrue(callable(durable.atomic_write))


if __name__ == "__main__":
    unittest.main()
