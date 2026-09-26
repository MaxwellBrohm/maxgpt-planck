"""Tests added by the 2026-09-26 verification: find_guard must pick the training job's guard record (not an eval
job that reused the tag), and validate must catch problems across entries. Run: python3 models/tests/
test_registry_cross.py. REGISTRY_UNDER_TEST=DIR runs them against copies of the two modules (mutation checks)."""
import copy, json, os, sys, tempfile, unittest
SRC = os.environ.get("REGISTRY_UNDER_TEST") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import registry as R  # noqa: E402
import regtools as T  # noqa: E402

SHA = "ab" * 32


def entry(i, **kw):
    e = {"schema": 1, "id": f"E999/tiny/s{i}", "experiment": "E999", "arm": "tiny", "role": "scored", "seed": i,
         "kind": "from_scratch", "base_model": None, "params_total": 96, "params_body": 16,
         "tokenizer": {"id": "tok", "sha256": SHA}, "data": {"identity": "stream", "sha256": SHA},
         "code": {"git_commit": "abc1234", "relation": "pre_run", "digest_sha256": None, "digest_source": None},
         "config_sha256": SHA, "result": "not scored", "status": "diagnostic",
         "run": {"state": "completed", "started": None, "ended": None, "log": None},
         "files": [{"path": "final.pt", "bytes": 10, "sha256": SHA}], "locations": [f"mac:runs/s{i}"],
         "created": "2026-09-25T10:00:00-04:00", "release": "undecided", "notes": ""}
    e.update(kw)
    return e


class FindGuard(unittest.TestCase):
    def test_eval_job_with_the_same_tag_is_not_the_training_record(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "logs"))
            jobs = (("ft_s1", ["py", "-B", "e004_ft_test.py", "org/m", "--seed", "1"], "2026-09-25 10:00:00"),
                    ("gen_s1", ["py", "-B", "gen_probe.py", "org/m", "../weights/org__m__s1", "--tag", "s1"],
                     "2026-09-25 12:00:00"),
                    ("al_s1", ["py", "-B", "e005_al_ft_test.py", "org/m", "--tag", "s1"], "2026-09-25 13:00:00"))
            for name, cmd, ended in jobs:
                with open(os.path.join(d, "logs", name + ".guard.json"), "w") as f:
                    json.dump({"cmd": cmd, "started": "2026-09-25 09:00:00", "ended": ended, "exit": 0,
                               "killed": None}, f)
            path, g = T.find_guard(d, "org/m", "s1")
            self.assertTrue(path.endswith("/ft_s1.guard.json"), path)
            self.assertEqual(T.run_args(g["cmd"])[0], "e004_ft_test.py")


class CrossEntry(unittest.TestCase):
    def test_distinct_entries_pass(self):
        es = [entry(1), entry(2), entry(3, locations=["mac:runs/s1"], files=[{"path": "stable_400.pt", "bytes": 1,
                                                                              "sha256": SHA}])]
        self.assertEqual(R.validate_all(es), [])

    def test_same_file_at_the_same_location_twice_is_rejected(self):
        es = [entry(1), entry(2, locations=["mac:runs/s1"])]
        err = R.validate_all(es)
        self.assertTrue(any("mac:runs/s1/final.pt" in m and "E999/tiny/s1" in m for m in err), err)

    def test_checkpoint_parent_must_be_registered(self):
        ck = entry(2, kind="checkpoint", parent="E999/tiny/s1", step=400)
        self.assertEqual(R.validate_all([entry(1), ck]), [])
        bad = copy.deepcopy(ck)
        bad["parent"] = "E999/tiny/s7"
        err = R.validate_all([entry(1), bad])
        self.assertTrue(any("E999/tiny/s7" in m for m in err), err)

    def test_malformed_entries_are_reported_not_raised(self):
        es = [entry(1, files=5), entry(2, locations={"a": 1}, id=["x"]), entry(3, kind="checkpoint", parent=[1], step=1)]
        self.assertTrue(len(R.validate_all(es)) >= 3)

    def test_real_registry_has_no_cross_entry_errors(self):
        self.assertEqual(R.validate_all(R.load(os.path.join(SRC, "registry.jsonl"))), [])


if __name__ == "__main__":
    unittest.main()
