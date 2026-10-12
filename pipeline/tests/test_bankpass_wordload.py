"""Bank pass W0: the word list loader (bankpass/wordload.py) on a fixture word list built by wordfam from a fixture
count table (no core v0 on the Mac). Every word here is FAKE fixture text. Clean cases load; every planted defect
(no provenance header, a hash that differs, a missing column) is refused; FAKE is the fallback."""
import gzip
import json
import os
import tempfile
import unittest
from unittest import mock

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import fake_data as F
import pools as P
from bankpass import plan, wordfam as WF, wordload as WLd, wordtable as WT

H = "ab" * 32


def fixture_list(d):
    """a word list TSV written by wordfam over a small count table; returns its path."""
    rows = {"walk": dict(verb=40, noun=5), "walks": dict(noun=2, verb=1), "walked": dict(verb=3),
            "dog": dict(noun=40, a=30), "dogs": dict(noun=9), "hour": dict(noun=20, an=12, a=1),
            "rice": dict(noun=9, mass=8, a=1), "big": dict(adj=30), "paris": dict(noun=5, mid=20, capmid=19)}
    head = {"docs": {g: 1000 for g in WT.GROUPS}, "input_manifest_sha256": H, "code_sha256": H,
            "columns": ["word"] + [f"df_{g}" for g in WT.GROUPS] + ["tf"] + list(WT.CTX_FEATS)}
    cp = os.path.join(d, "counts_X.tsv.gz")
    with gzip.open(cp, "wt") as f:
        f.write("#" + json.dumps(head) + "\n")
        for w, r in rows.items():
            r = dict({"mid": 10, "capmid": 0}, **r)
            f.write("\t".join(map(str, [w] + [60 if g in WT.VOTING else 0 for g in WT.GROUPS] + [200]
                                  + [r.get(c, 0) for c in WT.CTX_FEATS])) + "\n")
    return WF.write(WF.build(cp), d, "X")


class TestWordLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = fixture_list(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_families_lists_and_provenance(self):
        wl = WLd.read(self.path)
        self.assertEqual((wl.provenance, wl.ref[:9], wl.meta["input_manifest_sha256"]), ("computed", "wordlist@", H))
        self.assertEqual([wl.head(w) for w in ("walked", "Walks", "dogs", "zebra")], ["walk", "walk", "dog", "zebra"])
        self.assertTrue(wl.in_list("walking") is False and wl.in_list("walked") and wl.in_list("dog", "RS"))
        self.assertFalse(wl.in_list("paris"))                       # a proper-noun family has no rank, no list
        self.assertEqual(wl.required("noun"), [])                   # POS pending: no required word yet
        self.assertIn("dog", wl.seed_nouns())
        self.assertNotIn("big", wl.seed_nouns())
        self.assertLessEqual({"walk", "walked", "dogs", "paris"}, wl.english())

    def test_features_from_counts(self):
        wl = WLd.read(self.path)
        self.assertEqual(wl.features("hour"), {"article": "an", "number": "sg", "mass": False, "basis": "counted"})
        self.assertEqual(wl.features("dogs")["number"], "pl")
        self.assertEqual(wl.features("dogs")["article"], "")
        self.assertEqual(wl.features("rice")["article"], "")        # mass: "much rice" outnumbers "a rice"
        self.assertEqual(wl.features("umbrella stand"), {"article": "an", "number": "sg", "mass": False,
                                                          "basis": "vowel_rule"})

    def test_planted_files_refused(self):
        with self.assertRaises(WLd.Refused):
            WLd.read(self.path, expect_sha256="0" * 64)
        lines = open(self.path).read().split("\n")
        cases = {"header": [ln for ln in lines if "input manifest sha256" not in ln],
                 "column": [ln.replace("\trequired_ok", "\tok") if ln.startswith("headword") else ln for ln in lines]}
        for name, ls in cases.items():
            with self.subTest(case=name):
                p = os.path.join(self.tmp.name, name + ".tsv")
                with open(p, "w") as f:
                    f.write("\n".join(ls))
                with self.assertRaises(WLd.Refused):
                    WLd.read(p)

    def test_manifest_sha_and_expected_hash(self):
        mp = os.path.join(self.tmp.name, "MANIFEST.sha256")
        from bankpass import store
        with open(mp, "w") as f:
            f.write(f"{'1' * 64}  other.tsv\n{store.sha256_file(self.path)}  wordlist_v0.X.tsv\n")
        sha = WLd.manifest_sha(mp, "wordlist_v0.X.tsv")
        self.assertEqual(WLd.read(self.path, expect_sha256=sha).meta["file_sha256"], sha)
        with self.assertRaises(WLd.Refused):
            WLd.manifest_sha(mp, "missing.tsv")

    def test_fake_fallback_and_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(WLd.ENV, None)
            wl = WLd.load()
        self.assertEqual((wl.provenance, wl.ref), ("FAKE", "wordlist@fake:FAKE"))
        self.assertEqual(wl.required("noun"), list(dict.fromkeys(F.WORD_NOUNS)))
        with mock.patch.dict(os.environ, {WLd.ENV: self.path}):
            self.assertEqual(WLd.load().provenance, "computed")

    def test_install_english_and_plan_seeds(self):
        wl = WLd.read(self.path)
        self.assertNotIn("walked", P._ENGLISHISH)
        undo = WLd.install_english(wl)
        try:
            self.assertIn("walked", P._ENGLISHISH)
        finally:
            undo()
        self.assertNotIn("walked", P._ENGLISHISH)
        self.assertIn("window", P._ENGLISHISH)
        WLd.install_english(WLd.fake())()                          # undo removes only what it added
        self.assertIn("window", P._ENGLISHISH)
        calls = plan.pool_calls(wl.seed_nouns())
        self.assertTrue(all(set(c["seed_words"]) <= set(wl.seed_nouns()) for c in calls))
        self.assertTrue(all(set(c["seed_words"]) <= set(P.pool("req_noun").values) for c in plan.pool_calls()))


if __name__ == "__main__":
    unittest.main()
