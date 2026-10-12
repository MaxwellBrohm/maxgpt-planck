"""Bank pass W2: source fetch with licence evidence, and the human pools (names by era and sex, surnames, cities).
Fixture sources only (tests/_human_fx.py, FAKE names and places in a temp dir); no network, no model."""
import collections
import json
import os
import tempfile
import unittest
from unittest import mock

import _human_fx as FX
from bankpass import human, humanseed, sources


def by_text(items):
    return {it["text"]: it for it in items}


class TestSources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_fetch_records_hashes_and_licence(self):
        recs = FX.make_sources(self.root)
        for name in ("ssa_names", "census_surnames", "geonames_cities", "ldnoobw", "nemotron_personas_usa"):
            self.assertEqual(recs[name]["status"], "ok", recs[name]["problems"])
            rec = sources.require(self.root, name)
            self.assertEqual(rec["license"], sources.SOURCES[name]["license"])
            for v in rec["evidence"].values():
                self.assertTrue(v["licence_match"])

    def test_missing_licence_statement_refuses_the_source(self):
        recs = FX.make_sources(self.root, evidence={"geonames_cities": "All rights reserved."})
        self.assertEqual(recs["geonames_cities"]["status"], "refused")
        with self.assertRaises(sources.SourceRefused):
            sources.require(self.root, "geonames_cities")

    def test_pinned_hash_mismatch_is_a_problem(self):
        recs = FX.make_sources(self.root)
        p = os.path.join(self.root, "nemotron_personas_usa", FX.PARQUET)
        self.assertTrue(os.path.exists(p))
        rec = sources.fetch(self.root, "nemotron_personas_usa", get=lambda url, path: None)
        self.assertNotEqual(rec["status"], "ok")
        self.assertTrue(any("pinned" in p for p in rec["problems"]), rec["problems"])
        self.assertEqual(recs["nemotron_personas_usa"]["status"], "ok")

    def test_blocked_host_then_a_copy_placed_by_hand(self):
        FX.make_sources(self.root, skip=("ssa.gov/oact",))
        self.assertEqual(sources.read_record(self.root, "ssa_names")["status"], "incomplete")
        with self.assertRaises(sources.SourceRefused):
            sources.require(self.root, "ssa_names")
        with open(os.path.join(self.root, "ssa_names", "names.zip"), "wb") as f:
            f.write(FX.ssa_zip())
        rec = FX.make_sources(self.root, skip=("ssa.gov/oact",))["ssa_names"]
        self.assertEqual(rec["status"], "ok")
        self.assertTrue(rec["files"]["names.zip"]["retrieved"].startswith("by hand"))
        self.assertTrue(rec["notes"])

    def test_changed_file_or_edited_licence_refused(self):
        FX.make_sources(self.root)
        with open(os.path.join(self.root, "ldnoobw", "en"), "a") as f:
            f.write("extra\n")
        with self.assertRaises(sources.SourceRefused):
            sources.require(self.root, "ldnoobw")
        p = sources.record_path(self.root, "census_surnames")
        rec = json.load(open(p))
        rec["license"] = "cc-by-3.0"                             # an allowed licence, but not this source's
        json.dump(rec, open(p, "w"))
        with self.assertRaises(sources.SourceRefused):
            sources.require(self.root, "census_surnames")


class TestPools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        FX.make_sources(cls.tmp.name)
        cls.recs = {}
        for n in sources.SOURCES:
            cls.recs[n] = dict(sources.require(cls.tmp.name, n), _dir=os.path.join(cls.tmp.name, n))
        safety, _ = humanseed.build_safety(cls.recs["ldnoobw"], {"gate_hash": "x"})
        cls.ctx = human.Ctx(FX.wordlist(), human.unsafe_re(humanseed.safety_terms(safety)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_name_profiles_sex_and_era(self):
        names, totals, last = human.ssa_counts(os.path.join(self.recs["ssa_names"]["_dir"], "names.zip"))
        self.assertEqual(last, 2005)
        prof = human.name_profiles(names, totals)
        self.assertEqual([prof[n]["era"] for n in ("Agatha", "Corinne", "Elowen")], ["older", "middle", "younger"])
        self.assertEqual([prof[n]["sex"] for n in ("Agatha", "Bertram", "Jordan")], ["F", "M", "U"])
        self.assertEqual(prof["Jordan"]["share_f"], 0.5)
        self.assertEqual(prof["Jordan"]["births"], 1200)
        self.assertEqual((prof["Jordan"]["ssa_rank"], prof["Agatha"]["ssa_rank"]), (1, 2))

    def test_names_drops_and_provenance(self):
        items, st = human.build_names(self.recs["ssa_names"], self.ctx, target=60)
        t = by_text(items)
        self.assertEqual(t["May"]["drop"], "PROGRAM_COLLISION")
        self.assertEqual(t["Will"]["drop"], "WORD_COLLISION")
        self.assertEqual(t["Novak"]["drop"], "HELDOUT_POOL")      # an E004 alias surname
        self.assertEqual(t["Bimbo"]["drop"], "SAFETY_LIST")
        self.assertEqual(t["Jasper"]["status"], "kept")           # in the word list, but flagged proper
        self.assertEqual(t["Agatha"]["features"]["cell"], "older.F")
        src = t["Agatha"]["author"]["source"]
        sha = self.recs["ssa_names"]["files"]["names.zip"]["sha256"]
        self.assertEqual((src["license"], src["file_sha256"]), ("cc0", sha))
        self.assertEqual(t["Agatha"]["gates"]["gate_hash"], self.ctx.gate["gate_hash"])
        kept = [i for i in items if i["status"] == "kept"]
        self.assertEqual(len(kept), sum(sum(c["kept"].values()) for c in st["cells"].values()))
        self.assertEqual(len(kept), len(FX.SSA) - 1 - 4)          # Jordan counted once; 4 drops
        again, _ = human.build_names(self.recs["ssa_names"], self.ctx, target=60)
        self.assertEqual(items, again)

    def test_names_bands_and_quota(self):
        with mock.patch.object(human, "COMMON_DEPTH", 3), mock.patch.object(human, "RARE_MIN", 1):
            items, st = human.build_names(self.recs["ssa_names"], self.ctx, target=12)
        cell = [(i["text"], i["features"]["band"], i["drop"]) for i in items if i["features"]["cell"] == "older.F"]
        self.assertEqual(cell, [("Agatha", "common", None), ("Bimbo", "rare", "SAFETY_LIST")])
        self.assertEqual(st["cells"]["older.F"], {"quota": 2, "kept": {"common": 1, "rare": 0}, "ranked": 4,
                                                  "rare_candidates": 1})
        with mock.patch.object(human, "COMMON_DEPTH", 1), mock.patch.object(human, "RARE_MIN", 1):
            items, st = human.build_names(self.recs["ssa_names"], self.ctx, target=12)
        rare = [i["text"] for i in items if i["features"]["cell"] == "older.F" and i["features"]["band"] == "rare"
                and i["status"] == "kept"]
        self.assertEqual(rare, ["Gwendolyn"])
        kept_cells = collections.Counter(i["features"]["cell"] for i in items if i["status"] == "kept")
        self.assertTrue(all(n <= 2 for n in kept_cells.values()), kept_cells)

    def test_surnames_case_and_collisions(self):
        items, st = human.build_surnames(self.recs["census_surnames"], self.ctx, target=8)
        t = by_text(items)
        self.assertEqual(t["Smith"]["status"], "kept")
        self.assertEqual(t["Brown"]["drop"], "PROGRAM_COLLISION")
        self.assertEqual(t["Mcdonald"]["drop"], "CASE_AMBIG")
        self.assertEqual(t["Odonnell"]["drop"], "CASE_AMBIG")    # O + a listed surname
        self.assertEqual(t["Young"]["drop"], "WORD_COLLISION")
        self.assertEqual(t["Novak"]["drop"], "HELDOUT_POOL")
        self.assertEqual(t["Lambert"]["status"], "kept")          # LA + a remainder that is not a listed surname
        self.assertNotIn("All other names", t)
        self.assertEqual(st["kept"], {"common": 4})                # rows past 1,000: none in the fixture
        self.assertEqual(st["universe"], len(FX.CENSUS))

    def test_cities_feature_dup_form_country(self):
        items, st = human.build_cities(self.recs["geonames_cities"], self.ctx, target=10)
        drops = {(i["text"], i["features"]["geonameid"]): i["drop"] for i in items}
        self.assertIsNone(drops[("Vantorra", 1)])
        self.assertEqual(drops[("Kelmouth", 2)], "FEATURE")      # a section: does not claim the name
        self.assertEqual(drops[("Vantorra", 4)], "DUP_NAME")
        self.assertEqual(drops[("Saint-Ober", 5)], "FORM")
        self.assertEqual(drops[("Nice", 3)], "WORD_COLLISION")
        self.assertEqual(drops[("Orange", 6)], "PROGRAM_COLLISION")
        self.assertEqual(drops[("Brimsby", 8)], "SAFETY_LIST")
        kept = [i for i in items if i["status"] == "kept"]
        self.assertEqual([i["text"] for i in kept], ["Vantorra", "Kelmouth"])
        self.assertEqual(kept[0]["features"]["country"], "Aaland Test")
        self.assertEqual(kept[0]["features"]["same_name"], 2)
        few, _ = human.build_cities(self.recs["geonames_cities"], self.ctx, target=1)
        self.assertEqual([i["text"] for i in few if i["status"] == "kept"], ["Vantorra"])
        self.assertEqual(len(few), 1)


if __name__ == "__main__":
    unittest.main()
