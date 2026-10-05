"""Bank pass W2: persona seeds, the safety rubric, the build into one bank directory, its admit checks (aux banks,
pool and seed RECHECK) and the loader. Fixture sources only (tests/_human_fx.py); no network, no model."""
import json
import os
import tempfile
import unittest
from unittest import mock

import _human_fx as FX
import pools as P
from bankpass import admit, human, humanbuild, humanseed, load, sources, store


def codes(probs):
    return {c for ps in probs.values() for c, _ in ps}


class TestSeeds(unittest.TestCase):
    def test_persona_seeds(self):
        rec = {"name": "nemotron_personas_usa", "title": "t", "license": "cc-by-4.0", "revision": "r",
               "files": {FX.PARQUET: {"url": "u", "sha256": "s", "retrieved": "d"}}}
        items, st = humanseed.build_personas(rec, {"gate_hash": "g"}, target=10, candidates=FX.persona_rows())
        got = {i["features"]["uuid"]: (i["text"], i["drop"]) for i in items}
        self.assertEqual(got["u1"], ("someone who keeps a tidy garden and a calm head", None))
        self.assertEqual(got["u2"][1], "NAME_AGAIN")
        self.assertEqual(got["u3"][1], "HELDOUT_VOCAB")            # a digit
        self.assertEqual(items[2]["gates"]["hits"], [["HELDOUT_VOCAB", "digit"]])
        self.assertEqual(got["u4"][1], "MINOR")
        self.assertEqual(got["u5"][1], "DUP")
        self.assertEqual(got["u6"], ("someone who collects old postcards and fixes radios", None))
        self.assertEqual(got["u7"][1], "NO_LEAD")
        self.assertEqual(got["u8"], ("a calm cook who hums while he works", None))     # appositive kept as is
        self.assertEqual(got["u9"][1], "NO_LEAD")                                    # possessive lead
        self.assertEqual(st["kept"], 3)
        self.assertTrue(all("Ann" not in i["text"] and "Vega" not in i["text"] for i in items))
        self.assertEqual(items[0]["author"]["source"]["row"], f"{FX.PARQUET} row 0 uuid u1")
        two, _ = humanseed.build_personas(rec, {"gate_hash": "g"}, target=1, candidates=FX.persona_rows())
        self.assertEqual(len(two), 1)

    def test_clean_and_name(self):
        self.assertEqual(humanseed.clean("a calm\u2011minded \u2014 kind \u201cperson\u201d"), 'a calm-minded, kind "person"')
        row = {f: "Jo Ann Lee writes" for f in humanseed.FIELDS}
        self.assertEqual(humanseed.person_name(row), ["Jo", "Ann", "Lee"])
        row = dict({f: "Al Bo, a clerk" for f in humanseed.FIELDS}, persona="Al Bo is calm")
        self.assertEqual(humanseed.person_name(row), ["Al", "Bo"])           # "Bo," and "Bo" are one name word
        row = dict({f: "Ed Fry is here" for f in humanseed.FIELDS}, persona="Ed Fry de keeps")
        self.assertEqual(humanseed.person_name(row), ["Ed", "Fry"])          # no trailing particle
        row = dict({f: "Al Bo, a clerk" for f in humanseed.FIELDS}, persona="Al Bo, Texan by birth")
        self.assertEqual(humanseed.person_name(row), ["Al", "Bo,"])          # the comma ends the name
        row = dict({f: "A clerk, Cy Dunn" for f in humanseed.FIELDS}, persona="Cy Dunn keeps")
        self.assertEqual(humanseed.person_name(row), [])                     # no other field opens with it

    def test_safety_list(self):
        with tempfile.TemporaryDirectory() as d:
            FX.make_sources(d)
            rec = dict(sources.require(d, "ldnoobw"), _dir=os.path.join(d, "ldnoobw"))
            items, st = humanseed.build_safety(rec, {"gate_hash": "g"})
        self.assertEqual([i["text"] for i in items if i["status"] == "kept"], ["bimbo", "brimsby", "bad phrase"])
        self.assertEqual([i["drop"] for i in items if i["status"] == "dropped"], ["DUP"])
        rx = human.unsafe_re(humanseed.safety_terms(items))
        self.assertTrue(rx.search("a Bad Phrase here") and not rx.search("bimbos are fine") and not rx.search("xbimbo"))


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src, self.out = os.path.join(self.tmp.name, "src"), os.path.join(self.tmp.name, "out")
        patch = mock.patch.object(humanseed, "read_candidates", lambda rec, n: (FX.persona_rows(), 7))
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def build(self, evidence=None, **kw):
        if not os.path.exists(self.src):
            FX.make_sources(self.src, evidence=evidence)
        return humanbuild.build(self.src, self.out, "human_test", FX.wordlist(), **kw)

    def test_build_admits_and_loads(self):
        man, probs = self.build()
        self.assertTrue(admit.ok(probs), probs)
        self.assertEqual(sorted(man["banks"]), ["pool.city", "pool.name"])
        self.assertEqual(sorted(man["aux"]), ["pool.surname", "rubric.safety", "seed.persona"])
        self.assertEqual(man["missing"], {})
        self.assertEqual(man["sources"]["geonames_cities"]["license"], "cc-by-4.0")
        self.assertEqual(man["banks"]["pool.city"]["kept"], 2)
        self.assertEqual(man["banks"]["pool.city"]["stats"]["by_country"], {"Aaland Test": 2})
        self.assertEqual(man["aux"]["seed.persona"]["kept"], 3)
        self.assertNotIn("Vantorra", json.dumps(man))              # no item text in the manifest
        self.assertTrue(json.load(open(os.path.join(self.out, "admit.json")))["ok"])
        bs = load.from_dir(self.out)
        self.assertEqual(bs.pools["city"][0], ["Vantorra", "Kelmouth"])
        self.assertEqual(bs.pools["city"][1], "human:geonames_cities=100")    # the whole author mix, kind included
        self.assertEqual(bs.features["name"]["Agatha"]["era"], "older")
        self.assertNotIn("surname", bs.pools)
        self.assertEqual(sorted(bs.aux), ["pool.surname", "rubric.safety", "seed.persona"])
        self.assertEqual(bs.provenance, "FAKE")                     # every other bank is still FAKE
        undo = load.install(bs)
        try:
            self.assertEqual(P.pool("city").values, ["Vantorra", "Kelmouth"])
        finally:
            undo()

    def test_refused_source_is_listed_missing(self):
        man, probs = self.build(evidence={"census_surnames": "no statement"})
        self.assertIn("pool.surname", man["missing"])
        self.assertNotIn("pool.surname", man.get("aux", {}))
        self.assertTrue(admit.ok(probs), probs)

    def test_no_safety_list_no_pools(self):
        man, _ = self.build(evidence={"ldnoobw": "no statement"})
        self.assertEqual(sorted(man["missing"]), ["pool.city", "pool.name", "pool.surname", "rubric.safety"])

    def rewrite(self, bank, edit, part="banks"):
        """edit one bank file's records, then re-hash it into the manifest (so only the planted defect remains)."""
        path = os.path.join(self.out, bank + ".jsonl")
        recs = store.read_jsonl(path)
        edit(recs)
        sha = store.write_jsonl(path, recs)
        mp = os.path.join(self.out, "manifest.json")
        man = json.load(open(mp))
        man[part][bank]["sha256"] = sha
        json.dump(man, open(mp, "w"))
        return admit.check_dir(self.out)[1]

    def kept(self, recs):
        return next(r for r in recs if r["status"] == "kept")

    def test_planted_pool_value_fails_recheck(self):
        self.build()
        probs = self.rewrite("pool.city", lambda rs: self.kept(rs).update(text="Kelmouth Nine"))
        self.assertEqual(codes(probs), set())
        probs = self.rewrite("pool.city", lambda rs: self.kept(rs).update(text="Kel 9"))
        self.assertIn("RECHECK", codes(probs))

    def test_planted_seed_and_aux_defects(self):
        self.build()
        probs = self.rewrite("seed.persona", lambda rs: self.kept(rs).update(text="someone who naps at 3"), "aux")
        self.assertEqual(codes(probs), {"RECHECK"})
        self.build()
        probs = self.rewrite("rubric.safety", lambda rs: self.kept(rs).update(author={"kind": "computed"}), "aux")
        self.assertIn("AUTHOR_KIND", codes(probs))
        self.build()
        os.remove(os.path.join(self.out, "pool.surname.jsonl"))
        self.assertIn("BANK_FILE", codes(admit.check_dir(self.out)[1]))
        self.build()
        probs = self.rewrite("pool.surname", lambda rs: self.kept(rs)["author"]["source"].update(license="cc-by-nc"),
                             "aux")
        self.assertIn("AUTHOR", codes(probs))


if __name__ == "__main__":
    unittest.main()
