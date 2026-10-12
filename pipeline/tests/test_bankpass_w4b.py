"""Bank pass W4, the freeze end to end: a mini stage P run by the FAKE teacher (tests/_w3fake.py, as the W3 tests do),
a fixture human set (tests/_human_fx.py), then w4freeze.collect / write_all / freeze, admit and the loader. Plus the
skeleton-side wiring the frozen set drives (per-bank p_exact, attribute predicates, required words, counted
articles, provenance from refs). No model, no network; every line is FAKE test text."""
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import _human_fx as FX
import test_bankpass_w3 as W3T
import assemble as A
import banks as B
import banks_keys as BK
import fake_data as F
import pools as P
import skeleton as S
import topic_words as TW
from bankpass import (admit, humanbuild, humanseed, load, plan, store, w3state as WS, w4banks as WB, w4freeze as W4F,
                      w4gates as W4G, wordload)

FW = ("amber", "brisk", "cedar", "dune", "ember", "fable", "grove", "harbor", "inlet", "juniper", "kettle", "lantern")


def w4_wordlist():
    fams = {}
    for i, (w, pos) in enumerate([(w, "noun") for w in FW] + [("walk", "verb"), ("fold", "verb"), ("carry", "verb"),
                                                               ("warm", "adj"), ("quiet", "adj"), ("calmly", "adv"),
                                                               ("bright", "adj"), ("the", "func")], 1):
        fams[w] = {"pos": pos, "pos_status": "confirmed", "rank": i, "lists": {"RS", "RM", "RL"}, "forms": {},
                   "flags": set(), "required_ok": pos in ("noun", "verb", "adj"), "a": 3, "an": 0, "mass": 0, "tf": 9}
    return wordload.WordList(fams, "computed", "wordlist@test:computed", {"file_sha256": "t" * 64})


class TestFreezeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root, src, hum = (os.path.join(cls.tmp.name, x) for x in ("stage", "src", "human"))
        p, wl0 = W3T.mini_plan()
        for t in plan.TEACHER_ORDER + plan.TEACHER_ORDER[:2]:
            W3T.run_stage(root, t, p, wl0)
            WS.refresh(root)
        with open(os.path.join(root, "plan.json"), "w") as f:
            json.dump(p, f)
        with mock.patch.object(humanseed, "read_candidates", lambda rec, n: (FX.persona_rows(), 7)):
            FX.make_sources(src)
            humanbuild.build(src, hum, "human_test", FX.wordlist())
        cls.wl = w4_wordlist()
        cls.g = W4G.Gates(W4G.Ngrams(["one two three four five six seven eight"], 8), None,
                          W4G.Ngrams(["hello there a amber kind of day"], 6), re.compile(r"\bjuniper\b"))
        cls.banks, cls.aux, cls.facts = W4F.collect(root, wl0, cls.wl, cls.g)
        cls.out = os.path.join(cls.tmp.name, "banks_v1")
        author = {"kind": "computed", "code_sha256": "c" * 64, "input_sha256": "t" * 64}
        computed = {"pool.avoid_word": WB.avoid_bank(cls.wl, author, cls.g)}
        computed.update({f"pool.req_{p}": WB.req_bank(cls.wl, p, author, cls.g) for p in ("noun", "verb", "adj")})
        meta = W4F.write_all(cls.out, cls.banks, cls.aux, hum, computed)
        cls.excluded = W4F.freeze(cls.out, meta, {"test": True})
        with open(os.path.join(cls.out, "manifest.json")) as f:
            cls.man = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_admitted_and_failures_listed(self):
        _, probs = admit.check_dir(self.out)
        self.assertTrue(admit.ok(probs), probs)
        self.assertEqual(self.man["excluded"], self.excluded)
        for b, v in self.excluded.items():
            self.assertTrue(v["codes"], b)
            self.assertNotIn(b, self.man["banks"])
        self.assertIn("pool.city", self.man["banks"])
        self.assertIn("seed.persona", self.man["aux"])
        self.assertEqual(self.facts["selected_without_wordset_calls"], 0)

    def test_every_record_kept(self):
        """s2h: every stage item of a bank stays a record, kept or dropped with a code."""
        for b, (rs, _) in self.banks.items():
            for r in rs:
                self.assertIn(r["status"], ("kept", "dropped"), b)
                self.assertTrue(r["status"] == "kept" or r["drop"], (b, r["id"]))

    def test_w4_gates_drop_and_mark(self):
        drops = {c for m in list(self.man["banks"].values()) + list(self.man.get("aux", {}).values())
                 for c in m["drops"]}
        drops |= {r["drop"] for rs, _ in self.banks.values() for r in rs}
        self.assertIn("MINED_COPY", drops)
        self.assertIn("SAFETY_LIST", drops)
        for b, (rs, _) in self.banks.items():
            for r in rs:
                if r["status"] == "kept" and r["author"]["kind"] == "teacher":
                    self.assertEqual(r["gates"].get("w4"), [], (b, r["id"]))

    def test_thirds_p_exact_and_no_text(self):
        for b, m in self.man["banks"].items():
            if m["kind"] == "teacher" and m["kept"] >= 30:
                self.assertTrue(all(abs(s - 1 / 3) <= 0.05 for s in m["authors"].values()), (b, m["authors"]))
            if b in W3T.BANKS:
                self.assertEqual(m["p_exact"], round(min(0.7, 0.7 * m["kept"] / W3T.BANKS[b]), 4), b)
        blob = json.dumps(self.man)
        self.assertNotIn("kind of day", blob)
        self.assertNotIn("Vantorra", blob)
        self.assertNotIn("amber", blob)

    def test_load_install_and_build(self):
        bs = load.from_dir(self.out)
        self.assertTrue(bs.p_exact)
        self.assertEqual(bs.pools["city"][1], "human:geonames_cities=100")
        fake = load.BankSet()
        for vt in ("city", "name"):            # the fixture human set holds two names and two cities: too few to build
            bs.pools[vt] = fake.pools[vt]
        fake_lists = dict(B.LIST_NAMES)
        undo = load.install(bs)
        try:
            for b, pe in bs.p_exact.items():
                self.assertEqual(B.P_EXACT_BANK[b], pe)
            self.assertEqual(B.LIST_REFS, {**{vt: f"listname.{vt}@fake:FAKE" for vt in B.LIST_NAMES},
                                           **{b.split(".", 1)[1]: m["ref"] for b, m in self.man["banks"].items()
                                              if b.startswith("listname.")}})
            self.assertEqual(len(TW.REQ_WORDS), self.man["banks"]["topic"]["kept"])      # every kept topic its sets
            lists = [b.split(".", 1)[1] for b in self.man["banks"] if b.startswith("listname.")]
            self.assertTrue(lists)
            for lt in lists:
                self.assertEqual(B.LIST_NAMES[lt], bs.list_names[lt])
                self.assertNotEqual(B.LIST_NAMES[lt], fake_lists[lt])
            sk = [S.build(s) for s in range(30)]
            refs = {r for k in sk for r in k["provenance"]["pools"]}
            self.assertIn(f"avoid_word@{P.pool('avoid_word').sha}:computed=100", refs)
            for part in ("noun", "verb", "adj"):
                self.assertIn(f"req_{part}@{P.pool('req_' + part).sha}:computed=100", refs)
            lists = {r for k in sk for r in k["provenance"]["lists"]}
            self.assertEqual(lists, set(B.LIST_REFS.values()))
            self.assertTrue(any(not r.endswith(":FAKE") for r in lists))
            self.assertTrue(all(k["provenance"]["fake"] for k in sk))           # personas and other banks: FAKE
        finally:
            undo()
        self.assertEqual(B.P_EXACT_BANK, {})
        self.assertEqual(B.LIST_REFS, {vt: f"listname.{vt}@fake:FAKE" for vt in B.LIST_NAMES})
        self.assertEqual(TW.REQ_WORDS, {})


class TestSkeletonWiring(unittest.TestCase):
    def test_p_exact_per_bank(self):
        from events_base import Ctx
        import random
        opts = [("open.greet.0", "Hello there.")]
        for pe, mode in ((0.0, "guided"), (1.0, "exact")):
            with mock.patch.dict(B.P_EXACT_BANK, {"open.greet": pe}):
                ctx = Ctx(random.Random(1), 4, 0.7, "RM", False, "Ada")
                self.assertEqual(ctx.user(0, "e1", "x", opts, {}, "greet", [])["mode"], mode)
        self.assertEqual(B.p_exact("key.job.plant.3", 0.7), 0.7)
        with mock.patch.dict(B.P_EXACT_BANK, {"key.job.plant": 0.1}):
            self.assertEqual(B.p_exact("key.job.plant.3", 0.7), 0.1)
            self.assertEqual(B.p_exact("key.job.corr.3", 0.7), 0.7)

    def test_lookup_pred_and_req_words_and_features(self):
        self.assertEqual(B.lookup_pred("opening day", "weekday"), B.LOOKUP_PRED["weekday"])
        with mock.patch.dict(B.LOOKUP_PRED_ATTR, {"opening day": "first opens on {v}"}):
            self.assertEqual(B.lookup_pred("opening day", "weekday"), "first opens on {v}")
        t = next(iter(TW.TOPIC_WORDS))
        self.assertEqual(TW.req_words(t, "noun"), TW.words(t, "noun"))
        held = next(w for w in TW.heldout.HELDOUT_PHRASES if " " not in w)
        with mock.patch.dict(TW.REQ_WORDS, {t: {"noun": ["cedar", held], "verb": [], "adj": []}}):
            self.assertEqual(TW.req_words(t, "noun"), ("cedar",))                 # held-out words never
            self.assertEqual(TW.req_words(t, "verb"), ())
        self.assertEqual(P.features("job", "engineer")["article"], "an")
        with mock.patch.dict(P.VALUE_FEATURES, {"job": {"engineer": {"article": "a", "number": "sg"}}}):
            self.assertEqual(P.features("job", "engineer")["article"], "a")
            self.assertEqual(P.features("name", "Engineer")["article"], "")      # a capitalized value: no article

    def test_fake_refs(self):
        prov = {"banks": ["banks@x:BANKSET"], "pools": ["city@x:human:geonames_cities=100", "job@y:g33m33q33"],
                "lists": ["listname.city@z:g100"], "personas": "nemotron@p", "topic_words": "topicwords@t:g34m33q33"}
        self.assertEqual(A.fake_refs(prov), [])
        for k, v in (("banks", ["banks@x:FAKE"]), ("pools", ["job@y:FAKE"]), ("lists", ["listname.city@fake:FAKE"]),
                     ("personas", "FAKE"), ("topic_words", "FAKE")):
            self.assertEqual(len(A.fake_refs(dict(prov, **{k: v}))), 1, k)
        sk = S.build(3)
        self.assertTrue(sk["provenance"]["fake"])
        self.assertIn("lists", sk["provenance"])

    def test_skeletons_use_attr_preds_and_req_words(self):
        preds = {a: "zorbles on {v}" for a in F.ATTR_TYPES}
        req = {t: {"noun": ["quokka"], "verb": ["yodel"], "adj": ["snazzy"]} for t in TW.TOPIC_WORDS}
        with mock.patch.dict(B.LOOKUP_PRED_ATTR, preds), mock.patch.dict(TW.REQ_WORDS, req), \
                mock.patch.dict(B.P_EXACT_BANK, {"lookup.ctx": 1.0, "lookup.cf": 1.0}):
            sks = [S.build(s) for s in range(150)]
        texts = [t["text"] for k in sks for t in k["turns"] if t["role"] == "user" and t["mode"] == "exact"
                 and (t.get("bank_ref") or "").startswith(("lookup.ctx", "lookup.cf"))]
        self.assertTrue(texts)
        self.assertTrue(all("zorbles on" in x for x in texts))
        placed = [k["required_words"][p] for k in sks for p in ("noun", "verb", "adj")[:len(k["required_words"]
                                                                                            ["turn_hint"])]]
        self.assertTrue(placed)
        self.assertTrue(all(w in ("quokka", "yodel", "snazzy") for w in placed))

    def test_provenance_fake_follows_refs(self):
        with mock.patch.object(A, "fake_refs", lambda prov: []):
            self.assertFalse(S.build(5)["provenance"]["fake"])
        self.assertIn("label.job", load.BankSet().fake_banks)                  # a FAKE key label counts

    def test_entity_and_relation_install(self):
        bs = load.BankSet()
        bs.lookup = ({"bakery": ["opening day"]}, {"opening day": "weekday"}, {"opening day": "opens its doors on {v}"})
        bs.sex = ({"aunt"}, {"uncle"})
        undo = load.install(bs)
        try:
            self.assertEqual(F.ENTITY_KINDS, {"bakery": ["opening day"]})
            self.assertEqual(B.LOOKUP_PRED_ATTR, {"opening day": "opens its doors on {v}"})
            self.assertEqual((BK.FEMALE, BK.MALE), ({"aunt"}, {"uncle"}))
        finally:
            undo()
        self.assertIn("museum", F.ENTITY_KINDS)
        self.assertIn("sister", BK.FEMALE)
        self.assertEqual(B.LOOKUP_PRED_ATTR, {})


if __name__ == "__main__":
    unittest.main()
