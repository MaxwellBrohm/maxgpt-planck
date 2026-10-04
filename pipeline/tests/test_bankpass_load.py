"""Bank pass W0: per-bank admit checks and the loaders. A fixture bank directory (fixtures.make_bank_dir, marked
fixture: true) is admitted only with allow_fixture; every planted defect must refuse it with its code; install() must
put the fixture lines into the running skeleton code and undo() must restore the FAKE banks. No model is loaded;
every line is FAKE fixture text."""
import json
import os
import tempfile
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import banks as B
import banks_keys as BK
import pools as P
import skeleton
from bankpass import admit, fixtures as FX, load, store


def codes(probs):
    return {c for ps in probs.values() for c, _ in ps}


class TestAdmit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_fixture_needs_the_flag(self):
        FX.make_bank_dir(self.d)
        _, probs = admit.check_dir(self.d, allow_fixture=True)
        self.assertTrue(admit.ok(probs), probs)
        _, probs = admit.check_dir(self.d)
        self.assertEqual(codes(probs), {"MANIFEST"})
        with self.assertRaises(load.LoadRefused):
            load.from_dir(self.d)

    def planted(self, tamper=None, **kw):
        FX.make_bank_dir(self.d, tamper=tamper, **kw)
        return codes(admit.check_dir(self.d, allow_fixture=True)[1])

    def test_each_planted_defect_refuses(self):
        def edit(fn, bank="open.greet"):
            return {"records": lambda b, rs: fn(rs) if b == bank else rs}

        def first(fn):
            return edit(lambda rs: [fn(rs[0])] + rs[1:])

        def tamper_file(p):
            fp = os.path.join(p, "close.goodbye.jsonl")
            with open(fp, "a") as f:
                f.write("\n")
        cases = {
            "BANK_HASH": {"after": tamper_file},
            "GATE_STALE": first(lambda r: dict(r, gates=dict(r["gates"], gate_hash="0" * 16))),
            "GATE_HITS": first(lambda r: dict(r, gates=dict(r["gates"], hits=[["HELDOUT_ECHO", "x"]]))),
            "JUDGE": first(lambda r: dict(r, judges=r["judges"][:1])),
            "AUTHOR": first(lambda r: dict(r, author=dict(r["author"], revision="f" * 40))),
            "AUTHOR_KIND": first(lambda r: dict(r, author={"kind": "computed", "code_sha256": "a",
                                                           "input_sha256": "b"})),
            "RECHECK": first(lambda r: dict(r, text="Hello there \u2014 friend.")),
            "HOLES": edit(lambda rs: [dict(rs[0], text="Hi about {t} and {v}.")] + rs[1:], "open.topic"),
            "DUP": first(lambda r: dict(r, text="Hey there!")),
            "ITEM_SCHEMA": first(lambda r: dict(r, status="dropped", drop=None)),
            "EMPTY": edit(lambda rs: [dict(r, status="dropped", drop="X") for r in rs]),
        }
        for code, tamper in cases.items():
            with self.subTest(code=code):
                for f in os.listdir(self.d):
                    os.remove(os.path.join(self.d, f))
                banks = dict(FX.FIXTURE_LINES, **({"open.topic": ["Hi, so {t}.", "Hey, {t} again."]}
                                                  if code == "HOLES" else {}))
                self.assertIn(code, self.planted(tamper, banks=banks))

    def test_fake_items_and_status(self):
        self.assertIn("FAKE_PROVENANCE", self.planted(author_kind="fake", fixture=False))
        for f in os.listdir(self.d):
            os.remove(os.path.join(self.d, f))
        FX.make_bank_dir(self.d, status="dry")
        self.assertIn("MANIFEST", codes(admit.check_dir(self.d, allow_fixture=True)[1]))
        self.assertTrue(admit.ok(admit.check_dir(self.d, allow_fixture=True, allow_dry=True)[1]))

    def test_author_share(self):
        lines = [f"Hello number {w}." for w in "one two three four five six seven eight nine ten eleven twelve "
                 "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()] * 1
        lines += [f"Hey, friend {w}." for w in "one two three four five six seven eight nine ten eleven".split()]
        skew = {"records": lambda b, rs: FX.bank_records(b, lines, models=("qwen3.5-9b", "qwen3.5-9b",
                                                                           "gemma-4-12b"))}
        self.assertIn("AUTHOR_SHARE", self.planted(skew, banks={"open.greet": lines}))
        for f in os.listdir(self.d):
            os.remove(os.path.join(self.d, f))
        self.assertNotIn("AUTHOR_SHARE", self.planted(banks={"open.greet": lines}))
        for f in os.listdir(self.d):
            os.remove(os.path.join(self.d, f))
        q = "qwen3.5-9b"                                         # all three present, one at 60%
        skew3 = {"records": lambda b, rs: FX.bank_records(b, lines, models=(q, q, q, "gemma-4-12b", "ministral-3-8b"))}
        self.assertIn("AUTHOR_SHARE", self.planted(skew3, banks={"open.greet": lines}))


class TestLoad(unittest.TestCase):
    def test_fake_fallback(self):
        bs = load.load(None)
        self.assertEqual(bs.provenance, "FAKE")
        self.assertEqual(bs.banks, B.BANKS)
        self.assertIn("open.greet", bs.fake_banks)

    def test_install_and_undo(self):
        with tempfile.TemporaryDirectory() as d:
            FX.make_bank_dir(d)
            bs = load.from_dir(d, allow_fixture=True)
            self.assertNotIn("open.greet", bs.fake_banks)
            self.assertIn("open.topic", bs.fake_banks)              # not in the dir: FAKE, so still fail closed
            self.assertEqual(bs.provenance, "FAKE")
            fake_greet, fake_markers = list(B.BANKS["open.greet"]), list(B.MARKERS)
            fake_query = list(BK.KEYS["pet_name"]["query"])
            undo = load.install(bs)
            try:
                self.assertEqual(B.BANKS["open.greet"], sorted(FX.FIXTURE_LINES["open.greet"],
                                                                key=FX.FIXTURE_LINES["open.greet"].index))
                self.assertEqual(B.MARKERS, [m + " " for m in FX.FIXTURE_LINES["marker.fix"]])
                self.assertEqual(BK.KEYS["pet_name"]["query"], FX.FIXTURE_LINES["key.pet_name.query"])
                import events_base
                self.assertIs(events_base.KEYS, BK.KEYS)              # swapped in place, not rebound
                seen = set()
                for s in range(40):
                    sk = skeleton.build(s, "RM")
                    for t in sk["turns"]:
                        if (t.get("bank_ref") or "").startswith("open.greet."):
                            seen.add(t["text"])
                    self.assertTrue(sk["provenance"]["fake"])
                self.assertTrue(seen and seen <= set(FX.FIXTURE_LINES["open.greet"]))
            finally:
                undo()
            self.assertEqual((B.BANKS["open.greet"], B.MARKERS, BK.KEYS["pet_name"]["query"]),
                             (fake_greet, fake_markers, fake_query))
            self.assertEqual(B.PROVENANCE, "FAKE")

    def test_provenance_is_restored(self):
        bs = load.fake()
        bs.refs = {b: f"{b}@abc:g34m33q33" for b in bs.refs}
        self.assertEqual(bs.provenance, "BANKSET")
        undo = load.install(bs)
        try:
            self.assertEqual(B.PROVENANCE, "BANKSET")
        finally:
            undo()
        self.assertEqual(B.PROVENANCE, "FAKE")

    def test_unloadable_banks_refuse(self):
        for bank in ("intent", "pool.req_noun", "pool.weekday", "pool.topic", "label.no_such_key", "wordlist"):
            with self.subTest(bank=bank), tempfile.TemporaryDirectory() as d:
                FX.make_bank_dir(d, banks={bank: ["alpha beta", "gamma delta"], "open.greet": ["Hey there!"]})
                probs = admit.check_dir(d, allow_fixture=True)[1]
                self.assertIn(("UNLOADABLE", bank), probs[bank])
                self.assertEqual(probs["open.greet"], [])
                with self.assertRaises(load.LoadRefused):
                    load.from_dir(d, allow_fixture=True)

    def test_loader_refuses_a_bank_it_has_no_branch_for(self):
        from unittest import mock
        from bankpass import specs
        with tempfile.TemporaryDirectory() as d, mock.patch.object(specs, "loadable", return_value=True):
            FX.make_bank_dir(d, banks={"intent": ["ask about the weather"]})
            self.assertTrue(admit.ok(admit.check_dir(d, allow_fixture=True)[1]))
            with self.assertRaises(load.LoadRefused):
                load.from_dir(d, allow_fixture=True)

    def test_topic_bank_installs(self):
        topics = ["fixing a squeaky door hinge", "choosing paint for a small room", "a walk along the canal",
                  "sorting old photos into albums", "baking bread at home", "learning card tricks",
                  "repotting a big fern", "planning a picnic menu"]
        with tempfile.TemporaryDirectory() as d:
            FX.make_bank_dir(d, banks={"topic": topics})
            undo = load.install(load.from_dir(d, allow_fixture=True))
            try:
                self.assertEqual(P.pool("topic").values, topics)
                texts = set()
                for s in range(10):
                    sk = skeleton.build(s, "RM")
                    texts |= set(sk["topic_text"].values())
                    self.assertTrue(sk["provenance"]["fake"])             # other banks are still FAKE
                self.assertTrue(texts and texts <= set(topics))
            finally:
                undo()
        self.assertEqual(P.pool("topic").provenance, "FAKE")
        self.assertNotIn(topics[0], P.pool("topic").values)

    def test_pool_from_dir(self):
        with tempfile.TemporaryDirectory() as d:
            FX.make_bank_dir(d, banks={"pool.hobby": ["knitting", "whittling", "kite flying"]})
            man = json.load(open(os.path.join(d, "manifest.json")))
            bs = load.from_dir(d, allow_fixture=True)
            self.assertEqual(bs.pools["hobby"][0], ["knitting", "whittling", "kite flying"])
            self.assertFalse(bs.pools["hobby"][1].endswith("FAKE"))
            undo = load.install(bs)
            try:
                self.assertEqual(P.pool("hobby").values, ["knitting", "whittling", "kite flying"])
                self.assertEqual(P.pool("hobby").provenance, man["banks"]["pool.hobby"]["ref"].rsplit(":", 1)[1])
                self.assertTrue(P.pool("job").provenance == "FAKE")
            finally:
                undo()
            self.assertEqual(P.pool("hobby").provenance, "FAKE")


if __name__ == "__main__":
    unittest.main()
