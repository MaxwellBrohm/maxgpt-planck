"""Mutation tests (Max's rule: every check must be seen failing). Each test first shows the clean skeleton passing,
then plants one defect and asserts the gate, the gold derivation or the schema validator catches it."""
import copy
import hashlib
import os
import tempfile
import unittest

from _corpus import S
import gate
import golds
import heldout
import schema
import text_e004 as TX


def find(pred, limit=400):
    for s in range(limit):
        sk = S.build(s)
        hit = pred(sk)
        if hit:
            return sk, hit
    raise AssertionError("no skeleton matches the predicate")


def codes(sk):
    return {c for c, _ in gate.check(sk)}


def exact_user(sk):
    return next(t for t in sk["turns"] if t["role"] == "user" and t["mode"] == "exact")


def s1_exact(sk):
    by_i = {t["i"]: t for t in sk["turns"]}
    for e in sk["events"]:
        if e["kind"] == "S1" and all(by_i[e["turns"][r]]["mode"] == "exact" for r in ("plant", "query")):
            return e
    return None


class TestGateMutations(unittest.TestCase):
    def setUp(self):
        self.sk = S.build(3)
        self.assertEqual(gate.check(self.sk), [])

    def _planted(self, text, code="HELDOUT_VOCAB"):
        sk = copy.deepcopy(self.sk)
        t = exact_user(sk)
        t["text"] = t["text"] + " " + text
        self.assertIn(code, codes(sk), text)

    def test_vocab(self):
        for text in ("I also love tennis.", "Oops, I meant that.", "Ask Mr. Novak.", "The table size is fine.",
                     "Two yard sales today.", "Mrs Okafor called.", "On second thought, no.", "A balloon popped."):
            self._planted(text)

    def test_digit_anywhere(self):
        sk = copy.deepcopy(self.sk)
        sk["turns"][1]["must_include"].append("3")
        self.assertIn("HELDOUT_VOCAB", codes(sk))

    def test_echo_of_e004_text(self):
        import items_e004
        q = items_e004.draw("probe", ["H1"])["H1"][0]["question"]
        sk = copy.deepcopy(self.sk)
        exact_user(sk)["text"] = q
        self.assertIn("HELDOUT_ECHO", codes(sk))

    def test_dash(self):
        self._planted("so " + chr(0x2014) + " yes", "DASH")
        self._planted("so - yes", "DASH")

    def test_struct_distance_alias_types(self):
        sk, e = find(s1_exact)
        m = copy.deepcopy(sk)
        next(x for x in m["events"] if x["id"] == e["id"])["params"]["d"] = 11
        self.assertIn("HELDOUT_STRUCT", codes(m))
        m = copy.deepcopy(sk)
        m["events"][0]["params"]["alias"] = "the thing"
        self.assertIn("HELDOUT_STRUCT", codes(m))
        m = copy.deepcopy(sk)
        base = m["slots"][e["params"]["slot"]]
        for n in range(2):
            m["slots"][f"x{n}"] = dict(base, value=f"Zorvek{'ab'[n]}", counted=True)
        self.assertIn("HELDOUT_STRUCT", codes(m))

    def test_struct_four_corrections(self):
        sk, e = find(lambda k: next((x for x in k["events"]
                                     if x["kind"] == "S3" and x["params"]["variant"] == "update"), None))
        m = copy.deepcopy(sk)
        ev = next(x for x in m["events"] if x["id"] == e["id"])
        sets = [op for op in ev["params"]["ops"] if op["op"] == "set"]
        ev["params"]["ops"] += [dict(sets[0]) for _ in range(4 - len(sets))]
        self.assertIn("HELDOUT_STRUCT", codes(m))

    def test_struct_query_frame_echo(self):
        sk, e = find(s1_exact)
        m = copy.deepcopy(sk)
        by_i = {t["i"]: t for t in m["turns"]}
        plant = by_i[e["turns"]["plant"]]["text"]
        by_i[e["turns"]["query"]]["text"] = plant.rstrip(".") + ", right?"   # the query repeats the statement frame
        self.assertIn("HELDOUT_STRUCT", codes(m))

    def test_rc12_exports_are_used(self):
        sk = self.sk
        val = next(s["value"] for s in sk["slots"].values() if s["type"] != "assistant_name")
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "vocab.txt"), "w") as f:
            f.write(val + "\n")
        m = copy.deepcopy(sk)
        m13 = exact_user(m)
        m13["text"] = "we should walk the dog before dinner and then read a short book together tonight"
        w = TX.words(m13["text"])
        with open(os.path.join(d, "ngram13_sha256.txt"), "w") as f:
            f.write(hashlib.sha256(" ".join(w[:13]).encode()).hexdigest() + "\n")
        try:
            heldout.load_ext(d)
            self.assertEqual(heldout.RC12_STATUS, "loaded")
            self.assertIn("HELDOUT_VOCAB", codes(sk))
            self.assertTrue(any("rc12_13gram" in x for _, x in gate.check(m)))
        finally:
            heldout.load_ext()
        self.assertEqual(heldout.RC12_STATUS, "pending")
        self.assertEqual(gate.check(sk), [])


class TestGoldMutations(unittest.TestCase):
    def setUp(self):
        self.sk, self.e = find(s1_exact)
        d = golds.derive(self.sk)
        self.assertEqual(d[self.e["id"]], self.e["gold"])

    def _ev(self, sk):
        return next(x for x in sk["events"] if x["id"] == self.e["id"])

    def test_wrong_stored_gold_is_seen(self):
        m = copy.deepcopy(self.sk)
        self._ev(m)["gold"]["answer"] = "Zorvek"
        self.assertNotEqual(golds.derive(m)[self.e["id"]], self._ev(m)["gold"])

    def test_op_value_not_stated(self):
        m = copy.deepcopy(self.sk)
        self._ev(m)["params"]["ops"][0]["value"] = "Zorvek"
        self.assertRaises(golds.GoldError, golds.derive, m)

    def test_leak_between_plant_and_query(self):
        m = copy.deepcopy(self.sk)
        ev = self._ev(m)
        between = [t for t in m["turns"] if ev["turns"]["plant"] < t["i"] < ev["turns"]["query"]
                   and ev["id"] not in t["events"] and t["mode"] == "guided"]
        self.assertTrue(between)
        between[0]["must_exclude"].remove(ev["gold"]["answer"])
        self.assertRaises(golds.GoldError, golds.derive, m)
        m = copy.deepcopy(self.sk)
        ev = self._ev(m)
        t = next(t for t in m["turns"] if ev["turns"]["plant"] < t["i"] < ev["turns"]["query"] and t["role"] == "user"
                 and ev["id"] not in t["events"])
        t.update(mode="exact", text="By the way, " + ev["gold"]["answer"] + " came up.")
        self.assertRaises(golds.GoldError, golds.derive, m)

    def test_query_restates_answer(self):
        m = copy.deepcopy(self.sk)
        ev = self._ev(m)
        q = next(t for t in m["turns"] if t["i"] == ev["turns"]["query"])
        q["text"] = q["text"].rstrip("?") + ", is it " + ev["gold"]["answer"] + "?"
        self.assertRaises(golds.GoldError, golds.derive, m)

    def test_list_op_change(self):
        sk, e = find(lambda k: next((x for x in k["events"] if x["kind"] == "S2" and x["params"]["ops"][1:]
                                     and x["params"]["query"] in ("count", "last", "first")), None))
        self.assertEqual(golds.derive(sk)[e["id"]], e["gold"])
        m = copy.deepcopy(sk)
        ev = next(x for x in m["events"] if x["id"] == e["id"])
        op = ev["params"]["ops"][1]
        op["op"] = "add" if op["op"] != "add" else "remove"
        try:
            self.assertNotEqual(golds.derive(m)[e["id"]], ev["gold"])
        except (ValueError, golds.GoldError):
            pass


class TestSchemaMutations(unittest.TestCase):
    def test_schema_catches(self):
        sk = S.build(5)
        self.assertEqual(schema.validate(sk), [])
        muts = [lambda m: m.pop("turns"), lambda m: m["turns"][0].update(role="assistant"),
                lambda m: m["turns"][0].update(mask=1), lambda m: m["events"][0]["turns"].update(x=999),
                lambda m: m.update(slice="core") or m["turns"].insert(2, dict(m["turns"][1], role="tool")),
                lambda m: m["turns"][1]["must_exclude"].extend(m["turns"][1]["must_include"] or ["x"])
                or m["turns"][1]["must_include"].append("x"),
                lambda m: m["events"][0]["params"].update(slot="s999"), lambda m: m.update(register="XL")]
        for n, f in enumerate(muts):
            m = copy.deepcopy(sk)
            f(m)
            self.assertTrue(schema.validate(m), f"mutation {n} not caught")

    def test_fake_provenance_stamped(self):
        for s in range(50):
            sk = S.build(s)
            self.assertTrue(sk["provenance"]["fake"])
            self.assertTrue(all(b.endswith(":FAKE") for b in sk["provenance"]["banks"]))
            self.assertTrue(any(p.endswith(":FAKE") for p in sk["provenance"]["pools"]))


if __name__ == "__main__":
    unittest.main()
