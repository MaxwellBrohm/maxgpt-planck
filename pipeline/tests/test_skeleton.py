"""Skeleton generator tests on a 5,000-skeleton sample (register RM) plus small RS and RL samples: schema validity,
determinism, event frequencies against the SPEC section 3 targets, no held-out E004 (or loaded RC-12) vocabulary,
golds derivable from the skeleton alone, and diversity statistics. Mutation tests live in test_skeleton_mut.py.
Run from pipeline/:  python3 -B -m unittest discover -s tests -v"""
import copy
import json
import re
import unittest
from collections import Counter

from _corpus import corpus, S
import golds
import heldout
import pools as P
import schema
import heldout_e004 as HE  # read-only, already on sys.path through heldout.py

TOL_KIND = 0.015
# targets copied from SPEC.txt sections 2-3 (not read from the generator, so a wrong generator constant fails here)
SPEC_KINDS = {"S1": 18, "S2": 12, "S3": 20, "S4": 10, "S5": 7, "S6": 13, "S7": 10, "S8": 5}
SPEC_NEVENTS = {2: 25, 3: 45, 4: 30}
SPEC_LOOKUP_SLICE = 0.20
SPEC_S3 = {"update": 40, "no_update_twin": 20, "two_slot": 15, "revert": 10, "assistant_err": 15}
SPEC_S7 = {"abstain": 50, "given": 50}
SPEC_P_EXACT = 0.7


def events(sk_list):
    return [e for sk in sk_list for e in sk["events"]]


def share(counter, key):
    return counter[key] / max(1, sum(counter.values()))


class TestSchemaAndDeterminism(unittest.TestCase):
    def test_schema_valid(self):
        bad = [(sk["seed"], schema.validate(sk)[:2]) for sk in corpus() if schema.validate(sk)]
        self.assertEqual(bad, [])
        for reg in ("RS", "RL"):
            self.assertEqual([sk["seed"] for sk in corpus(300, reg) if schema.validate(sk)], [])

    def test_deterministic_and_hashed(self):
        for s in (0, 7, 123, 4999):
            a, b = S.build(s), S.build(s)
            self.assertEqual(a, b)
            self.assertEqual(json.loads(json.dumps(a)), a)
            self.assertEqual(a["skel_id"], "sk-" + S.skel_hash(a)[:16])
        ids = [sk["skel_id"] for sk in corpus()]
        self.assertEqual(len(set(ids)), len(ids))

    def test_register_bounds_and_token_cap(self):
        for reg, lim in (("RS", (20, 25)), ("RM", (25, 30)), ("RL", (30, 40))):
            sample = corpus(300, reg) if reg != "RM" else corpus()
            for sk in sample:
                for t in sk["turns"]:
                    if t["mode"] == "guided":
                        cap = lim[0] if t["role"] == "user" else lim[1]
                        self.assertLessEqual(t["max_w"], cap)
                self.assertLessEqual(sk["estimate"]["tokens_max"], 1800)
        mids = [sk["estimate"]["tokens_mid"] for sk in corpus()]
        self.assertTrue(200 <= sum(mids) / len(mids) <= 400, sum(mids) / len(mids))


class TestFrequencies(unittest.TestCase):
    def test_kind_shares(self):
        c = Counter(e["kind"] for e in events(corpus()) if e["kind"] != "S9")
        tot = sum(SPEC_KINDS.values())
        for k, w in SPEC_KINDS.items():
            self.assertAlmostEqual(share(c, k), w / tot, delta=TOL_KIND, msg=k)

    def test_event_counts_and_slices(self):
        sks = corpus()
        n = Counter(len(sk["events"]) for sk in sks)
        tot = sum(SPEC_NEVENTS.values())
        for k, w in SPEC_NEVENTS.items():
            self.assertAlmostEqual(share(n, k), w / tot, delta=0.025, msg=f"{k} events")
        look = [sk for sk in sks if sk["slice"] == "lookup"]
        self.assertAlmostEqual(len(look) / len(sks), SPEC_LOOKUP_SLICE, delta=0.02)
        self.assertTrue(all(any(e["kind"] == "S9" for e in sk["events"]) for sk in look))
        self.assertFalse(any(e["kind"] == "S9" for sk in sks if sk["slice"] == "core" for e in sk["events"]))
        n9 = Counter(sum(e["kind"] == "S9" for e in sk["events"]) for sk in look)
        self.assertGreater(share(n9, 1), 0.7)

    def test_variant_shares(self):
        ev = events(corpus())
        import events_b, events_c
        for kind, table, key, tol in (("S3", SPEC_S3, "variant", 0.03),
                                      ("S6", events_b.S6_VARIANTS, "variant", 0.04),
                                      ("S7", SPEC_S7, "variant", 0.04),
                                      ("S9", events_c.S9_NEED, "need_kind", 0.04)):
            c = Counter(e["params"][key] for e in ev if e["kind"] == kind)
            tot = sum(table.values())
            for v, w in table.items():
                self.assertAlmostEqual(share(c, v), w / tot, delta=tol, msg=f"{kind} {v}")
        acts = Counter(e["params"]["act"] for e in ev if e["kind"] == "S8")
        self.assertEqual(set(acts), set(events_c.S8_ACTS))
        self.assertTrue(all(0.12 <= share(acts, a) <= 0.28 for a in acts), acts)

    def test_exact_share_distances_turns(self):
        sks = corpus()
        ev_turns = [t for sk in sks for t in sk["turns"] if t["role"] == "user" and t["events"]
                    and not re.fullmatch(r"digress[2-9]", t.get("role_in_event", ""))]
        exact = sum(t["mode"] == "exact" for t in ev_turns) / len(ev_turns)
        self.assertAlmostEqual(exact, SPEC_P_EXACT, delta=0.03)
        d1 = Counter(e["params"]["d"] for e in events(sks) if e["kind"] == "S1")
        # d=10 needs the plant in the first user turn of a 12-turn chat: about 1 skeleton in 4,700 (step 3 measured
        # 3 in 14,000), so a 5,000 sample covered it by luck. Pinned seed 9615 proves it is reachable; re-pin it
        # if the pools change. The rarity itself is a generator finding (notes.txt, step 3).
        self.assertTrue(set(range(1, 10)) <= set(d1) <= set(range(1, 11)), sorted(d1))
        self.assertIn(10, [e["params"]["d"] for e in S.build(9615, "RM")["events"] if e["kind"] == "S1"])
        ds = [e["params"]["d"] for e in events(sks) if "d" in e["params"]]
        self.assertTrue(all(0 <= d <= heldout.MAX_D for d in ds))
        nu = Counter(sum(t["role"] == "user" for t in sk["turns"]) for sk in sks)
        self.assertEqual(set(nu), set(range(4, 13)))


class TestHeldout(unittest.TestCase):
    """independent of heldout.py's regexes: the E004 lists are read directly here."""
    TERMS = [t.lower() for t in HE.all_heldout_objects() + HE.SPORT + HE.ALIAS_SURNAMES]
    MARKERS = [m.rstrip(",:").lower() for m in HE.EVAL_MARKERS]

    def _strings(self, obj, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.append(str(k))
                self._strings(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._strings(v, out)
        elif isinstance(obj, str):
            out.append(obj)
        return out

    RE = re.compile("|".join(r"(?<![a-z0-9])" + re.escape(t[:-1] if t.endswith("s") else t) + r"(?:s|es)?(?![a-z0-9])"
                             for t in TERMS + MARKERS))
    HON = re.compile("|".join(r"(?<![A-Za-z])" + re.escape(h[:-1]) + r"\b" for h in HE.HONORIFICS))

    def _hits(self, s):
        low = re.sub(r"[^a-z0-9' ]", " ", s.lower())
        return self.RE.findall(low) + self.HON.findall(s)

    def test_no_heldout_terms_anywhere(self):
        strings = set()
        for sk in corpus():
            strings.update(self._strings(sk, []))
        self.assertEqual(sorted(s for s in strings if self._hits(s))[:5], [])
        self.assertTrue(self._hits("my yard sale") and self._hits("Oops, Tuesday") and self._hits("Mr. Novak"))

    def test_no_digits_in_text(self):
        import gate
        bad = [(sk["seed"], w) for sk in corpus() for w, s in gate.text_fields(sk) if re.search(r"\d", s)]
        self.assertEqual(bad[:5], [])

    def test_pools_and_banks_clean(self):
        import banks as B
        for pool in P.POOLS.values():
            self.assertEqual([v for v in pool.values if self._hits(v)], [], pool.vtype)
        self.assertEqual([t for _, t in B.all_lines() if self._hits(t)], [])

    def test_rc12_pending_stamped(self):
        self.assertEqual(heldout.RC12_STATUS, "pending")
        self.assertTrue(all(sk["rc12"] == "pending" for sk in corpus()))
        self.assertEqual({sk["heldout_gate"] for sk in corpus()}, {heldout.gate_hash()})

    def test_structure_limits(self):
        for sk in corpus():
            per_type = Counter(s["type"] for s in sk["slots"].values() if s["counted"])
            self.assertLessEqual(max(per_type.values(), default=0), 2, sk["seed"])
            for e in sk["events"]:
                sets = Counter(op["slot"] for op in e["params"].get("ops", []) if op.get("op") in ("set", "fix"))
                self.assertLessEqual(max(sets.values(), default=0), 3)


class TestGolds(unittest.TestCase):
    def test_golds_derivable(self):
        bad = []
        for sk in corpus():
            d = golds.derive(copy.deepcopy(sk))
            bad += [(sk["seed"], e["id"]) for e in sk["events"] if d[e["id"]] != e["gold"]]
        self.assertEqual(bad[:5], [])

    def test_notes_match(self):
        for sk in corpus()[:500]:
            n = golds.notes(sk)
            self.assertEqual({t["i"]: t["gold_note"] for t in sk["turns"] if t["role"] == "assistant"}, n)


class TestDiversity(unittest.TestCase):
    def test_slot_value_coverage(self):
        vals = {}
        for sk in corpus():
            for s in sk["slots"].values():
                if not s["nonce"]:
                    vals.setdefault(s["type"], set()).add(s["value"])
        for vt, seen in vals.items():
            n = len(P.pool(vt).values)
            self.assertGreaterEqual(len(seen) / n, 0.9, f"{vt}: {len(seen)}/{n}")

    def test_nonce_share(self):
        sl = [s for sk in corpus() for s in sk["slots"].values() if s["type"] in P.NONCE_TYPES and s["counted"]]
        self.assertAlmostEqual(sum(s["nonce"] for s in sl) / len(sl), P.NONCE_SHARE, delta=0.03)

    def test_topics_personas_openings_styles(self):
        sks = corpus()
        topics = {t for sk in sks for t in sk["topic_path"]}
        self.assertEqual(len(topics), len(P.pool("topic").values))
        personas = {sk["user"]["persona_seed"] for sk in sks}
        self.assertGreater(len(personas) / S.N_PERSONAS, 0.85)
        import banks as B
        refs = {sk["opening"] for sk in sks}
        for bank in ("open.greet", "open.topic"):
            self.assertTrue(all(B.line_id(bank, i) in refs for i in range(len(B.lines(bank)))), bank)
        texts = Counter(sk["turns"][0]["text"] or sk["turns"][0]["intent"] + sk["topic_path"][0] for sk in sks)
        self.assertLess(max(texts.values()) / len(sks), 0.08, texts.most_common(3))  # FAKE bank: 10 lines
        styles = Counter(sk["user"]["style"] for sk in sks)
        for st, w in S.F.STYLES.items():
            self.assertAlmostEqual(share(styles, st), w, delta=0.03)
        self.assertEqual(Counter(sk["closing"] for sk in sks).keys(), set(S.CLOSING))

    def test_shard_without_replacement(self):
        sh = S.shard("t", 300)
        personas = [sk["user"]["persona_seed"] for sk in sh]
        self.assertEqual(len(set(personas)), 300)
        triples = {(tuple(sk["topic_path"]), sk["user"]["persona_seed"],
                    tuple(sorted(e["kind"] for e in sk["events"]))) for sk in sh}
        self.assertEqual(len(triples), 300)
        self.assertEqual(len({t for sk in sh for t in sk["topic_path"]}), len(P.pool("topic").values))


if __name__ == "__main__":
    unittest.main()
