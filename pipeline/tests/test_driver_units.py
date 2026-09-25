"""Step 4 unit tests: dedup (exact, near, index rebuild), shards (torn lines, rotation), resumable iter_shard, and
stats.py (distinct n-grams, histograms, slot coverage, cap alarms). No model, no server."""
import json
import os
import random
import shutil
import tempfile
import unittest

import driver_fixtures as DF  # noqa: F401  (puts the pipeline on sys.path)
import dedup
import fake_teacher
import shards
import skeleton
import stats


def conv(skel):
    tx = fake_teacher.render(skel)
    return [{"role": t["role"], "text": tx[t["i"]]} for t in skel["turns"]]


class TestDedup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skels = skeleton.shard("dedup", 60)

    def test_exact_near_and_distinct(self):
        idx = dedup.Index()
        for s in self.skels:
            hit, keys = idx.check(conv(s), s["slots"])
            self.assertIsNone(hit, "two different fake chats flagged as duplicates")
            idx.add(s["skel_id"], keys)
        a = self.skels[5]
        hit, _ = idx.check(conv(a), a["slots"])
        self.assertEqual(hit[:2], ("DUP_EXACT", a["skel_id"]))
        near = next(filter(None, (DF.swap_value(s) for s in self.skels[5:])))
        orig = near["skel_id"].rsplit("-near", 1)[0]
        hit, _ = idx.check(conv(near), near["slots"])
        self.assertEqual(hit[:2], ("DUP_NEAR", orig))

    def test_near_threshold_sides(self):
        rng = random.Random(1)
        vocab = [f"w{i}" for i in range(5000)]
        base = [rng.choice(vocab) for _ in range(400)]

        def turns(words):
            return [{"role": "assistant", "text": " ".join(words)}]
        idx = dedup.Index()
        _, keys = idx.check(turns(base), {})
        idx.add("base", keys)
        close = list(base)
        for k in range(0, 400, 80):          # 5 changed words: Jaccard of 5-grams about 0.88
            close[k] = "zz" + str(k)
        far = list(base)
        for k in range(0, 400, 12):          # 34 changed words: Jaccard about 0.4
            far[k] = "yy" + str(k)
        sh = dedup.shingles(turns(base), [])
        self.assertGreater(dedup.jaccard(sh, dedup.shingles(turns(close), [])), 0.8)
        self.assertLess(dedup.jaccard(sh, dedup.shingles(turns(far), [])), 0.5)
        self.assertEqual(idx.check(turns(close), {})[0][0], "DUP_NEAR")
        self.assertIsNone(idx.check(turns(far), {})[0])

    def test_rebuild_matches(self):
        a, b = dedup.Index(), dedup.Index()
        for s in self.skels[:20]:
            _, keys = a.check(conv(s), s["slots"])
            a.add(s["skel_id"], keys)
            b.add(s["skel_id"], b.keys(conv(s), s["slots"]))
        for s in self.skels[:25]:
            self.assertEqual(a.check(conv(s), s["slots"])[0], b.check(conv(s), s["slots"])[0])

    def test_mask_uses_type_tags(self):
        m = dedup.mask("I live in Brno now, Brno is nice", [("Brno", "city")])
        self.assertNotIn("Brno", m)
        self.assertEqual(m.count("<city>"), 2)


class TestShards(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)

    def test_rotation_resume_and_torn_line(self):
        w = shards.Writer(self.d, "x", size=3, fsync=False)
        for k in range(7):
            w.write({"k": k})
        w.close()
        self.assertEqual(len(shards.paths(self.d, "x")), 3)
        last = shards.paths(self.d, "x")[-1]
        with open(last, "a") as f:
            f.write('{"k": 7, "torn')
        w = shards.Writer(self.d, "x", size=3, fsync=False)
        self.assertEqual(w.dropped, len('{"k": 7, "torn'))
        for k in range(7, 10):
            w.write({"k": k})
        w.close()
        self.assertEqual([r["k"] for r in shards.read(self.d, "x")], list(range(10)))
        self.assertEqual([shards._lines(p) for p in shards.paths(self.d, "x")], [3, 3, 3, 1])

    def test_repair_whole_file_torn(self):
        p = os.path.join(self.d, "x-00000.jsonl")
        with open(p, "w") as f:
            f.write('{"a"')
        self.assertEqual(shards.repair(p), 4)
        self.assertEqual(os.path.getsize(p), 0)

    def test_unparsable_middle_line_raises(self):
        p = os.path.join(self.d, "x-00000.jsonl")
        with open(p, "w") as f:
            f.write('{"a": 1}\nnot json\n{"a": 2}\n')
        with self.assertRaises(ValueError):
            list(shards.read(self.d, "x"))


class TestIterShard(unittest.TestCase):
    def test_resume_continues_exactly(self):
        def take(it, n):
            return [x for _, x in zip(range(n), it)]
        full = [s["skel_id"] for _, s in take(skeleton.iter_shard("it"), 40)]
        first = take(skeleton.iter_shard("it"), 25)
        j_last = first[-1][0]
        seen = {skeleton.triple(s) for _, s in first}
        rest = [s["skel_id"] for _, s in take(skeleton.iter_shard("it", start_j=j_last, seen=seen), 15)]
        self.assertEqual(rest, full[25:])
        self.assertEqual([s["skel_id"] for s in skeleton.shard("it", 40)], full)


def rec(turns, slots=None, events=(), style="terse"):
    return {"turns": [{"role": r, "text": t, "author": a} for r, t, a in turns], "slots": slots or {},
            "events": list(events), "est_tokens": 10, "register": "RM", "slice": "core", "topic_path": ["t1"],
            "user": {"style": style}}


class TestStats(unittest.TestCase):
    def test_distinct_and_hist(self):
        self.assertEqual(stats.distinct([["a", "a", "b"]], 1), round(2 / 3, 4))
        toks = [["a", "b", "c", "a", "b", "d"]]
        self.assertEqual(stats.distinct(toks, 2), 0.8)             # ab bc ca ab bd: 4 of 5
        self.assertEqual(stats.distinct(toks, 3), 1.0)             # abc bca cab abd: 4 of 4
        self.assertEqual(stats.distinct([["a", "b"]], 3), None)    # no trigram at all
        h = stats.hist([1, 3, 4, 7, 8, 30], [1, 4, 8, 100])
        self.assertEqual([c for _, _, c in h], [2, 2, 2])

    def test_compute_on_records(self):
        slots = {"s1": {"type": "city", "value": "Brno", "nonce": False}}
        ev = [{"kind": "S1", "params": {"d": 4}}]
        recs = [rec([("user", "Hi, I moved to Brno last week.", "bank:x"),
                     ("assistant", f"That sounds like a big change number {k}.", "teacher:m")], slots, ev)
                for k in ("one", "two", "three")]
        st = stats.compute(recs)
        self.assertEqual(st["conversations"], 3)
        self.assertEqual(st["slots"]["city"]["slots"], 3)
        self.assertEqual(st["slots"]["city"]["distinct"], 1)
        self.assertEqual(st["coverage"]["cells"], {"S1|-|d3-5": 3})
        self.assertEqual(sum(c for _, _, c in st["lengths"]["words_per_user_turn"]), 3)
        self.assertEqual(st["repetition"]["opening4"]["top"][0][:2], ["hi i moved to", 3])
        self.assertLess(st["text"]["assistant"]["distinct_1"], 1.0)
        json.dumps(st)

    def test_cap_alarm_needs_min_n_and_fires(self):
        mk = (lambda k, shared: rec([("user", f"question number {k} about item {k}", "teacher:m"),
                                     ("assistant", (shared if shared else
                                                    f"unique reply number {k} has these several words here") +
                                      f" end {k}", "teacher:m")]))
        shared = "that is a very good plan for the whole weekend ahead"
        n = stats.MIN_N + 200                          # 1,200: one conversation is under 0.1%
        recs = [mk(k, shared if k < 2 else None) for k in range(n)]
        rep = stats.compute(recs)["repetition"]["ngram8"]
        self.assertTrue(rep["alarm"], rep["max_share"])
        recs = [mk(k, None) for k in range(n)]
        rep = stats.compute(recs)["repetition"]["ngram8"]
        self.assertEqual(rep["max_share"], round(1 / n, 5), "the fixture must produce 8-grams")
        self.assertFalse(rep["alarm"])
        recs = [mk(k, shared if k < 2 else None) for k in range(stats.MIN_N - 1)]
        rep = stats.compute(recs)["repetition"]["ngram8"]
        self.assertGreater(rep["max_share"], rep["cap"])
        self.assertFalse(rep["alarm"], "no alarm below MIN_N conversations")


if __name__ == "__main__":
    unittest.main()
