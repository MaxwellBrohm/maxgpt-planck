"""Bank pass W1 code on fixture docs (no core v0 on the Mac): the word-list counter (features, samples), the count
table, families / POS / drops / lists, and the human-example miner. Every doc here is FAKE fixture text."""
import collections
import gzip
import json
import os
import tempfile
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import heldout
from bankpass import mine, wordfam as WF, wordlist as WL, wordstats as WS, wordtable as WT


def acc():
    return {f: collections.Counter() for f in WL.FEATS}


class TestCounter(unittest.TestCase):
    def test_features_of_one_doc(self):
        a = acc()
        WL.count_doc("I want to walk the dog. It was very big. It was big. An hour, a unit, café Paris. We saw Paris and "
                     "many apples, much rice. It\u2019s fine, na\u00efve. See the General Public License.", a)
        self.assertEqual(a["verb"]["walk"], 1)
        self.assertEqual(a["noun"]["dog"], 1)
        self.assertEqual(a["adj"]["big"], 2)              # "very big" and "was big."
        self.assertEqual((a["an"]["hour"], a["a"]["unit"]), (1, 1))
        self.assertEqual((a["pl"]["apples"], a["mass"]["rice"]), (1, 1))
        self.assertEqual((a["mid"]["paris"], a["capmid"]["paris"]), (1, 1))   # "saw Paris" mid; "café Paris" not
        self.assertNotIn("caf", a["tf"])                  # no ASCII fragment of an accented word
        self.assertNotIn("ve", a["tf"])
        self.assertEqual((a["mid"]["dog"], a["capmid"]["dog"]), (1, 0))
        self.assertEqual((a["mid"]["general"], a["mid"]["public"], a["mid"]["license"]), (1, 0, 0))  # Title Case
        self.assertEqual(a["tf"]["it's"], 1)              # curly apostrophe straightened, contraction kept
        self.assertEqual(a["df"]["paris"], 1)

    def test_samples_disjoint_and_seeded(self):
        labs = [WL.sample_of("x/s-0.jsonl.zst", i, ("A", "B")) for i in range(4000)]
        self.assertEqual(labs, [WL.sample_of("x/s-0.jsonl.zst", i, ("A", "B")) for i in range(4000)])
        a, b = labs.count("A"), labs.count("B")
        self.assertTrue(150 < a < 250 and 150 < b < 250, (a, b))
        self.assertEqual(WL.sample_of("x", 3, ("full",)), "full")


def write_table(path, rows, docs):
    head = {"docs": docs, "columns": ["word"] + [f"df_{g}" for g in WT.GROUPS] + ["tf"] + list(WT.CTX_FEATS),
            "input_manifest_sha256": "m", "code_sha256": "c"}
    with gzip.open(path, "wt") as f:
        f.write("#" + json.dumps(head) + "\n")
        for w, r in rows.items():
            f.write("\t".join(map(str, [w] + [r.get(f"df_{g}", 0) for g in WT.GROUPS] + [r.get("tf", 0)]
                                  + [r.get(c, 0) for c in WT.CTX_FEATS])) + "\n")


def row(df=50, **kw):
    r = {f"df_{g}": df for g in WT.VOTING}
    r.update(tf=df * 3, mid=10, capmid=0)
    r.update(kw)
    return r


class TestFamilies(unittest.TestCase):
    def build(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "counts_A.tsv.gz")
            write_table(p, rows, {g: 1000 for g in WT.GROUPS})
            return WF.build(p)

    def test_eligibility_score_and_families(self):
        rows = {"walk": row(80, verb=40, noun=5), "walks": row(40, noun=2, verb=1), "walked": row(30, verb=3),
                "walking": row(30, verb=6), "big": row(90, adj=30), "bigger": row(20, adj=5),
                "new": row(95, adj=40), "news": row(60, noun=30), "rare": row(50, adj=9, df_books=0, df_howto=0,
                                                                                   df_web=0, df_speech=0),
                "sat": row(40, verb=9), "paris": row(40, noun=5, mid=20, capmid=19), "parisian": row(30, adj=4),
                "dogs": row(30, noun=9), "q": row(60, noun=2), "hour": row(50, an=9, a=1),
                "hike": row(40, verb=8), "hiking": row(30, verb=7, noun=1),
                "mark": row(40, noun=9, mid=30, capmid=25), "marks": row(35, noun=9, mid=20, capmid=1),
                "has": row(90, verb=5), "his": row(80, noun=3), "ha": row(40, noun=9), "hi": row(40, noun=9),
                "doing": row(40, verb=9), "do": row(90, verb=40), "rice": row(40, noun=9, mass=7, a=1),
                "ab": row(40, noun=9), "abs": row(40, noun=9)}
        rows["walks"]["noun"] = 2
        res = self.build(rows)
        self.assertNotIn("rare", res["elig"])                       # 2 of 6 groups: not eligible
        fam = res["fam"]
        self.assertEqual(sorted(sum(fam["walk"]["forms"].values(), [])), ["walked", "walking", "walks"])
        self.assertEqual(fam["big"]["forms"], {"er": ["bigger"]})
        self.assertIn("news", fam)                                   # "new" is an adjective: no -s join
        self.assertEqual(fam["hike"]["forms"], {"ing": ["hiking"]})  # e-drop
        self.assertIn("proper", res["flags"]["paris"])
        self.assertIn("letter", res["flags"]["q"])
        self.assertNotIn("paris", res["ranked"])
        self.assertIn("marks", res["ranked"])                        # a flagged base takes no forms
        self.assertEqual((fam["ha"]["forms"], fam["hi"]["forms"]), ({}, {}))   # 2-letter bases take no forms
        self.assertEqual(fam["ab"]["forms"], {})
        self.assertIn("abs", fam)
        self.assertEqual(res["pos"]["has"][0], "func")
        self.assertIn("doing", fam)                                  # "do" is a function word: no family
        self.assertEqual(WF.article(res["rows"]["hour"], "hour"), ("an", "counted"))
        self.assertEqual(WF.article(res["rows"]["dogs"], "unit"), ("an", "vowel_rule"))
        self.assertEqual(WF.value_features(res, "hour")["article"], "an")
        self.assertEqual(WF.value_features(res, "walks"), {"article": "", "number": "pl", "mass": False,
                                                           "basis": "vowel_rule"})
        self.assertEqual(WF.value_features(res, "umbrella")["basis"], "vowel_rule")
        self.assertEqual(WF.value_features(res, "rice")["article"], "")

    def test_bad_bases_and_low_evidence(self):
        rows = {"make": row(90, verb=40), "making": row(60, verb=20), "mak": row(40, verb=9),
                "wat": row(40, adj=9), "water": row(90, noun=40, tf=5000), "old": row(80, adj=30),
                "older": row(40, adj=9), "after": row(90, adj=20, noun=10, tf=100000), "dog": row(60, noun=9)}
        res = self.build(rows)
        self.assertEqual((res["head_of"]["making"], res["fam"]["mak"]["forms"]), ("making", {}))   # make is func
        self.assertEqual((res["head_of"]["water"], res["fam"]["wat"]["forms"]), ("water", {}))     # wat too rare
        self.assertEqual(res["fam"]["old"]["forms"], {"er": ["older"]})
        self.assertEqual((res["pos"]["after"][0], res["pos"]["dog"][0]), ("unk", "noun"))
        self.assertFalse(WF.required_ok(res, "after"))

    def test_score_floor_and_pos_normalization(self):
        rows = {"the": row(900, noun=0), "website": row(50, df_books=0, noun=50),
                "happy": row(60, noun=200, adj=30), "dog": row(60, noun=2000)}
        res = self.build(rows)
        self.assertIn("website", res["elig"])
        self.assertGreater(res["score"]["website"], 0)               # floored, not zero
        self.assertEqual(res["pos"]["happy"][0], "adj")              # raw argmax would say noun
        self.assertEqual(res["pos"]["dog"][0], "noun")

    def test_heldout_word_is_flagged(self):
        w = next(t for t in heldout.HELDOUT_PHRASES if " " not in t and t.isalpha())
        h = next(t for t in heldout.HELDOUT_HEADS if " " not in t and t.isalpha() and not heldout.vocab_hits(t))
        res = self.build({w: row(50, noun=9), h: row(50, noun=9), "dog": row(60, noun=9), "cat": row(60, noun=9)})
        self.assertIn("heldout_vocab", res["flags"][w])
        self.assertNotIn(w, res["ranked"])
        self.assertEqual(res["flags"][h], ["heldout_head"])             # said freely, never a value or required
        self.assertIn(h, res["ranked"])
        self.assertEqual((WF.required_ok(res, h), WF.required_ok(res, "dog")), (False, True))

    def test_jaccard(self):
        a, b = {"ranked": ["a", "b", "c", "d"]}, {"ranked": ["a", "b", "x", "y"]}
        self.assertEqual(WS.jaccard_top(a, b, 2), 1.0)
        self.assertEqual(WS.jaccard_top(a, b, 4), round(2 / 6, 4))


class TestCli(unittest.TestCase):
    def test_vocab_restricted_run_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            core = os.path.join(d, "core")
            texts = {"cccc": "the dog likes the park and the dog runs", "gutenberg": "the dog sat by the fire",
                     "youtube": "hey the dog is here", "news": "the dog story ran today", "foodista": "the dog bowl",
                     "stackexchange": "the dog question here"}
            shards = []
            for src, t in texts.items():
                os.makedirs(os.path.join(core, src))
                with gzip.open(os.path.join(core, src, "s.jsonl.gz"), "wt") as f:
                    for i in range(30):
                        f.write(json.dumps({"id": f"{src}:{i}", "text": t + f" extra{i % 3}", "meta": {}}) + "\n")
                shards.append({"source": src, "shard": f"{src}/s.jsonl.gz"})
            with open(os.path.join(core, "MANIFEST.json"), "w") as f:
                json.dump({"shards": shards}, f)
            vp = os.path.join(d, "v.txt")
            with open(vp, "w") as f:
                f.write("# c\ndog\nthe\n")
            total, meta = WL.run(core, d, ("full",), workers=1, vocab=vp)
            words = {w for g in total["full"].values() for c in g["feats"].values() if c for w in c}
            self.assertEqual(words, {"dog", "the"})
            self.assertEqual(meta["vocab"]["words"], 2)
            full, _ = WL.run(core, d, ("full",), workers=1)
            self.assertIn("park", {w for g in full["full"].values() for w in g["feats"]["df"]})
            pa, pb = os.path.join(d, "counts_A.tsv.gz"), os.path.join(d, "counts_B.tsv.gz")
            write_table(pa, {"dog": row(50, noun=9), "cat": row(50, noun=9)}, {g: 1000 for g in WT.GROUPS})
            write_table(pb, {"dog": row(50, noun=9), "cow": row(50, noun=9)}, {g: 1000 for g in WT.GROUPS})
            WS.main([os.path.join(d, "out"), pa, pb])
            st = json.load(open(os.path.join(d, "out", "wordlist_stats.json")))
            cand = open(os.path.join(d, "out", "vocab_candidates.txt")).read().split()
        self.assertEqual(st["stability"]["jaccard_top2000"], round(1 / 3, 4))
        self.assertTrue(st["stability"]["full_read_needed"])
        self.assertTrue({"dog", "cat", "cow", "monday"} <= set(cand))


class TestMine(unittest.TestCase):
    def test_patterns_and_gates(self):
        got = {c for c, _, _ in mine.candidates("Hi there, quick one. My dog is called Rex. Sorry, I meant the cat. "
                                                "From now on, keep it short. Thanks, that's all for now, bye!",
                                                tuple(mine.PAT))}
        self.assertEqual(got, {"O_greet", "K_plant", "M_fix", "R_rule", "C_close"})
        self.assertIsNone(mine.gate_reason("My dog is a beagle."))
        self.assertIsNone(mine.gate_reason("Sorry, I meant the cat, I'm sure."))
        self.assertEqual([mine.gate_reason(s) for s in ("My dog is called Rex.", "I live in NY.")], ["NAME"] * 2)
        self.assertEqual(mine.gate_reason("My dog is 3 years old."), "CHARS")
        self.assertEqual(mine.gate_reason("Hello."), "LEN")
        self.assertEqual(mine.gate_reason("Hi, as an AI you know a lot."), "AI_ISM")
        self.assertEqual(mine.gate_reason(f"Well, {heldout.MARKER_TERMS[0]}, it is fine."), "HELDOUT_VOCAB")

    def test_main_drops_reserved_text(self):
        import oodh
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "core", "oasst2"))
            res_id = next(f"t{i}" for i in range(99) if oodh.in_oodh_reserve(f"t{i}"))
            ok_id = next(f"t{i}" for i in range(99) if not oodh.in_oodh_reserve(f"t{i}"))
            turn = "Hello friend, nice day. My dog is a beagle."
            doc = {"id": "oasst2:" + ok_id, "source": "oasst2", "text": "", "meta": {"tree_id": ok_id},
                   "turns": [{"role": "user", "text": turn}]}
            res_doc = dict(doc, meta={"tree_id": res_id})
            with gzip.open(os.path.join(d, "core", "oasst2", "o.jsonl.gz"), "wt") as f:
                f.write(json.dumps(doc) + "\n" + json.dumps(res_doc) + "\n")
            with open(os.path.join(d, "core", "MANIFEST.json"), "w") as f:
                json.dump({"shards": [{"source": "oasst2", "shard": "oasst2/o.jsonl.gz"}]}, f)
            with gzip.open(os.path.join(d, "raw.gz"), "wt") as f:
                f.write(json.dumps({"message_tree_id": res_id, "prompt": {"role": "prompter", "replies": [],
                                    "text": "Oh. Hello friend, nice day. Anyway, what now?"}}) + "\n")
            mine.main([os.path.join(d, "core"), os.path.join(d, "raw.gz"), os.path.join(d, "out"), "--workers", "1"])
            got = [json.loads(x) for x in open(os.path.join(d, "out", "mined_examples_v0.jsonl"))]
            st = json.load(open(os.path.join(d, "out", "mined_stats.json")))
        self.assertEqual([(r["class"], r["text"]) for r in got], [("K_plant", "My dog is a beagle.")])
        self.assertEqual(st["notes"]["oodh_reserve_hit"], 1)
        self.assertEqual(st["notes"]["drop_O_greet_OODH_TEXT"], 1)

    def test_pick_is_seeded_and_capped(self):
        rows = [{"class": "K_plant", "source": "oasst2", "text": f"My dog is {w}.", "doc_id": f"d{i}", "turn": 0,
                 "sent": 0} for i, w in enumerate("rex max bo pip lu".split())]
        rows.append(dict(rows[0], doc_id="dup"))
        a, b = mine.pick(rows, 3), mine.pick(list(reversed(rows)), 3)
        self.assertEqual([r["text"] for r in a], [r["text"] for r in b])
        self.assertEqual(len(a), 3)


if __name__ == "__main__":
    unittest.main()
