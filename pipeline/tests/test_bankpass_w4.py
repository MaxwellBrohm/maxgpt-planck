"""Bank pass W4, word labels and gates (bankpass/w4words.py, w4gates.py): the POS vote, -ly and irregular joins, the
v1 TSV that wordload reads, and the RC-12 dev, OOD-H, safety and mined-copy gates. No model; FAKE fixture text only."""
import gzip
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
from bankpass import plan, store, w4gates as W4G, w4words as WW, wordload

Q, M, G = plan.TEACHER_ORDER
H = "ab" * 32
COLS = ["headword", "rank", "lists", "pos_proxy", "pos_conf", "pos_status", "forms", "score", "tf", "article",
        "article_basis", "a", "an", "pl_ctx", "mass_ctx", "proper_share", "flags", "required_ok"]
ROWS = [("go", 1, "noun"), ("went", 2, "verb"), ("dog", 3, "noun"), ("quick", 4, "adj"), ("quickly", 5, "adj"),
        ("short", 6, "adj"), ("shortly", 7, "adv"), ("left", 8, "noun"), ("leave", 9, "verb"), ("hour", 10, "noun"),
        ("the", 11, "func"), ("walk", 12, "verb"), ("zing", 13, "verb"), ("hourly", 14, "adj")]


def write_v0(d):
    p = os.path.join(d, "v0.tsv")
    with open(p, "w") as f:
        f.write("# controlled word list v0 (fixture)\n")
        f.write(f"# input manifest sha256 {H}; wordlist.py sha256 {H}; wordfam.py sha256 {H}\n# docs\n# pos\n")
        f.write("\t".join(COLS) + "\n")
        for w, k, pos in ROWS:
            forms = "ed:walked;s:walks" if w == "walk" else ""
            f.write("\t".join(map(str, [w, k, "RS,RM,RL", pos, 1.0, "pending", forms, 0.1, 100, "a", "counted", 5, 0,
                                        0, 0, 0.0, "", 1])) + "\n")
    return p


def rec(kind, cid, lines, author=Q, **kw):
    return dict({"kind": kind, "call_id": cid, "lines": lines, "author": store.teacher_author(author)}, **kw)


def label_recs():
    out = []
    pos = {Q: ["go: verb", "went: verb", "dog: noun", "quick: adjective", "short: adjective", "left: adjective",
               "leave: verb", "hour: noun", "the: determiner", "quickly: adverb", "walk: verb"],
           M: ["go: verb", "went: verb", "dog: noun", "quick: adjective", "short: adjective", "left: adjective",
               "leave: verb", "hour: verb", "the: determiner", "quickly: adverb", "walk: verb"],
           G: ["go: verb", "went: noun", "dog: adjective", "quick: adjective", "short: noun", "left: verb",
               "leave: verb", "hour: adjective", "the: determiner", "quickly: adverb",
               "walk: verb"]}
    for t, lines in pos.items():
        out.append(rec("pos", f"pos2.{store.SHORT[t]}.0", lines, t))
        out.append(rec("pos", f"pos.{store.SHORT[t]}.0", ["go: noun"] * 1, t))           # version 1: superseded
        out.append(rec("ly", f"ly.{store.SHORT[t]}.0", ["quick, quickly: same", "short, shortly: " +
                                                        ("same" if t == Q else "different"), "hour, hourly: same"], t))
        out.append(rec("verbs", f"verbs.pos2.{store.SHORT[t]}.0.0", ["go: went, " + ("goed" if t == G else "gone"),
                                                                     "leave: left, left", "walk: walked, " +
                                                                     ("walked" if t == G else "walkt")], t))
    out.append(rec("pos", "pos2.q.1", ["zing: verb"], Q, problem="NO_END"))              # a parse problem: unused
    return out


class TestPosVote(unittest.TestCase):
    def test_three_of_four_votes(self):
        self.assertEqual(WW.decide({Q: "noun", M: "noun", G: "noun"}, "adj"), ("noun", "confirmed"))
        self.assertEqual(WW.decide({Q: "noun", M: "noun", G: "adj"}, "noun"), ("noun", "confirmed"))
        self.assertEqual(WW.decide({Q: "noun", M: "noun", G: "adj"}, "adj"), ("amb", "amb"))
        self.assertEqual(WW.decide({Q: "noun", M: "noun", G: "adj"}, "unk"), ("amb", "amb"))
        self.assertEqual(WW.decide({Q: "noun", M: "adj"}, "noun"), ("amb", "amb"))
        self.assertEqual(WW.decide({}, "verb"), ("verb", "pending"))
        self.assertEqual(WW.decide({Q: "func", M: "func"}, "func"), ("func", "confirmed"))

    def test_labels_read_current_calls_only(self):
        lab = WW.labels(label_recs())
        self.assertEqual(lab["pos"]["go"], {Q: "verb", M: "verb", G: "verb"})     # not the version 1 "noun"
        self.assertNotIn("zing", lab["pos"])
        self.assertEqual(len(lab["pos"]["walk"]), 3)                             # the problem record is skipped
        self.assertEqual(lab["verbs"]["go"][Q], ("went", "gone"))
        self.assertEqual(lab["ly"][("short", "shortly")], {Q: "same", M: "different", G: "different"})
        old = rec("verbs", "verbs.pos.q.0.0", ["dog: dogged, dogged"])
        self.assertNotIn("dog", WW.labels([old])["verbs"])


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.v0 = write_v0(self.tmp.name)
        self.out = os.path.join(self.tmp.name, "v1.tsv")
        elig = {w for w, _, _ in ROWS} | {"gone", "walked", "walks"}
        self.stats = WW.build(self.v0, label_recs(), elig, self.out, "fixture")
        self.wl = wordload.read(self.out)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pos_and_required(self):
        f = self.wl.families
        self.assertEqual((f["go"]["pos"], f["go"]["pos_status"]), ("verb", "confirmed"))   # teachers beat the proxy
        self.assertEqual((f["hour"]["pos"], f["hour"]["pos_status"]), ("amb", "amb"))
        self.assertEqual((f["dog"]["pos"], f["dog"]["pos_status"]), ("noun", "confirmed"))   # 2 + proxy
        self.assertEqual(f["zing"]["pos_status"], "pending")
        self.assertTrue(f["dog"]["required_ok"])
        self.assertFalse(f["zing"]["required_ok"])                     # a pending POS is never a required word
        self.assertFalse(f["hour"]["required_ok"])
        self.assertFalse(f["the"]["required_ok"])
        self.assertEqual(self.wl.required("noun"), ["dog"])

    def test_joins(self):
        self.assertEqual(self.wl.head("went"), "go")                  # irregular past, a confirmed verb row joins
        self.assertEqual(self.wl.head("gone"), "go")                  # eligible participle
        self.assertEqual(self.wl.head("quickly"), "quick")            # 3 of 3 "same"
        self.assertEqual(self.wl.head("shortly"), "shortly")          # 1 of 3 "same": stays apart
        self.assertEqual(self.wl.head("left"), "left")                # a confirmed adjective, not leave's past
        self.assertEqual(self.wl.head("walked"), "walk")              # regular forms untouched
        self.assertIsNone(self.wl.families["went"]["rank"])
        self.assertIn("joined:go", self.wl.families["went"]["flags"])
        self.assertEqual(self.wl.head("walkt"), "walkt")              # 2 of 3 agree, but not an eligible word
        self.assertEqual(self.stats["refused_LY_DIFFERENT"], 1)
        self.assertEqual(self.wl.head("hourly"), "hourly")            # 3 "same", but hour is amb, not an adjective
        self.assertEqual(self.stats["refused_LY_BASE_NOT_ADJ"], 1)
        self.assertEqual(self.stats["refused_IRR_OTHER_POS"], 2)
        self.assertEqual(self.stats["refused_IRR_NOT_ELIGIBLE"], 1)
        self.assertEqual(self.stats["join_irr"], 2)                    # went, gone; walked is the regular form

    def test_header_keeps_provenance_and_marks_v1(self):
        with open(self.out) as f:
            head = [ln for ln in f if ln.startswith("#")]
        self.assertTrue(any("wordfam.py sha256" in h for h in head))
        self.assertTrue(any("W4 v1" in h for h in head))
        self.assertEqual(self.stats["sha256"], store.sha256_file(self.out))


class TestGates(unittest.TestCase):
    def setUp(self):
        dev = ["could you tell me which day the spring fair opens this year please"]
        res = ["I have been learning to bake sourdough bread at home for weeks now"]
        mined = ["honestly I never thought I would enjoy gardening this much"]
        self.g = W4G.Gates(W4G.Ngrams(dev, 8), W4G.Ngrams(res, 8), W4G.Ngrams(mined, 6),
                           re.compile(r"(?<![a-z])(?:badword)(?![a-z])", re.I), W4G.Ngrams(dev, 13))

    def test_ngram_lengths(self):
        self.assertEqual(self.g.text_hits("so which day the spring fair opens this year"), [
            ("RC12DEV_ECHO", "which day the spring fair opens this year")])
        self.assertEqual(self.g.text_hits("which day the spring fair opens this"), [])          # 7 words
        self.assertEqual(self.g.text_hits("Learning to bake sourdough bread at home for fun")[0][0], "OODH_ECHO")
        self.assertEqual(self.g.text_hits("I never thought I would enjoy chess")[0][0], "MINED_COPY")
        self.assertEqual(self.g.text_hits("I never thought I would like chess"), [])
        self.assertEqual(self.g.text_hits("what a BadWord thing")[0][0], "SAFETY_LIST")

    def test_first_hit_in_text_order(self):
        ws = [f"w{i}" for i in range(40)]
        g = W4G.Gates(mined=W4G.Ngrams([" ".join(ws)], 6))
        self.assertEqual(g.text_hits(" ".join(ws[3:])), [("MINED_COPY", " ".join(ws[3:9]))])

    def test_template_runs_and_fills(self):
        hit = self.g.template_hits("{M}I never thought I would enjoy {v} this much.", "key.hobby.plant")
        self.assertEqual(hit[0][0], "MINED_COPY")                                     # inside a hole-free run
        g2 = W4G.Gates(mined=W4G.Ngrams(["she said pizza is my favourite thing"], 6))
        tpl = "I said {v} is my favourite thing."
        with mock.patch.object(W4G.gates, "sample_fill", lambda b, s, r: dict(v="soup")):
            self.assertEqual(g2.template_hits(tpl, "key.fav_food.plant"), [])
        with mock.patch.object(W4G.gates, "sample_fill", lambda b, s, r: dict(v="pizza")):
            self.assertEqual(g2.template_hits(tpl, "key.fav_food.plant"),
                             [("MINED_COPY", "fill said pizza is my favourite thing")])   # formed across the hole

    def test_holes_split_runs(self):
        g = W4G.Gates(mined=W4G.Ngrams(["honestly i said is my favourite thing"], 6))
        with mock.patch.object(W4G.gates, "sample_fill", lambda b, s, r: dict(v="soup")):
            self.assertEqual(g.template_hits("I said {v} is my favourite thing.", "key.fav_food.plant"), [])

    def test_apply_marks_drop_and_keeps_hits(self):
        it = {"status": "kept", "drop": None, "gates": {"gate_hash": "x", "hits": []}}
        W4G.apply(it, [("MINED_COPY", "a b c d e f")])
        self.assertEqual((it["status"], it["drop"]), ("dropped", "MINED_COPY"))
        self.assertEqual(it["gates"]["w4"], [["MINED_COPY", "a b c d e f"]])
        self.assertEqual(it["gates"]["gate_hash"], "x")

    def test_from_paths_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            dev, trees, mined, rub = (os.path.join(d, x) for x in ("dev.jsonl", "t.jsonl.gz", "m.jsonl", "r.jsonl"))
            with open(dev, "w") as f:
                f.write(json.dumps({"turns": [{"text": "one two three four five six seven eight nine"}]}) + "\n")
            with gzip.open(trees, "wt") as f:
                f.write(json.dumps({"message_tree_id": "x", "prompt": {"role": "prompter", "text": "hi"}}) + "\n")
            with open(mined, "w") as f:
                f.write(json.dumps({"text": "alpha beta gamma delta epsilon zeta eta"}) + "\n")
            store.write_jsonl(rub, [store.make_item("Z", "rubric.safety", 0, "zonk", {"kind": "human"})])
            g = W4G.Gates.from_paths(dev, trees, mined, rub)
        self.assertEqual(g.text_hits("one two three four five six seven eight")[0][0], "RC12DEV_ECHO")
        self.assertEqual(g.text_hits("two three four five six seven eight"), [])        # 7 of the 8
        self.assertEqual(g.text_hits("alpha beta gamma delta epsilon zeta")[0][0], "MINED_COPY")
        self.assertEqual(g.text_hits("alpha beta gamma delta epsilon"), [])            # 5 of the 6
        self.assertEqual(g.text_hits("a zonk")[0][0], "SAFETY_LIST")
        self.assertTrue(g.n13("one two three four five six seven eight nine ten") is False)
        self.assertEqual(g.describe()["rc12_dev"], {"n": 8, "texts": 1})

    def test_source_readers(self):
        with tempfile.TemporaryDirectory() as d:
            dev = os.path.join(d, "dev.jsonl")
            with open(dev, "w") as f:
                f.write(json.dumps({"turns": [{"text": "u one", "ideal": "a one"}],
                                    "probes": [{"question": "q one", "prefix": "p", "gold": "g", "ideal": "i"}]}) + "\n")
            self.assertEqual(W4G.rc12_dev_texts(dev), ["u one", "a one", "q one", "p g", "i"])
            sys.path.insert(0, os.path.join(os.path.dirname(corpus.__globals__["PIPE"]), "corpus"))
            import oodh
            ids = [f"tree{i}" for i in range(40)]
            res = [t for t in ids if oodh.in_oodh_reserve(t)]
            self.assertTrue(res and len(res) < len(ids))
            trees = os.path.join(d, "t.jsonl.gz")
            with gzip.open(trees, "wt") as f:
                for t in ids:
                    f.write(json.dumps({"message_tree_id": t, "prompt": {"role": "prompter", "text": "user " + t,
                                        "replies": [{"role": "assistant", "text": "bot " + t}]}}) + "\n")
            turns, n = W4G.oodh_reserved_turns(trees)
            self.assertEqual((sorted(turns), n), (sorted("user " + t for t in res), len(res)))


if __name__ == "__main__":
    unittest.main()
