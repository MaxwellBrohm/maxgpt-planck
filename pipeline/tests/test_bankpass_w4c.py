"""Bank pass W4 unit rules: the per-bank choices (w4banks), the derived features (w4feats: votes, paraphrase checks,
relations, entity attributes, the topic text check and groups) and the label re-vote (w4vote). No model; FAKE test
text only."""
import re
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
from bankpass import plan, store, w4banks as WB, w4feats as WF, w4gates as W4G, w4vote as WV
from bankpass import wordload

Q, M, G = plan.TEACHER_ORDER
NOG = W4G.Gates()


def it(iid, bank, text, author=Q, w3="kept", status="kept", feats=None, drop_w3=None):
    return {"id": iid, "bank": bank, "text": text, "status": status, "drop": None, "author": store.teacher_author(author),
            "features": feats or {}, "judges": [], "call": {"call_id": iid}, "gates": {"gate_hash": "x", "hits": []},
            "w3": w3, "drop_w3": drop_w3}


def wl_of(rows):
    fams = {}
    for i, (w, pos, extra) in enumerate(rows, 1):
        fams[w] = dict({"pos": pos, "pos_status": "confirmed", "rank": i, "lists": {"RS", "RM", "RL"}, "forms": {},
                        "flags": set(), "required_ok": pos in ("noun", "verb", "adj"), "a": 3, "an": 0, "mass": 0,
                        "tf": 9}, **extra)
    return wordload.WordList(fams, "computed", "wordlist@t:computed", {"file_sha256": "t" * 64})


WL = wl_of([("garden", "noun", {}), ("soil", "noun", {}), ("dig", "verb", {"forms": {"irr": ["dug"]}}),
            ("green", "adj", {}), ("seed", "noun", {}), ("tired", "adj", {}), ("rare", "noun", {"lists": {"RL"}}),
            ("lovely", "adj", {}), ("quickly", "adv", {}), ("Paris", "noun", {"flags": {"proper"}}),
            ("very", "adv", {}), ("kettle", "noun", {})])


class TestBanks(unittest.TestCase):
    def test_as_record(self):
        self.assertEqual(WB.as_record(it("a", "b", "t"))["status"], "kept")
        r = WB.as_record(it("a", "b", "t", w3="dropped", drop_w3="JUDGE_DROP"))
        self.assertEqual((r["status"], r["drop"]), ("dropped", "JUDGE_DROP"))
        self.assertEqual(WB.as_record(it("a", "b", "t", w3="pending"))["drop"], "JUDGE_MISSING")
        self.assertNotIn("w3", r)

    def test_pool_rules(self):
        names = [it("n1", "pool.assistant_name", "Garden", Q), it("n2", "pool.assistant_name", "Zorbit", M),
                 it("n3", "pool.assistant_name", "May", G)]
        rs, _ = WB.pool_bank(names, "assistant_name", NOG, WL)
        self.assertEqual([r["drop"] for r in rs], ["WORD_COLLISION", None, "PROGRAM_COLLISION"])
        pets = [it("p1", "pool.pet_name", "Garden", Q)]
        self.assertEqual(WB.pool_bank(pets, "pet_name", NOG, WL)[0][0]["status"], "kept")   # names only: assistant
        rel = [it("r1", "pool.relation", "aunt"), it("r2", "pool.relation", "crew", M), it("r3", "pool.relation", "boss", G),
               it("r4", "pool.relation", "fan", Q), it("r5", "pool.relation", "uncle", M)]
        rs, _ = WB.pool_bank(rel, "relation", NOG, WL, {"aunt": ("woman", "one"), "crew": ("either", "more"),
                                                        "boss": (None, "one"), "uncle": ("man", "one")})
        self.assertEqual([(r["drop"], r["features"].get("sex")) for r in rs],
                         [(None, "F"), ("REL_PLURAL", "U"), (None, "U"), ("REL_PLURAL", "U"), (None, "M")])
        many = [it(f"h{i}", "pool.hobby", f"hob{chr(97 + i)}", Q) for i in range(6)] + \
            [it("m1", "pool.hobby", "knit", M), it("m2", "pool.hobby", "sew", M), it("g1", "pool.hobby", "row", G),
             it("g2", "pool.hobby", "sail", G)]
        rs, _ = WB.pool_bank(many, "hobby", NOG, WL)
        kept = [r["author"]["model"] for r in rs if r["status"] == "kept"]
        self.assertEqual((kept.count(Q), kept.count(M), kept.count(G)), (2, 2, 2))        # thirds: min 2 x 1.1
        ents = [it("e1", "pool.entity_kind", "bakery"), it("e2", "pool.entity_kind", "museum", M)]
        rs, _ = WB.pool_bank(ents, "entity_kind", NOG, WL, attrs={"e1": [["opening day", "weekday", "opens on {v}"]] * 2,
                                                                  "e2": [["location", "city", "is in {v}"]]})
        self.assertEqual([r["drop"] for r in rs], [None, "ENTITY_FEW_ATTRS"])
        self.assertEqual(rs[0]["features"]["article"], "a")

    def test_req_and_generic(self):
        sets = {Q: {"nouns": ["garden", "soil", "rare", "Paris"], "verbs": ["dug", "seed"], "adjs": ["green", "tired"]},
                M: {"nouns": ["garden", "rare", "Paris"], "verbs": ["dig", "seed"], "adjs": ["green", "tired"]},
                G: {"nouns": ["soil", "kettle"], "verbs": ["seed"], "adjs": ["lovely"]}}
        req = WB.req_words(sets, WL, generic={"soil"})
        self.assertEqual(req, {"noun": ["garden"], "verb": ["dig"], "adj": ["green"]})   # dug counts for dig
        many = {f"t{i}": {Q: {"nouns": ["garden"] if i < 5 else ["seed"]}} for i in range(100)}
        self.assertEqual(WB.generic_families(many, WL), {"garden", "seed"})
        few = {f"t{i}": {Q: {"nouns": ["garden"] if i < 4 else ["seed"]}} for i in range(100)}
        self.assertEqual(WB.generic_families(few, WL), {"seed"})                       # 4 of 100 is under 5%

    def test_exact_dedup_and_computed(self):
        rs = [dict(it("b", "intent", "ask about it"), status="kept"), dict(it("a", "intent", "Ask about it."))]
        WB.exact_dedup(rs)
        self.assertEqual([r["drop"] for r in rs], ["DUP_EXACT", None])                 # id order: a stays
        tops = [it("t1", "topic", "a garden"), it("t2", "topic", "a kettle", M), it("t3", "topic", "a cat", G)]
        rs, _ = WB.topic_bank(tops, {"t1", "t3"}, {"t1": "home"}, NOG)
        self.assertEqual([(r["drop"], r["features"].get("group")) for r in rs],
                         [(None, "home"), ("NOT_SELECTED", None), (None, None)])
        WF.drop_without_sets(rs, {"t1"})
        self.assertEqual([r["drop"] for r in rs], [None, "NOT_SELECTED", "TOPIC_NO_WORDSETS"])
        author = {"kind": "computed", "code_sha256": "c", "input_sha256": "i"}
        av, _ = WB.avoid_bank(WL, author, NOG)
        self.assertEqual([r["text"] for r in av if r["status"] == "kept"], ["green", "tired", "lovely", "quickly"])
        rq, _ = WB.req_bank(WL, "noun", author, NOG)
        self.assertNotIn("rare", [r["text"] for r in rq])                              # RL only: not required


class TestFeats(unittest.TestCase):
    def test_votes(self):
        def v(bank, t, x, kind="vote"):
            return {"kind": kind, "bank": bank, "lines": [x], "plan": {"cand_ids": ["c1", "c2", "c3"]},
                    "author": store.teacher_author(t)}
        recs = [v("label.a", Q, "2"), v("label.a", M, "2"), v("label.a", G, "1"), v("label.b", Q, "67"),
                v("label.b", M, "3"), v("label.b", G, "1"), v("label.c", Q, "1", "vote2"), v("label.c", M, "1", "vote2")]
        self.assertEqual(WF.vote_winners(recs), {"label.a": "c2"})
        self.assertEqual(WF.vote_winners(recs, "vote2"), {"label.c": "c1"})

    def test_para_relation_entities(self):
        ok = {"kind": "check", "plan": {"para_id": "p1"}, "lines": ["1: yes", "2: yes", "added: no"]}
        bad = {"kind": "check", "plan": {"para_id": "p2"}, "lines": ["1: yes", "2: no", "added: no"]}
        add = {"kind": "check", "plan": {"para_id": "p3"}, "lines": ["1: yes", "added: yes"]}
        self.assertEqual(WF.para_checks([ok, bad, add]), {"p1": True, "p2": False, "p3": False})
        rf = [{"kind": "relfeat", "lines": ["aunt: woman, one", "crew: either, more"]},
              {"kind": "relfeat", "lines": ["aunt: woman, one", "crew: man, more"]},
              {"kind": "relfeat", "lines": ["aunt: either, one", "crew: either, one"]}]
        rf[0]["lines"].append("pal: woman, one")
        rf[1]["lines"].append("pal: man, one")
        rf[2]["lines"].append("pal: either, one")
        self.assertEqual(WF.relation_features(rf), {"aunt": ("woman", "one"), "crew": ("either", "more"),
                                                    "pal": (None, "one")})             # a 1-1-1 split: no sex
        its = {"a1": it("a1", "attr", "opening day", feats={"kind_id": "k1", "vtype": "weekday"}),
               "a2": it("a2", "attr", "opening day", feats={"kind_id": "k2", "vtype": "month"}),
               "a3": it("a3", "attr", "colour", feats={"kind_id": "k1", "vtype": "colour"}),
               "a4": it("a4", "attr", "hall", feats={"kind_id": "k1", "vtype": "city"}),
               "a5": it("a5", "attr", "opening day", feats={"kind_id": "k1", "vtype": "weekday"}),
               "p5": it("p5", "pred", "first opens on {v}", feats={"attr_id": "a5"}),
               "p1": it("p1", "pred", "opens on {v}", feats={"attr_id": "a1"}),
               "p2": it("p2", "pred", "opens in {v}", feats={"attr_id": "a2"}),
               "p4": it("p4", "pred", "is held in {v}", feats={"attr_id": "a4"})}
        out, drops = WF.entity_attrs(its, NOG)
        self.assertEqual(out, {"k1": [["opening day", "weekday", "opens on {v}", "a1", "p1"],
                                      ["hall", "city", "is held in {v}", "a4", "p4"]]})
        self.assertEqual(drops, {"ATTR_TYPE_CONFLICT": 1, "NO_PRED": 1, "ATTR_REPEATED": 1})

    def test_topic_text_check(self):
        items = [it("w1", "topicwords.raw", "x", Q, feats={"topic_id": "t1", "topic": "old text"}),
                 it("w2", "topicwords.raw", "x", M, feats={"topic_id": "t1", "topic": "new text"}),
                 it("w3", "topicwords.raw", "x", G, feats={"topic_id": "t1", "topic": "new text"}),
                 it("w4", "topicwords.raw", "x", Q, feats={"topic_id": "t2", "topic": "two"})]
        self.assertEqual(WF.topic_text_check(items, {"t1": "new text", "t2": "two"}), 1)
        self.assertEqual(items[0]["drop_w3"], "TOPIC_TEXT_MISMATCH")
        self.assertEqual(WF.wordset_topics(items), {"t1"})                             # t2: one teacher only
        groups = [{"kind": "group", "lines": ["home"], "plan": {"topic_id": "t1", "topic": "new text"}},
                  {"kind": "group", "lines": ["pets"], "plan": {"topic_id": "t2", "topic": "old two"}}]
        self.assertEqual(WF.topic_groups(groups, {"t1": "new text", "t2": "two"}), {"t1": "home"})

    def test_merge_topicwords(self):
        from bankpass import load
        recs = [{"features": {"topic": "a", "nouns": ["x", "y"], "verbs": ["v"], "adjs": [],
                              "req": {"noun": ["x"], "verb": [], "adj": []}}},
                {"features": {"topic": "a", "nouns": ["y", "z"], "verbs": [], "adjs": ["q"],
                              "req": {"noun": ["x"], "verb": ["v"], "adj": []}}}]
        words, req = load.merge_topicwords(recs)
        self.assertEqual(words, {"a": ("x y z", "v", "q")})
        self.assertEqual(req, {"a": {"noun": ["x"], "verb": ["v"], "adj": []}})


class TestVote(unittest.TestCase):
    def test_answer_regex_and_parse(self):
        rx = WV.answer_re(12)
        self.assertTrue(re.fullmatch(rx, "12") and re.fullmatch(rx, "1"))
        self.assertFalse(re.fullmatch(rx, "13") or re.fullmatch(rx, "0") or re.fullmatch(rx, "67"))
        self.assertEqual(WV.parse("7\nEND", 9), (["7"], None))
        self.assertEqual(WV.parse("10\nEND", 9), ([], "OUT_OF_RANGE"))
        self.assertEqual(WV.parse("7", 9)[1], "NO_END")
        c = {"call_id": "vote2.label.a.q", "teacher": Q, "what": "a thing", "cands": ["x", "y"], "seed": 3,
             "bank": "label.a"}
        prompt, spec = WV.render(c)
        self.assertIn("1. x", prompt)
        self.assertEqual(spec["regex"].split("\\n")[0], "(?:2|1)")

    def test_run_skips_done(self):
        import tempfile
        from teachers import decode as D

        class Cl:
            n = 0

            def generate(self, prompt, spec, seed, preset=None, call=None):
                Cl.n += 1
                return {"text": "1\nEND", "finish": "stop", "decode": {"regex_sha256": D.sha(spec["regex"])}}
        cs = [{"call_id": f"vote2.label.{k}.q", "teacher": Q, "what": "w", "cands": ["x", "y"], "seed": 1,
               "bank": f"label.{k}"} for k in ("a", "b")]
        with tempfile.TemporaryDirectory() as d:
            path = WV.path_of(d, Q)
            self.assertEqual((WV.run(Cl(), cs, path), WV.run(Cl(), cs, path), Cl.n), (2, 0, 2))
            self.assertEqual([r["lines"] for r in WV.records(d)], [["1"], ["1"]])


if __name__ == "__main__":
    unittest.main()
