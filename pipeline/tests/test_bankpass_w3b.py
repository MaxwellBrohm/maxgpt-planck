"""Bank pass W3, unit rules: server and client guards, the plan's overgeneration and presets, the pre-judge dedup and
the judged view, topic selection, word sets, hole specs and the judge's fresh fill. No model; FAKE test text only."""
import json
import math
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
from bankpass import (bankserve, client as CL, hold, judge3 as J3, plan, specs, store, w3deps as WD, w3fill,
                      w3items as WI, w3judge as WJ, w3plan, w3render as WR, w3state as WS, wordload)
from teachers import decode as D

Q, M, G = plan.TEACHER_ORDER


def item(iid, bank, text, author, status="kept", feats=None, judges=()):
    return {"id": iid, "bank": bank, "text": text, "status": status, "author": store.teacher_author(author),
            "features": feats or {}, "judges": list(judges), "call": {"call_id": iid}}


class TestServerClient(unittest.TestCase):
    INFO = {"structured_backend": "xgrammar", "line_sep": "\\n", "presets": ["card", "shared"], "dash_ban": {"n": 1},
            "phrase_ban": {"rule": "x"}}

    def test_bank_regex_accepted_and_guarded(self):
        req = bankserve.decode_request({"bank_regex": "a\\nEND", "ban": "dash"}, self.INFO)
        self.assertEqual((req["regex"], req["label_rule"]), ("a\\nEND", bankserve.BANK_RULE))
        with self.assertRaises(ValueError):
            bankserve.decode_request({"bank_regex": "x", "constraint": {"mode": "labels", "lines": [["U1", None]]}},
                                     self.INFO)
        with self.assertRaises(ValueError):
            bankserve.decode_request({"bank_regex": "x"}, dict(self.INFO, structured_backend=None))
        with self.assertRaises(ValueError):
            bankserve.decode_request({"bank_regex": ""}, self.INFO)

    def test_client_refuses_a_row_without_what_it_asked(self):
        cl = CL.Client("http://127.0.0.1:1", Q)
        spec = {"regex": "a\\nEND", "max_tokens": 9}
        good = {"decode": {"regex_sha256": D.sha("a\\nEND"), "ban": {"x": 1}, "phrases": {"x": 1}, "preset": "shared"}}
        cl.verify(good, spec, "shared")
        for bad in ({"regex_sha256": "nope"}, {"ban": None}, {"phrases": None}, {"preset": "card"}):
            with self.assertRaises(CL.CallError):
                cl.verify({"decode": dict(good["decode"], **bad)}, spec, "shared")
        with self.assertRaises(CL.CallError):                       # greedy calls must get temperature 0
            cl.verify({"decode": dict(good["decode"], sampling={"temperature": 0.8})}, spec, None)
        with self.assertRaises(ValueError):
            CL.Client("http://10.0.0.2:1", Q)


class TestPlanAndRender(unittest.TestCase):
    def test_line_overgen_thirds_and_presets(self):
        calls = w3plan.line_calls({"open.greet": 40, "key.job.query": 30}, w3fill.Inputs())
        for bank, t in (("open.greet", 40), ("key.job.query", 30)):
            mine = [c for c in calls if c["bank"] == bank]
            self.assertEqual(len(mine), math.ceil(t * w3plan.LINE_OVERGEN / 10))
            n = [sum(c["teacher"] == x for c in mine) for x in plan.TEACHER_ORDER]
            self.assertLessEqual(max(n) - min(n), 1)
        self.assertEqual(WR.preset(calls[0]), WR.AUTHOR_PRESET[calls[0]["teacher"]])
        self.assertIsNone(WR.preset(dict(calls[0], kind="judge")))
        self.assertEqual(WR.AUTHOR_PRESET["gemma-4-12b"], "card")   # Gemma has no shared preset (D4)

    def test_mined_examples_only_where_listed(self):
        inp = w3fill.Inputs(mined=[{"class": "K_plant", "text": "I live near the sea.", "doc_id": "d1",
                                    "license": "cc-by-4.0"}])
        self.assertIsNotNone(inp.example("x", "key.home_city.plant"))
        self.assertIsNone(inp.example("x", "key.home_city.query"))
        self.assertIsNone(inp.example("x", "system.plain"))

    def test_hole_specs_relaxed_for_old_and_p_obj(self):
        sp = specs.line_specs()
        self.assertNotIn("old", sp["key.fav_colour.corr"]["holes_required"])
        self.assertIn("old", sp["key.fav_colour.corr"]["holes_allowed"])
        self.assertNotIn("p_obj", sp["key.person_name.twin"]["holes_required"])
        self.assertIn("v", sp["key.person_name.twin"]["holes_required"])


class TestStateRules(unittest.TestCase):
    def test_prejudge_dedup_keeps_the_first_in_author_order(self):
        its = {"a": item("a", "open.greet", "Hello there.", Q), "b": item("b", "open.greet", "hello there", G),
               "c": item("c", "open.greet", "Hi.", M)}
        WS.prejudge_dedup(its)
        self.assertEqual((its["a"]["status"], its["b"]["drop"], its["c"]["status"]),
                         ("kept", "DUP_EXACT_PREJUDGE", "kept"))

    def test_judge_blocks_and_vote_states(self):
        qs = [["act", "q?", "yes", "yn"]]
        recs = [{"kind": "judge", "call_id": f"j{m}", "author": store.teacher_author(m), "prompt_sha256": "p",
                 "raw_sha256": "r", "raw": raw, "plan": {"item_id": "x", "qs": qs, "fill": {}}}
                for m, raw in ((M, "act: yes\nEND"), (G, "act: no\nEND"))]
        jb = WS.judge_blocks(recs)
        self.assertEqual([b["verdict"] for b in jb["x"]], ["keep", "drop"])
        jb2 = WS.judge_blocks(recs[:1] + recs[:1])
        self.assertEqual(len(jb2["x"]), 1)                          # a repeat keeps the first


class TestTopics(unittest.TestCase):
    def test_selection_dedups_and_caps_common(self):
        wl = wordload.fake()
        its, recs, calls = {}, [], []
        for t in plan.TEACHER_ORDER:
            calls.append({"call_id": f"topic.{t}", "kind": "topic", "teacher": t})
            recs.append({"call_id": f"topic.{t}", "kind": "topic", "author": store.teacher_author(t)})
        texts = ["a quiet garden window", "a quiet garden window box", "fixing a ladder", "a muddy river walk"]
        for i, txt in enumerate(texts):
            its[f"T{i}"] = item(f"T{i}", "topic", txt, Q, feats={"likelihood": "common"})
        old = dict(WD.TOPIC_QUOTA)
        WD.TOPIC_QUOTA.update({Q: 4, M: 1, G: 1})
        try:
            got = [it["id"] for it, _ in WD.select_topics(its, recs, wl, calls)]
        finally:
            WD.TOPIC_QUOTA.update(old)
        self.assertNotIn("T1", got)                                 # Jaccard 0.75 with T0
        self.assertEqual(got[:2], ["T0", "T2"])                     # common capped at half, then refilled
        self.assertEqual(len(got), 3)

    def test_wordset_words_are_filtered(self):
        rec = {"plan": {"topic_id": "T1", "topic": "a walk", "teacher": Q}, "prompt_id": "p", "prompt_sha256": "p",
               "call_id": "w", "decode_sha256": "d", "seed": 1, "raw_sha256": "r", "time": 0, "preset": "shared",
               "author": store.teacher_author(Q),
               "lines": ["nouns: path, path, bench, two3, dog park", "verbs: walk", "adjectives: brisk"]}
        it = WI.wordset_items(rec)[0]
        self.assertEqual(it["features"]["nouns"], ["path", "bench"])
        self.assertEqual(sorted(w for w, _ in it["features"]["dropped_words"]), ["dog park", "two3"])


class TestMoreRules(unittest.TestCase):
    def test_value_match_is_whole_value(self):
        self.assertFalse(J3.match("Bis", "Biscuit", "free"))

    def test_done_ids_give_up_after_two_errors(self):
        recs = [{"call_id": "a", "error": "x"}, {"call_id": "b", "error": "x"}, {"call_id": "b", "error": "y"},
                {"call_id": "c"}]
        self.assertEqual(WS.done_ids(recs), {"b", "c"})

    def test_group_and_check_go_to_the_next_teacher(self):
        it = item("T0", "topic", "a garden walk", Q)
        self.assertEqual([c["teacher"] for t in plan.TEACHER_ORDER for c in WJ.group_calls(t, [(it, Q)], set())], [M])
        p = item("Q.instr.para.0", "instr.para", "x", G, feats={"head": "h", "tail": "t"})
        self.assertEqual([c["teacher"] for t in plan.TEACHER_ORDER for c in WJ.check_calls(t, {"p": p}, set())], [Q])

    def _sel(self, rows, quota, done=True):
        wl, its, recs, calls = wordload.fake(), {}, [], []
        for t in plan.TEACHER_ORDER:
            calls.append({"call_id": f"topic.{t}", "kind": "topic", "teacher": t})
            if done:
                recs.append({"call_id": f"topic.{t}", "kind": "topic", "author": store.teacher_author(t)})
        for i, (txt, like) in enumerate(rows):
            its[f"T{i}"] = item(f"T{i}", "topic", txt, Q, feats={"likelihood": like})
        old = dict(WD.TOPIC_QUOTA)
        WD.TOPIC_QUOTA.update({Q: quota})
        try:
            return [it["id"] for it, _ in WD.select_topics(its, recs, wl, calls)]
        finally:
            WD.TOPIC_QUOTA.update(old)

    def test_common_topics_fill_at_most_half_first(self):
        rows = [("fixing a ladder", "common"), ("a muddy river walk", "common"), ("baking bread", "rare")]
        self.assertEqual(self._sel(rows, 2), ["T0", "T2"])

    def test_selection_waits_for_every_topic_call(self):
        self.assertEqual(self._sel([("fixing a ladder", "rare")], 2, done=False), [])

    def test_query_value_runs_inside_line_items(self):
        c = {"call_id": "key.plan_day.query.0", "bank": "key.plan_day.query", "class": "K", "kind": "line",
             "teacher": Q, "n": 1, "form": "full", "fills": [{"v": "Friday", "old": "Monday", "o": "book club"}],
             "mined": None}
        rec = {"call_id": c["call_id"], "kind": "line", "bank": c["bank"], "class": "K", "author": store.teacher_author(Q),
               "plan": c, "prompt": "p", "prompt_id": "p", "prompt_sha256": "p", "decode_sha256": "d", "seed": 1,
               "raw_sha256": "r", "time": 0, "lines": ["Is my book club still on Monday?"], "likelihoods": [None],
               "problem": None}
        self.assertEqual(WI.items_for(rec)[0]["drop"], "QUERY_VALUE")

    def test_run_calls_skips_done_ids(self):
        import tempfile
        import _w3fake as FK
        c = {"call_id": "listname.city.q", "kind": "listname", "bank": "listname.city", "class": "N", "teacher": Q,
             "n": 2, "vtype": "city", "seed": 1}
        with tempfile.TemporaryDirectory() as d:
            path = d + "/q.jsonl"
            store.append_jsonl(path, {"call_id": c["call_id"], "kind": "listname"})
            fake = FK.Fake(Q)
            hold.run_calls(fake, [c], path, 1e12, 2, {"by": "Max", "date": "x"}, "t", lambda m: None, set(), {})
            self.assertEqual(fake.n, 0)

    def test_every_bank_renders_without_a_hole(self):
        tgt = {b: 10 for b in specs.line_specs()}
        for c in w3plan.line_calls(tgt, w3fill.Inputs()):
            if c["call_id"].endswith(".0"):
                self.assertNotIn("{", WR.render(c)[0], c["bank"])


class TestFreshFill(unittest.TestCase):
    def test_fresh_fill_moves_off_the_writing_value(self):
        it = {"id": "K.key.pet_name.plant.7_1", "bank": "key.pet_name.plant", "seeds": {"fill": {}}}
        first = J3.fresh_fill(it, M)["v"]
        it["seeds"]["fill"] = {"v": first}
        self.assertNotEqual(J3.fresh_fill(it, M)["v"], first)


if __name__ == "__main__":
    unittest.main()
