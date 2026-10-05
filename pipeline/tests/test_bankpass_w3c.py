"""Bank pass W3 amendments 1 to 3 (2026-10-04, after Qwen's first hold): version 2 pool, POS and note-frame calls
replace version 1 one for one, superseded items cascade to their attributes, cut-off calls get one retry, the
correction-marker and topic gates. No model; FAKE test text only."""
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
from bankpass import plan, store, w3amend as AM, w3items as WI, w3state as WS

Q, M, G = plan.TEACHER_ORDER


def rec(call_id, kind, lines, plan_call, author=Q, **kw):
    r = {"call_id": call_id, "kind": kind, "bank": plan_call.get("bank", "x"), "class": plan_call.get("class", "K"),
         "author": store.teacher_author(author), "plan": plan_call, "prompt": "p", "prompt_id": "p",
         "prompt_sha256": "p", "decode_sha256": "d", "seed": 1, "raw_sha256": "r", "time": 0, "lines": lines,
         "likelihoods": [None] * len(lines), "problem": None}
    r.update(kw)
    return r


class TestAmendments(unittest.TestCase):
    def test_v2_replaces_v1_one_for_one(self):
        p = {"calls": [{"call_id": "pool.job.0", "kind": "pool", "teacher": Q, "seed_words": ["a"]},
                       {"call_id": "pos.q.0", "kind": "pos", "teacher": Q}, {"call_id": "notewas.q", "kind": "notewas",
                                                                              "teacher": Q},
                       {"call_id": "topic.0", "kind": "topic", "teacher": Q}]}
        ids = sorted(c["call_id"] for c in AM.base_calls(p))
        self.assertEqual(ids, ["notewas2.q", "pool2.job.0", "pos2.q.0", "topic2.0"])
        self.assertTrue(all(c.get("v") == 2 for c in AM.base_calls(p)))
        self.assertIsNone(next(c for c in AM.base_calls(p) if c["call_id"] == "pool2.job.0")["seed_words"])

    def test_superseded_calls_are_not_retried(self):
        r = rec("key.job.query.3v4", "line", [], {"call_id": "key.job.query.3v4", "bank": "key.job.query"},
                problem="NO_END", finish="length")
        r["bank"] = "key.job.query"
        self.assertEqual(AM.retries([r], Q), [])

    def test_retry_only_for_cut_off_calls_once(self):
        recs = [rec("a.1", "line", [], {"call_id": "a.1"}, problem="NO_END", finish="length"),
                rec("a.2", "line", [], {"call_id": "a.2"}, problem="LINES_9_OF_10", finish="stop"),
                rec("a.3", "line", [], {"call_id": "a.3"}, problem="NO_END", finish="length"),
                rec("a.3r1", "line", [], {"call_id": "a.3r1"}, problem="NO_END", finish="length"),
                rec("a.4", "judge", [], {"call_id": "a.4"}, problem="NO_END", finish="length"),
                rec("a.5", "line", [], {"call_id": "a.5"}, author=M, problem="NO_END", finish="length")]
        self.assertEqual([c["call_id"] for c in AM.retries(recs, Q)], ["a.1r1"])
        self.assertEqual(AM.retries([dict(recs[0], plan={"call_id": "a.1", "seed": 7})], Q)[0]["seed"], 8)

    def test_superseded_items_and_their_attributes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            c = {"call_id": "pool.entity_kind.0", "bank": "pool.entity_kind", "class": "P", "teacher": Q,
                 "seed_words": None}
            store.append_jsonl(WS.calls_path(root, Q), rec("pool.entity_kind.0", "pool", ["bakery"], c))
            a = {"call_id": "attr.P.pool.entity_kind.0_0", "bank": "attr", "class": "U", "teacher": Q, "n": 1,
                 "value": "bakery", "src_id": "P.pool.entity_kind.0_0"}
            store.append_jsonl(WS.calls_path(root, Q), rec(a["call_id"], "attr", ["opening day: day of the week"], a))
            WS.refresh(root)
            its = WS.view(root)
            self.assertEqual(its["P.pool.entity_kind.0_0"]["drop"], "SUPERSEDED_V1")
            self.assertEqual(its["U.attr.P.pool.entity_kind.0_0~0"]["drop"], "SUPERSEDED_V1")

    def test_correction_marker_and_topic_gates(self):
        c = {"call_id": "key.job.corr.0", "bank": "key.job.corr", "class": "K", "kind": "line", "teacher": Q, "n": 2,
             "form": "full", "fills": [{"v": "nurse", "old": "baker", "article": "a"}] * 2, "mined": None}
        out = WI.items_for(rec(c["call_id"], "line", ["Actually, my job is a nurse now.", "My job is a nurse now."], c))
        self.assertEqual([i["drop"] for i in out], ["CORR_MARKER", None])
        t = {"call_id": "topic.0", "bank": "topic", "class": "T", "teacher": Q, "seed_words": ["a", "b", "c"]}
        out = WI.items_for(rec("topic.0", "topic", ["planning a quiet garden party", "the cake my friend Bob baked",
                                                    "fixing my leaky garden hose", "na quiet garden plans"], t,
                               likelihoods=["rare"] * 4))
        self.assertEqual([i["drop"] for i in out], [None, "TOPIC_NAME", "TOPIC_PERSONAL", "TOPIC_FORM"])

    def test_v2_pool_prompt_has_no_seed_words(self):
        from bankpass import w3render as WR
        c = {"call_id": "pool2.job.0", "kind": "pool", "bank": "pool.job", "class": "P", "teacher": Q, "n": 20,
             "seed_words": None, "v": 2, "seed": 1}
        prompt = WR.render(c)[0]
        self.assertNotIn("nudge", prompt)
        self.assertIn("Do not put a, an, the or my in front", prompt)

    def test_bank_lines_refuse_digits_markdown_and_quotes(self):
        import re
        from bankpass import prompts as PR
        rx = re.compile(PR.line_regex())
        self.assertTrue(rx.fullmatch("Hi there, friend."))
        for bad in ("I have 2 dogs", "my **cat**", 'He said "hi"', "a_b"):
            self.assertIsNone(rx.fullmatch(bad), bad)

    def test_topics_v2_for_every_teacher(self):
        from bankpass import w3render as WR
        p = {"calls": [{"call_id": f"topic.{i}", "kind": "topic", "bank": "topic", "class": "T", "teacher": t, "n": 10,
                        "seed_words": ["a", "b", "c"], "seed": i} for i, t in enumerate(plan.TEACHER_ORDER)]}
        got = AM.base_calls(p)
        self.assertEqual(sorted(c["call_id"] for c in got), ["topic2.0", "topic2.1", "topic2.2"])
        prompt = WR.render(got[0])[0]
        self.assertIn("not a question, not a request", prompt)
        self.assertTrue(AM.superseded("topic.0"))
        t = {"call_id": "topic2.0", "bank": "topic", "class": "T", "teacher": Q, "seed_words": ["a", "b", "c"]}
        out = WI.items_for(rec("topic2.0", "topic", ["why is my oven so slow?"], t, likelihoods=["rare"]))
        self.assertIn(out[0]["drop"], ("TOPIC_PERSONAL", "TOPIC_FORM"))

    def test_paraphrase_part_labels_are_stripped(self):
        c = {"call_id": "para.3", "bank": "instr.para", "class": "Q", "teacher": Q}
        it = WI.items_for(rec("para.3", "para", ["Part one: Write a chat.", "Part two: End with END."], c))[0]
        self.assertEqual((it["features"]["head"], it["features"]["tail"]), ("Write a chat.", "End with END."))


class TestFixes(unittest.TestCase):
    def test_pronoun_templates_fill_and_job_corrections_hold_av(self):
        from bankpass import gates, specs
        for key in ("person_name", "person_city", "person_job"):
            bank = f"key.{key}.corr"
            tpl = "{M}{p} lives in {v} now." if key == "person_city" else "{M}{p} is {av} now." \
                if key == "person_job" else "{M}{p} is called {v} now."
            self.assertEqual(gates.heldout_checks(tpl, bank, specs.line_specs()[bank], n_fills=20), [])
        c = {"call_id": "key.person_job.corr.3", "bank": "key.person_job.corr", "class": "K", "kind": "line",
             "teacher": Q, "n": 1, "form": "pronoun", "mined": None,
             "fills": [{"v": "nurse", "old": "baker", "o": "sister", "p": "she", "article": "a"}]}
        it = WI.items_for(rec(c["call_id"], "line", ["She works as a nurse now."], c))[0]
        self.assertEqual((it["text"], it["status"]), ("{M}{p} works as {av} now.", "kept"))


class TestAmendment4(unittest.TestCase):
    def test_question_mark_and_literals_at_decoding(self):
        import re
        from bankpass import w3fill, w3plan, w3render as WR
        calls = w3plan.line_calls({"key.job.query": 10, "lookup.ctx": 10, "list.init.city": 10, "swap.q": 10},
                                  w3fill.Inputs())
        by = {c["bank"]: dict(c, call_id=c["call_id"] + "v4") for c in calls if c["teacher"] == Q}
        rx = re.compile(WR.render(by["key.job.query"])[1]["regex"])
        self.assertTrue(rx.fullmatch("\n".join(["What do I do for work?"] * by["key.job.query"]["n"] + ["END"])))
        self.assertFalse(rx.fullmatch("\n".join(["Tell me my job"] * by["key.job.query"]["n"] + ["END"])))
        for bank in ("lookup.ctx", "list.init.city", "swap.q"):
            spec = WR.render(by[bank])[1]
            self.assertEqual(len([x for x in spec["literals"] if x]), by[bank]["n"], bank)
        f = by["lookup.ctx"]["fills"][0]
        self.assertEqual(WR.render(by["lookup.ctx"])[1]["literals"][0], f"{f['e']} {f['pred']}")
        v6 = dict(by["key.job.query"], call_id=by["key.job.query"]["call_id"][:-2] + "v6")     # amendment 6
        self.assertTrue(re.compile(WR.render(v6)[1]["regex"]).fullmatch("\n".join(["Tell me my job"] * v6["n"]
                                                                                    + ["END"])))
        self.assertFalse(any(WR.render(dict(by["lookup.ctx"], call_id="lookup.ctx.0v6"))[1]["literals"]))

    def test_first_teacher_reruns_and_its_old_calls_are_superseded(self):
        p = {"calls": [{"call_id": "key.job.query.0", "kind": "line", "bank": "key.job.query", "teacher": Q},
                       {"call_id": "key.job.query.1", "kind": "line", "bank": "key.job.query", "teacher": M},
                       {"call_id": "key.job.plant.0", "kind": "line", "bank": "key.job.plant", "teacher": Q}]}
        ids = sorted(c["call_id"] for c in AM.base_calls(p))
        self.assertEqual(ids, ["key.job.plant.0", "key.job.query.0v6", "key.job.query.1v6"])
        mk = lambda cid, a, bank="key.job.query": {"call": {"call_id": cid}, "author": store.teacher_author(a),
                                                     "bank": bank}
        self.assertTrue(AM.superseded_item(mk("key.job.query.0", Q)))
        self.assertTrue(AM.superseded_item(mk("key.job.query.0r1", Q)))
        self.assertTrue(AM.superseded_item(mk("key.job.query.0v4", Q)))      # amendment 6 took v4 back out
        self.assertFalse(AM.superseded_item(mk("key.job.query.0v6", Q)))
        self.assertFalse(AM.superseded_item(mk("key.job.query.0v6r1", Q)))
        self.assertTrue(AM.superseded_item(mk("key.job.query.1", M)))
        self.assertFalse(AM.superseded_item(mk("key.job.plant.0", Q, "key.job.plant")))

    def test_twins_may_name_the_noun_and_swap_may_say_your(self):
        from bankpass import gates, specs
        sp = specs.line_specs()
        self.assertIn("o", sp["key.plan_day.twin"]["holes_allowed"])
        self.assertNotIn("o", sp["key.job.twin"]["holes_allowed"])
        self.assertFalse([h for h in gates.item_checks("And what is your {lab}?", "swap.q") if h[0] == "PERSPECTIVE"])

    def test_superseded_items_are_never_dedup_representatives(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            for cid, a in (("key.job.query.0", Q), ("key.job.query.1", M), ("key.job.query.1v6", M)):
                c = {"call_id": cid, "bank": "key.job.query", "class": "K", "kind": "line", "teacher": a, "n": 1,
                     "form": "full", "fills": [{"v": "nurse", "old": "baker"}], "mined": None}
                store.append_jsonl(WS.calls_path(root, a), rec(cid, "line", ["What is my job again?"], c, author=a))
            WS.refresh(root)
            its = WS.view(root)
            self.assertEqual(its["K.key.job.query.0_0"]["drop"], "SUPERSEDED_V1")
            self.assertEqual(its["K.key.job.query.1_0"]["drop"], "SUPERSEDED_V1")
            self.assertEqual(its["K.key.job.query.1v6_0"]["status"], "kept")     # not a duplicate of a superseded one


if __name__ == "__main__":
    unittest.main()
