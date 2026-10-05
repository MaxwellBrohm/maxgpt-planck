"""Bank pass W3 (stage P machinery): render, regex, items, judges, derived calls and the hold loop, end to end with a
FAKE teacher (tests/_w3fake.py). No model is loaded or called; every output line is FAKE test text."""
import json
import os
import tempfile
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import _w3fake as FK
from bankpass import (hold, judge3 as J3, plan, store, w3deps as WD, w3fill, w3judge as WJ, w3plan, w3render as WR,
                      w3state as WS, wordload)

MAX_OK = {"by": "Max", "date": "2026-10-04", "ref": "test"}
BANKS = {"key.pet_name.plant": 10, "key.pet_name.query": 10, "key.job.corr": 10, "key.person_name.twin": 10,
         "open.greet": 10, "open.topic": 10, "marker.fix": 10, "close.goodbye": 10, "social.thanks": 10,
         "rule.max_words": 10, "rule.override": 10, "list.init.grocery": 10, "list.add": 10, "list.q.ordinal": 10,
         "lookup.q.weekday": 10, "lookup.cf": 10, "swap.q": 10, "system.plain": 10, "identity.q.user": 10}


def mini_plan():
    old = (dict(plan.POOL_TARGETS), plan.TOPICS_STAGE_P, w3plan.PARA_CALLS)
    plan.POOL_TARGETS.clear()
    plan.POOL_TARGETS.update({"hobby": 40, "relation": 20, "entity_kind": 20, "object": 20})
    plan.TOPICS_STAGE_P, w3plan.PARA_CALLS = 40, 6
    try:
        wl, inputs = wordload.fake(), w3fill.Inputs()
        calls = (w3plan.line_calls(BANKS, inputs) + w3plan.value_calls(inputs, wl.seed_nouns())
                 + w3plan.label_calls() + w3plan.word_calls(wl) + w3plan.para_calls())
    finally:
        plan.POOL_TARGETS.clear()
        plan.POOL_TARGETS.update(old[0])
        plan.TOPICS_STAGE_P, w3plan.PARA_CALLS = old[1], old[2]
    return {"calls": calls, "targets": BANKS, "sha256": "test"}, wl


def run_stage(root, teacher, p, wl, fake=None, rounds=12):
    fake = fake or FK.Fake(teacher)
    bad, seen = set(), {}
    for _ in range(rounds):
        b, d, j, _ = hold.owed(root, teacher, p, wl)
        todo = [c for c in b + d + j if c["kind"] not in bad]
        if not todo:
            return bad, 0
        n, gone = hold.run_calls(fake, todo, WS.calls_path(root, teacher), 1e12, 8, MAX_OK, "test", lambda m: None,
                                 bad, seen)
        if gone:
            return bad, -1
    return bad, len(todo)


class TestRender(unittest.TestCase):
    def test_every_kind_renders_and_the_fake_fits_its_regex(self):
        p, wl = mini_plan()
        kinds = {}
        for c in p["calls"]:
            kinds.setdefault(c["kind"], c)
        for k, c in kinds.items():
            prompt, spec, mode = WR.render(c)
            text = FK.answer(c)
            self.assertTrue(FK.matches(spec, text), f"{k}: {text[:120]!r} vs {spec['regex'][:200]}")
            lines, _, prob = WR.parse(c, text, mode, spec)
            self.assertIsNone(prob, k)
            self.assertNotIn("{", prompt.replace("{o}", ""), k)       # no hole ever reaches a teacher

    def test_no_claude_example_line_in_line_prompts(self):
        p, _ = mini_plan()
        for c in p["calls"]:
            if c["kind"] == "line":
                prompt = WR.render(c)[0]
                self.assertNotIn("for example", prompt.lower())
                self.assertNotIn("e.g.", prompt.lower())


class TestJudge3(unittest.TestCase):
    def test_lenient_value_match(self):
        self.assertTrue(J3.match("A nurse.", "nurse", "free"))
        self.assertTrue(J3.match("my Biscuit", "Biscuit", "free"))
        self.assertFalse(J3.match("Rex", "Biscuit", "free"))
        self.assertTrue(J3.match("Not mentioned", J3.NOT_SAID, "free"))
        self.assertFalse(J3.match("Biscuit", J3.NOT_SAID, "free"))
        self.assertTrue(J3.match("milk, eggs and the bread", ["milk", "eggs", "bread"], "items"))
        self.assertFalse(J3.match("eggs, milk, bread", ["milk", "eggs", "bread"], "items"))
        self.assertTrue(J3.match("salt and pepper, milk", ["salt and pepper", "milk"], "items"))

    def test_fresh_fill_is_not_the_writing_fill(self):
        it = {"id": "K.key.pet_name.plant.0_0", "bank": "key.pet_name.plant",
              "seeds": {"fill": {"v": "Biscuit", "o": "dog"}}}
        for judge in plan.TEACHER_ORDER:
            f = J3.fresh_fill(it, judge)
            self.assertNotEqual(f["v"], "Biscuit")
            self.assertEqual(f, J3.fresh_fill(it, judge))             # seeded


class TestHoldLoop(unittest.TestCase):
    def test_stages_end_to_end(self):
        p, wl = mini_plan()
        with tempfile.TemporaryDirectory() as root:
            for t in plan.TEACHER_ORDER + plan.TEACHER_ORDER[:2]:
                bad, left = run_stage(root, t, p, wl)
                self.assertEqual((bad, left), (set(), 0), t)
                WS.refresh(root)
            for t in plan.TEACHER_ORDER:
                b, d, j, _ = hold.owed(root, t, p, wl)
                self.assertEqual((len(b), len(d), len(j)), (0, 0, 0), t)
            recs = WS.records(root)
            its = WS.view(root, recs)
            judged = [i for i in its.values() if i["status"] == "kept" and WS.needs_judges(i)]
            self.assertTrue(judged)
            for it in judged:
                models = sorted(j["model"] for j in it["judges"])
                self.assertEqual(len(models), 2, it["id"])
                self.assertNotIn(it["author"]["model"], models)
            from bankpass import w3amend
            chosen = WD.select_topics(its, recs, wl, w3amend.base_calls(p))
            self.assertTrue(chosen)
            ids = {r["call_id"] for r in recs}
            for it, author in chosen:
                for t in plan.TEACHER_ORDER:
                    self.assertIn(f"wordset.{store.SHORT[t]}.{it['id']}", ids)
                self.assertEqual(sum(WD.intent_k(it["id"], t) for t in plan.TEACHER_ORDER), 8)
            self.assertEqual({a for _, a in chosen}, set(plan.TEACHER_ORDER))
            tops = [i for i in its.values() if i["bank"] == "open.topic_spec"]
            self.assertEqual({i["author"]["model"] for i in tops}, set(plan.TEACHER_ORDER))
            tauth = {it["id"]: a for it, a in chosen}
            for i in tops:
                self.assertEqual(i["author"]["model"], tauth[i["features"]["topic_id"]])
            self.assertTrue(any(r["kind"] == "vote" for r in recs))
            self.assertTrue(any(r["kind"] == "pred" for r in recs))
            for r in recs:
                self.assertEqual(r.get("max_ok"), MAX_OK)

    def test_restart_and_server_gone(self):
        p, wl = mini_plan()
        with tempfile.TemporaryDirectory() as root:
            t = plan.TEACHER_ORDER[0]
            bad, left = run_stage(root, t, p, wl, fake=FK.Fake(t, gone_after=25))
            self.assertEqual(left, -1)
            n1 = len(WS.records(root, t))
            self.assertGreater(n1, 0)
            run_stage(root, t, p, wl)
            ids = [r["call_id"] for r in WS.records(root, t)]
            self.assertEqual(len(ids), len(set(ids)))                   # no call ran twice

    def test_bad_kind_brake(self):
        p, wl = mini_plan()
        with tempfile.TemporaryDirectory() as root:
            t = plan.TEACHER_ORDER[0]
            bad, _ = run_stage(root, t, p, wl, fake=FK.Fake(t, bad_kinds={"pos", "line"}), rounds=2)
            self.assertEqual(bad, {"line"})                             # pos has only 2 calls: under the floor
            n_line = sum(r["kind"] == "line" for r in WS.records(root, t))
            self.assertLess(n_line, hold.SMOKE_N + 8)

    def test_retries_never_trip_the_brake(self):
        p, wl = mini_plan()
        with tempfile.TemporaryDirectory() as root:
            t = plan.TEACHER_ORDER[0]
            calls = [dict(c, retry_of=c["call_id"], call_id=c["call_id"] + "r1") for c in p["calls"]
                     if c["kind"] == "line" and c["teacher"] == t][:10]
            bad = set()
            hold.run_calls(FK.Fake(t, bad_kinds={"line"}), calls, WS.calls_path(root, t), 1e12, 4, MAX_OK, "x",
                           lambda m: None, bad, {})
            self.assertEqual(bad, set())


class TestQueryValue(unittest.TestCase):
    def test_query_holding_a_value_is_caught_case_rules(self):
        from bankpass import w3items as WI
        self.assertTrue(WI.query_value("Is my {o} on Monday?", "plan_day"))
        self.assertFalse(WI.query_value("Which day is my {o} again?", "plan_day"))
        self.assertTrue(WI.query_value("Is my {o} in May?", "plan_month"))
        self.assertFalse(WI.query_value("may I ask which month my {o} is in?", "plan_month"))
        self.assertTrue(WI.query_value("Is it still blue?", "fav_colour"))


if __name__ == "__main__":
    unittest.main()
