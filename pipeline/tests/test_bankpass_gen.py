"""Bank pass W0: plan, prompts, gen (dry guard, restart, hold cap), items, judge and trim. No model is loaded or
called: the client is fixtures.FakeBankTeacher, and every output line is FAKE fixture text."""
import math
import os
import re
import tempfile
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
from bankpass import fixtures as FX, gen, itemize, judge, plan, prompts, store, trim


def small_plan():
    tgt = {"key.pet_name.plant": 20, "key.pet_name.query": 20, "open.greet": 30, "marker.fix": 20}
    return plan.line_calls(tgt, n_per=10)


class TestPlan(unittest.TestCase):
    def test_rates_targets_and_rotation(self):
        rates = plan.measure_rates(60)
        self.assertEqual(rates, plan.measure_rates(60))                     # deterministic
        tgt = plan.targets(rates)
        self.assertTrue(all(v >= 30 for b, v in tgt.items() if not b.startswith("marker.")))
        self.assertEqual((tgt["marker.fix"], tgt["marker.err"]), (60, 20))
        per_k = 1000 / rates["n"]
        self.assertEqual(tgt["open.greet"], max(30, math.ceil(rates["lines"]["open.greet"] * per_k * 2.5)))
        calls = plan.line_calls(tgt)
        self.assertEqual(calls, plan.line_calls(tgt))
        by = {}
        for c in calls:
            by.setdefault(c["bank"], []).append(c["teacher"])
        for bank, ts in by.items():
            n = [ts.count(t) for t in plan.TEACHER_ORDER]
            self.assertLessEqual(max(n) - min(n), 1, bank)

    def test_ready_and_hole_fills(self):
        calls = plan.line_calls({"open.topic": 20, "system.plain": 10, "rule.max_words": 10, "open.greet": 10})
        by = {c["bank"]: c for c in calls}
        self.assertTrue(all("t" in f for f in by["open.topic"]["fills"]))
        self.assertTrue(all("A" in f for f in by["system.plain"]["fills"]))
        self.assertEqual((by["rule.max_words"]["ready"], by["open.greet"]["ready"]), (False, True))
        with tempfile.TemporaryDirectory() as d:
            st = gen.run(calls, lambda t: FX.FakeBankTeacher(t), os.path.join(d, "dry", "c.dry.jsonl"))
        self.assertEqual((st["not_ready"], st["done"]), (1, len(calls) - 1))

    def test_pronoun_fills_are_gendered(self):
        calls = plan.line_calls({"key.person_name.corr": 40})
        for c in calls:
            if c["form"] == "pronoun":
                self.assertTrue(all(f["p"] in ("he", "she") for f in c["fills"]))


class TestPromptsDecode(unittest.TestCase):
    def test_no_example_line_and_literal_regex(self):
        c = small_plan()[0]
        p = prompts.prompt_for(c)
        self.assertNotIn("For example", p)
        self.assertIn(c["fills"][0]["v"], p)
        spec = prompts.decode_spec("qwen3.5-9b", ["Biscuit", "Rex"])
        rx = re.compile(spec["regex"])
        self.assertTrue(rx.fullmatch("My dog is Biscuit.\nWe call him Rex.\nEND"))
        self.assertFalse(rx.fullmatch("My dog is Max.\nWe call him Rex.\nEND"))      # literal missing
        self.assertFalse(rx.fullmatch("My dog is Biscuit END.\nWe call him Rex.\nEND"))
        verb = re.compile(prompts.decode_spec("gemma-4-12b", [None], verbalized=True)["regex"])
        self.assertTrue(verb.fullmatch("Ah, I mean, | rare\nEND"))
        self.assertFalse(verb.fullmatch("Ah, I mean,\nEND"))

    def test_parse_lines(self):
        self.assertEqual(prompts.parse_lines("a\n\nb\nEND", 2), (["a", "b"], [None, None], None))
        self.assertEqual(prompts.parse_lines("a\nb", 2)[2], "NO_END")
        self.assertEqual(prompts.parse_lines("a\nEND", 2)[2], "LINES_1_OF_2")
        self.assertEqual(prompts.parse_lines("a | rare\nEND", 1, True), (["a"], ["rare"], None))
        self.assertEqual(prompts.parse_lines("a | often\nEND", 1, True)[2], "NO_LIKELIHOOD")


class TestGen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "dry", "calls.dry.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_guards(self):
        for path, mode, ok in [("dry/x.dry.jsonl", "dry", None), ("x.dry.jsonl", "dry", None),
                               ("dry/x.jsonl", "dry", None), ("real/x.jsonl", "real", None),
                               ("real/x.jsonl", "real", {"by": "Claude", "date": "2026-10-03"}),
                               ("planck/data/dry/x.dry.jsonl", "dry", None), ("sealed/dry/x.dry.jsonl", "dry", None),
                               ("dry/x.dry.jsonl", "wet", None)]:
            if path == "dry/x.dry.jsonl" and mode == "dry":
                gen.check_out_path(os.path.join(self.tmp.name, path), mode, ok)
                continue
            with self.subTest(path=path, mode=mode), self.assertRaises(gen.Refused):
                gen.check_out_path(os.path.join(self.tmp.name, path), mode, ok)
        gen.check_out_path(os.path.join(self.tmp.name, "real/x.jsonl"), "real", {"by": "Max", "date": "2026-10-04"})

    def test_restart_skips_done_and_hold_cap(self):
        calls = small_plan()
        clients = {t: FX.FakeBankTeacher(t) for t in store.TEACHERS}
        tick = iter(range(0, 10 ** 6, 600))                     # 10 minutes per clock read
        st = gen.run(calls, clients.get, self.path, clock=lambda: next(tick))
        self.assertEqual(st["stopped"], "HOLD_CAP")
        first = st["done"]
        self.assertTrue(0 < first < len(calls))
        st2 = gen.run(calls, clients.get, self.path)
        self.assertEqual((st2["skipped"], st2["done"]), (first, len(calls) - first))
        recs = store.read_jsonl(self.path)
        self.assertEqual(len({r["call_id"] for r in recs}), len(calls))
        self.assertTrue(all(r["dry"] and r["mode"] == "dry" and r["problem"] is None for r in recs))

    def test_model_mismatch_and_bad_output(self):
        calls = small_plan()[:1]
        with self.assertRaises(RuntimeError):
            gen.run(calls, lambda t: FX.FakeBankTeacher(t, wrong_model=True), self.path)
        st = gen.run(calls, lambda t: FX.FakeBankTeacher(t, bad=True), self.path)
        self.assertEqual(st["problems"], 1)

    def test_items_from_calls(self):
        calls = [c for c in small_plan() if c["bank"] == "key.pet_name.plant"][:1]
        gen.run(calls, lambda t: FX.FakeBankTeacher(t), self.path)
        rec = store.read_jsonl(self.path)[0]
        its = itemize.items_for(rec, calls[0], n_fills=2)
        self.assertEqual(len(its), 10)
        kept = [r for r in its if r["status"] == "kept"]
        self.assertTrue(kept and all(r["text"] == "My {o} is called {v}." for r in kept))
        self.assertTrue(all(r["gates"]["gate_hash"] for r in kept))


class TestJudgeTrim(unittest.TestCase):
    def rec(self, text, model="qwen3.5-9b", bank="key.pet_name.plant", likelihood=None):
        return store.make_item("K", bank, len(text), text, store.teacher_author(model),
                               features={"likelihood": likelihood})

    def test_questions_parse_verdict(self):
        r = self.rec("My {o} is called {v}.")
        qs = judge.questions(r, {"v": "Biscuit"}, label="dog name")
        ans = judge.parse("act: Yes.\next: biscuit", qs)
        self.assertEqual(judge.verdict(ans, qs), "keep")
        self.assertEqual(judge.verdict(judge.parse("act: yes\next: not said", qs), qs), "drop")
        self.assertEqual(judge.verdict(judge.parse("act: yes\nact: no\next: biscuit", qs), qs), "drop")
        q = judge.questions(self.rec("What did we name the {o}?", bank="key.pet_name.query"), {}, label="dog name")
        self.assertEqual(dict((a, e) for a, _, e in q)["ext"], judge.NOT_SAID)

    def test_vote(self):
        r = self.rec("My {o} is called {v}.")
        self.assertIsNone(FX.judged(r))
        self.assertEqual(FX.judged(r, ("keep", "drop")), "JUDGE_DROP")
        r["judges"] = [{"model": "qwen3.5-9b", "verdict": "keep"}, {"model": "gemma-4-12b", "verdict": "keep"}]
        self.assertEqual(judge.vote(r), "JUDGE_MISSING")                      # the author judged itself
        r["judges"] = r["judges"][1:]
        self.assertEqual(judge.vote(r), "JUDGE_MISSING")

    def test_trim(self):
        rs = [self.rec("My {o} is called {v}.", "qwen3.5-9b"), self.rec("my {o} is called {v}", "gemma-4-12b"),
              self.rec("My {o} is called {v} now.", "ministral-3-8b"), self.rec("We named the {o} {v}.")]
        trim.dedup(rs)
        codes = sorted(r["drop"] or "kept" for r in rs)
        self.assertEqual(codes, ["DUP_EXACT", "DUP_NEAR", "kept", "kept"])
        self.assertEqual(trim.p_exact(50, 100), 0.35)
        self.assertEqual(trim.p_exact(500, 100), 0.7)
        many = [self.rec(f"My {{o}} is {w} {{v}}.", m) for w in ("called", "named", "known as", "just")
                for m in ("qwen3.5-9b",)] + [self.rec("I have a {o} named {v}.", "gemma-4-12b"),
                                             self.rec("Our {o} goes by {v}.", "ministral-3-8b")]
        kept = trim.author_thirds(many)
        self.assertEqual(sum(r["author"]["model"] == "qwen3.5-9b" for r in kept), 1)


if __name__ == "__main__":
    unittest.main()
