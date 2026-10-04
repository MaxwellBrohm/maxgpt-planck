"""Bank pass W0: store, specs, templatize and the item / held-out gates (pipeline/bankpass/). No model is loaded or
called; every line here is FAKE fixture text. Each planted case must fire its code and each clean case must pass."""
import os
import tempfile
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import heldout
from bankpass import gates, specs, store, templatize as T


class TestStore(unittest.TestCase):
    def test_straight_and_key(self):
        self.assertEqual(store.straight("I’m  “here” "), 'I\'m "here"')
        self.assertEqual(store.norm_key("My {o} is {v}!"), store.norm_key("my {x} IS {y}"))

    def test_author_problems_by_kind(self):
        self.assertEqual(store.author_problems(store.teacher_author("qwen3.5-9b")), [])
        bad = dict(store.teacher_author("qwen3.5-9b"), revision="0" * 40)
        self.assertIn("teacher revision", store.author_problems(bad))
        self.assertEqual(store.author_problems({"kind": "teacher", "model": "gpt-x"}), ["teacher not pinned"])
        human = {"kind": "human", "source": {"url": "u", "license": "cc-by-4.0", "retrieved": "d", "file_sha256": "h",
                                             "row": 3}}
        self.assertEqual(store.author_problems(human), [])
        human["source"]["license"] = "cc-by-nc-4.0"
        self.assertIn("human license", store.author_problems(human))
        self.assertTrue(store.author_problems({"kind": "computed"}))

    def test_jsonl_torn_tail_and_done_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.dry.jsonl")
            store.append_jsonl(p, {"call_id": "a"})
            store.append_jsonl(p, {"call_id": "b"})
            with open(p, "a") as f:
                f.write('{"call_id": "c"')            # a torn write
            self.assertEqual(store.done_ids(p), {"a", "b"})
            with self.assertRaises(ValueError):
                store.read_jsonl(p, tolerate_torn_tail=False)

    def test_mix_and_ref(self):
        sh = {"gemma-4-12b": 0.34, "ministral-3-8b": 0.33, "qwen3.5-9b": 0.33}
        self.assertEqual(store.mix_string(sh), "g34m33q33")
        self.assertEqual(store.mix_string({"fake": 1.0}), "FAKE")
        self.assertTrue(store.bank_ref("open.greet", "ab" * 32, sh).startswith("open.greet@abababababab:g34"))


class TestSpecs(unittest.TestCase):
    def test_hole_sets_from_fake_banks(self):
        sp = specs.line_specs()
        self.assertEqual(sp["key.pet_name.corr"]["holes_required"], ["M", "v"])
        self.assertIn("p", sp["key.pet_name.corr"]["holes_allowed"])
        self.assertEqual(sp["open.topic"]["holes_required"], ["t"])
        self.assertEqual(specs.hole_problems("open.topic", "Hi, about {t} and {v}."), ["stray {v}"])
        self.assertEqual(specs.hole_problems("open.topic", "Hi there."), ["missing {t}"])
        self.assertIn("repeated {t}", specs.hole_problems("open.topic", "{t}, {t}."))
        self.assertIn("unbalanced braces", specs.hole_problems("open.topic", "Hi {t} }"))


class TestTemplatize(unittest.TestCase):
    def test_plant_article_and_value(self):
        self.assertEqual(T.templatize("I work as a nurse these days.", {"v": "nurse", "article": "a"}),
                         ("I work as {av} these days.", None))
        self.assertEqual(T.templatize("My dog is called Biscuit.", {"v": "Biscuit", "o": "dog"}),
                         ("My {o} is called {v}.", None))

    def test_missing_and_repeated_drop(self):
        self.assertEqual(T.templatize("My dog has a name.", {"v": "Biscuit", "o": "dog"})[1], T.DROP_MISSING)
        self.assertEqual(T.templatize("Biscuit, yes, Biscuit is my dog.", {"v": "Biscuit", "o": "dog"})[1],
                         T.DROP_REPEATED)
        self.assertEqual(T.templatize("Maybe in May.", {"v": "May"})[0], "Maybe in {v}.")   # case-exact value
        self.assertEqual(T.templatize("I may come in May.", {"v": "May"})[0], "I may come in {v}.")

    def test_correction_forms(self):
        f = {"v": "Biscuit", "old": "Rex", "o": "dog"}
        self.assertEqual(T.templatize("My dog is Biscuit, not Rex.", f, form="full", correction=True)[0],
                         "{M}my {o} is {v}, not {old}.")
        self.assertEqual(T.templatize("The dog is Biscuit, not Rex.", f, form="full", correction=True)[1],
                         T.DROP_FORM)
        self.assertEqual(T.templatize("She's called Biscuit now.", dict(f, p="she"), form="pronoun",
                                      correction=True)[0], "{M}{p}'s called {v} now.")
        self.assertEqual(T.templatize("The dog is Biscuit now.", f, form="head", correction=True)[0],
                         "{M}the {o} is {v} now.")
        self.assertEqual(T.templatize("My dog is Biscuit now.", f, form="head", correction=True)[1], T.DROP_FORM)
        self.assertEqual(T.templatize("Biscuit, not Rex.", f, form="ellipsis", correction=True)[0],
                         "{M}{v}, not {old}.")
        self.assertEqual(T.templatize("I meant Biscuit, not Rex.", f, form="ellipsis", correction=True)[0],
                         "{M}I meant {v}, not {old}.")

    def test_items_join_forms(self):
        items = ["eggs", "milk", "bread"]
        self.assertEqual(T.templatize("I need eggs, milk, and bread.", {"items": items})[0], "I need {items}.")
        self.assertEqual(T.templatize("I need eggs, milk and bread.", {"items": items})[0], "I need {items}.")
        self.assertEqual(T.templatize("I need eggs plus milk and bread.", {"items": items})[1], T.DROP_JOIN)


class TestItemChecks(unittest.TestCase):
    def codes(self, text, bank, **kw):
        return [c for c, _ in gates.item_checks(text, bank, **kw)]

    def test_clean_lines_pass(self):
        self.assertEqual(self.codes("My {o} is called {v}.", "key.pet_name.plant"), [])
        self.assertEqual(self.codes("What did we name the {o}?", "key.pet_name.query"), [])
        self.assertEqual(self.codes("Ah, I mean,", "marker.fix"), [])

    def test_planted_lines_fire(self):
        cases = [("My {o} \u2014 {v}.", "key.pet_name.plant", "DASH"),
                 ("My {o} is called {v} 2.", "key.pet_name.plant", "DIGIT"),
                 ("My **{o}** is {v}.", "key.pet_name.plant", "MARKDOWN"),
                 ("Is my {o} called {v}?", "key.pet_name.plant", "SPEECH_FORM"),
                 ("What did we name the {o}.", "key.pet_name.query", "SPEECH_FORM"),
                 ("Is your {o} called {v}, I said?", "key.pet_name.query", "PERSPECTIVE"),
                 ("Hi, happy to help with {t}.", "open.topic", "AI_ISM"),
                 ("The user says hi about {t}.", "open.topic", "USER_VOICE"),
                 ("Hello END.", "open.greet", "END_IN_TURN"),
                 ("Bonjour, ça va?", "open.greet", "NON_ENGLISH"),
                 ("Ah well you see what I really mean,", "marker.fix", "LEN_ITEM"),
                 ("Actually", "marker.fix", "MARKER_FORM"),
                 ("My {o} is called {v} and {t}.", "key.pet_name.plant", "HOLE"),
                 ("User: hi there.", "open.greet", "ROLE_LABEL")]
        for text, bank, code in cases:
            with self.subTest(text=text):
                self.assertIn(code, self.codes(text, bank))

    def test_prompt_and_mined_echo(self):
        prompt = "Write ten different things a person might say to a chat assistant at any point"
        self.assertIn("BANK_PROMPT_ECHO", self.codes("Things a person might say to a chat, hi.", "open.greet",
                                                     prompt=prompt))
        self.assertNotIn("BANK_PROMPT_ECHO", self.codes("Hello there, friend.", "open.greet", prompt=prompt))
        mined = ["well hello there my good old friend how are you"]
        self.assertIn("BANK_PROMPT_ECHO", self.codes("Hello there my good old friend!", "open.greet", mined=mined))


class TestHeldoutGates(unittest.TestCase):
    def test_clean_template_and_fills_pass(self):
        self.assertEqual(gates.heldout_checks("My {o} is called {v}.", "key.pet_name.plant", n_fills=5), [])

    def test_planted_vocab_and_echo_fire(self):
        term = heldout.MARKER_TERMS[0]
        self.assertEqual(gates.heldout_checks(f"{term.capitalize()}, my {{o}} is {{v}}.", "key.pet_name.corr"
                                              .replace("corr", "plant"), n_fills=1)[0][0], "HELDOUT_VOCAB")
        g5 = " ".join(sorted(heldout.echo_sets()[0])[0])
        self.assertEqual(gates.heldout_checks(g5 + " {t}.", "open.topic", n_fills=1)[0][0], "HELDOUT_ECHO")

    def test_template_check_alone_and_fill_check_alone(self):
        term = heldout.MARKER_TERMS[0]
        self.assertEqual(gates.heldout_checks(f"{term.capitalize()}, hi.", "open.greet", n_fills=0)[0][0],
                         "HELDOUT_VOCAB")
        real = gates.sample_fill
        gates.sample_fill = lambda bank, spec, rng: dict(real(bank, spec, rng), v="Friday")
        try:
            hits = gates.heldout_checks("So we agreed on {v} for the big {o} then.", "key.plan_day.plant",
                                        n_fills=1)
        finally:
            gates.sample_fill = real
        self.assertEqual(hits, [("HELDOUT_ECHO", "fill 5gram:we agreed on friday for")])
        self.assertEqual(gates.heldout_checks("So we agreed on {v} for the big {o} then.", "key.plan_day.plant",
                                              n_fills=0), [])

    def test_pool_checks(self):
        self.assertEqual(gates.pool_checks("knitting", "hobby"), [])
        self.assertEqual(gates.pool_checks(heldout.HELDOUT_PHRASES[0], "hobby")[0][0], "HELDOUT_POOL")
        self.assertIn("LEN_ITEM", [c for c, _ in gates.pool_checks("a b c d e", "food")])

    def test_pair_echo_marks_and_drops(self):
        q, s, s2 = "When is the school {o} happening again?", "The school {o} happening again is on {v}.", \
            "My {o} is on {v}."
        bad, drop = gates.pair_echoes([q], [s])
        self.assertEqual((bad, drop), ({(q, s)}, {q, s}))
        bad, drop = gates.pair_echoes([q], [s, s2])          # q echoes half its partners: kept, pair marked
        self.assertEqual((bad, drop), ({(q, s)}, {s}))
        self.assertEqual(gates.pair_echoes(["Which colour is the {o} now?"], ["The {o} is {v}."]), (set(), set()))


if __name__ == "__main__":
    unittest.main()
