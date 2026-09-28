"""Tests for the 2026-09-27 teacher pilot fixes (reject audit; D1 to D4 in notes.txt): the render prompt, the
guidance wording, the skeleton frame and required words, and checker units the mutation harness does not reach.
The checker's clean and planted pairs run in the mutation harness (fixtures_pilot.py, test_checker). No model is
loaded or called; every text here is FAKE fixture text."""
import re
import unittest

from _corpus import corpus, S  # noqa: F401  (puts the pipeline on sys.path)
import assemble as A
import banks as B
import check_base
import checker
import fake_teacher
import gate
import heldout
import parse
import pools as P
import records
import render_intents as RI
import render_prompt as R
import topic_words as TW
from check_base import Ctx, stem

N = 400


def _first(pred):
    return next(sk for sk in corpus(N) if pred(sk))


def _lower_exact(sk):
    return sk["user"]["style"] == "lowercase" and any(
        t["mode"] == "exact" and t["role"] == "user" and t["text"] != t["text"].lower() for t in sk["turns"])


class TestPrompt(unittest.TestCase):
    def test_lowercase_is_a_user_style_only(self):
        sk = _first(lambda s: s["user"]["style"] == "lowercase")
        b = R.build(sk)
        self.assertIn("this is the user's style only, the assistant writes normal sentence case", b["blocks"]["card"])
        self.assertIn("The assistant writes normal sentence case", b["blocks"]["tail"])

    def test_rules_and_example_come_after_the_script(self):
        p = R.build(corpus(N)[3])["prompt"]
        script, rules = p.index("Script:"), p.index("Rules: no dashes (use a comma or a new sentence)")
        self.assertLess(script, rules)
        self.assertLess(rules, p.index("Example, another script"))
        for rule in ("No digits (numbers as words)", "normal sentence case", "my, never your",
                     "never answers as your X is Y"):
            self.assertIn(rule, p[rules:])

    def test_example_is_well_formed_and_clean(self):
        head, out = R.EXAMPLE.split("\ngives\n")
        labels = [ln.split(" ", 1)[0] for ln in head.split("\n")[1:] if ln]
        self.assertEqual(labels, ["U1", "A2"])      # the global label scheme of parse.plan
        plan = [(lab, n, "user" if lab[0] == "U" else "assistant") for n, lab in enumerate(labels)]
        p = parse.parse(out, plan)
        self.assertEqual(p["codes"], [])
        self.assertIn(" my ", f" {p['turns'][0]} ")
        for n, text in p["turns"].items():
            self.assertFalse(gate.DASH_RE.search(text) or heldout.DIGIT_RE.search(text), text)
            self.assertEqual(heldout.vocab_hits(text), [], text)
            self.assertEqual(text[:1].islower(), plan[n][2] == "user", text)   # lowercase user, sentence-case assistant
        fake_words = {w for pl in P.POOLS.values() if pl.provenance == "FAKE" for v in pl.values for w in [v.lower()]}
        low = R.EXAMPLE.lower()
        self.assertEqual([w for w in fake_words if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low)], [])
        bank_lines = [ln for bank in B.BANKS.values() for ln in bank if "{" not in ln and len(ln.split()) > 2]
        self.assertEqual([ln for ln in bank_lines if ln.lower() in low], [])

    def test_card_never_prints_the_user_name(self):
        sk = _first(lambda s: s["user"]["name"])
        card = R.card_block(sk)
        self.assertNotIn(R.user_name(sk), card)
        self.assertIn("Nobody in the chat uses a name before the script line that gives it.", card)

    def test_exact_user_lines_follow_the_lowercase_style(self):
        sk = _first(_lower_exact)
        script = R.script_block(sk)
        for t in sk["turns"]:
            if t["mode"] == "exact":
                want = t["text"].lower() if t["role"] == "user" else t["text"]
                self.assertIn(f"[copy exactly]: {want}", script)
        sk2 = _first(lambda s: s["user"]["style"] != "lowercase" and any(t["mode"] == "exact" for t in s["turns"]))
        t = next(t for t in sk2["turns"] if t["mode"] == "exact")
        self.assertEqual(parse.exact_text(sk2, t), t["text"])

    def test_variant_ids_mark_the_new_prompt(self):
        self.assertEqual([v[0] for v in R.VARIANTS], ["instr.fake.p3.0", "instr.fake.p3.1"])   # p3: 2026-09-28


class TestGuidance(unittest.TestCase):
    def guided(self, pred):
        return [(sk, t) for sk in corpus(N) for t in sk["turns"] if t["mode"] == "guided" and pred(t)]

    def test_ask_back_is_first_person(self):
        g = [RI.guidance(sk, t) for sk, t in self.guided(lambda t: t["role"] == "user" and (t["intent"] or "")
                                                          .startswith("ask for ") and "never said" not in t["intent"])]
        self.assertTrue([x for x in g if x.startswith("ask the assistant to remind you ")])
        self.assertEqual([x for x in g if re.match(r"ask the assistant (?:your|the|which|what|where) ", x)], [])

    def test_list_guidance_never_says_your(self):
        g = [RI.guidance(sk, t) for sk, t in self.guided(lambda t: (t["intent"] or "").startswith(("list ", "ask list ")))]
        self.assertTrue(g)
        self.assertEqual([x for x in g if " your " in f" {x} "], [])

    def test_correction_names_the_new_value(self):
        pairs = self.guided(lambda t: t["role"] == "user" and (t["intent"] or "").startswith("correct "))
        self.assertTrue(pairs)
        for sk, t in pairs:
            self.assertIn(f"it is {t['must_include'][0]} now", RI.guidance(sk, t))

    def test_assistant_wording(self):
        for sk, t in self.guided(lambda t: t["role"] == "assistant"):
            g, base = RI.guidance(sk, t), t["intent"].split(";")[0]
            if base == "acknowledge briefly":
                self.assertTrue(g.startswith("acknowledge in one short sentence"))
                self.assertEqual("not repeating the value" in g, not t["must_include"])
            if base.startswith("answer"):
                self.assertIn(RI.ONLY_ANSWER, g)
                start = any(r["verifier"] == "start_name" for r in t.get("rules", []))
                self.assertEqual("starting with the answer itself" in g,   # 09-28 wording (test_round2_fixes)
                                 base == "answer with the value first" and not start)
            if base == "pick the first topic back up and name it":
                self.assertIn(t["must_include"][0], g)

    def test_rule_wording_is_clean(self):
        self.assertEqual(heldout.echo_hits(RI.RULE["end_question"]), [])


class TestSkeletonFrame(unittest.TestCase):
    def test_openings_and_closings_are_bank_lines(self):
        for sk in corpus(N):
            t0, last = sk["turns"][0], sk["turns"][-2] if sk["turns"][-1]["role"] == "assistant" else None
            if t0.get("role_in_event") == "opening":
                self.assertEqual(t0["mode"], "exact", sk["seed"])
                self.assertTrue(str(sk["opening"]).startswith("open."), sk["seed"])
                want = "greet the user back" if sk["opening"].startswith("open.greet") else \
                    "greet back and engage with the topic"
                self.assertEqual(sk["turns"][1]["intent"], want, sk["seed"])
            if last and last.get("role_in_event") == "closing":
                self.assertEqual(last["mode"], "exact", sk["seed"])

    def test_terse_topic_turns_have_room_for_the_topic(self):
        n = 0
        for sk in corpus(N):
            if sk["user"]["style"] != "terse":
                continue
            for t in sk["turns"]:
                if t["role"] == "user" and t["mode"] == "guided" and t["intent"].startswith("topic:"):
                    k = len(sk["topic_text"][t["intent"].split(":")[1]].split())
                    self.assertEqual(t["max_w"], min(A.REGISTERS[sk["register"]]["user"][1], max(10, k + 6)))
                    n += k + 6 > 10
        self.assertGreater(n, 0)

    def test_required_words_are_topic_fitted_and_placed_on_topic_turns(self):
        topical = 0
        for sk in corpus(N):
            rw = sk["required_words"]
            self.assertLessEqual(len(rw["turn_hint"]), 3)
            self.assertEqual(rw["turn_hint"], sorted(rw["turn_hint"]))
            for i, part in zip(rw["turn_hint"], ("noun", "verb", "adj")):
                t = sk["turns"][i]
                self.assertIn(t["intent"].split(";")[0], A.TOPIC_TURNS)
                self.assertGreaterEqual(t["max_w"], A.REQ_MIN_W)
                self.assertNotIn(rw[part], TW.NOT_REQUIRED)
                topical += rw[part] in TW.words(A._turn_topic(sk, t), part)
                self.assertIn(f'use the word "{rw[part]}"', R.script_line(sk, t, R.req_word_turns(sk)))
            self.assertEqual(sk["provenance"]["topic_words"], "FAKE")
        self.assertGreater(topical, 0.9 * sum(len(sk["required_words"]["turn_hint"]) for sk in corpus(N)))

    def test_topic_word_table(self):
        self.assertLessEqual(set(P.pool("topic").values), set(TW.TOPIC_WORDS))
        for topic in TW.TOPIC_WORDS:
            for part in TW.PARTS:
                self.assertTrue(TW.words(topic, part), (topic, part))
                self.assertTrue(all(not heldout.vocab_hits(w) for w in TW.words(topic, part)))

    def test_versions(self):
        self.assertEqual((S.GEN_VERSION, S.RNG_VERSION), ("skel-v0.3", "skel-v0.1"))   # v0.3: 2026-09-28
        self.assertEqual(corpus(N)[0]["gen_version"], "skel-v0.3")


class TestCheckerUnits(unittest.TestCase):
    def test_stem_pairs(self):
        # 2026-09-28: Porter (test_round2_fixes) no longer merges noisy/noise, shaking/shaky or cookies/cook
        same = [("saving", "save"), ("shaded", "shade"), ("worried", "worry"), ("worries", "worry"),
                ("running", "runs"), ("stopped", "stop"), ("holidays", "holiday"), ("parties", "party"),
                ("boxes", "box")]
        for a, b in same:
            self.assertEqual(stem(a), stem(b), (a, b))
        for a, b in [("noise", "nose"), ("bus", "bush"), ("glass", "gla"), ("class", "clas")]:
            self.assertNotEqual(stem(a), stem(b), (a, b))

    def test_straight_quotes_keep_offsets(self):
        s = "I’m sure it’s “fine”"
        self.assertEqual(check_base.straight(s), "I'm sure it's \"fine\"")
        self.assertEqual(len(check_base.straight(s)), len(s))

    def test_det_swap_is_never_for_values(self):
        sk = _first(lambda s: any(t["role"] == "user" and t["mode"] == "guided" for t in s["turns"]))
        t = next(t for t in sk["turns"] if t["role"] == "user" and t["mode"] == "guided")
        t = dict(t, must_include=["the cellar"])
        skel = dict(sk, turns=[t if x["i"] == t["i"] else x for x in sk["turns"]])
        ctx = Ctx(skel, {t["i"]: "Is my cellar dry?"})
        self.assertTrue(ctx.has(t["i"], "the cellar"))
        self.assertFalse(Ctx(skel, {t["i"]: "Is your cellar dry?"}).has(t["i"], "the cellar"))
        ctx.values = ctx.values | {"the cellar"}
        self.assertFalse(ctx.has(t["i"], "the cellar"))

    def test_role_swap_question_may_say_your(self):
        from check_behav import own_nouns, _user_perspective
        sk = _first(lambda s: own_nouns(s) and any(e["kind"] == "S6" and e["params"]["variant"] == "role_swap" and
                                                   s["turns"][e["turns"]["query"]]["mode"] == "guided"
                                                   for e in s["events"]))
        q = next(e["turns"]["query"] for e in sk["events"] if e["kind"] == "S6" and e["params"]["variant"] == "role_swap")
        other = next(t["i"] for t in sk["turns"] if t["role"] == "user" and t["mode"] == "guided" and t["i"] != q)
        line = f"And what about your {own_nouns(sk)[0]}?"
        self.assertEqual(_user_perspective(Ctx(sk, {q: line})), [])
        self.assertEqual(len(_user_perspective(Ctx(sk, {other: line}))), 1)

    def test_record_keeps_lowercase_exact_lines_and_folded_spans(self):
        sk = _first(lambda s: _lower_exact(s) and not R.feasible(s))
        res = checker.run(sk, fake_teacher.raw(sk))
        self.assertTrue(res["ok"], res["hits"][:3])
        turns = checker.record_turns(sk, res)
        for t in sk["turns"]:
            if t["mode"] == "exact" and t["role"] == "user":
                self.assertEqual(turns[t["i"]]["text"], t["text"].lower())
        cap = [s for s in sk["slots"].values() if s["value"][:1].isupper()]
        text = " ".join(v["value"].lower() for v in cap)
        if cap:
            self.assertEqual(len(records.spans(text, sk["slots"], fold=True)) >= len(cap), True)
            self.assertEqual(records.spans(text, {k: v for k, v in sk["slots"].items() if v in cap}), [])


if __name__ == "__main__":
    unittest.main()
