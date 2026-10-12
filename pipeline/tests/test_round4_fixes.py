"""Tests for the 2026-10-04 round 4 fixes (dry pilot 4 v1 quality review; SPEC 16, notes.txt): the self-name answer's
guidance, the lowercase-chat example, the round 4 patterns and the parts of check_r4 no checker fixture reaches
cheaply (everyday-word values, idiom heads, lookup values). The checker's planted and clean pairs run in the mutation
harness (fixtures_r4.py, test_checker). No model is loaded or called; every text here is FAKE fixture text."""
import copy
import os
import sys
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import check_r4 as C4
import fake_teacher
import lexicons as L
import lexicons_r4 as L4
import pools as P
import render_intents as RI
import render_prompt as R
from check_base import Ctx

N = 400
sys.path.insert(0, os.path.join(os.path.dirname(C4.__file__), "teachers"))


def _first(pred, n=N):
    return next(sk for sk in corpus(n) if pred(sk))


def _ctx(sk, edits=None):
    tx = fake_teacher.render(sk)
    tx.update(edits or {})
    return Ctx(sk, tx, built=R.build(sk))


class TestSelfNameGuidance(unittest.TestCase):
    def test_self_name_lead_in_points_at_nothing(self):
        got = [(sk, t) for sk in corpus(N) for t in sk["turns"] if t["role"] == "assistant"
               and (t["intent"] or "").startswith("answer with the value after a short lead in")]
        selfs = [(sk, t) for sk, t in got if RI.self_name_answer(sk, t)]
        self.assertTrue(selfs and len(selfs) < len(got))
        for sk, t in selfs:
            g = RI.guidance(sk, t)
            self.assertTrue(g.startswith(RI.SELF_LEAD), g)
            self.assertNotIn("user said", g)
            self.assertIn(R.card_name(sk), t["must_include"])
            self.assertTrue(L.GUIDE_META_RE.search(g), g)              # a verbatim copy is a PROMPT_ECHO
        for sk, t in got:
            if not RI.self_name_answer(sk, t):
                self.assertIn("pointing back to when the user said it", RI.guidance(sk, t))


class TestLowercaseChats(unittest.TestCase):
    def test_req_span_stays_case_exact_on_assistant_turns(self):
        """task d (10-04): no case fold for an assistant's required name in a lowercase chat (D1); the lowercase
        example measured on Qwen did not help and is not in the prompt."""
        sk = _first(lambda s: s["user"]["style"] == "lowercase" and any(
            t["role"] == "assistant" and t["mode"] == "guided" and any(x[:1].isupper() for x in t["must_include"])
            for t in s["turns"]))
        t = next(t for t in sk["turns"] if t["role"] == "assistant" and t["mode"] == "guided"
                 and any(x[:1].isupper() for x in t["must_include"]))
        v = next(x for x in t["must_include"] if x[:1].isupper())
        ctx = _ctx(sk, {t["i"]: f"So it is {v.lower()} then."})
        self.assertFalse(ctx.has(t["i"], v))
        self.assertTrue(_ctx(sk, {t["i"]: f"So it is {v} then."}).has(t["i"], v))
        self.assertFalse(hasattr(R, "EXAMPLE_LOWER"))
        self.assertIn(R.EXAMPLE, R.build(sk)["blocks"]["tail"])


class TestPatterns(unittest.TestCase):
    def test_user_said_points_at_values_not_questions(self):
        for s in ("You called me Tavi.", "You've called me Kit", "As you said, it is noon.", "I remember you saying it",
                  "You mentioned that my name is Pim"):
            self.assertTrue(L4.USER_SAID_RE.search(s), s)
        for s in ("You asked about the museum.", "You never asked.", "I said it before."):
            self.assertFalse(L4.USER_SAID_RE.search(s), s)

    def test_remember_is_not_a_promise(self):
        for s in ("I remember you said Perth.", "I do remember that."):
            self.assertTrue(L4.REMEMBER_RE.search(s), s)
        for s in ("I'll remember Perth.", "I will remember that.", "I don't remember.", "Remember to rest."):
            self.assertFalse(L4.REMEMBER_RE.search(s), s)

    def test_earlier_words(self):
        for s in ("You called me Kit right at the start", "you asked me before", "I noted Ursula earlier",
                  "I remember you mentioning Bologna earlier", "At the start, you mentioned it"):
            self.assertTrue(L.EARLIER_RE.search(s), s)
        self.assertFalse(L.EARLIER_RE.search("You asked about the start time."))

    def test_service_substitutes_and_clean_lines(self):
        for s in ("I am functioning well and ready to help you now.", "I'm an assistant here to help.",
                  "Glad I could assist.", "how else can help?", "I can certainly do that for you.",
                  "I have that information stored right here for you.", "Would you like to remember anything else?"):
            self.assertTrue(L4.SERVICE_RE.search(s.lower()), s)
        for s in ("Nothing else is on the list.", "It matters more than anything else.", "The bus is ready to go.",
                  "I can see why."):
            self.assertFalse(L4.SERVICE_RE.search(s.lower()), s)


class TestCheckEdges(unittest.TestCase):
    def test_everyday_word_values_are_left_to_req_span(self):
        sk = copy.deepcopy(_first(lambda s: any(t["role"] == "assistant" and t["mode"] == "guided" and not
                                                t.get("lookup_call") for t in s["turns"])))
        a = next(t for t in sk["turns"] if t["role"] == "assistant" and t["mode"] == "guided")
        sid = next(iter(sk["slots"]))
        for val, text, fires in (("May", "You may want to rest.", False), ("Ghent", "You live in ghent now.", True)):
            sk["slots"][sid]["value"] = val
            got = C4.chk_case_names(_ctx(sk, {a["i"]: text}))
            self.assertEqual(bool(got), fires, (val, got))
        self.assertIn("may", L4.AMBIG_LOWER)
        self.assertNotIn("ghent", L4.AMBIG_LOWER)

    def test_idiom_heads_are_not_claims(self):
        sk = copy.deepcopy(_first(lambda s: any(t["role"] == "assistant" and t["mode"] == "guided"
                                                for t in s["turns"])))
        a = next(t for t in sk["turns"] if t["role"] == "assistant" and t["mode"] == "guided")
        s0 = next(s for s in sk["slots"].values() if s["owner"] != "assistant")
        s0["noun"] = "phone case"
        self.assertEqual(C4.chk_claims_user(_ctx(sk, {a["i"]: "In my case, I would wait."})), [])
        s0["noun"] = "book club"
        self.assertTrue(C4.chk_claims_user(_ctx(sk, {a["i"]: "My club begins soon."})))

    def test_a_lookup_value_is_not_a_false_memory(self):
        sk = _first(lambda s: any(t["role"] == "tool" for t in s["turns"]))
        tool = next(t for t in sk["turns"] if t["role"] == "tool")
        ans = next(t for t in sk["turns"] if t["i"] > tool["i"] and t["role"] == "assistant" and t["mode"] == "guided")
        v = next(v for v in sorted(C4.R3._names(Ctx(sk, fake_teacher.render(sk))))
                 if v in (tool.get("text") or ""))
        ctx = _ctx(sk, {ans["i"]: f"You said to check, and it is {v}."})
        self.assertEqual(C4.chk_never_said(ctx), [])


class TestBanIsOn(unittest.TestCase):
    def test_drive_turns_the_phrase_ban_on(self):
        import drive
        import serve_http
        src = open(drive.__file__).read()
        self.assertIn('argv.append("--ban-phrases")', src)
        self.assertIn("--no-ban-phrases", src)
        self.assertIn("default=True", open(serve_http.__file__).read().split('add_argument("--phrase-ban"', 1)[1][:80])

    def test_the_dp4_service_families_are_banned_bare(self):
        """the bare forms carry the closers and offers no context lists (dp4 counts in decode.py)."""
        import decode
        self.assertEqual(decode.PHRASE_RULE, "aiism-v4")
        for p in ("anything else", "Anything else", "here to help", "I can help", "I can certainly", "ready to help",
                  "I can definitely",
                  "love to help", "how else can I", "happy to help"):
            self.assertIn(p, decode.PHRASES)
        self.assertLessEqual(2 * len(decode.PHRASES), decode.BAD_WORDS_CAP)


if __name__ == "__main__":
    unittest.main()
