"""Tests for the 2026-10-03 round 3 fixes (dry pilot 3 v1 quality review; SPEC 15, notes.txt): the reworded guidance
(lead-in, abstain, entity, topic forms 0 and 5), the sentence-case instruction in lowercase chats, the guidance frame
rule on the round 2 wording the teachers copied, and the round 3 patterns. The checker's clean and planted pairs run
in the mutation harness (fixtures_r3.py, test_checker). No model is loaded or called; every text here is FAKE
fixture text modeled on the review's findings."""
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import check_r3 as C
import fake_data as F
import heldout
import lexicons as L
import render_intents as RI
import render_prompt as R

N = 400


def _guided(pred):
    return [(sk, t) for sk in corpus(N) for t in sk["turns"] if t["mode"] == "guided" and pred(t)]


class TestGuidanceWording(unittest.TestCase):
    def test_lead_in_names_the_user_as_the_source(self):
        g = [RI.guidance(sk, t) for sk, t in _guided(lambda t: (t["intent"] or "").startswith(
            "answer with the value after a short lead in"))]
        self.assertTrue(g)
        for x in g:
            self.assertNotIn("were told", x)
            self.assertTrue(x.startswith("answer, first pointing back to when the user said it"), x)
            self.assertTrue(L.GUIDE_META_RE.search(x), x)      # a verbatim copy is a PROMPT_ECHO

    def test_abstain_asks_in_a_question(self):
        g = {RI.guidance(sk, t) for sk, t in _guided(lambda t: (t["intent"] or "").startswith("say it was not mention"))}
        self.assertTrue(g)
        for x in g:
            self.assertNotIn("share it", x)
            self.assertTrue(L.GUIDE_META_RE.search(x), x)

    def test_entity_and_topic_forms_carry_no_frame(self):
        """the round 3 wording leaves no frame of three required words to copy; the round 2 wording did."""
        ent = _guided(lambda t: t["intent"] == "ask about the entity")
        self.assertTrue(ent)
        for sk, t in ent:
            self.assertIsNone(C.frame_re(C.guide_frame(sk, RI.guidance(sk, t))), RI.guidance(sk, t))
        sk = corpus(N)[0]
        topic = next(iter(sk["topic_text"].values()))
        for f in (F.INTENT_FORMS[0], F.INTENT_FORMS[5]):
            self.assertIsNone(C.frame_re(C.guide_frame(sk, f.replace("{t}", topic))), f)
        self.assertEqual(len(F.INTENT_FORMS), 6)          # the form draw keeps its range (same RNG streams)
        for f in F.INTENT_FORMS:
            self.assertEqual(heldout.echo5(f.replace("{t}", "training a puppy")), [], f)
        for s in ("Can you give me some advice on training a puppy?", "What are the next steps with training a puppy?"):
            self.assertEqual(heldout.echo5(s), [], s)


class TestLowercaseChats(unittest.TestCase):
    def test_rules_and_first_assistant_line_say_sentence_case(self):
        self.assertIn(R.CASE_CLAUSE, R.RULES)
        n_low = n_other = 0
        for sk in corpus(N)[:120]:
            b = R.build(sk)
            low = sk["user"]["style"] == "lowercase"
            n_low, n_other = n_low + low, n_other + (not low)
            self.assertEqual(R.LOWER_CLAUSE in b["blocks"]["tail"], low)
            self.assertEqual(R.CASE_CLAUSE in b["blocks"]["tail"], not low)
            first = "A" + str(R.first_assistant(sk) + 1) + " ["
            for ln in b["blocks"]["script"].split("\n")[1:]:
                self.assertEqual("; " + R.LOWER_PART + "]" in ln, low and ln.startswith(first), ln)
        self.assertTrue(n_low and n_other)


class TestFrameRule(unittest.TestCase):
    """the round 2 frames dry pilot 3 found copied, on FAKE topic and entity text."""

    def setUp(self):
        self.sk = next(s for s in corpus(N) if any(e["params"].get("entity") for e in s["events"]))
        self.topic = next(iter(self.sk["topic_text"].values()))
        e = next(e for e in self.sk["events"] if e["params"].get("entity"))
        self.ent, self.att = e["params"]["entity"], e["params"]["attribute"]

    def rx(self, g):
        return C.frame_re(C.guide_frame(self.sk, g))

    def test_round2_topic_frames_fire(self):
        t = self.topic
        self.assertTrue(self.rx(f"ask what usually helps with {t}").search(f"What usually helps with {t}?"))
        rx = self.rx(f"ask which step to try next with {t}")
        self.assertTrue(rx.search(f"Which step should I try next with {t}?"))
        self.assertFalse(rx.search(f"What's the next step with {t}?"))
        self.assertFalse(self.rx(f"ask what usually helps with {t}").search(f"What helps most with {t}?"))

    def test_entity_and_attribute_are_holes(self):
        toks = C.guide_frame(self.sk, f"ask about the {self.att} of the {self.ent}")
        self.assertFalse({w.lower() for w in (self.att + " " + self.ent).split()} & set(toks), toks)

    def test_round2_entity_frame_fires_only_on_report(self):
        rx = self.rx(f"ask the assistant to look up the {self.ent} and report its {self.att}")
        self.assertTrue(rx.search(f"Can you look up the {self.ent} and report its {self.att}?"))
        self.assertFalse(rx.search(f"Can you look up the {self.ent} and tell me its {self.att}?"))

    def test_small_frames_are_not_templates(self):
        self.assertIsNone(self.rx(f"mention a small worry about {self.topic}"))      # two required words
        self.assertIsNone(self.rx(f"share how {self.topic} is going"))               # no content word left
        self.assertIsNone(C.frame_re(["which", "one", "you", "picked"]))           # one content word
        self.assertIsNotNone(C.frame_re(["which", "one", "you", "picked", "today"]))

    def test_slot_nouns_are_holes(self):
        """"tell the assistant your guinea pig's name" leaves no frame: "My guinea pig's name is X" is the natural
        line (dp3: 4 accepted plants matched before slot nouns were masked)."""
        sk = next(s for s in corpus(N) if any(x.get("noun") == "guinea pig" for x in s["slots"].values()))
        self.assertIsNone(C.frame_re(C.guide_frame(sk, "tell the assistant your guinea pig's name")))


class TestRound3Patterns(unittest.TestCase):
    def test_told_frames(self):
        for s in ("As you were told before, your class is in May.", "As told before my name is Kit.",
                  "I told you, she lives in Porto.", "My name is Kit as told by my makers."):
            self.assertTrue(L.TOLD_FRAME_RE.search(s), s)
        for s in ("You told me your class is in May.", "As you said, it is in May.", "As told by you, it is May."):
            self.assertFalse(L.TOLD_FRAME_RE.search(s), s)

    def test_memory_patterns(self):
        self.assertTrue(L.FORGOT_RE.search("That sounds fun, but I've forgotten."))
        self.assertTrue(L.SELF_ERR_RE.search("Of course, I meant Lisbon."))
        self.assertFalse(L.SELF_ERR_RE.search("I meant to ask, how did it go?"))
        self.assertFalse(L.SELF_ERR_RE.search("Thanks for the correction, Lisbon it is."))
        self.assertTrue(L.EARLIER_RE.search("You mentioned a florist earlier."))

    def test_live_weather_and_hedges(self):
        for s in ("It is very cold outside with snow falling.", "The weather is so sunny today."):
            self.assertTrue(L.LIVE_RE.search(s), s)
        self.assertTrue(L.LIVE_HEDGE_RE.search("If it is cold outside"))
        for s in ("Rainy weekends suit a good film.", "The weather gets warmer in April."):
            self.assertFalse(L.LIVE_RE.search(s), s)

    def test_self_claim_and_service_substitutes(self):
        self.assertTrue(L.self_claims("I'd love to catch up with old friends."))
        self.assertTrue(L.self_claims("Omelettes, though I'm partial to the plain kind myself."))
        for s in ("I'd love to hear how it goes.", "I do not eat food myself.", "I will limit myself to five words."):
            self.assertFalse(L.self_claims(s), s)
        for s in ("I can certainly help you with that.", "Is there anything else you need?", "Need anything else?",
                  "Would you like to know anything else?"):
            self.assertTrue(L.ASSIST_ISM_RE.search(s), s)
        self.assertFalse(L.ASSIST_ISM_RE.search("Tacos beat anything else on the menu."))


if __name__ == "__main__":
    unittest.main()
