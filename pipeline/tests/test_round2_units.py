"""More tests for the 2026-09-28 round 2 fixes (SPEC 14, notes.txt): a lowercase-style user's script lines are shown
in lowercase, the lead-in answer guidance, the plant clause rule on continued subjects, the assistant-only service
phrases, and the fixes the full dp2 re-check read asked for (topic-side forms, rule notes after the move, the 3-word
rule, denials, "I called you", "No X was mentioned", "the one you added", embedded plants). No model is loaded or
called; every text here is FAKE fixture text."""
import unittest

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import lexicons as L
import skeleton
from check_base import content, stem
from check_behav import topic_set
from check_events import ONE_RE, PRONOUN_ONE
from stemmer import topic_forms
import render_intents as RI
import render_prompt as R
from check_events import stated

N = 400


def _items(line):
    """the quoted items of a script line's bracket."""
    head = line.split("]: ", 1)[0]
    return [x for n, x in enumerate(head.split('"')) if n % 2]


class TestLowercaseScriptLines(unittest.TestCase):
    def test_lowercase_user_lines_show_items_and_guidance_in_lowercase(self):
        seen = 0
        for sk in corpus(N):
            low = sk["user"]["style"] == "lowercase"
            rw = R.req_word_turns(sk)
            for t in sk["turns"]:
                if t["mode"] != "guided":
                    continue
                line = R.script_line(sk, t, rw)
                items = [x for x in t["must_include"] + t["must_exclude"] if x != x.lower()]
                if not items:
                    continue
                if low and t["role"] == "user":
                    seen += 1
                    self.assertEqual(line.split("]: ", 1)[1], RI.guidance(sk, t).lower())
                    self.assertEqual(_items(line), [x.lower() for x in _items(line)], line)
                    self.assertTrue(all(x.lower() in _items(line) for x in items if x in t["must_include"]), line)
        self.assertGreater(seen, 20)

    def test_other_users_and_the_assistant_keep_the_case(self):
        n = 0
        for sk in corpus(N):
            for t in sk["turns"]:
                cap = [x for x in t["must_include"] if x != x.lower()]
                if t["mode"] != "guided" or not cap:
                    continue
                if t["role"] == "assistant" or sk["user"]["style"] != "lowercase":
                    n += 1
                    self.assertIn(f'"{cap[0]}"', R.script_line(sk, t, {}))
        self.assertGreater(n, 50)


class TestGuidanceWording(unittest.TestCase):
    def test_greet_backs_offer_a_question_that_is_not_service_talk(self):
        g = {RI.guidance(sk, t).split("; end with")[0] for sk in corpus(N) for t in sk["turns"] if t["mode"] == "guided"
             and (t["intent"] or "").startswith(("greet the user back", "greet back and engage"))}
        self.assertEqual(g, {"greet the user back briefly, maybe asking how they are; no topic yet",
                             "greet the user back and remark on the topic"})

    def test_abstain_names_the_hedge_and_asks(self):
        g = {RI.guidance(sk, t).split(";")[0] for sk in corpus(N) for t in sk["turns"] if t["mode"] == "guided"
             and (t["intent"] or "").startswith("say it was not mentioned")}
        self.assertEqual(g, {"say the user has not told you, give no guess, and ask for it in a question"})  # round 3

    def test_lead_in_answers_start_from_what_the_user_said(self):
        g = [RI.guidance(sk, t) for sk in corpus(N) for t in sk["turns"] if t["mode"] == "guided"
             and (t["intent"] or "").startswith("answer with the value after a short lead in")]
        self.assertTrue(g)
        for x in g:
            self.assertTrue(x.startswith("answer, first pointing back to when the user said it; no other value"), x)
            self.assertNotIn("your", x.split(";")[0])
        self.assertTrue(L.GUIDE_META_RE.search("Leading in with what you told me, it is Porto."))


class TestGuidanceCopies(unittest.TestCase):
    def test_withholding_clause_and_its_paraphrases(self):
        for s in ("Can you remind me of my hobby without saying the answer?",
                  "Please remind me what my boss does for work without telling me the answer.",
                  "What is my cat called? Remind me without giving me the answer."):
            self.assertTrue(L.GUIDE_META_RE.search(s), s)
        for s in ("I left without telling my sister.", "I bought it without giving it much thought."):
            self.assertFalse(L.GUIDE_META_RE.search(s), s)


class TestPlantClauses(unittest.TestCase):
    def test_a_continued_subject_states(self):
        for text, v in [("My dog is old and is called Rex.", "Rex"), ("I moved and am now in Porto.", "Porto"),
                        ("My aunt was a nurse but is a pilot now.", "pilot")]:
            self.assertTrue(stated(text, v), text)

    def test_a_that_clause_states_inside_a_question(self):
        """the split keeps "that" with its clause: "that is purple" states even when the sentence ends in "?"."""
        self.assertTrue(stated("I bought a hat that is purple, do you like it?", "purple"))
        self.assertTrue(stated("Did you know that my dog is called Rex?", "Rex"))

    def test_a_question_after_a_greeting_still_asks(self):
        for text, v in [("Hi, can you tell me about my friend Demklal?", "Demklal"),
                        ("So, is my yoga class on Monday?", "Monday")]:
            self.assertFalse(stated(text, v), text)


class TestAssistantPhrases(unittest.TestCase):
    def test_service_phrases_and_their_substitutes(self):
        for s in ("Hello, how can I help you today?", "How can Ihelp?", "Is there anything else I can help with?",
                  "I am happy to assist.", "How can I be your helpful assistant?", "I'm here to chat.",
                  "I would be happy to and help you."):
            self.assertTrue(L.ASSIST_ISM_RE.search(s), s)
        # round 3 (10-03, dry pilot 3 MEDIUM 6): "I can (certainly) help you ..." is the substitute Gemma wrote in 95
        # accepted chats, so the round 2 clean case "I can help you plan the route." now fires (AI_ISM is never loosened)
        self.assertTrue(L.ASSIST_ISM_RE.search("I can help you plan the route."))
        for s in ("Patience matters more than anything else I can think of.", "That will help with the noise.",
                  "You can help him plan the route."):
            self.assertFalse(L.ASSIST_ISM_RE.search(s), s)


class TestTopicForms(unittest.TestCase):
    def test_forms_by_part_of_speech(self):
        want = {("noisy", "adj"): "noise", ("leaky", "adj"): "leak", ("sunny", "adj"): "sun", ("tasty", "adj"): "taste",
                ("monthly", "adj"): "month", ("speak", "verb"): "speaker", ("drive", "verb"): "driver",
                ("plan", "verb"): "planner", ("strong", "adj"): "stronger", ("smooth", "adj"): "smoothly"}
        for (w, p), f in want.items():
            self.assertIn(f, topic_forms(w, p), (w, p))
        for (w, p), f in {("busy", "adj"): "bus", ("cozy", "adj"): "coz", ("tidy", "adj"): "tide",
                          ("cat", "verb"): "cater", ("car", "noun"): "carer", ("beach", "noun"): "beacher",
                          ("let", "verb"): "letter", ("sew", "verb"): "sewer"}.items():
            self.assertNotIn(f, topic_forms(w, p), (w, p))

    def test_topic_side_only_and_the_review_collisions_stay_apart(self):
        for w, topic in [("noise", "a broken washing machine"), ("leaks", "a leaky tap"), ("speakers", "learning Spanish"),
                         ("drivers", "a long commute"), ("taste", "cooking dinner on a budget"), ("month",
                         "choosing a new phone plan"), ("sun", "planning a picnic")]:
            self.assertIn(stem(w), topic_set(topic), w)
        # care/car is apart at the word level (test_round2_fixes); the drive set also lists "careful", a true relative
        for w, topic in [("ready", "reading more books"), ("busy", "a long commute"),
                         ("cater", "a cat that scratches furniture"), ("letting", "writing a cover letter"),
                         ("plane", "planning a picnic")]:
            self.assertNotIn(stem(w), topic_set(topic), w)
        self.assertEqual(content("Noted."), set())


class TestPromptPlacement(unittest.TestCase):
    def test_rule_notes_follow_the_move_and_the_floor_is_said(self):
        n = 0
        for sk in corpus(N):
            for t in sk["turns"]:
                if t["mode"] != "guided" or not t.get("rules"):
                    continue
                line = R.script_line(sk, t, {})
                head, move = line.split("]: ", 1)
                self.assertNotIn("rule:", head)
                self.assertTrue(move.endswith(")") and " (rule: " in move, line)
                n += 1
        self.assertGreater(n, 50)
        self.assertIn("in three words or more", R.RULES)
        self.assertEqual({t["min_w"] for sk in corpus(N) for t in sk["turns"] if t["mode"] == "guided"}, {3})


class TestRecheckRefinements(unittest.TestCase):
    def test_hedges_denials_and_past_acts(self):
        for s in ("No friend was mentioned so far, want to share?", "Nothing has been mentioned about it."):
            self.assertTrue(L.HEDGE_RE.search(s), s)
        for s in ("Your dog was mentioned as Rex.", "No problem, it was mentioned as Rex."):
            self.assertFalse(L.HEDGE_RE.search(s), s)
        for s in ("I have no city to call my own.", "I don't have my own hobbies.", "I called you Dev earlier."):
            self.assertEqual(L.self_claims(s), [], s)
        for s in ("An assistant doesn't have a city like my own.", "I called a plumber for you."):
            self.assertTrue(L.self_claims(s), s)

    def test_the_one_before_a_word_is_a_pronoun(self):
        for s, pron in (("There are four, including the one you just added.", True),
                        ("There are six, or maybe just the one.", False), ("Only the one, I think.", False)):
            self.assertEqual(len(ONE_RE.findall(s)) == len(PRONOUN_ONE.findall(s)), pron, s)

    def test_embedded_statements_and_pseudo_clefts_state(self):
        for text, v, want in [("Hi Amara, did you know our team lunch is this Friday?", "Friday", True),
                              ("Where I live is actually the city of Porto.", "Porto", True),
                              ("What time my interview starts is noon.", "noon", True),
                              ("Where I live is Porto?", "Porto", False),
                              ("Do you know the name of my friend Demklal?", "Demklal", False),
                              ("Do you know when my lunch starts at noon?", "noon", False)]:
            self.assertEqual(stated(text, v), want, text)


if __name__ == "__main__":
    unittest.main()
