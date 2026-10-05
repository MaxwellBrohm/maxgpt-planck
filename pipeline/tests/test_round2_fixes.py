"""Tests for the 2026-09-28 round 2 fixes (v1 quality review and dry pilot 2; SPEC 14, notes.txt): the Porter stem,
the stored text, required-word forms, the plant clause rule, the guidance wording, the prompt rules and the skeleton
(skel-v0.3). The checker's clean and planted pairs run in the mutation harness (fixtures_r2.py, test_checker). No
model is loaded or called; every text here is FAKE fixture text."""
import re
import unittest

from _corpus import corpus, S  # noqa: F401  (puts the pipeline on sys.path)
import assemble as A
import checker
import fake_data as F
import fake_teacher
import heldout
import lexicons as L
import parse
import render_intents as RI
import render_prompt as R
import skeleton_shard
from check_base import content, stem
from check_events import stated
from check_lines import word_forms_re
from stemmer import porter

N = 400


def _first(pred):
    return next(sk for sk in corpus(N) if pred(sk))


def _guided(pred):
    return [(sk, t) for sk in corpus(N) for t in sk["turns"] if t["mode"] == "guided" and pred(t)]


class TestStem(unittest.TestCase):
    def test_porter_reference_outputs(self):
        ref = {"caresses": "caress", "ponies": "poni", "cats": "cat", "feed": "feed", "agreed": "agre",
               "plastered": "plaster", "motoring": "motor", "sing": "sing", "hopping": "hop", "falling": "fall",
               "filing": "file", "happy": "happi", "sky": "sky", "relational": "relat", "conditional": "condit",
               "digitizer": "digit", "hopefulness": "hope", "goodness": "good", "allowance": "allow",
               "adjustment": "adjust", "dependent": "depend", "adoption": "adopt", "effective": "effect",
               "probate": "probat", "rate": "rate", "cease": "ceas", "controlling": "control", "roll": "roll",
               "generalizations": "gener", "oscillators": "oscil"}
        self.assertEqual({w: porter(w) for w in ref}, ref)

    def test_review_collisions_are_apart(self):
        for a, b in [("ready", "reading"), ("busy", "bus"), ("care", "car"), ("cater", "cat"), ("letting", "letter"),
                     ("plane", "plan"), ("noise", "nose"), ("bus", "bush"), ("glass", "gla")]:
            self.assertNotEqual(stem(a), stem(b), (a, b))

    def test_inflections_still_meet(self):
        for a, b in [("saving", "save"), ("shaded", "shade"), ("worried", "worry"), ("worries", "worry"),
                     ("running", "runs"), ("stopped", "stop"), ("holidays", "holiday"), ("parties", "party"),
                     ("boxes", "box"), ("shaking", "shake"), ("learning", "learn"), ("planning", "plan"),
                     ("neighbour's", "neighbours")]:
            self.assertEqual(stem(a), stem(b), (a, b))

    def test_given_up_pairs(self):
        """the round 1 stripper merged these on both sides; Porter does not (stemmer.topic_forms adds the base of a
        topic set's -y adjective on the topic side only: test_round2_units.TestTopicForms)."""
        for a, b in [("noisy", "noise"), ("shaky", "shaking"), ("cookies", "cook")]:
            self.assertNotEqual(stem(a), stem(b), (a, b))

    def test_long_y_words_and_comparatives(self):
        """dp2 recheck: Porter alone lost photographer/photography and stronger/strong (on-topic lines rejected)."""
        from check_behav import topic_set
        for a, b in [("photography", "photographer"), ("squeaky", "squeak"), ("cloudy", "cloud")]:
            self.assertEqual(stem(a), stem(b), (a, b))
        for w, topic in [("stronger", "getting fit after winter"), ("louder", "a road trip playlist"),
                         ("slower", "a long commute")]:
            self.assertIn(stem(w), topic_set(topic), w)

    def test_noted_is_not_a_topic_word(self):
        self.assertEqual(content("Noted, thanks."), set())
        self.assertIn(stem("note"), content("Leave them a note."))


class TestRecordAndFake(unittest.TestCase):
    def test_teacher_lines_are_stored_straight(self):
        sk = _first(lambda s: not R.feasible(s))
        tx = fake_teacher.render(sk)
        i = next(t["i"] for t in sk["turns"] if t["role"] == "assistant" and t["mode"] == "guided")
        tx[i] = "It’s what you said, isn’t it."
        res = checker.run(sk, parse.serialize(sk, tx))
        self.assertEqual(checker.record_turns(sk, res)[i]["text"], "It's what you said, isn't it.")

    def test_lowercase_user_passes_with_bank_or_lowercased_copies(self):
        """USER_STYLE reads the user's own lines only: a copied exact line passes in either case (D1 folds it)."""
        sk = _first(lambda s: not R.feasible(s) and s["user"]["style"] == "lowercase" and any(
            t["mode"] == "exact" and t["role"] == "user" and t["text"] != t["text"].lower() for t in s["turns"]))
        tx = fake_teacher.render(sk)
        self.assertTrue(checker.run(sk, parse.serialize(sk, tx))["ok"])
        low = {t["i"]: parse.exact_text(sk, t) for t in sk["turns"] if t["mode"] == "exact"}
        self.assertTrue(checker.run(sk, parse.serialize(sk, {**tx, **low}))["ok"])


class TestCheckerUnits(unittest.TestCase):
    def test_word_forms_follow_the_part_of_speech(self):
        yes = [("smooth", "adj", "smoothly"), ("gentle", "adj", "gently"), ("cozy", "adj", "cozily"),
               ("shelf", "noun", "shelves"), ("knife", "noun", "knives"), ("leaf", "noun", "leaves"),
               ("bake", "verb", "baking"), ("stir", "verb", "stirring"), ("chef", "noun", "chefs")]
        no = [("friend", "noun", "friendly"), ("cafe", "noun", "caves"), ("chef", "noun", "cheves"),
              ("fair", "adj", "fairly"), ("short", "adj", "shortly"), ("clear", "adj", "clearly"),
              ("near", "adj", "nearly"), ("quiet", "verb", "quietly")]
        for w, part, form in yes:
            self.assertTrue(word_forms_re(w, part).search(form), (w, part, form))
        for w, part, form in no:
            self.assertFalse(word_forms_re(w, part).search(form), (w, part, form))

    def test_plant_clause_rule(self):
        cases = [("Can you tell me the name of my friend Demklal?", "Demklal", False),
                 ("What is your favourite colour brown?", "brown", False),
                 ("Do you know when my team lunch starts at half past four?", "half past four", False),
                 ("May I tell you my vet visit is in June?", "June", True),   # an embedded statement (dp2 re-check)
                 ("I believe the Skurum Parade is in Oaxaca, can you check that?", "Oaxaca", True),
                 ("My favourite colour is brown, can you remember?", "brown", True),
                 ("Can you remember that my dog is called Rex?", "Rex", True),
                 ("I could eat mac and cheese every day.", "mac and cheese", True),
                 ("By the way, I just bought a hat that is a lovely shade of purple.", "purple", True),
                 ("My friend, who is called Demklal, is nice.", "Demklal", True),
                 ("Where does my sister live in Porto if you do not know?", "Porto", False)]
        for text, v, want in cases:
            self.assertEqual(stated(text, v), want, text)
        self.assertFalse(stated("can you tell me the name of my friend demklal", "Demklal", fold=True))
        self.assertTrue(stated("my friend is demklal", "Demklal", fold=True))

    def test_end_and_meta_patterns(self):
        for s in ("Bye for now. END", "Bye. END. END.", "See you. End.", "The end.", "Wait, no, just the script and end."):
            self.assertTrue(L.END_TOKEN_RE.search(s) or L.END_TAIL_RE.search(s) or L.SCRIPT_TALK_RE.search(s), s)
        for s in ("See you at the end of the week.", "the end of my list is milk", "That depends on the ending."):
            self.assertFalse(L.END_TOKEN_RE.search(s) or L.END_TAIL_RE.search(s) or L.SCRIPT_TALK_RE.search(s), s)
        self.assertFalse(L.HEDGE_RE.search("It is Monday, as was mentioned earlier."))
        self.assertTrue(L.HEDGE_RE.search("That hasn't been mentioned yet."))


class TestGuidance(unittest.TestCase):
    def test_no_withholding_suffix_and_recall_queries_exclude_the_answer(self):
        pairs = _guided(lambda t: t["role"] == "user")
        self.assertEqual([RI.guidance(sk, t) for sk, t in pairs if "without saying" in RI.guidance(sk, t)], [])
        n = 0
        for sk in corpus(N):
            for e in sk["events"]:
                q, a = e["turns"].get("query"), e["gold"].get("answer")
                t = sk["turns"][q] if q is not None else None
                if t and t["mode"] == "guided" and isinstance(a, str) and t["intent"].startswith(A.RECALL_QUERY) \
                        and "never said" not in t["intent"]:
                    self.assertIn(a, t["must_exclude"], t["intent"])
                    self.assertIn(f'"{a.lower()}"', R.script_line(sk, t, {}).split("must not include:")[1].lower())
                    n += 1
        self.assertGreater(n, 50)

    def test_never_said_query_reads_like_a_recall_question(self):
        g = [RI.guidance(sk, t) for sk, t in _guided(lambda t: "never said" in (t["intent"] or ""))]
        self.assertTrue(g)
        self.assertEqual([x for x in g if not x.startswith("ask the assistant to remind you")], [])

    def test_list_rule_lookup_and_assistant_wording(self):
        for sk, t in _guided(lambda t: True):
            g, it = RI.guidance(sk, t), t["intent"]
            if it.startswith("list "):
                self.assertTrue(g.startswith("have "), g)
            if "ask for rule " in it:
                self.assertIn(RI.RULE_HEAD, g)
                self.assertNotIn("drop the earlier rule", g)
            if it == "ask about the entity":      # 10-03 (round 3): "look up the X and report its Y" was copied
                self.assertTrue(g.startswith("ask about the ") and "report its" not in g, g)
            base = it.split(";")[0]
            if base in ("answer with the value first", "answer with the value after a short lead in"):
                self.assertTrue(re.search(r"going on|pointing back to when the user said it|then a few words|the name you go by", g), g)
            if base == "agree and follow the rule":
                self.assertIn("keep the rule in this reply", g)
            if base.startswith("say it was not mentioned"):
                self.assertIn("ask for it in a question", g)   # 10-03 (round 3)

    def test_prompt_rules_and_intent_forms(self):
        self.assertIn("no dashes (use a comma or a new sentence)", R.RULES)
        self.assertNotIn(";", R.RULES)
        self.assertEqual(len(F.INTENT_FORMS), 6)      # the form draw keeps its range (same RNG streams)
        for f in F.INTENT_FORMS:
            self.assertNotIn("the assistant", f)
            self.assertNotIn("simple tip", f)          # drew "Can you give me a simple tip", an E004 5-gram
            self.assertEqual(heldout.echo5(f.replace("{t}", "training a puppy")), [], f)
        self.assertEqual(heldout.echo5("What usually helps with training a puppy?"), [])

    def test_prompt_echo_reads_the_built_script(self):
        """a user line equal to the guidance the teacher was SHOWN fires, whatever render_intents says today."""
        sk = _first(lambda s: not R.feasible(s) and any(
            t["role"] == "user" and t["mode"] == "guided" and (t["intent"] or "").startswith("topic:") for t in s["turns"]))
        t = next(t for t in sk["turns"] if t["role"] == "user" and t["mode"] == "guided" and t["intent"].startswith("topic:"))
        b = R.build(sk)
        lab = parse.ROLE_LETTER["user"] + str(t["i"] + 1) + " ["
        shown = "mention the rainy weather"   # under PROMPT_N words: only the equality rule can see it
        b["blocks"]["script"] = "\n".join(ln.split("]: ")[0] + "]: " + shown if ln.startswith(lab) else ln
                                          for ln in b["blocks"]["script"].split("\n"))
        tx = fake_teacher.render(sk)
        tx[t["i"]] = "Mention the rainy weather."
        res = checker.run(sk, parse.serialize(sk, tx), b)
        self.assertIn(("PROMPT_ECHO", t["i"], "guidance copied"), res["hits"])

    def test_guidance_meta_catches_copies_not_natural_lines(self):
        for s in ("Can you remind me of my dog without saying the answer?", "Request a rule: short replies.",
                  "Okay, I will keep the rule in this reply."):
            self.assertTrue(L.GUIDE_META_RE.search(s), s)
        for s in ("Can you remind me of my dog's name?", "Please add milk to my shopping list.",
                  "From now on, end every reply with a question."):
            self.assertFalse(L.GUIDE_META_RE.search(s), s)


class TestSkeleton(unittest.TestCase):
    def test_versions_and_shard_module(self):
        self.assertEqual((S.GEN_VERSION, S.RNG_VERSION), ("skel-v0.3", "skel-v0.1"))
        self.assertIs(S.shard, skeleton_shard.shard)
        self.assertIs(S.iter_shard, skeleton_shard.iter_shard)

    def test_goodbyes_carry_no_required_word(self):
        self.assertNotIn("say goodbye briefly", A.TOPIC_TURNS)
        for sk in corpus(N):
            for i in sk["required_words"]["turn_hint"]:
                self.assertNotEqual(sk["turns"][i]["intent"].split(";")[0], "say goodbye briefly")

    def test_end_question_rule_never_meets_a_bank_goodbye(self):
        n = 0
        for sk in corpus(N):
            s4 = [e for e in sk["events"] if e["kind"] == "S4"]
            fixed_bye = any(e["kind"] == "S8" and e["params"]["act"] == "goodbye" for e in sk["events"])
            if len(s4) == 1 and s4[0]["params"]["rules"][-1] == "end_question" and not fixed_bye:
                n += 1
                self.assertNotEqual(sk["closing"], "goodbye", sk["seed"])
        self.assertGreater(n, 5)


if __name__ == "__main__":
    unittest.main()
