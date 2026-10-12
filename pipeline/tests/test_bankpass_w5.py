"""Bank pass W5 unit rules: the OFFTOPIC / REQ_WORD / VOCAB_OOL measures on probe renders (w5measure: the family
matcher, the prev_assist variants, the swapped-line catch, computed generic families) and the dry build summary
(w5dry). No model; FAKE test text only."""
import unittest
from unittest import mock

from _corpus import corpus  # noqa: F401  (puts the pipeline on sys.path)
import topic_words as TW
from bankpass import w5dry, w5measure as WM
from test_bankpass_w4c import WL


def sk_of(turns, topics=None, slots=None):
    return {"skel_id": "s", "topic_text": topics or {"t1": "starting a vegetable garden"}, "turns": turns,
            "slots": slots or {}, "required_words": {"noun": "garden", "verb": "dig", "adj": "green",
                                                     "turn_hint": [1, 3, 5]},
            "provenance": {"banks": ["banks@x:BANKSET"], "pools": ["name@y:FAKE"], "personas": "FAKE",
                           "topic_words": "tw@z:g33m33q33", "lists": []}, "assistant": {"system_ref": None}}


def turn(i, role, intent, mode="guided", ref=None):
    return {"i": i, "role": role, "mode": mode, "intent": intent, "bank_ref": ref}


class TestMeasure(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(TW.TOPIC_WORDS, {"starting a vegetable garden": ("soil seed", "dig", "green"),
                                              "a broken kettle": ("kettle", "boil", "hot")})
        p.start()
        self.addCleanup(p.stop)
        self.fam = WM.Fam(WL, generic={"lovely"})
        self.sk = sk_of([turn(0, "user", "topic:t1:0"), turn(1, "assistant", "reply on the topic"),
                         turn(2, "user", "topic:t1:1"), turn(3, "assistant", "reply on the topic"),
                         turn(4, "assistant", "answer with the looked up value")])

    def test_flags_fam_and_prev(self):
        texts = {0: "I want to dig some beds", 1: "That paris kettle chat", 2: "lovely paris news", 3: "So lovely",
                 4: "kettle news"}
        got = {r: WM.flags_fam(self.sk, texts, self.fam, r) for r in ("cur", "fam2", "none")}
        self.assertEqual(got, {"cur": {1, 3}, "fam2": {1, 2, 3}, "none": {1, 2, 3}})    # half of own: cur only
        texts[2] = "paris kettle news"
        got = {r: WM.flags_fam(self.sk, texts, self.fam, r) for r in ("cur", "fam2", "none")}
        self.assertEqual(got, {"cur": {1, 3}, "fam2": {1, 3}, "none": {1, 2, 3}})       # 2 shared families
        self.assertEqual(WM.flags_fam(self.sk, {0: "we dug beds"}, self.fam, "none"), set())     # dug -> dig

    def test_swap_rates(self):
        other = sk_of([turn(0, "user", "topic:t9:0"), turn(1, "assistant", "reply on the topic")],
                      topics={"t9": "a broken kettle"})
        same = sk_of([turn(0, "user", "topic:t1:0"), turn(1, "assistant", "reply on the topic")])
        rows = [(self.sk, {0: "dig the soil", 1: "green seed beds", 2: "more soil", 3: "seed it", 4: "whatever"}),
                (other, {0: "the kettle broke", 1: "boil it again"}), (same, {0: "soil and seed", 1: "dig deeper"})]
        out = WM.swap_rates(rows, lambda sk, tx: WM.flags_fam(sk, tx, self.fam, "none"))
        self.assertEqual(out, {"user": [4, 0, 4], "assistant": [4, 0, 4]})   # never a donor from a chat on its topic

    def test_req_vocab_generic(self):
        rows = [(self.sk, {0: "dig the soil", 1: "green seed beds", 2: "more soil", 3: "seed it"})]
        self.assertEqual(WM.req_rates(rows, WL), {"placed": 3, "missing_checker_forms": 1, "missing_family_forms": 1})
        sk = dict(self.sk, slots={"s1": {"value": "Garden"}})
        rows2 = [(sk, {0: "we dug it", 1: "green", 2: "a garden", 3: ""})]
        self.assertEqual(WM.req_rates(rows2, WL)["missing_checker_forms"], 1)        # dug: not a checker form
        self.assertEqual(WM.req_rates(rows2, WL)["missing_family_forms"], 0)         # dug: the family lists it
        self.assertEqual(WM.vocab_ool(rows2, WL)["user"], 0.75)                      # we, it, a; the slot value left out
        cat = sk_of([turn(0, "user", "topic:t5:0"), turn(1, "assistant", "reply on the topic")],
                    topics={"t5": "a new cat"})
        other = sk_of([turn(0, "user", "topic:t9:0"), turn(1, "assistant", "reply on the topic")],
                      topics={"t9": "a broken kettle"})
        prow = [(self.sk, {1: "great soil"}), (other, {1: "great boil"}), (cat, {1: "great cat"})]
        self.assertEqual(WM.probe_generic(prow, WM.Fam(WL, set())), ["great"])

    def test_dry_summary(self):
        sk = sk_of([turn(0, "user", None, "exact", "open.greet.0"), turn(1, "assistant", "x"),
                    turn(2, "user", None, "exact", "key.job.plant.1"), turn(3, "user", "topic:t1:0")])
        rep = w5dry.summarize([sk, sk], {"open.greet.0": "gemma-4-12b", "key.job.plant.1": "qwen3.5-9b"})
        self.assertEqual(rep["fake_refs_by_name"], {"name": 2, "FAKE": 2})
        self.assertEqual(rep["exact_user_turns"], 4)
        self.assertEqual(rep["exact_author_shares"], {"gemma-4-12b": 0.5, "qwen3.5-9b": 0.5})
        self.assertEqual(rep["skeletons_fake"], 2)


if __name__ == "__main__":
    unittest.main()
