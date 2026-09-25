"""Step 3 tests: render prompt formats, the parser, and the checker's mutation harness (quick corpus, every 4th
checker mutant; the full run is `python3 -B mutation_checker.py`, about a minute). No model is loaded or called."""
import io
import unittest
from contextlib import redirect_stdout

from _corpus import S  # noqa: F401  (puts the pipeline on sys.path)
import checker
import fake_teacher
import fixtures_hand as FH
import mutation_checker
import parse
import render_prompt as R


class TestPrompt(unittest.TestCase):
    def setUp(self):
        self.skel = FH.load_skel(169)
        self.built = R.build(self.skel)

    def test_gemma_raw_format_thinking_off(self):
        raw = R.gemma_raw("PROMPT")
        self.assertEqual(raw, "<|turn>user\nPROMPT<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        body = R.completion_body(self.built)
        self.assertEqual(body["stop"], ["<turn|>", "<|turn>"])
        self.assertTrue(body["prompt"].endswith("<|channel>thought\n<channel|>"))
        self.assertEqual(body["max_tokens"], 2 * sum(t["max_w"] for t in self.skel["turns"]) + 40)
        self.assertEqual((body["temperature"], body["top_p"], body["top_k"], body["min_p"]), (1.0, 0.95, 64, 0.05))

    def test_chat_body(self):
        body = R.chat_body(self.built, "some-local-model")
        self.assertEqual(body["messages"], [{"role": "user", "content": self.built["prompt"]}])

    def test_script_lists_every_turn_once_in_order(self):
        labels = [ln.split(" ", 1)[0] for ln in self.built["blocks"]["script"].split("\n")[1:]]
        self.assertEqual(labels, [lab for lab, _, _ in parse.plan(self.skel)])
        for t in self.skel["turns"]:
            if t["mode"] == "exact":
                self.assertIn(t["text"], self.built["prompt"])

    def test_prompt_is_clean_and_recorded(self):
        p = self.built["prompt"]
        self.assertFalse(any(c in p for c in "\u2013\u2014"))
        self.assertEqual(len(self.built["prompt_sha256"]), 64)
        self.assertTrue(self.built["variant"].startswith("instr.fake."))
        self.assertEqual(self.built["provenance"], "FAKE")

    def test_no_card_name_when_no_system_text(self):
        card = R.card_name(self.skel)
        self.assertIsNone(self.skel["assistant"]["system_text"])
        self.assertNotIn(card, self.built["prompt"])


class TestParse(unittest.TestCase):
    def test_drift_tolerated_and_defects_rejected(self):
        skel = FH.load_skel(63)
        texts = dict(enumerate(FH.HAND[63]))
        raw = parse.serialize(skel, texts)
        for name, fn in FH.DRIFT:
            p = parse.parse(fn(raw), parse.plan(skel))
            self.assertEqual((p["codes"], p["turns"]), ([], texts), name)
        p = parse.parse(raw.replace("\nA4:", "\nA5:"), parse.plan(skel))
        self.assertIn("FORMAT_LINES", [c for c, _ in p["codes"]])


class TestChecker(unittest.TestCase):
    def test_record_keeps_bank_line_for_exact_turns(self):
        skel = FH.load_skel(63)
        res = checker.run(skel, parse.serialize(skel, dict(enumerate(FH.HAND[63]))))
        self.assertTrue(res["ok"], res["hits"])
        rec = checker.record_turns(skel, res)
        self.assertEqual([r["text"] for r in rec][0], skel["turns"][0]["text"])

    def test_fake_render_is_deterministic(self):
        s = S.build(5, "RM")
        self.assertEqual(fake_teacher.raw(s), fake_teacher.raw(s))

    def test_mutation_harness_quick(self):
        out = io.StringIO()
        with redirect_stdout(out):
            status = mutation_checker.main(["--quick", "--mutant-sample", "4"])
        self.assertEqual(status, 0, out.getvalue())
        self.assertIn("reason codes with no planted fixture: none", out.getvalue())


if __name__ == "__main__":
    unittest.main()
