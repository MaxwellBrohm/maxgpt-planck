"""Round 2 serve and driver changes (2026-09-28), CPU only anywhere (stub vLLM, Python re for xgrammar); DRY.
  ExactLiteral  labels_exact literals are parse.exact_text: a lowercase-style user's copied line is forced
                lowercased, assistant and tool lines unchanged (v1 review: the sentence-case literal broke the style).
  NoEnd         labels-v2: no free line contains END (so END is written once, at the end); no line is capped.
  PhraseBan     decode.PHRASES against lexicons.AI_ISM_RE and the greeting family; literals leave a phrase out;
                serve passes bad_words to SamplingParams.
    cd pipeline/teachers && python3 -B -m unittest test_round2 -v"""
import itertools
import os
import re
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import decode  # noqa: E402
import driver  # noqa: E402
import fake_engine as FE  # noqa: E402
import fake_teacher  # noqa: E402
import lexicons  # noqa: E402
import parse  # noqa: E402
import serve  # noqa: E402
import skeleton  # noqa: E402
import stub_vllm as S  # noqa: E402

SKELS = skeleton.shard("test-round2", 40)
LOWER = [k for k in SKELS if k["user"].get("style") == "lowercase" and any(
    t["mode"] == "exact" and t["role"] == "user" and t["text"] != t["text"].lower() for t in k["turns"])]
OTHER_EXACT = [k for k in SKELS if any(t["mode"] == "exact" and t["role"] != "user" for t in k["turns"])]


def full(sk, text, mode="labels_exact", sep="\\n", rule=decode.LABEL_RULE):
    return re.fullmatch(decode.label_regex(driver.constraint(sk, mode)["lines"], sep, rule), text) is not None


def swap_last(sk, new):
    """the canonical render with its last labeled line's text replaced."""
    lines = FE.canon(sk).split("\n")
    lab = lines[-2].split(":")[0]
    return "\n".join(lines[:-2] + [f"{lab}: {new}", "END"])


class ExactLiteral(unittest.TestCase):
    def test_lowercase_user_lines_are_forced_lowercased(self):
        self.assertGreater(len(LOWER), 3)
        for sk in LOWER:
            lines = dict((lab, lit) for lab, lit in driver.constraint(sk, "labels_exact")["lines"])
            for lab, i, role in parse.plan(sk):
                t = sk["turns"][i]
                if t["mode"] == "exact" and role == "user":
                    self.assertEqual(lines[lab], t["text"].lower())
                    self.assertEqual(lines[lab], parse.exact_text(sk, t))
            self.assertTrue(full(sk, FE.canon(sk)))
            self.assertFalse(full(sk, fake_teacher.raw(sk)), "the sentence-case bank line must not fit")
            self.assertTrue(full(sk, fake_teacher.raw(sk), "labels"))

    def test_assistant_and_tool_literals_keep_their_case(self):
        self.assertTrue(OTHER_EXACT)
        for sk in OTHER_EXACT:
            lines = dict((lab, lit) for lab, lit in driver.constraint(sk, "labels_exact")["lines"])
            for lab, i, role in parse.plan(sk):
                t = sk["turns"][i]
                if t["mode"] == "exact" and role != "user":
                    self.assertEqual(lines[lab], t["text"])

    def test_literals_are_the_exact_turns_only(self):
        for sk in SKELS[:10]:
            lit = driver.literals(sk)
            self.assertEqual(set(lit), {t["i"] for t in sk["turns"] if t["mode"] == "exact"})
            self.assertEqual(driver.constraint(sk, "labels")["lines"], [[lab, None] for lab, _, _ in parse.plan(sk)])


class NoEnd(unittest.TestCase):
    def test_line_class_is_no_end(self):
        rx = re.compile(decode.noend())
        for n in range(0, 9):
            for tup in itertools.product("ENDx", repeat=n):
                s = "".join(tup)
                ok = n >= 1 and not any(x in s for x in ("END", "EE", "ENE"))    # stricter only on capital EE, ENE
                self.assertEqual(rx.fullmatch(s) is not None, ok, s)
        self.assertIsNone(rx.fullmatch("see you\nEND"))
        for line in ("Emma, see you at ENT.", "Bye, EN", "Endless fun, Ed!", "I SENT it", "E"):
            self.assertIsNotNone(rx.fullmatch(line), line)

    def test_end_inside_any_free_line_is_refused(self):
        sk = SKELS[0]
        canon = FE.canon(sk)
        lines = canon.split("\n")
        k = next(k for k, (_, lit) in enumerate(driver.constraint(sk, "labels_exact")["lines"][:-1]) if lit is None)
        mid = "\n".join(lines[:k] + [lines[k] + " END"] + lines[k + 1:])
        for sep in decode.SEPS:
            self.assertTrue(full(sk, canon, sep=sep))
            for bad in (swap_last(sk, "See you later. Bye. END"), swap_last(sk, "Bye. END. END. END."),
                        swap_last(sk, "THE END"), mid):
                self.assertFalse(full(sk, bad, sep=sep), bad[-50:])
                self.assertTrue(full(sk, bad, "labels", sep, "labels-v1"), "v1 let it through")
            self.assertFalse(full(sk, canon + "\nEND", sep=sep))
            self.assertTrue(full(sk, swap_last(sk, "See you at the end of the week, Endo."), sep=sep))

    def test_no_line_is_capped(self):
        sk = SKELS[1]                     # a capped form costs too much in xgrammar (decode.py); max_tokens bounds
        self.assertTrue(full(sk, swap_last(sk, "Bye now. " * 100)))
        self.assertTrue(full(sk, swap_last(sk, "a" * 900)))
        self.assertNotIn("{", decode.label_regex([["U1", None], ["A2", None]]))

    def test_v1_is_the_round_1_regex(self):
        lines = driver.constraint(SKELS[2], "labels")["lines"]
        self.assertEqual(decode.label_regex(lines, "\\n", "labels-v1"),
                         "\\n".join([f"{lab}: [^\\n]+" for lab, _ in lines] + ["END"]))
        with self.assertRaises(ValueError):
            decode.label_regex(lines, "\\n", "labels-v3")
        self.assertEqual(decode.LABEL_RULE, "labels-v2")


def fold(s):
    return s.replace("’", "'").lower()


def ai_ism_alternatives():
    """the top-level alternatives of lexicons.AI_ISM_RE (between its letter guards)."""
    body = lexicons.AI_ISM_RE.pattern.split("(?:", 1)[1].rsplit(")(?![a-z])", 1)[0]
    out, depth, cur = [], 0, ""
    for c in body:
        depth += (c == "(") - (c == ")")
        if c == "|" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    return out + [cur]


class PhraseBan(unittest.TestCase):
    def test_every_phrase_is_a_checker_ai_ism_or_the_greeting_family(self):
        for p in decode.PHRASES:
            self.assertTrue(lexicons.AI_ISM_RE.search(fold(p)) or any(g in fold(p) for g in decode.GREETING), p)

    def test_every_plain_checker_phrase_is_banned(self):
        alts = ai_ism_alternatives()
        self.assertGreater(len(alts), 10)
        plain = [a for a in alts if re.fullmatch(r"[a-z' !,]+", a)]
        self.assertGreater(len(plain), 10)
        for a in plain:
            self.assertTrue(any(fold(p) in a for p in decode.PHRASES), f"AI_ISM {a!r} has no phrase at sampling")

    def test_greeting_family_and_case_forms(self):
        low = {fold(p) for p in decode.PHRASES}
        for g in decode.GREETING:
            self.assertIn(g, low)
        for p in decode.PHRASES:
            if p[0].islower() and not p.startswith(("and ", "be ", "am ")):     # those are mid-sentence contexts
                self.assertIn(p[0].upper() + p[1:], decode.PHRASES, "a lowercase start needs its sentence start")
        self.assertIn("I’m here to help", decode.PHRASES)           # Ministral writes the curly apostrophe
        self.assertLessEqual(2 * len(decode.PHRASES), decode.BAD_WORDS_CAP)

    def test_literals_leave_a_phrase_out(self):
        self.assertEqual(decode.phrase_words(), list(decode.PHRASES))
        self.assertEqual(decode.phrase_words(["The lamp is by the door."]), list(decode.PHRASES))
        got = decode.phrase_words(["Oh, HAPPY TO HELP, see you."])
        self.assertEqual(set(decode.PHRASES) - set(got), {"happy to help", "Happy to help"})
        got = decode.phrase_words(["well, i’m here to help now"])
        self.assertEqual(set(decode.PHRASES) - set(got), {"I'm here to help", "I’m here to help"})

    def test_server_request_merges_constraint_literals(self):
        import serve_http
        sk = next(k for k in SKELS if driver.literals(k))
        lit = next(iter(driver.literals(sk).values()))
        phrase = " ".join(lit.split()[:2])
        info = {"line_sep": "\\n", "structured_backend": "xgrammar", "presets": [], "dash_ban": None,
                "phrase_ban": {"rule": decode.PHRASE_RULE, "n_seqs": 60}}
        with mock.patch.object(decode, "PHRASES", decode.PHRASES + (phrase,)):
            req = serve_http.decode_request({"constraint": driver.constraint(sk, "labels_exact"),
                                             "phrases": "ai_ism"}, info)
            self.assertEqual(req["label_rule"], decode.LABEL_RULE)
            eng = serve_http.Engine(None, {}, None, 0.86, info["phrase_ban"])
            with mock.patch.object(serve, "sampling_record", lambda t, s: {
                    "preset": "x", "bad_words_n": len(s.get("bad_words", ())), "bad_words_sha256": "h"}):
                s, dec = eng.sampling_for({"req": req, "max_tokens": 9})
            self.assertNotIn(phrase, s["bad_words"])
            self.assertEqual(len(s["bad_words"]), len(decode.PHRASES) - 1)
            self.assertEqual((dec["phrases"]["left_out"], dec["label_rule"]), (1, decode.LABEL_RULE))
        for body, inf in (({"phrases": "all"}, info), ({"phrases": "ai_ism"}, {**info, "phrase_ban": None}),
                          ({"phrases": "ai_ism", "literals": "Hi"}, info)):
            with self.assertRaises(ValueError, msg=body):
                serve_http.decode_request(body, inf)

    def test_serve_passes_bad_words(self):
        with S.install(), mock.patch.object(serve, "tokenizer", S.tokenizer):
            t = serve.tokenizer_only("gemma-4-12b")
            t.engine = {"max_model_len": 4096}
            words = decode.phrase_words()
            sp, _ = serve._params(t, {"max_tokens": 50, "bad_words": words, "ban_ids": [3]}, 10, 1)
            self.assertEqual((sp.bad_words, sp.logit_bias), (words, {3: -100.0}))
            none, _ = serve._params(t, {"max_tokens": 50, "bad_words": []}, 10, 1)
            self.assertFalse(hasattr(none, "bad_words"))
            rec = serve.sampling_record(t, {"max_tokens": 5, "bad_words": words})
            self.assertEqual((rec["bad_words_n"], rec["bad_words_sha256"]), (len(words), decode.sha("\n".join(words))))
            self.assertEqual(serve.sampling_record(t, {"max_tokens": 5})["bad_words_n"], 0)


if __name__ == "__main__":
    unittest.main()
