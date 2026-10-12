"""Round 2 (2026-09-28) end to end: drive.py --serve through serve_client, serve_http and serve.py's real request
path to a FAKE engine (fake_engine.py): the phrase ban reaches SamplingParams.bad_words with the chat's literals left
out, the labels-v2 regex and parse.exact_text literals reach the engine, every record keeps all run flags, and a
resume cannot mix label rules. CPU only, no model anywhere; DRY.
    cd pipeline/teachers && python3 -B -m unittest test_round2_e2e -v"""
import json
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import decode  # noqa: E402
import drive_report  # noqa: E402
import driver  # noqa: E402
import fake_engine as FE  # noqa: E402
import teacher_client as TC  # noqa: E402
import test_decode_e2e as E  # noqa: E402

SKELS, FEASIBLE = E.SKELS, E.FEASIBLE
ALL = ["--structured", "labels_exact", "--ban-dashes", "--ban-phrases", "--preset", "card", "--repair"]
# a phrase taken from one skeleton's forced literal, so the left-out path runs end to end
LIT_SK = next(k for k in FEASIBLE if driver.literals(k))
LIT_PHRASE = " ".join(next(iter(driver.literals(LIT_SK).values())).split()[:2])


class PhraseBanEndToEnd(E.Base):
    def test_all_flags_reach_the_engine_and_every_record(self):
        phrases = decode.PHRASES + (LIT_PHRASE,)
        s = FE.Served(SKELS, E.good, phrases=True)
        with mock.patch.object(decode, "PHRASES", phrases):
            self.assertEqual(self.run_served(s, *ALL), 0)
        self.assertEqual(s.t.llm.violations, [])
        seen = {k["skel_id"]: sp for k, _, sp in s.t.llm.seen}
        self.assertEqual(set(seen), {k["skel_id"] for k in FEASIBLE})
        for sk in FEASIBLE:
            sp = seen[sk["skel_id"]]
            lits = list(driver.literals(sk).values())
            want = [p for p in phrases if not any(decode._fold(p) in decode._fold(x) for x in lits)]
            self.assertEqual(sp.bad_words, want)
            self.assertEqual(LIT_PHRASE in sp.bad_words, sk["skel_id"] != LIT_SK["skel_id"])
            rx = decode.label_regex(driver.constraint(sk, "labels_exact")["lines"], "\\n")
            self.assertEqual(sp.structured_outputs.regex, rx)
            self.assertIn("(?:[^E\\n]", rx)
            self.assertEqual(sp.logit_bias, {i: -100.0 for i in FE.BANNED})
        acc, rej = self.records()
        self.assertTrue(acc)
        flags = {"structured": "labels_exact", "ban": "dash", "phrases": "ai_ism", "preset": "card", "repair": True}
        for r in acc + rej:
            self.assertEqual(r["run_flags"], flags)
            self.assertIn(r["attempt_kind"], driver.KINDS)
        for r in acc + [r for r in rej if r["teacher"]]:
            dec = (r["provenance"]["teacher"] if "provenance" in r else r["teacher"])["decode"]
            self.assertEqual((dec["structured"], dec["label_rule"], dec["preset"]), ("labels_exact", "labels-v2", "card"))
            self.assertEqual(dec["phrases"]["rule"], decode.PHRASE_RULE)
            self.assertEqual(dec["phrases"]["n"], len(seen[r["skel_id"]].bad_words))
            self.assertEqual(dec["phrases"]["left_out"], len(phrases) - dec["phrases"]["n"])
            self.assertEqual(dec["ban"]["ids_sha256"], decode.ids_sha(FE.BANNED))
        with open(os.path.join(self.out, "decode.json")) as f:
            self.assertEqual(json.load(f), {**flags, "label_rule": "labels-v2", "phrase_rule": decode.PHRASE_RULE})
        calls = len(acc) + sum(1 for r in rej if r["teacher"])
        rep = drive_report.decode_applied(acc, rej)
        self.assertEqual((rep["phrase_ban"], rep["label_rules"], rep["run_flags"]), (calls, {"labels-v2": calls}, [flags]))

    def test_phrases_alone_send_no_regex_and_still_leave_literals_out(self):
        s = FE.Served(SKELS, E.good, phrases=True)
        with mock.patch.object(decode, "PHRASES", decode.PHRASES + (LIT_PHRASE,)):
            self.assertEqual(self.run_served(s, "--ban-phrases"), 0)
            for sk, _, sp in s.t.llm.seen:
                self.assertFalse(hasattr(sp, "structured_outputs") or hasattr(sp, "logit_bias"))
                self.assertEqual(sp.bad_words, decode.phrase_words(driver.literals(sk).values()))
                self.assertEqual(LIT_PHRASE in sp.bad_words, sk["skel_id"] != LIT_SK["skel_id"])
        with open(os.path.join(self.out, "decode.json")) as f:
            self.assertEqual(json.load(f)["phrase_rule"], decode.PHRASE_RULE)
            f.seek(0)
            self.assertNotIn("label_rule", json.load(f))


    def test_an_end_inside_a_line_could_not_have_been_written(self):
        def end_inside(sk, prompt, sp):                  # Ministral's dp2 run-on, which labels-v2 rules out
            lines = FE.canon(sk).split("\n")
            return "\n".join(lines[:-2] + [lines[-2] + " Bye. END. END.", "END"])
        s = FE.Served(SKELS, end_inside)
        self.run_served(s, "--structured", "labels_exact")
        self.assertEqual({sid for sid, _ in s.t.llm.violations}, {k["skel_id"] for k in FEASIBLE})


class Refusals(E.Base):
    def test_server_without_the_phrase_ban(self):
        with self.assertRaises(TC.RefuseRealTeacher):
            self.run_served(FE.Served(SKELS, E.good, phrases=False), "--ban-phrases")
        self.assertFalse(os.path.exists(self.out))

    def test_ban_is_on_by_default_and_can_be_turned_off(self):
        """10-04 (round 4): drive.py --serve asks for the phrase ban unless --no-ban-phrases is given."""
        with self.assertRaises(TC.RefuseRealTeacher):
            self.run_served(FE.Served(SKELS, E.good, phrases=False))
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.run_served(FE.Served(SKELS, E.good, phrases=False), "--no-ban-phrases", n=3), 0)
        acc, rej = self.records()
        self.assertTrue(acc + rej)
        self.assertTrue(all(r["run_flags"]["phrases"] is None for r in acc + rej))

    def test_plain_client_refuses_ban_phrases(self):
        with self.assertRaises(SystemExit):
            driver.main(["--endpoint", "http://127.0.0.1:9", "--out", self.out, "--n", "2", "--ban-phrases"])
        self.assertFalse(os.path.exists(self.out))

    def dropped(self, key, *flags):
        s = FE.Served(SKELS, E.good, phrases=True)
        real = s.eng.sampling_for

        def drop(j):
            j["req"] = {k: v for k, v in j["req"].items() if k != key}
            return real(j)
        s.eng.sampling_for = drop
        self.run_served(s, *flags)
        acc, rej = self.records()
        called = [r for r in rej if r["primary"] != "SKEL_INFEASIBLE"]
        self.assertEqual(acc, [])
        self.assertTrue(called)
        self.assertTrue(all(r["primary"] == "TEACHER_ERROR" and "decode mismatch" in r["error"] for r in called))

    def test_a_server_that_drops_the_phrase_ban_is_a_teacher_error(self):
        self.dropped("phrases", "--ban-phrases")

    def test_a_row_without_the_servers_label_rule_is_a_teacher_error(self):
        self.dropped("label_rule", "--structured", "labels")


class Resume(E.Base):
    def test_a_run_dir_of_another_label_rule_is_refused(self):
        self.assertEqual(self.run_served(FE.Served(SKELS, E.good), "--structured", "labels_exact", n=3), 0)
        path = os.path.join(self.out, "decode.json")
        with open(path) as f:
            pins = json.load(f)
        self.assertEqual(pins["label_rule"], "labels-v2")
        old = {k: v for k, v in pins.items() if k not in ("label_rule", "phrases")}    # a dp2 (round 1) run dir
        with open(path, "w") as f:
            json.dump(old, f)
        with self.assertRaisesRegex(SystemExit, "decoding flags differ"):
            self.run_served(FE.Served(SKELS, E.good), "--structured", "labels_exact", n=3)

    def test_a_run_dir_from_before_the_phrase_flag_resumes_with_it_off(self):
        # 10-04: the ban is on by default, so a resume of a run dir pinned before it says --no-ban-phrases
        self.assertEqual(self.run_served(FE.Served(SKELS, E.good), "--preset", "card", "--no-ban-phrases", n=3), 0)
        path = os.path.join(self.out, "decode.json")
        with open(path) as f:
            pins = json.load(f)
        self.assertEqual(pins, {**driver.FLAGS_OFF, "preset": "card"})
        with open(path, "w") as f:
            json.dump({k: v for k, v in pins.items() if k != "phrases"}, f)
        self.assertEqual(self.run_served(FE.Served(SKELS, E.good), "--preset", "card", "--no-ban-phrases", n=4), 0)
        with self.assertRaisesRegex(SystemExit, "decoding flags differ"):
            self.run_served(FE.Served(SKELS, E.good, phrases=True), "--preset", "card", "--ban-phrases", n=5)


if __name__ == "__main__":
    unittest.main()
