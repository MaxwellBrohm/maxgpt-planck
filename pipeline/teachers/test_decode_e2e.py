"""drive.py --serve with the decoding controls, end to end through serve_client, serve_http and serve.py's real
request path to a FAKE engine (fake_engine.py): the engine must see each skeleton's own regex, the cached dash ban
and the preset's values in its SamplingParams, and every record must carry the run flags and the server's account.
CPU only, no model anywhere; DRY.
    cd pipeline/teachers && python3 -B -m unittest test_decode_e2e -v"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import decode  # noqa: E402
import drive  # noqa: E402
import driver  # noqa: E402
import fake_engine as FE  # noqa: E402
import fake_teacher  # noqa: E402
import render_prompt as R  # noqa: E402
import serve  # noqa: E402
import serve_http  # noqa: E402
import shards  # noqa: E402
import skeleton  # noqa: E402
import teacher_client as TC  # noqa: E402

SEED, N = "test-decode-e2e", 10
SKELS = skeleton.shard(SEED, N)
FEASIBLE = [k for k in SKELS if not R.feasible(k)]


def good(sk, prompt, sp):
    return FE.canon(sk)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_decode_e2e_")
        self.out = os.path.join(self.tmp, "run")
        self.orig = TC.TeacherClient, driver.Driver.run

    def tearDown(self):
        TC.TeacherClient, driver.Driver.run = self.orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, url, *extra, n=N):
        return ["--serve", "--allow-real-teacher", "--endpoint", url, "--out", self.out, "--n", str(n),
                "--shard-seed", SEED, "--concurrency", "3", "--timeout", "10", "--http-tries", "1", "--no-fsync",
                "--quiet", *extra]

    def run_served(self, s, *extra, n=N):
        try:
            return drive.main(self.args(s.url, *extra, n=n))
        finally:
            s.close()
            TC.TeacherClient = self.orig[0]

    def records(self):
        acc = list(shards.read(os.path.join(self.out, "accepted"), "accepted"))
        rej = list(shards.read(os.path.join(self.out, "rejects"), "rejects"))
        return acc, rej


class FlagsReachTheEngine(Base):
    def test_structured_ban_and_card_preset(self):
        s = FE.Served(SKELS, good)
        self.assertEqual(self.run_served(s, "--structured", "labels_exact", "--ban-dashes", "--preset", "card"), 0)
        seen = s.t.llm.seen
        self.assertEqual({k["skel_id"] for k, _, _ in seen}, {k["skel_id"] for k in FEASIBLE})
        rx = {k["skel_id"]: decode.label_regex(driver.constraint(k, "labels_exact")["lines"], "\\n") for k in SKELS}
        self.assertEqual(len(set(rx.values())), len(rx), "the fixture's skeletons must have distinct regexes")
        for sk, prompt, sp in seen:
            self.assertEqual(sp.structured_outputs.regex, rx[sk["skel_id"]])
            self.assertEqual(sp.logit_bias, {i: -100.0 for i in FE.BANNED})
            self.assertEqual((sp.temperature, sp.top_p, sp.top_k, sp.min_p), (1.0, 0.95, 64, 0.0))
        acc, rej = self.records()
        self.assertTrue(acc)
        # 10-04 (round 4): the phrase ban is on by default for every served run
        flags = {"structured": "labels_exact", "ban": "dash", "phrases": "ai_ism", "preset": "card", "repair": False}
        self.assertTrue(all(sp.bad_words for _, _, sp in seen))
        self.assertEqual(s.t.llm.violations, [])
        for r in acc + rej:
            self.assertEqual((r["run_flags"], r["attempt_kind"]), (flags, "first" if r["attempt"] == 0 else "retry"))
        for r in acc + [r for r in rej if r["teacher"]]:
            tm = r["provenance"]["teacher"] if "provenance" in r else r["teacher"]
            dec = tm["decode"]
            sk = next(k for k in SKELS if k["skel_id"] == r["skel_id"])
            self.assertEqual(dec["regex_sha256"], decode.sha(rx[r["skel_id"]]))
            self.assertEqual((dec["structured"], dec["line_sep"], dec["preset"]), ("labels_exact", "\\n", "card"))
            self.assertEqual(dec["n_literal"], sum(t["mode"] == "exact" for t in sk["turns"]))
            self.assertEqual(dec["ban"]["ids_sha256"], decode.ids_sha(FE.BANNED))
            self.assertEqual((dec["sampling"]["min_p"], dec["gpu_memory_utilization"]), (0.0, 0.86))
        with open(os.path.join(self.out, "decode.json")) as f:
            self.assertEqual(json.load(f), {**flags, "label_rule": decode.LABEL_RULE, "phrase_rule": decode.PHRASE_RULE})
        with open(os.path.join(self.out, "yield.jsonl")) as f:
            self.assertEqual(json.loads(f.readlines()[-1])["run_flags"], flags)

    def test_off_sends_nothing_extra(self):
        s = FE.Served(SKELS, good)
        self.assertEqual(self.run_served(s), 0)
        self.assertTrue(s.t.llm.seen)
        for sk, prompt, sp in s.t.llm.seen:
            self.assertFalse(hasattr(sp, "structured_outputs") or hasattr(sp, "logit_bias"))
            self.assertEqual((sp.temperature, sp.min_p), (1.0, 0.05))
        acc, _ = self.records()
        dec = acc[0]["provenance"]["teacher"]["decode"]
        self.assertEqual((dec["preset"], dec["structured"], dec["ban"], dec["regex_sha256"]),
                         ("yld0926", None, None, None))


class Refusals(Base):
    def refused(self, s, *extra):
        with self.assertRaises(TC.RefuseRealTeacher):
            self.run_served(s, *extra)
        self.assertFalse(os.path.exists(self.out))

    def test_server_without_what_the_run_asks_for(self):
        self.refused(FE.Served(SKELS, good, ban=False), "--ban-dashes")
        self.refused(FE.Served(SKELS, good, backend=None), "--structured", "labels")
        self.refused(FE.Served(SKELS, good), "--preset", "shared")        # D4: Gemma has no shared preset

    def test_plain_client_refuses_the_controls(self):
        for extra in (["--structured", "labels"], ["--ban-dashes"], ["--preset", "card"]):
            with self.assertRaises(SystemExit):
                driver.main(["--endpoint", "http://127.0.0.1:9", "--out", self.out, "--n", "2", *extra])
            self.assertFalse(os.path.exists(self.out))

    def test_resume_with_other_flags_is_refused(self):
        self.assertEqual(self.run_served(FE.Served(SKELS, good), n=3), 0)
        with self.assertRaisesRegex(SystemExit, "decoding flags differ"):
            self.run_served(FE.Served(SKELS, good), "--preset", "card", n=3)
        with open(os.path.join(self.out, "run.json")) as f:
            self.assertEqual(json.load(f)["sessions"], 1, "a refused resume must not count as a session")
        os.remove(os.path.join(self.out, "decode.json"))                   # a run dir from before decode.json
        with self.assertRaisesRegex(SystemExit, "before decode.json"):
            self.run_served(FE.Served(SKELS, good), "--preset", "card", n=3)

    def test_a_server_that_drops_the_constraint_is_a_teacher_error(self):
        s = FE.Served(SKELS, good)
        real = s.eng.sampling_for

        def drop(j):
            j["req"] = {k: v for k, v in j["req"].items() if k not in ("constraint", "regex")}
            return real(j)
        s.eng.sampling_for = drop
        self.run_served(s, "--structured", "labels")
        acc, rej = self.records()
        self.assertEqual(acc, [])
        called = [r for r in rej if r["primary"] != "SKEL_INFEASIBLE"]
        self.assertTrue(called)
        self.assertTrue(all(r["primary"] == "TEACHER_ERROR" and "decode mismatch" in r["error"] for r in called))


class Requests(unittest.TestCase):
    INFO = {"line_sep": "\\n+", "structured_backend": "xgrammar", "presets": ["card", "yld0926"], "dash_ban": None}

    def test_ministral_separator_and_bad_requests(self):
        sk = FEASIBLE[0]
        req = serve_http.decode_request({"constraint": driver.constraint(sk, "labels")}, self.INFO)
        self.assertTrue(req["regex"].endswith("\\n+END"))
        self.assertEqual(req["regex"], decode.label_regex(driver.constraint(sk, "labels")["lines"], "\\n+"))
        for body in ({"ban": "dash"}, {"preset": "shared"}, {"constraint": {"mode": "labels", "lines": [["U1", "x"]]}},
                     {"constraint": {"mode": "labels", "lines": [["Q1", None]]}}):
            with self.assertRaises(ValueError, msg=body):
                serve_http.decode_request(body, self.INFO)
        with self.assertRaises(ValueError):
            serve_http.decode_request({"constraint": driver.constraint(sk, "labels")},
                                      {**self.INFO, "structured_backend": None})

    def test_server_info_line_separator_per_teacher(self):
        for name, sep in (("ministral-3-8b", "\\n+"), ("qwen3.5-9b", "\\n"), ("gemma-4-12b", "\\n")):
            info = serve_http.server_info(serve.Teacher(name, serve.TEACHERS[name], None), {}, None, "xgrammar", 0.86)
            self.assertEqual((info["line_sep"], info["presets"][-1]), (sep, "yld0926"))


if __name__ == "__main__":
    unittest.main()
