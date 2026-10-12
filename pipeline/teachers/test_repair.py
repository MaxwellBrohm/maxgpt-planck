"""The driver's --repair flag (2026-09-27; SPEC 1 pilot flag, off by default): which rejects are near misses, the note
that names what failed, PROMPT_ECHO coverage of that note, and end to end through drive.py --serve and a FAKE
engine (fake_engine.py) whose policy fixes a near miss only when the prompt carries the note. CPU only; DRY.
    cd pipeline/teachers && python3 -B -m unittest test_repair -v"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import checker  # noqa: E402
import drive  # noqa: E402
import drive_report  # noqa: E402
import driver  # noqa: E402
import fake_engine as FE  # noqa: E402
import fake_teacher  # noqa: E402
import parse  # noqa: E402
import render_prompt as R  # noqa: E402
import shards  # noqa: E402
import skeleton  # noqa: E402
import teacher_client as TC  # noqa: E402

SEED, N = "test-repair", 12
SKELS = skeleton.shard(SEED, N)
FEAS = [k for k in SKELS if not R.feasible(k)]


def rec(*hits):
    return {"codes": sorted({h[0] for h in hits}, key=checker.RANK.get), "hits": [list(h) for h in hits]}


def span_case(sk):
    """(turn, item, text): the canonical render with one must_include item swapped out, when that leaves exactly
    one REQ_SPAN hit and nothing else."""
    raw = fake_teacher.raw(sk)
    turns = parse.parse(raw, parse.plan(sk))["turns"]
    for t in sk["turns"]:
        if t["mode"] == "guided" and t["must_include"] and t["must_include"][0] in turns[t["i"]]:
            x = t["must_include"][0]
            text = parse.serialize(sk, {**turns, t["i"]: turns[t["i"]].replace(x, "the thing")})
            res = checker.run(sk, text)
            if driver.near_miss(rec(*res["hits"])) == "span":
                return t["i"], x, text
    return None


NO_END = [FEAS[0]["skel_id"], FEAS[1]["skel_id"]]
SPAN = {}
for _k in FEAS[2:]:
    if len(SPAN) < 2 and span_case(_k):
        SPAN[_k["skel_id"]] = span_case(_k)
DASHED = [k["skel_id"] for k in FEAS if k["skel_id"] not in NO_END and k["skel_id"] not in SPAN][:2]
LABS = {k["skel_id"]: {i: lab for lab, i, _ in parse.plan(k)} for k in SKELS}


def policy(sk, prompt, sp):
    sid, raw = sk["skel_id"], fake_teacher.raw(sk)
    if driver.REPAIR_HEAD in prompt:
        return raw
    if sid in NO_END:
        return raw.rsplit("\nEND", 1)[0]
    if sid in SPAN:
        return SPAN[sid][2]
    if sid in DASHED:
        return raw.replace("\nA", " \u2014 yes\nA", 1)
    return raw


class Units(unittest.TestCase):
    def test_fixture_is_not_vacuous(self):
        self.assertEqual(SKELS[0]["skel_id"], NO_END[0])
        self.assertEqual((len(SPAN), len(DASHED)), (2, 2))
        for sid in DASHED:
            sk = next(k for k in SKELS if k["skel_id"] == sid)
            res = checker.run(sk, policy(sk, "", None))
            self.assertIn("DASH", res["codes"])
            self.assertIsNone(driver.near_miss(rec(*res["hits"])))
        sk = SKELS[0]
        self.assertEqual(checker.run(sk, policy(sk, "", None))["codes"], ["FORMAT_LINES"])

    def test_near_miss_classes(self):
        self.assertEqual(driver.near_miss(rec(("FORMAT_LINES", None, "no END"))), "form")
        self.assertEqual(driver.near_miss(rec(("FORMAT_LINES", None, "no END"), ("FORMAT_EXTRA", None, "x"))), "form")
        self.assertEqual(driver.near_miss(rec(("REQ_SPAN", 3, "Perth"))), "span")
        self.assertEqual(driver.near_miss(rec(("FORBID_SPAN", 3, "Perth"))), "span")
        self.assertEqual(driver.near_miss(rec(("REQ_SPAN", 3, "Perth"), ("VOCAB_OOL", 3, "zz"))), "span")
        for hits in ([("REQ_SPAN", 3, "a"), ("REQ_SPAN", 5, "b")], [("REQ_SPAN", 3, "a"), ("FORMAT_LINES", None, "x")],
                     [("REQ_SPAN", None, "a")], [("REQ_WORD", None, "x")], [("DASH", 2, "x")],
                     [("VOCAB_OOL", 2, "zz")], [("FORBID_SPAN", 1, "a"), ("ANSWER_WRONG", 1, "a")]):
            self.assertIsNone(driver.near_miss(rec(*hits)), hits)
        self.assertIsNone(driver.near_miss({"codes": ["TEACHER_ERROR"], "primary": "TEACHER_ERROR"}))

    def test_note_names_each_failure_without_dashes(self):
        sk = SKELS[0]
        i = next(t["i"] for t in sk["turns"] if t["role"] == "assistant")
        lab = LABS[sk["skel_id"]][i]
        for hits, want in (([("REQ_SPAN", i, "Perth")], f'Line {lab} must include "Perth" exactly as written.'),
                           ([("FORBID_SPAN", i, "Perth")], f'Line {lab} must not include "Perth".'),
                           ([("FORMAT_LINES", None, "no END")], "The last line must be END, on a line of its own."),
                           ([("FORMAT_LINES", None, "missing ['A4', 'U5']")], "Missing last time: A4, U5."),
                           ([("FORMAT_LINES", None, "duplicate A6")], "A6 came twice last time."),
                           ([("FORMAT_EXTRA", None, "Sure thing")], "Write nothing before the first label or after END."),
                           ([("FORMAT_WRAP", None, "more")], "Keep each script line on a single line")):
            note = driver.repair_note(sk, rec(*hits))
            self.assertTrue(note.startswith(driver.REPAIR_HEAD + "\n"))
            self.assertIn(want, note)
            self.assertNotRegex(note, "[\u2012-\u2015\u2212]|--| - ")
            self.assertNotIn("Sure thing", note)                      # never the teacher's own words

    def test_prompt_echo_covers_the_note(self):
        sk = SKELS[0]
        built = R.build(sk)
        rb = driver.repair_built(built, driver.repair_note(sk, rec(("FORMAT_LINES", None, "no END"))))
        self.assertEqual(rb["prompt"], built["prompt"] + "\n\n" + rb["blocks"]["tail"].split("\n\n")[-1])
        self.assertNotEqual(rb["prompt_sha256"], built["prompt_sha256"])
        raw = fake_teacher.raw(sk)
        self.assertTrue(checker.run(sk, raw, rb)["ok"])
        turns = parse.parse(raw, parse.plan(sk))["turns"]
        i = next(t["i"] for t in sk["turns"] if t["role"] == "assistant" and t["mode"] == "guided")
        planted = parse.serialize(sk, {**turns, i: "Sure, I will write the whole chat again, every line."})
        self.assertIn("PROMPT_ECHO", checker.run(sk, planted, rb)["codes"])
        self.assertNotIn("PROMPT_ECHO", checker.run(sk, planted, built)["codes"])


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_repair_")
        self.out = os.path.join(self.tmp, "run")
        self.orig = TC.TeacherClient, driver.Driver.run

    def tearDown(self):
        TC.TeacherClient, driver.Driver.run = self.orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_served(self, *extra):
        s = FE.Served(SKELS, policy)
        try:
            rc = drive.main(["--serve", "--allow-real-teacher", "--endpoint", s.url, "--out", self.out, "--n", str(N),
                             "--shard-seed", SEED, "--timeout", "10", "--http-tries", "1", "--no-fsync", "--quiet",
                             *extra])
        finally:
            s.close()
            TC.TeacherClient = self.orig[0]
        return rc, s.t.llm.seen

    def records(self):
        return (list(shards.read(os.path.join(self.out, "accepted"), "accepted")),
                list(shards.read(os.path.join(self.out, "rejects"), "rejects")))

    def last_snap(self):
        with open(os.path.join(self.out, "yield.jsonl")) as f:
            return json.loads(f.readlines()[-1])

    def test_repair_names_the_failure_and_is_counted_apart(self):
        rc, seen = self.run_served("--repair", "--concurrency", "3")
        self.assertEqual(rc, 0)
        acc, rej = self.records()
        for sid in NO_END + list(SPAN):
            r = [x for x in acc if x["skel_id"] == sid]
            self.assertEqual(len(r), 1, sid)
            r = r[0]
            self.assertEqual((r["attempt"], r["attempt_kind"], r["repair"]["of_attempt"]), (1, "repair", 0))
            self.assertEqual(r["repair"]["near_miss"], "form" if sid in NO_END else "span")
            if sid in SPAN:
                i, x, _ = SPAN[sid]
                self.assertIn(f'Line {LABS[sid][i]} must include "{x}" exactly as written.', r["repair"]["note"])
            sk = next(k for k in SKELS if k["skel_id"] == sid)
            sent = [p for k, p, _ in seen if k["skel_id"] == sid and driver.REPAIR_HEAD in p]
            self.assertEqual(sent, [R.build(sk)["prompt"] + "\n\n" + r["repair"]["note"]])
        for sid in DASHED:
            rs = sorted((x for x in rej if x["skel_id"] == sid), key=lambda x: x["attempt"])
            self.assertEqual([x["attempt_kind"] for x in rs], ["first", "retry"])
            self.assertEqual(rs[0]["teacher"]["prompt_sha256"], rs[1]["teacher"]["prompt_sha256"])
        bk = self.last_snap()["by_attempt_kind"]
        self.assertEqual(bk["repair"], {"attempted": 4, "accepted": 4, "yield": 1.0})
        self.assertEqual((bk["retry"]["attempted"], bk["retry"]["accepted"]), (2, 0))
        self.assertEqual(bk["first"]["attempted"], N)
        self.assertEqual(drive_report.report(self.out, "fake")[0]["by_attempt_kind"], bk)

    def test_without_the_flag_the_retry_is_blind(self):
        rc, seen = self.run_served("--concurrency", "3")
        self.assertEqual(rc, 0)
        self.assertFalse([p for _, p, _ in seen if driver.REPAIR_HEAD in p])
        acc, rej = self.records()
        for sid in NO_END + list(SPAN):
            self.assertEqual(sorted(x["attempt_kind"] for x in rej if x["skel_id"] == sid), ["first", "retry"])
        self.assertEqual(self.last_snap()["by_attempt_kind"]["repair"]["attempted"], 0)

    def test_resume_rebuilds_the_pending_repair(self):
        rc, _ = self.run_served("--repair", "--concurrency", "1", "--stop-after", "1")
        self.assertEqual(rc, 3)
        acc, rej = self.records()
        self.assertEqual([(r["skel_id"], r["attempt"]) for r in acc + rej], [(NO_END[0], 0)])
        rc, seen = self.run_served("--repair", "--concurrency", "1")
        self.assertEqual(rc, 0)
        self.assertTrue([p for k, p, _ in seen if k["skel_id"] == NO_END[0] and driver.REPAIR_HEAD in p])
        acc, _ = self.records()
        r = next(x for x in acc if x["skel_id"] == NO_END[0])
        self.assertEqual((r["attempt"], r["attempt_kind"]), (1, "repair"))
        self.assertEqual(self.last_snap()["by_attempt_kind"]["first"]["attempted"], N)


if __name__ == "__main__":
    unittest.main()
