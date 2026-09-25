"""Step 4 end-to-end tests: driver.py against the FAKE local stub (fake_server.py). No model is loaded or called and
no real server is contacted: the client refuses any endpoint that is not the stub.

The stub plan (stub_plan.plan) fixes, for every (skeleton, attempt), the canned body the stub serves and the record
the driver must write; driver_fixtures.verify compares the run directory with it outcome by outcome."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import driver_fixtures as DF  # noqa: E402  (puts the pipeline on sys.path)
import driver
import fake_server
import stub_plan
import teacher_client as TC

HTTP_TRIES = 3


def run_driver(out, url, skel_file, n, mode="completions", **kw):
    client = TC.TeacherClient(url, "fake-teacher", mode=mode, http_tries=HTTP_TRIES, backoff=0, timeout=10)
    cfg = {"out": out, "n": n, "skeletons": skel_file, "concurrency": 5, "quiet": True, "fsync": False,
           "log_every": 0.2, **kw}
    return driver.Driver(cfg, client).run()


class NotAStub(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _404(self):
        NotAStub.seen.append((self.command, self.path))
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _404


class TestClientSafety(unittest.TestCase):
    def test_non_loopback_refused_before_any_socket(self):
        orig = urllib.request.urlopen

        def boom(*a, **k):
            raise AssertionError("a socket was opened")
        urllib.request.urlopen = boom
        try:
            for url in ("http://192.0.2.1:8000", "http://10.0.0.5:1234", "https://example.invalid"):
                with self.assertRaises(TC.RefuseRealTeacher):
                    TC.TeacherClient(url, "google/gemma-4-12b-qat").check_endpoint()
            TC.TeacherClient("http://192.0.2.1:8000", "m", allow_real=True).check_endpoint()   # no request made
        finally:
            urllib.request.urlopen = orig

    def test_loopback_server_that_is_not_the_stub_is_refused(self):
        NotAStub.seen = []
        srv = ThreadingHTTPServer(("127.0.0.1", 0), NotAStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        tmp = tempfile.mkdtemp()
        try:
            with self.assertRaises(TC.RefuseRealTeacher):
                TC.TeacherClient(url, "google/gemma-4-12b-qat").check_endpoint()
            skels = DF.skeleton.shard("safety", 3)
            DF.write_skeletons(skels, os.path.join(tmp, "sk.jsonl"))
            with self.assertRaises(TC.RefuseRealTeacher):
                run_driver(os.path.join(tmp, "run"), url, os.path.join(tmp, "sk.jsonl"), 3)
            self.assertEqual(NotAStub.seen, [("GET", "/planck-stub")] * 2, "only the identity probe may be sent")
            self.assertFalse(os.path.exists(os.path.join(tmp, "run", "accepted")))
        finally:
            srv.shutdown()
            srv.server_close()
            shutil.rmtree(tmp)

    def test_licenses(self):
        self.assertEqual(TC.license_of("google/gemma-4-12b-qat"), "Apache-2.0")
        self.assertEqual(TC.license_of("mistralai/Ministral-3-8B-Instruct"), "Apache-2.0")
        self.assertEqual(TC.license_of("qwen3.5-9b"), "UNKNOWN")


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.skels, cls.groups = DF.corpus(70)
        cls.skel_file = os.path.join(cls.tmp, "skeletons.jsonl")
        DF.write_skeletons(cls.skels, cls.skel_file)
        cls.plan = stub_plan.plan(cls.skels, seed=7, http_tries=HTTP_TRIES, groups=cls.groups)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def stub(self, plan=None, delay=0.0):
        stub, server, url = fake_server.start(plan or self.plan, delay=delay)
        self.addCleanup(fake_server.stop, server)
        return stub, url

    def test_plan_scenarios_all_present(self):
        sc = {p["scenario"] for p in self.plan["expected"]["per_skel"].values()}
        self.assertLessEqual(set(stub_plan.SCENARIOS) | {"infeasible"}, sc)

    def test_counts_match_plan_exactly(self):
        stub, url = self.stub(delay=0.005)
        out = os.path.join(self.tmp, "run_full")
        run_driver(out, url, self.skel_file, len(self.skels))
        DF.verify(self, out, self.plan, self.skels)
        self.assertEqual(len(stub.log), self.plan["expected"]["requests"], stub.counts())
        self.assertNotIn("unplanned", stub.counts())
        self.assertGreater(stub.max_inflight, 1)
        self.assertLessEqual(stub.max_inflight, 5)

    def test_resume_after_stop_and_torn_line(self):
        stub, url = self.stub()
        out = os.path.join(self.tmp, "run_stop")
        snap = run_driver(out, url, self.skel_file, len(self.skels), stop_after=25)
        self.assertFalse(snap["final"])
        acc, rej, _ = DF.load_run(out)
        self.assertEqual(len(acc) + len(rej), 25)
        with open(os.path.join(out, "accepted", "accepted-00000.jsonl"), "a") as f:
            f.write('{"conv_id": "torn-write", "turns": [{"role": "us')
        run_driver(out, url, self.skel_file, len(self.skels))
        _, _, final = DF.verify(self, out, self.plan, self.skels)
        self.assertTrue(final["torn_bytes"], "the torn line was not reported")
        self.assertEqual(final["dup_pairs"], [])
        self.assertNotIn("unplanned", stub.counts())

    def test_resume_after_sigkill(self):
        slow = json.loads(json.dumps(self.plan))
        for steps in slow["responses"].values():
            for s in steps:
                s["sleep"] = 0.03
        stub, url = self.stub(slow)
        out = os.path.join(self.tmp, "run_kill")
        cmd = [sys.executable, "-B", os.path.join(DF.PIPE, "driver.py"), "--out", out, "--endpoint", url,
               "--model", "fake-teacher", "--skeletons", self.skel_file, "--n", str(len(self.skels)),
               "--concurrency", "4", "--http-tries", str(HTTP_TRIES), "--backoff", "0", "--quiet"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        while time.time() < deadline:
            if _records(out) >= 30:
                break
            time.sleep(0.02)
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
        killed_at = _records(out)
        self.assertGreaterEqual(killed_at, 30)
        self.assertLess(killed_at, self.plan["expected"]["accepted"] + self.plan["expected"]["rejected"])
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        self.assertEqual(r.returncode, 0, r.stderr.decode()[-2000:])
        DF.verify(self, out, slow, self.skels)

    def test_checker_crash_is_a_reject_not_a_dead_run(self):
        skels = DF.skeleton.shard("crash", 6)
        plan = stub_plan.plan(skels, seed=5, http_tries=HTTP_TRIES, weights={"defect_clean": 1})
        stub, url = self.stub(plan)
        victim = next(s["skel_id"] for s in skels if plan["expected"]["per_skel"][s["skel_id"]]["scenario"]
                      == "defect_clean")
        real, calls = driver.checker.run, []

        def flaky_checker(sk, raw, built=None, wordlist=None):
            calls.append(sk["skel_id"])
            if sk["skel_id"] == victim and calls.count(victim) == 1:
                raise RuntimeError("planted checker bug")
            return real(sk, raw, built, wordlist)
        driver.checker.run = flaky_checker
        self.addCleanup(setattr, driver.checker, "run", real)
        sk_file = os.path.join(self.tmp, "crash.jsonl")
        DF.write_skeletons(skels, sk_file)
        out = os.path.join(self.tmp, "run_crash")
        run_driver(out, url, sk_file, len(skels))
        acc, rej, snaps = DF.load_run(out)
        mine = sorted((r["attempt"], r.get("primary", "accept")) for r in acc + rej if r["skel_id"] == victim)
        self.assertEqual(mine, [(0, "CHECKER_ERROR"), (1, "accept")])
        self.assertIn("planted checker bug", next(r for r in rej if r["skel_id"] == victim)["error"])
        self.assertEqual(len(acc), plan["expected"]["accepted"])
        self.assertTrue(snaps[-1]["final"])
        self.assertEqual(snaps[-1]["by_primary"].get("CHECKER_ERROR"), 1)

    def test_chat_mode(self):
        skels = DF.skeleton.shard("chat", 14)
        plan = stub_plan.plan(skels, seed=3, http_tries=HTTP_TRIES)
        stub, url = self.stub(plan)
        out = os.path.join(self.tmp, "run_chat")
        sk_file = os.path.join(self.tmp, "chat.jsonl")
        DF.write_skeletons(skels, sk_file)
        run_driver(out, url, sk_file, len(skels), mode="chat")
        DF.verify(self, out, plan, skels)
        self.assertEqual(len(stub.log), plan["expected"]["requests"])


def _records(out):
    n = 0
    for sub in ("accepted", "rejects"):
        d = os.path.join(out, sub)
        if os.path.isdir(d):
            for name in os.listdir(d):
                with open(os.path.join(d, name), "rb") as f:
                    n += f.read().count(b"\n")
    return n


if __name__ == "__main__":
    unittest.main()
