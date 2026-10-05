"""drive.py --serve when the served teacher goes away mid-run (CPU only, the FAKE engine of test_drive.py).

A server that stops after a few renders must stop the driver (serve_client.ServerGone), not turn every remaining
attempt into a TEACHER_ERROR reject; the same command then resumes against a new server and finishes the run.
    cd pipeline/teachers && python3 -B -m unittest test_drive_gone -v"""
import json
import os
import shutil
import socket
import tempfile
import threading
import unittest
import urllib.error

import test_drive as TD  # noqa: E402  (sets sys.path and the vllm.inputs stand-in)
import drive  # noqa: E402
import drive_report  # noqa: E402
import driver  # noqa: E402
import serve_client  # noqa: E402
import shards  # noqa: E402
import teacher_client as TC  # noqa: E402


def stop_after(s, n):
    """make Served s go away (503 while stopping, then connection refused) after n finished renders."""
    orig, count = s.eng._finish, [0]

    def kill():
        s.stop.set()
        s.eng.halt.set()
        s.srv.shutdown()
        s.srv.server_close()

    def fin(o):
        orig(o)
        count[0] += 1
        if count[0] == n:
            threading.Thread(target=kill, daemon=True).start()
    s.eng._finish = fin


class ServerGoneStopsTheRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_drive_gone_")
        self.out = os.path.join(self.tmp, "run")
        self.orig_client = TC.TeacherClient

    def tearDown(self):
        TC.TeacherClient = self.orig_client
        driver.Driver.run = drive._RUN
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, url):
        return ["--serve", "--allow-real-teacher", "--endpoint", url, "--out", self.out, "--n", str(TD.N),
                "--shard-seed", TD.SEED, "--concurrency", "4", "--timeout", "10", "--http-tries", "1", "--no-fsync",
                "--quiet", "--no-ban-phrases"]          # TD.Served offers no phrase ban (on by default since 10-04)

    def records(self):
        acc = list(shards.read(os.path.join(self.out, "accepted"), "accepted"))
        rej = list(shards.read(os.path.join(self.out, "rejects"), "rejects"))
        return acc, rej

    def test_gone_stops_then_resumes(self):
        s1 = TD.Served()
        stop_after(s1, 5)
        try:
            with self.assertRaises(serve_client.ServerGone):
                drive.main(self.args(s1.url))
        finally:
            s1.close()
        acc, rej = self.records()
        self.assertGreaterEqual(len(acc) + len(rej), 3, "the fixture must finish some renders before it stops")
        stray = [r for r in rej if r["primary"] == "TEACHER_ERROR" and r["skel_id"] not in TD.BROKEN]
        self.assertEqual(stray, [], "a gone server was recorded as TEACHER_ERROR rejects")
        before = len(acc) + len(rej)
        TC.TeacherClient = self.orig_client
        s2 = TD.Served()
        try:
            rc = drive.main(self.args(s2.url))
        finally:
            s2.close()
        self.assertEqual(rc, 0)
        acc, rej = self.records()
        self.assertGreater(len(acc) + len(rej), before)
        stray = [r for r in rej if r["primary"] == "TEACHER_ERROR" and r["skel_id"] not in TD.BROKEN]
        self.assertEqual(stray, [])
        done = {r["skel_id"] for r in acc} | {r["skel_id"] for r in rej if r["attempt"] == 1 or
                                               r["primary"] == "SKEL_INFEASIBLE"}
        self.assertEqual(done, {k["skel_id"] for k in TD.SKELS})
        with open(os.path.join(self.out, "yield.jsonl")) as f:
            snaps = [json.loads(ln) for ln in f if ln.strip()]
        self.assertTrue(snaps[-1]["final"])
        self.assertIn("server gone", [x.get("stopped") for x in snaps])
        rep, _, _ = drive_report.report(self.out, "fake")
        self.assertEqual(rep["sessions"], 2)
        self.assertGreater(rep["elapsed_s_all_sessions"], snaps[-1]["session"]["elapsed_s"])
        self.assertEqual(rep["attempts"], snaps[-1]["attempts"])


class GoneUnits(unittest.TestCase):
    """which failures on the last try mean the server is gone (stop the run) and which are one render's failure."""
    def client(self, *effects):
        c = serve_client.ServeClient("http://127.0.0.1:9", allow_real=True, http_tries=len(effects), backoff=0)
        c._checked, it = True, iter(effects)

        def post(path, body):
            e = next(it)
            if isinstance(e, BaseException):
                raise e
            return e
        c._post = post
        return c

    def render(self, *effects):
        return self.client(*effects).render({"prompt": "p", "max_tokens": 5}, 1)

    def test_refused_is_gone(self):
        with self.assertRaises(serve_client.ServerGone):
            self.render(urllib.error.URLError(ConnectionRefusedError(111, "refused")))

    def test_broken_pipe_is_gone(self):
        with self.assertRaises(serve_client.ServerGone):
            self.render(urllib.error.URLError(BrokenPipeError(32, "Broken pipe")))

    def test_einval_is_gone(self):
        with self.assertRaises(serve_client.ServerGone):
            self.render(urllib.error.URLError(OSError(22, "Invalid argument")))

    def test_wrapped_timeout_is_not_gone(self):
        with self.assertRaises(TC.TeacherError):
            self.render(urllib.error.URLError(TimeoutError("timed out")))

    def test_429_is_not_gone(self):
        with self.assertRaises(TC.TeacherError):
            self.render(urllib.error.HTTPError("u", 429, "busy", {}, None))

    def test_503_is_gone(self):
        with self.assertRaises(serve_client.ServerGone):
            self.render(urllib.error.HTTPError("u", 503, "stopping", {}, None))

    def test_timeout_is_not_gone(self):
        with self.assertRaises(TC.TeacherError):
            self.render(socket.timeout("slow"), socket.timeout("slow"))

    def test_gone_then_answered_is_not_gone(self):
        with self.assertRaises(TC.TeacherError):
            self.render(urllib.error.HTTPError("u", 503, "stopping", {}, None), {"finish": "stop"})


if __name__ == "__main__":
    unittest.main()
