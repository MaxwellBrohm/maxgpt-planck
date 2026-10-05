"""Tests for drive.py --serve, serve_client.py and serve_http.py's engine loop. CPU only, no model anywhere: a FAKE
engine stands in for vLLM (enqueue returns vLLM-style internal ids "<n>-<8 hex>", step finishes requests after a
random number of steps and reports the external id "<n>", as vLLM 0.30 does). Renders come from fake_teacher.raw
(a correct FAKE render) or a broken line, so both accepted and rejected records are written.
    cd pipeline/teachers && python3 -B -m unittest test_drive -v"""
import itertools
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
try:
    import vllm.inputs  # noqa: F401
except ImportError:  # the Mac: a stand-in for the one name serve_http imports from vllm
    _vi = types.ModuleType("vllm.inputs")
    _vi.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.inputs"] = _vi
import drive  # noqa: E402
import fake_teacher  # noqa: E402
import render_prompt as R  # noqa: E402
import serve  # noqa: E402
import serve_client  # noqa: E402
import serve_http  # noqa: E402
import shards  # noqa: E402
import skeleton  # noqa: E402
import teacher_client as TC  # noqa: E402

SEED, N = "test-drive", 12
SKELS = skeleton.shard(SEED, N)
INDEX = {R.build(k)["prompt"]: i for i, k in enumerate(SKELS)}
BROKEN = {SKELS[3]["skel_id"]}           # its render breaks a wire invariant: TEACHER_ERROR, never a checked text


class FakeEngine:
    def __init__(self):
        self.live, self.internal_ids = {}, []

    def has_unfinished_requests(self):
        return bool(self.live)

    def step(self):
        time.sleep(0.002)
        out = []
        for rid in list(self.live):
            self.live[rid][0] -= 1
            if self.live[rid][0] <= 0:
                text = self.live.pop(rid)[1]
                c = types.SimpleNamespace(text=text, token_ids=[1] * len(text.split()), finish_reason="stop")
                out.append(types.SimpleNamespace(request_id=rid.rsplit("-", 1)[0], finished=True, outputs=[c]))
        return out


class FakeLLM:
    def __init__(self):
        self.llm_engine, self.count = FakeEngine(), itertools.count()

    def enqueue(self, prompts, params, use_tqdm=False):
        ids = []
        for p, sp in zip(prompts, params):
            rid = f"{next(self.count)}-{random.randrange(16 ** 8):08x}"
            sk = SKELS[p["prompt_token_ids"][0]]
            good = sp["seed"] % 2 == 0
            self.llm_engine.live[rid] = [random.randint(1, 5), fake_teacher.raw(sk) if good else "U1: hi\nEND"]
            self.llm_engine.internal_ids.append(rid)
            ids.append(rid)
        return ids


def fake_render(t, prompt):
    i = INDEX[prompt]
    if SKELS[i]["skel_id"] in BROKEN:
        raise serve.RenderError("fake: bos twice")
    return [i], prompt


class Served:
    """serve_http's engine thread and HTTP handler around the fake engine, on a free loopback port."""
    def __init__(self, license="Apache-2.0", status="dry"):
        self.saved = serve.render, serve._params, serve.thought_in
        serve.render = fake_render
        serve._params = lambda t, s, n, seed: ({"seed": seed, **s}, False)
        serve.thought_in = lambda t, text, ids: []
        cfg = serve.TEACHERS["gemma-4-12b"]
        self.t = types.SimpleNamespace(name="gemma-4-12b", cfg=cfg, llm=FakeLLM(), thought_ids=set(),
                                       engine={"max_model_len": 2048})
        self.eng = serve_http.Engine(self.t, {})
        threading.Thread(target=self.eng.run, daemon=True).start()
        self.info = {"planck_serve": True, "status": status, "teacher": "gemma-4-12b", "repo": cfg["repo"],
                     "revision": cfg["revision"], "license": license, "model": f"{cfg['repo']}@{cfg['revision']}"}
        self.stop = threading.Event()
        self.srv = serve_http.ThreadingHTTPServer(("127.0.0.1", 0), serve_http.make_handler(self.eng, self.info,
                                                                                            self.stop))
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.eng.halt.set()
        self.srv.shutdown()
        self.srv.server_close()
        serve.render, serve._params, serve.thought_in = self.saved


class DriveServe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_drive_")
        self.out = os.path.join(self.tmp, "run")
        self.orig_client = TC.TeacherClient

    def tearDown(self):
        TC.TeacherClient = self.orig_client
        shutil.rmtree(self.tmp, ignore_errors=True)

    def args(self, url, *extra):       # this fake server offers no phrase ban (on by default since 10-04)
        return ["--serve", "--endpoint", url, "--out", self.out, "--n", str(N), "--shard-seed", SEED,
                "--concurrency", "4", "--timeout", "10", "--http-tries", "1", "--no-fsync", "--quiet",
                "--no-ban-phrases", *extra]

    def test_serve_needs_allow_real_teacher(self):
        with self.assertRaises(SystemExit):
            drive.main(self.args("http://127.0.0.1:9"))
        self.assertFalse(os.path.exists(self.out))

    def test_client_refuses_without_allow_real_and_off_loopback(self):
        s = Served()                       # a live, valid server: only the guard under test can refuse
        try:
            with self.assertRaisesRegex(TC.RefuseRealTeacher, "allow-real-teacher"):
                serve_client.ServeClient(s.url, allow_real=False).check_endpoint()
            c = serve_client.ServeClient(s.url, allow_real=True)
            c.check_endpoint()
            self.assertEqual(c.license, "Apache-2.0")
        finally:
            s.close()
        with self.assertRaisesRegex(TC.RefuseRealTeacher, "loopback"):
            serve_client.ServeClient("http://192.0.2.1:9", allow_real=True, timeout=1).check_endpoint()

    def test_license_not_allowed_writes_nothing(self):
        s = Served(license="CC-BY-NC-4.0")
        try:
            with self.assertRaises(TC.RefuseRealTeacher):
                drive.main(self.args(s.url, "--allow-real-teacher"))
            self.assertFalse(os.path.exists(self.out))
        finally:
            s.close()

    def test_server_not_marked_dry_is_refused(self):
        s = Served(status="live")
        try:
            with self.assertRaisesRegex(TC.RefuseRealTeacher, "not a dry serve_http"):
                serve_client.ServeClient(s.url, allow_real=True).check_endpoint()
        finally:
            s.close()

    def test_end_to_end_records(self):
        s = Served()
        try:
            rc = drive.main(self.args(s.url, "--allow-real-teacher"))
        finally:
            s.close()
        self.assertEqual(rc, 0)
        acc = list(shards.read(os.path.join(self.out, "accepted"), "accepted"))
        rej = list(shards.read(os.path.join(self.out, "rejects"), "rejects"))
        self.assertTrue(acc and rej)
        broken = [r for r in rej if r["skel_id"] in BROKEN]
        self.assertTrue(broken and all(r["primary"] == "TEACHER_ERROR" and "render_error" in r["error"]
                                       for r in broken))
        other = [r for r in rej if r["skel_id"] not in BROKEN]
        self.assertFalse([r for r in other if r["primary"] == "TEACHER_ERROR"], "a served request failed")
        model = s.info["model"]
        for r in acc:
            self.assertEqual(r["provenance"]["teacher"]["mode"], "serve")
            self.assertEqual(r["provenance"]["teacher"]["model"], model)
            self.assertEqual(r["provenance"]["teacher"]["license"], "Apache-2.0")
            self.assertFalse(r["trainable"])
            self.assertIn("FAKE_PROVENANCE", r["blocked"])
        self.assertTrue(all("-" in i for i in s.t.llm.llm_engine.internal_ids))
        self.assertLessEqual(s.eng.st["max_inflight"], 4)
        self.assertEqual(s.eng.st["done"] + s.eng.st["render_errors"], s.eng.st["received"])

    def test_a_refused_request_fails_alone(self):
        """vLLM refusing one request, or a prompt with no room under max_model_len, fails that request only."""
        s = Served()
        try:
            real = s.t.llm.enqueue

            def enqueue(prompts, params, use_tqdm=False):
                if prompts[0]["prompt_token_ids"][0] == 5:
                    raise ValueError("fake: vLLM refused this request")
                return real(prompts, params, use_tqdm)
            s.t.llm.enqueue = enqueue
            jobs = [s.eng.submit(R.build(SKELS[i])["prompt"], 2, 100) for i in (5, 6)]
            for j in jobs:
                self.assertTrue(j["done"].wait(10))
            self.assertEqual(jobs[0]["out"]["finish"], "render_error")
            self.assertIn("refused", jobs[0]["out"]["error"])
            self.assertEqual(jobs[1]["out"]["finish"], "stop")
            s.t.engine["max_model_len"] = 1
            j = s.eng.submit(R.build(SKELS[7])["prompt"], 2, 100)
            self.assertTrue(j["done"].wait(10))
            self.assertEqual(j["out"]["finish"], "render_error")
            self.assertIn("no room", j["out"]["error"])
            s.t.engine["max_model_len"] = 2048
            j = s.eng.submit(R.build(SKELS[8])["prompt"], 2, 100)
            self.assertTrue(j["done"].wait(10))
            self.assertEqual(j["out"]["finish"], "stop")
            self.assertIsNone(s.eng.dead)
        finally:
            s.close()


class MinistralThoughtText(unittest.TestCase):
    """Ministral's official tekken decodes think ids 34/35 as '<SPECIAL_34>'/'<SPECIAL_35>' (checked on the PC,
    2026-09-27). THOUGHT_TAG_RE does not match them; the checker still rejects such a turn, through DIGIT."""
    def test_special_think_text_is_rejected(self):
        import checker
        sk = SKELS[0]
        self.assertTrue(checker.run(sk, fake_teacher.raw(sk))["ok"])            # the clean render passes
        lines = fake_teacher.raw(sk).split("\n")
        lines[1] = lines[1].replace(": ", ": <SPECIAL_34>plan it<SPECIAL_35>", 1)
        res = checker.run(sk, "\n".join(lines))
        self.assertFalse(res["ok"])
        self.assertNotIn("THOUGHT_TAG", res["codes"])
        self.assertIn("DIGIT", res["codes"])
        self.assertEqual(serve.thought_in(types.SimpleNamespace(cfg=serve.TEACHERS["ministral-3-8b"], thought_ids={34}),
                                          lines[1], [34]), ["<SPECIAL_34>", "<SPECIAL_35>", "id34"])


if __name__ == "__main__":
    unittest.main()
