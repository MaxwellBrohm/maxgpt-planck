"""Test fixture: serve_http's engine loop and HTTP handler around a FAKE vLLM engine, with serve.py's REAL render,
prepare, _params, resolve and dash_ban (stub_vllm stands in for vllm, torch and the Gemma tokenizer). Each request's
SamplingParams (stub: .kw) are recorded next to its skeleton and plain prompt, and the output text comes from a
policy(skel, prompt, sp) the test chooses, so a test sees exactly what the engine would have been asked to do.
CPU only, no model anywhere; DRY."""
import itertools
import os
import random
import shutil
import tempfile
import threading
import time
import types
from unittest import mock

import render_prompt as R
import serve
import serve_http
import stub_vllm as S

VOCAB = [b"a", b" the", "\u2014".encode(), b"-", b"well-", b"-known", b" -", b"--", b" - ", b"x--y",
         "\u2013".encode(), " \u2212".encode(), b"\xe2\x80", "\u2019".encode(), b"", "\u2012\u2015".encode()]
BANNED = [2, 6, 7, 8, 9, 10, 11, 15]
HEAD = "<bos><|turn>user\n"
TAIL = "<turn|>\n<|turn>model\n"


class Steps:
    """llm_engine: finishes each request after 1 to 3 steps, reporting vLLM's external id '<n>'."""
    def __init__(self):
        self.live = {}

    def has_unfinished_requests(self):
        return bool(self.live)

    def step(self):
        time.sleep(0.001)
        out = []
        for rid in list(self.live):
            self.live[rid][0] -= 1
            if self.live[rid][0] <= 0:
                text = self.live.pop(rid)[1]
                c = types.SimpleNamespace(text=text, token_ids=[1] * max(1, len(text.split())), finish_reason="stop")
                out.append(types.SimpleNamespace(request_id=rid.rsplit("-", 1)[0], finished=True, outputs=[c]))
        return out


class LLM:
    def __init__(self, policy, skels):
        self.llm_engine, self.count, self.policy = Steps(), itertools.count(), policy
        self.by_prompt = {R.build(k)["prompt"]: k for k in skels}
        self.seen, self.lock = [], threading.Lock()

    def skel_of(self, prompt):
        for p, k in self.by_prompt.items():
            if prompt == p or prompt.startswith(p + "\n\n"):
                return k
        raise KeyError("a prompt the fixture did not build")

    def enqueue(self, prompts, params, use_tqdm=False):
        rids = []
        for p, sp in zip(prompts, params):
            text = S.decode(p["prompt_token_ids"])
            prompt = text[len(HEAD):text.index(TAIL)]
            sk = self.skel_of(prompt)
            with self.lock:
                self.seen.append((sk, prompt, sp))
            rid = f"{next(self.count)}-{random.randrange(16 ** 8):08x}"
            self.llm_engine.live[rid] = [random.randint(1, 3), self.policy(sk, prompt, sp)]
            rids.append(rid)
        return rids


class Served:
    def __init__(self, skels, policy, ban=True, backend="xgrammar", sampling=None, util=0.86):
        self.tmp = tempfile.mkdtemp(prefix="fake_engine_")
        self.patches = [S.install(), mock.patch.object(serve, "tokenizer", S.tokenizer),
                        mock.patch.object(serve, "token_bytes", lambda t: VOCAB)]
        for p in self.patches:
            p.start()
        t = serve.tokenizer_only("gemma-4-12b")
        t.engine, t.load_s, t.llm = {"max_model_len": 16384}, 0.0, LLM(policy, skels)
        self.t, self.ban = t, None
        if ban:
            d = os.path.join(self.tmp, "models", t.cfg["dir"])
            os.makedirs(d)
            with open(os.path.join(d, "tokenizer.json"), "w") as f:
                f.write("fake tokenizer")
            self.ban = serve.dash_ban(t, os.path.join(self.tmp, "models"), os.path.join(self.tmp, "cache"))
        self.eng = serve_http.Engine(t, sampling or {}, self.ban, util)
        self.th = threading.Thread(target=self.eng.run, daemon=True)
        self.th.start()
        self.info = serve_http.server_info(t, sampling or {}, self.ban, backend, util)
        self.stop = threading.Event()
        self.srv = serve_http.ThreadingHTTPServer(("127.0.0.1", 0),
                                                  serve_http.make_handler(self.eng, self.info, self.stop))
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.eng.halt.set()
        self.srv.shutdown()
        self.srv.server_close()
        self.th.join(timeout=5)
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)
