"""Teacher client (SPEC sections 0 and 5): one render request to an OpenAI-compatible server.

Two wire modes, both from render_prompt:
  completions  POST /v1/completions with the Gemma 4 raw prompt, thinking OFF (LM Studio raw completions, vLLM,
               llama.cpp server); the body is render_prompt.completion_body() plus a per-attempt seed;
  chat         POST /v1/chat/completions with render_prompt.chat_body() plus the seed (any other local server).

Safety (SPEC 0): without allow_real the client refuses every host that is not loopback, BEFORE opening a socket,
and on loopback it first asks GET /planck-stub and refuses unless the answer says it is the fake stub
(fake_server.py). A real LM Studio or vLLM server answers that path with a 404, and an unknown GET path never
loads a model. Only the driver's --allow-real-teacher flag sets allow_real; the tests never pass it.

Transient failures (connection refused or reset, timeout, HTTP 408/429/5xx, a body that is not JSON) are retried
up to http_tries times with exponential backoff; any other HTTP status fails at once. A request that never
succeeds raises TeacherError, which the driver records as a TEACHER_ERROR reject for that attempt."""
import hashlib
import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import render_prompt as R

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
STUB_PATH = "/planck-stub"
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
LICENSES = {"gemma-4": "Apache-2.0", "ministral-3": "Apache-2.0"}   # model id substring -> license (D8)


class RefuseRealTeacher(RuntimeError):
    pass


class TeacherError(RuntimeError):
    def __init__(self, detail, requests):
        super().__init__(detail)
        self.detail, self.requests = detail, requests


def render_seed(skel_id, attempt):
    """the sampling seed of one render attempt: new per attempt (SPEC 1, one retry with a new render seed) and
    stable across resumes, so the fake stub can key its canned answers on it."""
    return int(hashlib.sha256(f"{skel_id}:{attempt}".encode()).hexdigest()[:8], 16)


def license_of(model):
    m = (model or "").lower()
    for key, lic in LICENSES.items():
        if key in m:
            return lic
    return "UNKNOWN"


class TeacherClient:
    def __init__(self, base_url, model, mode="completions", allow_real=False, timeout=180.0, http_tries=4,
                 backoff=1.0, sampling=None, license=None):
        if mode not in ("completions", "chat"):
            raise ValueError(f"mode {mode!r}")
        self.base = base_url.rstrip("/")
        self.model, self.mode, self.allow_real = model, mode, allow_real
        self.timeout, self.http_tries, self.backoff = timeout, http_tries, backoff
        self.sampling = sampling
        self.license = license or license_of(model)
        self.is_stub = False
        self._checked = False

    # --- safety --------------------------------------------------------------------------------------------------
    def check_endpoint(self):
        """raise RefuseRealTeacher unless this endpoint may be used. Runs once, before the first render."""
        if self._checked:
            return
        u = urllib.parse.urlparse(self.base)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise RefuseRealTeacher(f"bad endpoint {self.base!r}")
        if not self.allow_real:
            if u.hostname not in LOOPBACK:
                raise RefuseRealTeacher(f"host {u.hostname} is not loopback; pass --allow-real-teacher to use it")
            try:
                ident = self._get(STUB_PATH)
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
                raise RefuseRealTeacher(f"{self.base} did not answer {STUB_PATH} as the fake stub ({e}); "
                                        "a real teacher needs --allow-real-teacher") from e
            if not (isinstance(ident, dict) and ident.get("planck_fake_stub") is True):
                raise RefuseRealTeacher(f"{self.base} is not the fake stub")
            self.is_stub = True
            self.license = "FAKE"
        self._checked = True

    # --- http ----------------------------------------------------------------------------------------------------
    def _get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=min(self.timeout, 10.0)) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            e.close()
            raise

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def body(self, built, seed):
        if self.mode == "completions":
            b = R.completion_body(built, self.model, self.sampling)
        else:
            b = R.chat_body(built, self.model, self.sampling)
        b["seed"] = seed
        return b

    def render(self, built, seed):
        """-> {text, finish, usage, model, requests, latency_s}. Raises TeacherError after the retries."""
        self.check_endpoint()
        path = "/v1/completions" if self.mode == "completions" else "/v1/chat/completions"
        body = self.body(built, seed)
        last, t0 = None, time.monotonic()
        for k in range(self.http_tries):
            if k and self.backoff:
                time.sleep(min(30.0, self.backoff * 2 ** (k - 1)) * (0.5 + random.random()))
            try:
                resp = self._post(path, body)
                return {**self._extract(resp), "requests": k + 1, "latency_s": round(time.monotonic() - t0, 3)}
            except urllib.error.HTTPError as e:
                e.close()
                last = f"http {e.code}"
                if e.code not in TRANSIENT_STATUS:
                    raise TeacherError(last, k + 1) from e
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                    http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as e:
                last = f"{type(e).__name__}: {e}"[:200]
        raise TeacherError(f"gave up after {self.http_tries} requests ({last})", self.http_tries)

    def _extract(self, resp):
        ch = resp["choices"][0]
        text = ch["text"] if self.mode == "completions" else ch["message"]["content"]
        if not isinstance(text, str):
            raise TypeError("no text in the response")
        return {"text": text, "finish": ch.get("finish_reason"), "usage": resp.get("usage") or {},
                "model": resp.get("model") or self.model}
