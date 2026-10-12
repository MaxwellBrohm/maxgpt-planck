"""HTTP client for bankserve.py (loopback only). generate(call) sends one rendered bank call and returns the server
row: text, finish, token counts and the "decode" record of what the server applied. A row whose decode record lacks
what the call asked for (the regex hash, the dash ban, the phrase ban, the preset) is an error, never a silent
record (serve_client's fail-closed rule). ServerGone (connection refused or reset, or 503) stops a hold so the next
hold resumes; nothing is recorded for the call."""
import json
import time
import urllib.error
import urllib.request

from teachers import decode as D

GEN, INFO = "/planck-serve/generate", "/planck-serve"


class ServerGone(RuntimeError):
    pass


class CallError(RuntimeError):
    pass


class Client:
    def __init__(self, base, teacher, timeout=1800.0, tries=3, phrases=True, dash=True):
        if not base.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("bank pass client talks to a loopback server only")
        self.base, self.teacher, self.timeout, self.tries = base.rstrip("/"), teacher, timeout, tries
        self.phrases, self.dash, self.info = phrases, dash, None

    def check(self):
        with urllib.request.urlopen(self.base + INFO, timeout=30) as r:
            info = json.loads(r.read())
        if info.get("teacher") != self.teacher:
            raise CallError(f"server is {info.get('teacher')!r}, this hold is {self.teacher!r}")
        if info.get("license") != "Apache-2.0" or not info.get("structured_backend"):
            raise CallError("server lacks the Apache-2.0 pin or a structured-output backend")
        if self.dash and not info.get("dash_ban"):
            raise CallError("server has no dash ban")
        if self.phrases and not info.get("phrase_ban"):
            raise CallError("server has no phrase ban")
        self.info = info
        return info

    def body(self, prompt, spec, seed, preset):
        b = {"prompt": prompt, "seed": int(seed) % (2 ** 31), "max_tokens": int(spec["max_tokens"]),
             "bank_regex": spec["regex"]}
        if preset:
            b["preset"] = preset
        if self.dash:
            b["ban"] = "dash"
        if self.phrases:
            b.update(phrases="ai_ism", literals=[x for x in spec.get("literals") or [] if x])
        return b

    def _post(self, body):
        req = urllib.request.Request(self.base + GEN, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def generate(self, prompt, spec, seed, preset=None, call=None):
        """call (the planned call) is unused here; the test fake answers from it."""
        body = self.body(prompt, spec, seed, preset)
        err = None
        for i in range(self.tries):
            try:
                row = self._post(body)
                break
            except urllib.error.HTTPError as e:
                if e.code == 503:
                    raise ServerGone(f"503 {e.read()[:200]!r}")
                raise CallError(f"HTTP {e.code}: {e.read()[:300]!r}")
            except (ConnectionError, urllib.error.URLError) as e:
                r = getattr(e, "reason", e)
                if isinstance(r, OSError) and not isinstance(r, TimeoutError):
                    raise ServerGone(str(e))
                err = e
                time.sleep(2.0 * (i + 1))
        else:
            raise CallError(f"no answer after {self.tries} tries: {err}")
        self.verify(row, spec, preset)
        return row

    def verify(self, row, spec, preset):
        d = row.get("decode") or {}
        bad = []
        if d.get("regex_sha256") != D.sha(spec["regex"]):
            bad.append("regex")
        if self.dash and not d.get("ban"):
            bad.append("dash ban")
        if self.phrases and not d.get("phrases"):
            bad.append("phrase ban")
        if preset and d.get("preset") != preset:
            bad.append(f"preset {d.get('preset')!r}")
        if not preset and (d.get("sampling") or {}).get("temperature") != 0.0:
            bad.append("greedy (server --sampling-json temperature 0) not applied")
        if row.get("finish") == "render_error":
            bad.append("render_error " + str(row.get("error"))[:120])
        if bad:
            raise CallError("server row does not match the request: " + ", ".join(bad))
