"""Driver-side client for serve_http.py: pipeline/driver.py renders through a vLLM teacher on the PC (DRY: FAKE
skeletons; every record stays trainable=false through FAKE_PROVENANCE and DECON_PENDING).

It keeps teacher_client.TeacherClient's interface (check_endpoint, render(built, seed) -> {text, finish, usage, model,
requests, latency_s}, model, license, mode, is_stub) and its retry rule (transient failures retried with backoff,
TeacherError after http_tries), so driver.py runs unchanged. Differences:
  - the body is the PLAIN render prompt plus seed and max_tokens; serve_http renders the teacher's own wire format
    (serve.render) and refuses a render that breaks its invariants (reported here as a TeacherError, a TEACHER_ERROR
    reject, never a checked text);
  - check_endpoint refuses unless allow_real is set (the driver's --allow-real-teacher), the host is loopback (the
    PC is reached through an SSH tunnel), and GET /planck-serve answers as a dry serve_http with an allowed license;
  - model and license come from the served teacher's pins (serve.TEACHERS: repo@revision, Apache-2.0), not from
    the driver's --model and --license, so teacher_client.LICENSES (which has no Qwen key) is not consulted.
Timeouts must cover queueing: with concurrency above what the KV pool holds a request waits for a free slot.

Decoding controls (2026-09-27, the driver's --structured / --ban-dashes / --preset, set through configure_decode):
check_endpoint also refuses a server that does not offer what the run asks for (a structured-output backend, a
loaded dash ban, the preset name); each body carries them (the constraint is the one driver.dispatch put in
built["constraint"]); each row's "decode" (what serve_http applied) goes into the result record, and a row whose
"decode" does not match the request is a TeacherError (fail closed: a record never claims a control it lacked)."""
import json
import os
import random
import socket
import sys
import time
import http.client
import urllib.error
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import teacher_client as TC  # noqa: E402

OK_LICENSES = {"Apache-2.0"}
INFO_PATH = "/planck-serve"
GEN_PATH = "/planck-serve/generate"


class ServerGone(RuntimeError):
    """the served teacher went away on the last try: a socket error other than a timeout (refused, reset, broken
    pipe, EINVAL: all seen in the test fixture), or serve_http's 503 while it stops or after its engine died. Not a
    TeacherError: the driver does not catch it, so the run stops with the finished records intact instead of writing
    a TEACHER_ERROR reject for every remaining attempt, and the same command resumes the unfinished ones against a
    new server (2026-09-26)."""


def _gone(e):
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 503
    r = getattr(e, "reason", e)      # a socket error on loopback other than a timeout (macOS also gives EINVAL)
    return isinstance(r, OSError) and not isinstance(r, TimeoutError)


class ServeClient(TC.TeacherClient):
    supports_decode = True

    def __init__(self, base_url, model=None, mode=None, allow_real=False, timeout=3600.0, http_tries=3, backoff=2.0,
                 sampling=None, license=None):
        super().__init__(base_url, model or "served", "completions", allow_real=allow_real, timeout=timeout,
                         http_tries=http_tries, backoff=backoff, sampling=sampling, license=license or "UNKNOWN")
        self.mode = "serve"
        self.info = None
        self.decode = dict(TC.DECODE_OFF)

    def configure_decode(self, cfg):
        self.decode = TC.decode_config(cfg)

    def check_endpoint(self):
        if self._checked:
            return
        u = urllib.parse.urlparse(self.base)
        if u.scheme != "http" or u.hostname not in TC.LOOPBACK:
            raise TC.RefuseRealTeacher(f"serve endpoint must be http on loopback (an SSH tunnel), got {self.base!r}")
        if not self.allow_real:
            raise TC.RefuseRealTeacher("a served teacher is a real model; pass --allow-real-teacher")
        try:
            info = self._get(INFO_PATH)
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
            raise TC.RefuseRealTeacher(f"{self.base} did not answer {INFO_PATH} ({e})") from e
        if not (isinstance(info, dict) and info.get("planck_serve") is True and info.get("status") == "dry"):
            raise TC.RefuseRealTeacher(f"{self.base} is not a dry serve_http")
        if info.get("license") not in OK_LICENSES:
            raise TC.RefuseRealTeacher(f"served teacher license {info.get('license')!r} is not allowed (D8)")
        want = self.decode
        if want["structured"] != "off" and not info.get("structured_backend"):
            raise TC.RefuseRealTeacher("the run asks for structured output; this server has no backend for it")
        if want["ban"] and not (info.get("dash_ban") or {}).get("n"):
            raise TC.RefuseRealTeacher("the run asks for the dash ban; this server has none (serve_http --dash-ban)")
        if want["preset"] and want["preset"] not in (info.get("presets") or ()):
            raise TC.RefuseRealTeacher(f"preset {want['preset']!r} is not offered: {info.get('presets')}")
        self.info, self.model, self.license = info, info["model"], info["license"]
        self._checked = True

    def body(self, built, seed):
        b = {"prompt": built["prompt"], "seed": seed, "max_tokens": built["max_tokens"]}
        want = self.decode
        if want["preset"]:
            b["preset"] = want["preset"]
        if want["ban"]:
            b["ban"] = want["ban"]
        if want["structured"] != "off":
            c = built.get("constraint")
            if not c or c.get("mode") != want["structured"]:
                raise ValueError(f"built carries no {want['structured']} constraint (driver.dispatch sets it)")
            b["constraint"] = c
        return b

    def mismatch(self, dec):
        """why a row's decode record does not show what this client asked for, or None."""
        want, dec = self.decode, dec or {}
        if want["structured"] != "off" and (dec.get("structured") != want["structured"] or not dec.get("regex_sha256")):
            return f"structured {want['structured']} not applied"
        if want["ban"] and (dec.get("ban") or {}).get("n") != (self.info.get("dash_ban") or {}).get("n"):
            return "dash ban not applied"
        if want["preset"] and dec.get("preset") != want["preset"]:
            return f"preset {want['preset']} not applied (got {dec.get('preset')})"
        return None

    def decode_meta(self, call):
        return (call or {}).get("decode")

    def render(self, built, seed):
        self.check_endpoint()
        body = self.body(built, seed)
        last, gone, t0 = None, False, time.monotonic()
        for k in range(self.http_tries):
            if k and self.backoff:
                time.sleep(min(30.0, self.backoff * 2 ** (k - 1)) * (0.5 + random.random()))
            try:
                resp = self._post(GEN_PATH, body)
            except urllib.error.HTTPError as e:
                e.close()
                last, gone = f"http {e.code}", _gone(e)
                if e.code not in TC.TRANSIENT_STATUS:
                    raise TC.TeacherError(last, k + 1) from e
                continue
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                    http.client.HTTPException, ValueError) as e:
                last, gone = f"{type(e).__name__}: {e}"[:200], _gone(e)
                continue
            gone = False
            if resp.get("finish") == "render_error":
                raise TC.TeacherError(f"render_error: {resp.get('error')}"[:300], k + 1)
            if not isinstance(resp.get("text"), str):
                last = "no text in the response"
                continue
            bad = self.mismatch(resp.get("decode"))
            if bad:
                raise TC.TeacherError(f"decode mismatch: {bad}", k + 1)
            return {"text": resp["text"], "finish": resp.get("finish"),
                    "usage": {"prompt_tokens": resp.get("n_prompt"), "completion_tokens": resp.get("n_out")},
                    "model": resp.get("model") or self.model, "requests": k + 1,
                    "latency_s": round(time.monotonic() - t0, 3), "thought": resp.get("thought") or [],
                    "clamped": bool(resp.get("clamped")), "decode": resp.get("decode")}
        if gone:
            raise ServerGone(f"served teacher gone after {self.http_tries} requests ({last})")
        raise TC.TeacherError(f"gave up after {self.http_tries} requests ({last})", self.http_tries)


def describe(client):
    """one line for logs: what the driver is talking to."""
    i = client.info or {}
    return json.dumps({k: i.get(k) for k in ("teacher", "repo", "revision", "license", "quant", "wire", "sampling",
                                             "presets", "structured_backend", "line_sep", "dash_ban",
                                             "gpu_memory_utilization")})
