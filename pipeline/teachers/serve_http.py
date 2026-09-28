"""Loopback HTTP front for serve.py on the PC, so pipeline/driver.py can render through a vLLM teacher (DRY: FAKE
skeleton renders; nothing it produces is training data).

    flock -w 7200 ~/planck/locks/gpu.lock timeout -k 60 2280 python serve_http.py --teacher gemma-4-12b --port 18763 \
        [--log THIS_PROCESS.log] [--stats OUT.dry.json] [--idle-s 240] [--first-request-s 900] [--sampling-json '{}'] \
        [--dash-ban] [--phrase-ban] [--cache-dir DIR] [--gpu-util 0.86] [--structured-backend xgrammar|none]

Loads one teacher with serve.load, then listens on 127.0.0.1 only:
  GET  /planck-serve            identity + pins {"planck_serve": true, "status": "dry", teacher, repo, revision, ...}
  POST /planck-serve/generate   {"prompt": build(skel)["prompt"], "seed", "max_tokens"} -> one serve.generate row
                                optional (decoding controls, 2026-09-27): "preset" (a D4 preset name), "constraint"
                                {"mode", "lines"} (structured output: decode.label_regex with this teacher's line
                                separator), "ban": "dash" (needs --dash-ban); the row's "decode" says what was applied
                                (2026-09-28) "phrases": "ai_ism" (needs --phrase-ban) with "literals" (the chat's forced
                                lines): vLLM bad_words = decode.phrase_words(literals and constraint literals)
  POST /planck-serve/stop       unload, write --stats, exit
One engine thread owns vLLM. It adds each request the moment it arrives (LLM.enqueue) and steps the engine itself
(llm_engine.step), so requests join and leave the running batch continuously and the caller's concurrency is the
batch bound (vLLM still runs only what its KV pool holds). The wire render and its invariants are serve.render's
(Gemma <bos> + empty thought channel, Ministral official tekken with no system prompt, Qwen enable_thinking=False);
sampling is the teacher's serve.TEACHERS sampling plus --sampling-json. The process unloads and exits on /stop,
after --idle-s with nothing in flight once a request was served, after --first-request-s with no request at all,
or on SIGTERM, so the caller's flock on gpu.lock is released with the GPU memory already back."""
import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode  # noqa: E402
import serve  # noqa: E402
from bench import gpu_used_mib  # noqa: E402


class Engine:
    def __init__(self, t, sampling, ban=None, util=None, phrases=None):
        self.t, self.sampling, self.ban, self.util, self.phrases = t, sampling, ban, util, phrases
        self.q = queue.Queue()
        self.pending = {}
        self.halt = threading.Event()
        self.dead = None
        self.lock = threading.Lock()
        self.st = {"received": 0, "done": 0, "render_errors": 0, "prompt_tokens": 0, "out_tokens": 0, "finish": {},
                   "thought_hits": 0, "clamped": 0, "max_inflight": 0, "step_s": 0.0, "first_request_t": None,
                   "last_done_t": None, "structured": 0, "banned": 0, "phrase_banned": 0, "presets": {}}

    def submit(self, prompt, seed, max_tokens, req=None):
        """req: the request's decoding controls, checked by the handler: preset, constraint + regex, ban."""
        job = {"prompt": prompt, "seed": seed, "max_tokens": max_tokens, "done": threading.Event(),
               "t0": time.time(), "out": None, "req": req or {}}
        with self.lock:
            self.st["received"] += 1
            if self.st["first_request_t"] is None:
                self.st["first_request_t"] = job["t0"]
        self.q.put(job)
        return job

    def inflight(self):
        return len(self.pending) + self.q.qsize()

    def sampling_for(self, j):
        """one request's sampling and the "decode" record that goes back with its row."""
        r = j["req"]
        s = {"max_tokens": j["max_tokens"]}
        s = {**s, "preset": r["preset"]} if r.get("preset") else {**self.sampling, **s}
        if r.get("regex"):
            s["regex"] = r["regex"]
        if r.get("ban"):
            if not self.ban:
                raise ValueError("this server has no dash ban (start it with --dash-ban)")
            s["ban_ids"] = self.ban["ids"]
        if r.get("phrases"):
            if not self.phrases:
                raise ValueError("this server has no phrase ban (start it with --phrase-ban)")
            s["bad_words"] = decode.phrase_words(r.get("literals"))
        dec = serve.sampling_record(self.t, s)
        if not r.get("preset") and self.sampling:
            dec["preset"] = "custom"                      # --sampling-json over the default
        c = r.get("constraint")
        dec.update(structured=c["mode"] if c else None, line_sep=r.get("line_sep"),
                   n_literal=sum(x[1] is not None for x in c["lines"]) if c else 0,
                   ban={k: self.ban[k] for k in ("rule", "n", "ids_sha256", "tokenizer_sha256")} if r.get("ban")
                   else None, gpu_memory_utilization=self.util, label_rule=r.get("label_rule"),
                   phrases={"rule": self.phrases["rule"], "n": dec["bad_words_n"], "sha256": dec["bad_words_sha256"],
                            "left_out": len(decode.PHRASES) - dec["bad_words_n"]} if r.get("phrases") else None)
        return s, dec

    def _add(self, jobs):
        """one enqueue per request: a request vLLM refuses (or serve.prepare refuses: a broken wire invariant, a
        prompt with no room under max_model_len) fails alone as render_error; the engine and the batch go on."""
        from vllm.inputs import TokensPrompt
        for j in jobs:
            try:
                s, j["decode"] = self.sampling_for(j)
                ids, sp, clamped = serve.prepare(self.t, j["prompt"], s, j["seed"])
                # enqueue returns vLLM's internal id ("<n>-<8 random chars>"); a RequestOutput carries the external "<n>"
                rid = self.t.llm.enqueue([TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False)[0]
            except (serve.RenderError, ValueError, TypeError) as e:
                j["out"] = {"finish": "render_error", "error": f"{type(e).__name__}: {e}"[:300]}
                self.st["render_errors"] += 1
                j["done"].set()
                continue
            j["n_prompt"], j["clamped"] = len(ids), clamped
            self.st["structured"] += bool(j["decode"]["structured"])
            self.st["banned"] += bool(j["decode"]["ban"])
            self.st["phrase_banned"] += bool(j["decode"]["phrases"])
            self.st["presets"][j["decode"]["preset"]] = self.st["presets"].get(j["decode"]["preset"], 0) + 1
            self.pending[rid.rsplit("-", 1)[0] if "-" in rid else rid] = j
        self.st["max_inflight"] = max(self.st["max_inflight"], len(self.pending))

    def _finish(self, o):
        j = self.pending.pop(o.request_id, None)
        if j is None:
            return
        c = o.outputs[0]
        ids = list(c.token_ids)
        th = serve.thought_in(self.t, c.text, ids)
        j["out"] = {"text": serve.strip_end(c.text), "raw": c.text, "finish": c.finish_reason,
                    "n_prompt": j["n_prompt"], "n_out": len(ids), "thought": th, "clamped": j["clamped"],
                    "server_s": round(time.time() - j["t0"], 3), "decode": j["decode"]}
        s = self.st
        s["done"] += 1
        s["prompt_tokens"] += j["n_prompt"]
        s["out_tokens"] += len(ids)
        s["finish"][str(c.finish_reason)] = s["finish"].get(str(c.finish_reason), 0) + 1
        s["thought_hits"] += bool(th)
        s["clamped"] += bool(j["clamped"])
        s["last_done_t"] = time.time()
        j["done"].set()

    def run(self):
        eng = self.t.llm.llm_engine
        try:
            while not self.halt.is_set():
                jobs, busy = [], eng.has_unfinished_requests()
                try:
                    jobs.append(self.q.get(block=not busy, timeout=None if busy else 0.2))
                except queue.Empty:
                    pass
                while True:
                    try:
                        jobs.append(self.q.get_nowait())
                    except queue.Empty:
                        break
                if jobs:
                    self._add(jobs)
                if eng.has_unfinished_requests():
                    t0 = time.time()
                    for o in eng.step():
                        if o.finished:
                            self._finish(o)
                    self.st["step_s"] += time.time() - t0
        except Exception as e:  # the engine died: fail every waiting request so the driver logs TEACHER_ERROR
            self.dead = f"{type(e).__name__}: {e}"[:300]
            print(f"[serve_http] engine thread died: {self.dead}", flush=True)
        for j in list(self.pending.values()):
            j["done"].set()


def decode_request(body, info):
    """the request's decoding controls, checked against what this server offers; ValueError -> HTTP 400."""
    req = {}
    if body.get("preset") is not None:
        if body["preset"] not in info.get("presets", ()):
            raise ValueError(f"unknown preset {body['preset']!r} (this server: {info.get('presets')})")
        req["preset"] = body["preset"]
    c = body.get("constraint")
    if c is not None:
        if not info.get("structured_backend"):
            raise ValueError("this server was loaded without a structured-output backend")
        if not decode.constraint_ok(c):
            raise ValueError("constraint must be {mode: labels|labels_exact, lines: [[label, literal or null]]}")
        req.update(constraint=c, line_sep=info["line_sep"], label_rule=decode.LABEL_RULE,
                   regex=decode.label_regex(c["lines"], info["line_sep"]))
    if body.get("ban") is not None:
        if body["ban"] != "dash" or not info.get("dash_ban"):
            raise ValueError(f"ban {body['ban']!r} is not loaded on this server (start it with --dash-ban)")
        req["ban"] = "dash"
    if body.get("phrases") is not None:
        if body["phrases"] != "ai_ism" or not info.get("phrase_ban"):
            raise ValueError(f"phrases {body['phrases']!r} is not loaded on this server (start it with --phrase-ban)")
        lits = body.get("literals") or []
        if not isinstance(lits, list) or not all(isinstance(x, str) for x in lits):
            raise ValueError("literals must be a list of strings")
        lits = lits + [x[1] for x in (c or {}).get("lines", ()) if x[1] is not None]
        req.update(phrases="ai_ism", literals=lits)
    return req


def make_handler(eng, info, stop_evt):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/planck-serve":
                return self._send(200, info)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            except ValueError:
                return self._send(400, {"error": "bad json"})
            if self.path == "/planck-serve/stop":
                stop_evt.set()
                return self._send(200, {"ok": True})
            if self.path != "/planck-serve/generate":
                return self._send(404, {"error": "not found"})
            p, seed, mt = body.get("prompt"), body.get("seed"), body.get("max_tokens")
            if not isinstance(p, str) or not p or not isinstance(seed, int) or not isinstance(mt, int) or mt < 1:
                return self._send(400, {"error": "need prompt (str), seed (int), max_tokens (int >= 1)"})
            try:
                req = decode_request(body, info)
            except ValueError as e:
                return self._send(400, {"error": str(e)[:300]})
            if stop_evt.is_set() or eng.dead:
                return self._send(503, {"error": f"stopping ({eng.dead or 'stop'})"})
            job = eng.submit(p, seed, mt, req)
            while not job["done"].wait(5.0):
                if eng.dead or eng.halt.is_set():
                    break
            if job["out"] is None:
                return self._send(503, {"error": f"engine gone ({eng.dead or 'halt'})"})
            self._send(200, {**job["out"], "model": info["model"]})

    return H


def server_info(t, sampling, ban, backend, util, kv=None, phrases=None):
    """GET /planck-serve: identity, pins, and the decoding controls this server offers."""
    cfg = t.cfg
    return {"planck_serve": True, "status": "dry", "teacher": t.name, "repo": cfg["repo"],
            "revision": cfg["revision"], "license": cfg["license"], "quant": cfg["quant"], "wire": cfg["wire"],
            "model": f"{cfg['repo']}@{cfg['revision']}", "sampling": {**cfg["sampling"], **sampling},
            "engine": {k: v for k, v in (t.engine or {}).items() if k not in ("tokenizer", "structured_outputs_config")},
            **(kv or {}), "load_s": t.load_s, "decode_controls": 1, "presets": serve.preset_names(t.name),
            "default_preset": "custom" if sampling else serve.DEFAULT_PRESET, "structured_backend": backend,
            "line_sep": cfg.get("line_sep", "\\n"), "gpu_memory_utilization": util,
            "label_rule": decode.LABEL_RULE if backend else None,
            "phrase_ban": {k: v for k, v in phrases.items() if k != "seqs"} if phrases else None,
            "dash_ban": {k: ban[k] for k in ("rule", "n", "ids_sha256", "tokenizer_sha256", "cached")} if ban else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=sorted(serve.TEACHERS))
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", default=None, help="this process's log (vLLM's KV capacity line is read from it)")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--idle-s", type=float, default=240.0)
    ap.add_argument("--first-request-s", type=float, default=900.0)
    ap.add_argument("--sampling-json", default="{}", help="merged over the teacher's serve.TEACHERS sampling for "
                    "requests that name no preset")
    ap.add_argument("--dash-ban", action="store_true", help="compute (or load from --cache-dir) the dash-token ban "
                    "before loading, so requests may ask for it")
    ap.add_argument("--phrase-ban", action="store_true", help="offer the AI-ism phrase ban (vLLM bad_words); its "
                    "token form is checked on the loaded engine's tokenizer before READY")
    ap.add_argument("--cache-dir", default=serve.BAN_CACHE)
    ap.add_argument("--gpu-util", type=float, default=serve.ENGINE["gpu_memory_utilization"])
    ap.add_argument("--structured-backend", default="xgrammar", choices=["xgrammar", "none"])
    a = ap.parse_args(argv)
    if a.gpu_util > serve.MAX_UTIL:
        raise SystemExit(f"--gpu-util {a.gpu_util} is above {serve.MAX_UTIL}")
    sampling = json.loads(a.sampling_json)
    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())
    ban = serve.dash_ban(serve.tokenizer_only(a.teacher), cache_dir=a.cache_dir) if a.dash_ban else None
    backend = None if a.structured_backend == "none" else a.structured_backend
    g0 = gpu_used_mib()
    t = serve.load(a.teacher, structured_backend=backend, gpu_memory_utilization=a.gpu_util)
    kv = None
    if a.log and os.path.exists(a.log):
        with open(a.log, encoding="utf-8", errors="replace") as f:
            kv = serve.kv_from_log(f.read())
    try:
        phrases = serve.phrase_ban(t.llm.get_tokenizer()) if a.phrase_ban else None
    except Exception:
        serve.unload(t)                          # the GPU memory goes back before the lock is released
        raise
    info = server_info(t, sampling, ban, backend, a.gpu_util, kv={"kv_vllm_log": kv, "kv_frontend": serve.kv_info(t)},
                       phrases=phrases)
    eng = Engine(t, sampling, ban, a.gpu_util, phrases)
    th = threading.Thread(target=eng.run, daemon=True)
    th.start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(eng, info, stop_evt))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ready = time.time()
    print(f"[serve_http] READY {a.teacher} port {a.port} load {t.load_s}s kv {kv} gpu {gpu_used_mib()} util "
          f"{a.gpu_util} structured {backend} dash_ban {info['dash_ban']} phrase_ban {phrases}", flush=True)
    reason, last_busy = None, ready
    while reason is None:
        stop_evt.wait(2.0)
        now = time.time()
        if eng.inflight():
            last_busy = now
        if stop_evt.is_set():
            reason = "stop"
        elif eng.dead:
            reason = f"engine died: {eng.dead}"
        elif not eng.st["received"] and now - ready > a.first_request_s:
            reason = "no request"
        elif eng.st["received"] and not eng.inflight() and now - last_busy > a.idle_s:
            reason = "idle"
    print(f"[serve_http] stopping ({reason}), in flight {eng.inflight()}", flush=True)
    eng.halt.set()
    th.join(timeout=60)
    srv.shutdown()
    g1 = gpu_used_mib()
    serve.unload(t)
    time.sleep(5)
    s = eng.st
    win = (s["last_done_t"] - s["first_request_t"]) if s["last_done_t"] and s["first_request_t"] else None
    out = {**info, "exit_reason": reason, "gpu_before_load": g0, "gpu_loaded": g1, "gpu_after_unload": gpu_used_mib(),
           "stats": {**{k: v for k, v in s.items() if not k.endswith("_t")}, "step_s": round(s["step_s"], 1),
                     "window_s": round(win, 1) if win else None,
                     "out_tok_per_s_window": round(s["out_tokens"] / win, 1) if win else None}}
    if a.stats:
        with open(a.stats + ".tmp", "w") as f:
            json.dump(out, f, indent=1)
        os.replace(a.stats + ".tmp", a.stats)
    print(f"[serve_http] EXIT {reason} {json.dumps(out['stats'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
