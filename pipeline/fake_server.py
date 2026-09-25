"""FAKE teacher: a local OpenAI-compatible HTTP stub that serves canned renders (no model anywhere).

It answers only on 127.0.0.1. Endpoints: GET /planck-stub (the identity the teacher client demands before it will
talk to a loopback server without --allow-real-teacher), GET /v1/models, POST /v1/completions (Gemma 4 raw prompt,
thinking off) and POST /v1/chat/completions.

A plan (stub_plan.plan) maps "<prompt sha256>:<seed>" to a list of steps. The n-th request for a key gets step n
(the last step repeats). A step is one of
  {"text": str, "finish": "stop"|"length"}   a completion
  {"status": 500}                            an HTTP error with a JSON body
  {"drop": true}                             the connection is closed with no response
and any step may add {"sleep": seconds}. A request whose key is not in the plan gets a 404 ("unplanned"), so a
driver that calls the teacher for a skeleton the plan says must not be rendered shows up as a mismatch.

  python3 -B fake_server.py --plan plan.json [--port 8765] [--delay 0.02]"""
import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.dont_write_bytecode = True

PREFIX = "<|turn>user\n"
SUFFIX = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def prompt_of_completion(body):
    p = body.get("prompt", "")
    if not (p.startswith(PREFIX) and p.endswith(SUFFIX)):
        return None
    return p[len(PREFIX):-len(SUFFIX)]


def prompt_of_chat(body):
    msgs = body.get("messages") or []
    return msgs[-1].get("content") if msgs else None


def key(prompt, seed):
    return f"{hashlib.sha256(prompt.encode()).hexdigest()}:{seed}"


class Stub:
    def __init__(self, plan, delay=0.0):
        self.model = plan.get("model", "fake-teacher")
        self.responses = plan["responses"]
        self.delay = delay
        self.lock = threading.Lock()
        self.hits = {}            # key -> requests served
        self.log = []             # (key or None, step kind)
        self.inflight = 0
        self.max_inflight = 0

    def next_step(self, k):
        with self.lock:
            steps = self.responses.get(k)
            if steps is None:
                self.log.append((k, "unplanned"))
                return None
            n = self.hits.get(k, 0)
            self.hits[k] = n + 1
            step = steps[min(n, len(steps) - 1)]
            kind = "text" if "text" in step else ("drop" if step.get("drop") else f"status{step.get('status')}")
            self.log.append((k, kind))
            return step

    def enter(self):
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)

    def leave(self):
        with self.lock:
            self.inflight -= 1

    def counts(self):
        with self.lock:
            out = {}
            for _, kind in self.log:
                out[kind] = out.get(kind, 0) + 1
            return out


def make_handler(stub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, status, obj):
            data = json.dumps(obj).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/planck-stub":
                return self._send(200, {"planck_fake_stub": True, "model": stub.model})
            if self.path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": stub.model, "object": "model"}]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            stub.enter()
            try:
                self._post()
            finally:
                stub.leave()

        def _post(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                return self._send(400, {"error": {"message": "bad json"}})
            chat = self.path == "/v1/chat/completions"
            if self.path not in ("/v1/completions", "/v1/chat/completions"):
                return self._send(404, {"error": {"message": "not found"}})
            prompt = prompt_of_chat(body) if chat else prompt_of_completion(body)
            if prompt is None:
                return self._send(400, {"error": {"message": "prompt is not in the Gemma 4 thinking-off format"}})
            step = stub.next_step(key(prompt, body.get("seed")))
            if stub.delay:
                time.sleep(stub.delay)
            if step is None:
                return self._send(404, {"error": {"message": "unplanned request"}})
            if step.get("sleep"):
                time.sleep(step["sleep"])
            if step.get("drop"):
                self.close_connection = True
                try:
                    self.connection.shutdown(2)
                except OSError:
                    pass
                return None
            if "status" in step:
                return self._send(step["status"], {"error": {"message": f"planned {step['status']}"}})
            fin = step.get("finish", "stop")
            usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(step["text"].split())}
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": step["text"]}, "finish_reason": fin}
                obj = {"object": "chat.completion", "model": stub.model, "choices": [choice], "usage": usage}
            else:
                choice = {"index": 0, "text": step["text"], "finish_reason": fin}
                obj = {"object": "text_completion", "model": stub.model, "choices": [choice], "usage": usage}
            self._send(200, obj)

    return Handler


def start(plan, port=0, delay=0.0):
    """-> (stub, server, base_url). The server runs in a daemon thread; call stop(server) when done."""
    stub = Stub(plan, delay)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(stub))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return stub, server, f"http://127.0.0.1:{server.server_address[1]}"


def stop(server):
    server.shutdown()
    server.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args(argv)
    with open(a.plan) as f:
        plan = json.load(f)
    stub, server, url = start(plan, a.port, a.delay)
    print(f"fake teacher stub on {url} ({len(plan['responses'])} planned keys)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop(server)


if __name__ == "__main__":
    main()
