"""Render driver (SPEC 1 stages sample -> gate -> prompt -> render -> parse -> check -> dedup -> write).

  python3 -B driver.py --out RUN_DIR --endpoint http://127.0.0.1:PORT --model M --n 1000 [--mode completions|chat]
      [--shard-seed S] [--register RM] [--skeletons FILE.jsonl] [--concurrency 8] [--max-attempts 2]
      [--allow-real-teacher] [--stop-after K] [--log-every 30]

Skeletons come from skeleton.iter_shard (or a fixed jsonl list), are written to the manifest, checked with
render_prompt.feasible (SKEL_INFEASIBLE costs no teacher call), rendered by up to --concurrency requests in flight,
parsed and checked (checker.run), then deduplicated (dedup.py) against every accepted conversation. Each attempt
writes exactly one record: accepted/*.jsonl or rejects/*.jsonl (primary code, all codes). A rejected attempt is
retried once with a new render seed (SPEC 1), except SKEL_INFEASIBLE. A request that fails after the client's
retries is a TEACHER_ERROR reject for that attempt. yield.jsonl gets a snapshot every --log-every seconds and at the
end. Killed at any point, the same command resumes: finished attempts are never redone (driver_state.py).

Only the fake stub (fake_server.py) can be used without --allow-real-teacher (teacher_client.py)."""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import signal
import sys
import time

sys.dont_write_bytecode = True

import checker  # noqa: E402
import driver_state  # noqa: E402
import heldout  # noqa: E402
import records  # noqa: E402
import render_prompt as R  # noqa: E402
import shards  # noqa: E402
import skeleton  # noqa: E402
import teacher_client as TC  # noqa: E402
import yieldlog  # noqa: E402

DEFAULTS = {"n": 100, "shard_seed": "s0", "register": "RM", "skeletons": None, "concurrency": 8,
            "max_attempts": 2, "shard_size": 5000, "log_every": 30.0, "stop_after": None, "near_t": 0.7,
            "fsync": True, "quiet": False}


def file_source(path):
    def src(start_j, seen):
        with open(path, encoding="utf-8") as f:
            for j, line in enumerate(f, 1):
                if j > start_j and line.strip():
                    yield j, json.loads(line)
    return src


def shard_source(shard_seed, register):
    def src(start_j, seen):
        return skeleton.iter_shard(shard_seed, register, start_j=start_j, seen=seen)
    return src


class Driver:
    def __init__(self, cfg, client):
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        self.client = client
        client.check_endpoint()          # fail closed before anything is written
        self.out = c["out"]
        c["gen_version"] = skeleton.GEN_VERSION
        c["gate_hash"] = heldout.gate_hash()
        c["source"] = f"file:{os.path.basename(c['skeletons'])}" if c["skeletons"] else "iter_shard"
        c.setdefault("created", round(time.time()))
        self.run_info = driver_state.pin(self.out, c)
        self.n = self.run_info["n"]
        self.tally = yieldlog.Tally()
        self.state = driver_state.State(self.out, c["max_attempts"], c["near_t"])
        self.resumed = self.state.load(self.tally, skeleton.triple)
        src = file_source(c["skeletons"]) if c["skeletons"] else shard_source(c["shard_seed"], c["register"])
        self.source = src(self.state.last_j, self.state.seen)
        self.w_acc = shards.Writer(os.path.join(self.out, "accepted"), "accepted", c["shard_size"], c["fsync"])
        self.w_rej = shards.Writer(os.path.join(self.out, "rejects"), "rejects", c["shard_size"], c["fsync"])
        self.w_man = shards.Writer(os.path.join(self.out, "skeletons"), "skeletons", c["shard_size"], c["fsync"])
        self.queue = collections.deque((sid, self.state.next_attempt(sid)) for sid in self.state.order)
        self.written = 0
        self.stopped = None

    def log(self, msg):
        if not self.cfg["quiet"]:
            print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

    # --- work ------------------------------------------------------------------------------------------------------
    def next_work(self):
        if self.queue:
            return self.queue.popleft()
        while self.state.n_manifest < self.n:
            try:
                j, sk = next(self.source)
            except StopIteration:
                self.n = self.state.n_manifest
                return None
            self.w_man.write({"j": j, "skel": sk})
            self.state.n_manifest += 1
            self.state.last_j = j
            self.state.skels[sk["skel_id"]] = sk
            return sk["skel_id"], 0
        return None

    def teacher(self, built, seed, call=None):
        return records.teacher_meta(self.client, call, built, seed)

    def emit(self, sk, attempt, ok, rec, call=None):
        (self.w_acc if ok else self.w_rej).write(rec)
        self.written += 1
        primary = None if ok else rec["primary"]
        self.tally.add(sk, ok, primary, rec.get("codes", ()), rec.get("est_tokens", 0) if ok else 0, call)
        self.state.mark(sk["skel_id"], attempt, primary)
        if not ok and not self.state.finished(sk["skel_id"]):
            self.queue.appendleft((sk["skel_id"], attempt + 1))

    def finish(self, sk, attempt, built, seed, call, err):
        tm = self.teacher(built, seed, call)
        if err is not None:
            rec = records.reject(sk, attempt, "TEACHER_ERROR", ["TEACHER_ERROR"], tm,
                                 extra={"error": err.detail, "requests": err.requests})
            return self.emit(sk, attempt, False, rec, {"requests": err.requests})
        try:
            res = checker.run(sk, call["text"], built)
        except Exception as e:  # a checker bug must not end an overnight run; it is logged as its own code
            rec = records.reject(sk, attempt, "CHECKER_ERROR", ["CHECKER_ERROR"], tm, call, None, call["text"],
                                 extra={"error": f"{type(e).__name__}: {e}"[:500]})
            return self.emit(sk, attempt, False, rec, call)
        if not res["ok"]:
            rec = records.reject(sk, attempt, res["primary"], res["codes"], tm, call, res, call["text"])
            return self.emit(sk, attempt, False, rec, call)
        turns = checker.record_turns(sk, res)
        dup, keys = self.state.index.check(turns, sk["slots"])
        if dup:
            code, match, jac = dup
            rec = records.reject(sk, attempt, code, [code], tm, call, None, call["text"],
                                 extra={"dup_of": match, "jaccard": jac})
            return self.emit(sk, attempt, False, rec, call)
        rec = records.accepted(sk, built, res, call, attempt, tm, turns, keys)
        self.state.index.add(rec["conv_id"], keys)
        self.emit(sk, attempt, True, rec, call)

    def dispatch(self, ex, inflight, item):
        sid, attempt = item
        sk = self.state.skels[sid]
        built = R.build(sk)
        seed = TC.render_seed(sid, attempt)
        pre = R.feasible(sk)
        if pre:
            codes = [c for c, _ in pre]
            rec = records.reject(sk, attempt, codes[0], sorted(set(codes)), None,
                                 extra={"detail": [d for _, d in pre][:5]})
            return self.emit(sk, attempt, False, rec)
        inflight[ex.submit(self.client.render, built, seed)] = (sk, attempt, built, seed)

    # --- loop ------------------------------------------------------------------------------------------------------
    def snapshot(self, final=False):
        snap = self.tally.snapshot({"final": final, "stopped": self.stopped, "n": self.n,
                                    "manifest": self.state.n_manifest, "torn_bytes": self.state.torn,
                                    "resumed_records": list(self.resumed), "dup_pairs": self.state.dup_pairs})
        yieldlog.append(os.path.join(self.out, "yield.jsonl"), snap)
        return snap

    def run(self):
        c = self.cfg
        self.log(f"start: {self.state.n_manifest}/{self.n} skeletons in the manifest, "
                 f"{len(self.queue)} unfinished, resumed records {self.resumed}")
        if self.state.torn:
            self.log(f"cut torn lines: {self.state.torn}")
        ex = cf.ThreadPoolExecutor(max_workers=c["concurrency"])
        inflight, last_log = {}, time.time()
        try:
            while True:
                while self.stopped is None and len(inflight) < c["concurrency"]:
                    item = self.next_work()
                    if item is None:
                        break
                    self.dispatch(ex, inflight, item)
                    self._maybe_stop()
                if not inflight:
                    if self.stopped is None and (self.queue or self.state.n_manifest < self.n):
                        continue
                    break
                done, _ = cf.wait(list(inflight), timeout=1.0, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    sk, attempt, built, seed = inflight.pop(fut)
                    try:
                        call, err = fut.result(), None
                    except TC.TeacherError as e:
                        call, err = None, e
                    if self.stopped is None:
                        self.finish(sk, attempt, built, seed, call, err)
                        self._maybe_stop()
                if self.stopped is not None:
                    break
                if time.time() - last_log >= c["log_every"]:
                    self.snapshot()
                    self.log(self.tally.line())
                    last_log = time.time()
        except KeyboardInterrupt:
            self.stopped = "interrupted"
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            for w in (self.w_acc, self.w_rej, self.w_man):
                w.close()
        snap = self.snapshot(final=self.stopped is None)
        self.log(("stopped (" + self.stopped + "): " if self.stopped else "done: ") + self.tally.line())
        return snap

    def _maybe_stop(self):
        if self.cfg["stop_after"] is not None and self.written >= self.cfg["stop_after"]:
            self.stopped = "stop_after"


def _sigterm(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="google/gemma-4-12b-qat")
    ap.add_argument("--mode", default="completions", choices=["completions", "chat"])
    ap.add_argument("--license", default=None, help="teacher license; default guessed from the model id")
    ap.add_argument("--allow-real-teacher", action="store_true")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--http-tries", type=int, default=4)
    ap.add_argument("--backoff", type=float, default=1.0)
    for k, v in DEFAULTS.items():
        if k in ("fsync", "quiet"):
            continue
        typ = type(v) if v is not None else (int if k == "stop_after" else str)
        ap.add_argument("--" + k.replace("_", "-"), type=typ, default=v)
    ap.add_argument("--no-fsync", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _sigterm)
    client = TC.TeacherClient(a.endpoint, a.model, a.mode, allow_real=a.allow_real_teacher, timeout=a.timeout,
                              http_tries=a.http_tries, backoff=a.backoff, license=a.license)
    cfg = {k: getattr(a, k) for k in DEFAULTS if k not in ("fsync", "quiet")}
    cfg.update(out=a.out, fsync=not a.no_fsync, quiet=a.quiet)
    snap = Driver(cfg, client).run()
    print(json.dumps({k: snap[k] for k in ("attempts", "accepted", "rejected", "yield", "by_primary", "final")}))
    return 0 if snap["final"] else 3


if __name__ == "__main__":
    sys.exit(main())
