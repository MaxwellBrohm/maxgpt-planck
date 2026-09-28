"""Render driver (SPEC 1 stages sample -> gate -> prompt -> render -> parse -> check -> dedup -> write).

  python3 -B driver.py --out RUN_DIR --endpoint http://127.0.0.1:PORT --model M --n 1000 [--mode completions|chat]
      [--shard-seed S] [--register RM] [--skeletons FILE.jsonl] [--concurrency 8] [--max-attempts 2]
      [--allow-real-teacher] [--stop-after K] [--log-every 30]
      [--structured off|labels|labels_exact] [--ban-dashes] [--preset NAME] [--repair]

Skeletons come from skeleton.iter_shard (or a fixed jsonl list), are written to the manifest, checked with
render_prompt.feasible (SKEL_INFEASIBLE costs no teacher call), rendered by up to --concurrency requests in flight,
parsed and checked (checker.run), then deduplicated (dedup.py) against every accepted conversation. Each attempt
writes exactly one record: accepted/*.jsonl or rejects/*.jsonl (primary code, all codes). A rejected attempt is
retried once with a new render seed (SPEC 1), except SKEL_INFEASIBLE. A request that fails after the client's
retries is a TEACHER_ERROR reject for that attempt. yield.jsonl gets a snapshot every --log-every seconds and at the
end. Killed at any point, the same command resumes: finished attempts are never redone (driver_state.py).

Only the fake stub (fake_server.py) can be used without --allow-real-teacher (teacher_client.py).

Flags added 2026-09-27 (all off by default; recorded as "run_flags" in every record and yield snapshot, pinned in
<out>/decode.json so a resume cannot mix them):
  --structured labels|labels_exact  each request carries its skeleton's planned label lines (constraint(), parse.plan
                                    order; labels_exact makes exact and tool lines literal) for serve_http's structured
                                    output; --ban-dashes asks for the dash-token ban; --preset names a D4 sampling
                                    preset. These three need a served teacher (drive.py --serve); the server's own
                                    account of what it applied is the record's teacher "decode".
  --repair                          targeted repair (SPEC 1: a pilot flag, off by default; SPEC 5 "no repair in v0"
                                    holds without it): when an attempt is a near miss (near_miss(): only FORMAT_*
                                    codes, or one REQ_SPAN / FORBID_SPAN hit and nothing else), the next attempt's
                                    prompt ends with a note naming what failed. Every record has "attempt_kind"
                                    (first, retry, repair) and yield is also logged per kind ("by_attempt_kind")."""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import signal
import sys
import time

sys.dont_write_bytecode = True

import checker  # noqa: E402
import driver_state  # noqa: E402
import heldout  # noqa: E402
import parse  # noqa: E402
import records  # noqa: E402
import render_prompt as R  # noqa: E402
import shards  # noqa: E402
import skeleton  # noqa: E402
import teacher_client as TC  # noqa: E402
import yieldlog  # noqa: E402

DEFAULTS = {"n": 100, "shard_seed": "s0", "register": "RM", "skeletons": None, "concurrency": 8,
            "max_attempts": 2, "shard_size": 5000, "log_every": 30.0, "stop_after": None, "near_t": 0.7,
            "fsync": True, "quiet": False, "structured": "off", "ban": None, "preset": None, "repair": False}
FLAGS = ("structured", "ban", "preset", "repair")
FLAGS_OFF = {"structured": "off", "ban": None, "preset": None, "repair": False}
KINDS = ("first", "retry", "repair")
FORM_CODES = {"FORMAT_LINES", "FORMAT_EXTRA", "FORMAT_WRAP"}
SPAN_CODES = {"REQ_SPAN", "FORBID_SPAN"}
REPAIR_HEAD = ("Note: an earlier try at this chat did not pass the automatic check. Write the whole chat again, every "
               "line from the first label to END, and fix this:")


def constraint(sk, mode):
    """serve_http's structured-output plan for one skeleton: [[label, literal or None]] in parse.plan order; with
    labels_exact an exact or tool line is its literal (the bank or program line the checker compares against)."""
    by_i = {t["i"]: t for t in sk["turns"]}
    lit = mode == "labels_exact"
    return {"mode": mode, "lines": [[lab, by_i[i]["text"] if lit and by_i[i]["mode"] == "exact" else None]
                                    for lab, i, _ in parse.plan(sk)]}


def attempt_kind(rec):
    """first, retry or repair; records written before 2026-09-27 carry no kind."""
    return rec.get("attempt_kind") or ("first" if rec["attempt"] == 0 else "retry")


def near_miss(rec):
    """'form' when every gating code of a reject is FORMAT_*, 'span' when its one gating failure is a single
    REQ_SPAN or FORBID_SPAN hit on a turn, else None. Driver codes (TEACHER_ERROR, DUP_*) are never near misses."""
    codes = [c for c in rec.get("codes") or () if c not in checker.REPORT_ONLY]
    if not codes:
        return None
    if set(codes) <= FORM_CODES:
        return "form"
    hits = [h for h in rec.get("hits") or () if h[0] not in checker.REPORT_ONLY]
    if len(codes) == 1 and codes[0] in SPAN_CODES and len(hits) == 1 and hits[0][1] is not None:
        return "span"
    return None


def repair_note(sk, rec):
    """the text that names what failed, one sentence per distinct failure (never the teacher's own words)."""
    labs = {i: lab for lab, i, _ in parse.plan(sk)}
    out = []
    for code, i, detail in rec.get("hits") or ():
        d = str(detail or "")
        if code in checker.REPORT_ONLY:
            continue
        if code == "REQ_SPAN":
            line = f'Line {labs[i]} must include "{d}" exactly as written.'
        elif code == "FORBID_SPAN":
            line = f'Line {labs[i]} must not include "{d}".'
        elif code == "FORMAT_LINES" and d == "no END":
            line = "The last line must be END, on a line of its own."
        elif code == "FORMAT_LINES" and d.startswith("missing"):
            line = ("Every script label needs its own line, in script order. Missing last time: "
                    + ", ".join(re.findall(r"[UAT]\d+", d)) + ".")
        elif code == "FORMAT_LINES" and d.startswith("duplicate"):
            line = f"Write each label once. {d.split()[-1]} came twice last time."
        elif code == "FORMAT_LINES":
            line = "Use exactly the labels of the script, once each, in script order."
        elif code == "FORMAT_EXTRA":
            line = "Write nothing before the first label or after END."
        else:
            line = "Keep each script line on a single line that starts with its label."
        if line not in out:
            out.append(line)
    return REPAIR_HEAD + "\n" + "\n".join(out)


def repair_built(built, note):
    """built with the note at the end of the prompt, and in the tail block so PROMPT_ECHO covers its wording."""
    b = {**built, "blocks": {**built["blocks"], "tail": built["blocks"]["tail"] + "\n\n" + note},
         "prompt": built["prompt"] + "\n\n" + note}
    b["prompt_sha256"] = hashlib.sha256(b["prompt"].encode()).hexdigest()
    return b


def pin_flags(out, flags, fresh):
    """<out>/decode.json: the run's flags; a resume with other flags is refused (so is one of an older run dir)."""
    path = os.path.join(out, "decode.json")
    if os.path.exists(path):
        with open(path) as f:
            have = json.load(f)
        if have != flags:
            raise SystemExit(f"refusing to resume {out}: decoding flags differ, run {have} != now {flags}")
        return
    if not fresh and flags != FLAGS_OFF:
        raise SystemExit(f"refusing to resume {out} (written before decode.json) with decoding flags {flags}")
    os.makedirs(out, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(flags, f, sort_keys=True)
    os.replace(path + ".tmp", path)


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
        self.flags = {k: c[k] for k in FLAGS}
        dec = {k: c[k] for k in ("structured", "ban", "preset")}
        if hasattr(client, "configure_decode"):
            client.configure_decode(dec)     # a client that cannot apply a control refuses it here
        elif any(dec[k] != FLAGS_OFF[k] for k in dec):
            raise TC.DecodeConfigError("this client has no decoding controls")
        client.check_endpoint()          # fail closed before anything is written
        self.out = c["out"]
        c["gen_version"] = skeleton.GEN_VERSION
        c["gate_hash"] = heldout.gate_hash()
        c["source"] = f"file:{os.path.basename(c['skeletons'])}" if c["skeletons"] else "iter_shard"
        c.setdefault("created", round(time.time()))
        pin_flags(self.out, self.flags, not os.path.exists(os.path.join(self.out, "run.json")))
        self.run_info = driver_state.pin(self.out, c)
        self.n = self.run_info["n"]
        self.tally = yieldlog.Tally()
        self.state = driver_state.State(self.out, c["max_attempts"], c["near_t"])
        self.resumed = self.state.load(self.tally, skeleton.triple)
        self.kinds = {k: [0, 0] for k in KINDS}      # attempted, accepted per attempt kind
        self.repair_src = {}                         # skel_id -> its last attempt's reject, when a near miss
        if any(self.resumed):
            self._rebuild_kinds()
        src = file_source(c["skeletons"]) if c["skeletons"] else shard_source(c["shard_seed"], c["register"])
        self.source = src(self.state.last_j, self.state.seen)
        self.w_acc = shards.Writer(os.path.join(self.out, "accepted"), "accepted", c["shard_size"], c["fsync"])
        self.w_rej = shards.Writer(os.path.join(self.out, "rejects"), "rejects", c["shard_size"], c["fsync"])
        self.w_man = shards.Writer(os.path.join(self.out, "skeletons"), "skeletons", c["shard_size"], c["fsync"])
        self.queue = collections.deque((sid, self.state.next_attempt(sid)) for sid in self.state.order)
        self.written = 0
        self.stopped = None

    def _rebuild_kinds(self):
        """per-kind counts and the pending repairs, from the records on disk (resume)."""
        last = {}
        for sub, ok in (("accepted", True), ("rejects", False)):
            for r in shards.read(os.path.join(self.out, sub), sub):
                k = attempt_kind(r)
                self.kinds[k][0] += 1
                self.kinds[k][1] += ok
                sid = r["skel_id"]
                if not ok and sid in self.state.skels and r["attempt"] > last.get(sid, {"attempt": -1})["attempt"]:
                    last[sid] = r
        if self.flags["repair"]:
            self.repair_src = {sid: r for sid, r in last.items() if near_miss(r)}

    def kind_of(self, sid, attempt):
        if attempt == 0:
            return "first"
        src = self.repair_src.get(sid)
        return "repair" if self.flags["repair"] and src is not None and src["attempt"] == attempt - 1 else "retry"

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
        tm = records.teacher_meta(self.client, call, built, seed)
        meta = getattr(self.client, "decode_meta", None)
        tm["decode"] = meta(call) if meta and call else None     # what the server applied to this request
        return tm

    def emit(self, sk, attempt, ok, rec, call=None, built=None):
        kind = (built or {}).get("attempt_kind") or ("first" if attempt == 0 else "retry")
        rec["attempt_kind"], rec["run_flags"] = kind, self.flags
        if kind == "repair":
            rec["repair"] = built["repair"]
        (self.w_acc if ok else self.w_rej).write(rec)
        self.written += 1
        self.kinds[kind][0] += 1
        self.kinds[kind][1] += ok
        sid = sk["skel_id"]
        primary = None if ok else rec["primary"]
        self.tally.add(sk, ok, primary, rec.get("codes", ()), rec.get("est_tokens", 0) if ok else 0, call)
        self.state.mark(sid, attempt, primary)
        self.repair_src.pop(sid, None)
        if not ok and not self.state.finished(sid):
            if self.flags["repair"] and near_miss(rec):
                self.repair_src[sid] = rec
            self.queue.appendleft((sid, attempt + 1))

    def finish(self, sk, attempt, built, seed, call, err):
        tm = self.teacher(built, seed, call)
        if err is not None:
            rec = records.reject(sk, attempt, "TEACHER_ERROR", ["TEACHER_ERROR"], tm,
                                 extra={"error": err.detail, "requests": err.requests})
            return self.emit(sk, attempt, False, rec, {"requests": err.requests}, built)
        try:
            res = checker.run(sk, call["text"], built)
        except Exception as e:  # a checker bug must not end an overnight run; it is logged as its own code
            rec = records.reject(sk, attempt, "CHECKER_ERROR", ["CHECKER_ERROR"], tm, call, None, call["text"],
                                 extra={"error": f"{type(e).__name__}: {e}"[:500]})
            return self.emit(sk, attempt, False, rec, call, built)
        if not res["ok"]:
            rec = records.reject(sk, attempt, res["primary"], res["codes"], tm, call, res, call["text"])
            return self.emit(sk, attempt, False, rec, call, built)
        turns = checker.record_turns(sk, res)
        dup, keys = self.state.index.check(turns, sk["slots"])
        if dup:
            code, match, jac = dup
            rec = records.reject(sk, attempt, code, [code], tm, call, None, call["text"],
                                 extra={"dup_of": match, "jaccard": jac})
            return self.emit(sk, attempt, False, rec, call, built)
        rec = records.accepted(sk, built, res, call, attempt, tm, turns, keys)
        self.state.index.add(rec["conv_id"], keys)
        self.emit(sk, attempt, True, rec, call, built)

    def dispatch(self, ex, inflight, item):
        sid, attempt = item
        sk = self.state.skels[sid]
        built = R.build(sk)
        kind = self.kind_of(sid, attempt)
        if kind == "repair":
            src = self.repair_src[sid]
            note = repair_note(sk, src)
            built = repair_built(built, note)
            built["repair"] = {"of_attempt": src["attempt"], "near_miss": near_miss(src), "note": note}
        built["attempt_kind"] = kind
        if self.flags["structured"] != "off":
            built["constraint"] = constraint(sk, self.flags["structured"])
        seed = TC.render_seed(sid, attempt)
        pre = R.feasible(sk)
        if pre:
            codes = [c for c, _ in pre]
            rec = records.reject(sk, attempt, codes[0], sorted(set(codes)), None,
                                 extra={"detail": [d for _, d in pre][:5]})
            return self.emit(sk, attempt, False, rec, None, built)
        inflight[ex.submit(self.client.render, built, seed)] = (sk, attempt, built, seed)

    # --- loop ------------------------------------------------------------------------------------------------------
    def snapshot(self, final=False):
        by_kind = {k: {"attempted": a, "accepted": n, "yield": round(n / a, 4) if a else None}
                   for k, (a, n) in self.kinds.items()}
        snap = self.tally.snapshot({"final": final, "stopped": self.stopped, "n": self.n,
                                    "manifest": self.state.n_manifest, "torn_bytes": self.state.torn,
                                    "resumed_records": list(self.resumed), "dup_pairs": self.state.dup_pairs,
                                    "run_flags": self.flags, "by_attempt_kind": by_kind})
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
    ap.add_argument("--structured", choices=TC.STRUCTURED, default="off",
                    help="structured output of the planned label lines (drive.py --serve only)")
    ap.add_argument("--ban-dashes", dest="ban", action="store_const", const="dash", default=None,
                    help="ban dash tokens at sampling (drive.py --serve only)")
    ap.add_argument("--preset", default=None, help="a D4 sampling preset of the served teacher: card, shared, ...")
    ap.add_argument("--repair", action="store_true", help="name what failed in the retry prompt after a near miss")
    for k, v in DEFAULTS.items():
        if k in ("fsync", "quiet") or k in FLAGS:
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
    try:
        drv = Driver(cfg, client)
    except TC.DecodeConfigError as e:
        raise SystemExit(str(e)) from e
    snap = drv.run()
    print(json.dumps({k: snap[k] for k in ("attempts", "accepted", "rejected", "yield", "by_primary", "final")}))
    return 0 if snap["final"] else 3


if __name__ == "__main__":
    sys.exit(main())
