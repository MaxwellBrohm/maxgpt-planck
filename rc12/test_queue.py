"""Test of queue_dev_baselines.sh + queue_status.py on the Mac with fakes (no model, no GPU; notes STEP 9 QUEUE, 9g).
test_queue_sessions.py holds the fakes: a flock stand-in with real locks (the Mac has none; logs every call, can
refuse the first N gpu.lock calls with exit 75) and a Q_PY wrapper that logs every dev_batch.py process (lock held at
its start and end, lock fds inherited) before running it. Checks: plan order (model, render template then plain,
seed, dev before owncf), gpu.lock taken once for a queue whose processes fit one session, every process under it and
without the queue's lock fds, one status line per run with the right counts and the session note, seeds and renders
reaching the runs, the own-cf twin, NOENGINE, a failing process skipping the rest of its render, lock-timeout
retries on the session lock, a restart adding nothing for finished runs and leaving them byte-identical, a
half-finished seed running only its missing run, the STOP file, the DONE marker, and engines.json's per-model
"python" (notes STEP 9c): a model with one runs under it (~ expanded), the others under Q_PY, and a model whose python
is not executable gets NOPYTHON lines and no process; a queue killed while it waits for gpu.lock frees its queue
lock at once and starts nothing; then the session cases (test_queue_sessions.sessions).
  python3 -B test_queue.py        (exit 0 = pass)"""
import fcntl
import hashlib
import json
import os
import signal
import sys
import tempfile
import time

from test_queue_sessions import check, fails, held, lock_calls, procs, queue, sessions, setup, status, until

def tree_hash(root):
    h = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f != "queue_status.txt":
                p = os.path.join(dp, f)
                h[p] = (hashlib.sha256(open(p, "rb").read()).hexdigest(), os.stat(p).st_mtime_ns)
    return h


PANEL = ["Qwen/Qwen2.5-0.5B-Instruct", "tiiuae/Falcon-H1-Tiny-90M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct",
         "SmallDoge/Doge-160M-Instruct", "LiquidAI/LFM2.5-230M", "LiquidAI/LFM2.5-350M",
         "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen3.5-0.8B", "LiquidAI/LFM2-2.6B", "unsloth/gemma-3-270m-it",
         "Qwen/Qwen3-0.6B", "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B"]   # comparator, core by size, extras by size


def status_unit():
    """queue_status on hand-made runs: dev and owncf differ, so a swapped or mixed-up run is visible."""
    import queue_status as QS
    b = tempfile.mkdtemp(prefix="rc12_queue_st_")
    for suf, n, stops, secs in (("", 3, ["eot"] * 11 + ["cap"], 10), ("_owncf", 2, ["eos"] * 12, 4)):
        d = f"{b}/M/template/2{suf}"
        os.makedirs(d)
        with open(f"{d}/transcripts.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps({"turns": [{"stop": s} for s in stops]}) + "\n")
        json.dump({"seconds": secs, "finished": "2026-09-26 10:00:00"}, open(f"{d}/meta.json", "w"))
        open(f"{d}/DONE", "w").close()
    t0 = QS.time.mktime(QS.time.strptime("2026-09-26 11:00:00", "%Y-%m-%d %H:%M:%S"))
    got = [QS.run_line(b, "org/M", "template", "2", "vllm", run, "0", t0, t0 + 30) for run in ("dev", "owncf")]
    check(got == ["M template 2 vllm dev start=2026-09-26T09:59:50 end=2026-09-26T10:00:00 exit=0 done=1 convs=3 "
                  "replies=36 stops=cap:3,eot:33 secs=10 lock_secs=30",
                  "M template 2 vllm owncf start=2026-09-26T09:59:56 end=2026-09-26T10:00:00 exit=0 done=1 convs=2 "
                  "replies=24 stops=eos:24 secs=4 lock_secs=30"], f"status unit: {got}")
    os.makedirs(f"{b}/M/template/3.partial")
    got = QS.run_line(b, "org/M", "template", "3", "vllm", "dev", "1", t0, t0 + 5, "x y")
    check(got == "M template 3 vllm dev start=2026-09-26T11:00:00 end=2026-09-26T11:00:05 exit=1 done=0 convs=0 "
                 "replies=0 stops=- secs=- lock_secs=5 note=x_y", f"status unit: unfinished {got}")
    got = QS.progress(b, ["org/M", "org/N"], ["template", "plain"], ["2", "3"])
    check(got[0] == "complete 2 of 16 runs" and "2/4 done, dev run mean 10 s, in progress: 3" in got[1]
          and "0/4 done" in got[2], f"status unit: progress {got}")


def c_python():
    b = tempfile.mkdtemp(prefix="rc12_queue_py_")
    setup(b)
    os.makedirs(f"{b}/venv-x/bin")
    wrap = f"{b}/venv-x/bin/python"
    open(wrap, "w").write(f'#!/bin/bash\necho "$*" >> "{b}/wrap.log"\nexec "{sys.executable}" "$@"\n')
    os.chmod(wrap, 0o755)
    eng = json.load(open(f"{b}/engines.json"))
    eng["models"]["IDEAL"]["python"] = "~/venv-x/bin/python"
    eng["models"]["SAMPLER"]["python"] = f"{b}/missing/bin/python"
    json.dump(eng, open(f"{b}/engines.json", "w"))
    check(queue(b, "IDEAL CAPPER SAMPLER", seeds="greedy", renders="template", HOME=b) == 0, "python: queue exit")
    pys = {x["args"][x["args"].index("--responder") + 1]: x["py"] for x in procs(b)}   # the Q_PY wrapper's log
    check(pys == {"fake:CAPPER": f"{b}/shim/fakepy"}, f"python: per model {pys}")
    wl = open(f"{b}/wrap.log").read().splitlines() if os.path.exists(f"{b}/wrap.log") else []
    check(len(wl) == 1 and "--responder fake:IDEAL" in wl[0], f"python: wrapper ran {wl}")
    st = {x["key"]: x for x in status(b)}
    check([st[f"{m} template greedy fake {r}"]["exit"] for m in ("IDEAL", "CAPPER") for r in ("dev", "owncf")]
          == ["0"] * 4, f"python: runs {list(st)}")
    miss = [st.get(f"SAMPLER template greedy fake {r}", {}) for r in ("dev", "owncf")]
    check(all(x.get("exit") == "NOPYTHON" and x.get("done") == "0" and "missing" in x.get("note", "") for x in miss),
          f"python: NOPYTHON {miss}")


def c_killed_waiting():
    """the queue SIGKILLed while it waits for gpu.lock (held here): its queue lock is free at once (the flock waiter
    must not inherit fd 8, or it keeps the lock up to LOCK_WAIT), and no process starts once gpu.lock comes free."""
    b = tempfile.mkdtemp(prefix="rc12_queue_kw_")
    setup(b)
    os.makedirs(f"{b}/locks")
    g = os.open(f"{b}/locks/gpu.lock", os.O_RDONLY | os.O_CREAT, 0o666)
    fcntl.flock(g, fcntl.LOCK_EX)
    q = queue(b, "IDEAL", seeds="greedy", renders="template", wait=False)
    until(lambda: lock_calls(b), 300)
    time.sleep(0.5)
    q.send_signal(signal.SIGKILL)
    q.wait()
    time.sleep(0.3)
    qheld = held(f"{b}/locks/rc12_dev_queue.lock")
    os.close(g)
    time.sleep(1)
    check(lock_calls(b) and not qheld and not procs(b, "start") and not held(f"{b}/locks/gpu.lock"),
          f"killed waiting: queue lock held {qheld}, processes {len(procs(b, 'start'))}")


def replies_of(d):
    p = f"{d}/transcripts.jsonl"
    return [t["reply"] for line in open(p) for t in json.loads(line)["turns"]] if os.path.exists(p) else None


def main():
    base = tempfile.mkdtemp(prefix="rc12_queue_")
    setup(base)
    models = "IDEAL CAPPER PARROT NOPE SAMPLER"
    check(queue(base, models) == 0, "fresh: queue exit")
    st = status(base)
    want = []
    for m in models.split():
        for r in ("template", "plain"):
            for s in ("greedy", "1"):
                for run in ("dev", "owncf"):
                    want.append(f"{m} {r} {s} {'-' if m == 'PARROT' else 'fake'} {run}")
    check([x["key"] for x in st] == want, f"fresh: status keys/order {[x['key'] for x in st][:6]}...")
    for x in st:
        m = x["key"].split()[0]
        if m in ("IDEAL", "CAPPER", "SAMPLER"):
            ok = x["exit"] == "0" and x["done"] == "1" and x["convs"] == "12" and x["replies"] == "144"
            stops = {k: int(v) for k, v in (kv.split(":") for kv in x["stops"].split(",") if ":" in kv)}
            check(ok and sum(stops.values()) == 144, f"fresh: counts {x}")
            check((m == "CAPPER") == ("cap" in stops), f"fresh: stop reasons {x['key']} {x['stops']}")
        elif m == "PARROT":
            check(x["exit"] == "NOENGINE" and x["done"] == "0", f"fresh: NOENGINE {x}")
        else:
            s = x["key"].split()[2]
            check((x["exit"] != "0" and x["exit"] != "SKIP") if s == "greedy" else x["exit"] == "SKIP",
                  f"fresh: failure then skip {x}")
    lc, pr = lock_calls(base), [x["args"] for x in procs(base) if x["held0"] and x["held1"] and not x["fds"]]
    check(lc == [["-E", "75", "-w", "7200", "9"]] and len(pr) == 14, f"fresh: lock calls {lc}, processes {len(pr)}")
    check([c[c.index("--responder") + 1] + " " + c[c.index("--render") + 1] + " " + c[c.index("--seeds") + 1]
           for c in pr][:5] == ["fake:IDEAL template greedy", "fake:IDEAL template 1", "fake:IDEAL plain greedy",
                                "fake:IDEAL plain 1", "fake:CAPPER template greedy"], "fresh: process order")
    notes = [x.get("note", "") for x in st if x["exit"] not in ("NOENGINE", "SKIP")]   # 14 processes, 2 lines each
    sid = notes[0].split("_")[1] if notes and notes[0].count("_") == 3 else "?"
    check(sid.endswith(".1") and notes == [f"session_{sid}_process_{k // 2 + 1}" for k in range(28)],
          f"fresh: session notes {notes[:2]}")
    R = f"{base}/root"
    for m in ("IDEAL", "CAPPER", "SAMPLER"):
        for r in ("template", "plain"):
            for s in ("greedy", "1"):
                for suf, own in (("", False), ("_owncf", True)):
                    d = f"{R}/{m}/{r}/{s}{suf}"
                    meta = json.load(open(f"{d}/meta.json")) if os.path.exists(f"{d}/DONE") else {}
                    rows = [json.loads(line) for line in open(f"{d}/transcripts.jsonl")] if meta else []
                    check(meta.get("render") == r and meta.get("seed") == (None if s == "greedy" else 1)
                          and meta.get("own_cf") == own and all((x["family"] == "OWN") == own or not own for x in rows)
                          and all(x["seed"] == meta.get("seed") and x["render"] == r for x in rows), f"fresh: {d}")
    a, b = replies_of(f"{R}/SAMPLER/template/greedy"), replies_of(f"{R}/SAMPLER/template/1")
    check(a and b and a != b, "fresh: seed reaches")
    done = f"{base}/logs/rc12_dev_queue.DONE"
    check(os.path.exists(done) and open(done).readline().strip() == "complete 24 of 40 runs", "fresh: DONE")
    before = tree_hash(R)
    n0 = len(st)
    check(queue(base, models) == 0, "restart: queue exit")
    st2 = status(base)[n0:]
    check(all(x["key"].split()[0] in ("PARROT", "NOPE") for x in st2) and len(st2) == 16, f"restart: new lines "
          f"{[x['key'] for x in st2]}")
    check(len(lock_calls(base)) == 1 and len(procs(base)) == 2, f"restart: lock calls {len(lock_calls(base))}")
    after = tree_hash(R)
    check({k: v for k, v in after.items() if "/NOPE/" not in k} == before, "restart: finished runs touched")
    for f in os.listdir(f"{R}/IDEAL/template/1_owncf"):
        os.remove(f"{R}/IDEAL/template/1_owncf/{f}")
    os.rmdir(f"{R}/IDEAL/template/1_owncf")
    n1 = len(status(base))
    queue(base, "IDEAL")
    st3 = status(base)[n1:]
    check([x["key"] for x in st3] == ["IDEAL template 1 fake owncf"] and st3[0]["done"] == "1", f"half: {st3}")
    check(tree_hash(f"{R}/IDEAL/template/1") == {k: v for k, v in before.items() if "/IDEAL/template/1/" in k},
          "half: the finished dev run touched")
    for fail_n, ok in ((2, True), (3, False)):
        b = tempfile.mkdtemp(prefix="rc12_queue_lock_")
        setup(b)
        queue(b, "IDEAL", renders="template", FLOCK_FAIL=str(fail_n))
        s4, lc4 = status(b), lock_calls(b)
        if ok:
            check(len(lc4) == 3 and [x["exit"] for x in s4] == ["0"] * 4 and len(procs(b)) == 2, f"lock retry: {s4}")
        else:
            check(len(lc4) == 3 and [x["exit"] for x in s4] == ["75", "75", "SKIP", "SKIP"], f"lock timeout: {s4}")
    b = tempfile.mkdtemp(prefix="rc12_queue_stop_")
    setup(b)
    os.makedirs(f"{b}/logs")
    open(f"{b}/logs/rc12_dev_queue.STOP", "w").close()
    queue(b, "IDEAL")
    check(not status(b) and not lock_calls(b) and not os.path.exists(f"{b}/logs/rc12_dev_queue.DONE"), "STOP")
    b = tempfile.mkdtemp(prefix="rc12_queue_plan_")       # the default plan: no model in engines.json, so NOENGINE
    setup(b)
    queue(b, "", Q_MODELS="", Q_RENDERS="", Q_SEEDS="")
    want = [f"{m.split('/')[-1]} {r} {s} - {run}" for m in PANEL for r in ("template", "plain")
            for s in ("greedy", "1", "2", "3") for run in ("dev", "owncf")]
    check([x["key"] for x in status(b)] == want and not lock_calls(b), "default plan: order or content")
    done = f"{b}/logs/rc12_dev_queue.DONE"
    check(os.path.exists(done) and open(done).readline().strip() == "complete 0 of 208 runs", "default plan: DONE")
    status_unit()
    c_python()
    c_killed_waiting()
    sessions()


if __name__ == "__main__":
    finished = False
    try:
        main()
        finished = True
    finally:   # the failures so far are printed even when a check crashes (then the traceback follows)
        print("".join(f"FAIL {f}\n" for f in fails) + "test_queue: " + ("PASS" if finished and not fails else
              f"{len(fails)} FAILURES" + ("" if finished else ", then a crash")))
    sys.exit(1 if fails else 0)
