"""Test of queue_dev_baselines.sh + queue_status.py on the Mac with fakes (no model, no GPU; notes STEP 9 QUEUE).
A shim flock (the Mac has none) logs every call, can refuse the first N gpu.lock calls with exit 75, and runs the
command. Checks: plan order (model, render template then plain, seed, dev before owncf), one lock call per process,
one status line per run with the right counts, seeds and renders reaching the runs, the own-cf twin, NOENGINE, a
failing process skipping the rest of its render, lock-timeout retries, a restart adding nothing for finished runs and
leaving them byte-identical, a half-finished seed running only its missing run, the STOP file, the DONE marker.
  python3 -B test_queue.py        (exit 0 = pass)"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = """#!/bin/bash
[ "$1" = -n ] && exit 0
echo "$*" >> "$SHIM_LOG"
n=$(wc -l < "$SHIM_LOG")
[ -n "$FLOCK_FAIL" ] && [ "$n" -le "$FLOCK_FAIL" ] && exit 75
shift 5; exec "$@"
"""
fails = []


def check(cond, what):
    if not cond:
        fails.append(what)


def queue(base, models, seeds="greedy 1", renders="template plain", **env):
    e = dict(os.environ, Q_CODE=HERE, Q_ROOT=f"{base}/root", Q_LOGS=f"{base}/logs", Q_LOCKS=f"{base}/locks",
             Q_PY=sys.executable, Q_ENGINES=f"{base}/engines.json", Q_MODELS=models, Q_SEEDS=seeds,
             Q_RENDERS=renders, Q_LIMIT="12", PATH=f"{base}/shim:" + os.environ["PATH"], SHIM_LOG=f"{base}/flock.log")
    e.update(env)
    if os.path.exists(e["SHIM_LOG"]):
        os.remove(e["SHIM_LOG"])
    return subprocess.run(["bash", os.path.join(HERE, "queue_dev_baselines.sh")], env=e, timeout=300).returncode


def setup(base):
    os.makedirs(f"{base}/shim")
    open(f"{base}/shim/flock", "w").write(SHIM)
    os.chmod(f"{base}/shim/flock", 0o755)
    json.dump({"models": {m: {"engine": "fake"} for m in ("IDEAL", "CAPPER", "SAMPLER", "NOPE")}},
              open(f"{base}/engines.json", "w"))


def status(base):
    p = f"{base}/root/queue_status.txt"
    return [dict([("key", " ".join(line.split()[:5]))] + [kv.split("=", 1) for kv in line.split()[5:]])
            for line in open(p)] if os.path.exists(p) else []


def lock_calls(base):
    p = f"{base}/flock.log"
    return [line.split() for line in open(p)] if os.path.exists(p) else []


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


def replies_of(d):
    return [t["reply"] for line in open(f"{d}/transcripts.jsonl") for t in json.loads(line)["turns"]]


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
            stops = {k: int(v) for k, v in (kv.split(":") for kv in x["stops"].split(","))}
            check(ok and sum(stops.values()) == 144, f"fresh: counts {x}")
            check((m == "CAPPER") == ("cap" in stops), f"fresh: stop reasons {x['key']} {x['stops']}")
        elif m == "PARROT":
            check(x["exit"] == "NOENGINE" and x["done"] == "0", f"fresh: NOENGINE {x}")
        else:
            s = x["key"].split()[2]
            check((x["exit"] != "0" and x["exit"] != "SKIP") if s == "greedy" else x["exit"] == "SKIP",
                  f"fresh: failure then skip {x}")
    lc = lock_calls(base)
    check(len(lc) == 14 and all(c[:4] == ["-E", "75", "-w", "7200"] and c[4] == f"{base}/locks/gpu.lock"
                                and "dev_batch.py" in c for c in lc), f"fresh: lock calls {len(lc)}")
    check([c[c.index("--responder") + 1] + " " + c[c.index("--render") + 1] + " " + c[c.index("--seeds") + 1]
           for c in lc][:5] == ["fake:IDEAL template greedy", "fake:IDEAL template 1", "fake:IDEAL plain greedy",
                                "fake:IDEAL plain 1", "fake:CAPPER template greedy"], "fresh: process order")
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
    check(replies_of(f"{R}/SAMPLER/template/greedy") != replies_of(f"{R}/SAMPLER/template/1"), "fresh: seed reaches")
    check(open(f"{base}/logs/rc12_dev_queue.DONE").readline().strip() == "complete 24 of 40 runs", "fresh: DONE")
    before = tree_hash(R)
    n0 = len(st)
    check(queue(base, models) == 0, "restart: queue exit")
    st2 = status(base)[n0:]
    check(all(x["key"].split()[0] in ("PARROT", "NOPE") for x in st2) and len(st2) == 16, f"restart: new lines "
          f"{[x['key'] for x in st2]}")
    check(len(lock_calls(base)) == 2, f"restart: lock calls {len(lock_calls(base))}")
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
            check(len(lc4) == 4 and [x["exit"] for x in s4] == ["0"] * 4, f"lock retry: {len(lc4)} {s4}")
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
    check(open(f"{b}/logs/rc12_dev_queue.DONE").readline().strip() == "complete 0 of 208 runs", "default plan: DONE")
    status_unit()
    for f in fails:
        print("FAIL", f)
    print("test_queue:", "PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
