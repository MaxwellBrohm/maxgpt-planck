"""queue_dev_baselines.sh with engines.json "shards" and its run file (notes STEP 14X), on test_queue_sessions.py's fakes
(real dev_batch.py with fake responders, Q_LIMIT 12; no model, no GPU). test_queue.py runs these cases too.
  S1 a model with shards 3: per seed 3 processes, --shard 0/3, 1/3, 2/3 in order, all in one gpu.lock session, each
     with its own log and status lines noted "shard I/3"; the merged runs byte-identical to the same model run
     unsharded; DONE "complete 4 of 4 runs"; the run file holds the effective settings and its run id is the start
     line's
  S2 a failing shard (1/3): shard 2 is not run and the next seed is skipped; a restart runs shards 1 and 2 only and
     the runs equal the unsharded ones
  S3 every part DONE but no merge (a cut): the queue runs only the last shard, which merges; with the twin's part 0
     missing too (S3b) it runs shard 0 (that part only) and the last
  S4 a shards value that is not an int >= 1: BADSHARDS status lines and no process
  python3 -B test_queue_shards.py   (exit 0 = pass)"""
import json
import os
import sys
import tempfile

from test_queue_sessions import check, fails, lock_calls, procs, queue, setup, status

LIMIT = "12"


def base(tag, shards):
    b = tempfile.mkdtemp(prefix=f"rc12_qsh{tag}_")
    setup(b)
    eng = json.load(open(f"{b}/engines.json"))
    if shards is not None:
        eng["models"]["SAMPLER"]["shards"] = shards
    json.dump(eng, open(f"{b}/engines.json", "w"))
    return b


def files(b, seed, suf=""):
    d = f"{b}/root/SAMPLER/template/{seed}{suf}"
    if not os.path.exists(f"{d}/DONE"):
        return None
    return [open(f"{d}/{f}").read() for f in ("transcripts.jsonl", "scores.jsonl")]


def same(a, b, seeds, tag):
    for s in seeds:
        for suf in ("", "_owncf"):
            fa, fb = files(a, s, suf), files(b, s, suf)
            check(fa is not None and fa == fb, f"{tag}: {s}{suf} differs from the unsharded run")
    left = [d for d, _, _ in os.walk(f"{b}/root") if d.endswith(".shards") or d.endswith(".partial")]
    check(not left, f"{tag}: left behind {left}")


def shard_args(b):
    out = []
    for x in procs(b):
        a = x["args"]
        out.append(a[a.index("--seeds") + 1] + " " + (a[a.index("--shard") + 1] if "--shard" in a else "-"))
    return out


def parts_only(tag, drop):
    """every part of seed greedy DONE and nothing merged (dev_batch.py with the merge stubbed), minus part drop."""
    import shutil
    import dev_batch as DB
    d = base(tag, 3)
    real, old = DB.SH.merge, sys.argv
    DB.SH.merge = lambda *x, **k: None
    try:
        for i in range(3):
            sys.argv = ["dev_batch.py", "--responder", "fake:SAMPLER", "--render", "template", "--seeds", "greedy",
                        "--root", f"{d}/root", "--engines", f"{d}/engines.json", "--limit", LIMIT, "--shard", f"{i}/3"]
            DB.main()
    finally:
        DB.SH.merge, sys.argv = real, old
    check(files(d, "greedy") is None, f"{tag}: the no-merge stand-in did not hold")
    if drop:
        shutil.rmtree(f"{d}/root/SAMPLER/template/{drop}")
    return d


def shards():
    ref = base("ref", None)
    check(queue(ref, "SAMPLER", seeds="greedy 1", renders="template") == 0, "ref: queue exit")
    a = base("S1", 3)
    check(queue(a, "SAMPLER", seeds="greedy 1", renders="template") == 0, "S1: queue exit")
    check(shard_args(a) == ["greedy 0/3", "greedy 1/3", "greedy 2/3", "1 0/3", "1 1/3", "1 2/3"],
          f"S1: processes {shard_args(a)}")
    lc, p = lock_calls(a), procs(a)
    check(len(lc) == 1 and all(x["held0"] and x["held1"] and not x["fds"] and x["rc"] == 0 for x in p),
          f"S1: lock calls {len(lc)}, processes {[(x['held0'], x['held1'], x['fds'], x['rc']) for x in p]}")
    st = status(a)
    notes = [x.get("note", "").split("_shard_")[-1] for x in st]
    check(len(st) == 12 and notes == [f"{i}/3" for i in range(3) for _ in "do"] * 2, f"S1: status notes {notes}")
    logs = sorted(os.listdir(f"{a}/logs/rc12_dev"))
    check(logs == sorted(f"SAMPLER_template_{s}_s{i}.log" for s in ("greedy", "1") for i in range(3)),
          f"S1: logs {logs}")
    same(ref, a, ("greedy", "1"), "S1")
    done = open(f"{a}/logs/rc12_dev_queue.DONE").readline().strip()
    check(done == "complete 4 of 4 runs", f"S1: DONE {done}")
    run = dict(line.rstrip("\n").split("=", 1) for line in open(f"{a}/logs/rc12_dev_queue.run"))
    want = {"Q_MODELS": "SAMPLER", "Q_RENDERS": "template", "Q_SEEDS": "greedy 1", "Q_ROOT": f"{a}/root",
            "Q_ENGINES": f"{a}/engines.json", "Q_TIMEOUT": "10800", "Q_SESSION_S": "1800", "Q_SESSION_TAIL": "300",
            "Q_LIMIT": LIMIT}
    start = [x for x in open(f"{a}/logs/rc12_dev_queue.log") if "queue: start," in x]
    check(all(run.get(k) == v for k, v in want.items()) and len(start) == 1
          and start[0].rstrip().endswith(f", run {run.get('run')}") and f", pid {run.get('pid')}," in start[0],
          f"S1: run file {run} vs start line {start}")
    c = base("S2", 3)
    queue(c, "SAMPLER", seeds="greedy 1", renders="template", FAKE_FAIL="--seeds greedy --root")
    st = status(c)
    check(shard_args(c) == ["greedy 0/3"] and [x["exit"] for x in st] == ["1", "1", "SKIP", "SKIP"],
          f"S2: first pass {shard_args(c)} {[x['exit'] for x in st]}")
    c2 = base("S2b", 3)
    queue(c2, "SAMPLER", seeds="greedy", renders="template", FAKE_FAIL="--shard 1/3")
    check(shard_args(c2) == ["greedy 0/3", "greedy 1/3"] and [x["exit"] for x in status(c2)] == ["0", "0", "1", "1"],
          f"S2: shard 1 fails {shard_args(c2)} {[x['exit'] for x in status(c2)]}")
    queue(c2, "SAMPLER", seeds="greedy", renders="template")
    check(shard_args(c2) == ["greedy 1/3", "greedy 2/3"], f"S2: restart {shard_args(c2)}")
    same(ref, c2, ("greedy",), "S2")
    for tag, drop in (("S3", None), ("S3b", "greedy_owncf.shards/0of3")):
        d = parts_only(tag, drop)
        queue(d, "SAMPLER", seeds="greedy", renders="template")
        want = ["greedy 2/3"] if drop is None else ["greedy 0/3", "greedy 2/3"]
        check(shard_args(d) == want, f"{tag}: processes {shard_args(d)}, want {want}")
        same(ref, d, ("greedy",), tag)
    e = base("S4", "3")
    queue(e, "SAMPLER", seeds="greedy", renders="template")
    st = status(e)
    check(not procs(e) and [x["exit"] for x in st] == ["BADSHARDS"] * 2, f"S4: {[x['exit'] for x in st]}")


if __name__ == "__main__":
    shards()
    print("\n".join(f"FAIL {f}" for f in fails) + f"\ntest_queue_shards: {'PASS' if not fails else f'{len(fails)} FAILURES'}")
    sys.exit(1 if fails else 0)
