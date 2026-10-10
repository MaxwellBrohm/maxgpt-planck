"""RC-12 dev_batch.py shards (rc12/shards.py; notes STEP 14X) without a model, in-process with fakes (lockstep, PerConv).
  equal      a run played as 3 shards (in order, and in the order 2, 0, 1) has transcripts.jsonl and scores.jsonl
             byte-identical to the same run played by one process, and the same meta values (conversations, seed,
             own_cf, R_ungated, loop_rate); the merged meta.json names shards 3 with 3 shard_seconds and shard_batches
             that add up; no .shards or .partial dir is left; seeds greedy and 1 of a seed-sensitive fake (SAMPLER)
  merge      no run dir exists before the last part; a done part is not replayed (mtimes kept); a stale part .partial
             is redone; every part DONE without a merge (a cut) is merged by the next call without loading an engine
  refuse     no --shard for a model engines.json shards; N other than engines.json's; a bad I/N; --shard for a
             model without shards; a non-int shards value; a part whose run config differs is not merged (all kept)
  verify     verify_dev_runs C1 passes the merged run under the engines.json that shards it and fails it under one
             that does not
Run: python3 -B test_dev_batch_shards.py   (exit 1 on any failure)"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import dev_batch as DB
import runner as R
import shards as SH
import verify_dev_runs as V

LIMIT = 40
fails = []


def check(cond, what):
    if not cond:
        fails.append(what)


def call(argv):
    old = sys.argv
    sys.argv = ["dev_batch.py"] + argv
    try:
        return DB.main()
    except SystemExit as e:
        return str(e.code)
    finally:
        sys.argv = old


def engines(path, shards):
    json.dump({"models": {"SAMPLER": {} if shards is None else {"shards": shards}}}, open(path, "w"))


def args(root, eng, seeds="greedy,1"):
    return ["--responder", "fake:SAMPLER", "--render", "plain", "--root", root, "--limit", str(LIMIT),
            "--engines", eng, "--seeds", seeds]


def runs(root):
    return [os.path.join(root, "SAMPLER", "plain", s) for s in ("greedy", "greedy_owncf", "1", "1_owncf")]


def files(d):
    return {f: open(os.path.join(d, f)).read() for f in ("transcripts.jsonl", "scores.jsonl")} \
        if os.path.exists(os.path.join(d, "DONE")) else None


def mtimes(root):
    return {os.path.join(d, f): os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs}


def leftovers(root):
    return [d for d, _, _ in os.walk(root) if d.endswith(".shards") or d.endswith(".partial")]


def same_as(ref, root, order, tag):
    for a, b in zip(runs(ref), runs(root)):
        fa, fb = files(a), files(b)
        check(fa is not None and fa == fb, f"equal {tag}: {os.path.basename(b)} files differ from one process")
        if fb is None:
            continue
        ma, mb = json.load(open(f"{a}/meta.json")), json.load(open(f"{b}/meta.json"))
        keys = ("conversations", "seed", "own_cf", "R_ungated", "loop_rate", "model", "render")
        check(all(ma[k] == mb[k] for k in keys), f"equal {tag}: meta {[k for k in keys if ma[k] != mb[k]]}")
        check(not set(ma) - set(mb), f"equal {tag}: merged meta lacks {sorted(set(ma) - set(mb))}")   # e.g. max_batch
        check(mb.get("shards") == 3 and len(mb.get("shard_seconds", [])) == 3
              and abs(sum(mb["shard_seconds"]) - mb["seconds"]) < 0.2
              and mb["batches"] == [x for p in mb["shard_batches"] for x in p] and len(mb["shard_batches"]) == 3,
              f"equal {tag}: shard fields {mb.get('shards')} {mb.get('shard_seconds')}")
    check(not leftovers(root), f"equal {tag}: left behind {leftovers(root)}")


def main():
    t = tempfile.mkdtemp(prefix="rc12_shards_")
    e1, e3 = f"{t}/e1.json", f"{t}/e3.json"
    engines(e1, None)
    engines(e3, 3)
    ref = f"{t}/ref"
    check(call(args(ref, e1)) == 0, "equal: the one-process run failed")
    a = f"{t}/a"
    for i in (0, 1):
        check(call(args(a, e3) + ["--shard", f"{i}/3"]) == 0, f"merge: shard {i} failed")
    check(not any(os.path.exists(r) for r in runs(a)) and all(SH.part_done(r, 0, 3) and SH.part_done(r, 1, 3)
                                                             for r in runs(a)), "merge: a run dir before the last part")
    before = mtimes(a)
    call(args(a, e3) + ["--shard", "1/3"])
    check(mtimes(a) == before, "merge: a done part was replayed")
    for i, sec in ((0, 1.5), (1, 2.25)):        # fake parts take 0.0 s: give two of them a time the merge must add
        pj = SH.part_dir(runs(a)[0], i, 3) + "/part.json"
        json.dump(dict(json.load(open(pj)), seconds=sec), open(pj, "w"))
    stale = SH.part_dir(runs(a)[0], 2, 3) + ".partial"
    os.makedirs(stale)
    open(f"{stale}/junk", "w").close()
    check(call(args(a, e3) + ["--shard", "2/3"]) == 0, "merge: shard 2 failed")
    same_as(ref, a, "0 1 2", "in order")
    m = json.load(open(f"{runs(a)[0]}/meta.json")) if os.path.exists(f"{runs(a)[0]}/meta.json") else {}
    sh = m.get("shard_seconds") or [0, 0, 0]
    check(sh[:2] == [1.5, 2.25] and m.get("seconds") == round(sum(sh), 1), f"merge: seconds {m.get('seconds')} {sh}")
    b = f"{t}/b"
    for i in (2, 0, 1):
        call(args(b, e3) + ["--shard", f"{i}/3"])
    same_as(ref, b, "2 0 1", "out of order")
    c = f"{t}/c"
    real = SH.merge
    DB.SH.merge = lambda *x, **k: None
    try:
        for i in range(3):
            call(args(c, e3) + ["--shard", f"{i}/3"])
    finally:
        DB.SH.merge = real
    check(not any(os.path.exists(r) for r in runs(c)), "merge: the no-merge stand-in did not hold")
    real_make = R.make_responder

    def boom(*x, **k):
        raise RuntimeError("engine loaded")
    R.make_responder = boom
    try:
        rc = call(args(c, e3) + ["--shard", "0/3"])
    except RuntimeError as e:
        rc = str(e)
    finally:
        R.make_responder = real_make
    check(rc == 0, f"merge: the cut run was not merged without an engine ({rc})")
    same_as(ref, c, "cut", "cut then merged")
    for argv, want, what in ((args(t + "/r", e3), "does not fit", "no --shard"),
                             (args(t + "/r", e3) + ["--shard", "1/4"], "does not fit", "N 4 vs 3"),
                             (args(t + "/r", e3) + ["--shard", "3/3"], "0 <= I < N", "I out of range"),
                             (args(t + "/r", e3) + ["--shard", "1"], "two integers", "no N"),
                             (args(t + "/r", e1) + ["--shard", "0/1"], "N >= 2", "N 1"),
                             (args(t + "/r", e1) + ["--shard", "0/2"], "does not fit", "model without shards")):
        rc = call(argv)
        check(want in str(rc), f"refuse: {what} ({rc})")
    bad = f"{t}/bad.json"
    json.dump({"models": {"SAMPLER": {"shards": "3"}}}, open(bad, "w"))
    rc = call(args(t + "/r", bad) + ["--shard", "0/3"])
    check("an int >= 1" in str(rc), f"refuse: shards '3' ({rc})")
    check(not os.path.exists(t + "/r"), "refuse: a refused call wrote a run")
    d = f"{t}/d"
    for i in (0, 1):
        call(args(d, e3, "greedy") + ["--shard", f"{i}/3"])
    p = SH.part_dir(runs(d)[0], 1, 3) + "/part.json"
    pj = json.load(open(p))
    pj["config"]["dtype"] = "float16"
    json.dump(pj, open(p, "w"))
    rc = call(args(d, e3, "greedy") + ["--shard", "2/3"])
    check("does not fit" in str(rc) and "config" in str(rc) and not os.path.exists(runs(d)[0])
          and SH.all_done(runs(d)[0], 3), f"refuse: a part with another config was merged ({str(rc)[:120]})")
    m = json.load(open(f"{runs(a)[0]}/meta.json"))
    run = {"meta": m, "rows": [json.loads(x) for x in open(f"{runs(a)[0]}/transcripts.jsonl")]}
    exp = (len(R.load(R.DEV, None, LIMIT)), len(R.load(R.DEV, ["OWN"], LIMIT)))
    V.FAILS.clear()
    with contextlib.redirect_stdout(io.StringIO()):          # verify's own PASS / FAIL / INFO lines
        V.c1_config("greedy", run, e3, exp)
        ok3 = not any("shards" in f for f in V.FAILS)
        V.FAILS.clear()
        V.c1_config("greedy", run, e1, exp)
    check(ok3 and any("shards" in f for f in V.FAILS), f"verify: C1 shards check (under e3 ok {ok3}, under e1 "
                                                       f"{V.FAILS})")
    V.FAILS.clear()
    shutil.rmtree(t, ignore_errors=True)
    print("\n".join(f"FAIL {f}" for f in fails) or "test_dev_batch_shards: PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
