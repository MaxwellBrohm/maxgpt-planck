#!/usr/bin/env python3
"""Stand-in for the venv python in queue_k.sh tests (PLANCK_PY), stdlib only: no model, no GPU. Fakes ktrain.py,
kconfig.py check, kpreflight.py, probes.py and bpb.py; bpb_lines.py and -c snippets run for real. Every call is
appended to $FAKE_LOG as {tool, args, gpu_lock_held}. $FAKE_PLAN {run: [behaviour per attempt]} drives ktrain.py
(ok | crash | diverge | stop | wait_stop: runs until the run's STOP file appears, at most 60 s, then exits
0 at a checkpoint | refuse: exits 2 before any step, as train.py's prereg gate and ktrain's k block do);
$FAKE_KCHECK_REFUSE and $FAKE_REFUSE (comma lists) refuse runs in kconfig check and kpreflight; $FAKE_PROBE_FAIL=1
makes probes.py fail."""
import json
import os
import re
import sys

args = sys.argv[1:]
tool = os.path.basename(args[0]) if args else ""
if not args or args[0] == "-c" or tool == "bpb_lines.py":
    os.execv(sys.executable, [sys.executable] + args)


def lock_held() -> bool:
    import fcntl
    with open(os.path.join(os.environ["PLANCK_HOME"], "locks", "gpu.lock"), "a") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(lk, fcntl.LOCK_UN)
        return False


with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"tool": tool, "args": args[1:], "gpu_lock_held": lock_held()}) + "\n")


def touch(p):
    open(p, "w").close()


def ktrain():
    cfg = args[1]
    t = open(cfg).read()
    name = re.search(r"^name: (\S+)", t, re.M)[1]
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(cfg)), re.search(r"^out_dir: (\S+)", t, re.M)[1]))
    total = int((re.search(r"total_steps: (\d+)", t) or [0, 7630])[1])
    if "--require-committed" not in args:
        return 7
    sp = os.environ["FAKE_STATE"]
    st = json.load(open(sp)) if os.path.exists(sp) else {}
    k = st.get(name, 0)
    st[name] = k + 1
    json.dump(st, open(sp, "w"))
    plan = json.loads(os.environ.get("FAKE_PLAN", "{}")).get(name, [])
    beh = plan[k] if k < len(plan) else "ok"
    os.makedirs(out, exist_ok=True)
    if beh == "crash":
        return 1
    if beh == "refuse":                         # 2026-11-23 (K review finding 3)
        return 2
    if beh == "stop":
        touch(os.path.join(out, "ckpt_00000005.pt"))
        return 0
    if beh == "wait_stop":                      # 2026-11-22: a run in progress, ended by the queue's STOP file
        import time
        t0 = time.time()
        while not os.path.exists(os.path.join(out, "STOP")) and time.time() - t0 < 60:
            time.sleep(0.05)
        touch(os.path.join(out, "ckpt_00000005.pt"))
        return 0
    if beh == "diverge":
        with open(os.path.join(out, "log.jsonl"), "a") as f:
            f.write(json.dumps({"step": 9, "error": "non-finite loss"}) + "\n")
        return 1
    for s in (total - 20, total - 10):
        touch(os.path.join(out, f"ckpt_{s:08d}.pt"))
    touch(os.path.join(out, f"final_{total:08d}.pt"))
    return 0


def kconfig():
    if args[1] != "check":
        return 9
    if args[2] in os.environ.get("FAKE_KCHECK_REFUSE", "").split(","):
        print(f"{args[2]}: refused (fake)")
        return 1
    print("ok: 1 configs")
    return 0


def kpreflight():
    name = os.path.basename(args[1])[:-5]
    out = args[args.index("--out") + 1]
    bad = ["fake refusal"] if name in os.environ.get("FAKE_REFUSE", "").split(",") else []
    json.dump({"ok": not bad, "runs": [{"name": name, "refusals": bad}]}, open(out, "w"))
    return 3 if bad else 0


def probes():
    if os.environ.get("FAKE_PROBE_FAIL"):
        return 1
    out, items = args[args.index("--out") + 1], args[args.index("--items") + 1:]
    with open(out, "w") as f:
        for p in items:
            f.write(json.dumps({"item": os.path.basename(p)}) + "\n")
    json.dump({"summary": {}}, open(out + ".summary.json", "w"))
    return 0


def bpb():
    ck, out = args[1], args[args.index("--out") + 1]
    sets = {s: {"bits": 100.0 * (i + 1), "bytes": 100, "tokens": 30, "windows": 2, "bpb": float(i + 1),
                "truncated": 0} for i, s in enumerate(("cccc", "gutenberg", "oasst2", "wikimedia"))}
    json.dump({"checkpoint": os.path.basename(ck), "step": int(re.findall(r"\d+", os.path.basename(ck))[0]),
               "tokens_trained": 1, "evalset_sha256": "e", "tokenizer": {"sha256": "t"}, "precision": "fp32",
               "max_windows": None, "sets": sets}, open(out, "w"))
    return 0


sys.exit({"ktrain.py": ktrain, "kconfig.py": kconfig, "kpreflight.py": kpreflight, "probes.py": probes,
          "bpb.py": bpb}[tool]())
