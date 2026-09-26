#!/usr/bin/env python3
"""Stand-in for the venv python in queue tests (PLANCK_PY). Stdlib only; no model, no GPU.

Dispatch on the script the queue runs: train.py, preflight.py and bpb.py are faked; bpb_lines.py and
`-c` snippets run for real. Every call is appended to $FAKE_LOG as one JSON line {tool, args}.
$FAKE_PLAN (JSON {run name: [behaviour per attempt]}) drives train.py; attempts are counted in $FAKE_STATE.
Behaviours: ok | crash | prereg (exit 2, the gate refused) | stop (exit 0, no final) | diverge@K (non-finite loss at step K; a trunk keeps its
stable checkpoints below K) | wait_stop (run until out_dir/STOP appears, up to 5 s, then exit 0; else ok).
$FAKE_REFUSE: comma list of run names preflight refuses (not an init_from refusal).
"""
import json
import os
import re
import sys
import time

args = sys.argv[1:]
tool = os.path.basename(args[0]) if args else ""
if not args or args[0] == "-c" or tool == "bpb_lines.py":
    os.execv(sys.executable, [sys.executable] + args)
def lock_held() -> bool:        # can a NEW open file description take gpu.lock? then nobody holds it
    import fcntl
    with open(os.path.join(os.environ["PLANCK_HOME"], "locks", "gpu.lock"), "a") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(lk, fcntl.LOCK_UN)
        return False


with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"tool": tool, "args": args[1:], "cwd": os.getcwd(), "gpu_lock_held": lock_held()}) + "\n")


def cfg_of(path: str) -> dict:
    t = open(path).read()
    g = lambda pat: (re.search(pat, t, re.M) or [None, None])[1]    # noqa: E731
    d = {"name": g(r"^name: (\S+)"), "out_dir": g(r"^out_dir: (\S+)"), "mode": g(r"mode: (\w+)") or "full",
         "init_from": g(r"init_from: ([^,}\s]+)"), "total": int(g(r"total_steps: (\d+)") or 0),
         "points": [int(x) for x in (g(r"branch_points: \[([^\]]*)\]") or "").split(",") if x.strip()]}
    base = os.path.dirname(os.path.abspath(path))
    d["out"] = os.path.normpath(os.path.join(base, d["out_dir"]))
    d["init"] = os.path.normpath(os.path.join(base, d["init_from"])) if d["init_from"] else None
    return d


def touch(p: str) -> None:
    open(p, "w").close()


def train(cfg_path: str) -> int:
    c = cfg_of(cfg_path)
    if "--require-committed" not in args:
        return 7
    st_p = os.environ["FAKE_STATE"]
    st = json.load(open(st_p)) if os.path.exists(st_p) else {}
    k = st.get(c["name"], 0)
    st[c["name"]] = k + 1
    json.dump(st, open(st_p, "w"))
    plan = json.loads(os.environ.get("FAKE_PLAN", "{}")).get(c["name"], [])
    beh = plan[k] if k < len(plan) else "ok"
    os.makedirs(c["out"], exist_ok=True)
    log = os.path.join(c["out"], "log.jsonl")
    if beh == "crash":
        return 1
    if beh == "prereg":
        print("[train] refusing to start: prereg.yaml is not committed unchanged to git")
        return 2
    if beh == "stop":
        touch(os.path.join(c["out"], f"ckpt_{5:08d}.pt"))
        return 0
    if beh.startswith("diverge@"):
        at = int(beh.split("@")[1])
        for p in c["points"]:
            if p < at:
                touch(os.path.join(c["out"], f"stable_{p:08d}.pt"))
        with open(log, "a") as f:
            f.write(json.dumps({"step": at, "error": "non-finite loss"}) + "\n")
        print(f"FloatingPointError: non-finite loss at step {at}")
        return 1
    if beh == "wait_stop":
        t0 = time.time()
        while time.time() - t0 < 5:
            if os.path.exists(os.path.join(c["out"], "STOP")):
                touch(os.path.join(c["out"], f"ckpt_{7:08d}.pt"))
                return 0
            time.sleep(0.05)
    if c["mode"] == "branch" and not (c["init"] and os.path.exists(c["init"])):
        return 5
    for p in c["points"]:
        touch(os.path.join(c["out"], f"stable_{p:08d}.pt"))
    for s in (c["total"] - 20, c["total"] - 10):
        touch(os.path.join(c["out"], f"ckpt_{s:08d}.pt"))
    touch(os.path.join(c["out"], f"final_{c['total']:08d}.pt"))
    with open(log, "a") as f:
        f.write(json.dumps({"step": c["total"], "loss": 3.0}) + "\n")
    return 0


def preflight() -> int:
    cfg_path = args[1]
    out = args[args.index("--out") + 1]
    code = os.path.abspath(args[args.index("--code") + 1])
    c = cfg_of(cfg_path)
    bad = []
    if c["name"] in os.environ.get("FAKE_REFUSE", "").split(","):
        bad.append("engine: fake refusal")
    rel = os.path.relpath(c["init"], code) if c["init"] else None
    if c["mode"] == "branch" and not os.path.exists(c["init"]):
        bad.append(f"branch init_from {rel} does not exist")
    rep = {"ok": not bad, "strict": "--strict" in args,
           "runs": [{"name": c["name"], "refusals": bad, "init_from": rel, "schedule": {"mode": c["mode"]}}]}
    json.dump(rep, open(out, "w"))
    return 0 if not bad else 3


def bpb() -> int:
    ck, out = args[1], args[args.index("--out") + 1]
    if args[args.index("--precision") + 1] != "fp32" or args[args.index("--batch-tokens") + 1] != "16384":
        return 9
    sets = {s: {"bits": 100.0 * (i + 1), "bytes": 100, "tokens": 30, "windows": 2, "bpb": float(i + 1),
                "truncated": 0} for i, s in enumerate(("cccc", "gutenberg", "oasst2", "wikimedia"))}
    json.dump({"checkpoint": os.path.basename(ck), "step": int(re.findall(r"\d+", os.path.basename(ck))[0]),
               "tokens_trained": 1, "evalset_sha256": "e", "tokenizer": {"sha256": "t"}, "precision": "fp32",
               "max_windows": None, "sets": sets}, open(out, "w"))
    return 0


sys.exit({"train.py": lambda: train(args[1]), "preflight.py": preflight, "bpb.py": bpb}[tool]())
