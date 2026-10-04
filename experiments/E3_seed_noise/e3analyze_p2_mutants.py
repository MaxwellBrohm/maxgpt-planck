"""Mutation test of e3analyze_p2.py's checks on scratch copies (never the repo). A mutant is killed when the set of
failing checks differs from the unmutated copy's (the 4 D2 code checks).
  python e3analyze_p2_mutants.py   (copies this folder's Part 2 inputs to a temp directory; writes nothing here)"""
import json, os, shutil, subprocess, sys, tempfile
SRC = os.path.dirname(os.path.abspath(__file__))
WD = tempfile.mkdtemp(prefix="e3p2_mut_")
PY = sys.executable

def fresh():
    d = os.path.join(WD, "E3"); shutil.rmtree(d, ignore_errors=True); os.makedirs(os.path.join(d, "out"))
    for f in ("e3analyze.py", "e3analyze_p2.py", "results.json", "tables.txt"): shutil.copy(os.path.join(SRC, f), d)
    for sub in ("logs", "configs", "plans"): shutil.copytree(os.path.join(SRC, sub), os.path.join(d, sub))
    for n in os.listdir(os.path.join(SRC, "out")):
        if n.startswith("e3_20m_"): shutil.copytree(os.path.join(SRC, "out", n), os.path.join(d, "out", n))
    return d

def fails(d):
    r = subprocess.run([PY, os.path.join(d, "e3analyze_p2.py"), "--checks-only"], capture_output=True, text=True)
    if r.returncode: return {"CRASH: " + r.stderr.strip().splitlines()[-1]}
    return {c["check"] for c in json.loads(r.stdout) if not c["ok"]}

def jl(p): return [json.loads(x) for x in open(p)]
def wjl(p, rows): open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def edit_json(p, fn): x = json.load(open(p)); fn(x); json.dump(x, open(p, "w"), indent=1, sort_keys=True)
def sub(p, a, b, count=1):
    t = open(p).read(); assert a in t, (p, a); open(p, "w").write(t.replace(a, b, count))

def m_tail_drawn(d):
    p = f"{d}/out/e3_20m_B_s2/train_tail.jsonl"; r = jl(p); r[5]["drawn_cccc"] += 1; wjl(p, r)
def m_end_drawn(d): edit_json(f"{d}/out/e3_20m_A_s3/run_records.json", lambda x: x[1].__setitem__("drawn_oasst2", x[1]["drawn_oasst2"] + 1))
def m_schedule(d): edit_json(f"{d}/out/e3_20m_A_s1/run_records.json", lambda x: x[0]["schedule"].__setitem__("decay_start", 12206))
def m_bpb_step(d):
    p = f"{d}/out/e3_20m_B_s1/bpb.jsonl"; r = jl(p); i = next(i for i, x in enumerate(r) if x["step"] == 15250); r.pop(i); wjl(p, r)
def m_runio(d): sub(f"{d}/logs/part2/code_sha256_20261003_004304.txt", "ffcdf24596ab46cb", "ffcdf24596ab46cc")
def m_order(d):
    p = f"{d}/logs/part2/queue_e3.txt"; t = open(p).read()
    t = t.replace("train e3_20m_A_s2.yaml", "train XX.yaml").replace("train e3_20m_B_s2.yaml", "train e3_20m_A_s2.yaml").replace("train XX.yaml", "train e3_20m_B_s2.yaml")
    open(p, "w").write(t)
def m_compile(d): edit_json(f"{d}/out/e3_20m_B_s3/preflight.json", lambda x: x["runs"][0]["engine"].__setitem__("train.compile", True))
def m_resumed(d): edit_json(f"{d}/out/e3_20m_A_s2/run_records.json", lambda x: x[0].__setitem__("resumed_from", "ckpt_00014945.pt"))
def m_seed_collision(d):
    for a in "AB":
        e1 = json.load(open(f"{d}/out/e3_20m_{a}_s1/run_records.json"))[1]
        edit_json(f"{d}/out/e3_20m_{a}_s2/run_records.json", lambda x: x[1].update({k: v for k, v in e1.items() if k.startswith(("drawn_", "dropped_"))}))
def m_results(d): open(f"{d}/results.json", "a").write(" ")
def m_lr(d): sub(f"{d}/configs/e3_20m_B_s1.yaml", "lr: 0.0015, embed_lr", "lr: 0.003, embed_lr")
def m_tail_missing(d):
    p = f"{d}/out/e3_20m_A_s3/train_tail.jsonl"; r = jl(p); wjl(p, r[:-1])
def m_evalset(d):
    p = f"{d}/out/e3_20m_A_s2/bpb.jsonl"; r = jl(p); r[3]["evalset_sha256"] = "0" * 64; wjl(p, r)
def m_two_trains(d):
    p = f"{d}/logs/part2/queue_e3.txt"; t = open(p).read(); i = t.index("done e3_20m_A_s1 rc 0"); j = t.index("\n", i)
    open(p, "w").write(t[:j + 1] + t[i - 20:j + 1] + t[j + 1:])
def m_prereg(d): sub(f"{d}/configs/prereg.yaml", "id: E3", "id: E3 ")
def m_max_windows(d):
    p = f"{d}/out/e3_20m_B_s3/bpb.jsonl"; r = jl(p); r[0]["max_windows"] = 100; wjl(p, r)
def m_nonfinite(d):
    p = f"{d}/out/e3_20m_B_s2/train_tail.jsonl"; r = jl(p); r[10]["loss"] = float("nan"); wjl(p, r)
def m_queue_commit(d): sub(f"{d}/logs/part2/queue_e3.txt", "code commit 3c102f75dc68", "code commit 3c102f75dc69")

MUTANTS = [m_tail_drawn, m_end_drawn, m_schedule, m_bpb_step, m_runio, m_order, m_compile, m_resumed, m_seed_collision,
           m_results, m_lr, m_tail_missing, m_evalset, m_two_trains, m_prereg, m_max_windows, m_nonfinite, m_queue_commit]
base = fails(fresh()); print("baseline failing:", len(base), sorted(base))
killed = 0
for m in MUTANTS:
    d = fresh(); m(d); f = fails(d); k = f != base; killed += k
    print(f"{'KILLED ' if k else 'SURVIVED'} {m.__name__:18s} new failures: {sorted(f - base)}")
print(f"{killed}/{len(MUTANTS)} killed"); shutil.rmtree(WD)
