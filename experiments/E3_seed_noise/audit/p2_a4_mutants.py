"""E3 Part 2 audit: mutation test of the auditor's own checks (audit/p2_a2_checks.py) and of the recompute chain
(p2_a1_recompute.py + p2_a6_vs_results.py). Every mutant edits a scratch copy of this folder, never the repo.
A check mutant is killed when the set of failing checks differs from the unmutated copy's (none fail).
"Consistent" mutants also edit the PC printout in the scratch copy, so the copy-vs-PC hash cannot be what kills them.
  ~/.venvs/planck/bin/python audit/p2_a4_mutants.py SCRATCH_DIR"""
import hashlib, json, os, shutil, subprocess, sys

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHK = os.path.join(SRC, "audit", "p2_a2_checks.py")
WD = os.path.abspath(sys.argv[1]); PY = sys.executable


def fresh():
    d = os.path.join(WD, "E3"); shutil.rmtree(WD, ignore_errors=True); os.makedirs(WD)
    shutil.copytree(SRC, d, ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(WD, "E2_lr_transfer"))
    shutil.copy(os.path.join(SRC, "..", "E2_lr_transfer", "results.json"), os.path.join(WD, "E2_lr_transfer"))
    return d


def fails(d):
    r = subprocess.run([PY, "-B", CHK, d], capture_output=True, text=True)
    last = r.stdout.splitlines()[-1] if r.stdout.strip() else ""
    if not last.endswith(" pass"):
        return {"CRASH " + (r.stderr.strip().splitlines() or ["?"])[-1]}
    return {ln[5:] for ln in r.stdout.splitlines() if ln.startswith("FAIL ")}


def jl(p): return [json.loads(x) for x in open(p)]
def wjl(p, rows): open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def ej(p, fn): x = json.load(open(p)); fn(x); open(p, "w").write(json.dumps(x, indent=1, sort_keys=True) + "\n")
def sub(p, a, b):
    t = open(p).read(); assert a in t, (p, a); open(p, "w").write(t.replace(a, b, 1))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def resync_pc(d, run, fname, key):   # make the PC printout agree with the mutated copy
    p = f"{d}/audit/p2_pc_read.out"; lines = open(p).read().splitlines()
    lines = [f"SHA {key} {sha(f'{d}/out/{run}/{fname}')}" if ln.startswith(f"SHA {key} ") else ln for ln in lines]
    open(p, "w").write("\n".join(lines) + "\n")


def bpb_edit(d, run, fn, consistent=True):
    p = f"{d}/out/{run}/bpb.jsonl"; r = jl(p); fn(r); wjl(p, r)
    if consistent: resync_pc(d, run, "bpb.jsonl", f"{run}/bpb.jsonl")


def m_bpb_value(d): bpb_edit(d, "e3_20m_B_s2", lambda r: next(x for x in r if x["step"] == 15259 and x["set"] == "CHAT").update(bpb=0.97), consistent=False)
def m_windows_consistent(d): bpb_edit(d, "e3_20m_B_s1", lambda r: r[0].update(windows=r[0]["windows"] + 1))
def m_prose_row_consistent(d):
    def f(r):
        x = next(x for x in r if x["step"] == 15259 and x["set"] == "PROSE"); x["bits"] += 1000; x["bpb"] = x["bits"] / x["bytes"]
    bpb_edit(d, "e3_20m_A_s3", f)
def m_max_windows_consistent(d): bpb_edit(d, "e3_20m_B_s3", lambda r: r[0].update(max_windows=100))
def m_evalset_consistent(d): bpb_edit(d, "e3_20m_A_s2", lambda r: r[4].update(evalset_sha256="0" * 64))
def m_drop_step_consistent(d): bpb_edit(d, "e3_20m_B_s1", lambda r: r.__setitem__(slice(None), [x for x in r if x["step"] != 15250]))
def m_tail_drawn(d):
    p = f"{d}/out/e3_20m_A_s1/train_tail.jsonl"; r = jl(p); r[4]["drawn_cccc"] += 1; wjl(p, r)
def m_tail_crn_consistent(d):
    p = f"{d}/out/e3_20m_B_s2/train_tail.jsonl"; r = jl(p); r[7]["drawn_irc"] += 1; wjl(p, r)
    q = f"{d}/audit/p2_pc_read.out"; L = open(q).read().splitlines(); i = [k for k, ln in enumerate(L) if ln.startswith("TAIL e3_20m_B_s2 ")][7]
    L[i] = "TAIL e3_20m_B_s2 " + json.dumps(r[7], sort_keys=True); open(q, "w").write("\n".join(L) + "\n")
def m_tail_missing_consistent(d):
    p = f"{d}/out/e3_20m_A_s3/train_tail.jsonl"; r = jl(p); wjl(p, r[:-1])
    q = f"{d}/audit/p2_pc_read.out"; L = open(q).read().splitlines(); i = [k for k, ln in enumerate(L) if ln.startswith("TAIL e3_20m_A_s3 ")][-1]
    del L[i]; open(q, "w").write("\n".join(L) + "\n")
def m_resumed(d): ej(f"{d}/out/e3_20m_A_s2/run_records.json", lambda x: x[0].update(resumed_from="ckpt_00014945.pt"))
def m_host_back(d): ej(f"{d}/out/e3_20m_B_s3/run_records.json", lambda x: x[0]["env"].update(host="SOMEPC"))
def m_seed_collision(d):
    e1 = json.load(open(f"{d}/out/e3_20m_A_s1/run_records.json"))[1]
    for a in "AB":
        ej(f"{d}/out/e3_20m_{a}_s2/run_records.json", lambda x: x[1].update({k: v for k, v in e1.items() if k.startswith(("drawn_", "dropped_"))}))
def m_config_lr(d): sub(f"{d}/configs/e3_20m_B_s3.yaml", "lr: 0.0015, embed", "lr: 0.003, embed")
def m_config_seed(d): sub(f"{d}/configs/e3_20m_A_s2.yaml", "seed: 2", "seed: 1")
def m_preflight_compile_consistent(d):
    ej(f"{d}/out/e3_20m_B_s3/preflight.json", lambda x: x["runs"][0]["engine"].update({"train.compile": True}))
    resync_pc(d, "e3_20m_B_s3", "preflight.json", "preflight/e3_20m_B_s3.json")
def m_queue_order(d):
    p = f"{d}/logs/part2/queue_e3.txt"; t = open(p).read()
    t = t.replace("train e3_20m_A_s2.yaml", "train XX").replace("train e3_20m_B_s2.yaml", "train e3_20m_A_s2.yaml").replace("train XX", "train e3_20m_B_s2.yaml")
    open(p, "w").write(t)
def m_queue_two_trains(d):
    p = f"{d}/logs/part2/queue_e3.txt"; t = open(p).read(); i = t.index("2026-10-03 02:29:40 done e3_20m_A_s1 rc 0"); j = t.index("\n", i)
    open(p, "w").write(t[:j + 1] + "2026-10-03 02:29:41 train e3_20m_A_s1.yaml --require-committed\n" + t[j + 1:])
def m_queue_commit(d): sub(f"{d}/logs/part2/queue_e3.txt", "code commit 3c102f75dc68a21c", "code commit 3c102f75dc68a21d")
def m_status(d): sub(f"{d}/logs/part2/status.jsonl", '"run": "e3_20m_B_s3", "outcome": "done and scored"', '"run": "e3_20m_B_s3", "outcome": "diverged"')
def m_codelist_runio(d): sub(f"{d}/logs/part2/code_sha256_20261003_004304.txt", "ffcdf24596ab46cb", "438d4f5438aeee79")
def m_results(d): open(f"{d}/results.json", "a").write(" ")
def m_e3analyze(d): open(f"{d}/e3analyze.py", "a").write("# x\n")
def m_prereg(d): sub(f"{d}/configs/prereg.yaml", "id: E3", "id: E3 ")
def m_plan(d): open(f"{d}/plans/part2.txt", "a").write("train e3_20m_A_s4\n")
def m_pc_logstat(d): sub(f"{d}/audit/p2_pc_read.out", "LOGSTAT e3_20m_A_s3 1526 10 15259 grid_ok True nonfinite 0", "LOGSTAT e3_20m_A_s3 1526 10 15259 grid_ok True nonfinite 1")
def m_pc_runs_count(d): sub(f"{d}/audit/p2_pc_read.out", '[["e3_20m", "start"], 6]', '[["e3_20m", "start"], 7]')
def m_pc_scoreerr(d): sub(f"{d}/audit/p2_pc_read2.out", "SCOREERR e3_20m_B_s1 bytes 0", "SCOREERR e3_20m_B_s1 bytes 12")
def m_pc_trainout(d): sub(f"{d}/audit/p2_pc_read2.out", "loss 2.633459888503051", "loss 2.633459888503052")
def m_pc_extra_folder(d): sub(f"{d}/audit/p2_pc_read.out", "e3_20m_B_s3 e3_5m_A_s1 ", "e3_20m_B_s3 e3_20m_B_s3_R2 e3_5m_A_s1 ")

MUT = [m_bpb_value, m_windows_consistent, m_prose_row_consistent, m_max_windows_consistent, m_evalset_consistent, m_drop_step_consistent,
       m_tail_drawn, m_tail_crn_consistent, m_tail_missing_consistent, m_resumed, m_host_back, m_seed_collision, m_config_lr, m_config_seed,
       m_preflight_compile_consistent, m_queue_order, m_queue_two_trains, m_queue_commit, m_status, m_codelist_runio, m_results, m_e3analyze,
       m_prereg, m_plan, m_pc_logstat, m_pc_runs_count, m_pc_scoreerr, m_pc_trainout, m_pc_extra_folder]

base = fails(fresh()); print("baseline failing:", sorted(base))
killed = 0
for m in MUT:
    d = fresh(); m(d); f = fails(d); k = f != base; killed += k
    print(f"{'KILLED  ' if k else 'SURVIVED'} {m.__name__:32s} {sorted(f - base)}")
print(f"checks: {killed}/{len(MUT)} killed")

# recompute chain: a changed bpb value must show up as mismatches against results_part2.json
for label, eps in (("unmutated", 0.0), ("F(CHAT) of B_s2 +1e-5 bpb", 1e-5)):
    d = fresh()
    if eps:
        p = f"{d}/out/e3_20m_B_s2/bpb.jsonl"; r = jl(p)
        x = next(x for x in r if x["step"] == 15259 and x["set"] == "oasst2" and x["split"] == "all"); x["bits"] += eps * x["bytes"]
        y = next(x for x in r if x["step"] == 15259 and x["set"] == "CHAT"); y["bits"] = x["bits"]; y["bpb"] = x["bits"] / x["bytes"]; wjl(p, r)
    subprocess.run([PY, "-B", f"{d}/audit/p2_a1_recompute.py"], capture_output=True, check=True)
    out = subprocess.run([PY, "-B", f"{d}/audit/p2_a6_vs_results.py"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"recompute chain, {label}: {out}")
shutil.rmtree(WD)
