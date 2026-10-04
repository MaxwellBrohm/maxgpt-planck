"""E3 Part 2 audit (independent): provenance and run-identity checks, own code (does not import e3analyze*).
  ~/.venvs/planck/bin/python audit/p2_a2_checks.py [E3_DIR]     (default: this folder; a scratch copy for mutants)
Reads E3_DIR's out/e3_20m_*, logs/, configs/, plans/, and audit/p2_pc_read*.out (printed on the PC, read-only, by
audit/p2_pc_read*.sh). The commit's own file contents come from git (3c102f7), never from E3_DIR. Exit 1 on any failure."""
import hashlib, json, math, os, re, subprocess, sys

E3 = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GIT_ENV = dict(os.environ, DEVELOPER_DIR="/Library/Developer/CommandLineTools")
COMMIT = "3c102f75dc68a21c1e7a4f36a187850f4b07a1cd"
BASE = "620c77c"   # tip of main when the audit began (2026-10-04); Part 1 files and screens_lib.py are read here
RUNS = [f"e3_20m_{a}_s{s}" for s in (1, 2, 3) for a in "AB"]
CK = [14945, 15250, 15259]
TAIL = list(range(14960, 15251, 10)) + [15259]
NOTES_CFG = {  # notes.txt PART 2 QUEUED (committed in 3c102f7)
    "e3_20m_A_s1": "83e28d5f80ddf646fdbf593e1949aa32664f3adb688667fdd063a211d78c580f",
    "e3_20m_A_s2": "aa1be48b280708985625ee2efcdfaa74037eb31c6d555a81da26dd89a5507475",
    "e3_20m_A_s3": "a6e329e86163d580940df1bad7379e42eaf2b9d02e46e7fd9c9a3bd9d7e0c2d7",
    "e3_20m_B_s1": "a2843208b338f67cf352264330c326092198dde65414da28c9ac6572e6a4ac4b",
    "e3_20m_B_s2": "a952ce4b1fcc9b0d3b1b1878d0346f6b2c7174b651879243ae236ed4586cdb62",
    "e3_20m_B_s3": "55908528a0f86c38a9f0a7563401be0d4a84c31bf101e50951b94e5698764b7f"}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def h(b): return hashlib.sha256(b).hexdigest()
def hf(p): return h(open(p, "rb").read())
def git(*a): return subprocess.run(["git", "-C", REPO, *a], capture_output=True, env=GIT_ENV, check=True).stdout
def jl(p): return [json.loads(x) for x in open(p)]
def P(*a): return os.path.join(E3, *a)


# ---- PC printouts ----
pc = open(P("audit", "p2_pc_read.out")).read().splitlines()
pc2 = open(P("audit", "p2_pc_read2.out")).read().splitlines()
pcsha = {ln.split()[1]: ln.split()[2] for ln in pc if ln.startswith("SHA ")}
pctail = {}
for ln in pc:
    if ln.startswith("TAIL "):
        _, run, js = ln.split(" ", 2); pctail.setdefault(run, []).append(json.loads(js))
pcruns = [json.loads(ln[5:]) for ln in pc if ln.startswith("RUNS ")]
check("PC: E3 holds exactly the 18 Part 1 and 6 Part 2 run folders",
      sorted(x for x in next(ln for ln in pc if ln.startswith("LS_E3")).split()[1:] if x.startswith("e3_"))
      == sorted([f"e3_5m_{a}_s{s}" for s in range(1, 9) for a in "AB"] + ["e3_5m_A_s1_R2", "e3_5m_A_s1_R3"] + RUNS))
check("PC: marks are exactly E3_PART_1_DONE and E3_PART_2_DONE", next(ln for ln in pc if ln.startswith("MARKS")).split()[1:] == ["E3_PART_1_DONE", "E3_PART_2_DONE"])
check("PC: runs.jsonl has 6 e3_20m starts and 6 ends, 18 and 18 for 5M, nothing else for E3",
      next(ln for ln in pc if ln.startswith("RUNS_E3_COUNTS")) == 'RUNS_E3_COUNTS [[["e3_20m", "end"], 6], [["e3_20m", "start"], 6], [["e3_5m_", "end"], 18], [["e3_5m_", "start"], 18]]')
for f, mine in (("queue_e3.txt", "queue_e3.txt"), ("status.jsonl", "status.jsonl"), ("code_sha256_20261003_004304.txt", "code_sha256_20261003_004304.txt")):
    check(f"copy == PC: logs/part2/{mine}", hf(P("logs", "part2", mine)) == pcsha[f])

# ---- per run ----
p1pf = json.load(open(P("out", "e3_5m_A_s1", "preflight.json")))["runs"][0]
p1bpb = {(x["set"], x["split"]): x for x in jl(P("out", "e3_5m_A_s1", "bpb.jsonl")) if x["step"] == 7630}
prereg_sha = h(git("show", f"{COMMIT}:experiments/E3_seed_noise/configs/prereg.yaml"))
check("prereg.yaml on disk == 3c102f7 == be89babb", hf(P("configs", "prereg.yaml")) == prereg_sha and prereg_sha.startswith("be89babb"))
ends, tails = {}, {}
for n in RUNS:
    arm, seed = n.split("_")[2], int(n[-1])
    bad = []
    bpb, tail, rec = jl(P("out", n, "bpb.jsonl")), jl(P("out", n, "train_tail.jsonl")), json.load(open(P("out", n, "run_records.json")))
    pf = json.load(open(P("out", n, "preflight.json")))
    # copies == PC
    if hf(P("out", n, "bpb.jsonl")) != pcsha[f"{n}/bpb.jsonl"]: bad.append("bpb.jsonl != PC")
    if hf(P("out", n, "preflight.json")) != pcsha[f"preflight/{n}.json"]: bad.append("preflight != PC")
    if tail != pctail.get(n): bad.append("train_tail != PC log.jsonl records with step > 14953")
    st_, en = [r for r in pcruns if r["run"] == n and r["event"] == "start"], [r for r in pcruns if r["run"] == n and r["event"] == "end"]
    if len(st_) != 1 or len(en) != 1: bad.append("PC runs.jsonl: not one start and one end")
    else:
        s0, e0 = rec[0], rec[1]
        for k in ("config_sha256", "n_params", "prereg_sha256", "resumed_from", "schedule", "step"):
            if s0.get(k) != st_[0][k]: bad.append(f"start {k} != PC")
        for k in ("step", "total_steps"):
            if e0.get(k) != en[0][k]: bad.append(f"end {k} != PC")
    lg = next(ln for ln in pc if ln.startswith(f"LOGSTAT {n} "))
    if lg != f"LOGSTAT {n} 1526 10 15259 grid_ok True nonfinite 0 nonloss_records 0": bad.append("PC log.jsonl: " + lg)
    if next(ln for ln in pc if ln.startswith(f"LATEST {n} ")) != f'LATEST {n} {{"path": "final_00015259.pt", "step": 15259}}': bad.append("latest.json")
    if not re.fullmatch(rf"CKPT {n} ckpt_00014945\.pt:\d+ ckpt_00015250\.pt:\d+ final_00015259\.pt:\d+ ", next(ln for ln in pc if ln.startswith(f"CKPT {n} "))): bad.append("checkpoints")
    to = next(ln for ln in pc2 if ln.startswith(f"TRAINOUT_LAST {n} "))
    if float(to.rsplit("loss ", 1)[1]) != rec[1]["loss"] or "finished at step 15259" not in to: bad.append("train.out final loss != end record")
    if next(ln for ln in pc2 if ln.startswith(f"TRAINOUT_GREP {n} ")) != f"TRAINOUT_GREP {n} resume:0 nan:0 error:0": bad.append("train.out resume/nan/error")
    if next(ln for ln in pc2 if ln.startswith(f"SCOREERR {n} ")) != f"SCOREERR {n} bytes 0 lines 0": bad.append("score.err not empty")
    # config identity
    cfg_b = open(P("configs", n + ".yaml"), "rb").read()
    git_b = git("show", f"{COMMIT}:experiments/E3_seed_noise/configs/{n}.yaml")
    if h(cfg_b) != NOTES_CFG[n] or cfg_b != git_b or pf["runs"][0]["file_sha256"] != NOTES_CFG[n]: bad.append("config bytes")
    txt = cfg_b.decode(); lr = {"A": "0.003", "B": "0.0015"}[arm]
    want = (f"extends: ../../E2_lr_transfer/configs/base20m.yaml\nname: {n}\nout_dir: ../../../planck_root/runs/E3/{n}\nseed: {seed}\n"
            f"optim: {{lr: {lr}, embed_lr: {lr}, scalar_lr: {lr}}}\nschedule: {{mode: full, decay_frac: 0.2}}\ntrain: {{total_steps: 15259, ckpt_every: 305}}\n")
    if txt.split("\n", 1)[1] != want: bad.append("config body (LR, seed, schedule, cadence)")
    # preflight
    r = pf["runs"][0]
    if not (pf["ok"] and pf["strict"] and r["ok"] and r["refusals"] == [] and r["name"] == n and r["prereg"] == {"committed": True, "id": "E3", "sha256": prereg_sha}):
        bad.append("preflight ok/strict/prereg")
    for k in ("engine", "engine_fixed", "shares", "n_files", "engine_not_read_by_harness"):
        if r[k] != p1pf[k]: bad.append(f"preflight {k} != Part 1")
    if r["n_params"] != 20001511 or r["schedule"] != {"mode": "full", "total_steps": 15259, "warmup_steps": 76}: bad.append("preflight n_params/schedule")
    if r["config_sha256"] != rec[0]["config_sha256"]: bad.append("preflight config_sha256 != start record")
    # run records
    s0, e0 = rec[0], rec[1]
    if len(rec) != 2 or s0["event"] != "start" or e0["event"] != "end" or s0["run"] != n or e0["run"] != n: bad.append("records shape")
    if s0["schedule"] != {"decay_start": 12207, "mode": "full", "total_steps": 15259, "warmup_steps": 76}: bad.append("schedule")
    if (s0["batch_tokens"], s0["precision"], s0["doc_attn"], s0["optim_batched"], s0["n_params"], s0["resumed_from"], s0["prereg_committed"], s0["prereg_id"], s0["prereg_sha256"]) \
            != (32768, "bf16", "varlen", True, 20001511, None, True, "E3", prereg_sha): bad.append("start settings")
    if s0["env"].get("gpu") != "NVIDIA GeForce RTX 5070" or s0["env"].get("device") != "cuda": bad.append("env")
    if e0["step"] != 15259 or e0["total_steps"] != 15259 or not math.isfinite(e0["loss"]): bad.append("end")
    # bpb records
    per = {}
    for x in bpb:
        per.setdefault((x["set"], x["split"]), []).append(x)
        if (x["run"], x["tokenizer_sha256"], x["evalset_sha256"], x["precision"], x["max_windows"]) != (n, p1bpb[("CHAT", "all")]["tokenizer_sha256"], p1bpb[("CHAT", "all")]["evalset_sha256"], "fp32", None):
            bad.append(f"bpb scorer fields {x['set']}/{x['split']}@{x['step']}"); break
        want_ck = "final_00015259.pt" if x["step"] == 15259 else f"ckpt_{x['step']:08d}.pt"
        if x["ckpt"] != want_ck: bad.append("bpb ckpt name"); break
    if sorted(per) != sorted(p1bpb) or any(sorted(y["step"] for y in v) != CK for v in per.values()): bad.append("bpb (set, split) x steps")
    for key, v in per.items():
        for x in v:
            p1 = p1bpb[key]
            if (x["windows"], x["bytes"], x.get("truncated")) != (p1["windows"], p1["bytes"], p1.get("truncated")): bad.append(f"windows/bytes/truncated {key}"); break
            if abs(x["bpb"] - x["bits"] / x["bytes"]) > 1e-12: bad.append(f"bpb != bits/bytes {key}"); break
    for stp in CK:
        g = {(x["set"], x["split"]): x for x in bpb if x["step"] == stp}
        if any((s, "all") not in g for s in ("cccc", "gutenberg", "wikimedia", "oasst2", "CHAT", "PROSE")):
            bad.append(f"rows missing @{stp}"); continue
        pro =sum(g[(s, "all")]["bits"] for s in ("cccc", "gutenberg", "wikimedia")) / sum(g[(s, "all")]["bytes"] for s in ("cccc", "gutenberg", "wikimedia"))
        if abs(g[("PROSE", "all")]["bpb"] - pro) > 1e-12 or abs(g[("CHAT", "all")]["bpb"] - g[("oasst2", "all")]["bpb"]) > 0: bad.append(f"stored CHAT/PROSE row @{stp}")
    if [t["step"] for t in tail] != TAIL or not all(math.isfinite(t["loss"]) for t in tail): bad.append("tail steps/finite")
    stream = lambda r: {k: v for k, v in r.items() if k.startswith(("drawn_", "dropped_long_")) or k in ("tokens", "sup_tokens")}  # noqa: E731
    if stream(tail[-1]) != stream(e0) or len(stream(e0)) != 11: bad.append("tail end stream != end record (or key count != 11)")
    ends[n], tails[n] = e0, tail
    check(f"{n}: copies == PC, config, preflight, records, scorer, one clean run", not bad, "; ".join(bad))

stream = lambda r: {k: v for k, v in r.items() if k.startswith(("drawn_", "dropped_long_")) or k in ("tokens", "sup_tokens")}  # noqa: E731
for s in (1, 2, 3):
    a, b = f"e3_20m_A_s{s}", f"e3_20m_B_s{s}"
    check(f"CRN seed {s}: A and B equal on all 11 stream counters at all 31 tail steps and at the end",
          [stream(x) for x in tails[a]] == [stream(x) for x in tails[b]] and stream(ends[a]) == stream(ends[b]))
    check(f"CRN seed {s}: A and B are different runs (losses differ at every tail step)", all(x["loss"] != y["loss"] for x, y in zip(tails[a], tails[b])))
check("3 seeds drew 3 different streams (end counters)", len({json.dumps(stream(ends[f'e3_20m_A_s{s}']), sort_keys=True) for s in (1, 2, 3)}) == 3)

# ---- queue log, status, plan ----
q = open(P("logs", "part2", "queue_e3.txt")).read()
q1 = open(P("logs", "queue_e3.txt")).read()
check("Part 2 queue log begins with Part 1's committed queue log byte for byte", q.startswith(q1) and q1 == git("show", f"{BASE}:experiments/E3_seed_noise/logs/queue_e3.txt").decode())
q2 = q[len(q1):].splitlines()
body = [ln[20:] for ln in q2]
plan_git = git("show", f"{COMMIT}:experiments/E3_seed_noise/plans/part2.txt")
expect = [f"queue start: exp E3, plan part2.txt (sha256 {h(plan_git)[:16]}), code commit {COMMIT}", "seen E2: E2 20M DONE"]
for n in RUNS:
    expect += ["waiting for gpu.lock", "GPU-HELD", f"train {n}.yaml --require-committed", f"done {n} rc 0",
               f"score {n}/final_00015259", f"score {n}/ckpt_00014945", f"score {n}/ckpt_00015250"]
expect += ["MARK E3 PART 2 DONE", "plan finished: part2.txt"]
got = [("GPU-HELD" if re.fullmatch(r"gpu\.lock held; GPU memory in use \d+ MiB", ln) else ln) for ln in body]
check("Part 2 queue section is exactly: one start at 3c102f7 with plan part2.txt, A1 B1 A2 B2 A3 B3 each trained once rc 0 and scored at 3 checkpoints, MARK, finish",
      got == expect, f"{len(got)} lines vs {len(expect)}")
check("plans/part2.txt and plans/CURRENT on disk == 3c102f7", open(P("plans", "part2.txt"), "rb").read() == plan_git
      and open(P("plans", "CURRENT"), "rb").read() == git("show", f"{COMMIT}:experiments/E3_seed_noise/plans/CURRENT"))
stt = open(P("logs", "part2", "status.jsonl")).read()
s1 = git("show", f"{BASE}:experiments/E3_seed_noise/logs/status_e3.jsonl").decode()
check("status: Part 1 prefix == committed status_e3.jsonl; then 6 Part 2 lines 'done and scored' in order",
      stt.startswith(s1) and [json.loads(x)["run"] for x in stt[len(s1):].splitlines()] == RUNS
      and all(json.loads(x)["outcome"] == "done and scored" for x in stt[len(s1):].splitlines()))

# ---- code list vs the commit ----
cl = [ln.split("  ", 1) for ln in open(P("logs", "part2", "code_sha256_20261003_004304.txt")).read().splitlines() if ln.strip()]
codemap = {p: x for x, p in cl}
bad = [p for x, p in cl if h(git("show", f"{COMMIT}:{p}")) != x]
check("code list: every listed file's sha256 == its bytes at 3c102f7", not bad, f"{len(cl)} files, {len(bad)} differ")
py = sorted((p for p in git("ls-tree", "-r", "--name-only", COMMIT, "harness", "data_prep").decode().split() if p.endswith(".py")), key=lambda s: s.encode())
dig = h("".join(f"{h(git('show', f'{COMMIT}:{p}'))}  {p}\n" for p in py).encode())
check("FIXED 2 recipe at 3c102f7 (git ls-files harness data_prep | grep .py | sort | sha256sum): 56b4a68f over 100 files, not 92deaef0",
      dig.startswith("56b4a68f38d43cc8") and len(py) == 100, f"{dig[:16]} over {len(py)}")
check("D4 pin: harness/runio.py == ffcdf245 in the code list", codemap.get("harness/runio.py", "").startswith("ffcdf24596ab46cb"))
p1cl = {p: x for x, p in (ln.split("  ", 1) for ln in open(P("logs", "code_sha256_20260928_162455.txt")).read().splitlines() if ln.strip())}
diff = sorted(p for p in set(codemap) | set(p1cl) if p.startswith(("harness/", "data_prep/")) and p.endswith(".py") and codemap.get(p) != p1cl.get(p))
check("harness/ + data_prep/ .py vs Part 1's code list: exactly the 8 files D4 / E2 DEVIATION 1 name",
      diff == sorted(["harness/runio.py", "harness/from_pipeline.py", "data_prep/eval_sets.py", "data_prep/prep_common.py", "data_prep/pretokenize.py",
                      "harness/durable.py", "harness/test_durable.py", "data_prep/tests/test_durable_writes.py"]), " ".join(diff))
check("train.py, trainer.py, model.py as Part 1 (805bf5f8, 15d1a971, f4e61556)",
      all(codemap[p] == p1cl[p] for p in ("harness/train.py", "harness/trainer.py", "harness/model.py"))
      and (codemap["harness/train.py"][:8], codemap["harness/trainer.py"][:8], codemap["harness/model.py"][:8]) == ("805bf5f8", "15d1a971", "f4e61556"))
check("data_prep/bpb.py and evalwin.py as Part 1", all(codemap[p] == p1cl[p] for p in ("data_prep/bpb.py", "data_prep/evalwin.py")))
check("base20m.yaml, base5m.yaml, engine.yaml at 3c102f7 == Part 1 commit e39112a",
      all(git("show", f"{COMMIT}:experiments/E2_lr_transfer/configs/{f}") == git("show", f"e39112a:experiments/E2_lr_transfer/configs/{f}") for f in ("base20m.yaml", "base5m.yaml", "engine.yaml")))
check("e3analyze.py on disk == code list (3c102f7) == 620c77c", hf(P("e3analyze.py")) == codemap["experiments/E3_seed_noise/e3analyze.py"]
      == h(git("show", f"{BASE}:experiments/E3_seed_noise/e3analyze.py")))

# ---- arms from E2 ----
q2r = json.load(open(os.path.join(REPO, "experiments", "E2_lr_transfer", "results.json")))["q2"]
fb = q2r["final_bpb_20m"]
check("arms: A20 = E2 q2 argmin (3e-3, 1); B20 = the neighbour with the lower 500M F(CHAT) (1.5e-3: 0.9749 < 6e-3: 0.9814)",
      q2r["A20"] == [0.003, 1.0] and q2r["B20"] == [0.0015, 1.0] and q2r["e2pick"]["grid_20m"]["argmin_g"] == 1.0
      and fb["0.0015"]["b500M"]["chat"] < fb["0.006"]["b500M"]["chat"] and fb["0.003"]["b500M"]["chat"] < fb["0.0015"]["b500M"]["chat"])

# ---- Part 1 files untouched (D5) ----
check("results.json, tables.txt, e3analyze.py == 620c77c, and AUDIT.md begins with 620c77c's (Part 1 audit untouched) (D5)",
      all(open(P(f), "rb").read() == git("show", f"{BASE}:experiments/E3_seed_noise/{f}") for f in ("results.json", "tables.txt", "e3analyze.py"))
      and open(P("AUDIT.md"), "rb").read().startswith(git("show", f"{BASE}:experiments/E3_seed_noise/AUDIT.md")))
sl = git("show", f"{BASE}:experiments/screens/screens_lib.py").decode()
check("D5 reason: screens_lib.py (620c77c) pins E3_SHA 1f017ecb and asserts it", 'E3_SHA = "1f017ecb1e6d14dd4f676c90ca703c20e9395b217fc841f83fffcb93e57994a2"' in sl and "sha256(e3p) == E3_SHA" in sl)

# ---- privacy of the records to be committed ----
leak = [f"{n}: env.host" for n in RUNS if "host" in json.load(open(P("out", n, "run_records.json")))[0].get("env", {})]
check("OUTPUTS: no machine name in run_records.json (Part 1's copies carry no env.host)", not leak, ", ".join(leak))

npass = sum(ok for _, ok, _ in results)
for name, ok, det in results:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{det}]" if det and not ok else (f"  [{det}]" if det else "")))
print(f"{npass}/{len(results)} pass")
sys.exit(0 if npass == len(results) else 1)
