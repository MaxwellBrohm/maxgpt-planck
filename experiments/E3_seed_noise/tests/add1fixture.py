"""A synthetic record set for E3 Part 1b and Part 3 at 20M (notes.txt ADDENDUM 2026-10-12), for testing e3analyze_p1b.py
and e3analyze_p3_20m.py before any of their values exists. Built from Parts 1 and 2's copied records (real files of a
known structure); nothing here is a Part 1b or Part 3 at 20M value, and no git is needed (the code list is Part 3's,
with this folder's E3 rows hashed from the files copied into the fixture).
  Part 1b   e3_5m_A_s12..s21: bpb rows and tail losses from VAL1B (Part 1's B seeds and A1's replays); streams from
            Part 1's arm A seeds with drawn_cccc moved by 1000 x (i + 1) at every tail step and the end (distinct).
  20M       INIT runs and s1_P3 take Part 2's e3_20m_A_s1 stream; data2, data3 Part 2's A_s2, A_s3; data4, data5 A_s1's
            moved as above. Values from VAL20 (Part 2's six runs, some shifted by OFF20 bpb so no two runs are equal).
build(root, code=None) writes <root>/experiments/E3_seed_noise (the analyzers from CODE, default this folder) and the
screens' BASE files at their paths under <root>; returns the E3 folder."""
from __future__ import annotations
import hashlib, json, os, shutil

E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(E3))
PY = ("e3analyze.py", "e3analyze_p2.py", "e3analyze_p3.py", "e3analyze_p3_tables.py", "e3add1_lib.py", "e3analyze_p1b.py",
      "e3analyze_p3_20m.py")
KEEP = ("results.json", "tables.txt", "results_part2.json", "tables_part2.txt", "results_part4d.json", "results_part3.json",
        "tables_part3.txt")
P1B = [f"e3_5m_A_s{s}" for s in range(12, 22)]
VAL1B = dict(zip(P1B, [f"e3_5m_B_s{k}" for k in range(1, 9)] + ["e3_5m_A_s1_R2", "e3_5m_A_s1_R3"]))
P20 = ["e3_20m_A_init2", "e3_20m_A_data2", "e3_20m_A_init3", "e3_20m_A_data3", "e3_20m_A_s1_P3", "e3_20m_A_init4",
       "e3_20m_A_data4", "e3_20m_A_init5", "e3_20m_A_data5"]
SEEDS20 = {"e3_20m_A_s1_P3": (1, 1), **{f"e3_20m_A_init{k}": (k, 1) for k in range(2, 6)},
           **{f"e3_20m_A_data{k}": (1, k) for k in range(2, 6)}}
VAL20 = {"e3_20m_A_s1_P3": "e3_20m_B_s1", "e3_20m_A_init2": "e3_20m_A_s2", "e3_20m_A_init3": "e3_20m_A_s3",
         "e3_20m_A_init4": "e3_20m_B_s2", "e3_20m_A_init5": "e3_20m_B_s3", "e3_20m_A_data2": "e3_20m_A_s1",
         "e3_20m_A_data3": "e3_20m_B_s2", "e3_20m_A_data4": "e3_20m_B_s3", "e3_20m_A_data5": "e3_20m_A_s2"}
OFF20 = {"e3_20m_A_init4": 0.0011, "e3_20m_A_init5": -0.0007, "e3_20m_A_data2": 0.0023, "e3_20m_A_data5": -0.0019}
STREAM20 = {**{n: ("e3_20m_A_s1", 0) for n in SEEDS20 if SEEDS20[n][1] == 1}, "e3_20m_A_data2": ("e3_20m_A_s2", 0),
            "e3_20m_A_data3": ("e3_20m_A_s3", 0), "e3_20m_A_data4": ("e3_20m_A_s1", 4000), "e3_20m_A_data5": ("e3_20m_A_s1", 5000)}
BASE = {"base_s101": "experiments/S001_attn_gate/out/base_s101/bpb.jsonl", "base_s102": "experiments/S001_attn_gate/out/base_s102/bpb.jsonl",
        **{f"base_s{s}": f"experiments/screens/base_out/base_s{s}/bpb.jsonl" for s in (103, 104, 105)}}
COMMIT, T0 = "0123456789abcdef0123456789abcdef01234567", "2026-10-13 00:00:00"
SKIP = {"c6"}


def jl(p): return [json.loads(x) for x in open(p)]
def wjl(p, rows): open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def src(run): return os.path.join(E3, "out", run)


def moved(rec, off):
    return {**rec, "drawn_cccc": rec["drawn_cccc"] + off} if off else dict(rec)


def synth(d, name, vsrc, ssrc, off, seed, dseed, start0, pf0, shift=0.0):
    o = os.path.join(d, "out", name); os.makedirs(o)
    wjl(os.path.join(o, "bpb.jsonl"), [{**r, "run": name, "bpb": r["bpb"] + shift} for r in jl(os.path.join(src(vsrc), "bpb.jsonl"))])
    vt = {r["step"]: r["loss"] for r in jl(os.path.join(src(vsrc), "train_tail.jsonl"))}
    wjl(os.path.join(o, "train_tail.jsonl"), [{**moved(r, off), "loss": vt[r["step"]]} for r in jl(os.path.join(src(ssrc), "train_tail.jsonl"))])
    end = moved(json.load(open(os.path.join(src(ssrc), "run_records.json")))[-1], off)
    vend = json.load(open(os.path.join(src(vsrc), "run_records.json")))[-1]
    cs, prereg = hashlib.sha256(name.encode()).hexdigest(), sha(os.path.join(E3, "configs", "prereg.yaml"))
    start = {**start0, "run": name, "config_sha256": cs, "prereg_sha256": prereg}
    start.pop("data_seed", None)
    if dseed is not None:
        start["data_seed"] = dseed
    json.dump([start, {**end, "run": name, "loss": vend["loss"]}], open(os.path.join(o, "run_records.json"), "w"), indent=1)
    pf = json.loads(json.dumps(pf0))
    pf["runs"][0].update({"name": name, "config": f"experiments/E3_seed_noise/configs/{name}.yaml",
                          "file_sha256": sha(os.path.join(E3, "configs", name + ".yaml")), "config_sha256": cs})
    pf["runs"][0]["prereg"]["sha256"] = prereg
    json.dump(pf, open(os.path.join(o, "preflight.json"), "w"), indent=1)


def qlines(names, ckpts):
    t = lambda x: f"{T0} {x}\n"   # noqa: E731
    out = ""
    for n in names:
        out += t("waiting for gpu.lock") + t("gpu.lock held; GPU memory in use 640 MiB") + t(f"train {n}.yaml --require-committed")
        out += t(f"done {n} rc 0") + "".join(t(f"score {n}/{c}") for c in ckpts)
    return out


def code_list(d):
    rows = {p: h for p, h in ((ln.split("  ")[1], ln.split("  ")[0]) for ln in
            open(os.path.join(E3, "logs", "part3", "code_sha256_20261011_131228.txt")).read().split("\n") if ln.strip())
            if not p.startswith("experiments/E3_seed_noise/")}
    for sub in ("", "configs", "plans"):
        for f in sorted(os.listdir(os.path.join(d, sub))):
            p = os.path.join(d, sub, f)
            if os.path.isfile(p) and f.endswith((".py", ".yaml", ".txt", ".sh")) and f != "CURRENT":
                rows["experiments/E3_seed_noise/" + (sub + "/" if sub else "") + f] = sha(p)
    return "".join(f"{h}  {p}\n" for p, h in sorted(rows.items()))


def build(root: str, code: str | None = None) -> str:
    code = code or E3
    d = os.path.join(root, "experiments", "E3_seed_noise")
    os.makedirs(os.path.join(d, "out"))
    for f in PY:
        shutil.copy(os.path.join(code, f), d)
    for f in KEEP:
        shutil.copy(os.path.join(E3, f), d)
    for sub in ("configs", "plans", "logs"):
        shutil.copytree(os.path.join(E3, sub), os.path.join(d, sub), ignore=shutil.ignore_patterns("part1b", "part3_20m"))
    for n in [f"e3_5m_A_s{s}" for s in range(1, 9)] + ["e3_20m_A_s1"]:
        shutil.copytree(src(n), os.path.join(d, "out", n), ignore=shutil.ignore_patterns(*SKIP))
    for b, p in BASE.items():
        os.makedirs(os.path.dirname(os.path.join(root, p)), exist_ok=True)
        shutil.copy(os.path.join(REPO, p), os.path.join(root, p))
    st5, pf5 = json.load(open(os.path.join(src("e3_5m_A_s1"), "run_records.json")))[0], json.load(open(os.path.join(src("e3_5m_A_s1"), "preflight.json")))
    for i, n in enumerate(P1B):
        synth(d, n, VAL1B[n], f"e3_5m_A_s{i % 8 + 1}", 1000 * (i + 1), int(n.rsplit("_s", 1)[1]), None, st5, pf5)
    st20, pf20 = json.load(open(os.path.join(src("e3_20m_A_s1"), "run_records.json")))[0], json.load(open(os.path.join(src("e3_20m_A_s1"), "preflight.json")))
    for n in P20:
        synth(d, n, VAL20[n], *STREAM20[n], *SEEDS20[n], st20, pf20, OFF20.get(n, 0.0))
    t = lambda x: f"{T0} {x}\n"   # noqa: E731
    q = open(os.path.join(E3, "logs", "part3", "queue_e3.txt")).read()
    q += t(f"queue start: exp E3, plan part1b_p3at20m.txt (sha256 {sha(os.path.join(d, 'plans', 'part1b_p3at20m.txt'))[:16]}), "
           f"code commit {COMMIT}") + t("seen E3: E3 PART 3 DONE") + t("seen SCREENS: SCREENS ROBUSTNESS K5 DONE")
    q += qlines(P1B, ("final_00007630", "ckpt_00007344", "ckpt_00007497")) + t("MARK E3 PART 1B DONE")
    cl = code_list(d)
    for sub, extra in (("part1b", ""), ("part3_20m", qlines(P20, ("final_00015259", "ckpt_00014945", "ckpt_00015250"))
                                         + t("MARK E3 PART 3 AT 20M DONE") + t("plan finished: part1b_p3at20m.txt"))):
        os.makedirs(os.path.join(d, "logs", sub))
        q += extra
        open(os.path.join(d, "logs", sub, "queue_e3.txt"), "w").write(q)
        open(os.path.join(d, "logs", sub, "code_sha256_20261013_000000.txt"), "w").write(cl)
    return d
