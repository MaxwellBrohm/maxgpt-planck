"""A synthetic E3 Part 3 record set for testing e3analyze_p3.py before any Part 3 value exists (notes.txt PART 3 REOPENED,
ANALYZER). Built from Part 1's copied records (out/e3_5m_*), which are real files of a known structure:
  streams   s1_P3 and every INIT run take e3_5m_A_s1's per-source counts (one stream); data k takes e3_5m_A_s{k}'s.
  values    bpb rows and tail losses come from VALUE_SRC (another Part 1 run per Part 3 run), so the families have spread.
  records   the start record gains data_seed; preflight carries the Part 3 config's sha256; the queue log is Part 2's copy
            plus the lines a clean Part 3 queue appends; the code list is commit 4962426's files hashed as queue_lib does.
Nothing here is a Part 3 value. CODE (default: the E3 folder) supplies e3analyze*.py, so a mutant analyzer can be tested."""
from __future__ import annotations
import hashlib, io, json, os, shutil, subprocess, tarfile

E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = "4962426b8f0d17738051bd499f9141a7cbc18c40"
ORDER = ["e3_5m_A_init2", "e3_5m_A_data2", "e3_5m_A_init3", "e3_5m_A_data3", "e3_5m_A_s1_P3", "e3_5m_A_init4",
         "e3_5m_A_data4", "e3_5m_A_init5", "e3_5m_A_data5"]
SEEDS = {"e3_5m_A_s1_P3": (1, 1), **{f"e3_5m_A_init{k}": (k, 1) for k in range(2, 6)}, **{f"e3_5m_A_data{k}": (1, k) for k in range(2, 6)}}
STREAM_SRC = {n: ("e3_5m_A_s1" if d == 1 else f"e3_5m_A_s{d}") for n, (s, d) in SEEDS.items()}
VALUE_SRC = {"e3_5m_A_s1_P3": "e3_5m_A_s1_R2", **{f"e3_5m_A_init{k}": f"e3_5m_B_s{k}" for k in range(2, 6)},
             **{f"e3_5m_A_data{k}": f"e3_5m_A_s{k + 3}" for k in range(2, 6)}}
P1_RUNS = sorted({*STREAM_SRC.values(), *VALUE_SRC.values(), "e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3"})
FILES = ("e3analyze.py", "e3analyze_p3.py", "e3analyze_p3_tables.py", "results.json", "tables.txt", "results_part2.json",
         "tables_part2.txt", "results_part4d.json")
T0 = "2026-10-11 13:12:28"


def jl(p): return [json.loads(x) for x in open(p)]
def wjl(p, rows): open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def code_list(commit: str = COMMIT) -> str:
    """queue_lib.sh code_hashes over `git archive commit`: sha256 of every .py/.yaml/.sh/.txt and tok_v0_8k.json."""
    env = {**os.environ, "DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}
    root = os.path.dirname(os.path.dirname(E3))     # git archive from a subfolder would take that subtree only
    tar = subprocess.run(["git", "-C", root, "archive", commit], capture_output=True, env=env, check=True).stdout
    rows = []
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            top = ("harness/", "data_prep/", "experiments/E2_lr_transfer/", "experiments/E3_seed_noise/", "tokenizer/", "corpus/")
            if m.isfile() and m.name.startswith(top) and "__pycache__" not in m.name and (
                    m.name.endswith((".py", ".yaml", ".sh", ".txt")) or os.path.basename(m.name) == "tok_v0_8k.json"):
                rows.append((m.name, hashlib.sha256(t.extractfile(m).read()).hexdigest()))
    return "".join(f"{h}  {p}\n" for p, h in sorted(rows))


def build(dst: str, code: str | None = None, codelist: str | None = None) -> str:
    code = code or E3
    os.makedirs(os.path.join(dst, "out"))
    for f in FILES:
        shutil.copy(os.path.join(code if f.endswith(".py") else E3, f), dst)
    for sub in ("configs", "plans", "logs"):
        shutil.copytree(os.path.join(E3, sub), os.path.join(dst, sub), ignore=shutil.ignore_patterns("part3"))
    for n in P1_RUNS:
        shutil.copytree(os.path.join(E3, "out", n), os.path.join(dst, "out", n), ignore=shutil.ignore_patterns("c6"))
    prereg = sha(os.path.join(E3, "configs", "prereg.yaml"))
    for name in ORDER:
        s_src, v_src = (os.path.join(E3, "out", x) for x in (STREAM_SRC[name], VALUE_SRC[name]))
        d = os.path.join(dst, "out", name); os.makedirs(d)
        wjl(os.path.join(d, "bpb.jsonl"), [{**r, "run": name} for r in jl(os.path.join(v_src, "bpb.jsonl"))])
        vt = {r["step"]: r["loss"] for r in jl(os.path.join(v_src, "train_tail.jsonl"))}
        wjl(os.path.join(d, "train_tail.jsonl"), [{**r, "loss": vt[r["step"]]} for r in jl(os.path.join(s_src, "train_tail.jsonl"))])
        cs = hashlib.sha256(name.encode()).hexdigest()
        st0 = json.load(open(os.path.join(E3, "out", "e3_5m_A_s1", "run_records.json")))[0]
        end = json.load(open(os.path.join(s_src, "run_records.json")))[-1]
        vend = json.load(open(os.path.join(v_src, "run_records.json")))[-1]
        start = {**st0, "run": name, "config_sha256": cs, "prereg_sha256": prereg, "data_seed": SEEDS[name][1]}
        json.dump([start, {**end, "run": name, "loss": vend["loss"]}], open(os.path.join(d, "run_records.json"), "w"), indent=1)
        pf = json.load(open(os.path.join(E3, "out", "e3_5m_A_s1", "preflight.json")))
        cfg = os.path.join(E3, "configs", name + ".yaml")
        pf["runs"][0].update({"name": name, "config": f"experiments/E3_seed_noise/configs/{name}.yaml",
                              "file_sha256": sha(cfg), "config_sha256": cs})
        pf["runs"][0]["prereg"]["sha256"] = prereg
        json.dump(pf, open(os.path.join(d, "preflight.json"), "w"), indent=1)
    lg = os.path.join(dst, "logs", "part3"); os.makedirs(lg)
    q = open(os.path.join(E3, "logs", "part2", "queue_e3.txt")).read()
    t = lambda x: f"{T0} {x}\n"   # noqa: E731
    q += t(f"queue start: exp E3, plan part3.txt (sha256 07018f0fec572994), code commit {COMMIT}") + t("seen E3: E3 PART 2 DONE")
    for name in ORDER:
        q += t("waiting for gpu.lock") + t("gpu.lock held; GPU memory in use 540 MiB") + t(f"train {name}.yaml --require-committed")
        q += t(f"done {name} rc 0") + "".join(t(f"score {name}/{c}") for c in ("final_00007630", "ckpt_00007344", "ckpt_00007497"))
    q += t("MARK E3 PART 3 DONE") + t("plan finished: part3.txt")
    open(os.path.join(lg, "queue_e3.txt"), "w").write(q)
    open(os.path.join(lg, "code_sha256_20261011_131228.txt"), "w").write(codelist if codelist is not None else code_list())
    return dst
