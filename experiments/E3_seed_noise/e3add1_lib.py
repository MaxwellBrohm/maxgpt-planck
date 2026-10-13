"""Shared checks for E3 Part 1b and Part 3 at 20M (notes.txt ADDENDUM 2026-10-12): the plan's queue log, its code lists
and e3analyze_p3.py's per-run checks at either size. Stdlib only. Written and committed before any value of either part
exists; imports e3analyze.py and e3analyze_p3.py unchanged.
  queue_new(sub)    (starts with Part 3's copy?, the lines logs/<sub>/queue_e3.txt adds to logs/part3/queue_e3.txt)
  code_checks(...)  one code list per start of this plan (logs/<sub>/code_sha256_*.txt, written just before it), all
                    equal, and equal to Part 3's (commit 4962426) on every path outside experiments/E3_seed_noise/, so
                    the FIXED 2 checks fail on Part 3's five and nothing else; this part's configs, the plan and the
                    analyzers in them as on disk; every start names the plan and one commit
  run_checks(...)   e3analyze_p3.py's per-run list at 5M or 20M, with or without data_seed"""
from __future__ import annotations
import hashlib, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3analyze as E  # noqa: E402  (registered, unchanged)
import e3analyze_p3 as P3  # noqa: E402  (registered, unchanged)
import functools  # noqa: E402
for _m, _f in ((E, "t_ppf"), (E, "chi2_ppf"), (E, "factor"), (E, "ub"), (P3, "f_ppf")):
    setattr(_m, _f, functools.lru_cache(maxsize=None)(getattr(_m, _f)))   # pure functions of their arguments: same values,
    # computed once per argument in this process (the files are unchanged; a process without this module is untouched)

PLAN, PLAN_SHA = "part1b_p3at20m.txt", "17c939a5e86f3d63f08ef88bbde6c85e780bab9635d6252f224f1026774a38e0"
E3X = "experiments/E3_seed_noise/"
P1B = [f"e3_5m_A_s{s}" for s in range(12, 22)]
P20 = ["e3_20m_A_init2", "e3_20m_A_data2", "e3_20m_A_init3", "e3_20m_A_data3", "e3_20m_A_s1_P3", "e3_20m_A_init4",
       "e3_20m_A_data4", "e3_20m_A_init5", "e3_20m_A_data5"]
CONFIG_SHA = {  # notes.txt ADDENDUM 2026-10-12, RUNS
    "e3_5m_A_s12": "0e26246756996c607ae06179ec5b2d6765a7e403ee8a8cc961f979b7c47aabda",
    "e3_5m_A_s13": "1b9682c4534c9c28493b2724583bdf67159fd86ef476eaa9ee23e9aa3f37f0f9",
    "e3_5m_A_s14": "f990526f0f4ace45bb5fbb344ee654ec9ffbe6866f345e85c3fecfb61f16f520",
    "e3_5m_A_s15": "30f725d70f3177c1beb2052033b692f32f18ce19b15a8684a87cd14406db49ab",
    "e3_5m_A_s16": "3852bd9903557b9d648fda3d7f6e6436cf60d6334fec3f7c0f70916767b5d178",
    "e3_5m_A_s17": "3f11ddb21f75fae1d0b6b7f2f67020ea2aa29e9aa6ecfdf84c2d6e1e0967e783",
    "e3_5m_A_s18": "cd57645e99d6ccc486f46117eafd77882be411f6f890a628037f7755c8b89cae",
    "e3_5m_A_s19": "3a7870e9cef76afa2a0b0ae598b140e10482fae49b0518232b8ce31765854ee3",
    "e3_5m_A_s20": "c6018058d2f8d300256830c0a283f502afdc6dc85f8641896cf8add8ee5d6a58",
    "e3_5m_A_s21": "31c6dddf5268af7f0d3a64fc5b513fe43a7cb8aeb77d95e2e34e5b2b96d6a218",
    "e3_20m_A_init2": "b5e32c5a5b4b7b6038a988a072a97e1cb20d7c26eacfa219983c2c519f23a62c",
    "e3_20m_A_data2": "6248a748e26a8407f103170a56cc88e341edb32bf004a0eacc034c331f17f9e7",
    "e3_20m_A_init3": "48542f67e263b6bf1443fcaa5454e6471a9c63da7f21cc6cfdbbb0a3e299257d",
    "e3_20m_A_data3": "61874f53f7694309afbf3b66ae1aca29b7ef0deeb809f2aa332a15260ceab9f0",
    "e3_20m_A_s1_P3": "7aef1ba48a07df190e02b966fbd1911063fd184ee36aeef7f6288d7c2d0ad76e",
    "e3_20m_A_init4": "b22014058ec65034715b84e38b165511c4ac83d57abe48e0accb8a0e2b7990d2",
    "e3_20m_A_data4": "c3fdfce857641a7f764110460d08a4dd44d9d28e69f924ca359e9c42fd2931f0",
    "e3_20m_A_init5": "a794bdaaa2077dfb7502e30a2133775096c927e12535849927cad76659422ed3",
    "e3_20m_A_data5": "4e65e3fc4ef4616bdb7ee70db66e4f21853e4dc175f39fbe4c5d50d2fa986733"}
ANALYZERS = ("e3analyze.py", "e3analyze_p2.py", "e3analyze_p3.py", "e3add1_lib.py", "e3analyze_p1b.py", "e3analyze_p3_20m.py")
P3_LIST = os.path.join(HERE, "logs", "part3", "code_sha256_20261011_131228.txt")
PINNED = {**P3.PINNED, "results_part3.json": "17b1614d0e0f26337c9e39cf7f91a7e5057c0155c84c110d9dc6f0ccd32bb71c",
          "tables_part3.txt": "e07bb9910e1cdf85b520d3a293111f845d7f093fd5dc3ba1839eac8c25148cb1"}
EXPECTED_FAIL = list(P3.EXPECTED_FAIL)
SIZE = {"5m": {"total": 7630, "ckpts": [7344, 7497, 7630], "decay": 6104, "tail": list(range(7480, 7631, 10)),
               "n_params": 5010133},
        "20m": {"total": 15259, "ckpts": [14945, 15250, 15259], "decay": 12207,
                "tail": list(range(14960, 15251, 10)) + [15259], "n_params": 20001511}}
START = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d queue start: exp E3, plan (\S+) \(sha256 ([0-9a-f]{16})\), "
                   r"code commit ([0-9a-f]{40})$")


def sha256(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def gapped(run): return any(os.path.exists(os.path.join(HERE, "out", run, m)) for m in ("GAP", "DIVERGED"))
def listrows(p): return dict((ln.split("  ")[1], ln.split("  ")[0]) for ln in open(p).read().split("\n") if ln.strip())


def queue_new(sub):
    qt = open(os.path.join(HERE, "logs", sub, "queue_e3.txt")).read()
    p3 = open(os.path.join(HERE, "logs", "part3", "queue_e3.txt")).read()
    return qt.startswith(p3), (qt[len(p3):] if qt.startswith(p3) else qt).split("\n")


def order(q, names):
    """The registered runs' train lines in log order, a run retrained right after a relaunch counted once."""
    t = [ln.split("train ")[1].split(".yaml")[0] for ln in q if " train " in ln and ln.endswith("--require-committed")]
    t = [x for x in t if x in names]
    return [x for i, x in enumerate(t) if i == 0 or x != t[i - 1]]


def code_checks(add, sub, q, names):
    d = os.path.join(HERE, "logs", sub)
    starts = [ln for ln in q if " queue start: " in ln]
    have = sorted(f for f in os.listdir(d) if f.startswith("code_sha256_"))
    secs = lambda t: int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])   # noqa: E731
    timed = [f for f, s in zip(have, starts) if f[12:20] == s[:10].replace("-", "")
             and 0 <= secs(s[11:19].replace(":", "")) - secs(f[21:27]) <= 5] if len(have) == len(starts) else []
    lists = [listrows(os.path.join(d, f)) for f in have]
    first = lists[0] if lists else {}
    hp = [f"{h}  {p}\n" for p, h in sorted(first.items()) if p.startswith(("harness/", "data_prep/")) and p.endswith(".py")]
    dig = hashlib.sha256("".join(sorted(hp, key=lambda s: s.split("  ")[1].encode())).encode()).hexdigest()
    add("code digest harness+data_prep .py == FIXED 2 (92deaef0)", dig == E.FIXED["code_digest"], f"{dig[:16]} over {len(hp)} files")
    for p, h in E.FIXED["files"].items():
        got, d4 = first.get(p, "")[:16], E.FIXED_D4.get(p)
        add(f"{p} == FIXED 2", got == h or got == d4, first.get(p, "missing")[:16] + (" (D4)" if got == d4 else ""))
    p3 = {p: h for p, h in listrows(P3_LIST).items() if not p.startswith(E3X)}
    add("one code list per start of this plan, written just before it", bool(starts) and have == timed,
        f"{len(have)} lists, {len(starts)} starts")
    add("every code list equal, and equal to Part 3's (4962426) outside experiments/E3_seed_noise/",
        bool(lists) and all(x == first for x in lists) and {p: h for p, h in first.items() if not p.startswith(E3X)} == p3)
    add("the code lists hold this part's configs, the plan and the analyzers as on disk",
        bool(first) and all(first.get(f"{E3X}configs/{n}.yaml") == CONFIG_SHA[n] for n in names)
        and first.get(f"{E3X}plans/{PLAN}") == PLAN_SHA
        and all(first.get(E3X + a) == sha256(os.path.join(HERE, a)) for a in ANALYZERS))
    m = [START.match(s) for s in starts]
    commits = {x.group(3) for x in m if x}
    add(f"every start: plan {PLAN} ({PLAN_SHA[:8]}), one commit", bool(starts) and all(
        x and x.group(1) == PLAN and x.group(2) == PLAN_SHA[:16] for x in m) and len(commits) == 1,
        f"{len(starts)} start(s), commits {sorted(c[:12] for c in commits)}")
    add("plans/part1b_p3at20m.txt on disk has the registered sha", sha256(os.path.join(HERE, "plans", PLAN)) == PLAN_SHA)
    add("Parts 1, 2, 3 and 4d results (and tables) unchanged", all(sha256(os.path.join(HERE, f)) == h for f, h in PINNED.items()))
    return commits.pop() if len(commits) == 1 else None


def run_checks(r, name, seed, dseed, size, q, prereg):
    """e3analyze_p3.py's per-run checks; dseed None = no data_seed in the config or the start record (Part 1b)."""
    z, bad = SIZE[size], []
    ev = {(x["evalset_sha256"], x["tokenizer_sha256"], x["precision"], x["max_windows"]) for x in r["bpb"]}
    if ev != {(E.FIXED["evalset"], E.FIXED["tokenizer"], "fp32", None)}: bad.append(f"scorer/eval/tokenizer {ev}")
    if {x["run"] for x in r["bpb"]} != {name}: bad.append("bpb run field")
    per = {}
    for x in r["bpb"]: per.setdefault((x["set"], x["split"]), []).append(x["step"])
    if any(sorted(v) != z["ckpts"] for v in per.values()) or len(per) < 13: bad.append("bpb steps")
    pf = r["pf"]; cfg = os.path.join(HERE, "configs", name + ".yaml"); txt = open(cfg).read()
    if sha256(cfg) != pf["file_sha256"] or sha256(cfg) != CONFIG_SHA[name]: bad.append("config file sha")
    want = f"\nseed: {seed}\n" + (f"data_seed: {dseed}\n" if dseed is not None else "")
    if "lr: 0.003, embed_lr: 0.003, scalar_lr: 0.003}" not in txt or want not in txt or (dseed is None and "data_seed" in txt):
        bad.append("config LR/seeds")
    if not (pf["ok"] and pf["refusals"] == [] and pf["prereg"]["committed"] and pf["prereg"]["sha256"] == prereg): bad.append("preflight ok/strict/prereg")
    if pf["engine"] != E.FIXED["engine"] or pf["engine_fixed"] != E.FIXED["engine_fixed"]: bad.append("engine")
    if pf["shares"] != E.FIXED["shares"] or pf["n_files"] != E.FIXED["n_files"]: bad.append("shares/files")
    s, e = r["recs"][0], r["recs"][-1]
    if (s["event"], e["event"]) != ("start", "end") or len(r["recs"]) != 2: bad.append("records")
    if s["config_sha256"] != pf["config_sha256"] or s["prereg_sha256"] != prereg or not s["prereg_committed"]: bad.append("config/prereg sha")
    if ("data_seed" in s) != (dseed is not None) or s.get("data_seed") != dseed: bad.append("start record data_seed")
    if s["schedule"] != {"decay_start": z["decay"], "mode": "full", "total_steps": z["total"], "warmup_steps": 76}: bad.append("schedule")
    if (s["batch_tokens"], s["precision"], s["doc_attn"], s["optim_batched"], s["n_params"], s["resumed_from"]) != (
            32768, "bf16", "varlen", True, z["n_params"], None): bad.append("run setting")
    if e["step"] != z["total"] or not math.isfinite(e["loss"]): bad.append("end step/loss")
    if [t["step"] for t in r["tail"]] != z["tail"] or not all(math.isfinite(t["loss"]) for t in r["tail"]): bad.append("tail steps/finite")
    if P3.stream(r["tail"][-1]) != P3.stream(e): bad.append("tail end != end record stream")
    if sum(ln.endswith(f"train {name}.yaml --require-committed") for ln in q) != 1 or sum(ln.endswith(f"done {name} rc 0") for ln in q) != 1:
        bad.append("queue: not exactly one clean train")
    return bad


def ends(runs, names): return [json.dumps(P3.drawn(runs[n]["recs"][-1]), sort_keys=True) for n in names if n in runs]
def fails(ck): return [c["check"] for c in ck if not c["ok"] and c["check"] not in EXPECTED_FAIL]
