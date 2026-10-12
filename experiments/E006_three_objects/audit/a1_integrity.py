"""A1. Provenance: every scored tag complete, tag/arm fields, one weights hash per tag equal to run.json, guard.json
and the PC's own hash of the saved file; reference hashes equal E005/E004's; seeds and arms in run.json; G's update
digest equals C's; the e005w1 vs e005w1_r2 records; BIG items rebuilt; prompt hashes disjoint across draws.
Usage: python a1_integrity.py <pc_hashes.out from pc_hashes.sh>"""
import json, os, sys, glob, re
from common import *

pc = open(sys.argv[1]).read() if len(sys.argv) > 1 else ""
pc_w = {m.group(2): m.group(1) for m in re.finditer(r"^([0-9a-f]{64}) weights/" + PFX + r"(\S+)/model", pc, re.M)}
OUT = os.path.join(E6, "out")
SETS = {"e004": ("plain", "chat"), "big": ("plain", "chat"), "h5l": ("plain", "chat"), "al": ("plain", "chat")}
FULL = ["e004__plain", "e004__chat", "gen_e004__plain", "gen_e004__chat", "big__plain", "big__chat",
        "gen_big__plain", "gen_big__chat", "h5l__plain", "h5l__chat", "gen_h5l__plain", "gen_h5l__chat",
        "al__plain", "al__chat", "gen_al__plain", "gen_al__chat", "new__plain", "new__chat", "extra__plain",
        "extra__chat", "old__plain", "uprobe__plain", "cross__plain", "cross__chat", "kbig__plain",
        "khard__plain", "gen_cont__plain"]
N = {"e004": 640, "big": 576, "h5l": 192, "al": 64, "new": 1728, "extra": 192, "old": 1116, "uprobe": 576,
     "cross": 384, "kbig": 441, "khard": 80, "cont": 416}
REF = {"e005w": "f846ac3d 4ef6230a aa5f99f1 1d330be9 131c8113".split(),
       "e004w": "91a3aab0 aee5a7fa efd68a20 8ad34021 1bb9f377".split()}
tags = ["base"] + [f"{a}{s}" for a in ("C", "P", "G", "e005w", "e004w") for s in SEEDS] + ["C1t", "C2t", "e005w1_r2"]
problems, seen_w = [], {}
for t in tags:
    files = sorted(glob.glob(os.path.join(OUT, f"{PFX}{t}__*.jsonl")))
    names = {os.path.basename(f)[len(PFX) + len(t) + 2:-6]: f for f in files}
    want = FULL if t[0] in "CPG" and not t.endswith("t") or t in ("base", "e005w1_r2") or re.match(r"e005w[2-5]$", t) \
        else [n for n in FULL if n.split("__")[0].replace("gen_", "") in ("e004", "big", "h5l", "al")] if t.startswith("e004w") \
        else [n for n in FULL if n.split("__")[0].replace("gen_", "") in ("e004", "big", "h5l")] if t.endswith("t") else None
    if t == "e005w1":
        want = None  # the killed attempt: reported separately
    hashes, short = set(), []
    for n, f in names.items():
        recs = [json.loads(l) for l in open(f) if l.strip()]
        base = n.split("__")[0].replace("gen_", "")
        if want and n in want and len(recs) != N[base]:
            short.append(f"{n} {len(recs)}/{N[base]}")
        for r in recs:
            hashes.add(r.get("weights_sha256"))
            if r.get("tag") != t:
                problems.append(f"{t} {n}: record tag {r.get('tag')}")
                break
    missing = [n for n in (want or []) if n not in names]
    rj = os.path.join(OUT, f"{PFX}{t}__run.json")
    run = json.load(open(rj)) if os.path.exists(rj) else {}
    gj = os.path.join(E6, "logs", f"{t}.guard.json")
    guard = json.load(open(gj)) if os.path.exists(gj) else {}
    w = sorted(h for h in hashes if h)
    seen_w.setdefault(tuple(w), []).append(t)
    line = (f"{t}: files {len(names)}, missing {missing or 0}, short {short or 0}, record hashes {[h[:8] for h in w]},"
            f" run.json {str(run.get('weights_sha256'))[:8]} arm {run.get('arm')} seed {run.get('seed')} steps"
            f" {run.get('steps')} lr {run.get('lr')} tf32 {run.get('tf32')} | guard exit {guard.get('exit')}"
            f" killed {guard.get('killed')} w {str(guard.get('weights_sha256'))[:8]}")
    if t in pc_w:
        line += f" | PC file {pc_w[t][:8]} {'==' if [pc_w[t]] == w else '!='}"
    print(line)
    if len(w) != 1 or (run and run.get("weights_sha256") != w[0]) or missing or short:
        if t != "e005w1":
            problems.append(f"{t}: hashes {len(w)} / run.json mismatch / missing / short")
    m = re.match(r"(e005w|e004w)(\d)", t)
    if m and w and not w[0].startswith(REF[m.group(1)][int(m.group(2)) - 1]):
        problems.append(f"{t}: weights {w[0][:8]} not the reference {REF[m.group(1)][int(m.group(2)) - 1]}")
    if t[0] in "CPG" and not t.endswith("t") and run:
        if run.get("arm") != t[0] or run.get("seed") != int(t[1]) or run.get("steps") != 400 or run.get("lr") != 1.5e-4:
            problems.append(f"{t}: run.json arm/seed/steps/lr {run.get('arm')} {run.get('seed')}")
        if run.get("tf32"):
            problems.append(f"{t}: tf32 on")
print("distinct weight sets:", len(seen_w), "; shared:", [v for v in seen_w.values() if len(v) > 1])

# chat probes: weights hash per transcript equals the scored tag's
for t in ["base"] + [f"{a}{s}" for a in ("C", "P", "G", "e005w") for s in SEEDS]:
    p = os.path.join(E6, "transcripts", f"{PFX}{t}__greedy.jsonl")
    tr = [json.loads(l) for l in open(p)]
    hs = {x["weights_sha256"] for x in tr}
    recs = load(t, "e004", "plain")
    ok = len(tr) == 53 and hs == {recs[0]["weights_sha256"]} and all(x["tag"] == t for x in tr)
    g = json.load(open(os.path.join(E6, "logs", f"chat_{t}.guard.json")))
    if not ok or g.get("exit") != 0:
        problems.append(f"chat {t}: n {len(tr)} hashes {[h[:8] for h in hs]} exit {g.get('exit')}")
print("chat probes checked: 21")

# G's update part equals C's (digests in run.json)
for s in SEEDS:
    c = json.load(open(os.path.join(OUT, f"{PFX}C{s}__run.json")))
    g = json.load(open(os.path.join(OUT, f"{PFX}G{s}__run.json")))
    p = json.load(open(os.path.join(OUT, f"{PFX}P{s}__run.json")))
    print(f"s{s}: C digest {c['update_digest']['sha256'][:12]} G {g['update_digest']['sha256'][:12]}"
          f" P {p['update_digest']['sha256'][:12]} | G replay threads {len(g.get('replay_threads') or [])}"
          f" | P rejected {p['train_stats']['rejected_long']} C rejected {c['train_stats']['rejected_long']}")
    if c["update_digest"] != g["update_digest"] or p["update_digest"] == c["update_digest"]:
        problems.append(f"s{s}: digest relation wrong")

# e005w1 (killed) vs e005w1_r2: every record equal apart from tag
os.environ["AUDIT_NOMAP"] = "1"
for f in sorted(glob.glob(os.path.join(OUT, f"{PFX}e005w1__*.jsonl"))):
    n = os.path.basename(f)[len(PFX) + 8:]
    a = [json.loads(l) for l in open(f) if l.strip()]
    b = [json.loads(l) for l in open(os.path.join(OUT, f"{PFX}e005w1_r2__{n}")) if l.strip()]
    strip = lambda r: {k: v for k, v in r.items() if k not in ("tag",)}
    same = all(strip(x) == strip(y) for x, y in zip(a, b))
    print(f"e005w1 {n}: {len(a)} vs r2 {len(b)}, equal on the first {len(a)}: {same}")
    if not same:
        problems.append(f"e005w1 {n} differs from r2")

# BIG rebuilt with E004's generator; H5L count; prompt hashes disjoint
code_path()
import items_e004 as I
big = [json.loads(l) for l in open(os.path.join(E6, "big", "big_items.jsonl"))]
for f in ("H5", "C_noupd", "C_twoslot"):
    mine = json.loads(json.dumps(I.build(6006, f, 192)))  # tuples -> lists, as the file stores them
    file_f = [x for x in big if x["family"] == f]
    print(f"BIG {f}: rebuilt == file: {mine == file_f}")
    if mine != file_f:
        problems.append(f"BIG {f} differs from items_e004.build(6006)")
print("big_items sha256", sha256_file(os.path.join(E6, "big", "big_items.jsonl"))[:16])
H = {}
for set_name in ("e004", "big", "h5l", "al"):
    H[set_name] = {r["h"] for r in load("base", set_name, "plain")}
for a in H:
    for b in H:
        if a < b and H[a] & H[b]:
            problems.append(f"prompt hashes shared {a}/{b}: {len(H[a] & H[b])}")
print("prompt hashes per set:", {k: len(v) for k, v in H.items()})
print("PROBLEMS:", problems or "none")
