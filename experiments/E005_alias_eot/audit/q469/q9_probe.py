from common import *
import hashlib
from collections import Counter, defaultdict
sys.path.insert(0, E5 + "/code")
from pools_train import POOLS

TR = f"{SLUG}__{{}}__greedy.jsonl"


def load(exp, tag):
    root = E5 if exp == "E005" else E4
    return [json.loads(l) for l in open(f"{root}/transcripts/{TR.format(tag)}")]


# training answer templates -> regexes. strict: {v} must be a value of a training value pool (any type);
# loose: {v} = any single word. {o} = 1-4 words.
allv = sorted({v for P in POOLS.values() for v in P["values"]}, key=len, reverse=True)
VSTRICT = "(?:" + "|".join(re.escape(v) for v in allv) + ")"
VLOOSE = r"[A-Za-z0-9]+"
OBJ = r"[A-Za-z][A-Za-z'\- ]{0,40}?"
tpls = sorted({t for P in POOLS.values() for key in ("ans", "ans_upd") for t in P[key]})


def tre(t, vpat):
    parts = re.split(r"(\{v\}|\{o\})", t)
    out = ""
    for p in parts:
        if p == "{v}":
            out += vpat
        elif p == "{o}":
            out += OBJ
        else:
            out += re.escape(p)
    return re.compile(r"(?<![A-Za-z])" + out, re.I)


RS = [tre(t, VSTRICT) for t in tpls]
RL = [tre(t, VLOOSE) for t in tpls]


def has_tpl(text, rs):
    return any(r.search(text) for r in rs)


def repeated_line(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    c = Counter(lines)
    return bool(c) and max(c.values()) >= 3


def prompt_sig(convs):
    s = [(c["id"], c["cat"], c["system"], [t["user"] for t in c["turns"]]) for c in convs]
    return hashlib.sha256(json.dumps(s).encode()).hexdigest()[:16]


rows = []
models = [("E005", "base")] + [("E005", t) for t in SEEDS] + [("E004", t) for t in SEEDS]
print(f"{'model':10} {'conv':>4} {'turns':>5} {'capped':>7} {'eos':>5} {'tplS':>6} {'tplL':>6} {'rep3':>5} {'leak':>5} {'user':>5} {'empty':>5} {'mean_tok':>8} {'checks':>7} {'promptsig':>17}")
for exp, t in models:
    C = load(exp, t)
    turns = [tt for c in C for tt in c["turns"]]
    n = len(turns)
    capped = sum(tt["flags"]["hit_max"] for tt in turns)
    # my own capped check: not stopped at eos and n_gen_tokens >= the max_new of that turn is not stored; use stopped_eos
    eos = sum(tt["stopped_eos"] for tt in turns)
    ts = sum(has_tpl(tt["assistant"], RS) for tt in turns)
    tl = sum(has_tpl(tt["assistant"], RL) for tt in turns)
    rep = sum(repeated_line(tt["assistant"]) for tt in turns)
    leak = sum(tt["flags"]["leaked_template"] for tt in turns)
    usr = sum(tt["flags"]["invented_user_turn"] for tt in turns)
    emp = sum(tt["flags"]["empty"] for tt in turns)
    mt = sum(tt["n_gen_tokens"] for tt in turns) / n
    ch = [v for c in C for v in c["checks"].values()]
    print(f"{exp + ' ' + t:10} {len(C):4d} {n:5d} {capped:3d}={capped / n:.2f} {eos:5d} {ts / n:6.2f} {tl / n:6.2f} {rep:5d} {leak:5d} {usr:5d} {emp:5d} {mt:8.1f} {sum(v is True for v in ch):3d}/{len(ch):3d}(None {sum(v is None for v in ch)}) {prompt_sig(C):>17}")
    rows.append((exp, t, C))

# not-stopped turns: are they all at the cap?
print("\nturns that did not stop at eos but are not flagged hit_max (should be 0):")
for exp, t, C in rows:
    x = sum((not tt["stopped_eos"]) and not tt["flags"]["hit_max"] for c in C for tt in c["turns"])
    print(f"  {exp} {t}: {x}")

# per-category capped share and check pass rate
print("\nper category: capped share / probe checks passed")
cats = sorted({c["cat"] for c in rows[0][2]})
print("  " + " ".join(f"{c:>14}" for c in cats))
for exp, t, C in rows:
    cs = []
    for cat in cats:
        tt = [x for c in C if c["cat"] == cat for x in c["turns"]]
        ch = [v for c in C if c["cat"] == cat for v in c["checks"].values()]
        cs.append(f"{sum(x['flags']['hit_max'] for x in tt) / len(tt):.2f}/{sum(v is True for v in ch)}of{len(ch)}")
    print(f"  {exp} {t:4} " + " ".join(f"{s:>14}" for s in cs))
