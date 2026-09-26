"""Auditor helpers (Q4/Q6/Q9). No torch, no model. Reads raw records only."""
import sys, os, json, re, pickle, random, math
sys.dont_write_bytecode = True
E5 = "REPO/experiments/E005_alias_eot"
E4 = "REPO/experiments/E004_general_updating"
SLUG = "HuggingFaceTB__SmolLM2-135M-Instruct"
SEEDS = ["s1", "s2", "s3", "s4", "s5"]
PASS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "C_noupd", "C_twoslot"]
FAMS = PASS + ["ID"]
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = HERE + "/items_eval_e005.pkl"


def recs(exp, tag, kind, render="plain"):
    root = E5 if exp == "E005" else E4
    p = f"{root}/out/{SLUG}__{tag}__{kind}__{render}.jsonl"
    return [json.loads(l) for l in open(p)]


def items_flat():
    """E004/E005 eval draw 4004 built from E005's code copy (item definitions only), flattened in family order."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    sys.path.insert(0, E5 + "/code")
    import items as I  # noqa: F401  (same import order as e004_sets)
    import items_e004 as E
    d = E.draw("eval")
    flat = [it for f in FAMS for it in d[f]]
    pickle.dump(flat, open(CACHE, "wb"))
    return flat


# ---------------- my own strict grader (clauses 1-7 of E004 notes (c), re-implemented) ----------------
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NUMW = {"2": ["two", "2nd"], "3": ["three", "third", "3rd"], "4": ["four", "fourth", "4th"],
        "5": ["five", "fifth", "5th"], "6": ["six", "sixth", "6th"], "7": ["seven", "seventh", "7th"],
        "8": ["eight", "eighth", "8th"], "9": ["nine", "ninth", "9th"]}


def mentions(text, v):
    forms = [v] + NUMW.get(v, []) + (["grey"] if v.lower() == "gray" else [])
    for f in forms:
        fl = 0 if f == "May" else re.I
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", text, fl):
            return True
    return False


def first_pos(text, v):
    forms = [v] + NUMW.get(v, []) + (["grey"] if v.lower() == "gray" else [])
    best = None
    for f in forms:
        fl = 0 if f == "May" else re.I
        m = re.search(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", text, fl)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def my_strict(reply, stop, gold, pool, obj):
    r = (reply or "").replace("’", "'").replace("‘", "'")
    low = r.lower()
    ws = WORD.findall(low)
    fails = []
    if not r.strip() or stop == "cap":
        fails.append(1)
    tri = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    if len(tri) != len(set(tri)) or any(ws[i] == ws[i + 1] == ws[i + 2] for i in range(len(ws) - 2)):
        fails.append(2)
    if not mentions(r, gold):
        fails.append(3)
    if any(mentions(r, v) for v in pool if v != gold):
        fails.append(4)
    neg = "n't" in low or any(re.search(r"(?<![a-z'])" + re.escape(p) + r"(?![a-z])", low)
                              for p in ["not", "no longer", "never", "instead", "rather than"]) \
        or re.search(r"(?<![a-z'])no\s*(?:[,.!;]|$)", low)
    if neg:
        fails.append(5)
    hed = ["i think", "maybe", "probably", "might", "perhaps", "not sure", "i don't know", "i believe",
           "may be", "could be", "i guess", "possibly"]
    if "?" in r or any(re.search(r"(?<![a-z'])" + re.escape(h) + r"(?![a-z])", low) for h in hed):
        fails.append(6)
    if low.lstrip().startswith(("user:", "assistant:")) or any(
            w == "my" and set(ws[i + 1:i + 4]) & obj for i, w in enumerate(ws)):
        fails.append(7)
    return (not fails), fails


def obj_words(it):
    sys.path.insert(0, E5 + "/code")
    from text_e004 import STOPWORDS
    phrase, head = it["objects"][it["asked"]]
    ws = set(WORD.findall(phrase.lower())) | {head.lower()}
    if it.get("alias_obj") == it["asked"] and it.get("alias"):
        ws |= set(WORD.findall(it["alias"].lower()))
    return {w for w in ws if w not in STOPWORDS}


# ---------------- stats ----------------
def binom_two_sided(b, c):
    """exact McNemar: two-sided binomial p on the discordant pairs (b, c), p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * s)


def boot_ci(diffs, B=4000, seed=12345):
    rng = random.Random(seed)
    n = len(diffs)
    ms = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        ms.append(s / n)
    ms.sort()
    return ms[int(0.025 * B)], ms[int(0.975 * B) - 1]
