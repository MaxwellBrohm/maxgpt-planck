"""DIAGNOSTIC ONLY (not a gate, not SPEC s5's cheaters): how often simple value-picking shortcuts land on the
gold of the real dev items' VAL probes. It reads user turns only and picks a VALUE; the real cheater responders
(next step) turn picks into replies and run through the real graders and the 12-turn loop.
  FIRST / LAST   first / latest in-context value of the probe's pool
  WORDING        value right after the longest word run a user turn shares with the question (E004 O4)
  OVERLAP_A / _B turn with most content words shared with the question, latest wins; its first / last value (O5)
  MARKER         latest turn with a correction marker, else the first value (O3)
BIND is scored per pair (both twins right). Writes logs/quick_shortcuts.txt."""
import os
import re
import sys
from collections import defaultdict

import pools_vals as V
import test_gens as T

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(T.HERE, "..", "experiments", "E004_general_updating", "code"))
import text_e004 as TE  # noqa: E402
sys.path.pop(0)

MARKERS = ["actually", "sorry", "update", "change of plan", "scratch that", "oh wait", "make that", "changed",
           "switched", "moved"]
NAMES = ["FIRST", "LAST", "WORDING", "OVERLAP_A", "OVERLAP_B", "MARKER"]
LOW = {v.lower() for v in V.ALL_VALUES}


def pool_values(p):
    if p.get("pool_values"):
        return p["pool_values"]
    return V.GUESS_POOL.get(p["pool"]) or V.POOLS.get(p["pool"], [])


def mentions(text, vals):
    """(position, value) for every mention, in order."""
    out = []
    for v in vals:
        out += [(m.start(), v) for m in V.value_re(v).finditer(text)]
    return sorted(out)


def content(text):
    return {w for w in TE.words(text) if w not in TE.STOPWORDS and w not in TE.OBJECT_FREE_VERBS and w not in LOW}


def longest_run(tw, qw):
    best, end = 0, None
    for i in range(len(tw)):
        for j in range(len(qw)):
            k = 0
            while i + k < len(tw) and j + k < len(qw) and tw[i + k] == qw[j + k]:
                k += 1
            if k > best:
                best, end = k, i + k
    return best, end


def picks(r, p):
    vals = pool_values(p)
    turns = [t for t in r["turns"][:p["turn"] - 1] if t["kind"] != "D"]
    hits = [(t, m) for t in turns for m in mentions(t["text"], vals)]
    if not hits:
        return {}
    out = {"FIRST": hits[0][1][1], "LAST": hits[-1][1][1]}
    qw = re.findall(r"[a-z0-9']+", p["question"].lower())
    best = (-1, -1, None)
    for t in turns:
        ms = mentions(t["text"], vals)
        if not ms:
            continue
        tw = re.findall(r"[a-z0-9']+", t["text"].lower())
        n, end = longest_run(tw, qw)
        after = [v for v in (x[1] for x in ms) if end is not None and v.lower() in tw[end:]]
        v = after[0] if after else ms[-1][1]
        if n >= best[0]:
            best = (n, t["i"], v)
    out["WORDING"] = best[2]
    qc = content(p["question"])
    ov = [(len(content(t["text"]) & qc), t["i"], t) for t in turns if mentions(t["text"], vals)]
    top = max(ov, key=lambda x: (x[0], x[1]))[2]
    ms = mentions(top["text"], vals)
    out["OVERLAP_A"], out["OVERLAP_B"] = ms[0][1], ms[-1][1]
    mk = [t for t in turns if mentions(t["text"], vals) and any(m in t["text"].lower() for m in MARKERS)]
    out["MARKER"] = mentions(mk[-1]["text"], vals)[-1][1] if mk else hits[0][1][1]
    return out


def main():
    recs = T.load()
    score = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["knowledge"]:
            continue
        for p in r["probes"]:
            if p["grader"] != "VAL" or not p["gold"]:
                continue
            got = picks(r, p)
            key = f"{r['family']}:{r['cell']}"
            for n in NAMES:
                right = got.get(n) == p["gold"]
                if r["family"] == "BIND":
                    pairs[key][(n, r["meta"]["pair_id"])].append(right)
                else:
                    score[key][n].append(right)
    for key, d in pairs.items():
        for (n, _), v in d.items():
            score[key][n].append(all(v))
    lines = [f"{'family:cell':24s} " + " ".join(f"{n:>9s}" for n in NAMES) + "   units"]
    fam_tot = defaultdict(lambda: defaultdict(list))
    for key in sorted(score):
        row = score[key]
        lines.append(f"{key:24s} " + " ".join(f"{sum(row[n]) / len(row[n]):9.2f}" for n in NAMES)
                     + f"   {len(row[NAMES[0]])}")
        if key.split(":")[0] == "T0" or key in ("TWOHOP:COMPOSE", "RECALL:abstain"):
            continue
        for n in NAMES:
            fam_tot[key.split(":")[0]][n] += row[n]
    lines.append("")
    lines.append("pooled over composite units only: T0 (a gate), TWOHOP:COMPOSE (diagnostic) and the X probes of")
    lines.append("RECALL:abstain (their unit also needs the abstain) are left out of the pooled rows")
    for f in sorted(fam_tot):
        row = fam_tot[f]
        lines.append(f"{f + ' (pooled)':24s} " + " ".join(f"{sum(row[n]) / len(row[n]):9.2f}" for n in NAMES))
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(T.HERE, "logs"), exist_ok=True)
    with open(os.path.join(T.HERE, "logs", "quick_shortcuts.txt"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
