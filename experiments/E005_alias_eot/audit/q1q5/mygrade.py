"""Independent strict grader written from E004 notes (c) clauses 1-7 plus the documented pre-run matching choices."""
import re, sys
sys.path.insert(0, "REPO/experiments/E005_alias_eot/code")
from text_e004 import STOPWORDS  # fixed list (data only)

NUM = {"2": ["two", "2nd"], "3": ["three", "third", "3rd"], "4": ["four", "fourth", "4th"], "5": ["five", "fifth", "5th"],
       "6": ["six", "sixth", "6th"], "7": ["seven", "seventh", "7th"], "8": ["eight", "eighth", "8th"], "9": ["nine", "ninth", "9th"]}

def toks(t):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", t.lower())

def mentions(text, v):
    forms = [v] + NUM.get(v, []) + (["grey"] if v.lower() == "gray" else [])
    pos = []
    for f in forms:
        fl = 0 if f == "May" else re.I
        pos += [m.start() for m in re.finditer(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(f), text, fl)]
    return sorted(pos)

NEG = ["not", "no longer", "never", "instead", "rather than"]
HEDGE = ["i think", "maybe", "probably", "might", "perhaps", "not sure", "i don't know", "i believe",
         "may be", "could be", "i guess", "possibly"]

def has_phrase(low, p):
    return re.search(r"(?<![a-z'])%s(?![a-z])" % re.escape(p), low) is not None

def objwords(phrase, head, alias):
    ws = set(toks(phrase)) | {head.lower()} | (set(toks(alias)) if alias else set())
    return {w for w in ws if w not in STOPWORDS}

def grade(reply, stop, gold, pool, obj):
    reply = (reply or "").replace("’", "'").replace("‘", "'")
    low = reply.lower(); ws = toks(reply)
    fails = []
    if not reply.strip() or stop == "cap": fails.append(1)
    tri = [tuple(ws[i:i+3]) for i in range(len(ws) - 2)]
    if len(set(tri)) != len(tri) or any(ws[i] == ws[i+1] == ws[i+2] for i in range(len(ws) - 2)): fails.append(2)
    if not mentions(reply, gold): fails.append(3)
    if any(mentions(reply, v) for v in pool if v != gold): fails.append(4)
    if "n't" in low or any(has_phrase(low, p) for p in NEG) or re.search(r"(?<![a-z'])no\s*(?:[,.!;]|$)", low): fails.append(5)
    if "?" in reply or any(has_phrase(low, h) for h in HEDGE): fails.append(6)
    if low.lstrip().startswith(("user:", "assistant:")) or any(w == "my" and set(ws[i+1:i+4]) & obj for i, w in enumerate(ws)): fails.append(7)
    return not fails, fails
