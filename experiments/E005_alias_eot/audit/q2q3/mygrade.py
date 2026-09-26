"""independent strict grader written from the clause list in the notes (not imported from gen_grade)."""
import re
NUMW = {"2": ["two", "2nd"], "3": ["three", "third", "3rd"], "4": ["four", "fourth", "4th"],
        "5": ["five", "fifth", "5th"], "6": ["six", "sixth", "6th"], "7": ["seven", "seventh", "7th"],
        "8": ["eight", "eighth", "8th"], "9": ["nine", "ninth", "9th"]}
STOP = None
def toks(t):
    return re.findall(r"[a-z0-9']+", t.lower())
def forms(v):
    f = [v] + NUMW.get(v, [])
    if v.lower() == "gray": f.append("grey")
    return f
def mentions(text, v):
    for f in forms(v):
        fl = 0 if f == "May" else re.I
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", text, fl):
            return True
    return False
def first_pos(text, v):
    ps = []
    for f in forms(v):
        fl = 0 if f == "May" else re.I
        ps += [m.start() for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", text, fl)]
    return min(ps) if ps else None
NEG = ["not", "no longer", "never", "instead", "rather than"]
HEDGE = ["i think", "maybe", "probably", "might", "perhaps", "not sure", "i don't know", "i believe", "may be",
         "could be", "i guess", "possibly"]
def cue(t, p):
    return re.search(r"(?<![a-z'])" + re.escape(p) + r"(?![a-z])", t) is not None
def strict(reply, stop, gold, pool, objw):
    r = (reply or "").replace("’", "'").replace("‘", "'")
    low = r.lower()
    ws = toks(r)
    if not r.strip() or stop == "cap": return False
    tri = [tuple(ws[i:i+3]) for i in range(len(ws)-2)]
    if len(tri) != len(set(tri)) or any(ws[i] == ws[i+1] == ws[i+2] for i in range(len(ws)-2)): return False
    if not mentions(r, gold): return False
    if any(mentions(r, v) for v in pool if v != gold): return False
    if "n't" in low or any(cue(low, p) for p in NEG) or re.search(r"(?<![a-z'])no\s*(?:[,.!;]|$)", low): return False
    if "?" in r or any(cue(low, h) for h in HEDGE): return False
    if low.lstrip().startswith(("user:", "assistant:")): return False
    if any(w == "my" and set(ws[i+1:i+4]) & objw for i, w in enumerate(ws)): return False
    return True
