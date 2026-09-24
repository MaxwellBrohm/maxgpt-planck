"""E004 hardened STRICT free-generation grader (notes.txt (c), clauses 1-7). Pure Python: no torch, no model.
Structure follows E002 gen_probe.py (grader + self-test here; the generation loop is gen_run.py).

grade(reply, stop, gold, pool, obj_words) -> dict. A reply is right (strict) only if ALL hold:
  1 non-empty and generation stopped before the cap (stop != "cap"; a capped reply is a loop)
  2 no word 3-gram occurs twice and no word occurs 3 times in a row
  3 the gold is mentioned (whole word, case-insensitive)
  4 no OTHER value of the item's WHOLE value pool is mentioned (in-context or not: guesses fail)
  5 no negation cue anywhere (not, n't, no longer, never, instead, rather than, interjection "no")
  6 no question mark and no hedge (I think, maybe, probably, might, perhaps, not sure, I don't know, I believe;
    plus may be, could be, I guess, possibly)
  7 not the user's voice: no "my" within 3 words before a word of the asked object (phrase words, head noun,
    alias words), and the reply does not start with "User:" or "Assistant:"
Lenient (diagnostic only): the first pool value mentioned is the gold.
Value matching (choices fixed 2026-09-24 before any E004 model run; notes.txt STEP 4):
  case-insensitive whole words, except the month "May", which must be capitalized (modal "may");
  curly apostrophes are normalized; "grey" counts as "gray"; the number pool (2-9) also matches number words
  (two..nine), digit ordinals (2nd..9th) and ordinal words (third..ninth; not "second", which is also a unit
  of time); "no," of the pre-registered cue list is read as the interjection "no" followed by , . ! ; or the end.
"""
import re

from text_e004 import words, STOPWORDS

NEG_WORDS = ["not", "no longer", "never", "instead", "rather than"]
HEDGES = ["i think", "maybe", "probably", "might", "perhaps", "not sure", "i don't know", "i believe",
          "may be", "could be", "i guess", "possibly"]   # last four added before any model run (notes STEP 4)
ROLE_TAGS = ("user:", "assistant:")
NUM_WORDS = {"2": ["two", "2nd"], "3": ["three", "third", "3rd"], "4": ["four", "fourth", "4th"],
             "5": ["five", "fifth", "5th"], "6": ["six", "sixth", "6th"], "7": ["seven", "seventh", "7th"],
             "8": ["eight", "eighth", "8th"], "9": ["nine", "ninth", "9th"]}
SPELLINGS = {"gray": ["grey"]}
CLAUSES = (1, 2, 3, 4, 5, 6, 7)


def norm(text):
    return (text or "").replace("’", "'").replace("‘", "'")


def _forms(v):
    return [v] + NUM_WORDS.get(v, []) + SPELLINGS.get(v.lower(), [])


def value_hits(text, v):
    """start offsets of every mention of value v (any accepted form) in text."""
    out = []
    for f in _forms(v):
        flags = 0 if f == "May" else re.I
        out += [m.start() for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(f) + r"(?![A-Za-z0-9])", text, flags)]
    return sorted(out)


def _cue(text, phrase):
    return re.search(r"(?<![a-z'])" + re.escape(phrase) + r"(?![a-z])", text) is not None


# ---------------- clauses: each takes the context dict c and returns True when the reply PASSES it ----------------
def c1_nonempty_stopped(c):
    return bool(c["reply"].strip()) and c["stop"] != "cap"


def c2_no_loop(c):
    ws = c["ws"]
    tri = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    if len(tri) != len(set(tri)):
        return False
    return not any(ws[i] == ws[i + 1] == ws[i + 2] for i in range(len(ws) - 2))


def c3_gold(c):
    return bool(value_hits(c["reply"], c["gold"]))


def c4_no_other(c):
    return not any(value_hits(c["reply"], v) for v in c["pool"] if v != c["gold"])


def c5_no_negation(c):
    t = c["low"]
    if "n't" in t or any(_cue(t, p) for p in NEG_WORDS):
        return False
    return re.search(r"(?<![a-z'])no\s*(?:[,.!;]|$)", t) is None


def c6_no_question_hedge(c):
    return "?" not in c["reply"] and not any(_cue(c["low"], h) for h in HEDGES)


def c7_assistant_voice(c):
    if c["low"].lstrip().startswith(ROLE_TAGS):
        return False
    ws, obj = c["ws"], c["obj"]
    return not any(w == "my" and set(ws[i + 1:i + 4]) & obj for i, w in enumerate(ws))


CHECKS = {1: c1_nonempty_stopped, 2: c2_no_loop, 3: c3_gold, 4: c4_no_other, 5: c5_no_negation,
          6: c6_no_question_hedge, 7: c7_assistant_voice}


def lenient(reply, gold, pool):
    firsts = [(h[0], v) for v in pool for h in [value_hits(reply, v)] if h]
    return bool(firsts) and min(firsts)[1] == gold


def grade(reply, stop, gold, pool, obj_words, checks=None, cands=None):
    """reply: the generated text (already cut at the stop); stop: "newline" | "user" | "eos" | "cap".
    gold: the gold value (no leading space); pool: the item's whole value pool (must contain gold);
    obj_words: lowercased words naming the asked object; cands (in-context values) is carried for mutants and
    diagnostics only, no clause reads it. Returns strict, lenient, fails, and format stats."""
    if gold not in pool:
        raise ValueError(f"gold {gold!r} not in pool")
    reply = norm(reply)
    c = dict(reply=reply, low=reply.lower(), ws=words(reply), stop=stop, gold=gold, pool=list(pool),
             obj={w.lower() for w in obj_words}, cands=list(cands or []))
    fails = [k for k, fn in (checks or CHECKS).items() if not fn(c)]
    n = len(c["ws"])
    return dict(strict=not fails, lenient=lenient(reply, gold, pool), fails=fails, n_words=n,
                bare=0 < n <= 2, long=n >= 4, capped=stop == "cap", voice=7 in fails)


# ---------------- item adapters: (gold, pool, obj_words) ----------------
def obj_words_of(phrase, head, alias=None):
    ws = set(words(phrase)) | {head.lower()} | (set(words(alias)) if alias else set())
    return {w for w in ws if w not in STOPWORDS}


def spec_e004(item):
    """E004 eval item (items_e004.build) -> (gold, pool, obj_words)."""
    phrase, head = item["objects"][item["asked"]]
    alias = item.get("alias") if item.get("alias_obj") == item["asked"] else None
    return item["gold"], list(item["values"]), obj_words_of(phrase, head, alias)


CONT_OBJ = {"day": ("dentist appointment", "appointment"), "color": ("new car", "car")}


def spec_cont(item, fam):
    """E001/E002 continuity item (items_new / uprobe_items; fam "day" or "color") -> (gold, pool, obj_words)."""
    import items_new as N
    pool = N.DAYS if fam == "day" else N.COLORS
    return item["cands"]["gold"].strip(), list(pool), obj_words_of(*CONT_OBJ[fam])
