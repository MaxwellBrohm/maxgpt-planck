"""Bank specs (BANKPASS s1, s3): every bank the pipeline reads, its class, the holes a line must and may hold, the
word bounds, the side that speaks it and how it is judged. The hole sets are read off the FAKE banks (banks.py,
banks_keys.py), so a real bank keeps exactly the shape the skeleton code fills: required = holes in EVERY FAKE line of
the bank, allowed = holes in ANY of them. Nothing here is a bank line."""
import re

import banks as B
from banks_keys import KEYS

# line classes (s1 codes) by bank-name prefix; first match wins
CLASS_PREFIX = (("key.", "K"), ("marker.", "M"), ("open.", "O"), ("close.", "C"), ("social.", "S"),
                ("identity.", "S"), ("recap", "S"), ("return", "S"), ("swap.", "S"), ("rule.", "R"), ("list.", "L"),
                ("lookup.", "U"), ("system.", "Y"))
# word bounds per class (register-free: the render's user bounds start at 1-3 words, the skeleton caps at 25-30)
WORDS = {"K": (2, 25), "M": (1, 4), "O": (1, 20), "C": (2, 20), "S": (2, 20), "R": (3, 25), "L": (3, 30),
         "U": (3, 25), "Y": (4, 40)}
VERBALIZED = {"M", "O"}        # s2b: verbalized sampling for openings and markers (and topics, a pool class)
POOL_TYPES = ("name", "surname", "city", "assistant_name", "pet_name", "pet_kind", "job", "hobby", "food", "grocery",
              "chore", "plan", "object", "relation", "avoid_word", "entity_kind")
HUMAN_POOLS = {"name": "SSA baby names", "surname": "Census 2010 surnames", "city": "GeoNames cities15000"}
COMPUTED_POOLS = {"avoid_word", "req_noun", "req_verb", "req_adj"}   # W4: req_* from wordlist_v1, as banks
KEY_ROLES = ("plant", "query", "bait", "corr", "twin")
# speech act a key line must carry (s2e); extraction gold: the fill for plant and corr, "not said" otherwise
KEY_ACTS = {"plant": "states", "corr": "replaces", "query": "asks", "bait": "asks", "twin": "mentions_not_own"}
_HOLE = re.compile(r"\{(\w+)\}")


def class_of(bank):
    for prefix, cls in CLASS_PREFIX:
        if bank.startswith(prefix):
            return cls
    if bank.startswith("pool."):
        return "P"
    raise KeyError(bank)


def _holes(lines):
    sets = [set(_HOLE.findall(t)) for t in lines]
    return (set.intersection(*sets) if sets else set()), (set.union(*sets) if sets else set())


def key_banks():
    """{"key.<key>.<role>": (fake lines, forms or None)}; query and corr carry (form, line) pairs."""
    out = {}
    for k, d in KEYS.items():
        for role in KEY_ROLES:
            raw = d[role]
            if role in ("query", "corr"):
                out[f"key.{k}.{role}"] = ([t for _, t in raw], sorted({f for f, _ in raw}))
            else:
                out[f"key.{k}.{role}"] = (list(raw), None)
    return out


def line_specs():
    """{bank: spec} for every line bank: class, required and allowed holes, word bounds, verbalized, forms, side."""
    out = {}
    raw = {b: (ls, None) for b, ls in B.BANKS.items()}
    raw.update(key_banks())
    raw["marker.fix"] = (list(B.MARKERS), None)
    raw["marker.err"] = (list(B.ERR_FIX_MARKERS), None)
    for bank, (lines, forms) in sorted(raw.items()):
        cls = class_of(bank)
        req, allowed = _holes(lines)
        if cls == "K":      # W3: one or two FAKE lines made {old} and {p_obj} required; a correction may leave out
            req -= {"old", "p_obj"}   # the value it replaces and a twin need not name the person (both stay allowed)
            if KEYS[bank.split(".")[1]]["noun"]:      # amendment 4: any line of a noun key may name its noun
                allowed |= {"o"}
        lo, hi = WORDS[cls]
        out[bank] = {"class": cls, "holes_required": sorted(req), "holes_allowed": sorted(allowed), "min_w": lo,
                     "max_w": hi, "verbalized": cls in VERBALIZED, "forms": forms,
                     "side": "system" if cls == "Y" else "user", "fake_n": len(lines),
                     "role": bank.split(".")[2] if cls == "K" else None,
                     "key": bank.split(".")[1] if cls == "K" else None}
    return out


def pool_specs():
    out = {}
    for vt in POOL_TYPES:
        kind = "human" if vt in HUMAN_POOLS else "computed" if vt in COMPUTED_POOLS else "teacher"
        out["pool." + vt] = {"class": "P", "vtype": vt, "author_kind": kind, "min_w": 1, "max_w": 4}
    out["topic"] = {"class": "T", "author_kind": "teacher", "min_w": 2, "max_w": 8}
    out["topicwords"] = {"class": "W", "author_kind": "teacher"}
    out["intent"] = {"class": "I", "author_kind": "teacher", "min_w": 3, "max_w": 16}
    out["label"] = {"class": "N", "author_kind": "teacher", "min_w": 1, "max_w": 3}
    out["wordlist"] = {"class": "V", "author_kind": "computed"}
    return out


def loadable(bank):
    """a bank load.py can install: every line bank, label.<key>, topic, topicwords, and pool.<type> for a pool the
    skeleton reads that is not PROGRAM (closed lists); req_* are computed banks since W4 (wordlist_v1's confirmed
    required families, so a frozen set carries every pool the skeleton draws, hashed). Any
    other bank in a manifest is UNLOADABLE at admit: its content would be dropped while its ref claimed it."""
    import pools as P
    if bank in line_specs() or bank in ("topic", "topicwords"):
        return True
    if bank.startswith("label."):
        return bank.split(".", 1)[1] in KEYS
    if bank.startswith("listname."):         # W4: banks.LIST_NAMES, one chosen name per list type
        return bank.split(".", 1)[1] in B.LIST_NAMES
    if bank.startswith("pool."):
        vt = bank.split(".", 1)[1]
        return vt in P.POOLS and P.POOLS[vt].provenance != "PROGRAM" and vt != "topic"
    return False


def hole_problems(bank, template, spec=None):
    """hole integrity (s2c): every required hole present, no hole outside the allowed set, balanced braces."""
    spec = spec or line_specs()[bank]
    holes = set(_HOLE.findall(template))
    out = [f"missing {{{h}}}" for h in spec["holes_required"] if h not in holes]
    out += [f"stray {{{h}}}" for h in sorted(holes - set(spec["holes_allowed"]))]
    found = _HOLE.findall(template)
    out += [f"repeated {{{h}}}" for h in sorted(holes) if found.count(h) > 1]
    rest = _HOLE.sub("", template)
    if "{" in rest or "}" in rest:
        out.append("unbalanced braces")
    return out
