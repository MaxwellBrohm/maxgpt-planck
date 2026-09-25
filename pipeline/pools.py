"""Slot pools (SPEC section 4): loaders, provenance, value features, and the programmatic nonce generator.
Closed programmatic lists (weekday, month, colour, time of day in words, number words, ordinals) carry provenance
"PROGRAM"; every other pool is FAKE in this build (fake_data.py) until the real sources and the teacher bank pass
exist. Every value passes heldout.pool_ok before use; dropped values are recorded per pool."""
import hashlib
import json
import random

import fake_data as F
import heldout

# value type -> (source list, provenance, source note, license)
_DEFS = {
    "name": (F.FIRST_NAMES, "FAKE", "stand-in for SSA baby names", "public domain (real source)"),
    "assistant_name": (F.ASSISTANT_NAMES, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "pet_name": (F.PET_NAMES, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "pet_kind": (F.PET_KINDS, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "city": (F.CITIES, "FAKE", "stand-in for GeoNames cities15000", "CC-BY-4.0 (real source)"),
    "job": (F.JOBS, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "hobby": (F.HOBBIES, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "food": (F.FOODS, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "grocery": (F.GROCERIES, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "chore": (F.CHORES, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "plan": (F.PLAN_NOUNS, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "object": (F.OWNED_OBJECTS, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "relation": (F.RELATIONS, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "colour": (F.COLOURS, "PROGRAM", "closed list", "n/a"),
    "weekday": (F.WEEKDAYS, "PROGRAM", "closed list", "n/a"),
    "month": (F.MONTHS, "PROGRAM", "closed list", "n/a"),
    "time": (F.TIMES, "PROGRAM", "closed list", "n/a"),
    "number_word": (F.NUMBER_WORDS, "PROGRAM", "closed list", "n/a"),
    "ordinal": (F.ORDINALS, "PROGRAM", "closed list", "n/a"),
    "topic": (F.TOPICS, "FAKE", "stand-in for teacher-written everyday topics", "Apache-2.0 (teacher)"),
    "entity_kind": (sorted(F.ENTITY_KINDS), "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "avoid_word": (F.AVOID_WORDS, "FAKE", "stand-in for a teacher-written list", "Apache-2.0 (teacher)"),
    "req_noun": (F.WORD_NOUNS, "FAKE", "stand-in for the controlled word list", "computed (real source)"),
    "req_verb": (F.WORD_VERBS, "FAKE", "stand-in for the controlled word list", "computed (real source)"),
    "req_adj": (F.WORD_ADJS, "FAKE", "stand-in for the controlled word list", "computed (real source)"),
}
# value types whose values may be replaced by a nonce string (SPEC section 4, share 25% default)
NONCE_TYPES = {"name", "pet_name"}
NONCE_SHARE = 0.25
CAP_TYPES = {"name", "assistant_name", "pet_name", "city", "weekday", "month"}


class Pool:
    def __init__(self, vtype, values, provenance, source, license_, dropped):
        self.vtype, self.values, self.provenance = vtype, values, provenance
        self.source, self.license, self.dropped = source, license_, dropped
        self.sha = hashlib.sha256(json.dumps(values).encode()).hexdigest()[:12]

    @property
    def ref(self):
        return f"{self.vtype}@{self.sha}:{self.provenance}"


def _load():
    out = {}
    for vtype, (vals, prov, src, lic) in _DEFS.items():
        keep = [v for v in vals if heldout.pool_ok(v)]
        drop = [v for v in vals if v not in keep]
        out[vtype] = Pool(vtype, keep, prov, src, lic, drop)
    return out


POOLS = _load()


def pool(vtype):
    return POOLS[vtype]


def features(vtype, value):
    """article (a/an for common nouns, '' for names and closed lists), number, capitalized."""
    cap = value[:1].isupper()
    if cap or vtype in ("weekday", "month", "time", "colour", "number_word", "ordinal"):
        art = ""
    else:
        art = "an" if value[:1].lower() in "aeiou" else "a"
    plural = vtype in ("grocery",) and value.endswith("s") or value in ("curtains", "boots", "headphones")
    return {"article": art, "number": "pl" if plural else "sg", "cap": cap}


# ---- nonce values -------------------------------------------------------------------------------------------------
_ONSETS = list("bdfgklmnprstvz") + ["br", "dr", "gr", "kl", "pl", "tr", "st", "sk"]
_VOWELS = list("aeiou") + ["ai", "ou"]
_CODAS = ["", "", "", "n", "l", "r", "s", "k", "m"]
_ENGLISHISH = set(w.lower() for p in POOLS.values() for v in p.values for w in v.split())
_ENGLISHISH |= {"banana", "salami", "tomato", "potato", "solo", "memo", "demo", "bravo", "tuba", "koala", "sofa",
                "tuna", "lima", "soda", "diva", "pasta", "polka", "karma", "sauna", "puma", "lama", "piano"}


def nonce(rng, cap=True):
    """pronounceable CV(C)CV(C) string that is not an English word from our lists and passes the gate."""
    for _ in range(100):
        w = (rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
             + rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS))
        if 4 <= len(w) <= 9 and w not in _ENGLISHISH and heldout.pool_ok(w):
            return w.capitalize() if cap else w
    raise RuntimeError("nonce generator exhausted")


def draw(rng, vtype, used, allow_nonce=True, nonce_share=NONCE_SHARE):
    """one value of vtype not in used (case-insensitive); returns (value, is_nonce). Adds it to used."""
    low = {u.lower() for u in used}
    if allow_nonce and vtype in NONCE_TYPES and rng.random() < nonce_share:
        for _ in range(20):
            v = nonce(rng)
            if v.lower() not in low:
                used.add(v)
                return v, True
    cands = [v for v in POOLS[vtype].values if v.lower() not in low]
    if not cands:
        raise ValueError("pool exhausted: " + vtype)
    v = rng.choice(cands)
    used.add(v)
    return v, False


def provenance_refs(vtypes):
    return sorted({POOLS[t].ref for t in vtypes})


def any_fake(refs):
    return any(r.endswith(":FAKE") for r in refs)


def seeded(*parts):
    """a Random seeded from a string of parts, independent of PYTHONHASHSEED."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))
