"""W3 call inputs (BANKPASS s2a, s3): voice seeds (one Nemotron-Personas-USA line, its age band, a length band),
mined human examples (cited by doc id) and the value fills for the line banks W0 left without them (rules, lists,
lookups, swap). Every pick is seeded by the call id, so a restart re-issues byte-identical calls. Values come from the
installed pools (human_v0's GeoNames cities where installed, FAKE elsewhere): they are holes after templatizing, and a
judge reads every line on a fresh fill it never saw."""
import json
import os

import banks as B
import banks_keys as BK
import fake_data as F
import pools as P
from bankpass import store

LENGTHS = ("short", "fairly short", "medium length")
AGE_WORDS = {"10s": "aged eighteen or nineteen", "20s": "in their twenties", "30s": "in their thirties",
             "40s": "in their forties", "50s": "in their fifties", "60s": "in their sixties",
             "70s": "in their seventies", "80s+": "in their eighties or older"}
# mined classes (mine.py) -> the banks whose prompts may show one; the rest stay zero-shot with no example at all
MINED_FOR = {"open.greet": "O_greet", "open.topic": "O_greet", "marker.fix": "M_fix", "marker.err": "M_fix",
             "close.goodbye": "C_close", "social.how_are_you": "S_social", "social.thanks": "S_social",
             "social.who_are_you": "S_social", "rule.": "R_rule", "key.": "K_plant"}
NO_VOICE = ("system.",)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


class Inputs:
    """the persona seeds and mined examples (PC: ~/planck/runs/bankpass/{human_v0,mined_v0}); empty is allowed for
    tests (then no voice and no example)."""

    def __init__(self, personas=(), mined=(), persona_sha=None, mined_sha=None):
        self.personas = [p for p in personas if p.get("status") == "kept"]
        self.mined = {}
        for m in mined:
            self.mined.setdefault(m["class"], []).append(m)
        self.persona_sha, self.mined_sha = persona_sha, mined_sha

    @classmethod
    def from_root(cls, bankpass_root):
        pp = os.path.join(bankpass_root, "human_v0", "seed.persona.jsonl")
        mp = os.path.join(bankpass_root, "mined_v0", "mined_examples_v0.jsonl")
        return cls(read_jsonl(pp), read_jsonl(mp), store.sha256_file(pp), store.sha256_file(mp))

    def voice(self, call_id, bank=""):
        if not self.personas or bank.startswith(NO_VOICE):
            return None
        rng = P.seeded("bankpass-voice", call_id)
        p = rng.choice(self.personas)
        return {"persona": p["text"], "age": AGE_WORDS.get(p["features"].get("age_band"), "an adult"),
                "length": rng.choice(LENGTHS), "persona_id": p["id"], "uuid": p["features"].get("uuid")}

    def example(self, call_id, bank):
        cls = next((c for pre, c in MINED_FOR.items() if bank == pre or (pre.endswith(".") and bank.startswith(pre))),
                   None)
        if not cls or not self.mined.get(cls) or (bank.startswith("key.") and not bank.endswith(".plant")):
            return None
        m = P.seeded("bankpass-mined", call_id).choice(self.mined[cls])
        return {"text": m["text"], "doc_id": m["doc_id"], "license": m["license"], "class": cls}


def _lookup_e(rng, vt):
    kinds = sorted(k for k, attrs in F.ENTITY_KINDS.items() if any(F.ATTR_TYPES[a] == vt for a in attrs))
    return f"{P.nonce(rng)} {rng.choice(kinds).capitalize()}"


def fill(bank, rng):
    """one value set for a rule, list, lookup, swap, topic-opening or system line (None: the bank takes no fill)."""
    if bank == "rule.max_words":
        return {"N": rng.choice([w for w in P.pool("number_word").values if w not in ("one", "two")])}
    if bank in ("rule.call_user", "rule.start_name"):
        return {"X" if bank == "rule.call_user" else "n": rng.choice(P.pool("name").values)}
    if bank == "rule.avoid_word":
        return {"W": rng.choice(P.pool("avoid_word").values)}
    if bank.startswith("list.init."):
        lt = bank.rsplit(".", 1)[1]
        return {"L": B.LIST_NAMES[lt], "items": rng.sample(P.pool(lt).values, rng.choice((3, 4, 5)))}
    if bank.startswith("list."):
        lt = rng.choice(sorted(B.LIST_NAMES))
        f = {"L": B.LIST_NAMES[lt]}
        if bank in ("list.add", "list.remove", "list.move_first", "list.move_last", "list.q.contains",
                    "list.q.other"):
            f["v"] = rng.choice(P.pool(lt).values)
        if bank == "list.q.ordinal":
            f["ord"] = rng.choice([o for o in P.pool("ordinal").values if o != "first"])
        return f
    if bank.startswith("lookup.q."):
        return {"e": _lookup_e(rng, bank.rsplit(".", 1)[1])}
    if bank in ("lookup.cf", "lookup.ctx"):
        vt = rng.choice(sorted(B.LOOKUP_PRED))
        return {"e": _lookup_e(rng, vt), "pred": B.LOOKUP_PRED[vt].format(v=rng.choice(P.pool(vt).values))}
    if bank == "swap.q":
        return {"lab": rng.choice([d["label"] for d in BK.KEYS.values() if not d["noun"]])}
    if bank == "open.topic":
        return {"t": rng.choice(P.pool("topic").values)}
    if bank.startswith("system."):
        return {"A": rng.choice(P.pool("assistant_name").values)}
    return None


def fills(bank, call_id, n):
    rng = P.seeded("bankpass-fill3", call_id)
    out = [fill(bank, rng) for _ in range(n)]
    return [] if out and out[0] is None else out
