"""Build context shared by the event planners: slot creation (distinct values, <= 2 event slots per value type),
user-turn claims (exact bank line with p_exact, else guided), assistant reply specs, leak windows, and a random
position picker with gap constraints. Positions are USER-TURN indices k (0..T-1); skeleton.py maps them to global
turn indices at assembly. Fail means "this layout does not fit"; the caller retries with new draws."""
from collections import Counter

import banks as B
import heldout
import pools as P
from banks_keys import KEYS


class Fail(Exception):
    pass


class Ctx:
    def __init__(self, rng, T, p_exact, register, lookup, card_name):
        self.rng, self.T, self.p_exact, self.register, self.lookup = rng, T, p_exact, register, lookup
        self.card_name = card_name
        self.free = set(range(T))
        self.used = {card_name}
        self.used_nouns = set()
        self.slots, self.persons = {}, {}
        self.type_count = Counter()
        self.uturn, self.aturn = {}, {}
        self.events, self.windows = [], []
        self.user_name_mentioned = False
        self.reserved_keys = set()

    # ---- slots ----------------------------------------------------------------------------------------------------
    def new_slot(self, key, count=True, vtype_override=None, reserved_ok=False):
        k = KEYS[key]
        vtype = vtype_override or k["vtype"]
        if count and self.type_count[vtype] >= heldout.MAX_SLOTS_PER_TYPE:
            raise Fail("type cap " + vtype)
        if not k["multi"] and ((key in self.reserved_keys and not reserved_ok)
                               or any(s["key"] == key and s["owner"] == "user" for s in self.slots.values())):
            raise Fail("singleton key used or reserved " + key)
        noun, owner = None, "user"
        if k["noun"]:
            noun = self.draw_noun(k["noun"])
        if k["owner"] == "person":
            owner = self.person_for(noun)
        value, is_nonce = P.draw(self.rng, vtype, self.used)
        sid = f"s{len(self.slots) + 1}"
        self.slots[sid] = {"type": vtype, "value": value, "nonce": is_nonce, "owner": owner, "key": key,
                           "noun": noun, "features": P.features(vtype, value), "counted": count}
        if key == "person_name":
            self.persons[owner]["name"] = sid
        if key == "user_name":
            self.user_name_mentioned = True
        if count:
            self.type_count[vtype] += 1
        return sid

    def item_slot(self, vtype, value, key="list_item"):
        """an uncounted slot for a value that is not a planted fact (list items, lookup values), so that R2 can
        re-draw it and provenance covers it."""
        sid = f"s{len(self.slots) + 1}"
        self.slots[sid] = {"type": vtype, "value": value, "nonce": False, "owner": "user", "key": key,
                           "noun": None, "features": P.features(vtype, value), "counted": False}
        return sid

    def draw_noun(self, pool_name):
        cands = [v for v in P.pool(pool_name).values if v not in self.used_nouns]
        noun = self.rng.choice(cands)
        self.used_nouns.add(noun)
        return noun

    def person_for(self, relation):
        pid = f"p{len(self.persons) + 1}"
        self.persons[pid] = {"relation": relation, "name": None}
        return pid

    def extra_value(self, vtype):
        """a further value of vtype (a stale or twin value): distinct in the conversation, not a slot."""
        return P.draw(self.rng, vtype, self.used, allow_nonce=False)[0]

    def holes(self, sid, old=None, marker=None):
        s = self.slots[sid]
        return B.key_holes(s["key"], s["noun"], s["value"], s["features"], old=old, marker=marker)

    def ref(self, sid, form):
        s = self.slots[sid]
        return B.refs(s["key"], s["noun"])[form]

    # ---- turns ----------------------------------------------------------------------------------------------------
    def claim(self, k):
        if k not in self.free:
            raise Fail("turn taken")
        self.free.discard(k)

    def user(self, k, eid, role, options, holes, intent, must_include, exact=None):
        """options: [(line_id, template)]; exact line with p_exact (or forced by exact=True/False)."""
        self.claim(k)
        usable = [(i, t) for i, t in options if B.can_fill(t, **holes)]
        use_exact = (self.rng.random() < self.p_exact) if exact is None else exact
        spec = {"mode": "guided", "text": None, "bank_ref": None, "intent": intent,
                "must_include": list(must_include), "must_exclude": [], "events": [eid], "role": role}
        if use_exact:
            if not usable:
                raise Fail("no fillable line")
            lid, t = self.rng.choice(usable)
            spec.update(mode="exact", text=B.fill(t, **holes), bank_ref=lid)
        self.uturn[k] = spec
        return spec

    def reply(self, k, eid, intent, must_include=(), must_exclude=(), mask=0, lookup=None):
        self.aturn[k] = {"intent": intent, "must_include": list(must_include), "must_exclude": list(must_exclude),
                         "mask": mask, "events": [eid], "lookup": lookup}

    def window(self, value, start, end, eid):
        """value must not appear in any turn strictly between user turns start and end that eid does not own."""
        self.windows.append({"value": value, "start": start, "end": end, "eid": eid})

    # ---- positions ------------------------------------------------------------------------------------------------
    def pick(self, gaps, first_range=None, tries=60):
        """increasing free positions p0 < p1 < ...; gaps[i] = (lo, hi) bounds on p[i+1] - p[i]."""
        free = sorted(self.free)
        starts = [p for p in free if first_range is None or first_range[0] <= p <= first_range[1]]
        for _ in range(tries):
            if not starts:
                return None
            seq = [self.rng.choice(starts)]
            ok = True
            for lo, hi in gaps:
                cands = [p for p in free if lo <= p - seq[-1] <= hi]
                if not cands:
                    ok = False
                    break
                seq.append(self.rng.choice(cands))
            if ok:
                return seq
        return None

    def event(self, kind, params, gold, turns_k):
        eid = f"e{len(self.events) + 1}"
        self.events.append({"id": eid, "kind": kind, "params": params, "gold": gold, "turns_k": turns_k})
        return eid

    def next_eid(self):
        return f"e{len(self.events) + 1}"


def key_options(key, role, form=None):
    d = KEYS[key]
    if role in ("query", "corr"):
        return [(f"key.{key}.{role}.{i}", t) for i, (f, t) in enumerate(d[role]) if form is None or f == form]
    return [(f"key.{key}.{role}.{i}", t) for i, t in enumerate(d[role])]


def bank_options(bank):
    return [(B.line_id(bank, i), t) for i, t in enumerate(B.lines(bank))]


def keys_of_type(vtype, multi_only=False):
    return [k for k, d in KEYS.items() if d["vtype"] == vtype and (d["multi"] or not multi_only)]


def pick_weighted(rng, table):
    """table: {name: weight}; returns a name."""
    names = list(table)
    return rng.choices(names, weights=[table[n] for n in names])[0]
