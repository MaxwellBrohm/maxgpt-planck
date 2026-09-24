"""E004 training pools: the four value types, shared ellipsis corrections, acknowledgements and the seven
training markers (notes (a)). Per-type pools live in pools_train_a.py (weekday, colour) and pools_train_b.py
(month, city); fillers in fillers_train.py. Everything here is TRAINING-only vocabulary: the eval builders use
their own pools, and heldout_e004.py lists what training must never contain."""
from pools_train_a import DAY, COLOUR
from pools_train_b import MONTH, CITY
from fillers_train import FILLERS_TRAIN

# ellipsis corrections: no reference to the object at all; shared by every value type, plus one per type
ELL_COMMON = ["Make that {v}.", "Scratch that, {v}.", "Let's say {v}.", "{v}, I mean.", "Change that to {v}."]
REV_ELL_COMMON = ["Back to {v}, as first planned.", "Go back to {v}, like before."]

POOLS = {"weekday": DAY, "colour": COLOUR, "month": MONTH, "city": CITY}
for _P in POOLS.values():
    _P["ell"] = ELL_COMMON + _P["ell"]
    _P["rev_ell"] = list(REV_ELL_COMMON)

# assistant acknowledgements after a statement (shared; {v} optional)
ACK_ORIG = ["Got it, {v}.", "Noted, {v} for that.", "Okay, I'll remember {v}.", "Sounds good.",
            "Thanks for letting me know.", "Great, {v} it is.", "Understood.", "Perfect, {v}."]
ACK_CORR = ["Okay, updated to {v}.", "Got it, {v} then.", "No problem, {v}.", "Thanks, switching to {v}.",
            "Understood, {v}.", "All right, changed.", "Noted, {v}."]
ACK_INCID = ["That sounds nice.", "Good to know.", "Fair enough.", "How fun.", "I see."]
ACKS = {"orig": ACK_ORIG, "corr": ACK_CORR, "rev": ACK_CORR, "incid": ACK_INCID}

# training markers (notes (a)); the eval uses a disjoint set (heldout_e004.EVAL_MARKERS)
MARKERS = ["actually", "wait", "change of plans", "sorry, I meant", "update:", "oh,", "never mind,"]
MARKER_FMT = {"actually": "Actually, {s}", "wait": "Wait, {s}", "change of plans": "Change of plans: {s}",
              "sorry, I meant": "Sorry, I meant: {s}", "update:": "Update: {s}", "oh,": "Oh, {s}",
              "never mind,": "Never mind, {s}"}
# on statements that are NOT corrections only the four markers that do not read as retracting the previous
# statement are used ("Sorry, I meant: <B's first statement>" would read as correcting A, so the gold for A
# would be arguable); corrections use all seven
NONCORR_MARKERS = ["actually", "wait", "oh,", "update:"]
P_NO_MARKER_CORR = 0.35            # corrections (and changes back): none 35%, else one of the 7
P_MARKER_B_ORIG = 0.30             # the second object's first statement
P_MARKER_OTHER = 0.10              # the first object's original and incidentals

# reference form of a correction; pronoun/ellipsis only directly after the same object's exchange.
# notes (a) set 35/25/20/20; moved 5 points each to 30/20/25/25 before any model run (notes DEVIATIONS 1):
# at 35/25/20/20 oracle O6 (latest statement naming the asked object) scored 0.707 on the stream (limit 0.70)
REF_FORMS = ["full", "head", "pron", "ell"]
REF_TARGET = {"full": 0.30, "head": 0.20, "pron": 0.25, "ell": 0.25}

CAP_VALUES = set(DAY["values"]) | set(MONTH["values"]) | set(CITY["values"])


def cap(s):
    return s[:1].upper() + s[1:]


def decap(s):
    """lowercase the first letter unless the first word is I/I'm/... or a capitalized value."""
    w = s.split(" ", 1)[0].rstrip(",.")
    if w == "I" or w.startswith("I'") or w in CAP_VALUES:
        return s
    return s[:1].lower() + s[1:]


def with_marker(s, marker):
    return s if marker is None else MARKER_FMT[marker].format(s=decap(s))


def pool_sizes():
    """{(vtype, role): n} for every per-type pool, plus shared pools under vtype '*'."""
    out = {}
    for vt, P in POOLS.items():
        for role in ("orig", "corr", "pron", "ell", "rev", "rev_pron", "rev_ell", "incid", "ask", "ans",
                     "ans_upd"):
            out[(vt, role)] = len(P[role])
    for role, pool in (("ack_orig", ACK_ORIG), ("ack_corr", ACK_CORR), ("ack_incid", ACK_INCID),
                       ("filler", FILLERS_TRAIN), ("marker", MARKERS)):
        out[("*", role)] = len(pool)
    return out
