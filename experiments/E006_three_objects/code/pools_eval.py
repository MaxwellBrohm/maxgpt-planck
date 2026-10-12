"""E004 eval pools, shared parts and assembly (notes (b)). POOLS_E: the four training value types with eval-only
objects (pools_eval_a/b.py); H6_POOLS: the four newtype cells (pools_eval_h6.py). Eval acknowledgements, the
shared eval ellipsis corrections, markers and alias names (held-out vocabulary from heldout_e004.py), and the
two filler pools: eval_fillers() = E001's pool (read-only import, no bytecode written) minus every filler that
names a training object ("Any advice for a first job interview?", per the step-2 note) or holds a value
("... the green fades ...") plus EVAL_FILLERS_NEW;
H7_FILLERS for H7. Everything here is EVAL-only vocabulary."""
import os
import re
import sys

import heldout_e004 as H
from pools_eval_a import WEEKDAY_E, COLOUR_E
from pools_eval_b import MONTH_E, CITY_E
from pools_eval_h6 import H6_POOLS
from fillers_eval import EVAL_FILLERS_NEW, H7_FILLERS
from pools_train import POOLS as TRAIN_POOLS

POOLS_E = {"weekday": WEEKDAY_E, "colour": COLOUR_E, "month": MONTH_E, "city": CITY_E}
for _k, _P in POOLS_E.items():
    _P["vtype"] = _k
TRAIN_VTYPES = ["weekday", "colour", "month", "city"]
H6_CELLS = ["weekday_delivery", "colour_decor", "sport_activity", "number_place"]
ALL_EVAL_POOLS = dict(POOLS_E, **H6_POOLS)

# ellipsis corrections shared by every eval pool (no reference to the object at all)
ELL_COMMON_EVAL = ["{v} instead.", "{v}, sorry.", "Let's go with {v}.", "Forget that, {v}."]

# assistant acknowledgements after a statement ({v} optional); none equals a training acknowledgement
ACK_ORIG_E = ["Alright, I have {v} down.", "Sure, {v} is noted.", "I'll keep that in mind.", "Okay, {v} then.",
              "Thanks, that's written down.", "Sure thing.", "Good, {v} works."]
ACK_CORR_E = ["Alright, now {v}.", "Okay, I've changed that to {v}.", "Sure, {v} from here on.",
              "Done, I have {v} down now.", "Thanks for the heads-up, {v}.", "Revised on my end.", "Okay, adjusted."]
ACKS_E = {"orig": ACK_ORIG_E, "corr": ACK_CORR_E}

HONORIFIC_CYCLE = ["Mr.", "Ms.", "Dr.", "Mrs."]
ALIASES = [f"{HONORIFIC_CYCLE[i % 4]} {s}" for i, s in enumerate(H.ALIAS_SURNAMES)]

VALUE_POOLS = {"weekday": WEEKDAY_E["values"], "colour": COLOUR_E["values"], "month": MONTH_E["values"],
               "city": CITY_E["values"], "sport": H.SPORT, "number": H.NUMBER}
ALL_VALUES = [v for vs in VALUE_POOLS.values() for v in vs]
CAP_VALUES = {v for vs in VALUE_POOLS.values() for v in vs if v[:1].isupper()}

TRAIN_OBJECT_WORDS = sorted({w for P in TRAIN_POOLS.values() for ph, h in P["objects"] for w in ph.split() + [h]})


def cap(s):
    return s[:1].upper() + s[1:]


def decap(s):
    """lowercase the first letter unless the first word is I/I'm/..., a capitalized value, an honorific or a
    digit."""
    raw = s.split(" ", 1)[0]
    w = raw.rstrip(",.")
    if w == "I" or w.startswith("I'") or w in CAP_VALUES or raw in H.HONORIFICS or w[:1].isdigit():
        return s
    return s[:1].lower() + s[1:]


def with_marker(s, marker):
    return s if marker is None else H.EVAL_MARKER_FMT[marker].format(s=decap(s))


def _e001_fillers():
    sys.dont_write_bytecode = True
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "E001_battery_and_probes", "code")
    sys.path.insert(0, os.path.abspath(d))
    import items as I1
    import items_new as I1n
    sys.path.pop(0)
    return list(I1.DISTRACTORS) + list(I1n.EXTRA_DISTRACTORS)


def names_training_object(text):
    """True if text names a training object phrase (a head noun alone, such as "practice", is allowed)."""
    t = text.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(ph) + r"s?(?![a-z])", t)
               for P in TRAIN_POOLS.values() for ph, _ in P["objects"])


_CACHE = {}


def has_value(text):
    """any value of any pool (training, eval or held-out) or any digit."""
    from text_e004 import values_in
    return bool(values_in(text, ALL_VALUES)) or re.search(r"\d", text) is not None


def eval_fillers():
    """default eval fillers: E001's pool without fillers that name a training object or contain a value (E001's
    "Why do leaves change color in autumn?" says "green"), plus EVAL_FILLERS_NEW."""
    if "default" not in _CACHE:
        bad = lambda f: names_training_object(f[0] + " " + f[1]) or has_value(f[0] + " " + f[1])
        _CACHE["default"] = [f for f in _e001_fillers() if not bad(f)] + list(EVAL_FILLERS_NEW)
        _CACHE["e001_dropped"] = [f for f in _e001_fillers() if bad(f)]
    return _CACHE["default"]


def filler_pool(kind):
    return list(H7_FILLERS) if kind == "h7" else eval_fillers()
