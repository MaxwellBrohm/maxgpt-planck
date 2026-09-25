"""FAKE line banks (SPEC section 4, "bank lines"): markers, openings, closings, social lines, rule phrasings,
return lines, list lines, lookup lines, recap requests and system texts, plus the key catalogue (banks_keys.py).
Claude wrote every line here, so each bank carries provenance "FAKE" and admit.py refuses records that used one.
Line ids are "<bank>.<index>" and stay stable; the bank hash goes into every skeleton's provenance."""
import hashlib
import json
import re

from banks_keys import KEYS, pronouns

PROVENANCE = "FAKE"

# correction markers used in training; none of E004's eval markers (heldout.MARKER_TERMS) may appear here
MARKERS = ["Actually, ", "Wait, ", "Sorry, ", "Scratch that, ", "My mistake, ", "No wait, ", "I got that wrong, "]
ERR_FIX_MARKERS = ["No, ", "That's not right, ", "Not quite, "]

BANKS = {
    "open.greet": ["Hi!", "Hello there.", "Hey, how's it going?", "Good morning!", "Hi, hope you're well."],
    "open.topic": ["Hi, can I ask you something about {t}?", "Hey, I could use some help with {t}.",
                   "Hello! I've been thinking about {t}.", "Hi there. So, {t}.", "Morning! Quick question about {t}."],
    "close.goodbye": ["Thanks, that's all for now. Bye!", "Okay, I'll let you go. Talk later!", "Great, bye for now.",
                      "That helps a lot. See you!", "Alright, I'm off. Thanks again."],
    "social.how_are_you": ["How are you doing today?", "How's your day going?", "And how are things with you?"],
    "social.thanks": ["Thanks, that really helps.", "Thank you, that's useful.", "Cheers, good to know."],
    "social.who_are_you": ["Who am I talking to, by the way?", "What's your name?", "Sorry, who are you again?"],
    "rule.max_words": ["From now on, keep your replies to {N} words or fewer.",
                       "Can you answer in {N} words or less from now on?"],
    "rule.one_sentence": ["Please reply in just one sentence from now on.",
                          "Keep every answer to a single sentence, okay?"],
    "rule.end_question": ["Could you end every reply with a question from now on?",
                          "Please finish each answer with a question."],
    "rule.call_user": ["Please call me {X} from now on.", "From now on, call me {X}."],
    "rule.avoid_word": ["Please stop using the word {W}.", "Can you not say {W} anymore?"],
    "rule.start_name": ["I'm {n}. Can you start every reply with my name?",
                        "My name's {n}, and please begin each reply with it."],
    "rule.override": ["Change of plan. ", "Forget that last rule. ", "New rule instead. "],
    "return": ["Anyway, back to what I was asking before.", "Sorry, I got sidetracked. Where were we?",
               "Okay, back to the first thing. Any more ideas?",
               "Let's go back to what we were talking about earlier."],
    "swap.q": ["What about you? What's your {lab}?", "And you, what's your {lab}?"],
    "identity.q.self": ["And what's your name, by the way?", "Remind me, what's your name?"],
    "identity.q.user": ["Do you know my name?", "What's my name, then?"],
    "recap": ["Can you sum up what I've told you so far?", "Quick check, what do you remember from what I said?"],
    "list.init.grocery": ["I need to buy {items}, that's my {L}.", "My {L} so far is {items}."],
    "list.init.city": ["My {L} goes {items}, in that order.", "On my trip I'm visiting {items}, that's the {L}."],
    "list.init.name": ["I'm inviting {items} to dinner, that's the {L}.", "The {L} is {items}."],
    "list.init.chore": ["Today my {L} is {items}, in that order.", "My {L} for today is {items}."],
    "list.add": ["Oh, add {v} to my {L} too.", "Put {v} on my {L} as well."],
    "list.remove": ["Actually, take {v} off my {L}.", "Cross {v} off the {L}."],
    "list.move_first": ["Move {v} to the top of my {L}.", "Put {v} first on the {L} instead."],
    "list.move_last": ["Put {v} at the end of my {L} instead.", "Move {v} to the bottom of the {L}."],
    "list.q.first": ["What's first on my {L}?", "Which one did I put first on the {L}?"],
    "list.q.last": ["What's the last thing on my {L}?", "Which one is at the end of my {L} now?"],
    "list.q.ordinal": ["What's {ord} on my {L}?", "Which one is {ord} on the {L} now?"],
    "list.q.count": ["How many things are on my {L} now?", "How many are left on the {L}?"],
    "list.q.contains": ["Is {v} still on my {L}?", "Did {v} stay on the {L}?"],
    "list.q.other": ["I've done {v} now. What was the other one on my {L}?", "{v} is sorted. What's left on the {L}?"],
    "lookup.q.weekday": ["Which day is the {e} open?", "Can you check what day the {e} opens?"],
    "lookup.q.city": ["Where is the {e}?", "Can you find out which city the {e} is in?"],
    "lookup.q.month": ["Which month is the {e}?", "Can you check when the {e} happens?"],
    "lookup.q.colour": ["What colour is the {e}?", "Can you look up the colour of the {e}?"],
    "lookup.cf": ["I'm pretty sure the {e} {pred}. Can you check?", "I think the {e} {pred}, but can you look it up?"],
    "lookup.ctx": ["A friend told me the {e} {pred}.", "I read somewhere that the {e} {pred}."],
    "system.plain": ["You are {A}, a friendly assistant.", "Your name is {A}. You chat with people and help out."],
    "system.lookup": ["You are {A}, a helpful assistant. To look something up, write it between lookup tags.",
                      "Your name is {A}. You can look facts up with a lookup tag when you need to."],
}
LIST_NAMES = {"grocery": "shopping list", "city": "route", "name": "guest list", "chore": "chore list"}
LOOKUP_PRED = {"weekday": "opens on {v}", "city": "is in {v}", "month": "is in {v}", "colour": "is {v}"}


def lines(bank):
    return BANKS[bank]


def line_id(bank, i):
    return f"{bank}.{i}"


_HOLE = re.compile(r"\{(\w+)\}")


def fill(template, **holes):
    """fill {holes}; a missing or None hole raises KeyError. A leading empty {M} capitalizes the result."""
    def rep(m):
        v = holes.get(m.group(1))
        if v is None:
            raise KeyError(m.group(1))
        return v
    out = _HOLE.sub(rep, template)
    if template.startswith("{M}") and not holes.get("M"):
        out = out[:1].upper() + out[1:]
    return out


def can_fill(template, **holes):
    try:
        fill(template, **holes)
        return True
    except KeyError:
        return False


def av(value, feats):
    return (feats["article"] + " " + value) if feats.get("article") else value


def pools_article(value, vtype):
    import pools
    return pools.features(vtype, value)["article"]


def join_items(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def key_holes(key, noun, value, feats, old=None, marker=None):
    """hole values for a key line of one slot instance."""
    p, p_obj = pronouns(key, noun)
    return dict(v=value, av=av(value, feats), o=noun, old=old, M=marker if marker is not None else "",
                p=p, p_obj=p_obj)


def refs(key, noun):
    """referring expressions for a key instance: full and head forms (strings a guided turn must include)."""
    if KEYS[key]["noun"] is None:
        return {"full": "my " + KEYS[key]["label"], "head": "my " + KEYS[key]["label"]}
    return {"full": "my " + noun, "head": "the " + noun}


def label(key, noun):
    return KEYS[key]["label"].replace("{o}", noun or "")


def bank_hash():
    blob = json.dumps({"banks": BANKS, "keys": KEYS, "markers": MARKERS, "err": ERR_FIX_MARKERS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def bank_ref():
    return f"banks@{bank_hash()}:{PROVENANCE}"


def all_lines():
    """every bank line (for the held-out gate test): (id, template)."""
    out = [(line_id(b, i), t) for b, ls in BANKS.items() for i, t in enumerate(ls)]
    for k, d in KEYS.items():
        for role in ("plant", "bait", "twin"):
            out += [(f"key.{k}.{role}.{i}", t) for i, t in enumerate(d[role])]
        for role in ("query", "corr"):
            out += [(f"key.{k}.{role}.{i}", t) for i, (_, t) in enumerate(d[role])]
    out += [(f"marker.{i}", m) for i, m in enumerate(MARKERS + ERR_FIX_MARKERS)]
    return out
