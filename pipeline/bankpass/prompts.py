"""Zero-shot bank prompts and the per-call decode spec (BANKPASS s0, s2a, s2b, s3). Claude wrote this wording: it is
prompt, never training text, and it holds NO example line (RC-12's sealed templates are Claude-written, so examples
would pull the banks toward that phrasing). When a class needs an example, it is a mined human sentence (mine.py),
shown with its doc id. Every prompt is hashed into the call record; a wording change is a new PROMPT_VERSION.

Output contract: exactly n lines, one per value set, in order, then END on its own line. Verbalized classes end each
line with " | " and a likelihood word (common, uncommon, rare), parsed off. The regex is built from
teachers/decode.noend, so no line can hold END (labels-v2's rule), and a line that must hold a value holds it
literally."""
from teachers import decode as D

from bankpass import store

PROMPT_VERSION = "bp-prompts-v0"
LIKELIHOOD = ("common", "uncommon", "rare")
SEP = {"ministral-3-8b": "\\n+"}                 # D3 probe: Ministral puts a blank line between lines
_RULES = ("Write plain everyday English, the way a real person types in a chat. No numbering, no quotation marks, "
          "no dashes, no digits (write numbers as words), no emojis, no lists inside a line.")
FORMS = {"full": "Refer to the {thing} as 'my' followed by its word from that line.",
         "head": "Refer to the {thing} as 'the' followed by its word from that line.",
         "pronoun": "Start the line with the pronoun given for that line, meaning the {thing}, and do not name it.",
         "ellipsis": "Give just the new value and what it replaces, without naming the {thing}."}
NOUN_WORD = {"pet_kind": "pet", "plan": "plan or event", "object": "thing", "relation": "person"}
KEY_ACT = {
    "plant": "In each line the person tells the assistant, as a plain statement and not a question, {lab}: the value "
             "given for that line.",
    "corr": "In each line the person corrects what they said a moment ago about {lab}: the new value, not the old "
            "one. Do not open with a word like sorry or actually; that part is added separately.",
    "query": "In each line the person asks the assistant to remind them of {lab}, which they said earlier in the "
             "chat. The line must not contain the answer.",
    "bait": "In each line the person asks whether the assistant remembers {lab}. The line must not contain the answer.",
    "twin": "In each line the person mentions the value given for that line, but makes clear that it is not {lab}.",
}
CLASS_ASK = {
    "marker.fix": "short ways a person starts a sentence when they fix something they themselves just said in a chat. "
                  "Each is one to four words and ends with a comma",
    "marker.err": "short ways a person starts a sentence when they point out that the assistant just said something "
                  "wrong. Each is one to four words and ends with a comma",
    "open.greet": "ways a person greets a chat assistant at the very start of a chat, with nothing else in the line",
    "open.topic": "ways a person starts a chat with an assistant by greeting it and bringing up the topic given for "
                  "that line, without saying anything about themselves",
    "close.goodbye": "ways a person ends a chat with an assistant and says goodbye, without asking a question",
    "social.how_are_you": "ways a person asks a chat assistant how it is doing",
    "social.thanks": "short ways a person thanks a chat assistant for an answer",
    "social.who_are_you": "ways a person asks a chat assistant who it is or what its name is",
    "identity.q.self": "ways a person asks a chat assistant for its name",
    "identity.q.user": "ways a person asks a chat assistant whether it knows their name, without saying the name",
    "recap": "ways a person asks an assistant to sum up what they have told it so far in the chat",
    "return": "ways a person steers a chat back to what they were asking about before a side topic",
    "system.plain": "short system texts that tell an assistant only that its name is the name given for that line and "
                    "that it is an assistant. Say nothing else about it",
    "system.lookup": "short system texts that tell an assistant only that its name is the name given for that line, "
                     "that it is an assistant, and that it can look something up by writing the query between "
                     "lookup tags. Say nothing else about it",
}


HOLE_WORD = {"t": "topic:", "A": "name:"}


def voice(seed):
    """the voice seed line: one persona line (Nemotron-Personas-USA once downloaded), an age band, a length band."""
    if not seed:
        return ""
    return (f" Write as this person would: {seed.get('persona', 'an ordinary adult')}, {seed.get('age', 'adult')}, "
            f"in {seed.get('length', 'short')} lines.")


def key_label(key):
    """the prompt's words for a key's fact: "their home city", or "the day of the plan or event given for that line"."""
    from banks_keys import KEYS
    d = KEYS[key]
    if not d["noun"]:
        return "their " + d["label"]
    return f"the {d['label'].replace('{o}', '').strip()} of the {NOUN_WORD[d['noun']]} given for that line"


def key_prompt(n, key, role, form, fills, seed=None):
    """one key call: one key, one role, one form, n value sets (s3 K)."""
    sets = []
    for i, f in enumerate(fills, 1):
        bits = [f"value {f['v']}"] if role in ("plant", "twin", "corr") else []
        if role == "corr":
            bits.append(f"old value {f['old']}")
        if f.get("o"):
            bits.append(f"the {f['o']}")
        if form == "pronoun" and f.get("p"):
            bits.append(f"pronoun {f['p']}")
        sets.append(f"Line {i}: " + (", ".join(bits) if bits else "no value"))
    from banks_keys import KEYS
    thing = NOUN_WORD.get(KEYS[key]["noun"], "thing")
    ask = KEY_ACT[role].format(lab=key_label(key))
    form_s = (" " + FORMS[form].format(thing=thing)) if form else ""
    return (f"Write {n} different things a person might say to a chat assistant at any point in a conversation, one "
            f"line each. {ask}{form_s} Details for each line:\n" + "\n".join(sets) + f"\n\n{_RULES}{voice(seed)}"
            "\nAfter the last line write END on its own line.")


def class_prompt(bank, n, fills=None, mined=None, seed=None):
    """a line bank call other than keys: openings, closings, markers, social lines, system texts."""
    ask = CLASS_ASK[bank]
    verb = bank.startswith(("marker.", "open."))
    tail = (" After each line write ' | ' and how common it is: common, uncommon or rare. Include uncommon and rare "
            "ones too.") if verb else ""
    det = ""
    if fills:
        det = " Details for each line:\n" + "\n".join(f"Line {i}: " + ", ".join(f"{HOLE_WORD.get(k, k)} {v}"
                                                                         for k, v in f.items())
                                                       for i, f in enumerate(fills, 1)) + "\n"
    ex = ""
    if mined:
        ex = "\nA sentence a real person wrote, for the tone only (do not copy it): " + mined["text"] + "\n"
    return (f"Write {n} different {ask}.{det}{tail}{ex}\n{_RULES}{voice(seed)}\n"
            "After the last line write END on its own line.")


# W3 (s2b "digit and markdown tokens banned"): a bank line's characters exclude digits, markdown marks and straight
# double quotes, in decode.noend()'s no-END form; judge answers keep decode.noend()
_X = "\\n0-9*#`\"_"
BANK_UNIT = f"(?:[^E{_X}]|E[^EN{_X}]|EN[^DE{_X}])"


def bank_noend():
    return f"(?:{BANK_UNIT}+(?:EN?)?|EN?)"


def line_regex(literal=None, verbalized=False, question=False):
    """one output line: a no-END line, holding `literal` once when given, then the likelihood tail if verbalized.
    question (W3 amendment 4, key queries and baits, s3 K "end in ?"): the line ends with a question mark."""
    u = BANK_UNIT
    if question:
        body = (f"{u}*{D.esc(literal)}" if literal else "") + f"{u}*(?:EN?)?\\?"
    else:
        body = bank_noend() if literal is None else f"{u}*{D.esc(literal)}{u}*(?:EN?)?"
    return body + (" \\| (?:" + "|".join(LIKELIHOOD) + ")" if verbalized else "")


def decode_spec(model, literals, verbalized=False, banned=(), question=False):
    """the s2b decode spec for one call: the structured regex, the dash ban, digits and markdown banned by the
    regex classes left to the checker, and the values a query must not say (vLLM bad_words)."""
    sep = SEP.get(model, "\\n")
    regex = sep.join([line_regex(lit, verbalized, question) for lit in literals] + ["END"])
    spec = {"structured": "bank-lines-v1", "regex": regex, "ban": "dash", "bad_words": sorted(set(banned)),
            "n": len(literals), "literals": list(literals), "verbalized": verbalized, "sep": sep}
    spec["sha256"] = store.sha256_text(repr(sorted(spec.items())))
    return spec


def parse_lines(text, n, verbalized=False):
    """-> (lines, likelihoods, problem). The output must be n non-empty lines then END (blank lines between are
    tolerated, as the Ministral separator allows); anything else is a problem and the call's lines are not used."""
    rows = [r.strip() for r in (text or "").replace("\r", "").split("\n") if r.strip()]
    if not rows or rows[-1] != "END":
        return [], [], "NO_END"
    rows = rows[:-1]
    if len(rows) != n:
        return [], [], f"LINES_{len(rows)}_OF_{n}"
    lines, likes = [], []
    for r in rows:
        if verbalized:
            body, _, like = r.rpartition(" | ")
            if like not in LIKELIHOOD or not body.strip():
                return [], [], "NO_LIKELIHOOD"
            lines.append(body.strip())
            likes.append(like)
        else:
            lines.append(r)
            likes.append(None)
    return lines, likes, None


POOL_ASK = {"job": "jobs people do for a living", "hobby": "hobbies people do in their free time",
            "food": "foods a person could name as a favourite", "grocery": "things people buy at a grocery shop",
            "chore": "household chores", "plan": "kinds of plans or events a person puts on their calendar",
            "object": "everyday things a person owns", "pet_kind": "kinds of animals people keep as pets",
            "pet_name": "names people give their pets", "assistant_name": "short first names for a chat assistant",
            "relation": "words for people in someone's family or social life, such as relatives and friends",
            "entity_kind": "kinds of places or events a person might look up"}


def pool_prompt(vtype, n, seed_words, seed=None):
    """s3 P: one call per type, seeded with three word-list nouns (corpus words, not Claude categories)."""
    return (f"List {n} different {POOL_ASK[vtype]}, one per line, each one to four words, in lowercase unless it is a "
            f"name. Let these words nudge you toward variety: {', '.join(seed_words)}.\n{_RULES}{voice(seed)}\n"
            "After the last line write END on its own line.")


def topic_prompt(n, seed_words, seed=None):
    """s3 T: everyday things to chat with an assistant about, as short noun phrases."""
    return (f"List {n} different everyday things a person might chat with an assistant about, each as a short noun "
            f"phrase of four to eight words. Let these words nudge you toward variety: {', '.join(seed_words)}. "
            "After each phrase write ' | ' and how common a chat topic it is: common, uncommon or rare."
            f"\n{_RULES}{voice(seed)}\nAfter the last line write END on its own line.")


def prompt_for(call):
    """the prompt of one planned call."""
    cls = call["class"]
    if cls == "K":
        _, key, role = call["bank"].split(".")
        return key_prompt(call["n"], key, role, call.get("form"), call["fills"], call.get("voice"))
    if cls == "P":
        return pool_prompt(call["bank"].split(".", 1)[1], call["n"], call["seed_words"], call.get("voice"))
    if cls == "T":
        return topic_prompt(call["n"], call["seed_words"], call.get("voice"))
    if call["bank"] in CLASS_ASK:
        return class_prompt(call["bank"], call["n"], call.get("fills") or None, call.get("mined"), call.get("voice"))
    raise KeyError(f"no prompt written yet for {call['bank']}")
