"""W3 prompts, part 1 (BANKPASS s3 R, L, U, S swap.q, N, P features): the line banks W0 left unwritten, and the label
and feature calls. Same contract as prompts.py: Claude wrote this wording (prompt, never training text), zero-shot,
NO example line; each call asks for n lines then END, and its decode regex comes from line_res() below. A wording
change is a new PROMPT_VERSION (the prompt sha256 is in every call record)."""
from teachers import decode as D

from bankpass import prompts as PR, store

PROMPT_VERSION = "bp-prompts3-v1"
LINE_ASK = {
    "rule.max_words": "ways a person asks a chat assistant to keep every reply, from now on, to at most the number of "
                      "words given for that line",
    "rule.one_sentence": "ways a person asks a chat assistant to answer in a single sentence from now on",
    "rule.end_question": "ways a person asks a chat assistant to end every reply with a question from now on",
    "rule.call_user": "ways a person asks a chat assistant to call them by the name given for that line from now on",
    "rule.avoid_word": "ways a person asks a chat assistant to stop using the word given for that line",
    "rule.start_name": "ways a person tells a chat assistant their own name, given for that line, and asks it to start "
                       "every reply with that name",
    "rule.override": "short ways a person says that what they ask next replaces a rule they gave the assistant "
                     "earlier. Each is three to six words and does not say the new rule yet",
    "list.init": "ways a person tells a chat assistant what is on their {kind}, using the list name and the items given "
                 "for that line, with the items in the given order",
    "list.add": "ways a person asks a chat assistant to add the item given for that line to their list, called by the "
                "list name given",
    "list.remove": "ways a person asks a chat assistant to take the item given for that line off their list, called by "
                   "the list name given",
    "list.move_first": "ways a person asks a chat assistant to move the item given for that line to the top of their "
                       "list, called by the list name given",
    "list.move_last": "ways a person asks a chat assistant to move the item given for that line to the very end of "
                      "their list, called by the list name given",
    "list.q.first": "ways a person asks a chat assistant what comes first on their list, called by the list name given",
    "list.q.last": "ways a person asks a chat assistant what comes last on their list, called by the list name given",
    "list.q.ordinal": "ways a person asks a chat assistant which item is in the position given for that line on their "
                      "list, called by the list name given",
    "list.q.count": "ways a person asks a chat assistant how many things are on their list now, called by the list "
                    "name given",
    "list.q.contains": "ways a person asks a chat assistant whether the item given for that line is still on their "
                       "list, called by the list name given",
    "list.q.other": "ways a person tells a chat assistant they have dealt with the item given for that line and asks "
                    "what the other one on their list was, called by the list name given",
    "lookup.q.weekday": "ways a person asks a chat assistant to look up the day of the week that matters for the place "
                        "or event given for that line (the day it opens, runs or leaves, whichever fits), without "
                        "guessing the day",
    "lookup.q.city": "ways a person asks a chat assistant to look up which city the place or event given for that line "
                     "is in, without guessing the city",
    "lookup.q.month": "ways a person asks a chat assistant to look up which month the place or event given for that "
                      "line happens in, without guessing the month",
    "lookup.q.colour": "ways a person asks a chat assistant to look up what colour the place or thing given for that "
                       "line is, without guessing the colour",
    "lookup.cf": "ways a person tells a chat assistant what they believe about the place or event given for that line, "
                 "using the fact given, and asks the assistant to check it",
    "lookup.ctx": "ways a person passes on to a chat assistant something they heard or read about the place or event "
                  "given for that line, using the fact given, as a statement that asks nothing",
    "swap.q": "ways a person turns a question back on a chat assistant and asks about its own {lab}, using those words",
}
LIST_KIND = {"grocery": ("shopping list", "the things they need to buy"),
             "city": ("route", "the cities they will visit, in order"),
             "name": ("guest list", "the people they are inviting"), "chore": ("chore list", "their chores for the day")}
HOLE_WORD = {"t": "topic:", "A": "name:", "N": "number:", "X": "name:", "W": "word:", "n": "name:", "L": "list name:",
             "v": "item:", "ord": "position:", "e": "place or event:", "pred": "fact:", "lab": "words:"}


def ask_key(bank):
    return "list.init" if bank.startswith("list.init.") else bank


def detail(f):
    bits = []
    for k, v in f.items():
        if k == "items":
            bits.append("items: " + ", ".join(v))
        elif k in HOLE_WORD:
            bits.append(f"{HOLE_WORD[k]} {v}")
    return ", ".join(bits)


def line_prompt(bank, n, fills, mined=None, seed=None):
    """a rule, list, lookup or swap call: n lines, one per value set."""
    ask = LINE_ASK[ask_key(bank)]
    if bank.startswith("list.init."):
        ask = ask.format(kind="{}, {}".format(*LIST_KIND[bank.rsplit(".", 1)[1]]))
    if "{lab}" in ask:
        ask = ask.replace("{lab}", "thing given for that line")
    det = ""
    if fills:
        det = " Details for each line:\n" + "\n".join(f"Line {i}: {detail(f)}" for i, f in enumerate(fills, 1)) + "\n"
    ex = ("\nA sentence a real person wrote, for the tone only (do not copy it): " + mined["text"] + "\n") if mined else ""
    return (f"Write {n} different {ask}, one line each.{det}{ex}\n{PR._RULES}{PR.voice(seed)}\n"
            "After the last line write END on its own line.")


# ---- labels and features (class N, pool features) -------------------------------------------------------------------
KEY_FACT = {"user_name": "the person's own first name", "home_city": "the city where the person lives",
            "job": "the person's job", "hobby": "the person's hobby", "fav_food": "the person's favourite food",
            "fav_colour": "the person's favourite colour", "pet_name": "the name of the person's {o}",
            "plan_day": "the day of the week of the person's {o}", "plan_month": "the month of the person's {o}",
            "plan_city": "the city of the person's {o}", "plan_time": "the time of day of the person's {o}",
            "item_colour": "the colour of the person's {o}", "person_name": "the name of the person's {o}",
            "person_city": "the city where the person's {o} lives", "person_job": "the job of the person's {o}"}


def label_prompt(key, n, o=None):
    fact = KEY_FACT[key].replace("{o}", o or "")
    tail = f" Every name must contain the word {o}." if o else ""
    return (f"Give {n} different short names, one to four words each, for this fact in a person's notes: {fact}. "
            f"Write them in lowercase, as a heading a person might write before the fact.{tail} One per line, no "
            "numbering.\nAfter the last line write END on its own line.")


NOTE_WAS_ASK = ("A note keeps facts about a person. When a fact changes, the note gives the new value and then, in "
                "brackets, a short word or phrase and the old value it replaced. Give {n} different such words or "
                "phrases, one to three words each, in lowercase, one per line.\nAfter the last line write END on its "
                "own line.")


def notewas_prompt(n):
    return NOTE_WAS_ASK.format(n=n)


NOTE_WAS_ASK2 = ("A note keeps facts about a person. When a fact changes, the note gives the new value, and then, in "
                 "brackets, a short word or phrase followed by the earlier value, which tells the reader that this "
                 "earlier value no longer holds. Give {n} different short words or phrases that could stand there, "
                 "one to three words each, in lowercase, one per line. Write only the word or phrase, not a fact or "
                 "a value.\nAfter the last line write END on its own line.")


def notewas_prompt2(n):
    return NOTE_WAS_ASK2.format(n=n)


def listname_prompt(vtype, n):
    return (f"Give {n} different short names, one to three words each, a person might use for their "
            f"list of {LIST_KIND[vtype][1]}. Write them in lowercase, one per line, no numbering.\n"
            "After the last line write END on its own line.")


def vote_prompt(what, cands):
    return (f"Which of these is the most natural short name for {what}? Answer with its number only.\n"
            + "\n".join(f"{i}. {c}" for i, c in enumerate(cands, 1)))


REL_ANS = ("woman", "man", "either")
NUM_ANS = ("one", "more")


def relfeat_prompt(values):
    return ("For each word below, which describes a person in someone's family or social life, write the word, a "
            "colon, who it can refer to (woman, man or either), a comma, and whether it means one person or more "
            "(one or more). One word per line, in the same order.\n" + "\n".join(values)
            + "\nAfter the last line write END on its own line.")


ATTR_TYPES = ("day of the week", "city", "month", "colour")


def attr_prompt(kind, n=3):
    return (f"Think of a {kind}. Give {n} different facts about a {kind} that a person might look up, where the "
            "answer is a day of the week, a city, a month or a colour. For each, write a short label for the fact "
            "(two to four words, lowercase, naming what it is, not the answer), a colon, and which of the four the "
            "answer is: day of the week, city, month or colour. One per line.\nAfter the last line write END on its "
            "own line.")


def pred_prompt(rows):
    """rows: [(name, attr, value)]: finish 'the <name> ...' so it states that its <attr> is <value>."""
    body = "\n".join(f"Line {i}: the {nm}, its {a} is {v}" for i, (nm, a, v) in enumerate(rows, 1))
    return ("For each line, write only the words that would follow the name in a plain sentence saying that fact, "
            "starting with a verb and holding the value exactly as given. No full stop.\n" + body
            + "\nAfter the last line write END on its own line.")


# ---- regex fragments ----------------------------------------------------------------------------------------------
def choice(words):
    return "(?:" + "|".join(D.esc(w) for w in words) + ")"


def line_res(kind, rows):
    """one regex fragment per output line for the label and feature calls."""
    if kind == "relfeat":
        return [D.esc(v) + ": " + choice(REL_ANS) + ", " + choice(NUM_ANS) for v in rows]
    if kind == "attr":
        return ["[a-z][a-z' ]*[a-z]: " + choice(ATTR_TYPES) for _ in rows]
    u = PR.BANK_UNIT
    if kind == "pred":
        return [f"{u}*{D.esc(v)}{u}*(?:EN?)?" for _, _, v in rows]
    if kind == "vote":
        return ["[1-9][0-9]?"]
    if kind == "para":     # the base block itself says END (its last output line) and "U1:", so END and digits pass
        return ["[^\\n*#`]+" for _ in rows]
    if kind == "label":
        return [(f"{u}*{D.esc(o)}{u}*(?:EN?)?" if o else PR.bank_noend()) for o in rows]
    return [PR.bank_noend() for _ in rows]


def spec(model, frags, max_tokens):
    """the decode spec of one non-line call: the fragments joined by the teacher's separator, then END."""
    sep = PR.SEP.get(model, "\\n")
    regex = sep.join(list(frags) + ["END"])
    s = {"structured": "bank-calls-v1", "regex": regex, "ban": "dash", "n": len(frags), "sep": sep,
         "max_tokens": max_tokens, "literals": []}
    s["sha256"] = store.sha256_text(repr(sorted(s.items())))
    return s


def rows_of(text, n):
    """the n non-empty lines before END, or (None, problem)."""
    rows = [r.strip() for r in (text or "").replace("\r", "").split("\n") if r.strip()]
    if not rows or rows[-1] != "END":
        return None, "NO_END"
    rows = rows[:-1]
    if len(rows) != n:
        return None, f"LINES_{len(rows)}_OF_{n}"
    return rows, None
