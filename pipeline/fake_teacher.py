"""FAKE teacher: a correct render of a skeleton, as the teacher's raw completion text (labels + END). Claude wrote
every frame here as a TEST FIXTURE (provenance FAKE); none of it may reach training text. mutation_checker.py
uses it as the clean baseline that every check must pass, then plants one defect at a time. Frames rotate by turn
index so consecutive assistant turns do not copy each other (SELF_COPY)."""
import parse
import render_prompt as R
from check_base import content, sentences, stem, words
from lexicons import STOPWORDS
from fake_user import candidates as user_candidates, styled, cap, _event

PROVENANCE = "FAKE"
ACK = [["Noted, {x} then.", "Okay, {x}, got that.", "Thanks, {x} it is."], ["Okay, noted that.", "Got that, thanks.",
                                                                       "Alright, noted down."]]
CHANGE = [["Okay, updated to {x}.", "Thanks, changed to {x}.", "Fixed, {x} now."],
          ["Okay, I have updated it.", "Thanks, changed that.", "Fixed, noted the change."]]
TOPIC = ["That makes sense, {w} can take some patience.", "Good point, {w} is worth a small plan.",
         "Nice, {w} sounds manageable step by step.", "I see, {w} gets easier with practice."]
TOPIC_SHORT = ["{W} takes patience.", "{W} gets easier.", "{W} is manageable.", "{W} needs a plan."]
FIRST = ["{A}, that is what you said.", "{A}, from what you told me.", "{A}, if I have it right."]
LEAD = ["You told me it was {a}.", "From what you said, it is {a}.", "As you said earlier, {a}."]
ITEM = ["That would be {a}.", "It is {a}.", "Looks like {a}."]
FIXED = {"acknowledge the list": ["Okay, I have your list.", "Got the list, thanks."],
         "react briefly": ["Oh, a close call then.", "Ah, nearly a different pick."],
         "agree and follow the rule": ["Sure, I can do that.", "Okay, will do."],
         "say it was not found and give no guess": ["Sorry, I could not find anything on that.",
                                                    "Nothing came up for that, sorry."],
         "say it was not mentioned, give no guess, offer to note it": ["You haven't told me that yet. Want to tell me?",
                                                                      "Not sure, you didn't mention it. Tell me?",
                                                                      "You haven't said. Share it?"],
         "say it has no such thing of its own, without inventing one": ["I don't have one of my own, I'm an assistant.",
                                                                       "I don't have one, I'm an assistant."],
         "respond to the greeting briefly, no new topic": ["Hello, nice to hear from you."],
         "respond to the how are you briefly, no new topic": ["Doing fine, thanks for asking."],
         "respond to the thanks briefly, no new topic": ["You're welcome, glad it helped."],
         "respond to the goodbye briefly, no new topic": ["Bye, take care."],
         "say goodbye briefly": ["Bye for now, take care.", "Take care, bye."]}
REQ = {"noun": ["A {w} could help.", "A {w} helps."], "verb": ["You could {w} it later.", "{W} it."],
       "adj": ["That sounds {w}.", "So {w}."]}


def topic_words(text):
    """real content words of a topic text (not stems), longest first."""
    ws = [w for w in words(text) if w not in STOPWORDS and len(w) > 2]
    return sorted(dict.fromkeys(ws), key=len, reverse=True)


def _topic_word(skel, prev_user_text, i):
    tw = topic_words(" ".join(skel["topic_text"].values()))
    prev = content(prev_user_text or "")
    pool = [w for w in tw if stem(w) in prev] or tw
    return pool[(i // 2) % len(pool)] if pool else "that"


def _role(e, i):
    return next((k for k, v in e["turns"].items() if v == i), None) if e else None


def assistant_core(skel, t, prev_user_text):
    """(long, short) sentence for a guided assistant turn, before rules, names and required words."""
    base, i = t["intent"].split(";")[0], t["i"]
    e = _event(skel, t)
    rot = (i // 2)
    names = {r["args"].get("name") for r in t.get("rules", [])}
    vals = [x for x in t["must_include"] if x not in names]
    x = " and ".join(vals)
    g = e["gold"] if e else {}
    if base == "acknowledge briefly":
        f = ACK[0 if vals else 1][rot % 3]
        return f.format(x=x), (f"Okay, {x}, noted." if vals else "Okay, noted that.")
    if base == "acknowledge the change":
        f = CHANGE[0 if vals else 1][rot % 3]
        return f.format(x=x), (f"Okay, {x}, updated." if vals else "Okay, updated that.")
    if base in FIXED:
        opts = FIXED[base]
        return opts[rot % len(opts)], opts[(rot + 1) % len(opts)], opts[-1]
    if base in ("greet back and engage with the topic", "reply on the topic", "respond helpfully on the first topic",
                "respond on the second topic"):
        w = _topic_word(skel, prev_user_text, i)
        if base.startswith("greet"):
            return f"Hi, happy to chat about {w}.", f"Hi, {w} it is."
        return TOPIC[rot % 4].format(w=w), TOPIC_SHORT[rot % 4].format(W=cap(w))
    if base == "greet the user by name":
        return f"Nice to meet you, {x}.", f"Hi, {x}."
    if base == "say its name and that it is an assistant":
        return f"I'm {R.card_name(skel)}, an assistant.", f"I'm {R.card_name(skel)}."
    if base == "pick the first topic back up and name it":
        return f"Sure, back to {x}.", f"Back to {x}."
    if base.startswith("recap, but state"):
        return f"So far you told me about {x}.", f"You mentioned {x}."
    a = g.get("answer")
    if base == "answer with the value first":
        return FIRST[rot % 3].format(A=cap(a)), f"{cap(a)}."
    if base == "answer with the value after a short lead in":
        return LEAD[rot % 3].format(a=a), f"It is {a}."
    if base == "answer from what the user said, without a lookup":
        return ["You mentioned {a} before.", "Earlier you said {a}.", "Going by what you said, {a}."][rot % 3] \
            .format(a=a), f"It is {a}."
    if base == "answer with the looked up value":
        return f"The result says {a}.", f"It says {a}."
    if base == "answer from the list":
        if g["query"] == "count":
            return f"There are {a} now.", f"{cap(a)} now."
        if g["query"] == "contains":
            return ("Yes, it is still on there.", "Yes, still on.") if a == "yes" else \
                ("No, it is not on there now.", "No, it is off.")
        return ITEM[rot % 3].format(a=a), f"{cap(a)}."
    raise KeyError(f"no fake assistant line for intent {base!r}")


LOWERABLE = {"Okay", "Noted", "Thanks", "That", "It", "You", "Sure", "Got", "From", "As", "So", "Good", "Nice", "Hi",
             "Fixed", "Looks", "There", "Yes", "No", "Sorry", "Nothing", "Bye", "Take", "Doing", "Hello", "Alright",
             "Oh", "Ah", "Not", "Back", "A", "Want", "Tell", "Anything", "Right"}


def _lower_first(s):
    w = s.split(" ", 1)[0].rstrip(",.!?")
    return s[:1].lower() + s[1:] if w in LOWERABLE else s


def _one_sentence(text):
    sents = sentences(text)
    out = [(x.rstrip(".!?") if n < len(sents) - 1 else x) for n, x in enumerate(sents)]
    return ", ".join([out[0]] + [_lower_first(x) for x in out[1:]])


def finish(skel, t, core, req_word):
    """add rule effects, the required word and the open-question ending to one core sentence."""
    rules = {r["verifier"]: r["args"] for r in t.get("rules", [])}
    text = core
    if req_word:
        kind, w, form = req_word
        text += " " + REQ[kind][form].format(w=w, W=cap(w))
    name = (rules.get("start_name") or rules.get("call_user") or {}).get("name")
    if name and not text.startswith(name):
        text = f"{name}, {_lower_first(text)}"
    wants_q = "end_question" in rules or "end with an open question" in t["intent"]
    if "one_sentence" in rules:
        text = _one_sentence(text)
    if wants_q and not text.endswith("?"):
        text = text.rstrip(".!") + (", right?" if "one_sentence" in rules else ". Anything else?")
    return text


def _copies(prev, text):
    """the fake's own guard: text repeats a 4-word run of the previous assistant line."""
    if not prev:
        return False
    g = lambda s: {tuple(words(s)[i:i + 4]) for i in range(len(words(s)) - 3)}  # noqa: E731
    return bool(g(prev) & g(text))


def assistant_line(skel, t, prev_user_text, req_word, prev_assist=None):
    """the first fitting variant that carries the required word; else the first fitting variant, skipping one that
    repeats the previous assistant line (two same-intent turns under a tight word cap, SELF_COPY)."""
    cores = assistant_core(skel, t, prev_user_text)
    fits = lambda x: t["min_w"] <= len(x.split()) <= t["max_w"]  # noqa: E731
    for rw in ([req_word + (0,), req_word + (1,)] if req_word else []):
        for core in cores:
            text = finish(skel, t, core, rw)
            if fits(text) and not _copies(prev_assist, text):
                return text, True
    ok = [x for x in dict.fromkeys(finish(skel, t, c, None) for c in cores) if fits(x)]
    ok = [x for x in ok if not _copies(prev_assist, x)] or ok
    return (ok[0] if ok else finish(skel, t, cores[-1], None)), False


def user_line(skel, t):
    opts = user_candidates(skel, t)
    fit = [o for o in opts if t["min_w"] <= len(o.split()) <= t["max_w"]]
    return styled(skel, (fit or opts[-1:])[0], t)


def render(skel):
    """{turn index: text}: a correct FAKE render. Required words go on their hinted turns, and move to the next
    guided assistant turn when a hinted turn has no room."""
    texts, prev_user, prev_assist = {}, None, None
    pending = [(k, skel["required_words"][k]) for k in R.REQ_WORD_ORDER]
    hint = dict(zip(skel["required_words"]["turn_hint"], pending))
    carry = []
    for t in skel["turns"]:
        if t["mode"] == "exact":
            texts[t["i"]] = t["text"]
        elif t["role"] == "user":
            texts[t["i"]] = user_line(skel, t)
        else:
            if t["i"] in hint:
                carry.append(hint[t["i"]])
            rw = carry[0] if carry else None
            texts[t["i"]], used = assistant_line(skel, t, prev_user, rw, prev_assist)
            prev_assist = texts[t["i"]]
            if used:
                carry.pop(0)
        if t["role"] == "user":
            prev_user = texts[t["i"]]
    return texts


def raw(skel):
    return parse.serialize(skel, render(skel))
