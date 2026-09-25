"""FAKE teacher, user side: a correct first-person user line for every guided user turn of a skeleton. Claude wrote
these frames as TEST FIXTURES ONLY (provenance FAKE); they never reach training text. Used by fake_teacher.py."""
import banks as B
from check_base import event_values

KEY_FRAME = {"user_name": "I'm {v}, by the way.", "home_city": "I live in {v} these days.",
             "job": "I work as {av}.", "hobby": "Lately I spend my free time on {v}.",
             "fav_food": "I could eat {v} every day.", "fav_colour": "My favourite colour is {v}.",
             "pet_name": "My {o} is called {v}.", "plan_day": "My {o} is on {v}.", "plan_month": "My {o} is in {v}.",
             "plan_city": "My {o} is in {v}.", "plan_time": "My {o} starts at {v}.", "item_colour": "The {o} I got is {v}.",
             "person_name": "My {o} is called {v}.", "person_city": "My {o} lives in {v}.",
             "person_job": "My {o} works as {av}."}
FILLER = {0: ["Any easy tip for {t}?", "Tips for {t}?", "{T}, tips?"],
          1: ["So far {t} is going okay.", "{T} is going okay.", "{T}, going okay."],
          2: ["Where should I start with {t}?", "Where to start with {t}?", "{T}, where to start?"],
          3: ["I worry a little about {t}.", "Worried about {t}.", "{T} worries me."],
          4: ["Okay, that helps with {t}.", "Okay, {t} then.", "Okay, {t}."],
          5: ["What should I try next with {t}?", "Next step for {t}?", "{T}, next step?"],
          "digress": ["Also, {t} has been on my mind.", "Also, {t}.", "{T} too."]}
SOCIAL = {"greeting": "Hello, nice to see you.", "how_are_you": "How are you doing today?",
          "thanks": "Thanks, that is useful.", "goodbye": "I need to go now, bye.",
          "who_are_you": "Who am I chatting with?"}
LIST_Q = {"first": "What is first on my {L}?", "last": "What is last on my {L}?", "ordinal": "What is {a} on my {L}?",
          "count": "How many are on my {L} now?", "contains": "Is {a} still on my {L}?",
          "other": "{A} is done, what is the other one on my {L}?"}
LIST_OP = {"add": "Please add {v} to my {L}.", "remove": "Please take {v} off my {L}.",
           "move_first": "Put {v} at the top of my {L}.", "move_last": "Put {v} at the end of my {L}."}
LIST_SHORT = {"add": "Add {v} to my {L}.", "remove": "Take {v} off my {L}.", "move_first": "{V} first on my {L}.",
              "move_last": "{V} last on my {L}."}
RULE = {"max_words": "Keep your replies to {word} words or fewer, please.",
        "one_sentence": "Answer in one sentence from now on, please.",
        "end_question": "End each reply with a question, please.", "call_user": "Please call me {name} from now on.",
        "avoid_word": "Please avoid the word {word} from now on.",
        "start_name": "I'm {name}, please start each reply with my name."}


def cap(s):
    return s[:1].upper() + s[1:]


def _event(skel, t):
    ids = t.get("events") or []
    return next((e for e in skel["events"] if e["id"] in ids), None)


def _op(e, t):
    return next((o for o in e["params"].get("ops", []) if e["turns"].get(o["turn"]) == t["i"]), None) if e else None


def _refs(skel, t):
    vals = event_values(skel)
    return [x for x in t["must_include"] if x not in vals]


def candidates(skel, t):
    """candidate lines for a guided user turn, longest first; fake_teacher picks the first that fits max_w."""
    it, e = t["intent"], _event(skel, t)
    role = t.get("role_in_event")
    topics = skel["topic_text"]
    if it.startswith("topic:"):
        _, tid, form = it.split(":")
        key = "digress" if form == "digress" else int(form)
        return [f.format(t=topics[tid], T=cap(topics[tid])) for f in FILLER[key]]
    if it == "open the chat about the topic":
        tt = topics[skel["topic_path"][0]]
        return [f"Hi, I want to talk about {tt}.", f"Hi, {tt}."]
    if it == "say goodbye":
        return ["Okay, that is all for today, bye.", "Okay, bye for now."]
    if it.startswith("social: "):
        return [SOCIAL[it.split(": ", 1)[1]]]
    p = e["params"] if e else {}
    op = _op(e, t)
    refs = _refs(skel, t)
    if it == "state the list":
        return [f"My {p['list_name']} is {B.join_items(op['items'])}.", f"{cap(p['list_name'])}: {', '.join(op['items'])}."]
    if it.startswith("list "):
        return [LIST_OP[op["op"]].format(v=op["value"], L=p["list_name"]),
                LIST_SHORT[op["op"]].format(v=op["value"], V=cap(op["value"]), L=p["list_name"])]
    if it.startswith("ask list "):
        a = p.get("query_arg") or ""
        return [LIST_Q[p["query"]].format(a=a, A=cap(a), L=p["list_name"])]
    if "ask for rule " in it:
        seg = p["segments"][0 if role == "rule" else 1]
        line = RULE[seg["verifier"]].format(**{"word": "", "name": "", **seg["args"]})
        return [("New rule instead. " if role == "override" else "") + line, line]
    if it == "mention a look-alike value that changes nothing":
        return [f"I almost went with {op['value']} instead.", f"Nearly picked {op['value']}."]
    if it.startswith("state ") and op and op.get("slot"):
        s = skel["slots"][op["slot"]]
        holes = B.key_holes(s["key"], s["noun"], op["value"], s["features"])
        return [KEY_FRAME[s["key"]].format(**holes), op["value"] + "."]
    if it.startswith("correct ") and op:
        v = op["value"]
        if refs:
            return [f"Sorry, I got it wrong. It's {v} for {refs[0]}.", f"Sorry, {v} for {refs[0]}."]
        return [f"No, it's {v}, you got that wrong.", f"No, it's {v}."]
    if it == "ask for a recap":
        return ["Can you go over what I told you?"]
    if it.startswith("ask for the ") and it.endswith(" name"):
        return ["And what should I call you?"] if "self" in it else ["Do you know what I am called?"]
    if it.startswith("ask the assistant for its own "):
        return [f"What about you, any {p['label']}?"]
    if it == "ask if the assistant remembers":
        return [f"Do you remember {refs[0]}?"]
    if it.startswith("ask for ") and "never said" in it:
        return [f"What was {refs[0]} again?"]
    if it.startswith("ask for "):
        return [f"Can you remind me about {refs[0]}?", f"Remind me, {refs[0]}?"]
    if it == "go back to the first topic":
        return ["Anyway, back to what we said first.", "Back to the first thing."]
    pred = B.LOOKUP_PRED.get(p.get("vtype"), "is {v}")
    if it == "ask about the entity":
        return [f"Can you find the {p['attribute']} of the {p['entity']}?", f"Find the {p['entity']}, please?"]
    if it == "state a belief, ask to check":
        b = pred.format(v=p["stale"][0])
        return [f"I think the {p['entity']} {b}, can you check?", f"I think the {p['entity']} {b}. Check?"]
    if it == "pass on a fact":
        v = skel["slots"][p["slot"]]["value"]
        return [f"Someone told me the {p['entity']} {pred.format(v=v)}.", f"The {p['entity']} {pred.format(v=v)}."]
    raise KeyError(f"no fake user line for intent {it!r}")


def styled(skel, text, t):
    """apply the user style to a fake line: lowercase, or one harmless typo outside required spans."""
    style = skel["user"]["style"]
    if style == "lowercase":
        return text.lower()
    if style == "typos" and " about " in text and not any(" about " in x for x in t["must_include"]):
        return text.replace(" about ", " abuot ", 1)
    return text
