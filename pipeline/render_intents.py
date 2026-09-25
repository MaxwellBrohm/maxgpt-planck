"""Skeleton intents -> teacher-facing guidance for guided script lines (SPEC section 5e). Claude wrote this wording;
it is prompt text, never training text, and PROMPT_ECHO rejects any render that copies 6 words of it in a row.
User guidance is written as an instruction to the user speaker ("tell the assistant ..."), because the teacher
copies wording literally and third-person descriptions ("the user says ...") come back as third-person lines."""
import banks as B
import fake_data as F
from banks_keys import KEYS

# what a slot key is about, as the owner would say it ({o} = noun)
KEY_PHRASE = {"user_name": "your name", "home_city": "the city you live in", "job": "your job",
              "hobby": "your hobby", "fav_food": "your favourite food", "fav_colour": "your favourite colour",
              "pet_name": "your {o}'s name", "plan_day": "which day your {o} is", "plan_month": "which month your {o} is",
              "plan_city": "which city your {o} is in", "plan_time": "what time your {o} starts",
              "item_colour": "the colour of the {o} you bought", "person_name": "your {o}'s name",
              "person_city": "where your {o} lives", "person_job": "what your {o} does for work"}
LIST_Q = {"first": "ask what is first on your {L}", "last": "ask what is last on your {L}",
          "ordinal": "ask what is {arg} on your {L}", "count": "ask how many things are on your {L} now",
          "contains": "ask whether {arg} is still on your {L}",
          "other": "say {arg} is done and ask what the other one on your {L} was"}
LIST_OP = {"add": "add {v} to your {L}", "remove": "take {v} off your {L}", "move_first": "move {v} to the top of your {L}",
           "move_last": "move {v} to the end of your {L}"}
RULE = {"max_words": "request a rule: replies of at most {word} words",
        "one_sentence": "request a rule: one sentence per reply",
        "end_question": "request a rule: a question mark at the end of each reply",
        "call_user": "request a rule: the assistant calls you {name}",
        "avoid_word": "request a rule: no {word} in replies",
        "start_name": "give your name, {name}, and request a rule: each reply opens with that name"}
SOCIAL = {"greeting": "say hello", "how_are_you": "ask how the assistant is doing", "thanks": "thank the assistant",
          "goodbye": "say goodbye", "who_are_you": "ask who you are talking to"}
ASSIST = {"answer with the value first": "answer, starting the reply with the answer itself",
          "answer with the value after a short lead in": "answer, with a few words before the answer",
          "answer from the list": "answer from the list as it is now",
          "greet back and engage with the topic": "greet the user back and pick up the topic",
          "reply on the topic": "reply to what the user just said", "react briefly": "react briefly",
          "respond helpfully on the first topic": "reply helpfully to what the user just said",
          "respond on the second topic": "reply to what the user just said"}


def phrase(slot):
    return KEY_PHRASE[slot["key"]].replace("{o}", slot.get("noun") or "")


def _event(skel, t):
    ids = t.get("events") or []
    return next((e for e in skel["events"] if e["id"] in ids), None)


def _op_slot(skel, e, t):
    """the slot stated by this turn's op, if any."""
    if not e:
        return None
    for op in e["params"].get("ops", []):
        if e["turns"].get(op["turn"]) == t["i"] and op.get("slot"):
            return skel["slots"][op["slot"]]
    return None


def _topic_text(skel, tid):
    return skel["topic_text"].get(tid, "the topic")


def user_guidance(skel, t):
    it, e = t["intent"], _event(skel, t)
    role = t.get("role_in_event")
    if it.startswith("topic:"):
        _, tid, form = it.split(":")
        text = _topic_text(skel, tid)
        if form == "digress":
            return f"bring up something new: {text}"
        return F.INTENT_FORMS[int(form)].replace("{t}", text).replace("the assistant", "you")
    if it == "open the chat about the topic":
        return f"open the chat about {_topic_text(skel, skel['topic_path'][0])}"
    if it == "say goodbye":
        return "say goodbye"
    if it.startswith("social: "):
        return SOCIAL[it.split(": ", 1)[1]]
    p = e["params"] if e else {}
    slot = _op_slot(skel, e, t)
    if it == "state the list":
        return f"tell the assistant your {p['list_name']}, in this order"
    if it.startswith("list "):
        op = next(o for o in p["ops"] if e["turns"].get(o["turn"]) == t["i"])
        return LIST_OP[op["op"]].format(v=op["value"], L=p["list_name"])
    if it.startswith("ask list "):
        return LIST_Q[p["query"]].format(arg=p.get("query_arg"), L=p["list_name"])
    if "ask for rule " in it:
        k = 0 if role == "rule" else 1
        seg = p["segments"][k]
        text = RULE[seg["verifier"]].format(**{**seg["args"], "word": seg["args"].get("word", "")})
        return ("drop the earlier rule, then " + text) if k else text
    if it == "mention a look-alike value that changes nothing":
        return f"mention a near choice that changes nothing about {phrase(slot)}" if slot else "mention a near choice"
    if it.startswith("state ") and slot:
        return f"tell the assistant {phrase(slot)}"
    if it.startswith("correct ") and slot:
        return f"correct what you said before about {phrase(slot)}"
    if it == "ask for a recap":
        return "ask the assistant to sum up what you have told it"
    if it.startswith("ask for the ") and it.endswith(" name"):
        return "ask the assistant its name" if "self" in it else "ask whether the assistant knows your name"
    if it.startswith("ask the assistant for its own "):
        return "ask the assistant about its own " + it.rsplit("its own ", 1)[1]
    if it == "ask if the assistant remembers":
        return f"ask whether the assistant remembers {phrase(skel['slots'][p['slot']])}, without saying it"
    if it.startswith("ask for ") and "never said" in it:
        return "ask about " + p.get("ref_expr", "it") + " as if you had said it before"
    if it.startswith("ask for "):
        q = p.get("queried") or p.get("slot")
        return f"ask the assistant {phrase(skel['slots'][q])}, without saying the answer" if q else it
    if it == "go back to the first topic":
        return "steer the chat back to the first topic"
    if it == "ask about the entity":
        return f"ask the assistant to find out the {p['attribute']} of the {p['entity']}"
    if it == "state a belief, ask to check":
        return f"say what you believe about the {p['attribute']} of the {p['entity']}, and ask the assistant to check"
    if it == "pass on a fact":
        return f"pass on something you heard about the {p['entity']}"
    return it


def assistant_guidance(skel, t):
    it = t["intent"]
    base, _, tail = it.partition("; ")
    text = ASSIST.get(base, base)
    if tail:
        text += "; " + tail
    return text


def guidance(skel, t):
    return user_guidance(skel, t) if t["role"] == "user" else assistant_guidance(skel, t)
