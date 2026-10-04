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
# list guidance says "the {L}", never "your {L}": the teachers copy the guidance's second person into the user line
# ("what is last on your shopping list?", 2026-09-27 audit), which check_behav now rejects as PERSPECTIVE
LIST_Q = {"first": "ask what comes first on the {L}", "last": "ask what comes last on the {L}",
          "ordinal": "ask what is {arg} on the {L}", "count": "ask how many things the {L} has now",
          "contains": "ask whether {arg} is still on the {L}",
          "other": "say {arg} is done and ask what the other one on the {L} was"}
# 2026-09-28: list, rule, lookup, recap and return guidance reworded so that a natural user line is not the guidance
# word for word (dp2: "Move cheese to the top of the shopping list." equal to "move {v} to the top of the {L}", and
# "Request a rule: every reply finishes by asking something." copied whole), while a copy still fires PROMPT_ECHO
LIST_OP = {"add": "have {v} added to the {L}", "remove": "have {v} taken off the {L}",
           "move_first": "have {v} put first on the {L}", "move_last": "have {v} put last on the {L}"}
RULE_HEAD = "ask for a rule for later replies: "
RULE = {"max_words": RULE_HEAD + "at most {word} words each", "one_sentence": RULE_HEAD + "a single sentence each",
        "end_question": RULE_HEAD + "each one closes with a question", "call_user": RULE_HEAD + "it calls you {name}",
        "avoid_word": RULE_HEAD + "the word {word} never comes up",
        "start_name": "give your name, {name}, and " + RULE_HEAD + "each one opens with that name"}
SOCIAL = {"greeting": "say hello", "how_are_you": "ask how the assistant is doing", "thanks": "thank the assistant",
          "goodbye": "say goodbye", "who_are_you": "ask who you are talking to"}
ONLY_ANSWER = "; no other value"
# 2026-09-28: answers go on after the value (dp2: bare "Durban." under the 3-word floor) and the lead in is what the
# user said (dp2: "Your friend is Mateo.", an E004 sentence frame: 94 of Gemma's 113 frame hits on the lead-in intent);
# abstains ask for the fact; the rule reply keeps the rule at once.
# 2026-10-03 (dry pilot 3 v1 review): "leading in with what you were told" came back as the assistant telling the user
# ("As you were told before, your shift is in December": Qwen 22 of 140 lead-in answers, Gemma 9 of 239), so the lead
# in names the user as the one who said it; "ask them to share it" came back as "please share it" glued on with no
# stop ("You have not told me the start time yet please share it": Qwen, 10 chats), so the abstain asks a question
ASSIST = {"answer with the value first": "answer, starting with the answer itself and going on" + ONLY_ANSWER,
          "answer with the value after a short lead in": "answer, first pointing back to when the user said it"
                                                         + ONLY_ANSWER,
          "say it was not mentioned, give no guess, offer to note it": "say the user has not told you, give no guess, "
                                                                       "and ask for it in a question",
          "agree and follow the rule": "agree, and keep the rule in this reply",
          "answer from the list": "answer from the list as it is now" + ONLY_ANSWER,
          "answer from what the user said, without a lookup": "answer from what the user said, without a lookup"
                                                              + ONLY_ANSWER,
          "answer with the looked up value": "answer with the looked up value" + ONLY_ANSWER,
          "greet back and engage with the topic": "greet the user back and remark on the topic",
          "greet the user back": "greet the user back briefly, maybe asking how they are; no topic yet",
          "acknowledge the change": "acknowledge the change in one short sentence",
          "acknowledge the list": "acknowledge the list in one short sentence",
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


def _op(skel, e, t):
    """the op this turn states, if any."""
    for op in (e["params"].get("ops", []) if e else []):
        if e["turns"].get(op["turn"]) == t["i"]:
            return op
    return None


def _ask_back(slot):
    """first-person, unambiguous ask-back (audit: "ask the assistant your favourite colour" came back as a question
    about the assistant's own favourite colour)."""
    ph = phrase(slot)
    return "ask the assistant to remind you " + (ph if ph.split()[0] in ("which", "what", "where") else "of " + ph)


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
        return ("say the old rule no longer holds, then " + text) if k else text
    if it == "mention a look-alike value that changes nothing":
        return f"mention a near choice that changes nothing about {phrase(slot)}" if slot else "mention a near choice"
    if it.startswith("state ") and slot:
        return f"tell the assistant {phrase(slot)}"
    if it.startswith("correct ") and slot:
        op = _op(skel, e, t)
        new = f": it is {op['value']} now" if op and op.get("value") else ""
        return f"correct what you said before about {phrase(slot)}{new}"
    if it == "ask for a recap":
        return "ask the assistant for a recap of the facts you gave"
    if it.startswith("ask for the ") and it.endswith(" name"):
        return "ask the assistant its name" if "self" in it else "ask whether the assistant knows your name"
    if it.startswith("ask the assistant for its own "):
        return "ask the assistant about its own " + it.rsplit("its own ", 1)[1]
    # 2026-09-28: no "without saying the answer" / "without saying it" (v1 review: copied into 13 accepted user
    # lines); the query line's must-not-include list now holds the answer (assemble.finish). The never-said query
    # reads like any recall question, so only memory tells the two sides of S7 apart.
    if it == "ask if the assistant remembers":
        return f"ask whether the assistant remembers {phrase(skel['slots'][p['slot']])}"
    if it.startswith("ask for ") and "never said" in it:
        return _ask_back({"key": p["key"], "noun": p.get("noun")}) if p.get("key") in KEY_PHRASE else \
            "ask the assistant to remind you about " + p.get("ref_expr", "it")
    if it.startswith("ask for "):
        q = p.get("queried") or p.get("slot")
        return _ask_back(skel["slots"][q]) if q else it
    if it == "go back to the first topic":
        return "go back to what you talked about first"
    if it == "ask about the entity":   # 10-03: "look up the X and report its Y" was copied (64 of 210 dp3 lines)
        return f"ask about the {p['attribute']} of the {p['entity']}"
    if it == "state a belief, ask to check":
        return f"say what you believe about the {p['attribute']} of the {p['entity']}, and ask the assistant to check"
    if it == "pass on a fact":
        return f"pass on something you heard about the {p['entity']}"
    return it


def assistant_guidance(skel, t):
    it = t["intent"]
    base, _, tail = it.partition("; ")
    text = ASSIST.get(base, base)
    if base == "answer with the value first" and any(r["verifier"] == "start_name" for r in t.get("rules", [])):
        text = "answer, with the answer right after the name the reply starts with, then a few words" + ONLY_ANSWER
    elif base == "acknowledge briefly":
        text = "acknowledge in one short sentence" + ("" if t["must_include"] else ", not repeating the value")
    elif base == "pick the first topic back up and name it" and t["must_include"]:
        text = f"go back to what was said before the detour and name {t['must_include'][0]} in the reply"
    elif base.startswith("recap, but state"):
        text = base + ", and name no other value"
    if tail:
        text += "; " + tail
    return text


def guidance(skel, t):
    return user_guidance(skel, t) if t["role"] == "user" else assistant_guidance(skel, t)
