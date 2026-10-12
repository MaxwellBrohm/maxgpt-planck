"""Round 4 fixtures (2026-10-04, SPEC 16; the dry pilot 4 v1 quality review, notes.txt): a planted case that must fire
and a clean twin that must pass for every round 4 rule (check_r4.py, the broadened EARLIER_RE). Same builder contract
as fixtures_r3: (skel, texts, built) -> [(name, skel, texts, expect, opts)]; mutation_checker runs them. FAKE fixture
text written by Claude, modeled on the review's lines; no teacher output is copied here."""
import pools as P
import render_intents as RI
import render_prompt as R
from check_base import first_scheduled
from defects_text import _copy
from fixtures_bound import _fillers, _topic_w
from fixtures_pilot import _events, _plain
from fixtures_r2 import _sty
from fixtures_r3 import _lead_answers


def _guided(skel, role, pred):
    return [t for t in skel["turns"] if t["role"] == role and t["mode"] == "guided" and pred(t)]


def r4_memory(skel, texts, built):
    """FALSE_MEMORY never said (self-name answers), remember and the new pointer and time words; the clean twins:
    the assistant's own name said plainly, a pointer clause about something else, "I'll remember", a recall answer."""
    out, card = [], R.card_name(skel)
    for t in _guided(skel, "assistant", lambda t: RI.self_name_answer(skel, t) and _plain(skel, t["i"])
                     and t["max_w"] >= 12)[:1]:
        i = t["i"]
        out += [("never_called_me_fire", skel, _copy(texts, i, f"You called me {card}, that is my name."),
                 {"FALSE_MEMORY"}, {}),
                ("never_mentioned_my_name_fire", skel, _copy(texts, i, f"You mentioned that my name is {card}."),
                 {"FALSE_MEMORY"}, {}),
                ("never_as_you_asked_fire", skel, _copy(texts, i, f"As you asked me before, my name is {card}."),
                 {"FALSE_MEMORY"}, {}),
                ("never_own_name_pass", skel, _copy(texts, i, f"Oh, my name is {card}, nice to chat."), "ok", {}),
                ("meta_short_lead_in_fire", skel, _copy(texts, i, f"With a short lead in, my name is {card}."),
                 {"PROMPT_ECHO"}, {}),
                ("never_self_clause_pass", skel, _copy(texts, i, f"You asked after you mentioned your name, and I am "
                                                                 f"{card}."), "ok", {})]
    ack = _guided(skel, "assistant", lambda t: _plain(skel, t["i"]) and t["intent"] == "acknowledge briefly"
                  and t["must_include"] and t["max_w"] >= 8 and skel["turns"][t["i"] - 1]["role"] == "user"
                  and t["must_include"][0] in texts.get(t["i"] - 1, ""))
    for t in ack[:1]:
        v = t["must_include"][0]
        out += [("remember_prev_line_fire", skel, _copy(texts, t["i"], f"I remember you saying {v}."),
                 {"FALSE_MEMORY"}, {}),
                ("remember_promise_pass", skel, _copy(texts, t["i"], f"I'll remember {v}, thanks."), "ok", {}),
                ("earlier_at_the_start_fire", skel, _copy(texts, t["i"], f"You mentioned {v} at the start."),
                 {"FALSE_MEMORY"}, {}),
                ("earlier_noted_fire", skel, _copy(texts, t["i"], f"Okay, I noted {v} earlier."), {"FALSE_MEMORY"}, {})]
    for t, a in _lead_answers(skel)[:1]:
        out += [("remember_recall_pass", skel, _copy(texts, t["i"], f"I remember you saying {a}."), "ok", {}),
                ("never_user_said_pass", skel, _copy(texts, t["i"], f"You mentioned {a}, so {a} it is."), "ok", {})]
    return out


def r4_frame(skel, texts, built):
    """PROMPT_ECHO on a first-person copy of the correction and recap frames, and "Ask me" for an ask-back; the
    clean twins keep the fake line and say the move in other words."""
    out = []
    for t in _guided(skel, "user", lambda t: (t["intent"] or "").startswith("correct ") and t["max_w"] >= 24
                     and t["must_include"])[:1]:
        i, v = t["i"], t["must_include"][0]
        out += [("frame_first_person_fire", skel, _copy(texts, i, _sty(skel, f"I need to correct what I said before, it"
                                                                             f" is {v} now. " + texts[i])),
                 {"PROMPT_ECHO"}, {}),
                ("frame_first_person_pass", skel, _copy(texts, i, _sty(skel, "Sorry, I got that wrong. " + texts[i])),
                 "ok", {})]
    for t in _guided(skel, "user", lambda t: t["intent"] == "ask for a recap" and t["max_w"] >= 10)[:1]:
        out += [("frame_recap_fire", skel, _copy(texts, t["i"], _sty(skel, "Can you recap the facts I gave you?")),
                 {"PROMPT_ECHO"}, {}),
                ("frame_recap_pass", skel, _copy(texts, t["i"], _sty(skel, "Can you go over what I told you so far?")),
                 "ok", {})]
    for t in _guided(skel, "user", lambda t: RI.user_guidance(skel, t).startswith("ask the assistant to remind")
                     and t["max_w"] >= len(texts[t["i"]].split()) + 3)[:1]:
        out.append(("frame_ask_me_fire", skel, _copy(texts, t["i"], _sty(skel, "Ask me this: " + texts[t["i"]])),
                    {"PROMPT_ECHO"}, {}))
    return out


def r4_people(skel, texts, built):
    """ASSIST_CASE on a required name in lowercase, ASSIST_VOICE on the user named in the third person, PERSPECTIVE
    on the user's name or two-word thing claimed, ANSWER_WRONG on a person given another relation or a job used as a
    person; each with its clean twin."""
    out, first = [], first_scheduled(skel)
    ans = _guided(skel, "assistant", lambda t: t["must_include"] and _plain(skel, t["i"]) and t["max_w"] >= 10
                  and (t["intent"] or "").startswith("answer with the value") and t["must_include"][0][:1].isupper()
                  and len(t["must_include"][0]) > 2 and t["must_include"][0] in texts[t["i"]])
    for t in ans[:1]:
        v = t["must_include"][0]
        out.append(("case_name_lower_fire", skel, _copy(texts, t["i"], texts[t["i"]].replace(v, v.lower())),
                    {"ASSIST_CASE"} | ({"REQ_SPAN"} if skel["user"]["style"] == "lowercase" else set()), {}))
    u = R.user_name(skel)
    ends = [max(e["turns"].values()) for e in skel["events"] if u and u in str(e["params"]) + str(e["gold"])]
    late = [t for t in _fillers(skel, texts) if t["max_w"] >= 12 and u and first.get(u, 99) < t["i"]
            and u not in t["must_exclude"] and all(x < t["i"] for x in ends)]
    for t in late[:1]:
        w = _topic_w(skel)
        out += [("third_user_says_fire", skel, _copy(texts, t["i"], f"{u} says {w} takes patience."),
                 {"ASSIST_VOICE"}, {}),
                ("third_vocative_pass", skel, _copy(texts, t["i"], f"{u}, {w} takes patience."), "ok", {}),
                ("claims_name_really_fire", skel, _copy(texts, t["i"], f"My name is really {u}, and {w} helps."),
                 {"PERSPECTIVE"}, {})]
    for e in [e for e in _events(skel, "S6", "user_vs_card") if e["params"].get("ask") == "user"][:1]:
        i = e["turns"]["answer"]
        if _plain(skel, i) and skel["turns"][i]["max_w"] >= 10 and skel["turns"][i]["mode"] == "guided":
            out += [("third_answer_says_fire", skel, _copy(texts, i, f"{u} says that is the name you gave."),
                     {"ASSIST_VOICE"}, {}),
                    ("third_answer_vocative_pass", skel, _copy(texts, i, f"{u}, will you keep that name?"), "ok", {})]
    two = [s for s in skel["slots"].values() if s["owner"] != "assistant" and len((s.get("noun") or "").split()) > 1
           and s["noun"].split()[-1] not in ("case", "night", "day", "time")]
    for t in [t for t in _fillers(skel, texts) if t["max_w"] >= 12][:1] if two else []:
        h, w = two[0]["noun"].split()[-1], _topic_w(skel)
        out += [("claims_my_thing_fire", skel, _copy(texts, t["i"], f"My {h} is soon, and {w} helps."),
                 {"PERSPECTIVE"}, {}),
                ("claims_your_thing_pass", skel, _copy(texts, t["i"], f"Your {h} is soon, and {w} helps."), "ok", {})]
    rels = list(P.POOLS["relation"].values)
    for e in _events(skel, "S6", "two_persons") + _events(skel, "S1"):
        i, a = e["turns"].get("answer"), e["gold"].get("answer")
        s = next((x for x in skel["slots"].values() if x["value"] == a and x["key"] in ("person_name", "person_job")),
                 None)
        if not s or not _plain(skel, i) or skel["turns"][i]["max_w"] < 10 or skel["turns"][i]["mode"] != "guided":
            continue
        r = s["noun"]
        other = next(x for x in rels if x != r)
        if s["key"] == "person_name":
            out += [("rel_wrong_fire", skel, _copy(texts, i, f"I believe your {other} is actually {a}."),
                     {"ANSWER_WRONG"}, {}),
                    ("rel_right_pass", skel, _copy(texts, i, f"I believe your {r} {a} is the one."), "ok", {})]
        else:
            out += [("job_as_person_fire", skel, _copy(texts, i, f"Ah, your {a} works hard."), {"ANSWER_WRONG"}, {}),
                    ("job_adjective_pass", skel, _copy(texts, i, f"Ah, your {a} {r} works hard."), "ok", {})]
        break
    return out


def r4_rules_service(skel, texts, built):
    """PLANT_MISSING on a rule ask that never states its rule (one_sentence, end_question), the stated rule passing;
    AI_ISM on the service substitutes, with clean lines on the same words."""
    out = []
    say = {"one_sentence": "From now on, keep each reply to one sentence.",
           "end_question": "From now on, end every reply with a question."}
    for e in _events(skel, "S4"):
        for k, seg in enumerate(e["params"]["segments"]):
            i = e["turns"].get("rule" if k == 0 else "override")
            t = skel["turns"][i]
            if seg["verifier"] in say and t["mode"] == "guided" and t["max_w"] >= 10 and not out:
                out += [("rule_unstated_fire", skel, _copy(texts, i, _sty(skel, "Can we agree on a rule for later "
                                                                                "replies?")), {"PLANT_MISSING"}, {}),
                        ("rule_stated_pass", skel, _copy(texts, i, _sty(skel, say[seg["verifier"]])), "ok", {})]
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 14]
    if f:
        w, i = _topic_w(skel), f[0]["i"]
        W = w[:1].upper() + w[1:]
        out += [("service_ready_fire", skel, _copy(texts, i, f"I am doing well and ready to help with {w}."),
                 {"AI_ISM"}, {}),
                ("service_glad_assist_fire", skel, _copy(texts, i, f"Glad I could assist with {w}."), {"AI_ISM"}, {}),
                ("service_certainly_do_fire", skel, _copy(texts, i, f"I can certainly do that, {w} is fun."),
                 {"AI_ISM"}, {}),
                ("service_here_to_help_fire", skel, _copy(texts, i, f"An assistant here to help with {w}."),
                 {"AI_ISM"}, {}),
                ("service_else_question_fire", skel, _copy(texts, i, f"{W} is fun. Would you like to remember anything "
                                                                     "else?"), {"AI_ISM"}, {}),
                ("service_is_there_else_fire", skel, _copy(texts, i, f"{W} is fun. Is there anything else, or is that "
                                                                     "all?"), {"AI_ISM"}, {}),   # round 3's pattern only
                ("service_here_to_answer_fire", skel, _copy(texts, i, f"I am here to answer questions on {w}."),
                 {"AI_ISM"}, {}),
                ("service_what_else_fire", skel, _copy(texts, i, f"{W} is fun. What else can I help with?"),
                 {"AI_ISM"}, {}),
                ("service_hope_pass", skel, _copy(texts, i, f"I hope {w} goes well today."), "ok", {}),
                ("service_happy_helps_fire", skel, _copy(texts, i, f"I would be happy to helps with {w}."), {"AI_ISM"}, {}),
                ("service_anything_more_fire", skel, _copy(texts, i, f"{W} is fun. Would you like to know anything more "
                                                                     "about it?"), {"AI_ISM"}, {}),
                ("service_in_mind_fire", skel, _copy(texts, i, f"{W} is fun. Did you have anything else in mind?"),
                 {"AI_ISM"}, {}),
                ("service_of_service_fire", skel, _copy(texts, i, f"How may I be of service with {w}?"), {"AI_ISM"}, {}),
                ("service_try_help_fire", skel, _copy(texts, i, f"I can try to help you with {w}."), {"AI_ISM"}, {}),
                ("service_anything_new_fire", skel, _copy(texts, i, f"{W} is fun. Is there anything new to discuss?"),
                 {"AI_ISM"}, {}),
                ("service_anything_more_pass", skel, _copy(texts, i, f"{W} needs little, nothing more than time."),
                 "ok", {}),
                ("service_ready_go_pass", skel, _copy(texts, i, f"{W} is ready to go now."), "ok", {})]
    return out


BUILDERS = [r4_memory, r4_frame, r4_people, r4_rules_service]
