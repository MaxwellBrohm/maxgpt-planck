"""Round 3 fixtures (2026-10-03, SPEC 15; the dry pilot 3 v1 quality review, notes.txt): a clean case that must pass
and a planted case that must fire for every round 3 change (D2). Same builder contract as fixtures_r2: (skel, texts,
built) -> [(name, skel, texts, expect, opts)]; mutation_checker runs them. FAKE fixture text written by Claude,
modeled on the review's lines; no teacher output is copied here."""
import render_prompt as R
from defects_text import _copy, filler_user
from fixtures_bound import _fillers, _topic_w
from fixtures_pilot import _events, _plain
from fixtures_r2 import _sty


def _lead_answers(skel):
    """(turn, value) of lead-in answers to a user value (not the card name) that a fixture may overwrite."""
    out = []
    for e in skel["events"]:
        i, a = e["turns"].get("answer"), e["gold"].get("answer")
        t = skel["turns"][i] if i is not None else None
        if t and isinstance(a, str) and a != R.card_name(skel) and t["mode"] == "guided" and _plain(skel, i) \
                and (t["intent"] or "").startswith("answer with the value after") and t["max_w"] >= 12:
            out.append((t, a))
    return out


def r3_told(skel, texts, built):
    """PERSPECTIVE on a told frame: "As you were told before" and "I told you" for what the user said fire (also when
    the assistant's acknowledgement repeated it); "You told me before" passes, and so does "As I said" when the
    assistant was the first to name the value (its card name in its greeting)."""
    out = []
    for t, a in _lead_answers(skel)[:1]:
        out += [("told_were_told_fire", skel, _copy(texts, t["i"], f"As you were told before, it is {a}."),
                 {"PERSPECTIVE"}, {}),
                ("told_i_told_you_fire", skel, _copy(texts, t["i"], f"I told you, it is {a}."), {"PERSPECTIVE"}, {}),
                ("told_as_told_fire", skel, _copy(texts, t["i"], f"As told before, it is {a}."), {"PERSPECTIVE"}, {}),
                ("told_you_told_me_pass", skel, _copy(texts, t["i"], f"You told me before that it is {a}."), "ok", {})]
    card = R.card_name(skel)
    for e in _events(skel, "S8")[:1]:
        i = e["turns"]["answer"]
        g = next((t for t in skel["turns"] if t["role"] == "assistant" and t["mode"] == "guided" and t["i"] < i
                  and t["intent"] == "greet the user back" and t["max_w"] >= 4), None)
        if e["params"]["act"] != "who_are_you" or not skel["assistant"].get("system_text") or not _plain(skel, i) \
                or skel["turns"][i]["max_w"] < 10 or not g:
            continue
        said = _copy(texts, g["i"], f"Hello, I'm {card}.")
        out += [("told_as_i_said_pass", skel, _copy(said, i, f"As I said, I'm {card}, an assistant."), "ok", {}),
                ("told_as_i_said_fire", skel, _copy(texts, i, f"As I said, I'm {card}, an assistant."),
                 {"PERSPECTIVE"}, {})]
    return out


def r3_frame(skel, texts, built):
    """PROMPT_ECHO on a copied guidance frame: form 2 ("a follow up question about {t}"), form 4 ("react briefly to the
    last reply about {t}") and the return line ("go back to what you talked about first"); a paraphrase passes, and a
    list question that is its guidance's own natural wording passes."""
    out = []
    for u in filler_user(skel):
        form, w = u["intent"].split(":")[2], _topic_w(skel, u)
        t = skel["topic_text"][u["intent"].split(":")[1]]
        if form == "2" and u["max_w"] >= 12 and not any(n.startswith("frame_follow") for n, *_ in out):
            out += [("frame_follow_up_fire", skel, _copy(texts, u["i"], _sty(skel, f"I have a follow up question "
                                                                                    f"about {t}.")), {"PROMPT_ECHO"}, {}),
                    ("frame_follow_up_pass", skel, _copy(texts, u["i"], _sty(skel, f"Also, about {w}, what else "
                                                                                    "should I know?")), "ok", {})]
        if form == "4" and u["max_w"] >= 12 and not any(n.startswith("frame_react") for n, *_ in out):
            out.append(("frame_react_fire", skel, _copy(texts, u["i"], _sty(skel, f"Briefly, about your last reply on "
                                                                                   f"{t}, nice.")), {"PROMPT_ECHO"}, {}))
    for t in skel["turns"]:
        if t["role"] == "user" and t["mode"] == "guided" and t["intent"] == "go back to the first topic" \
                and t["max_w"] >= 10:
            out.append(("frame_go_back_fire", skel, _copy(texts, t["i"], _sty(skel, "Can we go back to what you "
                                                                                    "talked about first?")),
                        {"PROMPT_ECHO"}, {}))
            break
    for e in _events(skel, "S2")[:1]:
        q = e["turns"]["query"]
        if e["gold"]["query"] == "first" and skel["turns"][q]["mode"] == "guided" and skel["turns"][q]["max_w"] >= 8:
            out.append(("frame_list_natural_pass", skel, _copy(texts, q, _sty(skel, "What comes first on "
                                                                                    f"my {e['params']['list_name']}?")),
                        "ok", {}))
    return out


def r3_memory(skel, texts, built):
    """FALSE_MEMORY: "I've forgotten" on an acknowledgement fires and "I don't recall" in an abstain passes; "correcting
    me" after the user corrected themselves fires and after an assistant_err fix passes; "earlier" one line after the
    user said it fires, and on an answer to a value planted well before passes."""
    out = []
    ack = [t for t in skel["turns"] if t["role"] == "assistant" and t["mode"] == "guided" and _plain(skel, t["i"])
           and t["intent"] == "acknowledge briefly" and t["must_include"] and t["max_w"] >= 8
           and skel["turns"][t["i"] - 1]["role"] == "user" and t["must_include"][0] in texts.get(t["i"] - 1, "")]
    for t in ack[:1]:
        v = t["must_include"][0]
        out += [("memory_forgot_ack_fire", skel, _copy(texts, t["i"], f"Sounds like fun, {v}, but I've forgotten."),
                 {"FALSE_MEMORY"}, {}),
                ("memory_earlier_prev_line_fire", skel, _copy(texts, t["i"], f"You mentioned {v} earlier, nice."),
                 {"FALSE_MEMORY"}, {})]
    for t, a in _lead_answers(skel)[:1]:
        out.append(("memory_earlier_answer_pass", skel, _copy(texts, t["i"], f"You mentioned {a} earlier, so {a}."),
                    "ok", {}))
    for e in _events(skel, "S7", "abstain")[:1]:
        i = e["turns"]["answer"]
        if _plain(skel, i) and skel["turns"][i]["max_w"] >= 12:
            out.append(("memory_abstain_recall_pass", skel, _copy(texts, i, "I don't recall that, you haven't told me "
                                                                            "yet. What is it?"), "ok", {}))
    for e in _events(skel, "S3"):
        sets = [(e["turns"][op["turn"]] + 1, op["value"]) for op in e["params"]["ops"] if op["op"] == "set"]
        fix = e["turns"].get("fix")
        for (i, new), name in ([(sets[0], "self")] if sets else []) + ([((fix + 1, None), "fix")] if fix is not None
                                                                         else []):
            t = skel["turns"][i]
            if t["role"] != "assistant" or t["mode"] != "guided" or not _plain(skel, i) or t["max_w"] < 6:
                continue
            if name == "self" and not any(x["turns"].get("fix", 99) < i for x in _events(skel, "S3", "assistant_err")):
                out += [("memory_correcting_me_fire", skel, _copy(texts, i, "Thank you for correcting me."),
                         {"FALSE_MEMORY"}, {}),
                        ("memory_thanks_correction_pass", skel, _copy(texts, i, "Thanks for the correction."), "ok", {}),
                        ("memory_change_ack_before_pass", skel, _copy(texts, i, f"So {new}, not the one you mentioned "
                                                                                "before."), "ok", {})]
            if name == "fix":
                out.append(("memory_fix_correcting_me_pass", skel, _copy(texts, i, "Thank you for correcting me."),
                            "ok", {}))
        r, err = e["turns"].get("recap_reply"), next((op for op in e["params"]["ops"] if op["op"] == "err"), None)
        if r is not None and err and _plain(skel, r) and skel["turns"][r]["max_w"] >= 8:   # the planned misstatement
            out.append(("memory_recap_earlier_pass", skel, _copy(texts, r, f"You mentioned {err['value']} earlier, I "
                                                                           "believe."), "ok", {}))
    return out


def r3_line(skel, texts, built):
    """LIVE_DATA, ASSIST_CASE, RUN_ON, SELF_CLAIM ("I'd love to", "myself"), AI_ISM substitutes and the round 3
    guidance clauses (PROMPT_ECHO), each with a clean twin where one exists."""
    out, f = [], [t for t in _fillers(skel, texts) if t["max_w"] >= 14]
    if f:
        w, i = _topic_w(skel), f[0]["i"]
        W = w[:1].upper() + w[1:]
        out += [("live_cold_outside_fire", skel, _copy(texts, i, f"It is quite cold outside today, so {w} can wait."),
                 {"LIVE_DATA"}, {}),
                ("live_weather_today_fire", skel, _copy(texts, i, f"With {w}, the weather is so sunny today."),
                 {"LIVE_DATA"}, {}),
                ("live_hope_pass", skel, _copy(texts, i, f"I hope it is sunny outside today for {w}."), "ok", {}),
                ("live_when_pass", skel, _copy(texts, i, f"Plan {w} for days when it is cold outside."), "ok", {}),
                ("run_on_yes_please_pass", skel, _copy(texts, i, f"Yes please tell me more about {w}."), "ok", {}),
                ("case_lower_start_fire", skel, _copy(texts, i, f"{w} takes some patience."), {"ASSIST_CASE"}, {}),
                ("case_lone_i_fire", skel, _copy(texts, i, f"{W} takes patience, i think."), {"ASSIST_CASE"}, {}),
                ("case_sentence_pass", skel, _copy(texts, i, f"{W} takes patience, I think."), "ok", {}),
                ("self_id_love_fire", skel, _copy(texts, i, f"I'd love to catch up with old friends over {w}."),
                 {"SELF_CLAIM"}, {}),
                ("self_myself_fire", skel, _copy(texts, i, f"{W} is fun, I would try it myself."), {"SELF_CLAIM"}, {}),
                ("self_id_love_hear_pass", skel, _copy(texts, i, f"I'd love to hear how {w} goes for you."), "ok", {}),
                ("ism_can_help_fire", skel, _copy(texts, i, f"I can certainly help you with {w}."), {"AI_ISM"}, {}),
                ("ism_else_fire", skel, _copy(texts, i, f"{W} takes time. Would you like to know anything else?"),
                 {"AI_ISM"}, {}),
                ("ism_else_pass", skel, _copy(texts, i, f"{W} matters more than anything else."), "ok", {}),
                ("ism_else_to_add_fire", skel, _copy(texts, i, f"{W} takes time. Anything else to add?"), {"AI_ISM"}, {}),
                ("meta_asking_question_fire", skel, _copy(texts, i, f"Asking for it in a question, how is {w} going?"),
                 {"PROMPT_ECHO"}, {}),
                ("memory_meant_to_pass", skel, _copy(texts, i, f"I meant to ask, how is {w} going?"), "ok", {})]
    for t, a in _lead_answers(skel)[:1]:
        out.append(("meta_pointing_back_fire", skel, _copy(texts, t["i"], f"Pointing back to when you said it, {a}."),
                    {"PROMPT_ECHO"}, {}))
    for e in _events(skel, "S7", "abstain")[:1]:
        i = e["turns"]["answer"]
        if _plain(skel, i) and skel["turns"][i]["max_w"] >= 12:
            out += [("run_on_yet_please_fire", skel, _copy(texts, i, "You have not told me that yet please share it."),
                     {"RUN_ON"}, {}),
                    ("run_on_comma_pass", skel, _copy(texts, i, "You have not told me that yet, please share it."),
                     "ok", {}),
                    ("run_on_so_please_pass", skel, _copy(texts, i, "You have not told me that yet so please share it."),
                     "ok", {}),
                    ("meta_in_a_question_fire", skel, _copy(texts, i, "You haven't told me, so I ask for it in a "
                                                                      "question. What is it?"), {"PROMPT_ECHO"}, {})]
    for e in _events(skel, "S6", "role_swap")[:1]:
        i = e["turns"]["answer"]
        if _plain(skel, i) and skel["turns"][i]["max_w"] >= 10:
            out += [("self_myself_denial_pass", skel, _copy(texts, i, "I don't have one myself, I'm an assistant."),
                     "ok", {}),
                    ("self_myself_neg_pass", skel, _copy(texts, i, "I don't eat it myself, I'm an assistant."), "ok", {})]
    return out


BUILDERS = [r3_told, r3_frame, r3_memory, r3_line]
