"""Round 2 fixtures, part two (2026-09-28): the clean cases behind the refinements the dry pilot 2 re-check asked for
(a plant in a relative "that" clause, comparatives of topic adjectives, a long -y word meeting its base) and planted
cases that pin one pattern each where fixtures_r2 hit two at once (END mid-turn, "I spend"), and r2b_refine /
r2b_offtopic_forms: the cases behind the fixes the full re-check read asked for (topic-side forms, the share rule for
a reply to the assistant, denials, "I called you", "No X was mentioned", "the one you added", embedded plants). Same
contract as
fixtures_r2. FAKE fixture text written by Claude, modeled on the recheck's patterns; no teacher output is copied."""
import topic_words as TW
from check_base import content, stem
from check_behav import topic_set
from fixtures_pilot import _events, _plain
from stemmer import comparatives, topic_forms
from defects_text import _copy, filler_user
from fixtures_bound import _fillers, _topic_w
from fixtures_r2 import _sty


def r2b_plant_that(skel, texts, built):
    """a plant stated in a relative clause ("I bought a hat that is purple": three dp2 accepts the first rule lost;
    "my aunt, who is called Nora, ...": a wh-word that opens a relative clause, not a question)."""
    out = []
    for e in skel["events"]:
        for op in e["params"].get("ops", []):
            i, s = e["turns"].get(op["turn"]), skel["slots"].get(op.get("slot") or "")
            if op["op"] != "plant" or not s or skel["turns"][i]["mode"] != "guided" or skel["turns"][i]["max_w"] < 10:
                continue
            if s["key"] == "item_colour":
                out.append(("plant_that_is_pass", skel, _copy(texts, i, _sty(
                    skel, f"I just bought a {s['noun']} that is {s['value']}, do you like it?")), "ok", {}))
            if s["key"] == "person_name":
                out.append(("plant_who_clause_pass", skel, _copy(texts, i, _sty(
                    skel, f"My {s['noun']}, who is called {s['value']}, says hello.")), "ok", {}))
            if s["key"] in ("person_name", "pet_name"):   # "is" after "and" continues the subject, it asks nothing
                out.append(("plant_and_is_pass", skel, _copy(texts, i, _sty(
                    skel, f"My {s['noun']} is lovely and is called {s['value']}.")), "ok", {}))
    return out


def r2b_offtopic(skel, texts, built):
    """clean: a reply on topic only through a comparative of a topic adjective, or through a long -y word of the
    user's last line (photographer / photography)."""
    out, topics = [], list(skel["topic_text"].values())
    all_topic = set().union(*(topic_set(x) for x in topics))
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 10 and t["i"] - 1 in texts]
    if not f:
        return out
    t = f[0]
    base = content(texts[t["i"] - 1])
    adjs = [a for x in topics if x in TW.TOPIC_WORDS for a in TW.words(x, "adj") if len(a) <= 6]
    comp = next((c for a in adjs for c in comparatives(a)[:1] if stem(c) != stem(a) and stem(c) not in base), None)
    rest = content("It all gets a while")
    if comp and not rest & (all_topic | base):
        out.append(("offtopic_comparative_pass", skel, _copy(texts, t["i"], f"It all gets {comp} after a while."),
                    "ok", {}))
    u = t["i"] - 1
    if skel["turns"][u]["role"] == "user" and skel["turns"][u]["mode"] == "guided" and skel["turns"][u]["max_w"] >= 8:
        w = _topic_w(skel, skel["turns"][u])
        reply = "Photography needs patience, stay with it."
        if not (content(reply) - {stem("photography")}) & (all_topic | content(w)) and stem("photograph") not in all_topic:
            b2 = _copy(texts, u, _sty(skel, f"A photographer friend helps me with {w}."))
            out.append(("offtopic_y_word_pass", skel, _copy(b2, t["i"], reply), "ok", {}))
    return out


def r2b_single(skel, texts, built):
    """END in the middle of a turn (no closing "End." to catch it instead), script talk in the last turn (the round 2
    serve probe: "Wait, no, just the script and end."), and "I spend" without "my evenings"."""
    out, last = [], max(t["i"] for t in skel["turns"])
    if skel["turns"][last]["role"] == "assistant" and skel["turns"][last]["mode"] == "guided":
        out += [("end_mid_fire", skel, _copy(texts, last, texts[last] + " END. Take care."), {"END_IN_TURN"}, {}),
                ("script_talk_fire", skel, _copy(texts, last, texts[last] + " That ends the script for now."),
                 {"END_IN_TURN"}, {})]
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 10]
    if f:
        out.append(("self_spend_fire", skel, _copy(texts, f[0]["i"], f"I spend a lot of time on {_topic_w(skel)}."),
                    {"SELF_CLAIM"}, {}))
    return out


def r2b_ism_meta(skel, texts, built):
    """AI_ISM: a glued "how can Ihelp" (a phrase-ban substitute, round 2 serve probe) fires; "anything else I can
    think of" is not the service phrase. PROMPT_ECHO: the lead-in guidance clause copied into an answer fires."""
    out, f = [], [t for t in _fillers(skel, texts) if t["max_w"] >= 12]
    if f:
        w, i = _topic_w(skel), f[0]["i"]
        out += [("assist_glued_help_fire", skel, _copy(texts, i, f"{w.capitalize()} is tricky, how can Ihelp?"),
                 {"AI_ISM"}, {}),
                ("assist_anything_else_pass", skel, _copy(texts, i, f"For {w}, patience matters more than anything else "
                                                                    "I can think of."), "ok", {})]
    for e in skel["events"]:
        i, a = e["turns"].get("answer"), e["gold"].get("answer")
        t = skel["turns"][i] if i is not None else None
        if t and isinstance(a, str) and t["mode"] == "guided" and (t["intent"] or "").startswith("answer with the value "
                                                                                                 "after") \
                and not t.get("rules") and t["max_w"] >= 12 and "open question" not in t["intent"]:
            out.append(("lead_in_copy_fire", skel, _copy(texts, i, f"Leading in with what you told me, it is {a}."),
                        {"PROMPT_ECHO"}, {}))
            break
    return out


def r2b_withhold(skel, texts, built):
    """PROMPT_ECHO on a query line with a withholding clause in any wording (a paraphrase of the old guidance
    suffix); the same clause on another user line is ordinary speech."""
    out = []
    q = next((skel["turns"][e["turns"]["query"]] for e in skel["events"] if e["turns"].get("query") is not None
              and skel["turns"][e["turns"]["query"]]["mode"] == "guided"
              and skel["turns"][e["turns"]["query"]]["role"] == "user"), None)
    if q and len(texts[q["i"]].split()) + 4 <= q["max_w"] and texts[q["i"]].endswith("?"):
        out.append(("query_withhold_paraphrase_fire", skel, _copy(texts, q["i"], texts[q["i"]][:-1] + _sty(
            skel, " without telling me directly?")), {"PROMPT_ECHO"}, {}))
    u = next((u for u in filler_user(skel) if u["max_w"] >= 8), None)
    if u:
        out.append(("user_without_telling_pass", skel, _copy(texts, u["i"], _sty(
            skel, f"I started on {_topic_w(skel, u)} without telling anyone.")), "ok", {}))
    return out



def r2b_refine(skel, texts, built):
    """clean: "to call my own" in a denial, "I called you" (the call-me rule), "No such thing was mentioned", "the one
    you added" in a count, "did you know my X is called Y", "Where I live is Y."; planted: "I called a ..." (a past
    act), "Do you know the name of my X Y?" and "Where I live is Y?" (questions, no statement)."""
    out, f = [], [t for t in _fillers(skel, texts) if t["max_w"] >= 12]
    if f:
        w, i = _topic_w(skel), f[0]["i"]
        out += [("self_called_you_pass", skel, _copy(texts, i, f"Sorry I called you by the wrong name, {w} matters more."),
                 "ok", {}),
                ("self_called_fire", skel, _copy(texts, i, f"I called a {w} expert for you yesterday."), {"SELF_CLAIM"}, {})]
    for e in _events(skel, "S6", "role_swap")[:1]:
        i = e["turns"]["answer"]
        if _plain(skel, i) and skel["turns"][i]["max_w"] >= 12:
            out.append(("deny_call_my_own_pass", skel, _copy(texts, i, f"I'm an assistant, so no {e['params']['label']} to "
                                                                       "call my own."), "ok", {}))
    for e in _events(skel, "S7", "abstain")[:1]:
        i = e["turns"]["answer"]
        if _plain(skel, i) and skel["turns"][i]["max_w"] >= 12:
            out.append(("abstain_no_x_mentioned_pass", skel, _copy(texts, i, "No such thing was mentioned so far, what is "
                                                                             "it?"), "ok", {}))
    for e in _events(skel, "S2"):
        g, i = e["gold"], e["turns"]["answer"]
        if g["query"] == "count" and g["answer"] != "one" and _plain(skel, i) and skel["turns"][i]["max_w"] >= 10:
            out.append(("count_the_one_you_pass", skel, _copy(texts, i, f"There are {g['answer']} now, including the one "
                                                                        "you just added."), "ok", {}))
            break
    for e in skel["events"]:
        for op in e["params"].get("ops", []):
            i, s = e["turns"].get(op["turn"]), skel["slots"].get(op.get("slot") or "")
            if op["op"] != "plant" or not s or skel["turns"][i]["mode"] != "guided" or skel["turns"][i]["role"] != "user" \
                    or skel["turns"][i]["max_w"] < 10:
                continue
            if s["key"] in ("pet_name", "person_name"):
                out += [("plant_did_you_know_pass", skel, _copy(texts, i, _sty(
                    skel, f"Did you know my {s['noun']} is called {s['value']}?")), "ok", {}),
                        ("plant_know_the_name_fire", skel, _copy(texts, i, _sty(
                            skel, f"Do you know the name of my {s['noun']} {s['value']}?")), {"PLANT_MISSING"}, {})]
            if s["key"] == "home_city":
                out += [("plant_cleft_pass", skel, _copy(texts, i, _sty(skel, f"Where I live is {s['value']}.")), "ok", {}),
                        ("plant_cleft_question_fire", skel, _copy(texts, i, _sty(skel, f"Where I live is {s['value']}?")),
                         {"PLANT_MISSING"}, {})]
    return out


def r2b_offtopic_forms(skel, texts, built):
    """clean: an assistant reply on topic only through a topic-side form (an agent noun: drive -> driver, speak ->
    speaker; the -y and -ly bases are pinned by tests/test_round2_units.py), and a user reply that shares one of its two content words with the assistant's last
    line ("I could eat X every day"); the planted one-word echo of four stays in fixtures_r2."""
    out, topics = [], list(skel["topic_text"].values())
    plain = set().union(*(content(x) | content(" ".join(TW.related(x))) for x in topics))
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 10 and t["i"] - 1 in texts]
    if f:
        prev = content(texts[f[0]["i"] - 1])
        forms = [x for tx in topics for w in TW.words(tx, "verb") for x in topic_forms(w, "verb")]
        form = next((x for x in forms if stem(x) not in plain | prev and stem(x) in set().union(
            *(topic_set(tx) for tx in topics))), None)
        line = f"The {form} part is where it gets hard." if form else ""
        if form and not (content(line) - {stem(form)}) & (plain | prev | set().union(*(topic_set(x) for x in topics))):
            out.append(("offtopic_topic_form_pass", skel, _copy(texts, f[0]["i"], line), "ok", {}))
    fill = {t["i"] for t in _fillers(skel, texts)}
    for u in filler_user(skel):
        tw = topic_set(skel["topic_text"][u["intent"].split(":")[1]])
        far = [w for x in TW.TOPIC_WORDS if x not in topics for w in TW.words(x, "noun")
               if stem(w) not in tw | plain and len(w) > 3]
        if u["i"] - 1 not in fill or u["max_w"] < 8 or not far or stem("eat") in tw | content(texts[u["i"] - 1]):
            continue
        base = _copy(texts, u["i"] - 1, f"{_topic_w(skel).capitalize()} goes well with a {far[0]}.")
        out.append(("offtopic_prev_share_pass", skel, _copy(base, u["i"], _sty(skel, f"I could eat {far[0]} every day.")),
                    "ok", {}))
        break
    return out


BUILDERS = [r2b_plant_that, r2b_offtopic, r2b_single, r2b_ism_meta, r2b_withhold, r2b_refine, r2b_offtopic_forms]
