"""Clean and planted fixtures for the 2026-09-28 checker changes (round 2; SPEC 14, notes.txt). The planted cases
include every case type of the v1 quality review's failing fixtures (scratchpad qv/fixtures_qv.py and run_store.py),
ported: each fired nothing on the round 1 checker and must fire now. Same builder contract as fixtures_bound and
fixtures_pilot: (skel, texts, built) -> [(name, skel, texts, expect, opts)]; mutation_checker runs them. FAKE fixture
text written by Claude; no teacher output is copied here."""
import copy
import re

import banks as B
import pools as P
import topic_words as TW
from banks_keys import KEYS
from check_base import content, event_values, stem
from check_behav import topic_set
from defects_event import _slot_answer_events
from defects_text import _copy, filler_user
from fixtures_bound import _fillers, _topic_w
from fixtures_pilot import _events, _plain

# collision word -> a topic whose word set the round 1 stemmer met through it (review: ready/read, busy/bus, care/car,
# noted/note, cater/cat, letting/letter, plane/plan)
COLLIDE = {"ready": "reading more books", "busy": "a long commute", "care": "learning to drive",
           "noted": "a noisy neighbour", "cater": "a cat that scratches furniture", "letting": "writing a cover letter",
           "plane": "planning a picnic"}
FAKE_VALUE = {"city": "Atlantis", "job": "astronaut", "name": "Zorbex", "pet_name": "Quibble", "colour": "mauve",
              "food": "haggis", "hobby": "falconry", "day": "Doomsday", "month": "Smarch", "time": "teatime"}


def _sty(skel, s):
    """a user line in the user's style (a lowercase-style user's lines carry no capital: USER_STYLE)."""
    return s.lower() if skel["user"]["style"] == "lowercase" else s


def _far(skel, want, n=2):
    """words of other topics' word sets that share no stem with want and are no value word."""
    vw = {w for v in event_values(skel) for w in str(v).lower().split()}
    others = [w for x in TW.TOPIC_WORDS if x not in skel["topic_text"].values() for w in TW.related(x)]
    return [w for w in dict.fromkeys(others) if stem(w) not in want and len(w) > 3 and w not in vw][:n]


def r_offtopic(skel, texts, built):
    out, topics = [], set(skel["topic_text"].values())
    all_topic = set().union(*(topic_set(x) for x in topics))
    for t in [t for t in _fillers(skel, texts) if t["max_w"] >= 10][:1]:
        want = all_topic | content(texts.get(t["i"] - 1, ""))
        far = _far(skel, want)
        for w, topic in COLLIDE.items():     # planted: one colliding word and off-topic nouns
            if len(far) == 2 and topic in topics and not content(w) & want:   # no true relative (care/careful)
                out.append((f"offtopic_collide_{w}", skel,
                            _copy(texts, t["i"], f"{w.capitalize()}, but the {far[0]} and the {far[1]} matter most."),
                            {"OFFTOPIC"}, {}))
        if "a noisy neighbour" in topics:    # clean: "note" is still a topic noun there
            out.append(("offtopic_note_noun_pass", skel, _copy(texts, t["i"], "A polite note to them could work."),
                        "ok", {}))
    u = next((u for u in filler_user(skel) if u["max_w"] >= 10 and skel["turns"][u["i"] - 1]["role"] == "assistant"
              and u["i"] - 1 in {t["i"] for t in _fillers(skel, texts)}), None)
    if u:
        tw = topic_set(skel["topic_text"][u["intent"].split(":")[1]])
        far = _far(skel, tw | {stem("patience"), stem("routine"), stem("steady")}, 1)
        base = _copy(texts, u["i"] - 1, f"{_topic_w(skel).capitalize()} takes some patience and a steady routine.")
        if far and not {stem("patience"), stem("routine")} & tw:
            out.append(("offtopic_prev_assist_one_fire", skel,
                        _copy(base, u["i"], _sty(skel, f"Speaking of patience, my {far[0]} is acting up again.")),
                        {"OFFTOPIC"}, {}))
            out.append(("offtopic_prev_assist_two_pass", skel,
                        _copy(base, u["i"], _sty(skel, "A steady routine takes patience I lack.")), "ok", {}))
    return out


def r_req_forms(skel, texts, built):
    """REQ_WORD: -ly only for an adjective and never a meaning-shifted adverb; -ves only for a real -ves noun."""
    rw, f = skel["required_words"], [t for t in _fillers(skel, texts) if t["max_w"] >= 12]
    if len(rw["turn_hint"]) < 3 or not f:
        return []
    try:
        from check_lines import word_forms_re
    except ImportError:                    # the round 1 checker (red run): forms did not depend on the part
        from check_text import word_forms_re as _w
        word_forms_re = lambda w, part=None: _w(w)  # noqa: E731
    drop = lambda ks: {i: _drop(s, [word_forms_re(rw[k], k) for k in ks]) for i, s in texts.items()}  # noqa: E731
    topic, out = _topic_w(skel), []
    for k, w, form in (("adj", "clear", "Clearly"), ("noun", "friend", "friendly"), ("adj", "fair", "fairly"),
                       ("adj", "short", "shortly"), ("noun", "cafe", "caves")):
        sk = copy.deepcopy(skel)
        sk["required_words"][k] = w
        if word_forms_re(w, k).search(" ".join(texts.values())):   # the word itself is already in the chat
            continue
        line = f"{form} a {topic} helps here." if form[0].isupper() else f"A {form} {topic} helps here."
        out.append((f"req_form_{w}_{form.lower()}_fire", sk, _copy(drop([k]), f[0]["i"], line), {"REQ_WORD"}, {}))
    base = drop(["noun", "adj"])
    sk = copy.deepcopy(skel)
    sk["required_words"].update(noun="knife", adj="gentle")
    out.append(("req_form_knives_gently_pass", sk, _copy(base, f[0]["i"], f"With {topic}, cut gently with knives."),
                "ok", {}))
    return out


def _drop(s, rxs):
    for rx in rxs:
        s = rx.sub("thing", s)
    return s


def r_abstain(skel, texts, built):
    out = []
    for e in _events(skel, "S7", "abstain"):
        i, vt, q = e["turns"]["answer"], e["params"]["vtype"], e["turns"]["query"]
        pool, fv = P.POOLS.get(vt), FAKE_VALUE.get(vt)
        if not _plain(skel, i) or skel["turns"][i]["max_w"] < 12 or not fv or (pool and fv in pool.values):
            continue
        ref = (e["params"].get("ref_expr") or "that").replace("my ", "your ")
        out.append(("abstain_was_mentioned_fire", skel, _copy(texts, i, f"{ref.capitalize()} was mentioned as {fv}, "
                                                                        "tell me if that changed."),
                    {"ABSTAIN_MISSING"}, {}))
        out.append(("abstain_havent_told_pass", skel, _copy(texts, i, "You haven't told me that yet, what is it?"),
                    "ok", {}))
        other = sorted(set(pool.values) - event_values(skel)) if pool else []
        if skel["turns"][q]["mode"] == "guided" and other and skel["turns"][q]["max_w"] >= 12:
            label = KEYS[e["params"]["key"]]["label"].replace("{o}", e["params"].get("noun") or "thing")
            out.append(("abstain_query_states_fire", skel, _copy(texts, q, _sty(
                skel, f"My {label} is {other[0]}. {texts[q]}")), {"QUERY_RESTATES"}, {}))
        break
    return out


def r_self_claim(skel, texts, built):
    out = []
    for e in _events(skel, "S6", "role_swap"):
        i, lab = e["turns"]["answer"], e["params"]["label"]
        if not _plain(skel, i) or skel["turns"][i]["max_w"] < 12:
            continue
        for name, line in (("deny_then_do_love_fire", f"An assistant doesn't need a {lab}, but I do love falconry."),
                           ("deny_then_spend_fire", "I have no hobby, though I spend my evenings painting."),
                           ("deny_like_my_own_fire", f"An assistant doesn't have a {lab} like my own.")):
            out.append((name, skel, _copy(texts, i, line), {"SELF_CLAIM"}, {}))
        out.append(("deny_of_my_own_pass", skel, _copy(texts, i, f"I have no {lab} of my own, so tell me yours."),
                    "ok", {}))
        break
    f = _fillers(skel, texts)
    if f and f[0]["max_w"] >= 8:
        out.append(("self_booked_fire", skel, _copy(texts, f[0]["i"], f"I booked a {_topic_w(skel)} class for us."),
                    {"SELF_CLAIM"}, {}))
    return out


def r_count(skel, texts, built):
    for e in _events(skel, "S2"):
        g, i = e["gold"], e["turns"]["answer"]
        if g["query"] != "count" or g["answer"] == "one" or not _plain(skel, i) or skel["turns"][i]["max_w"] < 10:
            continue
        n = g["answer"]
        return [("count_or_the_one_fire", skel, _copy(texts, i, f"There are {n}, or maybe just the one."),
                 {"LIST_STATE"}, {}),
                ("count_only_one_of_fire", skel, _copy(texts, i, f"There are {n} on it, but only one of them is left."),
                 {"LIST_STATE"}, {}),
                ("count_next_one_pass", skel, _copy(texts, i, f"There are {n} now, the next one is up to you."), "ok", {})]
    return []


def r_end(skel, texts, built):
    last, out = max(t["i"] for t in skel["turns"]), []
    if skel["turns"][last]["role"] == "assistant" and skel["turns"][last]["mode"] == "guided":
        out += [("end_token_fire", skel, _copy(texts, last, texts[last] + " END"), {"END_IN_TURN"}, {}),
                ("end_tail_fire", skel, _copy(texts, last, texts[last] + " End."), {"END_IN_TURN"}, {})]
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 10]
    if f:
        out.append(("end_word_pass", skel, _copy(texts, f[0]["i"], f"The {_topic_w(skel)} can wait until the end of "
                                                                   "the week."), "ok", {}))
    return out


def r_plant(skel, texts, built):
    out, done = [], set()
    for e in skel["events"]:
        for op in e["params"].get("ops", []):
            i, s = e["turns"].get(op["turn"]), skel["slots"].get(op.get("slot") or "")
            if op["op"] != "plant" or i is None or not s or skel["turns"][i]["mode"] != "guided" or \
                    skel["turns"][i]["role"] != "user" or skel["turns"][i]["max_w"] < 12:
                continue
            k = KEYS.get(s["key"], {})
            lab = k.get("label", s["type"]).replace("{o}", s.get("noun") or "")
            if s.get("noun") and "noun" not in done:
                done.add("noun")
                out += [("plant_asked_fire", skel, _copy(texts, i, _sty(
                    skel, f"Can you tell me the {lab} of my {s['noun']} {s['value']}?")), {"PLANT_MISSING"}, {}),
                        ("plant_stated_then_asked_pass", skel, _copy(texts, i, texts[i] + _sty(
                            skel, " Can you remember that?")), "ok", {}),
                        ("plant_remember_that_pass", skel, _copy(texts, i, _sty(
                            skel, f"Can you remember that my {s['noun']} is {s['value']}?")), "ok", {})]
            elif s["owner"] == "user" and not k.get("noun") and s["key"] not in ("user_name", "assistant_name") \
                    and "label" not in done:
                done.add("label")
                out += [("plant_your_label_asked_fire", skel, _copy(texts, i, _sty(
                    skel, f"What is your {lab} {s['value']}?")), {"PERSPECTIVE", "PLANT_MISSING"}, {}),
                        ("plant_your_label_fire", skel, _copy(texts, i, _sty(skel, f"Your {lab} is {s['value']}.")),
                         {"PERSPECTIVE"}, {}),
                        ("plant_my_label_pass", skel, _copy(texts, i, _sty(skel, f"My {lab} is {s['value']}.")), "ok",
                         {})]
    return out


def r_answers(skel, texts, built):
    for e in _slot_answer_events(skel):
        i, a = e["turns"]["answer"], e["gold"]["answer"]
        t = skel["turns"][i]
        if t["mode"] == "guided" and _plain(skel, i) and t["max_w"] >= 12:
            return [("given_as_was_mentioned_pass", skel, _copy(texts, i, f"It is {a}, as was mentioned earlier."),
                     "ok", {}),
                    ("given_not_mentioned_but_fire", skel, _copy(texts, i, f"It was not mentioned, but it is {a}."),
                     {"DEFLECT"}, {})]
    return []


def r_style_echo_ism(skel, texts, built):
    out, users = [], [t for t in skel["turns"] if t["role"] == "user" and t["mode"] == "guided"]
    if skel["user"]["style"] == "lowercase" and users:
        up = {t["i"]: texts[t["i"]][:1].upper() + texts[t["i"]][1:] for t in users}
        out.append(("lowercase_user_capitals_fire", skel, {**texts, **up}, {"USER_STYLE"}, {}))
    q = next((t for t in users if (t["intent"] or "").startswith("ask for ") and t["max_w"] >= 12 and
              [r for r in t["must_include"] if r not in event_values(skel)]), None)
    if q:
        ref = [r for r in q["must_include"] if r not in event_values(skel)][0]
        out += [("query_without_saying_fire", skel, _copy(texts, q["i"], _sty(
            skel, f"Can you remind me of {ref} without saying the answer?")), {"PROMPT_ECHO"}, {}),
                ("query_plain_pass", skel, _copy(texts, q["i"], _sty(skel, f"Can you remind me of {ref}?")), "ok", {})]
    for t in users:
        it, e = t["intent"] or "", next((e for e in skel["events"] if e["id"] in t["events"]), None)
        if it == "ask for rule end_question" and t["max_w"] >= 10:
            out += [("rule_copy_fire", skel, _copy(texts, t["i"], _sty(skel, "From now on each one closes with a "
                                                                              "question.")), {"PROMPT_ECHO"}, {}),
                    ("rule_natural_pass", skel, _copy(texts, t["i"], _sty(skel, "Please end every reply with a "
                                                                                "question from now on.")), "ok", {})]
        op = next((o for o in (e["params"].get("ops", []) if e else []) if e["turns"].get(o["turn"]) == t["i"]), None)
        if it.startswith("list ") and op and op["op"] == "add" and t["max_w"] >= 8:   # dp2: equal to the old guidance
            out.append(("list_add_natural_pass", skel, _copy(texts, t["i"], _sty(
                skel, f"Add {op['value']} to the {e['params']['list_name']}.")), "ok", {}))
    f, u = _fillers(skel, texts), next((u for u in filler_user(skel) if u["max_w"] >= 8), None)
    if f:
        out.append(("assist_how_can_i_help_fire", skel, _copy(texts, f[0]["i"], f"{_topic_w(skel).capitalize()} is "
                                                                                 "tricky, how can I help you?"),
                    {"AI_ISM"}, {}))
    if u:
        out.append(("user_how_can_i_help_pass", skel, _copy(texts, u["i"], _sty(
            skel, f"How can I help with {_topic_w(skel, u)} at home?")), "ok", {}))
    return out


BUILDERS = [r_offtopic, r_req_forms, r_abstain, r_self_claim, r_count, r_end, r_plant, r_answers, r_style_echo_ism]
