"""Clean and planted fixtures for the 2026-09-27 checker changes (teacher pilot reject audit, D1 and D2 in notes.txt).
Each loosening has a clean case modeled on an audited false reject (must pass now; the old checker failed it) and a
planted case with the real violation (must still fire). Each tightening has a planted case (fires now; the old
checker let it through). Same builder contract as fixtures_bound: (skel, texts, built) -> [(name, skel, texts,
expect, opts)]; mutation_checker runs them with the other boundary cases. FAKE fixture text written by Claude,
modeled on the audit's patterns; no teacher output is copied here."""
import copy
import re

import lexicons as L
import render_intents as RI
import topic_words as TW
from check_base import content, stem
from check_behav import own_nouns, topic_set
from defects_event import _slot_answer_events, _sub
from defects_text import _copy, filler_user
from fixtures_bound import _fillers, _topic_w


def _events(skel, kind, variant=None):
    return [e for e in skel["events"] if e["kind"] == kind and (variant is None or e["params"].get("variant") == variant)]


def _free_words(skel, want, pool):
    """words of pool that share no stem with the wanted set (for a reply that must read as off topic)."""
    return [w for w in pool if stem(w) not in want and len(w) > 3]


def _plain(skel, i):
    """a turn a fixture may overwrite with a plain sentence: no rules and no open-question ending."""
    t = skel["turns"][i]
    return not t.get("rules") and "open question" not in (t["intent"] or "")


def p_quotes(skel, texts, built):
    """curly apostrophes: the lexicons read the straight form (audit: 51 false hits, 35 missed real ones)."""
    out = []
    f = _fillers(skel, texts)
    if f:
        i = f[0]["i"]
        out.append(("curly_ai_ism_fire", skel, _copy(texts, i, texts[i] + " I’m here to help."), {"AI_ISM"}, {}))
    for e in _events(skel, "S6", "role_swap"):
        i = e["turns"]["answer"]
        if _plain(skel, i):
            out.append(("curly_denial_pass", skel, _copy(texts, i, "I’m an assistant, so there is none."), "ok", {}))
    return out


def _value_words(skel):
    from check_base import event_values
    return {w for v in event_values(skel) for w in str(v).lower().split()}


def p_offtopic(skel, texts, built):
    """OFFTOPIC: a topic-set word, a stem match and a reply to the previous turn pass; an unrelated reply fires."""
    out, topics = [], skel["topic_text"]
    all_topic = set().union(*(topic_set(x) for x in topics.values()))
    plain = set().union(*(content(x) for x in topics.values()))
    vw = _value_words(skel)
    for t in _fillers(skel, texts):
        if t["max_w"] < 8:
            continue
        prev = texts.get(t["i"] - 1, "")
        tid = next((u["intent"].split(":")[1] for u in reversed(skel["turns"][:t["i"]])
                    if u["role"] == "user" and (u["intent"] or "").startswith("topic:")), skel["topic_path"][0])
        rel = [w for w in TW.related(topics[tid]) if stem(w) not in plain | content(prev) and len(w) > 3 and w not in vw]
        reply = f"A {rel[0]} can change a lot here." if rel else ""
        if rel and not content(reply) & (plain | content(prev)):
            out.append(("offtopic_topicset_pass", skel, _copy(texts, t["i"], reply), "ok", {}))
        others = [w for x in TW.TOPIC_WORDS if x not in topics.values() for w in TW.related(x) if w not in vw]
        far = _free_words(skel, all_topic | content(prev), dict.fromkeys(others))
        bad = f"The {far[0]} and the {far[1]} matter most." if len(far) >= 2 else ""
        if bad and not content(bad) & (all_topic | content(prev)):
            out.append(("offtopic_far_fire", skel, _copy(texts, t["i"], bad), {"OFFTOPIC"}, {}))
            near = (prev + " " + " ".join(topics.values())).lower()
            gw = [w for w in sorted(L.GENERIC) if re.search(r"(?<![a-z])" + w + r"(?![a-z])", near)]
            if gw:   # a topic-neutral word shared with the user's line does not make a reply on topic
                out.append(("offtopic_generic_fire", skel, _copy(texts, t["i"], bad[:-1] + f", {gw[0]}."), {"OFFTOPIC"}, {}))
        break
    for u in filler_user(skel):
        a = [t for t in _fillers(skel, texts) if t["i"] == u["i"] + 1]
        reply = "Shaky ones happen, press firmly."
        if a and u["max_w"] >= 6 and u["min_w"] <= 6 and not content(reply) & all_topic:
            base = _copy(texts, u["i"], f"{_topic_w(skel, u)} tips? The jars keep shaking.")
            out.append(("offtopic_stem_pass", skel, _copy(base, a[0]["i"], reply), "ok", {}))
            break
    fill = {t["i"] for t in _fillers(skel, texts)}
    for u in filler_user(skel):
        cw = topic_set(topics[u["intent"].split(":")[1]])
        if u["i"] - 1 not in fill or u["max_w"] < 6 or stem("patience") in cw | {stem(w) for w in vw}:
            continue
        base = _copy(texts, u["i"] - 1, f"{_topic_w(skel)} takes some patience.")
        out.append(("offtopic_prev_assist_pass", skel, _copy(base, u["i"], "Patience is hard for me, honestly."),
                    "ok", {}))
        break
    return out


def p_req_forms(skel, texts, built):
    """REQ_WORD forms: an adverb and a -ves plural count (audit: "smoothly" read as missing "smooth")."""
    rw = skel["required_words"]
    if len(rw["turn_hint"]) < 3:
        return []
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 12]
    if not f:
        return []
    out = []
    sk = copy.deepcopy(skel)
    sk["required_words"].update(noun="shelf", adj="smooth")
    base = {i: s for i, s in texts.items()}
    from check_text import word_forms_re
    for k in ("noun", "adj"):
        base = {i: word_forms_re(rw[k]).sub("thing", s) for i, s in base.items()}
    w = _topic_w(skel)
    out.append(("req_forms_pass", sk, _copy(base, f[0]["i"], f"With {w}, the shelves slide in smoothly."), "ok", {}))
    out.append(("req_forms_fire", sk, _copy(base, f[0]["i"], f"With {w}, the boards slide in easily."), {"REQ_WORD"}, {}))
    return out


def p_placed(skel, texts, built):
    """only the placed required words are required: a skeleton that placed two words does not require the third."""
    rw = skel["required_words"]
    if len(rw["turn_hint"]) != 3:
        return []
    from check_text import word_forms_re
    sk = copy.deepcopy(skel)
    sk["required_words"]["turn_hint"] = rw["turn_hint"][:2]
    no_adj = {i: word_forms_re(rw["adj"]).sub("fine", s) for i, s in texts.items()}
    no_noun = {i: word_forms_re(rw["noun"]).sub("item", s) for i, s in texts.items()}
    if no_adj == texts or no_noun == texts:
        return []
    return [("placed_two_pass", sk, no_adj, "ok", {}), ("placed_two_fire", sk, no_noun, {"REQ_WORD"}, {})]


def p_abstain_deny(skel, texts, built):
    out = []
    for e in _events(skel, "S7", "abstain"):
        i = e["turns"]["answer"]
        if _plain(skel, i) and skel["turns"][i]["max_w"] >= 12:
            out.append(("hedge_mentioned_pass", skel, _copy(texts, i, "That was not mentioned yet, let me note it."),
                        "ok", {}))
            break
    for e in _events(skel, "S6", "role_swap"):
        i = e["turns"]["answer"]
        if _plain(skel, i):
            out.append(("deny_just_assistant_pass", skel, _copy(texts, i, "I'm just an assistant, so none here."),
                        "ok", {}))
            break
    return out


def p_det(skel, texts, built):
    """a user turn may say "my X" for a required "the X" and back; "your X" still fires."""
    from check_base import event_values
    vals = event_values(skel)
    for t in skel["turns"]:
        if t["role"] != "user" or t["mode"] != "guided":
            continue
        for x in t["must_include"]:
            det, _, rest = x.partition(" ")
            if det in ("the", "my") and x not in vals and x.lower() in texts[t["i"]].lower():
                alt = ("my " if det == "the" else "the ") + rest
                return [("det_swap_pass", skel, _copy(texts, t["i"], _sub(texts[t["i"]], x, alt)), "ok", {}),
                        ("det_your_fire", skel, _copy(texts, t["i"], _sub(texts[t["i"]], x, "your " + rest)),
                         {"REQ_SPAN"}, {})]
    return []


def p_exact_case(skel, texts, built):
    """D1: a lowercase-style user's exact lines compare without case; exact assistant and tool lines keep it."""
    if skel["user"]["style"] != "lowercase":
        return []
    out = []
    for t in skel["turns"]:
        if t["mode"] == "exact" and t["role"] == "user" and t["text"] != t["text"].lower():
            out.append(("exact_user_case_pass", skel, _copy(texts, t["i"], t["text"]), "ok", {}))
            out.append(("exact_user_lower_pass", skel, _copy(texts, t["i"], t["text"].lower()), "ok", {}))
            break
    for t in skel["turns"]:
        if t["mode"] == "exact" and t["role"] == "tool" and t["text"] != t["text"].lower():
            out.append(("exact_tool_case_fire", skel, _copy(texts, t["i"], t["text"].lower()), {"EXACT_MISMATCH"}, {}))
            break
    return out


def p_list_one(skel, texts, built):
    """LIST_STATE count: "which one" is a pronoun, "or maybe one" is a second count."""
    for e in _events(skel, "S2"):
        g, i = e["gold"], e["turns"]["answer"]
        if g["query"] == "count" and g["answer"] != "one" and _plain(skel, i) \
                and len(texts[i].split()) + 5 <= skel["turns"][i]["max_w"]:
            return [("count_which_one_pass", skel, _copy(texts, i, texts[i] + " Which one comes next?"), "ok", {}),
                    ("count_or_one_fire", skel, _copy(texts, i, texts[i] + " Or maybe one."), {"LIST_STATE"}, {})]
    return []


def p_hedge_given(skel, texts, built):
    """DEFLECT: a hedge on a given item, even with the right value (dry pilot: accepted "I do not know ..., but")."""
    for e in _slot_answer_events(skel):
        i, a = e["turns"]["answer"], e["gold"]["answer"]
        t = skel["turns"][i]
        if t["mode"] == "guided" and _plain(skel, i) and t["max_w"] >= 12:
            return [("hedge_given_fire", skel, _copy(texts, i, f"I do not know, but it is {a}."), {"DEFLECT"}, {})]
    return []


def p_guidance_copy(skel, texts, built):
    """PROMPT_ECHO: a guided user turn that is its own guidance word for word."""
    for u in filler_user(skel):
        g = RI.guidance(skel, u)
        if u["min_w"] <= len(g.split()) <= u["max_w"]:
            return [("guidance_copy_fire", skel, _copy(texts, u["i"], g[:1].upper() + g[1:] + "."), {"PROMPT_ECHO"}, {})]
    return []


def p_user_your(skel, texts, built):
    """PERSPECTIVE on a user turn: "your X" for the user's own X fires, "my X" passes."""
    ns = own_nouns(skel)
    for u in filler_user(skel):
        if ns and len(texts[u["i"]].split()) + 5 <= u["max_w"]:
            n = ns[-1]
            return [("user_my_pass", skel, _copy(texts, u["i"], texts[u["i"]] + f" Also, my {n} says hi."), "ok", {}),
                    ("user_your_fire", skel, _copy(texts, u["i"], texts[u["i"]] + f" Also, your {n} says hi."),
                     {"PERSPECTIVE"}, {})]
    return []


BUILDERS = [p_quotes, p_offtopic, p_req_forms, p_placed, p_abstain_deny, p_det, p_exact_case, p_list_one,
            p_hedge_given, p_guidance_copy, p_user_your]
