"""Boundary fixtures: pairs that sit exactly on a checker threshold (must fire) and one step inside it (must pass),
so a checker mutant that moves a threshold or a bound by one is killed. Plus skeleton-level cases: SKEL_INFEASIBLE,
HELDOUT_STRUCT, the report-only VOCAB_OOL, a capitalized answer written in lowercase, and a required word used
only in its plural. Each builder takes (skel, texts, built) and returns [(name, skel, texts, expect, opts)] or [];
expect is "ok", a set of codes that must fire, or ("report", codes). FAKE fixture text."""
import copy

import render_prompt as R
from fake_teacher import topic_words
from check_text import word_forms_re
from defects_text import filler_assist, filler_user, _copy
from defects_event import _slot_answer_events, _sub

WORDS = ("plenty of other small ordinary words keep coming along here until this line has grown far past every "
         "limit anyone would set for one short friendly reply in a chat today and then some more beyond that "
         "point").split()


def _free_of_req(skel, text):
    rw = skel["required_words"]
    return not any(word_forms_re(rw[k]).search(text) for k in ("noun", "verb", "adj"))


def _topic_w(skel, t=None):
    """a real topic word; for a user filler turn, a word of that turn's own topic."""
    it = (t or {}).get("intent") or ""
    text = skel["topic_text"][it.split(":")[1]] if it.startswith("topic:") else " ".join(skel["topic_text"].values())
    return topic_words(text)[0]


def _fillers(skel, texts):
    return [t for t in filler_assist(skel) if _free_of_req(skel, texts[t["i"]])]


def b_len(skel, texts, built):
    out, w = [], _topic_w(skel)
    f = _fillers(skel, texts)
    if f and f[0]["max_w"] < len(WORDS):
        t = f[0]
        out += [("len_max_fire", skel, _copy(texts, t["i"], w + " " + " ".join(WORDS[:t["max_w"]]) + "."), {"LEN_ASSIST"}, {}),
                ("len_max_pass", skel, _copy(texts, t["i"], w + " " + " ".join(WORDS[:t["max_w"] - 1]) + "."), "ok", {})]
    u = [t for t in filler_user(skel) if t["min_w"] == 3]
    if u:
        uw = _topic_w(skel, u[0])
        out += [("len_min_fire", skel, _copy(texts, u[0]["i"], f"{uw} tips?"), {"LEN_USER"}, {}),
                ("len_min_pass", skel, _copy(texts, u[0]["i"], f"Tips for {uw}?"), "ok", {})]
    return out


def b_repeat(skel, texts, built):
    f, w = _fillers(skel, texts), _topic_w(skel)
    if not f or f[0]["max_w"] < 22:
        return []
    three = f"With {w}, go slow and steady at first, then go slow and steady later, and go slow and steady at the end."
    two = f"With {w}, go slow and steady at first, then go slow and steady later, and rest at the end."
    c8, c8b = f"{w} {w} helps, so so we start now.", f"{w} truly helps, so so we start now."
    i = f[0]["i"]
    return [("repeat_fire", skel, _copy(texts, i, three), {"REPEAT_4GRAM"}, {}),
            ("repeat_pass", skel, _copy(texts, i, two), "ok", {}),
            ("consec_fire", skel, _copy(texts, i, c8), {"CONSEC_REP"}, {}),
            ("consec_pass", skel, _copy(texts, i, c8b), "ok", {})]


def b_self_copy(skel, texts, built):
    w = _topic_w(skel)
    prev = None
    ok = {t["i"] for t in _fillers(skel, texts)}
    for t in skel["turns"]:
        if t["role"] != "assistant" or t.get("lookup_call"):
            continue
        if prev in ok and t["i"] in ok:
            base = _copy(texts, prev, f"{w} needs a calm plan every day.")
            return [("self_copy_fire", skel, _copy(base, t["i"], f"{w} needs a calm plan, not luck."), {"SELF_COPY"}, {}),
                    ("self_copy_pass", skel, _copy(base, t["i"], f"{w} needs a calm mind and rest."), "ok", {})]
        prev = t["i"]
    return []


def b_echo(skel, texts, built):
    ok = {t["i"] for t in _fillers(skel, texts)}
    for u in filler_user(skel):
        w = _topic_w(skel, u)
        if u["i"] + 1 in ok and u["max_w"] >= 8:
            base = _copy(texts, u["i"], f"tell me more about {w} for a beginner")
            return [("echo_fire", skel, _copy(base, u["i"] + 1, f"Okay, tell me more about {w} for now."), {"ECHO_USER"}, {}),
                    ("echo_pass", skel, _copy(base, u["i"] + 1, f"Okay, tell me more about {w} and we plan."), "ok", {})]
    return []


def b_ngrams(skel, texts, built):
    f = _fillers(skel, texts)
    if not f:
        return []
    i, s = f[0]["i"], texts[f[0]["i"]]
    p = built["persona"].split()
    head = built["blocks"]["head"].split("\n")[0].split(".")[0].split()
    return [("persona_fire", skel, _copy(texts, i, s + " You sound like " + " ".join(p[:8]) + "."), {"PERSONA_LEAK"}, {}),
            ("persona_pass", skel, _copy(texts, i, s + " You sound like " + " ".join(p[:7]) + "."), "ok", {}),
            ("prompt_fire", skel, _copy(texts, i, s + " " + " ".join(head[:6]) + "."), {"PROMPT_ECHO"}, {}),
            ("prompt_pass", skel, _copy(texts, i, s + " " + " ".join(head[:5]) + "."), "ok", {})]


FILL = ("we can start small and then go a bit further every single morning until it feels quite natural for "
        "everyone around here while the rest keeps moving along nicely at its own pace").split()


def b_fine(skel, texts, built):
    """shares just under a threshold (the nearest a legal turn can reach): consec 6/25, self copy 12/25,
    echo 13/22. Each must pass; the lowered-threshold mutants fire on them."""
    out, w = [], _topic_w(skel)
    f = [t for t in _fillers(skel, texts) if t["max_w"] >= 28]
    if f:
        rep = [w, "we", "we", "can", "can", "start", "start", "small", "small", "and", "and", "then", "then"] + FILL[6:18]
        out.append(("consec_024_pass", skel, _copy(texts, f[0]["i"], " ".join(rep) + "."), "ok", {}))
    prev = None
    for t in skel["turns"]:
        if t["role"] != "assistant" or t.get("lookup_call"):
            continue
        if prev is not None and prev in f and t in f:
            p = [w] + FILL[:27]
            c = p[:15] + ["then", "rest", "well"]
            base = _copy(texts, prev["i"], " ".join(p) + ".")
            out.append(("self_copy_048_pass", skel, _copy(base, t["i"], " ".join(c) + "."), "ok", {}))
            break
        prev = t
    for u in filler_user(skel):
        a = [t for t in f if t["i"] == u["i"] + 1]
        if a and u["max_w"] >= 25:
            uw = _topic_w(skel, u)
            ut = [uw] + FILL[:24]
            base = _copy(texts, u["i"], " ".join(ut) + ".")
            out.append(("echo_059_pass", skel, _copy(base, a[0]["i"], " ".join(ut[:16] + ["then", "rest"]) + "."), "ok", {}))
            break
    return out


def b_contains_query(skel, texts, built):
    """a guided contains-query may say "yes or no": that is not restating the answer (must pass)."""
    for e in skel["events"]:
        if e["kind"] == "S2" and e["gold"]["query"] == "contains":
            q = skel["turns"][e["turns"]["query"]]
            if q["mode"] == "guided" and len(texts[q["i"]].split()) + 3 <= q["max_w"]:
                return [("contains_yes_or_no_pass", skel, _copy(texts, q["i"], texts[q["i"]].rstrip("?") + ", yes or no?"),
                         "ok", {})]
    return []


def b_social(skel, texts, built):
    for e in skel["events"]:
        if e["kind"] == "S8" and e["params"]["act"] != "who_are_you":
            t = skel["turns"][e["turns"]["answer"]]
            if not t.get("rules") and not t["must_include"] and t["max_w"] >= 21 and \
                    _free_of_req(skel, texts[t["i"]]):
                base = "Bye, take care " + " ".join(WORDS[:17])
                return [("social_fire", skel, _copy(texts, t["i"], base + " again."), {"LEN_ASSIST"}, {}),
                        ("social_pass", skel, _copy(texts, t["i"], base + "."), "ok", {})]
    return []


def b_case(skel, texts, built):
    for e in _slot_answer_events(skel):
        a, i = e["gold"]["answer"], e["turns"]["answer"]
        if a[:1].isupper() and a != R.card_name(skel):
            low = _sub(texts[i], a, a.lower())
            return [("case_fire", skel, _copy(texts, i, low), {"ANSWER_WRONG"}, {})]
    return []


def b_plural(skel, texts, built):
    rx = word_forms_re(skel["required_words"]["noun"])
    n = skel["required_words"]["noun"]
    new = {i: rx.sub(n + "s", s) for i, s in texts.items()}
    return [("req_plural_pass", skel, new, "ok", {})] if new != texts else []


def b_skel(skel, texts, built):
    out = []
    g = [t for t in skel["turns"] if t["mode"] == "guided" and t["must_include"]]
    if g:
        bad = copy.deepcopy(skel)
        bad["turns"][g[0]["i"]]["max_w"] = 0
        out.append(("infeasible_fire", bad, texts, {"SKEL_INFEASIBLE"}, {}))
    s1 = [n for n, e in enumerate(skel["events"]) if e["kind"] == "S1"]
    if s1:
        bad = copy.deepcopy(skel)
        bad["events"][s1[0]]["params"]["d"] = 11
        out.append(("struct_fire", bad, texts, {"HELDOUT_STRUCT"}, {}))
    out.append(("vocab_ool_report", skel, texts, ("report", {"VOCAB_OOL"}), {"wordlist": {"the"}}))
    return out


BUILDERS = [b_len, b_repeat, b_self_copy, b_echo, b_ngrams, b_fine, b_contains_query, b_social, b_case, b_plural,
            b_skel]
