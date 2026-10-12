"""Planted defects for abstention, role swap, identity, social replies, lists, topic return and lookups, plus the
event-level registry. Same contract as defects_text. FAKE test fixture text."""
import pools as P
import render_prompt as R
from defects_text import _copy, filler_assist
from defects_event import (_events, _sub, _on_answer, ans_wrong, ans_stale, ans_shotgun, ans_echo_q, ans_owner,
                           ans_role, ans_empty, deflect, plant_missing, plant_unbound, corr_missing, value_early, dist_leak,
                           query_restates, query_noref, self_claim, persist, open_end)


def _answer_of(kind, pred=lambda e: True):
    def pick(skel):
        return _events(skel, lambda e: e["kind"] == kind and pred(e))
    return pick


def _variant(v):
    return lambda e: e["params"].get("variant") == v


abstain_guess = _on_answer(lambda k, tx, e, i: _copy(tx, i, "You haven't told me, but maybe it is "
                                                   + (P.POOLS[e["params"]["vtype"]].values[0]
                                                      if e["params"]["vtype"] in P.POOLS else "Biscuit") + "."),
                           _answer_of("S7", _variant("abstain")))
abstain_nohedge = _on_answer(lambda k, tx, e, i: _copy(tx, i, "Okay, sounds good to me."),
                             _answer_of("S7", _variant("abstain")))
swap_claim = _on_answer(lambda k, tx, e, i: _copy(tx, i, f"Oh, I love {e['gold']['user_value']} too."),
                        _answer_of("S6", _variant("role_swap")))
swap_nodeny = _on_answer(lambda k, tx, e, i: _copy(tx, i, "That is a nice thing to ask about."),
                         _answer_of("S6", _variant("role_swap")))
identity_other = _on_answer(lambda k, tx, e, i: _copy(tx, i, f"I'm {'Bram' if R.card_name(k) != 'Bram' else 'Pim'},"
                                                               " an assistant."),
                            _answer_of("S8", lambda e: e["params"]["act"] == "who_are_you"))


def identity_claim(skel, texts, built):
    f = filler_assist(skel)
    other = "Bram" if R.card_name(skel) != "Bram" else "Pim"
    return _copy(texts, f[0]["i"], texts[f[0]["i"]] + f" I'm {other}, by the way.") if f else None


def social_topic(skel, texts, built):
    for e in _events(skel, lambda e: e["kind"] == "S8" and e["params"]["act"] != "who_are_you"):
        i = e["turns"]["answer"]
        req = set(skel["turns"][i]["must_include"])
        vals = [s["value"] for s in skel["slots"].values() if s["value"] not in req
                and s["value"] != R.card_name(skel)]
        if vals:
            return _copy(texts, i, texts[i].rstrip(".") + ", and " + vals[0] + ".")
    return None


def list_wrong(skel, texts, built):
    for e in _events(skel, lambda e: e["kind"] == "S2"):
        i, g = e["turns"]["answer"], e["gold"]
        if g["query"] == "contains":
            return _copy(texts, i, "No, it is not on there now." if g["answer"] == "yes" else "Yes, it is still on.")
        if g["query"] == "count":
            other = "two" if g["answer"] != "two" else "three"
            return _copy(texts, i, _sub(texts[i], g["answer"], other))
        others = [x for x in e["params"]["items"] if x != g["answer"]]
        return _copy(texts, i, _sub(texts[i], g["answer"], others[0]))
    return None


def topic_return(skel, texts, built):
    for e in _events(skel, lambda e: e["kind"] == "S5"):
        i = e["turns"]["answer"]
        return _copy(texts, i, f"Sure, back to {e['gold']['digression'][0]}.")
    return None


def topic_return_both(skel, texts, built):
    for e in _events(skel, lambda e: e["kind"] == "S5"):
        i, g = e["turns"]["answer"], e["gold"]
        return _copy(texts, i, f"Sure, back to {g['origin'][0]}, not {g['digression'][0]}.")
    return None


def list_count_extra(skel, texts, built):
    for e in _events(skel, lambda e: e["kind"] == "S2" and e["gold"]["query"] == "count"):
        i, a = e["turns"]["answer"], e["gold"]["answer"]
        other = "two" if a != "two" else "three"
        return _copy(texts, i, texts[i].rstrip(".") + f", or maybe {other}.")
    return None


def restates_count(skel, texts, built):
    for e in _events(skel, lambda e: e["kind"] == "S2" and e["gold"]["query"] == "count"):
        q = e["turns"]["query"]
        return _copy(texts, q, texts[q].rstrip("?") + ", still " + e["gold"]["answer"] + "?")
    return None


abstain_nooffer = _on_answer(lambda k, tx, e, i: _copy(tx, i, "You haven't told me that yet."),
                             _answer_of("S7", _variant("abstain")))


def _s9(need, result=None):
    return lambda e: e["kind"] == "S9" and e["gold"]["need"] == need and (result is None or e["gold"]["result"] == result)


def lookup_format(skel, texts, built):
    for e in _events(skel, _s9("needed")):
        i = e["turns"]["call"]
        return _copy(texts, i, texts[i].replace("</lookup>", " please</lookup>"))
    return None


def lookup_tool(skel, texts, built):
    for e in _events(skel, _s9("needed")):
        i = e["turns"]["tool"]
        return _copy(texts, i, texts[i].replace("<result>", "<result> "))
    return None


def lookup_missing(skel, texts, built):
    for e in _events(skel, _s9("needed")):
        return _copy(texts, e["turns"]["call"], "Let me check that for you.")
    return None


lookup_unneeded = _on_answer(lambda k, tx, e, i: _copy(tx, i, "Let me look it up first. " + tx[i]),
                             lambda skel: _events(skel, _s9("context")))


def lookup_stray_tag(skel, texts, built):
    f = filler_assist(skel)
    return _copy(texts, f[0]["i"], texts[f[0]["i"]] + " <lookup>weather</lookup>") if f else None


lookup_copy = _on_answer(lambda k, tx, e, i: _copy(tx, i, _sub(tx[i], e["gold"]["answer"], "somewhere")),
                         lambda skel: _events(skel, lambda e: e["kind"] == "S9" and e["gold"]["need"] == "needed"
                                              and e["gold"]["result"] != "empty"))
lookup_empty = _on_answer(lambda k, tx, e, i: _copy(tx, i, "It is probably in " + P.POOLS["city"].values[0] + "."
                                                    if e["params"]["vtype"] == "city" else
                                                    "It is probably " + P.POOLS[e["params"]["vtype"]].values[0] + "."),
                          lambda skel: _events(skel, _s9("needed", "empty")))
lookup_stale = _on_answer(lambda k, tx, e, i: _copy(tx, i, tx[i].rstrip(".") + ", not " + e["gold"]["stale"][0] + "."),
                          lambda skel: _events(skel, _s9("needed", "counterfactual")))

DEFECTS = [("plant_missing", "PLANT_MISSING", plant_missing), ("plant_unbound", "PLANT_UNBOUND", plant_unbound),
           ("corr_missing", "CORR_MISSING", corr_missing),
           ("value_early", "VALUE_EARLY", value_early), ("dist_leak", "DIST_LEAK", dist_leak),
           ("query_restates", "QUERY_RESTATES", query_restates), ("query_noref", "QUERY_NOREF", query_noref),
           ("ans_wrong", "ANSWER_WRONG", _on_answer(ans_wrong)), ("ans_stale", "ANSWER_STALE", _on_answer(ans_stale)),
           ("ans_shotgun", "ANSWER_SHOTGUN", _on_answer(ans_shotgun)), ("ans_echo_q", "ANSWER_WRONG", _on_answer(ans_echo_q)),
           ("ans_owner", "PERSPECTIVE", _on_answer(ans_owner)), ("ans_role", "ROLE_LABEL", ans_role),
           ("ans_empty", "EMPTY_TURN", ans_empty), ("deflect", "DEFLECT", deflect),
           ("claim_plans", "SELF_CLAIM", self_claim("I'm planning a trip this weekend.")),
           ("claim_family", "SELF_CLAIM", self_claim("My sister says the same.")),
           ("claim_body", "SELF_CLAIM", self_claim("My back is sore today.")),
           ("claim_past", "SELF_CLAIM", self_claim("I went through that last year.")),
           ("claim_prefs", "SELF_CLAIM", self_claim("I love quiet evenings.")),
           ("claim_places", "SELF_CLAIM", self_claim("I live by the sea.")),
           ("persist_max_words", "PERSIST_FAIL", persist("max_words")),
           ("persist_one_sentence", "PERSIST_FAIL", persist("one_sentence")),
           ("persist_end_question", "PERSIST_FAIL", persist("end_question")),
           ("persist_call_user", "PERSIST_FAIL", persist("call_user")),
           ("persist_avoid_word", "PERSIST_FAIL", persist("avoid_word")),
           ("persist_start_name", "PERSIST_FAIL", persist("start_name")), ("open_end", "OPEN_END", open_end),
           ("abstain_guess", "ABSTAIN_MISSING", abstain_guess), ("abstain_nohedge", "ABSTAIN_MISSING", abstain_nohedge),
           ("swap_claim", "PERSPECTIVE", swap_claim), ("swap_nodeny", "ANSWER_WRONG", swap_nodeny),
           ("identity_other", "IDENTITY", identity_other), ("identity_claim", "IDENTITY", identity_claim),
           ("social_topic", "SOCIAL_TOPIC", social_topic), ("list_wrong", "LIST_STATE", list_wrong),
           ("topic_return", "TOPIC_RETURN", topic_return), ("lookup_format", "LOOKUP_FORMAT", lookup_format),
           ("lookup_tool", "LOOKUP_FORMAT", lookup_tool), ("lookup_missing", "LOOKUP_MISSING", lookup_missing),
           ("lookup_unneeded", "LOOKUP_UNNEEDED", lookup_unneeded),
           ("lookup_stray_tag", "LOOKUP_UNNEEDED", lookup_stray_tag), ("lookup_copy", "LOOKUP_COPY", lookup_copy),
           ("lookup_empty", "LOOKUP_EMPTY", lookup_empty), ("lookup_stale", "ANSWER_STALE", lookup_stale),
           ("topic_return_both", "TOPIC_RETURN", topic_return_both), ("list_count_extra", "LIST_STATE", list_count_extra),
           ("restates_count", "QUERY_RESTATES", restates_count), ("abstain_nooffer", "ABSTAIN_MISSING", abstain_nooffer)]
