"""Planted defects, event and behaviour level. For every answer check the SPEC plants: an empty reply, a wrong
in-pool value, a shotgun answer, an echo of the question, a stale value, a wrong owner and a leaked role label.
Same contract as defects_text: (skel, texts, built) -> mutated {turn: text} or None. FAKE test fixture text."""
import re

import golds
import pools as P
from banks_keys import KEYS
from check_base import event_values, first_scheduled
from check_events import slot_event
from defects_text import _copy, filler_assist, NEUTRAL


def _events(skel, pred):
    return [e for e in skel["events"] if pred(e)]


def _sub(text, v, new):
    """replace value v in text, case-insensitively (a lowercase-style user writes values in lowercase)."""
    return re.compile(golds.value_re(v).pattern, re.I).sub(new, text)


def _slot_answer_events(skel):
    return _events(skel, lambda e: slot_event(e) and isinstance(e["gold"].get("answer"), str))


def _not_card(skel):
    card = skel["slots"][skel["assistant"]["name"]]["value"]
    return [e for e in _slot_answer_events(skel) if e["gold"]["answer"] != card]


def _on_answer(fn, pick=_slot_answer_events):
    def d(skel, texts, built):
        for e in pick(skel):
            out = fn(skel, texts, e, e["turns"]["answer"])
            if out is not None:
                return out
        return None
    return d


def _wrong_value(skel, v):
    vt = next((s["type"] for s in skel["slots"].values() if s["value"] == v), None)
    used = event_values(skel)
    pool = P.POOLS.get(vt)
    return next((x for x in pool.values if x not in used and not golds.value_re(x).search(v)), None) if pool else None


def ans_wrong(skel, texts, e, i):
    w = _wrong_value(skel, e["gold"]["answer"])
    return _copy(texts, i, _sub(texts[i], e["gold"]["answer"], w)) if w else None


def ans_stale(skel, texts, e, i):
    st = e["gold"].get("stale")
    return _copy(texts, i, _sub(texts[i], e["gold"]["answer"], st[0])) if st else None


def ans_shotgun(skel, texts, e, i):
    c = [x for x in e["gold"].get("candidates", []) + e["gold"].get("stale", [])
         if x not in ctx_required(skel, i)]
    return _copy(texts, i, texts[i].rstrip(".?!") + ", or maybe " + " or ".join(c) + ".") if c else None


def ctx_required(skel, i):
    return set(skel["turns"][i]["must_include"])


def ans_echo_q(skel, texts, e, i):
    return _copy(texts, i, texts[e["turns"]["query"]])


def ans_owner(skel, texts, e, i):
    sid = e["gold"].get("slot") or e["params"].get("slot")
    s = skel["slots"].get(sid) if sid else None
    if not s or s["key"] not in KEYS or s["key"] == "user_name":
        return None
    who = s["noun"] if KEYS[s["key"]]["noun"] else KEYS[s["key"]]["label"]
    return _copy(texts, i, f"My {who} is {e['gold']['answer']}, I think.")


ans_role = _on_answer(lambda k, tx, e, i: _copy(tx, i, "Assistant: " + tx[i]))
ans_empty = _on_answer(lambda k, tx, e, i: _copy(tx, i, ""))
deflect = _on_answer(lambda k, tx, e, i: _copy(tx, i, "I can't remember much, but " + tx[i][:1].lower() + tx[i][1:]))


def _ops_turns(skel, kinds_ops):
    for e in skel["events"]:
        for op in e["params"].get("ops", []):
            if op["op"] in kinds_ops:
                yield e, op, e["turns"][op["turn"]]


def plant_missing(skel, texts, built):
    for e, op, i in _ops_turns(skel, {"plant"}):
        return _copy(texts, i, _sub(texts[i], op["value"], "something"))
    return None


def plant_unbound(skel, texts, built):
    for e, op, i in _ops_turns(skel, {"plant"}):
        s = skel["slots"].get(op.get("slot") or "")
        t = skel["turns"][i]
        if s and s.get("noun") and t["mode"] == "guided" and t["role"] == "user":
            return _copy(texts, i, re.sub(r"(?i)\bmy " + re.escape(s["noun"]) + r"\b", "it", texts[i]))
    return None


def corr_missing(skel, texts, built):
    for e, op, i in _ops_turns(skel, {"set", "fix", "twin", "err", "add", "remove", "move_first", "move_last"}):
        return _copy(texts, i, _sub(texts[i], op["value"], "that one"))
    return None


def value_early(skel, texts, built):
    first = first_scheduled(skel)
    for v, f in sorted(first.items(), key=lambda x: -x[1]):
        early = [t for t in filler_assist(skel) if t["i"] < f]
        if early:
            i = early[0]["i"]
            return _copy(texts, i, texts[i].rstrip(".") + ", like " + v + ".")
    return None


def dist_leak(skel, texts, built):
    for e in _not_card(skel):
        a, q = e["gold"]["answer"], e["turns"]["query"]
        if e["kind"] == "S9":
            last = e["turns"]["context"]
        else:
            queried = e["params"].get("queried") or e["params"].get("slot")
            last = max(e["turns"][op["turn"]] for op in e["params"]["ops"]
                       if op.get("slot") == queried and op["op"] in ("plant", "set", "fix"))
        for t in skel["turns"]:
            if last < t["i"] < q and e["id"] not in t["events"] and t["mode"] == "guided" and not t["events"] \
                    and a not in t["must_include"]:
                return _copy(texts, t["i"], texts[t["i"]].rstrip(".?") + ", like " + a + ".")
    return None


def query_restates(skel, texts, built):
    for e in _not_card(skel):
        q = e["turns"]["query"]
        return _copy(texts, q, texts[q].rstrip("?") + ", is it " + e["gold"]["answer"] + "?")
    return None


def query_noref(skel, texts, built):
    vals = event_values(skel)
    for e in skel["events"]:
        q = e["turns"].get("query")
        t = skel["turns"][q] if q is not None else None
        if t and t["mode"] == "guided":
            refs = [x for x in t["must_include"] if x not in vals]
            if refs:
                return _copy(texts, q, re.sub(re.escape(refs[0]), "that", texts[q], flags=re.I))
    return None


def self_claim(sentence):
    def d(skel, texts, built):
        f = filler_assist(skel)
        return _copy(texts, f[0]["i"], texts[f[0]["i"]] + " " + sentence) if f else None
    return d


def _rule_turn(skel, verifier):
    for t in skel["turns"]:
        for r in t.get("rules", []):
            if r["verifier"] == verifier and t["mode"] == "guided":
                return t, r["args"]
    return None, None


def persist(verifier):
    def d(skel, texts, built):
        t, a = _rule_turn(skel, verifier)
        if not t:
            return None
        s = texts[t["i"]]
        new = {"max_words": lambda: s + " " + " ".join(NEUTRAL[:a.get("max", 0)]),
               "one_sentence": lambda: s + " Here is one more sentence.",
               "end_question": lambda: s.rstrip("?") + ".",
               "call_user": lambda: _sub(s, a.get("name", ""), "friend"),
               "avoid_word": lambda: s + " That is " + a.get("word", "") + " it.",
               "start_name": lambda: "So, " + s}[verifier]()
        return _copy(texts, t["i"], new)
    return d


def open_end(skel, texts, built):
    t = next((t for t in skel["turns"] if t["mode"] == "guided" and "open question" in (t["intent"] or "")
              and not any(r["verifier"] == "end_question" for r in t.get("rules", []))), None)
    return _copy(texts, t["i"], texts[t["i"]].rstrip("?") + ".") if t else None
