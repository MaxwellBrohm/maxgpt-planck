"""Behaviour checks (SPEC section 6): perspective, self claims, rule persistence, open endings, topicality,
abstention, role-swap denial, identity, social replies and lookups."""
import re

import lexicons as L
import parse
import pools as P
from banks_keys import KEYS
from check_base import content, sentences, mentioned, value_re_i

NAME_CLAIM_RE = re.compile(r"(?<![A-Za-z])(?:I'm|I am|my name is|my name's|call me|I'm called|I am called)\s+"
                           r"([A-Z][a-z]+)")
FILLER_ASSIST = ("reply on the topic", "respond helpfully on the first topic", "respond on the second topic",
                 "greet back and engage with the topic")


def _guided_assist(ctx):
    return [(t, s) for t, s in ctx.turns(role="assistant", mode="guided")]


def chk_perspective(ctx):
    out = []
    pats = []
    for s in ctx.skel["slots"].values():
        if s["owner"] == "assistant" or s["key"] not in KEYS or s["key"] in ("assistant_name",):
            continue
        k = KEYS[s["key"]]
        if s["key"] == "user_name":
            v = re.escape(s["value"])
            pats.append(re.compile(r"(?<![A-Za-z])(?:my name is|my name's|I'm|I am)\s+" + v + r"(?![A-Za-z])", re.I))
        elif k["noun"] and s.get("noun"):
            pats.append(re.compile(r"(?<![A-Za-z])my " + re.escape(s["noun"]) + r"(?![A-Za-z])", re.I))
        elif not k["noun"]:
            pats.append(re.compile(r"(?<![A-Za-z])my " + re.escape(k["label"]) + r"(?![A-Za-z])", re.I))
    for t, s in _guided_assist(ctx):
        for rx in pats:
            if rx.search(s):
                out.append(("PERSPECTIVE", t["i"], rx.search(s).group(0)))
                break
    for e in ctx.skel["events"]:
        i = e["turns"].get("answer")
        if e["kind"] == "S6" and e["params"]["variant"] == "role_swap" and i in ctx.text:
            v = e["gold"]["user_value"]
            for sent in sentences(ctx.text[i]):
                if value_re_i(v).search(sent) and re.search(r"\b(?:I|my|me)\b", sent) and \
                        not re.search(r"\byou(?:r|'re)?\b", sent, re.I):
                    out.append(("PERSPECTIVE", i, f"claims {v}"))
            if not L.DENY_RE.search(ctx.text[i]):
                out.append(("ANSWER_WRONG", i, f"{e['id']} no denial"))
    return out


def chk_self_claim(ctx):
    return [("SELF_CLAIM", t["i"], f"{c}: {m}") for t, s in _guided_assist(ctx) for c, m in L.self_claims(s)[:1]]


def _verify(ctx, i, rule):
    s, a = ctx.text[i], rule["args"]
    v = rule["verifier"]
    if v == "max_words":
        return len(s.split()) <= a["max"]
    if v == "one_sentence":
        return len(sentences(s)) == 1
    if v == "end_question":
        return s.rstrip().endswith("?")
    if v == "call_user":
        return ctx.has(i, a["name"])
    if v == "avoid_word":
        return not value_re_i(a["word"]).search(s)
    if v == "start_name":
        return re.match(r"\s*" + re.escape(a["name"]) + r"(?![A-Za-z])", s) is not None
    raise ValueError(v)


def chk_persist(ctx):
    out = []
    for t, s in ctx.turns(role="assistant"):
        for r in t.get("rules", []):
            if not _verify(ctx, t["i"], r):
                out.append(("PERSIST_FAIL", t["i"], r["verifier"]))
        if t["mode"] == "guided" and "end with an open question" in (t["intent"] or "") and not s.rstrip().endswith("?"):
            out.append(("OPEN_END", t["i"], ""))
    return out


def chk_offtopic(ctx):
    out, topics = [], ctx.skel["topic_text"]
    all_topic = set().union(*(content(x) for x in topics.values()))
    prev_user = None
    for t in ctx.skel["turns"]:
        s = ctx.text.get(t["i"])
        if t["role"] == "user":
            prev_user = s
        if s is None or t["mode"] != "guided":
            continue
        it = t["intent"] or ""
        if t["role"] == "user" and it.startswith("topic:"):
            want = content(topics[it.split(":")[1]])
        elif t["role"] == "user" and it == "open the chat about the topic":
            want = content(topics[ctx.skel["topic_path"][0]])
        elif t["role"] == "assistant" and it.split(";")[0] in FILLER_ASSIST:
            want = all_topic | content(prev_user or "")
        else:
            continue
        if not content(s) & want:
            out.append(("OFFTOPIC", t["i"], it[:30]))
    return out


def _pool_guess(ctx, i, vtype):
    pool = P.POOLS.get(vtype)
    return mentioned(ctx, i, pool.values) if pool else []


def chk_abstain(ctx):
    out = []
    for e in ctx.skel["events"]:
        i = e["turns"].get("answer")
        if e["kind"] != "S7" or e["params"]["variant"] != "abstain" or i not in ctx.text:
            continue
        s = ctx.text[i]
        guess = mentioned(ctx, i, e["gold"]["candidates"]) or _pool_guess(ctx, i, e["params"]["vtype"])
        if not L.HEDGE_RE.search(s):
            out.append(("ABSTAIN_MISSING", i, "no hedge"))
        elif not (L.OFFER_RE.search(s) or s.rstrip().endswith("?")):
            out.append(("ABSTAIN_MISSING", i, "no offer"))
        if guess:
            out.append(("ABSTAIN_MISSING", i, f"guess {guess}"))
    return out


def chk_identity(ctx):
    out = []
    for t, s in _guided_assist(ctx):
        for m in NAME_CLAIM_RE.finditer(s):
            if not ctx.has_sys or m.group(1) != ctx.card:
                out.append(("IDENTITY", t["i"], m.group(0)))
    for e in ctx.skel["events"]:
        i = e["turns"].get("answer")
        if e["kind"] != "S8" or i not in ctx.text:
            continue
        if e["params"]["act"] == "who_are_you":
            if not ctx.has(i, ctx.card):
                out.append(("IDENTITY", i, "no card name"))
            continue
        req = set(ctx.by_i[i]["must_include"])
        vals = mentioned(ctx, i, [s["value"] for s in ctx.skel["slots"].values() if s["value"] not in req])
        if vals:
            out.append(("SOCIAL_TOPIC", i, str(vals)))
    return out


def chk_lookup(ctx):
    out, sched = [], set()
    for e in ctx.skel["events"]:
        if e["kind"] != "S9":
            continue
        g, tr = e["gold"], e["turns"]
        i = tr.get("answer")
        if g["need"] == "context":
            if i in ctx.text and L.LOOKUP_TALK_RE.search(ctx.text[i]):
                out.append(("LOOKUP_UNNEEDED", i, L.LOOKUP_TALK_RE.search(ctx.text[i]).group(0)))
            continue
        sched |= {tr["call"], tr["tool"]}
        for j in (tr["call"], tr["tool"]):
            if j not in ctx.text:
                continue
            if j == tr["call"] and not L.TAG_RE.search(ctx.text[j]):
                out.append(("LOOKUP_MISSING", j, ctx.text[j][:30]))
            elif parse.normalize(ctx.text[j]) != parse.normalize(ctx.by_i[j]["text"]):
                out.append(("LOOKUP_FORMAT", j, ctx.text[j][:30]))
        if i not in ctx.text:
            continue
        if g["result"] == "empty":
            guess = _pool_guess(ctx, i, e["params"]["vtype"])
            if not L.NOT_FOUND_RE.search(ctx.text[i]) or guess:
                out.append(("LOOKUP_EMPTY", i, f"guess {guess}"))
        elif not ctx.has(i, g["answer"]):
            out.append(("LOOKUP_COPY", i, f"lacks {g['answer']}"))
        if g.get("stale") and mentioned(ctx, i, g["stale"]):
            out.append(("ANSWER_STALE", i, f"{e['id']} {g['stale']}"))
    for t, s in ctx.turns():
        if t["i"] not in sched and L.TAG_RE.search(s):
            out.append(("LOOKUP_UNNEEDED", t["i"], "unscheduled tag"))
    return out


CHECKS = [chk_perspective, chk_self_claim, chk_persist, chk_offtopic, chk_abstain, chk_identity, chk_lookup]
