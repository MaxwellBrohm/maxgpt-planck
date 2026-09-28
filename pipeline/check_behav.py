"""Behaviour checks (SPEC section 6): perspective, self claims, rule persistence, open endings, topicality,
abstention, role-swap denial, identity and social replies (lookups: check_lookup.py, moved 2026-09-28)."""
import re

import lexicons as L
import pools as P
import topic_words as TW
from banks_keys import KEYS
from check_base import content, sentences, mentioned, value_re_i
from stemmer import topic_forms

NAME_CLAIM_RE = re.compile(r"(?<![A-Za-z])(?:I'm|I am|my name is|my name's|call me|I'm called|I am called)\s+"
                           r"([A-Z][a-z]+)")
# a user topic turn off its own topic passes on the assistant's last reply only when it shares this many content stems
# with it, or when the shared ones are this share of its own (2026-09-28, v1 review: "Speaking of assistant, my meal is
# acting up again" passed on one echoed word; dp2 recheck: one-word reactions such as "I could eat X every day" are
# replies to that line and share half of their content)
PREV_ASSIST_MIN = 2
PREV_ASSIST_SHARE = 0.5
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
    out += _user_perspective(ctx)
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


def own_nouns(skel):
    """what the user owns and names with a noun: slot nouns of user slots, the user's persons, list names."""
    ns = {s["noun"] for s in skel["slots"].values() if s["owner"] != "assistant" and s.get("noun")}
    ns |= {p["relation"] for p in skel["persons"].values()}
    ns |= {e["params"]["list_name"] for e in skel["events"] if e["params"].get("list_name")}
    return sorted(ns, key=len, reverse=True)


def own_labels(skel):
    """{turn index: labels} of the user's own facts named without a noun (home city, job, hobby, favourite food or
    colour; never "name", which the user may ask the assistant for), on the turns of an event about that fact."""
    out = {}
    for e in skel["events"]:
        p = e["params"]
        sids = {op.get("slot") for op in p.get("ops", [])} | {p.get("slot"), p.get("queried")}
        labs = {KEYS[s["key"]]["label"] for sid, s in skel["slots"].items() if sid in sids and s["owner"] == "user"
                and s["key"] in KEYS and not KEYS[s["key"]]["noun"] and s["key"] != "user_name"}
        if e["kind"] == "S7" and p.get("key") in KEYS and not KEYS[p["key"]]["noun"] and p["key"] != "user_name":
            labs.add(KEYS[p["key"]]["label"])
        for i in e["turns"].values():
            out.setdefault(i, set()).update(labs)
    return out


def _user_perspective(ctx):
    """PERSPECTIVE on a guided user turn: the user calls one of their own things "your X" (the guidance's second
    person copied into the line: "what is last on your shopping list?"), or, on a turn about one of their own facts,
    its label ("What is your favourite colour brown?", 2026-09-28). The role-swap question asks about the assistant's
    own thing on purpose and is exempt."""
    swap_q = {e["turns"].get("query") for e in ctx.skel["events"]
              if e["kind"] == "S6" and e["params"].get("variant") == "role_swap"}
    ns, labs, out = own_nouns(ctx.skel), own_labels(ctx.skel), []
    for t, s in ctx.turns(role="user", mode="guided"):
        words = ns + sorted(labs.get(t["i"], ()), key=len, reverse=True)
        if t["i"] in swap_q or not words:
            continue
        m = re.search(r"(?<![A-Za-z])your (?:" + "|".join(re.escape(n) for n in words) + r")(?![A-Za-z])", s, re.I)
        if m:
            out.append(("PERSPECTIVE", t["i"], m.group(0)))
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


def topic_set(text):
    """stemmed content words of a topic text plus its topic word set (topic_words, FAKE until the bank pass) and the
    forms stemmer.topic_forms adds to that set's words by part of speech (comparatives, -ly, -y bases, agents)."""
    forms = [f for p in TW.PARTS for w in TW.words(text, p) for f in topic_forms(w, p)]
    return content(text) | content(" ".join(TW.related(text))) | content(" ".join(forms))


def chk_offtopic(ctx):
    """OFFTOPIC: a guided topic turn shares no stemmed content word with what it should be about. A user topic turn
    may pick up its topic or the assistant's last reply (PREV_ASSIST_MIN words, or PREV_ASSIST_SHARE of its own content
    words); an assistant filler reply may pick up any topic of the chat or the user's last line (audit 2026-09-27:
    synonyms and replies to the previous turn were read as off topic)."""
    out, topics = [], ctx.skel["topic_text"]
    all_topic = set().union(*(topic_set(x) for x in topics.values()))
    prev_user = prev_assist = None
    for t in ctx.skel["turns"]:
        s = ctx.text.get(t["i"])
        it = t["intent"] or ""
        want = None
        if s is not None and t["mode"] == "guided":
            if t["role"] == "user" and it.startswith("topic:"):
                want = topic_set(topics[it.split(":")[1]])
                shared = content(s) & content(prev_assist or "")
                if len(shared) >= PREV_ASSIST_MIN or len(shared) >= PREV_ASSIST_SHARE * len(content(s)) > 0:
                    want = want | content(prev_assist or "")
            elif t["role"] == "user" and it == "open the chat about the topic":
                want = topic_set(topics[ctx.skel["topic_path"][0]])
            elif t["role"] == "assistant" and it.split(";")[0] in FILLER_ASSIST:
                want = all_topic | content(prev_user or "")
        if want is not None and not content(s) & want:
            out.append(("OFFTOPIC", t["i"], it[:30]))
        if t["role"] == "user":
            prev_user = s
        elif t["role"] == "assistant" and not t.get("lookup_call"):
            prev_assist = s
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
        out += _abstain_query_value(ctx, e)
    return out


def _abstain_query_value(ctx, e):
    """QUERY_RESTATES on an abstain query: the user states a value of the asked type that the chat never gives
    (2026-09-28, v1 review: "My favourite colour is red. What's your favourite colour?" answered "it wasn't
    mentioned"; the hedge then contradicts the user)."""
    q = e["turns"].get("query")
    pool = P.POOLS.get(e["params"]["vtype"])
    if q not in ctx.text or ctx.by_i[q]["mode"] != "guided" or not pool:
        return []
    return [("QUERY_RESTATES", q, f"{e['id']} states {v}") for v in mentioned(ctx, q, set(pool.values) - ctx.values)]


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


CHECKS = [chk_perspective, chk_self_claim, chk_persist, chk_offtopic, chk_abstain, chk_identity]   # lookups: check_lookup
