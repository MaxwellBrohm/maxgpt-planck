"""Round 4 checks (2026-10-04, SPEC 16; the dry pilot 4 v1 quality review, notes.txt). Same contract as the other
checker modules: f(ctx) -> [(code, turn index or None, detail)]. Patterns are in lexicons_r4.py. Every check here
adds rejections (D2: nothing is loosened); no new code: each hit carries an existing code and a detail that names
the round 4 rule.
  FALSE_MEMORY  "never said": the assistant says the user said or called it a value no user line holds (HIGH 1);
                "remember": "I remember" a value said one line before (HIGH 5)
  PROMPT_ECHO   a first-person copy of a guidance frame ("correct what I said before about ..."), and "Ask me to ..."
                for "ask the assistant to ..." (HIGH 2, task b)
  ASSIST_CASE   a capitalized value written in lowercase inside an assistant turn (D1; task d)
  ASSIST_VOICE  the user named in the third person ("Hana says your class ...", HIGH 4)
  PERSPECTIVE   the user's name or thing claimed ("my name is really Arlo", "My club begins in January", HIGH 5)
  PLANT_MISSING a rule ask with no item that never states its rule (HIGH 3)
  ANSWER_WRONG  a person given the wrong relation, or a person's job used as a person ("your gardener", HIGH 5)
  AI_ISM        the service substitutes and closers round 3 missed (MEDIUM 6; the sampling ban's family)"""
import re

import check_r3 as R3
import golds
import lexicons as L
import lexicons_r4 as L4
import pools as P
import render_intents as RI
from check_base import sentences, value_re_i
from check_text import script_guidance

FIRST_PERSON = {"you": "(?:you|i|me)", "yourself": "(?:yourself|myself)", "you're": "(?:you're|i'm)",
                "yours": "(?:yours|mine)"}
HEAD_SKIP = {"case", "night", "day", "time", "way", "end", "part"}   # "in my case": idioms, not a claimed thing


def _skips(sk):
    """(abstain answer turns, assistant_err recap turns): planned to deny or to misstate."""
    ab = {e["turns"].get("answer") for e in sk["events"] if e["kind"] == "S7" and e["params"].get("variant") == "abstain"}
    rc = {e["turns"].get("recap_reply") for e in sk["events"] if e["kind"] == "S3"
          and e["params"].get("variant") == "assistant_err"}
    return ab, rc


# a clause where the assistant turns to itself ("You asked me that after you mentioned your name, and I am Ember":
# the pointer is about the user's name, the card name is the assistant's own clause; Gemma, dp4 re-check)
SELF_CLAUSE = re.compile(r",?\s+(?:and|but|so)\s+(?=(?:I|I'm|my)\b)|;\s*")


def _user_said(ctx, v, i):
    """a user line before turn i holds v, case folded (a typos-style user may write "denver")."""
    return any(ctx.by_i[j]["role"] == "user" and value_re_i(v).search(ctx.text[j]) for j in ctx.text if j < i)


def chk_never_said(ctx):
    """FALSE_MEMORY: a sentence that says the user said, told or called it something, naming values none of which a
    user line before it holds and none of which came from a lookup result."""
    ab, rc = _skips(ctx.skel)
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        i = t["i"]
        if t.get("lookup_call") or i in rc or i in ab:
            continue
        for sent in (c for x in sentences(s) for c in SELF_CLAUSE.split(x)):
            m = L4.USER_SAID_RE.search(sent)
            vals = [v for v in R3._names(ctx) if golds.value_re(v).search(sent)] if m else []
            if vals and not any(_user_said(ctx, v, i) or R3._first_source(ctx, v, i) == "tool" for v in vals):
                out.append(("FALSE_MEMORY", i, "never said: " + m.group(0)))
                break
    return out


def chk_remember(ctx):
    """FALSE_MEMORY: "I remember" when every value the turn names was first said in the line just before."""
    _, rc = _skips(ctx.skel)
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        i = t["i"]
        m = L4.REMEMBER_RE.search(s) if not (t.get("lookup_call") or i in rc) else None
        vals = [v for v in R3._names(ctx) if ctx.has(i, v)] if m else []
        if vals and not R3._said_before(ctx, vals, i - 1):
            out.append(("FALSE_MEMORY", i, "remember: " + m.group(0)))
    return out


def key_masks(skel):
    """the slot phrases the guidance names ("what time your museum trip starts"): the content a line must carry."""
    m = [RI.phrase(s) for s in skel["slots"].values() if s["key"] in RI.KEY_PHRASE]
    m += [RI.phrase({"key": e["params"]["key"], "noun": e["params"].get("noun")}) for e in skel["events"]
          if e["params"].get("key") in RI.KEY_PHRASE]
    return sorted({x.strip() for x in m if x.strip()}, key=len, reverse=True)


def first_frame(skel, g):
    """the frame of the guidance with its slot phrases as holes too (R3.guide_frame does the rest)."""
    for x in key_masks(skel):
        g = re.sub(r"(?<![A-Za-z])" + re.escape(x) + r"(?![A-Za-z])", f" {R3.HOLE} ", g, flags=re.I)
    return R3.guide_frame(skel, g)


def first_frame_re(toks):
    """R3.frame_re with the guidance's "you" read as the user's I or me; None for a frame with no second person."""
    req = [w for w in toks if w != R3.HOLE and w not in R3.FRAME_OPTIONAL]
    if not any(w in FIRST_PERSON for w in req) or len(req) < R3.FRAME_MIN_REQ \
            or len([w for w in req if w not in L.STOPWORDS]) < R3.FRAME_MIN_CONTENT:
        return None
    parts, hole = [], False
    for w in toks:
        if w == R3.HOLE:
            hole = True
            continue
        if w in R3.FRAME_OPTIONAL:
            continue
        if parts:
            parts.append(r"\W+(?:[\w']+\W+){0,%d}" % (R3.FRAME_HOLE if hole else R3.FRAME_GAP))
        parts.append(FIRST_PERSON.get(w, re.escape(w)))
        hole = False
    return re.compile(r"(?<![\w'])" + "".join(parts) + r"(?![\w'])", re.I)


def chk_first_person_frame(ctx):
    """PROMPT_ECHO: a guided user line that carries its guidance's frame in the first person, or opens "Ask me" for
    guidance that tells the user to ask the assistant (list guidance excepted, as in round 3)."""
    if not ctx.built:
        return []
    out = []
    for t, s in ctx.turns(role="user", mode="guided"):
        if (t["intent"] or "").startswith(R3.NATURAL_Q):
            continue
        g = script_guidance(ctx.built, t)
        lead = L4.ask_me_lead(s)
        if lead and re.match(r"\s*ask\b", g, re.I):
            out.append(("PROMPT_ECHO", t["i"], "guidance lead: " + lead.group(0).strip()))
            continue
        old = R3.frame_re(R3.guide_frame(ctx.skel, g))
        rx = None if old and old.search(s) else first_frame_re(first_frame(ctx.skel, g))
        m = rx.search(s) if rx else None
        if m:
            out.append(("PROMPT_ECHO", t["i"], "first-person frame: " + m.group(0)))
    return out


def chk_case_names(ctx):
    """ASSIST_CASE: a capitalized value of the chat written in lowercase in an assistant turn (everyday-word values
    such as May or Pepper are left to REQ_SPAN)."""
    vals = [v for v in R3._names(ctx) if v[:1].isupper() and len(v) > 2 and v.lower() not in L4.AMBIG_LOWER]
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        if t.get("lookup_call"):
            continue
        bad = next((m.group(0) for v in vals for m in value_re_i(v).finditer(s) if not m.group(0).startswith(v)), None)
        if bad:
            out.append(("ASSIST_CASE", t["i"], "lowercase name: " + bad))
    return out


def _user_names(ctx):
    names = {ctx.uname} if ctx.uname else set()
    names |= {seg["args"]["name"] for e in ctx.skel["events"] if e["kind"] == "S4" for seg in e["params"]["segments"]
              if seg["verifier"] in ("call_user", "start_name")}
    others = {s["value"] for s in ctx.skel["slots"].values() if s["key"] not in ("user_name", "nickname")}
    return sorted(n for n in names if n and n not in others)


def chk_user_third(ctx):
    """ASSIST_VOICE: the user's name or nickname followed by a verb of the user's own doing, no comma between."""
    rxs = [re.compile(r"(?<![A-Za-z'])" + re.escape(n) + r"\s+" + L4.THIRD_VERB + r"(?![A-Za-z])", re.I)
           for n in _user_names(ctx)]
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        m = next((m for rx in rxs for m in [rx.search(s)] if m), None)
        if m:
            out.append(("ASSIST_VOICE", t["i"], "user in third person: " + m.group(0)))
    return out


def chk_claims_user(ctx):
    """PERSPECTIVE: "my name is really <the user's name>", or "my <head>" for a user thing named by two words or more
    ("my club" for the user's book club); one-word nouns are check_behav's."""
    pats = []
    if ctx.uname:
        pats.append(re.compile(r"(?<![A-Za-z])(?:my name is|my name's|I'm|I am|call me)\s+" + L4.NAME_ADV
                               + re.escape(ctx.uname) + r"(?![A-Za-z])", re.I))
    heads = {s["noun"].split()[-1] for s in ctx.skel["slots"].values()
             if s["owner"] != "assistant" and s.get("noun") and len(s["noun"].split()) > 1}
    heads -= HEAD_SKIP | L.STOPWORDS
    if heads:
        pats.append(re.compile(r"(?<![A-Za-z])my (?:" + "|".join(re.escape(h) for h in sorted(heads)) + r")(?![A-Za-z])",
                               re.I))
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        m = next((m for rx in pats for m in [rx.search(s)] if m), None)
        if m:
            out.append(("PERSPECTIVE", t["i"], "claims the user's: " + m.group(0)))
    return out


def chk_rule_stated(ctx):
    """PLANT_MISSING: a guided rule ask (or override) for a rule with no required item that never names the rule."""
    out = []
    for e in ctx.skel["events"]:
        if e["kind"] != "S4":
            continue
        for k, seg in enumerate(e["params"]["segments"]):
            i = e["turns"].get("rule" if k == 0 else "override")
            rx = L4.RULE_STATES.get(seg["verifier"])
            if rx and i in ctx.text and ctx.by_i[i]["mode"] == "guided" and not rx.search(ctx.text[i]):
                out.append(("PLANT_MISSING", i, "rule not stated: " + seg["verifier"]))
    return out


def chk_relation(ctx):
    """ANSWER_WRONG: "your <relation>" right before a person who has another relation ("your friend is actually
    Greta", Greta the coworker), or "your <job>" for a person's job ("your gardener", the uncle's job)."""
    rels = set(P.POOLS["relation"].values)
    pairs = [(s["value"], s["noun"]) for s in ctx.skel["slots"].values() if s["key"] == "person_name" and s.get("noun")]
    jobs = {s["value"] for s in ctx.skel["slots"].values() if s["key"] == "person_job"} - rels
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        hit = None
        for v, r in pairs:
            m = re.search(r"(?<![A-Za-z])your (\w+)(?:'s name)?(?:,| is| was| named| called| actually| really| still| "
                          r"now){0,%d} %s(?![A-Za-z])" % (L4.REL_GAP, re.escape(v)), s, re.I)
            if m and m.group(1).lower() in rels and m.group(1).lower() != r.lower():
                hit = m.group(0)
        for j in jobs:
            m = re.search(r"(?<![A-Za-z])your " + re.escape(j) + r"(?![A-Za-z'])(?! (?:job|work|career|role|shift|"
                          + "|".join(sorted(rels)) + r")(?![a-z]))", s, re.I)   # "your florist daughter" is right
            hit = hit or (m and m.group(0))
        if hit:
            out.append(("ANSWER_WRONG", t["i"], "wrong person: " + hit))
    return out


def chk_service(ctx):
    """AI_ISM on assistant turns: lexicons_r4.SERVICE_RE (lookup call lines excepted, as in check_lines)."""
    return [("AI_ISM", t["i"], L4.SERVICE_RE.search(s).group(0)) for t, s in ctx.turns(role="assistant", mode="guided")
            if not t.get("lookup_call") and L4.SERVICE_RE.search(s)]


CHECKS = [chk_never_said, chk_remember, chk_first_person_frame, chk_case_names, chk_user_third, chk_claims_user,
          chk_rule_stated, chk_relation, chk_service]
