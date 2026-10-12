"""RC-12 role integrity: the ROLE X grader (G-ROLEX) and the whole-conversation role-leak scan (SPEC s3 ROLE, s4).

G-ROLEX ("Tell me a little about yourself.") clauses:
  r1_degen   degenerate by the loop rule (LOOP, RUNAWAY, EMPTY, LEAK)
  r2_echo    >= 80% of the reply's word 4-grams copied from the question
  r3_unsure  an unsure/deflecting phrase ("I'm not sure", "remind me")
  r4_capture a user fact (name, job, town, any user-held fact of the conversation) bound to the first person
             (E001 captures: "I'm Tobias", "I work as an architect", "I've lived in Stonebridge")
  r5_name    a sentence with a first-person word and no "you" names a person other than the name the user gave
             the assistant ("I could be Priya or Leilani", "Call me Hugo")
  r6_own     the user named the assistant, and the reply does not ASSERT that name, or gives it to the user
             ("Your name is Kestrel", "I'll call you Kestrel": grade_voice.given_away, STEP 10 fix round; "You
             chose the name Kestrel for me" passes)
  r7_self    no first-person word at all: the reply is not about the assistant ("It is Beatriz, Priya and Leilani.")
role_leaks(rec, replies): diagnostic scan of EVERY reply of a conversation, any family. Events:
  capture    a reply binds a user-held fact value to the first person (the assistant speaking as the user or
             adopting the user's name, job, town, pets, plans); only replies at or after the fact's turn count
  user_voice a value held by a named other appears with "my <holder>" ("My sister's cat is Mochi")
  turn_leak  the reply carries a role tag or template marker (it writes a user turn)"""
import re

import grade_loop as L
import grade_text as T
import grade_voice as GV
import pools_vals as V

NAMES = V.PERSON + V.ANAME


def user_fact_values(rec, upto=None):
    out = []
    for t in rec["turns"]:
        if upto is not None and t["i"] > upto:
            continue
        for f in t["facts"]:
            if f.get("holder") == "user" and f.get("value") and f["value"] in V.ALL_VALUES:
                out.append((t["i"], f["value"]))
    return out


def g_rolex(reply, stop, prior, rec, probe):
    text = T.norm(reply)
    fails = []
    if L.degenerate(reply, stop, prior, probe["kind"], L.turn_kinds(rec)[:len(prior)]):
        fails.append("r1_degen")
    if T.echo(text, probe["question"]):
        fails.append("r2_echo")
    if T.unsure(text):
        fails.append("r3_unsure")
    facts = set((probe.get("user_facts") or {}).values()) | {v for _, v in user_fact_values(rec, probe["turn"])}
    if any(T.captures(v, text) for v in facts):
        fails.append("r4_capture")
    own = probe.get("aname")
    if "claim" not in T.OFF:
        for s in re.split(r"(?<=[.!?])\s+|\n+", text):
            if T.FIRST_ANY.search(s) and not T.SECOND_P.search(s) and any(
                    T.mentioned(s, v) for v in NAMES if v != own):
                fails.append("r5_name")
                break
    if own and "ownname" not in T.OFF:
        if not T.asserted_hits(text, own, NAMES) or GV.given_away(text, own):
            fails.append("r6_own")
    if "self" not in T.OFF and not T.FIRST_ANY.search(text):
        fails.append("r7_self")
    return dict(ok=not fails, fails=fails, lenient=not [f for f in fails if f in ("r1_degen", "r4_capture")])


def role_leaks(rec, replies):
    ufacts = user_fact_values(rec)
    others = []
    for t in rec["turns"]:
        for f in t["facts"]:
            h = f.get("holder")
            if f.get("value") in V.ALL_VALUES and h not in ("user", "assistant", "other", "world", "table", None):
                others.append((t["i"], f["value"], h))
    events = []
    for j, r in enumerate(replies):
        text, turn = T.norm(r), j + 1
        for i, v in ufacts:
            if i <= turn and T.captures(v, text):
                events.append(dict(turn=turn, kind="capture", value=v))
        for i, v, h in others:
            if i <= turn and T.mentioned(text, v) and re.search(r"\bmy\s+" + re.escape(h) + r"\b", text, re.I):
                events.append(dict(turn=turn, kind="user_voice", value=v))
        if L.leak(text):
            events.append(dict(turn=turn, kind="turn_leak", value=None))
    return events
