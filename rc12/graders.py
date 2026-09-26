"""RC-12 per-probe graders (SPEC s4) and the conversation grader. Pure Python, no model.

grade_conv(rec, replies, stops) -> dict: probes (one result per graded probe), unit (the conversation's unit score
under rec.meta.unit), flags (loop-rule flags per turn), ack_repeat (per turn: None on asking turns, else whether the
reply repeats an earlier one; OD6 iii, reported), ack_of_answer (per turn: None on asking turns, else whether the
reply repeats an earlier reply to an asking turn; reported, and a LOOP in flags too since F1), leaks (role-leak scan,
diagnostic). Every loop-rule call passes the kinds of the earlier turns (grade_loop.turn_kinds).
replies[i] / stops[i] answer user turn i+1; stop is "eos" | "eot" | "role" | "cap".
A result: ok (strict), lenient (diagnostic), fails (names of the clauses that failed), plus grader extras, and
eq_only (grade_probe, every grader): the probe fails ONLY through the loop rule's equality clause, i.e. its one
failing clause is its degenerate clause (DEGEN_CLAUSES) and the loop rule's only reason is "equal" (a right answer
that repeats an earlier reply of the model). Reported per family by score.py (equality_only), never scored: the
measurement for strict case (d) and F3 (prereg draft s6, s17 OD6 follow-up (3); verifier 2026-09-25).

G-VAL clauses (a reply is right only if none fails):
  v1_degen    degenerate by the loop rule (LOOP, RUNAWAY, EMPTY, LEAK: grade_loop.py)
  v2_gold     no mention of the gold is ASSERTED (grade_text.why_not_asserted: negated within 3 words before,
              negated right after, question, hedge, example, if-clause, "X or Y", bare list line, "than X",
              greeting vocative)
  v2_unsure   an unsure/deflecting phrase anywhere ("I'm not sure", "remind me", "you haven't told me")
  v3_other    another in-context value of the type is asserted; a STALE value of the asked object is allowed only
              in a change phrase ("moved from Monday", "Monday was the old day") when the gold is asserted
  v3_guess    a value of the probe's pool that is NOT in the conversation is mentioned at all (a guess)
  v3_shotgun  every candidate (2+) is mentioned (not COMPOSE, whose answer compares the two holders)
  v4_voice    user voice: the gold bound to the first person (E001 captures) or "my" + an object/holder word
              (E004 clause 7); for assistant-held facts (OWN, "What did I call you?") the reverse: the value
              given to the user ("You picked Quizzards", "Your name is Kestrel"). Report frames (grade_voice.py,
              STEP 10): "You told me X" / "You asked me to X" do not bind a user fact to the first person;
              "I said X" / "I'd told you X" / "It's X, like I told you" do (the model speaks as the user), and so
              fail a fact the user gave the assistant; "You said X" / "You told me X" fail the assistant's own
              answer (OWN pick), a request ("You asked me to pick one, and X was my choice") does not. The
              reversal is per mode (grade_voice.reversed_to). World-held facts (K) keep E001 captures unchanged.
  v5_echo     >= 80% of the reply's word 4-grams copied from the probe question
G-ABS clauses: a1_degen, a2_cue (an abstain cue), a3_value (no value of the probe's pool mentioned at all),
  a4_voice (user voice: "my" + object word, or a first-person-bound sentence about the object with no "you").
  No echo clause: no probe question contains an abstain cue (checked by mutation_graders.py), so a2_cue already
  fails every echo. Lenient ABS: cue, and every mentioned value sits in a sentence naming its own holder or table.
FMT, DYN: grade_fmt_dyn.py. ROLEX and the role-leak scan: grade_role.py. LOOP probes: not degenerate and not an
echo of the user turn (deviation D7 in notes.txt)."""
import re

import grade_loop as L
import grade_text as T
import grade_voice as GV
import pools_vals as V

STOP = {"the", "a", "an", "of", "by", "on", "in", "at", "to", "for", "and", "my", "your", "is", "it", "that", "this"}
NON_PERSON = ("user", "assistant", "world", None)
STALE_BEFORE = T.rx(r"(?:\bfrom|\bchanged|\boriginally|\bpreviously|\bearlier|\bbefore|\bused to|\bwas|\bwere|\bold)"
                    r"\W+(?:[\w']+\W+){0,3}$")
STALE_AFTER = T.rx(r"^\W*(?:[\w']+\W+){0,1}(?:was|were|originally|previously|before|earlier|no longer)\b")


def pool_values(probe):
    if probe.get("pool_values"):
        return list(probe["pool_values"])
    p = probe.get("pool")
    return list(V.GUESS_POOL.get(p) or V.POOLS.get(p) or [])


def golds(probe, gold=None):
    if gold is not None:
        return [gold]
    return list(probe.get("accepted") or ([probe["gold"]] if probe.get("gold") else []))


def in_context(rec, probe, pool):
    text = " ".join(t["text"] for t in rec["turns"] if t["i"] <= probe["turn"])
    return [v for v in pool if T.mentioned(T.norm(text), v)]


def obj_words(probe):
    ws = {w.lower() for ph in probe.get("object_words") or [] for w in T.toks(ph)}
    h = probe.get("holder")
    if h not in NON_PERSON and not str(h).startswith("table"):
        ws |= set(T.lwords(h))
    return ws - STOP


def stale_ok(text, m):
    return bool(STALE_BEFORE.search(text[max(0, m.start() - 40):m.start()]) or
                STALE_AFTER.search(text[m.end():m.end() + 30]))


def g_val(reply, stop, prior, rec, probe, gold=None):
    text = T.norm(reply)
    gs = golds(probe, gold)
    pool = pool_values(probe)
    cands = list(probe.get("candidates") or [])
    ctx = in_context(rec, probe, pool)
    others = [v for v in dict.fromkeys(cands + ctx) if v not in gs]
    outside = [v for v in pool if v not in gs and v not in cands and v not in ctx]
    stale = set(probe.get("stale") or [])
    fails = []
    if L.degenerate(reply, stop, prior, probe["kind"], L.turn_kinds(rec)[:len(prior)]):
        fails.append("v1_degen")
    gold_ok = any(T.asserted_hits(text, g, pool) for g in gs)
    if not gold_ok:
        fails.append("v2_gold")
    if T.unsure(text):
        fails.append("v2_unsure")
    for v in others:
        hits = T.asserted_hits(text, v, pool)
        if "stale" not in T.OFF and v in stale and gold_ok:
            hits = [m for m in hits if not stale_ok(text, m)]
        if hits:
            fails.append("v3_other")
            break
    if "guess" not in T.OFF and any(T.mentioned(text, v) for v in outside):
        fails.append("v3_guess")
    if "shotgun" not in T.OFF and not probe.get("pool_values") and len(cands) >= 2 and all(T.mentioned(text, v) for v in cands):
        fails.append("v3_shotgun")
    mode = GV.mode_of(probe)
    voice = any(GV.voice_error(g, text, mode) for g in gs)
    if mode not in ("given", "own"):
        voice = voice or T.my_object(GV.unquote(text), obj_words(probe))
    if voice:
        fails.append("v4_voice")
    if T.echo(text, probe["question"]):
        fails.append("v5_echo")
    first = sorted((h.start(), v) for v in gs + others for h in T.value_hits(text, v)[:1])
    return dict(ok=not fails, fails=fails, lenient=bool(first) and first[0][1] in gs)


ABS_NOT_BIND = T.rx(r"\bi(?:'m| am) (?:not sure|afraid|sorry|unable|not able|not certain|happy|glad|here)\b")


def abs_voice(text, objwords):
    if T.my_object(text, objwords):
        return True
    if "absvoice" in T.OFF:
        return False
    for s in re.split(r"(?<=[.!?])\s+|\n+", ABS_NOT_BIND.sub(" ", text)):
        ws = set(T.lwords(s))
        if ws & objwords and T.FIRST_BIND.search(s) and not T.SECOND_P.search(s):
            return True
    return False


def holders_of(rec):
    out = {}
    for t in rec["turns"]:
        for f in t["facts"]:
            if f.get("role") == "table":
                for v in f["table"].values():
                    out.setdefault(v, set()).add(f["object"])
            elif f.get("value"):
                out.setdefault(f["value"], set()).add(str(f.get("holder")))
    return out


def g_abs(reply, stop, prior, rec, probe):
    text = T.norm(reply)
    pool = pool_values(probe)
    fails = []
    if L.degenerate(reply, stop, prior, probe["kind"], L.turn_kinds(rec)[:len(prior)]):
        fails.append("a1_degen")
    cue = "cue" in T.OFF or T.ABS_CUE.search(text) is not None
    if not cue:
        fails.append("a2_cue")
    named = [v for v in pool if T.mentioned(text, v)]
    if named and "absvalue" not in T.OFF:
        fails.append("a3_value")
    if abs_voice(text, obj_words(probe)):
        fails.append("a4_voice")
    hold = holders_of(rec)
    bound = all(any(any(w in T.lwords(text[slice(*T.sentence_span(text, m.start()))]) for w in T.lwords(h))
                    for h in hold.get(v, ()) if h != "user") for v in named for m in T.value_hits(text, v))
    lenient = not [f for f in fails if f != "a3_value"] and bound
    return dict(ok=not fails, fails=fails, lenient=lenient)


def g_loop(reply, stop, prior, rec, probe):
    flags, why = L.classify(reply, stop, prior, probe["kind"], L.turn_kinds(rec)[:len(prior)])
    fails = [f"loop_{f}" for f in flags]
    if T.echo(T.norm(reply), probe["question"]):
        fails.append("loop_echo")
    return dict(ok=not fails, fails=fails, lenient=not flags, why=why)


DEGEN_CLAUSES = ("v1_degen", "a1_degen", "f1_degen", "d1_degen", "r1_degen", "loop_LOOP")


def eq_only(res, reply, stop, prior, rec, probe):
    """whether the probe fails only through the equality clause (module docstring; reported, never scored)."""
    if len(res["fails"]) != 1 or res["fails"][0] not in DEGEN_CLAUSES:
        return False
    earlier = L.turn_kinds(rec)[:len(prior)]
    flags, why = L.classify(reply, stop, prior, probe["kind"], earlier)
    return flags == ["LOOP"] and why == ["equal"]


def grade_probe(rec, probe, replies, stops):
    import grade_fmt_dyn as FD
    import grade_role as R
    i = probe["turn"] - 1
    args = (replies[i], stops[i], replies[:i], rec, probe)
    fn = {"VAL": g_val, "ABS": g_abs, "LOOP": g_loop, "FMT": FD.g_fmt, "DYN": FD.g_dyn, "ROLEX": R.g_rolex}
    res = fn[probe["grader"]](*args)
    res.update(turn=probe["turn"], kind=probe["kind"], grader=probe["grader"], eq_only=eq_only(res, *args))
    return res


def unit_score(rec, results):
    if not results:
        return None
    if rec["meta"].get("unit") in ("all", "pair"):
        return float(all(r["ok"] for r in results))
    return sum(r["ok"] for r in results) / len(results)


def grade_conv(rec, replies, stops):
    import grade_role as R
    assert len(replies) == len(stops) == rec["n_turns"], rec["id"]
    results = [grade_probe(rec, p, replies, stops) for p in rec["probes"]]
    kinds = L.turn_kinds(rec)
    return dict(id=rec["id"], family=rec["family"], cell=rec["cell"], probes=results,
                unit=unit_score(rec, results), flags=L.conversation_flags(replies, stops, kinds),
                ack_repeat=L.conversation_acks(replies, kinds),
                ack_of_answer=L.conversation_answer_repeats(replies, kinds), leaks=R.role_leaks(rec, replies))
