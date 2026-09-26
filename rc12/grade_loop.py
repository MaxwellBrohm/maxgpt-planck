"""RC-12 loop rule (SPEC s4, pre-registered exactly; equality clause per OD6 (iii), ruled 2026-09-25, and its follow-ups
F1 and F2, decided 2026-09-25 by Claude under Max's standing permission, reversible before the lock), per reply r
answering user turn t, given the earlier replies:
  LOOP     some word 3-gram occurs >= 4 times in r (E001 battery.degenerate: WITHIN a turn); OR r, normalized, equals
           an earlier reply of its equality set, at any length (below); OR r has >= 12 words and >= 50% of its word
           4-grams occur in ONE earlier reply (self-copy: ACROSS turns, every earlier reply, whatever turn t is).
  RUNAWAY  r stopped at the 256-token cap.   EMPTY  r has no word characters.
  LEAK     a role tag at the start of any line of r, or a chat-template marker anywhere in r (the reply speaks a
           user turn or continues into one).
Degenerate = any of the four; the Level A loop criterion counts LOOP only (the rest are reported beside).
Equality set (equality_set):
  turn t ASKS something (kind P, X, Q, or D with asks: true; LOOP requests are P): EVERY earlier reply of the
    conversation, whatever turn it answered (OD6 iii). So the same answer to two different questions is a LOOP, and so is a question
    answered with a copy of an earlier acknowledgement ("Got it." after a statement, then "Got it." to a question).
  turn t is a statement turn (S, L, C, I, O, T, or a D filler with asks: false, F2): the earlier replies to ASKING
    turns only (F1). An answer said again
    on a statement turn is a LOOP (a parroted answer); an acknowledgement said again after an acknowledgement
    ("Got it." on two statement turns) is not: OD6 (iii) exempted consistent acknowledgements, not answers.
  Which turns ask is read from the kinds the caller passes: prior_kinds[j] is the kind of the turn prior[j] answered
  (turn_kinds(rec) in turn order; the probe graders and graders.grade_conv both take it from there).
  F2 (small-talk fillers): a D turn whose filler is a plain statement ("The sunrise was really pretty today.") carries
  asks: false in the record (the hand label of pools_fill_a / _b, copied by the builder; never inferred from the text
  or its punctuation). loop_kind maps it to the loop kind "d", which does not ask; the record kind stays D for
  every other purpose. Requests without a question mark ("Invent a silly holiday ...") keep asks: true.
ACK-REPEAT (OD6 iii, reported, never a flag by itself, never in Level A): turn t is a statement turn and r, normalized,
  equals ANY earlier reply. score.py reports its rate beside the loop rate. It includes the answer repeats below,
  which are LOOPs since F1, so ack_repeat is not "exempt repeats only".
ANSWER REPEAT (reported apart, verifier 2026-09-25): the subset where r equals an earlier reply to an ASKING turn.
  Since F1 it marks exactly the statement-turn replies the equality clause charges (the same test, computed apart as
  a cross-check), so ack_repeat_of_answers overlaps the loop rate; it stays reported beside it.
Normalization for equality: lowercase word tokens (punctuation and spacing ignored)."""
import re

import grade_text as T

LP = dict(tri_rep=4, selfcopy=0.50, selfcopy_min=12)
KINDS = frozenset("SLCIOTQDPXd")  # SPEC s1 turn kinds, plus d: a D filler that asks nothing (F2, loop_kind)
ASKS = frozenset("PXQD")          # turns that ask something: the equality clause applies (OD6 iii); d does not
ROLE_LINE = re.compile(r"^\s*[#*>\[\(]*\s*(?:user|assistant|human|system|ai|bot|model)\s*[\])*]*\s*:", re.I | re.M)
MARKER = re.compile(r"<\|[a-z_ ]+\|>|<\|?(?:im_start|im_end|endoftext|eot_id|start_header_id|end_header_id)\|?>|\[/?INST\]"
                    r"|</?s>|<start_of_turn>|<end_of_turn>|<\|user\|>|<\|assistant\|>", re.I)


def tri_repeat(ws):
    g = {}
    for i in range(len(ws) - 2):
        k = tuple(ws[i:i + 3])
        g[k] = g.get(k, 0) + 1
    return max(g.values(), default=0)


def asks(kind):
    assert kind in KINDS, f"unknown turn kind {kind!r}"
    return kind in ASKS


def equals_earlier(ws, prior_ws):
    return bool(ws) and any(ws == p for p in prior_ws)


def self_copy(ws, prior_ws):
    if len(ws) < LP["selfcopy_min"]:
        return False
    grams = [tuple(ws[i:i + 4]) for i in range(len(ws) - 3)]
    for p in prior_ws:
        pg = {tuple(p[i:i + 4]) for i in range(len(p) - 3)}
        if grams and sum(g in pg for g in grams) / len(grams) >= LP["selfcopy"]:
            return True
    return False


def leak(reply):
    if "leak" in T.OFF:
        return False
    return ROLE_LINE.search(reply) is not None or MARKER.search(reply) is not None


def loop_kind(t):
    """the kind the loop rule reads for user turn t: its record kind, except "d" for a D turn annotated asks: false
    (F2). Every D turn must carry the annotation (a record built without it fails here, never defaults)."""
    if t["kind"] != "D":
        return t["kind"]
    assert type(t.get("asks")) is bool, f"D turn u{t['i']} has no asks annotation (F2)"
    return "D" if t["asks"] else "d"


def turn_kinds(rec):
    """the loop-rule kinds of a record's user turns in turn order (kinds[i] is user turn i+1's; loop_kind)."""
    return [loop_kind(t) for t in sorted(rec["turns"], key=lambda t: t["i"])]


def equality_set(prior_ws, prior_kinds, kind):
    """the earlier replies the equality clause compares r with: every one when turn t asks (OD6 iii); on a statement
    turn only the replies to asking turns (F1). prior_kinds[j] is the kind of the turn prior_ws[j] answered."""
    assert len(prior_kinds) == len(prior_ws), (len(prior_kinds), len(prior_ws))
    if asks(kind):
        return prior_ws
    return [p for p, k in zip(prior_ws, prior_kinds) if asks(k)]


def classify(reply, stop, prior, kind, prior_kinds):
    """flags (subset of LOOP, RUNAWAY, EMPTY, LEAK) and the LOOP sub-reasons (tri, equal, selfcopy). kind: the kind of
    the user turn r answers; prior_kinds: the kinds of the turns the prior replies answered (equality_set)."""
    reply = T.norm(reply)
    ws = T.lwords(reply)
    prior_ws = [T.lwords(T.norm(p)) for p in prior]
    why = []
    if "tri" not in T.OFF and tri_repeat(ws) >= LP["tri_rep"]:
        why.append("tri")
    if "equal" not in T.OFF and equals_earlier(ws, equality_set(prior_ws, prior_kinds, kind)):
        why.append("equal")
    if "selfcopy" not in T.OFF and self_copy(ws, prior_ws):
        why.append("selfcopy")
    flags = []
    if why:
        flags.append("LOOP")
    if "runaway" not in T.OFF and stop == "cap":
        flags.append("RUNAWAY")
    if "empty" not in T.OFF and not re.search(r"\w", reply):
        flags.append("EMPTY")
    if leak(reply):
        flags.append("LEAK")
    return flags, why


def ack_repeat(reply, prior, kind):
    """OD6 (iii): None when the turn asks something (the equality clause is LOOP's there), else whether r, normalized,
    equals ANY earlier reply (an ack-repeat: reported; a LOOP too when the copy is of an answer, F1)."""
    if asks(kind):
        return None
    return equals_earlier(T.lwords(T.norm(reply)), [T.lwords(T.norm(p)) for p in prior])


def answer_repeat(reply, prior, prior_kinds, kind):
    """None when the turn asks something; else whether r, normalized, equals an earlier reply to an ASKING turn (an
    answer repeated on a statement turn). Reported; since F1 the equality clause charges the same replies as LOOP."""
    if asks(kind):
        return None
    answers = [p for p, k in zip(prior, prior_kinds) if asks(k)]
    return equals_earlier(T.lwords(T.norm(reply)), [T.lwords(T.norm(a)) for a in answers])


def degenerate(reply, stop, prior, kind, prior_kinds):
    return bool(classify(reply, stop, prior, kind, prior_kinds)[0])


def conversation_flags(replies, stops, kinds):
    """per-turn flags for a whole conversation (replies[i] answers user turn i+1, of kind kinds[i])."""
    assert len(replies) == len(stops) == len(kinds), (len(replies), len(stops), len(kinds))
    return [classify(r, s, replies[:i], k, kinds[:i])[0] for i, (r, s, k) in enumerate(zip(replies, stops, kinds))]


def conversation_acks(replies, kinds):
    """per-turn ack_repeat (None on asking turns, True / False on statement turns)."""
    assert len(replies) == len(kinds), (len(replies), len(kinds))
    return [ack_repeat(r, replies[:i], k) for i, (r, k) in enumerate(zip(replies, kinds))]


def conversation_answer_repeats(replies, kinds):
    """per-turn answer_repeat (None on asking turns, True / False on statement turns)."""
    assert len(replies) == len(kinds), (len(replies), len(kinds))
    return [answer_repeat(r, replies[:i], kinds[:i], k) for i, (r, k) in enumerate(zip(replies, kinds))]
