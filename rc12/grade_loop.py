"""RC-12 loop rule (SPEC s4, pre-registered exactly; equality clause per OD6 (iii), ruled 2026-09-25), per reply r
answering user turn t, given the earlier replies:
  LOOP     some word 3-gram occurs >= 4 times in r (E001 battery.degenerate: WITHIN a turn); OR turn t ASKS something
           (kind P, X, Q or D; LOOP requests are P) and r, normalized, equals an earlier reply of the conversation, at
           any length; OR r has >= 12 words and >= 50% of its word 4-grams occur in ONE earlier reply (self-copy:
           ACROSS turns).
  RUNAWAY  r stopped at the 256-token cap.   EMPTY  r has no word characters.
  LEAK     a role tag at the start of any line of r, or a chat-template marker anywhere in r (the reply speaks a
           user turn or continues into one).
Degenerate = any of the four; the Level A loop criterion counts LOOP only (the rest are reported beside).
ACK-REPEAT (OD6 iii): turn t is a statement turn (kind S, L, C, I, O or T) and r, normalized, equals an earlier reply
  ("Got it." twice). Never a flag, never degenerate, never in Level A; score.py reports its rate beside the loop rate.
  ANSWER REPEAT (reported apart, verifier 2026-09-25): the subset where r equals an earlier reply to an ASKING turn
  (an answer said again on a statement turn, not an acknowledgement said twice). Under OD6 (iii) as ruled it is an
  ack-repeat like any other, so a responder that parrots its last short answer on statement turns meets the Level A
  loop criterion; whether to charge it as a LOOP is a scope question left to Max (prereg draft s17 OD6).
Comparison set: "an earlier reply" is EVERY earlier reply of the conversation, whatever kind of turn it answered.
  OD6 changes which replies can be charged (only replies to asking turns), not what they are compared with (the
  same set as self-copy). So the same answer to two different questions is a LOOP, and so is a question answered
  with a copy of an earlier acknowledgement ("Got it." after a statement, then "Got it." to a filler question).
Normalization for equality: lowercase word tokens (punctuation and spacing ignored)."""
import re

import grade_text as T

LP = dict(tri_rep=4, selfcopy=0.50, selfcopy_min=12)
KINDS = frozenset("SLCIOTQDPX")   # SPEC s1 turn kinds
ASKS = frozenset("PXQD")          # turns that ask something: the equality clause applies (OD6 iii)
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


def classify(reply, stop, prior, kind):
    """flags (subset of LOOP, RUNAWAY, EMPTY, LEAK) and the LOOP sub-reasons (tri, equal, selfcopy). kind: the kind of
    the user turn r answers (the equality clause charges only replies to asking turns, OD6 iii)."""
    reply = T.norm(reply)
    ws = T.lwords(reply)
    prior_ws = [T.lwords(T.norm(p)) for p in prior]
    why = []
    if "tri" not in T.OFF and tri_repeat(ws) >= LP["tri_rep"]:
        why.append("tri")
    if "equal" not in T.OFF and asks(kind) and equals_earlier(ws, prior_ws):
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
    equals an earlier reply (an ack-repeat: reported, never a flag)."""
    if asks(kind):
        return None
    return equals_earlier(T.lwords(T.norm(reply)), [T.lwords(T.norm(p)) for p in prior])


def answer_repeat(reply, prior, prior_kinds, kind):
    """None when the turn asks something; else whether r, normalized, equals an earlier reply to an ASKING turn (an
    answer repeated on a statement turn; reported, never a flag)."""
    if asks(kind):
        return None
    answers = [p for p, k in zip(prior, prior_kinds) if asks(k)]
    return equals_earlier(T.lwords(T.norm(reply)), [T.lwords(T.norm(a)) for a in answers])


def degenerate(reply, stop, prior, kind):
    return bool(classify(reply, stop, prior, kind)[0])


def conversation_flags(replies, stops, kinds):
    """per-turn flags for a whole conversation (replies[i] answers user turn i+1, of kind kinds[i])."""
    assert len(replies) == len(stops) == len(kinds), (len(replies), len(stops), len(kinds))
    return [classify(r, s, replies[:i], k)[0] for i, (r, s, k) in enumerate(zip(replies, stops, kinds))]


def conversation_acks(replies, kinds):
    """per-turn ack_repeat (None on asking turns, True / False on statement turns)."""
    assert len(replies) == len(kinds), (len(replies), len(kinds))
    return [ack_repeat(r, replies[:i], k) for i, (r, k) in enumerate(zip(replies, kinds))]


def conversation_answer_repeats(replies, kinds):
    """per-turn answer_repeat (None on asking turns, True / False on statement turns)."""
    assert len(replies) == len(kinds), (len(replies), len(kinds))
    return [answer_repeat(r, replies[:i], kinds[:i], k) for i, (r, k) in enumerate(zip(replies, kinds))]
