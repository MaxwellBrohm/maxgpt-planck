"""RC-12 loop rule (SPEC s4, pre-registered exactly), per reply r at turn t given the earlier replies:
  LOOP     some word 3-gram occurs >= 4 times in r (E001 battery.degenerate: WITHIN a turn); OR r, normalized, equals
           an earlier reply of the conversation; OR r has >= 12 words and >= 50% of its word 4-grams occur in ONE
           earlier reply (self-copy: ACROSS turns).
  RUNAWAY  r stopped at the 256-token cap.   EMPTY  r has no word characters.
  LEAK     a role tag at the start of any line of r, or a chat-template marker anywhere in r (the reply speaks a
           user turn or continues into one).
Degenerate = any of the four; the Level A loop criterion counts LOOP only (the rest are reported beside).
Normalization for equality: lowercase word tokens (punctuation and spacing ignored)."""
import re

import grade_text as T

LP = dict(tri_rep=4, selfcopy=0.50, selfcopy_min=12)
ROLE_LINE = re.compile(r"^\s*[#*>\[\(]*\s*(?:user|assistant|human|system|ai|bot|model)\s*[\])*]*\s*:", re.I | re.M)
MARKER = re.compile(r"<\|[a-z_ ]+\|>|<\|?(?:im_start|im_end|endoftext|eot_id|start_header_id|end_header_id)\|?>|\[/?INST\]"
                    r"|</?s>|<start_of_turn>|<end_of_turn>|<\|user\|>|<\|assistant\|>", re.I)


def tri_repeat(ws):
    g = {}
    for i in range(len(ws) - 2):
        k = tuple(ws[i:i + 3])
        g[k] = g.get(k, 0) + 1
    return max(g.values(), default=0)


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


def classify(reply, stop, prior):
    """flags (subset of LOOP, RUNAWAY, EMPTY, LEAK) and the LOOP sub-reasons (tri, equal, selfcopy)."""
    reply = T.norm(reply)
    ws = T.lwords(reply)
    prior_ws = [T.lwords(T.norm(p)) for p in prior]
    why = []
    if "tri" not in T.OFF and tri_repeat(ws) >= LP["tri_rep"]:
        why.append("tri")
    if "equal" not in T.OFF and equals_earlier(ws, prior_ws):
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


def degenerate(reply, stop, prior):
    return bool(classify(reply, stop, prior)[0])


def conversation_flags(replies, stops):
    """per-turn flags for a whole conversation (replies[i] answers user turn i+1)."""
    return [classify(r, s, replies[:i])[0] for i, (r, s) in enumerate(zip(replies, stops))]
