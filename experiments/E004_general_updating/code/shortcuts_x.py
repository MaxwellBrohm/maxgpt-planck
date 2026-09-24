"""E004 extra cheap rules beyond O1-O8 (step 3: "any other cheap rule you can think of"), text-only like
oracles_e004.py, plus T, a topic tracker reported as a DIAGNOSTIC, not a shortcut: it assigns every statement
that names no object to the object of the previous statement and answers with the latest statement assigned to
the asked object. That is the updating skill itself (pronoun/ellipsis resolution + recency per object) without
alias resolution, so it is expected to score high everywhere except on alias items."""
from text_e004 import words
from oracles_e004 import View, has_marker, strip_marker, o1_first, o2_last, ORACLES

UPDATE_CUES = {"moved", "now", "instead", "changed", "switched", "shifted", "pushed", "rebook", "rescheduled",
               "delayed", "slid", "slipped", "relocated", "redone", "repainted", "exchanging", "sorry", "forget",
               "better", "reassigned", "swapped", "became", "turned", "switching", "rather", "after", "back"}


def x1_second_last(V):
    return V.stmts[-2][2] if len(V.stmts) >= 2 else o2_last(V)


def x2_update_cue(V):
    return V.latest(lambda ti, t, v: has_marker(t) or bool(set(words(t)) & UPDATE_CUES), o2_last(V))


def x3_full_phrase(V):
    return V.latest(lambda ti, t, v: V.names_asked(t, full_only=True), o1_first(V))


def x4_after_last_naming(V):
    """first value mentioned at or after the last user-turn mention of the asked object's phrase or head."""
    last = None
    for i, (ti, t, v) in enumerate(V.stmts):
        if V.names_asked(t):
            last = i
    if last is None:
        for ti, (u, a) in enumerate(V.turns):
            if V.names_asked(u):
                later = [s for s in V.stmts if s[0] >= ti]
                return later[0][2] if later else o2_last(V)
        return o2_last(V)
    return V.stmts[last][2]


def x5_after_echo(V):
    """the statement right after the last statement that frame-echoes the question or prefix."""
    idx = [i for i, (ti, t, v) in enumerate(V.stmts) if V.echoes(t)]
    if not idx:
        return o2_last(V)
    i = idx[-1]
    return V.stmts[i + 1][2] if i + 1 < len(V.stmts) else V.stmts[i][2]


def x6_no_overlap(V):
    qs = V.content(V.q) | V.content(V.p)
    return V.latest(lambda ti, t, v: not (V.content(t) & qs), o2_last(V))


def x7_first_naming(V):
    """earliest statement naming the asked object (the original, if named), else first mention."""
    for ti, t, v in V.stmts:
        if V.names_asked(t):
            return v
    return o1_first(V)


def x8_acked(V):
    """latest statement whose acknowledgement repeats its value."""
    return V.latest(lambda ti, t, v: v in V.values_in_order(V.turns[ti][1]), o2_last(V))


def x9_named_latest_unmarked(V):
    """latest statement naming the asked object that carries no marker, else O6's fallback."""
    return V.latest(lambda ti, t, v: V.names_asked(t) and not has_marker(t), o1_first(V))


def x10_longest(V):
    """value of the longest statement (most words), latest wins."""
    return V.best_latest(lambda t: len(words(strip_marker(t))))


def x11_third_last(V):
    """third-to-last statement (the gold's position in every U item where B follows A)."""
    return V.stmts[-3][2] if len(V.stmts) >= 3 else o1_first(V)


def topic_tracker(V):
    owner, prev = [], None
    for ti, t, v in V.stmts:
        named = [i for i, (ph, h) in enumerate(V.it["objects"]) if _names_obj(t, ph, h)]
        cur = named[0] if named else prev
        owner.append(cur)
        prev = cur
    for (ti, t, v), o in zip(reversed(V.stmts), reversed(owner)):
        if o == V.it["asked"]:
            return v
    return o1_first(V)


def _names_obj(t, ph, h):
    from oracles_e004 import names
    return names(t, ph) or names(t, h)


EXTRA = [("X1 second-to-last statement", x1_second_last), ("X2 latest with update cue", x2_update_cue),
         ("X3 latest naming asked, full phrase", x3_full_phrase), ("X4 value at last naming", x4_after_last_naming),
         ("X5 statement after last echo", x5_after_echo), ("X6 latest with no q/p overlap", x6_no_overlap),
         ("X7 first naming asked", x7_first_naming), ("X8 latest acked with value", x8_acked),
         ("X9 latest unmarked naming asked", x9_named_latest_unmarked), ("X10 longest statement", x10_longest),
         ("X11 third-to-last statement", x11_third_last)]
FAKE = ORACLES + EXTRA                      # every fake scorer held to the shortcut gate
DIAGNOSTIC = [("T topic tracker (diagnostic)", topic_tracker)]


def score(items, fn):
    right = 0
    for it in items:
        right += fn(View(it)) == it["gold"]
    return right / len(items)


def chance(items):
    return sum(1 / len(it["candidates"]) for it in items) / len(items)
