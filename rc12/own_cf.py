"""RC-12 OWN counterfactual-history DIAGNOSTIC (step 5 audit). Not the scored protocol, and never mixed into it.

Why: OWN asks the model to recall its own earlier commitment. The audit's CONSIST cheater scores OWN 1.00 without
ever reading its own reply: it picks option 1 at Q and later names option 1 of the USER's list (lists: it re-derives
the same list). A small model with a strong first-option bias does the same by accident. No generator change can
stop that, because any fixed policy is self-consistent.
What: with runner --own-cf, the harness rewrites the model's OWN reply at each OWN Q turn right after it is
generated, before it enters the history: a pick has its chosen option swapped with the next offered option (both
names exchanged everywhere, so "A, not B" stays coherent); a numbered / bulleted list of exactly 3 has items 2 and
3 exchanged. The rewritten reply is what the model sees in its history and what G-DYN parses (the model's own text
is kept in the transcript's raw field). A model that reads its history follows the swap; one that re-derives its
answer does not. Replies that cannot be parsed are left alone (they are source failures either way).
Report OWN under --own-cf beside the normal OWN score; a large drop means OWN was passed by consistency, not by
reading the conversation."""
import re

import grade_text as T

ITEM = re.compile(r"^(\s*(?:\d+[.)]|[-*•])\s+)(.*)$")


def _exchange(text, a, b):
    rx = lambda v: re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])")  # noqa: E731
    marked = rx(a).sub("\x00", text)
    return rx(b).sub(a, marked).replace("\x00", b)


def swap(rec, turn_i, text):
    """the reply at an OWN Q turn with its commitment swapped; None when turn_i is not an OWN source turn or the
    reply cannot be parsed."""
    for p in rec["probes"]:
        gf = p.get("gold_fn")
        if not gf or gf["src_turn"] != turn_i:
            continue
        if gf["type"] == "pick":
            opts = gf["options"]
            picked = [o for o in opts if T.asserted_hits(T.norm(text), o, opts)]
            if len(picked) != 1:
                return None
            return _exchange(text, picked[0], opts[(opts.index(picked[0]) + 1) % len(opts)])
        lines = text.split("\n")
        idx = [k for k, s in enumerate(lines) if ITEM.match(s)]
        if len(idx) != 3:
            return None
        ma, mb = ITEM.match(lines[idx[1]]), ITEM.match(lines[idx[2]])
        lines[idx[1]], lines[idx[2]] = ma.group(1) + mb.group(2), mb.group(1) + ma.group(2)
        return "\n".join(lines)
    return None
