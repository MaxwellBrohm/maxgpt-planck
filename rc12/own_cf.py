"""RC-12 OWN counterfactual-history run (step 5 audit; since OD1 b, 2026-09-25, a GATE on OWN in score.py).

Why: OWN asks the model to recall its own earlier commitment. The audit's CONSIST cheater scores OWN 1.00 without
ever reading its own reply: it picks option 1 at Q and later names option 1 of the USER's list (lists: it re-derives
the same list). A small model with a strong first-option bias does the same by accident. No generator change can
stop that, because any fixed policy is self-consistent.
What: with runner --own-cf, the harness rewrites the model's OWN reply at each OWN Q turn right after it is
generated, before it enters the history: a pick has its chosen option swapped with the next offered option (both
names exchanged everywhere, so "A, not B" stays coherent); a numbered / bulleted list of exactly 3 has items 2 and
3 exchanged. The rewritten reply is what the model sees in its history and what G-DYN parses (the model's own text
is kept in the transcript's raw field). A model that reads its history follows the swap; one that re-derives its
answer does not.
Parsing: swap finds the commitment with G-DYN's OWN rules (grade_fmt_dyn: asserted_hits over the normalized reply
for a pick, list_spans for a list: the same lines, the same head cut, str.splitlines), so every reply G-DYN can
parse is rewritten, extra marker lines and blank bullets included (verifier 2026-09-25: a looser line rule here let
a padded list go unswapped and ungated). A reply G-DYN cannot parse is left alone: it is a source failure in both
runs (unit 0). A pick rewrite that does not parse back as the swapped option returns UNSWAPPED: the runner keeps
the reply, marks the turn and row cf_unswapped, and score.py counts that OWN unit 0 in OWN_GATED.
OD1 (b): score.py joins these rows to the normal run by conversation id; an OWN unit counts in R only if it is
right in BOTH runs (OWN_GATED). OWN (normal run alone) and OWN_CF (this run alone) are reported beside it; a large
drop means OWN was passed by consistency, not by reading the conversation. The rows count in nothing else."""
import grade_fmt_dyn as FD
import grade_text as T
import pools_vals as V

UNSWAPPED = object()      # G-DYN parses the reply but the rewrite did not take (a pick exchange that parses back wrong)


def _exchange(text, a, b):
    """a and b exchanged everywhere, matched exactly as G-DYN matches option names (pools_vals.value_re)."""
    marked = V.value_re(a).sub("\x00", text)
    return V.value_re(b).sub(a, marked).replace("\x00", b)


def _picked(text, opts):
    return [o for o in opts if T.asserted_hits(T.norm(text), o, opts)]


def swap(rec, turn_i, text):
    """the reply at an OWN Q turn with its commitment swapped; None when turn_i is not an OWN source turn or G-DYN
    cannot parse the reply; UNSWAPPED when a pick rewrite does not parse back as the next option."""
    for p in rec["probes"]:
        gf = p.get("gold_fn")
        if not gf or gf["src_turn"] != turn_i:
            continue
        if gf["type"] == "pick":
            opts = gf["options"]
            picked = _picked(text, opts)
            if len(picked) != 1:
                return None
            new = _exchange(text, picked[0], opts[(opts.index(picked[0]) + 1) % len(opts)])
            return new if _picked(new, opts) == [opts[(opts.index(picked[0]) + 1) % len(opts)]] else UNSWAPPED
        spans = FD.list_spans(T.norm(text))    # T.norm maps character to character: the lines and spans line up
        if len(spans) != 3:
            return None
        lines = text.splitlines(keepends=True)
        (ka, (a0, a1), _), (kb, (b0, b1), _) = spans[1], spans[2]
        la, lb = lines[ka], lines[kb]
        lines[ka], lines[kb] = la[:a0] + lb[b0:b1] + la[a1:], lb[:b0] + la[a0:a1] + lb[b1:]
        return "".join(lines)
    return None
