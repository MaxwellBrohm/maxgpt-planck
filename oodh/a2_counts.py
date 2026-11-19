"""OOD-H Part 1 counts for lock step A2 (2026-10-19; rc12/notes.txt STEP 16; prereg s6, the A2 block; s14 (e)). The
sealed session runs it beside gate.py (rc12/LOCK_STEPS.md B7), the only place Part 1 may be read. Each A2 paraphrase
of rc12/audit_a2.py (the forms the OOD-H reviews saw fail, their guards, the new abstain cues) is graded as the reply
at each VAL or ABS probe it applies to, the other turns holding the fixed history and the probe IDEALs, through the
same graders as RC-12. Prints COUNTS ONLY: per form, how many probes pass and how many get the stated verdict; never
item text. Report only (exit 0 when the file reads), no gate: what a count should be is in the notes.
Since 2026-11-19 (A2 (v), grade_cues.neg_value_re) also VAL:a2_neg_forms: for every accepted form that holds a
negation, each other form of that negation as the reply ("It isn't X." for "is not X"), counted per form (want pass);
and (review r1, 2026-11-19) VAL:a2_neg_written: each such accepted form as written, as a sentence of its own ("Is not
X." for "is not X", "Can't Stop." for "Can't Stop"), so a value the matcher no longer finds as written shows here.
On both rows an accepted form in the first person ("I am not X", said as "I'm not X." or "I am not X.") fails v4
(the voice rule, not a matcher miss): read a shortfall there against the forms' person before calling it a miss.
Run: python3 -B oodh/a2_counts.py [--data FILE]"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import select  # noqa: E402,F401  the stdlib module, cached first: oodh/ holds a select.py of its own
sys.path[:0] = [HERE]

import argparse  # noqa: E402

import build as B  # noqa: E402  (puts rc12/ on sys.path)
import audit_a2 as A2  # noqa: E402
import grade_cues as GC  # noqa: E402
import graders as G  # noqa: E402


def replies_of(rec):
    """the fixed history on thread turns, the probe IDEALs on probe turns."""
    by = {p["turn"]: p for p in rec["probes"]}
    return [by[t["i"]]["ideal"] if t["i"] in by else t["history_reply"] for t in rec["turns"]]


def neg_replies(form):
    """replies that say accepted form `form` with its first negation in each other form of its group (an opening bare
    "not" also as "isn't", the one other form neg_value_re gives a bare "not"); [] when it holds none or only a later
    bare "not" ("vegan, not vegetarian": review r2, 2026-11-19)."""
    m = GC.NEG_IN_VALUE.search(form)
    if not m:
        return []
    group = GC.GROUP_OF.get(m.group(0).lower()) or (("not", "isn't") if m.start() == 0 else ())
    head, tail, out = form[:m.start()].rstrip(), form[m.end():], []
    for f in group:
        if f == m.group(0).lower():
            continue
        text = (head or "It") + ("" if f[0] == "'" else " ") + f + tail
        out.append(text[:1].upper() + text[1:] + ".")
    return out

def written_reply(form):
    """accepted form `form` said as written, a sentence of its own (first letter capitalized); None when it holds no
    negation."""
    return form[:1].upper() + form[1:] + "." if GC.NEG_IN_VALUE.search(form) else None


def counts(recs):
    """{(grader, name): [passes, applicable probes, right verdicts, wanted]} over every VAL / ABS probe."""
    out = {}
    for rec in recs:
        base = replies_of(rec)
        for p in rec["probes"]:
            table = {"VAL": A2.VAL, "ABS": A2.ABS}.get(p["grader"])
            if table is None or (p["grader"] == "VAL" and not p.get("gold")):
                continue
            lure = (p.get("pool_values") or p.get("candidates") or ["?"])[0]
            for name, tpl, want in table:
                if not A2.applicable(name, p):
                    continue
                reps = list(base)
                reps[p["turn"] - 1] = tpl.format(g=p.get("gold"), l=lure)
                ok = G.grade_probe(rec, p, reps, ["eos"] * len(reps))["ok"]
                c = out.setdefault((p["grader"], name), [0, 0, 0, want])
                c[0] += ok
                c[1] += 1
                c[2] += ok == want
            acc = (p.get("accepted") or []) if p["grader"] == "VAL" else []
            for name, text in [("a2_neg_forms", t) for f in acc for t in neg_replies(f)] + \
                    [("a2_neg_written", written_reply(f)) for f in acc if written_reply(f)]:
                reps = list(base)
                reps[p["turn"] - 1] = text
                ok = G.grade_probe(rec, p, reps, ["eos"] * len(reps))["ok"]
                c = out.setdefault(("VAL", name), [0, 0, 0, True])
                c[0] += ok
                c[1] += 1
                c[2] += ok
    return out


def lines(out):
    rows = [f"A2 counts (passes / probes the form applies to; right = the stated verdict): {len(out)} forms"]
    for (g, name), (ok, n, right, want) in sorted(out.items()):
        rows.append(f"  {g}:{name:22s} want {'pass' if want else 'fail'}  passes {ok:4d} of {n:4d}  right {right:4d}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(B.SEALED, "oodh_part1.jsonl"))
    a = ap.parse_args()
    print("\n".join(lines(counts(B.read_jsonl(a.data)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
