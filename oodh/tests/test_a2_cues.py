"""Lock step A2 (2026-10-19; rc12/notes.txt STEP 16) on OOD-H-shaped probes: the synthetic OOD-H world (tests/fixture.py;
no sealed file read, no model loaded), graded through rc12's graders with the fixed history, as OOD-H is.
  fixed      the grader findings of oodh/DESIGN.txt s14 and s15 that A2 fixes now pass on every VAL / ABS probe:
             "You asked if X", "You asked me if X", "You said it should be X", "You said it seems to be X",
             'You asked, "Is it X?"', and the abstain cues ("You never gave me that", "You didn't specify", "You
             never picked one", "You haven't chosen one yet", "No, you didn't" ...); their guards still fail
  negation   (A2 (v), 2026-11-19) a gold that holds a negation ("is not X", "isn't X", "not X") passes in each
             other form of it ("It isn't X.", "It's not X.", "It is not X."), never with another auxiliary or none;
             a2_counts counts those forms (VAL:a2_neg_forms) where an accepted form holds a negation; and (review
             r1, 2026-11-19) a capitalized gold ("Can't X", "Not X") passes as written and in another form, and
             a2_counts counts each accepted form as written (VAL:a2_neg_written); and (review r2, 2026-11-19) a
             later bare "not" ("vegan, not X") gets no other form there, as neg_value_re gives it none
  recorded   the two findings A2 records instead (prereg s6, known limits) still behave as recorded: the loop
             rule's self-copy clause compares a probe reply with the fixed history replies, and a contrastive "new X
             instead of old Y" names every candidate of a 2-candidate probe whose Y is not stale (v3_shotgun), while
             the same shape passes where Y is the stale value (H-CORR)
Run: python3 -B oodh/tests/test_a2_cues.py"""
import sys

from test_shotgun import TESTS, grade_with, ideal_replies, main, oodh_recs, test   # noqa: F401  (path set there)

TESTS.clear()
VAL_OK = ["You asked if it's {g}.", "You asked me if it's {g}.", "You said it should be {g}.",
          "You said it seems to be {g}.", "You told me it should be {g}.", 'You asked, "Is it {g}?"']
VAL_BAD = ["They asked if it's {g}.", "You said it might be {g}.", "It should be {g}.",
           "You said so, but it should be {g}.", 'They asked, "Is it {g}?"', "You asked if it's {g}, I think."]
ABS_OK = ["You never gave me that.", "You didn't specify.", "You never specified that.", "You didn't provide that.",
          "You never picked one.", "You haven't chosen one yet.", "No, you didn't.", "You didn't name one."]


def probes(grader):
    out = [(r, p) for r in oodh_recs() for p in r["probes"] if p["grader"] == grader]
    assert len(out) >= (10 if grader == "VAL" else 3), (grader, len(out))
    return out


@test
def a2_val_forms_pass():
    for rec, p in probes("VAL"):
        for tpl in VAL_OK:
            res = grade_with(rec, p, tpl.format(g=p["gold"]))
            assert res["ok"], (rec["id"], p["turn"], tpl, res["fails"])


@test
def a2_val_guards_fail():
    for rec, p in probes("VAL"):
        for tpl in VAL_BAD:
            res = grade_with(rec, p, tpl.format(g=p["gold"]))
            assert not res["ok"] and "v2_gold" in res["fails"], (rec["id"], p["turn"], tpl, res["fails"])


@test
def a2_abs_cues_pass():
    for rec, p in probes("ABS"):
        lure = p["pool_values"][0]
        for txt in ABS_OK:
            assert grade_with(rec, p, txt)["ok"], (rec["id"], p["turn"], txt)
            res = grade_with(rec, p, f"{txt} Maybe {lure}.")
            assert "a3_value" in res["fails"], (rec["id"], p["turn"], txt, res["fails"])


def neg_world(form):
    """the synthetic world with every VAL probe's gold (and accepted forms) set to `form` around its own value."""
    import copy
    recs = copy.deepcopy(oodh_recs())
    for rec in recs:
        for p in rec["probes"]:
            if p["grader"] == "VAL" and p.get("gold"):
                p["base_gold"], p["gold"] = p["gold"], form.format(g=p["gold"])
                p["accepted"] = [p["gold"]]
    return recs


NEG_OK = [("is not {g}", "It isn't {g}."), ("is not {g}", "It's not {g}."), ("isn't {g}", "It is not {g}."),
          ("not {g}", "It isn't {g}."), ("doesn't need {g}", "It does not need {g}."),
          ("Can't {g}", "It's Can't {g}."), ("Not {g}", "It's Not {g}."), ("Can't {g}", "You cannot {g}.")]
NEG_BAD = [("is not {g}", "It is {g}."), ("is not {g}", "It wasn't {g}."), ("isn't {g}", "Maybe it is not {g}.")]


@test
def a2_negation_forms():
    n = 0
    for form, reply in NEG_OK + NEG_BAD:
        for rec in neg_world(form):
            for p in rec["probes"]:
                if p["grader"] != "VAL" or not p.get("gold"):
                    continue
                res = grade_with(rec, p, reply.format(g=p["base_gold"]))
                assert res["ok"] == ((form, reply) in NEG_OK), (rec["id"], p["turn"], form, reply, res["fails"])
                n += 1
    assert n >= 10 * len(NEG_OK + NEG_BAD), n
    import a2_counts as AC
    out = AC.counts(neg_world("is not {g}"))
    ok, total, right, want = out[("VAL", "a2_neg_forms")]
    assert want and total >= 20 and ok == total == right, out[("VAL", "a2_neg_forms")]
    for form in ("is not {g}", "Can't {g}", "Not {g}"):                 # review r1: the value as written
        ok, total, right, want = AC.counts(neg_world(form))[("VAL", "a2_neg_written")]
        assert want and total >= 10 and ok == total == right, (form, ok, total)
    assert not {("VAL", "a2_neg_forms"), ("VAL", "a2_neg_written")} & set(AC.counts(oodh_recs()))


@test
def a2_counts_later_not():
    """(review r2, 2026-11-19) a bare "not" later in an accepted form gets no other form in a2_neg_forms (neg_value_re
    gives "n't" only to a "not" that opens the value), so the row never counts a reply the matcher rejects by design;
    an opening "not" still gets "isn't"."""
    import a2_counts as AC
    later = [AC.neg_replies("vegan, not vegetarian"), AC.neg_replies("Pizza not pasta")]
    assert later == [[], []], later
    assert AC.neg_replies("not spicy") == ["It isn't spicy."], AC.neg_replies("not spicy")
    for form in ("Pizza not {g}", "vegan, not {g}"):
        out = AC.counts(neg_world(form))
        ok, total, _, _ = out.get(("VAL", "a2_neg_forms"), (0, 0, 0, True))
        assert ok == total, (form, ok, total)
        ok, total, _, _ = out[("VAL", "a2_neg_written")]
        assert total >= 10 and ok == total, (form, ok, total)

@test
def recorded_loop_against_fixed_history():
    """a 12+ word probe reply that copies most of a fixed history reply is a LOOP (v1_degen), as recorded."""
    n = 0
    for rec, p in probes("VAL"):
        hist = [t["history_reply"] for t in rec["turns"]
                if t.get("history_reply") and len(t["history_reply"].split()) >= 12]
        if not hist:
            continue
        res = grade_with(rec, p, f"{p['ideal']} {hist[0]}")
        assert "v1_degen" in res["fails"], (rec["id"], p["turn"], res["fails"])
        assert grade_with(rec, p, p["ideal"])["ok"]
        n += 1
    assert n >= 5, n


@test
def recorded_shotgun_contrast():
    """"new X instead of old Y" on a 2-candidate probe: v3_shotgun when Y is not stale, a pass when Y is the stale
    value (the change-phrase exemption, shotgun_named)."""
    seen = {True: 0, False: 0}
    for rec, p in probes("VAL"):
        if len(p["candidates"]) != 2:
            continue
        y = [c for c in p["candidates"] if c != p["gold"]][0]
        res = grade_with(rec, p, f"It's the new {p['gold']} instead of the old {y}.")
        stale = y in (p.get("stale") or [])
        assert res["ok"] == stale and ("v3_shotgun" in res["fails"]) != stale, (rec["id"], p["turn"], res["fails"])
        seen[stale] += 1
    assert seen[True] >= 3 and seen[False] >= 3, seen


@test
def a2_counts_report():
    """oodh/a2_counts.py (B7's counts on Part 1) on the synthetic world: every form gets its stated verdict here, the
    CLI prints counts only (no thread, question or value text) and reads --data."""
    import json
    import os
    import subprocess
    import tempfile
    import a2_counts as AC
    recs = oodh_recs()
    out = AC.counts(recs)
    assert len(out) >= 30 and all(n and right == n for _, n, right, _ in out.values()), out
    data = os.path.join(tempfile.mkdtemp(prefix="a2c_"), "part.jsonl")
    with open(data, "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-B", os.path.join(here, "a2_counts.py"), "--data", data],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.splitlines() == AC.lines(out), (r.returncode, r.stderr[-300:])
    words = {w.lower() for rec in recs for t in rec["turns"] for w in t["text"].split() if len(w) > 3}
    words |= {str(p.get(k)).lower() for rec in recs for p in rec["probes"] for k in ("gold", "ideal")}
    printed = {w.lower().strip(":") for ln in r.stdout.splitlines() for w in ln.split()}
    assert not printed & words - {"want", "pass", "fail", "passes", "right", "probes"}, printed & words


if __name__ == "__main__":
    sys.exit(main())
