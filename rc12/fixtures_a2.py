"""RC-12 grader fixtures for lock step A2 (2026-10-19): the OOD-H reviews' grader findings (oodh/DESIGN.txt s14 and
s15; rc12/grade_cues.py, the A2 block; rc12/notes.txt STEP 16). Built from REAL dev items like fixtures_graders.py,
whose fx() they use; mutation_graders.py adds them to every VAL and ABS probe it picks.
  *_ok        a natural right reply the graders failed before the fix ("You asked if X", "You said it should be X",
              'You asked, "Is it X?"', "You never picked one", "No, you didn't") must pass
  the rest    the guards: each fails, and each is the fixture that kills one mutant of a fix clause (mutants_graders,
              the A2 block): another subject ("They asked if X", "They never picked one", "No, they didn't"), a
              probability hedge in a report ("You said it might be X"), a hedge with no report ("It should be X"), a
              break between the report and the value (comma, semicolon, "but"), a hedge after the value, a mention
              outside the quoted question, a value given with the new cues (a3_value)
Mode (grade_voice.mode_of): no fixture on assistant-held OWN picks (DYN grades them; "You asked if X" gives the
pick to the user there); the "me" forms only where "me" binds nothing (user and given: grade_voice's report
frames; K keeps E001 captures, where "You asked me if X" fails as before).
A2 (v) (2026-11-19; grade_cues.neg_value_re): no RC-12 gold holds a negation (0 of 257 dev values), so neg_records
copies real VAL records and sets the probe's gold to a negation phrase around its own value (marked synthetic; the
conversation tests of mutation_graders skip them): *_ok the other form of the negation must pass, the rest fail.
Review r1 (2026-11-19): fixtures for clauses no fixture held (each kills the mutant named beside it in mutants_a2):
a break after the value ("You said it should be X, so that's settled.", must pass), a report frame that does not end
right before the quote or the "if" ('You said so, but is it "X?"', "You asked me to check if it's X.", must fail),
"have not", "did not" and a denial with no comma; for (v) a capitalized negation value as written and in another form
(the case fix in grade_cues._alts) and the contraction of every group.
Review r2 (2026-11-19): the reported-hedge exemption refused for a frame after the hedge, a negation in or after the
frame, a first-person word or "so" after it, and each older break again without "so" (must fail); for (v) the
value's boundaries around a mid-value negation and at its end (must fail) and a curly apostrophe in it (must pass)."""
import copy

import grade_voice as GV
from fixtures_graders import cap, fx


def val_a2(rec, p, gold=None):
    gold = gold or p.get("gold") or (p.get("accepted") or [None])[0]
    mode = GV.mode_of(p)
    if not gold or mode == "own":
        return []
    out = [fx(rec, p, "a2_asked_if_ok", f"You asked if it's {gold}.", True),
           fx(rec, p, "a2_first_asked_ok", f"You first asked if it's {gold}.", True),
           fx(rec, p, "a2_said_should_ok", f"You said it should be {gold}.", True),
           fx(rec, p, "a2_said_seems_ok", f"You said it seems to be {gold}.", True),
           fx(rec, p, "a2_quote_asked_ok", f'You asked, "Is it {gold}?"', True),
           fx(rec, p, "a2_quote_said_ok", f'You said, "Can it be {gold}?"', True),
           fx(rec, p, "a2_they_asked_if", f"They asked if it's {gold}.", False),
           fx(rec, p, "a2_said_might", f"You said it might be {gold}.", False),
           fx(rec, p, "a2_should_bare", f"It should be {gold}.", False),
           fx(rec, p, "a2_said_but", f"You said so but it should be {gold}.", False),
           fx(rec, p, "a2_said_comma", f"You said so, yet it should be {gold}.", False),
           fx(rec, p, "a2_said_semi", f"You said so; it should be {gold}.", False),
           fx(rec, p, "a2_hedge_after", f"You said {gold}, and it seems right.", False),
           fx(rec, p, "a2_quote_they", f'They asked, "Is it {gold}?"', False),
           fx(rec, p, "a2_quote_outside", f'{cap(gold)} is what you asked, "right?"', False),
           # review r1 (2026-11-19): a break after the value is no break; the quote frame must end the text before
           # the quote
           fx(rec, p, "a2_said_should_after_ok", f"You said it should be {gold}, so that's settled.", True),
           fx(rec, p, "a2_quote_frame_far", f'You said so, but is it "{gold}?"', False),
           # review r2 (2026-11-19): a frame after the hedge; a negation in the frame or after it; a first-person
           # word after it (each passed on the graders before r2)
           fx(rec, p, "a2_should_said_after", f"It should be {gold}, you said.", False),
           fx(rec, p, "a2_never_wrote_should", f"You never wrote it should be {gold}.", False),
           fx(rec, p, "a2_never_once_should", f"You never once said it should be {gold}.", False),
           fx(rec, p, "a2_said_unsure_should", f"You said you weren't sure it should be {gold}.", False),
           fx(rec, p, "a2_said_i_should", f"You said I should be {gold}.", False),
           # the comma, semicolon and "but" breaks again without "so" (a2_said_comma, _semi, _but hold "You said so")
           fx(rec, p, "a2_said_that_comma", f"You said that, yet it should be {gold}.", False),
           fx(rec, p, "a2_said_that_semi", f"You said that; it should be {gold}.", False),
           fx(rec, p, "a2_said_that_but", f"You said that but it should be {gold}.", False)]
    if mode in ("user", "given"):
        out += [fx(rec, p, "a2_asked_me_if_ok", f"You asked me if it's {gold}.", True),
                fx(rec, p, "a2_told_should_ok", f"You told me it should be {gold}.", True),
                fx(rec, p, "a2_last_frame_ok", f"You told me so, and you said it should be {gold}.", True),
                fx(rec, p, "a2_quote_asked_me_ok", f'You asked me: "Is it {gold}?"', True),
                fx(rec, p, "a2_asked_to_check_if", f"You asked me to check if it's {gold}.", False),   # review r1
                fx(rec, p, "a2_told_nothing_so", f"You told me nothing so it should be {gold}.", False)]  # review r2
    return out


ABS_OK = [("a2_gave_ok", "You never gave me that."), ("a2_specify_ok", "You didn't specify."),
          ("a2_specified_ok", "You never specified that."), ("a2_provide_ok", "You didn't provide that."),
          ("a2_picked_ok", "You never picked one."), ("a2_chosen_ok", "You haven't chosen one yet."),
          ("a2_name_ok", "You didn't name one."), ("a2_rate_ok", "You didn't rate it."),
          ("a2_ordered_ok", "You haven't ordered anything yet."), ("a2_decided_ok", "You haven't decided yet."),
          ("a2_yet_ok", "You haven't yet picked one."), ("a2_no_didnt_ok", "No, you didn't."),
          ("a2_have_not_ok", "You have not chosen one yet."), ("a2_did_not_ok", "You did not specify."),
          ("a2_no_comma_ok", "No you didn't.")]                                     # the last three: review r1


def abs_a2(rec, p):
    c = (p.get("candidates") or [None])[0] or p.get("other_value")
    out = [fx(rec, p, label, text, True) for label, text in ABS_OK]
    out += [fx(rec, p, "a2_they_never", "They never picked one.", False),
            fx(rec, p, "a2_no_they", "No, they didn't.", False)]
    if c:
        out += [fx(rec, p, "a2_cue_value", f"You didn't specify, but it's {c}.", False),
                fx(rec, p, "a2_denial_value", f"No, you didn't. It's {c}.", False)]
    return out


NEG = [("a2_neg_bare_ok", "not {g}", "It isn't {g}.", True), ("a2_neg_is_ok", "is not {g}", "It isn't {g}.", True),
       ("a2_neg_clitic_ok", "is not {g}", "It's not {g}.", True), ("a2_neg_expand_ok", "isn't {g}", "It is not {g}.", True),
       ("a2_neg_do_ok", "do not want {g}", "You don't want {g}.", True),
       ("a2_neg_can_ok", "can't have {g}", "You cannot have {g}.", True),
       ("a2_neg_will_ok", "won't pick {g}", "You will not pick {g}.", True),
       ("a2_neg_have_ok", "haven't tried {g}", "You've not tried {g}.", True),
       ("a2_neg_mid_clitic_ok", "it is not {g}", "It's not {g}.", True),
       ("a2_neg_mid_ok", "it's not {g}", "It isn't {g}.", True),
       # review r1 (2026-11-19): a capitalized value as written and in another form (its negation in any case), and
       # the contraction of each other group (the clitics 're, 'd, 'll, 'm too)
       ("a2_neg_cap_ok", "Can't {g}", "It's Can't {g}.", True),
       ("a2_neg_cap_open_ok", "Not {g}", "It's Not {g}.", True),
       ("a2_neg_cap_mid_ok", "It Is Not {g}", "Yes: It Is Not {g}.", True),
       ("a2_neg_cap_other_ok", "Can't {g}", "It's Cannot {g}.", True),
       ("a2_neg_are_ok", "are not {g}", "They aren't {g}.", True),
       ("a2_neg_re_ok", "are not {g}", "They're not {g}.", True),
       ("a2_neg_am_ok", "I am not {g}", "You said: I'm not {g}.", True),
       ("a2_neg_was_ok", "was not {g}", "It wasn't {g}.", True),
       ("a2_neg_were_ok", "were not {g}", "They weren't {g}.", True),
       ("a2_neg_does_ok", "does not need {g}", "It doesn't need {g}.", True),
       ("a2_neg_did_ok", "did not want {g}", "You didn't want {g}.", True),
       ("a2_neg_has_ok", "has not got {g}", "It hasn't got {g}.", True),
       ("a2_neg_had_ok", "had not tried {g}", "You hadn't tried {g}.", True),
       ("a2_neg_would_ok", "would not pick {g}", "You wouldn't pick {g}.", True),
       ("a2_neg_d_ok", "would not pick {g}", "You'd not pick {g}.", True),
       ("a2_neg_ll_ok", "will not pick {g}", "You'll not pick {g}.", True),
       ("a2_neg_could_ok", "could not have {g}", "You couldn't have {g}.", True),
       ("a2_neg_should_ok", "should not have {g}", "You shouldn't have {g}.", True),
       ("a2_neg_must_ok", "must not have {g}", "You mustn't have {g}.", True),
       ("a2_neg_need_ok", "need not bring {g}", "You needn't bring {g}.", True),
       # review r2 (2026-11-19): the value's own boundaries around a mid-value negation and at its end, and a curly
       # apostrophe in the value
       ("a2_neg_mid_inside", "it is not {g}", "Bit isn't {g}.", False),
       ("a2_neg_tail_inside", "is not {g}", "It isn't {g}ish.", False),
       ("a2_neg_curly_ok", "isn\u2019t {g}", "It isn't {g}.", True),
       ("a2_neg_plain", "is not {g}", "It is {g}.", False), ("a2_neg_other_aux", "is not {g}", "It wasn't {g}.", False),
       ("a2_neg_hedge", "isn't {g}", "Maybe it is not {g}.", False), ("a2_neg_question", "not {g}", "It isn't {g}?", False)]

def neg_records(index, per=4):
    """(records, fixtures) for A2 (v): the first user-mode VAL probe of each of the first `per` families in index, one
    copy of its record per NEG kind (id "<id>~<label>"), the probe's gold and accepted set to the kind's phrase."""
    base = []
    for r in index.values():
        p = next((p for p in r["probes"] if p["grader"] == "VAL" and p.get("gold") and GV.mode_of(p) == "user"), None)
        if p and r["family"] not in {b["family"] for b, _ in base}:
            base.append((r, p))
    recs, out = {}, []
    for r, p in base[:per]:
        for label, form, reply, want in NEG:
            c = copy.deepcopy(r)
            c["id"], c["synthetic"] = f"{r['id']}~{label}", True
            q = next(x for x in c["probes"] if x["turn"] == p["turn"])
            q["gold"] = form.format(g=p["gold"])
            q["accepted"] = [q["gold"]]
            recs[c["id"]] = c
            out.append(fx(c, q, label, reply.format(g=p["gold"]), want))
    return recs, out
