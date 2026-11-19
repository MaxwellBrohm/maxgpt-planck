"""RC-12 audit_graders.py paraphrases for lock step A2 (2026-10-19): the OOD-H reviews' grader findings (oodh/DESIGN.txt
s14 and s15; grade_cues.py's A2 block; rc12/notes.txt STEP 16). audit_graders appends VAL and ABS to its tables and
asks applicable() for every name in NAMES. Same format: (name, template, expected pass); {g} the gold, {l} a lure.
The me-forms only where "me" binds nothing (user- and assistant-given facts; K keeps E001 captures).
NEG (A2 (v), 2026-11-19; grade_cues.neg_value_re): no RC-12 gold holds a negation, so neg_cases grades each on a copy
of the probe whose gold (and accepted) is the entry's negation phrase around the probe's own value {g}: (name, gold
phrase, reply template, expected pass); audit_graders reports them as NEG:<name>.
Review r1 (2026-11-19) added 3 VAL, 3 ABS and 7 NEG kinds (the forms of the r1 fixtures in fixtures_a2.py), review r2
(2026-11-19) 7 VAL and 3 NEG kinds (the r2 fixtures' forms and one must-pass gap word)."""
import copy

import grade_voice as GV

VAL = [
    ("a2_asked_if", "You asked if it's {g}.", True), ("a2_first_asked", "You first asked if it's {g}.", True),
    ("a2_me_asked_if", "You asked me if it's {g}.", True), ("a2_said_should", "You said it should be {g}.", True),
    ("a2_said_seems", "You said it seems to be {g}.", True), ("a2_mentioned_should", "You mentioned it should be {g}.", True),
    ("a2_me_told_seems", "You told me it seems to be {g}.", True), ("a2_quote_asked", 'You asked, "Is it {g}?"', True),
    ("a2_me_quote_asked", 'You asked me: "Is it {g}?"', True),
    ("a2_they_asked_if", "They asked if it's {g}.", False), ("a2_wonder_if", "I wonder if it's {g}.", False),
    ("a2_asked_if_think", "You asked if it's {g}, I think.", False), ("a2_said_might", "You said it might be {g}.", False),
    ("a2_said_maybe", "You said maybe {g}.", False),
    ("a2_said_but_should", "You said something else, but it should be {g}.", False),
    ("a2_should_now", "It should be {g} now.", False), ("a2_said_then_seems", "You said {g}, and it seems right.", False),
    ("a2_quote_they", 'They asked, "Is it {g}?"', False), ("a2_quote_i", 'I asked, "Is it {g}?"', False),
    ("a2_quote_outside", 'So {g} is what you asked, "right?"', False),
    # review r1 (2026-11-19): a break after the value; a frame that does not end right before the "if" or the quote
    ("a2_said_should_after", "You said it should be {g}, so that's settled.", True),
    ("a2_quote_said_far", 'You said so, but is it "{g}?"', False),
    ("a2_me_asked_to_check", "You asked me to check if it's {g}.", False),
    # review r2 (2026-11-19): a frame after the hedge; a negation in or after the frame; a first-person word or "so"
    # after it; a gap word that is no negation still frames
    ("a2_should_said_after", "It should be {g}, you said.", False),
    ("a2_never_wrote_should", "You never wrote it should be {g}.", False),
    ("a2_never_once_should", "You never once said it should be {g}.", False),
    ("a2_said_unsure_should", "You said you weren't sure it should be {g}.", False),
    ("a2_said_i_should", "You said I should be {g}.", False),
    ("a2_me_told_nothing_so", "You told me nothing so it should be {g}.", False),
    ("a2_also_said_should", "You also said it should be {g}.", True),
]
ABS = [
    ("a2_gave", "You never gave me that.", True), ("a2_specify", "You didn't specify.", True),
    ("a2_specified", "You never specified that.", True), ("a2_provide", "You didn't provide that.", True),
    ("a2_picked", "You never picked one.", True), ("a2_chosen", "You haven't chosen one yet.", True),
    ("a2_name", "You didn't name one.", True), ("a2_rate", "You didn't rate it.", True),
    ("a2_ordered", "You haven't ordered anything yet.", True), ("a2_decided", "You haven't decided yet.", True),
    ("a2_no_didnt", "No, you didn't.", True),
    ("a2_lure_specify", "You didn't specify, but it's {l}.", False), ("a2_lure_no", "No, you didn't. It's {l}.", False),
    ("a2_lure_picked", "You never picked one, so {l} it is.", False), ("a2_they_never", "They never picked one.", False),
    ("a2_have_not", "You have not chosen one yet.", True), ("a2_did_not", "You did not specify.", True),    # review r1
    ("a2_no_comma", "No you didn't.", True),
]
NAMES = {n for n, _, _ in VAL + ABS}


def applicable(name, p):
    if name.startswith("a2_me_"):
        return GV.mode_of(p) in ("user", "given")
    return GV.mode_of(p) != "own"

NEG = [
    ("a2_neg_isnt", "is not {g}", "It isn't {g}.", True), ("a2_neg_s_not", "is not {g}", "It's not {g}.", True),
    ("a2_neg_no_isnt", "is not {g}", "No, it isn't {g}.", True), ("a2_neg_expand", "isn't {g}", "It is not {g}.", True),
    ("a2_neg_bare_isnt", "not {g}", "It isn't {g}.", True), ("a2_neg_bare_arent", "not {g}", "They aren't {g}.", True),
    ("a2_neg_said", "is not {g}", "You said it isn't {g}.", True),
    ("a2_neg_dont", "do not want {g}", "You don't want {g}.", True),
    ("a2_neg_doesnt", "doesn't need {g}", "It does not need {g}.", True),
    ("a2_neg_cant", "can't have {g}", "You cannot have {g}.", True),
    ("a2_neg_wont", "won't pick {g}", "You will not pick {g}.", True),
    ("a2_neg_havent", "haven't tried {g}", "You've not tried {g}.", True),
    ("a2_neg_drop", "is not {g}", "It is {g}.", False), ("a2_neg_aux", "is not {g}", "It wasn't {g}.", False),
    ("a2_neg_think", "isn't {g}", "I think it is not {g}.", False), ("a2_neg_q", "not {g}", "It isn't {g}?", False),
    ("a2_neg_maybe", "is not {g}", "Maybe it's not {g}.", False),
    ("a2_neg_if", "is not {g}", "If it isn't {g}, tell me.", False),
    # review r1 (2026-11-19): a capitalized value as written and in another form; more groups
    ("a2_neg_cap_written", "Can't {g}", "It's Can't {g}.", True), ("a2_neg_cap_open", "Not {g}", "It's Not {g}.", True),
    ("a2_neg_cap_other", "Can't {g}", "Yes, Cannot {g}.", True),
    ("a2_neg_wasnt", "was not {g}", "It wasn't {g}.", True),
    ("a2_neg_didnt", "did not want {g}", "You didn't want {g}.", True),
    ("a2_neg_shouldnt", "should not have {g}", "You shouldn't have {g}.", True),
    ("a2_neg_youd", "would not pick {g}", "You'd not pick {g}.", True),
    # review r2 (2026-11-19): the value's boundaries (a word ending in its first word, a suffix after it); a curly
    # apostrophe in the value
    ("a2_neg_inside_word", "it is not {g}", "Bit isn't {g}.", False),
    ("a2_neg_suffix", "is not {g}", "It isn't {g}ish.", False),
    ("a2_neg_curly", "isn\u2019t {g}", "It isn't {g}.", True),
]


def neg_cases(p):
    """(name, probe copy with the negation gold, reply, expected pass) for each NEG entry; VAL probes that hold a
    gold and are not assistant-held OWN picks."""
    if p["grader"] != "VAL" or not p.get("gold") or GV.mode_of(p) == "own":
        return []
    out = []
    for name, form, tpl, want in NEG:
        q = copy.deepcopy(p)
        q["gold"] = form.format(g=p["gold"])
        q["accepted"] = [q["gold"]]
        out.append((name, q, tpl.format(g=p["gold"]), want))
    return out
