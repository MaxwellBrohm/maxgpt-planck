"""RC-12 grader fixtures for lock step A2b (2026-11-19; rc12/notes.txt STEP 16A2B, prereg s6 (x)): G-ABS on the
LOOKUP family's X probe (its one ABS probe: the key only the OTHER table holds) and the not-found replies the Planck
pipeline teaches (pipeline/SPEC.txt S9 "empty result reported as not found"; pipeline/lexicons.NOT_FOUND_RE).
grade_cues.LOOKUP_ABS, LOOKUP_ONLY and lookup_cue, applied by graders.g_abs on LOOKUP records only. Built from REAL
dev items with fixtures_graders.fx, over EVERY LOOKUP record (all 48 X probes and their P probes) and every ABS probe
of another family; mutation_graders.build adds the records it did not pick (records_and_fixtures).
  X *_ok      a not-found reply each cue form alone carries, with and without the key and the table, apologies before
              or after (every one failed a2_cue on the graders before A2b; "I couldn't find X", "I cannot locate X"
              and "I was unable to find X" also a4_voice), and the cut-at-the-cap copy of one (a2b_runaway_ok);
              X_OK_R1 (fix round, 2026-11-19, each failing on the A2b graders before it): no subject at a clause
              start after each mark, "we", "the lookup", "my search", each adverb, "seem to", "don't / do not see",
              "wasn't / was not able to", "wasn't / couldn't / cannot be found", "nothing (was) found", "I found
              nothing" (a4), "only on / in the other", each ONLY verb and object, a negation in the clause before
              the cue's (after a comma, semicolon, colon), and two pins: the table word capitalized, a negated hit
              then a clean one
  X the rest  guards, each must fail: a cue beside the lure or another pool value, a guess list, the cue negated
              or reversed ("nothing" and "nobody" too), "they" for "I", "only the <asked table> has", "only on the
              <asked table>", a far "has", an ONLY form whose object is not the key ("Only the old one has
              changed"), an adverb outside the list ("I almost couldn't find"), real user voice beside the frame,
              the degenerate replies (fixtures_graders.generic_degenerate around a must-pass reply)
  P           LOOKUP's VAL probe keeps its verdicts: the answer with a cue after it passes, the cue alone and the
              cue with the lure fail (G-VAL reads no abstain cue)
  scope       the same cues on every ABS probe of another family (RECALL abstain) fail, as before A2b, and so does a
              first-person not-found frame binding the asked object there (the a4 exemption is LOOKUP's only); and
              (late review) per cell a copy of a LOOKUP record relabelled OODH, OOD-H's family, fails a not-found
              reply with no value, so a scope widened to OOD-H alone is caught (a2b_scope_relabelled_oodh)
  X_OK_R2, X_FAIL_R2  the late review (2026-11-22, notes STEP 16A2B-R2): the openings that are no negation (each,
              after a comma, before a dash), a negation in the sentence before, "nothing" / "nobody" set off by a
              mark (each kind) and their window, LOOKUP_ON's anchors, the ONLY objects' ends, each passive found form,
              each "found nothing" end must pass; a negation before an aside, an opening mid-sentence or with no mark
              after, each NEG_PHRASE word and verb, LOOKUP_ON with no anchor, too far, across a mark, negated at
              "only" or with another preposition, the ONLY objects that do not end their clause, the active found
              forms, "found nothing" with a word after it, and the reviewer's subject / mark / auxiliary survivors
              must fail
  X_OK_R3, X_FAIL_R3  the late review's second round (2026-11-22, notes STEP 16A2B-R3): "none" / "no" set off by a
              mark, an opening after a comma and before a colon, an aside that keeps its comma or is followed by a
              subject, each verb that may follow an aside, quotes and colons after "nobody", the NEG_PHRASE word count,
              "not found" after a far "have" or after none, LOOKUP_ON's "'ll", "'d" and plural key, ONLY's plural key
              and each word that may end its object must pass; a negation before an aside of commas, dashes or
              brackets (with be, been, was, found or "been found" after it), a quoted "nothing", the new NEG_PHRASE
              subjects, words, asides and verbs ("no one" and "were" where the window cannot read them), an adverb
              inside the active perfect, a key or "the <key> <word>" or a row form that does not end its clause, and
              the second reviewer's survivors must fail
  X_OK_A2D, X_FAIL_A2D, X_CONSERVATIVE_A2D  lock step A2d (2026-11-23, notes STEP 16A2D: the cues read fail closed,
              grade_cues' module docstring): the forms it keeps (emphasis taken out, each sentence break, any clean
              cue, what may follow "no <word>", the ends and words of the ONLY and ON shapes) must pass; the wrong
              replies it closes (a negation inside or after an aside, in emphasis, farther back, each LOOKUP_NEG word
              and n't without its apostrophe, have / has / having before "not found" past a word or an aside, ONLY
              followed by ?, a comma, "and" or "but", ON with a predicate or more words, "no <word>" or "nothing came
              up" as a subject, "missing / absent from the <other table>", a cue asked as a question, an ellipsis, a
              struck-through cue) must fail; and the right replies it fails, moved from the must-pass lists under
              new names (a2d_conservative_<old name without a2b_ and _ok>), must fail and are recorded in prereg s6 (x)
  X_OK_A2D_FR, X_FAIL_A2D_FR, X_CONSERVATIVE_A2D_FR  A2d's fix round (2026-11-23, notes STEP 16A2D-R): the forms it
              keeps (emphasis round an opening, each mark an opening may take, a predicate word before the cue or in
              the next sentence, a question in the next sentence) must pass; the wrong replies it closes (a negation
              joined to a word by emphasis, a hyphen or quote or bracket after "No", each new LOOKUP_NEG word, n't
              without its apostrophe after the new auxiliaries, each apostrophe look-alike and 'nt, each predicate
              word after a quantifier or found-nothing cue, a question mark anywhere in the sentence, each strike
              mark and its reach, the other table's word as either of the two words after "missing / absent from",
              with any spacing, quote or bracket, a far "have") and the older quantifier and END rules with no
              predicate word to hide them must fail; and the right replies its closures fail (a2d_conservative_fr_*)
              must fail and are recorded in prereg s6 (x)
  echo        copies of the first LOOKUP record of each cell whose X question itself holds a not-found cue (marked
              synthetic; the conversation tests skip them): echoing that question fails (lookup_cue ignores the
              LOOKUP cues when the question holds one, so G-ABS still needs no echo clause, D12), for a find frame,
              an ONLY form and a curly apostrophe; a question whose cue is negated holds none (a2b_negq_ok: a reply's
              own cue counts there)
Table words: the asked table and the other are the record's table facts ("old" / "new" in fam_role_lookup, code, not
content); the reply never names the shape's label (content: the sealed session writes its own)."""
import copy

from fixtures_graders import cap, fx, generic_degenerate

X_OK = [
    ("a2b_couldnt_find_ok", "I couldn't find {k} on the {a} one."),
    ("a2b_could_not_anything_ok", "Sorry, I could not find anything on that."),       # pipeline fake teacher
    ("a2b_cant_see_ok", "I can't see {k} on the {a} list."),
    ("a2b_cannot_locate_ok", "I cannot locate {k} there."),
    ("a2b_didnt_spot_ok", "I didn't spot it, sorry."),
    ("a2b_did_not_find_ok", "I did not find {k} on the {a} one."),
    ("a2b_unable_find_ok", "I was unable to find that."),
    ("a2b_no_comma_ok", "No, I couldn't find {k} on the {a} one."),
    ("a2b_nothing_came_ok", "Nothing came up for that, sorry."),                       # pipeline fake teacher
    ("a2b_nothing_turned_ok", "Nothing turned up for {k} on the {a} one."),
    ("a2b_nothing_showed_ok", "Sorry, nothing showed up for {k}."),
    ("a2b_turned_up_nothing_ok", "Checking the {a} one turned up nothing for {k}."),
    ("a2b_found_nothing_ok", "Found nothing for {k} on the {a} list."),
    ("a2b_came_up_empty_ok", "{k} came up empty on the {a} one."),
    ("a2b_not_found_ok", "{k}: not found on the {a} one."),
    ("a2b_no_results_ok", "No results for {k} on the {a} one."),
    ("a2b_no_match_ok", "No match for {k}, sorry."),
    ("a2b_no_entry_ok", "No entry for {k} on the {a} list."),
    ("a2b_no_row_ok", "Sorry, no row for {k} there."),
    ("a2b_no_record_ok", "No record of {k} on the {a} one."),
    ("a2b_no_information_ok", "No information on {k} in the {a} one."),
    ("a2b_no_luck_ok", "No luck with {k} on the {a} list."),
    ("a2b_missing_ok", "{k} is missing from the {a} one."),
    ("a2b_absent_ok", "{k} is absent from the {a} list."),
    ("a2b_only_other_has_ok", "Only the {o} one has {k}."),
    ("a2b_only_other_lists_ok", "Only the other one lists {k}."),
    ("a2b_only_two_words_ok", "Only the {o} one there has {k}."),
    ("a2b_afraid_not_ok", "I'm afraid not. No results for {k} on the {a} one."),    # a negation in the sentence before
]
X_OK_R1 = [                                   # fix round (2026-11-19): forms the pipeline accepts and their kin
    ("a2b_subjless_start_ok", "Couldn't find {k} on the {a} one."),
    ("a2b_subjless_comma_ok", "Sorry, couldn't find anything on that."),
    ("a2b_subjless_period_ok", "Sorry. Couldn't find {k} on the {a} one."),
    ("a2b_subjless_bang_ok", "Sorry! Couldn't find {k}."),
    ("a2b_subjless_q_ok", "{k}? Couldn't find it on the {a} one."),
    ("a2b_subjless_semi_ok", "Sorry; couldn't find {k}."),
    ("a2b_subjless_colon_ok", "Checked the {a} one: couldn't find {k}."),
    ("a2b_we_ok", "We couldn't find {k} on the {a} one."),
    ("a2b_the_lookup_ok", "The lookup couldn't find {k}."),
    ("a2b_my_search_ok", "My search didn't find {k} on the {a} list."),
    ("a2b_still_ok", "I still couldn't find {k} on the {a} one."),
    ("a2b_just_ok", "I just can't find {k}."),
    ("a2b_really_ok", "I really can't find {k} on the {a} list."),
    ("a2b_honestly_ok", "I honestly couldn't find {k}."),
    ("a2b_simply_ok", "I simply couldn't find {k} there."),
    ("a2b_seem_ok", "I can't seem to find {k} on the {a} one."),
    ("a2b_dont_see_ok", "I don't see {k} on the {a} one."),
    ("a2b_do_not_see_ok", "I do not see {k} there."),
    ("a2b_wasnt_able_ok", "I wasn't able to find {k} on the {a} one."),
    ("a2b_was_not_able_ok", "I was not able to locate {k}."),
    ("a2b_wasnt_found_ok", "{k} wasn't found on the {a} one."),
    ("a2b_be_found_ok", "{k} couldn't be found on the {a} list."),
    ("a2b_cannot_be_found_ok", "{k} cannot be found there."),
    ("a2b_nothing_found_ok", "Nothing found for {k} on the {a} one."),
    ("a2b_nothing_was_found_ok", "Nothing was found for {k}."),
    ("a2b_i_found_nothing_ok", "I found nothing for {k} on the {a} one."),           # the a4 frame
    ("a2b_only_on_ok", "{k} is only on the {o} one."),
    ("a2b_only_in_ok", "{k} is only in the {o} list."),
    ("a2b_only_includes_ok", "Only the {o} one includes {k}."),
    ("a2b_only_has_it_ok", "Only the {o} one has it."),
    ("a2b_only_has_that_ok", "Only the {o} one has that."),
    ("a2b_only_cap_ok", "Only the {O} one has {k}."),                                # the table word in any case
    ("a2b_only_on_cap_ok", "{k} is only on the {O} one."),
    ("a2b_afraid_not_comma_ok", "I'm afraid not, I couldn't find {k} on the {a} one."),  # a negation in the clause
    ("a2b_dont_think_semi_ok", "I don't think so; nothing came up for {k}."),            # before, not in the cue's
    ("a2b_sadly_not_colon_ok", "Sadly not: no results for {k}."),
]
X_FAIL = [
    ("a2b_find_lure", "I couldn't find {k} on the {a} one, but it's {lure}."),
    ("a2b_results_guess", "No results, so probably {lure}."),
    ("a2b_nothing_q_lure", "Nothing came up. {lure}?"),
    ("a2b_missing_then_other", "{k} is missing from the {a} one; the {o} one has {lure}."),
    ("a2b_guess_list", "No match for {k}. Maybe {c1}, {c2} or {c3}?"),
    ("a2b_found_value", "I found {k}: {lure}."),
    ("a2b_didnt_fail", "I didn't fail to find it."),
    ("a2b_results_for", "There are results for {k} on the {a} one."),
    ("a2b_not_missing", "{k} isn't missing from the {a} one."),
    ("a2b_never_absent", "{k} is never absent from the {a} list."),
    ("a2b_not_only", "Not only the {o} one has {k}."),
    ("a2b_they_couldnt", "They couldn't find {k} on the {a} one."),
    ("a2b_only_asked", "Only the {a} one has {k}."),
    ("a2b_only_far", "Only the {o} one is short and the {a} one has {k}."),
    ("a2b_voice_on_key", "I couldn't find {k} on the {a} one. I am on {k} myself."),
    ("a2b_voice_my", "I couldn't find {k} on the {a} one. My {k} is free."),
    ("a2b_voice_same_sent", "I couldn't find {k} because I work {k}."),
    ("a2b_nothing_missing", "Nothing is missing from the {a} one."),                 # fix round (2026-11-19)
    ("a2b_nobody_missing", "Nobody is missing from the {a} list; {k} is covered."),
    ("a2b_only_changed", "Only the {o} one has changed; the {a} one still has {k}."),
    ("a2b_only_more_rows", "Only the other one has more rows, but the {a} one has {k} too."),
    ("a2b_only_on_asked", "{k} is only on the {a} one."),
    ("a2b_not_only_on", "Not only on the {o} one; {k} is on the {a} one too."),
    ("a2b_almost_couldnt", "I almost couldn't find {k} on the {a} one, but it's there."),
]
X_OK_R2 = [                                   # late review (2026-11-22, notes STEP 16A2B-R2)
    ("a2b_r2_afraid_ok", "Afraid not, I couldn't find {k} on the {a} one."),       # the openings (INTERJ)
    ("a2b_r2_i_am_afraid_ok", "I am afraid not, no results for {k}."),
    ("a2b_r2_unfortunately_ok", "Unfortunately not, nothing came up for {k} on the {a} one."),
    ("a2b_r2_guess_ok", "Guess not, couldn't find {k} on the {a} list."),
    ("a2b_r2_i_guess_ok", "I guess not, I couldn't find {k}."),
    ("a2b_r2_probably_ok", "Probably not, I couldn't find {k} on the {a} one."),
    ("a2b_r2_apparently_ok", "Apparently not, {k} is missing from the {a} one."),
    ("a2b_r2_do_not_think_ok", "I do not think so, no results for {k}."),
    ("a2b_r2_dont_think_noi_ok", "Don't think so, no results for {k}."),
    ("a2b_r2_not_sure_ok", "Not sure, but I couldn't find {k} on the {a} one."),
    ("a2b_r2_not_really_ok", "Not really, I couldn't find {k} on the {a} one."),
    ("a2b_r2_not_this_time_ok", "Not this time; I couldn't find {k}."),
    ("a2b_r2_interj_after_comma_ok", "Sorry, I'm afraid not, I couldn't find {k} on the {a} one."),
    ("a2b_r2_interj_dash_ok", "I'm afraid not - I couldn't find {k} on the {a} one."),
    ("a2b_r2_prev_sentence_ok", "Sorry, it isn't there. No results for {k} on the {a} one."),
    ("a2b_r2_on_it_ok", "It is only on the {o} one."),                                 # LOOKUP_ON's anchors
    ("a2b_r2_on_key_s_ok", "{k}'s only on the {o} one."),
    ("a2b_r2_only_it_end_ok", "Only the {o} one has it"),                              # ONLY objects
    ("a2b_r2_could_not_be_found_ok", "{k} could not be found on the {a} one."),        # the passive found forms
    ("a2b_r2_werent_found_ok", "Entries for {k} weren't found on the {a} one."),
    ("a2b_r2_isnt_found_ok", "{k} isn't found on the {a} list."),
    ("a2b_r2_arent_found_ok", "Rows for {k} aren't found on the {a} one."),
    ("a2b_r2_cant_be_found_ok", "{k} can't be found on the {a} one."),
    ("a2b_r2_wont_be_found_ok", "{k} won't be found on the {a} list."),
    ("a2b_r2_end_on_ok", "Nothing was found on the {a} one for {k}."),                # "found nothing" ends
    ("a2b_r2_end_in_ok", "Found nothing in the {a} one for {k}."),
    ("a2b_r2_end_about_ok", "Found nothing about {k}, sorry."),
    ("a2b_r2_end_mark_ok", "{k}? Nothing was found."),
    ("a2b_r2_end_end_ok", "{k} on the {a} one: found nothing"),
]
X_FAIL_R2 = [
    ("a2b_r2_isnt_aside", "{k} isn't, honestly, missing from the {a} one."),          # a negation before an aside
    ("a2b_r2_not_aside_only", "Not, mind you, only the {o} one has {k}; the {a} one has it too."),
    ("a2b_r2_interj_midsentence", "{k} is probably not, honestly, missing from the {a} one."),
    ("a2b_r2_interj_no_mark", "Not really missing from the {a} one: {k} is there."),
    ("a2b_r2_nobody_aside", "Nobody, it seems, is missing from the {a} list; {k} is covered."),   # NEG_PHRASE
    ("a2b_r2_nothing_aside_was", "Nothing, so far, was missing from the {a} one."),
    ("a2b_r2_no_one_aside_were", "No one, it seems, were missing from the {a} one."),
    ("a2b_r2_none_aside_are", "None, it seems, are missing from the {a} one."),
    ("a2b_r2_nothing_at_all", "Nothing at all is missing from the {a} one."),
    ("a2b_r2_nobodys_missing", "Nobody's missing from the {a} one; {k} is covered."),
    ("a2b_r2_on_extra_rows", "The {a} one has {k}; the extra rows are only on the {o} one."),     # LOOKUP_ON
    ("a2b_r2_on_changes", "The {a} one still has {k}; the changes are only in the {o} one."),
    ("a2b_r2_on_other_literal", "{k} is on the {a} one; the crossed-out names are only on the other one."),
    ("a2b_r2_on_far", "{k} is on the {a} one and the extra rows are only on the {o} one."),
    ("a2b_r2_on_marks", "The {a} one has {k}; changes are only on the {o} one."),
    ("a2b_r2_on_negated", "{k} is not only on the {o} one; the {a} one has it too."),
    ("a2b_r2_on_about", "It is only about the {o} one; {k} is on the {a} one."),
    ("a2b_r2_only_that_problem", "Only the {o} one has that problem; the {a} one has {k} covered."),  # ONLY objects
    ("a2b_r2_only_it_wrong", "Only the {o} one has it wrong; {k} is on the {a} one."),
    ("a2b_r2_only_line_crossed", "Only the {o} one has a line crossed out; {k} is on the {a} one."),
    ("a2b_r2_only_entry_blank", "Only the {o} one has an entry left blank; the {a} one has {k}."),
    ("a2b_r2_only_shows_it_crossed", "Only the {o} one shows it crossed out; the {a} one still has {k}."),
    ("a2b_r2_only_that_typo", "Only the {o} one contains that typo; {k} is fine on the {a} one."),
    ("a2b_r2_only_it_holiday", "Only the {o} one includes it as a holiday; the {a} one has {k} as normal."),
    ("a2b_r2_only_its_typo", "Only the {o} one has it's own row; {k} is on the {a} one."),
    ("a2b_r2_only_a_gap", "Only the {o} one has a gap; the {a} one has {k}."),
    ("a2b_r2_only_changed_it", "Only the {o} one changed it; the {a} one has {k}."),
    ("a2b_r2_only_this_problem", "Only the {o} one has this problem; the {a} one has {k}."),
    ("a2b_r2_havent_found", "I haven't found any gap. The {a} one has it covered."),          # active found forms
    ("a2b_r2_hasnt_found", "The office hasn't found a gap; {k} is on the {a} one."),
    ("a2b_r2_have_not_found", "We have not found any gap: {k} is on the {a} one."),
    ("a2b_r2_has_not_found", "The office has not found a problem; {k} is on the {a} one."),
    ("a2b_r2_had_not_found", "We had not found a gap; {k} is on the {a} one."),
    ("a2b_r2_ve_not_found", "We've not found any gap; {k} is on the {a} one."),
    ("a2b_r2_was_found", "{k} was found on the {a} one."),
    ("a2b_r2_found_nothing_wrong", "I found nothing wrong: {k} is on the {a} one."),           # "found nothing" ends
    ("a2b_r2_found_nothing_missing", "I found nothing missing from the {a} one; {k} is covered."),
    ("a2b_r2_nosubj_found_nothing", "Found nothing wrong: {k} is on the {a} one."),
    ("a2b_r2_nothing_found_wrong", "Nothing was found wrong with the {a} one; {k} is covered."),
    ("a2b_r2_nothing_new_found", "Nothing new found: {k} is on the {a} one."),
    ("a2b_r2_she_couldnt", "She couldn't find {k} on the {a} one."),                             # LOOKUP_FIND pins
    ("a2b_r2_and_couldnt", "They looked and couldn't find {k} on the {a} one."),
    ("a2b_r2_dash_couldnt", "They checked - couldn't find {k} on the {a} one."),
    ("a2b_r2_wouldnt_see", "I wouldn't see why not: {k} is on the {a} one."),
]
X_OK_R3 = [                                   # late review round 2 (2026-11-22, notes STEP 16A2B-R3)
    ("a2b_r3_no_dash_ok", "No - {k} is missing from the {a} one."),
    ("a2b_r3_no_sadly_ok", "No, sadly, I couldn't find {k} on the {a} one."),         # an aside keeps its comma
    ("a2b_r3_interj_comma_colon_ok", "Sorry, afraid not: no results for {k}."),       # an opening after a comma
    ("a2b_r3_has_gone_missing_ok", "{k} has gone missing from the {a} one."),         # the perfect: "not found" only
    ("a2b_r3_on_plural_ok", "{k}s are only on the {o} one."),
    ("a2b_r3_only_plural_ok", "Only the {o} one has {k}s."),                          # ONLY's key and its end
]
X_FAIL_R3 = [
    ("a2b_r3_nothing_dash_aside", "Nothing - really - is missing from the {a} one; {k} is on it."),   # asides
    ("a2b_r3_nobody_paren_aside", "Nobody (really) is missing from the {a} one; {k} is covered."),
    ("a2b_r3_nothing_em_aside", "Nothing {em} honestly {em} is missing from the {a} one."),
    ("a2b_r3_nothing_quoted", '"Nothing" is missing from the {a} one; {k} is on it.'),
    ("a2b_r3_nobody_words_dash", "Nobody at all - really - is missing from the {a} one; {k} is covered."),
    ("a2b_r3_nobody_words_paren", "Nobody at all (really) is missing from the {a} one; {k} is covered."),
    ("a2b_r3_nothing_words_comma", "Nothing at all, honestly, is missing from the {a} one; {k} is on it."),
    ("a2b_r3_isnt_long_aside", "{k} isn't, as far as I can tell, missing from the {a} one."),
    ("a2b_r3_not_dash_aside", "{k} is not - as far as I can tell - missing from the {a} one."),
    ("a2b_r3_not_paren_aside", "{k} is not (as far as I can tell) missing from the {a} one."),
    ("a2b_r3_never_aside_been", "{k} has never, in any version, been missing from the {a} one."),
    ("a2b_r3_will_not_aside_be", "{k} will not, in any version, be missing from the {a} one."),
    ("a2b_r3_nothing_aside_was_found", "Nothing - at all - was found missing from the {a} one."),
    ("a2b_r3_nothing_comma_found", "Nothing, honestly, was found missing from the {a} one; {k} is on it."),
    ("a2b_r3_not_aside_been_found", "{k} has not, as far as I know, been found missing from the {a} one."),
    ("a2b_r3_not_aside_found", "{k} was not, as far as I know, found missing from the {a} one."),
    ("a2b_r3_no_day_aside_was", "No day, it seems, was missing from the {a} one; {k} is on it."),
    ("a2b_r3_not_a_single", "Not a single day is missing from the {a} one; {k} is on it."),       # NEG_PHRASE
    ("a2b_r3_not_one", "Not one day is missing from the {a} one; {k} is on it."),
    ("a2b_r3_no_single", "No single day is missing from the {a} one, so {k} is covered."),
    ("a2b_r3_none_of_the", "None of the days is missing from the {a} one; {k} is on it."),
    ("a2b_r3_found_to_be", "Nothing was found to be missing from the {a} one; {k} is on it."),
    ("a2b_r3_nobody_has_been", "Nobody at all has been missing from the {a} one; {k} is covered."),
    ("a2b_r3_no_one_words_aside", "No one at all, it seems, is missing from the {a} one; {k} is covered."),
    ("a2b_r3_none_of_them_were", "None of them, it seems, were missing from the {a} one; {k} is on it."),
    ("a2b_r3_has_also_not_found", "The office has also not found a gap; {k} is on the {a} one."),  # the perfect
    ("a2b_r3_have_honestly_not", "We have honestly not found any problem: {k} is on the {a} one."),
    ("a2b_r3_had_really_not", "The office had really not found a gap; {k} is on the {a} one."),
    ("a2b_r3_ve_also_not", "We've also not found a gap; {k} is on the {a} one."),
    ("a2b_r3_only_key_crossed", "Only the {o} one has {k} crossed out; the {a} one has {k} too."),  # ONLY's end
    ("a2b_r3_only_key_twice", "Only the {o} one lists {k} twice; the {a} one lists it once."),
    ("a2b_r3_only_slot_crossed", "Only the {o} one has the {k} slot crossed out; the {a} one has it too."),
    ("a2b_r3_only_a_line", "Only the {o} one has a line; the {a} one has {k}."),
    ("a2b_r3_row_for_other", "Only the {o} one has a row for notes; the {a} one has {k}."),
    ("a2b_r3_split_only", "The split is only on the {o} one; {k} is on the {a} one."),    # the reviewer's survivors
    ("a2b_r3_nobodys_ever", "Nobody's ever really missing from the {a} one; {k} is covered."),
    ("a2b_r3_found_nothing_inconsistent", "I found nothing inconsistent: {k} is on the {a} one."),
    ("a2b_r3_nothing_found_online", "Nothing found online; {k} is on the {a} one."),
    ("a2b_r3_nothing_seems", "Nothing really seems missing from the {a} one; {k} is on it."),
]
X_OK_A2D = [                                  # A2d (2026-11-23, notes STEP 16A2D): the forms the rule keeps
    ("a2d_md_star_ok", "Only the {o} one has **{k}**."),                         # (a) * _ and backticks taken out
    ("a2d_md_under_ok", "Only the {o} one has _{k}_."),
    ("a2d_md_tick_ok", "Only the {o} one has `{k}`."),
    ("a2d_bang_ok", "Sadly, it isn't there! No results for {k} on the {a} one."),      # the sentence ends
    ("a2d_newline_ok", "Not there\nNo results for {k} on the {a} one."),
    ("a2d_q_break_ok", "Isn't it on the {a} one? No, {k} is missing from it."),
    ("a2d_quote_break_ok", 'It said "not here." No results for {k} on the {a} one.'),
    ("a2d_two_hits_ok", "Missing from the {a} one? Yes, {k} is missing from the {a} one."),   # any clean cue
    ("a2d_no_results_found_ok", "No results found for {k} on the {a} one."),     # (g) what may follow "no <word>"
    ("a2d_no_entry_in_ok", "No entry in the {a} one for {k}."),
    ("a2d_no_info_about_ok", "No information about {k} on the {a} one."),
    ("a2d_no_results_mark_ok", "Sorry, no results."),
    ("a2d_no_results_end_ok", "{k} on the {a} one: no results"),
    ("a2d_only_bang_ok", "Only the {o} one has {k}!"),                           # (e) the shapes' ends and words
    ("a2d_only_semi_ok", "Only the {o} one has {k}; the {a} one does not."),
    ("a2d_only_newline_ok", "Only the {o} one has {k}\nSorry about that."),
    ("a2d_on_was_ok", "{k} was only on the {o} one."),
    ("a2d_on_were_ok", "{k}s were only on the {o} one."),
    ("a2d_on_two_words_ok", "{k} is only on the {o} one there."),
]
X_FAIL_A2D = [
    ("a2d_dash_not", "{k} is - not - missing from the {a} one."),                 # (c) a negation inside an aside
    ("a2d_paren_not", "{k} is (not) missing from the {a} one."),
    ("a2d_paren_definitely_not", "{k} is (definitely not) missing from the {a} one."),
    ("a2d_em_definitely_not", "{k} is{em}definitely not{em}missing from the {a} one; it's there."),
    ("a2d_paren_no_longer", "{k} was (no longer) missing from the {a} one."),
    ("a2d_dash_isnt_now", "{k} was - but isn't now - missing from the {a} one."),
    ("a2d_em_not_anymore", "{k} is{em}not anymore{em}missing from the {a} one."),
    ("a2d_used_to_isnt", "{k} used to be, but isn't now, missing from the {a} one."),
    ("a2d_was_no_longer_is", "{k} was, but no longer is, missing from the {a} one."),
    ("a2d_comma_not", "{k} is, not, missing from the {a} one."),
    ("a2d_stacked_paren_comma", "{k} isn't (as far as I know), honestly, missing from the {a} one."),
    ("a2d_stacked_dash_comma", "{k} isn't - as far as I can tell - honestly, missing from the {a} one."),
    ("a2d_stacked_three", "{k} isn't, I think, honestly, really, missing from the {a} one."),
    ("a2d_dd_nothing", "Nothing--really--is missing from the {a} one; {k} is on it."),
    ("a2d_dd_isnt", "{k} isn't--as far as I can tell--missing from the {a} one."),
    ("a2d_bracket_isnt", "{k} isn't [as far as I can tell] missing from the {a} one."),
    ("a2d_hyphen_nothing", "Nothing-really-is missing from the {a} one; {k} is on it."),
    ("a2d_not_really_at_all", "{k} is not really at all missing from the {a} one."),
    ("a2d_nothing_from_other", "Nothing from the {o} one is missing from the {a} one; {k} is there."),
    ("a2d_md_bold_nothing", "**Nothing** is missing from the {a} one; {k} is on it."),       # (a) emphasis
    ("a2d_md_ital_nothing", "*Nothing* is missing from the {a} one; {k} is on it."),
    ("a2d_md_under_nothing", "_Nothing_ is missing from the {a} one; {k} is on it."),
    ("a2d_md_bold_nobody", "**Nobody** is missing from the {a} one; {k} is covered."),
    ("a2d_md_bold_not_single", "**Not a single** day is missing from the {a} one; {k} is on it."),
    ("a2d_md_bold_none_of", "**None** of the days is missing from the {a} one; {k} is on it."),
    ("a2d_nowhere", "Nowhere is {k} missing from the {a} one."),                   # (c) each word of the list
    ("a2d_neither", "Neither of them is missing from the {a} one; {k} is on it."),
    ("a2d_nor", "The next day is there, nor is {k} missing from the {a} one."),
    ("a2d_without", "{k} is on the {a} one without anything missing from it."),
    ("a2d_cannot", "{k} cannot be missing from the {a} one."),
    ("a2d_hardly", "{k} is hardly missing from the {a} one; it's right there."),
    ("a2d_scarcely", "{k} is scarcely absent from the {a} one."),
    ("a2d_barely", "{k} is barely missing from the {a} one; it's right there."),
    ("a2d_far_from", "{k} is far from missing from the {a} one."),
    ("a2d_anything_but", "{k} is anything but missing from the {a} one."),
    ("a2d_instead_of", "Instead of {k}, the next day is missing from the {a} one."),
    ("a2d_rather_than", "The next day rather than {k} is missing from the {a} one."),
    ("a2d_isnt", "{k} isnt missing from the {a} one."),                           # n't without the apostrophe
    ("a2d_arent", "{k}s arent missing from the {a} one."),
    ("a2d_wasnt", "{k} wasnt missing from the {a} one."),
    ("a2d_werent", "{k}s werent missing from the {a} one."),
    ("a2d_dont", "{k}s dont go missing from the {a} one."),
    ("a2d_doesnt", "{k} doesnt go missing from the {a} one."),
    ("a2d_didnt", "{k} didnt go missing from the {a} one."),
    ("a2d_hasnt", "{k} hasnt gone missing from the {a} one."),
    ("a2d_havent", "{k}s havent gone missing from the {a} one."),
    ("a2d_hadnt", "{k} hadnt gone missing from the {a} one."),
    ("a2d_cant", "{k} cant be missing from the {a} one."),
    ("a2d_couldnt", "{k} couldnt be missing from the {a} one."),
    ("a2d_wont", "{k} wont be missing from the {a} one."),
    ("a2d_has_comma_not_found", "The office has, so far, not found a gap; {k} is on the {a} one."),   # (d)
    ("a2d_has_dash_not_found", "The office has - so far - not found a gap; {k} is on the {a} one."),
    ("a2d_has_paren_not_found", "The office has (so far) not found a gap; {k} is on the {a} one."),
    ("a2d_has_two_words_not_found", "The office has also still not found a gap; {k} is on the {a} one."),
    ("a2d_having_not_found", "Having not found any gap, I can say {k} is on the {a} one."),
    ("a2d_only_q", "Only the {o} one has {k}? No, the {a} one has it too."),       # (e) ONLY: its end
    ("a2d_only_comma_but", "Only the {o} one has {k}, but the {a} one has it too."),
    ("a2d_only_and_swapped", "Only the {o} one has {k} and its neighbour swapped; the {a} one has them in order."),
    ("a2d_only_and_so_does", "Only the {o} one has {k} and so does the {a} one."),
    ("a2d_only_listed_and", "Only the {o} one has {k} listed and crossed out; the {a} one has it plainly."),
    ("a2d_only_row_and", "Only the {o} one has a row for {k} and the next one merged; the {a} one splits them."),
    ("a2d_only_slot_and", "Only the {o} one has the {k} slot and the next slot merged; the {a} one splits them."),
    ("a2d_only_but_not_bold", "Only the {o} one has {k} but not in bold; the {a} one has {k} too."),
    ("a2d_only_on_it_and", "Only the {o} one has {k} on it and in red; the {a} one has it in black."),
    ("a2d_only_the_gap", "Only the {o} one has the {k} gap; the {a} one fills it."),
    ("a2d_only_the_twice", "Only the {o} one lists the {k} twice; the {a} one has it."),
    ("a2d_on_other_side", "{k} is only on the other side of the {a} one."),       # (e) ON: its words and end
    ("a2d_on_other_column", "{k} is only in the other column of the {a} one."),
    ("a2d_on_tail_three", "{k} is only on the other side of it."),
    ("a2d_on_cleft", "{k} is on the {a} one; it's only on the {o} one that the order differs."),
    ("a2d_on_crossed", "{k} is crossed out only on the {o} one; the {a} one has it."),
    ("a2d_on_gap", "{k} shows a gap only on the {o} one."),
    ("a2d_on_typo", "{k} has a typo only in the {o} one; the {a} one lists it fine."),
    ("a2d_on_swapped", "{k} was swapped only on the {o} one."),
    ("a2d_on_plural_pred", "{k}s are busier only on the {o} one; the {a} one has {k} too."),
    ("a2d_on_itd", "{k} is on the {a} one; it'd matter only on the {o} one."),
    ("a2d_on_thats", "{k} is on the {a} one. The typo? That's only on the {o} one."),
    ("a2d_on_key_poss", "{k}'s slot is empty only on the {o} one."),
    ("a2d_on_needs_only", "{k} is on the {o} one."),
    ("a2d_no_rows_are", "No rows are missing from the {a} one; {k} is on it."),       # (g) "no <word>", "nothing came up"
    ("a2d_no_entries_are", "No entries are missing from the {a} one; {k} is there."),
    ("a2d_no_records_colon", "No records are missing: {k} is on the {a} one."),
    ("a2d_no_info_missing", "No information is missing: {k} is on the {a} one."),
    ("a2d_nothing_showed_as", "Nothing showed up as missing from the {a} one; {k} is on it."),
    ("a2d_nothing_came_missing", "Nothing came up missing on the {a} one; {k} is there."),
    ("a2d_nothing_turned_missing", "Nothing turned up missing from the {a} one; {k} is on it."),
    ("a2d_missing_from_other", "{k} is missing from the {o} one; the {a} one has it."),   # (j) the other table
    ("a2d_absent_from_other", "{k} is absent from the {o} one; the {a} one has it."),
    ("a2d_missing_from_other_word", "{k} is missing from the other one; the {a} one has it."),
    ("a2d_missing_from_other_cap", "{k} is missing from the {O} one; the {a} one has it."),
    ("a2d_was_missing_other_not_now", "{k} was missing from the {o} one, but is not, now, missing from the {a} one."),
    ("a2d_q_missing", "Missing from the {a} one? No, {k} is right there."),          # a cue asked as a question
    ("a2d_q_is_missing", "Is {k} missing from the {a} one? No, it's there."),
    ("a2d_q_statement", "{k} is missing from the {a} one? No, it's right there."),
    ("a2d_q_couldnt", "Couldn't find {k}? It's right there on the {a} one."),
    ("a2d_q_interrobang", "{k} is missing from the {a} one?! No, it's right there."),
    ("a2d_ellipsis_isnt", "{k} isn't... missing from the {a} one, it's right there."),   # a run of dots ends no sentence
    ("a2d_ellipsis_nothing", "Nothing... is missing from the {a} one; {k} is on it."),
    ("a2d_strike", "{k} is ~~missing from~~ on the {a} one."),                      # struck-through text
]
X_CONSERVATIVE_A2D = [                        # A2d (2026-11-23, notes STEP 16A2D): right replies the fail-closed
    # rule fails, moved from the must-pass lists (their old names: a2b_<name>_ok); recorded in prereg s6 (x)
    ("a2d_conservative_only_contains", "Only the {o} one contains {k}."),
    ("a2d_conservative_only_shows", "Only the {o} list shows {k}."),
    ("a2d_conservative_only_a_row", "Only the {o} one has a row for {k}."),
    ("a2d_conservative_only_an_entry", "Only the {o} one has an entry for {k}."),
    ("a2d_conservative_only_a_line", "Only the {o} one has a line for {k}."),
    ("a2d_conservative_only_the_key", "Only the {o} one has the {k} slot."),
    ("a2d_conservative_two_hits", "{k} isn't missing from the {o} one; it's missing from the {a} one."),
    ("a2d_conservative_r2_nobody_comma", "Nobody, {k} is missing from the {a} one."),
    ("a2d_conservative_r2_nobody_em", "Nobody {em} {k} is missing from the {a} one."),
    ("a2d_conservative_r2_nothing_en", "Nothing {en} {k} is missing from the {a} one."),
    ("a2d_conservative_r2_nobody_hyphen", "Nobody - {k} is missing from the {a} list."),
    ("a2d_conservative_r2_nothing_window", "Nothing on the {a} one covers {k} because it is missing from it."),
    ("a2d_conservative_r2_phrase_end", "Nothing else is wrong, but {k} is missing from the {a} one."),
    ("a2d_conservative_r2_on_that", "That day is only on the {o} one."),
    ("a2d_conservative_r2_on_three_words", "{k} is still listed only on the {o} one."),
    ("a2d_conservative_r2_only_entry_on", "Only the {o} one has an entry on {k}."),
    ("a2d_conservative_r3_none_comma", "None, {k} is missing from the {a} one."),
    ("a2d_conservative_r3_none_em", "None {em} {k} is missing from the {a} one."),
    ("a2d_conservative_r3_tail_word", "{k} isn't there, sadly, it's missing from the {a} one."),
    ("a2d_conservative_r3_not_others_is", "{k}, not others, is missing from the {a} one."),
    ("a2d_conservative_r3_not_others_are", "{k}s, not others, are missing from the {a} one."),
    ("a2d_conservative_r3_not_others_were", "{k}s, not others, were missing from the {a} one."),
    ("a2d_conservative_r3_nobody_quote", '"Nobody" - {k} is missing from the {a} one.'),
    ("a2d_conservative_r3_nobody_colon", "Nobody: {k} is missing from the {a} one."),
    ("a2d_conservative_r3_nobody_noticed", "Nobody has noticed that {k} is missing from the {a} one."),
    ("a2d_conservative_r3_have_here", "The {k} rows you have here are not found on the {a} one."),
    ("a2d_conservative_r3_on_itll", "It'll be only on the {o} one."),
    ("a2d_conservative_r3_on_thatd", "That'd be only on the {o} one."),
    ("a2d_conservative_r3_only_on_it", "Only the {o} one has {k} on it."),
    ("a2d_conservative_r3_only_in_there", "Only the {o} one has {k} in there."),
    ("a2d_conservative_r3_only_listed", "Only the {o} one has {k} listed."),
    ("a2d_conservative_r3_only_and", "Only the {o} one has {k} and the {a} one does not."),
    ("a2d_conservative_r3_only_but", "Only the {o} one has {k} but the {a} one does not."),
]
X_OK_A2D_FR = [                               # A2d's fix round (2026-11-23, notes STEP 16A2D-R): the forms it keeps
    ("a2d_fr_md_opener_ok", "**No**, I couldn't find {k} on the {a} one."),       # emphasis read as a space too
    ("a2d_fr_opener_dots_ok", "No... I couldn't find {k} on the {a} one."),       # each mark an opening may take
    ("a2d_fr_opener_en_ok", "No {en} {k} is missing from the {a} one."),
    ("a2d_fr_opener_em_ok", "No{em}{k} is missing from the {a} one."),
    ("a2d_fr_opener_ellipsis_ok", "No\u2026 I couldn't find {k} on the {a} one."),
    ("a2d_fr_pred_before_ok", "{k} is on the {o} one: no results on the {a} one."),   # a predicate only after the cue
    ("a2d_fr_pred_next_sentence_ok", "No results for {k} on the {a} one. It is on the {o} one."),
    ("a2d_fr_q_later_ok", "No results for {k} on the {a} one. Is it on the {o} one?"),   # "?" in its own sentence
]
X_FAIL_A2D_FR = [
    ("a2d_fr_md_join_star", "{k} is**not** missing from the {a} one."),           # (a) emphasis read as a space too
    ("a2d_fr_md_join_under", "{k} is__not__ missing from the {a} one."),
    ("a2d_fr_md_join_tick", "{k} is`not` missing from the {a} one."),
    ("a2d_fr_md_join_snake", "{k} is not_really missing from the {a} one."),
    ("a2d_fr_md_join_never", "{k} has**never** been missing from the {a} one."),
    ("a2d_fr_md_tick_isnt", "{k} isn`t missing from the {a} one."),              # and taken out
    ("a2d_fr_noone_hyphen", "No-one is missing from the {a} one; {k} is covered."),   # (b) an opening's mark
    ("a2d_fr_sorry_noone", "Sorry, no-one is missing from the {a} one; {k} is covered."),
    ("a2d_fr_no_quote", 'No "missing from the {a} one" here; {k} is on it.'),
    ("a2d_fr_no_paren", "No (missing from the {a} one) flag; {k} is on it."),
    ("a2d_fr_noone", "Noone is missing from the {a} one; {k} is covered."),       # (c) each word of the list
    ("a2d_fr_zero", "Zero days are missing from the {a} one; {k} is on it."),
    ("a2d_fr_nary", "Nary a day is missing from the {a} one; {k} is on it."),
    ("a2d_fr_nada", "Nada is missing from the {a} one; {k} is on it."),
    ("a2d_fr_naught", "Naught is missing from the {a} one; {k} is on it."),
    ("a2d_fr_nothin", "Nothin' is missing from the {a} one; {k} is on it."),
    ("a2d_fr_aint", "{k} aint missing from the {a} one."),
    ("a2d_fr_shouldnt", "{k} shouldnt be missing from the {a} one; it's there."),
    ("a2d_fr_wouldnt", "{k} wouldnt be missing from the {a} one; it's there."),
    ("a2d_fr_mustnt", "{k} mustnt be missing from the {a} one; it's there."),
    ("a2d_fr_neednt", "{k} neednt be missing from the {a} one; it's there."),
    ("a2d_fr_mightnt", "{k} mightnt be missing from the {a} one; it's there."),
    ("a2d_fr_apos_acute", "{k} isn\u00b4t missing from the {a} one."),           # apostrophe look-alikes
    ("a2d_fr_apos_modifier", "{k} isn\u02bct missing from the {a} one."),
    ("a2d_fr_apos_prime", "{k} isn\u2032t missing from the {a} one."),
    ("a2d_fr_apos_fullwidth", "{k} isn\uff07t missing from the {a} one."),
    ("a2d_fr_apos_reversed", "{k} isn\u201bt missing from the {a} one."),
    ("a2d_fr_apos_misplaced", "{k} is'nt missing from the {a} one."),
    ("a2d_fr_pred_is", "No row for {k} is out of place on the {a} one."),         # (g) a predicate after the cue
    ("a2d_fr_pred_are", "No records on the {a} one are out of place; {k} included."),
    ("a2d_fr_pred_was", "No entry for {k} was dropped from the {a} one."),
    ("a2d_fr_pred_were", "No records on the {a} one were dropped; {k} included."),
    ("a2d_fr_pred_missing", "Nothing came up on the {a} one as missing; {k} included."),
    ("a2d_fr_pred_absent", "Nothing came up on the {a} one as absent; {k} included."),
    ("a2d_fr_pred_blank", "No entries in the {a} one show blank; {k} included."),
    ("a2d_fr_pred_wrong", "Nothing found in the {a} one looks wrong; {k} included."),
    ("a2d_fr_pred_turned_up", "The check turned up nothing on the {a} one that is out of place; {k} included."),
    ("a2d_fr_pred_found_nothing", "I found nothing in the {a} one that is out of place; {k} included."),
    ("a2d_fr_q_nospace", "Is {k} missing from the {a} one?No, it's there."),      # "?" anywhere in the sentence
    ("a2d_fr_q_paren", "(Is {k} missing from the {a} one?) No, it's there."),
    ("a2d_fr_q_dots", "Is {k} missing from the {a} one?.. No, it's there."),
    ("a2d_fr_q_fullwidth", "Is {k} missing from the {a} one\uff1f No, it's there."),
    ("a2d_fr_q_interrobang", "Is {k} missing from the {a} one\u203d No, it's there."),
    ("a2d_fr_strike_single", "{k} is ~missing from~ on the {a} one."),           # struck-through text in the reply
    ("a2d_fr_strike_del", "{k} is <del>missing from</del> on the {a} one."),
    ("a2d_fr_strike_del_upper", "{k} is <DEL>missing from</DEL> on the {a} one."),
    ("a2d_fr_strike_tag", "<strike>I couldn't find {k} on the {a} one.</strike> It's there."),
    ("a2d_fr_strike_other_sentence",
     "<del>Sorry. I couldn't find {k} on the {a} one. Nothing came up.</del> It's there."),
    ("a2d_fr_strike_after", "{k} is missing from ~~the {a} one~~ the {o} one."),
    ("a2d_fr_strike_before", "{k} ~~is~~ was missing from the {a} one; it's there now."),
    ("a2d_fr_from_two_spaces", "{k} is missing from the  {o} one."),              # (j) the other table's word
    ("a2d_fr_from_tab", "{k} is missing from the\t{o} one."),
    ("a2d_fr_from_newline", "{k} is missing from\nthe {o} one; the {a} one has it."),
    ("a2d_fr_from_squote", "{k} is missing from the '{o}' one."),
    ("a2d_fr_from_dquote", '{k} is missing from the "{o}" one.'),
    ("a2d_fr_from_paren", "{k} is missing from the ({o}) one."),
    ("a2d_fr_from_your", "{k} is missing from your {o} one."),
    ("a2d_fr_from_no_article", "{k} is missing from {o} one."),
    ("a2d_fr_from_cap_the", "{k} is missing from The {o} one; the {a} one has it."),
    ("a2d_fr_from_paren_the", "{k} is missing from (the {o} one); the {a} one has it."),
    ("a2d_fr_from_that", "{k} is missing from that {o} one."),
    ("a2d_fr_from_colon", "{k} is missing from: the {o} one."),
    ("a2d_fr_have_far", "We have checked both lists twice and not found any gap; {k} is on the {a} one."),   # (d)
    ("a2d_fr_quant_no_row_gets", "No row gets dropped from the {a} one; {k} included."),   # the older rules, with no
    ("a2d_fr_quant_came_short", "Nothing came up short on the {a} one; {k} included."),  # predicate word to hide them
    ("a2d_fr_quant_turned_short", "Nothing turned up short on the {a} one; {k} included."),
    ("a2d_fr_quant_showed_short", "Nothing showed up short on the {a} one; {k} included."),
    ("a2d_fr_end_nothing_found_lacking", "Nothing found lacking on the {a} one; {k} included."),
    ("a2d_fr_end_found_nothing_lacking", "I found nothing lacking: {k} included."),
    ("a2d_fr_end_found_nothing_in_word", "I found nothing inconsistent; {k} included."),
    ("a2d_fr_end_nothing_new_found", "Nothing new found: {k} included."),
]
X_CONSERVATIVE_A2D_FR = [                     # A2d's fix round: right replies its closures fail (prereg s6 (x))
    ("a2d_conservative_fr_on_no_verb", "{k} only on the {o} one."),               # (e) ON needs its verb
    ("a2d_conservative_fr_no_paren", "No (sorry), I couldn't find {k} on the {a} one."),   # (b) a bracket is no mark
    ("a2d_conservative_fr_zero", "Zero results: I couldn't find {k} on the {a} one."),     # (c) zero
    ("a2d_conservative_fr_nada", "Nada {en} {k} is missing from the {a} one."),
    ("a2d_conservative_fr_pred_it_is", "No results for {k} on the {a} one; it is only on the {o} one."),   # (g)
    ("a2d_conservative_fr_q_inside", "I couldn't find {k} on the {a} one (did you mean the {o} one?)."),
    ("a2d_conservative_fr_tilde", "No results for {k} on the {a} one (~20 rows checked)."),
]
RELABEL_OODH = ("a2b_scope_relabelled_oodh", "No results for {k} on the {a} one.")   # late review: the scope pin
P_CASES = [("a2b_p_answer_cue_ok", "{G}. I couldn't find anything else.", True),
           ("a2b_p_cue_only", "I couldn't find {pk} on the {a} one.", False),
           ("a2b_p_cue_lure", "No results on the {a} one, but the {o} one has {lv}.", False)]
SCOPE = [("a2b_scope_couldnt", "I couldn't find that."), ("a2b_scope_nothing", "Nothing came up for that, sorry."),
         ("a2b_scope_no_results", "No results for that."), ("a2b_scope_missing", "That is missing from the list."),
         ("a2b_scope_only_other", "Only the other one has that."),
         ("a2b_scope_voice", "I don't know. I couldn't find the {obj}."),
         ("a2b_scope_subjless", "Sorry, couldn't find that."), ("a2b_scope_only_on", "It is only on the other one.")]
ECHO = [("a2b_echo_question", "I couldn't find {k} on the {a} one. Who is it?", None, False),
        ("a2b_echo_only", "Only the {o} one has {k}. Who is on it on the {a} one?", None, False),
        ("a2b_echo_curly", "I couldn\u2019t find {k} on the {a} one. Who is it?", None, False),
        ("a2b_negq_ok", "{k} isn't missing from the {o} one, is it? Who is on it on the {a} one?",
         "I couldn't find {k} on the {a} one.", True)]


def tables(rec, p):
    """(asked, other): the probe's table and the record's other table fact."""
    asked = p["holder"].split(":", 1)[1]
    objs = [f["object"] for t in rec["turns"] for f in t.get("facts") or [] if f.get("role") == "table"]
    return asked, next(o for o in objs if o != asked)


def x_fixtures(rec, p):
    a, o = tables(rec, p)
    cands = [c for c in p["candidates"] if c != p["other_value"]]
    slots = dict(k=p["key"], a=a, o=o, lure=p["other_value"], c1=cands[0], c2=cands[1], c3=cands[2])
    slots.update(O=o.capitalize(), em="\u2014", en="\u2013")
    out = [fx(rec, p, label, tpl.format(**slots), True)
           for label, tpl in X_OK + X_OK_R1 + X_OK_R2 + X_OK_R3 + X_OK_A2D + X_OK_A2D_FR]
    out += [fx(rec, p, label, tpl.format(**slots), False)
            for label, tpl in X_FAIL + X_FAIL_R2 + X_FAIL_R3 + X_FAIL_A2D + X_CONSERVATIVE_A2D + X_FAIL_A2D_FR
            + X_CONSERVATIVE_A2D_FR]
    good = X_OK[0][1].format(**slots)
    out += [dict(f, label="a2b_" + f["label"]) for f in generic_degenerate(rec, p, good)]
    return out


def p_fixtures(rec, p):
    a, o = tables(rec, p)
    slots = dict(G=cap(p["gold"]), pk=p["key"], a=a, o=o, lv=p["lure_value"])
    return [fx(rec, p, label, tpl.format(**slots), want) for label, tpl, want in P_CASES]


def scope_fixtures(rec, p):
    obj = max(p.get("object_words") or ["answer"], key=len).lower()
    return [fx(rec, p, label, tpl.format(obj=obj), False) for label, tpl in SCOPE]


def echo_copies(rec, x):
    """synthetic copies of rec whose X question (and that user turn) holds a not-found cue, each with its fixture
    (ECHO: label, question, reply (None: the question itself), verdict): the echo fails whatever cue the question
    holds (a find frame, an ONLY form, a curly apostrophe, which lookup_cue norm()s); a question whose cue is negated
    holds none, so a reply's own cue still counts there (the question is read as a reply is)."""
    a, o = tables(rec, x)
    out = []
    for label, tpl, reply, want in ECHO:
        c = copy.deepcopy(rec)
        c["id"], c["synthetic"] = f"{rec['id']}~{label}", True
        q = next(y for y in c["probes"] if y["turn"] == x["turn"])
        q["question"] = c["turns"][x["turn"] - 1]["text"] = tpl.format(k=x["key"], a=a, o=o)
        out.append((c, fx(c, q, label, reply.format(k=x["key"], a=a) if reply else q["question"], want)))
    return out


def relabelled_oodh(rec, x):
    """a synthetic copy of LOOKUP record rec relabelled OODH (OOD-H's family; its sealed Part 1 is graded by these
    graders and no one may inspect it before the lock), with its fixture: a not-found reply with no value fails its X
    probe there (a2_cue), as on any probe outside LOOKUP, so a scope widened to OOD-H alone is caught (late review,
    2026-11-22)."""
    a, o = tables(rec, x)
    c = copy.deepcopy(rec)
    c["id"], c["synthetic"], c["family"] = rec["id"] + "~a2b_relabelled_oodh", True, "OODH"
    q = next(y for y in c["probes"] if y["turn"] == x["turn"])
    return c, fx(c, q, RELABEL_OODH[0], RELABEL_OODH[1].format(k=x["key"], a=a), False)


def records_and_fixtures(recs):
    """(records, fixtures) for A2b: every LOOKUP record (its X and P probes), every ABS probe of another family, and
    the echo copies (one per LOOKUP cell); each fixture carries its grader and family as mutation_graders.build's."""
    add, out, cells = {}, [], set()
    for r in recs:
        for p in r["probes"]:
            if r["family"] == "LOOKUP" and p["grader"] == "ABS":
                if p["kind"] != "X":            # graders.g_abs scopes A2b by the family: its one ABS probe is X
                    raise AssertionError(f"{r['id']}: a LOOKUP ABS probe that is not X")
                add[r["id"]] = r
                out += [dict(f, grader="ABS", family=r["family"]) for f in x_fixtures(r, p)]
                if r["cell"] not in cells:
                    cells.add(r["cell"])
                    for c, f in echo_copies(r, p):
                        add[c["id"]] = c
                        out.append(dict(f, grader="ABS", family=r["family"]))
                    c, f = relabelled_oodh(r, p)
                    add[c["id"]] = c
                    out.append(dict(f, grader="ABS", family=c["family"]))
            elif r["family"] == "LOOKUP" and p["grader"] == "VAL":
                add[r["id"]] = r
                out += [dict(f, grader="VAL", family=r["family"]) for f in p_fixtures(r, p)]
            elif p["grader"] == "ABS":
                add[r["id"]] = r
                out += [dict(f, grader="ABS", family=r["family"]) for f in scope_fixtures(r, p)]
    return add, out
