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
    ("a2b_only_contains_ok", "Only the {o} one contains {k}."),
    ("a2b_only_shows_ok", "Only the {o} list shows {k}."),
    ("a2b_only_has_it_ok", "Only the {o} one has it."),
    ("a2b_only_has_that_ok", "Only the {o} one has that."),
    ("a2b_only_a_row_ok", "Only the {o} one has a row for {k}."),
    ("a2b_only_an_entry_ok", "Only the {o} one has an entry for {k}."),
    ("a2b_only_a_line_ok", "Only the {o} one has a line for {k}."),
    ("a2b_only_the_key_ok", "Only the {o} one has the {k} slot."),
    ("a2b_only_cap_ok", "Only the {O} one has {k}."),                                # the table word in any case
    ("a2b_only_on_cap_ok", "{k} is only on the {O} one."),
    ("a2b_afraid_not_comma_ok", "I'm afraid not, I couldn't find {k} on the {a} one."),  # a negation in the clause
    ("a2b_dont_think_semi_ok", "I don't think so; nothing came up for {k}."),            # before, not in the cue's
    ("a2b_sadly_not_colon_ok", "Sadly not: no results for {k}."),
    ("a2b_two_hits_ok", "{k} isn't missing from the {o} one; it's missing from the {a} one."),   # any clean hit
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
    ("a2b_r2_nobody_comma_ok", "Nobody, {k} is missing from the {a} one."),           # an answer word set off
    ("a2b_r2_nobody_em_ok", "Nobody {em} {k} is missing from the {a} one."),
    ("a2b_r2_nothing_en_ok", "Nothing {en} {k} is missing from the {a} one."),
    ("a2b_r2_nobody_hyphen_ok", "Nobody - {k} is missing from the {a} list."),
    ("a2b_r2_nothing_window_ok", "Nothing on the {a} one covers {k} because it is missing from it."),
    ("a2b_r2_phrase_end_ok", "Nothing else is wrong, but {k} is missing from the {a} one."),
    ("a2b_r2_on_it_ok", "It is only on the {o} one."),                                 # LOOKUP_ON's anchors
    ("a2b_r2_on_that_ok", "That day is only on the {o} one."),
    ("a2b_r2_on_key_s_ok", "{k}'s only on the {o} one."),
    ("a2b_r2_on_three_words_ok", "{k} is still listed only on the {o} one."),
    ("a2b_r2_only_it_end_ok", "Only the {o} one has it"),                              # ONLY objects
    ("a2b_r2_only_entry_on_ok", "Only the {o} one has an entry on {k}."),
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
    ("a2b_r3_none_comma_ok", "None, {k} is missing from the {a} one."),               # an answer word set off (INTERJ)
    ("a2b_r3_no_dash_ok", "No - {k} is missing from the {a} one."),
    ("a2b_r3_none_em_ok", "None {em} {k} is missing from the {a} one."),
    ("a2b_r3_no_sadly_ok", "No, sadly, I couldn't find {k} on the {a} one."),         # an aside keeps its comma
    ("a2b_r3_interj_comma_colon_ok", "Sorry, afraid not: no results for {k}."),       # an opening after a comma
    ("a2b_r3_tail_word_ok", "{k} isn't there, sadly, it's missing from the {a} one."),  # only verbs after an aside
    ("a2b_r3_not_others_is_ok", "{k}, not others, is missing from the {a} one."),
    ("a2b_r3_not_others_are_ok", "{k}s, not others, are missing from the {a} one."),
    ("a2b_r3_not_others_were_ok", "{k}s, not others, were missing from the {a} one."),
    ("a2b_r3_nobody_quote_ok", '"Nobody" - {k} is missing from the {a} one.'),         # NEG_MORE: quotes, marks
    ("a2b_r3_nobody_colon_ok", "Nobody: {k} is missing from the {a} one."),
    ("a2b_r3_nobody_noticed_ok", "Nobody has noticed that {k} is missing from the {a} one."),  # NEG_PHRASE's words
    ("a2b_r3_has_gone_missing_ok", "{k} has gone missing from the {a} one."),         # the perfect: "not found" only
    ("a2b_r3_have_here_ok", "The {k} rows you have here are not found on the {a} one."),
    ("a2b_r3_on_itll_ok", "It'll be only on the {o} one."),                           # LOOKUP_ON's anchor
    ("a2b_r3_on_thatd_ok", "That'd be only on the {o} one."),
    ("a2b_r3_on_plural_ok", "{k}s are only on the {o} one."),
    ("a2b_r3_only_plural_ok", "Only the {o} one has {k}s."),                          # ONLY's key and its end
    ("a2b_r3_only_on_it_ok", "Only the {o} one has {k} on it."),
    ("a2b_r3_only_in_there_ok", "Only the {o} one has {k} in there."),
    ("a2b_r3_only_listed_ok", "Only the {o} one has {k} listed."),
    ("a2b_r3_only_and_ok", "Only the {o} one has {k} and the {a} one does not."),
    ("a2b_r3_only_but_ok", "Only the {o} one has {k} but the {a} one does not."),
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
    out = [fx(rec, p, label, tpl.format(**slots), True) for label, tpl in X_OK + X_OK_R1 + X_OK_R2 + X_OK_R3]
    out += [fx(rec, p, label, tpl.format(**slots), False) for label, tpl in X_FAIL + X_FAIL_R2 + X_FAIL_R3]
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
