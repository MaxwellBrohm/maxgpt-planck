"""RC-12 audit_graders.py paraphrases for lock step A2b (2026-11-19; grade_cues.py's A2b block, rc12/notes.txt STEP
16A2B): other wordings of the not-found replies on every LOOKUP X probe (they must pass) and of their guards (they
must fail), and the same cues on every ABS probe of another family, which must still fail (the cues are LOOKUP's
only). audit_graders adds cases(rec, probe) to every ABS probe and reports them as ABS:<name>.
Slots: {k} the X key, {a} the asked table, {o} the other table, {l} the lure (the other table's value for {k}), {c}
another candidate, {q} the question; {w} the probe's longest object word (other families)."""

LOOKUP = [
    ("a2b_couldnt_find_table", "Sorry, I couldn't find {k} on the {a} table.", True),
    ("a2b_could_not_find_it", "I could not find it on the {a} one.", True),
    ("a2b_cant_find_any", "I can't find anything for {k}.", True),
    ("a2b_cannot_see", "I cannot see {k} anywhere in the {a} one.", True),
    ("a2b_didnt_find", "I didn't find {k} there, sorry.", True),
    ("a2b_did_not_see", "I did not see {k} on it.", True),
    ("a2b_unable_spot", "I was unable to spot {k} on the {a} list.", True),
    ("a2b_unable_locate", "Sorry, I was unable to locate that.", True),
    ("a2b_s9_teacher_1", "Sorry, I could not find anything on that.", True),
    ("a2b_s9_teacher_2", "Nothing came up for that, sorry.", True),
    ("a2b_nothing_came_key", "Nothing came up for {k} on the {a} one.", True),
    ("a2b_nothing_showed", "I'm sorry, nothing showed up.", True),
    ("a2b_turned_up_nothing", "Looking for {k} turned up nothing.", True),
    ("a2b_found_nothing", "Found nothing on that, sorry.", True),
    ("a2b_came_up_empty", "Sorry, {k} came up empty.", True),
    ("a2b_not_found", "{k} was not found on the {a} one.", True),
    ("a2b_no_results_sorry", "Sorry, no results for {k}.", True),
    ("a2b_no_matches", "No matches for {k} on the {a} one.", True),
    ("a2b_no_entries", "There were no entries for {k}.", True),
    ("a2b_no_rows", "No rows for {k} on the {a} list.", True),
    ("a2b_no_records", "No records for {k}.", True),
    ("a2b_no_info", "No information about {k} on the {a} one.", True),
    ("a2b_no_luck", "No luck finding {k} on the {a} one.", True),
    ("a2b_simply_missing", "{k} is simply missing from the {a} one.", True),
    ("a2b_absent", "{k} is absent from that one.", True),
    ("a2b_only_other_not_this", "Only the {o} one has {k}, not this one.", True),
    ("a2b_only_other_list", "Only the other list lists {k}.", True),
    ("a2b_lure_after", "I couldn't find {k}, so it must be {l}.", False),
    ("a2b_lure_before", "{l}? No, nothing came up for {k}.", False),
    ("a2b_other_value", "No results for {k}; maybe {c}.", False),
    ("a2b_found_it", "I found {k} on the {a} one.", False),
    ("a2b_results_exist", "The results for {k} are on the {a} one.", False),
    ("a2b_not_missing", "{k} is not missing from the {a} one.", False),
    ("a2b_never_absent", "{k} was never absent from the {a} one.", False),
    ("a2b_only_asked", "Only the {a} one has {k}.", False),
    ("a2b_they_couldnt", "They couldn't find {k} either.", False),
    ("a2b_voice_bind", "I couldn't find {k}. I'm on {k} myself.", False),
    ("a2b_voice_my", "Nothing came up for {k}, but my {k} is free.", False),
    ("a2b_echo_q", "{q}", False),
    # fix round (2026-11-19): the forms added then, and their guards, in other words than fixtures_a2b's
    ("a2b_r1_sorry_nosubj", "Sorry, couldn't find {k} anywhere on the {a} list.", True),
    ("a2b_r1_nosubj_it", "Couldn't find it on the {a} one, sorry.", True),
    ("a2b_r1_hmm_nosubj", "Hmm. Couldn't see {k} there.", True),
    ("a2b_r1_we_cant", "We can't find {k} on the {a} one.", True),
    ("a2b_r1_search_locate", "The search couldn't locate {k}.", True),
    ("a2b_r1_just_anywhere", "I just couldn't find {k} anywhere.", True),
    ("a2b_r1_still_see", "I still can't see {k} on it.", True),
    ("a2b_r1_seem_see", "I can't seem to see {k} on the {a} list.", True),
    ("a2b_r1_dont_see_any", "Sorry, I don't see any {k} there.", True),
    ("a2b_r1_wasnt_able_spot", "I wasn't able to spot {k} on the {a} one.", True),
    ("a2b_r1_wasnt_found", "Sorry, {k} wasn't found.", True),
    ("a2b_r1_could_not_be_found", "{k} could not be found on the {a} one.", True),
    ("a2b_r1_nothing_was_found", "Sorry, nothing was found for {k}.", True),
    ("a2b_r1_i_found_nothing", "I found nothing on {k}, sorry.", True),
    ("a2b_r1_only_in_other", "{k} is only in the other list.", True),
    ("a2b_r1_only_on_o", "That day is only on the {o} one.", True),
    ("a2b_r1_only_contains_it", "Only the {o} list contains it.", True),
    ("a2b_r1_only_shows_row", "Only the {o} one shows a row for {k}.", True),
    ("a2b_r1_afraid_not", "Afraid not, nothing came up for {k}.", True),
    ("a2b_r1_not_this_time", "Not this time; I couldn't find {k}.", True),
    ("a2b_r1_two_hits", "{k} is not absent from the {o} one, but it is absent from the {a} one.", True),
    ("a2b_r1_nothing_absent", "Nothing is absent from the {a} one.", False),
    ("a2b_r1_nobodys_missing", "Nobody's missing from the {a} one.", False),
    ("a2b_r1_only_more", "Only the {o} one has more entries; {k} is on the {a} one.", False),
    ("a2b_r1_only_in_asked", "{k} is only in the {a} one.", False),
    ("a2b_r1_she_couldnt", "She couldn't find {k} on the {a} one.", False),
    ("a2b_r1_nosubj_lure", "Couldn't find {k}, but it's {l}.", False),
    # late review (2026-11-22, notes STEP 16A2B-R2): its forms and guards, in other words than fixtures_a2b's
    ("a2b_r2_sorry_afraid", "Sorry, afraid not, nothing came up for {k}.", True),
    ("a2b_r2_unfortunately", "Unfortunately not, I can't find {k} on the {a} one.", True),
    ("a2b_r2_probably_not", "Probably not, no results for {k} on the {a} list.", True),
    ("a2b_r2_not_sure", "Not sure, but nothing turned up for {k}.", True),
    ("a2b_r2_guess_not_semi", "I guess not; {k} is absent from the {a} one.", True),
    ("a2b_r2_nobody_dash", "Nobody - {k} is absent from the {a} one.", True),
    ("a2b_r2_nothing_comma", "Nothing, sorry: {k} is missing from the {a} list.", True),
    ("a2b_r2_its_only_in", "It's only in the {o} one, sorry.", True),
    ("a2b_r2_listed_only_on", "{k} is listed only on the {o} list.", True),
    ("a2b_r2_only_has_it_semi", "Only the {o} one has it; this one does not.", True),
    ("a2b_r2_only_line_for", "Only the {o} list has a line for {k}.", True),
    ("a2b_r2_cant_be_found", "Sorry, {k} can't be found on the {a} one.", True),
    ("a2b_r2_isnt_found", "{k} isn't found anywhere on the {a} one.", True),
    ("a2b_r2_found_nothing_on", "I found nothing on the {a} one for {k}.", True),
    ("a2b_r2_nothing_found_period", "Sorry. Nothing was found.", True),
    ("a2b_r2_isnt_really", "{k} isn't, really, missing from the {a} list.", False),
    ("a2b_r2_never_honestly", "{k} was never, honestly, absent from the {a} one.", False),
    ("a2b_r2_nobody_aside", "Nobody, as far as I know, is missing from the {a} one.", False),
    ("a2b_r2_nothing_else", "Nothing else is missing from the {a} one.", False),
    ("a2b_r2_rows_only_in", "The {a} one has {k}, and the extra rows are only in the {o} one.", False),
    ("a2b_r2_only_that_issue", "Only the {o} one has that issue; {k} is on the {a} one.", False),
    ("a2b_r2_only_row_missing", "Only the {o} one has a row missing; the {a} one has {k}.", False),
    ("a2b_r2_havent_found_issue", "We haven't found an issue: {k} is on the {a} one.", False),
    ("a2b_r2_nothing_found_amiss", "Nothing was found amiss; {k} is on the {a} one.", False),
    ("a2b_r2_found_nothing_odd", "Found nothing odd: {k} is covered on the {a} one.", False),
    ("a2b_r2_he_couldnt", "He couldn't find {k} on the {a} one.", False),
    # late review, second round (2026-11-22, notes STEP 16A2B-R3): its forms and guards, in other words again
    ("a2b_r3_none_colon", "None: {k} is absent from the {a} one.", True),
    ("a2b_r3_no_em", "No \u2014 {k} is missing from the {a} list.", True),
    ("a2b_r3_itll_in", "It'll be only in the {o} one, sorry.", True),
    ("a2b_r3_thatd_on", "Sorry, that'd be only on the {o} list.", True),
    ("a2b_r3_plural_in", "{k}s are only in the {o} one.", True),
    ("a2b_r3_only_lists_plural", "Only the {o} list lists {k}s.", True),
    ("a2b_r3_only_key_in_it", "Only the {o} one has {k} in it, sorry.", True),
    ("a2b_r3_only_key_and", "Only the {o} list has {k} and this one doesn't.", True),
    ("a2b_r3_not_others_absent", "{k}, not others, is absent from the {a} one.", True),
    ("a2b_r3_nothing_dash_honestly", "Nothing - honestly - is absent from the {a} one.", False),
    ("a2b_r3_nobody_paren", "Nobody (honestly) is missing from the {a} one.", False),
    ("a2b_r3_isnt_aside_long", "{k} isn't, from what I can see, absent from the {a} one.", False),
    ("a2b_r3_never_aside_been", "{k} has never, as far as I know, been missing from the {a} one.", False),
    ("a2b_r3_not_single_entry", "Not a single entry is missing from the {a} one.", False),
    ("a2b_r3_none_of_the_rows", "None of the rows is missing from the {a} one.", False),
    ("a2b_r3_has_since_not_found", "The team has since not found any gap: {k} is on the {a} one.", False),
    ("a2b_r3_only_key_bold", "Only the {o} one has {k} in bold; the {a} one has it too.", False),
    ("a2b_r3_quoted_nobody", '"Nobody" is missing from the {a} one.', False),
    ("a2b_r3_nothing_aside_found", "Nothing, really, was found missing from the {a} one.", False),
]
SCOPE = [
    ("a2b_scope_couldnt", "Sorry, I couldn't find that.", False),
    ("a2b_scope_nothing", "Nothing came up for that.", False),
    ("a2b_scope_no_results", "No results, sorry.", False),
    ("a2b_scope_not_found", "Not found.", False),
    ("a2b_scope_voice", "I'm not sure. I cannot locate the {w}.", False),
    ("a2b_scope_r1_nosubj", "Couldn't find it, sorry.", False),
    ("a2b_scope_r1_only_in", "It's only in the other one.", False),
    ("a2b_scope_r2_afraid", "I'm afraid not, I couldn't find that.", False),
    ("a2b_scope_r3_itll", "It'll be only on the other one.", False),
]
TABLE = LOOKUP + SCOPE


def cases(rec, p):
    """(name, reply, expected pass) for an ABS probe: LOOKUP's X probe gets LOOKUP, any other ABS probe SCOPE."""
    if p["grader"] != "ABS":
        return []
    if rec["family"] != "LOOKUP":
        w = max(p.get("object_words") or ["answer"], key=len).lower()
        return [(n, t.format(w=w), want) for n, t, want in SCOPE]
    a = p["holder"].split(":", 1)[1]
    o = next(f["object"] for t in rec["turns"] for f in t.get("facts") or []
             if f.get("role") == "table" and f["object"] != a)
    c = next(x for x in p["candidates"] if x != p["other_value"])
    slots = dict(k=p["key"], a=a, o=o, l=p["other_value"], c=c, q=p["question"])
    return [(n, t.format(**slots), want) for n, t, want in LOOKUP]
