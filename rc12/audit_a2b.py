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
]
SCOPE = [
    ("a2b_scope_couldnt", "Sorry, I couldn't find that.", False),
    ("a2b_scope_nothing", "Nothing came up for that.", False),
    ("a2b_scope_no_results", "No results, sorry.", False),
    ("a2b_scope_not_found", "Not found.", False),
    ("a2b_scope_voice", "I'm not sure. I cannot locate the {w}.", False),
    ("a2b_scope_r1_nosubj", "Couldn't find it, sorry.", False),
    ("a2b_scope_r1_only_in", "It's only in the other one.", False),
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
