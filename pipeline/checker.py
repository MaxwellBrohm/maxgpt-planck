"""Render checker (SPEC section 6). run(skel, raw, built) parses the teacher text, runs EVERY check, logs every
failing code, and names the first code in ORDER as the primary reason. Codes marked report-only (VOCAB_OOL while
the word list is FAKE) are logged but do not reject.

Codes added beyond the SPEC list (see notes.txt): STAGE_DIR (bracketed stage directions, SPEC 5b), PLANT_UNBOUND
(a guided plant line states the value without the noun it belongs to: "it's in Brno" for the aunt's city), CORR_MISSING
(a correction, fix, twin or list-op value absent from its turn), VALUE_EARLY (a scheduled value said before the
turn that schedules it), USER_VOICE / ASSIST_VOICE (third-person narration: "the user says ..."), OPEN_END (an
"end with an open question" turn without a question mark), SOCIAL_TOPIC (an S8 social reply that brings in a slot
value), SKEL_INFEASIBLE (render_prompt.feasible, before any teacher call)."""
import check_behav
import check_events
import check_text
import parse
import render_prompt
from check_base import Ctx

ORDER = ["SKEL_INFEASIBLE", "FORMAT_LINES", "FORMAT_EXTRA", "FORMAT_WRAP", "THOUGHT_TAG", "EMPTY_TURN", "ROLE_LABEL",
         "MARKDOWN", "EMOJI", "DASH", "STAGE_DIR", "LEN_USER", "LEN_ASSIST", "DIGIT", "EXACT_MISMATCH", "REQ_SPAN",
         "FORBID_SPAN", "REQ_WORD", "VOCAB_OOL", "PLANT_MISSING", "PLANT_UNBOUND", "CORR_MISSING", "VALUE_EARLY", "DIST_LEAK",
         "QUERY_RESTATES", "QUERY_NOREF", "ANSWER_WRONG", "ANSWER_STALE", "ANSWER_SHOTGUN", "PERSPECTIVE",
         "SELF_CLAIM", "USER_VOICE", "ASSIST_VOICE", "PERSIST_FAIL", "OPEN_END", "OFFTOPIC", "ABSTAIN_MISSING",
         "DEFLECT", "AI_ISM", "TOPIC_RETURN", "IDENTITY", "SOCIAL_TOPIC", "LIST_STATE", "LOOKUP_FORMAT",
         "LOOKUP_UNNEEDED", "LOOKUP_MISSING", "LOOKUP_COPY", "LOOKUP_EMPTY", "REPEAT_4GRAM", "CONSEC_REP",
         "SELF_COPY", "ECHO_USER", "NON_ENGLISH", "SAFETY", "PERSONA_LEAK", "PROMPT_ECHO", "HELDOUT_VOCAB",
         "HELDOUT_ECHO", "HELDOUT_STRUCT"]
REPORT_ONLY = {"VOCAB_OOL"}
RANK = {c: n for n, c in enumerate(ORDER)}


def registry():
    """every check function, in module order; mutation_checker replaces entries to build checker mutants."""
    return list(check_text.CHECKS) + list(check_events.CHECKS) + list(check_behav.CHECKS)


REGISTRY = registry()


def check_turns(skel, turn_texts, built=None, wordlist=None, parse_codes=(), drift=()):
    ctx = Ctx(skel, turn_texts, built=built, wordlist=wordlist)
    hits = [(c, None, d) for c, d in parse_codes]
    for fn in REGISTRY:
        hits += fn(ctx)
    unknown = {c for c, _, _ in hits} - set(RANK)
    if unknown:
        raise ValueError(f"codes not in ORDER: {unknown}")
    codes = sorted({c for c, _, _ in hits}, key=RANK.get)
    failing = [c for c in codes if c not in REPORT_ONLY]
    return {"ok": not failing, "primary": failing[0] if failing else None, "codes": codes,
            "hits": sorted(hits, key=lambda h: (RANK[h[0]], -1 if h[1] is None else h[1])), "drift": list(drift)}


def run(skel, raw, built=None, wordlist=None):
    """teacher text -> check result. built = render_prompt.build(skel) (needed for PROMPT_ECHO and PERSONA_LEAK)."""
    built = built or render_prompt.build(skel)
    pre = render_prompt.feasible(skel)
    p = parse.parse(raw, parse.plan(skel))
    res = check_turns(skel, p["turns"], built, wordlist, parse_codes=pre + p["codes"], drift=p["drift"])
    res["turns"] = p["turns"]
    return res


def record_turns(skel, res):
    """accepted conversation turns for the record: the stored text of an exact line is the bank line itself."""
    out = []
    for t in skel["turns"]:
        text = t["text"] if t["mode"] == "exact" else res["turns"][t["i"]]
        out.append({"role": t["role"], "text": text, "mask": t["mask"]})
    return out
