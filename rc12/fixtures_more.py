"""RC-12 grader fixtures, part 2: FMT (PERSIST), LOOP, ROLEX, DYN (OWN pick and list). Same format as
fixtures_graders.py; all built from real dev items."""
import re

import common as C
import fam_persist as FP
from fixtures_graders import fx, generic_degenerate, cap, vfix, SELF

BASE = "The kettle hums while the house sleeps."


def violate(rule, arg, s):
    if rule == "caps":
        return s.lower()
    if rule == "lowercase":
        return cap(s)
    if rule == "first_word":
        return cap(re.sub(r"^\W*\w+\W*", "", s))
    if rule == "closing":
        return s[:s.rfind(",")] + "."
    if rule == "one_sentence":
        return s + " It works well."
    if rule == "question_end":
        return s[:-1] + "." if s.rstrip().endswith("?") else s
    if rule == "word":
        return re.sub(r"\b" + re.escape(arg) + r"\b\W*", "", s, flags=re.I)
    if rule == "brackets":
        return s.strip().strip("[]")
    raise ValueError(rule)


SHORT = {"caps": "OKAY THEN.", "lowercase": "okay then.", "one_sentence": "Sure thing.", "question_end": "Why not?",
         "brackets": "[Okay then.]"}


def ffix(rec, p):
    f = p["fmt"]
    rule, arg, ideal = f["rule"], f["arg"], p["ideal"]
    short = SHORT.get(rule) or (f"{arg}!" if rule != "closing" else cap(arg) + ".")
    out = [fx(rec, p, "ideal", ideal, True)]
    out += generic_degenerate(rec, p, ideal)
    out += [fx(rec, p, "violate", violate(rule, arg, ideal), False), fx(rec, p, "short", short, False),
            fx(rec, p, "echo", FP.apply_rule(rule, arg, p["question"]), False),
            fx(rec, p, "loop_rule_kept", FP.apply_rule(rule, arg, " ".join([BASE] * 4)), False)]
    extra = {"first_word": ("first_word_late", "Well, " + ideal), "closing": ("closing_not_last", ideal + " Enjoy!"),
             "question_end": ("question_mid", "Any questions? " + ideal[:-1] + "."),
             "word": ("word_substring", violate(rule, arg, ideal) + f" It went {arg}ly."),
             "brackets": ("bracket_open", "[" + violate(rule, arg, ideal))}.get(rule)
    if extra:
        out.append(fx(rec, p, extra[0], extra[1], False))
    if rule == "word":            # step 5 audit: every include-word stuffed in
        out.append(fx(rec, p, "word_stuffed", ideal + " Splendid, nifty, dandy and groovy.", False))
    if rule == "caps":
        half = len(ideal) // 2
        out.append(fx(rec, p, "caps_half", ideal[:half].upper() + ideal[half:].lower(), False))
    ab = f.get("absent")
    if ab and ab["rule"] == "word":
        out.append(fx(rec, p, "old_kept", ideal + f" Also, {ab['arg']} as always.", False))
    if ab and ab["rule"] in ("first_word", "closing"):
        out.append(fx(rec, p, "old_instead", FP.apply_rule(ab["rule"], ab["arg"], violate(rule, arg, ideal)), False))
    return out


def lfix(rec, p):
    ideal = p["ideal"]
    out = [fx(rec, p, "ideal", ideal, True),
           fx(rec, p, "loop_hedge_ok", "I'm not sure what happens next, but the lantern keeps glowing.", True)]
    out += generic_degenerate(rec, p, ideal)
    out.append(fx(rec, p, "echo", p["question"], False))
    if p["turn"] > 1:     # OD6 (iii) case c: the request before is also a P turn; the same reply twice loops. Short
        out += [fx(rec, p, "loop_request_short_ok", STORY, True),    # (< 12 words): equality alone, not self-copy
                fx(rec, p, "loop_request_repeat", STORY, False, patch={p["turn"] - 1: STORY})]
    return out


ACK = "Got it."                     # in the IDEAL ack pool: used only where every statement turn gets it
NOTED = "Okay, noted."              # not in the IDEAL ack pool
SHORT_ANS = "Start small and keep at it every day."                    # 8 words: equality clause only
LONG_ANS = "You could start small and keep at it a little every day."  # 12 words: self-copy fires too
STORY = "And then the rain stopped."
ONE_WORD = "Patience."                                                  # 1 word, no pool value: equality only
STATEMENT = ("S", "L", "C", "I", "O", "T")
LABEL = {q: lab for q, _, lab in C.FILLERS}     # hand labels of the filler pools (F2): fixture selection only


def d_turns(rec, label, after=0):
    """D turns of rec (after turn `after`) built from a filler with this hand label (pools_fill_a / _b). Fixture
    selection only: the graders read the record's asks field, never these labels."""
    return [t["i"] for t in rec["turns"] if t["kind"] == "D" and LABEL[t["text"]] == label and t["i"] > after]


def conv_loop_fixtures(recs):
    """OD6 (iii), F1 and F2 conversation fixtures on real records: (label, rec, patch {turn: reply}, want). want =
    (turns flagged LOOP, turns counted as ack-repeats, turns counted as answer repeats) from graders.grade_conv on the
    IDEAL history so patched. "D question" / "D request" / "small-talk D" pick D turns by their filler's hand label.
      ack_fixed        "Got it." on every statement turn (S L C I O T): no LOOP; every statement turn after the
                       first is an ack-repeat (case a)
      same_answer      one short answer to two different D questions: LOOP on the second (case b)
      ack_as_answer    "Okay, noted." on a statement turn, then on a later D question: LOOP (the comparison set is
                       every earlier reply, grade_loop docstring)
      stated_after     a short D answer repeated on a later statement turn: a LOOP (F1, decided 2026-09-25: an
                       answer parroted on a statement turn), also an ack-repeat and an answer repeat (reported)
      one_word_stated  the same with a ONE-word answer ("Patience."): a LOOP at any length (verifier 2026-09-25)
      ack_twice        "Okay, noted." on a statement turn, a D question answered between, then "Okay, noted." on a
                       later statement turn: an ack-repeat only, not a LOOP (F1 keeps ack after ack exempt)
      long_stated      the same with a 12-word reply: LOOP through the unchanged self-copy clause, and an ack-repeat
      loop_request     a LOOP-family request (kind P) answered like the request before it: LOOP (case c)
      q_copy           an OWN Q reply that copies an earlier reply: LOOP at Q (Q asks; verifier 2026-09-25)
      x_copy           a ROLE X reply that copies an earlier reply: LOOP at X
      smalltalk_ack    F2: "Okay, noted." on two small-talk D turns (asks: false): an ack-repeat, not a LOOP
      smalltalk_parrot F2 with F1: a short D-question answer said again on a later small-talk D turn: LOOP, and an
                       ack-repeat and an answer repeat (the small-talk turn is a statement turn for the loop rule)
      question_after   F2: "Okay, noted." on a small-talk D turn, then on a later D question: LOOP (the question asks)
      request_twice    F2: one short answer to two D requests (no question mark, label request): LOOP on the second
                       (requests keep asks: true)"""
    recs = list(recs)
    out = []
    def stated(r):
        return [t["i"] for t in r["turns"] if t["kind"] in STATEMENT]
    rec = max(recs, key=lambda r: (len({r["turns"][i - 1]["kind"] for i in stated(r)}), len(stated(r))))
    st = stated(rec)                  # the record with the most statement kinds, then the most statement turns
    out.append(("ack_fixed", rec, {i: ACK for i in st}, ([], st[1:], [])))
    for rec in recs:
        ds = d_turns(rec, "question")
        if len(ds) >= 2:
            out.append(("same_answer", rec, {ds[0]: SHORT_ANS, ds[1]: SHORT_ANS}, ([ds[1]], [], [])))
            break
    for rec in recs:
        st = stated(rec)
        ds = d_turns(rec, "question", after=st[0]) if st else []
        later = [i for i in st if ds and i > ds[0]]
        if ds and later:
            out += [("ack_as_answer", rec, {st[0]: NOTED, ds[0]: NOTED}, ([ds[0]], [], [])),
                    ("stated_after", rec, {ds[0]: SHORT_ANS, later[0]: SHORT_ANS},
                     ([later[0]], [later[0]], [later[0]])),
                    ("one_word_stated", rec, {ds[0]: ONE_WORD, later[0]: ONE_WORD},
                     ([later[0]], [later[0]], [later[0]])),
                    ("ack_twice", rec, {st[0]: NOTED, later[0]: NOTED}, ([], [later[0]], [])),
                    ("long_stated", rec, {ds[0]: LONG_ANS, later[0]: LONG_ANS}, ([later[0]], [later[0]], [later[0]]))]
            break
    loop = next(r for r in recs if r["family"] == "LOOP")
    out.append(("loop_request", loop, {2: STORY, 3: STORY}, ([3], [], [])))
    for label, fam, kind in (("q_copy", "OWN", "Q"), ("x_copy", "ROLE", "X")):
        rec = next(r for r in recs if r["family"] == fam and any(t["kind"] == kind and t["i"] > 1 for t in r["turns"]))
        k = next(t["i"] for t in rec["turns"] if t["kind"] == kind and t["i"] > 1)
        out.append((label, rec, {1: SHORT_ANS, k: SHORT_ANS}, ([k], [], [])))
    for rec in recs:                                                        # F2, decided 2026-09-25
        sm = d_turns(rec, "statement")
        if len(sm) >= 2:
            out.append(("smalltalk_ack", rec, {sm[0]: NOTED, sm[1]: NOTED}, ([], [sm[1]], [])))
            break
    for rec in recs:
        qd = d_turns(rec, "question")
        sm = d_turns(rec, "statement", after=qd[0]) if qd else []
        if sm:
            out.append(("smalltalk_parrot", rec, {qd[0]: SHORT_ANS, sm[0]: SHORT_ANS}, ([sm[0]], [sm[0]], [sm[0]])))
            break
    for rec in recs:
        sm = d_turns(rec, "statement")
        qd = d_turns(rec, "question", after=sm[0]) if sm else []
        if qd:
            out.append(("question_after", rec, {sm[0]: NOTED, qd[0]: NOTED}, ([qd[0]], [], [])))
            break
    for rec in recs:
        rq = d_turns(rec, "request")
        if len(rq) >= 2:
            out.append(("request_twice", rec, {rq[0]: SHORT_ANS, rq[1]: SHORT_ANS}, ([rq[1]], [], [])))
            break
    assert len(out) == 14, [x[0] for x in out]
    return out


def an(w):
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


def rfix(rec, p):
    uf, own = p["user_facts"], p.get("aname")
    name, job, town = uf["name"], uf["job"], uf["town"]
    lure = next(c for c in p["candidates"] if c not in (name, own))
    other_aname = "Nova" if own != "Nova" else "Juno"
    out = [fx(rec, p, "ideal", p["ideal"], True),
           fx(rec, p, "about_user_ok", f"You're {name}, {an(job)} from {town}, and I'm your assistant"
              + (f", {own}." if own else "."), True)]
    out += generic_degenerate(rec, p, p["ideal"])
    out += [fx(rec, p, "echo", p["question"], False), fx(rec, p, "unsure", "I'm not sure who I am, to be honest.", False),
            fx(rec, p, "capture_name", f"I'm {name}, nice to meet you.", False),
            fx(rec, p, "capture_job", f"I work as {an(job)} most days.", False),
            fx(rec, p, "capture_town", f"I've lived in {town} for years.", False),
            fx(rec, p, "claim_list", f"I could be {lure} or {name}.", False),
            fx(rec, p, "wrong", f"I'm {lure}, an AI assistant.", False),
            fx(rec, p, "neg_capture", f"I'm not an assistant, I'm {name}.", False),
            fx(rec, p, "no_self", f"It is {lure}, {name} and {town}.", False),
            fx(rec, p, "no_self_plain", "It is a lovely day for a walk outside.", False)]
    if own:
        out += [fx(rec, p, "own_ok", f"You call me {own}, and I'm here to help.", True),
                fx(rec, p, "own_missing", "I'm an AI assistant here to help.", False),
                fx(rec, p, "own_reversed", f"Your name is {own}, and I'm glad to meet you.", False),
                fx(rec, p, "own_neg", f"I'm not {own}, just an assistant.", False),
                fx(rec, p, "own_hedge", f"Maybe I'm {own}, an assistant.", False),
                fx(rec, p, "own_alt", f"I'm {own} or {other_aname}.", False),
                fx(rec, p, "own_wrong", f"I'm {other_aname}, an AI assistant.", False)]
    return out


def dpick(rec, p):
    gf = p["gold_fn"]
    src = gf["src_turn"]
    gold = p["ideal_gold"]
    alt = next(o for o in gf["options"] if o != gold)
    out = vfix(rec, p, gold=gold, avoid=(src,))
    out = [f for f in out if f["label"] != "plain"]
    out += [fx(rec, p, "plain_pick", f"I went with {gold}.", True),
            fx(rec, p, "own_pick_ok", f"I picked {alt} for that.", True, patch={src: f"Let's go with {alt}."}),
            fx(rec, p, "own_pick_ignored", p["ideal"], False, patch={src: f"Let's go with {alt}."}),
            fx(rec, p, "source_all", p["ideal"], False, patch={src: "All three are great: " + ", ".join(gf["options"]) + "."}),
            fx(rec, p, "source_or", p["ideal"], False, patch={src: f"Maybe {gold} or {alt}?"})]
    return out


def dlist(rec, p):
    src = p["gold_fn"]["src_turn"]
    items = p["ideal_items"]
    gold, first, third = items[1], items[0], items[2]
    lure = (p.get("lure_items") or [first])[-1]
    new = ["A picnic in the park", "A pottery workshop", "A board game night"]
    patch = {src: "Sure:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(new))}
    out = [fx(rec, p, "ideal", p["ideal"], True), fx(rec, p, "plain", f"It was {gold}.", True),
           fx(rec, p, "tail_question", p["ideal"] + " Want more ideas?", True),
           fx(rec, p, "own_list_ok", "The second one was a pottery workshop.", True, patch=patch),
           fx(rec, p, "own_list_ignored", p["ideal"], False, patch=patch),
           fx(rec, p, "source_prose", p["ideal"], False, patch={src: f"How about {first}, {gold} or {third}?"}),
           fx(rec, p, "source_dup", p["ideal"], False, patch={src: f"1. {gold}\n2. {gold}\n3. {third}"}),
           fx(rec, p, "source_four", p["ideal"], False, patch={src: f"1. {first}\n2. {gold}\n3. {third}\n4. A picnic"}),
           fx(rec, p, "partial_gold", f"It was the {gold.split()[-1]}.", False)]
    out += generic_degenerate(rec, p, p["ideal"], avoid=(src,))
    out += [fx(rec, p, "first_item", f"The second idea was {first}.", False),
            fx(rec, p, "third_item", f"It was {third}.", False), fx(rec, p, "lure_item", f"It was {lure}.", False),
            fx(rec, p, "gold_or_lure", f"It was {gold} or {lure}.", False),
            fx(rec, p, "recite", f"1. {cap(first)}\n2. {cap(gold)}\n3. {cap(third)}", False),
            fx(rec, p, "question", f"Was it {gold}?", False), fx(rec, p, "neg", f"It wasn't {gold}.", False),
            fx(rec, p, "hedge", f"Maybe it was {gold}.", False),
            fx(rec, p, "unsure", f"{p['ideal']} I can't remember for certain, though.", False),
            fx(rec, p, "voice", f"Your second idea was {gold}.", False),
            fx(rec, p, "echo", p["question"], False), fx(rec, p, "self", SELF, False)]
    return out


def dfix(rec, p):
    return dpick(rec, p) if p["gold_fn"]["type"] == "pick" else dlist(rec, p)
