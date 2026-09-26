"""RC-12 grader fixtures built from REAL dev items (SPEC L3, G6). A fixture replaces one probe reply (and, when it
needs to, one earlier reply) inside the conversation's IDEAL history and states the verdict the grader must give.
fixture = dict(rid, turn, label, reply, stop, patch {turn: text}, expect True (must pass) / False (must fail)).
VAL/DYN-pick fixtures: vfix; ABS: afix; FMT: ffix; LOOP: lfix; ROLEX: rfix; DYN-list: dlfix (fixtures_more.py).
Content fixtures that do not apply to a grader are listed in NOT_APPLICABLE with the reason."""
import re

import grade_voice as GV
import graders as G
import pools_vals as V

SELF = "I hope the rest of your week goes smoothly and calmly for you and your family."
NOT_APPLICABLE = {
    "FMT": "hedge, negation, guess list, wrong candidate: G-FMT grades format only (SPEC s4); a hedged or negated "
           "reply that obeys the rule complies. Its failing fixtures are rule violations, marker-only replies, "
           "echo, loop, copy, leak, cap.",
    "LOOP": "hedge, negation, guess list, wrong candidate: the LOOP family scores degeneration only (SPEC s3); a "
            "hedged continuation is not degenerate and must PASS (fixture loop_hedge_ok).",
    "ABS": "deflect ('I'm not sure, could you remind me?') is an abstain and PASSES G-ABS by SPEC s4; the unit "
           "rule (all probes) makes DEFLECT score 0 on RECALL-abstain and LOOKUP units.",
}


def fx(rec, p, label, reply, expect, stop="eos", patch=None):
    return dict(rid=rec["id"], turn=p["turn"], label=label, reply=reply, stop=stop, patch=patch or {},
                expect=expect)


def cap(s):
    return s[:1].upper() + s[1:]


def sub_word(text, old, new):
    return V.value_re(old).sub(new, text)


def earlier_turn(rec, p, avoid=()):
    """an earlier turn whose reply may be overwritten (not a source, Q or instruction turn)."""
    for t in rec["turns"]:
        if t["i"] < p["turn"] and t["kind"] in ("D", "L", "S") and t["i"] not in avoid:
            return t["i"]
    return None


def generic_degenerate(rec, p, good, avoid=()):
    """empty, runaway, leak, loops (within, equal to an earlier reply, self-copy) around a reply that would pass."""
    out = [fx(rec, p, "empty", "", False), fx(rec, p, "empty_punct", " ... ", False),
           fx(rec, p, "runaway", good, False, stop="cap"),
           fx(rec, p, "leak_turn", good + "\nUser: Thanks! What else?", False),
           fx(rec, p, "leak_tag", "Assistant: " + good, False),
           fx(rec, p, "leak_marker", good + " <|im_start|>user", False),
           fx(rec, p, "loop_tri", " ".join([good] * 4), False)]
    e = earlier_turn(rec, p, avoid)
    if e:
        out.append(fx(rec, p, "loop_equal", good, False, patch={e: good.lower() + "!"}))
        out.append(fx(rec, p, "loop_selfcopy", good + " " + SELF, False, patch={e: "To recap: " + good + " " + SELF}))
        share = (len(SELF.split()) - 3) / (len(SELF.split()) + len(good.split()) - 3)
        if 0.55 <= share <= 0.85:
            out.append(fx(rec, p, "loop_selfcopy_part", good + " " + SELF, False, patch={e: SELF}))
    return out


def val_parts(rec, p, gold):
    cands = list(p.get("candidates") or [])
    pool = G.pool_values(p)
    ctx = G.in_context(rec, p, pool)
    others = [c for c in dict.fromkeys(cands + ctx) if c != gold]
    outside = [v for v in pool if v not in ctx and v not in cands and v != gold]
    return cands, others, outside


def vfix(rec, p, gold=None, ideal=None, avoid=()):
    gold = gold or p.get("gold") or (p.get("accepted") or [None])[0]
    ideal = ideal or p["ideal"]
    cands, others, outside = val_parts(rec, p, gold)
    Gd = cap(gold)
    o = others[0] if others else None
    obj = " ".join(p.get("object_words") or []) or "answer"
    h = p.get("holder")
    out = [fx(rec, p, "ideal", ideal, True), fx(rec, p, "plain", f"It's {gold}.", True),
           fx(rec, p, "tail_question", ideal + " Anything else I can help with?", True),
           fx(rec, p, "tail_hedge", ideal + " You might want to write that down.", True)]
    out += generic_degenerate(rec, p, ideal, avoid)
    out += [fx(rec, p, "hedge", f"Maybe it's {gold}.", False), fx(rec, p, "hedge_think", f"I think it's {gold}.", False),
            fx(rec, p, "hedge_guess", f"I'd guess {gold}.", False),                       # step 5 audit cues
            fx(rec, p, "hedge_afaik", f"{Gd}, as far as I know.", False), fx(rec, p, "hedge_near", f"{Gd}. Probably.", False),
            fx(rec, p, "unsure_elsewhere", ideal + " I can't remember for certain, though.", False),
            fx(rec, p, "deflect", "I'm not sure, could you remind me?", False),
            fx(rec, p, "neg", f"It's not {gold}.", False), fx(rec, p, "neg_far", f"It is not really {gold}.", False),
            fx(rec, p, "neg_after", f"{Gd} isn't right.", False), fx(rec, p, "neg_nt", f"It isn't {gold}.", False),
            fx(rec, p, "no_interj_ok", f"No problem, it's {gold}.", True),
            fx(rec, p, "cond_closed_ok", f"If you ever forget, it's {gold}.", True),
            fx(rec, p, "voice_mixed_ok", f"From what I've heard, you said it's {gold}.", True),
            fx(rec, p, "question", f"Is it {gold}?", False),
            fx(rec, p, "hypo", f"For example, {gold} would work.", False),
            fx(rec, p, "cond", f"If it were {gold} that would be easy.", False),
            fx(rec, p, "than", f"It's later than {gold}.", False),
            fx(rec, p, "list_label_ok", f"Here is what I have:\n- Answer: {Gd}\n- Source: our chat\n- Status: set", True),
            fx(rec, p, "list_only", f"Here are some thoughts:\n- {Gd}\n- something else\n- another one", False),
            fx(rec, p, "alt_nonpool", f"It's {gold} or Easter.", False),
            fx(rec, p, "neg_list_ok", f"It's {gold}, not Easter or Christmas.", True),
            fx(rec, p, "but_list_ok", f"Not Easter or Christmas, but {gold}.", True),
            fx(rec, p, "alt_slash_nonpool", f"It's {gold}/Easter.", False),
            fx(rec, p, "echo", p["question"], False), fx(rec, p, "echo_plus", p["question"] + " " + Gd + ".", False)]
    qw = p["question"].rstrip("?").split()
    if 8 <= len(qw) <= 22:
        out.append(fx(rec, p, "echo_near", " ".join(qw[1:]) + "? " + Gd + ".", False))
    if h == "assistant":
        rev = f"Your name is {gold}." if p.get("pool") == "name" else f"You picked {gold}."
        out.append(fx(rec, p, "voice_reversed", rev, False))
    else:
        who = f"{h}'s " if h not in G.NON_PERSON and not str(h).startswith("table") else ""
        out += [fx(rec, p, "voice_my", f"My {who}{obj} is {gold}.", False), fx(rec, p, "voice_mine", f"Mine is {gold}.", False),
                ] + ([fx(rec, p, "voice_myobj", f"As for my own {obj}, you said it's {gold}.", False)]
                     if p.get("object_words") else [])
        if who:
            out.append(fx(rec, p, "voice_myholder", f"As for my own {h}, you said it's {gold}.", False))
    mode = GV.mode_of(p)                      # report frames (grade_voice.py, STEP 10 and its fix round)
    if mode == "user":
        out += [fx(rec, p, "voice_you_told_ok", f"You told me it was {gold}.", True),
                fx(rec, p, "voice_you_asked_ok", f"You asked me about the {gold}.", True),
                fx(rec, p, "voice_long_ok", f"You told me that the one in question was {gold}.", True),
                fx(rec, p, "voice_request_ok", f"You wanted me to remember it's {gold}.", True),
                fx(rec, p, "voice_let_me_ok", f"Let me check: it's {gold}.", True),
                fx(rec, p, "voice_shared_ok", f"You shared with me that it's {gold}.", True),
                fx(rec, p, "voice_quote_ok", f'You said, "Mine is {gold}."', True),
                fx(rec, p, "voice_quote_my_ok", f'You told me, "My {obj} is {gold}."', True),
                fx(rec, p, "voice_i_said", f"I said {gold}.", False),
                fx(rec, p, "voice_i_told_you", f"I told you it was {gold}.", False),
                fx(rec, p, "voice_i_mentioned", f"I mentioned {gold}.", False),
                fx(rec, p, "voice_know_i_said", f"You know I said {gold}.", False),
                fx(rec, p, "voice_id_said", f"I'd said it was {gold}.", False), fx(rec, p, "voice_i_did", f"I did {gold}.", False),
                fx(rec, p, "voice_after_i_told", f"It's {gold}, like I told you.", False),
                fx(rec, p, "voice_second_mention", f"You told me it was {gold}. I said {gold} too.", False),
                fx(rec, p, "voice_you_after_i_ok", f"I checked: your answer is {gold}.", True),
                fx(rec, p, "voice_far_mixed_ok", f"I'm happy you asked; the one we talked about earlier on was {gold}.", True)]
    elif mode == "given":
        out += [fx(rec, p, "given_you_told_ok", f"You told me {gold}.", True),
                fx(rec, p, "given_my_name_ok", f"My name is {gold}.", True),
                fx(rec, p, "given_chose_ok", f"You chose the name {gold} for me.", True),
                fx(rec, p, "given_quote_ok", f'You said, "I\'ll call you {gold}."', True),
                fx(rec, p, "given_i_said", f"I said {gold}.", False), fx(rec, p, "given_already", f"I already said {gold}.", False),
                fx(rec, p, "given_told_call_me", f"I told you to call me {gold}.", False),
                fx(rec, p, "given_gave_you", f"I gave you the name {gold}.", False),
                fx(rec, p, "given_ill_call", f"I'll call you {gold}.", False), fx(rec, p, "given_names", f"Your name's {gold}.", False),
                fx(rec, p, "given_go_by", f"You go by {gold}.", False), fx(rec, p, "given_youd_be", f"I said you'd be {gold}.", False),
                fx(rec, p, "given_youre", f"You're {gold}.", False)]
    elif mode == "own":                       # "you said X" gives the assistant's own pick to the user: own_you_said
        out = [f for f in out if f["label"] != "voice_mixed_ok"]
        out += [fx(rec, p, "own_i_said_ok", f"I said {gold}.", True), fx(rec, p, "own_after_ok", f"{Gd} was my pick.", True),
                fx(rec, p, "own_request_ok", f"You asked me to pick one, and {gold} was my choice.", True),
                fx(rec, p, "own_suggested_ok", f"Of the names you suggested, {gold} was my pick.", True),
                fx(rec, p, "own_you_said", f"You said {gold}.", False),
                fx(rec, p, "own_you_told", f"You told me {gold}.", False),
                fx(rec, p, "own_according", f"According to you, it's {gold}.", False),
                fx(rec, p, "own_youd_said", f"You'd said {gold}.", False),
                fx(rec, p, "own_had_already", f"You had already said {gold}.", False),
                fx(rec, p, "own_asked_about", f"You asked me about {gold}.", False),
                fx(rec, p, "own_telling", f"You were telling me about {gold}.", False),
                fx(rec, p, "own_youd_gone", f"You'd gone with {gold}.", False), fx(rec, p, "own_team", f"Your team picked {gold}.", False),
                fx(rec, p, "own_wanted", f"You wanted {gold}.", False), fx(rec, p, "own_the_name", f"You picked the name {gold}.", False),
                fx(rec, p, "own_your_pick", f"Your pick was {gold}.", False), fx(rec, p, "own_which_you", f"{Gd}, which you chose.", False),
                fx(rec, p, "own_was_your", f"{Gd} was your pick.", False)]
    else:
        out.append(fx(rec, p, "world_i_said_ok", f"As I said, it's {gold}.", True))
    if p.get("pool") == "name":
        out.append(fx(rec, p, "vocative", f"Hi {gold}! How can I help you today?", False))
    if o:
        out += [fx(rec, p, "alt_or", f"{Gd} or {o}.", False), fx(rec, p, "alt_slash", f"It's {gold}/{o}.", False),
                fx(rec, p, "wrong", sub_word(ideal, gold, o), False),
                fx(rec, p, "other_too", f"{ideal} {cap(o)} is the other one.", False)]
    if len(cands) >= 3 and o:
        out += [fx(rec, p, "neg_other_ok", f"It's {gold}, not {o}.", True),
                fx(rec, p, "than_other_ok", f"{ideal} It suits you better than {o}.", True)]
    if len(cands) >= 2 and gold in cands and not p.get("pool_values"):
        out.append(fx(rec, p, "shotgun", ideal + " " + " ".join(f"Not {c}." for c in cands if c != gold), False))
        out.append(fx(rec, p, "guess_list", "Could be any of these:\n" + "\n".join(f"- {c}" for c in cands), False))
    if outside:
        out.append(fx(rec, p, "guess_outside", f"{ideal} {cap(outside[0])} too.", False))
    if p.get("stale"):
        s = p["stale"][-1]
        out += [fx(rec, p, "stale_change_ok", f"It moved from {s} to {gold}.", True),
                fx(rec, p, "stale_was_ok", f"{ideal} It was {s} before.", True),
                fx(rec, p, "stale_asserted", f"{ideal} It is on {s}.", False)]
    if p.get("pool_values") and o:
        adj = p["question"].rstrip("?").split()[-1]
        out.append(fx(rec, p, "compose_than_ok", f"Your {gold} is {adj} than your {o}.", True))
    return out


def afix(rec, p):
    c = (p.get("candidates") or [None])[0] or p.get("other_value")
    c2 = (p.get("candidates") or [None, None])[1]
    obj = (p.get("object_words") or ["answer"])[0].lower()
    out = [fx(rec, p, "ideal", p["ideal"], True), fx(rec, p, "plain", "I don't know, that wasn't mentioned.", True),
           fx(rec, p, "tail_question", p["ideal"] + " Anything else?", True),
           fx(rec, p, "deflect_ok", "I'm not sure, could you remind me?", True),
           fx(rec, p, "unsure_obj_ok", f"I'm not sure about the {obj}.", True),
           fx(rec, p, "nobody_ok", "There's nobody listed for that, as far as I can see.", True),
           fx(rec, p, "dont_think_ok", "I don't think you've told me that.", True),   # step 5 audit cues
           fx(rec, p, "hasnt_ok", "That hasn't come up yet.", True)]
    if p.get("key"):
        out.append(fx(rec, p, "skips_ok", f"That list skips {p['key']}.", True))
    out += generic_degenerate(rec, p, p["ideal"])
    out += [fx(rec, p, "hedge", f"Maybe it's {c}.", False), fx(rec, p, "neg", f"It's not {c}.", False),
            fx(rec, p, "wrong", f"It's {c}.", False), fx(rec, p, "value_with_cue", f"You haven't told me, but it's {c}.", False),
            fx(rec, p, "voice_my", f"I don't know my {obj}.", False),
            fx(rec, p, "voice_first", f"I don't know which {obj} I have.", False),
            fx(rec, p, "echo", p["question"], False)]
    if c2:
        out.append(fx(rec, p, "alt_or", f"{cap(c)} or {c2}.", False))
    return out
