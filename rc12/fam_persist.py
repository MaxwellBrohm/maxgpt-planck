"""RC-12 DEV builder: PERSIST, instruction persistence across 6-12 turns (SPEC s3).

hold (32 = 8 rules x 4; I at u1 in half, u2 in half): every reply at distance 6-11 from I is a graded FMT probe.
override (16 = 4 rule pairs x 4; I1 at u1, O at u4 or u5): replies at distance >= 6 from O are graded: the new
rule followed AND the old one absent. Every other turn is a filler; IDEAL replies obey the rule in force.
Deviation from SPEC (notes.txt): two rules (lowercase, brackets) are added to SPEC's six so 32 hold
conversations balance exactly."""
import common as C
import pools_vals as V

FIRST_WORDS = ["Indeed", "Behold", "Ahoy", "Greetings"]
CLOSINGS = ["over and out", "stay curious", "that is all", "until next time"]
WORDS = ["splendid", "nifty", "dandy", "groovy"]
QUESTION_TAILS = ["Does that help?", "Want another idea?", "Have you tried that before?", "What do you think?",
                  "Shall I say more?", "Does that sound doable?", "Would that work for you?",
                  "Any other questions?", "Is that useful?", "Want me to go on?", "Does that make sense?",
                  "Should I add anything?"]
HOLD_RULES = ["caps", "lowercase", "first_word", "closing", "one_sentence", "question_end", "word", "brackets"]
OVERRIDE_PAIRS = [("first_word", "first_word"), ("closing", "closing"), ("word", "word"), ("caps", "lowercase")]
ACK_I = "Sure, I'll do that from now on."
ACK_O = "Okay, switching to the new rule now."

INSTR = {
    "caps": "From now on, please write every reply entirely in capital letters.",
    "lowercase": "From now on, please write every reply in lowercase letters only, with no capitals at all.",
    "first_word": "From now on, start every reply with the word {a}.",
    "closing": 'From now on, end every reply with the words "{a}".',
    "one_sentence": "From now on, keep every reply to exactly one sentence.",
    "question_end": "From now on, end every reply with a question for me.",
    "word": 'From now on, use the word "{a}" somewhere in every reply.',
    "brackets": "From now on, put your whole reply inside square brackets.",
}
OVERRIDE = {
    "first_word": "Change of plan: stop starting with {old}. From now on, start every reply with the word {a} "
                  "instead.",
    "closing": 'Change of plan: stop ending with "{old}". From now on, end every reply with the words "{a}" '
               'instead.',
    "word": 'Change of plan: never use the word "{old}" again. From now on, use "{a}" in every reply instead.',
    "lowercase": "Change of plan: no more capital letters. From now on, write every reply in lowercase letters "
                 "only.",
}


def rule_arg(rule, rng, exclude=()):
    pool = {"first_word": FIRST_WORDS, "closing": CLOSINGS, "word": WORDS}.get(rule)
    return rng.choice([x for x in pool if x not in exclude]) if pool else None


def apply_rule(rule, arg, s, k=0):
    """the IDEAL reply s rewritten to obey rule (k picks a question tail so tails vary inside a conversation)."""
    if rule == "caps":
        return s.upper()
    if rule == "lowercase":
        return s.lower()
    if rule == "first_word":
        return f"{arg}, {V.decap(s)}"
    if rule == "closing":
        return s.rstrip(".!?") + f", {arg}."
    if rule == "one_sentence":
        return s
    if rule == "question_end":
        return s + " " + QUESTION_TAILS[k % len(QUESTION_TAILS)]
    if rule == "word":
        return f"{V.cap(arg)}! {s}"
    if rule == "brackets":
        return f"[{s}]"
    raise ValueError(rule)


def _conv(rng, num, cell, rules, i_turn, o_turn, checked):
    """rules: [(rule, arg)] (one for hold, old+new for override)."""
    conv = C.Conv("PERSIST", cell, C.rid("PERSIST", num), rng)
    fillers = conv.filler_pool()
    rng.shuffle(fillers)
    (r0, a0) = rules[0]
    conv.put(i_turn, "I", INSTR[r0].format(a=a0), ideal=apply_rule(r0, a0, ACK_I),
             facts=[dict(rule=r0, arg=a0, role="instruction")])
    if o_turn:
        (r1, a1) = rules[1]
        conv.put(o_turn, "O", OVERRIDE[r1].format(a=a1, old=a0), ideal=apply_rule(r1, a1, ACK_O),
                 facts=[dict(rule=r1, arg=a1, role="override", old_rule=r0, old_arg=a0)])
    for t in range(1, C.N_TURNS + 1):
        if t in conv.turns:
            continue
        q, a, label = fillers.pop()
        if t < i_turn:
            ideal = a
        elif o_turn and t > o_turn:
            ideal = apply_rule(rules[1][0], rules[1][1], a, k=t)
        else:
            ideal = apply_rule(r0, a0, a, k=t)
        if t in checked:
            if o_turn:
                spec = dict(rule=rules[1][0], arg=rules[1][1], absent=dict(rule=r0, arg=a0))
                src = [i_turn, o_turn]
            else:
                spec = dict(rule=r0, arg=a0, absent=None)
                src = [i_turn]
            conv.add_probe(t, "P", "FMT", q, ideal, fmt=spec, src=src, d=t - (o_turn or i_turn),
                           gold=None, candidates=[], pool=None)
        else:
            conv.put(t, "D", q, ideal=ideal, asks=C.ASKING[label])
    conv.meta.update(unit="mean", rules=[list(r) for r in rules], i_turn=i_turn, o_turn=o_turn,
                     checked=sorted(checked))
    return conv.record()


def build_persist(n_hold=32, n_over=16):
    rng = C.stream("PERSIST")
    rows = [(r, it) for r in HOLD_RULES for it in (1, 1, 2, 2)]
    rng.shuffle(rows)
    out, num = [], 0
    for rule, it in rows:
        num += 1
        checked = [t for t in range(1, C.N_TURNS + 1) if 6 <= t - it <= 11]
        out.append(_conv(rng, num, "hold", [(rule, rule_arg(rule, rng))], it, None, checked))
    rows = [(pair, ot) for pair in OVERRIDE_PAIRS for ot in (4, 4, 5, 5)]
    rng.shuffle(rows)
    for (old, new), ot in rows:
        num += 1
        a0 = rule_arg(old, rng)
        a1 = rule_arg(new, rng, exclude=(a0,))
        checked = [t for t in range(1, C.N_TURNS + 1) if t - ot >= 6]
        out.append(_conv(rng, num, "override", [(old, a0), (new, a1)], 1, ot, checked))
    return out
