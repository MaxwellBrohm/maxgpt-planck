"""RC-12 DEV builder: BIND, whose-is-whose in both mention orders (SPEC s3). 30 pairs = 60 records.

owner (12 pairs) and third-party (8): an intro turn names both holders with no value ("My sister and I both got
new bikes this spring."); the two value statements bind by pronoun only ("Mine is teal." / "Hers is red."), so
the question's words ("my sister's bike") match the intro and neither value statement: first/last mention,
wording and overlap shortcuts all split across the twins. Both statements of a conversation use the SAME frame.
perspective (10): "You can call me Marcus." vs "Your name is Juno from now on."; "What's my name?" and "What did
I call you?" alternate; the two statements tie on words shared with the question and on length (step 5 audit).
Twins differ ONLY in the order of the two value statements (same turns, fillers, question)."""
import random

import balance as BL
import common as C
import pools_vals as V

# value type -> (intro for "my {h} and I" / "my {h1} and my {h2}", frames, question, IDEAL)
# frames use {Mine} (Mine/Hers/His) or {I} (I/She/He); {mine} lowercase variant
BIND_TYPES = {
    "colour": dict(obj="bike", intro="{A} both got new bikes this spring.",
                   frames=["{Mine} is {v}.", "{Mine} came in {v}."],
                   q="What color is {whose} bike?", ideal="{Whose} bike is {v}."),
    "petname": dict(obj="kitten", intro="{A} each adopted a kitten last month.",
                    frames=["{Mine} is called {v}.", "{Mine} got the name {v}."],
                    q="What's {whose} kitten called?", ideal="{Whose} kitten is called {v}."),
    "instrument": dict(obj="instrument", intro="{A} both started music lessons this year.",
                       frames=["{I} picked the {v}.", "{I} went with the {v}."],
                       q="Which instrument is {who_is} learning?", ideal="{Who} {is_are} learning the {v}."),
    "city": dict(obj="city", intro="{A} both moved away last year.",
                 frames=["{I} ended up in {v}.", "{I} settled in {v}."],
                 q="Which city did {who} move to?", ideal="{Who_cap} moved to {v}."),
    "food": dict(obj="cook", intro="{A} cooked dinner for each other last night.",
                 frames=["{I} made {v}.", "{I} served {v}."],
                 q="What did {who} cook?", ideal="{Who_cap} cooked {v}."),
    "month": dict(obj="trip", intro="{A} are both planning a trip abroad.",
                  frames=["{I} will go in {v}.", "{I} booked for {v}."],
                  q="Which month is {whose} trip?", ideal="{Whose} trip is in {v}."),
}
# step 5 audit: the old frames crossed the words (user "call", assistant "name"), so the statement that does NOT
# echo the question was always the gold (ANTI_OVERLAP 0.80 of pairs); now both sides have frames with and without
# each question's words and balance.choose() makes the two statements tie on echo and on length
USER_NAME_FRAMES = ["You can call me {n}.", "Everyone calls me {n}.", "Friends call me {n}.", "I go by the name {n}."]
ANAME_FRAMES = ["Your name is {a} from now on.", "I'll name you {a}, if that's okay.", "I'll call you {a}."]


def _forms(holder):
    """pronoun forms for a holder ("user" or a gendered relative)."""
    if holder == "user":
        return dict(Mine="Mine", I="I", whose="my", Whose="Your", who="I", Who="You", Who_cap="You",
                    who_is="I", is_are="are")
    she = V.PRONOUN[holder] == "she"
    return dict(Mine="Hers" if she else "His", I="She" if she else "He", whose=f"my {holder}'s",
                Whose=f"Your {holder}'s", who=f"my {holder}", Who=f"Your {holder}", Who_cap=f"Your {holder}",
                who_is=f"my {holder}", is_are="is")


def _q_fix(q, holder):
    # "Which instrument is I learning?" -> "Which instrument am I learning?"
    return q.replace("is I learning", "am I learning")


def _pair_turns(rng, p):
    """intro turn (or None) and the two value-statement turns inside u1..u6."""
    t = sorted(rng.sample(range(1, 7), 3))
    return t[0], t[1], t[2]


def _twin(rid, cell, state, texts, probe, fill_seed, pair_id, order):
    """one twin; acks and fillers come from the pair's shared state, so twins differ only in texts' order."""
    conv = C.Conv("BIND", cell, rid, random.Random(fill_seed))
    conv._acks = list(state["acks"])
    conv.avoid.update(state["avoid"])
    for (t, kind, text, facts) in texts:
        conv.put(t, kind, text, facts=facts)
    conv.add_probe(**probe, pair_id=pair_id)
    conv.meta.update(unit="pair", pair_id=pair_id, order=order)
    conv.fill()
    return conv.record()


def _pair(rng, cell, pnum, p, ask_first):
    """both twins of one pair; ask_first: the probe asks about h1 (owner: the user; third-party: the holder
    named first in the intro; perspective: "What's my name?")."""
    if cell == "perspective":
        n, a = rng.choice(V.PERSON), rng.choice(V.ANAME)
        if ask_first:
            q, gold, ideal, holder = "What's my name?", n, f"Your name is {n}.", "user"
        else:
            q, gold, ideal, holder = "What did I call you?", a, f"You called me {a}.", "assistant"
        opts = [[x for f in USER_NAME_FRAMES for x in BL.tailed(f.format(n=n))],
                [x for f in ANAME_FRAMES for x in BL.tailed(f.format(a=a))]]
        tu, ta = BL.choose(rng, opts, [0 if ask_first else 1], "none", question=q, values=V.ALL_VALUES)
        s_user = (tu, [dict(holder="user", object="name", value=n, role="user_name")])
        s_asst = (ta, [dict(holder="assistant", object="name", value=a, role="assistant_name")])
        stmts = [s_user, s_asst]
        intro = None
        vtype, objw, pool = "name", ["name", "call"], "name"
    else:
        vtype = rng.choice(list(BIND_TYPES))
        B = BIND_TYPES[vtype]
        if cell == "owner":
            other = rng.choice(V.HOLDER_F + V.HOLDER_M)
            h1, h2 = "user", other
            intro_a = f"My {other} and I"
        else:
            h1, h2 = rng.choice(V.HOLDER_F), rng.choice(V.HOLDER_M)
            if rng.random() < 0.5:
                h1, h2 = h2, h1
            intro_a = f"My {h1} and my {h2}"
        frame = rng.choice(B["frames"])
        v1, v2 = rng.sample(V.POOLS[vtype], 2)
        stmts = [(frame.format(v=v, **_forms(h)), [dict(holder=h, object=B["obj"], value=v, role="bound")])
                 for h, v in ((h1, v1), (h2, v2))]
        intro = B["intro"].format(A=intro_a)
        holder = h1 if ask_first else h2
        gold = v1 if ask_first else v2
        f = _forms(holder)
        q = _q_fix(B["q"].format(**f), holder)
        ideal = B["ideal"].format(v=gold, **f)
        objw, pool = [B["obj"]], vtype
    ti, t1, t2 = _pair_turns(rng, p)
    if intro is None:
        t1, t2 = sorted(rng.sample(range(1, 7), 2))
    state = dict(acks=[x for x in V.ACKS], avoid={"bike", "kitten", "instrument", "trip", "name"})
    rng.shuffle(state["acks"])
    fill_seed = rng.random()
    pair_id = f"bind-{pnum:02d}"
    out = []
    for order, (first, second) in (("a", (stmts[0], stmts[1])), ("b", (stmts[1], stmts[0]))):
        texts = [] if intro is None else [(ti, "S", intro, [])]
        texts += [(t1, "S", first[0], first[1]), (t2, "S", second[0], second[1])]
        gold_turn = t1 if first[1][0]["value"] == gold else t2
        cands = [first[1][0]["value"], second[1][0]["value"]]
        probe = dict(turn=p, kind="P", grader="VAL", question=q, ideal=ideal, gold=gold, candidates=cands,
                     pool=pool, holder=holder, object_words=objw, src=[gold_turn],
                     prefix=None, other_holder=[x for x in (stmts[0][1][0]["holder"], stmts[1][1][0]["holder"])
                                                if x != holder][0])
        out.append(_twin(C.rid("BIND", pnum, order), cell, state, texts, probe, fill_seed, pair_id,
                         "asked_first" if gold_turn == t1 else "asked_second"))
    return out


def build_bind():
    rng = C.stream("BIND")
    plan = [("owner", 12), ("perspective", 10), ("third-party", 8)]
    out, pnum = [], 0
    for cell, n in plan:
        asks = C.spread([True, False], n, rng)
        ps = [12] * (n - n // 4) + [10 + (i % 2) for i in range(n // 4)]
        rng.shuffle(ps)
        for ask_first, p in zip(asks, ps):
            pnum += 1
            out += _pair(rng, cell, pnum, p, ask_first)
    return out
