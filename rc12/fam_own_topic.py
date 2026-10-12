"""RC-12 DEV builders: OWN (references to the model's own earlier answer) and TOPIC (return after a
digression) (SPEC s3).

OWN: Q at u2-u5 makes the model commit (pick 32: one of three offered names; list 16: three numbered ideas);
P at Q+4..Q+9 asks for it back. The gold is a gold_fn on the model's OWN reply at Q (graded by G-DYN); the IDEAL
picks option 1, 2 or 3 in thirds. Lures: a same-type value bound to someone else at u1, and someone else's choice
after Q that frame-echoes the probe ("My friend would pick Saltwind for my boat.").
TOPIC: three projects, two explicitly closed (inline, or introduced and closed in a later turn), one left open
(first/middle/last in thirds; it can get a later "still undecided" update too, TOPIC_PLAN); 3-6 digression
fillers; the probe never names it. Statement wording is length- and echo-balanced (step 5 audit)."""
import balance as B
import common as C
import pools_vals as V
import pools_misc as M


def _own_turns(rng, p):
    q = rng.randint(max(2, p - 9), min(5, p - 4))
    after = rng.randint(q + 1, p - 1)
    return q, after


def build_own(n_pick=32, n_list=16):
    rng = C.stream("OWN")
    out, num = [], 0
    cats = C.spread(list(M.OWN_PICK), n_pick)
    picks = C.spread([0, 1, 2], n_pick)
    rows = list(zip(cats, picks))
    rng.shuffle(rows)
    mains = C.probe_turns(rng, n_pick)
    for (cat, gi), p in zip(rows, mains):
        num += 1
        P = M.OWN_PICK[cat]
        conv = C.Conv("OWN", "pick", C.rid("OWN", num), rng)
        conv.avoid.update(["boat", "puppy", "band", "quiz", "team"])
        a, b, c, x, y = rng.sample(V.POOLS[cat], 5)
        q, after = _own_turns(rng, p)
        g = [a, b, c][gi]
        conv.put(1, "L", P["before"].format(x=x), facts=[dict(holder="other", object=cat, value=x, role="lure")])
        conv.put(q, "Q", P["q"].format(a=a, b=b, c=c), ideal=P["ideal_q"].format(g=g),
                 facts=[dict(holder="assistant", object=cat, value=None, role="options", options=[a, b, c])])
        conv.put(after, "L", P["after"].format(y=y), facts=[dict(holder="other", object=cat, value=y, role="lure")])
        conv.add_probe(p, "P", "DYN", P["probe"], P["ideal_p"].format(g=g),
                       gold_fn=dict(type="pick", src_turn=q, options=[a, b, c]), candidates=[x, a, b, c, y],
                       pool=cat, holder="assistant", object_words=[cat], src=[q], ideal_gold=g, ideal_index=gi)
        conv.meta.update(unit="mean", category=cat, q_turn=q, gap=p - q)
        conv.fill()
        out.append(conv.record())
    topics = C.spread(list(M.OWN_LIST), n_list)
    rng.shuffle(topics)
    mains = C.probe_turns(rng, n_list)
    for t, p in zip(topics, mains):
        num += 1
        items, lures = M.OWN_LIST[t]
        conv = C.Conv("OWN", "list", C.rid("OWN", num), rng)
        conv.avoid.update(w for w in t.split() if len(w) > 3)
        q, after = _own_turns(rng, p)
        ideal_q = "Here are three ideas:\n" + "\n".join(f"{i + 1}. {V.cap(it)}" for i, it in enumerate(items))
        conv.put(1, "L", M.OWN_LIST_BEFORE.format(x=lures[0], t=t),
                 facts=[dict(holder="other", object=t, value=lures[0], role="lure")])
        conv.put(q, "Q", M.OWN_LIST_Q.format(t=t), ideal=ideal_q,
                 facts=[dict(holder="assistant", object=t, value=None, role="list_request")])
        conv.put(after, "L", M.OWN_LIST_AFTER.format(y=lures[1]),
                 facts=[dict(holder="other", object=t, value=lures[1], role="lure")])
        conv.add_probe(p, "P", "DYN", rng.choice(M.OWN_LIST_PROBES), f"The second idea was {items[1]}.",
                       gold_fn=dict(type="list2", src_turn=q), candidates=[], pool=None, holder="assistant",
                       object_words=[], src=[q], ideal_gold=items[1], ideal_items=items, lure_items=lures)
        conv.meta.update(unit="mean", topic=t, q_turn=q, gap=p - q)
        conv.fill()
        out.append(conv.record())
    return out


# ---------------- TOPIC ----------------
# step 5 audit: the update plan is a fixed table, (open position, projects updated later in this order by
# statement index) -> items. It keeps FIRST 16, LAST 14, second-to-last 16 of 48 on the open project, the gold
# never both the first and the last mention, and the open project updated about as often as a closed one
# (n = 1: 4 of 16, n = 2: 10 of 16; status-blind would be 5.3 and 10.7), so mention counts carry no status.
TOPIC_PLAN = [("first", (), 5), ("first", (1,), 3), ("first", (2,), 3), ("first", (0, 1), 1), ("first", (0, 2), 1),
              ("first", (1, 2), 1), ("first", (2, 1), 2), ("middle", (), 5), ("middle", (0,), 1),
              ("middle", (1,), 2), ("middle", (2,), 2), ("middle", (0, 1), 1), ("middle", (0, 2), 1),
              ("middle", (1, 0), 1), ("middle", (1, 2), 1), ("middle", (2, 0), 1), ("middle", (2, 1), 1),
              ("last", (), 6), ("last", (0,), 2), ("last", (1,), 1), ("last", (2,), 2), ("last", (0, 2), 1),
              ("last", (1, 0), 1), ("last", (1, 2), 1), ("last", (2, 0), 1), ("last", (2, 1), 1)]


def build_topic(n=48):
    rng = C.stream("TOPIC")
    rows = [(pos, pat) for pos, pat, c in TOPIC_PLAN for _ in range(c)]
    assert len(rows) == n
    rng.shuffle(rows)
    lens = C.spread(B.TARGETS, n, rng)          # gold length rank long / mid / short, 16 each
    mains = C.probe_turns(rng, n)
    out = []
    for k, ((pos, pat), p) in enumerate(zip(rows, mains)):
        out.append(_topic_one(rng, k + 1, pos, p, pat, lens[k]))
    return out


TOPIC_TPL = {"open": M.TOPIC_OPEN, "closed_inline": M.TOPIC_CLOSED_INLINE, "closed_intro": M.TOPIC_CLOSED_INTRO,
             "close_later": M.TOPIC_CLOSED_LATER, "open_later": M.TOPIC_OPEN_LATER}


def _topic_options(rng, seq):
    """candidate texts per event; the first statement is never additive; two events of one kind get disjoint
    templates (no template twice in a conversation); statements may carry a neutral tail."""
    taken, opts = {}, []
    for i, (kind, noun) in enumerate(seq):
        verb, purpose, pending = M.PROJECTS[noun]
        f = dict(noun=noun, verb=verb, purpose=purpose, pending=pending)
        tpl = [x for x in TOPIC_TPL[kind] if i > 0 or not any(a in x for a in M.TOPIC_ADDITIVE)]
        if sum(k == kind for k, _ in seq) == 2:
            if kind in taken:
                tpl = [x for x in tpl if x not in taken[kind]]
            else:
                tpl = list(tpl)
                rng.shuffle(tpl)
                tpl = tpl[:max(1, len(tpl) // 2)]
                taken[kind] = set(tpl)
        tails = B.TAILS[:2] if kind in ("open", "closed_inline", "closed_intro") else []
        opts.append([x for t in tpl for x in B.tailed(t.format(**f), tails)])
    return opts


def _topic_one(rng, num, pos, p, pattern, target):
    conv = C.Conv("TOPIC", "topic", C.rid("TOPIC", num), rng)     # one cell; the open position is a factor
    oi = ["first", "middle", "last"].index(pos)
    for _ in range(50):                  # redraw projects and wording until a balanced statement set exists
        nouns = rng.sample(list(M.PROJECTS), 3)
        seq = []
        for j, noun in enumerate(nouns):
            if j == oi:
                seq.append(("open", noun))
            else:
                seq.append(("closed_intro" if j in pattern else "closed_inline", noun))
        for j in pattern:                                            # after all three statements, in plan order
            seq.append(("open_later" if j == oi else "close_later", nouns[j]))
        question = rng.choice(M.TOPIC_PROBES)
        gold = nouns[oi]
        if seq[0][1] == gold and seq[-1][1] == gold:   # first AND last mention: FIRST and LAST would both pass
            continue
        texts = B.choose(rng, _topic_options(rng, seq), [i for i, (_, n) in enumerate(seq) if n == gold], target,
                         question=question, values=V.ALL_VALUES, cap=60000)
        if texts is not None:
            break
    else:
        raise ValueError(f"TOPIC {num}: no balanced statement set")
    # the three statements sit in u1..u5 (block start <= 3), then 3-6 digression fillers before the probe
    digress = rng.randint(max(3, p - len(seq) - 3), min(6, p - 1 - len(seq)))
    s0 = p - digress - len(seq)
    if not 1 <= s0 <= 3:
        raise ValueError("topic block does not fit")
    conv.avoid.update(nouns)
    src = []
    for i, ((kind, noun), text) in enumerate(zip(seq, texts)):
        t = s0 + i
        conv.put(t, "C" if kind.endswith("_later") else "S", text,
                 facts=[dict(holder="user", object="project", value=noun, role=kind)])
        if kind in ("open", "open_later"):              # an update of the open project is a source too
            src.append(t)
    ideal = M.TOPIC_IDEAL if M.PROJECTS[gold][2] in texts[[k for k, _ in seq].index("open")] else M.TOPIC_IDEAL_SHORT
    conv.add_probe(p, "P", "VAL", question,
                   ideal.format(noun=gold, pending=M.PROJECTS[gold][2]), gold=gold, candidates=nouns,
                   pool="project", holder="user", object_words=["project"], src=src,
                   closed=[n for n in nouns if n != gold])
    conv.meta.update(unit="mean", pos=pos, digression=digress, block_start=s0, block_len=len(seq),
                     n_later=len(pattern), open_later=oi in pattern, pattern=list(pattern), len_rank=target)
    conv.fill()
    return conv.record()
