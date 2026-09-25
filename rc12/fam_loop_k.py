"""RC-12 DEV builders: LOOP (degeneration stress) and K (knowledge-bearing follow-ups, P-045) (SPEC s3).

LOOP: a story request at u1, then 11 different continuation requests (never "repeat" or "summarize"). Every reply
is a graded LOOP probe; the IDEAL uses fresh story sentences (no sentence twice in a conversation).
K: "I'm flying to Portugal next month." ... "What language do they speak there?" (reference + world knowledge),
plus a one-turn control that names the referent ("What language do they speak in Portugal?"). Always flagged
knowledge-bearing, scored and reported separately, never pooled with the knowledge-free families."""
import common as C
import pools_misc as M


def build_loop(n=40):
    rng = C.stream("LOOP")
    topics = list(M.LOOP_TOPICS)
    rng.shuffle(topics)
    out = []
    for k in range(n):
        conv = C.Conv("LOOP", "story", C.rid("LOOP", k + 1), rng)
        prompts = rng.sample(M.LOOP_PROMPTS, C.N_TURNS - 1)
        sents = rng.sample(M.STORY_SENTENCES, 2 * C.N_TURNS)
        texts = [rng.choice(M.LOOP_START).format(t=topics[k % len(topics)])] + prompts
        for t, text in enumerate(texts, start=1):
            ideal = sents[2 * (t - 1)] + " " + sents[2 * (t - 1) + 1]
            conv.add_probe(t, "P", "LOOP", text, ideal, gold=None, candidates=[], pool=None)
        conv.meta.update(unit="mean", topic=topics[k % len(topics)])
        out.append(conv.record())
    return out


def build_k():
    rng = C.stream("K")
    items = list(M.K_ITEMS)
    rng.shuffle(items)
    mains = C.probe_turns(rng, len(items))
    out = []
    for k, ((qtype, country, gold), p) in enumerate(zip(items, mains)):
        pair_id = f"k-{k + 1:02d}"
        conv = C.Conv("K", "followup", C.rid("K", k + 1), rng, knowledge=True)
        r = rng.randint(1, min(6, p - 1))
        conv.put(r, "S", rng.choice(M.K_REF).format(c=country),
                 facts=[dict(holder="user", object="referent", value=country, role="referent")])
        conv.add_probe(p, "P", "VAL", M.K_Q[qtype], M.K_IDEAL[qtype].format(g=gold[0]), gold=gold[0],
                       accepted=gold, candidates=[], pool=qtype, holder="world", object_words=[], src=[],
                       d=p - r, ref_turn=r, referent=country, pair_id=pair_id)
        conv.meta.update(unit="mean", qtype=qtype, referent=country, pair_id=pair_id)
        conv.fill()
        out.append(conv.record())
        ctrl = C.Conv("K", "control", C.rid("K", k + 1, "c"), rng, n_turns=1, knowledge=True)
        ctrl.add_probe(1, "P", "VAL", M.K_CTRL[qtype].format(c=country), M.K_IDEAL[qtype].format(g=gold[0]),
                       gold=gold[0], accepted=gold, candidates=[], pool=qtype, holder="world", object_words=[],
                       src=[], d=0, ref_turn=1, referent=country, pair_id=pair_id)
        ctrl.meta.update(unit="mean", qtype=qtype, referent=country, pair_id=pair_id)
        out.append(ctrl.record())
    return out
