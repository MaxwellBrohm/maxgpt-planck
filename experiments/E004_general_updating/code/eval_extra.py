"""Extra eval variants for E002, built in the E001 item format (held-out wording).

noupd_incid : the E001 original statement, then an INCIDENTAL same-type mention that is not a
              statement about any tracked object ("The pool near my place is closed every Tuesday"),
              then the tail. Gold = the original value. "Latest mentioned value wins" fails it.
              Its incidental sentences are eval-only (train_data.py uses different ones).

crossed (POST HOC, added 2026-09-24 07:00 after seed 0's step-50 probe hit 1.0; not part of the
              pre-registered pass rule): both objects are stated and then both are corrected,
                [A: v0] [B: v1] [A corrected: v2] [B corrected: v3] + tail
              asked about A (gold v2) and, on the same dialogue, about B (gold v3). Candidates v0-v3.
              cross_pair = both right. first-mentioned fails both, last-mentioned / last-correction fail A,
              first-correction fails B; only tracking the latest statement per object passes the pair.
              The E001 controls cannot exclude "a correction marker wins, else the first value": every
              E001 correction starts with "Actually" and no E001 item corrects the other object.
"""
import random

import items_new as N

OTHER_CORR = {
    "day": lambda v: (f"Actually, my haircut is on {v} now.", f"Got it, your haircut is on {v}."),
    "color": lambda v: (f"Actually, the color of my new bike is {v} now.", f"Got it, the color of your new bike is {v}."),
}
OTHER_Q = {
    "day": ("What day is my haircut?", "Your haircut is on"),
    "color": ("What color is my new bike?", "The color of your new bike is"),
}

INCID = {
    "day": ("The pool near my place is closed every {v}, which is annoying.", "That is annoying, hopefully it changes."),
    "color": ("My neighbor's new puppy has a {v} collar.", "That sounds adorable."),
}


def build(n_scen=32, distances=(0, 4, 10), seed=4242, families=("day", "color")):
    rng = random.Random(seed)
    P = N.pool(N.EXCLUDE + ["pool", "puppy", "collar", "neighbor"])
    out = []
    for fam in families:
        F = N.FAMILIES[fam]
        for d in distances:
            for s in range(n_scen):
                v = rng.sample(F["values"], 2)
                fills = rng.sample(P, 1 + d)
                u, a = INCID[fam]
                turns = [F["key"](v[0]), fills[0], (u.format(v=v[1]), a)] + fills[1:]
                out.append(dict(task="N_noupd_incid", fam=fam, var="noupd_incid", d=d, sid=f"{fam}|{d}|x{s}", k=0,
                                turns=turns, question=F["question"], prefix=F["prefix"],
                                cands={"gold": " " + v[0], "later": " " + v[1]}))
    return out


def build_crossed(n_scen=32, distances=(0, 4, 10), seed=5151, families=("day", "color")):
    rng = random.Random(seed)
    P = N.pool(N.EXCLUDE)
    out = []
    for fam in families:
        F = N.FAMILIES[fam]
        for d in distances:
            for s in range(n_scen):
                v = rng.sample(F["values"], 4)
                fills = rng.sample(P, 3 + d)
                turns = [F["key"](v[0]), fills[0], F["other"](v[1]), fills[1], F["key_corr"](v[2]), fills[2],
                         OTHER_CORR[fam](v[3])] + fills[3:]
                sid = f"{fam}|{d}|c{s}"
                out.append(dict(task="N_cross_A", fam=fam, var="cross_A", d=d, sid=sid, k=1, turns=turns,
                                question=F["question"], prefix=F["prefix"],
                                cands={"gold": " " + v[2], "orig": " " + v[0], "other": " " + v[1], "other_corr": " " + v[3]}))
                q, pre = OTHER_Q[fam]
                out.append(dict(task="N_cross_B", fam=fam, var="cross_B", d=d, sid=sid, k=1, turns=turns,
                                question=q, prefix=pre,
                                cands={"gold": " " + v[3], "orig": " " + v[1], "other": " " + v[0], "other_corr": " " + v[2]}))
    return out
