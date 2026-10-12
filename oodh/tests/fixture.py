"""Synthetic OOD-H world for the build and gate tests: made-up threads (no OASST2 text), reserved made-up tree ids, and
probe batches in the three hand-written shapes build.py reads (batch_1: kind H-*, turn_kind, text; batch_2: RC-12
turns with asks; batch_3: kind X|P, probe_kind, turn_asks). Two thread types:
  A  u1 names a companion and a first city (S), u2 corrects the city (C); probes H-CORR, H-FACT, H-ABS.
  B  u1 asks for a poem about a pet (S), u2 names a neighbour's pet (L, a lure), u3 thanks (D, asks false);
     probes H-ASK, H-FACT."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.append(os.path.join(REPO, "corpus"))
from oodh import in_oodh_reserve  # noqa: E402  (corpus/oodh.py)

RESERVED = [t for t in (f"fixture-tree-{i}" for i in range(600)) if in_oodh_reserve(t)]
OUTSIDE = [t for t in (f"fixture-tree-{i}" for i in range(600)) if not in_oodh_reserve(t)]
RELS = ["sister", "brother", "cousin", "neighbor", "coworker", "roommate"]
CITIES = ["Porto", "Lisbon", "Madrid", "Seville", "Valencia", "Bilbao", "Granada", "Malaga", "Cordoba", "Toledo"]
PETS = ["cat", "dog", "rabbit", "hamster", "parrot", "turtle", "goldfish", "ferret", "pony", "lizard"]
MONEY = ["$500", "$800", "$1,000", "$1,500", "$2,000", "$300", "$250", "$3,000"]
KINS = ["sister", "brother", "cousin", "neighbor", "coworker", "roommate", "mother", "father", "aunt", "uncle"]
WORKS = ["poem", "song", "story", "letter", "essay", "speech", "limerick", "haiku", "toast"]


def pool(gold, values, extra=()):
    return list(dict.fromkeys([gold] + list(extra) + values))[:10]


def thread_a(k):
    rel, c0, c1 = RELS[k % len(RELS)], CITIES[k % 10], CITIES[(k + 3) % 10]
    users = [f"I am planning a trip to {c0} next spring with my {rel}.",
             f"Actually, make that {c1} instead. Can you suggest a few things to do there?"]
    replies = ["That sounds wonderful. Spring is a lovely season for a trip like that.",
               "Of course. You could walk the old town, try the local food and see a museum or two."]
    probes = [dict(pk="H-CORR", grader="VAL", q="Which city did I settle on in the end?", gold=c1, acc=[c1],
                   cands=[c1, c0], pool=pool(c1, CITIES, [c0]), stale=[c0], obj=["city"], src=2,
                   prefix="You settled on", ideal=f"You settled on {c1}."),
              dict(pk="H-FACT", grader="VAL", q="Who did I say I am going with?", gold=rel, acc=[rel], cands=[rel],
                   pool=pool(rel, KINS), stale=[], obj=["going"], src=1, prefix="You are going with your",
                   ideal=f"You are going with your {rel}."),
              dict(pk="H-ABS", grader="ABS", q="How much money did I say I can spend?", gold=None, acc=[], cands=[],
                   pool=list(MONEY), stale=[], obj=["money", "spend"], src=None, prefix=None,
                   ideal="You haven't told me your budget.")]
    return users, replies, [True, True], probes


def thread_b(k):
    pet, pet2 = PETS[k % 10], PETS[(k + 4) % 10]
    users = [f"Please write a short poem about my {pet} for a birthday card.",
             f"My neighbour has a {pet2} too.", "Thank you so much."]
    replies = ["Here is a short one: soft paws at the door, a nap on the floor, a friend I adore.",
               "That is sweet, they could be friends.", "You are welcome, have a lovely day.\n"]
    probes = [dict(pk="H-ASK", grader="VAL", q="What did I first ask you to write?", gold="poem", acc=["poem"],
                   cands=["poem"], pool=pool("poem", WORKS), stale=[], obj=["write"], src=1, prefix="You wanted a",
                   ideal="You wanted a short poem."),
              dict(pk="H-FACT", grader="VAL", q="Which animal did I want it to be about?", gold=pet, acc=[pet],
                   cands=[pet, pet2], pool=pool(pet, PETS, [pet2]), stale=[], obj=["animal"], src=1,
                   prefix="It is about your", ideal=f"It is about your {pet}.")]
    return users, replies, [True, False, False], probes


def shape(b, tid, ci, users, replies, asks, probes):
    n = len(users)
    mids = [tid] + [f"{tid}-m{j}" for j in range(1, 2 * n)]
    out = []
    for j, p in enumerate(probes):
        turn, kind = n + 1 + j, "P" if j == len(probes) - 1 else "X"
        src = [p["src"]] if p["src"] else []
        smid = mids[2 * (p["src"] - 1)] if p["src"] else None
        common = dict(turn=turn, grader=p["grader"], gold=p["gold"], accepted=p["acc"], candidates=p["cands"],
                      pool_values=p["pool"], stale=p["stale"], holder="user", object_words=p["obj"], src=src,
                      d=turn - src[0] if src else None, dc=None, prefix=p["prefix"], ideal=p["ideal"],
                      abstain=p["grader"] == "ABS")
        if b == 1:
            out.append(dict(common, n=j + 1, kind=p["pk"], turn_kind=kind, text=p["q"], source_message_id=smid,
                            source_turn=2 * (p["src"] - 1) if p["src"] else None, corr_source_turn=None))
        elif b == 2:
            out.append(dict(common, kind=kind, probe_kind=p["pk"], question=p["q"], source_msg_id=smid, pool=None,
                            gold_fn=None, pair_id=None, source_turn=p["src"]))
        else:
            out.append(dict(common, n=j + 1, kind=kind, probe_kind=p["pk"], text=p["q"], question=p["q"],
                            source_turn=p["src"], source_msg=2 * (p["src"] - 1) if p["src"] else None))
    rec = dict(tree_id=tid, message_ids=mids, path_rule="best", probes=out)
    if b == 2:
        turns = [dict(i=i + 1, kind="D", text=u, ideal=replies[i], history_reply=replies[i], facts=[], vals=[],
                      asks=asks[i], msg_id=mids[2 * i]) for i, u in enumerate(users)]
        rec.update(id=f"oodh-b2-{ci:03d}", split="oodh", family="OODH", meta=dict(unit="mean", cand_index=ci),
                   turns=turns, n_thread_user_turns=n)
    else:
        rec.update(cand_index=ci, batch=b, n_user_turns=n)
        if b == 3:
            rec.update(turn_asks=asks, user_turn_msg=[2 * i for i in range(n)])
    msgs = [dict(role=r, text=t) for u, a in zip(users, replies) for r, t in (("user", u), ("assistant", a))]
    cand = dict(tree_id=tid, message_ids=mids, path_rule="best", n_user_turns=n, turns=msgs)
    return rec, cand


def make_world(root, n_threads=9):
    """writes candidates.jsonl, batch_1..3.jsonl and batch_1_asks.json under root; returns the tree ids."""
    os.makedirs(root, exist_ok=True)
    cands, batches, asks1, tids = [], {1: [], 2: [], 3: []}, {}, []
    for k in range(n_threads):
        tid = RESERVED[k]
        users, replies, asks, probes = (thread_a if k % 2 == 0 else thread_b)(k)
        b = 1 + k % 3
        rec, cand = shape(b, tid, len(cands), users, replies, asks, probes)
        cands.append(cand)
        batches[b].append(rec)
        tids.append(tid)
        if b == 1:
            asks1[tid] = asks
    for name, rows in [("candidates.jsonl", cands)] + [(f"batch_{b}.jsonl", batches[b]) for b in (1, 2, 3)]:
        with open(os.path.join(root, name), "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
    with open(os.path.join(root, "batch_1_asks.json"), "w") as fh:
        json.dump(asks1, fh)
    return tids
