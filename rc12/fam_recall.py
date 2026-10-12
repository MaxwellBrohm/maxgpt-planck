"""RC-12 DEV builders: T0 (Tier 0 gate) and RECALL (SPEC s3). Step 5 audit: RECALL statements are length- and
echo-balanced (balance.py) and the abstain cell puts its answerable X before the abstain P in half the items."""
import balance as B
import common as C
import pools_vals as V
from pools_slots import RECALL_SLOTS

BINS = {"d1-3": (1, 3), "d4-7": (4, 7), "d8-11": (8, 11)}
POSITIONS = ["first", "middle", "last"]
STMT_MAX = 11          # RECALL statements sit in u1..u11 (deviation from SPEC u1..u10: notes.txt)


def _stmt(rng, slot, v, holder=None):
    if holder is None:
        return rng.choice(slot["gold"]).format(v=v)
    return rng.choice(slot["lure"]).format(v=v, h=holder)


def _options(slot, v, holder=None):
    """every frame of the statement, each bare and with every neutral tail (audit: length-rank balance)."""
    frames = slot["gold"] if holder is None else slot["lure"]
    return [x for f in frames for x in B.tailed(f.format(v=v, h=holder))]

def _probe_val(conv, turn, kind, slot, gold, cands, src, rng, q=None):
    q = q or rng.choice(slot["q"])
    return conv.add_probe(turn, kind, "VAL", q, slot["ideal"].format(v=gold), gold=gold, candidates=cands,
                          pool=slot["vtype"], holder="user", object_words=slot["obj"], src=src,
                          prefix=slot["prefix"])


# ---------------- T0 ----------------
def build_t0(n=48):
    rng = C.stream("T0")
    ds = C.spread([(1, 1), (1, 2), (2, 1), (2, 2)], n, rng)
    mains = C.probe_turns(rng, n)
    out = []
    for k in range(n):
        conv = C.Conv("T0", "t0", C.rid("T0", k + 1), rng)
        s1, s2 = rng.sample(RECALL_SLOTS, 2)
        while s1["vtype"] == s2["vtype"]:
            s1, s2 = rng.sample(RECALL_SLOTS, 2)
        d1, d2 = ds[k]
        p2 = mains[k]
        st2 = p2 - d2
        p1 = rng.randint(1 + d1, st2 - 1)
        st1 = p1 - d1
        v1, v2 = rng.choice(V.POOLS[s1["vtype"]]), rng.choice(V.POOLS[s2["vtype"]])
        conv.avoid.update(s1["obj"] + s2["obj"])
        conv.put(st1, "S", _stmt(rng, s1, v1), facts=[dict(holder="user", object=s1["key"], value=v1, role="gold")])
        conv.put(st2, "S", _stmt(rng, s2, v2), facts=[dict(holder="user", object=s2["key"], value=v2, role="gold")])
        _probe_val(conv, p1, "X", s1, v1, [v1], [st1], rng)
        _probe_val(conv, p2, "P", s2, v2, [v2], [st2], rng)
        conv.meta.update(unit="mean", d=[d1, d2])
        conv.fill()
        out.append(conv.record())
    return out


# ---------------- RECALL ----------------
def _feasible_ds(binname, pos, p):
    lo, hi = BINS[binname]
    ds = []
    for d in range(lo, hi + 1):
        s = p - d
        if s < 1 or s > STMT_MAX:
            continue
        after = min(p - 1, STMT_MAX) - s          # free statement turns after S
        before = s - 1
        need = {"first": (0, 2), "middle": (1, 1), "last": (2, 0)}[pos]
        if before >= need[0] and after >= need[1]:
            ds.append(d)
    return ds


def _recall_design(rng, n_main=48):
    items = []
    rot = {"d1-3": [6, 5, 5], "d4-7": [5, 6, 5], "d8-11": [5, 5, 6]}
    for b, counts in rot.items():
        for pos, c in zip(POSITIONS, counts):
            items += [(b, pos)] * c
    rng.shuffle(items)
    ps = [12] * n_main
    early = [9 + (i % 3) for i in range(n_main // 4)]
    for p in early:                      # give each early probe turn to a random item that can take it
        cands = [j for j in range(n_main) if ps[j] == 12 and _feasible_ds(*items[j], p)]
        ps[rng.choice(cands)] = p
    return items, ps


def build_recall(n_main=48, n_abs=12):
    rng = C.stream("RECALL")
    items, ps = _recall_design(rng, n_main)
    slots = C.spread(RECALL_SLOTS, n_main, rng)
    lens = {}
    for b in BINS:                       # audit: gold length rank long / mid / short, balanced inside each bin
        idx = [j for j, it in enumerate(items) if it[0] == b]
        for j, tg in zip(idx, C.spread(B.TARGETS, len(idx), rng)):
            lens[j] = tg
    out = []
    for k, ((b, pos), p) in enumerate(zip(items, ps)):
        slot = slots[k]
        conv = C.Conv("RECALL", b, C.rid("RECALL", k + 1), rng)
        conv.avoid.update(slot["obj"])
        d = rng.choice(_feasible_ds(b, pos, p))
        s = p - d
        before = list(range(1, s))
        after = list(range(s + 1, min(p - 1, STMT_MAX) + 1))
        nb = {"first": 0, "middle": 1, "last": 2}[pos]
        lure_turns = sorted(rng.sample(before, nb) + rng.sample(after, 2 - nb))
        gold, l1, l2 = rng.sample(V.POOLS[slot["vtype"]], 3)
        holders = rng.sample(V.HOLDERS, 2)
        q = rng.choice(slot["q"])
        texts = B.choose(rng, [_options(slot, gold)] + [_options(slot, lv, h) for lv, h in zip((l1, l2), holders)],
                         [0], lens[k], question=q, values=V.ALL_VALUES)
        if texts is None:
            raise ValueError(f"RECALL {k + 1}: no balanced statement set for {slot['key']} / {q}")
        conv.put(s, "S", texts[0], facts=[dict(holder="user", object=slot["key"], value=gold, role="gold")])
        for t, lv, h, text in zip(lure_turns, (l1, l2), holders, texts[1:]):
            conv.put(t, "L", text, facts=[dict(holder=h, object=slot["key"], value=lv, role="lure")])
        order = sorted([(s, gold)] + list(zip(lure_turns, (l1, l2))))
        _probe_val(conv, p, "P", slot, gold, [v for _, v in order], [s], rng, q=q)
        conv.meta.update(unit="mean", pos=pos, d=d, dbin=b, len_rank=lens[k])
        conv.fill()
        out.append(conv.record())
    out += _build_recall_abs(rng, n_abs, n_main)
    return out


def _build_recall_abs(rng, n, offset):
    mains = C.probe_turns(rng, n)             # the LATER of the two probes (audit: P or X may come first)
    out = []
    pslots = C.spread(RECALL_SLOTS, n, rng)
    xfirst = C.spread([True, False], n, rng)   # audit: the answerable X before the abstain P in half, after in half
    for k in range(n):
        slot = pslots[k]
        xslot = rng.choice([s for s in RECALL_SLOTS if s["vtype"] != slot["vtype"]])
        conv = C.Conv("RECALL", "abstain", C.rid("RECALL", offset + k + 1), rng)
        conv.avoid.update(slot["obj"] + xslot["obj"])
        while True:
            if xfirst[k]:
                p = mains[k]
                x = rng.choice(range(5, p))
            else:
                x = mains[k]
                p = rng.choice(range(5, x))
            sx = rng.randint(max(1, x - 8), x - 2)
            free = [t for t in range(1, min(p, STMT_MAX + 1)) if t not in (x, sx)]
            if sx != p and len(free) >= 2:
                break
        lure_turns = sorted(rng.sample(free, 2))
        l1, l2 = rng.sample(V.POOLS[slot["vtype"]], 2)
        xv = rng.choice(V.POOLS[xslot["vtype"]])
        holders = rng.sample(V.HOLDERS, 2)
        for t, lv, h in zip(lure_turns, (l1, l2), holders):
            conv.put(t, "L", _stmt(rng, slot, lv, h), facts=[dict(holder=h, object=slot["key"], value=lv,
                                                                   role="lure")])
        conv.put(sx, "S", _stmt(rng, xslot, xv), facts=[dict(holder="user", object=xslot["key"], value=xv,
                                                              role="gold")])
        conv.add_probe(p, "P", "ABS", rng.choice(slot["q"]), slot["abst"], gold=None, candidates=[l1, l2],
                       pool=slot["vtype"], holder="user", object_words=slot["obj"], src=[],
                       prefix=None, lure_holders=holders)
        _probe_val(conv, x, "X", xslot, xv, [xv], [sx], rng)
        conv.meta.update(unit="all", x_first=xfirst[k])
        conv.fill()
        out.append(conv.record())
    return out
