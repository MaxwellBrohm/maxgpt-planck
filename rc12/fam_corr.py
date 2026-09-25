"""RC-12 DEV builder: CORR, corrections applied 4+ turns later (SPEC s3; E004 dev.2 layout).

Cells (16 each): U-diff (A's latest correction by pronoun, ellipsis, demonstrative or alias: no word of the
question), U-same (latest by full phrase or head noun), C_noupd (A never corrected, B corrected 1-3 times),
C_twoslot (A corrected, B's corrections come after A's latest). B is a second object of the same value type. An
indirect correction (pronoun, ellipsis, demonstrative) always directly follows the statement it corrects.
dc = probe turn - turn of A's latest correction >= 4 in every cell that corrects A.
Step 5 audit (notes.txt): in the U cells something of the same type ALWAYS follows A's latest correction, and how
much varies (t_after 1, 2 or 3 values: B's statement and correction, and/or same-type LURES bound to another
holder, plain or marked, naming A's own object or a third one), so "latest value", "latest correction", "value
before the last new statement" and fixed offsets from the end stop finding the gold. C_twoslot has two layouts
(blocks: A then B; bfirst: B stated first, B corrected after A's latest) and B corrected once or twice (m)."""
import balance as BL
import common as C
import pools_vals as V
from pools_slots import (CORR_TYPES, CORR_OBJECTS, CORR_FORMS, INDIRECT_FORMS, CORR_LURE_PLAIN, CORR_LURE_MARKED,
                         CORR_LURE_ZERO)

CELLS = ["U-diff", "U-same", "C_noupd", "C_twoslot"]
# step 5 audit: what follows A's latest correction in the U cells, as (events, items) per B placement. Lm / Lp a
# marked / plain same-type lure of another holder, B0 B's statement, B1 B's own (marked, indirect) correction.
# Found by a search that keeps every structural rule (LAST, n-th value from the end, latest / n-th-last marked
# turn, value before the latest fresh statement, the same with the last value turn excluded) at <= 5 of 16.
# U-same never ends on A's latest (with its words and its marker, LAST, MARKER, OVERLAP and WORDING would all
# pass it: G4); U-diff does in 2 items (only LAST and MARKER pass those).
U_TAILS = {"U-diff": {"before": [((), 2), (("Lm", "Lm"), 2), (("Lm", "Lp"), 2), (("Lm",), 1), (("Lp",), 1)],
                      "after": [(("Lm", "B0", "B1"), 6), (("B0", "B1", "Lm"), 1), (("Lp", "B0", "B1"), 1)]},
           "U-same": {"before": [(("Lm",), 3), (("Lm", "Lm"), 2), (("Lp", "Lp"), 2), (("Lm", "Lp"), 1)],
                      "after": [(("Lm", "B0", "B1"), 5), (("B0", "B1", "Lm"), 3)]}}
VTYPES = ["weekday", "month", "colour", "city"]


def _text(rng, vtype, obj, ev, objA):
    phrase, head, alias = obj
    T = CORR_TYPES[vtype]
    v = ev["value"]
    if ev["role"] == "lure":
        if ev.get("zero"):
            text = CORR_LURE_ZERO["colour" if vtype == "colour" else "other"].format(n=ev["holder"], o=phrase, v=v)
            return "Actually, " + text if ev["form"] == "marked" else text
        if ev["form"] == "marked":
            return rng.choice(CORR_LURE_MARKED).format(h=ev["holder"], o=phrase, now=T["now"].format(v=v), v=v)
        return CORR_LURE_PLAIN[vtype].format(h=ev["holder"], o=phrase, v=v)
    if ev["role"] == "orig":
        tpl = T["alias_orig"] if ev.get("alias") else rng.choice(T["orig"])
        return tpl.format(o=phrase, v=v, alias=alias)
    now = T["now"].format(v=v)
    return rng.choice(CORR_FORMS[ev["form"]]).format(o=phrase, head=head, v=v, now=now, alias=alias, oa=objA[0])


def _lure(rng, mark, where="after", zero=False):
    """a same-type value bound to another holder: marked lures name A's own object (so "the latest correction
    naming the asked object" meets one), plain ones name it in half; a lure BEFORE A names a third object."""
    echo = False if where == "before" else True if mark == "m" else rng.random() < 0.5
    return dict(obj="L", role="lure", form="marked" if mark == "m" else "plain", echo=echo and not zero, where=where,
                zero=zero)


def _events(rng, cell, k, form, place, m, bform, tail):
    """ordered statement events: dict(obj A | B | L, role orig | corr | lure, form, value=None, alias)."""
    A = [dict(obj="A", role="orig", form=None, alias=(form == "alias" or rng.random() < 0.25))]
    B = [dict(obj="B", role="orig", form=None, alias=False)]
    if cell in ("U-diff", "U-same", "C_twoslot"):
        for j in range(k):
            f = form if j == k - 1 else rng.choice(["full", "head"])
            A.append(dict(obj="A", role="corr", form=f))
    if cell in ("U-diff", "U-same"):          # tail = the events after A's latest (U_TAILS)
        ev = {"B0": B[0], "B1": dict(obj="B", role="corr", form=bform)}
        zero = cell == "U-diff" and place == "before"        # audit: zero-echo lures (see CORR_LURE_ZERO)
        after = [ev[x] if x in ev else _lure(rng, x[1], zero=zero) for x in tail]
        if place == "before":
            return B + A + after
        seq = A + after
        return ([_lure(rng, "p", "before")] if len(seq) < 7 else []) + seq   # 7 = size of the weekday pool
    if cell == "C_twoslot":
        layout, m = place, tail
        if layout == "blocks":
            bc = [dict(obj="B", role="corr", form=bform)] + [dict(obj="B", role="corr", form=rng.choice(
                ["full", "head"])) for _ in range(m - 1)]
            return A + B + bc
        bc = [dict(obj="B", role="corr", form=rng.choice(["full", "head"])) for _ in range(m)]
        return B + A + bc
    # C_noupd: B orig + m corrections; A first, or between B's orig and B's latest
    bcorr = [dict(obj="B", role="corr", form=rng.choice(["full", "head"])) for _ in range(m)]
    tl, eb = tail
    if place == "first":
        seq = A + B + bcorr
    else:
        cut = rng.randint(0, m - 1)            # A goes after B's orig and before B's latest correction
        seq = B + bcorr[:cut] + A + bcorr[cut:]
    if tl or form == "anchor":                # a plain lure after B's statements (A-first half, and every anchor
        seq.append(_lure(rng, "p"))           # item, so the anchor is never simply "the newest turn")
    # audit: another holder's statement about A's own object, before A (eb) or after A but never last, so neither
    # "the first / latest / non-newest turn naming the asked object" finds the gold without the holder
    a0 = seq.index(A[0])
    seq.insert(a0 if eb else rng.randint(a0 + 1, len(seq) - 1), dict(obj="L", role="lure", form="plain", echo=True,
                                                                     where="after", zero=False))
    for j in range(1, len(seq)):              # indirect only right after the same object's statement
        if seq[j]["obj"] == "B" and seq[j]["role"] == "corr" and seq[j - 1]["obj"] == "B" and rng.random() < 0.5:
            seq[j]["form"] = rng.choice(INDIRECT_FORMS)
    if form == "anchor":                      # B's latest correction also names A as unchanged
        [ev for ev in seq if ev["obj"] == "B"][-1]["form"] = "anchor"
    return seq


def _blocks(seq):
    """group events into blocks: an indirect correction is glued to the event before it."""
    blocks = []
    for j, ev in enumerate(seq):
        if ev["role"] == "corr" and ev["form"] in INDIRECT_FORMS and j > 0:
            assert seq[j - 1]["obj"] == ev["obj"], "indirect correction must follow its own object"
            blocks[-1].append(j)
        else:
            blocks.append([j])
    return blocks


def _design(rng, cell, n):
    """rows (k, (latest form, place, tail), value type). place: U cells B before / after A; C_twoslot layout
    blocks / bfirst; C_noupd A first / middle. tail: U cells the marks of the lures after A's latest; C_twoslot
    B's number of corrections m (1 / 2); C_noupd A-first: a plain lure after B's statements (True / False)."""
    ks = C.spread([1, 2, 3], n)
    if cell in ("U-diff", "U-same"):
        forms = ["pronoun", "ellipsis", "alias", "demonstrative"] if cell == "U-diff" else ["full", "head"]
        combos = C.spread([(f, b) for f in forms for b in ["before", "after"]], n)
        tails = {b: [t for t, n in U_TAILS[cell][b] for _ in range(n)] for b in ("before", "after")}
        for t in tails.values():
            rng.shuffle(t)
        combos = [(f, b, tails[b].pop()) for f, b in combos]
    elif cell == "C_twoslot":
        combos = C.spread([(f, lay, m) for f in ["full", "head", "pronoun", "alias"] for lay in ["blocks", "bfirst"]
                           for m in (1, 2)], n)
    else:
        combos = C.spread([(f, a, (t, eb)) for f in ["anchor", None] for a, t in [
            ("first", True), ("first", False), ("middle", None), ("middle", None)] for eb in (True, False)], n)
    vts = C.spread(VTYPES, n)
    rows = list(zip(ks, combos, vts))
    rng.shuffle(rows)
    return rows


def build_corr(n_per_cell=16):
    rng = C.stream("CORR")
    out, num = [], 0
    for cell in CELLS:
        rows = _design(rng, cell, n_per_cell)
        mains = C.probe_turns(rng, n_per_cell)
        lens = C.spread(BL.TARGETS, n_per_cell, rng)       # audit: the gold statement's length rank
        for (k, (form, place, tail), vtype), p, lr in zip(rows, mains, lens):
            num += 1
            out.append(_one(rng, cell, num, k, form, place, tail, vtype, p, lr))
    return out


def _one(rng, cell, num, k, form, place, tail, vtype, p, len_rank):
    conv = C.Conv("CORR", cell, C.rid("CORR", num), rng)
    objA, objB, objC = rng.sample(CORR_OBJECTS[vtype], 3)
    m = rng.choice([1, 2, 3])
    # B's own correction: indirect in the U cells (E004 dev.2), any form in C_twoslot
    bform = rng.choice(INDIRECT_FORMS if cell in ("U-diff", "U-same") else INDIRECT_FORMS + ("full",))
    seq = _events(rng, cell, k, form, place, m, bform, tail)
    lure_h, lure_n = rng.sample(V.HOLDERS, 3), rng.sample(V.PERSON, 3)
    for ev in [e for e in seq if e["obj"] == "L"]:
        ev["holder"] = lure_n.pop() if ev.get("zero") else lure_h.pop()
    vals = rng.sample(V.POOLS[vtype], len(seq))
    for ev, v in zip(seq, vals):
        ev["value"] = v
    a_idx = [j for j, ev in enumerate(seq) if ev["obj"] == "A"]
    blocks = _blocks(seq)
    lens = [len(b) for b in blocks]
    corrected = cell != "C_noupd"

    def ok(starts):
        turn = {}
        for b, s0 in zip(blocks, starts):
            for off, j in enumerate(b):
                turn[j] = s0 + off
        return (not corrected) or turn[a_idx[-1]] <= p - 4

    starts = C.place_blocks(rng, lens, 1, p - 1, ok)
    turn = {}
    for b, s0 in zip(blocks, starts):
        for off, j in enumerate(b):
            turn[j] = s0 + off
    for w in (objA[0] + " " + objB[0] + " " + objC[0] + " " + objA[2] + " " + objB[2]).split():
        conv.avoid.add(w.strip(".,").lower())
    objs = [objA if ev["obj"] == "A" or (ev["obj"] == "L" and ev["echo"]) else objB if ev["obj"] == "B" else objC
            for ev in seq]
    texts = [_text(rng, vtype, o, ev, objA) for o, ev in zip(objs, seq)]
    gold_j = a_idx[-1]
    # audit: the gold statement at a balanced length rank (an ellipsis correction was the shortest turn: SHORTEST)
    opts = [BL.tailed(x) if j == gold_j else [x, x.rstrip(".") + rng.choice(BL.TAILS)] for j, x in enumerate(texts)]
    order = {"long": ("long", "mid", "short"), "mid": ("mid", "long", "short"), "short": ("short", "mid", "long")}
    for len_rank in order[len_rank]:          # a short ellipsis cannot always be the longest: fall back to mid,
        texts = BL.choose(rng, opts, [gold_j], len_rank)   # to short only as a last resort (test_gens_audit counts)
        if texts is not None:
            break
    else:
        raise ValueError(f"CORR {num}: no length-balanced statement set")
    for j, (ev, o, text) in enumerate(zip(seq, objs, texts)):
        kind = "L" if ev["obj"] == "L" else "S" if ev["role"] == "orig" else "C"
        conv.put(turn[j], kind, text, facts=[dict(holder=ev.get("holder", "user"), object=o[0], value=ev["value"],
                                                  role=ev["role"], ref=ev["obj"], form=ev["form"])])
    gold = seq[gold_j]["value"]
    stale = [seq[j]["value"] for j in a_idx[:-1]]
    cands = [seq[j]["value"] for j in sorted(range(len(seq)), key=lambda j: turn[j])]
    T = CORR_TYPES[vtype]
    prefix = T["prefix"].format(o=objA[0])
    conv.add_probe(p, "P", "VAL", rng.choice(T["q"]).format(o=objA[0]), f"{prefix} {gold}.", gold=gold,
                   candidates=cands, pool=vtype, stale=stale, holder="user", object_words=objA[0].split(),
                   src=[turn[gold_j]], dc=(p - turn[gold_j]) if corrected else None, prefix=prefix,
                   b_values=[ev["value"] for ev in seq if ev["obj"] == "B"], alias=objA[2],
                   lure_values=[ev["value"] for ev in seq if ev["obj"] == "L"])
    after = [j for j in range(len(seq)) if turn[j] > turn[gold_j]] if corrected else []
    conv.meta.update(unit="mean", k=len(a_idx) - 1, latest_form=seq[gold_j]["form"],
                     b_place=place if cell in ("U-diff", "U-same") else None,
                     a_pos=place if cell == "C_noupd" else None, layout=place if cell == "C_twoslot" else None,
                     t_after=len(after) if corrected else None, vtype=vtype, objA=objA[0], objB=objB[0],
                     len_rank=len_rank, anchor=(cell == "C_noupd" and form == "anchor"), m_b=sum(ev["obj"] == "B" and ev["role"] ==
                                                                             "corr" for ev in seq))
    conv.fill()
    return conv.record()
