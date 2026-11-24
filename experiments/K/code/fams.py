"""SPEC 4 families R, U, B, F, A, P. Each builder returns (units, asked key, extra) or None (the caller redraws).

Entity slot 0 is always the asked entity; names are assigned to slots at random by the caller, so slot 0 carries no
position. A unit is a list of statements that stays contiguous (an antecedent and its ellipsis). The order of units
is drawn at random and kept only if it meets the item's design flags (intro rank, follow, latest-statement role).
K3 round 5 (REVIEW 6a): B takes the entity at the flagged rank of a uniform order. K3 round 6 (REVIEW 7a S-1): R,
U and P build every key by one law and draw the asked key uniformly over the item's keys, the order from sym_order
(intro ranks independent of the structure), so given the item the asked key is any of its keys alike: a rule blind
to the question's name sees the asked attribute's holders as alike, one blind to its attribute the asked entity's
keys; only k and the intro ranks (and U's count pair) are block flags, each stratifying a law it does not tilt.
"""
from __future__ import annotations

from items import St, follows, intro_rank


def _form(r) -> str:
    return "full" if r.random() < 0.5 else "reord"


def _attrs(g, r, n, must=None, exclude=None) -> list[int]:
    pool = [a for a in g.attrs.tolist() if a != exclude and a != must]
    out = [int(a) for a in r.choice(pool, n - (must is not None), replace=False)]
    return ([must] if must is not None else []) + out


def arrange(units, ok, r, tries=300):
    for _ in range(tries):
        order = [units[i] for i in r.permutation(len(units))]
        if ok([s for u in order for s in u]):
            return order
    return None


def _rank_follow(fl, astar):
    def ok(flat):
        return intro_rank(flat, 0) == fl["rank"] and follows(flat, 0, astar) == bool(fl["follow"])
    return ok


def holder_ranks(flat, astar) -> tuple[tuple, list]:
    """R and P (K3 round 5): the asked attribute's holders in the order they state it (one statement each), and their
    intro ranks in that order."""
    seen, order = [], []
    for s in flat:
        if s.kind in ("full", "reord", "def") and s.ent not in seen:
            seen.append(s.ent)
        if s.attr == astar and s.kind != "def":
            order.append(s.ent)
    return tuple(seen.index(e) for e in order), order


def _relabel(order, e: int):
    """Swap entity labels e and 0 (slot 0 is the asked entity; names go to slots at random, so labels carry nothing)."""
    if e == 0:
        return order
    sw = {0: e, e: 0}
    return [[St(s.kind, sw.get(s.ent, s.ent), s.attr, s.val, s.alias) for s in u] for u in order]


def sigma_with(k: int, fixed: dict, r) -> list:
    """A uniform intro order (entity per rank) of k entities, conditioned on fixed {entity: rank}."""
    sig = [None] * k
    for e, rk in fixed.items():
        sig[rk] = e
    free_e = [e for e in range(k) if e not in fixed]
    free_r = [i for i in range(k) if sig[i] is None]
    for i, x in zip(free_r, r.permutation(len(free_e))):
        sig[i] = free_e[int(x)]
    return sig


def sym_order(units_of: dict, sigma: list, r) -> list:
    """K3 round 6 (REVIEW 7a S-1): R's, U's and P's order of units. Entities are introduced in the order sigma (drawn
    by the caller, independent of the structure), each by one of its units drawn uniformly; every other unit goes to
    a uniform gap after its entity's intro unit (the units taken in a random sequence). An entity's intro rank is so
    independent of its size and keys, so a block's exact (k, rank[, frank]) cells stratify the natural law and give
    no posterior cue, and nothing in the order depends on which of an entity's keys, or which holder, is asked."""
    seq, rest, intro = [], [], {}
    for e in sigma:
        us = units_of[e]
        i = int(r.integers(len(us)))
        intro[e] = us[i]
        seq.append(us[i])
        rest += [(e, u) for j, u in enumerate(us) if j != i]
    for x in r.permutation(len(rest)):
        e, u = rest[int(x)]
        at = next(i for i, v in enumerate(seq) if v is intro[e])
        seq.insert(int(r.integers(at + 1, len(seq) + 1)), u)
    return seq


R_Q, R_N = 3, (2, 3)    # R, P (K3 round 6): the attributes of the main component; each entity states 2-3 of them


def fam_R(g, fl, r, k=None, by=False):
    """K3 round 6 (REVIEW 7a S-1). Round 5 built the asked attribute's holders by the follow flag (2 or more in 80%
    of items) while the asked entity's other attributes got the random draw's holders, mostly fewer, so a tracker
    that reads the question's name and never its attribute picked the asked key by its holder count (TA-SHARE
    0.734-0.761, a fitted model 0.746-0.818, against T's 0.65). Now one law builds every key: the main component's
    km entities each state 2-3 of R_Q attributes (each key once, a uniform subset), the asked key is a uniform draw
    over all km x 2-3 keys (so the asked entity is drawn by its size and the asked attribute by its holders), and
    the order is sym_order's with the asked entity at the block's intro rank. Given the item every key is equally
    likely the asked one: a rule blind to the question's name sees the asked attribute's m holders as alike (best
    1/m: O2 and R-POST read E[1/m], about 0.43), a rule blind to its attribute sees the asked entity's n keys as
    alike (best 1/n: T and TA-* about 0.40). No flag but k and rank is conditioned on (follow and the holder count
    are natural). k (OOD ENTS, PLONG) overrides the block's entity count and rank. by (P): one more entity, a
    bystander stating 1-3 attributes no other entity states (its own component: P's earlier questions ask it, so
    no earlier answer touches the main component, skill._questions)."""
    if k is not None and "k_ood" not in fl:     # OOD ENTS and PLONG: their own entity count and a rank override the
        fl.update(k_ood=k + by, k=k + by, rank=int(r.integers(0, k + by)))     # block flags (kept over redraws)
    k = fl.setdefault("k", int(r.integers(2, 5)) + by)                         # (A's R contexts: per item)
    rank = fl.setdefault("rank", int(r.integers(0, k)))
    km = k - by
    attrs = [int(a) for a in r.choice(g.attrs, R_Q, replace=False)]
    keys = {e: [attrs[int(i)] for i in r.choice(R_Q, int(r.integers(R_N[0], R_N[1] + 1)), replace=False)]
            for e in range(km)}
    if by:
        other = [int(a) for a in g.attrs.tolist() if int(a) not in attrs]
        keys[km] = [int(a) for a in r.choice(other, int(r.integers(1, 4)), replace=False)]
    edges = [(e, a) for e in range(km) for a in keys[e]]
    e0, astar = edges[int(r.integers(len(edges)))]
    vals = iter(g.values(r, sum(len(v) for v in keys.values())))
    units = {e: [[St(_form(r), e, a, next(vals))] for a in ks] for e, ks in keys.items()}
    order = sym_order(units, sigma_with(k, {e0: rank}, r), r)
    return _relabel(order, e0), (0, astar), {"k": k}


U_TYPES = (("one2", .5), ("sym", .25), ("two", .25))     # U: an attribute's two holders' statement counts (round 6)
U_Q = {2: (2, 3), 3: (3, 4), 4: (4, 4)}                  # U: attributes (pairs) per entity count k


def _pair_counts(t: str, r) -> tuple:
    if t == "one2":
        return (1, 2)
    if t == "sym":
        return tuple(int(x) for x in r.choice([2, 3, 4], 2, replace=False))
    m = int(r.integers(2, 4))
    return (m, m)


def fam_U(g, fl, r, kmany=False):
    """K3 round 6 (REVIEW 7a S-1). Round 5 restated the asked key 2-4 times and gave it a foil while the asked entity's
    other keys were stated 1-2 times by it alone, so "the asked entity's key with the most statements in the item"
    (TA-CNT) read 0.69-0.75 and a fitted attribute-blind model 0.87-0.94. Now every attribute of a U item is held by
    exactly two entities (a pair, uniform over the k (k - 1) / 2 pairs, every entity in one or more) with a count
    pair of one law (U_TYPES: (1, 2) 1/2, two of 2-4 apart 1/4, (m, m) m 2-3 1/4, sides at random); the asked
    attribute is one of them with its type and side from the block (noupd: the single statement of a (1, 2) pair,
    upd "one" its double, "sym", twoslot), every other attribute's drawn from the law, so the asked key is a uniform
    draw over the item's keys: blind to the question's attribute the asked entity's keys are alike (TA about 0.45),
    blind to its name the two holders are (1/2). The order is sym_order's with the key's and the foil's intro ranks
    the block's (rank, frank); recency is natural (follow is no flag: conditioning it with the ranks would tilt the
    rank cells). kmany (OOD): the key stated 5-6 times."""
    var, sub = fl["var"], fl.get("sub", "sym")
    k = fl.setdefault("k", int(r.integers(2, 5)))
    rank = fl.setdefault("rank", int(r.integers(0, k)))
    fr = fl.get("frank")            # the block's foil rank (a stale one, k and rank drawn per item, is redrawn)
    frank = fl["frank"] = fr if fr is not None and fr < k and fr != rank else int(r.choice([x for x in range(k)
                                                                                           if x != rank]))
    q = int(r.integers(U_Q[k][0], U_Q[k][1] + 1))
    attrs = [int(a) for a in r.choice(g.attrs, q, replace=False)]
    allp = [(a, b) for a in range(k) for b in range(a + 1, k)]
    while True:
        pairs = [allp[int(r.integers(len(allp)))] for _ in range(q)]
        if len({e for p in pairs for e in p}) == k:
            break
    c0 = (1, 2) if var == "noupd" else (2, 1) if sub == "one" else _pair_counts("two" if var == "twoslot" else
                                                                              "sym", r)
    c0 = (int(r.integers(5, 7)), c0[1]) if kmany else c0
    side = int(r.integers(2))
    e0, e1 = pairs[0][side], pairs[0][1 - side]
    counts = {(e0, attrs[0]): c0[0], (e1, attrs[0]): c0[1]}
    for a, (x, y) in zip(attrs[1:], pairs[1:]):
        c = _pair_counts(str(r.choice([t for t, _ in U_TYPES], p=[p for _, p in U_TYPES])), r)
        counts[(x, a)], counts[(y, a)] = c if r.random() < 0.5 else c[::-1]
    vals = iter(g.values(r, sum(counts.values())))
    units = {e: [] for e in range(k)}
    for (e, a), n in counts.items():
        units[e] += [[St(_form(r), e, a, next(vals))] for _ in range(n)]
    order = sym_order(units, sigma_with(k, {e0: rank, e1: frank}, r), r)
    return _relabel(order, e0), (0, attrs[0]), {"k": k, "var": var, "sub": sub}


def fam_B(g, fl, r):
    """Every entity states the asked attribute [C2]. Names come from one cube (skill.Gen.cube: {a, a'} x {b, b'} x
    {c, c'}): struct items (3/4) use all 8, so every proper subset of every name's positions is held by another name
    and only the whole name is a key; the other items use 2-4 of them. The asked entity (slot 0) is uniform among
    the item's names. K3 round 2 (REVIEW 3a B-CENTER): with 4 names and every subset of the asked name held by a
    foil, the asked name was the unique centre (2 tokens shared with each foil, the foils 1 with each other), so
    "the latest value of the most central name" read 0.92 on B without reading the question's name. K3 round 5
    (REVIEW 6a S-2): the order is uniform and the asked entity is the one at the block's intro rank (every entity is
    a holder, so one is always there). Built, the order was drawn until the asked entity had that rank whatever its
    statement count, while a name with 2 statements is usually introduced early: a partial key with the tie broken
    toward the name whose rank is least expected for its count (B-SURPRISE) read 0.676-0.681 on training items."""
    cube = g.cube(r)
    if cube is None:
        return None, None, None
    k = fl.setdefault("k", 8 if fl["struct"] else int(r.integers(2, 5)))      # flags: skill.block_flags (BD-22)
    names = [cube[int(i)] for i in r.permutation(8)[:k]]
    rank = fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    keys = [(e, astar) for e in range(k)]
    for e in range(k):
        if r.random() < 0.5:
            keys.append((e, _attrs(g, r, 1, exclude=astar)[0]))
    vals = g.values(r, len(keys))
    units = [[St(_form(r), e, a, v)] for (e, a), v in zip(keys, vals)]
    order = [units[i] for i in r.permutation(len(units))]           # uniform (BD-1), nothing conditioned on
    seen = []
    for s in (s for u in order for s in u):
        if s.ent not in seen:
            seen.append(s.ent)
    return _relabel(order, seen[rank]), (0, astar), {"k": k, "names": names}


LAST_KEEP = 0.35    # F: share of arrangements kept when the item's last statement is the key's latest or the foil's
                    # reference statement (BD-12, BD-16; 0.5 until K3 round 2, BD-21 pushed references later)


def _ref_units(e, attrs, ref, al_e, late, stale, vals, r):
    """F: entity e states attrs[0] once in reference form (`ref`: an alias after its definition, or an ellipsis right
    after a named statement of attrs[1]); `late` adds a later named statement of attrs[0] (the latest: "full after"),
    `stale` an earlier named one. -> (units, reference statement, latest statement of attrs[0], stale statement)."""
    rest, units = list(attrs[1:]), []
    if ref == "alias":
        u = St("alias", e, attrs[0], next(vals), al_e)
        units += [[St("def", e, alias=al_e)], [u]]
    else:
        u = St("ell", e, attrs[0], next(vals))
        units.append([St(_form(r), e, rest.pop(0), next(vals)), u])
    last = St(_form(r), e, attrs[0], next(vals)) if late else u
    old = St(_form(r), e, attrs[0], next(vals)) if stale else None
    units += [[x] for x in (last, old) if x is not None and x is not u]
    return units + [[St(_form(r), e, a, next(vals))] for a in rest], u, last, old


def _defs_first(order):
    """Swap each alias unit with its definition's unit when the definition comes later (2-to-1 per alias onto the
    orders that meet the rule, so the kept orders stay uniform before ok() filters them)."""
    at = {u[0].alias: i for i, u in enumerate(order) if u[0].kind == "def"}
    for i, u in enumerate(order):
        for s in u:
            if s.kind == "alias" and at[s.alias] > i:
                j = at[s.alias]
                order[i], order[j] = order[j], order[i]
                at[s.alias] = i
    return order


def fam_F(g, fl, r):
    """Every entity states one attribute in a reference form (alias or ellipsis, 1/2 each); the asked entity and one
    foil state the asked attribute that way (review R-1, 2026-09-27: when only the asked entity used ellipsis, "the
    latest statement of the attribute that names no other entity" scored 0.908). Flags: kref = fref (the reference
    form of the key and the foil: one kind per item, alias in 1/2, BD-14: with mixed kinds, resolving one kind named
    the key by elimination in every mixed item, SURFe and SURFa 0.90), klate (the key's latest is a named statement
    after its reference: 1/4), follow (the foil's reference statement comes after the key's latest: 1/2), kstale,
    fstale (a named statement of the asked attribute before the reference one: 1/2 of the foils and of the keys that
    are not late). The foil is never restated after its reference and every stale statement precedes both reference
    statements, so the forms of the asked attribute's statements say little about which reference statement is the
    key's (SURF, purity.surf: design 0.625, "O4 on the late quarter, a coin elsewhere"). Orders whose last statement
    is the key's latest or the foil's reference statement are kept at LAST_KEEP (BD-12: without it F's last-statement
    rate read 0.27 against 0.20; BD-16, K3 R-3: thinning only the key's left LASTSKIP at 0.69). Both reference
    statements, an ellipsis with its antecedent, come after both entities' first explicit mention, definitions
    included (BD-21, K3 round 2: a reference before the asked name's first mention could only be the foil's, and an
    alias before the asked name's definition too, so F-INTRO read 0.65 and F-DEFX 0.68 without resolving anything;
    with only the reference statements after them, an ellipsis whose antecedent was its entity's first mention tied
    the order of the references to the asked name's intro rank, 0.54 on ellipsis items with it). K3 round 8 (REVIEW
    9a G-1, BD-48): on ellipsis items the key's and the foil's antecedents state one attribute (`ante`, drawn per
    item). Each entity states each other attribute once, so with antecedent attributes drawn apart the key never
    restated its own antecedent's attribute while the foil's was one the key states in 0.20-0.40 of items, and a rule
    reading only that attribute's frequency and the asked name's attribute count (F-ANTESIZE) read F 0.636."""
    k = fl.setdefault("k", int(r.integers(2, 5)))
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    al = [int(x) for x in r.choice(g.alias, k, replace=False)]
    foil = int(r.integers(1, k))
    vals = iter(g.values(r, 40))
    ante = _attrs(g, r, 1, exclude=astar)[0] if fl["kref"] == "ell" else None     # K3 round 8 (BD-48)
    units, keyst = [], {}
    for e in range(k):
        n_other = int(r.integers(1, 3))
        if e in (0, foil):
            attrs = _attrs(g, r, 1 + n_other, must=astar) if ante is None else [astar, ante] + [
                int(a) for a in r.choice([a for a in g.attrs.tolist() if a not in (astar, ante)], n_other - 1,
                                         replace=False)]
            ref = fl["kref"] if e == 0 else fl["fref"]
            late = e == 0 and fl["klate"]
            stale = fl["kstale"] if e == 0 else fl["fstale"]       # flags, so redraws cannot bias them
            us, u, last, old = _ref_units(e, attrs, ref, al[e], late, stale, vals, r)
            keyst[e] = (u, last, old)
        else:
            attrs = _attrs(g, r, 1 + n_other, exclude=astar)
            us = _ref_units(e, attrs, "alias" if r.random() < 0.5 else "ell", al[e], False, False, vals, r)[0]
        units += us
    base = _rank_follow(fl, astar)
    (ku, kl, ko), (fu, _, fo) = keyst[0], keyst[foil]

    def ok(flat):
        if not base(flat):
            return False
        keyed = [s for s in flat if s.ent == 0 and s.attr == astar]
        foiled = [s for s in flat if s.ent == foil and s.attr == astar]
        if keyed[-1] is not kl or foiled[-1] is not fu:
            return False
        if (flat[-1] is kl or flat[-1] is fu) and r.random() >= LAST_KEEP:    # SPEC 4's last-statement rate at
            return False                            # chance, thinned for the foil too (K3 R-3: LASTSKIP 0.69)
        first_ref = min(i for i, s in enumerate(flat) if s is ku or s is fu)
        intro = max(min(i for i, s in enumerate(flat) if s.ent == e and s.kind in ("full", "reord", "def"))
                    for e in (0, foil))
        defs = [i for i, s in enumerate(flat) if s.kind == "def" and s.ent in (0, foil)]
        if first_ref - (flat[first_ref].kind == "ell") <= max([intro] + defs):     # BD-21: both reference units
            return False                                                            # after both first mentions
        return all(i < first_ref for i, s in enumerate(flat) if s is ko or s is fo)

    for _ in range(600):
        order = _defs_first([units[i] for i in r.permutation(len(units))])
        if ok([s for u in order for s in u]):
            break
    else:
        order = None
    return order, (0, astar), {"k": k, "case": "full_after" if fl["klate"] else fl["kref"], "foil": foil}


def fam_A(g, fl, r):
    """Lures: "both" (the asked entity is stated with other attributes and other entities with the asked one), "ent"
    (only the entity), "attr" (only the attribute). A both-item is the context of an R item (3/8 since K3 round 7,
    skill.A_CTX) or of a U item (5/8, U's variants, no lure question) whose question names an entity of it that does
    not state the asked attribute, so no rule blind to the question's name can tell it from the R or U items, which
    outnumber it in every context (K3, 2026-09-27). With its own structure (one other entity stating the attribute)
    "the attribute stated by exactly one entity" was more often a both-item than an R item and the fitted cue model
    read 0.98 on A; built from R contexts alone it read 0.55 (the " none" prior beat three split values). The context
    kind is a block flag (K3 round 4: drawn per item, R contexts, which fail more, were redrawn into U contexts). K3
    round 6: with R's one-law design a both-item was 1/4 of the R-like items with a free entity and 3 holders, a tie
    with each holder's value that the gate broke to " none" (A 0.548-0.567 on fresh sets); R contexts went to 1/4.
    K3 round 7 (REVIEW 8a S-1). In a U context the asked entity was a non-holder of the context's asked attribute
    drawn uniformly, which among 3 names is one entity, mostly a small one, while a U item asks an entity drawn by its
    size, so a reader of the asked name's side said " none" where the asked name states one attribute among 3 names
    (A-ONEKEY3 0.56-0.57 on A with a mix gain; cuegate.ta_a_run, the TA side with " none", 0.55-0.58). U contexts:
    the asked entity is drawn uniformly among the context's k entities, the draw (context and all) repeated while it
    states every attribute the item states, the asked attribute one of those it lacks (uniform); within a cell of
    (k, the item's attributes) A's share against U's asked entity (drawn by its n keys) falls as 1/n, under a pick
    among its n keys and, where n is 1, under O4. The size law itself (the context's own asked entity, an attribute
    it lacks) held A at 0.35 of U's share in every cell where the entity lacks one, a tie with a pick among 3 keys
    (ta_a_run 0.523). R contexts keep round 6's draw: there every name lacking an attribute lacks exactly one and
    has 2 keys, so the two draws differ on the asked name's side only through the holders of that attribute (cell
    tables at most 0.72 of a key pick), while the uniform draw favoured asked attributes with few holders (R-like
    cells of the name-blind side where A beat a holder's value). At A_CTX's 1/4 R both draws left a tie on the TA
    side (ta_a_run 0.523 for the size law; 0.508, 0.510 and 0.523 for the uniform draw on three samples of
    57,344-64,512 items); R 3/8 with the uniform draw in R too failed the name-blind side (K-EVAL-ID 0.617). K3 ROUND
    7 has every reading. meta "cattr": the context's asked attribute (its d range, skill._in_range)."""
    lure = fl["lure"]
    astar = int(r.choice(g.attrs))
    if lure == "both":
        if (fl.get("ctx") or ("R" if r.random() < 0.375 else "U")) == "R":    # a block flag (skill.block_flags)
            c = "R"                                     # K3 round 6: R's k and rank as R items' (fam_R draws them)
            order, (_, a0), meta = fam_R(g, {}, r)
        else:
            c = str(r.choice(["one", "sym", "noupd", "twoslot"], p=[.25, .25, .25, .25]))
            kk = int(r.integers(2, 5))                  # K3 round 4: k and the (key, foil) rank pair as U items'
            rk, fk = (int(x) for x in r.choice(kk, 2, replace=False))
            fu = {"var": "upd" if c in ("one", "sym") else c, "sub": c, "k": kk, "rank": rk, "frank": fk}
            order, (_, a0), meta = fam_U(g, fu, r)
            c = "U" + c
        if order is None:
            return None, None, None
        if c == "R":            # R: the context's attribute, a non-holder of it uniformly (K3 round 6's draw)
            free = [e for e in range(meta["k"]) if not any(s.ent == e and s.attr == a0 for u in order for s in u)]
            if not free:
                return None, None, None
            return order, (int(r.choice(free)), a0), {"k": meta["k"], "lure": lure, "rctx": c, "cattr": a0}
        e = int(r.integers(meta["k"]))
        miss = sorted({s.attr for u in order for s in u if s.kind != "def"} - {s.attr for u in order for s in u
                                                                              if s.ent == e})
        if not miss:
            return None, None, None
        return order, (e, int(r.choice(miss))), {"k": meta["k"], "lure": lure, "rctx": c, "cattr": a0}
    k = fl.setdefault("k", int(r.integers(1, 5)))
    keys = []
    if lure != "attr":
        keys += [(0, a) for a in _attrs(g, r, int(r.integers(1, 4)), exclude=astar)]
    for e in range(1, max(k, 2) if lure == "attr" else k):
        keys += [(e, a) for a in _attrs(g, r, int(r.integers(1, 4)), must=astar if (e == 1 and lure != "ent")
                                        else None, exclude=astar if (lure == "ent" or e != 1) else None)]
    vals = g.values(r, len(keys))
    units = [[St(_form(r), e, a, v)] for (e, a), v in zip(keys, vals)]
    fl.setdefault("rank", int(r.integers(0, k)) if lure != "attr" else -1)

    def ok(flat):
        return lure == "attr" or intro_rank(flat, 0) == fl["rank"]
    return arrange(units, ok, r), (0, astar), {"k": max(k, 2) if lure == "attr" else k, "lure": lure}


def fam_P(g, fl, r, n_q=None):
    """R's statement design plus a system marker and one bystander entity of its own component (fam_R by), whose keys
    the earlier questions ask: 3-5 entities (PLONG too). K3 round 6: R's design gives every item a foil with the
    holders' natural share (no foil flag), and an exhausted builder returns None here instead of indexing its meta
    (REVIEW 7b O-1: round 5's fam_R returned (None, None, None) and this line raised TypeError)."""
    order, key, meta = fam_R(g, fl, r, k=int(r.integers(2, 5)) if n_q is not None else None, by=True)
    if order is None:
        return None, None, None
    meta["marker"] = int(r.choice(g.marker))
    return order, key, meta
