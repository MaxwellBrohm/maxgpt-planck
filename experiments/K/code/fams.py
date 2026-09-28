"""SPEC 4 families R, U, B, F, A, P. Each builder returns (units, asked key, extra) or None (the caller redraws).

Entity slot 0 is always the asked entity; names are assigned to slots at random by the caller, so slot 0 carries no
position. A unit is a list of statements that stays contiguous (an antecedent and its ellipsis). The order of units
is drawn at random and kept only if it meets the item's design flags (intro rank, follow, latest-statement role).
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


def fam_R(g, fl, r, k=None, by=False):
    """by (P, K3 round 4): one more entity, the last slot, a bystander (1-3 attributes, never the asked one), so a P
    item always has a bystander for its earlier questions to ask (SPEC 4) while its holders keep R's design, but for
    P's foil in every item (skill.block_flags). P drawn until it had one kept fewer holders: its cheap rules read
    0.01-0.04 over round 3's P and the gate P 0.675."""
    if k is not None and "k_ood" not in fl:     # OOD ENTS and PLONG: their own entity count and a rank override the
        fl.update(k_ood=k + by, k=k + by, rank=int(r.integers(0, k + by)))     # block flags (kept over redraws)
    k = fl.setdefault("k", int(r.integers(2 if fl["foil"] else 1, 5)) + by)     # (A's R contexts: per item)
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    keys = [(0, a) for a in _attrs(g, r, int(r.integers(1, 4)), must=astar)]
    foil_slot = int(r.integers(1, k - by)) if fl["foil"] else -1
    for e in range(1, k):
        keys += [(e, a) for a in _attrs(g, r, int(r.integers(1, 4)), must=astar if e == foil_slot else None,
                                        exclude=None if fl["foil"] and not (by and e == k - 1) else astar)]
    vals = g.values(r, len(keys))
    units = [[St(_form(r), e, a, v)] for (e, a), v in zip(keys, vals)]
    return arrange(units, _rank_follow(fl, astar), r), (0, astar), {"k": k}


def fam_U(g, fl, r, kmany=False):
    """The asked key and the foil (entity 1's statements of the asked attribute) are exchangeable in everything a
    rule can read without the question's name (K3, 2026-09-27: the fitted cue model read 0.87 on U when the key was
    the more repeated key and its entity the bigger one). Both entities state 2-3 attributes, the asked one
    included; every other key is stated 1-2 times; the (key, foil) count pair with its recency is mirrored: noupd
    (1, 2, the foil's latest after the key's) by upd "one" (2, 1, the key's latest after the foil's), and upd "sym"
    (a != b in 2-4) and twoslot (m, m, m in 2-3) take follow in 1/2 (skill.block_flags). K3 round 4 (REVIEW 5a S-2):
    the foil's intro rank is a block flag too (frank; the (key, foil) rank pair uniform, skill.rank_pairs): with the
    key's rank alone exact, the foil (a big entity) was introduced early, and the holder introduced last was the key
    in 0.541 of non-lure items."""
    var, sub = fl["var"], fl.get("sub", "sym")
    k = fl.setdefault("k", int(r.integers(2, 5)))
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    counts = {}
    for e in range(k):
        n_attr = int(r.integers(2, 4)) if e < 2 else int(r.integers(1, 4))
        for a in _attrs(g, r, n_attr, must=astar if e < 2 else None, exclude=None if e < 2 else astar):
            counts[(e, a)] = int(r.integers(1, 3))
    if var == "noupd":
        pair = (1, 2)
    elif var == "twoslot":
        pair = (int(r.integers(2, 4)),) * 2
    elif sub == "one":
        pair = (2, 1)
    else:
        pair = tuple(int(x) for x in r.choice([2, 3, 4], 2, replace=False))
    counts[(0, astar)], counts[(1, astar)] = (int(r.integers(5, 7)), pair[1]) if kmany else pair
    keys = [key for key, n in counts.items() for _ in range(n)]
    vals = g.values(r, len(keys))
    units = [[St(_form(r), e, a, v)] for (e, a), v in zip(keys, vals)]
    base, fr = _rank_follow(fl, astar), fl.get("frank")       # K3 round 4 (REVIEW 5a S-2): the foil's intro rank
    ok = base if fr is None else (lambda flat: base(flat) and intro_rank(flat, 1) == fr)   # is a block flag too
    return arrange(units, ok, r, tries=1000), (0, astar), {"k": k, "var": var, "sub": sub}


def fam_B(g, fl, r):
    """Every entity states the asked attribute [C2]. Names come from one cube (skill.Gen.cube: {a, a'} x {b, b'} x
    {c, c'}): struct items (3/4) use all 8, so every proper subset of every name's positions is held by another name
    and only the whole name is a key; the other items use 2-4 of them. The asked entity (slot 0) is uniform among
    the item's names. K3 round 2 (REVIEW 3a B-CENTER): with 4 names and every subset of the asked name held by a
    foil, the asked name was the unique centre (2 tokens shared with each foil, the foils 1 with each other), so
    "the latest value of the most central name" read 0.92 on B without reading the question's name."""
    cube = g.cube(r)
    if cube is None:
        return None, None, None
    k = fl.setdefault("k", 8 if fl["struct"] else int(r.integers(2, 5)))      # flags: skill.block_flags (BD-22)
    names = [cube[int(i)] for i in r.permutation(8)[:k]]
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    keys = [(e, astar) for e in range(k)]
    for e in range(k):
        if r.random() < 0.5:
            keys.append((e, _attrs(g, r, 1, exclude=astar)[0]))
    vals = g.values(r, len(keys))
    units = [[St(_form(r), e, a, v)] for (e, a), v in zip(keys, vals)]
    ok = lambda flat: intro_rank(flat, 0) == fl["rank"]        # noqa: E731  (uniform order: BD-1)
    return arrange(units, ok, r), (0, astar), {"k": k, "names": names}


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
    the order of the references to the asked name's intro rank, 0.54 on ellipsis items with it)."""
    k = fl.setdefault("k", int(r.integers(2, 5)))
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    al = [int(x) for x in r.choice(g.alias, k, replace=False)]
    foil = int(r.integers(1, k))
    vals = iter(g.values(r, 40))
    units, keyst = [], {}
    for e in range(k):
        n_other = int(r.integers(1, 3))
        if e in (0, foil):
            attrs = _attrs(g, r, 1 + n_other, must=astar)
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
    (only the entity), "attr" (only the attribute). A both-item is the context of an R item (1/2, R's foil and follow
    shares) or of a U item (1/2, U's variants, no lure question) whose question names an entity of it that does not
    state the asked attribute, so no rule blind to the question's name can tell it from the R or U items, which
    outnumber it in every context (K3, 2026-09-27). With its own structure (one other entity stating the attribute)
    "the attribute stated by exactly one entity" was more often a both-item than an R item and the fitted cue model
    read 0.98 on A; built from R contexts alone it read 0.55 (the " none" prior beat three split values). The context
    kind is a block flag (K3 round 4: drawn per item, R contexts, which fail more, were redrawn into U contexts)."""
    lure = fl["lure"]
    astar = int(r.choice(g.attrs))
    if lure == "both":
        if (fl.get("ctx") or ("R" if r.random() < 0.5 else "U")) == "R":     # a block flag (skill.block_flags)
            c = str(r.choice(["f1", "f0", "nofoil"], p=[.5, .3, .2]))
            order, (_, astar), meta = fam_R(g, {"foil": c != "nofoil", "follow": c == "f1"}, r)
        else:
            c = str(r.choice(["one", "sym", "noupd", "twoslot"], p=[.25, .25, .25, .25]))
            kk = int(r.integers(2, 5))                  # K3 round 4: k and the (key, foil) rank pair as U items'
            rk, fk = (int(x) for x in r.choice(kk, 2, replace=False))
            fu = {"var": "upd" if c in ("one", "sym") else c, "sub": c, "k": kk, "rank": rk, "frank": fk,
                  "follow": c == "noupd" or (c in ("sym", "twoslot") and r.random() < 0.5)}
            order, (_, astar), meta = fam_U(g, fu, r)
            c = "U" + c
        if order is None:
            return None, None, None
        hold = {s.ent for u in order for s in u if s.attr == astar}
        free = [e for e in range(meta["k"]) if e not in hold]
        if not free:
            return None, None, None
        return order, (int(r.choice(free)), astar), {"k": meta["k"], "lure": lure, "rctx": c}
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
    """R's statement design (foil and follow flags) plus a system marker; PLONG uses 2-4 entities. K3 round 4: plus one
    bystander entity (fam_R by), whose keys the earlier questions ask, and a foil in every item (skill.block_flags):
    3-5 entities (PLONG 3-5)."""
    order, key, meta = fam_R(g, fl, r, k=int(r.integers(2, 5)) if n_q is not None else None, by=True)
    meta["marker"] = int(r.choice(g.marker))
    return order, key, meta
