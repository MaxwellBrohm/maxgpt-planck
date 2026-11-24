"""Round 5's R, U and P designs (fams.py and skill.py as of K3 round 5), kept to show the round-6 checks catch them
(REVIEW 7a S-1: the asked key built unlike the asked entity's other keys). use(monkeypatch) installs them: the round-5
builders, their block flags (R's foil and follow, U's follow) and the round-5 question rule (no component rule).
Also A's round-6 both-item and the size law (K3 round 7), F's round-7 antecedents (fam_F_r7, K3 round 8) and
the DFAR pad of K3 rounds 4 to 13 (pad_r13, before G-2)."""
from __future__ import annotations

import fams as Fm
import skill as S
from items import St

R_HOLD, R_TRIES = 1 / 3, (60, 2000)


def fam_R(g, fl, r, k=None, by=False):
    """fams.fam_R of K3 round 5: m holders by the foil flag (2 + Binomial(k - 2, 1/3)), the key the holder the follow
    flag picks, the holders' rank tuple uniform; the asked entity's other attributes drawn at random (1-3 of 6)."""
    if k is not None and "k_ood" not in fl:
        fl.update(k_ood=k + by, k=k + by, rank=int(r.integers(0, k + by)))
    k = fl.setdefault("k", int(r.integers(2 if fl["foil"] else 1, 5)) + by)
    rank = fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    m = 2 + int(r.binomial(k - by - 2, R_HOLD)) if fl["foil"] else 1
    j = int(r.integers(0, m - 1)) if fl["follow"] else m - 1
    rest = [int(x) for x in r.choice([x for x in range(k) if x != rank], m - 1, replace=False)]
    t = tuple(rest[:j] + [rank] + rest[j:])
    for _ in range(R_TRIES[0]):
        keys = []
        for e in range(k):
            keys += [(e, a) for a in Fm._attrs(g, r, int(r.integers(1, 4)), must=astar if e < m else None,
                                               exclude=None if e < m else astar)]
        units = [[St(Fm._form(r), e, a, v)] for (e, a), v in zip(keys, g.values(r, len(keys)))]
        for _ in range(R_TRIES[1]):
            order = [units[i] for i in r.permutation(len(units))]
            got, ents = Fm.holder_ranks([s for u in order for s in u], astar)
            if got == t:
                return Fm._relabel(order, ents[j]), (0, astar), {"k": k}
    return None, None, None


def fam_U(g, fl, r, kmany=False):
    """fams.fam_U of K3 round 5: the asked key stated 2-4 times (upd), its foil per the variant, both entities 2-3
    attributes, every other key 1-2 times; the order drawn until the block's rank, frank and follow hold."""
    var, sub = fl["var"], fl.get("sub", "sym")
    k = fl.setdefault("k", int(r.integers(2, 5)))
    fl.setdefault("rank", int(r.integers(0, k)))
    astar = int(r.choice(g.attrs))
    counts = {}
    for e in range(k):
        n_attr = int(r.integers(2, 4)) if e < 2 else int(r.integers(1, 4))
        for a in Fm._attrs(g, r, n_attr, must=astar if e < 2 else None, exclude=None if e < 2 else astar):
            counts[(e, a)] = int(r.integers(1, 3))
    pair = ((1, 2) if var == "noupd" else (int(r.integers(2, 4)),) * 2 if var == "twoslot" else (2, 1) if sub == "one"
            else tuple(int(x) for x in r.choice([2, 3, 4], 2, replace=False)))
    counts[(0, astar)], counts[(1, astar)] = (int(r.integers(5, 7)), pair[1]) if kmany else pair
    keys = [key for key, n in counts.items() for _ in range(n)]
    units = [[St(Fm._form(r), e, a, v)] for (e, a), v in zip(keys, g.values(r, len(keys)))]
    base, fr = Fm._rank_follow(fl, astar), fl.get("frank")
    ok = base if fr is None else (lambda flat: base(flat) and Fm.intro_rank(flat, 1) == fr)
    return Fm.arrange(units, ok, r, tries=1000), (0, astar), {"k": k, "var": var, "sub": sub}


def fam_P(g, fl, r, n_q=None):
    order, key, meta = fam_R(g, fl, r, k=int(r.integers(2, 5)) if n_q is not None else None, by=True)
    if order is None:
        return None, None, None
    meta["marker"] = int(r.choice(g.marker))
    return order, key, meta


def block_flags(fam, n, r, ncube=None, _new=S.block_flags):
    """skill.block_flags of K3 round 5 for R, P (foil and follow) and U (follow); the others as now."""
    if fam in ("R", "P"):
        fl = [{"qform": q} for q in S.balanced(n, [(0, .5), (1, .5)], r)]
        cs = S.balanced(n, [("f1", .5), ("f0", .3), ("nofoil", .2)] if fam == "R" else [("f1", .5), ("f0", .5)], r)
        ks = [0] * n
        for foil, opts in ((True, (2, 3, 4)), (False, (1, 2, 3, 4))):
            idx = [i for i, c in enumerate(cs) if (c != "nofoil") == foil]
            for i, k in zip(idx, S.balanced(len(idx), [(k, 1 / len(opts)) for k in opts], r)):
                ks[i] = k + (fam == "P")
        for f, c, k, x in zip(fl, cs, ks, S.ranks(ks, r)):
            f.update(follow=c == "f1", foil=c != "nofoil", k=k, rank=x)
        return fl
    fl = _new(fam, n, r, ncube)
    if fam == "U":
        free = [i for i, f in enumerate(fl) if f["var"] == "twoslot" or f["sub"] == "sym"]
        fol = dict(zip(free, S.balanced(len(free), [(True, .5), (False, .5)], r)))
        for i, f in enumerate(fl):
            f["follow"] = fol[i] if i in fol else f["var"] == "noupd"
    return fl


def use(monkeypatch, fams="RUP"):
    """Install round 5's builders for `fams`, its block flags and its question rule (no component rule)."""
    for f in fams:
        monkeypatch.setattr(Fm, f"fam_{f}", {"R": fam_R, "U": fam_U, "P": fam_P}[f])
    monkeypatch.setattr(S, "block_flags", block_flags)
    monkeypatch.setattr(S, "_questions", questions)


def questions(turns, final, n_extra, lure_key, r, skip=frozenset(), comp=False, place=False,
              _q=S._questions):
    """skill._questions with round 5's rule: bystanders' keys, whatever their component."""
    return _q(turns, final, n_extra, lure_key, r, skip)


def fam_A_r6(g, fl, r, _new=Fm.fam_A):
    """fams.fam_A's both-item of K3 round 6 (REVIEW 8a S-1): the context's asked attribute kept, the asked entity a
    non-holder of it drawn uniformly, so the asked name's side did not follow the R or U law (A-ONEKEY3 0.56-0.57 on
    A with a mix gain). Other lures as fams.fam_A. Install with monkeypatch.setattr(fams, "fam_A", fam_A_r6)."""
    if fl["lure"] != "both":
        return _new(g, fl, r)
    if (fl.get("ctx") or ("R" if r.random() < 0.25 else "U")) == "R":
        c = "R"
        order, (_, astar), meta = Fm.fam_R(g, {}, r)
    else:
        c = str(r.choice(["one", "sym", "noupd", "twoslot"], p=[.25, .25, .25, .25]))
        kk = int(r.integers(2, 5))
        rk, fk = (int(x) for x in r.choice(kk, 2, replace=False))
        fu = {"var": "upd" if c in ("one", "sym") else c, "sub": c, "k": kk, "rank": rk, "frank": fk}
        order, (_, astar), meta = Fm.fam_U(g, fu, r)
        c = "U" + c
    if order is None:
        return None, None, None
    hold = {s.ent for u in order for s in u if s.attr == astar}
    free = [e for e in range(meta["k"]) if e not in hold]
    if not free:
        return None, None, None
    return order, (int(r.choice(free)), astar), {"k": meta["k"], "lure": "both", "rctx": c, "cattr": astar}


def fam_A_size(g, fl, r, _new=Fm.fam_A):
    """K3 round 7's first draft of the both-item (REVIEW 8a S-1's hint taken literally): the context's own asked entity
    (slot 0, drawn by R's or U's size law), an attribute it lacks; the context redrawn when it lacks none (here, as
    fam_A_r6 draws it, also when its asked attribute has no non-holder). A then held
    0.35 of U's share in every cell where the entity lacks an attribute, a tie with a pick among 3 keys there:
    cuegate.ta_a_run read A 0.523 cross-fitted on 57,344 fresh items (K3 ROUND 7)."""
    if fl["lure"] != "both":
        return _new(g, fl, r)
    order, key, meta = fam_A_r6(g, fl, r)
    if order is None:
        return None, None, None
    own = {s.attr for u in order for s in u if s.ent == 0}
    miss = sorted({s.attr for u in order for s in u if s.kind != "def"} - own)
    if not miss:
        return None, None, None
    return order, (0, int(r.choice(miss))), meta


def fam_F_r7(g, fl, r):
    """fams.fam_F of K3 round 7 (REVIEW 9a G-1): the key's and the foil's antecedents state attributes drawn apart (each
    entity's first other attribute), so the key never restated its own antecedent's attribute while the foil's was one
    the key states in 0.20-0.40 of items: F-ANTESIZE read F 0.636 on 81,920 fresh items. Install with
    monkeypatch.setattr(fams, "fam_F", fam_F_r7)."""
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
            attrs = Fm._attrs(g, r, 1 + n_other, must=astar)
            late, stale = e == 0 and fl["klate"], fl["kstale"] if e == 0 else fl["fstale"]
            us, u, last, old = Fm._ref_units(e, attrs, fl["kref"] if e == 0 else fl["fref"], al[e], late, stale,
                                             vals, r)
            keyst[e] = (u, last, old)
        else:
            attrs = Fm._attrs(g, r, 1 + n_other, exclude=astar)
            us = Fm._ref_units(e, attrs, "alias" if r.random() < 0.5 else "ell", al[e], False, False, vals, r)[0]
        units += us
    base = Fm._rank_follow(fl, astar)
    (ku, kl, ko), (fu, _, fo) = keyst[0], keyst[foil]

    def ok(flat):
        if not base(flat):
            return False
        keyed = [s for s in flat if s.ent == 0 and s.attr == astar]
        foiled = [s for s in flat if s.ent == foil and s.attr == astar]
        if keyed[-1] is not kl or foiled[-1] is not fu:
            return False
        if (flat[-1] is kl or flat[-1] is fu) and r.random() >= Fm.LAST_KEEP:
            return False
        first_ref = min(i for i, s in enumerate(flat) if s is ku or s is fu)
        intro = max(min(i for i, s in enumerate(flat) if s.ent == e and s.kind in ("full", "reord", "def"))
                    for e in (0, foil))
        defs = [i for i, s in enumerate(flat) if s.kind == "def" and s.ent in (0, foil)]
        if first_ref - (flat[first_ref].kind == "ell") <= max([intro] + defs):
            return False
        return all(i < first_ref for i, s in enumerate(flat) if s is ko or s is fo)

    for _ in range(600):
        order = Fm._defs_first([units[i] for i in r.permutation(len(units))])
        if ok([s for u in order for s in u]):
            return order, (0, astar), {"k": k, "case": "full_after" if fl["klate"] else fl["kref"], "foil": foil}
    return None, (0, astar), {"k": k, "case": "full_after" if fl["klate"] else fl["kref"], "foil": foil}


def pad_r13(ex, key, names_k, g, r, target):
    """skill._pad of K3 rounds 4 to 13 (before G-2, REVIEW 12b T-4): a pad's random entity and attribute could join an
    earlier question's entity to the final key's component (U.b0.i87 of gen.py's K-EVAL-OOD)."""
    hold = {s.ent for k, b in ex if k == "S" for s in b if s.attr == key[1] and s.kind != "def"}
    last = max(i for i, (k, b) in enumerate(ex) if k == "S" and any((s.ent, s.attr) == key for s in b))
    if last + 1 < len(ex) and ex[last + 1][0] == "S" and ex[last + 1][1][0].kind == "ell":
        return None
    d = len(ex) - 1 - last - 1
    used = {s.val for k, b in ex if k == "S" for s in b}
    free = [int(v) for v in g.vpool if int(v) not in used]
    vals = [int(v) for v in r.choice(free, max(0, target - d), replace=False)]
    pads = []
    for v in vals:
        e, a = int(r.integers(0, names_k)), int(r.choice(g.attrs))
        if (e, a) == key:
            e = (e + 1) % names_k if names_k > 1 else e
            a = a if (e, a) != key else int(r.choice([x for x in g.attrs.tolist() if x != key[1]]))
        if a == key[1] and e not in hold:
            a = int(r.choice([x for x in g.attrs.tolist() if x != key[1]]))
        pads.append(("S", [St(Fm._form(r), e, a, v)]))
    return ex[:last + 1] + pads + ex[last + 1:]
