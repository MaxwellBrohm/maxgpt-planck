"""SPEC 4 and 8: skill item streams and eval sets (names, values, design flags balanced per 512-item block).

split "train": S1-S3 names minus the K-EVAL-S triples; "eval": T1-T3 (K-EVAL-ID, OOD, DEV, PROBE); "evals": the
K-EVAL-S triples [C6]. Every item: 1-3 questions (P 2-4), the final one is the scored one; d in 0..12 (DFAR 16..24);
items over 2,048 tokens or outside their ranges are redrawn with the same design flags, never cut. B eval blocks
(K-EVAL-ID, K-EVAL-S) score cube groups (K3 round 3, BD-26): group().
"""
from __future__ import annotations

import numpy as np

import fams as Fm
import items as I
import kcommon as K

FAM_SHARE = {"R": 0.25, "U": 0.25, "B": 0.15, "F": 0.15, "A": 0.10, "P": 0.10}
ROUND = {"R": 5, "U": 5, "B": 3, "F": 3, "A": 2, "P": 2}       # blocks per round, the shares above
BLOCK = 512
OOD = ("DFAR", "KMANY", "ENTS", "VALH", "ATTH", "PLONG")
OOD_FAMS = {"DFAR": "RU", "KMANY": "U", "ENTS": "R", "VALH": "RUBFAP", "ATTH": "RUBFAP", "PLONG": "P"}
MAX_TOKENS, D_MAX = 2048, 12


class Gen:
    def __init__(self, pools, split: str = "train", ood: str | None = None):
        assert split in ("train", "eval", "evals") and (ood is None or ood in OOD)
        self.split, self.ood = split, ood
        self.attrs = pools["AH"] if ood == "ATTH" else pools["A"]
        self.vpool = pools["VH"] if ood == "VALH" else pools["V"]
        self.alias, self.marker = pools["AL"], pools["MK"]
        self.npos = pools.triple("T" if split == "eval" else "S")
        self.kevs = pools.kevs_set() if split == "train" else set()
        self.kevs_arr, self.kevs_cells = pools.kevs, pools.kevs_cells

    def name(self, r) -> tuple:
        if self.split == "evals":
            return tuple(int(x) for x in self.kevs_arr[int(r.integers(0, len(self.kevs_arr)))])
        while True:
            t = tuple(int(p[int(r.integers(0, len(p)))]) for p in self.npos)
            if t not in self.kevs:
                return t

    def cube(self, r) -> list | None:
        """B (K3 round 2): the 8 names {a, a'} x {b, b'} x {c, c'}, two tokens per position, in a fixed order (the
        caller shuffles). Every name shares 2 tokens with 3 others, 1 with 3 and none with 1, so no name is more
        central than another. train: no K-EVAL-S triple; evals: a K-EVAL-S cell (pools.kevs_cells)."""
        if self.split == "evals":
            c = self.kevs_cells[int(r.integers(0, len(self.kevs_cells)))]
            toks = [[int(x) for x in c[i]] for i in range(3)]
        else:
            for _ in range(100):
                toks = [[int(x) for x in r.choice(p, 2, replace=False)] for p in self.npos]
                if not any((a, b, c) in self.kevs for a in toks[0] for b in toks[1] for c in toks[2]):
                    break
            else:
                return None
        return [(a, b, c) for a in toks[0] for b in toks[1] for c in toks[2]]

    def values(self, r, n) -> list[int]:
        return [int(v) for v in r.choice(self.vpool, n, replace=False)]


def balanced(n: int, cats: list, r) -> list:
    """Exact category counts (largest remainder), in a random order."""
    raw = [n * f for _, f in cats]
    cnt = [int(x) for x in raw]
    for i in sorted(range(len(cats)), key=lambda i: -(raw[i] - cnt[i]))[:n - sum(cnt)]:
        cnt[i] += 1
    out = [c for (c, _), m in zip(cats, cnt) for _ in range(m)]
    return [out[i] for i in r.permutation(n)]


def nested(n: int, outer: list, inner, r) -> list:
    """(outer value, inner value) per item: outer counts exact, then inner(outer value)'s counts exact within each
    outer group (one joint draw would round the outer marginal off by one on odd group sizes)."""
    top = balanced(n, outer, r)
    out = [None] * n
    for v, _ in outer:
        idx = [i for i, t in enumerate(top) if t == v]
        for i, c in zip(idx, balanced(len(idx), inner(v), r)):
            out[i] = (v, c)
    return out


def rank_pairs(n: int, k: int, r) -> list[tuple]:
    """U (K3 round 4): n (key intro rank, foil intro rank) pairs, each role's rank exact (every rank within one of
    n / k) and every ordered pair within one of n / (k (k - 1)): whole copies of the k (k - 1) pairs, then whole
    cyclic shifts (rank i with i + s mod k: each rank once per role), then part of one more shift at distinct ranks."""
    q, rem = divmod(n, k * (k - 1))
    a, b = divmod(rem, k)
    shifts = [int(s) for s in r.permutation(np.arange(1, k))]
    out = [(i, (i + s) % k) for _ in range(q) for s in range(1, k) for i in range(k)]
    out += [(i, (i + s) % k) for s in shifts[:a] for i in range(k)]
    out += [(int(i), (int(i) + shifts[a]) % k) for i in r.choice(k, b, replace=False)] if b else []
    return [out[i] for i in r.permutation(n)]


def ranks(ks: list, r) -> list[int]:
    """An intro rank per item, exact per k: balanced over 0..k-1 among the items with each k (K3 round 4)."""
    out = [0] * len(ks)
    for k in sorted(set(ks)):
        idx = [i for i, x in enumerate(ks) if x == k]
        for i, x in zip(idx, balanced(len(idx), [(j, 1 / k) for j in range(k)], r)):
            out[i] = x
    return out


def block_flags(fam: str, n: int, r, ncube: int | None = None) -> list[dict]:
    """Exact per-block design flags. ncube (B eval blocks with groups, BD-26): that many cube items among n, the
    question frame split exactly within the cube items and within the others (a cube item carries 3 copies)."""
    fl = [{"qform": q} for q in balanced(n, [(0, .5), (1, .5)], r)]
    if fam in ("R", "P"):       # P's statements follow R's design (SPEC 4 leaves P's value structure open); K3 round
        cs = balanced(n, [("f1", .5), ("f0", .3), ("nofoil", .2)] if fam == "R" else [("f1", .5), ("f0", .5)], r)
        # 4: P has a foil in every item, follow in 1/2 (with R's no-foil fifth P's gate read 0.649-0.670 on gen.py
        # samples against a limit of 0.666: the fifth is free and 5/8 of foil items put the key's latest first)
        ks = [0] * n            # K3 round 4: k (foil items 2-4, the rest 1-4; P one more, its bystander) and the
        for foil, opts in ((True, (2, 3, 4)), (False, (1, 2, 3, 4))):     # intro rank exact per block (drawn per
            idx = [i for i, c in enumerate(cs) if (c != "nofoil") == foil]  # item, one K-EVAL-sized R block read
            for i, k in zip(idx, balanced(len(idx), [(k, 1 / len(opts)) for k in opts], r)):   # k 3 rank 1 at
                ks[i] = k + (fam == "P")                                    # 0.506, 3.3 SE)
        for f, c, k, x in zip(fl, cs, ks, ranks(ks, r)):
            f.update(follow=c == "f1", foil=c != "nofoil", k=k, rank=x)
    elif fam == "U":            # K3: noupd (key 1, foil 2, follow) mirrored by upd "one" (2, 1, no follow); upd
        var = nested(n, [("upd", .5), ("noupd", .25), ("twoslot", .25)],       # "sym" and twoslot: follow in 1/2
                     lambda v: [("one", .5), ("sym", .5)] if v == "upd" else [("-", 1.)], r)
        free = [i for i, (v, s) in enumerate(var) if v == "twoslot" or s == "sym"]
        fol = dict(zip(free, balanced(len(free), [(True, .5), (False, .5)], r)))
        # (k, intro rank) exact too: drawn in fam_U, the K-EVAL-ID draw read k 4 rank 2 at 0.147 (n 163, 3 SE).
        # K3 round 4 (REVIEW 5a S-2): the foil's intro rank with the key's, the (key, foil) rank pair uniform over the
        # k (k - 1) ordered pairs, exact per block: the foil, a big entity, kept an early intro (the holder introduced
        # last was the key in 0.541 of non-lure U items), so key and foil were not exchangeable to a question-blind rule
        ks = balanced(n, [(2, 1 / 3), (3, 1 / 3), (4, 1 / 3)], r)
        kr = [None] * n
        for k in (2, 3, 4):
            idx = [i for i, x in enumerate(ks) if x == k]
            for i, p in zip(idx, rank_pairs(len(idx), k, r)):
                kr[i] = (k, p)
        for i, (f, (v, s), (k, (rk, fk))) in enumerate(zip(fl, var, kr)):
            f.update(var=v, sub=s, follow=fol[i] if i in fol else v == "noupd", k=k, rank=rk, frank=fk)
    elif fam == "B":            # BD-1: B's order is uniform, so its follow rate is chance (1 - 1/k), not 1/2.
        share = .75 if ncube is None else ncube / n
        sk = nested(n, [(True, share), (False, 1 - share)],     # BD-22: the cube (k 8) in 3/4, k 2-4 of it in 1/4,
                    lambda c: [(8, 1.)] if c else [(2, 1 / 3), (3, 1 / 3), (4, 1 / 3)], r)      # and each k's intro
        rk = [0] * n                                            # rank exact too (per item they failed SPEC 5's
        for k in (2, 3, 4, 8):                                  # rank rule at 3.2 SE on one gen.py sample)
            idx = [i for i, (_, kk) in enumerate(sk) if kk == k]
            for i, x in zip(idx, balanced(len(idx), [(j, 1 / k) for j in range(k)], r)):
                rk[i] = x
        for f, (c, k), x in zip(fl, sk, rk):
            f.update(struct=c, k=k, rank=x)
        if ncube is not None:                                   # BD-26: frames exact within cube and small items
            for c in (True, False):
                idx = [i for i, (cc, _) in enumerate(sk) if cc == c]
                for i, q in zip(idx, balanced(len(idx), [(0, .5), (1, .5)], r)):
                    fl[i]["qform"] = q
    elif fam == "F":            # R-1: one reference kind for the key and the foil (BD-14), nested in follow, so a
        kinds = [("alias", .5), ("ell", .5)]        # foil alias follows in 1/4 (the tie order flips between the
        fol = nested(n, [(1, .5), (0, .5)], lambda v: kinds if v else kinds[::-1], r)   # groups: kind exact too)
        late = nested(n, [(1, .25), (0, .75)], lambda v: [(0, 1.)] if v else [(1, .5), (0, .5)], r)  # key's (late,
        fst = balanced(n, [(1, .5), (0, .5)], r)          # stale) and the foil's stale exact too
        ks = balanced(n, [(2, 1 / 3), (3, 1 / 3), (4, 1 / 3)], r)      # K3 round 4: F's (k, rank) exact too
        for f, (w, x), (kl, kst), fs, k, rk in zip(fl, fol, late, fst, ks, ranks(ks, r)):
            f.update(kref=x, klate=bool(kl), kstale=bool(kst), follow=bool(w), fref=x, fstale=bool(fs), k=k, rank=rk)
    elif fam == "A":            # K3 round 4: a both-item's context kind (R or U) exact too; drawn per item, the
        lu = nested(n, [("both", .5), ("ent", .25), ("attr", .25)],      # contexts that fail more (R's with no
                    lambda v: [("R", .5), ("U", .5)] if v == "both" else [("-", 1.)], r)    # bystander) were redrawn
        for f, (c, x) in zip(fl, lu):                                   # into U contexts (93 R vs 163 U of 256)
            f.update(lure=c, ctx=x)
    return fl


def _questions(turns, final: I.Qn, n_extra: int, lure_key, r, skip=frozenset()) -> list:
    """Earlier questions at random turn ends (never between an ellipsis and its antecedent), each asking a key stated
    so far. None asks the final question's attribute (K3 R-4: an earlier answer of that attribute named the gold, a
    copy, or excluded a foil's value), except U's lure: the final key itself, asked only where a later statement
    of it follows, so its answer is always a stale value (SPEC 4's lure). K3 round 4 (REVIEW 5a S-1): none names
    the final entity or any holder of the final attribute (a bystander's key only; U's lure aside). Drawn from every
    stated key, whether and how often the final name was asked told the asked entity's size (it states another
    attribute), and the holders' sizes are read off the statements (QFIT R 0.683, P 0.707). skip: entities never
    asked either (make_item: the bystander R and U items reserve)."""
    hold = {s.ent for t in turns for s in t if s.attr == final.attr and s.kind != "def"} | {final.ent} | set(skip)
    ok = [j for j in range(len(turns)) if j + 1 >= len(turns) or turns[j + 1][0].kind != "ell"]
    at = sorted(r.choice(ok, min(n_extra, len(ok)), replace=False).tolist()) if n_extra else []
    ex, stated = [], []
    for j, t in enumerate(turns):
        ex.append(("S", t))
        stated += [(s.ent, s.attr) for s in t if s.kind != "def" and (s.ent, s.attr) not in stated]
        pool = [k for k in stated if k[1] != final.attr and k[0] not in hold]
        lure = lure_key in stated and any((s.ent, s.attr) == lure_key for u in turns[j + 1:] for s in u)
        for _ in range(at.count(j)):
            if lure and r.random() < 0.5:
                key = lure_key
            elif pool:
                key = pool[int(r.integers(0, len(pool)))]
            else:
                continue
            ex.append(("Q", I.Qn(key[0], key[1], int(r.integers(0, 2)))))
    return ex + [("Q", final)]


def _pad(ex, key, names_k, g, r, target):
    """DFAR: insert one-statement exchanges after the asked key's latest statement until d = target. A pad states the
    asked attribute only for an entity that already does (K3 round 4: a bystander made a holder could be one an
    earlier question asked, or leave every remaining bystander asked, which _questions and make_item never make)."""
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
        pads.append(("S", [I.St(Fm._form(r), e, a, v)]))
    return ex[:last + 1] + pads + ex[last + 1:]


def make_item(g: Gen, fam: str, fl: dict, r, one_q: bool = False) -> I.Item | None:
    ood = g.ood
    if fam == "R":
        order, key, meta = Fm.fam_R(g, fl, r, k=int(r.integers(5, 7)) if ood == "ENTS" else None)
    elif fam == "U":
        order, key, meta = Fm.fam_U(g, fl, r, kmany=ood == "KMANY")
    elif fam == "P":
        order, key, meta = Fm.fam_P(g, fl, r, n_q=int(r.integers(5, 8)) if ood == "PLONG" else None)
    else:
        order, key, meta = {"B": Fm.fam_B, "F": Fm.fam_F, "A": Fm.fam_A}[fam](g, fl, r)
    if order is None:
        return None
    k = meta["k"]
    names = meta.pop("names", None) or _names(g, r, k)
    if names is None:
        return None
    turns = I.group(order, r)
    if fam == "P":
        n_extra = int(r.integers(4, 7)) if ood == "PLONG" else int(r.integers(1, 4))
    else:
        n_extra = 0 if (fam == "B" and one_q) else int(r.integers(0, 3))
    # K3 round 4: R and U items (A's contexts) reserve one bystander, never asked, as an A both-item's asked entity
    # is never asked (with every bystander askable, every bystander was asked in 0.350 of the R and U items that had
    # one and in no A both-item, and the gate read A 0.536 on gen.py's stream)
    skip = set()
    if fam in ("R", "U"):
        free = sorted({s.ent for u in order for s in u} - {s.ent for u in order for s in u if s.attr == key[1]})
        skip = {int(r.choice(free))} if free else set()
    ex = _questions(turns, I.Qn(key[0], key[1], fl["qform"]), n_extra, key if fam == "U" else None, r, skip)
    if fam == "P" and sum(k == "Q" for k, _ in ex) < (5 if ood == "PLONG" else 2):
        return None             # P: 2-4 questions (PLONG 5-7) though earlier ones skip the final attribute
    if ood == "DFAR":
        ex = _pad(ex, key, len(names), g, r, int(r.integers(16, 25)))
        if ex is None:
            return None
    it = I.Item(fam, names, ex, meta.pop("marker", -1), {**fl, **meta})
    if not _in_range(it, ood):
        return None
    return it


def _names(g: Gen, r, k: int):
    out = []
    while len(out) < k:
        t = g.name(r)
        if t not in out:
            out.append(t)
    return out


def _in_range(it: I.Item, ood) -> bool:
    rec = I.render(it, "x", full=True)
    if I.n_tokens(rec) > MAX_TOKENS:
        return False
    d = rec["qs"][-1]["d"]
    lo, hi = (16, 24) if ood == "DFAR" else (0, D_MAX)
    if "rctx" in it.meta:       # an A both-item keeps its R or U context's range: that key's d (fams.fam_A)
        a = [b for k, b in it.ex if k == "Q"][-1].attr
        last = max(xi for xi, (k, b) in enumerate(it.ex) if k == "S" and any((s.ent, s.attr) == (0, a) for s in b))
        d = len(it.ex) - 1 - last - 1
    if (it.fam != "A" or "rctx" in it.meta) and not lo <= d <= hi:
        return False
    defs, dent = {}, {}
    for xi, (kind, body) in enumerate(it.ex):       # alias used 1-10 exchanges after its definition
        for s in body if kind == "S" else []:
            if s.kind == "def":
                defs[s.alias], dent[s.ent] = xi, xi
            elif s.kind == "alias" and not 1 <= xi - defs[s.alias] <= 10:
                return False
    if it.fam == "F":           # BD-25 (K3 round 3, REVIEW 4a S-2): the key's and the foil's aliases of the asked
        a = [b for k, b in it.ex if k == "Q"][-1].attr      # attribute sit 1-10 exchanges after BOTH their
        pair = [e for e in (0, it.meta["foil"]) if e in dent]       # definitions (per alias, the foil's could share
        for xi, (kind, body) in enumerate(it.ex):                   # the asked name's definition turn: F-WINK)
            for s in body if kind == "S" else []:
                if s.kind == "alias" and s.attr == a and s.ent in (0, it.meta["foil"]) and \
                        not all(1 <= xi - dent[e] <= 10 for e in pair):
                    return False
    return True


def group(it: I.Item) -> list[I.Item]:
    """B eval group (K3 round 3, BD-26; REVIEW 4a S-1): a cube item (meta grp 0) and 3 copies of its statements, each
    asking one of the asked name's two-token neighbours (grp 1-3; the same attribute and frame), scored jointly: a
    group is right only if all 4 are. A rule that reads a proper subset of the question's name positions gives two
    of them the same answer (the asked name and the neighbour that agrees with it there), with any tie-break, so it
    gets at most 1/16 of groups with tie credit and none without; a reversed twin only ruled out recency."""
    q = it.ex[-1][1]
    flat = [s for k, b in it.ex if k == "S" for s in b]
    nb = [e for e, u in enumerate(it.names) if sum(x == y for x, y in zip(u, it.names[0])) == 2]
    assert len(nb) == 3, it.names
    return [I.Item(it.fam, it.names, it.ex, it.marker, {**it.meta, "grp": 0})] + [
        I.Item(it.fam, it.names, it.ex[:-1] + [("Q", I.Qn(e, q.attr, q.form))], it.marker,
               {**it.meta, "grp": j, "rank": I.intro_rank(flat, e)}) for j, e in enumerate(nb, 1)]


def block(g: Gen, fam: str, n: int, seed: tuple, groups: bool = False) -> list[I.Item]:
    """n items of one family, exact per-block flags. groups (B eval blocks, BD-26; n a multiple of 32): 3n/16 cube
    items (an even number), each followed by its 3 group copies, and n/4 items naming 2-4 of a cube; every B eval
    item carries only its final question (BD-7)."""
    r = np.random.default_rng([K.W, *seed])
    grp = groups and fam == "B"
    assert not grp or n % 32 == 0, n
    ncube = n * 3 // 16 if grp else None
    out = []
    for fl0 in block_flags(fam, n - 3 * ncube if grp else n, r, ncube):
        fl = dict(fl0)
        for att in range(2000):
            if att % 50 == 49:          # a (k, rank) the builder cannot meet: draw them again
                fl = dict(fl0)
            it = make_item(g, fam, fl, r, one_q=grp)
            if it is not None and (not grp or not fl["struct"] or all(_in_range(c, g.ood) for c in group(it)[1:])):
                break
        else:
            raise RuntimeError(f"{fam} {fl}: no item in 2000 draws")
        out += group(it) if grp and fl["struct"] else [it]
    return out
