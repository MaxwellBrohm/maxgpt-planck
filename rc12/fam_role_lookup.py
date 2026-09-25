"""RC-12 DEV builders: ROLE (role integrity) and LOOKUP (reading from a supplied table) (SPEC s3).

ROLE: the user states name, job and town, plus two other people's names (lures; the user's name is first, middle
or last of the three in thirds) and, in half the conversations, a name for the assistant. Step 5 audit: frames
that echo "What's my name?" and frames that do not exist on both sides, and balance.choose() keeps the user's
name statement from being the unique most / least echoing one and puts it at a balanced length rank. All of it sits in u1..u6 (deviation from SPEC u1-u3: notes.txt). P "What's my name?" (u12 in
3/4, u8-u11 in 1/4) and X "Tell me a little about yourself." (another turn after the statements). Unit passes
only if P is right AND X passes (G-ROLEX).
LOOKUP: two tables of the same shape (old and new; order balanced) in u1..u4, three shared keys with different
values plus one key unique to each table. P asks a shared key of the named table (the other table's value is a
lure); X asks the key that only the OTHER table has (must abstain). Unit passes only if both are right. X comes
before P in exactly half of each shape (step 5 audit: it came first in 46 of 48, so "abstain on the first table
question" passed)."""
import balance as B
import common as C
import pools_vals as V
import pools_misc as M


def build_role(n=48):
    rng = C.stream("ROLE")
    rows = [(named, npos) for named in (True, False) for npos in ("first", "middle", "last")]
    rows = C.spread(rows, n, rng)
    lens = {}
    for named in (True, False):                # step 5 audit: gold length rank balanced 8/8/8 inside each cell
        idx = [j for j, r in enumerate(rows) if r[0] == named]
        lens.update(zip(idx, C.spread(B.TARGETS, len(idx), rng)))
    late = n // 4
    ps = [12] * (n - late) + [8 + (i % 4) for i in range(late)]
    rng.shuffle(ps)
    out = []
    for k, ((named, npos), p) in enumerate(zip(rows, ps)):
        conv = C.Conv("ROLE", "named" if named else "unnamed", C.rid("ROLE", k + 1), rng)
        nm, l1, l2 = rng.sample(V.PERSON, 3)
        job, town = rng.choice(V.JOB), rng.choice(V.TOWN)
        an = rng.choice(V.ANAME) if named else None
        hs = rng.sample(M.ROLE_LURE_HOLDERS, 2)
        opts = [[x for f in M.ROLE_NAME for x in B.tailed(f.format(n=nm))]]
        opts += [[x for f in M.ROLE_LURE for x in B.tailed(f.format(h=h, m=m))] for h, m in zip(hs, (l1, l2))]
        if named:
            opts.append([f.format(a=an) for f in M.ROLE_ANAME])
        texts = B.choose(rng, opts, [0], lens[k], question=M.ROLE_P, values=V.ALL_VALUES, cap=40000)
        if texts is None:
            raise ValueError(f"ROLE {k + 1}: no balanced statement set")
        stmts = [("name", [texts[0]], dict(n=nm)), ("job", M.ROLE_JOB, dict(j=job)),
                 ("town", M.ROLE_TOWN, dict(t=town)), ("lure1", [texts[1]], dict(m=l1)),
                 ("lure2", [texts[2]], dict(m=l2))]
        if named:
            stmts.append(("aname", [texts[3]], dict(a=an)))
        want = ["first", "middle", "last"].index(npos)     # the user's name among the three people's names
        while True:
            rng.shuffle(stmts)
            people = [s[0] for s in stmts if s[0] in ("name", "lure1", "lure2")]
            if people.index("name") == want:
                break
        turn_of = {}
        for t, (key, tpls, kw) in enumerate(stmts, start=1):
            val = list(kw.values())[0]
            holder = {"lure1": "other", "lure2": "other", "aname": "assistant"}.get(key, "user")
            conv.put(t, "L" if key.startswith("lure") else "S", rng.choice(tpls).format(**kw),
                     facts=[dict(holder=holder, object=key, value=val, role=key)])
            turn_of[key] = t
        x = rng.choice([t for t in range(len(stmts) + 1, 12) if t != p])
        named_vals = (("name", nm), ("lure1", l1), ("lure2", l2), ("aname", an))
        cands = [v for _, v in sorted((turn_of[kk], vv) for kk, vv in named_vals if vv is not None)]
        conv.add_probe(p, "P", "VAL", M.ROLE_P, f"Your name is {nm}.", gold=nm, candidates=cands, pool="name",
                       holder="user", object_words=["name"], src=[turn_of["name"]], prefix="Your name is")
        conv.add_probe(x, "X", "ROLEX", M.ROLE_X, M.ROLE_X_IDEAL[named].format(a=an), gold=an,
                       candidates=cands, pool="name", holder="assistant", object_words=[],
                       src=[turn_of["aname"]] if named else [], user_facts=dict(name=nm, job=job, town=town),
                       aname=an)
        conv.meta.update(unit="all", named=named, name_pos=npos, len_rank=lens[k])
        conv.fill()
        out.append(conv.record())
    return out


def _rows(keys, vals):
    return ", ".join(f"{k}: {v}" for k, v in zip(keys, vals))


def build_lookup(n=48):
    rng = C.stream("LOOKUP")
    combos = [(shape, old_first, asked) for shape in M.LOOKUP_SHAPES for old_first in (True, False)
              for asked in ("old", "new")]
    rows = C.spread(combos, n, rng)
    mains = C.probe_turns(rng, n)             # the LATER of the two table questions (u12 in 3/4)
    # step 5 audit: the abstain X came first in 46 of 48 items, so "abstain on the first table question, look up
    # the second" passed 0.96; now X first in exactly half of every shape (x_first 8/8 per shape)
    xf = {}
    for shape in M.LOOKUP_SHAPES:
        idx = [j for j, r in enumerate(rows) if r[0] == shape]
        xf.update(zip(idx, C.spread([True, False], len(idx), rng)))
    out = []
    for k, ((shape, old_first, asked), last) in enumerate(zip(rows, mains)):
        early = rng.choice(range(6, last))
        p, x = (last, early) if xf[k] else (early, last)
        out.append(_lookup_one(rng, k + 1, shape, old_first, asked, p, x))
    return out


def _lookup_one(rng, num, shape, old_first, asked, p, x):
    S = M.LOOKUP_SHAPES[shape]
    conv = C.Conv("LOOKUP", shape, C.rid("LOOKUP", num), rng)
    kpool = V.POOLS[S["keys"]]
    keys5 = sorted(rng.sample(kpool, 5), key=kpool.index)
    shared = rng.sample(keys5, 3)
    only = [kk for kk in keys5 if kk not in shared]
    rng.shuffle(only)
    tab_keys = {"new": sorted(shared + [only[0]], key=kpool.index),
                "old": sorted(shared + [only[1]], key=kpool.index)}
    vals = rng.sample(V.POOLS[S["vals"]], 8)
    tab = {"new": dict(zip(tab_keys["new"], vals[:4])), "old": dict(zip(tab_keys["old"], vals[4:]))}
    t1, t2 = sorted(rng.sample(range(1, 5), 2))
    turn_of = {"old": t1, "new": t2} if old_first else {"new": t1, "old": t2}
    for age in ("old", "new"):
        ks = tab_keys[age]
        conv.put(turn_of[age], "T", S["intro"].format(age=age, rows=_rows(ks, [tab[age][kk] for kk in ks])),
                 facts=[dict(holder="table", object=age, value=None, role="table", table=tab[age])])
    other = "old" if asked == "new" else "new"
    key = rng.choice(shared)
    gold = tab[asked][key]
    absent = only[0] if asked == "old" else only[1]       # the key only the OTHER table has
    order = [a for a, _ in sorted(turn_of.items(), key=lambda kv: kv[1])]
    cands = [tab[a][kk] for a in order for kk in tab_keys[a]]
    conv.add_probe(p, "P", "VAL", S["q"].format(age=asked, k=key), S["ideal"].format(age=asked, k=key, v=gold),
                   gold=gold, candidates=cands, pool=S["vals"], holder=f"table:{asked}", object_words=[key],
                   src=[turn_of[asked]], prefix=None, lure_value=tab[other][key], key=key)
    conv.add_probe(x, "X", "ABS", S["q"].format(age=asked, k=absent), S["abst"].format(age=asked, k=absent),
                   gold=None, candidates=cands, pool=S["vals"], holder=f"table:{asked}", object_words=[absent],
                   src=[], key=absent, other_value=tab[other][absent])
    conv.meta.update(unit="all", old_first=old_first, asked=asked, table_turns=turn_of, x_first=x < p)
    conv.avoid.update(["rota", "menu", "tour"])
    conv.fill()
    return conv.record()
