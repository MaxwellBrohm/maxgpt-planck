"""Event planners S1 (recall at distance), S2 (reference and list state), S3 (correction and update).
Each kind has pre(rng) -> params with "need" (user turns it claims) and place(ctx, pre) -> event id, which
claims turns, writes user lines or guided intents, reply specs, leak windows, params (with an "ops" replay list)
and the gold. golds.py re-derives every gold from the skeleton alone, so ops must say what each turn states."""
import banks as B
from banks_keys import KEYS
from events_base import Fail, key_options, bank_options, keys_of_type, pick_weighted

S3_VARIANTS = {"update": 40, "no_update_twin": 20, "two_slot": 15, "revert": 10, "assistant_err": 15}
MARKER_NONE = 0.35


def _usable(ctx, k):
    if ctx.type_count[KEYS[k]["vtype"]] != 0:
        return False
    if KEYS[k]["multi"]:
        return True
    return k not in ctx.reserved_keys and not any(s["key"] == k for s in ctx.slots.values())


def _key_any(ctx, exclude=()):
    keys = [k for k in KEYS if k not in exclude and _usable(ctx, k)]
    if not keys:
        raise Fail("no key")
    return ctx.rng.choice(keys)


def _pair_keys(ctx):
    """two key instances of one value type with no event slot of that type yet."""
    pairs = []
    for k1 in KEYS:
        if not _usable(ctx, k1):
            continue
        pairs += [(k1, k2) for k2 in keys_of_type(KEYS[k1]["vtype"]) if _usable(ctx, k2)
                  and (k2 != k1 or KEYS[k1]["multi"])]
    if not pairs:
        raise Fail("no key pair")
    return ctx.rng.choice(pairs)


def _ds(ctx, lo=1, hi=10):
    ds = list(range(lo, min(hi, ctx.T - 2) + 1))
    ctx.rng.shuffle(ds)
    return ds


def _marker(ctx):
    if ctx.rng.random() < MARKER_NONE:
        return None, ""
    i = ctx.rng.randrange(len(B.MARKERS))
    return i, B.MARKERS[i]


def _plant(ctx, k, eid, sid, role="plant"):
    s = ctx.slots[sid]
    ctx.user(k, eid, role, key_options(s["key"], "plant"), ctx.holes(sid), f"state {B.label(s['key'], s['noun'])}",
             [s["value"]])


def _query(ctx, k, eid, sid, form, role="query", opts=None):
    s = ctx.slots[sid]
    if opts is None:
        opts = key_options(s["key"], "query", form) or key_options(s["key"], "query")
    ref = ctx.ref(sid, form if form in ("full", "head") else "full")
    ctx.user(k, eid, role, opts, ctx.holes(sid), f"ask for {B.label(s['key'], s['noun'])}", [ref])
    return ref


def _ack(ctx, k, eid, value, mention):
    ctx.reply(k, eid, "acknowledge briefly", [value] if mention else [], [] if mention else [value])


def _answer(ctx, k, eid, gold, bad, order):
    ctx.reply(k, eid, f"answer with the value {'first' if order == 'first' else 'after a short lead in'}",
              [gold], [b for b in bad if b != gold])


# ---- S1 ------------------------------------------------------------------------------------------------------------
def s1_pre(rng):
    dis = rng.random() < 0.5
    return {"distractor": dis, "answer_order": rng.choice(["first", "second"]),
            "ref_form": rng.choice(["full", "head"]), "ack_mentions": rng.random() < 0.5, "need": 2 + dis}


def s1_place(ctx, pre):
    eid = ctx.next_eid()
    if pre["distractor"]:
        key, dkey = _pair_keys(ctx)
    else:
        key, dkey = _key_any(ctx), None
    sid = ctx.new_slot(key)
    dsid = ctx.new_slot(dkey) if dkey else None
    for d in _ds(ctx):
        pos = ctx.pick([(d + 1, d + 1)])
        if not pos:
            continue
        p, q = pos
        dpos = None
        if dsid:
            opts = [x for x in ctx.free if x < q and x not in (p, q)]
            if not opts:
                continue
            dpos = ctx.rng.choice(opts)
        break
    else:
        raise Fail("S1 no positions")
    val = ctx.slots[sid]["value"]
    _plant(ctx, p, eid, sid)
    _ack(ctx, p, eid, val, pre["ack_mentions"])
    ops = [{"turn": "plant", "slot": sid, "value": val, "op": "plant"}]
    turns = {"plant": ["u", p], "query": ["u", q], "answer": ["a", q]}
    bad = []
    if dsid:
        dval = ctx.slots[dsid]["value"]
        _plant(ctx, dpos, eid, dsid, role="distractor")
        ctx.reply(dpos, eid, "acknowledge briefly", [], [val])
        ops.append({"turn": "distractor", "slot": dsid, "value": dval, "op": "plant"})
        turns["distractor"] = ["u", dpos]
        bad = [dval]
        ctx.window(dval, dpos, q, eid)
    ref = _query(ctx, q, eid, sid, pre["ref_form"])
    _answer(ctx, q, eid, val, bad, pre["answer_order"])
    ctx.window(val, p, q, eid)
    params = {"slot": sid, "plant_turn": p, "query_turn": q, "d": q - p - 1, "distractor": dsid,
              "distractor_pos": None if dpos is None else ("before" if dpos < p else "between"),
              "ref_form": pre["ref_form"], "ref_expr": ref, "answer_order": pre["answer_order"],
              "ack_mentions": pre["ack_mentions"], "ops": ops}
    return ctx.event("S1", params, {"answer": val, "stale": [], "candidates": bad, "slot": sid}, turns)


# ---- S2 ------------------------------------------------------------------------------------------------------------
S2_TYPES = ["grocery", "city", "name", "chore"]


def s2_pre(rng):
    n_ops = rng.choice([0, 1, 1, 2])
    return {"item_type": rng.choice(S2_TYPES), "n_items": rng.randint(3, 5), "n_ops": n_ops, "need": 2 + n_ops}


def list_query(rng, final, removed, n0):
    """(qtype, arg, answer) valid for the final list state."""
    opts = ["first", "last", "count", "contains"] + (["ordinal"] if len(final) >= 3 else [])
    opts += ["other"] if len(final) == 2 else []
    q = rng.choice(opts)
    if q == "first":
        return q, None, final[0]
    if q == "last":
        return q, None, final[-1]
    if q == "count":
        return q, None, ["zero", "one", "two", "three", "four", "five", "six", "seven"][len(final)]
    if q == "ordinal":
        i = rng.randint(1, min(4, len(final) - 1))
        return q, ["first", "second", "third", "fourth", "fifth"][i], final[i]
    if q == "other":
        i = rng.randrange(2)
        return q, final[i], final[1 - i]
    if removed and rng.random() < 0.5:
        return q, rng.choice(removed), "no"
    return q, rng.choice(final), "yes"


def s2_place(ctx, pre):
    eid = ctx.next_eid()
    vt = pre["item_type"]
    items = [ctx.extra_value(vt) for _ in range(pre["n_items"])]
    for v in items:
        ctx.item_slot(vt, v)
    pos = ctx.pick([(1, 3)] * (pre["n_ops"] + 1))
    if not pos:
        raise Fail("S2 no positions")
    cur, removed, ops = list(items), [], [{"turn": "init", "op": "init", "items": list(items)}]
    for j in range(pre["n_ops"]):
        choices = ["add", "remove"] + (["move_first", "move_last"] if len(cur) >= 2 else [])
        op = ctx.rng.choice(choices if len(cur) > 2 else ["add", "move_first", "move_last"])
        if op == "add":
            v = ctx.extra_value(vt)
            ctx.item_slot(vt, v)
            cur.append(v)
        else:
            cand = cur[1:] if op == "move_first" else cur[:-1] if op == "move_last" else cur
            v = ctx.rng.choice(cand)
            cur.remove(v)
            if op == "move_first":
                cur.insert(0, v)
            elif op == "move_last":
                cur.append(v)
            else:
                removed.append(v)
        ops.append({"turn": f"op{j + 1}", "op": op, "value": v})
    qtype, arg, ans = list_query(ctx.rng, cur, removed, len(items))
    k0, q = pos[0], pos[-1]
    L = B.LIST_NAMES[vt]
    ctx.user(k0, eid, "init", bank_options("list.init." + vt), {"items": B.join_items(items), "L": L},
             "state the list", items + [L])
    ctx.reply(k0, eid, "acknowledge the list", [], [])
    turns = {"init": ["u", k0], "query": ["u", q], "answer": ["a", q]}
    for j, o in enumerate(ops[1:]):
        k = pos[j + 1]
        ctx.user(k, eid, o["turn"], bank_options("list." + o["op"]), {"v": o["value"], "L": L},
                 f"list {o['op']}", [o["value"], L])
        ctx.reply(k, eid, "acknowledge the change", [], [])
        turns[o["turn"]] = ["u", k]
    holes = {"ord": arg, "v": arg, "L": L}
    ctx.user(q, eid, "query", bank_options("list.q." + qtype), holes, f"ask list {qtype}",
             ([arg] if arg else []) + [L])
    ever = items + [o["value"] for o in ops[1:] if o["op"] == "add"]
    bad = [x for x in ever if x != ans] if qtype not in ("count", "contains") else []
    ctx.reply(q, eid, "answer from the list", [ans] if qtype != "contains" else [], bad)
    for v in set(items + [o["value"] for o in ops[1:]]):
        ctx.window(v, k0, q, eid)
    params = {"item_type": vt, "list_name": L, "items": items, "ops": ops, "query": qtype, "query_arg": arg}
    return ctx.event("S2", params, {"answer": ans, "list": cur, "query": qtype}, turns)
