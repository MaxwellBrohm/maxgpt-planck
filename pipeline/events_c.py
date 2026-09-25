"""Event planners S7 (two-sided abstention, P-017), S8 (social basics) and S9 (lookup, lookup slice only, P-085).
S7 abstain asks for a fact that no event plants (the key or noun is reserved so nothing plants it later); its
candidates (same-type values in the conversation) are filled at assembly. S9 builds a fictional entity (nonce name
plus a kind), a program-written lookup call and tool result, and an answer turn."""
import banks as B
import fake_data as F
import heldout
import pools as P
from banks_keys import KEYS
from events_base import Fail, key_options, bank_options, pick_weighted
from events_a import _key_any, _ds, _plant, _answer

S7_VARIANTS = {"abstain": 50, "given": 50}
S8_ACTS = ["greeting", "how_are_you", "thanks", "who_are_you", "goodbye"]
S9_NEED = {"needed": 70, "context": 30}
S9_RESULT = {"hit": 70, "empty": 20, "counterfactual": 10}   # counterfactual arm: 0 / 10 / 25 (P-085)


# ---- S7 ------------------------------------------------------------------------------------------------------------
def s7_pre(rng):
    v = pick_weighted(rng, S7_VARIANTS)
    return {"variant": v, "need": 1 if v == "abstain" else 2}


def s7_place(ctx, pre):
    eid = ctx.next_eid()
    if pre["variant"] == "abstain":
        planted = {s["key"] for s in ctx.slots.values()} | ctx.reserved_keys
        keys = [k for k in KEYS if KEYS[k]["multi"] or k not in planted]
        key = ctx.rng.choice(keys)
        noun = ctx.draw_noun(KEYS[key]["noun"]) if KEYS[key]["noun"] else None
        if not KEYS[key]["multi"]:
            ctx.reserved_keys.add(key)
        free = [k for k in ctx.free if k > 0]
        if not free:
            raise Fail("S7 no positions")
        q = ctx.rng.choice(free)
        holes = B.key_holes(key, noun, "", {"article": ""})
        ref = B.refs(key, noun)["full"]
        ctx.user(q, eid, "query", key_options(key, "query", "full") or key_options(key, "query"), holes,
                 f"ask for {B.label(key, noun)}, which was never said", [ref])
        ctx.reply(q, eid, "say it was not mentioned, give no guess, offer to note it", [], [])
        params = {"variant": "abstain", "key": key, "noun": noun, "vtype": KEYS[key]["vtype"], "ref_expr": ref}
        return ctx.event("S7", params, {"answer": None, "abstain": True, "candidates": []},
                         {"query": ["u", q], "answer": ["a", q]})
    sid = ctx.new_slot(_key_any(ctx))
    pos = None
    for d in _ds(ctx):
        pos = ctx.pick([(d + 1, d + 1)])
        if pos:
            break
    if not pos:
        raise Fail("S7 no positions")
    p, q = pos
    s = ctx.slots[sid]
    _plant(ctx, p, eid, sid)
    ctx.reply(p, eid, "acknowledge briefly", [], [])
    ref = ctx.ref(sid, "full")
    ctx.user(q, eid, "query", key_options(s["key"], "bait"), ctx.holes(sid), "ask if the assistant remembers",
             [ref])
    _answer(ctx, q, eid, s["value"], [], ctx.rng.choice(["first", "second"]))
    ctx.window(s["value"], p, q, eid)
    params = {"variant": "given", "slot": sid, "d": q - p - 1, "ref_expr": ref,
              "ops": [{"turn": "plant", "slot": sid, "value": s["value"], "op": "plant"}]}
    return ctx.event("S7", params, {"answer": s["value"], "abstain": False, "candidates": [], "slot": sid},
                     {"plant": ["u", p], "query": ["u", q], "answer": ["a", q]})


# ---- S8 ------------------------------------------------------------------------------------------------------------
def s8_pre(rng):
    return {"act": rng.choice(S8_ACTS), "need": 1}


def s8_place(ctx, pre):
    eid = ctx.next_eid()
    act = pre["act"]
    if act == "greeting":
        k = 0
        bank = "open.greet"
    elif act == "goodbye":
        k = ctx.T - 1
        bank = "close.goodbye"
    else:
        free = [x for x in ctx.free if 0 < x < ctx.T - 1] or [x for x in ctx.free if x > 0]
        if not free:
            raise Fail("S8 no positions")
        k = ctx.rng.choice(free)
        bank = "social." + act
    ctx.user(k, eid, act, bank_options(bank), {}, f"social: {act}", [])
    if act == "who_are_you":
        ctx.reply(k, eid, "say its name and that it is an assistant", [ctx.card_name], [])
    else:
        ctx.reply(k, eid, f"respond to the {act.replace('_', ' ')} briefly, no new topic", [], [])
    gold = {"act": act, "identity": ctx.card_name if act == "who_are_you" else None, "max_w": 20}
    return ctx.event("S8", {"act": act, "needs_system": act == "who_are_you"}, gold,
                     {"social": ["u", k], "answer": ["a", k]})


# ---- S9 ------------------------------------------------------------------------------------------------------------
def s9_pre(rng):
    need = pick_weighted(rng, S9_NEED)
    res = pick_weighted(rng, S9_RESULT) if need == "needed" else "hit"
    return {"need_kind": need, "result": res, "need": 2 if need == "context" else 1}


def _entity(ctx):
    kind = ctx.rng.choice(P.pool("entity_kind").values)
    name = P.nonce(ctx.rng)
    while name in ctx.used:
        name = P.nonce(ctx.rng)
    ctx.used.add(name)
    attrs = [a for a in F.ENTITY_KINDS[kind] if ctx.type_count[F.ATTR_TYPES[a]] < heldout.MAX_SLOTS_PER_TYPE]
    if not attrs:
        raise Fail("S9 type cap")
    attr = ctx.rng.choice(attrs)
    return f"{name} {kind.capitalize()}", attr, F.ATTR_TYPES[attr]


def s9_place(ctx, pre):
    eid = ctx.next_eid()
    ent, attr, vt = _entity(ctx)
    if ctx.type_count[vt] >= heldout.MAX_SLOTS_PER_TYPE:
        raise Fail("S9 type cap")
    ctx.type_count[vt] += 1
    val = ctx.extra_value(vt)
    sid = ctx.item_slot(vt, val, key="lookup_value")
    ctx.slots[sid]["counted"] = True
    need, res = pre["need_kind"], pre["result"]
    qopts = bank_options("lookup.q." + vt)
    turns, stale = {}, []
    if need == "context":
        pos = ctx.pick([(2, 8)])
        if not pos:
            raise Fail("S9 no positions")
        s, q = pos
        pred = B.LOOKUP_PRED[vt].format(v=val)
        ctx.user(s, eid, "context", bank_options("lookup.ctx"), {"e": ent, "pred": pred}, "pass on a fact", [val])
        ctx.reply(s, eid, "acknowledge briefly", [], [])
        ctx.user(q, eid, "query", qopts, {"e": ent}, "ask about the entity", [ent])
        ctx.reply(q, eid, "answer from what the user said, without a lookup", [val], [])
        ctx.window(val, s, q, eid)
        turns = {"context": ["u", s], "query": ["u", q], "answer": ["a", q]}
        lookup = None
    else:
        free = [k for k in ctx.free if k > 0]
        if not free:
            raise Fail("S9 no positions")
        q = ctx.rng.choice(free)
        if res == "counterfactual":
            old = ctx.extra_value(vt)
            stale = [old]
            pred = B.LOOKUP_PRED[vt].format(v=old)
            ctx.user(q, eid, "query", bank_options("lookup.cf"), {"e": ent, "pred": pred}, "state a belief, ask to "
                     "check", [ent, old])
        else:
            ctx.user(q, eid, "query", qopts, {"e": ent}, "ask about the entity", [ent])
        call = f"<lookup>{ent} {attr}</lookup>"
        result = f"<result>{ent} {attr}: {'not found' if res == 'empty' else val}</result>"
        lookup = {"call": call, "result": result}
        if res == "empty":
            ctx.reply(q, eid, "say it was not found and give no guess", [], [val])
        else:
            ctx.reply(q, eid, "answer with the looked up value", [val], stale, lookup=lookup)
        ctx.aturn[q]["lookup"] = lookup
        turns = {"query": ["u", q], "call": ["a", q], "tool": ["t", q], "answer": ["a2", q]}
    params = {"entity": ent, "attribute": attr, "vtype": vt, "slot": sid, "need_kind": need, "result": res,
              "lookup": lookup, "stale": stale}
    gold = {"need": need, "result": res, "lookup": need == "needed", "answer": None if res == "empty" else val,
            "stale": stale}
    return ctx.event("S9", params, gold, turns)
