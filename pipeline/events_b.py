"""Event planners S4 (instruction persistence and override, P-018), S5 (topic return, P-044) and S6 (speaker and
perspective binding, P-007/P-008). S4 rule segments are closed at assembly: a rule holds from its start turn until
the next rule start (an override or a later S4) or the end, on every assistant reply that is not a lookup call."""
import banks as B
import pools as P
from banks_keys import KEYS
from events_base import Fail, key_options, bank_options, pick_weighted
from events_a import _key_any, _ds, _plant, _query, _ack, _answer

RULES = ["max_words", "one_sentence", "end_question", "call_user", "avoid_word", "start_name"]
MAX_WORDS = {"eight": 8, "ten": 10, "twelve": 12}
S6_VARIANTS = {"two_persons": 40, "user_vs_card": 30, "role_swap": 30}
SWAP_KEYS = ["home_city", "job", "hobby", "fav_food", "fav_colour"]


# ---- S4 ------------------------------------------------------------------------------------------------------------
def s4_pre(rng):
    ov = rng.random() < 0.3
    rules = [rng.choice(RULES)]
    if ov:
        rules.append(rng.choice([r for r in RULES if r != rules[0]]))
    return {"rules": rules, "override": ov, "need": 1 + ov}


def _rule_args(ctx, rule, eid):
    """(args, holes, must_include) for one rule instance."""
    if rule == "max_words":
        w = ctx.rng.choice(sorted(MAX_WORDS))
        return {"max": MAX_WORDS[w], "word": w}, {"N": w}, [w]
    if rule == "call_user":
        x, _ = P.draw(ctx.rng, "name", ctx.used, allow_nonce=False)
        sid = ctx.item_slot("name", x, key="nickname")
        return {"name": x, "slot": sid}, {"X": x}, [x]
    if rule == "avoid_word":
        cands = [w for w in P.pool("avoid_word").values if w not in ctx.used]
        w = ctx.rng.choice(cands)
        ctx.used.add(w)
        return {"word": w}, {"W": w}, [w]
    if rule == "start_name":
        sid = next((s for s, v in ctx.slots.items() if v["key"] == "user_name"), None) or ctx.new_slot("user_name")
        n = ctx.slots[sid]["value"]
        return {"name": n, "slot": sid}, {"n": n}, [n]
    return {}, {}, []


def s4_place(ctx, pre):
    eid = ctx.next_eid()
    gaps = [(1, 6)] if pre["override"] else []
    pos = ctx.pick(gaps)
    if not pos:
        raise Fail("S4 no positions")
    segs, turns = [], {}
    for j, (rule, k) in enumerate(zip(pre["rules"], pos)):
        args, holes, inc = _rule_args(ctx, rule, eid)
        opts = bank_options("rule." + rule)
        if j == 1:
            pre_ov = ctx.rng.choice(B.lines("rule.override"))
            opts = [(i + "+override", pre_ov + t) for i, t in opts]
        role = "rule" if j == 0 else "override"
        ctx.user(k, eid, role, opts, holes, f"{'replace the rule: ' if j else ''}ask for rule {rule}", inc)
        ctx.reply(k, eid, "agree and follow the rule", [], [])
        turns[role] = ["u", k]
        segs.append({"verifier": rule, "args": args, "start_k": k})
    return ctx.event("S4", {"rules": pre["rules"], "override": pre["override"], "segments": segs},
                     {"segments": []}, turns)


# ---- S5 ------------------------------------------------------------------------------------------------------------
def s5_pre(rng):
    n_o, g = rng.choice([1, 1, 2]), rng.randint(2, 4)
    return {"n_origin": n_o, "digression": g, "need": n_o + g + 1}


def s5_place(ctx, pre):
    eid = ctx.next_eid()
    n_o, g = pre["n_origin"], pre["digression"]
    pos = ctx.pick([(1, 1)] * (n_o + g))
    if not pos:
        raise Fail("S5 no positions")
    osids = []
    for j in range(n_o):
        osids.append(ctx.new_slot(_key_any(ctx)))
    dsid = ctx.new_slot(_key_any(ctx))
    ovals = [ctx.slots[s]["value"] for s in osids]
    dval = ctx.slots[dsid]["value"]
    turns, ops = {}, []
    for j, sid in enumerate(osids):
        _plant(ctx, pos[j], eid, sid, role=f"origin{j + 1}")
        ctx.reply(pos[j], eid, "respond helpfully on the first topic", [], [])
        turns[f"origin{j + 1}"] = ["u", pos[j]]
        ops.append({"turn": f"origin{j + 1}", "slot": sid, "value": ovals[j], "op": "plant"})
    for j in range(g):
        k = pos[n_o + j]
        role = f"digress{j + 1}"
        if j == 0:
            _plant(ctx, k, eid, dsid, role=role)
            ops.append({"turn": role, "slot": dsid, "value": dval, "op": "plant"})
        else:
            ctx.claim(k)
            ctx.uturn[k] = {"mode": "guided", "text": None, "bank_ref": None, "intent": "digression:second_topic",
                            "must_include": [], "must_exclude": list(ovals), "events": [eid], "role": role}
        ctx.uturn[k]["must_exclude"] = sorted(set(ctx.uturn[k]["must_exclude"]) | set(ovals))
        ctx.reply(k, eid, "respond on the second topic", [], ovals)
        turns[role] = ["u", k]
    r = pos[-1]
    ctx.user(r, eid, "return", bank_options("return"), {}, "go back to the first topic", [])
    ctx.reply(r, eid, "pick the first topic back up and name it", [ovals[0]], [dval])
    turns.update({"return": ["u", r], "answer": ["a", r]})
    params = {"origin_slots": osids, "digression_slot": dsid, "digression_turns": g, "ops": ops}
    return ctx.event("S5", params, {"origin": ovals, "digression": [dval]}, turns)


# ---- S6 ------------------------------------------------------------------------------------------------------------
def s6_pre(rng):
    v = pick_weighted(rng, S6_VARIANTS)
    return {"variant": v, "need": 3 if v == "two_persons" else 2, "ask": rng.choice(["self", "user"]),
            "answer_order": rng.choice(["first", "second"])}


def s6_place(ctx, pre):
    eid = ctx.next_eid()
    v = pre["variant"]
    if v == "two_persons":
        keys = [k for k in ("person_name", "person_city", "person_job") if not ctx.type_count[KEYS[k]["vtype"]]]
        if not keys:
            raise Fail("S6 type in use")
        key = ctx.rng.choice(keys)
        sa, sb = ctx.new_slot(key), ctx.new_slot(key)
        pos = None
        for d in _ds(ctx, 1, 9):
            pos = ctx.pick([(1, 3), (d + 1, d + 1)])
            if pos:
                break
        if not pos:
            raise Fail("S6 no positions")
        pa, pb, q = pos
        qs = ctx.rng.choice([sa, sb])
        other = sb if qs == sa else sa
        vals = {s: ctx.slots[s]["value"] for s in (sa, sb)}
        _plant(ctx, pa, eid, sa, role="plant_a")
        _plant(ctx, pb, eid, sb, role="plant_b")
        ctx.reply(pa, eid, "acknowledge briefly", [], [])
        ctx.reply(pb, eid, "acknowledge briefly", [], [vals[sa]])
        ref = _query(ctx, q, eid, qs, "full")
        _answer(ctx, q, eid, vals[qs], [vals[other]], pre["answer_order"])
        last = pa if qs == sa else pb
        ctx.window(vals[qs], last, q, eid)
        ops = [{"turn": "plant_a", "slot": sa, "value": vals[sa], "op": "plant"},
               {"turn": "plant_b", "slot": sb, "value": vals[sb], "op": "plant"}]
        turns = {"plant_a": ["u", pa], "plant_b": ["u", pb], "query": ["u", q], "answer": ["a", q]}
        params = {"variant": v, "slots": [sa, sb], "queried": qs, "ref_expr": ref, "d": q - last - 1, "ops": ops}
        gold = {"answer": vals[qs], "candidates": [vals[other]], "perspective": "your", "slot": qs}
        return ctx.event("S6", params, gold, turns)
    if v == "user_vs_card":
        sid = next((s for s, x in ctx.slots.items() if x["key"] == "user_name"), None)
        if sid:
            raise Fail("S6 user name already planted")
        sid = ctx.new_slot("user_name", reserved_ok=True)
        pos = ctx.pick([(2, 8)])
        if not pos:
            raise Fail("S6 no positions")
        p, q = pos
        uname = ctx.slots[sid]["value"]
        _plant(ctx, p, eid, sid)
        ctx.reply(p, eid, "greet the user by name", [uname], [])
        ask = pre["ask"]
        ctx.user(q, eid, "query", bank_options("identity.q." + ask), {}, f"ask for the {ask} name", [])
        gold_v, bad = (ctx.card_name, uname) if ask == "self" else (uname, ctx.card_name)
        _answer(ctx, q, eid, gold_v, [bad], pre["answer_order"])
        ops = [{"turn": "plant", "slot": sid, "value": uname, "op": "plant"}]
        params = {"variant": v, "slot": sid, "ask": ask, "needs_system": True, "d": q - p - 1, "ops": ops}
        gold = {"answer": gold_v, "candidates": [bad], "perspective": "self" if ask == "self" else "your"}
        return ctx.event("S6", params, gold, {"plant": ["u", p], "query": ["u", q], "answer": ["a", q]})
    keys = [k for k in SWAP_KEYS if ctx.type_count[KEYS[k]["vtype"]] == 0]
    if not keys:
        raise Fail("S6 no swap key")
    sid = ctx.new_slot(ctx.rng.choice(keys))
    pos = ctx.pick([(1, 6)])
    if not pos:
        raise Fail("S6 no positions")
    p, q = pos
    s = ctx.slots[sid]
    _plant(ctx, p, eid, sid)
    _ack(ctx, p, eid, s["value"], True)
    lab = B.label(s["key"], s["noun"])
    ctx.user(q, eid, "query", bank_options("swap.q"), {"lab": lab}, f"ask the assistant for its own {lab}", [])
    ctx.reply(q, eid, "say it has no such thing of its own, without inventing one", [], [])
    ops = [{"turn": "plant", "slot": sid, "value": s["value"], "op": "plant"}]
    params = {"variant": v, "slot": sid, "label": lab, "d": q - p - 1, "ops": ops}
    gold = {"answer": None, "deny_self": True, "user_value": s["value"], "perspective": "self"}
    return ctx.event("S6", params, gold, {"plant": ["u", p], "query": ["u", q], "answer": ["a", q]})
