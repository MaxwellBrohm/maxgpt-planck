"""Event planner S3, correction and update (P-009). Variants: update (k 1-3 corrections), no_update_twin (a
look-alike mention that changes nothing), two_slot (two same-type slots, one corrected), revert (change, then back
to the original) and assistant_err (the assistant misstates the value, that turn gets mask 1, the user fixes it).
Held-out limits: at most 3 corrections of one slot, d at most 10 (E004 H3/H4 stay unseen)."""
import banks as B
from banks_keys import KEYS
from events_base import Fail, key_options, bank_options, pick_weighted
from events_a import S3_VARIANTS, _key_any, _pair_keys, _ds, _marker, _plant, _query, _ack, _answer

NEED = {"update": None, "no_update_twin": 3, "two_slot": 4, "revert": 4, "assistant_err": 4}


def s3_pre(rng):
    v = pick_weighted(rng, S3_VARIANTS)
    k = rng.randint(1, 3) if v == "update" else (2 if v == "revert" else 1)
    need = 2 + k if v == "update" else NEED[v]
    return {"variant": v, "k": k, "need": need, "ref_form_q": rng.choice(["full", "head"]),
            "ack_mentions": rng.random() < 0.5, "answer_order": rng.choice(["first", "second"])}


def _corr_forms(ctx, sid):
    s = ctx.slots[sid]
    forms = sorted({f for f, _ in KEYS[s["key"]]["corr"]})
    p = B.key_holes(s["key"], s["noun"], s["value"], s["features"])["p"]
    return [f for f in forms if f != "pronoun" or p]


def _correction(ctx, k, eid, sid, new, old, role, markers=None):
    """one user correction turn: new value replaces old. Returns the params record of the correction."""
    s = ctx.slots[sid]
    form = ctx.rng.choice(_corr_forms(ctx, sid))
    if markers is None:
        mid, m = _marker(ctx)
    else:
        mid = ctx.rng.randrange(len(markers))
        m = markers[mid]
    feats = dict(s["features"], article=B.pools_article(new, s["type"]))
    holes = B.key_holes(s["key"], s["noun"], new, feats, old=old, marker=m)
    ref = None if form in ("pronoun", "ellipsis") else ctx.ref(sid, "full")
    ctx.user(k, eid, role, key_options(s["key"], "corr", form), holes, f"correct {B.label(s['key'], s['noun'])}",
             [new] + ([ref] if ref else []))
    ctx.reply(k, eid, "acknowledge the change", [], [])
    return {"turn": role, "slot": sid, "value": new, "op": "set", "old": old, "form": form,
            "marker_id": mid, "marker_set": "err" if markers is not None else "main"}


def s3_place(ctx, pre):
    v = pre["variant"]
    eid = ctx.next_eid()
    if v == "two_slot":
        ka, kb = _pair_keys(ctx)
        sid, sid_b = ctx.new_slot(ka), ctx.new_slot(kb)
    else:
        sid, sid_b = ctx.new_slot(_key_any(ctx)), None
    s = ctx.slots[sid]
    v0 = s["value"]
    pos = None
    for d in _ds(ctx):
        if v == "update":
            gaps = [(1, 3)] * pre["k"] + [(d + 1, d + 1)]
        elif v == "no_update_twin":
            gaps = [(1, 3), (d + 1, d + 1)]
        elif v == "two_slot":
            gaps = [(1, 3), (1, 3), (d + 1, d + 1)]
        elif v == "revert":
            gaps = [(1, 3), (1, 3), (d + 1, d + 1)]
        else:
            gaps = [(1, 3), (1, 1), (d + 1, d + 1)]
        pos = ctx.pick(gaps)
        if pos:
            break
    if not pos:
        raise Fail("S3 no positions")
    ops, turns, stale, cands = [], {}, [], []
    q = pos[-1]
    queried = sid
    if v == "two_slot":
        a_first = ctx.rng.random() < 0.5
        pa, pb = (pos[0], pos[1]) if a_first else (pos[1], pos[0])
        _plant(ctx, pa, eid, sid, role="plant_a")
        _plant(ctx, pb, eid, sid_b, role="plant_b")
        for kk in (pa, pb):
            ctx.reply(kk, eid, "acknowledge briefly", [], [])
        vb = ctx.slots[sid_b]["value"]
        ops += [{"turn": "plant_a", "slot": sid, "value": v0, "op": "plant"},
                {"turn": "plant_b", "slot": sid_b, "value": vb, "op": "plant"}]
        turns.update(plant_a=["u", pa], plant_b=["u", pb])
        new = ctx.extra_value(s["type"])
        ops.append(_correction(ctx, pos[2], eid, sid, new, v0, "corr1"))
        turns["corr1"] = ["u", pos[2]]
        queried = ctx.rng.choice([sid, sid_b])
        if queried == sid:
            gold, stale, cands, last = new, [v0], [vb], pos[2]
        else:
            gold, stale, cands, last = vb, [], [v0, new], pb
    else:
        p = pos[0]
        _plant(ctx, p, eid, sid)
        _ack(ctx, p, eid, v0, pre["ack_mentions"])
        ops.append({"turn": "plant", "slot": sid, "value": v0, "op": "plant"})
        turns["plant"] = ["u", p]
        hist = [v0]
        if v == "update":
            for j in range(pre["k"]):
                new = ctx.extra_value(s["type"])
                ops.append(_correction(ctx, pos[1 + j], eid, sid, new, hist[-1], f"corr{j + 1}"))
                turns[f"corr{j + 1}"] = ["u", pos[1 + j]]
                hist.append(new)
            gold, stale, last = hist[-1], hist[:-1], pos[-2]
        elif v == "revert":
            new = ctx.extra_value(s["type"])
            ops.append(_correction(ctx, pos[1], eid, sid, new, v0, "corr1"))
            ops.append(_correction(ctx, pos[2], eid, sid, v0, new, "corr2"))
            turns.update(corr1=["u", pos[1]], corr2=["u", pos[2]])
            gold, stale, last = v0, [new], pos[2]
        elif v == "no_update_twin":
            tv = ctx.extra_value(s["type"])
            holes = ctx.holes(sid)
            holes.update(v=tv, av=B.av(tv, dict(s["features"], article=B.pools_article(tv, s["type"]))))
            ctx.user(pos[1], eid, "twin", key_options(s["key"], "twin"), holes, "mention a look-alike value that "
                     "changes nothing", [tv])
            ctx.reply(pos[1], eid, "react briefly", [], [])
            ops.append({"turn": "twin", "slot": sid, "value": tv, "op": "twin"})
            turns["twin"] = ["u", pos[1]]
            gold, stale, cands, last = v0, [], [tv], p
        else:  # assistant_err
            r, f = pos[1], pos[2]
            w = ctx.extra_value(s["type"])
            ctx.user(r, eid, "recap", bank_options("recap"), {}, "ask for a recap", [])
            ctx.reply(r, eid, f"recap, but state the {B.label(s['key'], s['noun'])} wrongly as {w}", [w], [v0],
                      mask=1)
            ops.append({"turn": "recap_reply", "slot": sid, "value": w, "op": "err"})
            ops.append(_correction(ctx, f, eid, sid, v0, w, "fix", markers=B.ERR_FIX_MARKERS))
            ops[-1]["op"] = "fix"
            turns.update(recap=["u", r], recap_reply=["a", r], fix=["u", f])
            gold, stale, last = v0, [w], f
    if not 1 <= q - last - 1 <= 10:
        raise Fail("S3 distance out of range")
    ref = _query(ctx, q, eid, queried, pre["ref_form_q"])
    turns.update(query=["u", q], answer=["a", q])
    _answer(ctx, q, eid, gold, stale + cands, pre["answer_order"])
    ctx.window(gold, last, q, eid)
    params = {"variant": v, "slot": sid, "slot_b": sid_b, "queried": queried, "k": pre["k"], "d": q - last - 1,
              "ref_form": pre["ref_form_q"], "ref_expr": ref, "ack_mentions": pre["ack_mentions"],
              "answer_order": pre["answer_order"], "ops": ops}
    return ctx.event("S3", params, {"answer": gold, "stale": stale, "candidates": cands, "slot": queried}, turns)
