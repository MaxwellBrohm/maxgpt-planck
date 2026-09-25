"""Skeleton sampler (SPEC sections 2-3): build(seed, register) -> one skeleton dict, deterministic per seed.
Kinds are drawn first (S9 only through the lookup slice), then per-kind parameters, then a turn count at least as
large as the events need, then positions. A layout that does not fit is redrawn (parameters, then turn count);
after 40 failed layouts the kinds are redrawn. Every skeleton passes gate.check before it is returned.
shard(shard_seed, n) draws personas and topics without replacement and never repeats a (topic, persona, kind set)."""
import hashlib
import json

import assemble as A
import banks as B
import events_a
import events_b
import events_c
import events_s3
import gate
import pools as P
import fake_data as F
from events_base import Ctx, Fail, pick_weighted

GEN_VERSION = "skel-v0.1"
TURN_WEIGHTS = {4: 5, 5: 10, 6: 20, 7: 20, 8: 20, 9: 7.5, 10: 7.5, 11: 5, 12: 5}
NEVENT_WEIGHTS = {2: 25, 3: 45, 4: 30}
KIND_SHARES = {"S1": 18, "S2": 12, "S3": 20, "S4": 10, "S5": 7, "S6": 13, "S7": 10, "S8": 5}
LOOKUP_SLICE = 0.20
S9_PER_LOOKUP = {1: 80, 2: 20}
CLOSING = {"goodbye": 40, "open": 35, "abrupt": 25}
TOPIC_COUNT = {1: 40, 2: 40, 3: 20}
SYSTEM_SHARE = 0.4
P_EXACT = 0.7
N_PERSONAS = 2000
PLANNERS = {"S1": (events_a.s1_pre, events_a.s1_place), "S2": (events_a.s2_pre, events_a.s2_place),
            "S3": (events_s3.s3_pre, events_s3.s3_place), "S4": (events_b.s4_pre, events_b.s4_place),
            "S5": (events_b.s5_pre, events_b.s5_place), "S6": (events_b.s6_pre, events_b.s6_place),
            "S7": (events_c.s7_pre, events_c.s7_place), "S8": (events_c.s8_pre, events_c.s8_place),
            "S9": (events_c.s9_pre, events_c.s9_place)}
FIXED_ACTS = ("greeting", "goodbye")


def topic_ids():
    return [f"t{i:04d}" for i in range(len(P.pool("topic").values))]


def topic_text(tid):
    return P.pool("topic").values[int(tid[1:])]


def draw_kinds(rng):
    lookup = rng.random() < LOOKUP_SLICE
    n = pick_weighted(rng, NEVENT_WEIGHTS)
    n9 = min(pick_weighted(rng, S9_PER_LOOKUP), n - 1) if lookup else 0
    return lookup, ["S9"] * n9 + [pick_weighted(rng, KIND_SHARES) for _ in range(n - n9)]


def draw_pres(rng, kinds):
    """per-event parameters; a second S8 never repeats a fixed act, a second S2 never repeats an item type."""
    pres, fixed, list_types = [], set(), set()
    for k in kinds:
        pre = PLANNERS[k][0](rng)
        while (k == "S8" and pre["act"] in fixed) or (k == "S2" and pre["item_type"] in list_types):
            pre = PLANNERS[k][0](rng)
        while k == "S6" and pre["variant"] == "user_vs_card" and any(
                q.get("variant") == "user_vs_card" for q in pres if "variant" in q and "ask" in q):
            pre = PLANNERS[k][0](rng)
        if k == "S4" and "S6" in kinds:
            while "start_name" in pre["rules"]:
                pre = PLANNERS[k][0](rng)
        if k == "S8" and pre["act"] in FIXED_ACTS:
            fixed.add(pre["act"])
        if k == "S2":
            list_types.add(pre["item_type"])
        pres.append(pre)
    return pres, fixed


def _shrinkable(kinds, pres, fixed, info):
    out = []
    for i, (k, p) in enumerate(zip(kinds, pres)):
        if k == "S3" and p["variant"] == "update" and p["k"] > 1:
            out.append((i, "k"))
        elif k == "S5" and (p["digression"] > 2 or p["n_origin"] > 1):
            out.append((i, "digression" if p["digression"] > 2 else "n_origin"))
        elif (k == "S1" and p["distractor"]) or (k == "S2" and p["n_ops"]) or (k == "S4" and p["override"]):
            out.append((i, {"S1": "distractor", "S2": "n_ops", "S4": "override"}[k]))
    if info["closing"] == "goodbye" and "goodbye" not in fixed:
        out.append((None, "closing"))
    if not out and "greeting" not in fixed and not info.get("no_opening"):
        out.append((None, "no_opening"))
    return out


def shrink(rng, kinds, pres, fixed, info):
    """reduce within-event sizes (never kinds or variants) until the events fit in the largest turn count, so
    over-long draws do not bias event or variant shares. Changes k, digression length, distractors, list ops,
    overrides or the goodbye closing, and records nothing else."""
    while _need(pres, fixed, info) > max(TURN_WEIGHTS):
        opts = _shrinkable(kinds, pres, fixed, info)
        if not opts:
            return
        i, field = rng.choice(opts)
        if field == "closing":
            info["closing"] = "open"
            continue
        if field == "no_opening":
            info["no_opening"] = True
            continue
        p = pres[i]
        if field == "override":
            p["override"], p["rules"] = False, p["rules"][:1]
        elif field == "distractor":
            p["distractor"] = False
        else:
            p[field] -= 1
        p["need"] -= 1
        p["shrunk"] = p.get("shrunk", 0) + 1


def draw_T(rng, need):
    ok = {t: w for t, w in TURN_WEIGHTS.items() if t >= need}
    return pick_weighted(rng, ok) if ok else None


def _order(rng, kinds, pres):
    idx = list(range(len(kinds)))
    rng.shuffle(idx)
    return sorted(idx, key=lambda i: (-(kinds[i] == "S8" and pres[i]["act"] in FIXED_ACTS), -pres[i]["need"]))


def _layout(rng, info, kinds, pres, fixed, T):
    card = P.draw(rng, "assistant_name", set(), allow_nonce=False)[0]
    ctx = Ctx(rng, T, P_EXACT, info["register"], info["lookup"], card)
    ctx.digression_topic_needed = "S5" in kinds
    if any(k == "S6" and p["variant"] == "user_vs_card" for k, p in zip(kinds, pres)):
        ctx.reserved_keys.add("user_name")
    opening = "greeting" not in fixed and not info.get("no_opening")
    if opening:
        ctx.free.discard(0)
    if info["closing"] == "goodbye" and "goodbye" not in fixed:
        ctx.free.discard(T - 1)
    for i in _order(rng, kinds, pres):
        PLANNERS[kinds[i]][1](ctx, pres[i])
    if info["closing"] == "goodbye" and "goodbye" not in fixed:
        ctx.free.add(T - 1)
    if opening:
        ctx.free.add(0)
    A.fill_frame(ctx, info)
    for k, u in ctx.uturn.items():
        if u["intent"] == "digression:second_topic":
            u["intent"] = f"topic:{info['topic_path'][1]}:digress"
    A.apply_windows(ctx)
    return ctx


def _skeleton(ctx, info, seed, attempt):
    turns, index = A.build_turns(ctx, info["register"], info["style"])
    events = []
    for e in ctx.events:
        e = dict(e)
        e["turns"] = {role: index[(side, k)] for role, (side, k) in e.pop("turns_k").items()}
        events.append(e)
    needs_system = any(e["params"].get("needs_system") for e in events)
    use_sys = needs_system or ctx.rng.random() < SYSTEM_SHARE
    sys_bank = "system.lookup" if info["lookup"] else "system.plain"
    sys_i = ctx.rng.randrange(len(B.lines(sys_bank)))
    card_sid = ctx.item_slot("assistant_name", ctx.card_name, key="assistant_name")
    if not use_sys:
        for t in turns:
            if t["role"] == "assistant" and t["mode"] == "guided":
                t["must_exclude"] = sorted(set(t["must_exclude"]) | {ctx.card_name})
    uname = next((s for s, v in ctx.slots.items() if v["key"] == "user_name"), None)
    skel = {
        "skel_id": None, "seed": seed, "attempt": attempt, "gen_version": GEN_VERSION,
        "register": info["register"], "slice": "lookup" if info["lookup"] else "core",
        "topic_path": info["topic_path"], "topic_text": {t: info["topic_text"][t] for t in info["topic_path"]},
        "opening": ctx.uturn[0]["bank_ref"] or ctx.uturn[0]["intent"], "closing": info["closing"],
        "user": {"name": uname, "age_band": info["age_band"], "style": info["style"],
                 "persona_seed": info["persona"],
                 "facts": [s for s, v in ctx.slots.items() if v["owner"] == "user" and v["counted"]][:4]},
        "assistant": {"name": card_sid, "may_say": (["name"] if use_sys else []) + ["is_assistant"]
                      + (["can_look_up"] if info["lookup"] else []), "must_not": list(A.MUST_NOT),
                      "system_text": B.fill(B.lines(sys_bank)[sys_i], A=ctx.card_name) if use_sys else None,
                      "system_ref": B.line_id(sys_bank, sys_i) if use_sys else None},
        "slots": ctx.slots, "persons": ctx.persons, "events": events, "turns": turns,
    }
    A.apply_rules(skel)
    A.finish(skel, ctx)
    return skel


def skel_hash(skel):
    body = {k: v for k, v in skel.items() if k != "skel_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _info(rng, register, persona, topics):
    n_top = pick_weighted(rng, TOPIC_COUNT)
    tids = list(topics) if topics else rng.sample(topic_ids(), 3)
    return {"register": register, "closing": pick_weighted(rng, CLOSING), "style": pick_weighted(rng, F.STYLES),
            "age_band": rng.choice(F.AGE_BANDS), "persona": persona or f"np{rng.randrange(N_PERSONAS):05d}",
            "topic_path": tids[:n_top], "all_topics": tids,
            "topic_text": {t: topic_text(t) for t in tids}, "intent_forms": F.INTENT_FORMS}


def _need(pres, fixed, info):
    need = sum(p["need"] for p in pres) + ("greeting" not in fixed and not info.get("no_opening"))
    return need + (info["closing"] == "goodbye" and "goodbye" not in fixed)


def build(seed, register="RM", persona=None, topics=None, kind_draws=5, attempts=12):
    """one gated skeleton; deterministic in (seed, register, persona, topics). Kinds and event parameters come from
    their own stream and stay fixed while layouts are retried, so layout failures do not bias the event shares;
    they are redrawn only if no layout fits after all attempts."""
    for r in range(kind_draws):
        krng = P.seeded("planck-kinds", GEN_VERSION, register, seed, r)
        info = _info(krng, register, persona, topics)
        info["lookup"], kinds = draw_kinds(krng)
        pres, fixed = draw_pres(krng, kinds)
        shrink(krng, kinds, pres, fixed, info)
        if "goodbye" in fixed:
            info["closing"] = "goodbye"
        if "S5" in kinds and len(info["topic_path"]) < 2:
            info["topic_path"] = info["all_topics"][:2]
        need = _need(pres, fixed, info)
        for attempt in range(attempts):
            rng = P.seeded("planck-skel", GEN_VERSION, register, seed, r, attempt)
            T = draw_T(rng, need + attempt // 3)
            if T is None:
                break
            for _ in range(4):
                try:
                    ctx = _layout(rng, info, kinds, pres, fixed, T)
                    skel = _skeleton(ctx, info, seed, [r, attempt])
                except Fail:
                    continue
                if gate.check(skel):
                    continue
                skel["skel_id"] = "sk-" + skel_hash(skel)[:16]
                return skel
    raise RuntimeError(f"no skeleton for seed {seed}")


def triple(sk):
    """the (topic path, persona, kind set) triple that a shard never repeats."""
    return tuple(sk["topic_path"]), sk["user"]["persona_seed"], tuple(sorted(e["kind"] for e in sk["events"]))


def iter_shard(shard_seed, register="RM", start_j=0, seen=None):
    """endless generator of (j, skeleton) in shard order; shard() takes the first n. Resumable: pass the j of the
    last skeleton already produced and the triples seen so far, and the stream continues exactly where it stopped."""
    rng = P.seeded("planck-shard", GEN_VERSION, shard_seed)
    personas = [f"np{i:05d}" for i in range(N_PERSONAS)]
    tops = topic_ids()
    rng.shuffle(personas)
    rng.shuffle(tops)
    seen, j = set() if seen is None else set(seen), start_j
    while True:
        persona = personas[j % len(personas)]
        three = [tops[(3 * j + m) % len(tops)] for m in range(3)]
        j += 1
        sk = build(f"{shard_seed}:{j}", register, persona=persona, topics=three)
        tr = triple(sk)
        if tr in seen:
            continue
        seen.add(tr)
        yield j, sk


def shard(shard_seed, n, register="RM"):
    """n skeletons with personas and topics drawn without replacement (cycling when exhausted) and no repeated
    (topic path, persona, kind set) triple."""
    out = []
    if n <= 0:
        return out
    for _, sk in iter_shard(shard_seed, register):
        out.append(sk)
        if len(out) >= n:
            break
    return out
