"""Turn a placed Ctx into the skeleton JSON of SPEC section 2: opening, closing and filler turns, leak windows as
must_exclude, S4 rule constraints, S7 abstain candidates, global turn indices (lookup call and tool turns
inserted), word bounds per register and style, required words, gold notes, provenance and the gate hash."""
import banks as B
import golds
import heldout
import pools as P
from events_base import Fail, bank_options

REGISTERS = {"RS": {"user": (3, 20), "assistant": (3, 25)}, "RM": {"user": (3, 25), "assistant": (3, 30)},
             "RL": {"user": (3, 30), "assistant": (3, 40)}}
TOKENS_PER_WORD = 1.3
HARD_CAP_TOKENS = 1800
MUST_NOT = ["plans", "family", "body", "past", "preferences", "places"]


def _words(s):
    return len(s.split())


def fill_frame(ctx, info):
    """opening, closing and fillers on the free user turns, split across the topic path in order."""
    rng = ctx.rng
    if 0 in ctx.free:
        topic_text = info["topic_text"][info["topic_path"][0]]
        bank = "open.topic" if rng.random() < 0.7 else "open.greet"
        ctx.user(0, "opening", "opening", bank_options(bank), {"t": topic_text}, "open the chat about the topic",
                 [], exact=None)
        ctx.uturn[0]["events"] = []
        ctx.aturn[0] = {"intent": "greet back and engage with the topic", "must_include": [], "must_exclude": [],
                        "mask": 0, "events": [], "lookup": None}
    last = ctx.T - 1
    if info["closing"] == "goodbye" and last in ctx.free:
        ctx.user(last, "closing", "closing", bank_options("close.goodbye"), {}, "say goodbye", [])
        ctx.uturn[last]["events"] = []
        ctx.aturn[last] = {"intent": "say goodbye briefly", "must_include": [], "must_exclude": [], "mask": 0,
                           "events": [], "lookup": None}
    free = sorted(ctx.free)
    path = info["topic_path"]
    if ctx.digression_topic_needed and len(path) < 2:
        raise Fail("S5 needs two topics")
    for n, k in enumerate(free):
        tid = path[min(len(path) - 1, n * len(path) // max(1, len(free)))]
        form = rng.randrange(len(info["intent_forms"]))
        ctx.claim(k)
        ctx.uturn[k] = {"mode": "guided", "text": None, "bank_ref": None, "intent": f"topic:{tid}:{form}",
                        "must_include": [], "must_exclude": [], "events": [], "role": "filler"}
        ctx.aturn[k] = {"intent": "reply on the topic", "must_include": [], "must_exclude": [], "mask": 0,
                        "events": [], "lookup": None}
    if info["closing"] == "open":
        ctx.aturn[last]["intent"] += "; end with an open question"


def apply_windows(ctx):
    for w in ctx.windows:
        for k in range(w["start"] + 1, w["end"]):
            u, a = ctx.uturn[k], ctx.aturn[k]
            if w["eid"] in u["events"]:
                continue
            if u["mode"] == "exact" and golds.value_re(w["value"]).search(u["text"]):
                raise Fail("exact turn leaks a windowed value")
            for spec in (u, a):
                if w["value"] not in spec["must_exclude"] and w["value"] not in spec["must_include"]:
                    spec["must_exclude"].append(w["value"])
                elif w["value"] in spec["must_include"]:
                    raise Fail("window conflicts with a required mention")


def build_turns(ctx, reg, style):
    ulo, uhi = REGISTERS[reg]["user"]
    alo, ahi = REGISTERS[reg]["assistant"]
    if style == "terse":
        uhi = min(uhi, 10)
    turns, index = [], {}
    for k in range(ctx.T):
        u = ctx.uturn[k]
        t = dict(i=len(turns), role="user", mode=u["mode"], text=u["text"], bank_ref=u["bank_ref"],
                 intent=None if u["mode"] == "exact" else u["intent"], min_w=ulo, max_w=uhi,
                 must_include=u["must_include"], must_exclude=sorted(set(u["must_exclude"])), events=u["events"],
                 mask=0, gold_note=None, role_in_event=u["role"])
        if u["text"]:
            t["min_w"], t["max_w"] = min(ulo, _words(u["text"])), max(uhi, _words(u["text"]))
        index[("u", k)] = t["i"]
        turns.append(t)
        a = ctx.aturn[k]
        if a["lookup"]:
            index[("a", k)] = len(turns)
            turns.append(dict(i=len(turns), role="assistant", mode="exact", text=a["lookup"]["call"], bank_ref=None,
                              intent=None, min_w=1, max_w=_words(a["lookup"]["call"]), must_include=[],
                              must_exclude=[], events=a["events"], mask=0, gold_note=None, lookup_call=True))
            index[("t", k)] = len(turns)
            turns.append(dict(i=len(turns), role="tool", mode="exact", text=a["lookup"]["result"], bank_ref=None,
                              intent=None, min_w=1, max_w=_words(a["lookup"]["result"]), must_include=[],
                              must_exclude=[], events=a["events"], mask=0, gold_note=None))
            index[("a2", k)] = len(turns)
        else:
            index[("a", k)] = len(turns)
        turns.append(dict(i=len(turns), role="assistant", mode="guided", text=None, bank_ref=None,
                          intent=a["intent"], min_w=alo, max_w=ahi, must_include=a["must_include"],
                          must_exclude=sorted(set(a["must_exclude"]) - set(a["must_include"])), events=a["events"],
                          mask=a["mask"], gold_note=None))
    return turns, index


def apply_rules(skel):
    """S4 golds (segments with assistant turn lists) and the matching per-turn constraints."""
    segs = golds.rule_segments(skel)
    by_i = {t["i"]: t for t in skel["turns"]}
    for e in skel["events"]:
        if e["kind"] != "S4":
            continue
        e["gold"] = {"segments": segs[e["id"]]}
        for seg in segs[e["id"]]:
            for i in seg["turns"]:
                t = by_i[i]
                t.setdefault("rules", []).append({"verifier": seg["verifier"], "args": seg["args"]})
                a = seg["args"]
                if seg["verifier"] == "max_words":
                    t["max_w"] = min(t["max_w"], a["max"])
                    t["min_w"] = min(t["min_w"], t["max_w"])
                elif seg["verifier"] == "avoid_word":
                    if any(a["word"] in x.lower().split() for x in t["must_include"]):
                        raise Fail("avoid word required")
                    t["must_exclude"] = sorted(set(t["must_exclude"]) | {a["word"]})
                elif seg["verifier"] in ("call_user", "start_name"):
                    if a["name"] in t["must_exclude"]:
                        raise Fail("rule name excluded")
                    t["must_include"] = t["must_include"] + [a["name"]]


def finish(skel, ctx):
    """S7 candidates, required words, gold notes, token estimate, provenance."""
    by_i = {t["i"]: t for t in skel["turns"]}
    for e in skel["events"]:
        if e["kind"] == "S7" and e["params"]["variant"] == "abstain":
            e["gold"]["candidates"] = golds.same_type_values(skel, e["params"]["vtype"])
            t = by_i[e["turns"]["answer"]]
            if set(t["must_include"]) & set(e["gold"]["candidates"]):
                raise Fail("abstain candidate required by a rule")
            if e["params"]["key"] == "user_name" and any(x["key"] == "nickname" for x in skel["slots"].values()):
                raise Fail("abstain on the name after a call-me rule")
            t["must_exclude"] = sorted(set(t["must_exclude"]) | set(e["gold"]["candidates"]))
    rng = ctx.rng
    rw = {}
    for part in ("noun", "verb", "adj"):
        cands = [w for w in P.pool("req_" + part).values if w not in ctx.used]
        rw[part] = rng.choice(cands)
    fill = [t["i"] for t in skel["turns"] if t["role"] == "assistant" and not t["events"] and t["mode"] == "guided"]
    other = [t["i"] for t in skel["turns"] if t["role"] == "assistant" and t["mode"] == "guided" and t["i"] not in fill]
    hint = rng.sample(fill, min(3, len(fill)))
    hint += rng.sample(other, min(3 - len(hint), len(other)))
    rw["turn_hint"] = sorted(hint)
    skel["required_words"] = rw
    for i, note in golds.notes(skel).items():
        by_i[i]["gold_note"] = note
    est = sum(t["max_w"] for t in skel["turns"]) * TOKENS_PER_WORD
    if est > HARD_CAP_TOKENS:
        raise Fail("token cap")
    mid = sum((t["min_w"] + t["max_w"]) / 2 for t in skel["turns"]) * TOKENS_PER_WORD
    skel["estimate"] = {"tokens_mid": round(mid), "tokens_max": round(est), "flag": "word count x 1.3"}
    vtypes = {s["type"] for s in skel["slots"].values()} | {"topic", "relation", "plan", "object", "pet_kind",
                                                            "req_noun", "req_verb", "req_adj", "entity_kind"}
    skel["provenance"] = {"banks": [B.bank_ref()], "pools": P.provenance_refs(vt for vt in vtypes if vt in P.POOLS),
                          "personas": "FAKE", "fake": True}
    skel["heldout_gate"] = heldout.gate_hash()
    skel["rc12"] = heldout.RC12_STATUS
