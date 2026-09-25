"""Event checks (SPEC section 6): planting and corrections at their scheduled turns, values never said before they
are scheduled, distance integrity, query form, answer correctness (wrong, stale, shotgun, deflection), list state
and topic return. Golds come from the skeleton (golds.derive recomputes them; test_skeleton checks they agree)."""
import re

import lexicons as L
from check_base import event_values, same_type, slot_type, mentioned, value_re_i

PLANT_OPS = {"plant", "init"}
CORR_OPS = {"set", "fix", "err", "twin", "add", "remove", "move_first", "move_last"}
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve"]
YES_RE = re.compile(r"(?<![a-z])(?:yes|yep|yeah|still (?:on|there)|it is|it's (?:still )?(?:on|there))(?![a-z])", re.I)
NO_RE = re.compile(r"(?<![a-z])(?:no|nope|not|isn't|is not|off|removed|gone|took .{1,30} off)(?![a-z])", re.I)


def slot_event(e):
    k, v = e["kind"], e["params"].get("variant")
    return k in ("S1", "S3") or (k, v) in (("S6", "two_persons"), ("S6", "user_vs_card"), ("S7", "given")) or \
        (k == "S9" and e["params"]["need_kind"] == "context")


def chk_ops(ctx):
    out = []
    for e in ctx.skel["events"]:
        for op in e["params"].get("ops", []):
            i = e["turns"][op["turn"]]
            if i not in ctx.text:
                continue
            code = "PLANT_MISSING" if op["op"] in PLANT_OPS else "CORR_MISSING" if op["op"] in CORR_OPS else None
            vals = op.get("items") or [op.get("value")]
            out += [(code, i, f"{e['id']} {v}") for v in vals if code and v and not ctx.has(i, v)]
            s = ctx.skel["slots"].get(op.get("slot") or "")
            t = ctx.by_i[i]
            if op["op"] == "plant" and s and s.get("noun") and t["mode"] == "guided" and t["role"] == "user" \
                    and not value_re_i(s["noun"]).search(ctx.text[i]):
                out.append(("PLANT_UNBOUND", i, f"{e['id']} {s['noun']}"))
    return out


def chk_early(ctx):
    """VALUE_EARLY: a scheduled value said in a turn before the first turn that schedules it (a correction's new
    value before the correction, a list item before the list, the user's name before they give it)."""
    out = []
    for v, first in ctx.first.items():
        for t, s in ctx.turns():
            if t["i"] < first and ctx.has(t["i"], v):
                out.append(("VALUE_EARLY", t["i"], f"{v} (scheduled at {first})"))
    return out


def _answer_value(e):
    g = e["gold"]
    return g.get("answer") if isinstance(g.get("answer"), str) else None


def chk_distance(ctx):
    out = []
    for e in ctx.skel["events"]:
        a = _answer_value(e)
        if not slot_event(e) or a is None or a == ctx.card:
            continue
        q = e["turns"]["query"]
        if e["kind"] == "S9":
            last = e["turns"]["context"]
        else:
            queried = e["params"].get("queried") or e["params"].get("slot")
            last = max(e["turns"][op["turn"]] for op in e["params"]["ops"]
                       if op.get("slot") == queried and op["op"] in ("plant", "set", "fix"))
        for i in range(last + 1, q):
            if i in ctx.text and e["id"] not in ctx.by_i[i]["events"] and ctx.has(i, a):
                out.append(("DIST_LEAK", i, f"{e['id']} {a}"))
    return out


def chk_query(ctx):
    out, vals = [], event_values(ctx.skel)
    for e in ctx.skel["events"]:
        q = e["turns"].get("query")
        if q is None or q not in ctx.text:
            continue
        a = _answer_value(e)
        leakable = e["kind"] != "S2" or e["gold"]["query"] != "contains"
        if a and leakable and a != ctx.card and ctx.has(q, a):
            out.append(("QUERY_RESTATES", q, f"{e['id']} {a}"))
        t = ctx.by_i[q]
        if t["mode"] == "guided":
            for ref in t["must_include"]:
                if ref not in vals and not ctx.has(q, ref):
                    out.append(("QUERY_NOREF", q, f"{e['id']} {ref}"))
    return out


def chk_answers(ctx):
    out = []
    for e in ctx.skel["events"]:
        i = e["turns"].get("answer")
        if not slot_event(e) or i not in ctx.text:
            continue
        g, a = e["gold"], _answer_value(e)
        if a is None:
            continue
        present = ctx.has(i, a)
        if not present:
            out.append(("ANSWER_WRONG", i, f"{e['id']} lacks {a}"))
        stale = [v for v in g.get("stale", []) if ctx.has(i, v)]
        if stale:
            out.append(("ANSWER_STALE", i, f"{e['id']} {stale}"))
        required = set(ctx.by_i[i]["must_include"])
        cands = (set(g.get("candidates", [])) | set(g.get("stale", [])) |
                 same_type(ctx.skel, slot_type(ctx.skel, a))) - {a} - required
        named = mentioned(ctx, i, cands)
        if present + len(named) >= 2:
            out.append(("ANSWER_SHOTGUN", i, f"{e['id']} also {named}"))
        if L.DEFLECT_RE.search(ctx.text[i]):
            out.append(("DEFLECT", i, L.DEFLECT_RE.search(ctx.text[i]).group(0)))
    return out


def chk_list(ctx):
    out = []
    for e in ctx.skel["events"]:
        i = e["turns"].get("answer")
        if e["kind"] != "S2" or i not in ctx.text:
            continue
        g, s = e["gold"], ctx.text[i]
        items = set(e["params"]["items"]) | {op["value"] for op in e["params"]["ops"] if op.get("value")}
        if g["query"] == "contains":
            ok = bool(YES_RE.search(s)) and not re.match(r"\s*(?:no|nope|not)\b", s, re.I) if g["answer"] == "yes" \
                else bool(NO_RE.search(s)) and not re.match(r"\s*(?:yes|yep|yeah)\b", s, re.I)
            if not ok:
                out.append(("LIST_STATE", i, f"{e['id']} contains should be {g['answer']}"))
            continue
        if g["query"] == "count":
            others = mentioned(ctx, i, [w for w in NUMBER_WORDS if w != g["answer"]])
            if not ctx.has(i, g["answer"]) or others:
                out.append(("LIST_STATE", i, f"{e['id']} count {g['answer']} vs {others}"))
            continue
        wrong = mentioned(ctx, i, items - {g["answer"]} - set(ctx.by_i[i]["must_include"]))
        if not ctx.has(i, g["answer"]) or wrong:
            out.append(("LIST_STATE", i, f"{e['id']} {g['answer']} vs {wrong}"))
    return out


def chk_return(ctx):
    out = []
    for e in ctx.skel["events"]:
        i = e["turns"].get("answer")
        if e["kind"] != "S5" or i not in ctx.text:
            continue
        o = mentioned(ctx, i, e["gold"]["origin"])
        d = mentioned(ctx, i, e["gold"]["digression"])
        if not o or d:
            out.append(("TOPIC_RETURN", i, f"{e['id']} origin {o} digression {d}"))
    return out


CHECKS = [chk_ops, chk_early, chk_distance, chk_query, chk_answers, chk_list, chk_return]
